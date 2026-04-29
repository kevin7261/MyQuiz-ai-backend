"""
測驗題 LLM 生成（兩條路徑）：

1. generate_quiz — 有 FAISS RAG ZIP：向量檢索 → 以檢索片段為「課程內容」出題。
2. generate_quiz_transcription_only — 無向量庫（unit_type 2/3/4）：整段文字當 system 出題。

純 RAG（unit_type=1）請傳空的 system_supplement；與「逐字稿全文當唯一依據」不同。
"""

import json
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI

QUIZ_LLM_MODEL = "gpt-4o"
EMBEDDING_MODEL = "text-embedding-3-small"
RETRIEVAL_K = 5
DEFAULT_RETRIEVAL_QUERY = "課程重點概念"


def _normalize_quiz_llm_json(data: dict) -> dict:
    """統一 LLM 可能回傳的別名鍵為 quiz_reference_answer、quiz_hint。"""
    if "quiz_reference_answer" not in data and "reference_answer" in data:
        data["quiz_reference_answer"] = data.pop("reference_answer")
    if "quiz_reference_answer" not in data and "answer" in data:
        data["quiz_reference_answer"] = data.pop("answer")
    if "quiz_hint" not in data and "hint" in data:
        data["quiz_hint"] = data.pop("hint")
    return data


def _invoke_quiz_json_llm(client: OpenAI, messages: list) -> dict:
    response = client.chat.completions.create(
        model=QUIZ_LLM_MODEL,
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0.7,
    )
    raw = response.choices[0].message.content
    data = json.loads(raw or "{}")
    if not isinstance(data, dict):
        data = {}
    return _normalize_quiz_llm_json(data)


def generate_quiz_transcription_only(
    api_key: str,
    transcription: str,
    user_instruction: str = "",
) -> dict:
    """
    無 FAISS 時：system = 整段 transcription；user = user_instruction（可空）。

    用於 Rag_Unit.unit_type 為 2／3／4。若兩段文字皆不含「json」，會在 user 末尾補一行以符合
    OpenAI json_object 對 messages 須含「json」的要求。

    回傳：quiz_content, quiz_hint, quiz_reference_answer。
    """
    if not api_key or not api_key.strip():
        raise ValueError("請傳入 llm_api_key")
    system_text = (transcription or "").strip()
    if not system_text:
        raise ValueError("請傳入 transcription（作為 system 訊息，必填）")

    user_content = user_instruction if user_instruction is not None else ""
    if "json" not in (system_text + user_content).lower():
        sep = "\n\n" if (user_content or "").strip() else ""
        user_content = (
            user_content
            + sep
            + "請以 JSON 物件格式回傳 quiz_content、quiz_hint、quiz_reference_answer（鍵名請沿用英文）。"
        )

    client = OpenAI(api_key=api_key)
    return _invoke_quiz_json_llm(
        client,
        [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_content},
        ],
    )


def _system_prompt_faiss_quiz(system_supplement: str) -> str:
    """組出 FAISS 路徑的 system 訊息；system_supplement 為空時不帶「出題補充」區塊。"""
    extra = (system_supplement or "").strip()
    supplement_block = ""
    if extra:
        supplement_block = f"""
                【出題補充／參考文字】（與下方檢索到的課程內容並陳時，仍以課程內容為出題依據）
                {extra}
                """
    return f"""
                你是一位教授，請給學生設計測驗題目：
                【出題規範】
                根據輸入的「課程內容」設計測驗題目。
                使用繁體中文 (Traditional Chinese) 出題與撰寫答案提示 (quiz_hint) 及參考答案 (quiz_reference_answer)。
                {supplement_block}
                【回傳格式】
                以 JSON 格式回傳：
                {{ "quiz_content": "題目內容", 
                "quiz_hint": "答案提示內容", 
                "quiz_reference_answer": "參考答案內容" }}
            """


def generate_quiz(
    zip_path: Path,
    api_key: str,
    system_supplement: str = "",
    user_instruction: str = "",
) -> dict:
    """
    有 FAISS RAG ZIP：解壓 → 載入向量庫 → 檢索 → GPT 出題。

    僅支援 POST /rag/tab/build-rag-zip 產出的 RAG ZIP。

    Args:
        system_supplement: 選填。併入 system 的補充說明（例如單元／Rag 表層文字）。
            **純 RAG（unit_type=1）應傳空字串**，題目只依向量檢索到的片段；此欄位**不是**
            「向量庫內建的逐字稿」，也不取代檢索內容。
        user_instruction: 選填；非空時置於 user「課程內容」段落之前。

    回傳：quiz_content, quiz_hint, quiz_reference_answer（API 層可再附加 transcription 等）。
    """
    if not api_key or not api_key.strip():
        raise ValueError("請傳入 llm_api_key")

    extract_folder = Path(tempfile.mkdtemp())
    try:
        if not zipfile.is_zipfile(zip_path):
            raise ValueError("無效的 ZIP 檔")

        zip_kw: dict = {}
        if sys.version_info >= (3, 11):
            zip_kw["metadata_encoding"] = "utf-8"
        with zipfile.ZipFile(zip_path, "r", **zip_kw) as z:
            z.extractall(extract_folder)

        db_folder = None
        for root, _dirs, files in os.walk(extract_folder):
            if "index.faiss" in files and "index.pkl" in files:
                db_folder = root
                break
        if not db_folder:
            raise ValueError(
                "此 API 僅支援 RAG ZIP（由 POST /rag/tab/build-rag-zip 產出），請上傳含 FAISS 向量庫的 ZIP"
            )

        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=api_key)
        vectorstore = FAISS.load_local(
            db_folder, embeddings, allow_dangerous_deserialization=True
        )

        retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K})
        docs = retriever.invoke(DEFAULT_RETRIEVAL_QUERY)
        context_text = "\n\n".join(d.page_content for d in docs)

        system_prompt = _system_prompt_faiss_quiz(system_supplement)
        user_prompt = f"課程內容：\n{context_text}"
        extra = (user_instruction or "").strip()
        if extra:
            user_prompt = f"{extra}\n\n{user_prompt}"

        client = OpenAI(api_key=api_key)
        return _invoke_quiz_json_llm(
            client,
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
    finally:
        shutil.rmtree(extract_folder, ignore_errors=True)
