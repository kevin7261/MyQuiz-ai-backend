"""
從 RAG ZIP（FAISS）載入、檢索後以 LLM 生成測驗內容（generate_quiz），
或無向量庫時以逐字稿為 system 純 LLM 生成（generate_quiz_transcription_only）。
"""

# 引入 json 用於解析 LLM 回傳
import json
# 引入 os 用於 os.walk
import os
# 引入 shutil 用於刪除暫存目錄
import shutil
# 引入 sys 用於 version_info 判斷
import sys
# 引入 tempfile 建立暫存目錄
import tempfile
# 引入 zipfile 讀取 ZIP
import zipfile
# 引入 Path 用於路徑操作
from pathlib import Path

# LangChain OpenAI Embeddings
from langchain_openai import OpenAIEmbeddings
# FAISS 向量庫
from langchain_community.vectorstores import FAISS
# OpenAI 客戶端
from openai import OpenAI

from utils.course_name_utils import get_course_name_for_prompt


def _normalize_quiz_llm_json(data: dict) -> dict:
    """統一 quiz_reference_answer／quiz_hint 鍵名（與 generate_quiz 一致）。"""
    if "quiz_reference_answer" not in data and "reference_answer" in data:
        data["quiz_reference_answer"] = data.pop("reference_answer")
    if "quiz_reference_answer" not in data and "answer" in data:
        data["quiz_reference_answer"] = data.pop("answer")
    if "quiz_hint" not in data and "hint" in data:
        data["quiz_hint"] = data.pop("hint")
    return data


def generate_quiz_transcription_only(
    api_key: str,
    transcription: str,
    user_instruction: str = "",
) -> dict:
    """
    無 RAG ZIP／向量庫時，以 LLM 純生成題目（Rag_Unit.unit_type 為 2／3／4 時）。
    **system** = transcription；**user** = user_instruction（對應 quiz_user_prompt_text，可空字串）。
    若 system 與 user 皆未含「json」字樣，會於 user 末尾加上一行 JSON 輸出說明（OpenAI API 要求）。
    回傳 {"quiz_content", "quiz_hint", "quiz_reference_answer"}。
    """
    if not api_key or not api_key.strip():
        raise ValueError("請傳入 llm_api_key")
    ts = (transcription or "").strip()
    if not ts:
        raise ValueError("請傳入 transcription（作為 system 訊息，必填）")

    client = OpenAI(api_key=api_key)
    user_content = user_instruction if user_instruction is not None else ""
    # OpenAI：使用 response_format json_object 時，messages 須出現「json」字樣（不分大小寫）
    if "json" not in (ts + user_content).lower():
        sep = "\n\n" if (user_content or "").strip() else ""
        user_content = (
            user_content
            + sep
            + "請以 JSON 物件格式回傳 quiz_content、quiz_hint、quiz_reference_answer（鍵名請沿用英文）。"
        )
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": ts},
            {"role": "user", "content": user_content},
        ],
        response_format={"type": "json_object"},
        temperature=0.7,
    )
    content = response.choices[0].message.content
    data = json.loads(content or "{}")
    if not isinstance(data, dict):
        data = {}
    _normalize_quiz_llm_json(data)
    return data


def generate_quiz(
    zip_path: Path,
    api_key: str,
    transcription: str,
    user_instruction: str = "",
) -> dict:
    """
    從現成 RAG ZIP（含 FAISS 向量庫）解壓 → 載入向量庫 → 檢索 → 呼叫 GPT-4o 出題。
    僅支援由 POST /rag/tab/build-rag-zip 產出的 RAG ZIP，不支援一般講義 ZIP。
    transcription 為必填，由後端自 **Rag.transcription**／**Rag_Unit.transcription** 解析（非純 API body）。
    user_instruction 為選填；非空時置於「課程內容」user 訊息之前。POST /exam/tab/quiz/llm-generate 會傳入含 **全部 body 欄位** 之標示文字（含 exam_quiz_id、rag_unit_id、quiz_user_prompt_text 等）；RAG 路徑則多為出題補充（quiz_user_prompt_text）。
    回傳 {"quiz_content", "quiz_hint", "quiz_reference_answer"}；API 層可再加上 transcription 等。
    """
    if not api_key or not api_key.strip():
        raise ValueError("請傳入 llm_api_key")
    if not transcription or not transcription.strip():
        raise ValueError("請傳入 transcription（出題用補充／逐字稿文字，必填）")

    extract_folder = Path(tempfile.mkdtemp())  # 建立暫存目錄
    try:
        if not zipfile.is_zipfile(zip_path):
            raise ValueError("無效的 ZIP 檔")

        # 使用 UTF-8 解壓，避免非 ASCII 檔名或路徑造成 'ascii' codec can't encode 錯誤（Python 3.11+）
        zip_kw: dict = {}
        if sys.version_info >= (3, 11):
            zip_kw["metadata_encoding"] = "utf-8"  # Python 3.11+ 支援 UTF-8 檔名
        with zipfile.ZipFile(zip_path, "r", **zip_kw) as z:
            z.extractall(extract_folder)

        # 尋找含 index.faiss 與 index.pkl 的目錄（FAISS 向量庫）
        db_folder = None
        for root, _dirs, files in os.walk(extract_folder):
            if "index.faiss" in files and "index.pkl" in files:
                db_folder = root
                break
        if not db_folder:
            raise ValueError("此 API 僅支援 RAG ZIP（由 POST /rag/tab/build-rag-zip 產出），請上傳含 FAISS 向量庫的 ZIP")

        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)
        vectorstore = FAISS.load_local(  # 載入 FAISS 向量庫
            db_folder, embeddings, allow_dangerous_deserialization=True
        )

        # 檢索查詢：課程重點概念與操作步驟
        query = "課程重點概念"
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        docs = retriever.invoke(query)
        context_text = "\n\n".join([d.page_content for d in docs])

        course_name = get_course_name_for_prompt()
        final_system_prompt = f"""
            你是一個「{course_name}」課程的教授，請給學生設計測驗題目：
            【出題規範】
            請根據輸入的「課程內容」設計測驗題目。
            請使用繁體中文 (Traditional Chinese) 出題與撰寫答案提示 (quiz_hint) 及參考答案 (quiz_reference_answer)。
            **{transcription}**
            【回傳格式】
            請以 JSON 格式回傳：
            {{ "quiz_content": "題目內容", 
              "quiz_hint": "答案提示內容", 
              "quiz_reference_answer": "參考答案內容" }}
        """
        user_prompt_text = f"課程內容：\n{context_text}"
        extra_ui = (user_instruction or "").strip()
        if extra_ui:
            user_prompt_text = f"{extra_ui}\n\n{user_prompt_text}"

        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": final_system_prompt},
                {"role": "user", "content": user_prompt_text},
            ],
            response_format={"type": "json_object"},
            temperature=0.7,
        )

        content = response.choices[0].message.content
        data = json.loads(content or "{}")  # 解析 JSON
        if not isinstance(data, dict):
            data = {}
        _normalize_quiz_llm_json(data)
        return data
    finally:
        shutil.rmtree(extract_folder, ignore_errors=True)  # 清理暫存
