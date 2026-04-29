"""
測驗題 LLM 生成：路由依 Rag_Unit.unit_type 擇一呼叫。

- generate_quiz：有 FAISS RAG ZIP。檢索片段 → user「課程內容」；system 僅出題規範（Markdown）。
- generate_quiz_transcription_only：無向量庫（典型 unit_type 2/3/4）。與 RAG 相同：system 僅出題規範；逐字稿置於 user「課程內容」區塊。

重要（維持行為時請留意）：
- unit_type=1：僅依向量檢索片段出題；2/3/4 依整份逐字稿（格式與 RAG 的 user 課程區塊對齊）。
- 兩路徑皆 response_format=json_object；system 範本含「JSON」字樣，無需再補尾段。

檔案結構（由上而下）：
1. 模型／檢索常數
2. LLM Prompt 範本（system 共用；user 分 RAG 片段／逐字稿全文）
3. 回傳正規化與共用 API 呼叫
4. 公開函式（exam／grade 路由動態 import）
"""

import json
import os
import shutil
import sys
import tempfile
import textwrap
import zipfile
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI

# ---------------------------------------------------------------------------
# 模型與檢索常數
# ---------------------------------------------------------------------------
# 下列數值會直接影響：API 成本、embedding 維度相容、檢索到的 chunk 數與內容。
# 若調整 DEFAULT_RETRIEVAL_QUERY，歷史產出的題目可能無法與舊 log 對照。

QUIZ_LLM_MODEL = "gpt-4o"
EMBEDDING_MODEL = "text-embedding-3-small"
RETRIEVAL_K = 5
# 出題時「固定查詢句」送進向量檢索；與批改 run_grade_job（以題幹當查詢）刻意不同。
DEFAULT_RETRIEVAL_QUERY = "課程重點概念"


# ---------------------------------------------------------------------------
# LLM Prompt 範本（集中維護：語氣、Markdown、JSON 鍵名請只改本區）
# ---------------------------------------------------------------------------
# SYSTEM_PROMPT_FAISS_QUIZ：僅角色與規範，不含課文；課文一律在 user（RAG：USER_PROMPT_FAISS_COURSE；逐字稿：USER_PROMPT_TRANSCRIPTION_COURSE）。
# USER_PROMPT_FAISS_COURSE：佔位符 {context_text} = 向量檢索多段 chunk 以空行串接。
# USER_PROMPT_TRANSCRIPTION_COURSE：佔位符 {context_text} = 整份逐字稿／課程全文（與上者區塊結構一致）。

SYSTEM_PROMPT_FAISS_QUIZ = textwrap.dedent("""
    # 角色

    你是一位教授，請為學生設計測驗題目。

    ## 出題規範

    - 根據使用者訊息中的 **課程內容** 區塊設計測驗題目。
    - 題目、`quiz_hint`（答案提示）、`quiz_reference_answer`（參考答案）一律使用 **繁體中文（Traditional Chinese）**。

    ## 回傳格式（JSON）

    請回傳一個 JSON 物件，鍵名固定為（英文）：

    - `quiz_content`：題目內容
    - `quiz_hint`：答案提示內容
    - `quiz_reference_answer`：參考答案內容
    """).strip()

USER_PROMPT_FAISS_COURSE = textwrap.dedent("""
    ## 課程內容

    下列為自課程向量庫檢索之片段（出題**僅能**依此與上文條件為據）：

    {context_text}
    """).strip()

USER_PROMPT_TRANSCRIPTION_COURSE = textwrap.dedent("""
    ## 課程內容

    下列為課程逐字稿／全文（出題**僅能**依此與上文條件為據）：

    {context_text}
    """).strip()


# ---------------------------------------------------------------------------
# LLM 回傳正規化與呼叫（非 prompt 文字）
# ---------------------------------------------------------------------------

def _normalize_quiz_llm_json(data: dict) -> dict:
    """
    將 LLM 偶發的別名鍵對齊為 API 約定鍵。

    背景：模型有時回 reference_answer／answer／hint，前端與 DB 已統一用
    quiz_reference_answer、quiz_hint；此處就地改 dict，避免路由層重複處理。
    """
    if "quiz_reference_answer" not in data and "reference_answer" in data:
        data["quiz_reference_answer"] = data.pop("reference_answer")
    if "quiz_reference_answer" not in data and "answer" in data:
        data["quiz_reference_answer"] = data.pop("answer")
    if "quiz_hint" not in data and "hint" in data:
        data["quiz_hint"] = data.pop("hint")
    return data


def _invoke_quiz_json_llm(client: OpenAI, messages: list) -> dict:
    """
    呼叫 GPT 並解析 JSON 物件回應。

    - temperature=0.7：出題需要一定變化；批改在 services/grading 用 0.3，兩者勿混用語意。
    - 若 parse 結果非 dict（極少見），當成空 dict 再 normalize，避免呼叫端 KeyError。
    """
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


# ---------------------------------------------------------------------------
# 公開 API（由 routers/exam.py、routers/grade.py 動態 import）
# ---------------------------------------------------------------------------

def generate_quiz_transcription_only(
    api_key: str,
    transcription: str,
    user_instruction: str = "",
) -> dict:
    """
    無 FAISS：與 generate_quiz 相同訊息結構——system 為出題規範；逐字稿置於 user「課程內容」。

    路由：Rag_Unit.unit_type 為 2／3／4 時使用（與 generate_quiz 互斥）。
    回傳：quiz_content, quiz_hint, quiz_reference_answer。

    Args:
        api_key: OpenAI API Key（與 embeddings 無關，本路徑不建向量）。
        transcription: 課程全文或逐字稿，填入 user 課程內容區塊；不可空。
        user_instruction: 出題補充（如 quiz_user_prompt_text）；非空時置於課程內容區塊前，與 RAG 路徑相同以 --- 分隔。
    """
    if not api_key or not api_key.strip():
        raise ValueError("請傳入 llm_api_key")
    context_text = (transcription or "").strip()
    if not context_text:
        raise ValueError("請傳入 transcription（課程內容區塊，必填）")

    user_prompt = USER_PROMPT_TRANSCRIPTION_COURSE.format(context_text=context_text)
    prefix_md = (user_instruction or "").strip()
    if prefix_md:
        user_prompt = f"{prefix_md}\n\n---\n\n{user_prompt}"

    client = OpenAI(api_key=api_key)
    return _invoke_quiz_json_llm(
        client,
        [
            {"role": "system", "content": SYSTEM_PROMPT_FAISS_QUIZ},
            {"role": "user", "content": user_prompt},
        ],
    )


def generate_quiz(
    zip_path: Path,
    api_key: str,
    user_instruction: str = "",
) -> dict:
    """
    有 FAISS RAG ZIP：解壓 → 載入向量庫 → 檢索 → 組 Markdown user → LLM。

    zip_path 須為 POST /rag/tab/build-rag-zip 產物。user_instruction 非空時置於「課程內容」區塊前。
    回傳：quiz_content, quiz_hint, quiz_reference_answer。

    Args:
        zip_path: 本機路徑，指向已下載之 RAG ZIP（內含 index.faiss / index.pkl）。
        api_key: 同時用於 OpenAIEmbeddings 與 Chat Completions。
        user_instruction: 選填；exam 路由會傳入 API 參數＋quiz_user_prompt 的 Markdown 前綴。
    """
    if not api_key or not api_key.strip():
        raise ValueError("請傳入 llm_api_key")

    extract_folder = Path(tempfile.mkdtemp())
    try:
        if not zipfile.is_zipfile(zip_path):
            raise ValueError("無效的 ZIP 檔")

        # Python 3.11+：ZIP 內 UTF-8 檔名需 metadata_encoding，否則解壓可能觸發 ascii 編碼錯誤。
        zip_kw: dict = {}
        if sys.version_info >= (3, 11):
            zip_kw["metadata_encoding"] = "utf-8"
        with zipfile.ZipFile(zip_path, "r", **zip_kw) as z:
            z.extractall(extract_folder)

        # 僅接受 LangChain FAISS.save_local 目錄結構，與自建講義 ZIP 區隔。
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
        # 本機自建 ZIP：須允許 pickle 反序列化（LangChain 預設關閉）。
        vectorstore = FAISS.load_local(
            db_folder, embeddings, allow_dangerous_deserialization=True
        )

        retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K})
        docs = retriever.invoke(DEFAULT_RETRIEVAL_QUERY)
        # page_content：每段 chunk 純文字；多段之間空行分隔，便於模型閱讀邊界。
        context_text = "\n\n".join(d.page_content for d in docs)

        user_prompt = USER_PROMPT_FAISS_COURSE.format(context_text=context_text)
        prefix_md = (user_instruction or "").strip()
        if prefix_md:
            # --- 分隔：前段為「請求／出題條件」，後段為「檢索課文」，避免模型混淆兩類來源。
            user_prompt = f"{prefix_md}\n\n---\n\n{user_prompt}"

        client = OpenAI(api_key=api_key)
        return _invoke_quiz_json_llm(
            client,
            [
                {"role": "system", "content": SYSTEM_PROMPT_FAISS_QUIZ},
                {"role": "user", "content": user_prompt},
            ],
        )
    finally:
        # 暫存目錄必清，避免磁碟堆積與路徑外洩。
        shutil.rmtree(extract_folder, ignore_errors=True)
