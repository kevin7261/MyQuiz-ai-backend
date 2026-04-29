"""
測驗題 LLM 生成：路由依 Rag_Unit.unit_type 擇一呼叫。

- generate_quiz：有 FAISS RAG ZIP。檢索片段 → user「課程內容」；system 僅出題規範（Markdown）。
- system／user 訊息與使用者 **出題補充** 皆為 **Markdown**（課程原文以 fenced code block 包覆）。

重要（維持行為時請留意）：
- unit_type=1：僅依向量檢索片段出題；2/3/4 依整份逐字稿（格式與 RAG 的 user 課程區塊對齊）。
- 兩路徑皆 response_format=json_object；system 範本含「JSON」字樣，無需再補尾段。

檔案結構（由上而下）：
1. 模型／檢索常數
2. **LLM Prompt 全文**（見「出題 Prompt」區：system → user；`{quiz_user_prompt_text}` 與版面皆在模板內）
3. LLM 回傳正規化與 API 呼叫（僅 `_context_as_markdown_fenced` 將課文套入 `{context_md}`）
4. 公開函式（exam／grade 路由動態 import）
"""

import json
import os
import re
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


# -----------------------------------------------------------------------------
# LLM 出題 Prompt（閱讀順序：[system] → [user]，user 以 .format(quiz_user_prompt_text=…, context_md=…) 填入）
# -----------------------------------------------------------------------------
# Chat messages：
#   role=system … SYSTEM_PROMPT_FAISS_QUIZ
#   role=user … USER_PROMPT_FAISS_COURSE 或 USER_PROMPT_TRANSCRIPTION_COURSE；
#   其中 `{context_md}` 僅由向量檢索／逐字稿經 _context_as_markdown_fenced 產生，其餘不重組字串。
#

SYSTEM_PROMPT_FAISS_QUIZ = textwrap.dedent("""
    # 角色

    你是一位教授，請為學生設計測驗題目。

    ## 訊息格式

    - 系統與使用者訊息皆為 **Markdown**（標題、清單、粗體、水平線、`---`、課程原文之 fenced code block 等）。
    - **課程內容** 以 code fence（```text …```）包住逐字／檢索原文；區段內唯純引用，勿將標記語法本身當成教學內容。
    - 將 `---` 視為區段分隔。
    - **出題補充** 區塊（`## 出題補充`）以下若無實質文字（僅留白或空字串），請**完全忽略**該節，僅依 **課程內容** 出題。
    - 若出題補充有文字，則同時斟酌 **出題補充** 與 **課程內容** 出題。

    ## 出題規範

    - `quiz_content`、`quiz_hint`、`quiz_answer_reference` 之字串值皆為 **Markdown**（段落、清單、`**強調**` 等），並使用 **繁體中文（Traditional Chinese）**。

    ## 回傳格式（JSON）

    請回傳一個 JSON 物件，鍵名固定為（英文）：

    - `quiz_content`：題目（Markdown 字串）
    - `quiz_hint`：答案提示（Markdown 字串）
    - `quiz_answer_reference`：參考答案（Markdown 字串）
    """).strip()

USER_PROMPT_FAISS_COURSE = textwrap.dedent("""
    ## 出題補充

    {quiz_user_prompt_text}

    ---

    ## 課程內容

    下列為自課程向量庫檢索之片段（出題**僅能**依下列引用與上文條件為據）。

    {context_md}
    """).strip()

USER_PROMPT_TRANSCRIPTION_COURSE = textwrap.dedent("""
    ## 出題補充

    {quiz_user_prompt_text}

    ---

    ## 課程內容

    下列為課程逐字稿／全文（出題**僅能**依下列引用與上文條件為據）。

    {context_md}
    """).strip()


def _context_as_markdown_fenced(context_text: str) -> str:
    """產出 USER_PROMPT_* 占位 {context_md}：Markdown fenced block（標記為 text）；圍欄長度避開內文反引號。"""
    inner = (context_text or "").rstrip()
    max_run = 0
    for m in re.finditer(r"`+", inner):
        max_run = max(max_run, len(m.group(0)))
    n = max(3, max_run + 1)
    fence = "`" * n
    return f"{fence}text\n{inner}\n{fence}"


# ---------------------------------------------------------------------------
# LLM 回傳正規化與呼叫（非 prompt 文字）
# ---------------------------------------------------------------------------

def _normalize_quiz_llm_json(data: dict) -> dict:
    """
    將 LLM 偶發的別名鍵對齊為資料庫欄位鍵。

    主鍵：`quiz_answer_reference`。相容舊別名：`quiz_reference_answer`、`reference_answer`、`answer`；
    `quiz_hint` 相容 `hint`。
    """
    if "quiz_answer_reference" not in data:
        if "quiz_reference_answer" in data:
            data["quiz_answer_reference"] = data.pop("quiz_reference_answer")
        elif "reference_answer" in data:
            data["quiz_answer_reference"] = data.pop("reference_answer")
        elif "answer" in data:
            data["quiz_answer_reference"] = data.pop("answer")
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
    quiz_user_prompt_text: str = "",
) -> dict:
    """
    無 FAISS：與 generate_quiz 相同訊息結構——system 為出題規範；逐字稿置於 user「課程內容」。

    路由：Rag_Unit.unit_type 為 2／3／4 時使用（與 generate_quiz 互斥）。
    回傳：quiz_content, quiz_hint, quiz_answer_reference。

    Args:
        api_key: OpenAI API Key（與 embeddings 無關，本路徑不建向量）。
        transcription: 課程全文或逐字稿，填入 user 課程內容區塊；不可空。
        quiz_user_prompt_text: 填入 USER_PROMPT_* 之「出題補充」占位；空字串時依 system 指示略過該節。
    """
    if not api_key or not api_key.strip():
        raise ValueError("請傳入 llm_api_key")
    raw_tc = transcription if transcription is not None else ""
    if not raw_tc.strip():
        raise ValueError("請傳入 transcription（課程內容區塊，必填）")
    context_text = raw_tc
    context_md = _context_as_markdown_fenced(context_text)
    qup = (quiz_user_prompt_text or "").strip()

    user_prompt = USER_PROMPT_TRANSCRIPTION_COURSE.format(
        quiz_user_prompt_text=qup,
        context_md=context_md,
    )

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
    quiz_user_prompt_text: str = "",
) -> dict:
    """
    有 FAISS RAG ZIP：解壓 → 載入向量庫 → 檢索 → 組 Markdown user → LLM。

    zip_path 須為 POST /rag/tab/build-rag-zip 產物。`quiz_user_prompt_text` 見 USER_PROMPT_FAISS_COURSE。
    回傳：quiz_content, quiz_hint, quiz_answer_reference。

    Args:
        zip_path: 本機路徑，指向已下載之 RAG ZIP（內含 index.faiss / index.pkl）。
        api_key: 同時用於 OpenAIEmbeddings 與 Chat Completions。
        quiz_user_prompt_text: 填入 USER_PROMPT_FAISS_COURSE 之「出題補充」占位；空字串時依 system 指示略過該節。
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
        context_md = _context_as_markdown_fenced(context_text)
        qup = (quiz_user_prompt_text or "").strip()

        user_prompt = USER_PROMPT_FAISS_COURSE.format(
            quiz_user_prompt_text=qup,
            context_md=context_md,
        )

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
