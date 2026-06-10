"""Bank 出題／批改之單元內容存取與題目產生 glue（bank 與 quiz 共用）。

抽自原 `routers.bank.group_helpers` 與 `routers.quiz.helpers` 之重複邏輯；**僅內部重構，不改任何
API input/output**。涵蓋：
- 讀取出題／批改所需之 Bank_Unit 欄位。
- 由 Bank_Unit.upload_file_name 解析 RAG ZIP 的 page_id。
- 呼叫 bank 出題 LLM，回傳一題的 question_* 欄位（不寫 DB）。
- 批改前置：依 unit_type 決定逐字稿／RAG ZIP，建立 work_dir（錯誤以 BankAnswerSetupError 拋出，由呼叫端轉成 JSONResponse）。

此模組屬 bank 家族（與 rag／exam 無關）；quiz 因搭配 bank、重用 bank 管線，故一併取用。
"""

import shutil
import tempfile
import zipfile
from pathlib import Path

from fastapi import HTTPException

from services.bank_generation import (
    format_bank_quiz_history_prompt_for_llm,
    generate_bank_quiz,
    generate_bank_quiz_transcript_only,
)
from services.bank_answering import cleanup_answer_workspace
from utils.bank_course import (
    execute_with_course_id_fallback,
    select_without_course_id_if_needed,
)
from utils.bank_storage import get_zip_path
from utils.bank_stem import transcript_from_row
from utils.fs import safe_unlink

from .helpers import (
    BANK_UNIT_TYPE_MP3,
    BANK_UNIT_TYPE_TEXT,
    BANK_UNIT_TYPE_YOUTUBE,
)

# unit_type 2／3／4 走逐字稿純 LLM；其餘（1）走 RAG ZIP 向量檢索。
TRANSCRIPT_UNIT_TYPES = (BANK_UNIT_TYPE_TEXT, BANK_UNIT_TYPE_MP3, BANK_UNIT_TYPE_YOUTUBE)


def fetch_bank_unit_for_llm(supabase, bank_unit_id: int, course_id: int) -> dict | None:
    """取出題／批改所需之 Bank_Unit 欄位（不含 folder_combination，避免舊表 42703）。"""

    def build(with_course_filter: bool):
        cols = select_without_course_id_if_needed(
            "Bank_Unit",
            "bank_unit_id, bank_page_id, person_id, unit_name, unit_type, transcript, upload_file_name, course_id",
            with_course_filter,
        )
        q = (
            supabase.table("Bank_Unit")
            .select(cols)
            .eq("bank_unit_id", bank_unit_id)
            .eq("deleted", False)
        )
        if with_course_filter and course_id is not None:
            q = q.eq("course_id", course_id)
        return q.limit(1)

    sel = execute_with_course_id_fallback("Bank_Unit", build, course_id)
    return sel.data[0] if sel.data else None


def rag_zip_page_id_from_unit(unit_row: dict) -> str:
    """由 Bank_Unit.upload_file_name（{stem}_rag.zip）取出 rag ZIP 的 page_id（{stem}_rag）。"""
    rf = (unit_row.get("upload_file_name") or "").strip()
    if rf.lower().endswith(".zip"):
        rid = rf[:-4].strip()
        if rid and "/" not in rid and "\\" not in rid:
            return rid
    return ""


def generate_question_fields_from_bank_unit(
    supabase,
    *,
    bank_unit_id: int,
    bank_page_id_fallback: str,
    course_id: int,
    api_key: str,
    llm_model: str,
    qup: str,
    qsp: str,
    prior_items: list[dict],
    ask_history_body: str = "",
) -> dict:
    """呼叫 bank 出題 LLM 產生一題（不寫 DB）。

    回傳含 question_content／question_hint／question_answer_reference／question_reason／
    bank_unit_id／bank_page_id。LLM 失敗時 raise（由呼叫端轉 llm_error_json_response）；
    驗證錯誤 raise HTTPException。`prior_items` 為「已出過題目（勿重複）」題幹清單。
    """
    if int(bank_unit_id or 0) <= 0:
        raise HTTPException(status_code=400, detail="該題組對應的 bank_unit_id 無效")
    unit = fetch_bank_unit_for_llm(supabase, bank_unit_id, course_id)
    if not unit:
        raise HTTPException(status_code=404, detail=f"找不到 bank_unit_id={bank_unit_id} 的 Bank_Unit")
    bank_page_id = (unit.get("bank_page_id") or bank_page_id_fallback or "").strip()

    prompt_for_llm = format_bank_quiz_history_prompt_for_llm(prior_items)

    try:
        unit_type_val = int(unit.get("unit_type") or 0)
    except (TypeError, ValueError):
        unit_type_val = 0
    transcript_text = transcript_from_row(unit)

    path: Path | None = None
    try:
        if unit_type_val in TRANSCRIPT_UNIT_TYPES:
            if not transcript_text:
                raise HTTPException(
                    status_code=400,
                    detail="單元類型 2／3／4 需有逐字稿：請於 Bank_Unit 設定 transcript，或經 build-zip 寫入",
                )
            result = generate_bank_quiz_transcript_only(
                api_key=api_key,
                transcript=transcript_text,
                quiz_user_prompt_text=qup,
                quiz_history_list_prompt_text=prompt_for_llm,
                ask_history_body=ask_history_body,
                llm_model=llm_model,
                quiz_system_prompt_text=qsp,
            )
        else:
            rag_zip_page_id = rag_zip_page_id_from_unit(unit)
            if not rag_zip_page_id:
                raise HTTPException(
                    status_code=400,
                    detail="該單元尚無 RAG ZIP（upload_file_name 為空）；請先執行 build-zip 建立向量庫",
                )
            path = get_zip_path(rag_zip_page_id)
            if not path or not path.exists():
                raise HTTPException(
                    status_code=404,
                    detail=f"找不到 RAG ZIP（page_id={rag_zip_page_id}），請確認該單元已 build-zip",
                )
            result = generate_bank_quiz(
                path,
                api_key=api_key,
                quiz_user_prompt_text=qup,
                quiz_history_list_prompt_text=prompt_for_llm,
                ask_history_body=ask_history_body,
                llm_model=llm_model,
                quiz_system_prompt_text=qsp,
            )

        return {
            "question_content": (result.get("quiz_content") or "").strip(),
            "question_hint": (result.get("quiz_hint") or "").strip(),
            "question_answer_reference": (result.get("quiz_answer_reference") or "").strip(),
            "question_reason": (result.get("question_reason") or "").strip(),
            "bank_unit_id": bank_unit_id,
            "bank_page_id": bank_page_id,
        }
    finally:
        if path is not None:
            safe_unlink(path)


class BankAnswerSetupError(Exception):
    """批改前置失敗；呼叫端據 status_code／detail 轉成 JSONResponse（與原行為一致）。"""

    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


def prepare_bank_answer_workspace(
    supabase,
    *,
    bank_unit_id: int,
    course_id: int,
    prefix: str,
) -> tuple[int, str | None, Path]:
    """批改前置：取單元、決定逐字稿／RAG ZIP，並建立 work_dir（zip 模式已含 ref.zip）。

    回傳 (unit_type, transcript_answer, work_dir)。`transcript_answer` 非空時走逐字稿純 LLM 批改。
    失敗時 raise BankAnswerSetupError（status_code 與 detail 同原 JSONResponse）。
    `prefix` 為暫存資料夾前綴（如 "myquizai_bank_answer" / "myquizai_quiz_answer"）。
    """
    unit = fetch_bank_unit_for_llm(supabase, bank_unit_id, course_id) if bank_unit_id > 0 else None
    try:
        unit_type = int(unit.get("unit_type") or 0) if unit else 0
    except (TypeError, ValueError):
        unit_type = 0
    transcript_text = transcript_from_row(unit) if unit else ""

    transcript_answer: str | None = None
    if unit_type in TRANSCRIPT_UNIT_TYPES:
        if not transcript_text.strip():
            raise BankAnswerSetupError(400, "批改用 transcript 未設定：請於 Bank_Unit 設定 transcript（單元 2／3／4）")
        transcript_answer = transcript_text
        work_dir = Path(tempfile.mkdtemp(prefix=f"{prefix}_tx_"))
        return unit_type, transcript_answer, work_dir

    rag_zip_page_id = rag_zip_page_id_from_unit(unit or {})
    rag_zip_path = get_zip_path(rag_zip_page_id) if rag_zip_page_id else None
    if not rag_zip_path or not rag_zip_path.exists():
        raise BankAnswerSetupError(404, f"找不到 RAG ZIP（page_id={rag_zip_page_id}），請確認該單元已 build-zip")
    work_dir = Path(tempfile.mkdtemp(prefix=f"{prefix}_"))
    zip_source_path = work_dir / "ref.zip"
    extract_folder = work_dir / "extract"
    extract_folder.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy(rag_zip_path, zip_source_path)
        if not zipfile.is_zipfile(zip_source_path):
            cleanup_answer_workspace(work_dir)
            raise BankAnswerSetupError(400, "無效的 ZIP 檔")
    except BankAnswerSetupError:
        raise
    except Exception as e:
        cleanup_answer_workspace(work_dir)
        raise BankAnswerSetupError(500, str(e))
    finally:
        safe_unlink(rag_zip_path)
    return unit_type, transcript_answer, work_dir


__all__ = [
    "TRANSCRIPT_UNIT_TYPES",
    "BankAnswerSetupError",
    "fetch_bank_unit_for_llm",
    "rag_zip_page_id_from_unit",
    "generate_question_fields_from_bank_unit",
    "prepare_bank_answer_workspace",
]
