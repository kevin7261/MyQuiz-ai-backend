"""Exam 端點 course_id 驗證。"""

from typing import Any

from fastapi import HTTPException

from utils.supabase import get_supabase


def require_exam_row(
    *,
    course_id: int,
    exam_id: int = 0,
    exam_tab_id: str = "",
    person_id: str | None = None,
) -> dict[str, Any]:
    """依 exam_id 或 exam_tab_id + course_id 載入 Exam（deleted=false）。person_id 非空時須一致。"""
    supabase = get_supabase()
    eid = int(exam_id or 0)
    et = (exam_tab_id or "").strip()
    if eid > 0:
        q = (
            supabase.table("Exam")
            .select("exam_id, exam_tab_id, tab_name, person_id, course_id, local, deleted")
            .eq("exam_id", eid)
            .eq("course_id", course_id)
            .eq("deleted", False)
            .limit(1)
        )
    elif et and et != "0":
        q = (
            supabase.table("Exam")
            .select("exam_id, exam_tab_id, tab_name, person_id, course_id, local, deleted")
            .eq("exam_tab_id", et)
            .eq("course_id", course_id)
            .eq("deleted", False)
            .limit(1)
        )
    else:
        raise HTTPException(status_code=400, detail="請傳入 exam_id 或 exam_tab_id")
    if person_id is not None:
        q = q.eq("person_id", person_id)
    sel = q.execute()
    if not sel.data:
        raise HTTPException(status_code=404, detail="找不到對應的 Exam 資料，或不屬於此 course_id")
    return sel.data[0]
