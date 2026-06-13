"""
課程設定（Course_Setting）API 模組，掛載於 /rag。
- GET /rag/course-members：依 course_id 列出該課程所有使用者；須為有效登入使用者；必填 query course_id。
- POST /rag/course-members/add：新增一筆課程成員（person_id、name、user_type）；僅 user_type 1／2。
  User 表已有相同 college_id + person_id 時僅新增選課，否則先建立 User 再加入課程。
- POST /rag/course-members/add-batch：批次新增該課程學生（每筆 person_id、name；user_type 固定 3）；僅 user_type 1／2。
- PUT /rag/course-members/edit/{person_id}：編輯課程成員（name、user_type）；僅 user_type 1／2。
- PUT /rag/course-members/delete/{person_id}：自課程移除成員（User_Course_Relation deleted=true）；僅 user_type 1／2。
- GET /rag/person-analysis-user-prompt-text：取得個人分析指令（Course_Setting key=person_analysis_user_prompt_text）；須為有效登入使用者；必填 query course_id。
- PUT /rag/person-analysis-user-prompt-text：寫入 Course_Setting（依 course_id upsert）；僅 user_type 1／2。
- GET /rag/course-analysis-user-prompt-text：取得課程分析指令（Course_Setting key=course_analysis_user_prompt_text）；須為有效登入使用者；必填 query course_id。
- PUT /rag/course-analysis-user-prompt-text：寫入 Course_Setting（依 course_id upsert）；僅 user_type 1／2。

LLM API Key 亦存於 Course_Setting（rag-api-key／exam-api-key）；見 GET/PUT /v1/rag/llm-api-key、/v1/rag/llm-model、/v1/exam/llm-api-key。
"""

from typing import Annotated, Optional

from fastapi import APIRouter, Body, HTTPException, Path
from pydantic import BaseModel, Field

from dependencies.course_id import CourseId
from dependencies.person_id import CurrentUser, PersonId
from utils.course_setting import (
    COURSE_SETTING_COURSE_ANALYSIS_USER_PROMPT_TEXT_KEY,
    COURSE_SETTING_PERSON_ANALYSIS_USER_PROMPT_TEXT_KEY,
    fetch_course_setting_text,
    upsert_course_setting_and_get_row,
)
from utils.db_schema import (
    ACTIVE_DELETED_FILTER,
    COLLEGE_TABLE,
    COURSE_TABLE,
    USER_COURSE_RELATION_TABLE,
    USER_TABLE,
)
from utils.openapi import openapi_body
from utils.supabase import get_supabase
from utils.taipei_time import now_taipei_iso

DEFAULT_NEW_MEMBER_PASSWORD = "0000"

router = APIRouter(prefix="/rag", tags=["rag"])


def _user_type_for_active_person(person_id: str, course_id: int) -> Optional[int]:
    """依 person_id、course_id 查 User_Course_Relation.user_type；無列或非有效帳號時回傳 None。"""
    pid = (person_id or "").strip()
    if not pid:
        return None
    try:
        supabase = get_supabase()
        resp = (
            supabase.table(USER_COURSE_RELATION_TABLE)
            .select("user_type")
            .eq("person_id", pid)
            .eq("course_id", course_id)
            .or_(ACTIVE_DELETED_FILTER)
            .order("course_user_id")
            .limit(1)
            .execute()
        )
        if not resp.data or len(resp.data) == 0:
            return None
        ut = resp.data[0].get("user_type")
        if ut is None:
            return None
        return int(ut)
    except Exception:
        return None


def _require_active_person(person_id: str, college_id: Optional[object] = None) -> None:
    """person_id 須對應有效 User（未刪除）。身分 = person_id + college_id：
    帶 college_id 時限定為「該學校」的有效帳號（跨校同 person_id 只認登入學校那一筆）。"""
    pid = (person_id or "").strip()
    if not pid:
        raise HTTPException(status_code=404, detail="找不到該使用者")
    cid = str(college_id or "").strip()
    try:
        supabase = get_supabase()
        query = (
            supabase.table(USER_TABLE)
            .select("user_id")
            .eq("person_id", pid)
        )
        if cid:
            query = query.eq("college_id", cid)
        resp = query.or_(ACTIVE_DELETED_FILTER).limit(1).execute()
        if not resp.data:
            raise HTTPException(status_code=404, detail="找不到該使用者")
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=404, detail="找不到該使用者")


def _require_developer_or_manager_for_course_setting_write(
    person_id: str, course_id: int
) -> None:
    """變更課程設定：僅開發者（1）或管理者（2）。"""
    ut = _user_type_for_active_person(person_id, course_id)
    if ut is None:
        raise HTTPException(status_code=404, detail="找不到該使用者")
    if ut not in (1, 2):
        raise HTTPException(status_code=403, detail="僅開發者或管理者可變更課程設定")


# 供 answer／exam api_key 端點沿用之別名
_require_developer_or_manager_for_analysis_prompt_write = (
    _require_developer_or_manager_for_course_setting_write
)
_upsert_setting_and_get_row = upsert_course_setting_and_get_row


class PersonAnalysisUserPromptTextResponse(BaseModel):
    """GET/PUT /rag/person-analysis-user-prompt-text 回應（資料來自 Course_Setting）。"""

    course_id: Optional[int] = None
    person_analysis_user_prompt_text: Optional[str] = None


class PutPersonAnalysisUserPromptTextRequest(BaseModel):
    """PUT /rag/person-analysis-user-prompt-text 的 body。"""

    person_analysis_user_prompt_text: str = Field(..., description="個人分析使用者 Prompt 文字")


class CourseAnalysisUserPromptTextResponse(BaseModel):
    """GET/PUT /rag/course-analysis-user-prompt-text 回應（資料來自 Course_Setting）。"""

    course_id: Optional[int] = None
    course_analysis_user_prompt_text: Optional[str] = None


class PutCourseAnalysisUserPromptTextRequest(BaseModel):
    """PUT /rag/course-analysis-user-prompt-text 的 body。"""

    course_analysis_user_prompt_text: str = Field(..., description="課程分析使用者 Prompt 文字")


class CourseMemberItem(BaseModel):
    """GET /rag/course-members 單筆成員。"""

    course_user_id: int
    user_id: int
    person_id: Optional[str] = None
    name: Optional[str] = None
    password: Optional[str] = None
    user_type: Optional[int] = None
    college_id: Optional[int] = None


class ListCourseMembersResponse(BaseModel):
    """GET /rag/course-members 回應。"""

    course_id: int
    members: list[CourseMemberItem]
    count: int


class AddCourseMemberRequest(BaseModel):
    """POST /rag/course-members/add 的 body。"""

    person_id: str = Field(..., description="登入帳號（id）")
    name: str = Field(..., description="姓名")
    user_type: int = Field(..., description="身份：1 開發者、2 管理者、3 學生")


class EditCourseMemberRequest(BaseModel):
    """PUT /rag/course-members/edit/{person_id} 的 body。"""

    name: str = Field(..., description="姓名")
    user_type: int = Field(..., description="身份：1 開發者、2 管理者、3 學生")


class BatchCourseMemberRow(BaseModel):
    """批次新增單筆：僅 person_id、name；user_type 固定為 3（學生）。"""

    person_id: str
    name: str


class BatchCourseMemberFailure(BaseModel):
    person_id: str
    detail: str


class BatchAddCourseMembersResponse(BaseModel):
    """POST /rag/course-members/add-batch 回應。"""

    created: list[CourseMemberItem]
    failed: list[BatchCourseMemberFailure]
    created_count: int
    failed_count: int


def _validate_course_member_fields(
    target_person_id: str,
    name: str,
    user_type: int,
) -> tuple[str, str]:
    pid = (target_person_id or "").strip()
    display_name = (name or "").strip()
    if not pid:
        raise HTTPException(status_code=400, detail="person_id 不可為空")
    if not display_name:
        raise HTTPException(status_code=400, detail="name 不可為空")
    if user_type not in (1, 2, 3):
        raise HTTPException(status_code=400, detail="user_type 須為 1、2 或 3")
    return pid, display_name


def _get_active_course_member_relation(
    supabase,
    course_id: int,
    target_person_id: str,
) -> dict:
    pid = (target_person_id or "").strip()
    resp = (
        supabase.table(USER_COURSE_RELATION_TABLE)
        .select("course_user_id, user_id, person_id, name, user_type, college_id")
        .eq("person_id", pid)
        .eq("course_id", course_id)
        .or_(ACTIVE_DELETED_FILTER)
        .limit(1)
        .execute()
    )
    if not resp.data:
        raise HTTPException(status_code=404, detail="找不到該課程成員")
    return resp.data[0]


def _fetch_user_row_for_member(supabase, user_id: int) -> dict | None:
    resp = (
        supabase.table(USER_TABLE)
        .select("user_id, person_id, name, password, college_id")
        .eq("user_id", user_id)
        .or_(ACTIVE_DELETED_FILTER)
        .limit(1)
        .execute()
    )
    return resp.data[0] if resp.data else None


def _course_member_item_from_rows(rel: dict, user_row: dict | None) -> CourseMemberItem:
    uid = int(rel["user_id"])
    college_raw = rel.get("college_id")
    college_id = int(college_raw) if college_raw is not None and int(college_raw or 0) != 0 else None
    if user_row:
        name = (user_row.get("name") or "").strip() or (rel.get("name") or "").strip() or None
        pid = (user_row.get("person_id") or rel.get("person_id") or "").strip() or None
        password = user_row.get("password")
    else:
        name = (rel.get("name") or "").strip() or None
        pid = (rel.get("person_id") or "").strip() or None
        password = None
    return CourseMemberItem(
        course_user_id=int(rel["course_user_id"]),
        user_id=uid,
        person_id=pid,
        name=name,
        password=password,
        user_type=rel.get("user_type"),
        college_id=college_id,
    )


def _fetch_course_college(supabase, course_id: int) -> tuple[int, str | None]:
    course_resp = (
        supabase.table(COURSE_TABLE)
        .select("course_id, college_id")
        .eq("course_id", course_id)
        .or_(ACTIVE_DELETED_FILTER)
        .limit(1)
        .execute()
    )
    if not course_resp.data:
        raise HTTPException(status_code=400, detail="找不到指定課程")
    college_id = int(course_resp.data[0].get("college_id") or 0)
    if not college_id:
        raise HTTPException(status_code=400, detail="課程未設定所屬學院")
    college_resp = (
        supabase.table(COLLEGE_TABLE)
        .select("college_id, college_name")
        .eq("college_id", college_id)
        .or_(ACTIVE_DELETED_FILTER)
        .limit(1)
        .execute()
    )
    if not college_resp.data:
        raise HTTPException(status_code=400, detail="找不到所屬學院")
    college_name = (college_resp.data[0].get("college_name") or "").strip() or None
    return college_id, college_name


def _add_course_member(
    supabase,
    *,
    course_id: int,
    target_person_id: str,
    name: str,
    user_type: int,
) -> CourseMemberItem:
    pid, display_name = _validate_course_member_fields(target_person_id, name, user_type)

    college_id, college_name = _fetch_course_college(supabase, course_id)

    existing_rel = (
        supabase.table(USER_COURSE_RELATION_TABLE)
        .select("course_user_id")
        .eq("person_id", pid)
        .eq("course_id", course_id)
        .or_(ACTIVE_DELETED_FILTER)
        .limit(1)
        .execute()
    )
    if existing_rel.data:
        raise HTTPException(status_code=409, detail="該使用者已在課程中")

    user_resp = (
        supabase.table(USER_TABLE)
        .select("user_id, person_id, name, password, college_id")
        .eq("person_id", pid)
        .eq("college_id", college_id)
        .or_(ACTIVE_DELETED_FILTER)
        .limit(1)
        .execute()
    )
    ts = now_taipei_iso()
    if user_resp.data:
        user_row = user_resp.data[0]
        user_id = int(user_row["user_id"])
        password = user_row.get("password")
        member_name = (user_row.get("name") or "").strip() or display_name
    else:
        other_college = (
            supabase.table(USER_TABLE)
            .select("user_id, college_id")
            .eq("person_id", pid)
            .or_(ACTIVE_DELETED_FILTER)
            .limit(1)
            .execute()
        )
        if other_college.data:
            raise HTTPException(
                status_code=409,
                detail="person_id 已存在於其他學院，無法加入此課程",
            )
        ins = (
            supabase.table(USER_TABLE)
            .insert(
                {
                    "person_id": pid,
                    "name": display_name,
                    "password": DEFAULT_NEW_MEMBER_PASSWORD,
                    "college_id": college_id,
                    **({"college_name": college_name} if college_name else {}),
                    "deleted": False,
                    "updated_at": ts,
                    "created_at": ts,
                }
            )
            .execute()
        )
        user_row = ins.data[0] if ins.data else None
        if not user_row:
            raise HTTPException(status_code=500, detail="新增使用者失敗")
        user_id = int(user_row["user_id"])
        password = DEFAULT_NEW_MEMBER_PASSWORD
        member_name = display_name

    rel_ins = (
        supabase.table(USER_COURSE_RELATION_TABLE)
        .insert(
            {
                "user_id": user_id,
                "person_id": pid,
                "name": display_name,
                "course_id": course_id,
                "college_id": college_id,
                "user_type": user_type,
                "deleted": False,
                "updated_at": ts,
                "created_at": ts,
            }
        )
        .execute()
    )
    rel_row = rel_ins.data[0] if rel_ins.data else None
    if not rel_row:
        raise HTTPException(status_code=500, detail="新增課程成員失敗")

    return CourseMemberItem(
        course_user_id=int(rel_row["course_user_id"]),
        user_id=user_id,
        person_id=pid,
        name=member_name,
        password=password,
        user_type=user_type,
        college_id=college_id,
    )


def _edit_course_member(
    supabase,
    *,
    course_id: int,
    target_person_id: str,
    name: str,
    user_type: int,
) -> CourseMemberItem:
    pid, display_name = _validate_course_member_fields(target_person_id, name, user_type)
    rel = _get_active_course_member_relation(supabase, course_id, pid)
    user_id = int(rel["user_id"])
    ts = now_taipei_iso()

    supabase.table(USER_COURSE_RELATION_TABLE).update(
        {
            "name": display_name,
            "user_type": user_type,
            "updated_at": ts,
        }
    ).eq("course_user_id", rel["course_user_id"]).execute()

    user_row = _fetch_user_row_for_member(supabase, user_id)
    if user_row:
        supabase.table(USER_TABLE).update(
            {"name": display_name, "updated_at": ts}
        ).eq("user_id", user_id).execute()
        user_row = {**user_row, "name": display_name}

    rel_out = {
        **rel,
        "name": display_name,
        "user_type": user_type,
    }
    return _course_member_item_from_rows(rel_out, user_row)


def _soft_delete_course_member(
    supabase,
    *,
    course_id: int,
    target_person_id: str,
) -> CourseMemberItem:
    pid = (target_person_id or "").strip()
    if not pid:
        raise HTTPException(status_code=400, detail="person_id 不可為空")
    rel = _get_active_course_member_relation(supabase, course_id, pid)
    user_row = _fetch_user_row_for_member(supabase, int(rel["user_id"]))
    ts = now_taipei_iso()
    supabase.table(USER_COURSE_RELATION_TABLE).update(
        {"deleted": True, "updated_at": ts}
    ).eq("course_user_id", rel["course_user_id"]).execute()
    return _course_member_item_from_rows(rel, user_row)


def _fetch_course_members(supabase, course_id: int) -> list[CourseMemberItem]:
    """依 course_id 自 User_Course_Relation 與 User 組出課程成員列表。"""
    rel_resp = (
        supabase.table(USER_COURSE_RELATION_TABLE)
        .select("course_user_id, user_id, person_id, name, user_type, college_id")
        .eq("course_id", course_id)
        .or_(ACTIVE_DELETED_FILTER)
        .order("course_user_id")
        .execute()
    )
    rels = rel_resp.data or []
    if not rels:
        return []

    user_ids = [int(r["user_id"]) for r in rels if r.get("user_id") is not None]
    if not user_ids:
        return []

    user_resp = (
        supabase.table(USER_TABLE)
        .select("user_id, person_id, name, password")
        .in_("user_id", list(dict.fromkeys(user_ids)))
        .or_(ACTIVE_DELETED_FILTER)
        .execute()
    )
    user_by_id = {
        int(u["user_id"]): u for u in (user_resp.data or []) if u.get("user_id") is not None
    }

    members: list[CourseMemberItem] = []
    for r in rels:
        uid_raw = r.get("user_id")
        if uid_raw is None:
            continue
        uid = int(uid_raw)
        user_row = user_by_id.get(uid)
        if not user_row:
            continue
        name = (user_row.get("name") or "").strip() or (r.get("name") or "").strip() or None
        pid = (user_row.get("person_id") or r.get("person_id") or "").strip() or None
        college_raw = r.get("college_id")
        college_id = int(college_raw) if college_raw is not None and int(college_raw or 0) != 0 else None
        members.append(
            CourseMemberItem(
                course_user_id=int(r["course_user_id"]),
                user_id=uid,
                person_id=pid,
                name=name,
                password=user_row.get("password"),
                user_type=r.get("user_type"),
                college_id=college_id,
            )
        )
    return members


@router.get("/course-members", response_model=ListCourseMembersResponse)
def list_course_members(caller: CurrentUser, course_id: CourseId):
    """List course members：依 course_id 列出該課程所有使用者（User_Course_Relation + User，不含已刪除）。"""
    _require_active_person(caller.person_id, caller.college_id)
    try:
        supabase = get_supabase()
        members = _fetch_course_members(supabase, course_id)
        return ListCourseMembersResponse(course_id=course_id, members=members, count=len(members))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/course-members", response_model=CourseMemberItem, status_code=201)
def add_course_member(
    body: openapi_body(
        AddCourseMemberRequest,
        {"person_id": "string", "name": "string", "user_type": 3},
    ),
    person_id: PersonId,
    course_id: CourseId,
):
    """Add course member：新增一筆課程成員（person_id、name、user_type）。
    User 表已有相同 college_id + person_id 時僅新增選課；否則建立 User（預設密碼 0000）後再加入課程。"""
    _require_developer_or_manager_for_course_setting_write(person_id, course_id)
    try:
        supabase = get_supabase()
        return _add_course_member(
            supabase,
            course_id=course_id,
            target_person_id=body.person_id,
            name=body.name,
            user_type=body.user_type,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


def _batch_add_course_members(
    supabase,
    *,
    course_id: int,
    rows: list[BatchCourseMemberRow],
) -> BatchAddCourseMembersResponse:
    created: list[CourseMemberItem] = []
    failed: list[BatchCourseMemberFailure] = []
    for row in rows:
        pid = (row.person_id or "").strip()
        if not pid:
            failed.append(
                BatchCourseMemberFailure(
                    person_id=row.person_id if row.person_id is not None else "",
                    detail="person_id 不可為空",
                )
            )
            continue
        try:
            member = _add_course_member(
                supabase,
                course_id=course_id,
                target_person_id=pid,
                name=row.name,
                user_type=3,
            )
            created.append(member)
        except HTTPException as he:
            failed.append(BatchCourseMemberFailure(person_id=pid, detail=str(he.detail)))
        except Exception as e:
            failed.append(BatchCourseMemberFailure(person_id=pid, detail=str(e)))
    return BatchAddCourseMembersResponse(
        created=created,
        failed=failed,
        created_count=len(created),
        failed_count=len(failed),
    )


@router.post("/course-members/batch", response_model=BatchAddCourseMembersResponse, status_code=201)
def batch_add_course_members(
    body: Annotated[
        list[BatchCourseMemberRow],
        Body(
            openapi_examples={
                "default": {
                    "summary": "Default",
                    "value": [{"person_id": "string", "name": "string"}],
                }
            }
        ),
    ],
    person_id: PersonId,
    course_id: CourseId,
):
    """Add course members (batch)：批次新增該課程學生（每筆 person_id、name；user_type 固定 3）。
    User 表已有相同 college_id + person_id 時僅新增選課；否則建立 User（預設密碼 0000）後再加入課程。
    已存在於課程或失敗之 person_id 會列入 failed，其餘仍會繼續寫入。"""
    _require_developer_or_manager_for_course_setting_write(person_id, course_id)
    if not body:
        raise HTTPException(status_code=400, detail="請至少傳入一筆學生")

    try:
        supabase = get_supabase()
        return _batch_add_course_members(supabase, course_id=course_id, rows=body)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.patch(
    "/course-members/{member_person_id}",
    response_model=CourseMemberItem,
    summary="Edit course member",
    operation_id="rag_course_members_edit",
)
def edit_course_member(
    target_person_id: Annotated[
        str, Path(alias="member_person_id", description="要編輯的成員 person_id")
    ],
    body: openapi_body(
        EditCourseMemberRequest,
        {"name": "string", "user_type": 3},
    ),
    person_id: PersonId,
    course_id: CourseId,
):
    """Edit course member：更新課程成員 name、user_type（以 path member_person_id 識別成員）。"""
    _require_developer_or_manager_for_course_setting_write(person_id, course_id)
    try:
        supabase = get_supabase()
        return _edit_course_member(
            supabase,
            course_id=course_id,
            target_person_id=target_person_id,
            name=body.name,
            user_type=body.user_type,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.delete(
    "/course-members/{member_person_id}",
    response_model=CourseMemberItem,
    summary="Remove course member",
    operation_id="rag_course_members_delete",
)
def soft_delete_course_member(
    target_person_id: Annotated[
        str, Path(alias="member_person_id", description="要移出課程的成員 person_id")
    ],
    person_id: PersonId,
    course_id: CourseId,
):
    """Remove course member：自課程軟刪除成員（User_Course_Relation deleted=true，不刪 User 表）。"""
    _require_developer_or_manager_for_course_setting_write(person_id, course_id)
    try:
        supabase = get_supabase()
        return _soft_delete_course_member(
            supabase,
            course_id=course_id,
            target_person_id=target_person_id,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/person-analysis-user-prompt-text", response_model=PersonAnalysisUserPromptTextResponse)
def get_person_analysis_user_prompt_text_setting(caller: CurrentUser, course_id: CourseId):
    """取得個人分析指令（Course_Setting key=person_analysis_user_prompt_text，依 course_id）。"""
    _require_active_person(caller.person_id, caller.college_id)
    try:
        text = fetch_course_setting_text(
            COURSE_SETTING_PERSON_ANALYSIS_USER_PROMPT_TEXT_KEY, course_id
        )
        return PersonAnalysisUserPromptTextResponse(
            course_id=course_id,
            person_analysis_user_prompt_text=text or None,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.put("/person-analysis-user-prompt-text", response_model=PersonAnalysisUserPromptTextResponse)
def put_person_analysis_user_prompt_text_setting(
    body: openapi_body(
        PutPersonAnalysisUserPromptTextRequest,
        {"person_analysis_user_prompt_text": "string"},
    ),
    person_id: PersonId,
    course_id: CourseId,
):
    """寫入個人分析指令至 Course_Setting（依 course_id upsert；傳空字串可清除）。"""
    _require_developer_or_manager_for_course_setting_write(person_id, course_id)
    value_to_save = (body.person_analysis_user_prompt_text or "").strip()
    try:
        row = upsert_course_setting_and_get_row(
            get_supabase(),
            COURSE_SETTING_PERSON_ANALYSIS_USER_PROMPT_TEXT_KEY,
            value_to_save,
            course_id,
        )
        if not row:
            raise HTTPException(status_code=500, detail="寫入 Course_Setting 失敗")
        return PersonAnalysisUserPromptTextResponse(
            course_id=course_id,
            person_analysis_user_prompt_text=value_to_save or None,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/course-analysis-user-prompt-text", response_model=CourseAnalysisUserPromptTextResponse)
def get_course_analysis_user_prompt_text_setting(caller: CurrentUser, course_id: CourseId):
    """取得課程分析指令（Course_Setting key=course_analysis_user_prompt_text，依 course_id）。"""
    _require_active_person(caller.person_id, caller.college_id)
    try:
        text = fetch_course_setting_text(
            COURSE_SETTING_COURSE_ANALYSIS_USER_PROMPT_TEXT_KEY, course_id
        )
        return CourseAnalysisUserPromptTextResponse(
            course_id=course_id,
            course_analysis_user_prompt_text=text or None,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.put("/course-analysis-user-prompt-text", response_model=CourseAnalysisUserPromptTextResponse)
def put_course_analysis_user_prompt_text_setting(
    body: openapi_body(
        PutCourseAnalysisUserPromptTextRequest,
        {"course_analysis_user_prompt_text": "string"},
    ),
    person_id: PersonId,
    course_id: CourseId,
):
    """寫入課程分析指令至 Course_Setting（依 course_id upsert；傳空字串可清除）。"""
    _require_developer_or_manager_for_course_setting_write(person_id, course_id)
    value_to_save = (body.course_analysis_user_prompt_text or "").strip()
    try:
        row = upsert_course_setting_and_get_row(
            get_supabase(),
            COURSE_SETTING_COURSE_ANALYSIS_USER_PROMPT_TEXT_KEY,
            value_to_save,
            course_id,
        )
        if not row:
            raise HTTPException(status_code=500, detail="寫入 Course_Setting 失敗")
        return CourseAnalysisUserPromptTextResponse(
            course_id=course_id,
            course_analysis_user_prompt_text=value_to_save or None,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
