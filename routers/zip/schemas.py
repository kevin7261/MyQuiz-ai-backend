"""routers.zip schemas（自 zip.py 拆分）。"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ListRagResponse(BaseModel):
    """GET /rag/pages 回應：每筆 Rag 含 units（Rag_Unit），每個 unit 含 quizzes（Rag_Quiz，含 follow_up、quiz_history_list）；皆已依 query course_id 篩選。"""
    rags: list[dict]
    count: int


class RagUnitMp3FileResponse(BaseModel):
    """GET /rag/units/{rag_unit_id}/mp3-file 回應。"""
    rag_unit_id: int
    rag_page_id: str
    audio_base64: str
    media_type: str
    filename: str
    transcript: str = ""


class UpdateRagUnitNameRequest(BaseModel):
    """PATCH /rag/pages/{rag_page_id}：請求僅含 tab_name；定位用 path 參數 rag_page_id。"""
    tab_name: str = Field(..., description="新的顯示名稱，寫入 Rag 表 tab_name 欄位")


class PackRequest(BaseModel):
    """
    person_id（同 public.Rag）→ unit_list → Rag_Unit 相關欄（unit_name、unit_type、transcript、rag_chunk_*）。
    rag_page_id 改由 path 參數帶入（POST /rag/pages/{rag_page_id}/build-zip），於 handler 內回填本物件。
    rag_chunk_size／rag_chunk_overlap：全批預設（寫入 Rag_Unit、建 FAISS 時用）。
    rag_chunk_sizes／rag_chunk_overlaps：可選逗號字串或整數陣列（JSON），與 unit_list 解出之任務數同序；某段空白則該段用 rag_chunk_size／rag_chunk_overlap。
    unit_names：可選逗號字串或字串陣列（JSON），與任務同序；某段 strip 後非空則覆寫該單元 Rag_Unit.unit_name（顯示名），空白則與 folder_combination 相同（皆為檔名 stem）。
    """

    # rag_page_id 不是 request body 欄位，改由 path 參數帶入後於 handler 內回填本物件（extra=allow 才能 setattr）。
    model_config = ConfigDict(extra="allow")

    person_id: str
    unit_list: str  # 指定要打包的資料夾；例："220222+220301"（加號=同一 ZIP 多資料夾）；結果存入 Rag_Unit 表
    unit_names: str | list[str] | None = Field(
        default="",
        description="可選；逗號字串或 JSON 字串陣列，與 packed 任務同序；該段 strip 後非空則覆寫 Rag_Unit.unit_name（顯示名），空段則與 folder_combination 相同（檔名 stem）",
    )
    unit_types: str = ""
    transcripts: list[str] | None = Field(
        default=None,
        description="可選；與 unit_list 逗號分段對齊；非空時覆寫逐字稿全文",
    )
    rag_chunk_size: int = 1000
    rag_chunk_overlap: int = 200
    rag_chunk_sizes: str = Field(
        "",
        description="可選；逗號字串或 [1000,800] 陣列，與 packed 任務同序；空段→該任務用 rag_chunk_size",
    )
    rag_chunk_overlaps: str = Field(
        "",
        description="可選；逗號字串或 [200,100] 陣列，與 packed 任務同序；空段→該任務用 rag_chunk_overlap",
    )

    @field_validator("rag_chunk_sizes", "rag_chunk_overlaps", mode="before")
    @classmethod
    def _coerce_chunk_segments_csv(cls, v: Any) -> str:
        """相容前端傳 JSON 陣列；統一成逗號字串供 _chunk_params_per_task 解析。"""
        if v is None:
            return ""
        if isinstance(v, list):
            parts: list[str] = []
            for x in v:
                try:
                    parts.append(str(int(x)))
                except (TypeError, ValueError):
                    parts.append("")
            return ",".join(parts)
        if isinstance(v, str):
            return v.strip()
        return str(v)

    build_faiss: bool | None = Field(
        default=None,
        description="省略時依 User_Course_Relation.user_type；False 時僅複製 repack 至 rag；True 時強制建向量 RAG ZIP",
    )

    @field_validator("unit_names", mode="before")
    @classmethod
    def _coerce_unit_names(cls, v: Any) -> str | list[str]:
        if v is None:
            return ""
        if isinstance(v, list):
            return [("" if x is None else str(x)) for x in v]
        if isinstance(v, str):
            return v.strip()
        return str(v)


class InsertRagQuizRowRequest(BaseModel):
    """
    POST /rag/quizzes：欄位順序對齊 public.Rag_Quiz 之關聯欄（rag_page_id、rag_unit_id）。
    `rag_page_id` 與 `rag_unit_id` 二擇一定位 Rag_Unit：
    - `rag_unit_id > 0`：以主鍵載入；若同傳 `rag_page_id`（非空）則須與該列一致。
    - `rag_unit_id == 0`：`rag_page_id`（非空）須在該名下**唯一**一筆未刪除之 Rag_Unit，否則 400。
    """

    rag_page_id: str = Field("", description="Rag tab 識別；與 rag_unit_id 併用見上")
    rag_unit_id: int = Field(0, ge=0, description="Rag_Unit 主鍵；0 表示改由 rag_page_id 唯一解析")


class UpdateRagQuizQuizNameRequest(BaseModel):
    """PATCH /rag/quizzes/{rag_quiz_id}：更新 Rag_Quiz 的 quiz_name；定位用 path 參數 rag_quiz_id。"""
    quiz_name: str = Field(..., description="新的 quiz_name")
