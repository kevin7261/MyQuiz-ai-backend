"""routers.bank schemas（自 routers.zip 複製，Bank/Bank_Unit 版；不含 quiz）。"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ListBankResponse(BaseModel):
    """GET /bank/pages 回應：每筆 Bank 含 units（Bank_Unit）；皆已依 query course_id 篩選。"""
    banks: list[dict]
    count: int


class BankUnitMp3FileResponse(BaseModel):
    """GET /bank/units/{bank_unit_id}/mp3-file 回應。"""
    bank_unit_id: int
    bank_page_id: str
    audio_base64: str
    media_type: str
    filename: str
    transcript: str = ""


class BankUnitTextResponse(BaseModel):
    """GET /bank/units/{bank_unit_id}/text 回應：unit_type=2 文字單元逐字稿（DB 優先，無則讀 upload ZIP）。"""
    bank_unit_id: int
    bank_page_id: str
    folder_name: str = ""
    text_file_name: str = ""
    transcript: str = ""


class BankUnitTextPreviewResponse(BaseModel):
    """GET /bank/pages/{bank_page_id}/unit-preview/text 回應：建置前自 upload ZIP 讀取之文字單元逐字稿。"""
    bank_page_id: str
    folder_name: str
    text_file_name: str = ""
    transcript: str = ""


class BankUnitMp3FilePreviewResponse(BaseModel):
    """GET /bank/pages/{bank_page_id}/unit-preview/mp3-file 回應：建置前自 upload ZIP 擷取之音訊與同資料夾文字檔逐字稿。"""
    bank_page_id: str
    folder_name: str
    audio_base64: str
    media_type: str
    filename: str
    text_file_name: str = ""
    transcript: str = ""


class BankUnitYoutubeUrlPreviewResponse(BaseModel):
    """GET /bank/pages/{bank_page_id}/unit-preview/youtube-url 回應：建置前自 upload ZIP 解析 watch URL 與文字檔第二行起逐字稿。"""
    bank_page_id: str
    folder_name: str
    youtube_url: str = Field(..., description="https://www.youtube.com/watch?v=…")
    text_file_name: str = ""
    transcript: str = ""


class BankUnitYoutubeUrlResponse(BaseModel):
    """GET /bank/units/{bank_unit_id}/youtube-url 回應：unit_type=4 之 watch URL 與逐字稿（DB 優先，無則讀 upload ZIP）。"""
    bank_unit_id: int
    bank_page_id: str
    folder_name: str = ""
    youtube_url: str = Field(..., description="https://www.youtube.com/watch?v=…")
    text_file_name: str = ""
    transcript: str = ""


class UpdateBankUnitNameRequest(BaseModel):
    """PATCH /bank/pages/{bank_page_id}：請求僅含 tab_name；定位用 path 參數 bank_page_id。"""
    tab_name: str = Field(..., description="新的顯示名稱，寫入 Bank 表 tab_name 欄位")


class PackRequest(BaseModel):
    """
    person_id（同 public.Bank）→ unit_list → Bank_Unit 相關欄（unit_name、unit_type、transcript、rag_chunk_*）。
    bank_page_id 改由 path 參數帶入（POST /bank/pages/{bank_page_id}/build-zip），於 handler 內回填本物件。
    rag_chunk_size／rag_chunk_overlap：全批預設（寫入 Bank_Unit、建 FAISS 時用）。
    rag_chunk_sizes／rag_chunk_overlaps：可選逗號字串或整數陣列（JSON），與 unit_list 解出之任務數同序；某段空白則該段用 rag_chunk_size／rag_chunk_overlap。
    unit_names：可選逗號字串或字串陣列（JSON），與任務同序；某段 strip 後非空則覆寫該單元 Bank_Unit.unit_name（顯示名），空白則與 folder_combination 相同（皆為檔名 stem）。
    """

    # bank_page_id 不是 request body 欄位，改由 path 參數帶入後於 handler 內回填本物件（extra=allow 才能 setattr）。
    model_config = ConfigDict(extra="allow")

    person_id: str | None = Field(
        default=None,
        description="選填；未傳以 token 解析的呼叫者為準；有傳須與呼叫者一致",
    )
    unit_list: str  # 指定要打包的資料夾；例："220222+220301"（加號=同一 ZIP 多資料夾）；結果存入 Bank_Unit 表
    unit_names: str | list[str] | None = Field(
        default="",
        description="可選；逗號字串或 JSON 字串陣列，與 packed 任務同序；該段 strip 後非空則覆寫 Bank_Unit.unit_name（顯示名），空段則與 folder_combination 相同（檔名 stem）",
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
