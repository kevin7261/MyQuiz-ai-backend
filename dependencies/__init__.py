"""共用 FastAPI 依賴。"""

from dependencies.person_id import PersonId, require_person_id

__all__ = ["PersonId", "require_person_id"]
