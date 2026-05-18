"""共用 FastAPI 依賴。"""

from dependencies.course_id import CourseId, require_course_id
from dependencies.person_id import PersonId, require_person_id

__all__ = ["PersonId", "require_person_id", "CourseId", "require_course_id"]
