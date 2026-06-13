"""routers.quiz LLM 設定 routes：quiz 專屬 API Key／模型（與 bank-／exam-／rag- 完全分開）。

與 bank 版同形（僅 Bank／Quiz 命名與 Course_Setting key 不同），端點由
routers.llm_settings.build_llm_settings_router 依 quiz 常數產生；
路徑、operation_id、回應 model 與說明文字皆與原手寫版一致。
"""

from utils.course_setting import (
    COURSE_SETTING_QUIZ_API_KEY,
    COURSE_SETTING_QUIZ_LLM_MODEL,
)
from utils.quiz_llm_key import (
    fetch_quiz_api_key_setting_row,
    fetch_quiz_llm_model_setting_row,
    quiz_api_key_exists,
)
from routers.llm_settings import build_llm_settings_router

router = build_llm_settings_router(
    prefix="quiz",
    title="Quiz",
    api_key_setting_key=COURSE_SETTING_QUIZ_API_KEY,
    llm_model_setting_key=COURSE_SETTING_QUIZ_LLM_MODEL,
    api_key_exists=quiz_api_key_exists,
    fetch_api_key_setting_row=fetch_quiz_api_key_setting_row,
    fetch_llm_model_setting_row=fetch_quiz_llm_model_setting_row,
)
