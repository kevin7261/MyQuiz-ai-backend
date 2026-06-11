"""routers.quiz LLM 設定 routes：quiz 專屬 API Key／模型（與 bank-／exam-／rag- 完全分開）。

與 bank 版同形（僅 Bank／Quiz 命名與 Course_Setting key 不同），端點由
routers.llm_settings.build_llm_settings_router 依 quiz 常數產生；
路徑、operation_id、回應 model 與說明文字皆與原手寫版一致。
"""

from utils.course_setting import (
    COURSE_SETTING_QUIZ_ANSWER_USER_PROMPT_TEXT,
    COURSE_SETTING_QUIZ_API_KEY,
    COURSE_SETTING_QUIZ_LLM_MODEL,
    COURSE_SETTING_QUIZ_QUESTION_SYSTEM_PROMPT_TEXT,
    COURSE_SETTING_QUIZ_QUESTION_USER_PROMPT_TEXT,
)
from utils.quiz_llm_key import (
    fetch_quiz_api_key_setting_row,
    fetch_quiz_llm_model_setting_row,
    quiz_api_key_exists,
)
from routers.llm_settings import build_llm_settings_router
from routers.quiz_module_analysis_prompts import register_quiz_module_analysis_prompt_routes

router = build_llm_settings_router(
    prefix="quiz",
    title="Quiz",
    group_table="Quiz_Group",
    api_key_setting_key=COURSE_SETTING_QUIZ_API_KEY,
    llm_model_setting_key=COURSE_SETTING_QUIZ_LLM_MODEL,
    question_system_prompt_key=COURSE_SETTING_QUIZ_QUESTION_SYSTEM_PROMPT_TEXT,
    question_user_prompt_key=COURSE_SETTING_QUIZ_QUESTION_USER_PROMPT_TEXT,
    answer_user_prompt_key=COURSE_SETTING_QUIZ_ANSWER_USER_PROMPT_TEXT,
    api_key_exists=quiz_api_key_exists,
    fetch_api_key_setting_row=fetch_quiz_api_key_setting_row,
    fetch_llm_model_setting_row=fetch_quiz_llm_model_setting_row,
)
register_quiz_module_analysis_prompt_routes(router)
