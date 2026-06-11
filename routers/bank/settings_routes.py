"""routers.bank LLM 設定 routes：bank 專屬 API Key／模型（與 rag 的 /v1/rag/llm-* 完全分開）。

與 quiz 版同形（僅 Bank／Quiz 命名與 Course_Setting key 不同），端點由
routers.llm_settings.build_llm_settings_router 依 bank 常數產生；
路徑、operation_id、回應 model 與說明文字皆與原手寫版一致。
"""

from utils.course_setting import (
    COURSE_SETTING_BANK_ANSWER_USER_PROMPT_TEXT,
    COURSE_SETTING_BANK_API_KEY,
    COURSE_SETTING_BANK_LLM_MODEL,
    COURSE_SETTING_BANK_QUESTION_SYSTEM_PROMPT_TEXT,
    COURSE_SETTING_BANK_QUESTION_USER_PROMPT_TEXT,
)
from utils.bank_llm_key import (
    bank_api_key_exists,
    fetch_bank_api_key_setting_row,
    fetch_bank_llm_model_setting_row,
)
from routers.llm_settings import build_llm_settings_router

router = build_llm_settings_router(
    prefix="bank",
    title="Bank",
    group_table="Bank_Group",
    api_key_setting_key=COURSE_SETTING_BANK_API_KEY,
    llm_model_setting_key=COURSE_SETTING_BANK_LLM_MODEL,
    question_system_prompt_key=COURSE_SETTING_BANK_QUESTION_SYSTEM_PROMPT_TEXT,
    question_user_prompt_key=COURSE_SETTING_BANK_QUESTION_USER_PROMPT_TEXT,
    answer_user_prompt_key=COURSE_SETTING_BANK_ANSWER_USER_PROMPT_TEXT,
    api_key_exists=bank_api_key_exists,
    fetch_api_key_setting_row=fetch_bank_api_key_setting_row,
    fetch_llm_model_setting_row=fetch_bank_llm_model_setting_row,
)
