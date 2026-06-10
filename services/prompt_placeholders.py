"""
GET /v1/prompt-templates 占位符說明（與 services/quiz_generation、answering、weakness_report、rag_prompts 填入邏輯對齊）。
"""


def prompt_placeholder_descriptions() -> dict[str, dict[str, str]]:
    """依 API 區塊回傳 `{占位符}` → 說明。"""
    return {
        "rag": {
            "quiz_content": (
                "僅 llm_answer 向量檢索：以本題題幹（Rag_Quiz.quiz_content 或請求 body）"
                "作為 retriever 查詢句，非 Chat user 模板占位。"
            ),
        },
        "llm_generate": {
            "quiz_user_prompt_text": (
                "教師出題指令；POST .../llm-generate 之 body 或 Rag_Quiz.quiz_user_prompt_text；"
                "空字串時模板該節留白，依 system 指示略過。"
            ),
            "quiz_history_body": (
                "已出過題目區塊正文；由 quiz_history_list（字串陣列）格式化成編號列表，"
                "或無列表時為「未提供」說明句。"
            ),
            "quiz_history_qa_body": (
                "僅 followup user 模板：先前問答紀錄；由 quiz_history_list（物件陣列，"
                "含 quiz_content、answer_content、quiz_answer_reference、answer_critique）"
                "逐組排版，或無資料時為占位說明。"
            ),
            "context_md": (
                "課程內容：unit_type=1 為 FAISS 檢索片段、2/3/4 為 Rag_Unit.transcript 全文，"
                "經 _context_as_markdown_fenced 包成 ```text …``` fenced block。"
            ),
        },
        "llm_answer": {
            "id_block": (
                "選填；有 exam_quiz_id 或 rag_quiz_id 時為「## 關聯識別」Markdown 區塊"
                "（含兩 id 列點），否則空字串。"
            ),
            "quiz_user_prompt_text": (
                "出題 user prompt；Rag_Quiz.quiz_user_prompt_text；"
                "空則顯示「（未提供）」。"
            ),
            "answer_user_prompt_text": (
                "作答／批改 user prompt；請求 body 或 Rag_Quiz.answer_user_prompt_text；"
                "空則顯示「（未提供）」。"
            ),
            "quiz_content": (
                "本題題幹；Rag_Quiz.quiz_content 或請求 body。"
            ),
            "quiz_answer": (
                "學生作答；請求 body 之 answer_content／quiz_answer。"
            ),
            "context_md": (
                "課程內容：逐字稿路徑為 Rag_Unit.transcript 全文；"
                "FAISS 路徑為以 quiz_content 檢索之 chunk，同樣包成 ```text …```。"
            ),
        },
        "bank": {
            "quiz_system_prompt_text": (
                "僅出題 system：題組連續出題規定；Bank_Group.question_system_prompt_text"
                "（或請求 override），織入「指令優先級」區塊為最高優先；空則該節不出現。"
            ),
            "quiz_user_prompt_text": (
                "出題／批改 user prompt；Bank_Group.question_user_prompt_text"
                "（或 Bank_QA.question_user_prompt_text／請求 override）；"
                "空字串時模板該節留白。"
            ),
            "quiz_history_body": (
                "僅出題 user：已出過題目區塊正文；由同題組既有 Bank_QA 題幹"
                "格式化成編號列表，或無題目時為「未提供」說明句。"
            ),
            "id_block": (
                "僅批改 user：有 bank_qa_id 時為「## 關聯識別」Markdown 區塊"
                "（含 bank_qa_id 列點），否則空字串。"
            ),
            "answer_user_prompt_text": (
                "僅批改 user：作答／批改 user prompt；Bank_Group.answer_user_prompt_text"
                "（或 Bank_QA.answer_user_prompt_text）；空則顯示「（未提供）」。"
            ),
            "quiz_content": (
                "僅批改 user：本題題幹；Bank_QA.quiz_content 或請求 body。"
            ),
            "quiz_answer": (
                "僅批改 user：學生作答；請求 body 之 answer_content／quiz_answer。"
            ),
            "context_md": (
                "課程內容：unit_type=1 為 FAISS 檢索片段、2/3/4 為逐字稿全文，"
                "包成 ```text …``` fenced block。"
            ),
        },
        "person_analysis": {
            "analysis_user_prompt_text": (
                "個人分析指令；Course_Setting key=person_analysis_user_prompt_text"
                "（GET/PUT /rag/person-analysis-user-prompt-text，依 course_id）；"
                "空則「（未提供）」。"
            ),
            "material_md": (
                "測驗素材 Markdown：已批改時為各題 answer_critique 摘要列表；"
                "否則為題幹、參考答案、學生作答、quiz_rate 等摘要。"
            ),
        },
        "course_analysis": {
            "analysis_user_prompt_text": (
                "課程分析指令；Course_Setting key=course_analysis_user_prompt_text"
                "（GET/PUT /rag/course-analysis-user-prompt-text，依 course_id）；"
                "空則「（未提供）」。"
            ),
            "material_md": (
                "同 person_analysis.material_md，但素材範圍為整課 course_id 下已作答 Exam_Quiz。"
            ),
        },
        "quiz": {
            "question_user_prompt_text": (
                "追問 user：題組出題規定；Quiz_Group.question_user_prompt_text；"
                "空則顯示「（未提供）」。"
            ),
            "quiz_qa_history_body": (
                "追問 user：本題組全部 Quiz_QA 紀錄正文；含題目、提示、參考答案、"
                "學生作答、評閱（後三者可能為「（未提供）」）。"
            ),
            "ask_history_body": (
                "追問 user：本題組已完成之 Quiz_Ask 紀錄正文；"
                "含每次提問與回答，不含本次學生提問。"
            ),
            "ask_user_prompt_text": (
                "追問 user：本次學生提問；POST .../llm-ask body；空則顯示「（未提供）」。"
            ),
            "context_md": (
                "追問 user：課程內容；unit_type=2/3/4 為逐字稿全文，"
                "其餘為以本次提問檢索之 FAISS 片段，包成 ```text …```。"
            ),
            "id_block": (
                "追問 user：有 quiz_group_id／bank_group_id 時為「## 關聯識別」"
                "Markdown 區塊，否則空字串。"
            ),
        },
    }
