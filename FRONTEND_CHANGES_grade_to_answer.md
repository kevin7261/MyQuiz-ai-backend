# 前端異動通知:`grade` 全面改名為 `answer`

> 日期:2026-06-09
> 影響:**Breaking change**。API 上已不再存在任何 `grade` 字樣,全部改為 `answer`。

---

## 1. 端點路徑

| 舊 | 新 |
|---|---|
| `POST /v1/exam/quizzes/llm-grade` | `POST /v1/exam/quizzes/llm-answer` |
| `POST /v1/rag/quizzes/llm-grade` | `POST /v1/rag/quizzes/llm-answer` |
| `POST /v1/rag/quizzes/llm-grade-db` | `POST /v1/rag/quizzes/llm-answer-db` |
| `GET /v1/exam/quizzes/grade-result/{job_id}` | `GET /v1/exam/quizzes/answer-result/{job_id}` |
| `GET /v1/rag/quizzes/grade-result/{job_id}` | `GET /v1/rag/quizzes/answer-result/{job_id}` |
| `PUT /v1/exam/quizzes/{exam_quiz_id}/grade-rate` | `PUT /v1/exam/quizzes/{exam_quiz_id}/answer-rate` |
| `POST /v1/exam/quizzes/grade`(隱藏舊別名) | `POST /v1/exam/quizzes/answer` |

## 2. 欄位 / JSON key

| 舊 | 新 | 出現位置 |
|---|---|---|
| `grade_rate` | `answer_rate` | Exam_Quiz 回應欄位;`answer-rate` 的 request body `{ "answer_rate": -1/0/1 }` |
| `grade_llm_model` | `answer_llm_model` | 評分 202 回應 `{ "job_id", "answer_llm_model" }`;Exam_Quiz 回應 |
| `llm_grade`(設定區塊) | `llm_answer` | `GET /v1/prompt-templates` 回應;RAG 檢索設定 config |
| `grade_embedding_model` | `answer_embedding_model` | `GET /v1/prompt-templates` 回應的設定區塊 |

## 3. OpenAPI schema 名稱(若前端用 codegen / typed client 會受影響)

| 舊 | 新 |
|---|---|
| `ExamQuizGradeRequest` | `ExamQuizAnswerRequest` |
| `QuizGradeRequest` | `QuizAnswerRequest` |
| `QuizGradeDbOnlyRequest` | `QuizAnswerDbOnlyRequest` |
| `ExamQuizGradeRateRequest` | `ExamQuizAnswerRateRequest` |

summary 標籤:`Exam Grade Quiz` → `Exam Answer Quiz`、`Get Grade Result` → `Get Answer Result` 等。

## 4. 一句話摘要

> 所有 `grade` 一律改成 `answer`:
> - 路徑:`llm-grade` → `llm-answer`、`grade-result` → `answer-result`、`grade-rate` → `answer-rate`
> - 欄位:`grade_rate` → `answer_rate`、`grade_llm_model` → `answer_llm_model`
> - 設定 key:`llm_grade` → `llm_answer`、`grade_embedding_model` → `answer_embedding_model`
>
> API 上已不存在任何 `grade`。

---

## 5. 後端 DB 端需同步(非程式碼,提醒事項)

- Supabase 欄位:`Exam_Quiz.grade_rate` → `answer_rate`、`grade_llm_model` → `answer_llm_model`
- Course_Setting 內已存的 prompt-template JSON:`llm_grade` key → `llm_answer`

未同步前,讀寫會對不上。
