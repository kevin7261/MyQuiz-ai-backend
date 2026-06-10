# 前端異動通知：新增 Quiz（試卷／Test）模組

> 日期：2026-06-10
> 影響：**純新增**，無 breaking change。既有 bank／exam／rag 端點完全不動。
> 完整串接文件（含每支端點的 request/response JSON 範例）：`docs/QUIZ_API.md`

---

## 1. 這是什麼

搭配 **Bank（題庫）** 的「試卷／應試」層，定位等同 `exam` 之於 `rag`：

```
Quiz(試卷) ─< Quiz_Group(題組, 自 Bank_Group 快照) ─< Quiz_QA(題目)
   quiz_page_id      quiz_group_id                     quiz_qa_id
```

- 老師在 Bank 設好題組（prompt、題數 `qa_count`）並標 `for_exam=true`。
- 學生端建一份 Quiz → 挑 Bank 題組加入（**快照**，之後 Bank 改設定不影響已建立的試卷）→ 逐題出題 → 作答 → LLM 批改 → 評分。
- **無追問模式**（與 exam 不同）。

## 2. 新端點總覽（15 支，全在 `/v1/quiz` 下）

| 區塊 | 端點 |
|---|---|
| 試卷 | `GET/POST /v1/quiz/pages`、`PATCH/DELETE /v1/quiz/pages/{quiz_page_id}` |
| 挑題組 | `GET /v1/quiz/bank-groups`（列 `for_exam=true` 的 Bank_Group） |
| 題組 | `POST /v1/quiz/pages/{quiz_page_id}/groups`（body `{bank_group_id}`）、`GET/PATCH/DELETE /v1/quiz/groups/{quiz_group_id}` |
| 出題 | `POST /v1/quiz/groups/{quiz_group_id}/qa/llm-generate`（出下一題）、`POST /v1/quiz/qa/{quiz_qa_id}/llm-regenerate`（原地重出） |
| 批改 | `POST /v1/quiz/qa/{quiz_qa_id}/llm-answer`（202+job_id）→ `GET /v1/quiz/qa/answer-result/{job_id}`（輪詢） |
| 評分／刪題 | `PUT /v1/quiz/qa/{quiz_qa_id}/question-rate`、`PUT .../answer-rate`、`DELETE /v1/quiz/qa/{quiz_qa_id}` |
| LLM 設定 | `GET/PUT /v1/quiz/llm-api-key`、`GET /v1/quiz/llm-api-key/exists`、`GET/PUT /v1/quiz/llm-model` |

驗證與其他模組相同：`Authorization: Bearer <token>` ＋ 必填 query `?course_id=<int>`。

## 3. 前端要特別注意的 5 件事

1. **出題是逐題的**：一份題組要出滿，前端要自己迴圈呼叫 `llm-generate` 共 `qa_count` 次。回應有 `generated_count`／`qa_count` 可判斷進度；出滿再呼叫回 **409**。
2. **LLM 失敗仍回 HTTP 200**：出題／重出失敗時 body 帶 `llm_error` 字串、題目欄位為空。**要判斷 `llm_error`，不能只看狀態碼。**
3. **批改是非同步的**：`llm-answer` 回 202＋`job_id`，輪詢 `answer-result` 直到 `status: ready`（建議 1.5～3 秒一次）；`ready` 時回應的 `quiz_qa` 是整列最新資料（含 `answer_critique`）。404 表示服務重啟，需重送批改。
4. **先決條件**：出題前該課程要設好 `quiz-api-key`（與 bank/exam/rag 的 key **互相獨立**，要分開設定頁）；對應 Bank 單元要已 `build-zip`、題組 `for_exam=true`，否則回 400/404。
5. **評分有兩個**：`question_rate`（對題目）與 `answer_rate`（對批改結果），皆 -1/0/1。重出題（`llm-regenerate`）後該題舊作答／批改／評分會被清空。

## 4. 典型流程（一句話版）

> 建試卷 → `GET /bank-groups` 挑題組 → 加入試卷（快照）→ 迴圈出題到 `qa_count` → 作答＋輪詢批改 → 評分 → `GET /quiz/pages` 重載整份巢狀資料。

## 5. OpenAPI

Swagger（`/docs`）已可直接看到 `quiz` tag 下全部端點，排序在 bank 之後、exam 之前。若用 codegen，新增的 schema 名稱皆以 `Quiz` 開頭（`CreateQuizRequest`、`ListQuizResponse`、`QuizQaAnswerRequest` 等），不與既有名稱衝突。
