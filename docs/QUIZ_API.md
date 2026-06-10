# Quiz（試卷／Test）API 文件

給前端串接用。Quiz 是搭配 **Bank（題庫）** 的「試卷／應試」模組，定位等同 `exam` 之於 `rag`：
先建一份試卷，挑一個既有的 **Bank 題組**快照進來，再在這個題組下**逐題出題**（題數上限＝`qa_count`），最後作答／批改／評分。**無追問模式**。

> 內容（課程講義）來自 Bank：出題／批改讀的是 Bank 單元的 RAG 向量庫或逐字稿。所以使用前，對應的 Bank 單元要先 `build-zip` 完成，且題組要 `for_exam=true`。

---

## 0. 共通約定

- **Base URL**：所有端點都在 `/v1` 之下，例：`https://<host>/v1/quiz/pages`
- **驗證**：每支 API 都要帶
  ```
  Authorization: Bearer <access_token>
  ```
  （`access_token` 由 `POST /v1/auth/login` 取得、`POST /v1/auth/refresh` 換發）。未帶或失效 → **401**。
  後端用 token 解析呼叫者 `person_id`，**不需**自己在 query／body 帶。
- **course_id**：每支 API 都**必填** query 參數 `?course_id=<int>`。未帶 → **400**。
- **擁有權**：只有 `person_id == token 持有者` 的資源才能讀寫，否則 **403**。
- **軟刪**：刪除都是 `deleted=true`，不是真的刪列。
- **常見錯誤碼**：`400` 參數錯、`401` 未登入、`403` 無權、`404` 找不到／已刪、`409` 超過 `qa_count` 上限、`500` 伺服器錯、`502/503` 儲存或 LLM 暫時失敗。
- **LLM 出題／批改失敗的特例**：出題端點若 LLM 呼叫失敗，**仍回 HTTP 200**，但 body 帶 `llm_error` 字串、題目欄位為空字串（見 2.6）。前端要判斷 `llm_error` 是否存在，而非只看 HTTP 狀態碼。

### 資料階層

```
Quiz(試卷/page) ─< Quiz_Group(題組, 自 Bank_Group 快照) ─< Quiz_QA(題目)
   quiz_page_id          quiz_group_id                      quiz_qa_id
                          └ 來源: bank_page_id / bank_unit_id / bank_group_id
```

Quiz_Group 是建立當下從 Bank_Group「快照」過來的：出題／批改 prompt、`qa_count`、模型、單元名稱／類型都複製進 Quiz_Group，**之後改 Bank 不會影響已建立的試卷**。

### 單元類型 `unit_type`（沿用 Bank，影響出題／批改的課程來源）

| 值 | 意義 | 課程來源 |
|----|------|------|
| 1 | RAG（FAISS 向量庫） | 向量檢索片段 |
| 2 / 3 / 4 | 文字 / 音訊 / YouTube | `transcript` 逐字稿（純 LLM） |

### LLM 設定（quiz 專屬，與 bank/exam/rag 完全分開）

出題／批改要先設定 **quiz 自己的**課程層級金鑰與模型（獨立端點、獨立 Course_Setting key）：
- `PUT /v1/quiz/llm-api-key`　body `{ "api_key": "sk-..." }`（key=`quiz-api-key`）
- `PUT /v1/quiz/llm-model`　body `{ "llm_model": "gpt-5.4" }`（key=`quiz-llm-model`；未設定有預設）

題組也可在 `question_llm_model` / `answer_llm_model` 各自指定模型，留空則用上面的課程設定。

---

## 1. 試卷（Quiz）

### 1.1 列出試卷　`GET /v1/quiz/pages`

回傳該 `course_id`、該登入者、`deleted=false` 的所有 Quiz，**巢狀**帶出 groups → qas（qas 依 `question_series_index` 升序）。

**Query**：`course_id`（必）

**Response 200**
```json
{
  "quizzes": [
    {
      "quiz_id": 12,
      "quiz_page_id": "u123_260610153000",
      "person_id": "u123",
      "course_id": 5,
      "tab_name": "第一回測驗",
      "deleted": false,
      "updated_at": "2026-06-10T15:30:00+08:00",
      "created_at": "2026-06-10T15:30:00+08:00",
      "quiz_groups": [
        {
          "quiz_group_id": 34,
          "quiz_page_id": "u123_260610153000",
          "bank_page_id": "u123_250101...",
          "bank_unit_id": 7,
          "bank_group_id": 9,
          "unit_name": "第三章 細胞",
          "unit_type": 1,
          "group_name": "細胞題組",
          "qa_count": 5,
          "question_system_prompt_text": "...",
          "question_user_prompt_text": "...",
          "question_llm_model": "",
          "answer_user_prompt_text": "...",
          "answer_llm_model": "",
          "deleted": false,
          "qas": [ { "quiz_qa_id": 101, "question_series_index": 1, "question_content": "...", "...": "..." } ]
        }
      ]
    }
  ],
  "count": 1
}
```

### 1.2 建立試卷　`POST /v1/quiz/pages`

**Query**：`course_id`（必）
**Body**（皆選填）
```json
{ "quiz_page_id": "", "person_id": "", "tab_name": "第一回測驗" }
```
- `quiz_page_id` 不傳則後端產生；`person_id` 不傳用 token 持有者（有傳須與 token 一致，否則 400）。

**Response 201**：完整的 Quiz 列（同 1.1 單筆，不含 `quiz_groups`）。

### 1.3 更名　`PATCH /v1/quiz/pages/{quiz_page_id}`

**Body** `{ "tab_name": "新名稱" }` → **200** `{ quiz_id, quiz_page_id, tab_name, person_id, updated_at }`

### 1.4 刪除　`DELETE /v1/quiz/pages/{quiz_page_id}`

軟刪（不動底下 group/qa）→ **200** `{ message, quiz_page_id, person_id, updated_at }`

---

## 2. 題組與出題

### 2.1 列出可選用的 Bank 題組　`GET /v1/quiz/bank-groups`

挑題組前先呼叫這支，拿到本課程裡 `for_exam=true`、未刪除的 Bank_Group 清單（含單元名稱／類型）。

**Query**：`course_id`（必）

**Response 200**
```json
{
  "groups": [
    {
      "bank_group_id": 9,
      "bank_page_id": "u123_250101...",
      "bank_unit_id": 7,
      "unit_name": "第三章 細胞",
      "unit_type": 1,
      "group_name": "細胞題組",
      "qa_count": 5,
      "question_system_prompt_text": "...",
      "question_user_prompt_text": "...",
      "answer_user_prompt_text": "...",
      "for_exam": true
    }
  ],
  "count": 1
}
```

### 2.2 把 Bank 題組加入試卷（快照）　`POST /v1/quiz/pages/{quiz_page_id}/groups`

**不呼叫 LLM**。把指定 Bank_Group 的設定快照成一筆 Quiz_Group 掛在此試卷下。

**Query**：`course_id`（必）
**Body**
```json
{ "bank_group_id": 9, "group_name": "" }
```
- `group_name` 留空則沿用 Bank_Group 的名稱。

**Response 201**：完整的 Quiz_Group 列（含 `quiz_group_id`、來源 `bank_*` 鍵、快照的 prompt／`qa_count`／模型）。

### 2.3 讀單一題組　`GET /v1/quiz/groups/{quiz_group_id}`

**Response 200**：Quiz_Group 列 ＋ `qas`（Quiz_QA 陣列，依 `question_series_index` 升序）。

### 2.4 更新題組快照　`PATCH /v1/quiz/groups/{quiz_group_id}`

只更新有傳的欄位（可改 `group_name`、`qa_count`、`question_system_prompt_text`、`question_user_prompt_text`、`question_llm_model`、`answer_user_prompt_text`、`answer_llm_model`）。**200** 回更新後整列。

### 2.5 刪題組　`DELETE /v1/quiz/groups/{quiz_group_id}`

軟刪（不動底下 qa）→ **200** `{ message, quiz_group_id, person_id, updated_at }`

### 2.6 逐題出題　`POST /v1/quiz/groups/{quiz_group_id}/qa/llm-generate`

產生**下一題**（同步、會等 LLM）。用題組既有的 prompt；同題組已出過的題幹會當作「勿重複」一起送進去。**前端要呼叫 `qa_count` 次**才出滿一份。

**Query**：`course_id`（必）
**Body**（皆選填，本次覆寫用，不寫回題組）
```json
{ "question_user_prompt_text": "", "question_system_prompt_text": "" }
```

**Response 200（成功）**
```json
{
  "question_llm_model": "gpt-5.4",
  "qa_count": 5,
  "generated_count": 1,
  "quiz_qa_id": 101,
  "quiz_group_id": 34,
  "quiz_page_id": "u123_260610153000",
  "question_series_index": 1,
  "question_content": "題幹…",
  "question_hint": "提示…",
  "question_answer_reference": "參考答案…",
  "question_reason": "出題理由…",
  "question_rate": 0,
  "answer_content": "",
  "answer_critique": null,
  "answer_rate": 0,
  "...": "其餘 Quiz_QA 欄位"
}
```
- `generated_count` = 目前這份已出到第幾題。前端可用 `generated_count >= qa_count` 判斷出滿。

**Response 409**：已達 `qa_count` 上限（`{ "detail": "本題組已達 qa_count 上限（5 題），無法再出題" }`）。

**Response 200（LLM 失敗特例）**：HTTP 仍 200，但帶 `llm_error`：
```json
{ "llm_error": "…錯誤訊息…", "quiz_group_id": 34, "question_content": "", "question_hint": "", "question_answer_reference": "", "question_llm_model": "gpt-5.4" }
```

### 2.7 原地重出同一題　`POST /v1/quiz/qa/{quiz_qa_id}/llm-regenerate`

只重產這一題的 `question_*` 並覆寫回同一 `quiz_qa_id`（不新增、不改 `question_series_index`、不檢查上限）。同題組「此題之前」的題作為勿重複。重出後該題舊的作答／批改／評分會清空。Body、回應同 2.6。

---

## 3. 作答、批改、評分

### 3.1 送出作答（非同步批改）　`POST /v1/quiz/qa/{quiz_qa_id}/llm-answer`

**Query**：`course_id`（必）
**Body** `{ "answer_content": "學生作答文字" }`

**Response 202**
```json
{ "job_id": "f3c2…", "answer_llm_model": "gpt-5.4" }
```
拿到 `job_id` 後輪詢 3.2。

### 3.2 取批改結果　`GET /v1/quiz/qa/answer-result/{job_id}`

**Query**：`course_id`（必）。建議每 1.5～3 秒輪詢一次。

**Response 200**
```json
{
  "status": "pending | ready | error",
  "result": { "...": "LLM 原始批改結果（ready 時）" },
  "error": null,
  "llm_error": null,
  "quiz_qa": { "quiz_qa_id": 101, "answer_content": "…", "answer_critique": "批改評語…", "...": "整列 Quiz_QA" }
}
```
- `status=ready`：批改完成，`quiz_qa` 為回讀的整列（含 `answer_critique`）。
- `status=pending`：還在跑，繼續輪詢。
- `status=error` 或有 `llm_error`：批改失敗。
- **404**：`job_id` 不存在（多半是服務重啟／冷啟動，請重送 3.1）。

### 3.3 評分

| 端點 | Body | 200 回應 |
|---|---|---|
| `PUT /v1/quiz/qa/{quiz_qa_id}/question-rate` | `{ "question_rate": 1 }` | `{ quiz_qa_id, question_rate, updated_at }` |
| `PUT /v1/quiz/qa/{quiz_qa_id}/answer-rate` | `{ "answer_rate": 1 }` | `{ quiz_qa_id, answer_rate, updated_at }` |

`rate` 僅接受 `-1` / `0` / `1`。

### 3.4 刪題　`DELETE /v1/quiz/qa/{quiz_qa_id}`

軟刪 → **200** `{ message, quiz_qa_id, quiz_group_id, person_id, updated_at }`

---

## 3b. 追問（對題組對應的 Bank 課程內容發問）

出題後，學生若對課程內容仍有不懂，可針對**整個題組**對應的 Bank 單元內容發問（不綁單題）。同步呼叫 LLM，依課程內容（逐字稿／向量檢索）作答，每問存一列 `Quiz_Ask`。

### 3b.1 發問　`POST /v1/quiz/groups/{quiz_group_id}/llm-ask`

**Query**：`course_id`（必）
**Body** `{ "ask_user_prompt_text": "想再問：粒線體與葉綠體的內共生證據有哪些？" }`（亦接受 `ask`／`question` 別名）

**Response 200**：新增的 Quiz_Ask 列
```json
{
  "quiz_ask_id": 5,
  "quiz_group_id": 34,
  "quiz_page_id": "u123_260610153000",
  "bank_page_id": "u123_250101...",
  "bank_unit_id": 7, "bank_group_id": 9,
  "unit_name": "第三章 細胞", "unit_type": 1, "group_name": "細胞題組",
  "ask_user_prompt_text": "想再問：…",
  "answer_content": "依課程內容，內共生證據包括…（Markdown 純文字）",
  "answer_rate": 0,
  "created_at": "2026-06-10T16:00:00+08:00"
}
```
LLM 失敗時 HTTP 仍 200，帶 `{ "llm_error": "...", "quiz_group_id": 34, "answer_content": "" }`。

### 3b.2 列出歷次提問　`GET /v1/quiz/groups/{quiz_group_id}/asks`

**Response 200** `{ "asks": [ Quiz_Ask, ... ], "count": N }`（依 `created_at` 由舊到新）

### 3b.3 評分／刪除

| 方法 | 路徑 | Body | 說明 |
|---|---|---|---|
| PUT | `/v1/quiz/asks/{quiz_ask_id}/answer-rate` | `{ "answer_rate": 1 }` | 對回答評分（-1/0/1） |
| DELETE | `/v1/quiz/asks/{quiz_ask_id}` | — | 軟刪單筆提問 |

---

## 3.5 追問（對 Bank 課程內容發問）

出題後，使用者可針對**該題組對應的 Bank 單元課程內容**自由發問（不綁特定題目）。每問一次新增一列 `Quiz_Ask`。

### 發問　`POST /v1/quiz/groups/{quiz_group_id}/llm-ask`

同步（會等 LLM 數秒）。`unit_type` 2/3/4 用逐字稿回答，其餘用該單元 RAG ZIP 檢索後回答。

**Query**：`course_id`（必）
**Body**
```json
{ "ask_user_prompt_text": "想再問：粒線體與葉綠體的內共生證據有哪些？" }
```
（也接

| 方法 | 路徑 | Body | 說明 |
|---|---|---|---|
| GET | `/v1/quiz/llm-api-key/exists` | — | `{ course_id, exists }` 是否已設金鑰 |
| GET | `/v1/quiz/llm-api-key` | — | 讀金鑰（需權限）`{ course_setting_id, course_id, api_key }` |
| PUT | `/v1/quiz/llm-api-key` | `{ "api_key": "sk-..." }` | 寫金鑰（key=`quiz-api-key`） |
| GET | `/v1/quiz/llm-model` | — | 讀模型 `{ course_setting_id, course_id, llm_model }` |
| PUT | `/v1/quiz/llm-model` | `{ "llm_model": "gpt-5.4" }` | 寫模型（key=`quiz-llm-model`，出題／批改共用） |

> 讀／寫金鑰與模型需該課程的開發者／管理者權限，否則 403。`exists` 一般成員可呼叫。

---

## 5. 典型前端流程

```
（一次性）PUT /v1/quiz/llm-api-key                       設定金鑰
1. POST   /v1/quiz/pages                                 建一份試卷 → 取得 quiz_page_id
2. GET    /v1/quiz/bank-groups                           列可選 Bank 題組
3. POST   /v1/quiz/pages/{quiz_page_id}/groups           選 bank_group_id 加入 → 取得 quiz_group_id（含 qa_count）
4. 迴圈 qa_count 次：
   POST   /v1/quiz/groups/{quiz_group_id}/qa/llm-generate   逐題出題（檢查 llm_error；滿了會 409）
5. 學生作答：
   POST   /v1/quiz/qa/{quiz_qa_id}/llm-answer            → job_id
   GET    /v1/quiz/qa/answer-result/{job_id}             輪詢到 status=ready
6. PUT    /v1/quiz/qa/{quiz_qa_id}/question-rate|answer-rate   評分（選用）
7. GET    /v1/quiz/pages                                 重新載入整份試卷（巢狀 groups→qas）
```

每支請求記得帶 `Authorization: Bearer <token>` 與 `?course_id=<int>`。
