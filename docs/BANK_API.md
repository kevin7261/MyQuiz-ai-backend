# Bank（測試題庫）API 文件

給前端串接用。Bank 是獨立的題庫模組，結構複製自 RAG，但**出題以「題組（Group）」為單位、無追問**。

---

## 0. 共通約定

- **Base URL**：所有端點都在 `/v1` 之下，例：`https://<host>/v1/bank/pages`
- **驗證**：每支 API 都要帶
  ```
  Authorization: Bearer <access_token>
  ```
  （`access_token` 由 `POST /v1/auth/login` 取得、`POST /v1/auth/refresh` 換發）。未帶或失效 → **401**。
  後端用 token 解析出呼叫者 `person_id`，**不需**在 query／body 自己帶。
- **course_id**：每支 API 都**必填** query 參數 `?course_id=<int>`。未帶 → **400**。
- **擁有權**：只有資源的 `person_id == token 持有者` 才能讀寫，否則 **403**。
- **軟刪**：刪除都是 `deleted=true`，不是真的刪列。
- **常見錯誤碼**：`400` 參數錯、`401` 未登入、`403` 無權、`404` 找不到/已刪、`409` 超過題數上限、`500` 伺服器錯、`502/503` 儲存或 LLM 暫時失敗。

### 資料階層

```
Bank(題庫頁面/page) ─< Bank_Unit(單元) ─< Bank_Group(題組) ─< Bank_QA(題目)
        bank_page_id        bank_unit_id        bank_group_id
```

### 單元類型 `unit_type`

| 值 | 意義 | 出題/批改的課程來源 |
|----|------|------|
| 0 | 未選 | （需先 build-zip） |
| 1 | RAG（FAISS 向量庫） | 向量檢索片段 |
| 2 | 文字 | `transcript` 逐字稿 |
| 3 | 音訊 mp3 | `transcript` 逐字稿 |
| 4 | YouTube | `transcript` 逐字稿 |

### LLM 設定（bank 專屬，與 RAG 完全分開）

出題／批改要先設定 **bank 自己的** 課程層級金鑰與模型（獨立端點、獨立 Course_Setting key，與 rag 無關）：
- `PUT /v1/bank/llm-api-key`　body `{ "api_key": "sk-..." }`（key=`bank-api-key`）
- `PUT /v1/bank/llm-model`　body `{ "llm_model": "gpt-5.4" }`（key=`bank-llm-model`；未設定有預設）

詳見第 3 章。題組也可在 `question_llm_model` / `answer_llm_model` 各自指定模型，留空則用上面的課程設定。
出題／批改的 prompt 也是 bank 專屬（後端 `services/bank_generation.py`、`services/bank_answering.py`），與 rag 無關。

---

## 1. 題庫頁面與單元（檔案管理）

### 1.1 列出題庫頁面 `GET /v1/bank/pages`

回傳該 `course_id`、該登入者、`deleted=false` 的所有 Bank，**巢狀**帶出 units → groups → qas。

**Query**：`course_id`（必）

**Response 200**
```json
{
  "banks": [
    {
      "bank_id": 1,
      "bank_page_id": "abc",
      "person_id": "u_123",
      "course_id": 10,
      "tab_name": "第一章",
      "file_size": 1.2,
      "file_metadata": { "...": "..." },
      "rag_metadata": { "...": "..." },
      "deleted": false,
      "updated_at": "2026-06-09T12:00:00",
      "created_at": "2026-06-09T11:00:00",
      "units": [
        {
          "bank_unit_id": 5,
          "bank_page_id": "abc",
          "unit_name": "單元一",
          "unit_type": 2,
          "transcript": "...",
          "mp3_audio_url": "/bank/units/5/mp3-file?bank_page_id=abc&course_id=10",
          "youtube_url_api": "/bank/units/5/youtube-url?bank_page_id=abc&course_id=10",
          "groups": [
            {
              "bank_group_id": 7,
              "bank_unit_id": 5,
              "group_name": "第一回測驗",
              "qa_count": 5,
              "for_exam": false,
              "qas": [
                { "bank_qa_id": 11, "question_series_index": 1, "question_content": "...", "answer_content": "...", "answer_critique": "..." }
              ]
            }
          ]
        }
      ]
    }
  ],
  "count": 1
}
```
> `mp3_audio_url` 僅音訊單元(3)且有檔時出現；`youtube_url_api` 僅 YouTube 單元(4)出現。皆為相對於後端 origin 的路徑。

### 1.2 建立題庫並上傳 ZIP `POST /v1/bank/pages/upload-zip`

`multipart/form-data`。

**Query**：`course_id`（必）
**Form 欄位**：`file`（.zip，必）、`bank_page_id`（必）、`tab_name`（必）、`person_id`（選，未傳用 token）

**Response 201**
```json
{
  "bank_id": 1, "bank_page_id": "abc", "tab_name": "第一章",
  "person_id": "u_123", "course_id": 10,
  "created_at": "2026-06-09T11:00:00",
  "file_metadata": { "bank_page_id": "abc", "filename": "x.zip", "second_folders": ["..."], "file_size": 1.2 }
}
```

### 1.3 更新頁面名稱 `PATCH /v1/bank/pages/{bank_page_id}`

**Query**：`course_id`　**Body**：`{ "tab_name": "新名稱" }`
**Response 200**：`{ bank_id, bank_page_id, person_id, tab_name, updated_at }`

### 1.4 刪除頁面 `DELETE /v1/bank/pages/{bank_page_id}`

軟刪該頁面、其所有單元，並刪除 storage 資料夾。
**Query**：`course_id`
**Response 200**：`{ message, bank_page_id, person_id, bank_updated, folder_deleted }`

### 1.5 打包／建 RAG `POST /v1/bank/pages/{bank_page_id}/build-zip`

依先前上傳的 ZIP 與 `unit_list` 重新打包；整批成功時自動建立對應 `Bank_Unit` 並更新 `Bank.rag_metadata`。

**Query**：`course_id`、`repack_only`（選，bool）
**Body**（JSON）
```json
{
  "unit_list": "folder1+folder2",
  "unit_names": "",
  "unit_types": "",
  "transcripts": null,
  "rag_chunk_size": 1000,
  "rag_chunk_overlap": 200,
  "rag_chunk_sizes": "",
  "rag_chunk_overlaps": "",
  "build_faiss": null
}
```
**Response**：**NDJSON 串流**（`application/x-ndjson`，HTTP 固定 200）。請用 `fetch` 讀 `response.body` 逐行解析，**不要**用 `response.json()`。事件：
- `{"type":"start", ...}`
- `{"type":"building","index":i,"total":N, ...}`
- `{"type":"unit","index":i,"output":{...}}`
- `{"type":"complete","success":true|false,"outputs":[...], ...}` ← 以最後這筆的 `success` 判斷整批成敗。

### 1.6 列出單元 `GET /v1/bank/pages/{bank_page_id}/units`

**Query**：`course_id`
**Response 200**：`{ "units": [ { ...Bank_Unit, "groups": [ { ...group, "qas": [...] } ] } ], "count": n }`

### 1.7 取單元內容

不需 person_id（後端依 `bank_page_id` 解析擁有者）。

| 端點 | 適用 unit_type | 回傳重點 |
|------|------|------|
| `GET /v1/bank/units/{bank_unit_id}/text` | 2 文字 | `text_file_name`, `transcript` |
| `GET /v1/bank/units/{bank_unit_id}/mp3-file` | 3 音訊 | `audio_base64`, `media_type`, `filename`, `transcript` |
| `GET /v1/bank/units/{bank_unit_id}/youtube-url` | 4 YouTube | `youtube_url`, `text_file_name`, `transcript` |

**Query（三者共通）**：`course_id`（必）、`bank_page_id`（必）

### 1.8 建置前預覽（單元尚未建立時，直接讀 upload ZIP）

| 端點 | 回傳 |
|------|------|
| `GET /v1/bank/pages/{bank_page_id}/unit-preview/text` | `text_file_name`, `transcript` |
| `GET /v1/bank/pages/{bank_page_id}/unit-preview/mp3-file` | `audio_base64`, `media_type`, `filename`, `text_file_name`, `transcript` |
| `GET /v1/bank/pages/{bank_page_id}/unit-preview/youtube-url` | `youtube_url`, `text_file_name`, `transcript` |

**Query（共通）**：`course_id`（必）、`folder_name`（必，upload ZIP 內資料夾名）。呼叫者須為該頁面擁有者。

---

## 2. 題組與題目（LLM 出題／批改）

> URL 慣例：**建立／列表**巢狀在 page/unit 下；**單一資源**（題組、題目）用自己的主鍵走淺路徑。

### 2.1 建立題組 `POST /v1/bank/pages/{bank_page_id}/units/{bank_unit_id}/groups`

不呼叫 LLM，只建一個題組並設定出題規則與題數上限。

**Query**：`course_id`
**Body**
```json
{
  "group_name": "第一回測驗",
  "qa_count": 5,
  "question_system_prompt_text": "請連續出題，題目越來越深入且彼此不重複。",
  "question_user_prompt_text": "請就課程內容出一道問答題。",
  "question_llm_model": "",
  "answer_user_prompt_text": "請依參考答案批改，指出學生答得不足之處。",
  "answer_llm_model": "",
  "for_exam": false
}
```
| 欄位 | 說明 |
|------|------|
| `qa_count` | 本題組要出的**題數上限**（0＝不限） |
| `question_system_prompt_text` | **連續出題規定**（最高優先，例：越來越難、勿重複） |
| `question_user_prompt_text` | 出題 user prompt |
| `answer_user_prompt_text` | 批改 user prompt |
| `question_llm_model` / `answer_llm_model` | 各自指定模型，空則用課程設定 |

**Response 201**：建立的 Bank_Group 整列（含 `bank_group_id`）。

### 2.2 列出題組 `GET /v1/bank/pages/{bank_page_id}/units/{bank_unit_id}/groups`

**Query**：`course_id`
**Response 200**：`{ "groups": [ { ...Bank_Group, "qas": [...] } ], "count": n }`

### 2.3 讀單一題組 `GET /v1/bank/groups/{bank_group_id}`

**Query**：`course_id`
**Response 200**：Bank_Group 整列 + `"qas": [...]`（依 `question_series_index` 升序）。

### 2.4 更新題組 `PATCH /v1/bank/groups/{bank_group_id}`

**只更新有傳入的欄位**（沒傳的欄位不會被動到，可單獨更新任一欄）。
**Query**：`course_id`
**可更新欄位（任意子集）**：

| 欄位 | 說明 |
|------|------|
| `group_name` | 題組顯示名稱 |
| `qa_count` | 題數上限 |
| `question_system_prompt_text` | 連續出題規定（出題 system prompt） |
| `question_user_prompt_text` | 出題 user prompt |
| `question_llm_model` | 出題 LLM 模型 |
| `answer_user_prompt_text` | 批改 user prompt |
| `answer_llm_model` | 批改 LLM 模型 |

**Body**（任意子集；下例為全欄位，只送要改的即可）
```json
{
  "group_name": "新名稱",
  "qa_count": 8,
  "question_system_prompt_text": "請連續出題，題目越來越深入且彼此不重複。",
  "question_user_prompt_text": "請就課程內容出一道問答題。",
  "question_llm_model": "",
  "answer_user_prompt_text": "請依參考答案批改，指出學生答得不足之處。",
  "answer_llm_model": ""
}
```
例如只改連續出題規定：`{ "question_system_prompt_text": "每題換一個考查角度" }`。
**Response 200**：更新後整列。

### 2.5 設定 for_exam `PUT /v1/bank/groups/{bank_group_id}/for-exam`

**Query**：`course_id`　**Body**：`{ "for_exam": true }`
**Response 200**：更新後整列。

### 2.6 刪除題組 `DELETE /v1/bank/groups/{bank_group_id}`

**僅軟刪此題組**（deleted=true）；**不會**動到底下的 Bank_QA（如需刪題請個別呼叫 `DELETE /v1/bank/qa/{bank_qa_id}`）。
**Query**：`course_id`
**Response 200**：`{ message, bank_group_id, person_id, updated_at }`

### 2.7 出下一題 `POST /v1/bank/groups/{bank_group_id}/qa/llm-generate`

**同步**呼叫 LLM 出**一道**新題並寫入 Bank_QA。使用題組既有的 `question_system_prompt_text` + `question_user_prompt_text`；同題組已出過的題幹（出題歷史）會放進 system 作為「接續出題依據／勿重複」。
**理由先行**：同一次呼叫即一併產出 `question_reason`（出題理由）——模型先依「出題歷史＋題組規則(question_system_prompt_text)＋出題規則(question_user_prompt_text)＋課程內容」決定出題理由，再據此寫題；`question_reason` 會說明本題如何呼應這些規則與歷史（若有）。

**Query**：`course_id`
**Body**（皆選填，非空才覆寫本次，不寫回題組）
```json
{ "question_user_prompt_text": "", "question_system_prompt_text": "" }
```
**Response 200**
```json
{
  "question_llm_model": "gpt-5.4",
  "qa_count": 5,
  "generated_count": 1,
  "bank_qa_id": 11,
  "bank_group_id": 7,
  "bank_unit_id": 5,
  "bank_page_id": "abc",
  "question_series_index": 1,
  "question_system_prompt_text": "（出題當下自題組複製、凍結）",
  "question_user_prompt_text": "（出題當下自題組複製、凍結）",
  "question_content": "...",
  "question_hint": "...",
  "question_answer_reference": "...",
  "question_reason": "本題考察…",
  "answer_user_prompt_text": "（出題當下自題組複製、凍結，供批改用）",
  "answer_llm_model": "",
  "answer_content": "",
  "answer_critique": null,
  "created_at": "..."
}
```
> 出題時：`question_system_prompt_text`／`question_user_prompt_text`／`answer_user_prompt_text` 為自題組複製的凍結 prompt；`question_llm_model` 為本次出題用的模型。`answer_llm_model` 出題時為空，**批改完成後**才填入批改實際用的模型（同時 `answer_content`／`answer_critique` 也在批改後填）。
- 已出題數達 `qa_count` 上限 → **409**。
- LLM 呼叫失敗 → 回 JSON `{ "llm_error": "...", "bank_group_id": 7, "question_content": "", ... }`（HTTP 200，前端以 `llm_error` 判斷）。

### 2.8 批改作答 `POST /v1/bank/qa/{bank_qa_id}/llm-answer`

**非同步**。使用該題所屬題組的 `answer_user_prompt_text` 批改學生作答，結果寫回 Bank_QA。

**Query**：`course_id`　**Body**：`{ "answer_content": "學生作答文字" }`
**Response 202**：`{ "job_id": "uuid", "answer_llm_model": "gpt-5.4" }`
→ 接著輪詢下面 2.9。

### 2.9 取批改結果 `GET /v1/bank/qa/answer-result/{job_id}`

**Query**：`course_id`
**Response 200**
```json
{
  "status": "pending | ready | error",
  "result": { "quiz_comments": ["..."], "bank_qa_id": 11 },
  "error": null,
  "llm_error": null,
  "bank_qa": { "bank_qa_id": 11, "answer_content": "...", "answer_critique": "...", "...": "..." }
}
```
- `status=ready` 時 `bank_qa` 為批改後整列。
- `status=ready` 但 `llm_error` 非空 → LLM 批改失敗。
- 找不到 job（服務重啟等）→ HTTP 404 + `status=error`。

> 輪詢建議：送出後每 1–2 秒打一次，直到 `status` 變 `ready` 或 `error`。

### 2.10 刪除題目 `DELETE /v1/bank/qa/{bank_qa_id}`

**Query**：`course_id`
**Response 200**：`{ message, bank_qa_id, bank_group_id, person_id, updated_at }`

---

## 3. LLM 設定端點（bank 專屬）

權限：`exists` 須登入；讀寫金鑰/模型須該課程的 developer／manager。

| 方法 | 路徑 | 說明 |
|------|------|------|
| GET | `/v1/bank/llm-api-key/exists` | 查是否已設定金鑰（不回傳內容）→ `{ course_id, exists }` |
| GET | `/v1/bank/llm-api-key` | 讀金鑰 → `{ course_setting_id?, course_id, api_key? }` |
| PUT | `/v1/bank/llm-api-key` | 寫金鑰，body `{ "api_key": "sk-..." }` |
| GET | `/v1/bank/llm-model` | 讀模型 → `{ course_setting_id?, course_id, llm_model? }` |
| PUT | `/v1/bank/llm-model` | 寫模型，body `{ "llm_model": "gpt-5.4" }` |

**Query（皆必填）**：`course_id`。這些是 bank 專屬設定（Course_Setting key=`bank-api-key`／`bank-llm-model`），與 rag 的 `/v1/rag/llm-*` 互不影響。

---

## 4. 欄位參考

### Bank_Group
`bank_group_id, bank_page_id, bank_unit_id, person_id, course_id, group_name, question_system_prompt_text, question_user_prompt_text, qa_count, question_llm_model, answer_user_prompt_text, answer_llm_model, for_exam, deleted, updated_at, created_at`

### Bank_QA
`bank_qa_id, bank_page_id, bank_unit_id, bank_group_id, person_id, course_id, question_series_index, question_system_prompt_text, question_user_prompt_text, question_content, question_hint, question_answer_reference, question_reason, question_llm_model, answer_user_prompt_text, answer_llm_model, answer_content, answer_critique, deleted, updated_at, created_at`

> **prompt 凍結 / model 記錄**（隨 QA 一起回傳：GET pages/units、GET group、llm-generate 回應、answer-result）：
> - **prompt**：QA 列記「這一題各次 LLM 呼叫**實際用了什麼**」——`question_system_prompt_text`／`question_user_prompt_text` 於**出題／重出當下**寫入本次出題實際用的規則；`answer_user_prompt_text` 出題當下先複製題組值，**批改完成後覆寫為本次批改實際使用的規則**（重出時重置為題組現值）。出題／重出／批改**每次都用題組現值**（批改時重抓 Bank_Group；題組欄位空才回退此題快照，舊資料相容）。
> - **model**：記「該次 LLM 呼叫實際用的模型」——`question_llm_model` 出題後即填；`answer_llm_model` **批改完成後才填**（出題當下為空字串，批改後才有值）。批改模型取「題組 `answer_llm_model` 現值，空則課程 `bank-llm-model` 現值」，不沿用 QA 上次批改的模型。

---

## 5. 典型前端流程

1. **設定金鑰/模型**（一次）：`PUT /v1/bank/llm-api-key`、`PUT /v1/bank/llm-model`。
2. **建題庫**：`POST /v1/bank/pages/upload-zip` 上傳 ZIP → `POST /v1/bank/pages/{bank_page_id}/build-zip`（讀 NDJSON 串流）建立單元。
3. **建題組**：`POST /v1/bank/pages/{bank_page_id}/units/{bank_unit_id}/groups`，設好 `qa_count` 與 prompt，拿到 `bank_group_id`。
4. **逐題出題**：重複 `POST /v1/bank/groups/{bank_group_id}/qa/llm-generate`，直到拿到想要的題數或回 409（達上限）。
5. **作答批改**：`POST /v1/bank/qa/{bank_qa_id}/llm-answer` → 輪詢 `GET /v1/bank/qa/answer-result/{job_id}`。
6. **檢視**：`GET /v1/bank/pages`（巢狀全部）或 `GET /v1/bank/groups/{bank_group_id}`（單一題組含題目）。
