# 前端異動通知：Quiz 分析 LLM 設定獨立

> 日期：2026-06-13  
> 影響：**Quiz 分析模組**（`user-analyses`、`quiz-analyses`）的 LLM API Key／模型設定路徑變更。  
> 完整串接文件：`docs/QUIZ_ANALYSIS_API.md`

---

## 1. 改了什麼

先前 **User Analysis** / **Quiz Analysis** 的 `POST …/llm-analysis` 共用 Quiz 出題設定：

- 金鑰：`PUT /v1/quiz/llm-api-key`（key=`quiz-api-key`）
- 模型：`PUT /v1/quiz/llm-model`（key=`quiz-llm-model`）

現在改為**各自獨立**，與 Quiz 出題／批改分開設定：

| 模組 | API Key 路徑 | Course_Setting key | LLM 模型路徑 | Course_Setting key |
|------|-------------|-------------------|-------------|-------------------|
| **個人弱點分析** | `/v1/user-analyses/llm-api-key` | `user-analysis-api-key` | `/v1/user-analyses/llm-model` | `user-analysis-llm-model` |
| **課程弱點分析** | `/v1/quiz-analyses/llm-api-key` | `quiz-analysis-api-key` | `/v1/quiz-analyses/llm-model` | `quiz-analysis-llm-model` |
| Quiz 出題／批改（不變） | `/v1/quiz/llm-api-key` | `quiz-api-key` | `/v1/quiz/llm-model` | `quiz-llm-model` |

---

## 2. 新增端點

端點形狀與 `/v1/bank/llm-api-key`、`/v1/bank/llm-model` **完全相同**，只是前綴不同。

### User Analysis

| Method | Path | 說明 |
|--------|------|------|
| GET | `/v1/user-analyses/llm-api-key/exists` | 是否已設定金鑰（**不回傳 key**；一般成員可呼叫） |
| GET | `/v1/user-analyses/llm-api-key` | 讀取金鑰（需開發者／管理者權限） |
| PUT | `/v1/user-analyses/llm-api-key` | 寫入金鑰 |
| GET | `/v1/user-analyses/llm-model` | 讀取模型（需開發者／管理者權限） |
| PUT | `/v1/user-analyses/llm-model` | 寫入模型 |

### Quiz Analysis（課程彙整）

| Method | Path | 說明 |
|--------|------|------|
| GET | `/v1/quiz-analyses/llm-api-key/exists` | 是否已設定金鑰（**不回傳 key**；一般成員可呼叫） |
| GET | `/v1/quiz-analyses/llm-api-key` | 讀取金鑰（需開發者／管理者權限） |
| PUT | `/v1/quiz-analyses/llm-api-key` | 寫入金鑰 |
| GET | `/v1/quiz-analyses/llm-model` | 讀取模型（需開發者／管理者權限） |
| PUT | `/v1/quiz-analyses/llm-model` | 寫入模型 |

**共通 query**（與其他模組相同）：

```
?course_id=<int>
Authorization: Bearer <access_token>
```

---

## 3. Request / Response 範例

### 查是否已設定金鑰（學生端提示用）

```http
GET /v1/user-analyses/llm-api-key/exists?course_id=1
Authorization: Bearer <token>
```

```json
{
  "course_id": 1,
  "exists": false
}
```

### 寫入金鑰（教師／管理者設定頁）

```http
PUT /v1/user-analyses/llm-api-key?course_id=1
Authorization: Bearer <token>
Content-Type: application/json

{ "api_key": "sk-..." }
```

**Response 200**

```json
{
  "course_setting_id": 42,
  "course_id": 1,
  "api_key": "sk-..."
}
```

### 寫入模型

```http
PUT /v1/user-analyses/llm-model?course_id=1
Authorization: Bearer <token>
Content-Type: application/json

{ "llm_model": "gpt-5.4" }
```

**Response 200**

```json
{
  "course_setting_id": 43,
  "course_id": 1,
  "llm_model": "gpt-5.4"
}
```

未設定模型時，`llm-analysis` 會 fallback 後端預設值（與 bank/quiz 相同）；`GET …/llm-model` 未設定時 `llm_model` 可為 `null`。

---

## 4. 前端要改的地方

### 4.1 設定頁 UI

若分析功能有獨立的「LLM 設定」區塊，請**不要**再綁 `/v1/quiz/llm-api-key` 或 `/v1/quiz/llm-model`：

| 功能頁 | 金鑰端點 | 模型端點 |
|--------|---------|---------|
| 個人弱點分析（user-analyses） | `PUT /v1/user-analyses/llm-api-key` | `PUT /v1/user-analyses/llm-model` |
| 課程弱點分析（quiz-analyses） | `PUT /v1/quiz-analyses/llm-api-key` | `PUT /v1/quiz-analyses/llm-model` |
| Quiz 出題／批改 | `PUT /v1/quiz/llm-api-key`（不變） | `PUT /v1/quiz/llm-model`（不變） |

Key 與模型皆**互不共用**，設定頁需分開存取。

### 4.2 產生報告前的檢查

呼叫 `POST …/llm-analysis` 前，建議先打對應的 `…/llm-api-key/exists`：

- 個人分析 → `GET /v1/user-analyses/llm-api-key/exists`
- 課程分析 → `GET /v1/quiz-analyses/llm-api-key/exists`

`exists: false` 時提示老師先設定 API Key，避免 LLM 呼叫失敗。

### 4.3 錯誤訊息（`llm_error` 字串）

未設定金鑰時，`POST …/llm-analysis` 仍回 **HTTP 200**，body 的 `llm_error` 會帶：

| 模組 | `llm_error` 前綴 |
|------|-----------------|
| user-analyses | `未設定 API Key：PUT /v1/user-analyses/llm-api-key…` |
| quiz-analyses | `未設定 API Key：PUT /v1/quiz-analyses/llm-api-key…` |

前端可依路徑導向正確的設定頁。

---

## 5. 典型流程（含一次性設定）

```
（教師，一次性）
PUT /v1/user-analyses/llm-api-key
PUT /v1/user-analyses/llm-model
PUT /v1/quiz-analyses/llm-api-key
PUT /v1/quiz-analyses/llm-model
PUT /v1/user-analyses/analysis-user-prompt-text
PUT /v1/quiz-analyses/analysis-user-prompt-text

（學生／教師，分析流程）
1. GET  /v1/{user-analyses|quiz-analyses}/llm-api-key/exists   → 確認已設定
2. POST /v1/{prefix}                                            → 建空白分析列
3. POST /v1/{prefix}/{id}/llm-analysis                          → 產生報告
4. GET  /v1/{prefix}                                            → 讀取結果
```

---

## 6. 權限

| 端點 | 一般成員 | 開發者／管理者 |
|------|---------|---------------|
| `GET …/llm-api-key/exists` | ✅ | ✅ |
| `GET …/llm-api-key` | ❌ 403 | ✅ |
| `PUT …/llm-api-key` | ❌ 403 | ✅ |
| `GET …/llm-model` | ❌ 403 | ✅ |
| `PUT …/llm-model` | ❌ 403 | ✅ |

與 `/v1/bank/llm-api-key`、`/v1/bank/llm-model` 權限規則相同。

---

## 7. 與其他分析模組對照

| 模組 | 資料來源 | API Key | LLM 模型 |
|------|---------|---------|----------|
| person-analyses（RAG 個人） | Exam_Quiz | `PUT /v1/exam/llm-api-key` | `PUT /v1/rag/llm-model` |
| course-analyses（RAG 課程） | Exam_Quiz | `PUT /v1/course-analyses/llm-api-key` | `PUT /v1/rag/llm-model` |
| **user-analyses（Quiz 個人）** | Quiz_QA | **`PUT /v1/user-analyses/llm-api-key`** | **`PUT /v1/user-analyses/llm-model`** |
| **quiz-analyses（Quiz 課程）** | Quiz_QA | **`PUT /v1/quiz-analyses/llm-api-key`** | **`PUT /v1/quiz-analyses/llm-model`** |

---

## 8. OpenAPI

Swagger（`/docs`）已新增端點，tag 分別在 **user analysis**、**quiz analysis** 下。  
operation_id 範例：`user_analysis_get_llm_model`、`quiz_analysis_put_llm_model` 等。
