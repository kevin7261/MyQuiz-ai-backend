# MyQuiz.ai Backend

[MyQuiz.ai](https://myquiz-ai.vercel.app) 的 **FastAPI 後端**：課程測驗平台之 REST API，負責使用者與課程管理、教材 ZIP 上傳與 RAG 向量庫建置、LLM 自動出題／批改、測驗（Exam）管理，以及個人／課程弱點分析報告。

---

## 目錄

- [專案概覽](#專案概覽)
- [系統架構](#系統架構)
- [技術棧](#技術棧)
- [目錄結構](#目錄結構)
- [資料模型](#資料模型)
- [核心業務流程](#核心業務流程)
- [環境變數](#環境變數)
- [本機開發](#本機開發)
- [部署（Render）](#部署render)
- [認證與授權](#認證與授權)
- [API 回傳格式](#api-回傳格式)
- [API 目錄](#api-目錄)
- [API 詳細文件](#api-詳細文件)

---

## 專案概覽

MyQuiz.ai 是一套 **AI 輔助測驗與學習分析** 系統。教師上傳課程教材（Office、PDF、Markdown、音訊、YouTube 逐字稿等），後端依單元類型建置 FAISS 向量庫或逐字稿 ZIP，再透過 OpenAI **gpt-5.4** 自動出題、批改與產生弱點報告；學生則在 Exam 模組完成測驗並取得 AI 評語。

| 模組 | 說明 |
|------|------|
| **使用者／課程** | Supabase Auth + `User`／`User_Course_Relation` 管理登入、選課與角色 |
| **RAG（`/rag`）** | 教材 ZIP 上傳、建庫、單元管理、練習題出題與批改 |
| **Exam（`/exam`）** | 正式測驗卷、從 RAG 匯入題目、追問出題、評分與題目評價 |
| **分析** | 個人／課程層級弱點報告（LLM Markdown） |
| **Log** | 每次業務 API 請求寫入 `Log` 表供稽核 |

**重要資料約定**

- 學生作答欄位：請求 body 使用 **`quiz_answer`**（仍相容舊欄位 `answer`）；寫入 DB 欄位 **`answer_content`**。
- 批改評語寫入 **`answer_critique`**（純文字 Markdown，非 JSON）。
- 已無獨立 `Rag_Answer`／`Exam_Answer` 表，亦無 `quiz_grade`／`answer_grade` 數值評分欄位。
- LLM API Key 存於 **`Course_Setting`**（依 `course_id`：`rag-api-key`、`exam-api-key`），以 GET/PUT `/rag/llm_api_key`、`/exam/api_key` 管理。
- LLM 模型（出題、批改、弱點分析共用）存於 **`Course_Setting`** key=`llm-model`，以 GET/PUT `/rag/llm_model` 管理；未設定時預設 **`gpt-5.4`**（`services/quiz_generation.QUIZ_LLM_MODEL`）。

---

## 系統架構

### 整體架構圖

```mermaid
flowchart TB
    subgraph Client["用戶端"]
        FE["MyQuiz.ai 前端<br/>(Vercel / 本機 dev server)"]
    end

    subgraph Backend["MyQuiz.ai Backend — FastAPI"]
        direction TB
        CORS["CORSMiddleware"]
        LOG["APILogMiddleware<br/>寫入 Log 表"]
        ROUTERS["Routers 層<br/>users / zip / grade / exam / …"]
        DEPS["Dependencies<br/>person_id / course_id"]
        SERVICES["Services 層"]
        UTILS["Utils 層"]
    end

    subgraph ServicesDetail["Services"]
        QG["quiz_generation<br/>LLM 出題"]
        GR["grading<br/>LLM 批改 + FAISS 檢索"]
        WR["weakness_report<br/>弱點分析"]
        RAGP["rag_prompts / prompt_placeholders"]
    end

    subgraph External["外部服務"]
        PG[("Supabase PostgreSQL<br/>User, Rag, Exam, Log…")]
        ST["Supabase Storage<br/>bucket: MyQuiz-ai"]
        OAI["OpenAI API<br/>gpt-5.4 + embeddings"]
        DG["Deepgram API<br/>音訊轉文字（選用）"]
    end

    FE -->|"HTTPS<br/>person_id + course_id query"| CORS
    CORS --> LOG
    LOG --> ROUTERS
    ROUTERS --> DEPS
    ROUTERS --> SERVICES
    ROUTERS --> UTILS
    SERVICES --> QG
    SERVICES --> GR
    SERVICES --> WR
    SERVICES --> RAGP
    UTILS --> PG
    UTILS --> ST
    QG --> OAI
    GR --> OAI
    GR --> ST
    WR --> OAI
    UTILS --> DG
```

### 請求處理流程

```mermaid
sequenceDiagram
    participant FE as 前端
    participant MW as APILogMiddleware
    participant RT as Router
    participant SV as Service
    participant DB as Supabase DB
    participant ST as Supabase Storage
    participant LLM as OpenAI

    FE->>MW: HTTP 請求 (person_id, course_id)
    MW->>RT: 轉發（略過 OPTIONS/HEAD/docs）
    RT->>RT: Depends 驗證 person_id / course_id
    alt 讀寫資料
        RT->>DB: PostgREST CRUD
    end
    alt 需要 RAG ZIP
        RT->>ST: 下載 upload/repack/rag ZIP
    end
    alt LLM 出題（同步）
        RT->>SV: generate_quiz*
        SV->>LLM: Chat Completions (JSON)
        SV->>DB: 更新 Rag_Quiz / Exam_Quiz
    end
    alt LLM 批改（非同步）
        RT->>FE: 202 + job_id
        RT->>SV: BackgroundTasks run_grade_job_background
        SV->>LLM: Chat Completions (JSON)
        SV->>DB: 寫入 answer_critique
        FE->>RT: GET grade-result/{job_id} 輪詢
    end
    RT->>MW: 回應
    MW->>DB: 非同步寫 Log（失敗不影響回應）
    MW->>FE: JSON 回應
```

### 模組分層

| 層級 | 目錄 | 職責 |
|------|------|------|
| **入口** | `main.py` | FastAPI app、CORS、Middleware、路由掛載 |
| **路由** | `routers/` | HTTP 端點、Pydantic 模型、權限檢查 |
| **服務** | `services/` | LLM 出題／批改／弱點報告、Prompt 模板 |
| **工具** | `utils/` | Supabase、Storage、FAISS、ZIP、序列化 |
| **依賴** | `dependencies/` | 全域 `person_id`、`course_id` 注入 |
| **中介** | `middleware/` | API 呼叫紀錄、敏感欄位遮罩 |

路由掛載順序（`main.py`）：`zip` → `grade` → `exam` → `person_analysis` → `course_analysis` → `users` → `college` → `course` → `course_settings`（掛在 `/rag`）→ `prompt` → `log`。

### Supabase Storage 路徑

Bucket 名稱由環境變數 `SUPABASE_RAG_BUCKET` 指定（預設 `MyQuiz-ai`）：

```
{person_id}/{rag_page_id}/upload/{rag_page_id}.zip   ← 原始上傳 ZIP
{person_id}/{rag_page_id}/repack/{unit_id}.zip      ← 單元 repack
{person_id}/{rag_page_id}/rag/{unit_id}_rag.zip     ← FAISS 或 transcript ZIP
```

根目錄 `_metadata.json` 記錄 page_id → 路徑對照，供 `get_zip_path()` 查詢。

---

## 技術棧

| 類別 | 技術 |
|------|------|
| 語言／Runtime | Python 3.10.12 |
| Web 框架 | FastAPI + Uvicorn |
| 資料庫 | Supabase（PostgreSQL + PostgREST） |
| 檔案儲存 | Supabase Storage |
| 向量檢索 | LangChain + FAISS-CPU + OpenAI Embeddings (`text-embedding-3-small`) |
| LLM | OpenAI **gpt-5.4**（出題、批改、弱點分析共用；可經 `/rag/llm_model` 依課程覆寫） |
| 文件解析 | PyPDF、python-docx、python-pptx、unstructured、docx2txt |
| 語音（選用） | Deepgram（音訊轉逐字稿） |
| 部署 | Render（`runtime.txt` 指定 Python 版本） |

---

## 目錄結構

```
MyQuiz-ai-backend/
├── main.py                 # FastAPI 入口、CORS、路由掛載
├── requirements.txt        # Python 依賴
├── runtime.txt             # Render Python 版本
├── .env.example            # 環境變數範例
├── dependencies/
│   ├── person_id.py        # 全域必填 query person_id
│   └── course_id.py        # RAG 相關必填 query course_id
├── middleware/
│   └── api_log_middleware.py
├── routers/
│   ├── users.py            # /user — 登入、使用者 CRUD
│   ├── college.py          # /college — 學院列表
│   ├── course.py           # /course — 課程列表
│   ├── zip.py              # /rag — ZIP 上傳、建庫、Rag CRUD
│   ├── grade.py            # /rag — 出題、批改、API Key、分析 prompt
│   ├── exam.py             # /exam — 測驗 CRUD、出題、批改
│   ├── course_analysis.py  # /course-analysis
│   ├── person_analysis.py  # /person-analysis
│   ├── course_settings.py  # /rag — 分析 prompt（Course_Setting）
│   ├── prompt.py           # /prompt — 內建 LLM 模板查詢
│   └── log.py              # /log — API 呼叫紀錄
├── services/
│   ├── quiz_generation.py  # LLM 出題（FAISS / 逐字稿）
│   ├── grading.py          # LLM 批改、Background Job
│   ├── weakness_report.py  # 個人／課程弱點報告
│   ├── rag_prompts.py      # RAG 檢索參數
│   └── prompt_placeholders.py
└── utils/
    ├── supabase.py         # Supabase client
    ├── zip_storage.py      # Storage 上傳／下載
    ├── rag_faiss.py        # FAISS 建庫
    ├── rag_transcript.py   # 逐字稿抽取（unit_type 2/3/4）
    ├── course_setting.py   # Course_Setting 讀寫
    ├── db_schema.py        # 表名與 SELECT 欄位常數
    └── …
```

---

## 資料模型

### ER 關係圖（簡化）

```mermaid
erDiagram
    College ||--o{ Course : contains
    User ||--o{ User_Course_Relation : enrolls
    Course ||--o{ User_Course_Relation : has
    Course ||--o{ Course_Setting : configures
    Course ||--o{ Rag : owns
    Rag ||--o{ Rag_Unit : contains
    Rag_Unit ||--o{ Rag_Quiz : has
    Course ||--o{ Exam : owns
    Exam ||--o{ Exam_Quiz : has
    Exam_Quiz }o--|| Rag_Quiz : "optional link"
    User ||--o{ Log : generates

    User {
        int user_id PK
        string person_id UK
        string name
        bool deleted
    }
    User_Course_Relation {
        int course_user_id PK
        int user_id FK
        int course_id FK
        int user_type
    }
    Rag {
        int rag_id PK
        string rag_page_id UK
        int course_id
        json file_metadata
    }
    Rag_Unit {
        int rag_unit_id PK
        string rag_page_id FK
        int unit_type
        string transcript
        string rag_file_name
    }
    Rag_Quiz {
        int rag_quiz_id PK
        string quiz_content
        string answer_content
        string answer_critique
        bool for_exam
        bool follow_up
    }
    Exam {
        int exam_id PK
        string exam_page_id UK
        int course_id
    }
    Exam_Quiz {
        int exam_quiz_id PK
        string exam_page_id FK
        int rag_quiz_id FK
        int quiz_rate
        string answer_content
        string answer_critique
        int grade_rate
        bool follow_up
        bool deleted
    }
    Course_Setting {
        int course_setting_id PK
        int course_id FK
        string key
        string value
    }
    Log {
        int log_id PK
        string person_id
        int course_id
        string api
        json api_metadata
    }
```

### 主要資料表

| 表名 | 說明 |
|------|------|
| `User` | 使用者（`person_id` 為登入帳號） |
| `User_Course_Relation` | 選課關係；**`user_type` 依課程**（1＝可建 FAISS 之管理者／教師、2＝教師、3＝學生） |
| `College` / `Course` | 學院與課程 |
| `Course_Setting` | 課程級 key-value 設定（API Key、分析 prompt） |
| `Rag` / `Rag_Unit` / `Rag_Quiz` | RAG 分頁、教材單元、練習題 |
| `Exam` / `Exam_Quiz` | 測驗卷與測驗題（可連結 `rag_quiz_id`） |
| `Log` | API 呼叫紀錄 |

### Rag_Unit.unit_type

| 值 | 名稱 | 建庫行為 | 出題／批改 context |
|----|------|----------|-------------------|
| 0 | 未指定 | repack 複製至 rag | 依 ZIP 內容推斷 |
| 1 | RAG（Office/PDF/MD） | 建 **FAISS** 向量 ZIP | 向量檢索 top-k chunks |
| 2 | 文字 | repack + **transcript.md** ZIP | 全文逐字稿 |
| 3 | MP3 音訊 | repack + transcript ZIP | 音訊同資料夾文字檔逐字稿 |
| 4 | YouTube | repack + transcript ZIP | URL 第一行 + 第二行起逐字稿 |

> `user_type == 1` 且 `unit_type == 1` 時，`build-rag-zip` 才會實際建 FAISS；其餘類型以逐字稿 ZIP 供 LLM 使用。

### Course_Setting 常用 key

| key | 用途 | 管理端點 |
|-----|------|----------|
| `rag-api-key` | RAG 出題／批改 OpenAI Key | GET/PUT `/rag/llm_api_key` |
| `llm-model` | 出題／批改／弱點分析 LLM 模型名 | GET/PUT `/rag/llm_model` |
| `exam-api-key` | Exam 出題／批改 OpenAI Key | GET/PUT `/exam/api_key` |
| `person_analysis_user_prompt_text` | 個人弱點分析指令 | GET/PUT `/rag/person_analysis_user_prompt_text` |
| `course_analysis_user_prompt_text` | 課程弱點分析指令 | GET/PUT `/rag/course_analysis_user_prompt_text` |

---

## 核心業務流程

### RAG 建置流程（build-rag-zip）

```mermaid
flowchart LR
    A["上傳 ZIP<br/>POST /rag/tab/upload-zip"] --> B["選擇單元<br/>unit_list"]
    B --> C{"unit_type?"}
    C -->|1 + user_type=1| D["解析 Office/PDF/MD<br/>chunk + embedding"]
    D --> E["建 FAISS<br/>上傳 rag/*.zip"]
    C -->|2/3/4| F["抽取逐字稿<br/>transcript.md ZIP"]
    C -->|0 等| G["repack 複製"]
    E --> H["寫入 Rag_Unit<br/>NDJSON 串流回報"]
    F --> H
    G --> H
```

- 回應格式：`application/x-ndjson`，逐行 JSON（`start` → `building` → `unit` → … → `complete`）。
- 前端須用 `fetch` 逐行讀取，**不可** `response.json()`。

### LLM 出題流程

```mermaid
flowchart TD
    REQ["POST .../llm-generate"] --> UT{"Rag_Unit.unit_type"}
    UT -->|1| FAISS["下載 rag ZIP<br/>FAISS 檢索 k=5<br/>query: 課程重點概念"]
    UT -->|2/3/4| TR["讀 Rag_Unit.transcript"]
    FAISS --> LLM["gpt-5.4 JSON 出題<br/>quiz_content / hint / reference"]
    TR --> LLM
    LLM --> DB["更新 Rag_Quiz 或 Exam_Quiz<br/>清空 answer_content / answer_critique"]
```

- **追問出題**（`llm-generate-followup`）：依前次作答與評語決定追問弱點或出新題。
- **`-db` 變體**：body 不帶 prompt 文字，沿用 DB 既有 `quiz_user_prompt_text` 或 `answer_user_prompt_text`。

### LLM 批改流程（非同步）

```mermaid
stateDiagram-v2
    [*] --> Pending: POST llm-grade → 202 job_id
    Pending --> Running: BackgroundTasks 啟動
    Running --> Ready: LLM 成功 + DB 寫入
    Running --> Error: LLM 或 DB 失敗
    Ready --> [*]: GET grade-result status=ready
    Error --> [*]: GET grade-result status=error
    Pending --> Pending: GET grade-result status=pending
```

1. `POST .../llm-grade` 立即回 **HTTP 202** + `{ "job_id": "uuid" }`。
2. 背景執行 `run_grade_job_background`：有逐字稿走 transcript 路徑，否則 FAISS 檢索（以**題幹**為 query，k=5）。
3. LLM 回傳 JSON → 解析 `quiz_comments` → 合併寫入 `answer_critique`（純文字）。
4. 前端輪詢 `GET .../grade-result/{job_id}` 直到 `status` 為 `ready` 或 `error`。
5. Job 結果存於**程序記憶體**（重啟後遺失；生產環境若多 instance 需另行設計持久化）。

### 弱點分析流程

- **個人**：`GET /person-analysis/quizzes/{person_id}?course_id=` → 彙整該生已作答 `Exam_Quiz` → LLM 產生 Markdown 報告。
- **課程**：`GET /course-analysis/quizzes?course_id=` → 彙整全課程已作答題目 → LLM 產生報告。
- 分析指令來自 `Course_Setting` 對應 prompt key；模板全文可從 `GET /prompt/templates` 查閱。

---

## 環境變數

複製 `.env.example` 為 `.env` 並填入：

| 變數 | 必填 | 說明 |
|------|------|------|
| `SUPABASE_URL` | ✅ | Supabase 專案 URL |
| `SUPABASE_ANON_KEY` | ✅* | 公開 anon key |
| `SUPABASE_SERVICE_ROLE_KEY` | ✅* | 後端 service role（略過 RLS） |
| `SUPABASE_RAG_BUCKET` | 選 | Storage bucket 名（預設 `MyQuiz-ai`） |
| `CORS_EXTRA_ORIGINS` | 選 | 額外 CORS origin，逗號分隔 |
| `DEEPGRAM_API_KEY` | 選 | 音訊轉文字（使用 Deepgram 時） |
| `DEEPGRAM_MODEL` | 選 | 預設 `nova-2` |
| `DEEPGRAM_REQUEST_TIMEOUT_SECONDS` | 選 | 預設 900 |

\* `ANON_KEY` 與 `SERVICE_ROLE_KEY` 至少需其一；後端建議使用 service role。

LLM API Key **不**放在 `.env`，而是依課程存入 `Course_Setting`。

---

## 本機開發

### 前置需求

- Python 3.10+
- Supabase 專案（PostgreSQL + Storage bucket）
- OpenAI API Key（寫入各課程 `Course_Setting`）

### 啟動步驟

```bash
# 1. 環境變數
cp .env.example .env
# 編輯 .env 填入 Supabase 憑證

# 2. 依賴
pip install -r requirements.txt

# 3. 啟動（熱重載）
uvicorn main:app --reload
```

- API 文件：`http://127.0.0.1:8000/docs`（Swagger UI）
- 健康檢查：`GET /?person_id=YOUR_ID`

### macOS 注意事項

`main.py` 啟動時設定 `KMP_DUPLICATE_LIB_OK=TRUE`，避免 FAISS/NumPy 多份 OpenMP runtime 衝突。

### CORS

內建允許：`localhost`／`127.0.0.1` 埠 8080–8086、5173（Vite）、4173、3000，以及 `https://myquiz-ai.vercel.app`。區網 IP 或其他網域請設 `CORS_EXTRA_ORIGINS`。

---

## 部署（Render）

1. 連接 GitHub repo，Build Command：`pip install -r requirements.txt`
2. Start Command：`uvicorn main:app --host 0.0.0.0 --port $PORT`
3. Environment 新增與 `.env` 相同變數（至少 Supabase 三項）
4. Python 版本由 `runtime.txt`（`python-3.10.12`）指定

**Render 代理逾時**：同步 LLM 請求若超過約 30 秒可能回 502；批改已改非同步（202 + 輪詢）以避免此問題。若 502 回應不含 CORS 標頭，前端會看到跨域錯誤——應確認使用輪詢端點而非長時間阻塞 POST。

---

## 認證與授權

本後端**不使用 JWT middleware**；身分以 query **`person_id`** 識別呼叫者（所有端點必填，少數媒體端點例外）。

| 機制 | 說明 |
|------|------|
| `person_id` | 必填 query；對應 `User.person_id` |
| `course_id` | RAG／Exam／Log 等端點必填；對應目前操作課程 |
| `user_type` | 存於 `User_Course_Relation`；`1` 可建 FAISS、`1`/`2` 可寫分析 prompt 與 API Key |
| 資源擁有權 | 多數寫入操作驗證 `person_id` 與 Rag/Exam 擁有者一致 |

登入流程：`POST /user/login`（body：`person_id` + `password`）→ 回傳使用者與選課列表；前端自行保存 `person_id` 供後續 API 使用。

---

## API 回傳格式

所有端點皆需 query 參數 `person_id`（RAG／Exam 相關端點另需 `course_id`；少數媒體端點如 `GET /rag/tab/unit/mp3-file` 不需 `person_id`）。

### 通用錯誤格式

HTTP 4xx / 5xx 時統一回傳：

```json
{ "detail": "錯誤說明文字" }
```

評分非同步端點（llm-grade）在排程前檢查到錯誤時，直接回傳相應 HTTP 狀態碼加上：

```json
{ "error": "錯誤說明文字" }
```

---

## API 目錄

RAG 與 Exam 採相同層級：**分頁（tab）→ 單元（unit，僅 RAG）→ 題目（quiz：create → name → delete → 標記 → 出題 → 評分）→ 設定**。Swagger（`/docs`）路徑順序由 `utils/openapi_order.py` 統一排序。

| 方法 | 路徑 | 說明 |
|------|------|------|
| GET | [`/`](#get-) | 健康檢查 |
| **學院／課程** | | |
| GET | [`/college/colleges`](#get-collegecolleges) | 列出學院（含 courses） |
| GET | [`/course/courses`](#get-coursecourses) | 列出課程（含 college_name） |
| **使用者** | | |
| GET | [`/user/users`](#get-userusers) | 列出所有使用者 |
| POST | [`/user/users`](#post-userusers) | 新增使用者 |
| POST | [`/user/users/batch`](#post-userusersbatch) | 批次新增使用者 |
| PUT | [`/user/users/delete`](#put-userusersdelete) | 軟刪除使用者 |
| POST | [`/user/login`](#post-userlogin) | 登入 |
| PATCH | [`/user/profile`](#patch-userprofile) | 更新個人資料 |
| **RAG ZIP 管理** | | |
| GET | [`/rag/tabs`](#get-ragtabs) | 列出 Rag（含 units→quizzes） |
| POST | [`/rag/tab/create`](#post-ragtabcreate) | 建立 Rag |
| POST | [`/rag/tab/create-upload-zip`](#post-ragtabcreate-upload-zip) | 建立 Rag 並上傳 ZIP |
| PUT | [`/rag/tab/tab-name`](#put-ragtabtab-name) | 更新 Rag tab_name |
| PUT | [`/rag/tab/delete/{rag_page_id}`](#put-ragtabdeleterag_page_id) | 軟刪除 Rag |
| POST | [`/rag/tab/upload-zip`](#post-ragtabupload-zip) | 上傳 ZIP |
| POST | [`/rag/tab/build-rag-zip`](#post-ragtabbuild-rag-zip) | 打包 RAG ZIP（NDJSON 串流） |
| GET | [`/rag/tab/units`](#get-ragtabunits) | 列出 Rag_Unit（含 quizzes） |
| PUT | [`/rag/tab/unit/unit-name`](#put-ragtabunitunit-name) | 更新 Rag_Unit unit_name |
| GET | [`/rag/tab/unit/mp3-file`](#get-ragtabunitmp3-file) | 取得音訊（unit_type=3） |
| GET | [`/rag/tab/unit/youtube-url`](#get-ragtabunityoutube-url) | 取得 YouTube URL（unit_type=4） |
| POST | [`/rag/tab/unit/quiz/create`](#post-ragtabunitquizcreate) | 新增空白 Rag_Quiz |
| PUT | [`/rag/tab/unit/quiz/quiz-name`](#put-ragtabunitquizquiz-name) | 更新 Rag_Quiz quiz_name |
| PUT | [`/rag/tab/unit/quiz/delete/{rag_quiz_id}`](#put-ragtabunitquizdeleterag_quiz_id) | 軟刪除 Rag_Quiz |
| POST | [`/rag/tab/unit/quiz/followup`](#post-ragtabunitquizfollowup) | 更新 follow_up 標記 |
| POST | [`/rag/tab/unit/quiz/for-exam`](#post-ragtabunitquizfor-exam) | 更新 for_exam 標記 |
| **RAG 出題與評分** | | |
| POST | [`/rag/tab/unit/quiz/llm-generate`](#post-ragtabunitquizllm-generate) | LLM 出題 |
| POST | [`/rag/tab/unit/quiz/llm-generate-db`](#post-ragtabunitquizllm-generate-db) | LLM 出題（沿用 DB prompt） |
| POST | [`/rag/tab/unit/quiz/llm-generate-followup`](#post-ragtabunitquizllm-generate-followup) | LLM 追問出題 |
| POST | [`/rag/tab/unit/quiz/llm-generate-followup-db`](#post-ragtabunitquizllm-generate-followup-db) | LLM 追問出題（沿用 DB prompt） |
| POST | [`/rag/tab/unit/quiz/llm-grade`](#post-ragtabunitquizllm-grade) | 非同步評分（202 + job_id） |
| POST | [`/rag/tab/unit/quiz/llm-grade-db`](#post-ragtabunitquizllm-grade-db) | 非同步評分（沿用 DB prompt） |
| GET | [`/rag/tab/unit/quiz/grade-result/{job_id}`](#get-ragtabunitquizgrade-resultjob_id) | 輪詢評分結果 |
| GET | [`/rag/unit/text`](#get-ragunittext) | 取得文字單元逐字稿 |
| GET | [`/rag/unit/mp3-file`](#get-ragunitmp3-file) | 取得音訊與同資料夾文字檔逐字稿 |
| GET | [`/rag/unit/youtube-url`](#get-raguniityoutube-url) | 解析 YouTube URL 與第二行起逐字稿 |
| **RAG 課程設定** | | |
| GET | [`/rag/llm_api_key`](#get-ragllm_api_key) | 讀取 rag-api-key（user_type 1/2） |
| PUT | [`/rag/llm_api_key`](#put-ragllm_api_key) | 寫入 rag-api-key |
| GET | [`/rag/llm_model`](#get-ragllm_model) | 讀取 llm-model（user_type 1/2） |
| PUT | [`/rag/llm_model`](#put-ragllm_model) | 寫入 llm-model |
| GET | [`/rag/person_analysis_user_prompt_text`](#get-ragperson_analysis_user_prompt_text) | 取得個人分析 prompt |
| PUT | [`/rag/person_analysis_user_prompt_text`](#put-ragperson_analysis_user_prompt_text) | 寫入個人分析 prompt |
| GET | [`/rag/course_analysis_user_prompt_text`](#get-ragcourse_analysis_user_prompt_text) | 取得課程分析 prompt |
| PUT | [`/rag/course_analysis_user_prompt_text`](#put-ragcourse_analysis_user_prompt_text) | 寫入課程分析 prompt |
| **測驗** | | |
| GET | [`/exam/tabs`](#get-examtabs) | 列出 Exam（含 quizzes） |
| GET | [`/exam/rag-for-exams`](#get-examrag-for-exams) | 列出 for_exam RAG 單元 |
| POST | [`/exam/tab/create`](#post-examtabcreate) | 建立 Exam |
| PUT | [`/exam/tab/tab-name`](#put-examtabtab-name) | 更新 Exam tab_name |
| PUT | [`/exam/tab/delete/{exam_page_id}`](#put-examtabdeleteexam_page_id) | 軟刪除 Exam |
| POST | [`/exam/tab/quiz/create`](#post-examtabquizcreate) | 新增空白 Exam_Quiz |
| PUT | [`/exam/tab/quiz/delete/{exam_quiz_id}`](#put-examtabquizdeleteexam_quiz_id) | 軟刪除 Exam_Quiz |
| POST | [`/exam/tab/quiz/llm-generate`](#post-examtabquizllm-generate) | LLM 出題 |
| POST | [`/exam/tab/quiz/llm-generate-followup`](#post-examtabquizllm-generate-followup) | LLM 追問出題 |
| POST | [`/exam/tab/quiz/create-llm-generate`](#post-examtabquizcreate-llm-generate) | 建立並 LLM 出題 |
| POST | [`/exam/tab/quiz/create-llm-generate-followup`](#post-examtabquizcreate-llm-generate-followup) | 建立並 LLM 追問出題 |
| POST | [`/exam/tab/quiz/llm-grade`](#post-examtabquizllm-grade) | 非同步評分（202 + job_id） |
| GET | [`/exam/tab/quiz/grade-result/{job_id}`](#get-examtabquizgrade-resultjob_id) | 輪詢評分結果 |
| POST | [`/exam/tab/quiz/quiz-rate`](#post-examtabquizquiz-rate) | 更新 quiz_rate |
| POST | [`/exam/tab/quiz/grade-rate`](#post-examtabquizgrade-rate) | 更新 grade_rate |
| GET | [`/exam/api_key`](#get-examapi_key) | 讀取 exam-api-key |
| PUT | [`/exam/api_key`](#put-examapi_key) | 寫入 exam-api-key |
| **課程分析** | | |
| GET | [`/course-analysis/quizzes`](#get-course-analysisquizzes) | 課程作答分析（含弱點報告） |
| **個人分析** | | |
| GET | [`/person-analysis/quizzes/{person_id}`](#get-person-analysisquizzesperson_id) | 個人作答分析（含弱點報告） |
| **Prompt 模板** | | |
| GET | [`/prompt/templates`](#get-prompttemplates) | 內建 LLM prompt 模板全文 |
| **Log** | | |
| GET | [`/log/logs`](#get-loglogs) | 列出 API 呼叫紀錄 |

---

## API 詳細文件

### 健康檢查

#### `GET /`

確認伺服器正常運行（需帶 query `person_id`）。

```json
{ "status": "Server is running" }
```

---

### 學院 `/college`

#### `GET /college/colleges`

列出所有未刪除學院，含所屬課程列表。需 query `person_id`。

```json
{
  "colleges": [
    {
      "college_id": 1,
      "college_name": "string",
      "courses": [
        {
          "course_id": 1,
          "college_id": 1,
          "semester": "113-1",
          "course_name": "string"
        }
      ],
      "updated_at": "2024-01-01T00:00:00+08:00",
      "created_at": "2024-01-01T00:00:00+08:00"
    }
  ],
  "count": 1
}
```

---

### 課程 `/course`

#### `GET /course/courses`

列出所有未刪除課程，含 `college_id`、`college_name`。需 query `person_id`。

```json
{
  "courses": [
    {
      "course_id": 1,
      "college_id": 1,
      "college_name": "string",
      "semester": "113-1",
      "course_name": "string",
      "updated_at": "2024-01-01T00:00:00+08:00",
      "created_at": "2024-01-01T00:00:00+08:00"
    }
  ],
  "count": 1
}
```

---

### 使用者 `/user`

#### `GET /user/users`

列出所有未刪除使用者。

```json
{
  "users": [
    {
      "user_id": 1,
      "person_id": "string",
      "name": "string",
      "courses": [],
      "user_metadata": null,
      "updated_at": "2024-01-01T00:00:00+08:00",
      "created_at": "2024-01-01T00:00:00+08:00"
    }
  ],
  "count": 1
}
```

---

#### `POST /user/users`

新增單一使用者。

```json
{
  "user": {
    "user_id": 1,
    "person_id": "string",
    "name": "string",
    "user_type": 3,
    "user_metadata": null,
    "updated_at": "2024-01-01T00:00:00+08:00",
    "created_at": "2024-01-01T00:00:00+08:00"
  },
  "courses": []
}
```

---

#### `POST /user/users/batch`

批次新增使用者（user_type 固定 3，密碼預設 0000）。

```json
{
  "created": [
    {
      "user_id": 1,
      "person_id": "string",
      "name": "string",
      "user_type": 3,
      "user_metadata": null,
      "updated_at": "2024-01-01T00:00:00+08:00",
      "created_at": "2024-01-01T00:00:00+08:00"
    }
  ],
  "failed": [
    { "person_id": "string", "detail": "person_id 已存在" }
  ],
  "created_count": 1,
  "failed_count": 0
}
```

---

#### `PUT /user/users/delete`

軟刪除指定使用者（courses 固定為空列表）。

```json
{
  "user": {
    "user_id": 1,
    "person_id": "string",
    "name": "string",
    "user_type": 3,
    "user_metadata": null,
    "updated_at": "2024-01-01T00:00:00+08:00",
    "created_at": "2024-01-01T00:00:00+08:00"
  },
  "courses": []
}
```

---

#### `POST /user/login`

登入成功時回傳使用者資訊與所有課程列表。

```json
{
  "user": {
    "user_id": 1,
    "person_id": "string",
    "name": "string",
    "user_type": 3,
    "user_metadata": null,
    "updated_at": "2024-01-01T00:00:00+08:00",
    "created_at": "2024-01-01T00:00:00+08:00"
  },
  "courses": [
    {
      "course_user_id": 1,
      "course_id": 1,
      "course_name": "string",
      "user_type": 3
    }
  ]
}
```

---

#### `PATCH /user/profile`

更新個人資料（name、user_type）；`user_type` 更新會套用至該使用者所有選課列。

```json
{
  "user": {
    "user_id": 1,
    "person_id": "string",
    "name": "string",
    "user_type": 3,
    "user_metadata": null,
    "updated_at": "2024-01-01T00:00:00+08:00",
    "created_at": "2024-01-01T00:00:00+08:00"
  },
  "courses": []
}
```

---

### RAG ZIP 管理 `/rag`

#### `GET /rag/tabs`

列出 Rag（含 units→quizzes）。音訊單元含 `mp3_audio_url`；YouTube 單元含 `youtube_url_api`。

```json
{
  "rags": [
    {
      "rag_id": 1,
      "rag_page_id": "string",
      "tab_name": "string",
      "person_id": "string",
      "course_id": 1,
      "local": false,
      "deleted": false,
      "file_metadata": { "filename": "...", "second_folders": [], "file_size": 1.23 },
      "updated_at": "2024-01-01T00:00:00+08:00",
      "created_at": "2024-01-01T00:00:00+08:00",
      "units": [
        {
          "rag_unit_id": 1,
          "rag_page_id": "string",
          "person_id": "string",
          "course_id": 1,
          "unit_name": "string",
          "folder_combination": "string",
          "unit_type": 1,
          "repack_file_name": "string",
          "rag_file_name": "string",
          "rag_file_size": 1.23,
          "rag_chunk_size": 1000,
          "rag_chunk_overlap": 200,
          "transcript": "string",
          "text_file_name": "string",
          "mp3_file_name": "string",
          "youtube_url": "string",
          "deleted": false,
          "updated_at": "2024-01-01T00:00:00+08:00",
          "created_at": "2024-01-01T00:00:00+08:00",
          "mp3_audio_url": "/rag/tab/unit/mp3-file?rag_page_id=...&rag_unit_id=...&course_id=...",
          "youtube_url_api": "/rag/tab/unit/youtube-url?rag_page_id=...&rag_unit_id=...&course_id=...",
          "quizzes": [
            {
              "rag_quiz_id": 1,
              "rag_page_id": "string",
              "rag_unit_id": 1,
              "person_id": "string",
              "quiz_name": "string",
              "quiz_user_prompt_text": "string",
              "quiz_content": "string",
              "quiz_hint": "string",
              "quiz_answer_reference": "string",
              "answer_user_prompt_text": "string",
              "answer_content": "string",
              "answer_critique": "string | null",
              "for_exam": false,
              "follow_up": false,
              "deleted": false,
              "updated_at": "2024-01-01T00:00:00+08:00",
              "created_at": "2024-01-01T00:00:00+08:00"
            }
          ]
        }
      ]
    }
  ],
  "count": 1
}
```

> `mp3_audio_url` 僅 unit_type=3 且 mp3_file_name 非空時出現；`youtube_url_api` 僅 unit_type=4 且 youtube_url 非空時出現。

---

#### `POST /rag/tab/create`

建立一筆 Rag。

```json
{
  "rag_id": 1,
  "rag_page_id": "string",
  "tab_name": "string",
  "person_id": "string",
  "course_id": 1,
  "local": false,
  "created_at": "2024-01-01T00:00:00+08:00"
}
```

---

#### `POST /rag/tab/create-upload-zip`

建立 Rag 並上傳 ZIP（multipart/form-data）。

```json
{
  "rag_id": 1,
  "rag_page_id": "string",
  "tab_name": "string",
  "person_id": "string",
  "course_id": 1,
  "local": false,
  "created_at": "2024-01-01T00:00:00+08:00",
  "file_metadata": {
    "rag_id": 1,
    "rag_page_id": "string",
    "created_at": "2024-01-01T00:00:00+08:00",
    "filename": "upload.zip",
    "second_folders": ["folder1", "folder2"],
    "file_size": 1.23
  }
}
```

---

#### `PUT /rag/tab/tab-name`

更新 Rag 的 tab_name。

```json
{
  "rag_id": 1,
  "rag_page_id": "string",
  "person_id": "string",
  "tab_name": "新名稱",
  "updated_at": "2024-01-01T00:00:00+08:00"
}
```

---

#### `PUT /rag/tab/delete/{rag_page_id}`

軟刪除 Rag 及其 Rag_Unit，並刪除 Storage 資料夾。

```json
{
  "message": "已將 RAG 資料標記為刪除並刪除儲存資料夾",
  "rag_page_id": "string",
  "person_id": "string",
  "rag_updated": true,
  "folder_deleted": true
}
```

---

#### `POST /rag/tab/upload-zip`

上傳 ZIP 並更新 Rag.file_metadata（multipart/form-data）。

```json
{
  "rag_id": 1,
  "rag_page_id": "string",
  "created_at": "2024-01-01T00:00:00+08:00",
  "filename": "upload.zip",
  "second_folders": ["folder1", "folder2"],
  "file_size": 1.23
}
```

---

#### `POST /rag/tab/build-rag-zip`

NDJSON 串流回應（`application/x-ndjson`）。請以 `fetch` 逐行讀取，**勿**使用 `response.json()`。HTTP 狀態碼恒為 200，以最後一行 `type==="complete"` 的 `success` 判斷成敗。

**第 1 行 — start**
```json
{
  "type": "start",
  "total": 2,
  "source_rag_page_id": "string",
  "unit_list": "folder1+folder2",
  "user_type": 1,
  "build_faiss_request": null,
  "repack_only": false,
  "allow_faiss": true
}
```

**第 N 行 — building（每單元前一行）**
```json
{
  "type": "building",
  "index": 1,
  "total": 2,
  "completed_before": 0,
  "filename": "folder1.zip"
}
```

**第 N+1 行 — unit（每單元結果）**
```json
{
  "type": "unit",
  "index": 1,
  "total": 2,
  "output": {
    "filename": "folder1.zip",
    "folder_combination": "folder1",
    "unit_name": "folder1",
    "repack_filename": "abc123.zip",
    "rag_filename": "abc123_rag.zip",
    "unit_type": 1,
    "rag_mode": "faiss",
    "transcript_plain": "string",
    "text_file_name": "string",
    "mp3_file_name": "string",
    "youtube_url": "string",
    "rag_chunk_size": 1000,
    "rag_chunk_overlap": 200,
    "file_size": 0.45,
    "rag_error": "string（僅失敗時出現）"
  }
}
```

> `rag_mode`：`"faiss"`（向量庫）、`"transcript_md"`（逐字稿 md ZIP）、`"repack_copy"`（與 repack 同內容）。
> `rag_chunk_size`／`rag_chunk_overlap` 於 unit_type≠1 時回傳 0。

**最後一行 — complete**
```json
{
  "type": "complete",
  "success": true,
  "source_rag_page_id": "string",
  "unit_list": "folder1+folder2",
  "outputs": [ /* 同 unit.output */ ],
  "total": 2,
  "built_ok": 2,
  "built_failed": 0,
  "message": "RAG ZIP 建立失敗（請修正後重試）（僅失敗時出現）"
}
```

---

#### `GET /rag/tab/units`

依 rag_page_id 列出所有 Rag_Unit（含 quizzes）。

```json
{
  "units": [
    {
      "rag_unit_id": 1,
      "rag_page_id": "string",
      "person_id": "string",
      "unit_name": "string",
      "folder_combination": "string",
      "unit_type": 1,
      "repack_file_name": "string",
      "rag_file_name": "string",
      "rag_file_size": 1.23,
      "rag_chunk_size": 1000,
      "rag_chunk_overlap": 200,
      "transcript": "string",
      "text_file_name": "string",
      "mp3_file_name": "string",
      "youtube_url": "string",
      "deleted": false,
      "updated_at": "2024-01-01T00:00:00+08:00",
      "created_at": "2024-01-01T00:00:00+08:00",
      "quizzes": [ /* Rag_Quiz[] */ ]
    }
  ],
  "count": 1
}
```

---

#### `PUT /rag/tab/unit/unit-name`

更新 Rag_Unit 的 unit_name。

```json
{
  "rag_unit_id": 1,
  "rag_page_id": "string",
  "person_id": "string",
  "unit_name": "新名稱",
  "updated_at": "2024-01-01T00:00:00+08:00"
}
```

---

#### `GET /rag/tab/unit/mp3-file`

依 rag_page_id、rag_unit_id 回傳音訊（unit_type=3）。不需 person_id。

```json
{
  "rag_unit_id": 1,
  "rag_page_id": "string",
  "audio_base64": "base64 encoded audio string",
  "media_type": "audio/mpeg",
  "filename": "audio.mp3",
  "transcript": "string"
}
```

---

#### `GET /rag/tab/unit/youtube-url`

依 rag_page_id、rag_unit_id 回傳 YouTube URL（unit_type=4）。不需 person_id。

```json
{
  "rag_unit_id": 1,
  "rag_page_id": "string",
  "youtube_url": "https://www.youtube.com/watch?v=...",
  "transcript": "string"
}
```

---

#### `POST /rag/tab/unit/quiz/create`

新增空白 Rag_Quiz（不呼叫 LLM）。

```json
{
  "rag_quiz_id": 1,
  "rag_page_id": "string",
  "rag_unit_id": 1,
  "person_id": "string",
  "quiz_name": "string",
  "quiz_user_prompt_text": "",
  "quiz_content": "",
  "quiz_hint": "",
  "quiz_answer_reference": "",
  "answer_user_prompt_text": "",
  "quiz_answer": "",
  "answer_content": "",
  "answer_critique": null,
  "for_exam": false,
  "follow_up": false,
  "deleted": false,
  "updated_at": "2024-01-01T00:00:00+08:00",
  "created_at": "2024-01-01T00:00:00+08:00"
}
```

---

#### `PUT /rag/tab/unit/quiz/quiz-name`

更新 Rag_Quiz 的 quiz_name。

```json
{
  "rag_quiz_id": 1,
  "rag_page_id": "string",
  "rag_unit_id": 1,
  "person_id": "string",
  "quiz_name": "新名稱",
  "updated_at": "2024-01-01T00:00:00+08:00"
}
```

---

#### `PUT /rag/tab/unit/quiz/delete/{rag_quiz_id}`

軟刪除 Rag_Quiz。

```json
{
  "message": "已將 Rag_Quiz 標記為刪除",
  "rag_quiz_id": 1,
  "rag_page_id": "string",
  "rag_unit_id": 1,
  "person_id": "string",
  "rag_quiz_updated": true,
  "updated_at": "2024-01-01T00:00:00+08:00"
}
```

---

#### `POST /rag/tab/unit/quiz/followup`

更新 Rag_Quiz.follow_up（`followup=true` 標記追問、`false` 取消）。

---

#### `POST /rag/tab/unit/quiz/for-exam`

更新 Rag_Quiz.for_exam 標記。

```json
{
  "rag_quiz_id": 1,
  "rag_page_id": "string",
  "rag_unit_id": 1,
  "person_id": "string",
  "quiz_name": "string",
  "quiz_user_prompt_text": "string",
  "quiz_content": "string",
  "quiz_hint": "string",
  "quiz_answer_reference": "string",
  "answer_user_prompt_text": "string",
  "answer_content": "string",
  "quiz_answer": "string",
  "answer_critique": "string | null",
  "for_exam": true,
  "follow_up": false,
  "deleted": false,
  "updated_at": "2024-01-01T00:00:00+08:00",
  "created_at": "2024-01-01T00:00:00+08:00"
}
```

---

### RAG 出題與評分 `/rag`

#### `POST /rag/tab/unit/quiz/llm-generate`
#### `POST /rag/tab/unit/quiz/llm-generate-db`
#### `POST /rag/tab/unit/quiz/llm-generate-followup`
#### `POST /rag/tab/unit/quiz/llm-generate-followup-db`

LLM 出題後更新 Rag_Quiz 並回傳出題結果。`follow_up` 在 followup 端點為 true。

```json
{
  "rag_quiz_id": 1,
  "quiz_name": "string",
  "quiz_content": "題幹",
  "quiz_hint": "提示",
  "quiz_answer_reference": "參考答案",
  "quiz_user_prompt_text": "出題 prompt",
  "answer_user_prompt_text": "批改 prompt",
  "transcript": "逐字稿（unit_type=1 時為空字串）",
  "rag_output": {
    "rag_page_id": "stem string",
    "unit_name": "stem string",
    "filename": "stem.zip"
  },
  "follow_up": false,
  "quiz_llm_model": "gpt-5.4"
}
```

`quiz_llm_model` 為本次出題實際使用的模型（`Course_Setting` key=`llm-model`；未設定時為程式預設 `gpt-5.4`）。

---

#### `POST /rag/tab/unit/quiz/llm-grade`
#### `POST /rag/tab/unit/quiz/llm-grade-db`

非同步評分。回傳 **HTTP 202**。

```json
{
  "job_id": "uuid-string",
  "grade_llm_model": "gpt-5.4"
}
```

`grade_llm_model` 為本次排程批改實際使用的模型（與出題共用 `llm-model` 設定）。

---

#### `GET /rag/tab/unit/quiz/grade-result/{job_id}`

輪詢評分結果。`status` 為 `"pending"` | `"ready"` | `"error"`。

**pending / error 時**
```json
{
  "status": "pending",
  "result": null,
  "error": null
}
```

**ready 時**（另附 rag_quiz 整列）
```json
{
  "status": "ready",
  "result": {
    "quiz_comments": ["評語 Markdown 段落 1", "評語 Markdown 段落 2"],
    "rag_quiz_id": 1,
    "rag_answer_id": 1
  },
  "error": null,
  "rag_quiz": {
    "rag_quiz_id": 1,
    "rag_page_id": "string",
    "rag_unit_id": 1,
    "person_id": "string",
    "quiz_name": "string",
    "quiz_content": "string",
    "quiz_hint": "string",
    "quiz_answer_reference": "string",
    "answer_content": "學生作答",
    "answer_critique": "批改評語純文字（非 JSON 物件）",
    "for_exam": false,
    "follow_up": false,
    "deleted": false,
    "updated_at": "2024-01-01T00:00:00+08:00",
    "created_at": "2024-01-01T00:00:00+08:00"
  }
}
```

> `result.quiz_comments`：字串陣列（Markdown）。DB 的 `Rag_Quiz.answer_critique` 寫入合併後純文字，非 JSON 物件。`rag_answer_id` 為 `rag_quiz_id` 的向下相容別名。

---

#### `GET /rag/unit/text`

回傳文字單元（unit_type=2）的 `text_file_name` 與 `transcript`（全文）。`folder_name` 與 `rag_unit_id` **二擇一**。

- **folder_name**：自 upload ZIP 讀取（與 build-rag-zip unit_type=2 一致）；須傳 `person_id`。
- **rag_unit_id**：自 DB 讀取，不需 `person_id`；DB 無逐字稿時改讀 upload ZIP。

```json
{
  "rag_page_id": "string",
  "folder_name": "string",
  "rag_unit_id": 1,
  "text_file_name": "content.md",
  "transcript": "全文 Markdown 內容"
}
```

---

#### `GET /rag/unit/mp3-file`

自 upload ZIP 依 `folder_name` 回傳音訊與同資料夾內**至多一個**文字檔全文（與 build-rag-zip unit_type=3 一致）。需 person_id。

```json
{
  "rag_page_id": "string",
  "folder_name": "string",
  "audio_base64": "base64 encoded audio string",
  "media_type": "audio/mpeg",
  "filename": "audio.mp3",
  "text_file_name": "transcript.md",
  "transcript": "文字檔全文"
}
```

---

#### `GET /rag/unit/youtube-url`

自 upload ZIP 解析 YouTube URL；文字檔**第一行**為 URL，**第二行起**為 `transcript`（與 build-rag-zip unit_type=4 一致）。需 person_id。

```json
{
  "rag_page_id": "string",
  "folder_name": "string",
  "youtube_url": "https://www.youtube.com/watch?v=VIDEO_ID",
  "text_file_name": "unit.md",
  "transcript": "第二行起的逐字稿"
}
```

---

### 測驗 `/exam`

#### `GET /exam/tabs`

列出 Exam（含 quizzes）。

```json
{
  "exams": [
    {
      "exam_id": 1,
      "exam_page_id": "string",
      "tab_name": "string",
      "person_id": "string",
      "course_id": 1,
      "local": false,
      "deleted": false,
      "updated_at": "2024-01-01T00:00:00+08:00",
      "created_at": "2024-01-01T00:00:00+08:00",
      "quizzes": [
        {
          "exam_quiz_id": 1,
          "exam_page_id": "string",
          "rag_page_id": "string",
          "rag_unit_id": 1,
          "rag_quiz_id": 1,
          "person_id": "string",
          "course_id": 1,
          "unit_name": "string",
          "quiz_name": "string",
          "quiz_user_prompt_text": "string",
          "quiz_content": "string",
          "quiz_hint": "string",
          "quiz_answer_reference": "string",
          "quiz_rate": 0,
          "answer_user_prompt_text": "string",
          "answer_content": "string | null",
          "answer_critique": "string | null",
          "grade_rate": 0,
          "follow_up": false,
          "follow_up_exam_quiz_id": 0,
          "updated_at": "2024-01-01T00:00:00+08:00",
          "created_at": "2024-01-01T00:00:00+08:00",
          "follow_up_quiz": { /* 下一筆 follow_up Exam_Quiz，結構相同，可遞迴 */ }
        }
      ]
    }
  ],
  "count": 1
}
```

---

#### `GET /exam/rag-for-exams`

列出 for_exam=true 的 RAG 單元與題目。

```json
{
  "units": [
    {
      "rag_unit_id": 1,
      "rag_page_id": "string",
      "unit_name": "string",
      "unit_type": 1,
      "for_exam": true,
      "deleted": false,
      "updated_at": "2024-01-01T00:00:00+08:00",
      "created_at": "2024-01-01T00:00:00+08:00",
      "quizzes": [
        {
          "rag_quiz_id": 1,
          "rag_page_id": "string",
          "rag_unit_id": 1,
          "person_id": "string",
          "course_id": 1,
          "follow_up": false,
          "quiz_name": "string",
          "quiz_user_prompt_text": "string",
          "quiz_content": "string",
          "quiz_hint": "string",
          "quiz_answer_reference": "string",
          "answer_user_prompt_text": "string"
        }
      ]
    }
  ],
  "count": 1
}
```

---

#### `POST /exam/tab/create`

建立一筆 Exam。

```json
{
  "exam_id": 1,
  "exam_page_id": "string",
  "tab_name": "string",
  "person_id": "string",
  "course_id": 1,
  "local": false,
  "deleted": false,
  "updated_at": "2024-01-01T00:00:00+08:00",
  "created_at": "2024-01-01T00:00:00+08:00"
}
```

---

#### `PUT /exam/tab/tab-name`

更新 Exam 的 tab_name。

```json
{
  "exam_id": 1,
  "exam_page_id": "string",
  "tab_name": "新名稱",
  "person_id": "string",
  "updated_at": "2024-01-01T00:00:00+08:00"
}
```

---

#### `PUT /exam/tab/delete/{exam_page_id}`

軟刪除 Exam。

```json
{
  "message": "已將 Exam 標記為刪除",
  "exam_page_id": "string",
  "person_id": "string"
}
```

---

#### `POST /exam/tab/quiz/create`

新增空白 Exam_Quiz（不呼叫 LLM）。回傳 Exam_Quiz 完整列。

```json
{
  "exam_quiz_id": 1,
  "exam_page_id": "string",
  "rag_page_id": "",
  "rag_unit_id": null,
  "rag_quiz_id": null,
  "person_id": "string",
  "course_id": 1,
  "follow_up": false,
  "follow_up_exam_quiz_id": 0,
  "unit_name": "",
  "quiz_name": "",
  "quiz_user_prompt_text": null,
  "quiz_content": "",
  "quiz_hint": "",
  "quiz_answer_reference": "",
  "quiz_rate": 0,
  "answer_user_prompt_text": null,
  "answer_content": null,
  "answer_critique": null,
  "grade_rate": 0,
  "updated_at": "2024-01-01T00:00:00+08:00",
  "created_at": "2024-01-01T00:00:00+08:00"
}
```

---

#### `PUT /exam/tab/quiz/delete/{exam_quiz_id}`

軟刪除 Exam_Quiz（`deleted=true`）；並一併軟刪除 `follow_up_exam_quiz_id` 指向該題的追問子題鏈。`GET /exam/tabs` 等僅回傳未刪除題（`deleted=false` 或 `null`）。

```json
{
  "message": "已將 Exam_Quiz 標記為刪除",
  "exam_quiz_id": 1,
  "exam_page_id": "string",
  "person_id": "string",
  "exam_quiz_updated": true,
  "updated_at": "2024-01-01T00:00:00+08:00"
}
```

---

#### `POST /exam/tab/quiz/llm-generate`
#### `POST /exam/tab/quiz/create-llm-generate`

LLM 出題後更新 Exam_Quiz 並回傳出題結果。

```json
{
  "exam_quiz_id": 1,
  "quiz_name": "string",
  "quiz_content": "題幹",
  "quiz_hint": "提示",
  "quiz_answer_reference": "參考答案",
  "quiz_user_prompt_text": "出題 prompt",
  "answer_user_prompt_text": "批改 prompt",
  "unit_name": "string",
  "rag_page_id": "string",
  "rag_unit_id": 1,
  "rag_quiz_id": 1,
  "transcript": "逐字稿（unit_type=1 時為空字串）",
  "rag_output": {
    "rag_page_id": "stem string",
    "unit_name": "stem string",
    "filename": "stem.zip"
  },
  "created_at": "2024-01-01T00:00:00+08:00",
  "quiz_llm_model": "gpt-5.4"
}
```

`quiz_llm_model` 為本次出題實際使用的模型（`Course_Setting` key=`llm-model`；未設定時為程式預設 `gpt-5.4`）。

---

#### `POST /exam/tab/quiz/llm-generate-followup`
#### `POST /exam/tab/quiz/create-llm-generate-followup`

接續追問出題。`follow_up_exam_quiz_id > 0` 時回傳帶 follow_up 欄位的結果。

```json
{
  "exam_quiz_id": 2,
  "quiz_name": "string",
  "quiz_content": "追問題幹",
  "quiz_hint": "提示",
  "quiz_answer_reference": "參考答案",
  "quiz_user_prompt_text": "出題 prompt",
  "answer_user_prompt_text": "批改 prompt",
  "unit_name": "string",
  "rag_page_id": "string",
  "rag_unit_id": 1,
  "rag_quiz_id": 1,
  "transcript": "string",
  "rag_output": {
    "rag_page_id": "stem string",
    "unit_name": "stem string",
    "filename": "stem.zip"
  },
  "follow_up": true,
  "follow_up_exam_quiz_id": 1,
  "quiz_history_list": [
    {
      "quiz_content": "string",
      "answer_content": "string",
      "quiz_answer_reference": "string",
      "answer_critique": "string"
    }
  ],
  "created_at": "2024-01-01T00:00:00+08:00",
  "quiz_llm_model": "gpt-5.4"
}
```

> `follow_up`、`follow_up_exam_quiz_id`、`quiz_history_list` 僅在 `follow_up_exam_quiz_id > 0` 且有傳入歷史問答時才出現。

---

#### `POST /exam/tab/quiz/llm-grade`

非同步評分。回傳 **HTTP 202**。

```json
{
  "job_id": "uuid-string",
  "grade_llm_model": "gpt-5.4"
}
```

`grade_llm_model` 為本次排程批改實際使用的模型（與出題共用 `llm-model` 設定）。

---

#### `GET /exam/tab/quiz/grade-result/{job_id}`

輪詢評分結果。`status` 為 `"pending"` | `"ready"` | `"error"`。

**pending / error 時**
```json
{
  "status": "pending",
  "result": null,
  "error": null
}
```

**ready 時**（另附 exam_quiz 整列）
```json
{
  "status": "ready",
  "result": {
    "quiz_comments": ["評語 Markdown 段落 1", "評語 Markdown 段落 2"],
    "exam_quiz_id": 1
  },
  "error": null,
  "exam_quiz": {
    "exam_quiz_id": 1,
    "exam_page_id": "string",
    "rag_page_id": "string",
    "rag_unit_id": 1,
    "rag_quiz_id": 1,
    "person_id": "string",
    "course_id": 1,
    "unit_name": "string",
    "quiz_name": "string",
    "quiz_content": "string",
    "quiz_hint": "string",
    "quiz_answer_reference": "string",
    "quiz_rate": 0,
    "answer_content": "學生作答",
    "answer_critique": "批改評語純文字（非 JSON 物件）",
    "grade_rate": 0,
    "follow_up": false,
    "follow_up_exam_quiz_id": 0,
    "updated_at": "2024-01-01T00:00:00+08:00",
    "created_at": "2024-01-01T00:00:00+08:00"
  }
}
```

> `result.quiz_comments`：字串陣列（Markdown）。DB 的 `Exam_Quiz.answer_critique` 寫入合併後純文字，非 JSON 物件。

---

#### `POST /exam/tab/quiz/quiz-rate`

更新 Exam_Quiz.quiz_rate（僅 -1、0、1）。

```json
{
  "exam_quiz_id": 1,
  "quiz_rate": 0,
  "updated_at": "2024-01-01T00:00:00+08:00",
  "created_at": "2024-01-01T00:00:00+08:00",
  "message": "已更新 quiz_rate"
}
```

---

#### `POST /exam/tab/quiz/grade-rate`

更新 Exam_Quiz.grade_rate（僅 -1、0、1）。

```json
{
  "exam_quiz_id": 1,
  "grade_rate": 0,
  "updated_at": "2024-01-01T00:00:00+08:00",
  "created_at": "2024-01-01T00:00:00+08:00",
  "message": "已更新 grade_rate"
}
```

---

### 課程分析 `/course-analysis`

#### `GET /course-analysis/quizzes`

依 course_id 取得已作答 Exam_Quiz，並產生弱點報告。必填 query `course_id`（弱點報告會讀取該課程之 `course_analysis_user_prompt_text` 設定）。

```json
{
  "exams": [
    {
      "exam_id": 1,
      "exam_page_id": "string",
      "tab_name": "string",
      "person_id": "string",
      "quizzes": [ /* Exam_Quiz[]（含 answer_content 非空者），結構同 GET /exam/tabs */ ]
    }
  ],
  "count": 1,
  "weakness_report": "Markdown 弱點報告全文（LLM 成功時非 null）"
}
```

> `weakness_report` 為 LLM `message.content` 原文 Markdown；無 API Key、無可分析題目或失敗時為 null。

---

### 個人分析 `/person-analysis`

#### `GET /person-analysis/quizzes/{person_id}`

依 person_id、course_id 取得已作答 Exam_Quiz，並產生弱點報告。必填 query `course_id`（題目與弱點報告皆限該課程；弱點報告會讀取 `person_analysis_user_prompt_text` 設定）。

```json
{
  "exams": [
    {
      "exam_id": 1,
      "exam_page_id": "string",
      "tab_name": "string",
      "person_id": "string",
      "quizzes": [ /* Exam_Quiz[]（含 answer_content 非空者），結構同 GET /exam/tabs */ ]
    }
  ],
  "count": 1,
  "weakness_report": "Markdown 弱點報告全文（LLM 成功時非 null）"
}
```

> `weakness_report` 為 LLM `message.content` 原文 Markdown；無 API Key、無可分析題目或失敗時為 null。

---

### Log `/log`

#### `GET /log/logs`

依 course_id 列出 API 呼叫紀錄，依 log_id 降冪。

```json
{
  "logs": [
    {
      "log_id": 1,
      "person_id": "string",
      "course_id": 1,
      "api": "POST /rag/tab/unit/quiz/llm-generate",
      "api_metadata": { "key": "value" },
      "updated_at": "2024-01-01T00:00:00+08:00",
  "count": 1
}
```

---

### RAG 課程設定 `/rag`

#### `GET /rag/llm_api_key`

讀取 `Course_Setting` key=`rag-api-key`（依 query `course_id`）。僅該課程 `user_type` 1／2 可讀。

```json
{
  "course_id": 1,
  "api_key": "sk-…（無設定時為 null）"
}
```

---

#### `PUT /rag/llm_api_key`

寫入 `rag-api-key`。僅 `user_type` 1／2 可操作。Body：`{ "api_key": "sk-…" }`。

---

#### `GET /rag/llm_model`

讀取 `Course_Setting` key=`llm-model`（依 query `course_id`）。出題、批改、弱點分析共用同一模型；僅該課程 `user_type` 1／2 可讀。無設定時 `llm_model` 為 null（執行時 fallback 至程式預設 `gpt-5.4`）。

```json
{
  "course_id": 1,
  "llm_model": "gpt-5.4（無設定時為 null）"
}
```

---

#### `PUT /rag/llm_model`

寫入 `llm-model`。僅 `user_type` 1／2 可操作。Body：`{ "llm_model": "gpt-5.4" }`。

適用範圍：

- RAG／Exam **出題**（`QUIZ_LLM_MODEL`）
- RAG／Exam **批改**（`GRADE_LLM_MODEL`）
- **個人／課程弱點分析**（`WEAKNESS_LLM_MODEL`）

---

#### `GET /rag/person_analysis_user_prompt_text`

取得個人分析 user prompt（所有有效使用者皆可讀取）。必填 query `course_id`。

```json
{
  "course_setting_id": 1,
  "course_id": 1,
  "person_analysis_user_prompt_text": "string（無設定時為 null）"
}
```

---

#### `PUT /rag/person_analysis_user_prompt_text`

寫入個人分析 user prompt（僅該課程 `user_type` 1／2 可操作）。必填 query `course_id`。

```json
{
  "course_setting_id": 1,
  "course_id": 1,
  "person_analysis_user_prompt_text": "string"
}
```

---

#### `GET /rag/course_analysis_user_prompt_text`

取得課程分析 user prompt（所有有效使用者皆可讀取）。必填 query `course_id`。

```json
{
  "course_setting_id": 1,
  "course_id": 1,
  "course_analysis_user_prompt_text": "string（無設定時為 null）"
}
```

---

#### `PUT /rag/course_analysis_user_prompt_text`

寫入課程分析 user prompt（僅該課程 `user_type` 1／2 可操作）。必填 query `course_id`。

```json
{
  "course_setting_id": 1,
  "course_id": 1,
  "course_analysis_user_prompt_text": "string"
}
```

---

### Exam 課程設定 `/exam`

#### `GET /exam/api_key`

讀取 `Course_Setting` key=`exam-api-key`（依 query `course_id`）。僅 `user_type` 1／2 可讀。

```json
{
  "course_id": 1,
  "api_key": "sk-…（無設定時為 null）"
}
```

---

#### `PUT /exam/api_key`

寫入 `exam-api-key`。僅 `user_type` 1／2 可操作。Body：`{ "api_key": "sk-…" }`。

---

### Prompt 模板 `/prompt`

#### `GET /prompt/templates`

回傳程式內建 LLM prompt 模板全文（**非** DB 動態值）。含占位符說明、RAG 檢索參數、出題／批改／弱點分析模板。

```json
{
  "placeholders": { "llm_generate": {}, "llm_grade": {} },
  "rag": {
    "llm_generate": { "retrieval_query": "課程重點概念", "retrieval_k": 5 },
    "llm_grade": { "retrieval_k": 5 },
    "build_defaults": { "chunk_size": 1000, "chunk_overlap": 200 }
  },
  "llm_generate": { "system": "…", "user": "…", "system_followup": "…", "user_followup": "…" },
  "llm_grade": { "system": "…", "user_transcript_course": "…", "user_faiss_course": "…" },
  "person_analysis": { "system": "…", "user": "…" },
  "course_analysis": { "system": "…", "user": "…" }
}
```

---

## 開發備註

| 主題 | 說明 |
|------|------|
| OpenAPI | `http://127.0.0.1:8000/docs`；部分 legacy 路徑 `include_in_schema=False` |
| 時區 | 時間戳記以台北時區（`utils/taipei_time.py`）格式化 |
| 軟刪除 | `deleted=true` 標記；查詢 filter `deleted.eq.false,deleted.is.null` |
| 評分 Job | 記憶體 dict；單 process 有效；Render 重啟或 scale-out 後 job_id 可能失效 |
| 敏感 Log | Middleware 遮罩 password、api_key、token 等欄位 |
| LLM 模型 | 出題／批改／弱點分析皆預設 **`gpt-5.4`**（`QUIZ_LLM_MODEL`／`GRADE_LLM_MODEL`／`WEAKNESS_LLM_MODEL` 共用）；可經 GET/PUT `/rag/llm_model`（`llm-model`）依課程覆寫；embedding 固定 `text-embedding-3-small` |

---

## 授權

本專案為 MyQuiz.ai 後端私有程式碼。使用前請遵守 OpenAI、Supabase、Deepgram 等第三方服務條款。
