# Prompt 撰寫指南（如何下 Prompt）

本文件說明本後端各模組的 prompt 怎麼下、存在哪、有哪些可用佔位符、優先級如何決定，以及實際的 API 操作。

> 核心觀念：**system / user 模板是寫死在程式碼裡的，你不能改模板本身。** 你能改的只有「填進佔位符的那段文字」（例如出題指令、批改指令、分析指令）。出題／批改指令存在**題組欄位**（由前端輸入，無預設）；分析指令存在 `Course_Setting`，皆透過 API 設定。

---

## 1. 三種你能下的 Prompt

| 種類 | 你下的內容 | 填入的佔位符 | 影響 |
|------|-----------|-------------|------|
| **出題 system prompt** | 連續出題的整體規定（題型、難度遞增、不重複） | `{quiz_system_prompt_text}` | 織入 system「指令優先級」區塊，**最高優先** |
| **出題 user prompt** | 這一題要怎麼出 | `{quiz_user_prompt_text}` | 單題出題指令 |
| **批改 user prompt** | 批改的標準與重點 | `{answer_user_prompt_text}` | 批改指令 |
| **分析 user prompt** | 弱點／課程分析的語氣、結構、聚焦 | `{analysis_user_prompt_text}` | 分析報告指令 |

---

## 2. Prompt 存在哪？

- **出題／批改 prompt（Bank／Quiz）** 存在**題組欄位**（`Bank_Group`／`Quiz_Group` 的 `question_system_prompt_text`／`question_user_prompt_text`／`answer_user_prompt_text`）。一律由前端輸入，**後端不提供任何預設**。
- **分析 prompt（user／quiz／person／course analysis）** 仍存在 **`Course_Setting`** 表，以 `(course_id, key)` 唯一（`utils/course_setting.py`）。

### Course_Setting key 對照（分析指令）

| 模組 / 用途 | key |
|------------|-----|
| User Analysis（個人弱點） | `user_analysis_user_prompt_text` |
| Quiz Analysis（課程整體） | `quiz_analysis_user_prompt_text` |
| Person Analysis | `person_analysis_user_prompt_text` |
| Course Analysis | `course_analysis_user_prompt_text` |

### prompt 來源（出題／批改）

建題組時，三個 prompt 欄位**直接存前端送入的值**：

```
建題組 body 的值 → 原樣存進題組欄位（送空字串就存空字串）
```

之後可用 per-group `PUT /v1/bank/groups/{id}/…-prompt-text`（或 quiz 對應端點）改某一項。

> 沒有任何 fallback：**沒有課程預設、也沒有程式內建預設**。留空就是空字串，呼叫 LLM 時該佔位符即為空。要客製就自己填文字。

---

## 3. 可用佔位符清單

下表是各區塊 user/system 模板裡會被自動填入的 `{佔位符}`。**這些由系統填，你不需要也不能在自己的指令裡手動寫它們**，列出來是讓你理解「模型實際會看到什麼」。
（來源：`services/prompt_placeholders.py`，可由 `GET /v1/prompt-templates` 取得完整說明）

### 出題（llm-generate）

| 佔位符 | 內容來源 |
|--------|---------|
| `{quiz_system_prompt_text}` | 題組出題 system 規定（你下的） |
| `{quiz_user_prompt_text}` | 出題 user 指令（你下的） |
| `{quiz_history_body}` | 同題組已出過的題目（自動帶，避免重複） |
| `{ask_history_body}` | 追問紀錄（僅 Quiz 路徑） |
| `{context_md}` | 課程內容：FAISS 檢索片段或逐字稿全文 |

### 批改（llm-answer）

| 佔位符 | 內容來源 |
|--------|---------|
| `{quiz_user_prompt_text}` | 出題 user prompt（空則「（未提供）」） |
| `{answer_user_prompt_text}` | 批改 user 指令（你下的） |
| `{quiz_content}` | 本題題幹 |
| `{quiz_answer}` | 學生作答 |
| `{context_md}` | 課程內容參考 |
| `{id_block}` | 關聯識別區塊（選填） |

### 分析（user/quiz/person/course analysis）

| 佔位符 | 內容來源 |
|--------|---------|
| `{analysis_user_prompt_text}` | 分析指令（你下的，空則「（未提供）」） |
| `{material_md}` | 測驗素材：已批改題目的 `answer_critique` 摘要 |

---

## 4. 怎麼下 Prompt（API 操作）

### 4.1 分析指令（User / Quiz Analysis）

設定課程級分析指令：

```http
PUT /user-analyses/analysis-user-prompt-text?person_id=teacher001&course_id=123
Content-Type: application/json

{ "analysis_user_prompt_text": "請著重分析該生的計算錯誤模式，給出具體練習方向。" }
```

讀取目前設定：

```http
GET /user-analyses/analysis-user-prompt-text?person_id=teacher001&course_id=123
→ { "course_id": 123, "analysis_user_prompt_text": "..." }
```

Quiz Analysis 同形式，路徑改 `/quiz-analyses/analysis-user-prompt-text`。

### 4.2 Bank / Quiz 題組出題、批改指令

**建題組時直接帶 prompt**（原樣存入；留空即存空字串，無預設）：

```http
POST /bank/pages/{bank_page_id}/units/{bank_unit_id}/groups?caller_person_id=teacher001&course_id=123

{
  "group_name": "第一回測驗",
  "qa_count": 5,
  "question_system_prompt_text": "",
  "question_user_prompt_text": "",
  "answer_user_prompt_text": ""
}
```

**之後只改某一項**：

```http
PUT /bank/groups/{bank_group_id}/question-user-prompt-text?caller_person_id=teacher001&course_id=123

{ "question_user_prompt_text": "請根據向量分析主題出一道深入思考的題目。" }
```

**觸發出題**（可用 body 臨時覆蓋，不填則用題組設定）：

```http
POST /bank/groups/{bank_group_id}/qa/llm-generate?caller_person_id=teacher001&course_id=123
{ "question_user_prompt_text": "" }
```

### 4.3 查所有模板與佔位符說明

```http
GET /v1/prompt-templates?person_id=user001
```

回傳所有模組（bank / quiz / rag / exam / 各 analysis）的 system/user 模板原文 + 佔位符說明，是「想知道模型實際看到什麼」時的權威來源。

---

## 5. 撰寫建議

1. **出題 system 寫「整體規則」，出題 user 寫「這一題」。** system 控制全題組行為（難度遞增、題型、不重複），user 控制單題焦點。
2. **指令會壓過課程內容與泛化規則。** 模板明定：你下的 user/system 指令優先級高於課程片段與內建規範。要嚴格遵守某題型／格式，直接寫清楚。
3. **不用自己塞佔位符。** `{context_md}`、`{quiz_history_body}` 等由系統填，你只寫自然語言指令。
4. **批改指令聚焦「評什麼」。** 注意批改輸出是純文字評語（`answer_critique`），**不含分數**，不要要求模型給分。
5. **分析指令控制報告形狀。** 語氣、結構、弱點聚焦、篇幅都在 `analysis_user_prompt_text` 裡指定，輸出為 Markdown 報告。
6. **留空就是空。** 出題／批改 prompt 沒有任何 fallback——留空即存空字串，該佔位符對模型而言就是空。要有內容就自己填。

---

## 6. 模型與金鑰

各模組金鑰／模型分開存於 `Course_Setting`：

| 模組 | API key | 模型 key |
|------|---------|---------|
| RAG | `rag-api-key` | `llm-model` |
| Exam | `exam-api-key` | `llm-model` |
| Bank | `bank-api-key` | `bank-llm-model` |
| Quiz | `quiz-api-key` | `quiz-llm-model` |

出題／批改／分析皆以 `response_format=json_object`（分析為 Markdown）呼叫，預設模型 `gpt-5.4`，題組層級可用 `question_llm_model` / `answer_llm_model` 覆蓋。

---

## 參考檔案

- `utils/course_setting.py` — Course_Setting 讀寫、key 常數（分析指令等）
- `services/prompt_placeholders.py` — 佔位符完整說明
- `services/quiz_generation.py` / `services/bank_generation.py` — 出題模板與組裝
- `services/answering.py` / `services/bank_answering.py` — 批改模板
- `services/weakness_report.py` — 分析報告模板
- `routers/analysis_prompt_settings.py` — 分析指令 GET/PUT 端點
- `routers/prompt.py` — `GET /v1/prompt-templates`
