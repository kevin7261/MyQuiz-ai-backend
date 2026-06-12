# 前端調整通知：批改結果輪詢 — LLM 失敗改回 `status: "error"`

> 變更日期：2026-06-13
> 影響面：**批改結果輪詢（answer-result）** 4 個端點
> 風險等級：低（回應結構、欄位名、HTTP 狀態碼皆不變，只有 LLM 失敗時 `status` 的「值」改變）

---

## 一句話總結

批改時若 **LLM 呼叫失敗**，後端**過去**會回 `status: "ready"` 且 `quiz_comments` 為空陣列，前端會誤判成「批改成功但沒有評語」。
**現在**改回 `status: "error"`，與「LLM 完成但寫入 DB 失敗」的既有行為一致。請前端把這種情況當成**批改失敗**處理。

---

## 受影響端點（4 個，行為一致）

| 端點 | `ready` 時附帶的回讀欄位 |
|------|--------------------------|
| `GET /v1/bank/qa/answer-result/{job_id}` | `bank_qa` |
| `GET /v1/quiz/qa/answer-result/{job_id}` | `quiz_qa` |
| `GET /v1/exam/quizzes/answer-result/{job_id}` | `exam_quiz` |
| `GET /v1/rag/quizzes/answer-result/{job_id}` | `rag_quiz` |

這 4 個端點共用同一個回應外殼，這次的行為改變對 4 個**完全一致**。

---

## 回應結構（沒有改變）

```jsonc
{
  "status": "pending" | "ready" | "error",
  "result": object | null,
  "error": string | null,
  "llm_error": string | null
  // 僅當 status == "ready" 時，才額外帶該端點的回讀欄位：
  // bank_qa / quiz_qa / exam_quiz / rag_quiz
}
```

- HTTP 狀態碼仍為 **200**（job 存在時），`status` 欄位才是真正的狀態來源。
- 只有「**查無此 job**」（服務重啟／冷啟動／job_id 打錯）才回 **HTTP 404**，內容為 `status: "error"` 帶說明。

---

## 差異對照（只有「LLM 批改失敗」這一種情況改變）

### 變更前 ❌（會誤導）

LLM 失敗時：

```jsonc
{
  "status": "ready",                       // ← 假的成功
  "result": {
    "llm_error": "...",
    "quiz_comments": [],                   // ← 空陣列，前端看起來像「沒有評語」
    "bank_qa_id": 123                      // (其他端點為 rag_quiz_id / exam_quiz_id ...)
  },
  "error": null,
  "llm_error": "...",
  "bank_qa": { /* 回讀的整列 */ }          // ← 因為 status=ready 而被一併回讀
}
```

### 變更後 ✅

LLM 失敗時：

```jsonc
{
  "status": "error",                       // ← 明確失敗
  "result": null,                          // ← 不再是物件
  "error": "<LLM 失敗原因>",
  "llm_error": "<LLM 失敗原因>"
  // 不再有 bank_qa / quiz_qa / exam_quiz / rag_quiz 欄位
}
```

> 成功（`status: "ready"`）與「進行中」（`status: "pending"`）的回應**完全沒有改變**。

---

## 前端要做的調整

1. **把 `status === "error"` 當成輪詢的終止失敗狀態**：停止輪詢，向使用者顯示錯誤（取 `error`，它與 `llm_error` 同值）。
   - 若你目前只處理 `pending` / `ready` 兩種，請補上 `error` 分支。
   - 「LLM 完成但寫 DB 失敗」本來就會回 `error`，所以多數情況下你**可能已有**這個分支；這次只是讓「LLM 失敗」也走同一條路。

2. **不要在 `status !== "ready"` 時讀 `result.*`**：失敗時 `result` 為 `null`，請用可選鏈（`result?.quiz_comments`）或先判斷 `status`。

3. **不要在 `status !== "ready"` 時期待回讀欄位**：`bank_qa` / `quiz_qa` / `exam_quiz` / `rag_quiz` 只在 `ready` 時出現。

4. **可移除舊的權宜判斷**（若有）：先前若靠「`status === "ready"` 但 `result.llm_error` 有值 / `quiz_comments` 為空」來偵測失敗，現在可直接改用 `status === "error"`。舊判斷留著也不會誤觸發，但建議清掉。

---

## 建議的輪詢處理（pseudo-code）

```js
const res = await fetch(`/v1/bank/qa/answer-result/${jobId}`, { headers });

if (res.status === 404) {
  // job 遺失（服務重啟／冷啟動）→ 提示重新送出批改
  stopPolling();
  showError("批改任務已遺失，請重新送出");
  return;
}

const data = await res.json();

switch (data.status) {
  case "pending":
    // 繼續輪詢
    break;

  case "error":
    stopPolling();
    showError(data.error || data.llm_error || "批改失敗");   // ← 新增 / 確認有這個分支
    break;

  case "ready":
    stopPolling();
    renderComments(data.result?.quiz_comments ?? []);
    renderRow(data.bank_qa);   // 其他端點對應 quiz_qa / exam_quiz / rag_quiz
    break;
}
```

---

## 不需要改的部分

- 請求方法、路徑、Query/Path 參數、Header（含 Bearer token）：**全部不變**。
- 回應的欄位名稱與結構：**不變**。
- 成功與進行中的回應：**不變**。
- 唯一改變：**LLM 批改失敗** 時 `status` 從 `"ready"` → `"error"`，且 `result` 由物件變為 `null`。
