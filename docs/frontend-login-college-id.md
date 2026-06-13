# 前端異動說明：登入身分改為 person_id + college_id

> 異動日期：2026-06-13　影響範圍：登入與所有需認證的 API

## TL;DR — 前端要改的 3 件事

1. **登入表單加「學校」欄位**，`POST /v1/auth/login` 的 body 多帶 `college_id`（int，必填）。
2. **全域攔截 401**：所有舊 token（改版前發出的）會被後端拒絕，需清掉本地 token、導回登入頁。**這是強制的，舊 token 不再相容。**
3. **處理新錯誤碼**：登入時 `422`（沒帶 college_id）、`403`（學校不符）。

> 後續一般 API 的呼叫方式**完全不變**（照舊帶 `Authorization: Bearer <token>`）；`college_id` 已包在 token 裡，前端不用自己存、也不用每次帶。

---

## 1. 登入 API 變更 `POST /v1/auth/login`

`college_id`（學校 id）**變成必填**。

### Request body

```json
{
  "person_id": "string",
  "password": "string",
  "college_id": 1
}
```

| 欄位 | 型別 | 必填 | 說明 |
|------|------|:---:|------|
| person_id | string | ✅ | 帳號 |
| password | string | ✅ | 密碼 |
| college_id | int | ✅ | 學校（學院）id |

### Response（成功 200，結構與原本相同）

```json
{
  "user": { "user_id": 1, "person_id": "...", "college_id": "1", "courses": [] },
  "courses": [],
  "access_token": "xxxx.yyyy",
  "token_type": "bearer",
  "expires_in": 2592000
}
```

`access_token` 內部已包含這次登入的 `college_id`，後續請求照舊只放 `Authorization: Bearer <access_token>`。

### 登入錯誤碼

| 狀態碼 | 情境 | 前端處理 |
|--------|------|------|
| 422 | 沒帶 `college_id` 或型別錯 | 檢查表單是否帶到學校 |
| 401 | 帳號或密碼錯誤 | 提示帳密錯誤 |
| 403 | 此帳號不屬於指定的學校 | 提示選錯學校 |

---

## 2. 舊 token 一律失效（務必處理）

身分改版後，**改版前發出、不含 `college_id` 的舊 token，呼叫任何需認證的端點都會回 401**：

```json
{ "detail": "token 缺少學校資訊（college_id），請重新登入" }
```

前端需在 API 層全域攔截：**收到 401 → 清掉本地 token → 導回登入頁**。範例（axios）：

```js
api.interceptors.response.use(
  (res) => res,
  (err) => {
    if (err.response?.status === 401) {
      localStorage.removeItem("access_token"); // 清掉舊 token
      router.push("/login");                   // 導回登入
    }
    return Promise.reject(err);
  }
);
```

> 上線後所有現有使用者都會被登出一次、需重新登入，這是預期行為。

---

## 3. 行為更正確的端點（呼叫方式不用改）

以下端點現在會依 token 內的 `college_id` 鎖定「登入學校那一筆」帳號，前端**不用改**，只是行為更正確：

| 端點 | 變化 |
|------|------|
| `GET /v1/users/me` | 跨校同 person_id 時，只回傳**登入學校**那一筆 profile |
| `PUT /v1/users/me/password` | 只改**登入學校**那一筆密碼（修正：原本會一次改掉所有同 person_id 帳號）|
| `POST /v1/auth/refresh` | 換發的新 token 沿用原本的 person_id + college_id |

---

## 4. 常見問題

**Q：後續每支 API 要不要自己帶 college_id？**
不用。token 內已含 college_id，後端自動解析。

**Q：`expires_in` 有變嗎？**
沒有，預設一樣 30 天。token 接近到期可呼叫 `POST /v1/auth/refresh` 換發（保留學校情境）。

**Q：同一個人在兩間學校怎麼處理？**
視為兩個不同帳號，各自用對應的 `college_id` 登入、拿到各自的 token。

**Q：學校清單（college_id）哪裡來？**
`GET /v1/colleges` **已開放免 token**，登入頁未登入即可直接呼叫取得學校下拉清單（回應結構不變，沿用原本下拉做法即可），讓使用者選後送出時帶該校的 `college_id`。
