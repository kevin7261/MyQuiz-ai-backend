-- 讓 anon / Publishable key 可以讀取 "User" 表（RLS 政策）
-- 在 Supabase 後台：SQL Editor → 貼上整段 → Run

-- 若表名在 DB 裡是小寫 "user"，把下面 "User" 改成 "user"
ALTER TABLE "User" ENABLE ROW LEVEL SECURITY;

-- 允許 anon 角色 SELECT（只讀）
CREATE POLICY "Allow anon to read User"
  ON "User"
  FOR SELECT
  TO anon
  USING (true);

-- 若之後要限制「只給登入者看」可改成：
-- CREATE POLICY "Allow authenticated to read User"
--   ON "User" FOR SELECT TO authenticated USING (true);
