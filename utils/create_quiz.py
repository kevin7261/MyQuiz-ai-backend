"""
從 RAG ZIP（FAISS 向量庫）載入後檢索 Context，呼叫 GPT-4o 生成測驗。
"""

# 引入 json 用於解析 LLM 回傳
import json
# 引入 os 用於 os.walk
import os
# 引入 shutil 用於刪除暫存目錄
import shutil
# 引入 sys 用於 version_info 判斷
import sys
# 引入 tempfile 建立暫存目錄
import tempfile
# 引入 zipfile 讀取 ZIP
import zipfile
# 引入 Path 用於路徑操作
from pathlib import Path

# LangChain OpenAI Embeddings
from langchain_openai import OpenAIEmbeddings
# FAISS 向量庫
from langchain_community.vectorstores import FAISS
# OpenAI 客戶端
from openai import OpenAI

def generate_quiz(
    zip_path: Path,
    api_key: str,
    quiz_level: str,
    system_prompt_instruction: str,
) -> dict:
    """
    從現成 RAG ZIP（含 FAISS 向量庫）解壓 → 載入向量庫 → 檢索 → 呼叫 GPT-4o 出題。
    僅支援由 build-rag-zip 產出的 RAG ZIP，不支援一般講義 ZIP。
    system_prompt_instruction 為必填參數，由 API 呼叫端傳入出題系統指令。
    回傳 {"quiz_content", "quiz_hint", "reference_answer"}；API 層會再加上 system_prompt_instruction、quiz_level 等。
    """
    if not api_key or not api_key.strip():
        raise ValueError("請傳入 llm_api_key")
    if not system_prompt_instruction or not system_prompt_instruction.strip():
        raise ValueError("請傳入 system_prompt_instruction（出題系統指令，必填）")

    extract_folder = Path(tempfile.mkdtemp())  # 建立暫存目錄
    try:
        if not zipfile.is_zipfile(zip_path):
            raise ValueError("無效的 ZIP 檔")

        # 使用 UTF-8 解壓，避免非 ASCII 檔名或路徑造成 'ascii' codec can't encode 錯誤（Python 3.11+）
        zip_kw: dict = {}
        if sys.version_info >= (3, 11):
            zip_kw["metadata_encoding"] = "utf-8"  # Python 3.11+ 支援 UTF-8 檔名
        with zipfile.ZipFile(zip_path, "r", **zip_kw) as z:
            z.extractall(extract_folder)

        # 尋找含 index.faiss 與 index.pkl 的目錄（FAISS 向量庫）
        db_folder = None
        for root, _dirs, files in os.walk(extract_folder):
            if "index.faiss" in files and "index.pkl" in files:
                db_folder = root
                break
        if not db_folder:
            raise ValueError("此 API 僅支援 RAG ZIP（由 build-rag-zip 產出），請上傳含 FAISS 向量庫的 ZIP")

        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)
        vectorstore = FAISS.load_local(  # 載入 FAISS 向量庫
            db_folder, embeddings, allow_dangerous_deserialization=True
        )

        # 檢索查詢：空間分析 + 難度 + 重點概念與操作步驟
        query = f"課程 {quiz_level} 的重點概念"
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        docs = retriever.invoke(query)
        context_text = "\n\n".join([d.page_content for d in docs])

        final_system_prompt = f"""
            【出題規範】
            請根據輸入的「參考內容」設計測驗題目。
            請使用繁體中文 (Traditional Chinese) 出題與撰寫提示及參考答案。
            題目難度：{quiz_level}。
            **{system_prompt_instruction}**
            【回傳格式】
            請以 JSON 格式回傳：
            {{ "quiz_content": "題目內容", 
              "quiz_hint": "答案提示內容", 
              "reference_answer": "參考答案內容" }}
        """
        user_prompt_text = f"參考內容：\n{context_text}"

        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": final_system_prompt},
                {"role": "user", "content": user_prompt_text},
            ],
            response_format={"type": "json_object"},
            temperature=0.7,
        )

        content = response.choices[0].message.content
        data = json.loads(content)  # 解析 JSON
        # 統一使用 reference_answer（參考答案）；相容舊回傳 answer
        if "reference_answer" not in data and "answer" in data:
            data["reference_answer"] = data.pop("answer")
        # 統一使用 quiz_hint；相容舊回傳 hint
        if "quiz_hint" not in data and "hint" in data:
            data["quiz_hint"] = data.pop("hint")
        return data
    finally:
        shutil.rmtree(extract_folder, ignore_errors=True)  # 清理暫存
