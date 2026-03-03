"""
從 RAG ZIP（FAISS 向量庫）載入後檢索 Context，呼叫 GPT-4o 生成題目。
"""

import json
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from openai import OpenAI


# 常見 GIS / 資料檔副檔名，用於提供給 AI 的檔案列表
GIS_EXTENSIONS = {".shp", ".tif", ".tiff", ".gpkg", ".csv", ".rds", ".geojson", ".json"}

# 預設出題系統指令（會與 API 傳入的 system_prompt_instruction 一併使用，放在其上方）
SYSTEM_INSTRUCTION_PREDEFINE = """
            1. **請務必使用繁體中文 (Traditional Chinese) 出題。**
            2. 在 'question_content' (題目) 中：只說明**任務目標**。嚴禁直接列出步驟 1, 2, 3。請保留思考空間給學生。
            3. 在 'hint' (提示) 中：才列出詳細的解題步驟。
        """


def get_gis_filenames(extract_folder: Path) -> list[str]:
    """掃描解壓縮目錄，回傳 GIS/資料相關檔名列表（供 AI 出題時選擇）。"""
    names: list[str] = []
    extract_folder = Path(extract_folder)
    if not extract_folder.exists():
        return names
    for root, _dirs, files in os.walk(extract_folder):
        for f in files:
            if Path(f).suffix.lower() in GIS_EXTENSIONS:
                names.append(f)
    return sorted(set(names))


def generate_question(
    zip_path: Path,
    api_key: str,
    qtype: str,
    level: str,
    system_prompt_instruction: str,
    course_name: str,
) -> dict:
    """
    從現成 RAG ZIP（含 FAISS 向量庫）解壓 → 載入向量庫 → 檢索 → 呼叫 GPT-4o 出題。
    僅支援由 /zip/pack 產出的 RAG ZIP，不支援一般講義 ZIP。
    system_prompt_instruction 為必填參數，由 API 呼叫端傳入出題系統指令。
    course_name 為課程名稱，會帶入出題 prompt 中。
    回傳 {"question_content": "...", "hint": "...", "answer": "..."}。
    """
    if not api_key or not api_key.strip():
        raise ValueError("請傳入 openai_api_key")
    if not system_prompt_instruction or not system_prompt_instruction.strip():
        raise ValueError("請傳入 system_prompt_instruction（出題系統指令，必填）")
    if not course_name or not course_name.strip():
        raise ValueError("請傳入 course_name（課程名稱，必填）")

    extract_folder = Path(tempfile.mkdtemp())
    try:
        if not zipfile.is_zipfile(zip_path):
            raise ValueError("無效的 ZIP 檔")

        # 使用 UTF-8 解壓，避免非 ASCII 檔名或路徑造成 'ascii' codec can't encode 錯誤
        zip_kw: dict = {}
        if sys.version_info >= (3, 11):
            zip_kw["metadata_encoding"] = "utf-8"
        with zipfile.ZipFile(zip_path, "r", **zip_kw) as z:
            z.extractall(extract_folder)

        db_folder = None
        for root, _dirs, files in os.walk(extract_folder):
            if "index.faiss" in files and "index.pkl" in files:
                db_folder = root
                break
        if not db_folder:
            raise ValueError("此 API 僅支援 RAG ZIP（由 /zip/pack 產出），請上傳含 FAISS 向量庫的 ZIP")

        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)
        vectorstore = FAISS.load_local(
            db_folder, embeddings, allow_dangerous_deserialization=True
        )

        file_names = get_gis_filenames(extract_folder)
        file_names_str = ", ".join(file_names) if file_names else "None"

        query = f"空間分析 {level} {qtype} 重點概念與操作步驟"
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        docs = retriever.invoke(query)
        context_text = "\n\n".join([d.page_content for d in docs])

        sys_role = f"你是頂尖的「{course_name}」課程助教。請使用 GPT-4o 的強大邏輯來出題。"

        task_instruction = f"目前的題型任務是：【{qtype}】。難度：{level}。"
        core_point = "🔥 **本次題目核心考點：請根據以下參考講義內容設計**"

        final_system_prompt = f"""
            {sys_role}
            {task_instruction}
            {core_point}
            (Please design the question around the core concept above.)
            【出題重要規範】
            {SYSTEM_INSTRUCTION_PREDEFINE}
            {system_prompt_instruction}
            請以 JSON 格式回傳：
            {{ "question_content": "Question content (Markdown)...", 
              "hint": "Hint for students...", 
              "answer": "Answer for students..." }}
        """
        user_prompt_text = f"參考講義內容：\n{context_text}"

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
        return json.loads(content)
    finally:
        shutil.rmtree(extract_folder, ignore_errors=True)
