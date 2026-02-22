"""
從 RAG ZIP（FAISS 向量庫）載入後檢索 Context，呼叫 GPT-4o 生成題目。
"""

import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from openai import OpenAI

from utils.rag import process_zip_to_docs


# 常見 GIS / 資料檔副檔名，用於提供給 AI 的檔案列表
GIS_EXTENSIONS = {".shp", ".tif", ".tiff", ".gpkg", ".csv", ".rds", ".geojson", ".json"}


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
) -> dict:
    """
    從 RAG ZIP（或內含 FAISS 的 zip）解壓 → 載入向量庫 → 檢索 → 呼叫 GPT-4o 出題。
    回傳 {"question_content": "...", "hint": "...", "target_filename": "..."}。
    """
    if not api_key or not api_key.strip():
        raise ValueError("請傳入 openai_api_key")

    extract_folder = Path(tempfile.mkdtemp())
    try:
        if not zipfile.is_zipfile(zip_path):
            raise ValueError("無效的 ZIP 檔")

        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(extract_folder)

        is_rag_db = False
        db_folder = None
        for root, _dirs, files in os.walk(extract_folder):
            if "index.faiss" in files and "index.pkl" in files:
                is_rag_db = True
                db_folder = root
                break

        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)

        if is_rag_db and db_folder:
            vectorstore = FAISS.load_local(
                db_folder, embeddings, allow_dangerous_deserialization=True
            )
        else:
            all_documents = process_zip_to_docs(zip_path, extract_folder)
            if not all_documents:
                raise ValueError("ZIP 內無支援的講義文件（需 .pdf 或 .txt）")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200
            )
            split_docs = text_splitter.split_documents(all_documents)
            vectorstore = FAISS.from_documents(split_docs, embeddings)

        file_names = get_gis_filenames(extract_folder)
        file_names_str = ", ".join(file_names) if file_names else "None"

        query = f"空間分析 {level} {qtype} 重點概念與操作步驟"
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        docs = retriever.invoke(query)
        context_text = "\n\n".join([d.page_content for d in docs])

        sys_role = "你是頂尖的空間分析助教。請使用 GPT-4o 的強大邏輯來出題。"
        r_rules = """⚠️ 嚴格限制：
1. 實作內容必須限定使用 **R 語言** (例如使用 sf, terra, tmap, tidyverse 等套件)。
2. 🚫 禁止提及 "ArcGIS", "QGIS" 或通用的 "GIS 軟體" 字眼。
3. 題目應引導學生寫出 R 程式碼來解決問題。
4. **請務必使用繁體中文 (Traditional Chinese) 出題。**"""

        system_instruction = f"""你必須從提供的「真實檔案列表」中選擇一個檔案來設計操作任務。
真實檔案列表: [{file_names_str}]
(若選擇 Shapefile，請只提及 .shp 檔，不要提及 .dbf 或 .shx)
{r_rules}
【出題重要規範】
1. 在 'question_content' (題目) 中：只說明**任務目標**與**使用資料**。❌ 嚴禁直接列出步驟 1, 2, 3。請保留思考空間給學生。
2. 在 'hint' (提示) 中：才列出詳細的解題步驟、建議使用的 R 套件與函數。"""

        task_instruction = f"目前的題型任務是：【{qtype}】。難度：{level}。"
        core_point = "🔥 **本次題目核心考點：請根據以下參考講義內容設計**"

        final_system_prompt = f"""
{sys_role}
{task_instruction}
{core_point}
(Please design the question around the core concept above.)
{system_instruction}
請以 JSON 格式回傳：
{{ "question_content": "Question content (Markdown)...", "hint": "Hint for students...", "target_filename": "AI選擇的檔案名稱" }}
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
