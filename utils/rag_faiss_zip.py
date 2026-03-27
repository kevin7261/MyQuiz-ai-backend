"""
從講義 ZIP 建 FAISS 向量庫並打包成 RAG ZIP；含 process_zip_to_docs（與建庫相同副檔名規則）。
Embeddings 由呼叫端傳入 API key，不從環境變數讀取。
"""

# 引入 io 模組，用於 BytesIO 等
import io
# 引入 os 模組，用於 os.walk
import os
# 引入 shutil，用於刪除暫存目錄
import shutil
# 引入 tempfile，用於建立暫存目錄
import tempfile
# 引入 zipfile，用於讀寫 ZIP
import zipfile
# 引入 Path 用於路徑操作
from pathlib import Path

# LangChain 文字切分器
from langchain_text_splitters import RecursiveCharacterTextSplitter
# OpenAI Embeddings
from langchain_openai import OpenAIEmbeddings
# FAISS 向量庫
from langchain_community.vectorstores import FAISS
# 文件載入器
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader
# Document 型別
from langchain_core.documents import Document

# 引入 zip_utils 的 fix_encoding，用於修正檔名亂碼
from utils.zip_utils import fix_encoding

# ZIP 解壓後會嘗試載入的副檔名（小寫）；.doc / .ppt 依賴 unstructured，環境缺少套件或系統工具時可能略過
_SUPPORTED_DOC_EXTS: frozenset[str] = frozenset(
    {".pdf", ".doc", ".docx", ".ppt", ".pptx"}
)


def _documents_from_pptx(path: Path) -> list[Document]:
    from pptx import Presentation

    prs = Presentation(str(path))
    chunks: list[str] = []
    for idx, slide in enumerate(prs.slides):
        lines: list[str] = []
        for shape in slide.shapes:
            if not hasattr(shape, "text"):
                continue
            t = (shape.text or "").strip()
            if t:
                lines.append(t)
        if lines:
            chunks.append(f"Slide {idx + 1}\n" + "\n".join(lines))
    text = "\n\n".join(chunks).strip()
    if not text:
        return []
    return [Document(page_content=text, metadata={"source": str(path)})]


def _load_docs_from_file(path: Path) -> list[Document]:
    ext = path.suffix.lower()
    if ext not in _SUPPORTED_DOC_EXTS:
        return []
    try:
        if ext == ".pdf":
            return PyPDFLoader(str(path)).load()
        if ext == ".docx":
            return Docx2txtLoader(str(path)).load()
        if ext == ".doc":
            from langchain_community.document_loaders import UnstructuredWordDocumentLoader

            return UnstructuredWordDocumentLoader(str(path)).load()
        if ext == ".pptx":
            return _documents_from_pptx(path)
        if ext == ".ppt":
            from langchain_community.document_loaders import UnstructuredPowerPointLoader

            return UnstructuredPowerPointLoader(str(path)).load()
    except Exception:
        return []
    return []


def process_zip_to_docs(zip_path: Path, extract_dir: Path) -> list[Document]:
    """
    解壓 ZIP 到 extract_dir，載入支援的檔案為 Document 列表。
    副檔名：.pdf, .doc, .docx, .ppt, .pptx（不分大小寫）。
    過濾 __MACOSX、.DS_Store，並修正編碼。
    """
    all_docs: list[Document] = []  # 存放所有 Document
    extract_dir = Path(extract_dir)  # 確保為 Path
    extract_dir.mkdir(parents=True, exist_ok=True)  # 建立解壓目錄

    with zipfile.ZipFile(zip_path, "r") as z:  # 開啟 ZIP
        for raw_name in z.namelist():  # 遍歷 ZIP 內所有檔名
            # 跳過目錄
            if raw_name.endswith("/"):
                continue
            # 修正編碼
            decoded = fix_encoding(raw_name)
            # 跳過 macOS 隱藏檔
            if "__MACOSX" in decoded or ".DS_Store" in decoded:
                continue
            # 解壓到與 decoded 相同的相對路徑，將 .. 替換為 _ 防止路徑穿越
            safe_path = extract_dir / decoded.replace("..", "_")
            safe_path.parent.mkdir(parents=True, exist_ok=True)  # 確保父目錄存在
            safe_path.write_bytes(z.read(raw_name))  # 寫入檔案內容

    for path in extract_dir.rglob("*"):
        if not path.is_file():
            continue
        all_docs.extend(_load_docs_from_file(path))

    return all_docs


def build_faiss_zip_from_docs(
    documents: list[Document],
    api_key: str,
    chunk_size: int,
    chunk_overlap: int,
) -> bytes:
    """
    將 Document 列表做切分、Embedding、FAISS 建索引，再打包成 ZIP 的 bytes。
    """
    if not documents:
        raise ValueError("無文件可處理")

    # 建立文字切分器
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    split_docs = text_splitter.split_documents(documents)

    # 建立 Embeddings（須與評分時一致）
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)
    vectorstore = FAISS.from_documents(split_docs, embeddings)

    tmpdir = Path(tempfile.mkdtemp())
    try:
        vectorstore.save_local(str(tmpdir))  # 將 FAISS 存到 tmpdir
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _dirs, files in os.walk(tmpdir):
                for f in files:
                    fp = Path(root) / f
                    arcname = os.path.relpath(fp, tmpdir)
                    zf.write(fp, arcname)
        return buf.getvalue()
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)  # 清理暫存


def make_rag_zip_from_zip_path(
    zip_path: Path,
    api_key: str,
    chunk_size: int,
    chunk_overlap: int,
) -> bytes:
    """
    從一個 ZIP 路徑：解壓 → 載入文件 → 切分 → Embedding → FAISS → 打包成 ZIP。
    回傳 ZIP 的 bytes。
    """
    tmp_extract = Path(tempfile.mkdtemp())
    try:
        all_docs = process_zip_to_docs(zip_path, tmp_extract)
        if not all_docs:
            raise ValueError(
                "ZIP 內無可讀文件（支援：.pdf .doc .docx .ppt .pptx）"
            )
        return build_faiss_zip_from_docs(
            all_docs, api_key, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
    finally:
        shutil.rmtree(tmp_extract, ignore_errors=True)  # 清理暫存
