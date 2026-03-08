"""
從 ZIP 製作 RAG 用 FAISS 向量庫，並打包成 ZIP 供下載。
LLM API key 由呼叫端傳入，不從環境變數讀取。
"""

import io
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

from utils.zip_utils import fix_encoding


def process_zip_to_docs(zip_path: Path, extract_dir: Path) -> list[Document]:
    """
    解壓 ZIP 到 extract_dir，載入支援的檔案（.pdf, .txt）為 Document 列表。
    過濾 __MACOSX、.DS_Store，並修正編碼。
    """
    all_docs: list[Document] = []
    extract_dir = Path(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as z:
        for raw_name in z.namelist():
            if raw_name.endswith("/"):
                continue
            decoded = fix_encoding(raw_name)
            if "__MACOSX" in decoded or ".DS_Store" in decoded:
                continue
            # 解壓到與 decoded 相同的相對路徑
            safe_path = extract_dir / decoded.replace("..", "_")
            safe_path.parent.mkdir(parents=True, exist_ok=True)
            safe_path.write_bytes(z.read(raw_name))

    for ext, loader_class in [(".pdf", PyPDFLoader), (".txt", TextLoader)]:
        for path in extract_dir.rglob(f"*{ext}"):
            try:
                loader = loader_class(str(path))
                docs = loader.load()
                all_docs.extend(docs)
            except Exception:
                continue

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

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    split_docs = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)
    vectorstore = FAISS.from_documents(split_docs, embeddings)

    tmpdir = Path(tempfile.mkdtemp())
    try:
        vectorstore.save_local(str(tmpdir))
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _dirs, files in os.walk(tmpdir):
                for f in files:
                    fp = Path(root) / f
                    arcname = os.path.relpath(fp, tmpdir)
                    zf.write(fp, arcname)
        return buf.getvalue()
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


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
            raise ValueError("ZIP 內無支援的文件（需 .pdf 或 .txt）")
        return build_faiss_zip_from_docs(
            all_docs, api_key, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
    finally:
        shutil.rmtree(tmp_extract, ignore_errors=True)
