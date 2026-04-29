"""從 RAG Storage（upload ZIP）依單元資料夾名擷取音訊、YouTube 或文字（.md），供 /rag/transcript/* 使用。"""

from __future__ import annotations

import logging
import re
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Iterator

from utils.english_system_transcript import parse_youtube_video_id
from utils.zip_storage import UPLOAD_DEFAULT_PERSON, get_zip_path_by_person
from utils.zip_utils import fix_encoding

logger = logging.getLogger(__name__)

_AUDIO_EXTS = frozenset({
    ".mp3", ".wav", ".m4a", ".webm", ".ogg", ".flac", ".opus", ".mp4", ".mpeg", ".mpga", ".aac", ".wma",
})

_MD_SUFFIXES = frozenset({".md", ".markdown"})


def _is_markdown_path(decoded_path: str) -> bool:
    return Path(decoded_path).suffix.lower() in _MD_SUFFIXES


def _zip_members(z: zipfile.ZipFile) -> Iterator[tuple[str, str]]:
    for raw in z.namelist():
        if raw.endswith("/"):
            continue
        dec = fix_encoding(raw)
        if "__MACOSX" in dec or ".DS_Store" in dec:
            continue
        yield raw, dec


def path_has_folder_segment(decoded_path: str, folder_name: str) -> bool:
    """路徑任一段與 folder_name 完全相同即視為該單元底下。"""
    needle = (folder_name or "").strip()
    if not needle or "/" in needle or "\\" in needle:
        return False
    parts = decoded_path.replace("\\", "/").strip("/").split("/")
    return needle in parts


def read_upload_zip_bytes(person_id: str, rag_tab_id: str) -> bytes:
    """自 Supabase RAG bucket 的 upload 區下載該 tab 的 ZIP，讀成 bytes（暫存檔會刪除）。"""
    pid = (person_id or "").strip() or UPLOAD_DEFAULT_PERSON
    rid = (rag_tab_id or "").strip()
    if not rid or "/" in rid or "\\" in rid:
        raise ValueError("無效的 rag_tab_id")
    tmp = get_zip_path_by_person(pid, rid)
    if not tmp or not tmp.exists():
        raise FileNotFoundError(f"找不到 upload ZIP（person_id={pid}, rag_tab_id={rid}）")
    try:
        return tmp.read_bytes()
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            logger.debug("刪除暫存 upload ZIP 失敗", exc_info=True)


def pick_audio_from_upload_zip(zip_bytes: bytes, folder_name: str) -> tuple[bytes, str, str]:
    """回傳 (音訊 bytes, 副檔名含點, ZIP 內解碼路徑)。"""
    fn = (folder_name or "").strip()
    if not fn:
        raise ValueError("請傳入 folder_name（ZIP 內單元資料夾名稱）")
    if "/" in fn or "\\" in fn:
        raise ValueError("folder_name 不可含路徑分隔字元")

    with zipfile.ZipFile(BytesIO(zip_bytes), "r") as z:
        candidates: list[tuple[str, str, str]] = []
        for raw, dec in _zip_members(z):
            if not path_has_folder_segment(dec, fn):
                continue
            suf = Path(dec).suffix.lower()
            if suf not in _AUDIO_EXTS:
                continue
            candidates.append((raw, dec, suf))

        candidates.sort(key=lambda x: x[1])
        if not candidates:
            has_md = any(
                path_has_folder_segment(dec, fn) and _is_markdown_path(dec)
                for _raw, dec in _zip_members(z)
            )
            if has_md:
                raise ValueError(
                    f"於資料夾「{fn}」下沒有音訊檔，但偵測到 Markdown（.md）。"
                    "若為 **YouTube 單元**（.md 內含影片連結），請改呼叫 **GET /rag/transcript/youtube**，"
                    "使用相同的 person_id、rag_tab_id、folder_name。"
                )
            raise ValueError(
                f"於資料夾「{fn}」下找不到支援的音訊檔（副檔名: {', '.join(sorted(_AUDIO_EXTS))}）"
            )
        raw, dec, suf = candidates[0]
        data = z.read(raw)
        if not data:
            raise ValueError("音訊檔為空")
        return data, suf or ".mp3", dec


_YT_URL_RE = re.compile(
    r"(https?://(?:www\.)?(?:youtube\.com/watch\?v=[\w-]{11}|youtu\.be/[\w-]{11})[^\s]*)",
    re.I,
)


def extract_video_id_from_unit_md(text: str) -> str | None:
    """自 Markdown 內容解析 YouTube video_id。"""
    t = (text or "").strip()
    if not t:
        return None
    for chunk in (t, *[ln.strip() for ln in t.splitlines() if ln.strip()]):
        vid = parse_youtube_video_id(chunk)
        if vid:
            return vid
    m = _YT_URL_RE.search(t)
    if m:
        return parse_youtube_video_id(m.group(1))
    return None


def read_youtube_video_id_from_upload_zip(zip_bytes: bytes, folder_name: str) -> tuple[str, str]:
    """
    該資料夾下須**恰好一個** .md；自其內文解析 YouTube。
    回傳 (video_id, 該 .md 在 ZIP 內的解碼路徑)。
    """
    fn = (folder_name or "").strip()
    if not fn:
        raise ValueError("請傳入 folder_name（ZIP 內單元資料夾名稱）")
    if "/" in fn or "\\" in fn:
        raise ValueError("folder_name 不可含路徑分隔字元")

    with zipfile.ZipFile(BytesIO(zip_bytes), "r") as z:
        md_files: list[tuple[str, str]] = []
        for raw, dec in _zip_members(z):
            if not path_has_folder_segment(dec, fn):
                continue
            if not _is_markdown_path(dec):
                continue
            md_files.append((raw, dec))
        md_files.sort(key=lambda x: x[1])
        if not md_files:
            raise ValueError(
                f"於資料夾「{fn}」下找不到 .md／.markdown（請放**一個**含 YouTube 連結或 video_id 的檔案）"
            )
        if len(md_files) > 1:
            names = ", ".join(d for _, d in md_files[:5])
            more = f" 等共 {len(md_files)} 個" if len(md_files) > 5 else ""
            raise ValueError(
                f"於資料夾「{fn}」下僅允許**一個** .md，目前有 {len(md_files)} 個：{names}{more}"
            )
        raw, dec = md_files[0]
        try:
            raw_bytes = z.read(raw)
            text = raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            text = raw_bytes.decode("utf-8", errors="replace")
        vid = extract_video_id_from_unit_md(text)
        if not vid:
            raise ValueError(f"於「{dec}」內找不到有效的 YouTube 連結或 video_id")
        return vid, dec


def read_single_text_md_from_upload_zip(zip_bytes: bytes, folder_name: str) -> tuple[str, str]:
    """
    該資料夾下須**恰好一個** .md；回傳 (檔案全文, ZIP 內解碼路徑)。
    """
    fn = (folder_name or "").strip()
    if not fn:
        raise ValueError("請傳入 folder_name（ZIP 內單元資料夾名稱）")
    if "/" in fn or "\\" in fn:
        raise ValueError("folder_name 不可含路徑分隔字元")

    with zipfile.ZipFile(BytesIO(zip_bytes), "r") as z:
        md_files: list[tuple[str, str]] = []
        for raw, dec in _zip_members(z):
            if not path_has_folder_segment(dec, fn):
                continue
            if not _is_markdown_path(dec):
                continue
            md_files.append((raw, dec))
        md_files.sort(key=lambda x: x[1])
        if not md_files:
            raise ValueError(f"於資料夾「{fn}」下找不到 .md／.markdown（請放**一個** Markdown 檔）")
        if len(md_files) > 1:
            names = ", ".join(d for _, d in md_files[:5])
            more = f" 等共 {len(md_files)} 個" if len(md_files) > 5 else ""
            raise ValueError(
                f"於資料夾「{fn}」下僅允許**一個** .md，目前有 {len(md_files)} 個：{names}{more}"
            )
        raw, dec = md_files[0]
        raw_b = z.read(raw)
        try:
            text = raw_b.decode("utf-8")
        except UnicodeDecodeError:
            text = raw_b.decode("utf-8", errors="replace")
        return (text or "").strip(), dec
