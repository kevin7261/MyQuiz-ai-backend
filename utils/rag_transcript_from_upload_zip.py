"""從 RAG Storage（upload ZIP）依單元資料夾名擷取音訊、YouTube 或文字檔，供 /rag/transcript/* 使用。"""

from __future__ import annotations

import logging
import re
import tempfile
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Iterator

from utils.media_transcript import (
    parse_youtube_video_id,
    transcribe_audio_bytes_deepgram,
    youtube_transcript_plain_text,
)
from utils.zip_storage import UPLOAD_DEFAULT_PERSON, get_zip_path_by_person
from utils.zip_utils import fix_encoding

logger = logging.getLogger(__name__)

_AUDIO_EXTS = frozenset({
    ".mp3", ".wav", ".m4a", ".webm", ".ogg", ".flac", ".opus", ".mp4", ".mpeg", ".mpga", ".aac", ".wma",
})

# 文字單元／YouTube 連結檔：該資料夾下僅允許一個此類檔案
_TRANSCRIPT_TEXT_EXTS = frozenset({".md", ".markdown", ".txt", ".doc", ".docx"})


def _is_transcript_text_path(decoded_path: str) -> bool:
    return Path(decoded_path).suffix.lower() in _TRANSCRIPT_TEXT_EXTS


def _decode_transcript_file_bytes(raw_bytes: bytes, ext: str) -> str:
    """依副檔名將 ZIP 內檔案 bytes 轉成純文字（供 /rag/transcript/text、youtube）。"""
    suf = (ext or "").lower()
    if suf in (".md", ".markdown", ".txt"):
        try:
            return raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            return raw_bytes.decode("utf-8", errors="replace")
    if suf == ".docx":
        import docx2txt

        with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
            try:
                tmp.write(raw_bytes)
                tmp.flush()
                t = docx2txt.process(tmp.name)
            finally:
                Path(tmp.name).unlink(missing_ok=True)
        return (t or "").strip()
    if suf == ".doc":
        from langchain_community.document_loaders import UnstructuredWordDocumentLoader

        with tempfile.NamedTemporaryFile(suffix=".doc", delete=False) as tmp:
            try:
                tmp.write(raw_bytes)
                tmp.flush()
                docs = UnstructuredWordDocumentLoader(tmp.name).load()
            finally:
                Path(tmp.name).unlink(missing_ok=True)
        return "\n\n".join((d.page_content or "").strip() for d in docs).strip()
    raise ValueError(f"不支援的文字檔副檔名: {suf}")


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
            has_text = any(
                path_has_folder_segment(dec, fn) and _is_transcript_text_path(dec)
                for _raw, dec in _zip_members(z)
            )
            if has_text:
                raise ValueError(
                    f"於資料夾「{fn}」下沒有音訊檔，但偵測到文字檔（.md .txt .doc .docx 等）。"
                    "若為 **YouTube 單元**（檔內為影片連結），請改呼叫 **GET /rag/transcript/youtube**，"
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
    """自文字檔內容解析 YouTube video_id（連結或純 id）。"""
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
    該資料夾下須**恰好一個**文字檔（.md .txt .doc .docx 等）；自其內文解析 YouTube（通常檔內僅連結）。
    回傳 (video_id, 該檔在 ZIP 內的解碼路徑)。
    """
    fn = (folder_name or "").strip()
    if not fn:
        raise ValueError("請傳入 folder_name（ZIP 內單元資料夾名稱）")
    if "/" in fn or "\\" in fn:
        raise ValueError("folder_name 不可含路徑分隔字元")

    with zipfile.ZipFile(BytesIO(zip_bytes), "r") as z:
        text_files: list[tuple[str, str]] = []
        for raw, dec in _zip_members(z):
            if not path_has_folder_segment(dec, fn):
                continue
            if not _is_transcript_text_path(dec):
                continue
            text_files.append((raw, dec))
        text_files.sort(key=lambda x: x[1])
        if not text_files:
            raise ValueError(
                f"於資料夾「{fn}」下找不到文字檔（請放**一個** .md／.txt／.doc／.docx，內含 YouTube 連結或 video_id）"
            )
        if len(text_files) > 1:
            names = ", ".join(d for _, d in text_files[:5])
            more = f" 等共 {len(text_files)} 個" if len(text_files) > 5 else ""
            raise ValueError(
                f"於資料夾「{fn}」下僅允許**一個**文字檔，目前有 {len(text_files)} 個：{names}{more}"
            )
        raw, dec = text_files[0]
        raw_bytes = z.read(raw)
        suf = Path(dec).suffix.lower()
        text = _decode_transcript_file_bytes(raw_bytes, suf)
        vid = extract_video_id_from_unit_md(text)
        if not vid:
            raise ValueError(f"於「{dec}」內找不到有效的 YouTube 連結或 video_id")
        return vid, dec


def read_single_transcript_text_from_upload_zip(zip_bytes: bytes, folder_name: str) -> tuple[str, str]:
    """
    該資料夾下須**恰好一個**文字檔（.md .txt .doc .docx 等）；回傳 (正文, ZIP 內解碼路徑)。
    """
    fn = (folder_name or "").strip()
    if not fn:
        raise ValueError("請傳入 folder_name（ZIP 內單元資料夾名稱）")
    if "/" in fn or "\\" in fn:
        raise ValueError("folder_name 不可含路徑分隔字元")

    with zipfile.ZipFile(BytesIO(zip_bytes), "r") as z:
        text_files: list[tuple[str, str]] = []
        for raw, dec in _zip_members(z):
            if not path_has_folder_segment(dec, fn):
                continue
            if not _is_transcript_text_path(dec):
                continue
            text_files.append((raw, dec))
        text_files.sort(key=lambda x: x[1])
        if not text_files:
            raise ValueError(
                f"於資料夾「{fn}」下找不到文字檔（請放**一個** .md／.txt／.doc／.docx）"
            )
        if len(text_files) > 1:
            names = ", ".join(d for _, d in text_files[:5])
            more = f" 等共 {len(text_files)} 個" if len(text_files) > 5 else ""
            raise ValueError(
                f"於資料夾「{fn}」下僅允許**一個**文字檔，目前有 {len(text_files)} 個：{names}{more}"
            )
        raw, dec = text_files[0]
        raw_b = z.read(raw)
        suf = Path(dec).suffix.lower()
        text = _decode_transcript_file_bytes(raw_b, suf)
        return (text or "").strip(), dec


def build_transcript_md_zip_bytes(transcript: str, arcname: str = "transcript.md") -> bytes:
    """將逐字稿包成單檔 ZIP（供 Storage `rag` 區仍使用 .zip 物件鍵）。"""
    body = transcript if transcript is not None else ""
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(arcname.strip() or "transcript.md", body.encode("utf-8"))
    return buf.getvalue()


def _transcript_text_members_in_zip(z: zipfile.ZipFile) -> list[tuple[str, str]]:
    """ZIP 內副檔名屬文字逐字稿來源者（與 _TRANSCRIPT_TEXT_EXTS 一致），供 build-rag-zip 單元 ZIP 解析。"""
    rows: list[tuple[str, str]] = []
    for raw, dec in _zip_members(z):
        if Path(dec).suffix.lower() not in _TRANSCRIPT_TEXT_EXTS:
            continue
        rows.append((raw, dec))
    rows.sort(key=lambda x: x[1])
    return rows


def _audio_members_in_zip(z: zipfile.ZipFile) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    for raw, dec in _zip_members(z):
        suf = Path(dec).suffix.lower()
        if suf not in _AUDIO_EXTS:
            continue
        rows.append((raw, dec, suf))
    rows.sort(key=lambda x: x[1])
    return rows


def extract_transcript_for_rag_build(zip_bytes: bytes, unit_type: int) -> dict[str, str]:
    """
    自 build-rag-zip 產出之「單元小 ZIP」擷取逐字稿與 Rag_Unit 附帶欄位。
    回傳 transcript、text_file_name、mp3_file_name、youtube_url（非適用之欄位為空字串）。
    unit_type：2=文字（**恰好一個** .md/.txt/.doc/.docx；**text_file_name** 為該檔 basename）、
    3=音訊（第一個支援副檔名＋Deepgram）、4=YouTube（**恰好一個**上述文字檔含連結＋en 字幕）。
    """
    out: dict[str, str] = {
        "transcript": "",
        "text_file_name": "",
        "mp3_file_name": "",
        "youtube_url": "",
    }
    if unit_type not in (2, 3, 4):
        raise ValueError(f"extract_transcript_for_rag_build 僅支援 unit_type 2/3/4，收到 {unit_type}")

    with zipfile.ZipFile(BytesIO(zip_bytes), "r") as z:
        if unit_type == 2:
            text_rows = _transcript_text_members_in_zip(z)
            if not text_rows:
                raise ValueError(
                    "單元 ZIP 內找不到文字檔（unit_type=2 須恰好一個 .md／.txt／.doc／.docx）"
                )
            if len(text_rows) > 1:
                raise ValueError(
                    "unit_type=2 須恰好一個文字檔，目前有 "
                    f"{len(text_rows)} 個："
                    + ", ".join(d for _, d in text_rows[:5])
                )
            raw, dec = text_rows[0]
            raw_b = z.read(raw)
            suf = Path(dec).suffix.lower()
            text = _decode_transcript_file_bytes(raw_b, suf)
            text = (text or "").strip()
            if not text:
                raise ValueError(f"{Path(dec).name} 內容為空")
            out["transcript"] = text
            out["text_file_name"] = Path(dec).name
            return out

        if unit_type == 3:
            aud = _audio_members_in_zip(z)
            if not aud:
                raise ValueError(
                    "單元 ZIP 內找不到音訊檔（unit_type=3；支援副檔名: "
                    + ", ".join(sorted(_AUDIO_EXTS))
                    + "）"
                )
            raw, dec, suf = aud[0]
            data = z.read(raw)
            if not data:
                raise ValueError("音訊檔為空")
            try:
                transcript, _elapsed = transcribe_audio_bytes_deepgram(data, suffix=suf or ".mp3")
            except RuntimeError as e:
                raise ValueError(str(e)) from e
            except ValueError as e:
                raise ValueError(str(e)) from e
            transcript = (transcript or "").strip()
            if not transcript:
                raise ValueError("Deepgram 逐字稿為空（unit_type=3）")
            out["transcript"] = transcript
            out["mp3_file_name"] = Path(dec).name
            return out

        # unit_type == 4
        text_rows = _transcript_text_members_in_zip(z)
        if not text_rows:
            raise ValueError(
                "單元 ZIP 內找不到文字檔（unit_type=4 須恰好一個 .md／.txt／.doc／.docx，內含 YouTube 連結）"
            )
        if len(text_rows) > 1:
            raise ValueError(
                "unit_type=4 須恰好一個文字檔，目前有 "
                f"{len(text_rows)} 個："
                + ", ".join(d for _, d in text_rows[:5])
            )
        raw, dec = text_rows[0]
        raw_b = z.read(raw)
        suf = Path(dec).suffix.lower()
        raw_text = _decode_transcript_file_bytes(raw_b, suf)
        vid = extract_video_id_from_unit_md(raw_text)
        if not vid:
            raise ValueError(f"{Path(dec).name} 內找不到有效的 YouTube 連結或 video_id")
        try:
            cap, _elapsed = youtube_transcript_plain_text(vid, languages=["en"])
        except Exception as e:
            raise ValueError(f"擷取 YouTube 英文字幕失敗: {e!s}") from e
        cap = (cap or "").strip()
        if not cap:
            raise ValueError("YouTube 英文字幕為空（請確認影片有 en 字幕）")
        out["transcript"] = cap
        out["youtube_url"] = f"https://www.youtube.com/watch?v={vid}"
        return out
