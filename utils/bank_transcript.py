"""Bank 專屬：自 Bank Storage（upload ZIP）依單元資料夾名擷取音訊／YouTube／文字檔（自 utils.rag_transcript 複製，與 rag 無關）。"""

from __future__ import annotations

import logging
import re
import tempfile
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Iterator

from utils.media import parse_youtube_video_id
from utils.bank_storage import UPLOAD_DEFAULT_PERSON, get_zip_path, get_zip_path_by_person
from utils.bank_zip_utils import fix_encoding

logger = logging.getLogger(__name__)

_AUDIO_EXTS = frozenset({
    ".mp3", ".wav", ".m4a", ".webm", ".ogg", ".flac", ".opus", ".mp4", ".mpeg", ".mpga", ".aac", ".wma",
})

_TRANSCRIPT_TEXT_EXTS = frozenset({".md", ".markdown", ".txt", ".doc", ".docx"})


def _is_transcript_text_path(decoded_path: str) -> bool:
    return Path(decoded_path).suffix.lower() in _TRANSCRIPT_TEXT_EXTS


def _decode_transcript_file_bytes(raw_bytes: bytes, ext: str) -> str:
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
    needle = (folder_name or "").strip()
    if not needle or "\\" in needle:
        return False
    parts = decoded_path.replace("\\", "/").strip("/").split("/")
    if "/t" not in needle:
        return needle in parts
    segs = [s.strip() for s in needle.split("/t") if s.strip()]
    return any(s in parts for s in segs)


def read_upload_zip_bytes(person_id: str, bank_page_id: str) -> bytes:
    """自 Bank bucket 的 upload 區下載該 tab 的 ZIP，讀成 bytes（暫存檔會刪除）。"""
    pid = (person_id or "").strip() or UPLOAD_DEFAULT_PERSON
    rid = (bank_page_id or "").strip()
    if not rid or "/" in rid or "\\" in rid:
        raise ValueError("無效的 bank_page_id")
    tmp = get_zip_path_by_person(pid, rid)
    if not tmp or not tmp.exists():
        raise FileNotFoundError(f"找不到 upload ZIP（person_id={pid}, bank_page_id={rid}）")
    try:
        return tmp.read_bytes()
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            logger.debug("刪除暫存 upload ZIP 失敗", exc_info=True)


def read_repack_zip_bytes(repack_file_name: str) -> bytes:
    """自 Bank bucket 的 repack 區下載該單元 ZIP（對齊 Bank_Unit.repack_file_name，例如 `stem.zip`）。"""
    raw = (repack_file_name or "").strip()
    stem = Path(raw).stem.strip() if raw else ""
    if not stem or "/" in stem or "\\" in stem:
        raise ValueError("無效的 repack_file_name（無法取得 repack tab id）")
    tmp = get_zip_path(stem)
    if not tmp or not tmp.exists():
        raise FileNotFoundError(f"找不到 repack ZIP（repack_file_name={raw!r}, page_id={stem!r}）")
    try:
        return tmp.read_bytes()
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            logger.debug("刪除暫存 repack ZIP 失敗", exc_info=True)


def _first_level_folder_names_in_zip(zip_bytes: bytes) -> list[str]:
    order: list[str] = []
    seen: set[str] = set()
    with zipfile.ZipFile(BytesIO(zip_bytes), "r") as z:
        for _raw, dec in _zip_members(z):
            parts = dec.replace("\\", "/").strip("/").split("/")
            if len(parts) < 2:
                continue
            top = parts[0].strip()
            if not top or top in seen:
                continue
            seen.add(top)
            order.append(top)
    return order


def pick_audio_from_upload_zip_with_folder_fallback(
    zip_bytes: bytes,
    preferred_folder: str,
    *,
    allow_scan_other_top_folders: bool,
) -> tuple[bytes, str, str]:
    pref = (preferred_folder or "").strip()
    last_err: str | None = None
    if pref and "/" not in pref and "\\" not in pref:
        try:
            return pick_audio_from_upload_zip(zip_bytes, pref)
        except ValueError as e:
            last_err = str(e)
    if not allow_scan_other_top_folders:
        if last_err:
            raise ValueError(last_err) from None
        raise ValueError("請傳入有效的 folder_name（ZIP 內單元資料夾名）")
    tried: set[str] = set()
    if pref:
        tried.add(pref)
    for fn in _first_level_folder_names_in_zip(zip_bytes):
        if fn in tried:
            continue
        tried.add(fn)
        try:
            return pick_audio_from_upload_zip(zip_bytes, fn)
        except ValueError as e:
            last_err = str(e)
    msg = last_err or "於 ZIP 內找不到支援的音訊檔"
    raise ValueError(msg)


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
        if len(candidates) > 1:
            names = ", ".join(d for _, d, _ in candidates[:5])
            more = f" 等共 {len(candidates)} 個" if len(candidates) > 5 else ""
            raise ValueError(
                f"於資料夾「{fn}」下僅允許**一個**音訊檔，目前有 {len(candidates)} 個：{names}{more}"
            )
        if not candidates:
            has_text = any(
                path_has_folder_segment(dec, fn) and _is_transcript_text_path(dec)
                for _raw, dec in _zip_members(z)
            )
            if has_text:
                raise ValueError(
                    f"於資料夾「{fn}」下沒有音訊檔，但偵測到文字檔（.md .txt .doc .docx 等）。"
                    "若為 **YouTube 單元**（檔內為影片連結），請改用該單元之 youtube-url 端點。"
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


def _extract_youtube_video_id_url_only(text: str) -> str | None:
    t = (text or "").strip()
    if not t:
        return None
    for line in [t, *[ln.strip() for ln in t.splitlines() if ln.strip()]]:
        v = line.strip()
        if not v or re.fullmatch(r"[\w-]{11}", v):
            continue
        vid = parse_youtube_video_id(v)
        if vid:
            return vid
    m = _YT_URL_RE.search(t)
    if m:
        return parse_youtube_video_id(m.group(1))
    return None


def read_youtube_video_id_from_upload_zip(zip_bytes: bytes, folder_name: str) -> tuple[str, str]:
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
                f"於資料夾「{fn}」下找不到文字檔（請放一個 .md／.txt／.doc／.docx，第一行為 YouTube URL）"
            )
        raw, dec = text_files[0]
        raw_bytes = z.read(raw)
        content = _decode_transcript_file_bytes(raw_bytes, Path(dec).suffix.lower()) or ""
        first_line = (content.splitlines() or [""])[0].strip()
        vid = _extract_youtube_video_id_url_only(first_line)
        if not vid:
            raise ValueError(f"於「{dec}」第一行找不到有效的 YouTube URL")
        return vid, dec


def read_supplementary_text_from_youtube_unit(zip_bytes: bytes, folder_name: str) -> tuple[str, str]:
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
            raise ValueError(f"於資料夾「{fn}」下找不到文字檔")
        raw, dec = text_files[0]
        raw_b = z.read(raw)
        content = _decode_transcript_file_bytes(raw_b, Path(dec).suffix.lower()) or ""
        lines = content.splitlines()
        transcript = "\n".join(lines[1:]).strip()
        return transcript, dec


def read_mp3_unit_transcript_from_upload_zip(zip_bytes: bytes, folder_name: str) -> tuple[str, str]:
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
                f"於資料夾「{fn}」下找不到逐字稿文字檔"
                "（unit_type=3 須一個音訊檔與一個 .md／.txt／.doc／.docx）"
            )
        if len(text_files) > 1:
            names = ", ".join(d for _, d in text_files[:5])
            more = f" 等共 {len(text_files)} 個" if len(text_files) > 5 else ""
            raise ValueError(
                f"於資料夾「{fn}」下僅允許**一個**文字檔，目前有 {len(text_files)} 個：{names}{more}"
            )
        raw, dec = text_files[0]
        raw_b = z.read(raw)
        text = _decode_transcript_file_bytes(raw_b, Path(dec).suffix.lower()) or ""
        if not text.strip():
            raise ValueError(f"{Path(dec).name} 內容為空（unit_type=3 逐字稿不可為空）")
        return text, Path(dec).name


def read_single_transcript_text_from_upload_zip(zip_bytes: bytes, folder_name: str) -> tuple[str, str]:
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
    """將逐字稿包成單檔 ZIP（供 Storage rag 區仍使用 .zip 物件鍵）。"""
    body = transcript if transcript is not None else ""
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(arcname.strip() or "transcript.md", body.encode("utf-8"))
    return buf.getvalue()


def _transcript_text_members_in_zip(z: zipfile.ZipFile) -> list[tuple[str, str]]:
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


def infer_unit_type_when_unspecified(declared_unit_type: int, zip_bytes: bytes) -> int:
    """build-zip 未帶 unit_types 時推斷單元類型（僅 0 時推斷；2／3／4 不覆寫；不自動推 4）。"""
    try:
        d = int(declared_unit_type)
    except (TypeError, ValueError):
        d = 0
    if d != 0:
        return d
    try:
        with zipfile.ZipFile(BytesIO(zip_bytes), "r") as z:
            aud = _audio_members_in_zip(z)
            text_rows = _transcript_text_members_in_zip(z)
            if len(aud) == 1 and len(text_rows) == 1:
                raw, dec = text_rows[0]
                raw_b = z.read(raw)
                suf = Path(dec).suffix.lower()
                text = _decode_transcript_file_bytes(raw_b, suf) or ""
                if text.strip():
                    return 3
            if aud:
                return 0
            if len(text_rows) != 1:
                return 0
            raw, dec = text_rows[0]
            raw_b = z.read(raw)
            suf = Path(dec).suffix.lower()
            text = _decode_transcript_file_bytes(raw_b, suf) or ""
            if not text.strip():
                return 0
            return 2
    except (zipfile.BadZipFile, OSError, ValueError):
        logger.debug("infer_unit_type_when_unspecified: 無法解析 ZIP", exc_info=True)
    return 0


def _extract_transcript_type2(z: zipfile.ZipFile, out: dict[str, str]) -> dict[str, str]:
    text_rows = _transcript_text_members_in_zip(z)
    if not text_rows:
        raise ValueError("單元 ZIP 內找不到文字檔（unit_type=2 須恰好一個 .md／.txt／.doc／.docx）")
    if len(text_rows) > 1:
        raise ValueError(
            "unit_type=2 須恰好一個文字檔，目前有 " f"{len(text_rows)} 個：" + ", ".join(d for _, d in text_rows[:5])
        )
    raw, dec = text_rows[0]
    raw_b = z.read(raw)
    text = _decode_transcript_file_bytes(raw_b, Path(dec).suffix.lower()) or ""
    if not text.strip():
        raise ValueError(f"{Path(dec).name} 內容為空")
    out["transcript"] = text
    out["text_file_name"] = Path(dec).name
    return out


def _extract_transcript_type3(z: zipfile.ZipFile, out: dict[str, str]) -> dict[str, str]:
    aud = _audio_members_in_zip(z)
    if not aud:
        raise ValueError("單元 ZIP 內找不到音訊檔（unit_type=3；支援副檔名: " + ", ".join(sorted(_AUDIO_EXTS)) + "）")
    if len(aud) > 1:
        raise ValueError("unit_type=3 須恰好一個音訊檔，目前有 " f"{len(aud)} 個：" + ", ".join(d for _, d, _ in aud[:5]))
    text_rows = _transcript_text_members_in_zip(z)
    if not text_rows:
        raise ValueError("unit_type=3 須附一個逐字稿文字檔（.md／.txt／.doc／.docx）")
    if len(text_rows) > 1:
        raise ValueError("unit_type=3 僅允許一個文字檔，目前有 " f"{len(text_rows)} 個：" + ", ".join(d for _, d in text_rows[:5]))
    _, dec, _ = aud[0]
    out["mp3_file_name"] = Path(dec).name
    raw, text_dec = text_rows[0]
    raw_b = z.read(raw)
    text = _decode_transcript_file_bytes(raw_b, Path(text_dec).suffix.lower()) or ""
    if not text.strip():
        raise ValueError(f"{Path(text_dec).name} 內容為空（unit_type=3 逐字稿不可為空）")
    out["transcript"] = text
    out["text_file_name"] = Path(text_dec).name
    return out


def _extract_transcript_type4(z: zipfile.ZipFile, out: dict[str, str]) -> dict[str, str]:
    if _audio_members_in_zip(z):
        raise ValueError("unit_type=4 不可包含音訊檔（僅允許文字檔）")
    text_rows = _transcript_text_members_in_zip(z)
    if not text_rows:
        raise ValueError("單元 ZIP 內找不到文字檔（unit_type=4 須恰好一個 .md／.txt／.doc／.docx）")
    if len(text_rows) != 1:
        raise ValueError(f"unit_type=4 須恰好一個文字檔，目前有 {len(text_rows)} 個：" + ", ".join(d for _, d in text_rows[:5]))
    raw, dec = text_rows[0]
    raw_b = z.read(raw)
    content = _decode_transcript_file_bytes(raw_b, Path(dec).suffix.lower()) or ""
    lines = content.splitlines()
    if not lines:
        raise ValueError(f"{Path(dec).name} 內容為空")
    youtube_vid = _extract_youtube_video_id_url_only(lines[0].strip())
    if not youtube_vid:
        raise ValueError(
            f"{Path(dec).name} 第一行須為有效的 YouTube URL（不接受裸 video_id），例：https://www.youtube.com/watch?v=..."
        )
    out["youtube_url"] = f"https://www.youtube.com/watch?v={youtube_vid}"
    out["transcript"] = "\n".join(lines[1:]).strip()
    out["text_file_name"] = Path(dec).name
    return out


def extract_transcript_for_rag_build(zip_bytes: bytes, unit_type: int) -> dict[str, str]:
    """自 build-zip 單元小 ZIP 擷取逐字稿與 Bank_Unit 附帶欄位（unit_type 2／3／4）。"""
    out: dict[str, str] = {"transcript": "", "text_file_name": "", "mp3_file_name": "", "youtube_url": ""}
    if unit_type not in (2, 3, 4):
        raise ValueError(f"extract_transcript_for_rag_build 僅支援 unit_type 2/3/4，收到 {unit_type}")
    with zipfile.ZipFile(BytesIO(zip_bytes), "r") as z:
        if unit_type == 2:
            return _extract_transcript_type2(z, out)
        if unit_type == 3:
            return _extract_transcript_type3(z, out)
        return _extract_transcript_type4(z, out)
