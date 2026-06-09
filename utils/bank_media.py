"""Bank 專屬媒體小工具（自 utils.media 複製，與 rag 無關）：音訊 MIME 類型、YouTube 影片 ID 解析。"""

from __future__ import annotations

import re
from urllib.parse import parse_qs, urlparse


def _suffix_to_content_type(suffix: str) -> str:
    s = suffix.lower() if suffix.startswith(".") else f".{suffix.lower()}"
    return {
        ".mp3": "audio/mpeg",
        ".wav": "audio/wav",
        ".m4a": "audio/mp4",
        ".webm": "audio/webm",
        ".ogg": "audio/ogg",
        ".flac": "audio/flac",
        ".opus": "audio/opus",
        ".mp4": "audio/mp4",
        ".mpeg": "audio/mpeg",
        ".mpga": "audio/mpeg",
        ".aac": "audio/aac",
        ".wma": "audio/x-ms-wma",
    }.get(s, "application/octet-stream")


def audio_media_type_for_suffix(suffix: str) -> str:
    """HTTP ``Content-Type`` for an audio filename suffix (e.g. ``.mp3`` → ``audio/mpeg``)."""
    return _suffix_to_content_type(suffix)


def parse_youtube_video_id(raw: str) -> str | None:
    """接受 11 字元 video_id 或常見 YouTube 網址，回傳 video_id；無法辨識則 None。"""
    v = (raw or "").strip()
    if not v:
        return None
    if re.fullmatch(r"[\w-]{11}", v):
        return v
    if "youtu.be/" in v:
        part = v.split("youtu.be/", 1)[1].split("?")[0].split("/")[0]
        return part if re.fullmatch(r"[\w-]{11}", part) else None
    parsed = urlparse(v)
    host = (parsed.netloc or "").lower()
    if "youtube.com" in host or "youtube-nocookie.com" in host:
        qs = parse_qs(parsed.query)
        if "v" in qs and qs["v"]:
            vid = (qs["v"][0] or "").strip()
            if re.fullmatch(r"[\w-]{11}", vid):
                return vid
        path_parts = [p for p in (parsed.path or "").split("/") if p]
        if len(path_parts) >= 2 and path_parts[0].lower() == "shorts":
            vid = path_parts[1].strip()
            return vid if re.fullmatch(r"[\w-]{11}", vid) else None
    return None
