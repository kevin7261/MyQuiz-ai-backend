"""音訊 Deepgram 逐字稿、YouTube 字幕擷取（供 RAG transcript／評分模組使用）。"""

from __future__ import annotations

import logging
import os
import re
import time
from typing import Any
from urllib.parse import parse_qs, urlparse

import requests

from utils.llm_api_key_utils import get_deepgram_api_key

logger = logging.getLogger(__name__)

_DEEPGRAM_LISTEN_URL = "https://api.deepgram.com/v1/listen"


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


def transcribe_audio_bytes_deepgram(
    data: bytes,
    *,
    suffix: str,
    model: str | None = None,
    api_key: str | None = None,
) -> tuple[str, float]:
    """
    以 Deepgram 預錄 API 轉錄音訊 bytes。
    回傳 (全文逐字稿, 耗時秒數)。
    API Key 優先序：參數 api_key → 環境變數 DEEPGRAM_API_KEY → System_Setting key=deepgram_api_key。
    模型預設 nova-2，可覆寫 DEEPGRAM_MODEL。
    """
    k = (api_key or "").strip() or (os.environ.get("DEEPGRAM_API_KEY") or "").strip()
    if not k:
        k = (get_deepgram_api_key() or "").strip()
    if not k:
        raise RuntimeError(
            "未設定 Deepgram API Key：請在 System_Setting 新增 key=deepgram_api_key 的 value，"
            "或設定環境變數 DEEPGRAM_API_KEY。"
        )
    m = (model or os.environ.get("DEEPGRAM_MODEL") or "nova-2").strip()
    if not re.match(r"^[\w.-]+$", m):
        raise ValueError(f"不支援的 Deepgram model 名稱: {m!r}")

    ct = _suffix_to_content_type(suffix if suffix else ".mp3")
    t0 = time.perf_counter()
    try:
        resp = requests.post(
            _DEEPGRAM_LISTEN_URL,
            params={"model": m},
            headers={
                "Authorization": f"Token {k}",
                "Content-Type": ct,
            },
            data=data,
            timeout=int(os.environ.get("DEEPGRAM_REQUEST_TIMEOUT_SECONDS", "900")),
        )
    except requests.RequestException as e:
        raise RuntimeError(f"Deepgram 連線失敗: {e!s}") from e

    elapsed = time.perf_counter() - t0
    if resp.status_code >= 400:
        body = (resp.text or "")[:1200]
        raise RuntimeError(f"Deepgram API 回應 {resp.status_code}: {body}")

    try:
        payload: dict[str, Any] = resp.json()
    except ValueError as e:
        raise RuntimeError("Deepgram 回應非 JSON") from e

    try:
        ch0 = payload["results"]["channels"][0]["alternatives"][0]
        text = (ch0.get("transcript") or "").strip()
    except (KeyError, IndexError, TypeError):
        logger.warning("Deepgram JSON 結構異常，略過 channels: keys=%s", list(payload.keys()))
        text = ""

    return text, elapsed


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


def youtube_transcript_plain_text(video_id: str, languages: list[str] | None) -> tuple[str, float]:
    """
    擷取 YouTube 字幕並併成單一純文字（與 Colab 範例一致：以空白連接各段）。
    回傳 (全文, 耗時秒數)。
    """
    from youtube_transcript_api import YouTubeTranscriptApi

    langs = languages if languages else ["en"]
    t0 = time.perf_counter()
    api = YouTubeTranscriptApi()
    if hasattr(api, "fetch"):
        fetched = api.fetch(video_id, languages=langs)
        if hasattr(fetched, "to_raw_data"):
            transcript = fetched.to_raw_data()
        else:
            transcript = [{"text": getattr(s, "text", "")} for s in fetched]
    else:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=langs)

    texts: list[str] = []
    for entry in transcript:
        if isinstance(entry, dict):
            text_content = entry.get("text", "")
        else:
            text_content = getattr(entry, "text", "")
        texts.append(str(text_content).replace("\n", " "))
    full_text = " ".join(texts).strip()
    elapsed = time.perf_counter() - t0
    return full_text, elapsed
