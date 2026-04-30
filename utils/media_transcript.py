"""音訊 Deepgram 逐字稿、YouTube 字幕擷取（供 RAG transcript／評分模組使用）。"""

from __future__ import annotations

import logging
import os
import re
import time
from typing import Any
from urllib.parse import parse_qs, urlparse

import requests

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


def audio_media_type_for_suffix(suffix: str) -> str:
    """HTTP ``Content-Type`` for an audio filename suffix (e.g. ``.mp3`` → ``audio/mpeg``)."""
    return _suffix_to_content_type(suffix)


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
    API Key 優先序：參數 api_key → 環境變數 DEEPGRAM_API_KEY。
    模型預設 nova-2，可覆寫 DEEPGRAM_MODEL。
    """
    k = (api_key or "").strip() or (os.environ.get("DEEPGRAM_API_KEY") or "").strip()
    if not k:
        raise RuntimeError(
            "未設定 Deepgram API Key：請設定環境變數 DEEPGRAM_API_KEY（例如 .env 或部署平台 Environment）。"
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


def _youtube_transcript_api_from_env():
    """
    建立 YouTubeTranscriptApi。若設定代理相關環境變數，可避免雲端 IP 被 YouTube 封鎖。

    Webshare 住宅代理（與套件 README 一致）：
      YOUTUBE_TRANSCRIPT_WEBSHARE_USERNAME、YOUTUBE_TRANSCRIPT_WEBSHARE_PASSWORD
      選填 YOUTUBE_TRANSCRIPT_WEBSHARE_LOCATIONS：逗號分隔國碼，例如 us,tw

    自訂 HTTP/HTTPS 代理：
      YOUTUBE_TRANSCRIPT_HTTP_PROXY、YOUTUBE_TRANSCRIPT_HTTPS_PROXY
      （值格式如 http://user:pass@host:port；可只設其一）

    若同時設定 Webshare 帳密與 HTTP 代理，優先使用 Webshare。
    """
    from youtube_transcript_api import YouTubeTranscriptApi

    ws_user = (os.environ.get("YOUTUBE_TRANSCRIPT_WEBSHARE_USERNAME") or "").strip()
    ws_pass = (os.environ.get("YOUTUBE_TRANSCRIPT_WEBSHARE_PASSWORD") or "").strip()
    if ws_user and ws_pass:
        try:
            from youtube_transcript_api.proxies import WebshareProxyConfig
        except ImportError as e:
            raise RuntimeError(
                "已設定 YOUTUBE_TRANSCRIPT_WEBSHARE_*，但目前的 youtube-transcript-api 不支援 "
                "WebshareProxyConfig；請升級至 README 建議版本（含 proxies 模組）。"
            ) from e
        loc_raw = (os.environ.get("YOUTUBE_TRANSCRIPT_WEBSHARE_LOCATIONS") or "").strip()
        locations = [x.strip().lower() for x in loc_raw.split(",") if x.strip()] or None
        cfg = WebshareProxyConfig(
            proxy_username=ws_user,
            proxy_password=ws_pass,
            filter_ip_locations=locations,
        )
        logger.info("YouTube transcript: WebshareProxyConfig（住宅代理）已啟用")
        return YouTubeTranscriptApi(proxy_config=cfg)

    http_u = (os.environ.get("YOUTUBE_TRANSCRIPT_HTTP_PROXY") or "").strip()
    https_u = (os.environ.get("YOUTUBE_TRANSCRIPT_HTTPS_PROXY") or "").strip()
    if http_u or https_u:
        try:
            from youtube_transcript_api.proxies import GenericProxyConfig
        except ImportError as e:
            raise RuntimeError(
                "已設定 YOUTUBE_TRANSCRIPT_HTTP(S)_PROXY，但目前的 youtube-transcript-api 不支援 "
                "GenericProxyConfig；請升級套件。"
            ) from e
        cfg = GenericProxyConfig(
            http_url=http_u or None,
            https_url=https_u or None,
        )
        logger.info("YouTube transcript: GenericProxyConfig 已啟用")
        return YouTubeTranscriptApi(proxy_config=cfg)

    return YouTubeTranscriptApi()


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
    api = _youtube_transcript_api_from_env()
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
