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


def _env_generic_proxy_config_for_youtube(
    http_url: str | None,
    https_url: str | None,
):
    """
    套件內建 GenericProxyConfig 的 retries_when_blocked 恒為 0，遇 RequestBlocked 不會重試。
    輪換出口 IP 的代理需「關連線 + 重試」才有機會換到未被封的 IP。
    """
    from youtube_transcript_api.proxies import GenericProxyConfig

    class _EnvGenericProxy(GenericProxyConfig):
        @property
        def retries_when_blocked(self) -> int:
            raw = (os.environ.get("YOUTUBE_TRANSCRIPT_PROXY_RETRIES_WHEN_BLOCKED") or "").strip()
            if not raw:
                return 8
            try:
                return max(0, min(30, int(raw)))
            except ValueError:
                return 8

        @property
        def prevent_keeping_connections_alive(self) -> bool:
            v = (os.environ.get("YOUTUBE_TRANSCRIPT_PROXY_CLOSE_CONNECTION") or "1").strip().lower()
            return v not in ("0", "false", "no", "off")

    return _EnvGenericProxy(http_url=http_url, https_url=https_url)


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


def _deepgram_start_seconds_to_tag(start: Any) -> str:
    """將 Deepgram utterance/word 的 start（秒）格式化成 ``[MM:SS]`` 或 ``[H:MM:SS]``。"""
    try:
        sec = float(start)
    except (TypeError, ValueError):
        sec = 0.0
    if sec < 0:
        sec = 0.0
    h = int(sec // 3600)
    rem = sec % 3600
    m = int(rem // 60)
    s = int(rem % 60)
    if h > 0:
        return f"[{h}:{m:02d}:{s:02d}]"
    return f"[{m:02d}:{s:02d}]"


def resolve_timestamp_merge_seconds(explicit: float | None) -> float:
    """
    時間標記合併間隔（秒）。
    ``explicit`` 有傳（含 ``0``）時優先；否則讀環境變數 ``TRANSCRIPT_TIMESTAMP_MERGE_SECONDS``；
    皆無效時預設 ``10``（約每 10 秒一個標記；片段過長會於時間窗前切段）。
    ``0`` 表示不合併（維持字幕／utterance 原始粒度）。
    """
    if explicit is not None:
        try:
            return max(0.0, float(explicit))
        except (TypeError, ValueError):
            return 10.0
    raw = (os.environ.get("TRANSCRIPT_TIMESTAMP_MERGE_SECONDS") or "").strip()
    if raw:
        try:
            return max(0.0, float(raw))
        except ValueError:
            pass
    return 10.0


def merge_transcript_segments_by_interval(
    segments: list[tuple[float, str]],
    interval_sec: float,
) -> list[tuple[float, str]]:
    """依時間將連續片段合併為較長區塊；``interval_sec <= 0`` 時不改變順序與粒度。"""
    cleaned: list[tuple[float, str]] = []
    for start_any, text_any in segments:
        try:
            start_f = float(start_any)
        except (TypeError, ValueError):
            start_f = 0.0
        piece = (text_any or "").strip()
        if not piece:
            continue
        cleaned.append((start_f, piece))
    cleaned.sort(key=lambda x: x[0])
    if interval_sec <= 0 or len(cleaned) <= 1:
        return cleaned

    merged: list[tuple[float, str]] = []
    anchor: float | None = None
    buf: list[str] = []

    def flush() -> None:
        nonlocal anchor, buf
        if anchor is None or not buf:
            anchor = None
            buf = []
            return
        merged.append((anchor, " ".join(buf)))
        anchor = None
        buf = []

    for start_f, piece in cleaned:
        if anchor is None:
            anchor = start_f
            buf = [piece]
            continue
        if start_f >= anchor + interval_sec and buf:
            flush()
            anchor = start_f
            buf = [piece]
        else:
            buf.append(piece)
    flush()
    return merged


def format_timestamp_segments(segments: list[tuple[float, str]]) -> str:
    """將 ``(秒起點, 正文)`` 列格式化為多行 ``[MM:SS] text``。"""
    lines = [f"{_deepgram_start_seconds_to_tag(s)} {t}" for s, t in segments]
    return "\n".join(lines).strip()


def _deepgram_segments_from_utterances(payload: dict[str, Any]) -> list[tuple[float, str]] | None:
    raw = payload.get("results")
    if not isinstance(raw, dict):
        return None
    utterances = raw.get("utterances")
    if not isinstance(utterances, list) or not utterances:
        return None
    out: list[tuple[float, str]] = []
    for u in utterances:
        if not isinstance(u, dict):
            continue
        chunk = (u.get("transcript") or "").strip()
        if not chunk:
            words = u.get("words")
            if isinstance(words, list):
                chunk = _join_deepgram_word_tokens(words).strip()
        if not chunk:
            continue
        try:
            st = float(u.get("start", 0) or 0)
        except (TypeError, ValueError):
            st = 0.0
        out.append((st, chunk))
    return out if out else None


def _join_deepgram_word_tokens(words: list[Any]) -> str:
    parts: list[str] = []
    for w in words:
        if not isinstance(w, dict):
            continue
        tok = (w.get("punctuated_word") or w.get("word") or "").strip()
        if tok:
            parts.append(tok)
    return " ".join(parts)


def _deepgram_segments_from_paragraph_sentences(ch0: dict[str, Any]) -> list[tuple[float, str]] | None:
    """備援：paragraphs=true 時 alternatives 內段落／句子的 start／text。"""
    paras = ch0.get("paragraphs")
    if not isinstance(paras, dict):
        return None
    blocks = paras.get("paragraphs")
    if not isinstance(blocks, list) or not blocks:
        return None
    out: list[tuple[float, str]] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        sents = block.get("sentences")
        if isinstance(sents, list):
            for sent in sents:
                if not isinstance(sent, dict):
                    continue
                t = (sent.get("text") or "").strip()
                if not t:
                    continue
                try:
                    st = float(sent.get("start", 0) or 0)
                except (TypeError, ValueError):
                    st = 0.0
                out.append((st, t))
        else:
            t = (block.get("text") or "").strip()
            if not t:
                continue
            try:
                st = float(block.get("start", 0) or 0)
            except (TypeError, ValueError):
                st = 0.0
            out.append((st, t))
    return out if out else None


def _deepgram_segments_from_words_by_pause(ch0: dict[str, Any], *, gap_sec: float = 0.85) -> list[tuple[float, str]] | None:
    """備援：依字級時間與停頓切段（gap_sec 與 Deepgram utt_split 預設相近）。"""
    words = ch0.get("words")
    if not isinstance(words, list) or not words:
        return None
    segments: list[tuple[float, str]] = []
    buf: list[str] = []
    line_start: float | None = None
    prev_end: float | None = None

    def flush() -> None:
        nonlocal buf, line_start
        if not buf:
            return
        st = line_start if line_start is not None else 0.0
        segments.append((st, " ".join(buf)))
        buf = []
        line_start = None

    for w in words:
        if not isinstance(w, dict):
            continue
        tok = (w.get("punctuated_word") or w.get("word") or "").strip()
        if not tok:
            continue
        start_w = w.get("start")
        try:
            start_f = float(start_w) if start_w is not None else None
        except (TypeError, ValueError):
            start_f = None
        end_w = w.get("end")
        try:
            end_f = float(end_w) if end_w is not None else start_f
        except (TypeError, ValueError):
            end_f = start_f

        if line_start is None and start_f is not None:
            line_start = start_f
        if prev_end is not None and start_f is not None and (start_f - prev_end) >= gap_sec and buf:
            flush()
            line_start = start_f

        buf.append(tok)
        prev_end = end_f if end_f is not None else prev_end

    flush()
    return segments if segments else None


def transcribe_audio_bytes_deepgram(
    data: bytes,
    *,
    suffix: str,
    model: str | None = None,
    api_key: str | None = None,
    with_timestamps: bool = False,
    timestamp_merge_seconds: float | None = None,
) -> tuple[str, float]:
    """
    以 Deepgram 預錄 API 轉錄音訊 bytes。
    回傳 (全文逐字稿, 耗時秒數)。
    ``with_timestamps=True`` 時會啟用 utterances／punctuate／paragraphs，並優先依 utterances 輸出
    含時間之段落；再以 :func:`resolve_timestamp_merge_seconds` 依時間窗合併（預設約每 10 秒一標）。
    ``timestamp_merge_seconds=0`` 表示不按時間窗合併（維持 utterance／句級粒度；即「一段一標」）。
    無法分段時依段落句子、字級停頓備援；皆不可用時於全文前加上 ``[00:00]``。
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
    params: dict[str, str] = {"model": m}
    if with_timestamps:
        params["utterances"] = "true"
        params["punctuate"] = "true"
        params["paragraphs"] = "true"
    t0 = time.perf_counter()
    try:
        resp = requests.post(
            _DEEPGRAM_LISTEN_URL,
            params=params,
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
        ch0 = {}

    if with_timestamps and text:
        merge_sec = resolve_timestamp_merge_seconds(timestamp_merge_seconds)
        segments: list[tuple[float, str]] | None = None
        segments = _deepgram_segments_from_utterances(payload)
        if not segments and isinstance(ch0, dict):
            segments = _deepgram_segments_from_paragraph_sentences(ch0)
        if not segments and isinstance(ch0, dict):
            segments = _deepgram_segments_from_words_by_pause(ch0)
        if segments:
            segments = merge_transcript_segments_by_interval(segments, merge_sec)
            text = format_timestamp_segments(segments)
        else:
            text = f"{_deepgram_start_seconds_to_tag(0)} {text}"
            logger.warning(
                "Deepgram 已請求分段（utterances／paragraphs／words），仍無可用時間列；於全文前加上 [00:00]"
            )

    return text, elapsed


def _youtube_transcript_api_from_env():
    """
    建立 YouTubeTranscriptApi。若設定代理相關環境變數，可避免雲端 IP 被 YouTube 封鎖。

    Webshare 住宅代理（與套件 README 一致）：
      YOUTUBE_TRANSCRIPT_WEBSHARE_USERNAME、YOUTUBE_TRANSCRIPT_WEBSHARE_PASSWORD
      選填 YOUTUBE_TRANSCRIPT_WEBSHARE_LOCATIONS：逗號分隔國碼，例如 us,tw

    自訂 HTTP/HTTPS 代理（優先序）：
      YOUTUBE_TRANSCRIPT_HTTP_PROXY、YOUTUBE_TRANSCRIPT_HTTPS_PROXY
      （值格式如 http://user:pass@host:port；結尾 / 可省略；可只設其一，HTTPS 會沿用 HTTP）
      若以上皆未設定，會再讀取標準 HTTP_PROXY／HTTPS_PROXY（或 http_proxy／https_proxy）。

    選填（僅自訂 Generic 代理時）：
      YOUTUBE_TRANSCRIPT_PROXY_RETRIES_WHEN_BLOCKED：遇阻擋時重試次數，預設 8（0=不重試；輪換代理可加大）。
      YOUTUBE_TRANSCRIPT_PROXY_CLOSE_CONNECTION：預設 1；設 0/false/off 則不強制 Connection: close（靜態單一 IP 可試）。

    字幕語言預設優先序（未傳入 languages 時）：YOUTUBE_TRANSCRIPT_LANGUAGES，或內建 en→中文→日韓。

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

    http_u = (os.environ.get("YOUTUBE_TRANSCRIPT_HTTP_PROXY") or "").strip().rstrip("/")
    https_u = (os.environ.get("YOUTUBE_TRANSCRIPT_HTTPS_PROXY") or "").strip().rstrip("/")
    proxy_source = "YOUTUBE_TRANSCRIPT_*"
    if not http_u and not https_u:
        http_u = (os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy") or "").strip().rstrip("/")
        https_u = (os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy") or "").strip().rstrip("/")
        proxy_source = "HTTP_PROXY/HTTPS_PROXY"
    if http_u or https_u:
        try:
            cfg = _env_generic_proxy_config_for_youtube(
                http_url=http_u or None,
                https_url=https_u or None,
            )
        except ImportError as e:
            raise RuntimeError(
                "已設定代理環境變數，但目前的 youtube-transcript-api 不支援 "
                "GenericProxyConfig；請升級套件。"
            ) from e
        logger.info(
            "YouTube transcript: GenericProxyConfig 已啟用（來源 %s，retries_when_blocked=%s）",
            proxy_source,
            cfg.retries_when_blocked,
        )
        return YouTubeTranscriptApi(proxy_config=cfg)

    return YouTubeTranscriptApi()


def youtube_transcript_api_user_message(exc: BaseException) -> str:
    """
    給 HTTP／ValueError 用的簡短說明；避免 youtube_transcript_api 的 str(exc) 含 README、
    Webshare 推銷與 GitHub issue 長段文字。
    """
    from youtube_transcript_api._errors import (
        CouldNotRetrieveTranscript,
        InvalidVideoId,
        IpBlocked,
        NoTranscriptFound,
        PoTokenRequired,
        RequestBlocked,
        TranscriptsDisabled,
        VideoUnavailable,
        YouTubeTranscriptApiException,
    )

    vid = (getattr(exc, "video_id", None) or "").strip()
    label = f"影片 {vid} " if vid else ""

    if isinstance(exc, InvalidVideoId):
        return (
            f"{label}YouTube 影片 ID 無法辨識（請使用 11 字元 id，勿把整段 watch URL 當成 id）。"
        )
    if isinstance(exc, VideoUnavailable):
        return f"{label}不存在或無法存取。"
    if isinstance(exc, TranscriptsDisabled):
        return f"{label}已關閉字幕。"
    if isinstance(exc, NoTranscriptFound):
        codes = getattr(exc, "_requested_language_codes", None) or ()
        try:
            lang_part = "、".join(str(c) for c in codes) if codes else "指定語言"
        except (TypeError, ValueError):
            lang_part = "指定語言"
        return f"{label}找不到 {lang_part} 的字幕。"
    if isinstance(exc, IpBlocked):
        return (
            f"{label}無法取得字幕：YouTube 回傳驗證／反機器人頁或封鎖此來源 IP。"
            "請改用住宅代理、更換乾淨的代理節點，或在本機一般寬頻重試。"
        )
    if isinstance(exc, RequestBlocked):
        if getattr(exc, "_proxy_config", None) is not None:
            return (
                f"{label}無法取得字幕：後端已透過代理連線，且自訂代理路徑已預設重試數次，"
                "YouTube 仍拒絕。常見原因是出口仍為機房／資料中心 IP；請改用住宅型（residential）"
                "代理、加大輪換池，或在本機網路測試。靜態代理可設 "
                "YOUTUBE_TRANSCRIPT_PROXY_RETRIES_WHEN_BLOCKED=0 略過重試。"
            )
        return (
            f"{label}無法取得字幕：連線被 YouTube 視為異常或機房流量。"
            "請設定 YOUTUBE_TRANSCRIPT_HTTP_PROXY／HTTPS_PROXY（或 HTTP_PROXY／HTTPS_PROXY），"
            "格式 http://使用者:密碼@主機:埠。"
        )
    if isinstance(exc, PoTokenRequired):
        return (
            f"{label}目前需額外驗證權杖才能擷取字幕（YouTube 端限制）。"
            "請將 youtube-transcript-api 升級至最新版並留意 release 說明；或改用手動／其它來源的逐字稿。"
        )
    if isinstance(exc, CouldNotRetrieveTranscript):
        return (
            f"{label}無法取得字幕（可能被 YouTube 暫時阻擋或需登入限制）。"
            "若在受限網路，請設定 YOUTUBE_TRANSCRIPT_HTTP_PROXY／HTTPS_PROXY。"
        )
    if isinstance(exc, YouTubeTranscriptApiException):
        line = (str(exc) or "").strip().split("\n", 1)[0]
        return (line[:400] + "…") if len(line) > 400 else line
    return (str(exc) or "未知錯誤")[:400]


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


def youtube_transcript_default_languages() -> list[str]:
    """
    字幕語言嘗試順序（前者優先，套件會依序嘗試至成功）。
    環境變數 YOUTUBE_TRANSCRIPT_LANGUAGES：逗號分隔代碼，例如 ``en,zh-Hant,zh-TW``。
    """
    raw = (os.environ.get("YOUTUBE_TRANSCRIPT_LANGUAGES") or "").strip()
    if raw:
        out = [x.strip() for x in raw.split(",") if x.strip()]
        return out if out else ["en"]
    return ["en", "zh-Hant", "zh-TW", "zh-Hans", "zh-CN", "ja", "ko"]


def _youtube_transcript_entry_text_and_start(entry: Any) -> tuple[str, float]:
    """自字幕項目取出正文（換行改空白）與起點秒數。"""
    if isinstance(entry, dict):
        raw_text = entry.get("text", "")
        try:
            start = float(entry.get("start", 0) or 0)
        except (TypeError, ValueError):
            start = 0.0
    else:
        raw_text = getattr(entry, "text", "")
        try:
            start = float(getattr(entry, "start", 0) or 0)
        except (TypeError, ValueError):
            start = 0.0
    return str(raw_text).replace("\n", " ").strip(), start


def youtube_transcript_plain_text(
    video_id: str,
    languages: list[str] | None = None,
    *,
    with_timestamps: bool = False,
    timestamp_merge_seconds: float | None = None,
) -> tuple[str, float]:
    """
    擷取 YouTube 字幕並併成單一字串。
    ``with_timestamps=False``（預設）：各段以空白連接（與 Colab 範例一致）。
    ``with_timestamps=True``：依各字幕片段 ``start`` 輸出時間標記；再以
    :func:`resolve_timestamp_merge_seconds` 合併相近片段（預設約每 10 秒一個標記）。
    ``timestamp_merge_seconds=0`` 表示維持每一字幕片段一行。
    回傳 (全文, 耗時秒數)。

    languages 為 None 或空串列時，使用 :func:`youtube_transcript_default_languages`。
    """
    from youtube_transcript_api import YouTubeTranscriptApi

    langs = languages if languages else youtube_transcript_default_languages()
    t0 = time.perf_counter()
    api = _youtube_transcript_api_from_env()
    if hasattr(api, "fetch"):
        fetched = api.fetch(video_id, languages=langs)
        if hasattr(fetched, "to_raw_data"):
            transcript = fetched.to_raw_data()
        else:
            transcript = [
                {"text": getattr(s, "text", ""), "start": float(getattr(s, "start", 0) or 0)}
                for s in fetched
            ]
    else:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=langs)

    if with_timestamps:
        segments: list[tuple[float, str]] = []
        for entry in transcript:
            text_content, start_sec = _youtube_transcript_entry_text_and_start(entry)
            if not text_content:
                continue
            segments.append((start_sec, text_content))
        merge_sec = resolve_timestamp_merge_seconds(timestamp_merge_seconds)
        segments = merge_transcript_segments_by_interval(segments, merge_sec)
        full_text = format_timestamp_segments(segments)
    else:
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
