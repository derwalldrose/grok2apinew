#!/usr/bin/env python3
"""
Telegram Bot for Grok2API public endpoints.

Features:
- Chat
- Waterfall image generation
- Video generation
- Video extension

Env:
- TG_BOT_TOKEN (required)
- GROK_BASE_URL (default: http://127.0.0.1:18083)
- GROK_PUBLIC_KEY (required if public_key enabled)
- TG_ALLOWED_CHAT_IDS (optional, comma-separated chat IDs)
- TG_CHAT_MODEL (default: grok-3)
- TG_VIDEO_RATIO (default: 3:2)
- TG_VIDEO_RESOLUTION (default: 480p)
- TG_VIDEO_PRESET (default: normal)
- TG_VIDEO_CONCURRENT (default: 4)
- TG_IMAGE_RATIO (default: 2:3)
- TG_IMAGE_WATERFALL_COUNT (default: 4)
- TG_VIDEO_TIMEOUT_SEC (default: 600)
- TG_VIDEO_UPLOAD_MAX_MB (default: 45, Telegram fallback upload limit)
"""

from __future__ import annotations

import asyncio
import argparse
import base64
import json
import os
import re
import signal
import time
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple
from urllib.parse import urlencode

import httpx


VIDEO_URL_RE = re.compile(r"https?://[^\s\"'<>]+\.mp4[^\s\"'<>]*", re.IGNORECASE)
UUID_RE = re.compile(r"[0-9a-fA-F-]{32,36}")
POST_ID_PATTERNS = (
    re.compile(r"/generated/([0-9a-fA-F-]{32,36})(?:/|$)"),
    re.compile(r"/([0-9a-fA-F-]{32,36})/generated_video"),
)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except Exception:
        return default


def now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def log(msg: str) -> None:
    print(f"[{now_ts()}] {msg}", flush=True)


def sanitize_url_candidate(value: str) -> str:
    raw = str(value or "").strip().strip("\"'")
    if not raw:
        return ""
    raw = raw.replace("\\/", "/")
    while raw and raw[-1] in "\\]),};":
        raw = raw[:-1]
    return raw.strip()


def normalize_url(base_url: str, maybe_url: str) -> str:
    raw = sanitize_url_candidate(maybe_url)
    if not raw:
        return ""
    if raw.startswith("http://") or raw.startswith("https://"):
        return raw
    if raw.startswith("/"):
        return f"{base_url.rstrip('/')}{raw}"
    return f"{base_url.rstrip('/')}/{raw.lstrip('/')}"


def _extract_post_id_from_text(text: str) -> str:
    payload = str(text or "")
    if not payload:
        return ""
    for pattern in POST_ID_PATTERNS:
        m = pattern.search(payload)
        if m:
            return m.group(1)
    key_match = re.search(
        r"(?:post_id|videoPostId|parent_post_id|extend_post_id|file_attachment_id)"
        r"[\"'=\s:]+([0-9a-fA-F-]{32,36})",
        payload,
    )
    if key_match:
        return key_match.group(1)
    return ""


def _extract_known_post_id_recursive(payload: Any) -> str:
    keys = {
        "post_id",
        "videoPostId",
        "parent_post_id",
        "extend_post_id",
        "original_post_id",
        "file_attachment_id",
    }
    if isinstance(payload, dict):
        for key in keys:
            val = sanitize_url_candidate(str(payload.get(key) or ""))
            if val and UUID_RE.fullmatch(val):
                return val
        for val in payload.values():
            found = _extract_known_post_id_recursive(val)
            if found:
                return found
    elif isinstance(payload, list):
        for item in payload:
            found = _extract_known_post_id_recursive(item)
            if found:
                return found
    return ""


def extract_video_url(payload: Any, base_url: str) -> str:
    if isinstance(payload, dict):
        for key in ("video_url", "url"):
            val = sanitize_url_candidate(str(payload.get(key) or ""))
            if val and ".mp4" in val.lower():
                return normalize_url(base_url, val)
    text = json.dumps(payload, ensure_ascii=False) if payload is not None else ""
    # Prefer generated_video*.mp4 links if present.
    matches = VIDEO_URL_RE.findall(text)
    if matches:
        for candidate in matches:
            if "generated_video" in candidate:
                return normalize_url(base_url, candidate)
        return normalize_url(base_url, matches[0])
    # Local file route fallback.
    local_match = re.search(
        r"/v1/(?:files/video|public/video/splice)/[^\s\"'<>]+\.mp4[^\s\"'<>]*",
        text,
    )
    if local_match:
        return normalize_url(base_url, local_match.group(0))
    local_match = re.search(
        r"/v1/(?:files/video|public/video/splice)/[^\s\"'<>]+",
        text,
    )
    if local_match:
        return normalize_url(base_url, local_match.group(0))
    return ""


def extract_post_id(payload: Any, video_url: str = "") -> str:
    from_payload = _extract_known_post_id_recursive(payload)
    if from_payload:
        return from_payload

    if video_url:
        from_url = _extract_post_id_from_text(video_url)
        if from_url:
            return from_url

    text = json.dumps(payload, ensure_ascii=False) if payload is not None else ""
    from_text = _extract_post_id_from_text(text)
    if from_text:
        return from_text
    return ""


def extract_text_message(resp: Dict[str, Any]) -> str:
    choices = resp.get("choices") or []
    if not choices:
        return json.dumps(resp, ensure_ascii=False)
    msg = choices[0].get("message") or {}
    content = msg.get("content")
    if isinstance(content, str):
        return content.strip() or "(empty)"
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text":
                parts.append(str(item.get("text") or ""))
        text = "\n".join([p for p in parts if p]).strip()
        return text or json.dumps(content, ensure_ascii=False)
    return json.dumps(resp, ensure_ascii=False)


def decode_b64_image(raw_b64: str) -> Optional[bytes]:
    text = str(raw_b64 or "").strip()
    if not text:
        return None
    if text.startswith("data:") and "," in text:
        text = text.split(",", 1)[1]
    try:
        return base64.b64decode(text, validate=True)
    except Exception:
        return None


@dataclass
class UserState:
    last_video_url: str = ""
    last_video_post_id: str = ""
    last_file_attachment_id: str = ""
    last_video_duration_sec: float = 0.0
    pending_mode: str = ""
    mode_image_count: int = 0
    mode_video_count: int = 0


class TelegramApi:
    def __init__(self, token: str):
        self.token = token
        self.base = f"https://api.telegram.org/bot{token}"
        self.timeout = httpx.Timeout(connect=10.0, read=60.0, write=30.0, pool=10.0)
        self.client = httpx.AsyncClient(timeout=self.timeout)

    async def close(self) -> None:
        await self.client.aclose()

    async def call(
        self,
        method: str,
        data: Optional[Dict[str, Any]] = None,
        files: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        url = f"{self.base}/{method}"
        resp = await self.client.post(url, data=data, files=files)
        payload = resp.json()
        if not payload.get("ok"):
            raise RuntimeError(f"telegram_{method}_failed: {payload}")
        return payload

    async def get_updates(self, offset: int, timeout_sec: int = 30) -> List[Dict[str, Any]]:
        query = urlencode(
            {
                "offset": offset,
                "timeout": timeout_sec,
                "allowed_updates": json.dumps(["message"]),
            }
        )
        url = f"{self.base}/getUpdates?{query}"
        resp = await self.client.get(url, timeout=httpx.Timeout(connect=10.0, read=timeout_sec + 5.0, write=30.0, pool=10.0))
        payload = resp.json()
        if not payload.get("ok"):
            raise RuntimeError(f"telegram_getUpdates_failed: {payload}")
        return payload.get("result") or []

    async def send_message(self, chat_id: int, text: str, reply_to: Optional[int] = None) -> None:
        chunks = [text[i : i + 3500] for i in range(0, len(text), 3500)] or [text]
        for chunk in chunks:
            data: Dict[str, Any] = {"chat_id": str(chat_id), "text": chunk}
            if reply_to:
                data["reply_to_message_id"] = str(reply_to)
            await self.call("sendMessage", data=data)

    async def send_photo(
        self,
        chat_id: int,
        *,
        photo_url: str = "",
        photo_bytes: Optional[bytes] = None,
        caption: str = "",
    ) -> None:
        data = {"chat_id": str(chat_id)}
        if caption:
            data["caption"] = caption
        if photo_bytes is not None:
            files = {"photo": ("image.jpg", photo_bytes, "image/jpeg")}
            await self.call("sendPhoto", data=data, files=files)
            return
        data["photo"] = photo_url
        await self.call("sendPhoto", data=data)

    async def send_video(
        self,
        chat_id: int,
        video_url: str = "",
        video_bytes: Optional[bytes] = None,
        caption: str = "",
        filename: str = "video.mp4",
    ) -> None:
        data = {"chat_id": str(chat_id)}
        if caption:
            data["caption"] = caption
        if video_bytes is not None:
            files = {"video": (filename, video_bytes, "video/mp4")}
            await self.call("sendVideo", data=data, files=files)
            return
        data["video"] = video_url
        await self.call("sendVideo", data=data)


class BackendApi:
    def __init__(self, base_url: str, public_key: str):
        self.base_url = base_url.rstrip("/")
        self.public_key = public_key.strip()
        self.timeout = httpx.Timeout(connect=12.0, read=120.0, write=30.0, pool=12.0)
        self.client = httpx.AsyncClient(timeout=self.timeout)

    def auth_headers(self) -> Dict[str, str]:
        if not self.public_key:
            return {}
        return {"Authorization": f"Bearer {self.public_key}"}

    async def close(self) -> None:
        await self.client.aclose()

    async def post_json(self, path: str, body: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        headers = {"Content-Type": "application/json", **self.auth_headers()}
        resp = await self.client.post(url, headers=headers, json=body)
        if resp.status_code >= 400:
            raise RuntimeError(f"{path} failed: {resp.status_code} {resp.text}")
        return resp.json()

    async def chat(self, prompt: str, model: str) -> str:
        body = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }
        resp = await self.post_json("/v1/public/chat/completions", body)
        return extract_text_message(resp)

    async def imagine_start(self, prompt: str, aspect_ratio: str, nsfw: Optional[bool]) -> str:
        body: Dict[str, Any] = {"prompt": prompt, "aspect_ratio": aspect_ratio}
        if nsfw is not None:
            body["nsfw"] = bool(nsfw)
        resp = await self.post_json("/v1/public/imagine/start", body)
        task_id = str(resp.get("task_id") or "").strip()
        if not task_id:
            raise RuntimeError(f"imagine start invalid response: {resp}")
        return task_id

    async def video_start(self, body: Dict[str, Any]) -> str:
        task_ids = await self.video_start_tasks(body)
        return task_ids[0]

    async def video_start_tasks(self, body: Dict[str, Any]) -> List[str]:
        resp = await self.post_json("/v1/public/video/start", body)
        task_ids: List[str] = []
        raw_task_ids = resp.get("task_ids") or []
        if isinstance(raw_task_ids, list):
            for item in raw_task_ids:
                val = str(item or "").strip()
                if val:
                    task_ids.append(val)
        if not task_ids:
            task_id = str(resp.get("task_id") or "").strip()
            if task_id:
                task_ids.append(task_id)
        if not task_ids:
            raise RuntimeError(f"video start invalid response: {resp}")
        return task_ids

    async def download_binary(self, url: str, max_bytes: int) -> bytes:
        final_url = normalize_url(self.base_url, url)
        headers = self.auth_headers()
        resp = await self.client.get(final_url, headers=headers)
        if resp.status_code >= 400:
            raise RuntimeError(f"download failed: {resp.status_code} {resp.text[:200]}")
        blob = bytes(resp.content or b"")
        if not blob:
            raise RuntimeError("download failed: empty content")
        if len(blob) > max(1, int(max_bytes)):
            raise RuntimeError(
                f"download too large: {len(blob)} bytes > {int(max_bytes)} bytes"
            )
        return blob

    async def iter_sse_json(
        self, path: str, params: Dict[str, Any], timeout_sec: float
    ) -> AsyncGenerator[Dict[str, Any], None]:
        url = f"{self.base_url}{path}"
        headers = {"Accept": "text/event-stream", **self.auth_headers()}
        timeout = httpx.Timeout(connect=12.0, read=timeout_sec, write=30.0, pool=12.0)
        async with self.client.stream("GET", url, headers=headers, params=params, timeout=timeout) as resp:
            if resp.status_code >= 400:
                text = await resp.aread()
                raise RuntimeError(
                    f"SSE {path} failed: {resp.status_code} {text.decode(errors='ignore')[:300]}"
                )
            event_name = ""
            data_lines: List[str] = []
            async for line in resp.aiter_lines():
                if line is None:
                    continue
                raw = line.strip("\r")
                if raw == "":
                    if not data_lines:
                        event_name = ""
                        continue
                    data_text = "\n".join(data_lines).strip()
                    event_name = event_name or "message"
                    data_lines = []
                    if data_text == "[DONE]":
                        yield {"_event": event_name, "_done": True}
                        break
                    try:
                        payload = json.loads(data_text)
                        if isinstance(payload, dict):
                            payload.setdefault("_event", event_name)
                            yield payload
                        else:
                            yield {"_event": event_name, "_raw": payload}
                    except Exception:
                        yield {"_event": event_name, "_raw_text": data_text}
                    event_name = ""
                    continue
                if raw.startswith("event:"):
                    event_name = raw[6:].strip()
                elif raw.startswith("data:"):
                    data_lines.append(raw[5:].strip())

    async def wait_video_url(self, task_id: str, timeout_sec: float) -> Tuple[str, str]:
        start = time.time()
        async for payload in self.iter_sse_json(
            "/v1/public/video/sse",
            {"task_id": task_id, "t": str(int(time.time() * 1000))},
            timeout_sec=timeout_sec,
        ):
            if "error" in payload:
                raise RuntimeError(f"video sse error: {payload.get('error')}")
            video_url = extract_video_url(payload, self.base_url)
            if video_url:
                post_id = extract_post_id(payload, video_url)
                return video_url, post_id
            if payload.get("_done"):
                break
            if time.time() - start > timeout_sec:
                break
        raise RuntimeError("video result timeout or missing url")


class TgGrokBot:
    def __init__(self):
        self.bot_token = os.getenv("TG_BOT_TOKEN", "").strip()
        self.backend_url = os.getenv("GROK_BASE_URL", "http://127.0.0.1:18083").strip()
        self.public_key = os.getenv("GROK_PUBLIC_KEY", "").strip()
        self.allowed_chat_ids = self._parse_allowed_chat_ids(os.getenv("TG_ALLOWED_CHAT_IDS", ""))
        self.chat_model = os.getenv("TG_CHAT_MODEL", "grok-3").strip()
        self.video_ratio = os.getenv("TG_VIDEO_RATIO", "3:2").strip()
        self.video_resolution = os.getenv("TG_VIDEO_RESOLUTION", "480p").strip()
        self.video_preset = os.getenv("TG_VIDEO_PRESET", "normal").strip()
        self.video_concurrent = max(1, min(4, _env_int("TG_VIDEO_CONCURRENT", 4)))
        self.video_length_sec = 6
        self.video_extend_length_sec = 10
        self.image_ratio = os.getenv("TG_IMAGE_RATIO", "2:3").strip()
        self.image_waterfall_count = _env_int("TG_IMAGE_WATERFALL_COUNT", 4)
        self.video_timeout_sec = _env_float("TG_VIDEO_TIMEOUT_SEC", 600.0)
        self.video_upload_max_mb = _env_int("TG_VIDEO_UPLOAD_MAX_MB", 45)
        self.running = True
        self.offset = 0
        self.user_state: Dict[int, UserState] = {}
        self.chat_locks: Dict[int, asyncio.Lock] = {}
        self.telegram = TelegramApi(self.bot_token)
        self.backend = BackendApi(self.backend_url, self.public_key)
        self.bg_tasks: set[asyncio.Task] = set()

    @staticmethod
    def _parse_allowed_chat_ids(raw: str) -> set[int]:
        out: set[int] = set()
        for part in raw.split(","):
            p = part.strip()
            if not p:
                continue
            try:
                out.add(int(p))
            except Exception:
                continue
        return out

    def _is_allowed_chat(self, chat_id: int) -> bool:
        if not self.allowed_chat_ids:
            return True
        return chat_id in self.allowed_chat_ids

    def _get_lock(self, chat_id: int) -> asyncio.Lock:
        if chat_id not in self.chat_locks:
            self.chat_locks[chat_id] = asyncio.Lock()
        return self.chat_locks[chat_id]

    def _get_state(self, chat_id: int) -> UserState:
        if chat_id not in self.user_state:
            self.user_state[chat_id] = UserState()
        st = self.user_state[chat_id]
        if st.mode_image_count <= 0:
            st.mode_image_count = max(1, int(self.image_waterfall_count))
        if st.mode_video_count <= 0:
            st.mode_video_count = max(1, int(self.video_concurrent))
        return st

    @staticmethod
    def _clamp_count(value: int, minimum: int = 1, maximum: int = 16) -> int:
        return max(minimum, min(maximum, int(value)))

    @staticmethod
    def _parse_count_and_prompt(raw: str) -> Tuple[Optional[int], str]:
        text = str(raw or "").strip()
        if not text:
            return None, ""
        first, sep, rest = text.partition(" ")
        if re.fullmatch(r"\d{1,3}", first):
            try:
                return int(first), rest.strip()
            except Exception:
                return None, text
        return None, text

    def _spawn_user_task(
        self,
        chat_id: int,
        message_id: int,
        label: str,
        coro: Any,
    ) -> None:
        async def _runner() -> None:
            try:
                await coro
            except Exception as e:
                await self.telegram.send_message(
                    chat_id,
                    f"{label}失败: {e}",
                    reply_to=message_id,
                )

        task = asyncio.create_task(_runner())
        self.bg_tasks.add(task)
        task.add_done_callback(self.bg_tasks.discard)

    async def _send_video_with_fallback(self, chat_id: int, video_url: str, caption: str) -> None:
        try:
            await self.telegram.send_video(chat_id, video_url=video_url, caption=caption)
            return
        except Exception as e:
            log(f"sendVideo(url) failed: {e}")
        try:
            max_bytes = max(1, int(self.video_upload_max_mb)) * 1024 * 1024
            video_bytes = await self.backend.download_binary(video_url, max_bytes=max_bytes)
            await self.telegram.send_video(
                chat_id,
                video_bytes=video_bytes,
                caption=f"{caption}\n(已通过上传回退发送)",
            )
            return
        except Exception as e:
            log(f"sendVideo(upload) failed: {e}")
        await self.telegram.send_message(chat_id, f"{caption}\nvideo_url={video_url}")

    async def stop(self) -> None:
        self.running = False
        for task in list(self.bg_tasks):
            task.cancel()
        await asyncio.gather(*list(self.bg_tasks), return_exceptions=True)
        await self.backend.close()
        await self.telegram.close()

    async def run(self) -> None:
        if not self.bot_token:
            raise RuntimeError("TG_BOT_TOKEN is required")
        log("TG bot started")
        while self.running:
            try:
                updates = await self.telegram.get_updates(self.offset, timeout_sec=30)
                for upd in updates:
                    self.offset = max(self.offset, int(upd.get("update_id", 0)) + 1)
                    task = asyncio.create_task(self.handle_update(upd))
                    self.bg_tasks.add(task)
                    task.add_done_callback(self.bg_tasks.discard)
            except asyncio.CancelledError:
                break
            except Exception as e:
                log(f"poll error: {e}")
                await asyncio.sleep(2.0)

    async def handle_update(self, update: Dict[str, Any]) -> None:
        msg = update.get("message") or {}
        text = str(msg.get("text") or "").strip()
        chat = msg.get("chat") or {}
        chat_id = int(chat.get("id") or 0)
        message_id = int(msg.get("message_id") or 0)
        if not chat_id or not text:
            return
        if not self._is_allowed_chat(chat_id):
            return
        lock = self._get_lock(chat_id)
        async with lock:
            try:
                await self._dispatch_text(chat_id, message_id, text)
            except Exception as e:
                await self.telegram.send_message(chat_id, f"处理失败: {e}", reply_to=message_id)

    async def _dispatch_text(self, chat_id: int, message_id: int, text: str) -> None:
        st = self._get_state(chat_id)

        if text.startswith("/start") or text.startswith("/help"):
            await self.telegram.send_message(
                chat_id,
                "\n".join(
                    [
                        "Grok TG Bot 可用命令:",
                        "/chat - 进入聊天模式（后续文本按聊天处理）",
                        "/chat <问题> - 直接聊天",
                        "/img [张数] - 进入生图模式（如 /img 8）",
                        "/img [张数] <提示词> - 直接生图",
                        "/video [路数] - 进入生视频模式（如 /video 8）",
                        "/video [路数] <提示词> - 直接生视频",
                        "/extend [起始秒] [提示词] - 延长最近视频（默认视频末尾）",
                        "/setpost <post_id> - 手动设置延长基准 post_id",
                        "/cancel - 退出当前命令模式",
                        "/state - 查看当前会话状态",
                    ]
                ),
                reply_to=message_id,
            )
            return

        if text.startswith("/state"):
            await self.telegram.send_message(
                chat_id,
                f"last_video_post_id={st.last_video_post_id or '-'}\n"
                f"last_file_attachment_id={st.last_file_attachment_id or '-'}\n"
                f"last_video_url={st.last_video_url or '-'}\n"
                f"last_video_duration_sec={st.last_video_duration_sec:.3f}\n"
                f"pending_mode={st.pending_mode or '-'}\n"
                f"mode_image_count={st.mode_image_count}\n"
                f"mode_video_count={st.mode_video_count}",
                reply_to=message_id,
            )
            return

        if text.startswith("/cancel"):
            st.pending_mode = ""
            await self.telegram.send_message(chat_id, "已退出命令模式。", reply_to=message_id)
            return

        if text.startswith("/setpost "):
            post_id = text.split(" ", 1)[1].strip()
            if not UUID_RE.fullmatch(post_id):
                await self.telegram.send_message(chat_id, "post_id 格式无效", reply_to=message_id)
                return
            st.pending_mode = ""
            st.last_video_post_id = post_id
            st.last_video_duration_sec = 0.0
            if not st.last_file_attachment_id:
                st.last_file_attachment_id = post_id
            await self.telegram.send_message(chat_id, f"已设置 post_id: {post_id}", reply_to=message_id)
            return

        chat_match = re.match(r"^/chat(?:@\w+)?(?:\s+(.+))?$", text)
        if chat_match:
            prompt = (chat_match.group(1) or "").strip()
            st.pending_mode = "chat"
            if not prompt:
                await self.telegram.send_message(
                    chat_id,
                    "已进入聊天模式。后续普通文本都将走聊天。\n发送 /cancel 可退出模式。",
                    reply_to=message_id,
                )
                return
            self._spawn_user_task(
                chat_id,
                message_id,
                "聊天",
                self.handle_chat(chat_id, message_id, prompt),
            )
            return

        img_match = re.match(r"^/img(?:@\w+)?(?:\s+(.+))?$", text)
        if img_match:
            rest = (img_match.group(1) or "").strip()
            count, prompt = self._parse_count_and_prompt(rest)
            if count is not None:
                st.mode_image_count = self._clamp_count(count, 1, 16)
            st.pending_mode = "img"
            target_count = st.mode_image_count
            if not prompt:
                await self.telegram.send_message(
                    chat_id,
                    f"已进入生图模式。当前每次目标张数={target_count}。\n"
                    "后续普通文本将直接生图；发送 /cancel 退出模式。",
                    reply_to=message_id,
                )
                return
            self._spawn_user_task(
                chat_id,
                message_id,
                "生图",
                self.handle_image_flow(chat_id, message_id, prompt, target_count=target_count),
            )
            return

        video_match = re.match(r"^/video(?:@\w+)?(?:\s+(.+))?$", text)
        if video_match:
            rest = (video_match.group(1) or "").strip()
            count, prompt = self._parse_count_and_prompt(rest)
            if count is not None:
                st.mode_video_count = self._clamp_count(count, 1, 16)
            target_lanes = st.mode_video_count
            st.pending_mode = "video"
            if not prompt:
                await self.telegram.send_message(
                    chat_id,
                    f"已进入生视频模式。当前每次并发路数={target_lanes}。\n"
                    "后续普通文本将直接生视频；发送 /cancel 退出模式。",
                    reply_to=message_id,
                )
                return
            self._spawn_user_task(
                chat_id,
                message_id,
                "生视频",
                self.handle_video(chat_id, message_id, prompt, lanes=target_lanes),
            )
            return

        if text == "/extend" or text.startswith("/extend "):
            rest = text[len("/extend") :].strip()
            start_sec: Optional[float] = None
            prompt = ""
            if rest:
                parts = rest.split(" ", 1)
                try:
                    start_sec = float(parts[0].strip())
                    prompt = parts[1].strip() if len(parts) > 1 else ""
                except Exception:
                    # If the first token is not a number, treat all as prompt,
                    # and default start to current video end.
                    start_sec = None
                    prompt = rest
            self._spawn_user_task(
                chat_id,
                message_id,
                "视频延长",
                self.handle_extend(chat_id, message_id, start_sec, prompt),
            )
            return

        if text.startswith("/"):
            await self.telegram.send_message(
                chat_id,
                "未知命令。发送 /help 查看可用命令。",
                reply_to=message_id,
            )
            return

        # Non-command text: route by mode; default fallback is chat.
        prompt = text.strip()
        if not prompt:
            await self.telegram.send_message(chat_id, "请输入内容", reply_to=message_id)
            return

        if st.pending_mode == "img":
            self._spawn_user_task(
                chat_id,
                message_id,
                "生图",
                self.handle_image_flow(chat_id, message_id, prompt, target_count=st.mode_image_count),
            )
            return
        if st.pending_mode == "video":
            self._spawn_user_task(
                chat_id,
                message_id,
                "生视频",
                self.handle_video(chat_id, message_id, prompt, lanes=st.mode_video_count),
            )
            return

        self._spawn_user_task(
            chat_id,
            message_id,
            "聊天",
            self.handle_chat(chat_id, message_id, prompt),
        )

    async def handle_chat(self, chat_id: int, message_id: int, prompt: str) -> None:
        await self.telegram.send_message(chat_id, "聊天处理中...", reply_to=message_id)
        text = await self.backend.chat(prompt, model=self.chat_model)
        await self.telegram.send_message(chat_id, text or "(empty)")

    async def handle_image_flow(
        self, chat_id: int, message_id: int, prompt: str, target_count: Optional[int] = None
    ) -> None:
        target = self._clamp_count(target_count or self.image_waterfall_count, 1, 16)
        await self.telegram.send_message(
            chat_id,
            f"开始瀑布流生图，目标 {target} 张...",
            reply_to=message_id,
        )
        task_id = await self.backend.imagine_start(prompt, aspect_ratio=self.image_ratio, nsfw=None)
        got = 0
        sent_final_ids: set[str] = set()
        sent_urls: set[str] = set()
        async for payload in self.backend.iter_sse_json(
            "/v1/public/imagine/sse",
            {"task_id": task_id, "t": str(int(time.time() * 1000))},
            timeout_sec=300.0,
        ):
            p_type = str(payload.get("type") or "")
            if p_type == "error":
                await self.telegram.send_message(chat_id, f"生图失败: {payload.get('message') or payload}")
                return
            # Filter out preview/medium partial frames; keep final images only.
            stage = str(payload.get("stage") or "").lower()
            is_final = bool(payload.get("is_final")) or stage == "final" or p_type == "image_generation.completed"
            if not is_final:
                continue

            image_id = str(payload.get("image_id") or "").strip()
            if image_id and image_id in sent_final_ids:
                continue
            image_url = normalize_url(self.backend_url, str(payload.get("url") or ""))
            image_b64 = str(payload.get("b64_json") or "")
            if not image_url and not image_b64:
                continue
            if image_url and image_url in sent_urls:
                continue
            got += 1
            caption = f"瀑布流第 {got} 张"
            if image_url:
                await self.telegram.send_photo(chat_id, photo_url=image_url, caption=caption)
                sent_urls.add(image_url)
            else:
                image_bytes = decode_b64_image(image_b64)
                if image_bytes:
                    await self.telegram.send_photo(chat_id, photo_bytes=image_bytes, caption=caption)
            if image_id:
                sent_final_ids.add(image_id)
            if got >= target:
                break
        await self.telegram.send_message(chat_id, f"瀑布流结束，共 {got} 张")

    async def handle_video(
        self, chat_id: int, message_id: int, prompt: str, lanes: Optional[int] = None
    ) -> None:
        lanes = self._clamp_count(lanes or self.video_concurrent, 1, 16)
        await self.telegram.send_message(
            chat_id,
            f"视频生成中，目标 {lanes} 路并发，通常需 30-90 秒...",
            reply_to=message_id,
        )
        task_ids: List[str] = []
        remaining = lanes
        while remaining > 0:
            chunk = min(4, remaining)
            chunk_task_ids = await self.backend.video_start_tasks(
                {
                    "prompt": prompt,
                    "aspect_ratio": self.video_ratio,
                    "video_length": self.video_length_sec,
                    "resolution_name": self.video_resolution,
                    "preset": self.video_preset,
                    "concurrent": chunk,
                    "reasoning_effort": "low",
                }
            )
            if not chunk_task_ids:
                raise RuntimeError("video start returned empty task_ids")
            task_ids.extend(chunk_task_ids[:chunk])
            remaining -= len(chunk_task_ids[:chunk])
        total = len(task_ids)
        await self.telegram.send_message(chat_id, f"已提交 {total} 路任务，等待结果...")

        async def _wait_lane(lane: int, task_id: str) -> Tuple[int, str, str, Optional[str]]:
            try:
                video_url, post_id = await self.backend.wait_video_url(
                    task_id, timeout_sec=self.video_timeout_sec
                )
                return lane, video_url, post_id, None
            except Exception as e:
                return lane, "", "", str(e)

        jobs: List[asyncio.Task] = [
            asyncio.create_task(_wait_lane(i + 1, tid)) for i, tid in enumerate(task_ids)
        ]
        successes: List[Tuple[int, str, str]] = []
        failures: List[Tuple[int, str]] = []
        for fut in asyncio.as_completed(jobs):
            lane, video_url, post_id, err = await fut
            if err or not video_url:
                failures.append((lane, err or "missing video url"))
                await self.telegram.send_message(chat_id, f"第 {lane}/{total} 路失败: {err or 'unknown'}")
                continue
            successes.append((lane, video_url, post_id))
            await self._send_video_with_fallback(
                chat_id,
                video_url,
                caption=f"视频第 {lane}/{total} 路完成\npost_id={post_id or extract_post_id({}, video_url) or '-'}",
            )

        for job in jobs:
            if not job.done():
                job.cancel()
        await asyncio.gather(*jobs, return_exceptions=True)

        if not successes:
            raise RuntimeError(f"{total}路视频全部失败，请稍后重试")

        # Deterministic default base for extension: lane-1 success first.
        successes.sort(key=lambda x: x[0])
        default_lane, default_url, default_post_id = successes[0]
        st = self._get_state(chat_id)
        st.last_video_url = default_url
        st.last_video_post_id = default_post_id or extract_post_id({}, default_url)
        st.last_file_attachment_id = st.last_video_post_id
        st.last_video_duration_sec = float(self.video_length_sec)
        await self.telegram.send_message(
            chat_id,
            f"{total}路生成结束：成功 {len(successes)} 路，失败 {len(failures)} 路。\n"
            f"默认延长基准=第 {default_lane} 路，时长≈{st.last_video_duration_sec:.0f}s\n"
            f"post_id={st.last_video_post_id or '-'}",
        )

    async def handle_extend(
        self, chat_id: int, message_id: int, start_sec: Optional[float], prompt: str
    ) -> None:
        st = self._get_state(chat_id)
        if not st.last_video_post_id:
            await self.telegram.send_message(
                chat_id,
                "没有可延长的 post_id，请先 /video 生成或 /setpost 手动设置",
                reply_to=message_id,
            )
            return
        if start_sec is None:
            if st.last_video_duration_sec > 0:
                start_sec = float(st.last_video_duration_sec)
            else:
                # If duration is unknown (e.g. manual /setpost), default near end cap.
                start_sec = 30.0
        start_sec = max(0.0, float(start_sec))
        await self.telegram.send_message(
            chat_id,
            f"开始延长，post_id={st.last_video_post_id}, start={start_sec:.3f}s",
            reply_to=message_id,
        )
        body = {
            "prompt": prompt,
            "aspect_ratio": self.video_ratio,
            "video_length": self.video_extend_length_sec,
            "resolution_name": self.video_resolution,
            "preset": "spicy" if not prompt.strip() else self.video_preset,
            "reasoning_effort": "low",
            "concurrent": 1,
            "is_video_extension": True,
            "extend_post_id": st.last_video_post_id,
            "video_extension_start_time": max(0.0, float(start_sec)),
            "original_post_id": st.last_video_post_id,
            "file_attachment_id": st.last_file_attachment_id or st.last_video_post_id,
            "stitch_with_extend": True,
        }
        task_id = await self.backend.video_start(body)
        video_url, new_post_id = await self.backend.wait_video_url(task_id, timeout_sec=self.video_timeout_sec)
        st.last_video_url = video_url
        if new_post_id:
            st.last_video_post_id = new_post_id
            st.last_file_attachment_id = new_post_id
        base_duration = max(0.0, float(st.last_video_duration_sec or 0.0))
        estimated_duration = max(base_duration, start_sec + float(self.video_extend_length_sec))
        st.last_video_duration_sec = min(30.0, estimated_duration)
        await self._send_video_with_fallback(
            chat_id,
            video_url,
            caption=(
                f"视频延长完成\nnew_post_id={st.last_video_post_id or '-'}\n"
                f"estimated_duration≈{st.last_video_duration_sec:.0f}s"
            ),
        )


async def amain() -> None:
    bot = TgGrokBot()
    stop_event = asyncio.Event()

    def _handle_sig() -> None:
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _handle_sig)
        except NotImplementedError:
            pass

    run_task = asyncio.create_task(bot.run())
    wait_task = asyncio.create_task(stop_event.wait())
    done, pending = await asyncio.wait(
        {run_task, wait_task}, return_when=asyncio.FIRST_COMPLETED
    )
    for task in pending:
        task.cancel()
    await bot.stop()
    if run_task in done:
        exc = run_task.exception()
        if exc:
            raise exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Telegram bot for Grok2API public endpoints")
    parser.add_argument("--tg-token", dest="tg_token", default="", help="Telegram bot token")
    parser.add_argument(
        "--tg-chat-id",
        dest="tg_chat_id",
        default="",
        help="Allowed Telegram chat id(s), comma-separated",
    )
    parser.add_argument("--backend-url", dest="backend_url", default="", help="Grok API base URL")
    parser.add_argument("--public-key", dest="public_key", default="", help="Grok public key")
    args = parser.parse_args()

    if args.tg_token.strip():
        os.environ["TG_BOT_TOKEN"] = args.tg_token.strip()
    if args.tg_chat_id.strip():
        os.environ["TG_ALLOWED_CHAT_IDS"] = args.tg_chat_id.strip()
    if args.backend_url.strip():
        os.environ["GROK_BASE_URL"] = args.backend_url.strip()
    if args.public_key.strip():
        os.environ["GROK_PUBLIC_KEY"] = args.public_key.strip()

    try:
        asyncio.run(amain())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
