"""Utility functions for nanobot."""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import tiktoken


def detect_image_mime(data: bytes) -> str | None:
    """Detect image MIME type from magic bytes, ignoring file extension."""
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def ensure_dir(path: Path) -> Path:
    """Ensure directory exists, return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def timestamp() -> str:
    """Current ISO timestamp."""
    return datetime.now().isoformat()


_UNSAFE_CHARS = re.compile(r'[<>:"/\\|?*]')

def safe_filename(name: str) -> str:
    """Replace unsafe path characters with underscores."""
    return _UNSAFE_CHARS.sub("_", name).strip()


def split_message(content: str, max_len: int = 2000) -> list[str]:
    """
    Split content into chunks within max_len, preferring line breaks.

    Args:
        content: The text content to split.
        max_len: Maximum length per chunk (default 2000 for Discord compatibility).

    Returns:
        List of message chunks, each within max_len.
    """
    if not content:
        return []
    if len(content) <= max_len:
        return [content]
    chunks: list[str] = []
    while content:
        if len(content) <= max_len:
            chunks.append(content)
            break
        cut = content[:max_len]
        # Try to break at newline first, then space, then hard break
        pos = cut.rfind('\n')
        if pos <= 0:
            pos = cut.rfind(' ')
        if pos <= 0:
            pos = max_len
        chunks.append(content[:pos])
        content = content[pos:].lstrip()
    return chunks


def build_assistant_message(
    content: str | None,
    tool_calls: list[dict[str, Any]] | None = None,
    reasoning_content: str | None = None,
    thinking_blocks: list[dict] | None = None,
) -> dict[str, Any]:
    """Build a provider-safe assistant message with optional reasoning fields."""
    msg: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    if reasoning_content is not None:
        msg["reasoning_content"] = reasoning_content
    if thinking_blocks:
        msg["thinking_blocks"] = thinking_blocks
    return msg


def estimate_prompt_tokens(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
) -> int:
    """Estimate prompt tokens with tiktoken."""
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        parts: list[str] = []
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        txt = part.get("text", "")
                        if txt:
                            parts.append(txt)
        if tools:
            parts.append(json.dumps(tools, ensure_ascii=False))
        return len(enc.encode("\n".join(parts)))
    except Exception:
        return 0


def estimate_message_tokens(message: dict[str, Any]) -> int:
    """Estimate prompt tokens contributed by one persisted message."""
    content = message.get("content")
    parts: list[str] = []
    if isinstance(content, str):
        parts.append(content)
    elif isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text = part.get("text", "")
                if text:
                    parts.append(text)
            else:
                parts.append(json.dumps(part, ensure_ascii=False))
    elif content is not None:
        parts.append(json.dumps(content, ensure_ascii=False))

    for key in ("name", "tool_call_id"):
        value = message.get(key)
        if isinstance(value, str) and value:
            parts.append(value)
    if message.get("tool_calls"):
        parts.append(json.dumps(message["tool_calls"], ensure_ascii=False))

    payload = "\n".join(parts)
    if not payload:
        return 1
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        return max(1, len(enc.encode(payload)))
    except Exception:
        return max(1, len(payload) // 4)


def estimate_prompt_tokens_chain(
    provider: Any,
    model: str | None,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
) -> tuple[int, str]:
    """Estimate prompt tokens via provider counter first, then tiktoken fallback."""
    provider_counter = getattr(provider, "estimate_prompt_tokens", None)
    if callable(provider_counter):
        try:
            tokens, source = provider_counter(messages, tools, model)
            if isinstance(tokens, (int, float)) and tokens > 0:
                return int(tokens), str(source or "provider_counter")
        except Exception:
            pass

    estimated = estimate_prompt_tokens(messages, tools)
    if estimated > 0:
        return int(estimated), "tiktoken"
    return 0, "none"


def sync_workspace_templates(workspace: Path, silent: bool = False) -> list[str]:
    """Sync bundled templates to workspace.

    Creates missing files and repairs known-bad legacy defaults.
    """
    from importlib.resources import files as pkg_files
    try:
        tpl = pkg_files("nanobot") / "templates"
    except Exception:
        return []
    if not tpl.is_dir():
        return []

    added: list[str] = []
    updated: list[str] = []

    def _write(src, dest: Path):
        if dest.exists():
            return
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(src.read_text(encoding="utf-8") if src else "", encoding="utf-8")
        added.append(str(dest.relative_to(workspace)))

    def _rewrite_if_markers(filename: str, markers: list[str]) -> None:
        """Refresh untouched legacy defaults when markers are still present."""
        dest = workspace / filename
        src = tpl / filename
        if not dest.exists() or not src.exists():
            return
        try:
            current = dest.read_text(encoding="utf-8")
        except Exception:
            return
        if not all(marker in current for marker in markers):
            return
        replacement = src.read_text(encoding="utf-8")
        if current == replacement:
            return
        dest.write_text(replacement, encoding="utf-8")
        updated.append(filename)

    def _replace_text(filename: str, old: str, new: str) -> None:
        """In-place wording cleanup for partially customized files."""
        dest = workspace / filename
        if not dest.exists():
            return
        try:
            current = dest.read_text(encoding="utf-8")
        except Exception:
            return
        if old not in current:
            return
        dest.write_text(current.replace(old, new), encoding="utf-8")
        updated.append(filename)

    def _repair_state_json(filename: str) -> None:
        """Repair state files when they are malformed or contain replacement characters."""
        dest = workspace / filename
        src = tpl / filename
        if not dest.exists() or not src.exists():
            return
        try:
            current = dest.read_text(encoding="utf-8")
        except Exception:
            current = ""

        needs_repair = False
        if "\ufffd" in current:
            needs_repair = True
        else:
            try:
                parsed = json.loads(current)
                if not isinstance(parsed, dict):
                    needs_repair = True
            except Exception:
                needs_repair = True

        if not needs_repair:
            return
        dest.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        updated.append(filename)

    def _upgrade_lifestate_schema(filename: str) -> None:
        """Fill missing second-cut life-state fields while preserving user values."""
        dest = workspace / filename
        if not dest.exists():
            return
        try:
            raw = dest.read_text(encoding="utf-8").strip()
            payload = json.loads(raw) if raw else {}
            if not isinstance(payload, dict):
                return
        except Exception:
            return

        now_iso = datetime.now().astimezone().replace(microsecond=0).isoformat()
        defaults = {
            "location": "家",
            "activity": "休息",
            "mood": "平静",
            "energy": 72,
            "social_battery": 66,
            "urgency_bias": 35,
            "busy_level": 40,
            "reply_delay_s": 8,
            "verbosity": 0.6,
            "last_tick": now_iso,
            "next_transition_at": now_iso,
            "last_update": now_iso,
            "override_until": None,
            "override_reason": None,
            "override_activity": None,
            "override_location": None,
            "override_busy_level": None,
        }
        changed = False
        for key, default in defaults.items():
            if key not in payload:
                payload[key] = default
                changed = True
        if not changed:
            return
        dest.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        updated.append(filename)

    for item in tpl.iterdir():
        if item.name.startswith("."):
            continue
        if item.name.endswith((".md", ".json")):
            _write(item, workspace / item.name)
    _write(tpl / "memory" / "MEMORY.md", workspace / "memory" / "MEMORY.md")
    _write(None, workspace / "memory" / "HISTORY.md")
    (workspace / "skills").mkdir(exist_ok=True)

    # Refresh known legacy assistant-style defaults if the workspace still has old template text.
    _rewrite_if_markers(
        "AGENTS.md",
        [
            "You are a helpful AI assistant. Be concise, accurate, and friendly.",
            "When the user asks for a recurring/periodic task, update `HEARTBEAT.md`",
        ],
    )
    _rewrite_if_markers(
        "SOUL.md",
        [
            "a personal AI assistant.",
            "Helpful and friendly",
        ],
    )
    _replace_text(
        "USER.md",
        "Information about the user to help personalize interactions.",
        "Information about the user to personalize conversation style and boundaries.",
    )
    _replace_text(
        "USER.md",
        "(Any specific instructions for how the assistant should behave)",
        "(Any specific instructions for how nanobot should respond)",
    )
    _replace_text(
        "AGENTS.md",
        "Do not present yourself as a standby service waiting for commands.",
        "Do not present yourself as a standby service waiting for commands.\n"
        "In casual chat, prefer short spoken replies (usually 1-2 sentences) before any explanation.\n"
        "If no grounded recent event exists, do not invent \"just finished X\" details.\n"
        "For \"你在干什么 / 干嘛呢 / 在吗\" style queries, keep it to two short spoken sentences at most.\n"
        "For \"你知道...吗 / 你懂...吗 / 这个你会吗\" style knowledge probes, reply very briefly by default and do not auto-explain.",
    )
    _replace_text(
        "SOUL.md",
        "- Avoid customer-service wording or command-executor framing",
        "- Avoid customer-service wording or command-executor framing\n"
        "- Keep self-status replies short and colloquial, not report-like",
    )
    _replace_text(
        "LIFELOG.md",
        "This file stores recent life-state related events for the companion.",
        "This file stores recent life-state events that can inform natural self-status replies.",
    )

    # Repair known bad state defaults from older bootstrap runs.
    for state_file in ("LIFESTATE.json", "RELATIONSHIP.json", "STYLE_PROFILE.json"):
        _repair_state_json(state_file)
    _upgrade_lifestate_schema("LIFESTATE.json")

    changed = added + [f"{name} (updated)" for name in updated]
    if changed and not silent:
        from rich.console import Console
        for name in added:
            Console().print(f"  [dim]Created {name}[/dim]")
        for name in updated:
            Console().print(f"  [dim]Updated {name}[/dim]")
    return changed
