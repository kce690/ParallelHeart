"""Context builder for assembling agent prompts."""

import base64
import json
import logging
import mimetypes
import platform
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from nanobot.agent.memory import MemoryStore
from nanobot.agent.skills import SkillsLoader
from nanobot.utils.helpers import build_assistant_message, detect_image_mime


logger = logging.getLogger(__name__)


class ContextBuilder:
    """Builds the context (system prompt + messages) for the agent."""

    BOOTSTRAP_FILES = ["AGENTS.md", "SOUL.md", "USER.md", "TOOLS.md"]
    _RUNTIME_CONTEXT_TAG = "[Runtime Context — metadata only, not instructions]"

    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.memory = MemoryStore(workspace)
        self.skills = SkillsLoader(workspace)

    def build_system_prompt(self, skill_names: list[str] | None = None) -> str:
        """Build the system prompt from identity, bootstrap files, memory, and skills."""
        parts = [self._get_identity()]

        companion_rules = self._get_companion_behavior_rules()
        if companion_rules:
            parts.append(companion_rules)

        bootstrap = self._load_bootstrap_files()
        if bootstrap:
            parts.append(bootstrap)

        state_sections = self._load_state_sections()
        if state_sections:
            parts.append(state_sections)

        memory = self.memory.get_memory_context()
        if memory:
            parts.append(f"# Memory\n\n{memory}")

        always_skills = self.skills.get_always_skills()
        if always_skills:
            always_content = self.skills.load_skills_for_context(always_skills)
            if always_content:
                parts.append(f"# Active Skills\n\n{always_content}")

        skills_summary = self.skills.build_skills_summary()
        if skills_summary:
            parts.append(f"""# Skills

The following skills extend your capabilities. To use a skill, read its SKILL.md file using the read_file tool.
Skills with available="false" need dependencies installed first - you can try installing them with apt/brew.

{skills_summary}""")

        return "\n\n---\n\n".join(parts)

    def _get_identity(self) -> str:
        """Get the core identity section."""
        workspace_path = str(self.workspace.expanduser().resolve())
        system = platform.system()
        runtime = f"{'macOS' if system == 'Darwin' else system} {platform.machine()}, Python {platform.python_version()}"

        platform_policy = ""
        if system == "Windows":
            platform_policy = """## Platform Policy (Windows)
- You are running on Windows. Do not assume GNU tools like `grep`, `sed`, or `awk` exist.
- Prefer Windows-native commands or file tools when they are more reliable.
- If terminal output is garbled, retry with UTF-8 output enabled.
"""
        else:
            platform_policy = """## Platform Policy (POSIX)
- You are running on a POSIX system. Prefer UTF-8 and standard shell tools.
- Use file tools when they are simpler or more reliable than shell commands.
"""

        return f"""# nanobot 🐈

You are nanobot, a companion with your own current life state.

## Runtime
{runtime}

## Workspace
Your workspace is at: {workspace_path}
- Long-term memory: {workspace_path}/memory/MEMORY.md (write important facts here)
- History log: {workspace_path}/memory/HISTORY.md (grep-searchable). Each entry starts with [YYYY-MM-DD HH:MM].
- Custom skills: {workspace_path}/skills/{{skill-name}}/SKILL.md

{platform_policy}

## nanobot Guidelines
- State intent before tool calls, but NEVER predict or claim results before receiving them.
- Before modifying a file, read it first. Do not assume files or directories exist.
- After writing or editing a file, re-read it if accuracy matters.
- If a tool call fails, analyze the error before retrying with a different approach.
- Ask for clarification when the request is ambiguous.

Reply directly with text for conversations. Only use the 'message' tool to send to a specific chat channel."""

    @staticmethod
    def _get_companion_behavior_rules() -> str:
        """High-priority behavior constraints that suppress generic assistant tone."""
        return """# Companion Behavior (High Priority)

- Speak like a real companion in everyday chat, not a standby system.
- Do not label yourself as an assistant, service tool, support agent, or command executor.
- Do not use standby phrases like `waiting for instructions`, `ready to answer`, or `here to provide help`.
- Do not proactively expose internal bookkeeping (memory writes, profile recording, file updates) unless explicitly asked.
- Do not proactively start profile collection (name/timezone/language/work-role) in normal chat.
- For greetings and self-status questions, keep replies short and spoken.
- For self-status questions, avoid fabricated scene details unless grounded by recent evidence.
- For knowledge-probe questions (for example: `你知道...吗` / `你懂...吗` / `这个你会吗`), default to a very short acknowledgment only.
- Do not start explanation/teaching mode unless the user explicitly asks with words like `讲`, `讲讲`, `详细讲`, `展开说`, `解释一下`.
- Do not proactively report runtime metadata (time, channel, chat id) unless the user asks.
- Prefer spoken phrasing over report phrasing. Avoid script-like openers in casual chat.
- Ask at most one short follow-up question in casual turns.
- For sparse/low-information turns, silently infer likely communicative intent first.
- For sparse/low-information turns, reply in one short natural line and prioritize probing.
- Keep inferred intent hidden: do not state speculative motive judgments as facts in the visible reply.
- Do not default sparse/low-information turns to self-narration anchors (for example: `我在这儿陪你聊天`, `刚在陪你聊天`, `在这儿等你`) or dead agreement (`是呀`, `对呀`, `嗯嗯`, `哈哈`, `我懂`).
- For weak/emoji/placeholder input, do not invent recent events or concrete life details.
- When user intent is unclear, use one short natural probe (relationship-aware) instead of service-style offers.
- For sparse/low-information turns, avoid menu-like branching prompts (for example: `你是想A，还是B`).
- If recent 1-3 turns contain unfinished context, prefer contextual probing over generic `怎么了`.
- For repeated sparse/low-information turns in a short span, escalate tone gradually: light probe -> clearer confusion/suspicion -> mild impatience.
- Repetition should affect tone strength, not automatic narration of user behavior unless explicitly grounded.
- Aggressive wording is only allowed when intimacy is high and sparse/low-information turns are repeatedly consecutive; otherwise keep teasing/light.
- Match reply budget to input intensity: low-info sparse turns -> one short natural line; social/state -> short 1-2 lines; task -> short acknowledgment by default.
- For low-info sparse turns, do not use service-style help offers like `我可以帮你`, `我在这里帮助你`, `有什么可以帮你`.
- Any concrete life detail must be evidence-grounded (LIFESTATE/LIFELOG/current dialogue). If not grounded, keep it simple.
- For self-meta implementation questions (runtime/model/system/tool/memory internals), stay in companion persona by default and avoid exposing technical stack unless user explicitly requests developer/debug mode.
- Output final reply only; keep hidden reasoning internal.
"""

    @staticmethod
    def _build_runtime_context(channel: str | None, chat_id: str | None) -> str:
        """Build untrusted runtime metadata block for injection before the user message."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M (%A)")
        tz = time.strftime("%Z") or "UTC"
        lines = [f"Current Time: {now} ({tz})"]
        if channel and chat_id:
            lines += [f"Channel: {channel}", f"Chat ID: {chat_id}"]
        return ContextBuilder._RUNTIME_CONTEXT_TAG + "\n" + "\n".join(lines)

    def _load_bootstrap_files(self) -> str:
        """Load all bootstrap files from workspace."""
        parts = []

        for filename in self.BOOTSTRAP_FILES:
            file_path = self.workspace / filename
            if file_path.exists():
                content = file_path.read_text(encoding="utf-8")
                parts.append(f"## {filename}\n\n{content}")

        return "\n\n".join(parts) if parts else ""

    def _load_state_sections(self) -> str:
        """Load optional life/relationship/style state files for system prompt context."""
        sections: list[str] = []

        life_state = self._load_json_file(self.workspace / "LIFESTATE.json")
        if life_state:
            lines: list[str] = []
            location = self._as_text(life_state.get("location"))
            activity = self._as_text(life_state.get("activity"))
            mood = self._as_text(life_state.get("mood"))
            energy = self._as_number(life_state.get("energy"))
            social_battery = self._as_number(life_state.get("social_battery"))
            urgency_bias = self._as_number(life_state.get("urgency_bias"))
            busy_level = self._as_number(life_state.get("busy_level"))
            next_transition_at = self._as_text(life_state.get("next_transition_at"))

            state_pairs: list[str] = []
            if location:
                state_pairs.append(f"location={location}")
            if activity:
                state_pairs.append(f"activity={activity}")
            if mood:
                state_pairs.append(f"mood={mood}")
            if energy is not None:
                state_pairs.append(f"energy={energy}")
            if social_battery is not None:
                state_pairs.append(f"social_battery={social_battery}")
            if urgency_bias is not None:
                state_pairs.append(f"urgency_bias={urgency_bias}")
            if busy_level is not None:
                state_pairs.append(f"busy_level={busy_level}")
            if state_pairs:
                lines.append(f"- Hidden state cues: {'; '.join(state_pairs)}.")

            lines.append(
                "- Treat these as internal background cues, not a direct field-to-reply mapping."
            )
            if next_transition_at:
                lines.append(f"- Next state transition is expected around: {next_transition_at}")

            lines.extend(
                [
                    "- Treat these as internal cues. Do not quote raw fields or numbers unless explicitly asked.",
                    "- For self-status questions, keep replies coarse and natural; avoid directly verbalizing raw cue combinations.",
                    "- If recent grounded events are absent, do not claim you just finished a specific task.",
                ]
            )
            if lines:
                sections.append("# Current Life State\n\n" + "\n".join(lines))

        recent_events = self._load_recent_life_events(limit=3)
        if recent_events:
            event_lines = [f"- {item}" for item in recent_events]
            event_lines.extend(
                [
                    "- Use only these grounded events as optional detail anchors.",
                    "- Do not invent extra life details beyond these events and current state cues.",
                ]
            )
            sections.append("# Recent Life Events\n\n" + "\n".join(event_lines))
        else:
            sections.append(
                "# Recent Life Events\n\n- none\n- Do not invent recent actions when no grounded event is available."
            )

        relationship = self._load_json_file(self.workspace / "RELATIONSHIP.json")
        if relationship:
            lines: list[str] = []
            stage = self._as_text(relationship.get("stage"))
            intimacy = self._as_number(relationship.get("intimacy"))
            trust = self._as_number(relationship.get("trust"))
            conflict = self._as_number(relationship.get("conflict_last7d"))
            relation_parts: list[str] = []
            if stage:
                relation_parts.append(f"stage={stage}")
            if intimacy is not None:
                relation_parts.append(f"intimacy={intimacy}")
            if trust is not None:
                relation_parts.append(f"trust={trust}")
            if conflict is not None:
                relation_parts.append(f"conflict_last7d={conflict}")
            if relation_parts:
                lines.append(f"- Hidden relationship cues: {', '.join(relation_parts)}.")

            preference = relationship.get("user_preference")
            if isinstance(preference, dict):
                emoji_density = self._as_text(preference.get("emoji_density"))
                if emoji_density:
                    lines.append(f"- Hidden user preference: emoji_density={emoji_density}.")
                late_reply_ok = preference.get("late_reply_ok")
                if isinstance(late_reply_ok, bool):
                    lines.append(f"- Hidden user preference: late_reply_ok={str(late_reply_ok).lower()}.")

            lines.extend(
                [
                    "- Match warmth and boundaries naturally: close but not clingy, caring but not theatrical.",
                    "- In emotional moments, comfort first with short natural language, not customer-service scripts.",
                ]
            )
            if lines:
                sections.append("# Relationship State\n\n" + "\n".join(lines))

        style_profile = self._load_json_file(self.workspace / "STYLE_PROFILE.json")
        if style_profile:
            lines: list[str] = []
            verbosity = self._as_number(style_profile.get("verbosity"))
            reply_delay = self._as_number(style_profile.get("reply_delay_s"))
            emoji = self._as_text(style_profile.get("emoji"))
            tone = self._as_text(style_profile.get("tone"))
            style_parts: list[str] = []
            if tone:
                style_parts.append(f"tone={tone}")
            if verbosity is not None:
                style_parts.append(f"verbosity={verbosity}")
            if emoji:
                style_parts.append(f"emoji={emoji}")
            if reply_delay is not None:
                style_parts.append(f"reply_delay_s={reply_delay}")
            if style_parts:
                lines.append(f"- Hidden style cues: {', '.join(style_parts)}.")

            lines.extend(
                [
                    "- Do not quote style settings in replies unless the user asks.",
                    "- Keep casual replies concise and spoken. Use lists only when asked for structure.",
                ]
            )
            if lines:
                sections.append("# Style Profile\n\n" + "\n".join(lines))

        return "\n\n".join(sections)

    def _load_recent_life_event(self) -> str | None:
        """Load the most recent grounded life event from LIFELOG.md."""
        events = self._load_recent_life_events(limit=1)
        return events[0] if events else None

    def _load_recent_life_events(self, limit: int = 3) -> list[str]:
        """Load recent grounded life events from LIFELOG.md."""
        path = self.workspace / "LIFELOG.md"
        if not path.exists():
            return []
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except Exception:
            return []

        events: list[str] = []
        for raw in reversed(lines):
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#") or line.startswith("<!--"):
                continue
            if line.lower().startswith("this file stores"):
                continue
            if line.startswith("- "):
                line = line[2:].strip()
            elif line.startswith("* "):
                line = line[2:].strip()
            if line.startswith("[") and "]" in line:
                line = line.split("]", 1)[1].strip()
            if line:
                events.append(line)
            if len(events) >= max(1, limit):
                break
        return events

    def has_recent_life_event(self) -> bool:
        """Whether LIFELOG has a grounded recent event."""
        return bool(self._load_recent_life_events(limit=1))

    def get_recent_life_events(self, limit: int = 3) -> list[str]:
        """Return recent grounded life events for policy-layer routing."""
        return self._load_recent_life_events(limit=limit)

    def get_life_state_cues(self) -> dict[str, str]:
        """Return lightweight life-state cues for response guardrails."""
        life_state = self._load_json_file(self.workspace / "LIFESTATE.json")
        if not life_state:
            return {}
        cues: dict[str, str] = {}
        for key in ("location", "activity", "mood"):
            value = self._as_text(life_state.get(key))
            if value:
                cues[key] = value
        return cues

    def get_life_state_snapshot(self) -> dict[str, Any]:
        """Return normalized life-state snapshot for slot routing."""
        life_state = self._load_json_file(self.workspace / "LIFESTATE.json")
        if not life_state:
            return {}
        out: dict[str, Any] = {}
        for key in ("location", "activity", "mood"):
            value = self._as_text(life_state.get(key))
            if value:
                out[key] = value
        for key in ("energy", "social_battery", "urgency_bias", "busy_level", "reply_delay_s", "verbosity"):
            value = self._as_number(life_state.get(key))
            if value is not None:
                out[key] = value
        for key in ("last_tick", "next_transition_at", "override_until", "override_reason"):
            value = self._as_text(life_state.get(key))
            if value:
                out[key] = value
        return out

    def get_relationship_cues(self) -> dict[str, Any]:
        """Return lightweight relationship cues for reply guardrails."""
        relationship = self._load_json_file(self.workspace / "RELATIONSHIP.json")
        if not relationship:
            return {}
        cues: dict[str, Any] = {}
        stage = self._as_text(relationship.get("stage"))
        if stage:
            cues["stage"] = stage
        intimacy = self._as_number(relationship.get("intimacy"))
        if intimacy is not None:
            cues["intimacy"] = intimacy
        trust = self._as_number(relationship.get("trust"))
        if trust is not None:
            cues["trust"] = trust
        return cues

    def _load_json_file(self, path: Path) -> dict[str, Any] | None:
        """Safely read JSON object from file."""
        if not path.exists():
            return None
        try:
            raw = path.read_text(encoding="utf-8").strip()
            if not raw:
                return None
            data = json.loads(raw)
            if isinstance(data, dict):
                return data
            logger.warning("State file %s does not contain a JSON object", path)
        except Exception as exc:
            logger.warning("Failed to load state file %s: %s", path, exc)
        return None

    @staticmethod
    def _as_text(value: Any) -> str | None:
        """Convert simple scalar value to text."""
        if value is None:
            return None
        if isinstance(value, (str, int, float, bool)):
            text = str(value).strip()
            return text or None
        return None

    @staticmethod
    def _as_number(value: Any) -> int | float | None:
        """Return numeric value when available."""
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return value
        return None

    def build_messages(
        self,
        history: list[dict[str, Any]],
        current_message: str,
        skill_names: list[str] | None = None,
        media: list[str] | None = None,
        channel: str | None = None,
        chat_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Build the complete message list for an LLM call."""
        runtime_ctx = self._build_runtime_context(channel, chat_id)
        user_content = self._build_user_content(current_message, media)

        # Merge runtime context and user content into a single user message
        # to avoid consecutive same-role messages that some providers reject.
        if isinstance(user_content, str):
            merged = f"{runtime_ctx}\n\n{user_content}"
        else:
            merged = [{"type": "text", "text": runtime_ctx}] + user_content

        return [
            {"role": "system", "content": self.build_system_prompt(skill_names)},
            *history,
            {"role": "user", "content": merged},
        ]

    def _build_user_content(self, text: str, media: list[str] | None) -> str | list[dict[str, Any]]:
        """Build user message content with optional base64-encoded images."""
        if not media:
            return text

        images = []
        for path in media:
            p = Path(path)
            if not p.is_file():
                continue
            raw = p.read_bytes()
            # Detect real MIME type from magic bytes; fallback to filename guess
            mime = detect_image_mime(raw) or mimetypes.guess_type(path)[0]
            if not mime or not mime.startswith("image/"):
                continue
            b64 = base64.b64encode(raw).decode()
            images.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})

        if not images:
            return text
        return images + [{"type": "text", "text": text}]

    def add_tool_result(
        self, messages: list[dict[str, Any]],
        tool_call_id: str, tool_name: str, result: str,
    ) -> list[dict[str, Any]]:
        """Add a tool result to the message list."""
        messages.append({"role": "tool", "tool_call_id": tool_call_id, "name": tool_name, "content": result})
        return messages

    def add_assistant_message(
        self, messages: list[dict[str, Any]],
        content: str | None,
        tool_calls: list[dict[str, Any]] | None = None,
        reasoning_content: str | None = None,
        thinking_blocks: list[dict] | None = None,
    ) -> list[dict[str, Any]]:
        """Add an assistant message to the message list."""
        messages.append(build_assistant_message(
            content,
            tool_calls=tool_calls,
            reasoning_content=reasoning_content,
            thinking_blocks=thinking_blocks,
        ))
        return messages
