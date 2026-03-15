"""Agent loop: the core processing engine."""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from contextlib import AsyncExitStack
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from loguru import logger

from nanobot.agent.context import ContextBuilder
from nanobot.agent.memory import MemoryConsolidator
from nanobot.agent.subagent import SubagentManager
from nanobot.agent.tools.cron import CronTool
from nanobot.agent.tools.filesystem import EditFileTool, ListDirTool, ReadFileTool, WriteFileTool
from nanobot.agent.tools.life_state import LifeStateGetTool, LifeStateSetOverrideTool
from nanobot.agent.tools.message import MessageTool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.shell import ExecTool
from nanobot.agent.tools.spawn import SpawnTool
from nanobot.agent.tools.web import WebFetchTool, WebSearchTool
from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMProvider
from nanobot.session.manager import Session, SessionManager

if TYPE_CHECKING:
    from nanobot.companion.life_state.service import LifeStateService
    from nanobot.config.schema import ChannelsConfig, ExecToolConfig, WebSearchConfig
    from nanobot.cron.service import CronService


class AgentLoop:
    """
    The agent loop is the core processing engine.

    It:
    1. Receives messages from the bus
    2. Builds context with history, memory, skills
    3. Calls the LLM
    4. Executes tool calls
    5. Sends responses back
    """

    _TOOL_RESULT_MAX_CHARS = 16_000
    _SHORT_REPLY_MAX_CHARS = 12
    _SHORT_REPLY_END_PUNCT = "。！？!?!.~～…"
    _KNOWLEDGE_PROBE_HINTS = (
        "你知道", "你懂", "你会", "这个你会吗", "这个你知道吗",
        "这个你懂吗", "什么意思吗", "这个什么意思", "这个你知道什么意思吗",
    )
    _EXPLAIN_REQUEST_HINTS = ("讲", "讲讲", "详细讲", "展开说", "解释一下")
    _INTENT_PROBE_HELPER_MARKERS = (
        "我能怎么帮助你",
        "有什么可以帮你",
        "请告诉我你的需求",
        "我在这里为你服务",
        "需要我帮你",
    )
    _INTERNAL_FALLBACK_MARKERS = (
        "i've completed processing but have no response to give",
        "no response",
        "empty response",
        "internal fallback",
        "background task completed",
    )
    _META_SELF_MARKERS = (
        "python", "windows", "runtime", "prompt", "system", "tool", "memory file",
        "模型", "程序", "ai", "机器人", "写的", "实现", "运行", "系统", "提示词", "后端", "框架",
    )
    _DEBUG_MODE_MARKERS = (
        "调试", "debug", "开发者", "技术模式", "实现细节", "系统信息", "认真说", "技术回答", "模型细节",
    )
    _LOW_INFO_DEAD_ACKS = ("是呀", "对呀", "嗯嗯", "哈哈", "我懂")
    _LOW_INFO_INTENT_READING_PATTERNS = (
        r"你这是在",
        r"你是想",
        r"你是不是",
        r"你这是终于",
    )
    _LOW_INFO_MENU_PROBE_PATTERNS = (
        r"你是想[^。！？!?]{0,20}还是[^。！？!?]{0,20}",
        r"你想让我[^。！？!?]{0,20}还是[^。！？!?]{0,20}",
        r"你想我[^。！？!?]{0,20}还是[^。！？!?]{0,20}",
    )

    def __init__(
        self,
        bus: MessageBus,
        provider: LLMProvider,
        workspace: Path,
        model: str | None = None,
        max_iterations: int = 40,
        context_window_tokens: int = 65_536,
        web_search_config: WebSearchConfig | None = None,
        web_proxy: str | None = None,
        exec_config: ExecToolConfig | None = None,
        cron_service: CronService | None = None,
        life_state_service: LifeStateService | None = None,
        restrict_to_workspace: bool = False,
        session_manager: SessionManager | None = None,
        mcp_servers: dict | None = None,
        channels_config: ChannelsConfig | None = None,
    ):
        from nanobot.config.schema import ExecToolConfig, WebSearchConfig

        self.bus = bus
        self.channels_config = channels_config
        self.provider = provider
        self.workspace = workspace
        self.model = model or provider.get_default_model()
        self.max_iterations = max_iterations
        self.context_window_tokens = context_window_tokens
        self.web_search_config = web_search_config or WebSearchConfig()
        self.web_proxy = web_proxy
        self.exec_config = exec_config or ExecToolConfig()
        self.cron_service = cron_service
        self.life_state_service = life_state_service
        self.restrict_to_workspace = restrict_to_workspace

        self.context = ContextBuilder(workspace)
        self.sessions = session_manager or SessionManager(workspace)
        self.tools = ToolRegistry()
        self.subagents = SubagentManager(
            provider=provider,
            workspace=workspace,
            bus=bus,
            model=self.model,
            web_search_config=self.web_search_config,
            web_proxy=web_proxy,
            exec_config=self.exec_config,
            restrict_to_workspace=restrict_to_workspace,
        )

        self._running = False
        self._intent_probe_counter = 0
        self._mcp_servers = mcp_servers or {}
        self._mcp_stack: AsyncExitStack | None = None
        self._mcp_connected = False
        self._mcp_connecting = False
        self._active_tasks: dict[str, list[asyncio.Task]] = {}  # session_key -> tasks
        self._recent_slot_replies: dict[str, list[dict[str, str]]] = {}
        self._processing_lock = asyncio.Lock()
        self.memory_consolidator = MemoryConsolidator(
            workspace=workspace,
            provider=provider,
            model=self.model,
            sessions=self.sessions,
            context_window_tokens=context_window_tokens,
            build_messages=self.context.build_messages,
            get_tool_definitions=self.tools.get_definitions,
        )
        self._register_default_tools()

    def _register_default_tools(self) -> None:
        """Register the default set of tools."""
        allowed_dir = self.workspace if self.restrict_to_workspace else None
        for cls in (ReadFileTool, WriteFileTool, EditFileTool, ListDirTool):
            self.tools.register(cls(workspace=self.workspace, allowed_dir=allowed_dir))
        self.tools.register(ExecTool(
            working_dir=str(self.workspace),
            timeout=self.exec_config.timeout,
            restrict_to_workspace=self.restrict_to_workspace,
            path_append=self.exec_config.path_append,
        ))
        self.tools.register(WebSearchTool(config=self.web_search_config, proxy=self.web_proxy))
        self.tools.register(WebFetchTool(proxy=self.web_proxy))
        self.tools.register(MessageTool(send_callback=self.bus.publish_outbound))
        self.tools.register(SpawnTool(manager=self.subagents))
        if self.cron_service:
            self.tools.register(CronTool(self.cron_service))
        if self.life_state_service:
            self.tools.register(LifeStateGetTool(self.life_state_service))
            self.tools.register(LifeStateSetOverrideTool(self.life_state_service))

    async def _connect_mcp(self) -> None:
        """Connect to configured MCP servers (one-time, lazy)."""
        if self._mcp_connected or self._mcp_connecting or not self._mcp_servers:
            return
        self._mcp_connecting = True
        from nanobot.agent.tools.mcp import connect_mcp_servers
        try:
            self._mcp_stack = AsyncExitStack()
            await self._mcp_stack.__aenter__()
            await connect_mcp_servers(self._mcp_servers, self.tools, self._mcp_stack)
            self._mcp_connected = True
        except BaseException as e:
            logger.error("Failed to connect MCP servers (will retry next message): {}", e)
            if self._mcp_stack:
                try:
                    await self._mcp_stack.aclose()
                except Exception:
                    pass
                self._mcp_stack = None
        finally:
            self._mcp_connecting = False

    def _set_tool_context(self, channel: str, chat_id: str, message_id: str | None = None) -> None:
        """Update context for all tools that need routing info."""
        for name in ("message", "spawn", "cron"):
            if tool := self.tools.get(name):
                if hasattr(tool, "set_context"):
                    tool.set_context(channel, chat_id, *([message_id] if name == "message" else []))

    @staticmethod
    def _strip_think(text: str | None) -> str | None:
        """Remove <think>…</think> blocks that some models embed in content."""
        if not text:
            return None
        return re.sub(r"<think>[\s\S]*?</think>", "", text).strip() or None

    @staticmethod
    def _tool_hint(tool_calls: list) -> str:
        """Format tool calls as concise hint, e.g. 'web_search("query")'."""
        def _fmt(tc):
            args = (tc.arguments[0] if isinstance(tc.arguments, list) else tc.arguments) or {}
            val = next(iter(args.values()), None) if isinstance(args, dict) else None
            if not isinstance(val, str):
                return tc.name
            return f'{tc.name}("{val[:40]}…")' if len(val) > 40 else f'{tc.name}("{val}")'
        return ", ".join(_fmt(tc) for tc in tool_calls)

    @classmethod
    def _normalize_user_text(cls, text: str) -> tuple[str, str]:
        """Return (stripped, flat-no-space) user text after markup cleanup."""
        stripped = cls._strip_weak_input_markup((text or "").strip())
        flat = re.sub(r"\s+", "", stripped)
        return stripped, flat

    @classmethod
    def _classify_input_intensity(cls, text: str) -> str:
        """Classify incoming user text into ping/social/state/task/intent_probe."""
        raw = (text or "").strip()
        stripped, flat = cls._normalize_user_text(raw)
        if not flat:
            return "ping"
        flat_lower = flat.lower()

        length = len(flat)
        question_like = bool(re.search(r"[?？]", raw) or re.search(r"[吗嘛么呢啊呀]$", flat))
        unique_ratio = len(set(flat)) / max(1, length)
        low_density = length <= 4 and (unique_ratio < 0.85 or question_like)
        greeting_signal = bool(re.fullmatch(r"(hi+|hello+|hey+|yo+)", flat_lower))
        explicit_social_signal = bool(re.search(r"(想.{0,2}我|想不想我|想我了吗|陪我|抱抱|亲亲|爱不爱)", flat))
        explicit_task_request = bool(re.search(r"(帮我|请|麻烦|解释|讲讲|详细|展开|分析|总结|翻译|查|算|写|告诉我|怎么做)", flat))
        explicit_why_or_meaning = bool(re.search(r"(定义|原理|公式|步骤|推导|证明|什么意思|是什么|为什么|如何)", flat))
        presence_ping = bool(re.fullmatch(r"(你)?在吗|在不在|在么|在嘛", flat))
        weak_like_social = bool(
            cls._is_weak_input(raw)
            or greeting_signal
            or presence_ping
            or low_density
            or bool(re.search(r"(戳一戳|戳你|戳)", flat))
        )

        state_signal = bool(
            re.search(
                r"(你.{0,3}(在|忙|干|哪|吃|睡|方便|啥)|你在哪|你在干什么|你在干啥|告诉我你在干什么|忙吗|吃饭了没|干嘛呢|干啥呢)",
                flat,
            )
        ) and not presence_ping
        social_signal = explicit_social_signal
        task_signal = (
            cls._is_knowledge_probe(raw)
            or explicit_task_request
            or explicit_why_or_meaning
        )

        # state is higher priority for self-status queries.
        if state_signal:
            return "state"

        if task_signal:
            return "task"
        if social_signal:
            return "social"

        intent_probe_signal = weak_like_social and not task_signal and not state_signal and not social_signal
        if intent_probe_signal:
            return "intent_probe"

        if cls._is_weak_input(raw):
            return "ping"
        if low_density:
            return "ping"
        if question_like and length <= 12 and not explicit_why_or_meaning:
            return "social"
        if length <= 6 and greeting_signal:
            return "intent_probe"
        if length <= 6:
            return "social"
        return "task" if question_like else "social"

    @classmethod
    def _reply_budget(cls, category: str) -> dict[str, int | bool]:
        budgets: dict[str, dict[str, int | bool]] = {
            "ping": {"max_sentences": 1, "max_chars": 6, "min_chars": 1, "allow_followup": False, "allow_explain": False, "allow_detail": False, "strip_punct": True},
            "intent_probe": {"max_sentences": 1, "max_chars": 8, "min_chars": 2, "allow_followup": False, "allow_explain": False, "allow_detail": False, "strip_punct": True},
            "low_info": {"max_sentences": 1, "max_chars": 16, "min_chars": 2, "allow_followup": True, "allow_explain": False, "allow_detail": False, "strip_punct": True},
            "social": {"max_sentences": 2, "max_chars": 16, "min_chars": 2, "allow_followup": True, "allow_explain": False, "allow_detail": False, "strip_punct": True},
            "state": {"max_sentences": 2, "max_chars": 18, "min_chars": 4, "allow_followup": True, "allow_explain": False, "allow_detail": True, "strip_punct": True},
            "task": {"max_sentences": 1, "max_chars": 8, "min_chars": 2, "allow_followup": False, "allow_explain": False, "allow_detail": False, "strip_punct": True},
            "task_debug": {"max_sentences": 3, "max_chars": 160, "min_chars": 8, "allow_followup": True, "allow_explain": True, "allow_detail": True, "strip_punct": False},
        }
        return budgets.get(category, budgets["social"])

    @classmethod
    def _route_answer_slot(cls, text: str, category: str) -> str:
        """Route user query into answer slot for evidence-aware reply policy."""
        raw = (text or "").strip()
        _, flat = cls._normalize_user_text(raw)
        if not flat:
            return "unknown"

        if cls._is_greeting_input(raw):
            return "greeting"
        if cls._is_meta_self_query(raw):
            return "meta_self"

        has_time_back_ref = bool(re.search(r"(刚才|刚刚|之前|方才|前面|上一会|刚那会)", flat))
        asks_activity = bool(re.search(r"(在干|在做|忙什么|做什么|在忙|在哪|在干嘛|在干啥)", flat))
        asks_meal = bool(re.search(r"(吃饭|午饭|晚饭|早饭|早餐|吃了没|吃了吗|吃的什么|饭吃了没)", flat))
        asks_mood = bool(re.search(r"(心情|开心|难受|烦|情绪|状态怎么样|是不是不开心)", flat))
        asks_availability = bool(re.search(r"(方便吗|有空吗|能聊吗|是不是刚忙完|忙完了没|现在忙吗)", flat))

        if asks_meal:
            return "meal"
        if asks_mood:
            return "mood"
        if asks_availability:
            return "availability"
        if has_time_back_ref and asks_activity:
            return "previous_activity"
        if has_time_back_ref and category == "state":
            return "previous_activity"
        if asks_activity or category == "state":
            return "current_activity"
        return "unknown"

    @classmethod
    def _is_greeting_input(cls, text: str) -> bool:
        """Detect short greeting openings that should not read state."""
        _, flat = cls._normalize_user_text(text or "")
        if not flat:
            return False
        flat_lower = flat.lower()
        if re.fullmatch(r"(hi+|hello+|hey+|yo+)", flat_lower):
            return True
        return bool(re.fullmatch(r"(嗨|你好|哈喽|hello|hi|hey|在吗|在不在)", flat, flags=re.IGNORECASE))

    @classmethod
    def _is_low_info_turn(cls, text: str) -> bool:
        """Detect low information density turns using coarse heuristics only."""
        raw = (text or "").strip()
        stripped, flat = cls._normalize_user_text(raw)
        if not flat:
            return True

        compact_len = len(flat)
        if compact_len <= 3:
            return True
        if compact_len >= 16:
            return False

        # Multi-line or structured payloads are usually not low-info.
        if "\n" in raw or "：" in raw or ":" in raw or "```" in raw:
            return False

        payload_units = re.findall(r"[A-Za-z0-9\u4e00-\u9fff]+", stripped)
        unit_count = len(payload_units)
        has_multi_clause = bool(re.search(r"[，,；;。.!?？]", stripped))
        has_question_mark = bool(re.search(r"[?？]", stripped))
        unique_ratio = len(set(flat)) / max(1, compact_len)

        if compact_len >= 8 and unit_count >= 3:
            return False
        if has_multi_clause and compact_len >= 8:
            return False
        if has_question_mark and compact_len >= 7 and unit_count >= 2:
            return False
        if compact_len >= 8 and unique_ratio >= 0.72 and unit_count <= 2:
            return False

        if compact_len <= 8:
            return True
        if compact_len <= 12 and unit_count <= 2:
            return True
        return compact_len <= 14 and unit_count <= 1

    @classmethod
    def _is_meta_self_query(cls, text: str) -> bool:
        """Detect implementation-layer self questions."""
        raw = re.sub(r"\s+", "", (text or ""))
        if not raw:
            return False
        if not any(marker in raw.lower() for marker in cls._META_SELF_MARKERS):
            return False
        return bool(
            re.search(
                r"(你是.*(程序|AI|机器人|模型|写的|做的)|你用什么.*(写|做)|你运行在|你背后|你的模型|你的系统|你是不是.*(程序|AI|机器人))",
                raw,
                flags=re.IGNORECASE,
            )
        )

    @classmethod
    def _allow_meta_technical_reply(cls, text: str) -> bool:
        """Only allow implementation details in explicit debug/developer context."""
        raw = re.sub(r"\s+", "", (text or "")).lower()
        if not raw:
            return False
        return any(marker in raw for marker in cls._DEBUG_MODE_MARKERS)

    def _build_meta_self_floor_reply(self, user_text: str) -> str:
        """Companion-style default response for implementation-layer questions."""
        pool = ("我又不是程序", "你问这个干嘛", "我哪懂这些", "你怎么突然问这个", "你猜")
        base = re.sub(r"\s+", "", self._strip_weak_input_markup(user_text))
        seed = sum(ord(ch) for ch in (base or user_text or "0"))
        idx = (seed + self._intent_probe_counter) % len(pool)
        self._intent_probe_counter = (self._intent_probe_counter + 1) % 10_000
        return pool[idx]

    def _build_greeting_reply(self, user_text: str) -> str:
        """Short greeting reply that does not use life-state details."""
        pool = ("嗨", "在呢", "来啦", "哈喽", "你好呀")
        base = re.sub(r"\s+", "", self._strip_weak_input_markup(user_text))
        seed = sum(ord(ch) for ch in (base or user_text or "0"))
        idx = (seed + self._intent_probe_counter) % len(pool)
        self._intent_probe_counter = (self._intent_probe_counter + 1) % 10_000
        return pool[idx]

    @classmethod
    def _is_status_query(cls, text: str) -> bool:
        return cls._classify_input_intensity(text) == "state"

    @classmethod
    def _is_explain_request(cls, text: str) -> bool:
        msg = (text or "").strip()
        return any(hint in msg for hint in cls._EXPLAIN_REQUEST_HINTS)

    @classmethod
    def _is_knowledge_probe(cls, text: str) -> bool:
        msg = re.sub(r"\s+", "", (text or ""))
        if not msg:
            return False
        if cls._is_explain_request(msg):
            return False
        if "吗" not in msg and "？" not in msg and "?" not in msg:
            return False
        if any(hint in msg for hint in cls._KNOWLEDGE_PROBE_HINTS):
            return True
        return bool(re.search(r"(知道|懂|会|什么意思|是什么|能不能).*([吗么]|[?？])$", msg))

    @staticmethod
    def _strip_weak_input_markup(text: str) -> str:
        """Remove channel markup/tag payload from weak-input candidates."""
        clean = text or ""
        clean = re.sub(r"<[^>\n]{1,256}>", " ", clean)
        clean = re.sub(r"\[CQ:[^\]]+\]", " ", clean)
        clean = re.sub(
            r"\[(?:表情|图片|动画表情|qq表情|emoji|face|sticker|image)\]",
            " ",
            clean,
            flags=re.IGNORECASE,
        )
        clean = clean.replace("\u200b", " ")
        return re.sub(r"\s+", " ", clean).strip()

    @classmethod
    def _is_weak_input(cls, text: str) -> bool:
        """Detect emoji/placeholder/low-semantic input that should not trigger full generation."""
        raw = (text or "").strip()
        if not raw:
            return True
        if re.fullmatch(r"(?:<[^>\n]{1,256}>|\[CQ:[^\]]+\]|\s)+", raw):
            return True

        stripped = cls._strip_weak_input_markup(raw)
        if not stripped:
            return True

        flat = re.sub(r"\s+", "", stripped)
        if not flat:
            return True

        if re.fullmatch(r"[^\w\u4e00-\u9fff]+", flat, flags=re.UNICODE):
            return True
        if len(flat) == 1 and re.fullmatch(r"[A-Za-z0-9\u4e00-\u9fff]", flat):
            return True
        if len(flat) <= 2:
            return True
        if len(flat) <= 4 and len(set(flat)) <= 2:
            return True
        return False

    @classmethod
    def _is_social_ping(cls, text: str) -> bool:
        """Detect short social ping/chitchat turns that should get short replies."""
        return cls._classify_input_intensity(text) == "social"

    @classmethod
    def _is_internal_fallback_output(cls, text: str | None) -> bool:
        """Detect internal placeholders/fallbacks that must not be sent or persisted."""
        if text is None:
            return True
        msg = text.strip()
        if not msg:
            return True
        lowered = msg.lower()
        if any(marker in lowered for marker in cls._INTERNAL_FALLBACK_MARKERS):
            return True
        if lowered in {"n/a", "(empty)", "none"}:
            return True
        return False

    @classmethod
    def _apply_evidence_constraint(
        cls,
        user_text: str,
        reply: str | None,
        *,
        answer_slot: str = "unknown",
        recent_events: list[str] | None = None,
        has_recent_event: bool,
    ) -> str | None:
        """Remove unsupported concrete life details when evidence is weak."""
        if not reply:
            return reply
        events = recent_events or []
        has_meal_event = any(re.search(r"(吃|饭|午饭|晚饭|早饭|早餐)", str(item or "")) for item in events)

        if answer_slot == "meal" and not has_meal_event:
            text = re.sub(r"\s+", " ", reply).strip()
            text = re.sub(r"(刚刚?|刚才)?(在)?(外面|家里|学校)?(随便|简单)?吃了[^，。！？!?]{0,24}", "", text).strip(" ，,。！？!?")
            text = re.sub(r"(午饭|晚饭|早饭|早餐)吃了[^，。！？!?]{0,24}", "", text).strip(" ，,。！？!?")
            return text or "就普通吃的"

        if answer_slot == "previous_activity" and not has_recent_event:
            text = re.sub(r"\s+", " ", reply).strip()
            text = re.sub(r"(刚刚?|刚才|之前)[^，。！？!?]{0,24}", "", text).strip(" ，,。！？!?")
            return text or "刚刚就那样"

        if has_recent_event:
            return reply

        user_flat = re.sub(r"\s+", "", cls._strip_weak_input_markup(user_text))
        user_mentions_detail = bool(re.search(r"(刚|整理|文件|窗外|午饭|忙完|处理|更新|吃完|有空)", user_flat))
        if user_mentions_detail:
            return reply

        text = re.sub(r"\s+", " ", reply).strip()
        text = re.sub(r"(，|,)?刚刚?[^，。！？!?]{0,26}", "", text).strip(" ，,")
        text = re.sub(r"(，|,)?刚才[^，。！？!?]{0,26}", "", text).strip(" ，,")
        text = re.sub(r"(整理文件|看看窗外|处理完一些事情|忙完一些工作|更新了一些内容|正好有空)", "", text)
        text = re.sub(r"\s+", " ", text).strip(" ，,。！？!?")
        return text or None

    @classmethod
    def _short_task_ack(cls, user_text: str) -> str:
        msg = re.sub(r"\s+", "", user_text or "")
        if "会" in msg or "懂" in msg:
            return "会一点"
        if "什么意思" in msg:
            return "知道"
        return "知道啊"

    @staticmethod
    def _is_low_information_state_reply(text: str) -> bool:
        compact = re.sub(r"\s+", "", text or "")
        return compact in {
            "嗯", "在呢", "怎么了", "咋啦", "干嘛", "你说",
            "知道", "知道啊", "会一点",
        }

    def _build_state_floor_reply(self) -> str:
        """Build a coarse safe fallback for current-activity slot."""
        return "在上课呢"

    @staticmethod
    def _extract_latest_event(events: list[str]) -> str | None:
        """Pick latest event summary when available."""
        for item in events:
            text = re.sub(r"\s+", " ", str(item or "")).strip()
            if text:
                return text
        return None

    @staticmethod
    def _extract_latest_meal_event(events: list[str]) -> str | None:
        """Pick latest meal-related event."""
        for item in events:
            text = re.sub(r"\s+", " ", str(item or "")).strip()
            if not text:
                continue
            if re.search(r"(吃|饭|午饭|晚饭|早餐|早饭)", text):
                return text
        return None

    @staticmethod
    def _mood_floor(snapshot: dict[str, Any]) -> str:
        mood = str(snapshot.get("mood") or "")
        if re.search(r"(烦|低落|丧|差)", mood):
            return "有点闷"
        if re.search(r"(开心|不错|好)", mood):
            return "还挺好的"
        if re.search(r"(一般|平静|平稳)", mood):
            return "还行吧"
        return "就还行"

    @staticmethod
    def _availability_floor(snapshot: dict[str, Any]) -> str:
        busy = snapshot.get("busy_level")
        urgency = snapshot.get("urgency_bias")
        activity = str(snapshot.get("activity") or "")
        if isinstance(busy, (int, float)) and busy >= 72:
            return "这会儿还在忙"
        if isinstance(urgency, (int, float)) and urgency >= 75:
            return "还在忙一会儿"
        if re.search(r"(学习|通勤|开会|工作|忙)", activity):
            return "还在忙呢"
        return "这会儿能聊"

    def _build_slot_floor_reply(
        self,
        answer_slot: str,
        user_text: str,
        snapshot: dict[str, Any],
        recent_events: list[str],
    ) -> str | None:
        """Build evidence-grounded floor reply for each answer slot."""
        if answer_slot == "greeting":
            return self._build_greeting_reply(user_text)

        if answer_slot == "meta_self":
            return self._build_meta_self_floor_reply(user_text)

        if answer_slot == "current_activity":
            return self._build_state_floor_reply()

        if answer_slot == "previous_activity":
            latest = self._extract_latest_event(recent_events)
            if latest:
                compact = re.sub(r"\s+", "", latest)
                if len(compact) > 12:
                    return compact[:12]
                return compact
            return "刚刚就那样"

        if answer_slot == "meal":
            meal_event = self._extract_latest_meal_event(recent_events)
            if meal_event:
                compact = re.sub(r"\s+", "", meal_event)
                if len(compact) > 12:
                    return compact[:12]
                return compact
            return "就普通吃的"

        if answer_slot == "mood":
            return self._mood_floor(snapshot)

        if answer_slot == "availability":
            latest = self._extract_latest_event(recent_events)
            if latest and re.search(r"(刚|停|歇|回到|忙完)", latest):
                return "刚停下来"
            return self._availability_floor(snapshot)

        return None

    @staticmethod
    def _normalize_reply_signature(text: str) -> str:
        compact = re.sub(r"[\s，,。！？!?~～…]", "", text or "")
        compact = re.sub(r"(这会儿|现在|刚刚|刚才|就是|有点|一下|呢|呀|啦|吧)$", "", compact)
        return compact[:10]

    def _apply_anti_repeat_guard(
        self,
        *,
        session_key: str,
        answer_slot: str,
        reply: str | None,
        slot_floor_reply: str | None,
    ) -> str | None:
        """Avoid repeated same-slot semantic loops across recent turns."""
        if not reply:
            return reply
        if answer_slot not in {"current_activity", "previous_activity", "meal", "availability", "mood", "greeting"}:
            return reply

        history = self._recent_slot_replies.get(session_key, [])
        if not history:
            return reply
        recent_same_slot = [item for item in reversed(history[-4:]) if item.get("slot") == answer_slot]
        if not recent_same_slot:
            return reply

        sig = self._normalize_reply_signature(reply)
        last_sig = recent_same_slot[0].get("sig", "")
        if not sig or not last_sig:
            return reply
        repeated = sig == last_sig or sig in last_sig or last_sig in sig
        if not repeated:
            return reply

        if slot_floor_reply and self._normalize_reply_signature(slot_floor_reply) != last_sig:
            return slot_floor_reply

        variant = self._anti_repeat_variant(answer_slot)
        if variant and self._normalize_reply_signature(variant) != last_sig:
            return variant

        compact = re.sub(r"\s+", "", reply).strip(self._SHORT_REPLY_END_PUNCT)
        if len(compact) > 8:
            return compact[:8]
        return compact or reply

    def _anti_repeat_variant(self, answer_slot: str) -> str | None:
        """Return short variant when same-slot semantics repeat."""
        pools: dict[str, tuple[str, ...]] = {
            "current_activity": ("这会儿就这样", "刚停下来", "嗯就那样"),
            "previous_activity": ("就刚才那样", "刚那会儿"),
            "meal": ("就普通吃的", "就简单吃了"),
            "availability": ("刚停下来", "现在能聊"),
            "mood": ("就还行", "还行吧"),
            "greeting": ("嗨", "在呢", "来啦"),
        }
        pool = pools.get(answer_slot)
        if not pool:
            return None
        idx = self._intent_probe_counter % len(pool)
        self._intent_probe_counter = (self._intent_probe_counter + 1) % 10_000
        return pool[idx]

    def _record_reply_signature(self, session_key: str, answer_slot: str, reply: str) -> None:
        """Record reply signature for anti-repeat checks."""
        if not reply:
            return
        sig = self._normalize_reply_signature(reply)
        bucket = self._recent_slot_replies.setdefault(session_key, [])
        bucket.append({"slot": answer_slot, "sig": sig, "text": reply})
        if len(bucket) > 12:
            del bucket[:-12]

    @classmethod
    def _is_assistant_offer_style(cls, text: str) -> bool:
        compact = re.sub(r"\s+", "", text or "")
        if not compact:
            return False
        if any(marker in compact for marker in cls._INTENT_PROBE_HELPER_MARKERS):
            return True
        return bool(re.search(r"(帮助你|帮到你|你的需求|为你服务|等待你的指令|待命)", compact))

    def _relationship_probe_tier(self) -> str:
        """Return low/mid/high tier from relationship cues."""
        cues = self.context.get_relationship_cues()
        stage = str(cues.get("stage") or "")
        score = cues.get("intimacy")
        if score is None:
            score = cues.get("trust")

        level: float | None = None
        if isinstance(score, (int, float)):
            level = float(score)
            if 0.0 <= level <= 1.5:
                level *= 100.0

        stage_high = bool(re.search(r"(恋人|情侣|伴侣|暧昧|亲密|close|partner)", stage, flags=re.IGNORECASE))
        stage_low = bool(re.search(r"(陌生|初识|客套|formal|distant)", stage, flags=re.IGNORECASE))
        if level is None:
            if stage_high:
                return "high"
            if stage_low:
                return "low"
            return "mid"
        if level >= 70:
            return "high"
        if level < 35:
            return "low"
        return "mid"

    def _build_intent_probe_reply(self, user_text: str) -> str:
        """Build a short relationship-aware probe reply for unknown intent."""
        tier = self._relationship_probe_tier()
        pools: dict[str, tuple[str, ...]] = {
            "low": ("怎么了", "有事吗", "干嘛呀", "咋啦"),
            "mid": ("怎么啦", "干嘛呢", "找我呀", "你想说啥"),
            "high": ("怎么啦宝宝", "干嘛呀宝宝", "是不是想我了", "想我啦"),
        }
        pool = pools.get(tier, pools["mid"])
        base = re.sub(r"\s+", "", self._strip_weak_input_markup(user_text))
        seed = sum(ord(ch) for ch in (base or user_text or "0"))
        idx = (seed + self._intent_probe_counter) % len(pool)
        self._intent_probe_counter = (self._intent_probe_counter + 1) % 10_000
        return pool[idx]

    @staticmethod
    def _short_context_fragment(text: str | None, *, max_chars: int = 12) -> str | None:
        raw = re.sub(r"\s+", " ", text or "").strip()
        if not raw:
            return None
        raw = raw.split("\n", 1)[0].strip()
        raw = re.sub(r"[。！？!?~～…]+$", "", raw).strip()
        if not raw:
            return None
        compact = re.sub(r"\s+", "", raw)
        if len(compact) > max_chars:
            compact = compact[:max_chars]
        return compact or None

    def _recent_low_info_context_hint(self, session: Session, *, max_user_turns: int = 3) -> str | None:
        """Pick a short anchor from the latest non-low-info user turns."""
        seen_users = 0
        for msg in reversed(session.messages):
            if msg.get("role") != "user":
                continue
            content = str(msg.get("content") or "").strip()
            if not content or content.startswith("/"):
                continue
            seen_users += 1
            if seen_users > max_user_turns:
                break
            if self._is_low_info_turn(content):
                continue
            hint = self._short_context_fragment(content, max_chars=12)
            if hint and not self._is_weak_low_info_anchor(hint):
                return hint
        return None

    def _recent_low_info_dialogue(self, session: Session, *, limit: int = 3) -> list[str]:
        """Collect short recent dialogue snippets for low-info strategy prompting."""
        snippets: list[str] = []
        for msg in reversed(session.messages):
            role = msg.get("role")
            if role not in {"user", "assistant"}:
                continue
            content = str(msg.get("content") or "").strip()
            if not content:
                continue
            if role == "assistant" and self._is_internal_fallback_output(content):
                continue
            short = self._short_context_fragment(content, max_chars=16)
            if not short:
                continue
            label = "user" if role == "user" else "bot"
            snippets.append(f"{label}:{short}")
            if len(snippets) >= limit:
                break
        snippets.reverse()
        return snippets

    def _count_recent_low_info_streak(
        self,
        session: Session,
        current_text: str,
        *,
        max_user_turns: int = 4,
        max_minutes: int = 20,
    ) -> int:
        """Count consecutive low-info user turns in a short recent window."""
        user_turns: list[str] = [current_text]
        cutoff = datetime.now() - timedelta(minutes=max(1, max_minutes))
        for msg in reversed(session.messages):
            if msg.get("role") != "user":
                continue
            ts = msg.get("timestamp")
            if isinstance(ts, str):
                try:
                    if datetime.fromisoformat(ts) < cutoff:
                        break
                except ValueError:
                    pass
            content = str(msg.get("content") or "").strip()
            if not content:
                continue
            user_turns.append(content)
            if len(user_turns) >= max(1, max_user_turns):
                break

        streak = 0
        for turn in user_turns:
            category = self._classify_input_intensity(turn)
            if self._route_answer_slot(turn, category) != "unknown":
                break
            if not self._is_low_info_turn(turn):
                break
            streak += 1
        return max(1, streak)

    def _build_low_info_probe_reply(
        self,
        user_text: str,
        *,
        streak: int = 1,
        context_hint: str | None = None,
        intimacy_tier: str | None = None,
    ) -> str:
        """Build one short natural probe for sparse low-info turns."""
        tier = intimacy_tier or self._relationship_probe_tier()
        streak_level = max(1, int(streak))
        anchor = self._short_context_fragment(context_hint, max_chars=10)
        if anchor:
            if streak_level <= 1:
                return f"「{anchor}」这块想继续吗"
            if streak_level == 2:
                return f"刚说到「{anchor}」，补一句呗"
            if tier == "high":
                return f"还绕着「{anchor}」呢，别吊我胃口了"
            return f"「{anchor}」这句再说完整点"

        if streak_level <= 1:
            pool = ("怎么啦", "你想说啥", "咋了")
        elif streak_level == 2:
            pool = ("卡住了就多说一句", "这句有点悬", "到底想说啥呀")
        elif tier == "high":
            pool = ("你再这样我真急了", "别吊我胃口了", "行吧你到底想说啥")
        else:
            pool = ("说完整点嘛", "我有点接不住了", "别让我一直猜呀")

        base = re.sub(r"\s+", "", self._strip_weak_input_markup(user_text))
        seed = sum(ord(ch) for ch in (base or user_text or "0"))
        idx = (seed + self._intent_probe_counter) % len(pool)
        self._intent_probe_counter = (self._intent_probe_counter + 1) % 10_000
        return pool[idx]

    def _low_info_strategy_system_prompt(
        self,
        *,
        low_info_streak: int,
        intimacy_tier: str,
        context_hint: str | None = None,
        recent_dialogue: list[str] | None = None,
    ) -> str:
        """System-side strategy for sparse low-information user turns."""
        lines = [
            "Low-info turn strategy:",
            "- User turn is sparse/ambiguous; silently infer likely intent from conversation flow.",
            "- Keep inferred intent hidden in final text; do not state motives as fact.",
            "- Do not output explicit mind-reading phrasing like `你是想...` / `你是不是...` / `你这是在...` unless evidence is very strong.",
            "- Output one short natural probe line only, as nanobot (not a service assistant).",
            "- Prefer probing over dead agreement or self-narration anchors.",
            "- Avoid service offers like `我可以帮你` / `有什么可以帮你`.",
            "- Do not use menu-style option prompts (for example `你是想A，还是B`).",
            "- Contextual probing is allowed only when the context anchor is concrete and useful.",
            "- Repetition should strengthen tone only: light probe -> puzzled probe -> mild impatience.",
            "- Do not narrate repeated user behavior (for example `连发问号` / `一直戳我`) unless it is explicitly grounded by recent messages.",
            f"- Recent low-info streak: {max(1, low_info_streak)} (1=light probe, 2=clear confusion, 3+=mild impatience).",
            f"- Relationship intimacy tier: {intimacy_tier}.",
            "- Aggressive wording is allowed only when intimacy is high and streak >= 3.",
            "- Do not output reasoning.",
        ]
        anchor = self._short_context_fragment(context_hint, max_chars=12)
        if anchor:
            lines.append(f"- Context anchor candidate: {anchor}")
        if recent_dialogue:
            lines.append("- Recent turns: " + " | ".join(recent_dialogue[:3]))
        return "\n".join(lines)

    @classmethod
    def _is_dead_agreement_reply(cls, text: str | None) -> bool:
        compact = re.sub(r"[\s，,。！？!?~～…]", "", text or "")
        if not compact:
            return False
        if compact in cls._LOW_INFO_DEAD_ACKS:
            return True
        return compact in {"是啊", "对啊", "嗯", "哈"}

    @classmethod
    def _is_self_narration_reply(cls, text: str | None) -> bool:
        compact = re.sub(r"\s+", "", text or "")
        if not compact:
            return False
        return bool(
            re.search(
                r"(我(在|就)?(这儿|这里|这边).{0,8}(陪你聊|陪你聊天|等你)|刚在.{0,8}陪你聊|在这儿等你)",
                compact,
            )
        )

    @classmethod
    def _is_aggressive_reply(cls, text: str | None) -> bool:
        compact = re.sub(r"\s+", "", text or "")
        if not compact:
            return False
        return bool(re.search(r"(滚|闭嘴|烦死|别烦|有病|神经|欠骂|欠揍|怼你|懒得理)", compact))

    @classmethod
    def _is_intent_reading_reply(cls, text: str | None) -> bool:
        compact = re.sub(r"\s+", "", text or "")
        if not compact:
            return False
        return any(re.search(pattern, compact) for pattern in cls._LOW_INFO_INTENT_READING_PATTERNS)

    @classmethod
    def _is_menu_style_probe_reply(cls, text: str | None) -> bool:
        compact = re.sub(r"\s+", "", text or "")
        if not compact or "还是" not in compact:
            return False
        return any(re.search(pattern, compact) for pattern in cls._LOW_INFO_MENU_PROBE_PATTERNS)

    @classmethod
    def _is_behavior_narration_reply(cls, text: str | None) -> bool:
        compact = re.sub(r"\s+", "", text or "")
        if not compact:
            return False
        return bool(re.search(r"(连发问号|一直戳我|又在这样|在看你连发问号|老是这样|总是这样)", compact))

    @classmethod
    def _is_weak_low_info_anchor(cls, text: str | None) -> bool:
        compact = re.sub(r"\s+", "", text or "")
        if not compact:
            return True
        if len(compact) <= 2:
            return True
        return bool(re.fullmatch(r"(聊天|聊聊|在吗|你好|哈喽|继续聊|说说看|说说)", compact))

    @classmethod
    def _is_grounded_behavior_narration(
        cls,
        user_text: str,
        reply: str,
        *,
        streak: int,
        intimacy_tier: str,
    ) -> bool:
        """Allow behavior narration only when explicit evidence exists in current turn."""
        compact_user = re.sub(r"\s+", "", user_text or "")
        compact_reply = re.sub(r"\s+", "", reply or "")
        if not compact_user or not compact_reply:
            return False
        if streak < 2:
            return False
        if intimacy_tier not in {"mid", "high"}:
            return False
        if "问号" in compact_reply and len(re.findall(r"[?？]", user_text or "")) >= 2:
            return True
        if "戳" in compact_reply and "戳" in compact_user and intimacy_tier == "high":
            return True
        return False

    def _apply_low_info_output_guard(
        self,
        user_text: str,
        reply: str | None,
        *,
        streak: int = 1,
        context_hint: str | None = None,
        intimacy_tier: str | None = None,
    ) -> str | None:
        """Last-resort low-info output guard for quality and tone gating."""
        if not reply:
            return reply
        tier = intimacy_tier or self._relationship_probe_tier()
        fallback = self._build_low_info_probe_reply(
            user_text,
            streak=streak,
            context_hint=context_hint,
            intimacy_tier=tier,
        )
        if (
            self._is_assistant_offer_style(reply)
            or self._is_dead_agreement_reply(reply)
            or self._is_self_narration_reply(reply)
            or self._is_intent_reading_reply(reply)
            or self._is_menu_style_probe_reply(reply)
        ):
            return fallback
        if self._is_behavior_narration_reply(reply):
            if not self._is_grounded_behavior_narration(
                user_text,
                reply,
                streak=streak,
                intimacy_tier=tier,
            ):
                return fallback
        if self._is_aggressive_reply(reply):
            if not (tier == "high" and streak >= 3):
                return fallback
        return reply

    @classmethod
    def _enforce_reply_budget(
        cls,
        category: str,
        user_text: str,
        reply: str | None,
        *,
        answer_slot: str = "unknown",
        recent_events: list[str] | None = None,
        has_recent_event: bool,
        state_floor_reply: str | None = None,
        intent_probe_floor_reply: str | None = None,
        low_info_floor_reply: str | None = None,
    ) -> str | None:
        """Apply unified response budget and behavior limits by category."""
        if not reply:
            return reply

        text = re.sub(r"\s+", " ", reply).strip()
        if not text:
            return None

        if category == "task" and answer_slot != "meta_self" and not cls._is_explain_request(user_text):
            return cls._short_task_ack(user_text)
        if category == "intent_probe" and cls._is_assistant_offer_style(text):
            text = intent_probe_floor_reply or "怎么了"
        if category == "low_info" and (
            cls._is_assistant_offer_style(text)
            or cls._is_dead_agreement_reply(text)
            or cls._is_self_narration_reply(text)
            or cls._is_intent_reading_reply(text)
            or cls._is_menu_style_probe_reply(text)
        ):
            text = low_info_floor_reply or "怎么啦"

        if category == "state":
            text = cls._shape_status_reply(
                user_text,
                text,
                answer_slot=answer_slot,
                has_recent_event=has_recent_event,
            ) or text

        text = cls._apply_evidence_constraint(
            user_text,
            text,
            answer_slot=answer_slot,
            recent_events=recent_events,
            has_recent_event=has_recent_event,
        ) or text

        budget = cls._reply_budget(category)
        parts = [p.strip(" ，,。！？!?") for p in re.split(r"[。！？!?]", text) if p.strip()]
        if not parts:
            return None
        parts = parts[: int(budget["max_sentences"])]

        if not bool(budget["allow_followup"]) and len(parts) > 1:
            parts = parts[:1]

        joined = " ".join(parts).strip()
        if re.search(r"(请问|安排|帮助|服务|计划|有什么安排)", joined):
            joined = re.sub(r"(请问|安排|帮助|服务|计划|有什么安排)", "", joined).strip(" ，,")

        max_chars = int(budget["max_chars"])
        if len(re.sub(r"\s+", "", joined)) > max_chars:
            compact = re.sub(r"\s+", "", joined)
            joined = compact[:max_chars].strip()

        min_chars = int(budget.get("min_chars", 1))
        compact_len = len(re.sub(r"\s+", "", joined))
        if category == "state" and answer_slot in {"current_activity", "availability"} and (
            compact_len < min_chars or cls._is_low_information_state_reply(joined)
        ):
            joined = (state_floor_reply or "这会儿歇着呢").strip()
        elif category == "low_info" and (
            compact_len < min_chars
            or cls._is_assistant_offer_style(joined)
            or cls._is_dead_agreement_reply(joined)
            or cls._is_self_narration_reply(joined)
            or cls._is_intent_reading_reply(joined)
            or cls._is_menu_style_probe_reply(joined)
        ):
            joined = (low_info_floor_reply or "怎么啦").strip()
        elif category == "intent_probe" and (
            compact_len < min_chars or cls._is_assistant_offer_style(joined)
        ):
            joined = (intent_probe_floor_reply or "怎么了").strip()
        elif compact_len < min_chars:
            if category == "social":
                joined = "在呢"
            elif category == "task":
                joined = cls._short_task_ack(user_text)
            elif category == "task_debug":
                joined = "说细一点我再认真讲"
            else:
                joined = "嗯"

        if bool(budget["strip_punct"]):
            joined = joined.rstrip(cls._SHORT_REPLY_END_PUNCT).strip()
        if not joined:
            if category in {"ping", "social"}:
                return "在呢"
            if category == "low_info":
                return low_info_floor_reply or "怎么啦"
            if category == "intent_probe":
                return intent_probe_floor_reply or "怎么了"
            if category == "state":
                return "在家休息呢"
            if category == "task_debug":
                return "你具体想查哪块"
            return cls._short_task_ack(user_text)
        return joined

    @classmethod
    def _strip_short_reply_terminal_punct(cls, user_text: str, reply: str | None) -> str | None:
        """For short casual replies, remove sentence-final punctuation."""
        if not reply:
            return reply
        text = reply.strip()
        if not text:
            return reply

        forced = (
            cls._is_status_query(user_text)
            or cls._is_knowledge_probe(user_text)
            or cls._is_weak_input(user_text)
            or cls._is_social_ping(user_text)
            or cls._classify_input_intensity(user_text) == "intent_probe"
        )

        if forced and re.search(r"[。！？!?]", text):
            pieces = [p.strip() for p in re.split(r"[。！？!?]+", text) if p.strip()]
            if 1 <= len(pieces) <= 2 and all(len(re.sub(r"\s+", "", p)) <= 10 for p in pieces):
                return " ".join(pieces)

        core = text.rstrip(cls._SHORT_REPLY_END_PUNCT).strip()
        if core == text:
            return reply
        if not core or "\n" in core:
            return core or reply
        if re.search(r"[。！？!?]", core):
            return reply

        compact_len = len(re.sub(r"\s+", "", core))
        if forced or compact_len <= cls._SHORT_REPLY_MAX_CHARS:
            return core
        return reply

    @staticmethod
    def _shape_status_reply(
        user_text: str,
        reply: str | None,
        *,
        answer_slot: str,
        has_recent_event: bool,
    ) -> str | None:
        """Constrain casual self-status replies to short, spoken, non-report style."""
        if not reply or not AgentLoop._is_status_query(user_text):
            return reply
        if answer_slot not in {"current_activity", "availability"}:
            return reply

        text = re.sub(r"\s+", " ", reply).strip()
        if not text:
            return reply

        parts = [p.strip(" ，,。！？!?") for p in re.split(r"[。！？!?]", text) if p.strip()]
        if not parts:
            return reply

        first = parts[0]
        if not has_recent_event:
            first = re.sub(r"(，|,)?刚[^，。！？!?]{0,24}", "", first).strip(" ，,")
            first = re.sub(r"(，|,)?刚刚[^，。！？!?]{0,24}", "", first).strip(" ，,")
            first = re.sub(r"(，|,)?刚才[^，。！？!?]{0,24}", "", first).strip(" ，,")

        first = re.sub(r"^我(现在|正在)", "", first).strip()
        first = re.sub(r"(在这里)?等着?你", "", first).strip(" ，,")
        first = re.sub(r"待命", "", first).strip(" ，,")
        if len(first) > 18:
            first = first[:18].rstrip(" ，,")
        if not first:
            first = "在呢"

        second = ""
        if len(parts) > 1:
            candidate = parts[1]
            if not has_recent_event:
                candidate = re.sub(r"(，|,)?刚[^，。！？!?]{0,24}", "", candidate).strip(" ，,")
            if not re.search(r"(安排|帮助|帮忙|请问|需要|服务|计划|问题)", candidate):
                if len(candidate) <= 8:
                    second = candidate
                else:
                    m = re.search(r"(你呢|咋啦|怎么了|怎么啦|干嘛|在吗)", candidate)
                    if m:
                        second = m.group(1)

        return f"{first}。{second + '。' if second else ''}"

    @classmethod
    def _slot_reply_matches(cls, answer_slot: str, reply: str | None) -> bool:
        """Check whether reply is semantically aligned with slot intent."""
        if not reply:
            return False
        text = re.sub(r"\s+", "", reply)
        if not text:
            return False
        if answer_slot == "current_activity":
            return bool(re.search(r"(在|忙|歇|外面|在家|路上|休息|学习|通勤)", text))
        if answer_slot == "previous_activity":
            return bool(re.search(r"(刚|刚才|之前|方才|刚刚)", text))
        if answer_slot == "meal":
            return bool(re.search(r"(吃|饭|没吃|吃过|吃了)", text))
        if answer_slot == "mood":
            return bool(re.search(r"(心情|开心|烦|还行|平静|一般|低落|不错)", text))
        if answer_slot == "availability":
            return bool(re.search(r"(忙|有空|方便|能聊|刚停|这会儿)", text))
        if answer_slot == "meta_self":
            return True
        return True

    @classmethod
    def _enforce_slot_answer(
        cls,
        answer_slot: str,
        reply: str | None,
        *,
        slot_floor_reply: str | None,
        allow_meta_technical: bool,
    ) -> str | None:
        """Keep slot answer aligned to evidence source and persona boundaries."""
        if answer_slot == "meta_self" and not allow_meta_technical:
            return slot_floor_reply or "你问这个干嘛"
        if not reply:
            return slot_floor_reply or reply
        if answer_slot in {"unknown", "meta_self"}:
            return reply
        if cls._slot_reply_matches(answer_slot, reply):
            return reply
        return slot_floor_reply or reply

    @staticmethod
    def _is_rule_first_slot(answer_slot: str) -> bool:
        """Whether this slot should bypass LLM and answer from rule/evidence first."""
        return answer_slot in {
            "greeting",
            "current_activity",
            "previous_activity",
            "meal",
            "mood",
            "availability",
            "meta_self",
        }

    async def _run_agent_loop(
        self,
        initial_messages: list[dict],
        on_progress: Callable[..., Awaitable[None]] | None = None,
    ) -> tuple[str | None, list[str], list[dict]]:
        """Run the agent iteration loop."""
        messages = initial_messages
        iteration = 0
        final_content = None
        tools_used: list[str] = []

        while iteration < self.max_iterations:
            iteration += 1

            tool_defs = self.tools.get_definitions()

            response = await self.provider.chat_with_retry(
                messages=messages,
                tools=tool_defs,
                model=self.model,
            )

            if response.has_tool_calls:
                if on_progress:
                    thought = self._strip_think(response.content)
                    if thought:
                        await on_progress(thought)
                    await on_progress(self._tool_hint(response.tool_calls), tool_hint=True)

                tool_call_dicts = [
                    tc.to_openai_tool_call()
                    for tc in response.tool_calls
                ]
                messages = self.context.add_assistant_message(
                    messages, response.content, tool_call_dicts,
                    reasoning_content=response.reasoning_content,
                    thinking_blocks=response.thinking_blocks,
                )

                for tool_call in response.tool_calls:
                    tools_used.append(tool_call.name)
                    args_str = json.dumps(tool_call.arguments, ensure_ascii=False)
                    logger.info("Tool call: {}({})", tool_call.name, args_str[:200])
                    result = await self.tools.execute(tool_call.name, tool_call.arguments)
                    messages = self.context.add_tool_result(
                        messages, tool_call.id, tool_call.name, result
                    )
            else:
                clean = self._strip_think(response.content)
                # Don't persist error responses to session history — they can
                # poison the context and cause permanent 400 loops (#1303).
                if response.finish_reason == "error":
                    logger.error("LLM returned error: {}", (clean or "")[:200])
                    final_content = clean or "Sorry, I encountered an error calling the AI model."
                    break
                messages = self.context.add_assistant_message(
                    messages, clean, reasoning_content=response.reasoning_content,
                    thinking_blocks=response.thinking_blocks,
                )
                final_content = clean
                break

        if final_content is None and iteration >= self.max_iterations:
            logger.warning("Max iterations ({}) reached", self.max_iterations)

        return final_content, tools_used, messages

    async def run(self) -> None:
        """Run the agent loop, dispatching messages as tasks to stay responsive to /stop."""
        self._running = True
        await self._connect_mcp()
        logger.info("Agent loop started")

        while self._running:
            try:
                msg = await asyncio.wait_for(self.bus.consume_inbound(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            cmd = msg.content.strip().lower()
            if cmd == "/stop":
                await self._handle_stop(msg)
            elif cmd == "/restart":
                await self._handle_restart(msg)
            else:
                task = asyncio.create_task(self._dispatch(msg))
                self._active_tasks.setdefault(msg.session_key, []).append(task)
                task.add_done_callback(lambda t, k=msg.session_key: self._active_tasks.get(k, []) and self._active_tasks[k].remove(t) if t in self._active_tasks.get(k, []) else None)

    async def _handle_stop(self, msg: InboundMessage) -> None:
        """Cancel all active tasks and subagents for the session."""
        tasks = self._active_tasks.pop(msg.session_key, [])
        cancelled = sum(1 for t in tasks if not t.done() and t.cancel())
        for t in tasks:
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
        sub_cancelled = await self.subagents.cancel_by_session(msg.session_key)
        total = cancelled + sub_cancelled
        content = f"Stopped {total} task(s)." if total else "No active task to stop."
        await self.bus.publish_outbound(OutboundMessage(
            channel=msg.channel, chat_id=msg.chat_id, content=content,
        ))

    async def _handle_restart(self, msg: InboundMessage) -> None:
        """Restart the process in-place via os.execv."""
        await self.bus.publish_outbound(OutboundMessage(
            channel=msg.channel, chat_id=msg.chat_id, content="Restarting...",
        ))

        async def _do_restart():
            await asyncio.sleep(1)
            # Use -m nanobot instead of sys.argv[0] for Windows compatibility
            # (sys.argv[0] may be just "nanobot" without full path on Windows)
            os.execv(sys.executable, [sys.executable, "-m", "nanobot"] + sys.argv[1:])

        asyncio.create_task(_do_restart())

    async def _dispatch(self, msg: InboundMessage) -> None:
        """Process a message under the global lock."""
        async with self._processing_lock:
            try:
                response = await self._process_message(msg)
                if response is not None:
                    await self.bus.publish_outbound(response)
                elif msg.channel == "cli":
                    await self.bus.publish_outbound(OutboundMessage(
                        channel=msg.channel, chat_id=msg.chat_id,
                        content="", metadata=msg.metadata or {},
                    ))
            except asyncio.CancelledError:
                logger.info("Task cancelled for session {}", msg.session_key)
                raise
            except Exception:
                logger.exception("Error processing message for session {}", msg.session_key)
                await self.bus.publish_outbound(OutboundMessage(
                    channel=msg.channel, chat_id=msg.chat_id,
                    content="Sorry, I encountered an error.",
                ))

    async def close_mcp(self) -> None:
        """Close MCP connections."""
        if self._mcp_stack:
            try:
                await self._mcp_stack.aclose()
            except (RuntimeError, BaseExceptionGroup):
                pass  # MCP SDK cancel scope cleanup is noisy but harmless
            self._mcp_stack = None

    def stop(self) -> None:
        """Stop the agent loop."""
        self._running = False
        logger.info("Agent loop stopping")

    async def _process_message(
        self,
        msg: InboundMessage,
        session_key: str | None = None,
        on_progress: Callable[[str], Awaitable[None]] | None = None,
    ) -> OutboundMessage | None:
        """Process a single inbound message and return the response."""
        # System messages: parse origin from chat_id ("channel:chat_id")
        if msg.channel == "system":
            channel, chat_id = (msg.chat_id.split(":", 1) if ":" in msg.chat_id
                                else ("cli", msg.chat_id))
            logger.info("Processing system message from {}", msg.sender_id)
            key = f"{channel}:{chat_id}"
            session = self.sessions.get_or_create(key)
            await self.memory_consolidator.maybe_consolidate_by_tokens(session)
            self._set_tool_context(channel, chat_id, msg.metadata.get("message_id"))
            history = session.get_history(max_messages=0, include_assistant_text=False)
            messages = self.context.build_messages(
                history=history,
                current_message=msg.content, channel=channel, chat_id=chat_id,
            )
            final_content, _, all_msgs = await self._run_agent_loop(messages)
            self._save_turn(session, all_msgs, 1 + len(history))
            self.sessions.save(session)
            await self.memory_consolidator.maybe_consolidate_by_tokens(session)
            if self._is_internal_fallback_output(final_content):
                return None
            return OutboundMessage(channel=channel, chat_id=chat_id, content=final_content)

        preview = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
        logger.info("Processing message from {}:{}: {}", msg.channel, msg.sender_id, preview)

        key = session_key or msg.session_key
        session = self.sessions.get_or_create(key)

        # Slash commands
        cmd = msg.content.strip().lower()
        if cmd == "/new":
            try:
                if not await self.memory_consolidator.archive_unconsolidated(session):
                    return OutboundMessage(
                        channel=msg.channel,
                        chat_id=msg.chat_id,
                        content="Memory archival failed, session not cleared. Please try again.",
                    )
            except Exception:
                logger.exception("/new archival failed for {}", session.key)
                return OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content="Memory archival failed, session not cleared. Please try again.",
                )

            session.clear()
            self.sessions.save(session)
            self.sessions.invalidate(session.key)
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                  content="New session started.")
        if cmd == "/help":
            lines = [
                "🐈 nanobot commands:",
                "/new — Start a new conversation",
                "/stop — Stop the current task",
                "/restart — Restart the bot",
                "/help — Show available commands",
            ]
            return OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id, content="\n".join(lines),
            )
        await self.memory_consolidator.maybe_consolidate_by_tokens(session)

        category = self._classify_input_intensity(msg.content)
        answer_slot = self._route_answer_slot(msg.content, category)
        low_info_strategy = bool(
            self._is_low_info_turn(msg.content)
            and answer_slot == "unknown"
            and category in {"ping", "intent_probe", "social"}
        )
        low_info_streak = 1
        low_info_context_hint: str | None = None
        low_info_recent_dialogue: list[str] = []
        low_info_intimacy_tier = "mid"
        if low_info_strategy:
            low_info_streak = self._count_recent_low_info_streak(session, msg.content)
            low_info_context_hint = self._recent_low_info_context_hint(session)
            low_info_recent_dialogue = self._recent_low_info_dialogue(session, limit=3)
            low_info_intimacy_tier = self._relationship_probe_tier()
        allow_meta_technical = answer_slot == "meta_self" and self._allow_meta_technical_reply(msg.content)
        effective_category = (
            "low_info"
            if low_info_strategy
            else ("task_debug" if answer_slot == "meta_self" and allow_meta_technical else category)
        )
        state_snapshot = self.context.get_life_state_snapshot()
        recent_events = self.context.get_recent_life_events(limit=5)
        has_recent_event = bool(recent_events)
        state_floor_reply = self._build_state_floor_reply() if category == "state" else None
        intent_probe_floor_reply = self._build_intent_probe_reply(msg.content) if category == "intent_probe" else None
        low_info_floor_reply = (
            self._build_low_info_probe_reply(
                msg.content,
                streak=low_info_streak,
                context_hint=low_info_context_hint,
                intimacy_tier=low_info_intimacy_tier,
            )
            if low_info_strategy
            else None
        )
        slot_floor_reply = self._build_slot_floor_reply(
            answer_slot,
            msg.content,
            state_snapshot,
            recent_events,
        )

        # Slot-first, rule-first: explicit slots bypass free LLM generation by default.
        if self._is_rule_first_slot(answer_slot) and not (answer_slot == "meta_self" and allow_meta_technical):
            final_content = self._enforce_slot_answer(
                answer_slot,
                slot_floor_reply,
                slot_floor_reply=slot_floor_reply,
                allow_meta_technical=allow_meta_technical,
            )
            final_content = self._apply_anti_repeat_guard(
                session_key=key,
                answer_slot=answer_slot,
                reply=final_content,
                slot_floor_reply=slot_floor_reply,
            ) or final_content
            final_content = self._strip_short_reply_terminal_punct(msg.content, final_content) or final_content
            if not final_content:
                return None
            from datetime import datetime

            session.messages.append({"role": "user", "content": msg.content, "timestamp": datetime.now().isoformat()})
            session.messages.append({"role": "assistant", "content": final_content, "timestamp": datetime.now().isoformat()})
            session.updated_at = datetime.now()
            self._record_reply_signature(key, answer_slot, final_content)
            self.sessions.save(session)
            await self.memory_consolidator.maybe_consolidate_by_tokens(session)
            preview = final_content[:120] + "..." if len(final_content) > 120 else final_content
            logger.info("Rule-first response to {}:{}: {}", msg.channel, msg.sender_id, preview)
            return OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id, content=final_content,
                metadata=msg.metadata or {},
            )

        self._set_tool_context(msg.channel, msg.chat_id, msg.metadata.get("message_id"))
        if message_tool := self.tools.get("message"):
            if isinstance(message_tool, MessageTool):
                message_tool.start_turn()

        history = session.get_history(max_messages=0, include_assistant_text=False)
        initial_messages = self.context.build_messages(
            history=history,
            current_message=msg.content,
            media=msg.media if msg.media else None,
            channel=msg.channel, chat_id=msg.chat_id,
        )
        if low_info_strategy:
            initial_messages.insert(
                -1,
                {
                    "role": "system",
                    "content": self._low_info_strategy_system_prompt(
                        low_info_streak=low_info_streak,
                        intimacy_tier=low_info_intimacy_tier,
                        context_hint=low_info_context_hint,
                        recent_dialogue=low_info_recent_dialogue,
                    ),
                },
            )

        async def _bus_progress(content: str, *, tool_hint: bool = False) -> None:
            meta = dict(msg.metadata or {})
            meta["_progress"] = True
            meta["_tool_hint"] = tool_hint
            await self.bus.publish_outbound(OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id, content=content, metadata=meta,
            ))

        final_content, _, all_msgs = await self._run_agent_loop(
            initial_messages, on_progress=on_progress or _bus_progress,
        )

        final_content = self._enforce_reply_budget(
            effective_category,
            msg.content,
            final_content,
            answer_slot=answer_slot,
            recent_events=recent_events,
            has_recent_event=has_recent_event,
            state_floor_reply=state_floor_reply,
            intent_probe_floor_reply=intent_probe_floor_reply,
            low_info_floor_reply=low_info_floor_reply,
        ) or final_content

        final_content = self._enforce_slot_answer(
            answer_slot,
            final_content,
            slot_floor_reply=slot_floor_reply,
            allow_meta_technical=allow_meta_technical,
        ) or final_content

        final_content = self._apply_anti_repeat_guard(
            session_key=key,
            answer_slot=answer_slot,
            reply=final_content,
            slot_floor_reply=slot_floor_reply,
        ) or final_content

        final_content = self._strip_short_reply_terminal_punct(
            msg.content,
            final_content,
        ) or final_content
        if low_info_strategy:
            final_content = self._apply_low_info_output_guard(
                msg.content,
                final_content,
                streak=low_info_streak,
                context_hint=low_info_context_hint,
                intimacy_tier=low_info_intimacy_tier,
            ) or final_content

        if self._is_internal_fallback_output(final_content):
            logger.warning("Suppressing internal fallback output for {}:{}", msg.channel, msg.sender_id)
            self._save_turn(session, all_msgs, 1 + len(history))
            safe_short = None
            if low_info_strategy:
                safe_short = (low_info_floor_reply or "怎么啦").strip()
            elif category == "intent_probe":
                safe_short = (intent_probe_floor_reply or "怎么了").strip()
            elif category in {"ping", "social"}:
                safe_short = "在呢"
            elif answer_slot == "meta_self" and not allow_meta_technical:
                safe_short = slot_floor_reply or "你问这个干嘛"
            if safe_short:
                from datetime import datetime
                session.messages.append(
                    {"role": "assistant", "content": safe_short, "timestamp": datetime.now().isoformat()}
                )
                self._record_reply_signature(key, answer_slot, safe_short)
            self.sessions.save(session)
            await self.memory_consolidator.maybe_consolidate_by_tokens(session)
            if safe_short:
                return OutboundMessage(
                    channel=msg.channel, chat_id=msg.chat_id, content=safe_short,
                    metadata=msg.metadata or {},
                )
            return None

        self._replace_last_assistant_content(all_msgs, final_content)
        self._save_turn(session, all_msgs, 1 + len(history))
        self._record_reply_signature(key, answer_slot, final_content)
        self.sessions.save(session)
        await self.memory_consolidator.maybe_consolidate_by_tokens(session)

        if (mt := self.tools.get("message")) and isinstance(mt, MessageTool) and mt._sent_in_turn:
            return None

        preview = final_content[:120] + "..." if len(final_content) > 120 else final_content
        logger.info("Response to {}:{}: {}", msg.channel, msg.sender_id, preview)
        return OutboundMessage(
            channel=msg.channel, chat_id=msg.chat_id, content=final_content,
            metadata=msg.metadata or {},
        )

    @staticmethod
    def _replace_last_assistant_content(messages: list[dict], content: str) -> None:
        """Replace the final assistant text in transient message list with guarded output."""
        for i in range(len(messages) - 1, -1, -1):
            item = messages[i]
            if item.get("role") != "assistant":
                continue
            if item.get("tool_calls"):
                continue
            item["content"] = content
            break

    def _save_turn(self, session: Session, messages: list[dict], skip: int) -> None:
        """Save new-turn messages into session, truncating large tool results."""
        from datetime import datetime
        for m in messages[skip:]:
            entry = dict(m)
            role, content = entry.get("role"), entry.get("content")
            if role == "assistant" and not content and not entry.get("tool_calls"):
                continue  # skip empty assistant messages — they poison session context
            if role == "tool":
                continue  # never persist tool traces to session history/memory
            if role == "assistant":
                if entry.get("tool_calls"):
                    continue  # skip assistant tool-call scaffolding
                if self._is_internal_fallback_output(content if isinstance(content, str) else None):
                    continue  # skip internal fallback/status placeholders
            elif role == "user":
                if isinstance(content, str) and content.startswith(ContextBuilder._RUNTIME_CONTEXT_TAG):
                    # Strip the runtime-context prefix, keep only the user text.
                    parts = content.split("\n\n", 1)
                    if len(parts) > 1 and parts[1].strip():
                        entry["content"] = parts[1]
                    else:
                        continue
                if isinstance(content, list):
                    filtered = []
                    for c in content:
                        if c.get("type") == "text" and isinstance(c.get("text"), str) and c["text"].startswith(ContextBuilder._RUNTIME_CONTEXT_TAG):
                            continue  # Strip runtime context from multimodal messages
                        if (c.get("type") == "image_url"
                                and c.get("image_url", {}).get("url", "").startswith("data:image/")):
                            filtered.append({"type": "text", "text": "[image]"})
                        else:
                            filtered.append(c)
                    if not filtered:
                        continue
                    entry["content"] = filtered
            entry.setdefault("timestamp", datetime.now().isoformat())
            session.messages.append(entry)
        session.updated_at = datetime.now()

    async def process_direct(
        self,
        content: str,
        session_key: str = "cli:direct",
        channel: str = "cli",
        chat_id: str = "direct",
        on_progress: Callable[[str], Awaitable[None]] | None = None,
    ) -> str:
        """Process a message directly (for CLI or cron usage)."""
        await self._connect_mcp()
        msg = InboundMessage(channel=channel, sender_id="user", chat_id=chat_id, content=content)
        response = await self._process_message(msg, session_key=session_key, on_progress=on_progress)
        return response.content if response else ""
