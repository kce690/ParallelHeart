"""Tests for answer-slot routing, anti-repeat, and meta-self guard."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.loop import AgentLoop
from nanobot.bus.events import InboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMResponse


def _make_loop(tmp_path: Path, llm_response: LLMResponse) -> AgentLoop:
    bus = MessageBus()
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    provider.estimate_prompt_tokens.return_value = (10_000, "test")
    provider.chat_with_retry = AsyncMock(return_value=llm_response)
    loop = AgentLoop(
        bus=bus,
        provider=provider,
        workspace=tmp_path,
        model="test-model",
        context_window_tokens=65_536,
    )
    loop.provider.chat_with_retry = provider.chat_with_retry
    loop.tools.get_definitions = MagicMock(return_value=[])
    return loop


def test_route_answer_slot() -> None:
    assert AgentLoop._route_answer_slot("你在干什么", "state") == "current_activity"
    assert AgentLoop._route_answer_slot("你刚才在干什么", "state") == "previous_activity"
    assert AgentLoop._route_answer_slot("你午饭吃的什么", "state") == "meal"
    assert AgentLoop._route_answer_slot("你心情怎么样", "state") == "mood"
    assert AgentLoop._route_answer_slot("你现在方便吗", "state") == "availability"
    assert AgentLoop._route_answer_slot("你是用什么写的", "task") == "meta_self"


@pytest.mark.asyncio
async def test_meta_self_default_uses_persona_reply_and_skips_llm(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path, LLMResponse(content="我是用Python写的，运行在Windows", tool_calls=[]))
    msg = InboundMessage(channel="qq", sender_id="u1", chat_id="c1", content="你是用什么写的")

    out = await loop._process_message(msg)

    assert out is not None
    text = out.content.lower()
    assert "python" not in text
    assert "windows" not in text
    assert loop.provider.chat_with_retry.await_count == 0


@pytest.mark.asyncio
async def test_meta_self_debug_mode_allows_technical_reply(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path, LLMResponse(content="我是用Python写的，运行在Windows", tool_calls=[]))
    msg = InboundMessage(
        channel="qq",
        sender_id="u1",
        chat_id="c1",
        content="我在调试你 现在用技术模式回答 你是用什么写的",
    )

    out = await loop._process_message(msg)

    assert out is not None
    text = out.content.lower()
    assert ("python" in text) or ("windows" in text)
    assert loop.provider.chat_with_retry.await_count == 1


@pytest.mark.asyncio
async def test_state_slot_anti_repeat_changes_second_reply(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path, LLMResponse(content="在外面待着呢", tool_calls=[]))

    out1 = await loop._process_message(
        InboundMessage(channel="qq", sender_id="u1", chat_id="c1", content="你在干什么")
    )
    out2 = await loop._process_message(
        InboundMessage(channel="qq", sender_id="u1", chat_id="c1", content="你在干什么")
    )

    assert out1 is not None and out2 is not None
    assert out1.content != out2.content


@pytest.mark.asyncio
async def test_greeting_rule_first_does_not_use_state_or_llm(tmp_path: Path) -> None:
    (tmp_path / "LIFESTATE.json").write_text(
        '{"location":"外面","activity":"闲逛","mood":"平静","energy":70}',
        encoding="utf-8",
    )
    loop = _make_loop(tmp_path, LLMResponse(content="嗨 在外面随便逛", tool_calls=[]))
    out = await loop._process_message(
        InboundMessage(channel="qq", sender_id="u1", chat_id="c1", content="hi")
    )

    assert out is not None
    assert "外面" not in out.content
    assert loop.provider.chat_with_retry.await_count == 0


@pytest.mark.asyncio
async def test_meal_slot_prefers_meal_event_and_not_state(tmp_path: Path) -> None:
    (tmp_path / "LIFESTATE.json").write_text(
        '{"location":"外面","activity":"闲逛","mood":"平静","energy":70}',
        encoding="utf-8",
    )
    (tmp_path / "LIFELOG.md").write_text(
        "# Life Log\n- [2026-03-14 17:00] 刚吃饭了\n",
        encoding="utf-8",
    )
    loop = _make_loop(tmp_path, LLMResponse(content="在外面随便吃了点", tool_calls=[]))
    out = await loop._process_message(
        InboundMessage(channel="qq", sender_id="u1", chat_id="c1", content="你午饭吃的什么")
    )

    assert out is not None
    assert "吃" in out.content
    assert "外面" not in out.content
    assert loop.provider.chat_with_retry.await_count == 0


@pytest.mark.asyncio
async def test_ack_followup_does_not_parrot_input(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path, LLMResponse(content="我也是", tool_calls=[]))
    out = await loop._process_message(
        InboundMessage(channel="qq", sender_id="u1", chat_id="c1", content="我也是")
    )

    assert out is not None
    assert out.content != "我也是"
    assert loop.provider.chat_with_retry.await_count == 0
