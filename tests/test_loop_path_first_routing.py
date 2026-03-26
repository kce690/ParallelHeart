from datetime import datetime, timedelta
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


@pytest.mark.asyncio
async def test_identity_path_does_not_retrieve_or_use_life_detail_memory(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path, LLMResponse(content="我刚在外面吃饭", tool_calls=[]))
    loop.life_state_service = MagicMock()
    loop.life_state_service.retrieve_memory_evidence = AsyncMock(
        return_value={
            "recall_level": "detail",
            "evidence": [
                {"id": "m1", "recall_level": "detail", "text": "刚在外面吃饭", "source_kind": "generated_detail", "coarse_type": "meal"}
            ],
            "prompt_block": "DETAIL evidence:\n- [m1|generated_detail] 刚在外面吃饭",
        }
    )

    out = await loop._process_message(
        InboundMessage(channel="qq", sender_id="u1", chat_id="c1", content="你有没有性别")
    )

    assert out is not None
    assert out.content == "我是女的"
    loop.life_state_service.retrieve_memory_evidence.assert_not_awaited()


@pytest.mark.asyncio
async def test_body_profile_path_ignores_generated_detail_memory(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path, LLMResponse(content="我刚在外面看消息", tool_calls=[]))
    loop.life_state_service = MagicMock()
    loop.life_state_service.retrieve_memory_evidence = AsyncMock()

    out = await loop._process_message(
        InboundMessage(channel="qq", sender_id="u1", chat_id="c1", content="你多高啊")
    )

    assert out is not None
    assert "看消息" not in out.content
    loop.life_state_service.retrieve_memory_evidence.assert_not_awaited()


@pytest.mark.asyncio
async def test_meal_path_filters_memory_to_meal_only(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path, LLMResponse(content="刚在外面吃了碗面", tool_calls=[]))
    loop.life_state_service = MagicMock()
    loop.life_state_service.retrieve_memory_evidence = AsyncMock(
        return_value={
            "recall_level": "detail",
            "evidence": [
                {"id": "a1", "recall_level": "detail", "text": "在外面看消息", "gist_summary": "在外面看消息", "source_kind": "generated_detail", "coarse_type": "activity"},
                {"id": "m1", "recall_level": "detail", "text": "吃了碗面", "gist_summary": "吃了点东西", "source_kind": "generated_detail", "coarse_type": "meal"},
            ],
            "prompt_block": "ignored",
        }
    )
    loop.life_state_service.reinforce_memory_evidence = AsyncMock(return_value=0)
    loop.life_state_service.record_generated_detail = AsyncMock(return_value=None)

    out = await loop._process_message(
        InboundMessage(channel="qq", sender_id="u1", chat_id="c1", content="吃的什么")
    )

    assert out is not None
    call = loop.provider.chat_with_retry.await_args
    messages = call.kwargs["messages"]
    memory_prompts = [m["content"] for m in messages if m.get("role") == "system" and "Memory retrieval policy:" in str(m.get("content") or "")]
    assert memory_prompts
    assert "吃了碗面" in memory_prompts[-1]
    assert "看消息" not in memory_prompts[-1]


@pytest.mark.asyncio
async def test_low_info_unknown_does_not_let_memory_detail_take_over(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path, LLMResponse(content="刚在外面吃饭", tool_calls=[]))
    loop.life_state_service = MagicMock()
    loop.life_state_service.retrieve_memory_evidence = AsyncMock()

    out = await loop._process_message(
        InboundMessage(channel="qq", sender_id="u1", chat_id="c1", content="111")
    )

    assert out is not None
    loop.life_state_service.retrieve_memory_evidence.assert_not_awaited()


@pytest.mark.asyncio
async def test_previous_activity_without_memory_stays_on_path_and_generates_floor(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path, LLMResponse(content="internal fallback", tool_calls=[]))

    out = await loop._process_message(
        InboundMessage(channel="qq", sender_id="u1", chat_id="c1", content="刚才你在干什么")
    )

    assert out is not None
    assert out.content in {"刚在忙点自己的事", "刚在弄点东西"}
