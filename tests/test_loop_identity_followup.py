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


@pytest.mark.parametrize(
    "text",
    [
        "你是男的还是女的",
        "你是男生还是女生",
        "你有性别吗",
        "你有没有生理性别",
        "你是男的吗",
        "你是女的吗",
        "你有没有身体",
        "你是真人吗",
        "你是不是AI",
        "你到底是什么",
    ],
)
def test_route_answer_slot_detects_identity_questions(text: str) -> None:
    assert AgentLoop._route_answer_slot(text, "social") == "identity"


@pytest.mark.parametrize(
    "text",
    ["所以到底是什么", "那到底呢", "所以呢", "那你到底算什么", "所以到底是不是"],
)
def test_identity_followup_detector_matches_short_followups(text: str) -> None:
    assert AgentLoop._is_identity_followup_query(text) is True


@pytest.mark.asyncio
async def test_identity_question_prefers_core_fact_reply(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path, LLMResponse(content="知道啊", tool_calls=[]))

    out = await loop._process_message(
        InboundMessage(channel="qq", sender_id="u1", chat_id="c1", content="你到底是什么")
    )

    assert out is not None
    assert out.content == "我是女生"


@pytest.mark.asyncio
async def test_short_followup_inherits_previous_identity_path(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path, LLMResponse(content="知道啊", tool_calls=[]))

    first = await loop._process_message(
        InboundMessage(channel="qq", sender_id="u1", chat_id="c1", content="你是男的还是女的")
    )
    second = await loop._process_message(
        InboundMessage(channel="qq", sender_id="u1", chat_id="c1", content="所以到底是什么")
    )

    assert first is not None
    assert second is not None
    assert first.content == "我是女的"
    assert second.content == "我是女生"


@pytest.mark.asyncio
async def test_short_followup_can_inherit_after_meta_self_turn(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path, LLMResponse(content="我是Python写的", tool_calls=[]))

    first = await loop._process_message(
        InboundMessage(channel="qq", sender_id="u1", chat_id="c1", content="你是用什么写的")
    )
    second = await loop._process_message(
        InboundMessage(channel="qq", sender_id="u1", chat_id="c1", content="那你到底算什么")
    )

    assert first is not None
    assert second is not None
    assert first.content == "我是AI助手，不展开实现细节"
    assert second.content == "我是女生"
