"""Tests for stable IDENTITY_PROFILE generation and rule-first identity replies."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.loop import AgentLoop
from nanobot.bus.events import InboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMResponse


def _make_loop(tmp_path: Path, *, life_state_service=None) -> AgentLoop:
    bus = MessageBus()
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    provider.estimate_prompt_tokens.return_value = (10_000, "test")
    provider.chat_with_retry = AsyncMock(return_value=LLMResponse(content="fallback", tool_calls=[]))
    loop = AgentLoop(
        bus=bus,
        provider=provider,
        workspace=tmp_path,
        model="test-model",
        context_window_tokens=65_536,
        life_state_service=life_state_service,
    )
    loop.provider.chat_with_retry = provider.chat_with_retry
    loop.tools.get_definitions = MagicMock(return_value=[])
    return loop


def _load_identity_profile(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.asyncio
async def test_identity_profile_first_generate_then_reuse_same_values(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path)
    identity_path = tmp_path / "IDENTITY_PROFILE.json"
    assert not identity_path.exists()

    out1 = await loop._process_message(
        InboundMessage(channel="qq", sender_id="u1", chat_id="c1", content="你是男的还是女的")
    )
    assert out1 is not None
    assert out1.content == "我是女的"
    assert identity_path.exists()

    profile1 = _load_identity_profile(identity_path)
    assert profile1["gender"] == "female"
    assert profile1["gender_label"] == "女生"
    assert profile1["identity_style"] == "女性人设"
    assert profile1["source"] == "persona_profile"
    assert profile1["updated_at"]

    out2 = await loop._process_message(
        InboundMessage(channel="qq", sender_id="u1", chat_id="c1", content="你是女生吗")
    )
    assert out2 is not None
    assert out2.content == "我是女生"
    assert _load_identity_profile(identity_path) == profile1
    assert loop.provider.chat_with_retry.await_count == 0


@pytest.mark.asyncio
async def test_identity_followups_stay_on_stable_female_persona(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path)

    first = await loop._process_message(
        InboundMessage(channel="qq", sender_id="u1", chat_id="c1", content="你到底是什么性别")
    )
    second = await loop._process_message(
        InboundMessage(channel="qq", sender_id="u1", chat_id="c1", content="所以到底是什么")
    )
    third = await loop._process_message(
        InboundMessage(channel="qq", sender_id="u1", chat_id="c1", content="那你算女的吗")
    )
    fourth = await loop._process_message(
        InboundMessage(channel="qq", sender_id="u1", chat_id="c1", content="那你是女孩子吗")
    )

    assert first is not None and first.content == "我是女的"
    assert second is not None and second.content == "我是女生"
    assert third is not None and third.content == "算女的"
    assert fourth is not None and fourth.content == "算女孩子这边"
    assert loop.provider.chat_with_retry.await_count == 0


@pytest.mark.asyncio
async def test_identity_slot_early_returns_without_memory_retrieval_or_generated_detail(tmp_path: Path) -> None:
    life_state_service = MagicMock()
    life_state_service.retrieve_memory_evidence = AsyncMock(side_effect=AssertionError("should not be called"))
    life_state_service.record_generated_detail = AsyncMock(side_effect=AssertionError("should not be called"))
    life_state_service.record_recalled_event = AsyncMock(side_effect=AssertionError("should not be called"))
    life_state_service.reinforce_memory_evidence = AsyncMock(side_effect=AssertionError("should not be called"))
    life_state_service.get_state = AsyncMock(return_value={})
    life_state_service.get_recent_events = AsyncMock(return_value=[])
    life_state_service.set_override = AsyncMock(return_value={})
    life_state_service.clear_override = AsyncMock(return_value={})
    life_state_service.get_prehistory_summary = AsyncMock(return_value={})
    life_state_service.regenerate_prehistory = AsyncMock(return_value={})

    loop = _make_loop(tmp_path, life_state_service=life_state_service)

    out = await loop._process_message(
        InboundMessage(channel="qq", sender_id="u1", chat_id="c1", content="你是女生吗")
    )
    assert out is not None
    assert out.content == "我是女生"
    assert (tmp_path / "IDENTITY_PROFILE.json").exists()
    assert loop.provider.chat_with_retry.await_count == 0
    life_state_service.retrieve_memory_evidence.assert_not_awaited()
    life_state_service.record_generated_detail.assert_not_awaited()
    life_state_service.record_recalled_event.assert_not_awaited()
    life_state_service.reinforce_memory_evidence.assert_not_awaited()
