"""Tests for stable BODY_PROFILE generation and rule-first body replies."""

from __future__ import annotations

import json
import re
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


def _load_body_profile(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.asyncio
async def test_body_profile_first_generate_then_reuse_same_values(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path)
    body_path = tmp_path / "BODY_PROFILE.json"
    assert not body_path.exists()

    out1 = await loop._process_message(
        InboundMessage(channel="qq", sender_id="u1", chat_id="c1", content="你多高")
    )
    assert out1 is not None
    assert re.search(r"\d+\s*cm", out1.content)
    assert body_path.exists()

    profile1 = _load_body_profile(body_path)
    assert profile1.get("height_cm")
    assert profile1.get("weight_kg")
    assert profile1.get("age")
    assert profile1.get("appearance")
    assert profile1.get("source")
    assert profile1.get("generated_at")

    out2 = await loop._process_message(
        InboundMessage(channel="qq", sender_id="u1", chat_id="c1", content="你多高")
    )
    assert out2 is not None
    assert out2.content == out1.content
    assert _load_body_profile(body_path) == profile1
    assert loop.provider.chat_with_retry.await_count == 0

    out_w = await loop._process_message(
        InboundMessage(channel="qq", sender_id="u1", chat_id="c1", content="你多重")
    )
    out_a = await loop._process_message(
        InboundMessage(channel="qq", sender_id="u1", chat_id="c1", content="你几岁")
    )
    out_l = await loop._process_message(
        InboundMessage(channel="qq", sender_id="u1", chat_id="c1", content="你长什么样")
    )
    assert out_w is not None and str(profile1["weight_kg"]) in out_w.content
    assert out_a is not None and str(profile1["age"]) in out_a.content
    assert out_l is not None and str(profile1["appearance"]) in out_l.content


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "seed_content",
    [
        "{",
        "{}",
        json.dumps({"height_cm": None, "weight_kg": "", "age": 0, "appearance": " ", "source": "", "generated_at": ""}),
    ],
)
async def test_body_profile_regenerates_when_file_invalid_or_incomplete(
    tmp_path: Path,
    seed_content: str,
) -> None:
    body_path = tmp_path / "BODY_PROFILE.json"
    body_path.write_text(seed_content, encoding="utf-8")
    loop = _make_loop(tmp_path)

    out = await loop._process_message(
        InboundMessage(channel="qq", sender_id="u1", chat_id="c1", content="你多重")
    )
    assert out is not None
    assert re.search(r"\d+\s*kg", out.content.lower())

    profile = _load_body_profile(body_path)
    assert isinstance(profile.get("height_cm"), int) and profile["height_cm"] > 0
    assert isinstance(profile.get("weight_kg"), int) and profile["weight_kg"] > 0
    assert isinstance(profile.get("age"), int) and profile["age"] > 0
    assert isinstance(profile.get("appearance"), str) and profile["appearance"].strip()
    assert isinstance(profile.get("source"), str) and profile["source"].strip()
    assert isinstance(profile.get("generated_at"), str) and profile["generated_at"].strip()
    assert loop.provider.chat_with_retry.await_count == 0


@pytest.mark.asyncio
async def test_body_profile_slot_early_returns_without_memory_retrieval_or_llm(tmp_path: Path) -> None:
    life_state_service = MagicMock()
    life_state_service.retrieve_memory_evidence = AsyncMock(side_effect=AssertionError("should not be called"))
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
        InboundMessage(channel="qq", sender_id="u1", chat_id="c1", content="你长什么样")
    )
    assert out is not None
    assert (tmp_path / "BODY_PROFILE.json").exists()
    assert loop.provider.chat_with_retry.await_count == 0
    life_state_service.retrieve_memory_evidence.assert_not_awaited()
    life_state_service.record_recalled_event.assert_not_awaited()
    life_state_service.reinforce_memory_evidence.assert_not_awaited()
    assert not (tmp_path / "memory" / "LIFE_EVENTS.jsonl").exists()
    assert not (tmp_path / "memory" / "LIFE_MEMORY_INDEX.json").exists()


def test_route_answer_slot_detects_body_profile_questions() -> None:
    assert AgentLoop._route_answer_slot("你多高", "social") == "body_profile"
    assert AgentLoop._route_answer_slot("你多重", "social") == "body_profile"
    assert AgentLoop._route_answer_slot("你几岁", "social") == "body_profile"
    assert AgentLoop._route_answer_slot("你长什么样", "social") == "body_profile"
    assert AgentLoop._route_answer_slot("你胖吗", "social") == "body_profile"
    assert AgentLoop._route_answer_slot("你瘦吗", "social") == "body_profile"
