"""Tests for fact-layer storage and current-activity integration."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.loop import AgentLoop
from nanobot.companion.life_state.fact_store import LifeFactStore
from nanobot.companion.life_state.memory_utils import now_local, to_iso
from nanobot.companion.life_state.service import LifeStateService
from nanobot.providers.base import LLMResponse


def _make_loop(tmp_path: Path, llm_response: LLMResponse) -> AgentLoop:
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    provider.estimate_prompt_tokens.return_value = (10_000, "test")
    provider.chat_with_retry = AsyncMock(return_value=llm_response)
    loop = AgentLoop(
        bus=MagicMock(),
        provider=provider,
        workspace=tmp_path,
        model="test-model",
        context_window_tokens=65_536,
    )
    loop.provider.chat_with_retry = provider.chat_with_retry
    loop.tools.get_definitions = MagicMock(return_value=[])
    return loop


def test_preference_query_does_not_route_to_current_activity() -> None:
    assert AgentLoop._route_answer_slot("你喜欢哪首歌", "state") == "unknown"


def test_current_activity_prefers_strong_public_fact(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path, LLMResponse(content="ok", tool_calls=[]))
    now_iso = datetime.now().astimezone().replace(microsecond=0).isoformat()
    state = loop._resolve_current_activity_state(
        session_key="qq:c1",
        user_text="你在干什么",
        snapshot={"activity": "学习", "busy_level": 92, "last_tick": now_iso},
        recent_events=[],
        recent_facts=[
            {
                "fact_id": "fact_public_1",
                "fact_type": "activity",
                "content": "随便听歌",
                "source": "system_event",
                "confidence": "strong",
                "publicly_answerable": True,
                "start_at": now_iso,
                "expires_at": (datetime.now().astimezone() + timedelta(minutes=20)).replace(microsecond=0).isoformat(),
            }
        ],
        memory_evidence=[],
        memory_recall_level="none",
        prefer_recent_commitment=False,
    )
    assert state.get("source") == "fact_strong"
    assert state.get("layer") == "fact"
    assert "听歌" in str(state.get("reply") or "")


def test_current_activity_masks_nonpublic_fact(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path, LLMResponse(content="ok", tool_calls=[]))
    now_iso = datetime.now().astimezone().replace(microsecond=0).isoformat()
    state = loop._resolve_current_activity_state(
        session_key="qq:c1",
        user_text="你在干什么",
        snapshot={},
        recent_events=[],
        recent_facts=[
            {
                "fact_id": "fact_private_1",
                "fact_type": "activity",
                "content": "整理私信",
                "source": "tool",
                "confidence": "strong",
                "publicly_answerable": False,
                "start_at": now_iso,
                "expires_at": (datetime.now().astimezone() + timedelta(minutes=20)).replace(microsecond=0).isoformat(),
            }
        ],
        memory_evidence=[],
        memory_recall_level="none",
        prefer_recent_commitment=False,
    )
    assert state.get("source") == "fact_guarded"
    assert "私信" not in str(state.get("reply") or "")
    assert state.get("publicly_answerable") is False


def test_current_activity_builds_structured_event_for_synthesized_state(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path, LLMResponse(content="ok", tool_calls=[]))
    now_iso = datetime.now().astimezone().replace(microsecond=0).isoformat()
    state = loop._resolve_current_activity_state(
        session_key="qq:c1",
        user_text="你在忙什么",
        snapshot={"activity": "学习", "location": "家里", "last_tick": now_iso},
        recent_events=[],
        recent_facts=[],
        memory_evidence=[],
        memory_recall_level="none",
        prefer_recent_commitment=False,
    )

    event = state.get("event") or {}
    assert event.get("activity") == "学习"
    assert event.get("scene") == "家里"
    assert event.get("source") in {"extracted", "synthesized"}
    assert event.get("status")


def test_previous_activity_prefers_fact_when_timeline_is_coarse(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path, LLMResponse(content="ok", tool_calls=[]))
    now_iso = datetime.now().astimezone().replace(microsecond=0).isoformat()
    state = loop._resolve_previous_activity_state(
        user_text="刚才你在干什么",
        recent_events=["这会儿在外面"],
        recent_facts=[
            {
                "fact_id": "fact_public_prev_1",
                "fact_type": "activity",
                "content": "checking messages",
                "source": "dialogue",
                "confidence": "strong",
                "publicly_answerable": True,
                "start_at": now_iso,
                "expires_at": (datetime.now().astimezone() + timedelta(minutes=20)).replace(microsecond=0).isoformat(),
            }
        ],
        memory_evidence=[],
        memory_recall_level="none",
    )
    assert state.get("source") == "fact_strong"
    assert state.get("fact_id") == "fact_public_prev_1"


def test_fact_store_append_read_filter_and_prune(tmp_path: Path) -> None:
    store = LifeFactStore(tmp_path)
    now = now_local()

    store.append_fact(
        {
            "fact_type": "activity",
            "content": "随便听歌",
            "source": "system_event",
            "confidence": "strong",
            "publicly_answerable": True,
            "start_at": to_iso(now - timedelta(minutes=2)),
            "ttl_seconds": 1800,
        }
    )
    store.append_fact(
        {
            "fact_type": "event",
            "content": "整理消息",
            "source": "tool",
            "confidence": "weak",
            "publicly_answerable": False,
            "start_at": to_iso(now - timedelta(minutes=1)),
            "ttl_seconds": 1800,
        }
    )
    store.append_fact(
        {
            "fact_type": "activity",
            "content": "已过期内容",
            "source": "system_event",
            "confidence": "strong",
            "publicly_answerable": True,
            "start_at": to_iso(now - timedelta(minutes=20)),
            "ttl_seconds": 30,
        }
    )

    filtered = store.read_facts(
        limit=10,
        fact_types=["activity"],
        confidences=["strong"],
        publicly_answerable=True,
    )
    assert len(filtered) == 1
    assert filtered[0]["content"] == "随便听歌"

    removed = store.prune()
    assert removed >= 1


@pytest.mark.asyncio
async def test_dialogue_activity_fact_writeback_requires_non_generic_event_detail(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path, LLMResponse(content="ok", tool_calls=[]))
    loop.life_state_service = MagicMock()
    loop.life_state_service.record_dialogue_activity_fact = AsyncMock(return_value={})

    await loop._record_activity_fact_from_turn(
        answer_slot="current_activity",
        resolved_state={
            "source": "uncertain",
            "event": {
                "activity": "整理东西",
                "subject": "手头的小事",
                "status": "做了一会儿",
                "scene": "家里",
                "source": "synthesized",
                "confidence": "low",
            },
        },
        final_reply="在弄点东西",
        message_id="m1",
    )
    loop.life_state_service.record_dialogue_activity_fact.assert_not_awaited()

    await loop._record_activity_fact_from_turn(
        answer_slot="current_activity",
        resolved_state={
            "source": "uncertain",
            "event": {
                "activity": "学习",
                "subject": "线代例题",
                "status": "做了一会儿",
                "scene": "家里",
                "source": "synthesized",
                "confidence": "medium",
            },
            "publicly_answerable": True,
        },
        final_reply="在看线代例题",
        message_id="m2",
    )
    loop.life_state_service.record_dialogue_activity_fact.assert_awaited_once()


@pytest.mark.asyncio
async def test_previous_activity_generated_detail_backfills_timeline_when_missing(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path, LLMResponse(content="ok", tool_calls=[]))
    loop.life_state_service = MagicMock()
    loop.life_state_service.record_generated_timeline_event = AsyncMock(return_value={"summary": "刚在看消息"})

    out = await loop._record_timeline_backfill_from_turn(
        answer_slot="previous_activity",
        final_reply="刚在看消息",
        message_id="m-prev-1",
        recent_events=["这会儿在外面"],
        current_activity_state=None,
    )

    assert out is not None
    loop.life_state_service.record_generated_timeline_event.assert_awaited_once()


@pytest.mark.asyncio
async def test_service_consolidates_high_value_fact_into_memory(tmp_path: Path) -> None:
    service = LifeStateService(tmp_path)
    service._ensure_prehistory_bootstrap_locked = lambda **_kwargs: {}  # type: ignore[method-assign]
    start = now_local() - timedelta(minutes=10)

    row = await service.append_fact(
        fact_type="activity",
        content="整理消息",
        source="system_event",
        confidence="strong",
        publicly_answerable=True,
        start_at=start,
        ttl_seconds=3600,
        metadata={"coarse_type": "default"},
        consolidate_to_memory=True,
    )

    assert row is not None
    entries = service.memory_engine._load_entries()
    assert any(e.source_kind == "fact_layer_consolidation" for e in entries)
    facts = await service.read_facts(limit=10, fact_types=["activity"])
    assert any(
        isinstance(item.get("metadata"), dict)
        and item["metadata"].get("memory_pipeline") == "fact_layer"
        for item in facts
    )
