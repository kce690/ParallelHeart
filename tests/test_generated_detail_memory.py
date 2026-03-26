from __future__ import annotations

from datetime import timedelta

import pytest

from nanobot.agent.loop import AgentLoop
from nanobot.companion.life_state.memory_engine import LifeMemoryEngine
from nanobot.companion.life_state.memory_utils import now_local, to_iso
from nanobot.companion.life_state.service import LifeStateService
from nanobot.utils.mojibake import analyze_mojibake


@pytest.mark.asyncio
async def test_record_generated_detail_writes_volatile_memory(tmp_path) -> None:
    service = LifeStateService(tmp_path)
    out = await service.record_generated_detail(
        slot="meal",
        content="\u521a\u968f\u4fbf\u5403\u4e86\u70b9\u9762\uff0c\u5728\u5916\u9762\u5f85\u7740",
        source_turn="turn_1",
    )

    assert out is not None
    assert out.get("source_kind") == "generated_detail"
    assert out.get("fact_source") == "model_generated"
    assert out.get("coarse_type") == "meal"
    assert out.get("decay_profile") == "generated_detail"
    assert out.get("pinned_flag") is False

    entries = service.memory_engine._load_entries()
    assert entries
    latest = entries[-1]
    assert latest.source_kind == "generated_detail"
    assert latest.decay_profile == "generated_detail"
    assert latest.permanence_tier == "volatile"
    assert latest.pinned_flag is False
    assert latest.source_confidence <= 0.45


def test_generated_detail_profile_forgets_faster(tmp_path) -> None:
    engine = LifeMemoryEngine(tmp_path)
    t0 = now_local()
    engine.ingest_event(
        {
            "time": to_iso(t0),
            "event_time_start": to_iso(t0),
            "event_time_end": to_iso(t0 + timedelta(minutes=30)),
            "summary": "just had noodles outside",
            "type": "generated_detail_meal",
            "source": "model_generated",
            "source_kind": "generated_detail",
            "decay_profile": "generated_detail",
            "coarse_type": "meal",
            "source_confidence": 0.42,
        }
    )

    early = engine.retrieve("noodles", now=t0 + timedelta(hours=1), limit=3)
    later = engine.retrieve("noodles", now=t0 + timedelta(hours=48), limit=3)
    far = engine.retrieve("noodles", now=t0 + timedelta(days=8), limit=3)

    assert early
    assert early[0].recall_level in {"detail", "gist"}
    if later:
        assert later[0].recall_level in {"gist", "trace"}
    assert far == []


@pytest.mark.asyncio
async def test_hard_fact_downgrades_conflicting_generated_detail(tmp_path) -> None:
    service = LifeStateService(tmp_path)
    out = await service.record_generated_detail(
        slot="current_activity",
        content="\u5728\u5bb6\u6574\u7406\u62bd\u5c49",
        source_turn="turn_gd",
    )
    assert out is not None
    memory_id = str(out.get("memory_id") or "")
    assert memory_id

    before = next(e for e in service.memory_engine._load_entries() if e.id == memory_id)
    await service.append_fact(
        fact_type="activity",
        content="\u5728\u516c\u53f8\u5f00\u4f1a",
        source="system_event",
        confidence="strong",
        publicly_answerable=True,
        start_at=now_local(),
        consolidate_to_memory=False,
    )
    after = next(e for e in service.memory_engine._load_entries() if e.id == memory_id)
    assert after.detail_strength <= before.detail_strength
    assert after.gist_strength <= before.gist_strength


def test_extract_generated_detail_fragments() -> None:
    fragments = AgentLoop._extract_generated_detail_fragments(
        "meal",
        "\u521a\u968f\u4fbf\u5403\u4e86\u70b9\u9762\uff0c\u5728\u5916\u9762\u5f85\u7740",
    )
    assert fragments
    assert any("\u5403" in item for item in fragments)
    assert any("\u5916\u9762" in item for item in fragments)


def test_detects_known_mojibake_samples() -> None:
    assert analyze_mojibake("在忙瀛︿範")[0] is True
    assert analyze_mojibake("杩欎細鍎垮湪瀹舵瓏鐫€")[0] is True
    assert analyze_mojibake("我这会儿在忙着学习呢")[0] is False


def test_enforce_fact_grounded_reply_keeps_normal_reply_when_anchor_is_mojibake() -> None:
    out = AgentLoop._enforce_fact_grounded_reply(
        answer_slot="current_activity",
        reply="我这会儿在忙着学习呢",
        slot_floor_reply="在忙瀛︿範",
        current_activity_state={
            "source": "fact_strong",
            "fact": "杩欎細鍎垮湪瀹舵瓏鐫€",
            "uncertain": False,
        },
        memory_evidence=None,
        memory_recall_level="detail",
    )

    assert out == "我这会儿在忙着学习呢"


def test_extract_generated_detail_fragments_drops_mojibake() -> None:
    fragments = AgentLoop._extract_generated_detail_fragments(
        "current_activity",
        "在忙瀛︿範",
    )
    assert fragments == []


def test_normalize_state_payload_repairs_known_mojibake_labels(tmp_path) -> None:
    service = LifeStateService(tmp_path)
    state = service._normalize_state_payload(
        {"location": "瀹?", "activity": "瀛︿範"},
        now=now_local(),
    )

    assert state["location"] == "家里"
    assert state["activity"] == "学习"


@pytest.mark.asyncio
async def test_record_generated_detail_drops_mojibake(tmp_path) -> None:
    service = LifeStateService(tmp_path)
    out = await service.record_generated_detail(
        slot="current_activity",
        content="在忙瀛︿範",
        source_turn="turn_dirty",
    )

    assert out is None
    assert service.memory_engine._load_entries() == []
