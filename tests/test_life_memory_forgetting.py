from __future__ import annotations

import json
from datetime import timedelta

import pytest

from nanobot.agent.loop import AgentLoop
from nanobot.companion.life_state.memory_config import MemoryForgettingConfig
from nanobot.companion.life_state.memory_engine import LifeMemoryEngine
from nanobot.companion.life_state.memory_interference import apply_interference_penalty
from nanobot.companion.life_state.memory_utils import now_local, to_iso
from nanobot.companion.life_state.service import LifeStateService


def _mk_engine(tmp_path, cfg: MemoryForgettingConfig | None = None) -> LifeMemoryEngine:
    return LifeMemoryEngine(tmp_path, config=cfg or MemoryForgettingConfig())


def test_detail_decays_faster_than_gist(tmp_path) -> None:
    cfg = MemoryForgettingConfig()
    cfg.decay.lambda_detail = 0.20
    cfg.decay.lambda_gist = 0.05
    engine = _mk_engine(tmp_path, cfg)
    engine.ingest_event({"time": to_iso(now_local()), "summary": "Completed an important exam", "importance": 3})

    before = engine._load_entries()[0]
    base_detail = before.detail_strength
    base_gist = before.gist_strength
    later = now_local() + timedelta(hours=24)
    engine.decay_to(later)
    after = engine._load_entries()[0]

    assert after.detail_strength < base_detail
    assert after.gist_strength < base_gist
    assert after.detail_strength < after.gist_strength


def test_pinned_memory_remains_available(tmp_path) -> None:
    engine = _mk_engine(tmp_path)
    engine.ingest_event(
        {
            "time": to_iso(now_local()),
            "summary": "Identity milestone promise with strong emotional impact",
            "importance": 3,
            "source": "manual",
            "core_memory": True,
        }
    )
    far = now_local() + timedelta(days=365)
    engine.decay_to(far)
    entry = engine._load_entries()[0]

    assert entry.pinned_flag is True
    assert entry.gist_strength >= engine.config.permanence.pinned_gist_floor
    assert entry.detail_strength >= engine.config.permanence.pinned_detail_floor
    evidence = engine.retrieve("promise", now=far, limit=3)
    assert evidence
    assert evidence[0].recall_level in {"detail", "gist", "trace"}


def test_volatile_memory_drops_below_threshold_quickly(tmp_path) -> None:
    cfg = MemoryForgettingConfig()
    cfg.permanence.volatile_salience_threshold = 0.70
    cfg.decay.lambda_detail = 0.25
    cfg.decay.lambda_gist = 0.12
    engine = _mk_engine(tmp_path, cfg)
    base = now_local()
    for i in range(8):
        engine.ingest_event(
            {
                "time": to_iso(base + timedelta(minutes=i)),
                "summary": "Had lunch as usual",
                "importance": 0.1,
                "source": "timer",
            }
        )

    engine.decay_to(base + timedelta(hours=36))
    volatile_entries = [e for e in engine._load_entries() if e.permanence_tier == "volatile"]
    assert volatile_entries
    assert any(e.gist_strength < cfg.retrieval.T_gist for e in volatile_entries)


def test_retrieval_strengthens_memory(tmp_path) -> None:
    engine = _mk_engine(tmp_path)
    engine.ingest_event({"time": to_iso(now_local()), "summary": "Reviewed a difficult topic", "importance": 2})
    evidence = engine.retrieve("difficult topic", now=now_local(), limit=2)
    assert evidence
    memory_id = evidence[0].id
    before = next(e for e in engine._load_entries() if e.id == memory_id)
    detail_before = before.detail_strength
    gist_before = before.gist_strength

    reinforced = engine.reinforce([memory_id], now=now_local() + timedelta(minutes=10))
    assert reinforced == 1
    after = next(e for e in engine._load_entries() if e.id == memory_id)
    assert after.retrieval_count == 1
    assert after.detail_strength > detail_before
    assert after.gist_strength > gist_before


def test_repeated_similar_events_increase_interference_on_detail(tmp_path) -> None:
    engine = _mk_engine(tmp_path)
    base = now_local()
    for i in range(6):
        engine.ingest_event(
            {
                "time": to_iso(base + timedelta(minutes=i)),
                "summary": "Routine lunch event",
                "importance": 1,
                "source": "timer",
            }
        )

    entry = engine._load_entries()[-1]
    assert entry.similarity_cluster_pressure > 0.0
    detail_eff, gist_eff = apply_interference_penalty(entry, engine.config)
    assert detail_eff < entry.detail_strength
    assert gist_eff <= entry.gist_strength
    assert (entry.detail_strength - detail_eff) >= (entry.gist_strength - gist_eff)


def test_gist_survives_after_detail_falls_below_threshold(tmp_path) -> None:
    cfg = MemoryForgettingConfig()
    cfg.decay.lambda_detail = 0.30
    cfg.decay.lambda_gist = 0.03
    engine = _mk_engine(tmp_path, cfg)
    t0 = now_local()
    engine.ingest_event({"time": to_iso(t0), "summary": "Finished exam review session", "importance": 3})

    evidence = engine.retrieve("exam review", now=t0 + timedelta(hours=16), limit=3)
    assert evidence
    assert evidence[0].recall_level == "gist"


def test_query_time_gate_detail_then_gist_then_trace_then_none(tmp_path) -> None:
    cfg = MemoryForgettingConfig()
    cfg.decay.lambda_detail = 0.50
    cfg.decay.lambda_gist = 0.20
    cfg.permanence.durable_salience_threshold = 0.98
    engine = _mk_engine(tmp_path, cfg)
    t0 = now_local()
    engine.ingest_event({"time": to_iso(t0), "summary": "Completed intense practice session", "importance": 3})

    early = engine.retrieve("practice session", now=t0, limit=2)
    mid = engine.retrieve("practice session", now=t0 + timedelta(hours=5), limit=2)
    late = engine.retrieve("practice session", now=t0 + timedelta(hours=20), limit=2)
    very_late = engine.retrieve("practice session", now=t0 + timedelta(days=40), limit=2)

    assert early and early[0].recall_level == "detail"
    assert mid and mid[0].recall_level == "gist"
    assert late and late[0].recall_level == "trace"
    assert very_late == []


def test_study_profile_same_day_detail_then_week_gist(tmp_path) -> None:
    cfg = MemoryForgettingConfig()
    cfg.decay.lambda_detail = 0.08
    cfg.decay.lambda_gist = 0.003
    cfg.retrieval.T_detail = 0.25
    cfg.permanence.durable_salience_threshold = 0.99
    engine = _mk_engine(tmp_path, cfg)
    t0 = now_local()
    engine.ingest_event(
        {
            "time": to_iso(t0),
            "event_time_start": to_iso(t0),
            "summary": "Studied linear algebra at 9am and reviewed determinants",
            "source": "dialog",
            "importance": 1.8,
            "decay_profile": "study",
            "trace_summary": "Spent time on study-related activities.",
        }
    )

    same_day = engine.retrieve("studied linear algebra determinants", now=t0 + timedelta(hours=10), limit=3)
    one_week = engine.retrieve("studied linear algebra determinants", now=t0 + timedelta(days=6), limit=3)

    assert same_day and same_day[0].recall_level == "detail"
    assert one_week and one_week[0].recall_level == "gist"


def test_meal_profile_event_becomes_forgotten_after_trace_window(tmp_path) -> None:
    engine = _mk_engine(tmp_path)
    t0 = now_local()
    engine.ingest_event(
        {
            "time": to_iso(t0),
            "event_time_start": to_iso(t0),
            "summary": "中午12点吃了牛肉面",
            "source": "dialog",
            "importance": 0.6,
            "decay_profile": "meal",
            "trace_summary": "Had a meal around that time.",
        }
    )

    far = engine.retrieve("中午吃了什么", now=t0 + timedelta(days=20), limit=3)
    assert far == []


def test_trace_level_does_not_leak_detail_tokens(tmp_path) -> None:
    cfg = MemoryForgettingConfig()
    cfg.decay.lambda_detail = 0.08
    cfg.decay.lambda_gist = 0.003
    cfg.permanence.durable_salience_threshold = 0.99
    engine = _mk_engine(tmp_path, cfg)
    t0 = now_local()
    engine.ingest_event(
        {
            "time": to_iso(t0),
            "event_time_start": to_iso(t0),
            "summary": "Studied linear algebra at 9am and reviewed determinants",
            "source": "dialog",
            "importance": 1.8,
            "decay_profile": "study",
            "trace_summary": "Spent time on study-related activities.",
        }
    )

    evidence = engine.retrieve("studied linear algebra determinants", now=t0 + timedelta(days=10), limit=3)
    assert evidence
    assert evidence[0].recall_level == "trace"
    assert "9am" not in evidence[0].text.lower()
    assert "linear algebra" not in evidence[0].text.lower()
    assert "determinants" not in evidence[0].text.lower()


def test_old_index_entries_without_trace_fields_are_compatible(tmp_path) -> None:
    engine = _mk_engine(tmp_path)
    t0 = now_local()
    engine.store.save_memory_index(
        {
            "version": 1,
            "updated_at": to_iso(t0),
            "entries": [
                {
                    "id": "mem_legacy_1",
                    "event_ids": ["evt_legacy_1"],
                    "timestamp_first": to_iso(t0),
                    "timestamp_last": to_iso(t0),
                    "event_time_start": to_iso(t0),
                    "event_time_end": to_iso(t0),
                    "memory_type": "life_event",
                    "gist_summary": "General memory: studied in that period.",
                    "detail_text": "今天上午9点学线性代数，看到行列式",
                    "importance": 0.8,
                    "salience": 0.8,
                    "detail_strength": 0.7,
                    "gist_strength": 0.8,
                    "detail_strength_base": 0.7,
                    "gist_strength_base": 0.8,
                }
            ],
        }
    )

    evidence = engine.retrieve("上午学了什么", now=t0 + timedelta(days=2), limit=3)
    payload = engine.build_prompt_evidence("上午学了什么", now=t0 + timedelta(days=2), limit=3)

    assert isinstance(evidence, list)
    assert isinstance(payload, dict)
    assert "recall_level" in payload


@pytest.mark.asyncio
async def test_record_recalled_event_sets_decay_profile_and_kind(tmp_path) -> None:
    service = LifeStateService(tmp_path)
    t0 = now_local()
    out = await service.record_recalled_event(
        summary="中午12点吃了牛肉面",
        source_turn="turn_x",
        event_time_start=t0,
    )

    assert out is not None
    assert out.get("decay_profile") == "meal"
    assert out.get("type") == "recalled_meal"


@pytest.mark.asyncio
async def test_offline_catchup_updates_memory_index_consistently(tmp_path) -> None:
    service = LifeStateService(tmp_path)
    now = now_local()
    stale = {
        "location": "home",
        "activity": "rest",
        "mood": "calm",
        "energy": 62,
        "social_battery": 50,
        "urgency_bias": 45,
        "last_tick": to_iso(now - timedelta(hours=5)),
        "next_transition_at": to_iso(now - timedelta(hours=4, minutes=45)),
    }
    (tmp_path / "LIFESTATE.json").write_text(json.dumps(stale), encoding="utf-8")

    steps = await service.fast_forward_to_now()
    assert steps >= 1
    assert (tmp_path / "memory" / "LIFE_MEMORY_INDEX.json").exists()
    assert service.memory_engine._load_entries()
    payload = await service.retrieve_memory_evidence("what happened", limit=3)
    assert isinstance(payload, dict)
    assert "recall_level" in payload


def test_rebuild_memory_index_from_raw_event_log(tmp_path) -> None:
    engine = _mk_engine(tmp_path)
    t0 = now_local()
    for i in range(3):
        engine.ingest_event({"time": to_iso(t0 + timedelta(minutes=i)), "summary": f"event-{i}", "importance": 1})

    raw_count = len(list(engine.store.iter_raw_events()))
    engine.store.memory_index_path.write_text("{}", encoding="utf-8")
    rebuilt = engine.rebuild_from_raw_events()
    entries = engine._load_entries()

    assert rebuilt == raw_count
    assert len(entries) == raw_count
    assert all(e.id and e.event_ids for e in entries)


def test_response_side_gist_gate_prevents_fabricated_detail() -> None:
    out = AgentLoop._apply_evidence_constraint(
        "你刚才在干嘛",
        "刚刚在家整理完文件，还顺便看了窗外",
        answer_slot="previous_activity",
        recent_events=[],
        has_recent_event=False,
        memory_recall_level="gist",
    )
    assert out == "只记得个大概"
