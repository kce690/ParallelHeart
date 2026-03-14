from __future__ import annotations

import json
from datetime import timedelta

import pytest

from nanobot.companion.life_state.service import LifeStateService, _now_local, _to_iso


@pytest.mark.asyncio
async def test_life_state_default_shape(tmp_path):
    service = LifeStateService(tmp_path)
    state = await service.get_state()

    required = {
        "location",
        "activity",
        "mood",
        "energy",
        "social_battery",
        "urgency_bias",
        "last_tick",
        "next_transition_at",
    }
    for key in required:
        assert key in state


@pytest.mark.asyncio
async def test_life_state_offline_catchup_and_events(tmp_path):
    service = LifeStateService(tmp_path)
    now = _now_local()
    stale = {
        "location": "家",
        "activity": "休息",
        "mood": "平静",
        "energy": 62,
        "social_battery": 50,
        "urgency_bias": 45,
        "last_tick": _to_iso(now - timedelta(hours=6)),
        "next_transition_at": _to_iso(now - timedelta(hours=5, minutes=45)),
    }
    (tmp_path / "LIFESTATE.json").write_text(
        json.dumps(stale, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    steps = await service.fast_forward_to_now()
    assert steps >= 1

    events = await service.get_recent_events(limit=3)
    assert isinstance(events, list)
