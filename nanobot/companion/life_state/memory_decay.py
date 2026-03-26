"""Dual-track memory decay and retrieval reinforcement rules."""

from __future__ import annotations

import math
from datetime import datetime

from nanobot.companion.life_state.memory_config import MemoryForgettingConfig
from nanobot.companion.life_state.memory_models import MemoryEntry
from nanobot.companion.life_state.memory_utils import parse_iso, to_iso

_PROFILE_DECAY_MULTIPLIERS: dict[str, tuple[float, float]] = {
    "generated_detail": (2.4, 1.9),
}


def decay_entry(
    entry: MemoryEntry,
    *,
    now: datetime,
    cfg: MemoryForgettingConfig,
) -> bool:
    """Apply continuous exponential forgetting to one memory entry."""
    last = parse_iso(entry.last_decay_at) or parse_iso(entry.timestamp_last)
    if not last:
        entry.last_decay_at = to_iso(now)
        return True

    delta_hours = (now - last).total_seconds() / 3600.0
    if delta_hours <= 0.0:
        return False

    protection = _protection_factor(entry, cfg)
    tier_mul = cfg.permanence.tier_decay_multipliers.get(entry.permanence_tier, 1.0)
    detail_interference = 1.0 + cfg.interference.gamma_detail * max(0.0, entry.similarity_cluster_pressure)
    gist_interference = 1.0 + cfg.interference.gamma_gist * max(0.0, entry.similarity_cluster_pressure)

    detail_hours = delta_hours * detail_interference / max(cfg.decay.min_protection, protection)
    gist_hours = delta_hours * gist_interference / max(cfg.decay.min_protection, protection)

    lambda_detail = float(entry.decay_overrides.get("lambda_detail", cfg.decay.lambda_detail))
    lambda_gist = float(entry.decay_overrides.get("lambda_gist", cfg.decay.lambda_gist))
    mul_detail, mul_gist = _profile_decay_multiplier(entry)
    lambda_detail *= mul_detail
    lambda_gist *= mul_gist
    detail_factor = math.exp(-max(0.0, lambda_detail) * max(0.0, tier_mul) * detail_hours)
    gist_factor = math.exp(-max(0.0, lambda_gist) * max(0.0, tier_mul) * gist_hours)

    floor_detail, floor_gist = _strength_floors(entry, cfg)
    entry.detail_strength = max(floor_detail, entry.detail_strength * detail_factor)
    entry.gist_strength = max(floor_gist, entry.gist_strength * gist_factor)
    entry.last_decay_at = to_iso(now)
    return True


def reinforce_entry(
    entry: MemoryEntry,
    *,
    now: datetime,
    cfg: MemoryForgettingConfig,
) -> None:
    """Strengthen memory from retrieval with saturating bounded gain."""
    entry.retrieval_count = max(0, int(entry.retrieval_count)) + 1
    n = float(entry.retrieval_count)
    max_strength = cfg.strength.max_strength

    shape_detail = 1.0 - math.exp(-cfg.reinforcement.k_detail * n)
    shape_gist = 1.0 - math.exp(-cfg.reinforcement.k_gist * n)
    detail_delta = cfg.reinforcement.r_detail * shape_detail * max(0.0, max_strength - entry.detail_strength)
    gist_delta = cfg.reinforcement.r_gist * shape_gist * max(0.0, max_strength - entry.gist_strength)

    entry.detail_strength = min(max_strength, entry.detail_strength + detail_delta)
    entry.gist_strength = min(max_strength, entry.gist_strength + gist_delta)
    stamp = to_iso(now)
    entry.last_recalled_at = stamp
    entry.last_decay_at = stamp


def _protection_factor(entry: MemoryEntry, cfg: MemoryForgettingConfig) -> float:
    base = (
        1.0
        + cfg.decay.salience_shield * entry.salience
        + cfg.decay.emotional_shield * entry.emotional_weight
        + cfg.decay.relationship_shield * entry.relationship_relevance
        + cfg.decay.self_relevance_shield * entry.self_relevance
    )
    rehearse = 1.0 + cfg.decay.retrieval_shield * math.log1p(max(0.0, float(entry.retrieval_count)))
    return max(cfg.decay.min_protection, base * rehearse)


def _strength_floors(entry: MemoryEntry, cfg: MemoryForgettingConfig) -> tuple[float, float]:
    if entry.pinned_flag or entry.permanence_tier == "pinned":
        return cfg.permanence.pinned_detail_floor, cfg.permanence.pinned_gist_floor

    tier_floor = cfg.permanence.tier_min_floors.get(entry.permanence_tier, {})
    detail = float(tier_floor.get("detail", 0.0))
    gist = float(tier_floor.get("gist", 0.0))
    return max(0.0, detail), max(0.0, gist)


def _profile_decay_multiplier(entry: MemoryEntry) -> tuple[float, float]:
    source_kind = str(entry.source_kind or "").strip().lower()
    if source_kind == "generated_detail":
        return _PROFILE_DECAY_MULTIPLIERS["generated_detail"]
    profile = str(entry.decay_profile or "").strip().lower()
    return _PROFILE_DECAY_MULTIPLIERS.get(profile, (1.0, 1.0))
