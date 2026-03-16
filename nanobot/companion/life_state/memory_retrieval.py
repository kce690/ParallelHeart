"""Query-time retrieval and detail/gist threshold gating."""

from __future__ import annotations

import math
import re
from datetime import datetime

from nanobot.companion.life_state.memory_config import MemoryForgettingConfig
from nanobot.companion.life_state.memory_interference import apply_interference_penalty
from nanobot.companion.life_state.memory_models import MemoryEntry, MemoryEvidence
from nanobot.companion.life_state.memory_utils import parse_iso, tokenize

_TRACE_MIN_SIGNAL = 0.06
_PROFILE_WINDOWS_HOURS: dict[str, dict[str, float | None]] = {
    "meal": {"detail": 24.0, "gist": 72.0, "trace": 336.0},
    "study": {"detail": 48.0, "gist": 168.0, "trace": 720.0},
    "relationship": {"detail": 168.0, "gist": 720.0, "trace": None},
    "anchor": {"detail": 168.0, "gist": 720.0, "trace": None},
    "default": {"detail": 48.0, "gist": 168.0, "trace": 720.0},
}


def retrieve_memories(
    entries: list[MemoryEntry],
    *,
    query: str,
    now: datetime,
    cfg: MemoryForgettingConfig,
    limit: int | None = None,
) -> list[MemoryEvidence]:
    """Retrieve memories with threshold-based detail/gist gating."""
    out: list[tuple[float, MemoryEvidence]] = []
    query_tokens = tokenize(query)
    max_results = limit or cfg.retrieval.max_results

    for entry in entries:
        relevance = _relevance_score(entry, query_tokens=query_tokens, now=now)
        detail_eff, gist_eff = apply_interference_penalty(entry, cfg)
        recall_level = _recall_level(entry, detail_eff, gist_eff, cfg, now=now)
        if recall_level == "none":
            continue
        if relevance < cfg.retrieval.min_relevance and not entry.pinned_flag:
            continue

        if recall_level == "detail":
            score = relevance * detail_eff
            text = entry.detail_text
        elif recall_level == "gist":
            score = relevance * gist_eff * 0.92
            text = entry.gist_summary
        else:
            score = relevance * max(gist_eff, detail_eff, _TRACE_MIN_SIGNAL) * 0.80
            text = _trace_text(entry)
        evidence = MemoryEvidence(
            id=entry.id,
            recall_level=recall_level,
            text=text,
            gist_summary=entry.gist_summary,
            event_ids=list(entry.event_ids),
            relevance_score=relevance,
            detail_strength_effective=detail_eff,
            gist_strength_effective=gist_eff,
            similarity_cluster_pressure=entry.similarity_cluster_pressure,
            permanence_tier=entry.permanence_tier,
            pinned_flag=entry.pinned_flag,
        )
        out.append((score, evidence))

    out.sort(key=lambda item: item[0], reverse=True)
    return [item[1] for item in out[: max(1, int(max_results))]]


def _recall_level(
    entry: MemoryEntry,
    detail_eff: float,
    gist_eff: float,
    cfg: MemoryForgettingConfig,
    *,
    now: datetime,
) -> str:
    profile = _decay_profile(entry)
    windows = _PROFILE_WINDOWS_HOURS.get(profile, _PROFILE_WINDOWS_HOURS["default"])
    age_hours = _event_age_hours(entry, now=now)
    allow_detail = _within(age_hours, windows.get("detail"))
    allow_gist = _within(age_hours, windows.get("gist"))
    allow_trace = _within(age_hours, windows.get("trace"))

    if allow_detail and detail_eff >= cfg.retrieval.T_detail:
        return "detail"
    if allow_gist and gist_eff >= cfg.retrieval.T_gist:
        return "gist"
    if allow_trace and (
        profile in {"anchor", "relationship"} or max(detail_eff, gist_eff) >= _TRACE_MIN_SIGNAL
    ):
        return "trace"
    return "none"


def _event_age_hours(entry: MemoryEntry, *, now: datetime) -> float | None:
    stamp = (
        parse_iso(entry.event_time_start)
        or parse_iso(entry.timestamp_last)
        or parse_iso(entry.stored_time)
    )
    if stamp is None:
        return None
    return max(0.0, (now - stamp).total_seconds() / 3600.0)


def _within(age_hours: float | None, limit_hours: float | None) -> bool:
    if age_hours is None:
        return True
    if limit_hours is None:
        return True
    return age_hours <= float(limit_hours)


def _decay_profile(entry: MemoryEntry) -> str:
    explicit = str(entry.decay_profile or "").strip().lower()
    if explicit in _PROFILE_WINDOWS_HOURS:
        return explicit
    if entry.pinned_flag or entry.permanence_tier == "pinned":
        return "anchor"

    text = f"{entry.memory_type} {entry.gist_summary}".lower()
    if re.search(r"(identity|promise|milestone|anchor|身份|承诺|里程碑|锚点)", text):
        return "anchor"
    if re.search(r"(relationship|friend|family|partner|关系|朋友|家人|恋人)", text):
        return "relationship"
    if re.search(r"(meal|lunch|dinner|breakfast|吃|饭|早餐|午饭|晚饭)", text):
        return "meal"
    if re.search(r"(study|class|course|exam|学习|上课|复习|考试)", text):
        return "study"
    return "default"


def _trace_text(entry: MemoryEntry) -> str:
    explicit = str(entry.trace_summary or "").strip()
    if explicit:
        return explicit
    profile = _decay_profile(entry)
    if profile == "meal":
        return "Had a meal around that time."
    if profile == "study":
        return "Spent time on study-related activities."
    if profile == "relationship":
        return "Had a relationship-relevant interaction."
    if profile == "anchor":
        return "A long-term core milestone was involved."
    return "A past event happened in that period."


def _relevance_score(
    entry: MemoryEntry,
    *,
    query_tokens: list[str],
    now: datetime,
) -> float:
    gist_tokens = set(tokenize(entry.gist_summary))
    detail_tokens = set(tokenize(entry.detail_text))
    query_set = set(query_tokens)
    if not query_set:
        overlap = 0.08
    else:
        gist_overlap = len(query_set & gist_tokens) / max(1.0, float(len(query_set | gist_tokens)))
        detail_overlap = len(query_set & detail_tokens) / max(1.0, float(len(query_set | detail_tokens)))
        overlap = 0.62 * detail_overlap + 0.38 * gist_overlap

    stamp = parse_iso(entry.timestamp_last) or now
    age_hours = max(0.0, (now - stamp).total_seconds() / 3600.0)
    recency = 0.15 * math.exp(-age_hours / 336.0)
    pin_bonus = 0.10 if entry.pinned_flag else 0.0
    return max(0.0, min(1.0, overlap + recency + pin_bonus))
