"""Data models for life-memory index and retrieval evidence."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class MemoryEntry:
    """Persisted memory entry with dual-track strengths."""

    id: str
    event_ids: list[str] = field(default_factory=list)
    timestamp_first: str = ""
    timestamp_last: str = ""
    event_time_start: str = ""
    event_time_end: str = ""
    mentioned_time: str | None = None
    stored_time: str = ""
    source_turn: str = ""
    source_kind: str = ""
    memory_type: str = "life_event"
    gist_summary: str = ""
    detail_text: str = ""
    trace_summary: str = ""
    importance: float = 0.0
    salience: float = 0.0
    self_relevance: float = 0.0
    relationship_relevance: float = 0.0
    emotional_weight: float = 0.0
    novelty: float = 0.0
    source_confidence: float = 0.0
    retrieval_count: int = 0
    similarity_cluster_id: str = ""
    similarity_cluster_pressure: float = 0.0
    pinned_flag: bool = False
    permanence_tier: str = "normal"
    decay_profile: str = "default"
    detail_strength: float = 0.0
    gist_strength: float = 0.0
    detail_strength_base: float = 0.0
    gist_strength_base: float = 0.0
    last_recalled_at: str | None = None
    last_accessed_time: str | None = None
    last_decay_at: str = ""
    decay_overrides: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MemoryEntry":
        data = dict(payload or {})
        return cls(
            id=str(data.get("id") or ""),
            event_ids=[str(x) for x in data.get("event_ids") or []],
            timestamp_first=str(data.get("timestamp_first") or ""),
            timestamp_last=str(data.get("timestamp_last") or ""),
            event_time_start=str(data.get("event_time_start") or data.get("timestamp_first") or ""),
            event_time_end=str(data.get("event_time_end") or data.get("timestamp_last") or data.get("timestamp_first") or ""),
            mentioned_time=data.get("mentioned_time"),
            stored_time=str(data.get("stored_time") or data.get("timestamp_first") or ""),
            source_turn=str(data.get("source_turn") or ""),
            source_kind=str(data.get("source_kind") or data.get("source") or ""),
            memory_type=str(data.get("memory_type") or "life_event"),
            gist_summary=str(data.get("gist_summary") or ""),
            detail_text=str(data.get("detail_text") or ""),
            trace_summary=str(data.get("trace_summary") or ""),
            importance=float(data.get("importance") or 0.0),
            salience=float(data.get("salience") or 0.0),
            self_relevance=float(data.get("self_relevance") or 0.0),
            relationship_relevance=float(data.get("relationship_relevance") or 0.0),
            emotional_weight=float(data.get("emotional_weight") or 0.0),
            novelty=float(data.get("novelty") or 0.0),
            source_confidence=float(data.get("source_confidence") or 0.0),
            retrieval_count=int(data.get("retrieval_count") or 0),
            similarity_cluster_id=str(data.get("similarity_cluster_id") or ""),
            similarity_cluster_pressure=float(data.get("similarity_cluster_pressure") or 0.0),
            pinned_flag=bool(data.get("pinned_flag") or False),
            permanence_tier=str(data.get("permanence_tier") or "normal"),
            decay_profile=str(data.get("decay_profile") or "default"),
            detail_strength=float(data.get("detail_strength") or 0.0),
            gist_strength=float(data.get("gist_strength") or 0.0),
            detail_strength_base=float(data.get("detail_strength_base") or data.get("detail_strength") or 0.0),
            gist_strength_base=float(data.get("gist_strength_base") or data.get("gist_strength") or 0.0),
            last_recalled_at=data.get("last_recalled_at"),
            last_accessed_time=data.get("last_accessed_time"),
            last_decay_at=str(data.get("last_decay_at") or ""),
            decay_overrides=dict(data.get("decay_overrides") or {}),
        )


@dataclass
class MemoryEvidence:
    """Query-time retrieval result after threshold gating."""

    id: str
    recall_level: str
    text: str
    gist_summary: str
    event_ids: list[str]
    relevance_score: float
    detail_strength_effective: float
    gist_strength_effective: float
    similarity_cluster_pressure: float
    permanence_tier: str
    pinned_flag: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
