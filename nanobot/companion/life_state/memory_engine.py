"""Life-memory orchestration: ingest, decay, retrieve, reinforce, rebuild."""

from __future__ import annotations

import threading
import re
from datetime import datetime
from typing import Any

from nanobot.companion.life_state.memory_config import MemoryForgettingConfig
from nanobot.companion.life_state.memory_decay import decay_entry, reinforce_entry
from nanobot.companion.life_state.memory_interference import (
    estimate_cluster_pressure,
    recompute_cluster_pressure,
)
from nanobot.companion.life_state.memory_models import MemoryEntry, MemoryEvidence
from nanobot.companion.life_state.memory_retrieval import retrieve_memories
from nanobot.companion.life_state.memory_scoring import score_event
from nanobot.companion.life_state.memory_store import LifeMemoryStore
from nanobot.companion.life_state.memory_utils import now_local, parse_iso, to_iso


class LifeMemoryEngine:
    """Stateful engine implementing dual-track forgetting architecture."""

    def __init__(
        self,
        workspace,
        *,
        config: MemoryForgettingConfig | None = None,
        store: LifeMemoryStore | None = None,
    ):
        self.workspace = workspace
        self.config = config or MemoryForgettingConfig.from_workspace(workspace)
        self.store = store or LifeMemoryStore(workspace)
        self._lock = threading.RLock()

    def ingest_event(self, event: dict[str, Any]) -> MemoryEntry | None:
        """Append raw event and index it as a retrievable memory entry."""
        summary = str(event.get("summary") or "").strip()
        if not summary:
            return None

        with self._lock:
            raw = self.store.append_raw_event(event)
            event_time = parse_iso(raw.get("time")) or now_local()
            event_time_start = str(raw.get("event_time_start") or raw.get("time") or to_iso(event_time))
            event_time_end = str(raw.get("event_time_end") or event_time_start)
            mentioned_time = raw.get("mentioned_time")
            stored_time = str(raw.get("stored_time") or to_iso(event_time))
            source_turn = str(raw.get("source_turn") or "")
            source_kind = str(raw.get("source_kind") or raw.get("source") or "")

            entries = self._load_entries()
            changed = False
            for entry in entries:
                changed = decay_entry(entry, now=event_time, cfg=self.config) or changed
            recompute_cluster_pressure(entries, now=event_time, cfg=self.config)

            bootstrap = score_event(raw, self.config, cluster_pressure=0.0)
            pressure = estimate_cluster_pressure(
                entries,
                cluster_id=bootstrap["similarity_cluster_id"],
                now=event_time,
                cfg=self.config,
            )
            scored = score_event(raw, self.config, cluster_pressure=pressure)
            decay_profile = self._resolve_decay_profile(
                raw=raw,
                scored_memory_type=scored["memory_type"],
                pinned_flag=bool(scored["pinned_flag"]),
            )
            trace_summary = self._derive_trace_summary(
                raw=raw,
                decay_profile=decay_profile,
            )

            event_id = str(raw.get("event_id") or raw.get("id") or "")
            entry = MemoryEntry(
                id=f"mem_{event_id}",
                event_ids=[event_id] if event_id else [],
                timestamp_first=to_iso(event_time),
                timestamp_last=to_iso(event_time),
                event_time_start=event_time_start,
                event_time_end=event_time_end,
                mentioned_time=mentioned_time,
                stored_time=stored_time,
                source_turn=source_turn,
                source_kind=source_kind,
                memory_type=scored["memory_type"],
                gist_summary=scored["gist_summary"],
                detail_text=scored["detail_text"],
                trace_summary=trace_summary,
                importance=scored["importance"],
                salience=scored["salience"],
                self_relevance=scored["self_relevance"],
                relationship_relevance=scored["relationship_relevance"],
                emotional_weight=scored["emotional_weight"],
                novelty=scored["novelty"],
                source_confidence=scored["source_confidence"],
                retrieval_count=0,
                similarity_cluster_id=scored["similarity_cluster_id"],
                similarity_cluster_pressure=scored["similarity_cluster_pressure"],
                pinned_flag=scored["pinned_flag"],
                permanence_tier=scored["permanence_tier"],
                decay_profile=decay_profile,
                detail_strength=scored["detail_strength"],
                gist_strength=scored["gist_strength"],
                detail_strength_base=scored["detail_strength_base"],
                gist_strength_base=scored["gist_strength_base"],
                last_recalled_at=None,
                last_accessed_time=None,
                last_decay_at=to_iso(event_time),
            )
            entries.append(entry)
            recompute_cluster_pressure(entries, now=event_time, cfg=self.config)
            self._save_entries(entries)
            return entry

    def decay_to(self, now: datetime | None = None) -> int:
        """Decay all memory strengths forward to target time."""
        target = now or now_local()
        with self._lock:
            entries = self._load_entries()
            changed = 0
            for entry in entries:
                if decay_entry(entry, now=target, cfg=self.config):
                    changed += 1
            if changed:
                recompute_cluster_pressure(entries, now=target, cfg=self.config)
                self._save_entries(entries)
            return changed

    def retrieve(
        self,
        query: str,
        *,
        now: datetime | None = None,
        limit: int | None = None,
    ) -> list[MemoryEvidence]:
        """Retrieve currently recallable memories for a query."""
        target = now or now_local()
        with self._lock:
            entries = self._load_entries()
            changed = False
            for entry in entries:
                changed = decay_entry(entry, now=target, cfg=self.config) or changed
            recompute_cluster_pressure(entries, now=target, cfg=self.config)
            evidence = retrieve_memories(
                entries,
                query=query,
                now=target,
                cfg=self.config,
                limit=limit,
            )
            stamp = to_iso(target)
            used_ids = {item.id for item in evidence if item.id}
            if used_ids:
                for entry in entries:
                    if entry.id in used_ids and entry.last_accessed_time != stamp:
                        entry.last_accessed_time = stamp
                        changed = True
            if changed:
                self._save_entries(entries)
            return evidence

    def reinforce(self, memory_ids: list[str], *, now: datetime | None = None) -> int:
        """Apply retrieval-based strengthening to used memory entries."""
        if not memory_ids:
            return 0
        target = now or now_local()
        id_set = {str(x) for x in memory_ids if str(x)}
        with self._lock:
            entries = self._load_entries()
            reinforced = 0
            for entry in entries:
                if entry.id not in id_set:
                    continue
                reinforce_entry(entry, now=target, cfg=self.config)
                entry.last_accessed_time = to_iso(target)
                reinforced += 1
            if reinforced:
                recompute_cluster_pressure(entries, now=target, cfg=self.config)
                self._save_entries(entries)
            return reinforced

    def rebuild_from_raw_events(self) -> int:
        """Rebuild memory index deterministically from immutable raw event log."""
        with self._lock:
            raw_events = list(self.store.iter_raw_events())
            raw_events.sort(key=lambda e: str(e.get("time") or ""))
            entries: list[MemoryEntry] = []

            for raw in raw_events:
                summary = str(raw.get("summary") or "").strip()
                if not summary:
                    continue
                event_time = parse_iso(raw.get("time")) or now_local()
                event_time_start = str(raw.get("event_time_start") or raw.get("time") or to_iso(event_time))
                event_time_end = str(raw.get("event_time_end") or event_time_start)
                mentioned_time = raw.get("mentioned_time")
                stored_time = str(raw.get("stored_time") or to_iso(event_time))
                source_turn = str(raw.get("source_turn") or "")
                source_kind = str(raw.get("source_kind") or raw.get("source") or "")

                for existing in entries:
                    decay_entry(existing, now=event_time, cfg=self.config)
                recompute_cluster_pressure(entries, now=event_time, cfg=self.config)

                bootstrap = score_event(raw, self.config, cluster_pressure=0.0)
                pressure = estimate_cluster_pressure(
                    entries,
                    cluster_id=bootstrap["similarity_cluster_id"],
                    now=event_time,
                    cfg=self.config,
                )
                scored = score_event(raw, self.config, cluster_pressure=pressure)
                decay_profile = self._resolve_decay_profile(
                    raw=raw,
                    scored_memory_type=scored["memory_type"],
                    pinned_flag=bool(scored["pinned_flag"]),
                )
                trace_summary = self._derive_trace_summary(
                    raw=raw,
                    decay_profile=decay_profile,
                )
                event_id = str(raw.get("event_id") or raw.get("id") or "")
                entries.append(
                    MemoryEntry(
                        id=f"mem_{event_id}",
                        event_ids=[event_id] if event_id else [],
                        timestamp_first=to_iso(event_time),
                        timestamp_last=to_iso(event_time),
                        event_time_start=event_time_start,
                        event_time_end=event_time_end,
                        mentioned_time=mentioned_time,
                        stored_time=stored_time,
                        source_turn=source_turn,
                        source_kind=source_kind,
                        memory_type=scored["memory_type"],
                        gist_summary=scored["gist_summary"],
                        detail_text=scored["detail_text"],
                        trace_summary=trace_summary,
                        importance=scored["importance"],
                        salience=scored["salience"],
                        self_relevance=scored["self_relevance"],
                        relationship_relevance=scored["relationship_relevance"],
                        emotional_weight=scored["emotional_weight"],
                        novelty=scored["novelty"],
                        source_confidence=scored["source_confidence"],
                        retrieval_count=0,
                        similarity_cluster_id=scored["similarity_cluster_id"],
                        similarity_cluster_pressure=scored["similarity_cluster_pressure"],
                        pinned_flag=scored["pinned_flag"],
                        permanence_tier=scored["permanence_tier"],
                        decay_profile=decay_profile,
                        detail_strength=scored["detail_strength"],
                        gist_strength=scored["gist_strength"],
                        detail_strength_base=scored["detail_strength_base"],
                        gist_strength_base=scored["gist_strength_base"],
                        last_recalled_at=None,
                        last_accessed_time=None,
                        last_decay_at=to_iso(event_time),
                    )
                )
                recompute_cluster_pressure(entries, now=event_time, cfg=self.config)

            self._save_entries(entries)
            return len(entries)

    def build_prompt_evidence(
        self,
        query: str,
        *,
        now: datetime | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """Retrieve memories and format a prompt-safe evidence block."""
        evidence = self.retrieve(query, now=now, limit=limit)
        detail = [e for e in evidence if e.recall_level == "detail"]
        gist = [e for e in evidence if e.recall_level == "gist"]
        trace = [e for e in evidence if e.recall_level == "trace"]

        lines = [
            "Memory retrieval policy:",
            "- DETAIL evidence may be stated with specifics.",
            "- GIST evidence is coarse only; do not invent missing details.",
            "- TRACE evidence is very coarse only; do not restore specifics.",
            "- If no evidence is strong enough, say memory is unclear.",
        ]
        if detail:
            lines.append("DETAIL evidence:")
            for item in detail[:4]:
                lines.append(f"- [{item.id}] {item.text}")
        if gist:
            lines.append("GIST_ONLY evidence:")
            for item in gist[:4]:
                lines.append(f"- [{item.id}] {item.gist_summary}")
        if trace:
            lines.append("TRACE_ONLY evidence:")
            for item in trace[:4]:
                lines.append(f"- [{item.id}] {item.text}")
        if not detail and not gist and not trace:
            lines.append("No reliable long-term memory evidence for this query.")

        recall_level = "detail" if detail else ("gist" if gist else ("trace" if trace else "none"))
        return {
            "recall_level": recall_level,
            "evidence": [item.to_dict() for item in evidence],
            "prompt_block": "\n".join(lines),
        }

    @staticmethod
    def _resolve_decay_profile(
        *,
        raw: dict[str, Any],
        scored_memory_type: str,
        pinned_flag: bool,
    ) -> str:
        explicit = str(raw.get("decay_profile") or "").strip().lower()
        valid = {"meal", "study", "relationship", "anchor", "default"}
        if explicit in valid:
            return explicit
        if pinned_flag:
            return "anchor"

        text = " ".join(
            str(raw.get(k) or "")
            for k in ("type", "summary", "gist", "detail")
        ).lower()
        text = f"{text} {str(scored_memory_type or '').lower()}"
        if re.search(r"(identity|promise|milestone|anchor|身份|承诺|里程碑|锚点)", text):
            return "anchor"
        if re.search(r"(relationship|friend|family|partner|关系|朋友|家人|恋人)", text):
            return "relationship"
        if re.search(r"(meal|lunch|dinner|breakfast|吃|饭|早餐|午饭|晚饭)", text):
            return "meal"
        if re.search(r"(study|class|course|exam|学习|上课|复习|考试)", text):
            return "study"
        return "default"

    @staticmethod
    def _derive_trace_summary(*, raw: dict[str, Any], decay_profile: str) -> str:
        explicit = str(raw.get("trace_summary") or raw.get("trace") or "").strip()
        if explicit:
            return explicit
        if decay_profile == "meal":
            return "Had a meal around that time."
        if decay_profile == "study":
            return "Spent time on study-related activities."
        if decay_profile == "relationship":
            return "Had a relationship-relevant interaction."
        if decay_profile == "anchor":
            return "A long-term core milestone was involved."
        return "A past event happened in that period."

    def _load_entries(self) -> list[MemoryEntry]:
        payload = self.store.load_memory_index()
        entries_raw = payload.get("entries") or []
        out: list[MemoryEntry] = []
        for item in entries_raw:
            if isinstance(item, dict):
                entry = MemoryEntry.from_dict(item)
                if entry.id:
                    out.append(entry)
        return out

    def _save_entries(self, entries: list[MemoryEntry]) -> None:
        self.store.save_memory_index(
            {
                "entries": [entry.to_dict() for entry in entries],
                "entry_count": len(entries),
            }
        )
