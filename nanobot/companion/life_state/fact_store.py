"""Durable short-horizon fact storage for current activity/event grounding."""

from __future__ import annotations

import json
import os
import uuid
from datetime import timedelta
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.companion.life_state.memory_utils import now_local, parse_iso, to_iso
from nanobot.utils.helpers import ensure_dir


class LifeFactStore:
    """Persistent store for expiring activity/event facts."""

    _MAX_FACTS = 240
    _MAX_APPEND_DEDUPE_SCAN = 24
    _FACT_TYPES = {"activity", "event", "observation"}
    _FACT_SOURCES = {"state_transition", "system_event", "inference", "tool", "dialogue", "memory_pipeline"}
    _FACT_CONFIDENCE = {"strong", "medium", "weak"}
    _DEFAULT_TTL_SECONDS = {
        "activity": 30 * 60,
        "event": 90 * 60,
        "observation": 20 * 60,
    }

    def __init__(self, workspace: Path):
        memory_dir = ensure_dir(workspace / "memory")
        self.fact_log_path = memory_dir / "LIFE_FACTS.jsonl"

    def append_fact(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        """Append one fact record with normalization, dedupe, prune, and capacity control."""
        record = self._normalize_fact(payload)
        if not record:
            return None

        records = self._load_records()
        touched = self._touch_recent_duplicate(records, record)
        if touched is not None:
            self._write_records(self._prune_records(records, now=now_local()))
            return touched

        records.append(record)
        records = self._prune_records(records, now=now_local())
        self._write_records(records)
        return record

    def read_facts(
        self,
        *,
        limit: int = 12,
        fact_types: list[str] | None = None,
        confidences: list[str] | None = None,
        publicly_answerable: bool | None = None,
        include_expired: bool = False,
    ) -> list[dict[str, Any]]:
        """Read facts with optional filter set and recency ordering."""
        records = self._load_records()
        target_types = {
            str(x).strip().lower()
            for x in (fact_types or [])
            if str(x).strip()
        }
        target_conf = {
            str(x).strip().lower()
            for x in (confidences or [])
            if str(x).strip()
        }
        now = now_local()

        out: list[dict[str, Any]] = []
        for item in records:
            if target_types and str(item.get("fact_type") or "").strip().lower() not in target_types:
                continue
            if target_conf and str(item.get("confidence") or "").strip().lower() not in target_conf:
                continue
            if publicly_answerable is not None and bool(item.get("publicly_answerable")) is not publicly_answerable:
                continue
            if not include_expired and self._is_expired(item, now=now):
                continue
            out.append(dict(item))

        out.sort(key=self._sort_key, reverse=True)
        if limit <= 0:
            return out
        return out[:limit]

    def prune(self) -> int:
        """Drop expired facts and enforce max capacity; return removed count."""
        records = self._load_records()
        pruned = self._prune_records(records, now=now_local())
        removed = len(records) - len(pruned)
        if removed:
            self._write_records(pruned)
        return removed

    def load_all(self) -> list[dict[str, Any]]:
        """Load all fact records without filtering."""
        return self._load_records()

    def replace_all(self, records: list[dict[str, Any]]) -> None:
        """Replace full fact log atomically."""
        self._write_records(self._prune_records(records, now=now_local()))

    def _load_records(self) -> list[dict[str, Any]]:
        if not self.fact_log_path.exists():
            return []
        rows: list[dict[str, Any]] = []
        try:
            for line in self.fact_log_path.read_text(encoding="utf-8").splitlines():
                text = line.strip()
                if not text:
                    continue
                try:
                    obj = json.loads(text)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    rows.append(obj)
        except Exception as exc:
            logger.warning("Life fact store: failed reading {}: {}", self.fact_log_path, exc)
            return []
        return rows

    def _write_records(self, records: list[dict[str, Any]]) -> None:
        self.fact_log_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [json.dumps(item, ensure_ascii=False) for item in records if isinstance(item, dict)]
        data = ("\n".join(lines) + "\n") if lines else ""
        tmp = self.fact_log_path.with_suffix(self.fact_log_path.suffix + ".tmp")
        tmp.write_text(data, encoding="utf-8")
        os.replace(tmp, self.fact_log_path)

    def _prune_records(self, records: list[dict[str, Any]], *, now) -> list[dict[str, Any]]:
        kept = [dict(x) for x in records if isinstance(x, dict) and not self._is_expired(x, now=now)]
        kept.sort(key=self._sort_key, reverse=True)
        if len(kept) > self._MAX_FACTS:
            kept = kept[: self._MAX_FACTS]
        kept.reverse()  # Keep file in chronological append order.
        return kept

    def _touch_recent_duplicate(
        self,
        records: list[dict[str, Any]],
        new_record: dict[str, Any],
    ) -> dict[str, Any] | None:
        """If near-identical recent fact exists, refresh it instead of appending another row."""
        if not records:
            return None
        now = now_local()
        content = str(new_record.get("content") or "")
        ftype = str(new_record.get("fact_type") or "")
        source = str(new_record.get("source") or "")
        conf = str(new_record.get("confidence") or "")
        pub = bool(new_record.get("publicly_answerable"))
        start_dt = parse_iso(new_record.get("start_at")) or now

        scanned = 0
        for existing in reversed(records):
            if scanned >= self._MAX_APPEND_DEDUPE_SCAN:
                break
            scanned += 1
            if self._is_expired(existing, now=now):
                continue
            if str(existing.get("content") or "") != content:
                continue
            if str(existing.get("fact_type") or "") != ftype:
                continue
            if str(existing.get("source") or "") != source:
                continue
            if str(existing.get("confidence") or "") != conf:
                continue
            if bool(existing.get("publicly_answerable")) is not pub:
                continue
            exist_start = parse_iso(existing.get("start_at")) or now
            if abs((start_dt - exist_start).total_seconds()) > 3 * 60:
                continue
            existing["expires_at"] = self._max_iso(
                str(existing.get("expires_at") or ""),
                str(new_record.get("expires_at") or ""),
            )
            meta = existing.get("metadata")
            if not isinstance(meta, dict):
                meta = {}
            meta["touch_count"] = int(meta.get("touch_count") or 0) + 1
            meta["last_touched_at"] = to_iso(now)
            existing["metadata"] = meta
            return existing
        return None

    def _normalize_fact(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        if not isinstance(payload, dict):
            return None
        content = str(payload.get("content") or "").strip()
        if not content:
            return None

        now = now_local()
        fact_type = str(payload.get("fact_type") or "observation").strip().lower()
        if fact_type not in self._FACT_TYPES:
            fact_type = "observation"
        source = str(payload.get("source") or "inference").strip().lower()
        if source not in self._FACT_SOURCES:
            source = "inference"
        confidence = str(payload.get("confidence") or "weak").strip().lower()
        if confidence not in self._FACT_CONFIDENCE:
            confidence = "weak"

        start_dt = parse_iso(payload.get("start_at")) or now
        end_dt = parse_iso(payload.get("end_at"))
        if end_dt and end_dt < start_dt:
            end_dt = start_dt

        ttl_raw = payload.get("ttl_seconds", payload.get("ttl"))
        ttl_seconds = self._coerce_ttl_seconds(ttl_raw, default=self._DEFAULT_TTL_SECONDS[fact_type])
        expires_dt = parse_iso(payload.get("expires_at"))
        if expires_dt is None:
            expires_dt = start_dt + timedelta(seconds=ttl_seconds)
        if expires_dt < start_dt:
            expires_dt = start_dt

        metadata = payload.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}

        return {
            "fact_id": str(payload.get("fact_id") or f"fact_{uuid.uuid4().hex}"),
            "fact_type": fact_type,
            "content": content,
            "source": source,
            "confidence": confidence,
            "publicly_answerable": bool(payload.get("publicly_answerable")),
            "start_at": to_iso(start_dt),
            "end_at": to_iso(end_dt) if end_dt else None,
            "expires_at": to_iso(expires_dt),
            "ttl_seconds": ttl_seconds,
            "metadata": metadata,
        }

    @staticmethod
    def _coerce_ttl_seconds(value: Any, *, default: int) -> int:
        if isinstance(value, bool):
            return default
        if isinstance(value, (int, float)):
            return max(30, min(7 * 24 * 3600, int(value)))
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return default
            try:
                parsed = int(float(text))
            except Exception:
                return default
            return max(30, min(7 * 24 * 3600, parsed))
        return default

    @staticmethod
    def _is_expired(item: dict[str, Any], *, now) -> bool:
        expires = parse_iso(item.get("expires_at"))
        if not expires:
            return False
        return expires <= now

    @staticmethod
    def _sort_key(item: dict[str, Any]) -> tuple[float, str]:
        start = parse_iso(item.get("start_at"))
        score = start.timestamp() if start else 0.0
        return score, str(item.get("fact_id") or "")

    @staticmethod
    def _max_iso(a: str, b: str) -> str:
        dt_a = parse_iso(a)
        dt_b = parse_iso(b)
        if dt_a and dt_b:
            return to_iso(dt_a if dt_a >= dt_b else dt_b)
        if dt_a:
            return to_iso(dt_a)
        if dt_b:
            return to_iso(dt_b)
        return ""
