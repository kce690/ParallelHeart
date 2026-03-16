"""Durable storage for raw life events and derived memory index."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any, Iterable

from loguru import logger

from nanobot.companion.life_state.memory_utils import now_local, to_iso
from nanobot.utils.helpers import ensure_dir


class LifeMemoryStore:
    """Persistent store with append-only raw log and atomic index writes."""

    _INDEX_VERSION = 1

    def __init__(self, workspace: Path):
        memory_dir = ensure_dir(workspace / "memory")
        self.raw_event_log_path = memory_dir / "LIFE_EVENTS.jsonl"
        self.memory_index_path = memory_dir / "LIFE_MEMORY_INDEX.json"

    def append_raw_event(self, event: dict[str, Any]) -> dict[str, Any]:
        """Append one immutable raw event record."""
        record = dict(event or {})
        record_id = str(record.get("event_id") or record.get("id") or f"evt_{uuid.uuid4().hex}")
        record["event_id"] = record_id
        now_iso = to_iso(now_local())
        stored_time = str(record.get("stored_time") or "").strip() or now_iso
        mentioned_time_raw = str(record.get("mentioned_time") or "").strip()
        mentioned_time = mentioned_time_raw or None
        event_time_start = str(record.get("event_time_start") or record.get("time") or "").strip()
        if not event_time_start:
            event_time_start = mentioned_time or stored_time
        event_time_end = str(record.get("event_time_end") or "").strip() or event_time_start

        record["event_time_start"] = event_time_start
        record["event_time_end"] = event_time_end
        record["mentioned_time"] = mentioned_time
        record["stored_time"] = stored_time
        record.setdefault("source_turn", str(record.get("source_turn") or ""))
        record.setdefault("source_kind", str(record.get("source_kind") or record.get("source") or ""))
        record.setdefault("time", event_time_start)
        line = json.dumps(record, ensure_ascii=False)

        self.raw_event_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.raw_event_log_path.open("a", encoding="utf-8") as fp:
            fp.write(line + "\n")
            fp.flush()
            try:
                os.fsync(fp.fileno())
            except OSError:
                pass
        return record

    def iter_raw_events(self) -> Iterable[dict[str, Any]]:
        """Iterate parsed raw events in append order."""
        if not self.raw_event_log_path.exists():
            return []
        records: list[dict[str, Any]] = []
        try:
            for line in self.raw_event_log_path.read_text(encoding="utf-8").splitlines():
                text = line.strip()
                if not text:
                    continue
                try:
                    obj = json.loads(text)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    records.append(obj)
        except Exception as exc:
            logger.warning("Life memory: failed to read raw events {}: {}", self.raw_event_log_path, exc)
        return records

    def load_memory_index(self) -> dict[str, Any]:
        """Load memory index JSON with safe defaults."""
        if not self.memory_index_path.exists():
            return self._default_index()
        try:
            raw = self.memory_index_path.read_text(encoding="utf-8").strip()
            payload = json.loads(raw) if raw else {}
            if not isinstance(payload, dict):
                return self._default_index()
            payload.setdefault("version", self._INDEX_VERSION)
            payload.setdefault("updated_at", to_iso(now_local()))
            payload.setdefault("entries", [])
            if not isinstance(payload.get("entries"), list):
                payload["entries"] = []
            return payload
        except Exception as exc:
            logger.warning("Life memory: failed to parse index {}: {}", self.memory_index_path, exc)
            return self._default_index()

    def save_memory_index(self, payload: dict[str, Any]) -> None:
        """Atomically write memory index JSON."""
        obj = dict(payload or {})
        obj["version"] = self._INDEX_VERSION
        obj["updated_at"] = to_iso(now_local())
        obj.setdefault("entries", [])
        data = json.dumps(obj, ensure_ascii=False, indent=2) + "\n"

        self.memory_index_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.memory_index_path.with_suffix(self.memory_index_path.suffix + ".tmp")
        tmp.write_text(data, encoding="utf-8")
        os.replace(tmp, self.memory_index_path)

    def _default_index(self) -> dict[str, Any]:
        return {
            "version": self._INDEX_VERSION,
            "updated_at": to_iso(now_local()),
            "entries": [],
        }
