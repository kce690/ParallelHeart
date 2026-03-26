"""Data models for prehistory bootstrap generation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nanobot.companion.life_state.prehistory_templates import (
    DEFAULT_IDENTITY_FACTS,
    DEFAULT_INTERESTS,
    DEFAULT_PERSONALITY_TRAITS,
    DEFAULT_ROUTINE_PHASES,
)


@dataclass
class PrehistoryProfile:
    """Structured persona inputs for prehistory generation."""

    age_range: str = "20-29"
    life_stage: str = "young_adult"
    role: str = "student"
    gender: str = "female"
    gender_label: str = "女生"
    identity_style: str = "女性人设"
    city: str | None = None
    locale: str | None = None
    timezone: str | None = None
    personality_traits: list[str] = field(default_factory=lambda: list(DEFAULT_PERSONALITY_TRAITS))
    interests: list[str] = field(default_factory=lambda: list(DEFAULT_INTERESTS))
    routine_phases: list[str] = field(default_factory=lambda: list(DEFAULT_ROUTINE_PHASES))
    identity_facts: list[str] = field(default_factory=lambda: list(DEFAULT_IDENTITY_FACTS))
    seed_facts: list[str] = field(default_factory=list)
    relationship_with_user: bool = False
    relationship_stage: str = "unknown"
    relationship_tone: str = "neutral"
    relationship_trust: float = 0.0
    relationship_conflict_last7d: float = 0.0
    explicit_seed: int | None = None

    @classmethod
    def from_workspace(
        cls,
        workspace: Path,
        *,
        overrides: dict[str, Any] | None = None,
    ) -> "PrehistoryProfile":
        """Load persona profile from optional structured workspace files."""
        custom = _load_json_object(workspace / "PREHISTORY_PROFILE.json")
        identity_profile = _load_json_object(workspace / "IDENTITY_PROFILE.json")
        relationship = _load_json_object(workspace / "RELATIONSHIP.json")
        life_state = _load_json_object(workspace / "LIFESTATE.json")

        data = dict(custom)
        if isinstance(identity_profile, dict):
            for key in ("gender", "gender_label", "identity_style"):
                if identity_profile.get(key) not in (None, "") and key not in data:
                    data[key] = identity_profile.get(key)
        data.update(overrides or {})
        profile = cls()

        profile.age_range = _text(data.get("age_range"), profile.age_range)
        profile.life_stage = _text(data.get("life_stage"), profile.life_stage)
        profile.role = _text(data.get("role"), profile.role).lower()
        profile.gender = _text(data.get("gender"), profile.gender).lower()
        profile.gender_label = _text(data.get("gender_label"), profile.gender_label)
        profile.identity_style = _text(data.get("identity_style"), profile.identity_style)
        profile.city = _text_or_none(data.get("city"))
        profile.locale = _text_or_none(data.get("locale"))
        profile.timezone = _text_or_none(data.get("timezone"))

        profile.personality_traits = _list_text(
            data.get("personality_traits"),
            fallback=profile.personality_traits,
        )
        profile.interests = _list_text(
            data.get("interests"),
            fallback=profile.interests,
        )
        profile.routine_phases = _list_text(
            data.get("routine_phases"),
            fallback=profile.routine_phases,
        )
        profile.identity_facts = _list_text(
            data.get("identity_facts"),
            fallback=profile.identity_facts,
        )
        profile.seed_facts = _list_text(data.get("seed_facts"), fallback=[])
        profile.explicit_seed = _int_or_none(data.get("seed"))

        if isinstance(relationship, dict):
            stage = _text(relationship.get("stage"), profile.relationship_stage)
            profile.relationship_stage = stage
            profile.relationship_trust = _float01(relationship.get("trust"), profile.relationship_trust)
            profile.relationship_conflict_last7d = _float01(
                relationship.get("conflict_last7d"),
                profile.relationship_conflict_last7d,
            )
            profile.relationship_tone = _text(
                relationship.get("tone"),
                profile.relationship_tone,
            )

            explicit = data.get("relationship_with_user")
            if isinstance(explicit, bool):
                profile.relationship_with_user = explicit
            else:
                intimacy = _float01(relationship.get("intimacy"), 0.0)
                profile.relationship_with_user = (
                    stage not in {"unknown", "warming_up", "new", "stranger"}
                    or intimacy >= 0.55
                    or profile.relationship_trust >= 0.65
                )
        else:
            explicit = data.get("relationship_with_user")
            profile.relationship_with_user = bool(explicit) if isinstance(explicit, bool) else False

        if isinstance(life_state, dict):
            profile.city = profile.city or _text_or_none(life_state.get("city"))
            profile.timezone = profile.timezone or _text_or_none(life_state.get("timezone"))

        return profile

    def profile_hash(self) -> str:
        """Stable profile fingerprint for auditable generation metadata."""
        payload = {
            "age_range": self.age_range,
            "life_stage": self.life_stage,
            "role": self.role,
            "gender": self.gender,
            "gender_label": self.gender_label,
            "identity_style": self.identity_style,
            "city": self.city,
            "locale": self.locale,
            "timezone": self.timezone,
            "personality_traits": list(self.personality_traits),
            "interests": list(self.interests),
            "routine_phases": list(self.routine_phases),
            "identity_facts": list(self.identity_facts),
            "seed_facts": list(self.seed_facts),
            "relationship_with_user": self.relationship_with_user,
            "relationship_stage": self.relationship_stage,
            "relationship_tone": self.relationship_tone,
            "relationship_trust": round(float(self.relationship_trust), 4),
            "relationship_conflict_last7d": round(float(self.relationship_conflict_last7d), 4),
            "explicit_seed": self.explicit_seed,
        }
        text = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha1(text.encode("utf-8")).hexdigest()

    def to_metadata(self) -> dict[str, Any]:
        """Compact profile fields for prehistory metadata."""
        return {
            "age_range": self.age_range,
            "life_stage": self.life_stage,
            "role": self.role,
            "gender": self.gender,
            "gender_label": self.gender_label,
            "identity_style": self.identity_style,
            "city": self.city,
            "locale": self.locale,
            "timezone": self.timezone,
            "personality_traits": list(self.personality_traits),
            "interests": list(self.interests),
            "routine_phases": list(self.routine_phases),
            "identity_facts": list(self.identity_facts),
            "seed_facts": list(self.seed_facts),
            "relationship_with_user": self.relationship_with_user,
            "relationship_stage": self.relationship_stage,
            "relationship_tone": self.relationship_tone,
            "relationship_trust": self.relationship_trust,
            "relationship_conflict_last7d": self.relationship_conflict_last7d,
            "profile_hash": self.profile_hash(),
        }


@dataclass
class PrehistoryEvent:
    """Generated historical event aligned with raw memory ingestion format."""

    time: str
    type: str
    summary: str
    event_id: str = ""
    source: str = "prehistory"
    importance: float = 1.0
    self_relevance: float = 0.85
    relationship_relevance: float = 0.2
    emotional_weight: float = 0.3
    novelty: float = 0.4
    source_confidence: float = 0.92
    pinned: bool = False
    core_memory: bool = False
    gist: str = ""
    detail: str = ""
    location: str = ""
    activity: str = ""
    tags: list[str] = field(default_factory=list)

    def to_raw_event(self) -> dict[str, Any]:
        """Convert generated event to raw-event record for memory ingestion."""
        payload: dict[str, Any] = {
            "time": self.time,
            "type": self.type,
            "summary": self.summary,
            "source": self.source,
            "importance": float(self.importance),
            "self_relevance": float(self.self_relevance),
            "relationship_relevance": float(self.relationship_relevance),
            "emotional_weight": float(self.emotional_weight),
            "novelty": float(self.novelty),
            "source_confidence": float(self.source_confidence),
        }
        if self.event_id:
            payload["event_id"] = self.event_id
        if self.pinned:
            payload["pinned"] = True
        if self.core_memory:
            payload["core_memory"] = True
        if self.gist:
            payload["gist"] = self.gist
        if self.detail:
            payload["detail"] = self.detail
        if self.location:
            payload["location"] = self.location
        if self.activity:
            payload["activity"] = self.activity
        if self.tags:
            payload["tags"] = list(self.tags)
        return payload


@dataclass
class PrehistoryResult:
    """Output bundle persisted by prehistory bootstrap."""

    events: list[PrehistoryEvent]
    final_state: dict[str, Any]
    metadata: dict[str, Any]
    recent_log_events: list[PrehistoryEvent]
    summary: dict[str, Any]


def _load_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        raw = path.read_text(encoding="utf-8").strip()
        payload = json.loads(raw) if raw else {}
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass
    return {}


def _text(value: Any, default: str) -> str:
    if isinstance(value, (str, int, float, bool)):
        txt = str(value).strip()
        if txt:
            return txt
    return default


def _text_or_none(value: Any) -> str | None:
    if isinstance(value, (str, int, float, bool)):
        txt = str(value).strip()
        return txt or None
    return None


def _list_text(value: Any, *, fallback: list[str]) -> list[str]:
    if isinstance(value, str):
        parts = [x.strip() for x in value.split(",")]
        out = [x for x in parts if x]
        return out or list(fallback)
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            if isinstance(item, (str, int, float, bool)):
                txt = str(item).strip()
                if txt:
                    out.append(txt)
        return out or list(fallback)
    return list(fallback)


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value.strip()))
        except Exception:
            return None
    return None


def _float01(value: Any, default: float) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return max(0.0, min(1.0, float(value)))
    if isinstance(value, str):
        try:
            return max(0.0, min(1.0, float(value.strip())))
        except Exception:
            return default
    return default
