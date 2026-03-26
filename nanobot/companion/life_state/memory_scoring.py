"""Initial memory scoring, gist/detail generation, and permanence assignment."""

from __future__ import annotations

import hashlib
import re
from typing import Any

from nanobot.companion.life_state.memory_config import MemoryForgettingConfig
from nanobot.companion.life_state.memory_utils import compact_text, tokenize


_STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "at", "for",
    "is", "am", "are", "was", "were", "it", "this", "that", "just", "now",
    "i", "me", "my", "we", "our", "you", "your", "today", "tonight", "currently",
    "刚", "刚刚", "这会儿", "现在", "就", "了", "在", "有点",
}

_EMOTION_STRONG = (
    "love", "afraid", "panic", "anxious", "cry", "angry", "sad", "grief",
    "开心", "难过", "伤心", "害怕", "焦虑", "生气", "激动", "崩溃",
)
_EMOTION_MID = (
    "happy", "tired", "upset", "excited", "stressed",
    "累", "烦", "平静", "不错", "低落", "紧张",
)
_RELATION_KEYWORDS = (
    "friend", "family", "partner", "promise", "relationship", "user",
    "朋友", "家人", "恋人", "对象", "关系", "约定", "承诺", "你",
)
_PIN_KEYWORDS = (
    "identity", "promise", "goal", "trauma", "milestone",
    "身份", "承诺", "目标", "创伤", "纪念", "里程碑", "重要决定",
)
_ROUTINE_KEYWORDS = (
    "lunch", "dinner", "study", "rest", "commute", "sleep",
    "吃饭", "午饭", "晚饭", "上课", "学习", "休息", "通勤", "睡觉",
)


def score_event(
    event: dict[str, Any],
    cfg: MemoryForgettingConfig,
    *,
    cluster_pressure: float,
) -> dict[str, Any]:
    """Score one raw life event into memory entry fields."""
    summary = compact_text(str(event.get("summary") or ""))
    memory_type = classify_memory_type(event, summary)
    gist_summary = derive_gist_summary(event, summary=summary)
    detail_text = derive_detail_text(event, summary=summary)
    cluster_id = assign_similarity_cluster(memory_type, gist_summary)

    importance = _score_importance(event)
    self_relevance = _score_self_relevance(event)
    relationship_relevance = _score_relationship_relevance(summary)
    emotional_weight = _score_emotional_weight(summary)
    novelty = _score_novelty(cluster_pressure)
    source_confidence = _score_source_confidence(event)

    w = cfg.scoring
    salience = _clamp01(
        w.importance * importance
        + w.self_relevance * self_relevance
        + w.relationship_relevance * relationship_relevance
        + w.emotional_weight * emotional_weight
        + w.novelty * novelty
        + w.source_confidence * source_confidence
    )

    pinned_flag = _is_pinned_event(event, summary, salience, emotional_weight, relationship_relevance, cfg)
    permanence_tier = assign_permanence_tier(
        salience=salience,
        novelty=novelty,
        emotional_weight=emotional_weight,
        relationship_relevance=relationship_relevance,
        pinned_flag=pinned_flag,
        cfg=cfg,
    )

    m0 = salience
    detail_strength = _clamp01(m0 * cfg.strength.alpha_detail, hi=cfg.strength.max_strength)
    gist_strength = _clamp01(m0 * cfg.strength.alpha_gist, hi=cfg.strength.max_strength)

    return {
        "memory_type": memory_type,
        "gist_summary": gist_summary,
        "detail_text": detail_text,
        "importance": importance,
        "salience": salience,
        "self_relevance": self_relevance,
        "relationship_relevance": relationship_relevance,
        "emotional_weight": emotional_weight,
        "novelty": novelty,
        "source_confidence": source_confidence,
        "similarity_cluster_id": cluster_id,
        "similarity_cluster_pressure": max(0.0, cluster_pressure),
        "pinned_flag": pinned_flag,
        "permanence_tier": permanence_tier,
        "detail_strength": detail_strength,
        "gist_strength": gist_strength,
        "detail_strength_base": detail_strength,
        "gist_strength_base": gist_strength,
    }


def assign_similarity_cluster(memory_type: str, gist_summary: str) -> str:
    """Build a deterministic similarity-cluster id from type and gist tokens."""
    tokens = [tok for tok in tokenize(gist_summary) if tok not in _STOPWORDS]
    signature = "|".join(tokens[:5]) if tokens else "generic"
    digest = hashlib.sha1(signature.encode("utf-8")).hexdigest()[:10]
    return f"{memory_type}:{digest}"


def derive_gist_summary(event: dict[str, Any], *, summary: str) -> str:
    """Create semantic gist that can survive after detail fades."""
    explicit_gist = compact_text(str(event.get("gist") or ""))
    if explicit_gist:
        return explicit_gist

    source = compact_text(str(event.get("source") or ""))
    lowered = summary.lower()

    if re.search(r"(吃|饭|lunch|dinner|breakfast)", summary, flags=re.IGNORECASE):
        return "Had a meal in that period."
    if re.search(r"(学习|上课|study|class)", summary, flags=re.IGNORECASE):
        return "Was occupied with study/work tasks."
    if re.search(r"(睡|休息|sleep|rest)", summary, flags=re.IGNORECASE):
        return "Was resting and recovering energy."
    if re.search(r"(通勤|路上|commute|travel)", summary, flags=re.IGNORECASE):
        return "Was in transit between places."
    if source == "override":
        return "A temporary explicit state override happened."
    if "override" in lowered:
        return "A temporary explicit state override happened."
    if summary:
        cleaned = re.sub(r"^(just|currently|刚|刚刚|这会儿|现在)\s*", "", summary, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" .,!?:;")
        if cleaned:
            return f"General memory: {cleaned}."
    return "A life-state transition happened."


def derive_detail_text(event: dict[str, Any], *, summary: str) -> str:
    """Create detail channel text separate from gist."""
    explicit_detail = compact_text(str(event.get("detail") or ""))
    if explicit_detail:
        return explicit_detail

    parts = []
    if summary:
        parts.append(summary)
    if event.get("location"):
        parts.append(f"location={event.get('location')}")
    if event.get("activity"):
        parts.append(f"activity={event.get('activity')}")
    if event.get("source"):
        parts.append(f"source={event.get('source')}")
    if event.get("importance") is not None:
        parts.append(f"importance={event.get('importance')}")
    if event.get("override_reason"):
        parts.append(f"reason={event.get('override_reason')}")
    if not parts:
        return "Life-state event"
    return " | ".join(str(x) for x in parts if str(x).strip())


def classify_memory_type(event: dict[str, Any], summary: str) -> str:
    event_type = str(event.get("type") or "").strip().lower()
    if event_type:
        return event_type
    if re.search(r"(promise|承诺|约定)", summary, flags=re.IGNORECASE):
        return "promise"
    if re.search(r"(goal|目标|计划)", summary, flags=re.IGNORECASE):
        return "goal"
    if re.search(r"(identity|身份)", summary, flags=re.IGNORECASE):
        return "identity"
    if re.search(r"(吃饭|睡|学习|通勤|rest|study|commute|meal)", summary, flags=re.IGNORECASE):
        return "routine"
    return "life_event"


def assign_permanence_tier(
    *,
    salience: float,
    novelty: float,
    emotional_weight: float,
    relationship_relevance: float,
    pinned_flag: bool,
    cfg: MemoryForgettingConfig,
) -> str:
    if pinned_flag:
        return "pinned"
    if (
        salience >= cfg.permanence.durable_salience_threshold
        or emotional_weight >= 0.75
        or relationship_relevance >= 0.72
    ):
        return "durable"
    if salience <= cfg.permanence.volatile_salience_threshold and novelty < 0.45:
        return "volatile"
    return "normal"


def _score_importance(event: dict[str, Any]) -> float:
    raw = event.get("importance")
    if isinstance(raw, (int, float)):
        base = float(raw)
        if base > 1.0:
            return _clamp01(base / 3.0)
        return _clamp01(base)
    return 0.45


def _score_self_relevance(event: dict[str, Any]) -> float:
    raw = event.get("self_relevance")
    if isinstance(raw, (int, float)):
        return _clamp01(float(raw))
    return 0.85


def _score_relationship_relevance(text: str) -> float:
    lowered = (text or "").lower()
    strong = sum(1 for token in _RELATION_KEYWORDS if token in lowered)
    if strong >= 2:
        return 0.85
    if strong == 1:
        return 0.62
    return 0.30


def _score_emotional_weight(text: str) -> float:
    lowered = (text or "").lower()
    strong = sum(1 for token in _EMOTION_STRONG if token in lowered)
    mid = sum(1 for token in _EMOTION_MID if token in lowered)
    if strong >= 2:
        return 0.92
    if strong == 1:
        return 0.78
    if mid >= 2:
        return 0.64
    if mid == 1:
        return 0.52
    return 0.28


def _score_novelty(cluster_pressure: float) -> float:
    return _clamp01(1.0 / (1.0 + max(0.0, cluster_pressure)))


def _score_source_confidence(event: dict[str, Any]) -> float:
    explicit = event.get("source_confidence", event.get("certainty"))
    if isinstance(explicit, (int, float)) and not isinstance(explicit, bool):
        value = float(explicit)
        if value > 1.0:
            value = value / 100.0
        return _clamp01(value)
    source = str(event.get("source") or "").strip().lower()
    if source in {"override", "manual"}:
        return 0.95
    if source in {"model_generated"}:
        return 0.45
    if source in {"timer", "offline"}:
        return 0.82
    return 0.76


def _is_pinned_event(
    event: dict[str, Any],
    summary: str,
    salience: float,
    emotional_weight: float,
    relationship_relevance: float,
    cfg: MemoryForgettingConfig,
) -> bool:
    if bool(event.get("pinned") or event.get("core_memory")):
        return True
    lowered = (summary or "").lower()
    if any(token in lowered for token in _PIN_KEYWORDS):
        return True
    if salience >= cfg.permanence.pinned_salience_threshold and emotional_weight >= 0.78:
        return True
    return relationship_relevance >= 0.90 and emotional_weight >= 0.70


def looks_routine(text: str) -> bool:
    lowered = (text or "").lower()
    return any(token in lowered for token in _ROUTINE_KEYWORDS)


def _clamp01(value: float, *, hi: float = 1.0) -> float:
    return max(0.0, min(hi, float(value)))
