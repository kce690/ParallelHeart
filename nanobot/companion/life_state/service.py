"""Rule-based life-state service with sparse wake-up and offline catch-up."""

from __future__ import annotations

import asyncio
import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from loguru import logger


def _now_local() -> datetime:
    """Return local aware datetime without microseconds."""
    return datetime.now().astimezone().replace(microsecond=0)


def _to_iso(dt: datetime) -> str:
    """Serialize datetime to local ISO string."""
    return dt.astimezone().replace(microsecond=0).isoformat()


def _parse_iso(value: Any) -> datetime | None:
    """Parse ISO datetime while tolerating naive values."""
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip())
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=_now_local().tzinfo)
    return parsed.astimezone().replace(microsecond=0)


def _clamp_int(value: Any, low: int, high: int, default: int) -> int:
    """Clamp a value to [low, high] and fall back on invalid input."""
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return max(low, min(high, int(value)))
    if isinstance(value, str):
        try:
            parsed = int(float(value.strip()))
            return max(low, min(high, parsed))
        except Exception:
            return default
    return default


def _clamp_float(value: Any, low: float, high: float, default: float) -> float:
    """Clamp a float to [low, high]."""
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return max(low, min(high, float(value)))
    if isinstance(value, str):
        try:
            parsed = float(value.strip())
            return max(low, min(high, parsed))
        except Exception:
            return default
    return default


def _coerce_text(value: Any, default: str) -> str:
    """Coerce scalar value to text."""
    if value is None:
        return default
    if isinstance(value, (str, int, float, bool)):
        text = str(value).strip()
        return text or default
    return default


def _mood_to_score(value: Any) -> int:
    """Convert existing mood field to a numeric score."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return max(-5, min(5, int(value)))
    text = _coerce_text(value, "")
    if not text:
        return 1
    if any(k in text for k in ("低落", "烦", "丧", "差")):
        return -2
    if any(k in text for k in ("一般", "还行", "普通")):
        return 0
    if any(k in text for k in ("开心", "不错", "好")):
        return 3
    if any(k in text for k in ("平静", "平稳", "淡定")):
        return 1
    return 1


def _score_to_mood(score: int) -> str:
    """Map score to compact mood label."""
    if score <= -3:
        return "低落"
    if score <= -1:
        return "有点烦"
    if score == 0:
        return "一般"
    if score <= 2:
        return "平静"
    return "心情不错"


class LifeStateService:
    """State machine that advances companion life state without LLM calls."""

    _SCHEDULE_WINDOWS: tuple[tuple[int, int, str], ...] = (
        (0, 420, "deep_night"),     # 00:00-07:00
        (420, 510, "morning"),      # 07:00-08:30
        (510, 720, "study_am"),     # 08:30-12:00
        (720, 780, "lunch"),        # 12:00-13:00
        (780, 1080, "afternoon"),   # 13:00-18:00
        (1080, 1200, "dinner"),     # 18:00-20:00
        (1200, 1410, "evening"),    # 20:00-23:30
        (1410, 1440, "pre_sleep"),  # 23:30-24:00
    )
    _MIN_SLEEP_SECONDS = 5
    _MAX_CATCHUP_STEPS = 768

    def __init__(self, workspace: Path, enabled: bool = True):
        self.workspace = workspace
        self.enabled = enabled
        self.state_path = workspace / "LIFESTATE.json"
        self.life_log_path = workspace / "LIFELOG.md"
        self._running = False
        self._task: asyncio.Task | None = None
        self._lock = asyncio.Lock()
        self._rng = random.Random()

    @staticmethod
    def _default_state(now: datetime) -> dict[str, Any]:
        """Default life-state payload."""
        return {
            "location": "家",
            "activity": "休息",
            "mood": "平静",
            "mood_score": 1,
            "energy": 72,
            "social_battery": 66,
            "urgency_bias": 35,
            "busy_level": 40,
            "reply_delay_s": 8,
            "verbosity": 0.6,
            "last_tick": _to_iso(now),
            "next_transition_at": _to_iso(now + timedelta(minutes=35)),
            "last_update": _to_iso(now),  # Backward compatibility for first-cut readers.
            "override_until": None,
            "override_reason": None,
            "override_activity": None,
            "override_location": None,
            "override_busy_level": None,
        }

    def _load_state(self, now: datetime | None = None) -> dict[str, Any]:
        """Read state file and normalize to current schema."""
        now = now or _now_local()
        payload: dict[str, Any] = {}
        if self.state_path.exists():
            try:
                raw = self.state_path.read_text(encoding="utf-8").strip()
                loaded = json.loads(raw) if raw else {}
                if isinstance(loaded, dict):
                    payload = loaded
            except Exception as exc:
                logger.warning("LifeState: failed to parse {}: {}", self.state_path, exc)

        state = self._default_state(now)
        state.update(payload)

        state["location"] = _coerce_text(state.get("location"), "家")
        state["activity"] = _coerce_text(state.get("activity"), "休息")
        mood_score = _mood_to_score(state.get("mood_score", state.get("mood")))
        state["mood_score"] = max(-5, min(5, mood_score))
        state["mood"] = _score_to_mood(state["mood_score"])
        state["energy"] = _clamp_int(state.get("energy"), 0, 100, 72)
        state["social_battery"] = _clamp_int(state.get("social_battery"), 0, 100, 66)
        state["urgency_bias"] = _clamp_int(state.get("urgency_bias"), 0, 100, 35)
        state["busy_level"] = _clamp_int(state.get("busy_level"), 0, 100, 40)
        state["reply_delay_s"] = _clamp_int(state.get("reply_delay_s"), 0, 90, 8)
        state["verbosity"] = round(_clamp_float(state.get("verbosity"), 0.2, 0.95, 0.6), 2)
        state["override_reason"] = _coerce_text(state.get("override_reason"), "") or None
        state["override_activity"] = _coerce_text(state.get("override_activity"), "") or None
        state["override_location"] = _coerce_text(state.get("override_location"), "") or None
        state["override_busy_level"] = (
            _clamp_int(state.get("override_busy_level"), 0, 100, state["busy_level"])
            if state.get("override_busy_level") is not None
            else None
        )

        last_tick = _parse_iso(state.get("last_tick")) or _parse_iso(state.get("last_update"))
        if last_tick is None:
            last_tick = now
        state["last_tick"] = _to_iso(last_tick)
        state["last_update"] = _to_iso(last_tick)

        override_until = _parse_iso(state.get("override_until"))
        state["override_until"] = _to_iso(override_until) if override_until else None

        next_at = _parse_iso(state.get("next_transition_at"))
        if next_at is None:
            next_at = self._compute_next_transition(state, last_tick)
        state["next_transition_at"] = _to_iso(next_at)
        return state

    def _save_state(self, state: dict[str, Any]) -> None:
        """Persist normalized state to disk."""
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.state_path.write_text(
                json.dumps(state, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        except Exception as exc:
            logger.error("LifeState: failed to write {}: {}", self.state_path, exc)

    def _slot_at(self, now: datetime) -> tuple[str, int]:
        """Return current schedule slot and end minute-of-day."""
        minute_of_day = now.hour * 60 + now.minute
        for start, end, slot in self._SCHEDULE_WINDOWS:
            if start <= minute_of_day < end:
                return slot, end
        return "deep_night", 420

    def _compute_next_transition(self, state: dict[str, Any], now: datetime) -> datetime:
        """Compute sparse next wake-up time using schedule + jitter + override."""
        slot, end_minute = self._slot_at(now)
        day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        if end_minute >= 1440:
            end_minute = 1439
        candidate = day_start + timedelta(minutes=end_minute)
        if candidate <= now:
            candidate += timedelta(days=1)
            if end_minute == 420:  # deep-night slot wraps to next day 07:00
                candidate = day_start + timedelta(days=1, hours=7)

        jitter = self._rng.randint(-8, 10)
        if slot == "deep_night":
            jitter = self._rng.randint(5, 25)
        candidate = candidate + timedelta(minutes=jitter)

        min_allowed = now + timedelta(minutes=15)
        if candidate < min_allowed:
            candidate = min_allowed

        override_until = _parse_iso(state.get("override_until"))
        if override_until and override_until > now and override_until < candidate:
            candidate = override_until

        return candidate.replace(second=0, microsecond=0)

    def _derive_activity(self, state: dict[str, Any], now: datetime) -> tuple[str, str, int]:
        """Derive location/activity/busy from schedule + constraints + stochastic."""
        slot, _ = self._slot_at(now)
        energy = int(state.get("energy", 72))
        social = int(state.get("social_battery", 66))

        if slot == "deep_night":
            return "家", "睡觉", 10
        if slot == "morning":
            return "家", ("起床" if self._rng.random() < 0.55 else "吃饭"), 35
        if slot == "study_am":
            return ("学校" if self._rng.random() < 0.75 else "外面"), "学习", 78
        if slot == "lunch":
            return ("外面" if self._rng.random() < 0.6 else "家"), "吃饭", 40
        if slot == "afternoon":
            if energy < 28:
                return "家", "休息", 30
            candidates = [("学校", "学习", 75), ("家", "休息", 35), ("外面", "娱乐", 45), ("路上", "通勤", 60)]
            weights = [0.45, 0.25, 0.2, 0.1]
            if social < 25:
                weights = [0.35, 0.45, 0.1, 0.1]
            return self._rng.choices(candidates, weights=weights, k=1)[0]
        if slot == "dinner":
            if energy < 24:
                return "家", "休息", 20
            return ("家" if self._rng.random() < 0.7 else "外面"), ("吃饭" if self._rng.random() < 0.65 else "休息"), 32
        if slot == "evening":
            candidates = [("家", "娱乐", 28), ("家", "休息", 24), ("家", "学习", 55)]
            weights = [0.45, 0.35, 0.2]
            if int(state.get("urgency_bias", 35)) > 70:
                weights = [0.25, 0.25, 0.5]
            return self._rng.choices(candidates, weights=weights, k=1)[0]
        return "家", "睡前", 18

    def _activity_delta(self, activity: str) -> tuple[int, int]:
        """Return (energy_delta, social_delta)."""
        if activity == "睡觉":
            return self._rng.randint(12, 22), self._rng.randint(6, 12)
        if activity == "起床":
            return self._rng.randint(-1, 3), self._rng.randint(-1, 2)
        if activity == "吃饭":
            return self._rng.randint(4, 10), self._rng.randint(-1, 3)
        if activity == "学习":
            return self._rng.randint(-12, -5), self._rng.randint(-8, -3)
        if activity == "休息":
            return self._rng.randint(4, 9), self._rng.randint(4, 9)
        if activity == "通勤":
            return self._rng.randint(-10, -4), self._rng.randint(-7, -2)
        if activity == "娱乐":
            return self._rng.randint(-6, 1), self._rng.randint(-2, 4)
        if activity == "睡前":
            return self._rng.randint(-4, 1), self._rng.randint(0, 4)
        return 0, 0

    def _apply_override(self, state: dict[str, Any], now: datetime) -> None:
        """Apply temporary override if still active; clear it when expired."""
        override_until = _parse_iso(state.get("override_until"))
        if not override_until:
            return
        if override_until <= now:
            state["override_until"] = None
            state["override_reason"] = None
            state["override_activity"] = None
            state["override_location"] = None
            state["override_busy_level"] = None
            return
        if state.get("override_activity"):
            state["activity"] = _coerce_text(state.get("override_activity"), state["activity"])
        if state.get("override_location"):
            state["location"] = _coerce_text(state.get("override_location"), state["location"])
        if state.get("override_busy_level") is not None:
            state["busy_level"] = _clamp_int(state.get("override_busy_level"), 0, 100, state["busy_level"])

    def _event_from_transition(
        self,
        previous: dict[str, Any],
        state: dict[str, Any],
        at: datetime,
        source: str,
    ) -> dict[str, Any] | None:
        """Build a grounded short life event on meaningful changes."""
        prev_activity = _coerce_text(previous.get("activity"), "")
        prev_location = _coerce_text(previous.get("location"), "")
        activity = _coerce_text(state.get("activity"), "")
        location = _coerce_text(state.get("location"), "")

        summary: str | None = None
        importance = 1

        if activity != prev_activity:
            summary_map = {
                "睡觉": "这会儿睡了",
                "起床": "刚起床",
                "吃饭": "刚吃饭了",
                "学习": "这会儿在忙学习",
                "休息": "这会儿在家歇着" if location == "家" else "这会儿在歇会儿",
                "通勤": "在路上",
                "娱乐": "在放松",
                "睡前": "准备睡了",
            }
            summary = summary_map.get(activity)
            importance = 2
        elif location != prev_location:
            if location == "路上":
                summary = "刚出门在路上"
            elif location == "外面":
                summary = "这会儿在外面"
            elif location == "家":
                summary = "回到家了"
            importance = 2
        else:
            prev_energy = _clamp_int(previous.get("energy"), 0, 100, 60)
            energy = _clamp_int(state.get("energy"), 0, 100, 60)
            if prev_energy > 35 and energy <= 28:
                summary = "有点累了"
                importance = 1

        if not summary:
            return None

        return {
            "time": _to_iso(at),
            "type": "state_transition",
            "summary": summary,
            "source": source,
            "importance": importance,
        }

    def _append_event(self, event: dict[str, Any]) -> None:
        """Append event to LIFELOG.md in compact markdown."""
        summary = _coerce_text(event.get("summary"), "")
        when = _parse_iso(event.get("time")) or _now_local()
        if not summary:
            return
        if not self.life_log_path.exists():
            self.life_log_path.write_text(
                "# Life Log\n\nThis file stores recent life-state events that can inform natural self-status replies.\n",
                encoding="utf-8",
            )
        stamp = when.strftime("%Y-%m-%d %H:%M")
        line = f"- [{stamp}] {summary}\n"
        try:
            with self.life_log_path.open("a", encoding="utf-8") as fp:
                fp.write(line)
        except Exception as exc:
            logger.warning("LifeState: failed to append life event: {}", exc)

    def _advance_once(
        self,
        previous: dict[str, Any],
        tick_at: datetime,
        source: str,
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        """Advance state by one transition step."""
        state = dict(previous)

        location, activity, busy_level = self._derive_activity(state, tick_at)
        state["location"] = location
        state["activity"] = activity
        state["busy_level"] = busy_level

        self._apply_override(state, tick_at)

        energy_delta, social_delta = self._activity_delta(_coerce_text(state.get("activity"), "休息"))
        state["energy"] = _clamp_int(int(state.get("energy", 72)) + energy_delta, 0, 100, 72)
        state["social_battery"] = _clamp_int(int(state.get("social_battery", 66)) + social_delta, 0, 100, 66)

        mood_score = _clamp_int(state.get("mood_score"), -5, 5, _mood_to_score(state.get("mood")))
        mood_score += self._rng.choice((-1, 0, 1))
        if state["energy"] <= 25:
            mood_score -= 1
        if state["social_battery"] <= 20:
            mood_score -= 1
        if state["energy"] >= 70 and state["activity"] in {"休息", "娱乐", "吃饭"}:
            mood_score += 1
        mood_score = max(-5, min(5, mood_score))
        state["mood_score"] = mood_score
        state["mood"] = _score_to_mood(mood_score)

        if state["activity"] in {"学习", "通勤"}:
            urgency_base = self._rng.randint(62, 85)
        elif state["activity"] in {"睡觉", "睡前"}:
            urgency_base = self._rng.randint(10, 28)
        else:
            urgency_base = self._rng.randint(22, 56)
        state["urgency_bias"] = _clamp_int(urgency_base, 0, 100, 35)

        if state["busy_level"] >= 70 or state["activity"] in {"学习", "通勤"}:
            state["reply_delay_s"] = self._rng.randint(12, 30)
            verbosity = self._rng.uniform(0.3, 0.56)
        elif state["activity"] in {"睡觉", "睡前"}:
            state["reply_delay_s"] = self._rng.randint(18, 40)
            verbosity = self._rng.uniform(0.25, 0.48)
        else:
            state["reply_delay_s"] = self._rng.randint(4, 12)
            verbosity = self._rng.uniform(0.48, 0.78)
        if state["social_battery"] < 30:
            verbosity -= 0.1
        state["verbosity"] = round(max(0.2, min(0.9, verbosity)), 2)

        state["last_tick"] = _to_iso(tick_at)
        state["last_update"] = _to_iso(tick_at)
        state["next_transition_at"] = _to_iso(self._compute_next_transition(state, tick_at))

        event = self._event_from_transition(previous, state, tick_at, source)
        return state, event

    async def get_state(self) -> dict[str, Any]:
        """Return normalized current state snapshot."""
        async with self._lock:
            return self._load_state()

    async def get_recent_events(self, limit: int = 3) -> list[str]:
        """Read recent life events from LIFELOG.md."""
        if limit <= 0:
            return []
        if not self.life_log_path.exists():
            return []
        try:
            lines = self.life_log_path.read_text(encoding="utf-8").splitlines()
        except Exception:
            return []

        events: list[str] = []
        for raw in reversed(lines):
            line = raw.strip()
            if not line or line.startswith("#") or line.startswith("<!--"):
                continue
            if line.lower().startswith("this file stores"):
                continue
            if line.startswith("- "):
                line = line[2:].strip()
            if line.startswith("* "):
                line = line[2:].strip()
            if not line:
                continue
            # Keep only summary body for prompt injection.
            m = line.split("]", 1) if line.startswith("[") and "]" in line else None
            if m and len(m) == 2:
                line = m[1].strip()
            events.append(line)
            if len(events) >= limit:
                break
        return events

    async def fast_forward_to(self, now: datetime | None = None) -> int:
        """Offline catch-up state transitions until now. Returns step count."""
        now = (now or _now_local()).replace(microsecond=0)
        async with self._lock:
            state = self._load_state(now)
            steps = 0
            events: list[dict[str, Any]] = []
            while steps < self._MAX_CATCHUP_STEPS:
                next_at = _parse_iso(state.get("next_transition_at"))
                if not next_at:
                    next_at = self._compute_next_transition(state, _parse_iso(state.get("last_tick")) or now)
                    state["next_transition_at"] = _to_iso(next_at)
                if next_at > now:
                    break
                state, event = self._advance_once(state, next_at, source="offline")
                if event:
                    events.append(event)
                steps += 1
            self._save_state(state)
            for item in events:
                self._append_event(item)
            if steps:
                logger.info("LifeState: offline catch-up advanced {} step(s)", steps)
            return steps

    async def fast_forward_to_now(self) -> int:
        """Catch up transitions to current local time."""
        return await self.fast_forward_to(_now_local())

    async def step(self, now: datetime | None = None, source: str = "timer") -> dict[str, Any]:
        """Advance state once at `now` and persist."""
        at = (now or _now_local()).replace(microsecond=0)
        async with self._lock:
            current = self._load_state(at)
            updated, event = self._advance_once(current, at, source=source)
            self._save_state(updated)
            if event:
                self._append_event(event)
            return updated

    async def set_override(
        self,
        *,
        duration_minutes: int = 90,
        reason: str | None = None,
        activity: str | None = None,
        location: str | None = None,
        busy_level: int | None = None,
    ) -> dict[str, Any]:
        """Set temporary override and persist immediately."""
        duration_minutes = max(5, min(24 * 60, int(duration_minutes)))
        now = _now_local()
        until = now + timedelta(minutes=duration_minutes)
        async with self._lock:
            state = self._load_state(now)
            if activity is not None:
                state["override_activity"] = _coerce_text(activity, "") or None
            if location is not None:
                state["override_location"] = _coerce_text(location, "") or None
            if busy_level is not None:
                state["override_busy_level"] = _clamp_int(busy_level, 0, 100, state.get("busy_level", 40))
            state["override_reason"] = _coerce_text(reason, "") or state.get("override_reason")
            state["override_until"] = _to_iso(until)

            # Apply once immediately so context can reflect override right away.
            self._apply_override(state, now)
            state["last_tick"] = _to_iso(now)
            state["last_update"] = _to_iso(now)
            state["next_transition_at"] = _to_iso(min(_parse_iso(state["next_transition_at"]) or until, until))
            self._save_state(state)

            summary_parts = []
            if state.get("override_location"):
                summary_parts.append(f"在{state['override_location']}")
            if state.get("override_activity"):
                summary_parts.append(state["override_activity"])
            if not summary_parts and state.get("override_reason"):
                summary_parts.append(state["override_reason"])
            if summary_parts:
                self._append_event(
                    {
                        "time": _to_iso(now),
                        "type": "override",
                        "summary": f"临时状态：{'，'.join(summary_parts)}",
                        "source": "override",
                        "importance": 2,
                    }
                )
            return state

    async def clear_override(self) -> dict[str, Any]:
        """Clear override fields immediately."""
        now = _now_local()
        async with self._lock:
            state = self._load_state(now)
            state["override_until"] = None
            state["override_reason"] = None
            state["override_activity"] = None
            state["override_location"] = None
            state["override_busy_level"] = None
            state["last_tick"] = _to_iso(now)
            state["last_update"] = _to_iso(now)
            state["next_transition_at"] = _to_iso(self._compute_next_transition(state, now))
            self._save_state(state)
            return state

    async def start(self) -> None:
        """Start sparse wake-up loop."""
        if not self.enabled:
            logger.info("LifeState: disabled")
            return
        if self._running:
            logger.warning("LifeState: already running")
            return
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info("LifeState: service started")

    def stop(self) -> None:
        """Stop sparse wake-up loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            self._task = None

    async def _run_loop(self) -> None:
        """Sparse runtime loop: sleep until next_transition_at, then step."""
        await self.fast_forward_to_now()
        while self._running:
            try:
                state = await self.get_state()
                now = _now_local()
                next_at = _parse_iso(state.get("next_transition_at"))
                if not next_at or next_at <= now:
                    await self.step(now=now, source="timer")
                    continue
                wait_s = max(self._MIN_SLEEP_SECONDS, int((next_at - now).total_seconds()))
                await asyncio.sleep(wait_s)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("LifeState loop error: {}", exc)
                await asyncio.sleep(15)

