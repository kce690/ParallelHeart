"""Life-state tools: get current state and set temporary overrides."""

from __future__ import annotations

import json
from typing import Any

from nanobot.agent.tools.base import Tool
from nanobot.companion.life_state.service import LifeStateService


class LifeStateGetTool(Tool):
    """Read current life-state snapshot."""

    def __init__(self, service: LifeStateService):
        self._service = service

    @property
    def name(self) -> str:
        return "life_state_get"

    @property
    def description(self) -> str:
        return "Get current life-state snapshot and recent grounded events."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {},
            "required": [],
        }

    async def execute(self, **kwargs: Any) -> str:
        state = await self._service.get_state()
        events = await self._service.get_recent_events(limit=3)
        payload = {
            "current": {
                "location": state.get("location"),
                "activity": state.get("activity"),
                "mood": state.get("mood"),
                "energy": state.get("energy"),
                "social_battery": state.get("social_battery"),
                "urgency_bias": state.get("urgency_bias"),
                "busy_level": state.get("busy_level"),
                "next_transition_at": state.get("next_transition_at"),
                "override_until": state.get("override_until"),
                "override_reason": state.get("override_reason"),
            },
            "recent_events": events,
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)


class LifeStateSetOverrideTool(Tool):
    """Set temporary life-state override for short-term context alignment."""

    def __init__(self, service: LifeStateService):
        self._service = service

    @property
    def name(self) -> str:
        return "life_state_set_override"

    @property
    def description(self) -> str:
        return "Set temporary override for life-state (activity/location/busy) with expiry."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "duration_minutes": {
                    "type": "integer",
                    "minimum": 5,
                    "maximum": 1440,
                    "description": "Override duration in minutes.",
                },
                "reason": {
                    "type": "string",
                    "description": "Optional short reason.",
                },
                "activity": {
                    "type": "string",
                    "description": "Override activity, e.g. 忙/休息/外出/通勤.",
                },
                "location": {
                    "type": "string",
                    "description": "Override location, e.g. 家/学校/外面/路上.",
                },
                "busy_level": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 100,
                    "description": "Override busy level.",
                },
                "clear": {
                    "type": "boolean",
                    "description": "Clear active override immediately.",
                },
            },
            "required": [],
        }

    async def execute(
        self,
        duration_minutes: int = 90,
        reason: str | None = None,
        activity: str | None = None,
        location: str | None = None,
        busy_level: int | None = None,
        clear: bool = False,
        **kwargs: Any,
    ) -> str:
        if clear:
            state = await self._service.clear_override()
        else:
            state = await self._service.set_override(
                duration_minutes=duration_minutes,
                reason=reason,
                activity=activity,
                location=location,
                busy_level=busy_level,
            )
        payload = {
            "ok": True,
            "override_until": state.get("override_until"),
            "override_reason": state.get("override_reason"),
            "location": state.get("location"),
            "activity": state.get("activity"),
            "busy_level": state.get("busy_level"),
            "next_transition_at": state.get("next_transition_at"),
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

