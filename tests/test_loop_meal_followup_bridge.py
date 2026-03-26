from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.loop import AgentLoop
from nanobot.bus.events import InboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMResponse


def _make_loop(tmp_path: Path, llm_response: LLMResponse) -> AgentLoop:
    bus = MessageBus()
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    provider.estimate_prompt_tokens.return_value = (10_000, "test")
    provider.chat_with_retry = AsyncMock(return_value=llm_response)
    loop = AgentLoop(
        bus=bus,
        provider=provider,
        workspace=tmp_path,
        model="test-model",
        context_window_tokens=65_536,
    )
    loop.provider.chat_with_retry = provider.chat_with_retry
    loop.tools.get_definitions = MagicMock(return_value=[])
    return loop


def _seed_food_commitment(loop: AgentLoop, msg: InboundMessage, *, fact: str, scene: str = "家里") -> None:
    loop._recent_state_commitments[msg.session_key] = {
        "slot": "current_activity",
        "fact": fact,
        "reply": fact,
        "event": {
            "activity": "吃饭",
            "subject": "吃的东西",
            "scene": scene,
            "status": "正在进行",
            "source": "snapshot_activity",
            "confidence": "medium",
            "publicly_answerable": True,
        },
        "source": "snapshot_activity",
        "rank": 2,
        "uncertain": False,
        "updated_at": datetime.now(),
        "expires_at": datetime.now() + timedelta(minutes=8),
    }


@pytest.mark.asyncio
async def test_meal_followup_bridges_current_activity_commitment(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path, LLMResponse(content="随便吃了点", tool_calls=[]))
    loop.life_state_service = MagicMock()
    loop.life_state_service.retrieve_memory_evidence = AsyncMock(
        return_value={"recall_level": "none", "evidence": [], "prompt_block": ""}
    )
    loop.life_state_service.record_generated_detail = AsyncMock(return_value=None)
    msg = InboundMessage(channel="qq", sender_id="u1", chat_id="c1", content="吃什么啊")
    _seed_food_commitment(loop, msg, fact="在弄吃的")

    out = await loop._process_message(msg)

    assert out is not None
    loop.life_state_service.retrieve_memory_evidence.assert_awaited_once()
    loop.life_state_service.record_generated_detail.assert_not_awaited()
    call = loop.provider.chat_with_retry.await_args
    messages = call.kwargs["messages"]
    soft_prompts = [m["content"] for m in messages if m.get("role") == "system" and "answer_slot hint:" in str(m.get("content") or "")]
    assert soft_prompts
    assert "answer_slot hint: meal" in soft_prompts[-1]
    assert "bridge_source: current_activity" in soft_prompts[-1]


@pytest.mark.asyncio
async def test_meal_followup_bridge_can_answer_location_from_current_activity_context(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path, LLMResponse(content="不告诉你", tool_calls=[]))
    loop.life_state_service = MagicMock()
    loop.life_state_service.retrieve_memory_evidence = AsyncMock(
        return_value={"recall_level": "none", "evidence": [], "prompt_block": ""}
    )
    loop.life_state_service.record_generated_detail = AsyncMock(return_value=None)
    msg = InboundMessage(channel="qq", sender_id="u1", chat_id="c1", content="在哪吃的")
    _seed_food_commitment(loop, msg, fact="刚吃饭了", scene="家里")
    loop._recent_state_commitments[msg.session_key]["event"]["status"] = "刚做完"

    out = await loop._process_message(msg)

    assert out is not None
    assert out.content == "在家里"
    loop.life_state_service.record_generated_detail.assert_not_awaited()


@pytest.mark.asyncio
async def test_standalone_meal_followup_without_food_context_does_not_bridge(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path, LLMResponse(content="你想吃什么？", tool_calls=[]))
    loop.life_state_service = MagicMock()
    loop.life_state_service.retrieve_memory_evidence = AsyncMock()

    out = await loop._process_message(
        InboundMessage(channel="qq", sender_id="u1", chat_id="c1", content="吃什么啊")
    )

    assert out is not None
    loop.life_state_service.retrieve_memory_evidence.assert_not_awaited()
