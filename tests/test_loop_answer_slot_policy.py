"""Tests for answer-slot routing, low-info strategy, and policy guards."""

import inspect
import re
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


def test_route_answer_slot() -> None:
    assert AgentLoop._route_answer_slot("你在干什么", "state") == "current_activity"
    assert AgentLoop._route_answer_slot("你刚才在干什么", "state") == "previous_activity"
    assert AgentLoop._route_answer_slot("你午饭吃的什么", "state") == "meal"
    assert AgentLoop._route_answer_slot("你心情怎么样", "state") == "mood"
    assert AgentLoop._route_answer_slot("你现在方便吗", "state") == "availability"
    assert AgentLoop._route_answer_slot("这么晚还上课吗", "state") == "current_activity"
    assert AgentLoop._route_answer_slot("你是用什么写的", "task") == "meta_self"


@pytest.mark.parametrize("text", ["emmm", "我也是", "对呀", "好吧"])
def test_sparse_turns_no_longer_semantic_route_to_ack_slot(text: str) -> None:
    assert AgentLoop._route_answer_slot(text, "social") == "unknown"
    assert not AgentLoop._is_rule_first_slot("ack_social_followup")


@pytest.mark.asyncio
async def test_sparse_low_info_uses_llm_strategy_and_blocks_dead_ack(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path, LLMResponse(content="是呀", tool_calls=[]))
    out = await loop._process_message(
        InboundMessage(channel="qq", sender_id="u1", chat_id="c1", content="emmm")
    )

    assert out is not None
    assert out.content not in {"是呀", "对呀", "嗯嗯", "哈哈", "我懂"}
    assert loop.provider.chat_with_retry.await_count == 1


@pytest.mark.asyncio
async def test_phrase_like_sparse_turn_uses_low_info_llm_path(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path, LLMResponse(content="对呀", tool_calls=[]))
    out = await loop._process_message(
        InboundMessage(channel="qq", sender_id="u1", chat_id="c1", content="我也是")
    )

    assert out is not None
    assert out.content not in {"是呀", "对呀", "嗯嗯", "哈哈", "我懂"}
    assert loop.provider.chat_with_retry.await_count == 1


@pytest.mark.asyncio
async def test_low_info_blocks_self_narration_anchor(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path, LLMResponse(content="我在这儿陪你聊天", tool_calls=[]))
    out = await loop._process_message(
        InboundMessage(channel="qq", sender_id="u1", chat_id="c1", content="嗯")
    )

    assert out is not None
    assert "陪你聊天" not in out.content
    assert "等你" not in out.content


@pytest.mark.asyncio
async def test_low_info_blocks_explicit_intent_reading_phrasing(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path, LLMResponse(content="你是想我认真回吗", tool_calls=[]))
    out = await loop._process_message(
        InboundMessage(channel="qq", sender_id="u1", chat_id="c1", content="嗯")
    )

    assert out is not None
    assert "你是想" not in out.content
    assert out.content != "你是想我认真回吗"


@pytest.mark.asyncio
async def test_low_info_blocks_menu_style_probe(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path, LLMResponse(content="你是想闲聊，还是让我认真回", tool_calls=[]))
    out = await loop._process_message(
        InboundMessage(channel="qq", sender_id="u1", chat_id="c1", content="嗯")
    )

    assert out is not None
    assert "还是" not in out.content
    assert out.content != "你是想闲聊，还是让我认真回"


@pytest.mark.asyncio
async def test_low_info_prefers_contextual_probe_when_recent_context_exists(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path, LLMResponse(content="ok", tool_calls=[]))
    loop.provider.chat_with_retry.side_effect = [
        LLMResponse(content="好", tool_calls=[]),
        LLMResponse(content="是呀", tool_calls=[]),
    ]

    _ = await loop._process_message(
        InboundMessage(channel="qq", sender_id="u1", chat_id="c1", content="我今天跟老板聊崩了")
    )
    out = await loop._process_message(
        InboundMessage(channel="qq", sender_id="u1", chat_id="c1", content="嗯")
    )

    assert out is not None
    assert ("老板" in out.content) or ("聊崩" in out.content)


@pytest.mark.asyncio
async def test_repeated_low_info_turns_escalate_tone(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path, LLMResponse(content="嗯嗯", tool_calls=[]))

    out1 = await loop._process_message(
        InboundMessage(channel="qq", sender_id="u1", chat_id="c1", content="嗯")
    )
    out2 = await loop._process_message(
        InboundMessage(channel="qq", sender_id="u1", chat_id="c1", content="哦")
    )
    out3 = await loop._process_message(
        InboundMessage(channel="qq", sender_id="u1", chat_id="c1", content="啊")
    )

    assert out1 is not None and out2 is not None and out3 is not None
    assert out1.content != out2.content
    assert out2.content != out3.content
    assert any(mark in out3.content for mark in ("完整", "太短", "接不住", "吊我胃口"))
    assert not any(mark in out3.content for mark in ("连发问号", "一直戳我", "又在这样"))


def test_low_info_ungrounded_behavior_narration_is_blocked(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path, LLMResponse(content="ok", tool_calls=[]))
    narrated = "在看你连发问号呢"

    guarded = loop._apply_low_info_output_guard(
        "嗯",
        narrated,
        streak=3,
        intimacy_tier="high",
    )

    assert guarded != narrated
    assert "连发问号" not in guarded


def test_low_info_grounded_behavior_narration_requires_repetition_and_intimacy(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path, LLMResponse(content="ok", tool_calls=[]))
    narrated = "干嘛一直戳我"

    low_streak = loop._apply_low_info_output_guard(
        "戳戳你",
        narrated,
        streak=1,
        intimacy_tier="high",
    )
    low_intimacy = loop._apply_low_info_output_guard(
        "戳戳你",
        narrated,
        streak=3,
        intimacy_tier="low",
    )
    high_tier_repeated = loop._apply_low_info_output_guard(
        "戳戳你",
        narrated,
        streak=3,
        intimacy_tier="high",
    )

    assert low_streak != narrated
    assert low_intimacy != narrated
    assert high_tier_repeated == narrated


def test_low_info_aggressive_tone_is_gated_by_intimacy(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path, LLMResponse(content="ok", tool_calls=[]))
    aggressive = "你别烦我了"

    low_tier = loop._apply_low_info_output_guard(
        "嗯",
        aggressive,
        streak=3,
        intimacy_tier="low",
    )
    high_tier_but_not_repeated = loop._apply_low_info_output_guard(
        "嗯",
        aggressive,
        streak=2,
        intimacy_tier="high",
    )
    high_tier_repeated = loop._apply_low_info_output_guard(
        "嗯",
        aggressive,
        streak=3,
        intimacy_tier="high",
    )

    assert low_tier != aggressive
    assert high_tier_but_not_repeated != aggressive
    assert high_tier_repeated == aggressive


def test_low_info_detection_has_no_semantic_phrase_table_patterns() -> None:
    source = inspect.getsource(AgentLoop._is_low_info_turn)
    banned_patterns = ["帮我", "讲讲", "解释", "我也是", "好吧", "emmm", "对呀"]
    assert all(pattern not in source for pattern in banned_patterns)


@pytest.mark.asyncio
async def test_meta_self_default_uses_persona_reply_and_skips_llm(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path, LLMResponse(content="我是用Python写的，运行在Windows", tool_calls=[]))
    msg = InboundMessage(channel="qq", sender_id="u1", chat_id="c1", content="你是用什么写的")

    out = await loop._process_message(msg)

    assert out is not None
    text = out.content.lower()
    assert "python" not in text
    assert "windows" not in text
    assert loop.provider.chat_with_retry.await_count == 0


@pytest.mark.asyncio
async def test_meta_self_debug_mode_allows_technical_reply(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path, LLMResponse(content="我是用Python写的，运行在Windows", tool_calls=[]))
    msg = InboundMessage(
        channel="qq",
        sender_id="u1",
        chat_id="c1",
        content="我在调试你，现在用技术模式回答：你是用什么写的",
    )

    out = await loop._process_message(msg)

    assert out is not None
    text = out.content.lower()
    assert ("python" in text) or ("windows" in text)
    assert loop.provider.chat_with_retry.await_count == 1


@pytest.mark.asyncio
async def test_current_activity_uses_coarse_fallback_not_cue_mapping(tmp_path: Path) -> None:
    (tmp_path / "LIFESTATE.json").write_text(
        '{"location":"外面","activity":"通勤","mood":"平静","energy":70}',
        encoding="utf-8",
    )
    loop = _make_loop(tmp_path, LLMResponse(content="在外面呢", tool_calls=[]))
    out = await loop._process_message(
        InboundMessage(channel="qq", sender_id="u1", chat_id="c1", content="你在干什么")
    )

    assert out is not None
    assert out.content == "在路上"
    assert "上课" not in out.content
    assert loop.provider.chat_with_retry.await_count == 0


@pytest.mark.asyncio
async def test_state_slot_repeat_keeps_same_state_commitment(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path, LLMResponse(content="在外面待着呢", tool_calls=[]))

    out1 = await loop._process_message(
        InboundMessage(channel="qq", sender_id="u1", chat_id="c1", content="你在干什么")
    )
    out2 = await loop._process_message(
        InboundMessage(channel="qq", sender_id="u1", chat_id="c1", content="你在干什么")
    )

    assert out1 is not None and out2 is not None
    assert out1.content == out2.content


@pytest.mark.asyncio
async def test_state_related_followups_stay_on_same_source_chain(tmp_path: Path) -> None:
    (tmp_path / "LIFESTATE.json").write_text(
        '{"location":"学校","activity":"学习","mood":"平静","energy":70}',
        encoding="utf-8",
    )
    loop = _make_loop(tmp_path, LLMResponse(content="这会儿不走规则", tool_calls=[]))

    out1 = await loop._process_message(
        InboundMessage(channel="qq", sender_id="u1", chat_id="c1", content="你在干什么")
    )
    out2 = await loop._process_message(
        InboundMessage(channel="qq", sender_id="u1", chat_id="c1", content="这么晚还上课吗")
    )
    out3 = await loop._process_message(
        InboundMessage(channel="qq", sender_id="u1", chat_id="c1", content="你上课上到几点")
    )

    assert out1 is not None and out2 is not None and out3 is not None
    assert loop.provider.chat_with_retry.await_count == 0
    sig1 = AgentLoop._normalize_state_fact(out1.content)
    assert sig1
    assert sig1 == AgentLoop._normalize_state_fact(out2.content)
    assert sig1 == AgentLoop._normalize_state_fact(out3.content)


@pytest.mark.asyncio
async def test_current_activity_without_evidence_uses_vague_non_scene_reply(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path, LLMResponse(content="这条不会用到", tool_calls=[]))

    out = await loop._process_message(
        InboundMessage(channel="qq", sender_id="u1", chat_id="c1", content="你在干什么")
    )

    assert out is not None
    assert "上课" not in out.content
    assert re.search(r"(有点事|忙点事|弄点东西)", out.content)
    assert not re.search(r"(学校|教室|通勤|开会)", out.content)
    assert loop.provider.chat_with_retry.await_count == 0


@pytest.mark.asyncio
async def test_historical_memory_does_not_pollute_current_activity_resolution(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path, LLMResponse(content="杩欐潯涓嶄細鐢ㄥ埌", tool_calls=[]))
    stale_tick = (datetime.now().astimezone() - timedelta(hours=8)).replace(microsecond=0).isoformat()

    state = loop._resolve_current_activity_state(
        session_key="qq:c1",
        user_text="你现在在干嘛",
        snapshot={"activity": "在学线代", "last_tick": stale_tick},
        recent_events=[],
        memory_evidence=[{"recall_level": "detail", "text": "上午9点学线性代数，看到行列式"}],
        memory_recall_level="detail",
        prefer_recent_commitment=False,
    )

    assert state.get("source") == "uncertain"
    assert "线代" not in str(state.get("reply") or "")


def test_memory_recall_level_accepts_trace() -> None:
    assert AgentLoop._memory_recall_level({"recall_level": "trace"}) == "trace"


@pytest.mark.asyncio
async def test_state_correction_is_explicit_not_silent(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path, LLMResponse(content="这条不会用到", tool_calls=[]))

    out1 = await loop._process_message(
        InboundMessage(channel="qq", sender_id="u1", chat_id="c1", content="你在干什么")
    )
    assert out1 is not None
    assert "更正" not in out1.content

    (tmp_path / "LIFESTATE.json").write_text(
        '{"location":"路上","activity":"通勤","mood":"平静","energy":68}',
        encoding="utf-8",
    )
    out2 = await loop._process_message(
        InboundMessage(channel="qq", sender_id="u1", chat_id="c1", content="这么晚还上课吗")
    )

    assert out2 is not None
    assert "更正" in out2.content
    assert "上课" not in out2.content


@pytest.mark.asyncio
async def test_greeting_rule_first_does_not_use_state_or_llm(tmp_path: Path) -> None:
    (tmp_path / "LIFESTATE.json").write_text(
        '{"location":"外面","activity":"闲逛","mood":"平静","energy":70}',
        encoding="utf-8",
    )
    loop = _make_loop(tmp_path, LLMResponse(content="嗨，在外面随便逛", tool_calls=[]))
    out = await loop._process_message(
        InboundMessage(channel="qq", sender_id="u1", chat_id="c1", content="hi")
    )

    assert out is not None
    assert "外面" not in out.content
    assert loop.provider.chat_with_retry.await_count == 0


@pytest.mark.asyncio
async def test_meal_slot_prefers_meal_event_and_not_state(tmp_path: Path) -> None:
    (tmp_path / "LIFESTATE.json").write_text(
        '{"location":"外面","activity":"闲逛","mood":"平静","energy":70}',
        encoding="utf-8",
    )
    (tmp_path / "LIFELOG.md").write_text(
        "# Life Log\n- [2026-03-14 17:00] 刚吃饭了\n",
        encoding="utf-8",
    )
    loop = _make_loop(tmp_path, LLMResponse(content="在外面随便吃了点", tool_calls=[]))
    out = await loop._process_message(
        InboundMessage(channel="qq", sender_id="u1", chat_id="c1", content="你午饭吃的什么")
    )

    assert out is not None
    assert "吃" in out.content
    assert "外面" not in out.content
    assert loop.provider.chat_with_retry.await_count == 0
