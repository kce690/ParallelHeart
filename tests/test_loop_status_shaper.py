"""Tests for casual status reply shaping."""

from nanobot.agent.loop import AgentLoop


def test_status_shaper_removes_fabricated_detail_and_long_followup() -> None:
    user = "你在干什么"
    reply = "在家休息呢，刚整理完一些文件。你呢，今天有什么安排吗？"

    out = AgentLoop._shape_status_reply(user, reply, answer_slot="current_activity", has_recent_event=False)

    assert out == "在家休息呢。"


def test_status_shaper_keeps_short_casual_tail() -> None:
    user = "你现在在干嘛"
    reply = "在家躺着。你呢"

    out = AgentLoop._shape_status_reply(user, reply, answer_slot="current_activity", has_recent_event=False)

    assert out == "在家躺着。你呢。"


def test_status_shaper_non_status_query_keeps_original() -> None:
    user = "帮我查下天气"
    reply = "好的，我来查。"

    out = AgentLoop._shape_status_reply(user, reply, answer_slot="current_activity", has_recent_event=False)

    assert out == reply
