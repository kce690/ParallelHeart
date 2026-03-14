"""Tests for unified input classification, budgeting, and evidence constraints."""

from nanobot.agent.loop import AgentLoop


def test_input_classifier_ping_social_state_task() -> None:
    assert AgentLoop._classify_input_intensity("嘻嘻") == "ping"
    assert AgentLoop._classify_input_intensity("哈哈") == "ping"
    assert AgentLoop._classify_input_intensity("嗯哼") == "ping"
    assert AgentLoop._classify_input_intensity("嗨，想我了吗") == "social"
    assert AgentLoop._classify_input_intensity("你在干什么呢") == "state"
    assert AgentLoop._classify_input_intensity("你知道行列式吗") == "task"


def test_task_budget_defaults_to_short_ack_without_explain() -> None:
    out = AgentLoop._enforce_reply_budget(
        "task",
        "你知道行列式吗",
        "知道。行列式是从方阵到标量的映射，满足多线性与交替性",
        has_recent_event=False,
    )
    assert out == "知道啊"


def test_evidence_constraint_removes_unsupported_detail() -> None:
    out = AgentLoop._apply_evidence_constraint(
        "你在干什么",
        "在家休息呢，刚整理完一些文件，顺便看看窗外",
        has_recent_event=False,
    )
    assert out is not None
    assert "整理" not in out
    assert "窗外" not in out


def test_social_budget_keeps_short_non_template_reply() -> None:
    out = AgentLoop._enforce_reply_budget(
        "social",
        "想我了吗",
        "想呀。你今天有什么安排吗？",
        has_recent_event=False,
    )
    assert out is not None
    assert len(out) <= 16
    assert "安排" not in out
