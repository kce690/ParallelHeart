from __future__ import annotations

from nanobot.agent.loop import AgentLoop


def test_reply_layer_body_profile_beats_generated_detail() -> None:
    layer = AgentLoop._infer_reply_evidence_layer(
        answer_slot="body_profile",
        final_reply="165cm左右",
        slot_floor_reply="165cm左右",
        current_activity_state=None,
        memory_evidence=[
            {"source_kind": "generated_detail", "recall_level": "detail", "text": "在外面"},
        ],
        memory_recall_level="detail",
    )
    assert layer == "body_profile_structured"


def test_reply_layer_fact_detail_beats_generated_detail() -> None:
    layer = AgentLoop._infer_reply_evidence_layer(
        answer_slot="previous_activity",
        final_reply="刚在弄吃的",
        slot_floor_reply=None,
        current_activity_state=None,
        memory_evidence=[
            {"source_kind": "generated_detail", "recall_level": "detail", "text": "看消息"},
            {"source_kind": "fact_layer_consolidation", "recall_level": "detail", "text": "在弄吃的"},
        ],
        memory_recall_level="detail",
    )
    assert layer == "fact_detail"


def test_fact_grounding_corrects_chat_drift() -> None:
    corrected = AgentLoop._enforce_fact_grounded_reply(
        answer_slot="current_activity",
        reply="在和你聊天呀，刚看到你消息",
        slot_floor_reply="这会儿在弄吃的呢",
        current_activity_state={
            "source": "fact_strong",
            "fact": "在弄吃的",
            "uncertain": False,
        },
        memory_evidence=[],
        memory_recall_level="none",
    )
    assert corrected is not None
    assert "吃" in corrected
    assert "聊天" not in corrected


def test_generated_detail_extraction_keeps_core_fragment_only() -> None:
    fragments, cleaned, dropped = AgentLoop._extract_generated_detail_fragments_debug(
        "current_activity",
        "刚才我在看你发来的消息呀你这一晚上",
    )
    assert fragments
    assert len(fragments) <= 2
    assert any("看消息" in x for x in fragments)
    assert all("你这一晚上" not in x for x in fragments)
    assert cleaned
    assert dropped or cleaned
