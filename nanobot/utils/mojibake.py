from __future__ import annotations

import re
from typing import Any

_STRONG_FRAGMENTS = (
    "\u701b\ufe3f",
    "\u6769\u6b0e",
    "\u934e\u57ae",
    "\u942b\u20ac",
    "\u7039?",
    "\u6d93\u5a03",
    "\u6d93\u6483",
)
_STRONG_CHARS = {
    "\u6769",  # 杩
    "\u934e",  # 鍎
    "\u942b",  # 鐫
    "\u9428",  # 鐨
    "\ufe3f",  # ︿
    "\u6d93",  # 涓
}
_WEAK_CHARS = {
    "\u701b",  # 瀛
}


def analyze_mojibake(text: Any) -> tuple[bool, str]:
    compact = re.sub(r"\s+", "", str(text or ""))
    if not compact:
        return False, "empty"
    if "\ufffd" in compact:
        return True, "replacement_char"
    for fragment in _STRONG_FRAGMENTS:
        if fragment in compact:
            return True, f"fragment:{fragment.encode('unicode_escape').decode()}"
    strong_hits = sum(1 for ch in compact if ch in _STRONG_CHARS)
    weak_hits = sum(1 for ch in compact if ch in _WEAK_CHARS)
    if strong_hits >= 2:
        return True, f"strong_hits:{strong_hits}"
    if strong_hits >= 1 and weak_hits >= 1:
        return True, f"mixed_hits:{strong_hits}+{weak_hits}"
    if weak_hits >= 2 and len(compact) <= 16:
        return True, f"weak_hits:{weak_hits}"
    return False, "clean"


def is_mojibake_text(text: Any) -> bool:
    return analyze_mojibake(text)[0]
