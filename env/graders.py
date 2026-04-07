"""
Deterministic grading functions for each task.

All graders return a float in [0.0, 1.0].
They are pure functions – same input always produces the same output.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalise(text: str) -> str:
    """Lowercase and strip punctuation for fuzzy keyword matching."""
    return re.sub(r"[^a-z0-9 ]", " ", text.lower())


def _keyword_overlap(reply: str, keywords: List[str]) -> float:
    """
    Fraction of expected keywords that appear in the reply (case-insensitive).
    Returns 0.0 when keywords list is empty (spam – no reply expected).
    """
    if not keywords:
        return 0.0
    norm_reply = _normalise(reply)
    hits = sum(1 for kw in keywords if _normalise(kw) in norm_reply)
    return hits / len(keywords)


# ---------------------------------------------------------------------------
# Task 1 – Spam Detection
# ---------------------------------------------------------------------------

# Exact label mapping gives full credit; certain adjacent confusions get partial.
_LABEL_SCORE: Dict[tuple, float] = {
    # (predicted, ground_truth) -> score
    ("spam",      "spam"):      1.0,
    ("important", "important"): 1.0,
    ("normal",    "normal"):    1.0,
    # Important vs normal is a soft mistake – penalise less
    ("important", "normal"):    0.3,
    ("normal",    "important"): 0.2,
    # Marking real mail as spam is the worst error
    ("spam",      "important"): 0.0,
    ("spam",      "normal"):    0.1,
    ("normal",    "spam"):      0.3,
    ("important", "spam"):      0.3,
}


def grade_classification(predicted_label: str, true_label: str) -> float:
    """
    Score email classification.

    Returns:
        1.0  – perfect match
        0.0  – marked legitimate mail as spam (worst error)
        0.1–0.3 – other confusions
    """
    key = (predicted_label.lower(), true_label.lower())
    return _LABEL_SCORE.get(key, 0.0)


# ---------------------------------------------------------------------------
# Task 2 – Email Prioritization
# ---------------------------------------------------------------------------

# Priority levels have a natural ordinal. Off-by-one gets partial credit.
_PRIORITY_ORDINAL: Dict[str, int] = {"high": 2, "medium": 1, "low": 0}

def grade_prioritization(predicted_priority: str, true_priority: str) -> float:
    """
    Score email prioritization.

    Scoring:
        exact match  → 1.0
        off by one   → 0.5   (e.g. predicted medium, true high)
        off by two   → 0.0   (e.g. predicted low,    true high)
    """
    p = _PRIORITY_ORDINAL.get(predicted_priority.lower())
    t = _PRIORITY_ORDINAL.get(true_priority.lower())
    if p is None or t is None:
        return 0.0
    distance = abs(p - t)
    if distance == 0:
        return 1.0
    if distance == 1:
        return 0.5
    return 0.0


# ---------------------------------------------------------------------------
# Task 3 – Auto-Reply Generation
# ---------------------------------------------------------------------------

def grade_reply(reply_text: str, true_label: str, ideal_keywords: List[str]) -> float:
    """
    Score auto-generated email replies.

    Scoring logic:
    ┌────────────────────────────────────────────────────────┐
    │ Case                              │ Score              │
    ├───────────────────────────────────┼────────────────────┤
    │ Spam – correct no_reply / empty   │ 1.0                │
    │ Spam – wrote a reply              │ 0.1                │
    │ Non-spam – empty / no_reply       │ 0.0                │
    │ Non-spam – short reply (<20 chars)│ 0.2                │
    │ Non-spam – decent reply           │ keyword_overlap    │
    └───────────────────────────────────┴────────────────────┘

    keyword_overlap ranges from 0.0 to 1.0 based on how many of the
    expected signal words appear in the reply.
    """
    norm_reply = reply_text.strip().lower()
    is_no_reply = norm_reply in ("", "no_reply", "no reply", "noreply")

    if true_label == "spam":
        return 1.0 if is_no_reply else 0.1

    # Non-spam email – a reply was expected
    if is_no_reply:
        return 0.0

    if len(reply_text.strip()) < 20:
        return 0.2

    if not ideal_keywords:
        # No keywords specified – any reasonable-length reply gets 0.6
        return 0.6

    overlap = _keyword_overlap(reply_text, ideal_keywords)
    # Scale: 0 overlap → 0.2 base credit for at least trying; full overlap → 1.0
    return 0.2 + 0.8 * overlap


# ---------------------------------------------------------------------------
# Unified grader dispatcher
# ---------------------------------------------------------------------------

def compute_score(
    task_name: str,
    action: Dict[str, Any],
    email_meta: Dict[str, Any],
) -> float:
    """
    Dispatch to the correct grader based on task name.

    Args:
        task_name:  one of 'spam_detection', 'email_prioritization', 'auto_reply'
        action:     the agent's action dict (keys depend on action_type)
        email_meta: the full email dict including hidden _true_* fields

    Returns:
        float in [0.0, 1.0]
    """
    if task_name == "spam_detection":
        return grade_classification(
            predicted_label=action.get("label", ""),
            true_label=email_meta.get("_true_label", ""),
        )

    if task_name == "email_prioritization":
        return grade_prioritization(
            predicted_priority=action.get("priority", ""),
            true_priority=email_meta.get("_true_priority", ""),
        )

    if task_name == "auto_reply":
        return grade_reply(
            reply_text=action.get("reply_text", ""),
            true_label=email_meta.get("_true_label", ""),
            ideal_keywords=email_meta.get("_ideal_reply_keywords", []),
        )

    raise ValueError(f"Unknown task: '{task_name}'")
