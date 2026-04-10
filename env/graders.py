# """
# Deterministic grading functions for each task.

# All graders return a float STRICTLY in (0.0, 1.0) – exclusive of both bounds.
# They are pure functions – same input always produces the same output.
# """

# from __future__ import annotations

# import re
# import math
# from typing import Any, Dict, List, Optional


# # ---------------------------------------------------------------------------
# # Master safety clamp – applied to EVERY return value in this module
# # ---------------------------------------------------------------------------

# def _safe(score: float) -> float:
#     """
#     Guarantee the score is a float strictly inside (0.0, 1.0).
#     Maps 0.0  → 0.001
#     Maps 1.0  → 0.999
#     Maps <0.0 → 0.001
#     Maps >1.0 → 0.999
#     """
#     try:
#         v = float(score)
#         if math.isnan(v):
#             v = 0.5
#     except (TypeError, ValueError):
#         v = 0.5
    
#     if v <= 0.0:
#         return 0.001
#     if v >= 1.0:
#         return 0.999
#     return v


# # ---------------------------------------------------------------------------
# # Helpers
# # ---------------------------------------------------------------------------

# def _normalise(text: str) -> str:
#     """Lowercase and strip punctuation for fuzzy keyword matching."""
#     return re.sub(r"[^a-z0-9 ]", " ", text.lower())


# def _keyword_overlap(reply: str, keywords: List[str]) -> float:
#     """
#     Fraction of expected keywords that appear in the reply (case-insensitive).
#     Always returns a value strictly in (0.0, 1.0).
#     """
#     if not keywords:
#         return 0.5   # neutral – no keywords to check against
#     norm_reply = _normalise(reply)
#     hits = sum(1 for kw in keywords if _normalise(kw) in norm_reply)
#     raw = hits / len(keywords)   # in [0.0, 1.0]
#     # Map [0,1] to (0.001, 0.999)
#     return _safe(raw * 0.998 + 0.001)


# # ---------------------------------------------------------------------------
# # Task 1 – Spam Detection
# # ---------------------------------------------------------------------------

# # All values are strictly in (0, 1)
# _LABEL_SCORE: Dict[tuple, float] = {
#     # (predicted, ground_truth) -> score
#     ("spam",      "spam"):      0.95,
#     ("important", "important"): 0.95,
#     ("normal",    "normal"):    0.95,
#     # Important vs normal is a soft mistake – penalise less
#     ("important", "normal"):    0.30,
#     ("normal",    "important"): 0.20,
#     # Marking real mail as spam is the worst error
#     ("spam",      "important"): 0.05,
#     ("spam",      "normal"):    0.10,
#     ("normal",    "spam"):      0.30,
#     ("important", "spam"):      0.30,
# }


# def grade_classification(predicted_label: str, true_label: str, *args, **kwargs) -> float:
#     """
#     Score email classification.

#     Returns a float strictly in (0, 1).
#     """
#     try:
#         key = (str(predicted_label).lower().strip(), str(true_label).lower().strip())
#         score = _LABEL_SCORE.get(key, 0.05)
#     except Exception:
#         score = 0.05
#     return _safe(score)


# # ---------------------------------------------------------------------------
# # Task 2 – Email Prioritization
# # ---------------------------------------------------------------------------

# _PRIORITY_ORDINAL: Dict[str, int] = {"high": 2, "medium": 1, "low": 0}


# def grade_prioritization(predicted_priority: str, true_priority: str, *args, **kwargs) -> float:
#     """
#     Score email prioritization.

#     Scoring:
#         exact match  -> 0.95
#         off by one   -> 0.50
#         off by two   -> 0.05
#     Returns a float strictly in (0, 1).
#     """
#     try:
#         p = _PRIORITY_ORDINAL.get(str(predicted_priority).lower().strip())
#         t = _PRIORITY_ORDINAL.get(str(true_priority).lower().strip())
#         if p is None or t is None:
#             return _safe(0.05)
#         distance = abs(p - t)
#         if distance == 0:
#             return _safe(0.95)
#         if distance == 1:
#             return _safe(0.50)
#         return _safe(0.05)
#     except Exception:
#         return _safe(0.05)


# # ---------------------------------------------------------------------------
# # Task 3 – Auto-Reply Generation
# # ---------------------------------------------------------------------------

# def grade_reply(reply_text: str, true_label: str, ideal_keywords: List[str] = None, *args, **kwargs) -> float:
#     """
#     Score auto-generated email replies.

#     Scoring logic:
#     ┌────────────────────────────────────────────────────────┐
#     │ Case                              │ Score              │
#     ├───────────────────────────────────┼────────────────────┤
#     │ Spam – correct no_reply / empty   │ 0.95               │
#     │ Spam – wrote a reply              │ 0.10               │
#     │ Non-spam – empty / no_reply       │ 0.05               │
#     │ Non-spam – short reply (<20 chars)│ 0.20               │
#     │ Non-spam – decent reply           │ keyword_overlap    │
#     └───────────────────────────────────┴────────────────────┘
#     All returned values are strictly in (0, 1).
#     """
#     try:
#         reply_text = str(reply_text) if reply_text is not None else ""
#         true_label = str(true_label).lower().strip() if true_label is not None else ""
#         if ideal_keywords is None:
#             ideal_keywords = []

#         norm_reply = reply_text.strip().lower()
#         is_no_reply = norm_reply in ("", "no_reply", "no reply", "noreply")

#         if true_label == "spam":
#             return _safe(0.95) if is_no_reply else _safe(0.10)

#         # Non-spam email – a reply was expected
#         if is_no_reply:
#             return _safe(0.05)

#         if len(reply_text.strip()) < 20:
#             return _safe(0.20)

#         if not ideal_keywords:
#             # No keywords specified – any reasonable-length reply gets partial credit
#             return _safe(0.60)

#         overlap = _keyword_overlap(reply_text, ideal_keywords)
#         # Scale: low overlap → 0.20 base credit; high overlap → 0.95
#         raw = 0.20 + 0.75 * overlap
#         return _safe(raw)

#     except Exception:
#         return _safe(0.05)


# # ---------------------------------------------------------------------------
# # Unified grader dispatcher
# # ---------------------------------------------------------------------------

# def compute_score(
#     task_name: str,
#     action: Dict[str, Any],
#     email_meta: Dict[str, Any],
# ) -> float:
#     """
#     Dispatch to the correct grader based on task name.

#     Returns:
#         float strictly in (0.01, 0.99)
#     """
#     try:
#         if task_name == "spam_detection":
#             result = grade_classification(
#                 predicted_label=action.get("label", ""),
#                 true_label=email_meta.get("_true_label", ""),
#             )

#         elif task_name == "email_prioritization":
#             result = grade_prioritization(
#                 predicted_priority=action.get("priority", ""),
#                 true_priority=email_meta.get("_true_priority", ""),
#             )

#         elif task_name == "auto_reply":
#             result = grade_reply(
#                 reply_text=action.get("reply_text", ""),
#                 true_label=email_meta.get("_true_label", ""),
#                 ideal_keywords=email_meta.get("_ideal_reply_keywords", []),
#             )

#         else:
#             result = 0.05

#         # Final safety net – must ALWAYS be strictly in (0, 1)
#         return _safe(result)

#     except Exception:
#         return _safe(0.05)






"""
Deterministic grading functions for each task.

All graders return a float STRICTLY in (0.0, 1.0) – exclusive of both bounds.
They are pure functions – same input always produces the same output.
"""

from __future__ import annotations

import re
import math
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Master safety clamp – applied to EVERY return value in this module
# ---------------------------------------------------------------------------

def _safe(score: float) -> float:
    """
    Guarantee the score is a float strictly inside (0.0, 1.0).
      <= 0.0  →  0.001
      >= 1.0  →  0.999
      NaN     →  0.500
    """
    try:
        v = float(score)
        if math.isnan(v) or math.isinf(v):
            return 0.5
    except (TypeError, ValueError):
        return 0.5

    if v <= 0.0:
        return 0.001
    if v >= 1.0:
        return 0.999
    return v


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalise(text: str) -> str:
    """Lowercase and strip punctuation for fuzzy keyword matching."""
    return re.sub(r"[^a-z0-9 ]", " ", text.lower())


def _keyword_overlap(reply: str, keywords: List[str]) -> float:
    """
    Fraction of expected keywords present in the reply (case-insensitive).
    Returns a value strictly in (0.001, 0.999).
    """
    if not keywords:
        return 0.5  # neutral — nothing to check against

    norm_reply = _normalise(reply)
    hits = sum(1 for kw in keywords if _normalise(kw) in norm_reply)
    raw = hits / len(keywords)  # in [0.0, 1.0]

    # Squeeze [0.0, 1.0] → (0.001, 0.999) then apply _safe as final guard
    return _safe(raw * 0.998 + 0.001)


#