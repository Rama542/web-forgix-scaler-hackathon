"""
Data models for the Email Management RL Environment.

Typed Pydantic models defining the full observation, action, and reward
structures used throughout the environment.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class EmailLabel(str, Enum):
    SPAM      = "spam"
    IMPORTANT = "important"
    NORMAL    = "normal"


class PriorityLevel(str, Enum):
    HIGH   = "high"
    MEDIUM = "medium"
    LOW    = "low"


class ActionType(str, Enum):
    CLASSIFY   = "classify_email"
    PRIORITIZE = "prioritize_email"
    REPLY      = "reply_email"


class TaskDifficulty(str, Enum):
    EASY   = "easy"
    MEDIUM = "medium"
    HARD   = "hard"


# ---------------------------------------------------------------------------
# Shared clamp utility
# ---------------------------------------------------------------------------

def _clamp(v: Any) -> float:
    """Return a float strictly inside (0.0, 1.0) — never 0.0 or 1.0."""
    try:
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return 0.5
    except (TypeError, ValueError):
        return 0.5
    if f <= 0.0:
        return 0.001
    if f >= 1.0:
        return 0.999
    return f


# ---------------------------------------------------------------------------
# Core domain model
# ---------------------------------------------------------------------------

class EmailMessage(BaseModel):
    """Represents a single email in the inbox."""

    email_id:     str           = Field(..., description="Unique identifier for the email")
    subject:      str           = Field(..., description="Email subject line")
    sender:       str           = Field(..., description="Sender's email address")
    body:         str           = Field(..., description="Full email body text")
    urgency_hint: Optional[str] = Field(default=None, description="Optional urgency cue")
    timestamp:    str           = Field(..., description="ISO-8601 send time")

    # Ground-truth labels (hidden from agent, used by graders)
    _true_label:             Optional[str] = None
    _true_priority:          Optional[str] = None
    _ideal_reply_keywords:   List[str]     = []


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class Observation(BaseModel):
    """What the agent sees at each timestep."""

    email_id:         str
    subject:          str
    sender:           str
    body:             str
    urgency_hint:     Optional[str]  = None
    timestamp:        str
    task:             str            = Field(..., description="Current task name")
    difficulty:       TaskDifficulty
    step_number:      int            = Field(default=0, ge=0)
    remaining_emails: int            = Field(default=0, ge=0)
    instructions:     str            = Field(..., description="Task instructions shown to agent")


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class Action(BaseModel):
    """An action the agent can take."""

    action_type: ActionType
    label:       Optional[EmailLabel]   = None   # for classify_email
    priority:    Optional[PriorityLevel] = None  # for prioritize_email
    reply_text:  Optional[str]          = None   # for reply_email

    def validate_for_task(self, task_name: str) -> bool:
        mapping = {
            "spam_detection":      ActionType.CLASSIFY,
            "email_prioritization": ActionType.PRIORITIZE,
            "auto_reply":          ActionType.REPLY,
        }
        expected = mapping.get(task_name)
        return expected is None or self.action_type == expected


# ---------------------------------------------------------------------------
# Reward  ← KEY FIX: validator enforces (0, 1) at model construction time
# ---------------------------------------------------------------------------

class Reward(BaseModel):
    """Reward signal returned after each step."""

    # NOTE: removed gt/lt constraints from Field — we enforce via validator
    # so construction never raises a ValidationError on boundary values.
    value:       float            = Field(..., description="Reward strictly in (0, 1)")
    breakdown:   Dict[str, float] = Field(default_factory=dict)
    explanation: str              = Field(default="")

    @field_validator("value", mode="before")
    @classmethod
    def clamp_value(cls, v: Any) -> float:
        """Silently clamp to (0.001, 0.999) — never raise on boundary values."""
        return _clamp(v)

    @field_validator("breakdown", mode="before")
    @classmethod
    def clamp_breakdown(cls, d: Any) -> Dict[str, float]:
        """Clamp every numeric value in the breakdown dict."""
        if not isinstance(d, dict):
            return {}
        return {
            k: _clamp(v) if isinstance(v, (int, float)) else v
            for k, v in d.items()
        }


# ---------------------------------------------------------------------------
# Step result
# ---------------------------------------------------------------------------

class StepResult(BaseModel):
    """Full result returned by env.step()."""

    observation: Optional[Observation]
    reward:      Reward
    done:        bool
    info:        Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Environment state snapshot
# ---------------------------------------------------------------------------

class EnvState(BaseModel):
    """Complete environment state (returned by env.state())."""

    task_name:         str
    difficulty:        TaskDifficulty
    current_step:      int
    total_emails:      int
    emails_processed:  int
    cumulative_reward: float
    scores:            List[float] = Field(default_factory=list)
    done:              bool        = False