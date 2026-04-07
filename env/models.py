"""
Data models for the Email Management RL Environment.

Typed Pydantic models defining the full observation, action, and reward
structures used throughout the environment.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class EmailLabel(str, Enum):
    SPAM = "spam"
    IMPORTANT = "important"
    NORMAL = "normal"


class PriorityLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ActionType(str, Enum):
    CLASSIFY = "classify_email"
    PRIORITIZE = "prioritize_email"
    REPLY = "reply_email"


class TaskDifficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


# ---------------------------------------------------------------------------
# Core domain model
# ---------------------------------------------------------------------------

class EmailMessage(BaseModel):
    """Represents a single email in the inbox."""

    email_id: str = Field(..., description="Unique identifier for the email")
    subject: str = Field(..., description="Email subject line")
    sender: str = Field(..., description="Sender's email address")
    body: str = Field(..., description="Full email body text")
    urgency_hint: Optional[str] = Field(
        default=None,
        description="Optional urgency cue extracted from metadata (e.g. 'deadline today')"
    )
    timestamp: str = Field(..., description="ISO-8601 send time")

    # Ground-truth labels (hidden from agent, used by graders)
    _true_label: Optional[str] = None
    _true_priority: Optional[str] = None
    _ideal_reply_keywords: List[str] = []


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class Observation(BaseModel):
    """What the agent sees at each timestep."""

    email_id: str
    subject: str
    sender: str
    body: str
    urgency_hint: Optional[str] = None
    timestamp: str
    task: str = Field(..., description="Current task name")
    difficulty: TaskDifficulty
    step_number: int = Field(default=0, ge=0)
    remaining_emails: int = Field(default=0, ge=0)
    instructions: str = Field(..., description="Human-readable task instructions shown to the agent")


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class Action(BaseModel):
    """An action the agent can take."""

    action_type: ActionType
    # For classify_email
    label: Optional[EmailLabel] = None
    # For prioritize_email
    priority: Optional[PriorityLevel] = None
    # For reply_email
    reply_text: Optional[str] = None

    def validate_for_task(self, task_name: str) -> bool:
        """Light sanity-check that the action matches the active task type."""
        mapping = {
            "spam_detection": ActionType.CLASSIFY,
            "email_prioritization": ActionType.PRIORITIZE,
            "auto_reply": ActionType.REPLY,
        }
        expected = mapping.get(task_name)
        return expected is None or self.action_type == expected


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------

class Reward(BaseModel):
    """Reward signal returned after each step."""

    value: float = Field(..., ge=-1.0, le=1.0, description="Reward in [-1, 1]")
    breakdown: Dict[str, float] = Field(
        default_factory=dict,
        description="Per-component reward breakdown for interpretability"
    )
    explanation: str = Field(default="", description="Human-readable explanation")


# ---------------------------------------------------------------------------
# Step result
# ---------------------------------------------------------------------------

class StepResult(BaseModel):
    """Full result returned by env.step()."""

    observation: Optional[Observation]
    reward: Reward
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Environment state snapshot
# ---------------------------------------------------------------------------

class EnvState(BaseModel):
    """Complete environment state (returned by env.state())."""

    task_name: str
    difficulty: TaskDifficulty
    current_step: int
    total_emails: int
    emails_processed: int
    cumulative_reward: float
    scores: List[float] = Field(default_factory=list)
    done: bool = False
