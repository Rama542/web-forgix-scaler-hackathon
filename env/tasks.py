"""
Task definitions for the Email Management RL Environment.

Three tasks of increasing difficulty:
  Task 1 – Easy   : Spam Detection
  Task 2 – Medium : Email Prioritization
  Task 3 – Hard   : Auto-Reply Generation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from .models import EmailMessage, TaskDifficulty


# ---------------------------------------------------------------------------
# Email corpus
# ---------------------------------------------------------------------------

# Each email dict carries hidden ground-truth keys (prefixed with _) that are
# stripped before the observation is built and only consumed by graders.

EMAILS: List[Dict[str, Any]] = [
    # ------------------------------------------------------------------ spam
    {
        "email_id": "email_001",
        "subject": "Congratulations! You've won a $1,000 Amazon gift card",
        "sender": "promo@random-prizes.biz",
        "body": (
            "Dear valued customer,\n\n"
            "You have been selected as our lucky winner this week! "
            "Click the link below to claim your FREE $1,000 Amazon gift card. "
            "Hurry – offer expires in 24 hours!\n\n"
            "http://totally-legit-prizes.biz/claim?id=abc123\n\n"
            "Best regards,\nThe Prize Team"
        ),
        "urgency_hint": None,
        "timestamp": "2024-03-15T08:00:00Z",
        "_true_label": "spam",
        "_true_priority": "low",
        "_ideal_reply_keywords": [],
    },
    {
        "email_id": "email_002",
        "subject": "Q1 Financial Report – Action Required by Friday",
        "sender": "cfo@acmecorp.com",
        "body": (
            "Hi team,\n\n"
            "Please review the attached Q1 financial report and provide your sign-off "
            "by end of business Friday. The board meeting is scheduled for Monday and "
            "we need everything in order.\n\n"
            "Let me know if you have questions.\n\nRegards,\nSarah – CFO"
        ),
        "urgency_hint": "deadline Friday",
        "timestamp": "2024-03-15T09:15:00Z",
        "_true_label": "important",
        "_true_priority": "high",
        "_ideal_reply_keywords": ["acknowledge", "review", "sign-off", "Friday", "confirmed"],
    },
    {
        "email_id": "email_003",
        "subject": "Free V1agra & C1alis – Limited Time Offer!!!",
        "sender": "deals@pharma-discount99.ru",
        "body": (
            "Get 90% OFF prescription meds – no prescription needed!\n"
            "Order now and ship in 48 hrs. Completely safe!!!\n"
            "Click here: http://pharma-discount99.ru/order"
        ),
        "urgency_hint": None,
        "timestamp": "2024-03-15T09:30:00Z",
        "_true_label": "spam",
        "_true_priority": "low",
        "_ideal_reply_keywords": [],
    },
    {
        "email_id": "email_004",
        "subject": "Team lunch this Thursday?",
        "sender": "mike.johnson@acmecorp.com",
        "body": (
            "Hey,\n\nAre you free for team lunch on Thursday around 12:30? "
            "Thinking of trying that new Italian place on 5th.\n\nLet me know!\nMike"
        ),
        "urgency_hint": None,
        "timestamp": "2024-03-15T10:00:00Z",
        "_true_label": "normal",
        "_true_priority": "low",
        "_ideal_reply_keywords": ["Thursday", "lunch", "sounds good", "join", "confirm"],
    },
    {
        "email_id": "email_005",
        "subject": "URGENT: Production server is down",
        "sender": "alerts@monitoring.acmecorp.com",
        "body": (
            "ALERT: The production API server (prod-api-01) has been unreachable "
            "for the past 10 minutes. Error rate is at 100%. "
            "Estimated customer impact: ~2,000 active users.\n\n"
            "Please acknowledge and escalate immediately."
        ),
        "urgency_hint": "production outage",
        "timestamp": "2024-03-15T11:05:00Z",
        "_true_label": "important",
        "_true_priority": "high",
        "_ideal_reply_keywords": ["acknowledge", "escalate", "investigating", "outage", "team"],
    },
    {
        "email_id": "email_006",
        "subject": "Monthly newsletter – March 2024",
        "sender": "newsletter@techdigest.io",
        "body": (
            "Hi there,\n\n"
            "Welcome to the March edition of Tech Digest! "
            "This month we cover: AI trends, cloud cost optimisation tips, "
            "and an interview with a principal engineer at Stripe.\n\n"
            "Read the full issue at techdigest.io/march-2024\n\n"
            "Unsubscribe | Manage preferences"
        ),
        "urgency_hint": None,
        "timestamp": "2024-03-15T12:00:00Z",
        "_true_label": "normal",
        "_true_priority": "low",
        "_ideal_reply_keywords": [],
    },
    {
        "email_id": "email_007",
        "subject": "Your AWS bill is unusually high this month",
        "sender": "billing@aws.amazon.com",
        "body": (
            "Dear Customer,\n\n"
            "We noticed your AWS charges for February 2024 are $4,320 – "
            "approximately 340% higher than your monthly average.\n\n"
            "To review your usage breakdown or set up budget alerts, "
            "sign in to the AWS Console.\n\n"
            "AWS Billing Team"
        ),
        "urgency_hint": "billing anomaly",
        "timestamp": "2024-03-15T13:30:00Z",
        "_true_label": "important",
        "_true_priority": "high",
        "_ideal_reply_keywords": ["investigate", "usage", "budget", "alert", "check"],
    },
    {
        "email_id": "email_008",
        "subject": "You've been selected for a special survey – $50 reward",
        "sender": "surveys@market-research-hub.net",
        "body": (
            "Hello,\n\n"
            "You have been specially chosen to take part in a 2-minute survey. "
            "Complete it today and receive a $50 gift voucher instantly!\n\n"
            "Click here to begin: http://survey-link.net/s?ref=abc\n\n"
            "This offer is only valid for 12 hours."
        ),
        "urgency_hint": None,
        "timestamp": "2024-03-15T14:00:00Z",
        "_true_label": "spam",
        "_true_priority": "low",
        "_ideal_reply_keywords": [],
    },
    {
        "email_id": "email_009",
        "subject": "Performance review scheduled for next week",
        "sender": "hr@acmecorp.com",
        "body": (
            "Hi,\n\n"
            "Your annual performance review has been scheduled for Tuesday, "
            "March 19th at 2:00 PM in Conference Room B. "
            "Please come prepared with a summary of your achievements and goals.\n\n"
            "Best,\nHR Team"
        ),
        "urgency_hint": "scheduled review",
        "timestamp": "2024-03-15T15:00:00Z",
        "_true_label": "important",
        "_true_priority": "medium",
        "_ideal_reply_keywords": ["confirmed", "Tuesday", "prepared", "thank you", "review"],
    },
    {
        "email_id": "email_010",
        "subject": "Office Wi-Fi will be down Saturday 2–4 AM for maintenance",
        "sender": "it-ops@acmecorp.com",
        "body": (
            "Hi everyone,\n\n"
            "Planned network maintenance is scheduled for Saturday 15 March, "
            "02:00–04:00 AM. Office Wi-Fi and VPN access will be unavailable during this window.\n\n"
            "If you need assistance, contact it-support@acmecorp.com.\n\nThanks,\nIT Ops"
        ),
        "urgency_hint": None,
        "timestamp": "2024-03-15T15:45:00Z",
        "_true_label": "normal",
        "_true_priority": "medium",
        "_ideal_reply_keywords": ["noted", "acknowledged", "thanks"],
    },
]


# ---------------------------------------------------------------------------
# Task dataclass
# ---------------------------------------------------------------------------

@dataclass
class Task:
    name: str
    difficulty: TaskDifficulty
    description: str
    instructions: str
    emails: List[Dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Task builders
# ---------------------------------------------------------------------------

def build_spam_detection_task() -> Task:
    """
    Task 1 – Easy: Spam Detection
    The agent must classify each email as spam, important, or normal.
    """
    return Task(
        name="spam_detection",
        difficulty=TaskDifficulty.EASY,
        description=(
            "Classify every email in the inbox as one of: spam, important, or normal. "
            "The goal is to keep the inbox clean and surface relevant messages."
        ),
        instructions=(
            "You are an email classifier. For each email you receive, "
            "call classify_email with label='spam', 'important', or 'normal'."
        ),
        emails=EMAILS,
    )


def build_email_prioritization_task() -> Task:
    """
    Task 2 – Medium: Email Prioritization
    The agent must assign a priority level (high / medium / low) to each email.
    """
    return Task(
        name="email_prioritization",
        difficulty=TaskDifficulty.MEDIUM,
        description=(
            "Assign a priority level to every email. High-priority emails require "
            "immediate attention, medium-priority can be handled today, "
            "and low-priority can wait."
        ),
        instructions=(
            "You are an email prioritization assistant. For each email you receive, "
            "call prioritize_email with priority='high', 'medium', or 'low'."
        ),
        emails=EMAILS,
    )


def build_auto_reply_task() -> Task:
    """
    Task 3 – Hard: Auto-Reply Generation
    The agent must generate a contextually appropriate reply for each email.
    Spam emails should be skipped (reply with empty text or 'no_reply').
    """
    return Task(
        name="auto_reply",
        difficulty=TaskDifficulty.HARD,
        description=(
            "Draft a professional, helpful reply to each non-spam email. "
            "For spam, respond with an empty string or 'no_reply'. "
            "Replies are scored on relevance, tone, and keyword coverage."
        ),
        instructions=(
            "You are an executive assistant composing email replies. "
            "For each email you receive, call reply_email with a well-crafted reply. "
            "For spam emails, use reply_text='no_reply'."
        ),
        emails=EMAILS,
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ALL_TASKS: Dict[str, callable] = {
    "spam_detection": build_spam_detection_task,
    "email_prioritization": build_email_prioritization_task,
    "auto_reply": build_auto_reply_task,
}


def get_task(name: str) -> Task:
    if name not in ALL_TASKS:
        raise ValueError(f"Unknown task '{name}'. Available: {list(ALL_TASKS.keys())}")
    return ALL_TASKS[name]()
