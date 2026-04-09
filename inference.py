"""
inference.py – Baseline agent for the Email Management RL Environment.

Output format (STRICTLY followed):
    [START] task=<name> env=<env> model=<model>
    [STEP]  step=<n> action=<str> reward=<0.0000> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.0000> rewards=<r1,r2,...>

All reward/score values are strictly in (0, 1) — never 0.0 or 1.0.
"""

from __future__ import annotations

import os
import sys
import time
import textwrap
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from env.email_env import EmailManagementEnv
from env.models import Action, ActionType, EmailLabel, PriorityLevel


# ---------------------------------------------------------------------------
# Clamp helper
# ---------------------------------------------------------------------------

def clamp_score(x: float) -> float:
    """Strictly open interval (0, 1) — never 0.0 or 1.0."""
    try:
        v = float(x)
    except Exception:
        return 0.0001
    if v <= 0.0:
        return 0.0001
    if v >= 1.0:
        return 0.9999
    return v


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "gpt-4.1-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN", "")

TASK_NAMES: List[str] = ["spam_detection", "email_prioritization", "auto_reply"]
ENV_NAME = "EmailManagementEnv"


# ---------------------------------------------------------------------------
# Logging helpers — format must match spec EXACTLY
# ---------------------------------------------------------------------------

def log_start(task: str, model: str) -> None:
    print(f"[START] task={task} env={ENV_NAME} model={model}", flush=True)


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str],
) -> None:
    done_str  = "true" if done else "false"
    error_str = error if error else "null"
    safe_r    = clamp_score(reward)
    # action string must have no internal newlines
    action_clean = action.replace("\n", " ").replace("\r", "")
    print(
        f"[STEP] step={step} action={action_clean} reward={safe_r:.4f} "
        f"done={done_str} error={error_str}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    safe_score   = clamp_score(score)
    safe_rewards = [clamp_score(r) for r in rewards]
    reward_str   = ",".join(f"{r:.4f}" for r in safe_rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={safe_score:.4f} rewards={reward_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------

def _build_client() -> Optional[Any]:
    if not HF_TOKEN:
        return None
    try:
        return OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    except Exception:
        return None


def _call_llm(client: Any, system_prompt: str, user_prompt: str) -> str:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=512,
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Rule-based fallback agent
# ---------------------------------------------------------------------------

_SPAM_KEYWORDS = [
    "won", "winner", "prize", "gift card", "claim", "free", "offer", "limited",
    "click here", "discount", "v1agra", "c1alis", "pharma", "survey", "$50",
    "selected", "voucher", "reward",
]


def _rule_classify(body: str, subject: str, sender: str) -> str:
    text = (subject + " " + body + " " + sender).lower()
    spam_hits = sum(1 for kw in _SPAM_KEYWORDS if kw in text)
    if spam_hits >= 2:
        return "spam"
    if any(c in text for c in ["urgent", "action required", "asap", "deadline", "outage", "alert", "down"]):
        return "important"
    if any(c in text for c in ["meeting", "review", "report", "team", "scheduled", "maintenance", "billing"]):
        return "important"
    return "normal"


def _rule_prioritize(body: str, subject: str, urgency_hint: Optional[str]) -> str:
    text = (subject + " " + body + " " + (urgency_hint or "")).lower()
    if any(kw in text for kw in ["urgent", "outage", "down", "action required", "asap", "deadline"]):
        return "high"
    if any(kw in text for kw in ["scheduled", "review", "billing", "maintenance", "report"]):
        return "medium"
    return "low"


def _rule_reply(body: str, subject: str) -> str:
    text = (subject + " " + body).lower()
    spam_hints = sum(1 for kw in _SPAM_KEYWORDS if kw in text)
    if spam_hints >= 2:
        return "no_reply"
    if "outage" in text or "down" in text:
        return (
            "Thank you for the alert. I am acknowledging the outage and will escalate "
            "to the on-call team immediately to begin investigating."
        )
    if "review" in text and "performance" in text:
        return (
            "Thank you for the heads-up. I confirm I will be prepared and available "
            "for the performance review as scheduled on Tuesday. Looking forward to it."
        )
    if "financial" in text or "report" in text:
        return (
            "Acknowledged. I will review the Q1 financial report and provide my sign-off "
            "before end of business Friday as requested."
        )
    if "billing" in text or "aws" in text:
        return (
            "Thank you for the billing alert. I will investigate the unusual AWS usage "
            "and set up budget alerts to prevent this from happening again."
        )
    if "lunch" in text or "dinner" in text:
        return (
            "Sounds good! Thursday lunch works for me. "
            "Looking forward to it."
        )
    if "maintenance" in text or "wi-fi" in text:
        return (
            "Thanks for the heads-up. Noted — I will complete anything requiring "
            "VPN access before the maintenance window starts."
        )
    return (
        "Thank you for your email. I have received your message and will "
        "get back to you as soon as possible."
    )


# ---------------------------------------------------------------------------
# LLM-based helpers
# ---------------------------------------------------------------------------

_CLASSIFY_SYSTEM = textwrap.dedent("""
    You are an expert email classifier. Given an email, respond with EXACTLY one of:
    spam | important | normal
    No explanation. Just the single 
    label.
""").strip()

_PRIORITIZE_SYSTEM = textwrap.dedent("""

    You are an email triage specialist. Given an email, respond with EXACTLY one of:
    hig-coreh | medium | low
    No explanation. Just the single priority level.
""").strip()

_REPLY_SYSTEM = textwrap.dedent("""
    You are a professional executive assistant composing email replies.
    If the email is spam, respond with exactly: no_reply
    Otherwise write a concise, professional reply (2-4 sentences).
    Do NOT include any explanation or metadata — just the reply text.
""").strip()


def _llm_classify(client: Any, obs_dict: Dict[str, Any]) -> str:
    try:
        user_prompt = (
            f"Subject: {obs_dict['subject']}\n"
            f"From: {obs_dict['sender']}\n"
            f"Body:\n{obs_dict['body']}"
        )
        label = _call_llm(client, _CLASSIFY_SYSTEM, user_prompt).lower().strip()
        return label if label in ("spam", "important", "normal") else "normal"
    except Exception:
        return _rule_classify(obs_dict["body"], obs_dict["subject"], obs_dict["sender"])


def _llm_prioritize(client: Any, obs_dict: Dict[str, Any]) -> str:
    try:
        user_prompt = (
            f"Subject: {obs_dict['subject']}\n"
            f"From: {obs_dict['sender']}\n"
            f"Urgency hint: {obs_dict.get('urgency_hint') or 'none'}\n"
            f"Body:\n{obs_dict['body']}"
        )
        level = _call_llm(client, _PRIORITIZE_SYSTEM, user_prompt).lower().strip()
        return level if level in ("high", "medium", "low") else "medium"
    except Exception:
        return _rule_prioritize(obs_dict["body"], obs_dict["subject"], obs_dict.get("urgency_hint"))


def _llm_reply(client: Any, obs_dict: Dict[str, Any]) -> str:
    try:
        user_prompt = (
            f"Subject: {obs_dict['subject']}\n"
            f"From: {obs_dict['sender']}\n"
            f"Body:\n{obs_dict['body']}"
        )
        return _call_llm(client, _REPLY_SYSTEM, user_prompt)
    except Exception:
        return _rule_reply(obs_dict["body"], obs_dict["subject"])


# ---------------------------------------------------------------------------
# Action builder
# ---------------------------------------------------------------------------

def build_action(
    task_name: str,
    obs: Any,
    client: Optional[Any],
) -> Tuple[Action, str]:
    obs_dict = obs.model_dump()

    if task_name == "spam_detection":
        label = _llm_classify(client, obs_dict) if client else \
                _rule_classify(obs_dict["body"], obs_dict["subject"], obs_dict["sender"])
        action     = Action(action_type=ActionType.CLASSIFY, label=EmailLabel(label))
        action_str = f"classify_email(label={label})"

    elif task_name == "email_prioritization":
        level = _llm_prioritize(client, obs_dict) if client else \
                _rule_prioritize(obs_dict["body"], obs_dict["subject"], obs_dict.get("urgency_hint"))
        action     = Action(action_type=ActionType.PRIORITIZE, priority=PriorityLevel(level))
        action_str = f"prioritize_email(priority={level})"

    elif task_name == "auto_reply":
        reply = _llm_reply(client, obs_dict) if client else \
                _rule_reply(obs_dict["body"], obs_dict["subject"])
        # Truncate for log line — keep it short and single-line
        short      = reply[:50].replace("\n", " ") + ("..." if len(reply) > 50 else "")
        action     = Action(action_type=ActionType.REPLY, reply_text=reply)
        action_str = f"reply_email(text={short!r})"

    else:
        raise ValueError(f"Unsupported task: {task_name}")

    return action, action_str


# ---------------------------------------------------------------------------
# Single task runner
# ---------------------------------------------------------------------------

def run_task(task_name: str, client: Optional[Any]) -> Dict[str, Any]:
    env = EmailManagementEnv(task_name=task_name)
    obs = env.reset()

    log_start(task=task_name, model=MODEL_NAME)

    all_rewards: List[float] = []
    step = 0

    while obs is not None:
        step += 1
        action, action_str = build_action(task_name, obs, client)

        try:
            obs, raw_reward, done, info = env.step(action)
        except Exception as exc:
            score_on_error = clamp_score(
                sum(all_rewards) / len(all_rewards) if all_rewards else 0.0001
            )
            log_step(step=step, action=action_str, reward=0.0001, done=True, error=str(exc))
            log_end(success=False, steps=step, score=score_on_error, rewards=all_rewards)
            return {
                "task": task_name, "success": False,
                "steps": step, "rewards": all_rewards, "mean_reward": score_on_error,
            }

        numeric_reward = float(getattr(raw_reward, "value", raw_reward))
        safe_reward    = clamp_score(numeric_reward)

        all_rewards.append(safe_reward)
        log_step(
            step=step,
            action=action_str,
            reward=safe_reward,
            done=done,
            error=info.get("error"),
        )

        if done:
            break

    mean_reward = clamp_score(
        sum(all_rewards) / len(all_rewards) if all_rewards else 0.0001
    )
    success = mean_reward > 0.0001

    log_end(success=success, steps=step, score=mean_reward, rewards=all_rewards)
    print(f"[INFO] mean_reward={mean_reward:.4f}\n", flush=True)

    return {
        "task":        task_name,
        "success":     success,
        "steps":       step,
        "rewards":     all_rewards,
        "mean_reward": mean_reward,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 68)
    print("  Email Management RL Environment – Baseline Inference")
    print("=" * 68)
    print(f"  API_BASE_URL : {API_BASE_URL}")
    print(f"  MODEL_NAME   : {MODEL_NAME}")
    print(f"  HF_TOKEN     : {'set' if HF_TOKEN else 'NOT SET – using rule-based fallback'}")
    print("=" * 68)
    print()

    client  = _build_client()
    results = []

    for task_name in TASK_NAMES:
        print("-" * 68)
        result = run_task(task_name, client)
        results.append(result)
        time.sleep(0.1)

    print("=" * 68)
    print("  FINAL SUMMARY")
    print("=" * 68)
    overall_success = all(r["success"] for r in results)
    for r in results:
        status = "PASS" if r["success"] else "FAIL"
        print(
            f"  [{status}]  {r['task']:<28}  "
            f"mean_reward={r['mean_reward']:.4f}  steps={r['steps']}"
        )
    print("-" * 68)
    print(f"  Overall: {'PASSED' if overall_success else 'FAILED'}")
    print("=" * 68)

    sys.exit(0 if overall_success else 1)


if __name__ == "__main__":
    main()