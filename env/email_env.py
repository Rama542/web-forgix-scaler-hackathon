

"""
EmailManagementEnv – OpenEnv-compatible reinforcement learning environment.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Dict, List, Optional, Tuple

from .graders import compute_score
from .models import (
    Action,
    ActionType,
    EnvState,
    Observation,
    Reward,
    StepResult,
    TaskDifficulty,
)
from .tasks import Task, get_task


class EmailManagementEnv:

    # Reward constants – ALL strictly in (0.0, 1.0)
    _REWARD_CORRECT    = 0.95
    _REWARD_PARTIAL_HI = 0.70
    _REWARD_PARTIAL_LO = 0.40
    _REWARD_WRONG      = 0.10
    _REWARD_BAD_ACTION = 0.05

    @staticmethod
    def _safe(v: float) -> float:
        """Clamp to strictly open interval (0.0, 1.0) — never 0.0 or 1.0."""
        try:
            v = float(v)
            if math.isnan(v) or math.isinf(v):
                return 0.5
        except (TypeError, ValueError):
            return 0.5
        if v <= 0.0:
            return 0.001
        if v >= 1.0:
            return 0.999
        return v

    def __init__(self, task_name: str = "spam_detection") -> None:
        self._task_name = task_name
        self._task: Optional[Task] = None
        self._emails: List[Dict[str, Any]] = []
        self._cursor: int = 0
        self._cumulative_reward: float = 0.0
        self._scores: List[float] = []
        self._done: bool = False
        self._step_count: int = 0

    def reset(self) -> Observation:
        self._task = get_task(self._task_name)
        self._emails = copy.deepcopy(self._task.emails)
        self._cursor = 0
        self._cumulative_reward = 0.0
        self._scores = []
        self._done = False
        self._step_count = 0
        return self._build_observation()

    def step(self, action: Action) -> Tuple[Optional[Observation], Reward, bool, Dict[str, Any]]:
        if self._done:
            raise RuntimeError("Episode is finished. Call reset() to start a new episode.")
        if self._task is None:
            raise RuntimeError("Environment not initialised. Call reset() first.")

        email = self._emails[self._cursor]
        reward, info = self._compute_reward(action, email)

        # ✅ Always clamp reward.value before accumulating
        safe_val = self._safe(reward.value)
        reward = Reward(
            value=safe_val,
            breakdown=reward.breakdown,
            explanation=reward.explanation,
        )

        self._cumulative_reward += safe_val
        self._scores.append(self._safe(info.get("score", 0.5)))
        self._step_count += 1
        self._cursor += 1

        if self._cursor >= len(self._emails):
            self._done = True

        next_obs = None if self._done else self._build_observation()

        mean_score = self._safe(sum(self._scores) / len(self._scores))
        info.update({
            "email_id":          email["email_id"],
            "step":              self._step_count,
            "cumulative_reward": self._cumulative_reward,
            "mean_score":        mean_score,
            # ✅ Clamp score in info too
            "score":             self._safe(info.get("score", 0.5)),
        })

        return next_obs, reward, self._done, info

    def state(self) -> EnvState:
        if self._task is None:
            raise RuntimeError("Environment not initialised. Call reset() first.")
        return EnvState(
            task_name=self._task_name,
            difficulty=self._task.difficulty,
            current_step=self._step_count,
            total_emails=len(self._emails),
            emails_processed=self._cursor,
            cumulative_reward=self._cumulative_reward,
            scores=list(self._scores),
            done=self._done,
        )

    def _build_observation(self) -> Observation:
        assert self._task is not None
        email = self._emails[self._cursor]
        return Observation(
            email_id=email["email_id"],
            subject=email["subject"],
            sender=email["sender"],
            body=email["body"],
            urgency_hint=email.get("urgency_hint"),
            timestamp=email["timestamp"],
            task=self._task_name,
            difficulty=self._task.difficulty,
            step_number=self._step_count,
            remaining_emails=len(self._emails) - self._cursor,
            instructions=self._task.instructions,
        )

    def _compute_reward(
        self, action: Action, email: Dict[str, Any]
    ) -> Tuple[Reward, Dict[str, Any]]:

        if not action.validate_for_task(self._task_name):
            # ✅ _safe() on all values going into Reward
            bad_val = self._safe(self._REWARD_BAD_ACTION)
            reward = Reward(
                value=bad_val,
                breakdown={"type_penalty": bad_val},
                explanation=(
                    f"Wrong action type '{action.action_type}' for task "
                    f"'{self._task_name}'."
                ),
            )
            return reward, {"score": self._safe(0.05), "error": "wrong_action_type"}

        action_dict: Dict[str, Any] = {"action_type": action.action_type.value}
        if action.label is not None:
            action_dict["label"] = action.label.value
        if action.priority is not None:
            action_dict["priority"] = action.priority.value
        if action.reply_text is not None:
            action_dict["reply_text"] = action.reply_text

        # ✅ Clamp score immediately after grader returns
        raw_score = compute_score(
            task_name=self._task_name,
            action=action_dict,
            email_meta=email,
        )
        score = self._safe(raw_score)  # ← clamp HERE, right after grader

        reward_value, explanation = self._score_to_reward(score, action, email)
        # ✅ Clamp reward_value too before storing in Reward object
        safe_reward_value = self._safe(reward_value)

        reward = Reward(
            value=safe_reward_value,
            breakdown={
                "correctness":    score,           # already safe
                "scaled_reward":  safe_reward_value,
            },
            explanation=explanation,
        )
        return reward, {"score": score, "error": None}

    def _score_to_reward(
        self,
        score: float,   # already clamped by caller
        action: Action,
        email: Dict[str, Any],
    ) -> Tuple[float, str]:
        # All return values are hardcoded strictly inside (0, 1)
        if score >= 0.9:
            return self._REWARD_CORRECT,    f"Excellent! Score {score:.2f} – fully correct."
        if score >= 0.5:
            return self._REWARD_PARTIAL_HI, f"Good. Score {score:.2f} – partially correct."
        if score >= 0.3:
            return self._REWARD_PARTIAL_LO, f"Marginal. Score {score:.2f} – barely correct."
        return self._REWARD_WRONG,          f"Incorrect. Score {score:.2f} – wrong answer."

    @property
    def task_name(self) -> str:
        return self._task_name

    @property
    def is_done(self) -> bool:
        return self._done

    def close(self) -> None:
        """No-op teardown — satisfies server.py session cleanup calls."""
        pass

    def summary(self) -> Dict[str, Any]:
        if not self._scores:
            return {"status": "not started"}
        return {
            "task":              self._task_name,
            "total_steps":      self._step_count,
            "mean_score":       round(sum(self._scores) / len(self._scores), 4),
            "cumulative_reward": round(self._cumulative_reward, 4),
            "scores":           [round(s, 4) for s in self._scores],
            "done":             self._done,
        }