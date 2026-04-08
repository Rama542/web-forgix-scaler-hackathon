"""
EmailManagementEnv – OpenEnv-compatible reinforcement learning environment.

Simulates an email inbox where an AI agent must classify, prioritise, and
reply to a stream of incoming emails across three tasks of increasing difficulty.
"""

from __future__ import annotations

import copy
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


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class EmailManagementEnv:
    """
    OpenEnv-compatible environment for email management tasks.

    Usage:
        env = EmailManagementEnv(task_name="spam_detection")
        obs  = env.reset()

        while True:
            action = agent.act(obs)
            obs, reward, done, info = env.step(action)
            if done:
                break
    """

    # Reward constants – strictly in (0.0, 1.0), no negatives allowed
    _REWARD_CORRECT    =  0.99
    _REWARD_PARTIAL_HI =  0.70
    _REWARD_PARTIAL_LO =  0.40
    _REWARD_WRONG      =  0.10
    _REWARD_BAD_ACTION =  0.01   # wrong action type for the task

    def __init__(self, task_name: str = "spam_detection") -> None:
        self._task_name = task_name
        self._task: Optional[Task] = None
        self._emails: List[Dict[str, Any]] = []
        self._cursor: int = 0
        self._cumulative_reward: float = 0.0
        self._scores: List[float] = []
        self._done: bool = False
        self._step_count: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> Observation:
        """
        Reset environment to the beginning of the task.

        Returns:
            The first observation.
        """
        self._task = get_task(self._task_name)
        self._emails = copy.deepcopy(self._task.emails)
        self._cursor = 0
        self._cumulative_reward = 0.0
        self._scores = []
        self._done = False
        self._step_count = 0
        return self._build_observation()

    def step(self, action: Action) -> Tuple[Optional[Observation], Reward, bool, Dict[str, Any]]:
        """
        Apply an action and advance the environment by one step.

        Args:
            action: An Action instance produced by the agent.

        Returns:
            (observation, reward, done, info)
            observation is None when done=True.
        """
        if self._done:
            raise RuntimeError("Episode is finished. Call reset() to start a new episode.")

        if self._task is None:
            raise RuntimeError("Environment not initialised. Call reset() first.")

        email = self._emails[self._cursor]
        reward, info = self._compute_reward(action, email)

        self._cumulative_reward += reward.value
        self._scores.append(info.get("score", 0.0))
        self._step_count += 1
        self._cursor += 1

        if self._cursor >= len(self._emails):
            self._done = True

        next_obs = None if self._done else self._build_observation()

        info.update({
            "email_id":         email["email_id"],
            "step":             self._step_count,
            "cumulative_reward": self._cumulative_reward,
            "mean_score":       sum(self._scores) / len(self._scores),
        })

        return next_obs, reward, self._done, info

    def state(self) -> EnvState:
        """Return a snapshot of the current environment state."""
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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_observation(self) -> Observation:
        """Construct the public-facing observation for the current email."""
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
        """
        Compute the reward for the given action on the given email.

        Wrong action type → immediate penalty.
        Otherwise → grader score is mapped to a reward in [-1, 1].
        """
        # --- validate action type -----------------------------------------
        if not action.validate_for_task(self._task_name):
            reward = Reward(
                value=self._REWARD_BAD_ACTION,
                breakdown={"type_penalty": self._REWARD_BAD_ACTION},
                explanation=(
                    f"Wrong action type '{action.action_type}' for task '{self._task_name}'. "
                    f"Expected action type matching the task."
                ),
            )
            return reward, {"score": 0.01, "error": "wrong_action_type"}

        # --- build action payload -----------------------------------------
        action_dict: Dict[str, Any] = {"action_type": action.action_type.value}
        if action.label is not None:
            action_dict["label"] = action.label.value
        if action.priority is not None:
            action_dict["priority"] = action.priority.value
        if action.reply_text is not None:
            action_dict["reply_text"] = action.reply_text

        # --- grade --------------------------------------------------------
        score = compute_score(
            task_name=self._task_name,
            action=action_dict,
            email_meta=email,
        )

        # --- map score to reward ------------------------------------------
        reward_value, explanation = self._score_to_reward(score, action, email)

        reward = Reward(
            value=reward_value,
            breakdown={"correctness": score, "scaled_reward": reward_value},
            explanation=explanation,
        )
        return reward, {"score": score, "error": None}

    def _score_to_reward(
        self,
        score: float,
        action: Action,
        email: Dict[str, Any],
    ) -> Tuple[float, str]:
        """
        Convert a [0, 1] grader score into a [-1, 1] reward value.

        Thresholds:
            score >= 0.9  → +1.0  (correct)
            score >= 0.5  → +0.5  (partially correct)
            score >= 0.2  → +0.2  (marginal)
            score <  0.2  → -0.5  (wrong)
        """
        if score >= 0.9:
            return self._REWARD_CORRECT, f"Excellent! Score {score:.2f} – fully correct."
        if score >= 0.5:
            return self._REWARD_PARTIAL_HI, f"Good. Score {score:.2f} – partially correct."
        if score >= 0.3:
            return self._REWARD_PARTIAL_LO, f"Marginal. Score {score:.2f} – barely correct."
        return self._REWARD_WRONG, f"Incorrect. Score {score:.2f} – wrong answer."

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def task_name(self) -> str:
        return self._task_name

    @property
    def is_done(self) -> bool:
        return self._done

    def summary(self) -> Dict[str, Any]:
        """Return a human-readable episode summary dict."""
        if not self._scores:
            return {"status": "not started"}
        return {
            "task":             self._task_name,
            "total_steps":      self._step_count,
            "mean_score":       round(sum(self._scores) / len(self._scores), 4),
            "cumulative_reward": round(self._cumulative_reward, 4),
            "scores":           [round(s, 4) for s in self._scores],
            "done":             self._done,
        }
