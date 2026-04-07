"""
app.py – Gradio web interface for the Email Management RL Environment.

This file is the HuggingFace Spaces entry point.  It wraps the environment
in a simple interactive UI so anyone can try the agent without a CLI.
"""

from __future__ import annotations

import json
import textwrap
from typing import Any

import gradio as gr

from env.email_env import EmailManagementEnv
from env.models import Action, ActionType, EmailLabel, PriorityLevel

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TASK_DESCRIPTIONS = {
    "spam_detection": "🗑️  Easy – Classify emails as spam / important / normal",
    "email_prioritization": "📊  Medium – Assign priority (high / medium / low)",
    "auto_reply": "✉️  Hard – Draft a professional reply to each email",
}


def _fmt_obs(obs: Any) -> str:
    if obs is None:
        return "✅ Episode complete – no more emails."
    lines = [
        f"📧 **Email ID**: {obs.email_id}",
        f"📌 **Subject**: {obs.subject}",
        f"👤 **From**: {obs.sender}",
        f"📅 **Timestamp**: {obs.timestamp}",
    ]
    if obs.urgency_hint:
        lines.append(f"⚡ **Urgency**: {obs.urgency_hint}")
    lines += [
        "",
        "**Body:**",
        "```",
        textwrap.fill(obs.body, width=72),
        "```",
        "",
        f"📬 Emails remaining: **{obs.remaining_emails}**",
        f"🔢 Step: **{obs.step_number + 1}**",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Gradio app
# ---------------------------------------------------------------------------

# We keep a single env instance per session via gr.State.

def start_episode(task_name: str, state: dict) -> tuple:
    env = EmailManagementEnv(task_name=task_name)
    obs = env.reset()
    state["env"]   = env
    state["obs"]   = obs
    state["log"]   = []
    state["step"]  = 0
    state["rewards"] = []
    return (
        _fmt_obs(obs),
        "Episode started. Make your first action above.",
        "",   # clear summary
        state,
    )


def take_action(
    task_name: str,
    label: str,
    priority: str,
    reply_text: str,
    state: dict,
) -> tuple:
    env: EmailManagementEnv = state.get("env")
    obs = state.get("obs")

    if env is None or obs is None:
        return "⚠️ Start an episode first.", "", "", state

    # Build action based on task
    try:
        if task_name == "spam_detection":
            action = Action(action_type=ActionType.CLASSIFY, label=EmailLabel(label))
        elif task_name == "email_prioritization":
            action = Action(action_type=ActionType.PRIORITIZE, priority=PriorityLevel(priority))
        else:
            action = Action(action_type=ActionType.REPLY, reply_text=reply_text or "no_reply")
    except Exception as exc:
        return obs and _fmt_obs(obs), f"❌ Invalid action: {exc}", "", state

    next_obs, reward, done, info = env.step(action)
    state["step"] += 1
    state["rewards"].append(reward.value)
    state["obs"] = next_obs

    reward_color = "🟢" if reward.value > 0 else ("🟡" if reward.value == 0 else "🔴")
    log_line = (
        f"Step {state['step']:>3} │ {reward_color} reward={reward.value:+.2f} │ "
        f"score={info.get('score', 0):.2f} │ {reward.explanation}"
    )
    state["log"].append(log_line)

    log_text = "\n".join(state["log"][-15:])  # last 15 lines

    if done:
        scores = state["rewards"]
        mean_r = sum(scores) / len(scores)
        summary_md = (
            f"### 🏁 Episode Complete\n"
            f"- **Steps**: {state['step']}\n"
            f"- **Mean reward**: {mean_r:.4f}\n"
            f"- **Total reward**: {sum(scores):.4f}\n"
            f"- **Result**: {'✅ PASSED' if mean_r > 0 else '❌ FAILED'}\n"
        )
        return "✅ All emails processed.", log_text, summary_md, state

    return _fmt_obs(next_obs), log_text, "", state


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

with gr.Blocks(title="Email Management RL Env", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 📬 Email Management RL Environment
        ### An OpenEnv-compatible reinforcement learning environment

        Choose a task, start an episode, and let the agent (or you!) process the inbox.
        """
    )

    session_state = gr.State({})

    with gr.Row():
        task_dropdown = gr.Dropdown(
            choices=list(TASK_DESCRIPTIONS.keys()),
            value="spam_detection",
            label="Task",
            info="Select task difficulty",
        )
        start_btn = gr.Button("▶ Start Episode", variant="primary")

    with gr.Row():
        obs_box = gr.Markdown(value="*Start an episode to see the first email.*", label="Current Email")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Action")
            label_radio    = gr.Radio(["spam", "important", "normal"], value="normal",  label="Classification label")
            priority_radio = gr.Radio(["high", "medium", "low"],       value="medium",  label="Priority level")
            reply_input    = gr.Textbox(lines=3, placeholder="Type your reply here…",   label="Reply text")
            action_btn     = gr.Button("⚡ Submit Action", variant="secondary")

        with gr.Column(scale=1):
            gr.Markdown("### Step Log")
            log_box     = gr.Textbox(lines=12, interactive=False, label="Recent steps")
            summary_box = gr.Markdown(value="")

    # Wiring
    start_btn.click(
        start_episode,
        inputs=[task_dropdown, session_state],
        outputs=[obs_box, log_box, summary_box, session_state],
    )

    action_btn.click(
        take_action,
        inputs=[task_dropdown, label_radio, priority_radio, reply_input, session_state],
        outputs=[obs_box, log_box, summary_box, session_state],
    )

    gr.Markdown(
        """
        ---
        *Built for the Scaler Hackathon · Email Management RL Environment v1.0*
        """
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
