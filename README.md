# 📬 Email Management RL Environment

An **OpenEnv-compatible reinforcement learning environment** that simulates a
real-world corporate email inbox.  An AI agent must classify, prioritise, and
reply to a stream of emails — skills that directly mirror the daily workload of
knowledge workers everywhere.

---

## 🌍 Real-World Use Case

Every professional drowns in email.  Studies show knowledge workers spend
**2–3 hours per day** just managing their inbox.  This environment trains an
agent to:

- **Filter spam** accurately without over-blocking legitimate mail
- **Surface high-priority messages** so critical items are never missed
- **Draft professional replies** that save time and maintain quality

A well-trained agent could power smart email assistants, enterprise inbox
products, or automated customer-support systems.

---

## 🏗️ Project Structure

```
email-rl-env/
│
├── env/
│   ├── __init__.py       – package exports
│   ├── models.py         – Pydantic data models (Observation, Action, Reward, …)
│   ├── tasks.py          – 10-email corpus + 3 task definitions
│   ├── graders.py        – deterministic scoring functions
│   └── email_env.py      – main OpenEnv environment class
│
├── inference.py          – baseline agent + structured logging
├── app.py                – Gradio web UI (HuggingFace Spaces entry point)
├── openenv.yaml          – OpenEnv specification file
├── requirements.txt      – Python dependencies
├── Dockerfile            – container build instructions
└── README.md             – you are here
```

---

## 📊 Observation Space

Each step the agent receives an `Observation` object:

| Field              | Type    | Description                                    |
|--------------------|---------|------------------------------------------------|
| `email_id`         | string  | Unique email identifier                        |
| `subject`          | string  | Email subject line                             |
| `sender`           | string  | Sender email address                           |
| `body`             | string  | Full email body                                |
| `urgency_hint`     | string? | Optional urgency cue (e.g. "deadline Friday")  |
| `timestamp`        | string  | ISO-8601 send time                             |
| `task`             | string  | Active task name                               |
| `difficulty`       | string  | easy / medium / hard                           |
| `step_number`      | int     | Current step index                             |
| `remaining_emails` | int     | Emails left in the episode                     |
| `instructions`     | string  | Human-readable task instructions               |

---

## ⚙️ Action Space

Three action types, one per task:

```python
# Task 1 – Spam Detection
Action(action_type="classify_email",   label="spam" | "important" | "normal")

# Task 2 – Prioritization
Action(action_type="prioritize_email", priority="high" | "medium" | "low")

# Task 3 – Auto-Reply
Action(action_type="reply_email",      reply_text="<your reply>")
```

Using the wrong action type for the active task incurs an immediate `-1.0`
penalty.

---

## 🏁 Tasks

### Task 1 – Spam Detection *(Easy)*

The inbox contains a mix of obvious spam, work emails, and newsletters.
The agent must call `classify_email` with the correct label for each one.

**Grading:**

| Prediction → Truth          | Score |
|-----------------------------|-------|
| Exact match                 | 1.0   |
| Important → Normal          | 0.3   |
| Normal → Important          | 0.2   |
| Spam → Normal               | 0.1   |
| Spam → Important            | 0.0   |
| Normal/Important → Spam     | 0.1–0.3 |

---

### Task 2 – Email Prioritisation *(Medium)*

Same inbox; the agent assigns a priority level.  Priority has a natural ordinal
(high > medium > low) so off-by-one errors receive partial credit.

**Grading:**

| Distance | Score |
|----------|-------|
| 0 (exact) | 1.0 |
| 1 (off by one) | 0.5 |
| 2 (off by two) | 0.0 |

---

### Task 3 – Auto-Reply Generation *(Hard)*

For non-spam emails the agent drafts a reply; for spam it returns `"no_reply"`.
Replies are scored on keyword coverage — how many of the expected signal words
appear in the response.

**Grading:**

| Condition                                   | Score                        |
|---------------------------------------------|------------------------------|
| Spam → `no_reply` / empty                   | 1.0                          |
| Spam → wrote a reply                        | 0.1                          |
| Non-spam → empty / `no_reply`               | 0.0                          |
| Non-spam → reply < 20 chars                 | 0.2                          |
| Non-spam → reply with keyword overlap *k*   | 0.2 + 0.8 × *k*              |

---

## 🎯 Reward Function

Grader scores are mapped to rewards in **[-1.0, 1.0]**:

| Score range | Reward | Meaning                  |
|-------------|--------|--------------------------|
| ≥ 0.9       | +1.0   | Correct                  |
| ≥ 0.5       | +0.5   | Partially correct        |
| ≥ 0.2       | +0.2   | Marginal                 |
| < 0.2       | -0.5   | Wrong                    |
| Wrong type  | -1.0   | Bad action for this task |

---

## 🛠️ Setup

### Requirements
- Python 3.10+
- pip

### Install

```bash
git clone https://github.com/your-username/email-rl-env
cd email-rl-env
pip install -r requirements.txt
```

---

## 🚀 Running the Inference Script

### Without an API key (rule-based fallback agent)

```bash
python inference.py
```

### With an OpenAI-compatible API

```bash
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export HF_TOKEN="sk-..."
python inference.py
```

### With a HuggingFace Inference Endpoint

```bash
export API_BASE_URL="https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3/v1"
export MODEL_NAME="mistralai/Mistral-7B-Instruct-v0.3"
export HF_TOKEN="hf_..."
python inference.py
```

---

## 📋 Expected Output

```
====================================================================
  Email Management RL Environment – Baseline Inference
====================================================================
  API_BASE_URL : https://api.openai.com/v1
  MODEL_NAME   : gpt-3.5-turbo
  HF_TOKEN     : NOT SET – using rule-based fallback
====================================================================

--------------------------------------------------------------------
[START] task=spam_detection env=EmailManagementEnv model=gpt-3.5-turbo
[STEP] step=1 action=classify_email(label=spam) reward=1.0000 done=false error=null
[STEP] step=2 action=classify_email(label=important) reward=1.0000 done=false error=null
...
[STEP] step=10 action=classify_email(label=normal) reward=1.0000 done=true error=null
[END] success=true steps=10 rewards=[1.0, 1.0, 1.0, ...]
[INFO] mean_reward=0.8500

[START] task=email_prioritization env=EmailManagementEnv model=gpt-3.5-turbo
...
[END] success=true steps=10 rewards=[...]

[START] task=auto_reply env=EmailManagementEnv model=gpt-3.5-turbo
...
[END] success=true steps=10 rewards=[...]

====================================================================
  FINAL SUMMARY
====================================================================
  ✓ PASS  spam_detection              mean_reward=0.8500  steps=10
  ✓ PASS  email_prioritization        mean_reward=0.8000  steps=10
  ✓ PASS  auto_reply                  mean_reward=0.6500  steps=10
--------------------------------------------------------------------
  Overall: PASSED
====================================================================
```

---

## 🐳 Docker

### Build

```bash
docker build -t email-rl-env .
```

### Run inference

```bash
docker run --rm \
  -e API_BASE_URL="https://api.openai.com/v1" \
  -e MODEL_NAME="gpt-4o-mini" \
  -e HF_TOKEN="sk-..." \
  email-rl-env
```

### Run Gradio UI

```bash
docker run --rm -p 7860:7860 email-rl-env python app.py
```

Then open http://localhost:7860

---

## 🤗 HuggingFace Spaces

Deploy directly to HuggingFace Spaces:

1. Create a new Space with **Gradio** SDK
2. Upload all project files
3. Set secrets: `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`
4. The Space auto-runs `app.py`

---

## 📖 API Reference

```python
from env import EmailManagementEnv, Action
from env.models import ActionType, EmailLabel, PriorityLevel

env = EmailManagementEnv(task_name="spam_detection")
obs = env.reset()                       # → Observation

action = Action(
    action_type=ActionType.CLASSIFY,
    label=EmailLabel.SPAM,
)
obs, reward, done, info = env.step(action)   # → StepResult components

state = env.state()                     # → EnvState snapshot
summary = env.summary()                 # → dict with episode stats
```

---

## 📜 License

MIT — free to use, modify, and distribute.

---

*Built for the Scaler AI Hackathon · v1.0.0*
