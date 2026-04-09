---
title: Email Management RL Env
emoji: 📧
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
---
# 📬 Email Management RL Environment

An **OpenEnv-compatible reinforcement learning environment** that simulates a
real-world corporate email inbox. An AI agent must classify, prioritise, and
reply to a stream of emails across three tasks of increasing difficulty.

---

## 🌍 Real-World Use Case

Every professional drowns in email. Studies show knowledge workers spend
**2–3 hours per day** just managing their inbox. This environment trains an
agent to:

- **Filter spam** accurately without over-blocking legitimate mail
- **Surface high-priority messages** so critical items are never missed
- **Draft professional replies** that save time and maintain quality

---

## 🏗️ Project Structure

```
email-rl-env/
│
├── env/
│   ├── __init__.py       – package exports
│   ├── models.py         – Pydantic data models
│   ├── tasks.py          – 10-email corpus + 3 task definitions
│   ├── graders.py        – deterministic scoring functions (scores strictly in (0,1))
│   └── email_env.py      – main OpenEnv environment class
│
├── server.py             – FastAPI server exposing /reset and /step
├── inference.py          – baseline agent + structured logging
├── app.py                – Gradio web UI
├── openenv.yaml          – OpenEnv specification file
├── requirements.txt      – Python dependencies
├── Dockerfile            – container (runs server.py via uvicorn)
└── README.md             – you are here
```

---

## 📊 Observation Space

| Field              | Type    | Description                                    |
|--------------------|---------|------------------------------------------------|
| `email_id`         | string  | Unique email identifier                        |
| `subject`          | string  | Email subject line                             |
| `sender`           | string  | Sender email address                           |
| `body`             | string  | Full email body                                |
| `urgency_hint`     | string? | Optional urgency cue                           |
| `timestamp`        | string  | ISO-8601 send time                             |
| `task`             | string  | Active task name                               |
| `difficulty`       | string  | easy / medium / hard                           |
| `step_number`      | int     | Current step index                             |
| `remaining_emails` | int     | Emails left in the episode                     |
| `instructions`     | string  | Human-readable task instructions               |

---

## ⚙️ Action Space

```python
# Task 1 – Spam Detection
Action(action_type="classify_email",   label="spam" | "important" | "normal")

# Task 2 – Prioritization
Action(action_type="prioritize_email", priority="high" | "medium" | "low")

# Task 3 – Auto-Reply
Action(action_type="reply_email",      reply_text="<your reply>")
```

---

## 🏁 Tasks

### Task 1 – Spam Detection *(Easy)*
Classify each email as spam, important, or normal.

### Task 2 – Email Prioritisation *(Medium)*
Assign priority: high, medium, or low.

### Task 3 – Auto-Reply Generation *(Hard)*
Draft professional replies; respond `no_reply` for spam.

---

## 🎯 Score Function

All scores are strictly in **(0.001, 0.999)** — never exactly 0.0 or 1.0.

| Grader outcome   | Score  |
|------------------|--------|
| Exact match      | 0.950  |
| Off by one       | 0.500  |
| Wrong            | 0.050–0.100 |

---

## 🚀 Running

### Server (what the validator uses)
```bash
uvicorn server:app --host 0.0.0.0 --port 7860
```

### Inference script
```bash
export HF_TOKEN="hf_..."
python inference.py
```

### Docker
```bash
docker build -t email-rl-env .
docker run --rm -p 7860:7860 email-rl-env
```

---

## 📖 API

```
POST /reset   { "task_name": "spam_detection" }
              → { "observation": {...}, "info": {...} }

POST /step    { "action": { "action_type": "classify_email", "label": "spam" } }
              → { "observation": {...}, "reward": 0.95, "score": 0.95, "done": false, "info": {...} }
```

---

*Built for the Scaler AI Hackathon · v1.0.0*