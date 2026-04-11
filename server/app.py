import uvicorn
from fastapi import FastAPI, Request
from env.email_env import EmailManagementEnv
from env.models import Action

app = FastAPI(title="Email Management OpenEnv Server")
env_instance = None

def clamp(v):
    try:
        v = float(v)
    except:
        return 0.01
    if v <= 0.0: return 0.01
    if v >= 1.0: return 0.99
    return v

@app.post("/reset")
async def reset(request: Request):
    global env_instance
    try:
        body = await request.json()
    except:
        body = {}
    task_name = body.get("task_name", "spam_detection")
    env_instance = EmailManagementEnv(task_name=task_name)
    obs = env_instance.reset()
    return {"observation": obs.model_dump(), "info": {"task_name": task_name}}

@app.post("/step")
async def step(request: Request):
    global env_instance
    if env_instance is None:
        return {"error": "Call /reset first."}
    body = await request.json()
    action = Action(**body.get("action", {}))
    next_obs, reward, done, info = env_instance.step(action)
    raw = float(getattr(reward, "value", reward))
    safe = clamp(raw)
    return {
        "observation": next_obs.model_dump() if next_obs else None,
        "reward": safe,
        "score": safe,
        "done": done,
        "info": info
    }

@app.get("/state")
async def state():
    global env_instance
    if env_instance is None:
        return {"tasks": ["spam_detection", "email_prioritization", "auto_reply"]}
    return {"state": {}, "tasks": ["spam_detection", "email_prioritization", "auto_reply"]}

@app.get("/tasks")
async def tasks():
    return {
        "tasks": [
            {"id": "spam_detection", "grader": "grade_classification", "score_range": [0.01, 0.99]},
            {"id": "email_prioritization", "grader": "grade_prioritization", "score_range": [0.01, 0.99]},
            {"id": "auto_reply", "grader": "grade_reply", "score_range": [0.01, 0.99]},
        ]
    }

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/")
async def root():
    return {"status": "ready", "endpoints": ["POST /reset", "POST /step", "GET /state", "GET /tasks"]}

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=7860)
