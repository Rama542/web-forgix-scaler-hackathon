from fastapi import FastAPI, Request
from pydantic import BaseModel
import math
from typing import Any, Dict, Optional

from env.email_env import EmailManagementEnv
from env.models import Action

app = FastAPI(title="Email Management OpenEnv Server")
env_instance = None


def _clamp(v: Any) -> float:
    try:
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return 0.5
    except Exception:
        return 0.5
    if f <= 0.0:
        return 0.001
    if f >= 1.0:
        return 0.999
    return f


@app.post("/reset")
async def reset(request: Request):
    global env_instance
    try:
        body = await request.json()
    except Exception:
        body = {}
        
    task_name = body.get("task_name", "spam_detection")
    env_instance = EmailManagementEnv(task_name=task_name)
    obs = env_instance.reset()
    
    return {
        "observation": obs.model_dump(),
        "info": {"task_name": task_name}
    }

@app.post("/step")
async def step(request: Request):
    global env_instance
    if env_instance is None:
        return {"error": "Environment not initialized. Call /reset first."}
        
    body = await request.json()
    action_dict = body.get("action", {})
    
    # Map raw dict back to Action model
    action = Action(**action_dict)
    
    next_obs, reward, done, info = env_instance.step(action)
    
    # Clamp reward to strictly (0, 1) as required by evaluator
    raw_reward = float(reward.value) if hasattr(reward, "value") else float(reward)
    safe_reward = _clamp(raw_reward)

    safe_info = {}
    for k, v in info.items():
        if isinstance(v, float) or isinstance(v, int):
             safe_info[k] = _clamp(v)
        else:
             safe_info[k] = v

    # Ensure "score" and "mean_score" exist in info and are safely clamped
    if "score" in info:
        safe_info["score"] = _clamp(info["score"])
    if "mean_score" in info:
        safe_info["mean_score"] = _clamp(info["mean_score"])
    
    return {
        "observation": next_obs.model_dump() if next_obs else None,
        "reward": safe_reward,
        "score": safe_reward,
        "done": done,
        "info": safe_info
    }

import uvicorn

@app.get("/")
async def root():
    return {
        "status": "ready",
        "message": "Email RL Environment Server is running.",
        "endpoints": ["POST /reset", "POST /step"]
    }

def run_server():
    uvicorn.run("server:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    run_server()
