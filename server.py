from fastapi import FastAPI, Request
from pydantic import BaseModel
import math
from typing import Any, Dict, Optional

from env.email_env import EmailManagementEnv
from env.models import Action

app = FastAPI(title="Email Management OpenEnv Server")
env_instance = None

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
    try:
        raw_reward = float(reward.value) if hasattr(reward, 'value') else float(reward)
        if math.isnan(raw_reward):
            raw_reward = 0.5
    except Exception:
        raw_reward = 0.5

    if raw_reward <= 0.0:
    safe_reward = 0.01
elif raw_reward >= 1.0:
    safe_reward = 0.99
    else:
        safe_reward = raw_reward
    
    return {
        "observation": next_obs.model_dump() if next_obs else None,
        "reward": safe_reward,
        "done": done,
        "info": info
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
