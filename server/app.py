import uvicorn
from fastapi import FastAPI, Request
from pydantic import BaseModel
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
    
    action = Action(**action_dict)
    
    next_obs, reward, done, info = env_instance.step(action)
    
    return {
        "observation": next_obs.model_dump() if next_obs else None,
        "reward": reward.value, 
        "done": done,
        "info": info
    }

@app.get("/")
async def root():
    return {
        "status": "ready",
        "message": "Email RL Environment Server is running.",
        "endpoints": ["POST /reset", "POST /step"]
    }

def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
