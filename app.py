"""
server.py – FastAPI OpenEnv server for the Email Management RL Environment.
The validator calls POST /reset and POST /step on this server.
All scores/rewards are clamped to strictly open interval (0, 1).
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from env.email_env import EmailManagementEnv
from env.models import Action

app = FastAPI(title="Email Management OpenEnv Server")

# One global env instance (the validator calls /reset then /step sequentially)
env_instance: Optional[EmailManagementEnv] = None


# ---------------------------------------------------------------------------
# Clamp helper — strictly open interval (0, 1), NEVER 0.0 or 1.0
# ---------------------------------------------------------------------------

def _clamp(v: Any) -> float:
    try:
        v = float(v)
        if math.isnan(v) or math.isinf(v):
            return 0.5
    except Exception:
        return 0.5
    if v <= 0.0:
        return 0.001
    if v >= 1.0:
        return 0.999
    return v


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    return {
        "status": "ready",
        "message": "Email RL Environment Server is running.",
        "endpoints": ["POST /reset", "POST /step"],
    }


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

    return JSONResponse({
        "observation": obs.model_dump(),
        "info": {"task_name": task_name},
    })


@app.post("/step")
async def step(request: Request):
    global env_instance
    if env_instance is None:
        return JSONResponse(
            {"error": "Environment not initialized. Call /reset first."},
            status_code=400,
        )

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body."}, status_code=400)

    action_dict = body.get("action", {})

    try:
        action = Action(**action_dict)
    except Exception as exc:
        return JSONResponse({"error": f"Invalid action: {exc}"}, status_code=422)

    try:
        next_obs, reward, done, info = env_instance.step(action)
    except Exception as exc:
        return JSONResponse({"error": f"step() failed: {exc}"}, status_code=500)

    # ── Clamp reward ────────────────────────────────────────────────────────
    raw_reward = float(reward.value) if hasattr(reward, "value") else float(reward)
    safe_reward = _clamp(raw_reward)

    # ── Clamp every numeric field in info ───────────────────────────────────
    safe_info: Dict[str, Any] = {}
    for k, v in info.items():
        if isinstance(v, float):
            safe_info[k] = _clamp(v)
        else:
            safe_info[k] = v

    # ── Build response ───────────────────────────────────────────────────────
    # The validator reads BOTH "reward" and "score" — we return both,
    # both clamped to strictly (0, 1).
    return JSONResponse({
        "observation": next_obs.model_dump() if next_obs else None,
        "reward":      safe_reward,          # step-level reward
        "score":       safe_reward,          # task-level score (same value)
        "done":        done,
        "info":        safe_info,
    })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

import uvicorn

def run_server():
    uvicorn.run("server:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    run_server()