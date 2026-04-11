from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import math
import uuid
from typing import Any, Dict, Optional

from env.email_env import EmailManagementEnv
from env.models import Action

app = FastAPI(title="Email Management OpenEnv Server")

_envs: Dict[str, EmailManagementEnv] = {}
_current_session: Optional[str] = None
_SCORE_KEYS = {"score", "mean_score", "reward", "partial_score", "correctness"}


def _clamp(v: Any) -> float:
    try:
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return 0.5
    except Exception:
        return 0.5
    if f <= 0.0:
        return 0.01
    if f >= 1.0:
        return 0.99
    return f


def _sanitise_info(info: Dict[str, Any]) -> Dict[str, Any]:
    safe: Dict[str, Any] = {}
    for k, v in (info or {}).items():
        if k in _SCORE_KEYS and isinstance(v, (int, float)):
            safe[k] = _clamp(float(v))
        else:
            safe[k] = v
    return safe


@app.get("/")
async def root():
    return JSONResponse(content={
        "status": "ready",
        "message": "Email RL Environment Server is running.",
        "endpoints": ["POST /reset", "POST /step", "GET /state", "GET /tasks", "GET /health"],
    })


@app.get("/health")
async def health():
    return JSONResponse(content={"status": "ok"})


@app.get("/tasks")
async def tasks():
    return JSONResponse(content={
        "tasks": [
            {"id": "spam_detection", "grader": "grade_classification", "score_range": [0.01, 0.99]},
            {"id": "email_prioritization", "grader": "grade_prioritization", "score_range": [0.01, 0.99]},
            {"id": "auto_reply", "grader": "grade_reply", "score_range": [0.01, 0.99]},
        ]
    })


@app.post("/reset")
async def reset(request: Request):
    global _current_session
    try:
        body = await request.json()
    except Exception:
        body = {}

    task_name = body.get("task_name", "spam_detection")

    if _current_session and _current_session in _envs:
        try:
            _envs[_current_session].close()
        except Exception:
            pass
        del _envs[_current_session]

    session_id = str(uuid.uuid4())
    _current_session = session_id

    env = EmailManagementEnv(task_name=task_name)
    obs = env.reset()
    _envs[session_id] = env

    return JSONResponse(content={
        "observation": obs.model_dump(),
        "session_id": session_id,
        "info": {"task_name": task_name},
    })


@app.post("/step")
async def step(request: Request):
    global _current_session

    if not _current_session or _current_session not in _envs:
        raise HTTPException(status_code=400, detail="Call /reset first.")

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=422, detail="Invalid JSON body.")

    action_dict = body.get("action", {})
    try:
        action = Action(**action_dict)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid action: {e}")

    env = _envs[_current_session]

    try:
        next_obs, reward, done, info = env.step(action)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"env.step() failed: {e}")

    raw_reward = float(reward.value) if hasattr(reward, "value") else float(reward)
    safe_reward = _clamp(raw_reward)

    safe_info = _sanitise_info(info)
    safe_info["score"] = _clamp(float(info.get("score", raw_reward)))

    if done:
        try:
            env.close()
        except Exception:
            pass
        if _current_session in _envs:
            del _envs[_current_session]
        _current_session = None

    return JSONResponse(content={
        "observation": next_obs.model_dump() if next_obs else None,
        "reward": safe_reward,
        "score": safe_info["score"],
        "done": done,
        "info": safe_info,
    })


@app.get("/state")
async def state():
    global _current_session

    if not _current_session or _current_session not in _envs:
        return JSONResponse(content={
            "status": "not_initialized",
            "session_id": None,
            "state": None,
        })

    env = _envs[_current_session]
    try:
        env_state = env.state().model_dump()
    except Exception:
        env_state = {}

    return JSONResponse(content={
        "status": "ready",
        "session_id": _current_session,
        "state": env_state,
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=7860, reload=False, workers=1)