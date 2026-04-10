# from fastapi import FastAPI, Request
# from pydantic import BaseModel
# import math
# from typing import Any, Dict, Optional

# from env.email_env import EmailManagementEnv
# from env.models import Action

# app = FastAPI(title="Email Management OpenEnv Server")
# env_instance = None


# def _clamp(v: Any) -> float:
#     try:
#         f = float(v)
#         if math.isnan(f) or math.isinf(f):
#             return 0.5
#     except Exception:
#         return 0.5
#     if f <= 0.0:
#         return 0.001
#     if f >= 1.0:
#         return 0.999
#     return f


# @app.post("/reset")
# async def reset(request: Request):
#     global env_instance
#     try:
#         body = await request.json()
#     except Exception:
#         body = {}
        
#     task_name = body.get("task_name", "spam_detection")
#     env_instance = EmailManagementEnv(task_name=task_name)
#     obs = env_instance.reset()
    
#     return {
#         "observation": obs.model_dump(),
#         "info": {"task_name": task_name}
#     }

# @app.post("/step")
# async def step(request: Request):
#     global env_instance
#     if env_instance is None:
#         return {"error": "Environment not initialized. Call /reset first."}
        
#     body = await request.json()
#     action_dict = body.get("action", {})
    
#     # Map raw dict back to Action model
#     action = Action(**action_dict)
    
#     next_obs, reward, done, info = env_instance.step(action)
    
#     # Clamp reward to strictly (0, 1) as required by evaluator
#     raw_reward = float(reward.value) if hasattr(reward, "value") else float(reward)
#     safe_reward = _clamp(raw_reward)

#     safe_info = {}
#     for k, v in info.items():
#         if isinstance(v, float) or isinstance(v, int):
#              safe_info[k] = _clamp(v)
#         else:
#              safe_info[k] = v

#     # Ensure "score" and "mean_score" exist in info and are safely clamped
#     if "score" in info:
#         safe_info["score"] = _clamp(info["score"])
#     if "mean_score" in info:
#         safe_info["mean_score"] = _clamp(info["mean_score"])
    
#     return {
#         "observation": next_obs.model_dump() if next_obs else None,
#         "reward": safe_reward,
#         "score": safe_reward,
#         "done": done,
#         "info": safe_info
#     }

# import uvicorn

# @app.get("/")
# async def root():
#     return {
#         "status": "ready",
#         "message": "Email RL Environment Server is running.",
#         "endpoints": ["POST /reset", "POST /step"]
#     }

# def run_server():
#     uvicorn.run("server:app", host="0.0.0.0", port=7860)

# if __name__ == "__main__":
#     run_server()


from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import math
import uuid
from typing import Any, Dict, Optional

from env.email_env import EmailManagementEnv
from env.models import Action

app = FastAPI(title="Email Management OpenEnv Server")

# Use a dict so each session gets its own env (basic thread-safety)
_envs: Dict[str, EmailManagementEnv] = {}
_current_session: Optional[str] = None


def _clamp(v: Any) -> float:
    """Clamp to strictly open interval (0, 1)."""
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


def _is_score_key(k: str) -> bool:
    """Only clamp keys that are actual scores/rewards, not counts."""
    score_keys = {"score", "mean_score", "reward", "partial_score"}
    return k in score_keys


@app.post("/reset")
async def reset(request: Request):
    global _current_session

    try:
        body = await request.json()
    except Exception:
        body = {}

    task_name = body.get("task_name", "spam_detection")

    # Clean up old env if exists
    if _current_session and _current_session in _envs:
        try:
            _envs[_current_session].close()
        except Exception:
            pass
        del _envs[_current_session]

    # Create new session
    session_id = str(uuid.uuid4())
    _current_session = session_id

    env = EmailManagementEnv(task_name=task_name)
    obs = env.reset()
    _envs[session_id] = env

    return JSONResponse(content={
        "observation": obs.model_dump(),
        "session_id": session_id,
        "info": {"task_name": task_name}
    })


@app.post("/step")
async def step(request: Request):
    global _current_session

    if not _current_session or _current_session not in _envs:
        raise HTTPException(
            status_code=400,
            detail="Environment not initialized. Call /reset first."
        )

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

    # Clamp reward — only reward/score values, not counters
    raw_reward = float(reward.value) if hasattr(reward, "value") else float(reward)
    safe_reward = _clamp(raw_reward)

    safe_info = {}
    for k, v in (info or {}).items():
        if _is_score_key(k) and isinstance(v, (float, int)):
            safe_info[k] = _clamp(v)
        else:
            safe_info[k] = v  # pass through unchanged (counts, strings, etc.)

    # Clean up env when episode is done
    if done:
        try:
            env.close()
        except Exception:
            pass
        del _envs[_current_session]
        _current_session = None

    return JSONResponse(content={
        "observation": next_obs.model_dump() if next_obs else None,
        "reward": safe_reward,
        "score": safe_reward,
        "done": done,
        "info": safe_info
    })


@app.get("/state")
async def state():
    """
    Required by OpenEnv spec — returns current environment state.
    """
    global _current_session

    if not _current_session or _current_session not in _envs:
        return JSONResponse(content={
            "status": "not_initialized",
            "session_id": None,
            "state": None
        })

    env = _envs[_current_session]

    try:
        env_state = env.get_state() if hasattr(env, "get_state") else {}
    except Exception:
        env_state = {}

    return JSONResponse(content={
        "status": "ready",
        "session_id": _current_session,
        "state": env_state
    })


@app.get("/")
async def root():
    return JSONResponse(content={
        "status": "ready",
        "message": "Email RL Environment Server is running.",
        "endpoints": [
            "POST /reset",
            "POST /step",
            "GET  /state",
            "GET  /"
        ]
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=7860, reload=False)