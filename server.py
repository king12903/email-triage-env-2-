from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn

from environment import EmailTriageEnv, Action

app = FastAPI(title="Email Triage OpenEnv", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

envs = {}

class StepRequest(BaseModel):
    task_id: str = "easy"
    priority: str
    category: str
    should_reply: bool
    reply_text: Optional[str] = None

@app.get("/")
def root():
    return {"status": "ok", "env": "EmailTriageEnv", "version": "1.0.0"}

@app.post("/reset")
async def reset(request: Request):
    """Works with empty body OR {task_id: 'easy'|'medium'|'hard'}"""
    try:
        body = await request.json()
        task_id = body.get("task_id", "easy") if isinstance(body, dict) else "easy"
    except Exception:
        task_id = "easy"

    env = EmailTriageEnv(task_id=task_id)
    envs[task_id] = env
    envs["default"] = env
    obs = env.reset()
    return obs.dict()

@app.post("/step")
def step(req: StepRequest):
    env = envs.get(req.task_id) or envs.get("default")
    if env is None:
        raise HTTPException(status_code=400, detail="Call /reset first")

    action = Action(
        priority=req.priority,
        category=req.category,
        should_reply=req.should_reply,
        reply_text=req.reply_text
    )
    obs, reward, done, info = env.step(action)
    return {
        "observation": obs.dict() if obs else None,
        "reward": reward.dict(),
        "done": done,
        "info": info
    }

@app.get("/state")
def state(task_id: str = "easy"):
    env = envs.get(task_id) or envs.get("default")
    if env is None:
        raise HTTPException(status_code=400, detail="Call /reset first")
    return env.state().dict()

@app.get("/tasks")
def get_tasks():
    return {
        "tasks": [
            {
                "id": "easy",
                "name": "Spam Detection",
                "description": "Classify emails as spam or not-spam, set priority.",
                "difficulty": "easy",
                "action_schema": {
                    "priority": "high | medium | low",
                    "category": "billing | support | spam | inquiry",
                    "should_reply": "boolean",
                    "reply_text": "string (optional)"
                }
            },
            {
                "id": "medium",
                "name": "Priority & Category Triage",
                "description": "Assign correct priority and category to each email.",
                "difficulty": "medium",
                "action_schema": {
                    "priority": "high | medium | low",
                    "category": "billing | support | spam | inquiry",
                    "should_reply": "boolean",
                    "reply_text": "string (optional)"
                }
            },
            {
                "id": "hard",
                "name": "Full Email Triage",
                "description": "Complete triage with priority, category, reply decision and reply text.",
                "difficulty": "hard",
                "action_schema": {
                    "priority": "high | medium | low",
                    "category": "billing | support | spam | inquiry",
                    "should_reply": "boolean",
                    "reply_text": "string (required when should_reply=true)"
                }
            }
        ]
    }

@app.post("/grader")
async def grader(request: Request):
    try:
        body = await request.json()
        task_id = body.get("task_id", "easy") if isinstance(body, dict) else "easy"
    except Exception:
        task_id = "easy"

    env = envs.get(task_id) or envs.get("default")
    if env is None:
        raise HTTPException(status_code=400, detail="Call /reset first")

    s = env.state()
    final_score = round(s.cumulative_reward / max(s.current_step, 1), 3)
    return {
        "task_id": task_id,
        "score": final_score,
        "cumulative_reward": s.cumulative_reward,
        "steps_completed": s.current_step,
        "done": s.done
    }

@app.post("/baseline")
def baseline():
    results = {}
    for task_id in ["easy", "medium", "hard"]:
        env = EmailTriageEnv(task_id=task_id)
        obs = env.reset()
        total_reward, steps = 0.0, 0

        while True:
            subject = obs.email.subject.lower()
            if any(w in subject for w in ["won", "prize", "click", "free", "!!!"]):
                priority, category, should_reply = "low", "spam", False
            elif any(w in subject for w in ["urgent", "invoice", "overdue", "payment", "suspended"]):
                priority, category, should_reply = "high", "billing", True
            elif any(w in subject for w in ["bug", "error", "broken", "fix", "api"]):
                priority, category, should_reply = "high", "support", True
            elif any(w in subject for w in ["question", "pricing", "upgrade", "plan"]):
                priority, category, should_reply = "medium", "inquiry", True
            else:
                priority, category, should_reply = "medium", "inquiry", False

            action = Action(priority=priority, category=category, should_reply=should_reply)
            obs, reward, done, _ = env.step(action)
            total_reward += reward.value
            steps += 1
            if done:
                break

        results[task_id] = {
            "score": round(total_reward / steps, 3),
            "total_reward": round(total_reward, 3),
            "steps": steps
        }

    return {"baseline_scores": results}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
