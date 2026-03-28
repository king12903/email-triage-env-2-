from pydantic import BaseModel
from typing import Optional
import random

# -----------------------------
# Models
# -----------------------------
class Email(BaseModel):
    id: str
    subject: str
    body: str
    sender: str

class Observation(BaseModel):
    email: Email
    step_number: int
    total_steps: int
    task_description: str

class Action(BaseModel):
    priority: str
    category: str
    should_reply: bool
    reply_text: Optional[str] = None

class Reward(BaseModel):
    value: float
    reason: str

class State(BaseModel):
    current_step: int
    total_steps: int
    cumulative_reward: float
    done: bool
    task_id: str

# -----------------------------
# Dataset
# -----------------------------
EMAILS = [
    Email(id="e001", subject="URGENT: Payment failed", body="Account suspended due to failed payment", sender="client@company.com"),
    Email(id="e002", subject="Meeting confirmation", body="Confirming meeting at 3pm", sender="colleague@work.com"),
    Email(id="e003", subject="You won $1,000,000!!!", body="Click here to claim", sender="spam@xyz.com"),
    Email(id="e004", subject="Pricing plans?", body="Tell me about plans", sender="customer@gmail.com"),
    Email(id="e005", subject="Invoice overdue", body="Payment pending", sender="billing@vendor.com"),
    Email(id="e006", subject="API bug urgent", body="Endpoint failing", sender="dev@partner.com"),
]

GROUND_TRUTH = {
    "e001": {"priority": "high", "category": "billing", "should_reply": True},
    "e002": {"priority": "medium", "category": "inquiry", "should_reply": False},
    "e003": {"priority": "low", "category": "spam", "should_reply": False},
    "e004": {"priority": "medium", "category": "inquiry", "should_reply": True},
    "e005": {"priority": "high", "category": "billing", "should_reply": True},
    "e006": {"priority": "high", "category": "support", "should_reply": True},
}

# -----------------------------
# Environment
# -----------------------------
class EmailTriageEnv:
    def __init__(self, task_id="easy"):
        self.task_id = task_id
        self.current_step = 0
        self.done = False
        self.cumulative_reward = 0.0
        self.emails = EMAILS[:]
        self.current_email = None

    def reset(self):
        random.shuffle(self.emails)
        self.current_step = 0
        self.done = False
        self.cumulative_reward = 0.0
        self.current_email = self.emails[0]
        return self._obs()

    def step(self, action: Action):
        if self.done:
            raise ValueError("Reset first")

        reward = self._reward(action)
        self.cumulative_reward += reward.value
        self.current_step += 1

        if self.current_step >= len(self.emails):
            self.done = True
            return None, reward, True, {}

        self.current_email = self.emails[self.current_step]
        return self._obs(), reward, False, {}

    def state(self):
        return State(
            current_step=self.current_step,
            total_steps=len(self.emails),
            cumulative_reward=round(self.cumulative_reward, 3),
            done=self.done,
            task_id=self.task_id
        )

    def _obs(self):
        return Observation(
            email=self.current_email,
            step_number=self.current_step,
            total_steps=len(self.emails),
            task_description="Classify email and decide response"
        )

    def _reward(self, action):
        truth = GROUND_TRUTH[self.current_email.id]
        score = 0

        if action.priority == truth["priority"]:
            score += 0.4
        if action.category == truth["category"]:
            score += 0.4
        if action.should_reply == truth["should_reply"]:
            score += 0.2

        if action.should_reply and truth["category"] == "spam":
            score -= 0.3

        if action.reply_text and len(action.reply_text) > 10:
            score += 0.1

        score = max(0, min(1, round(score, 3)))

        return Reward(value=score, reason="scored")