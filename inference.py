"""
Baseline inference script for Email Triage OpenEnv.
Uses OpenAI API to run a language model agent against the environment.

Usage:
    export OPENAI_API_KEY=your_key_here
    python baseline.py
"""

import os
import json
import requests

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:7860")

def call_llm(prompt: str) -> dict:
    """Call OpenAI API and parse JSON action."""
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert email triage assistant. "
                    "Given an email, respond ONLY with a valid JSON object with these fields:\n"
                    '{"priority": "high|medium|low", "category": "billing|support|spam|inquiry", '
                    '"should_reply": true|false, "reply_text": "string or null"}'
                )
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    raw = response.choices[0].message.content.strip()
    # Remove markdown code fences if present
    raw = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)


def run_task(task_id: str) -> float:
    """Run the LLM agent on one task and return the average score."""
    print(f"\n{'='*50}")
    print(f"Running task: {task_id}")
    print(f"{'='*50}")

    # Reset environment
    res = requests.post(f"{BASE_URL}/reset", json={"task_id": task_id})
    obs = res.json()
    total_reward = 0.0
    steps = 0

    while True:
        email = obs["email"]
        prompt = (
            f"Subject: {email['subject']}\n"
            f"From: {email['sender']}\n"
            f"Body: {email['body']}\n\n"
            f"Task: {obs['task_description']}"
        )

        try:
            action = call_llm(prompt)
        except Exception as e:
            print(f"  LLM error: {e}, using fallback")
            action = {"priority": "medium", "category": "inquiry", "should_reply": False, "reply_text": None}

        # Send action to environment
        step_res = requests.post(f"{BASE_URL}/step", json={
            "task_id": task_id,
            "priority": action.get("priority", "medium"),
            "category": action.get("category", "inquiry"),
            "should_reply": action.get("should_reply", False),
            "reply_text": action.get("reply_text")
        })

        result = step_res.json()
        reward = result["reward"]["value"]
        total_reward += reward
        steps += 1

        print(f"  Step {steps}: reward={reward:.2f} | {result['reward']['reason'][:60]}")

        if result["done"]:
            break
        obs = result["observation"]

    avg_score = round(total_reward / steps, 3)
    print(f"  Final score for '{task_id}': {avg_score}")
    return avg_score


def main():
    print("Email Triage OpenEnv — Baseline Inference Script")
    print("="*50)

    if not OPENAI_API_KEY:
        print("WARNING: OPENAI_API_KEY not set. Please export it first.")
        return

    scores = {}
    for task_id in ["easy", "medium", "hard"]:
        scores[task_id] = run_task(task_id)

    print("\n" + "="*50)
    print("BASELINE SCORES SUMMARY")
    print("="*50)
    for task_id, score in scores.items():
        print(f"  {task_id:8s}: {score:.3f}")
    overall = round(sum(scores.values()) / len(scores), 3)
    print(f"  {'overall':8s}: {overall:.3f}")
    print("="*50)


if __name__ == "__main__":
    main()
