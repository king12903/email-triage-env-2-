---
title: Email Triage Env
emoji: 📧
colorFrom: blue
colorTo: green
sdk: docker
python_version: "3.9"
pinned: false
tags:
  - openenv
---

# 📧 Email Triage OpenEnv

A real-world OpenEnv environment where an AI agent learns to **prioritize, categorize, and respond to emails**.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/reset` | POST | Start new episode |
| `/step` | POST | Take an action |
| `/state` | GET | Get current state |
| `/tasks` | GET | List all tasks |
| `/grader` | POST | Get episode score |
| `/baseline` | POST | Run baseline agent |

## Tasks

| Task | Difficulty | Description |
|------|------------|-------------|
| `easy` | Easy | Spam vs not-spam detection |
| `medium` | Medium | Priority + category assignment |
| `hard` | Hard | Full triage with reply |

## Action Space

```json
{
  "priority": "high | medium | low",
  "category": "billing | support | spam | inquiry",
  "should_reply": true,
  "reply_text": "optional string"
}
```

## Setup

```bash
pip install -r requirements.txt
python server.py
```

## Docker

```bash
docker build -t email-triage-env .
docker run -p 7860:7860 email-triage-env
```
