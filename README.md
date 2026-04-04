---
title: Clinical Triage Gym
emoji: 🏥
colorFrom: blue
colorTo: red
sdk: docker
pinned: false
---

# 🏥 Hospital ER Simulator (Clinical Triage Gym)

Welcome to the **Hospital ER Simulator**! This is an interactive AI testing environment built on top of OpenEnv.

Instead of testing an AI on simple multiple-choice questions, this simulator acts like a real-world Emergency Room. The AI plays the role of the hospital receptionist. When a simulated patient walks in, the AI must decide whether to ask clarifying questions or immediately assign them to a doctor based on how sick they are (**this process is known as "Triage"**).

---

## ⚙️ OpenEnv Fundamentals

This project leans heavily on [OpenEnv](https://github.com/openenv-project), which handles all the complex networking and JSON schemas behind the scenes.
* **Pure Python:** We only write regular Python classes (`Pydantic` models). OpenEnv automatically translates these into **OpenAI Tool Schemas** so any LLM can natively understand your environment as a "function call".
* **Automated Servers:** `create_app()` in OpenEnv spins up an asynchronous FastAPI web server, WebSocket channels, and the Hugging Face Web UI automatically.
* **MCP Standards:** The simulator outputs standardized Model Context Protocol (MCP) data, ensuring it plays nicely with any modern AI agent frameworks.

---

## 📂 File Structure: Why Do We Need These Files?

Here is how the engine actually works:

* **`models.py`**  
  **What it does:** Defines the strict rules for how the AI must communicate. It contains the data shapes (like `TriageAction` and `TriageObservation`) ensuring the AI can only output valid commands (like `ask_question` or `assign_doctor`).
* **`data/scenarios.json`**  
  **What it does:** This is the database of simulated patients. Each patient has public vitals and *secret* information that the AI can only discover if it asks the right questions!
* **`server/triage_environment.py`**  
  **What it does:** The "brain" of the simulator. It controls the episode loop. When you run `reset()`, it picks a patient from the JSON file. When you run `step()`, it processes the AI's question, unlocks secret information, and updates the timer.
* **`tasks.py`**  
  **What it does:** The "Grader". Once the AI agent makes a final decision on where to route the patient, this file checks if the AI was correct and calculates a final score from $0.0$ to $1.0$.
* **`server/app.py`**  
  **What it does:** The OpenEnv entry point that converts the python logic into the live server you see on Hugging Face.

---

## 🎮 How the AI Plays the Game

### What the AI receives (Observation Space)
Every turn, the environment sends the AI the patient's:
`chief_complaint`, `vitals`, `medical_history`, `newly_uncovered_secrets`, `valid_hospital_departments`, and `timer_count`.

### What the AI can do (Action Space)

| Type | Description |
|------|-------------|
| `assess` | Just acknowledge reading the patient's chart. |
| `clarify` | Ask a direct question to the patient (Max 2 allowed). |
| `triage` | **Final Decision:** Assign the patient a Priority Level (1-5) and route them to a medical department (like Cardiology). |

---

## 💯 The Grading System (Reward Function)

The AI is graded based on safety and efficiency.

| Scoring Rule | Points | Explanation |
|-----------|--------|-------------|
| **Perfect Priority Match** | `+0.30` | Did the agent correctly predict exactly how critical the patient was? |
| **Correct Department** | `+0.30` | Was the patient sent to the right specialist? |
| **Asked Questions** | `+0.15` | Rewarded if the AI asked clarifying questions rather than blindly guessing. |
| **Efficiency Bonus** | `+0.10` | The AI loses points if it wastes time asking too many irrelevant questions. |
| 🚨 **Catastrophic Penalty** | **`-0.50`** | If a highly critical, dying patient is sent to a non-critical doctor (like the skin doctor), the AI suffers a massive point deduction. |

---

## Quick Start (For Developers)

```bash
pip install openenv-core[core] faker
python -m uvicorn server.app:app --host 0.0.0.0 --port 7860
```
