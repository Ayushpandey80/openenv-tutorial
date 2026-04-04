"""
Baseline Inference Script for Clinical Triage Gym.

Connects to the environment via WebSocket (which maintains session state
across reset/step calls) and runs a deterministic heuristic baseline agent
across all three difficulty tiers.

Usage:
    # Against a local server:
    API_BASE_URL=http://localhost:7860 python inference.py

    # Against the HF Space:
    API_BASE_URL=https://suppayp-clinical-triage-gym.hf.space python inference.py

Required environment variables:
    API_BASE_URL  - The environment server URL (default: http://localhost:7860)
    MODEL_NAME    - Model identifier (for logging, default: baseline-heuristic)
    HF_TOKEN      - HuggingFace token (optional, for authenticated spaces)
"""

import asyncio
import json
import os
import sys
from typing import Any, Dict, List, Optional

import websockets

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:7860")
MODEL_NAME = os.environ.get("MODEL_NAME", "baseline-heuristic")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

NUM_EPISODES_PER_TASK = 3


# ---------------------------------------------------------------------------
# WebSocket helpers (matching OpenEnv protocol exactly)
# ---------------------------------------------------------------------------
def _ws_url() -> str:
    """Convert HTTP URL to WebSocket URL."""
    base = API_BASE_URL.rstrip("/")
    if base.startswith("https://"):
        base = "wss://" + base[len("https://"):]
    elif base.startswith("http://"):
        base = "ws://" + base[len("http://"):]
    return f"{base}/ws"


async def ws_reset(ws, task_id: str, seed: Optional[int] = None) -> Dict[str, Any]:
    """Send reset via OpenEnv WS protocol: {type: 'reset', data: {...}}."""
    data: Dict[str, Any] = {"task_id": task_id}
    if seed is not None:
        data["seed"] = seed
    msg = {"type": "reset", "data": data}
    await ws.send(json.dumps(msg))
    resp = json.loads(await ws.recv())
    # Response format: {"type": "observation", "data": {...}}
    return resp.get("data", resp)


async def ws_step(ws, action: Dict[str, Any]) -> Dict[str, Any]:
    """Send step via OpenEnv WS protocol: {type: 'step', data: {...}}."""
    msg = {"type": "step", "data": action}
    await ws.send(json.dumps(msg))
    resp = json.loads(await ws.recv())
    return resp.get("data", resp)


# ---------------------------------------------------------------------------
# Heuristic decision logic
# ---------------------------------------------------------------------------
KEYWORD_SPECIALTY_MAP = {
    "chest pressure": "cardiology",
    "chest pain": "cardiology",
    "heart": "cardiology",
    "cardiac": "cardiology",
    "stemi": "cardiology",
    "tearing pain": "cardiology",
    "arm": "cardiology",
    "leg swelling": "cardiology",
    "worst headache": "neurology",
    "thunderclap": "neurology",
    "headache": "neurology",
    "seizure": "neurology",
    "stroke": "neurology",
    "weakness": "neurology",
    "tingling": "neurology",
    "fracture": "orthopedics",
    "wrist pain": "orthopedics",
    "ankle": "orthopedics",
    "bone": "orthopedics",
    "shortness of breath": "pulmonology",
    "breath": "pulmonology",
    "cough": "pulmonology",
    "lung": "pulmonology",
    "epigastric": "gastroenterology",
    "abdominal pain": "gastroenterology",
    "nausea": "gastroenterology",
    "vomit": "gastroenterology",
    "diarrhea": "gastroenterology",
    "urin": "urology",
    "kidney": "nephrology",
    "catheter": "internal_medicine",
    "confusion": "internal_medicine",
    "fever": "internal_medicine",
    "sore throat": "pediatrics",
    "child": "pediatrics",
    "purpuric rash": "pediatrics",
    "petechial": "pediatrics",
    "non-blanching": "pediatrics",
    "rash": "dermatology",
    "itchy": "dermatology",
    "eye": "ophthalmology",
    "ear": "ent",
    "pregnancy": "obstetrics_gynecology",
    "menstrual": "obstetrics_gynecology",
    "vaginal": "obstetrics_gynecology",
    "ectopic": "obstetrics_gynecology",
    "psych": "psychiatry",
    "anxiety": "psychiatry",
    "back pain": "family_medicine",
    "low back": "family_medicine",
}


def heuristic_urgency(vitals: Optional[Dict[str, Any]], complaint: str) -> int:
    """Estimate ESI urgency from vitals and chief complaint."""
    if vitals is None:
        return 3

    hr = vitals.get("hr", 80)
    spo2 = vitals.get("spo2", 99)
    temp = vitals.get("temp", 98.6)
    rr = vitals.get("rr", 16)
    bp_str = str(vitals.get("bp", "120/80"))
    try:
        systolic = int(bp_str.split("/")[0])
    except (ValueError, IndexError):
        systolic = 120

    c = complaint.lower()

    # ESI 1 — immediate life threat
    if spo2 < 90 or hr > 150 or rr > 30:
        return 1
    if any(kw in c for kw in ["unresponsive", "cardiac arrest", "not breathing", "pulseless"]):
        return 1

    # ESI 2 — emergent
    if any(kw in c for kw in [
        "chest pressure", "chest pain", "worst headache", "thunderclap",
        "tearing pain", "non-blanching", "purpuric rash", "lethargic",
        "radiating to", "severe confusion"
    ]):
        return 2
    if spo2 < 94 or hr > 120 or temp > 103 or systolic < 90 or rr > 26:
        return 2

    # ESI 3 — urgent
    if hr > 100 or temp > 101 or rr > 22 or systolic < 100:
        return 3

    # ESI 4 — less urgent
    if temp > 99.5 or hr > 90:
        return 4

    return 5


def heuristic_specialty(complaint: str, additional_info: Optional[str] = None) -> str:
    """Pick best specialty via keyword matching (longest match wins)."""
    text = (complaint + " " + (additional_info or "")).lower()
    best_match = "emergency_medicine"
    best_score = 0
    for keyword, specialty in KEYWORD_SPECIALTY_MAP.items():
        if keyword in text:
            if len(keyword) > best_score:
                best_score = len(keyword)
                best_match = specialty
    return best_match


def pick_clarify_question(complaint: str) -> str:
    """Choose a relevant clarifying question."""
    c = complaint.lower()
    if "chest" in c:
        return "Does the pain radiate to the arm or jaw? Any ECG findings?"
    if "headache" in c:
        return "Was the onset sudden or gradual? Any neurological deficits?"
    if "confusion" in c or "altered" in c:
        return "Are there signs of infection such as fever or urinary changes?"
    if "breath" in c:
        return "Are breath sounds equal bilaterally? Any chest pain?"
    if "abdomin" in c or "epigastric" in c:
        return "Is the abdomen rigid? Any guarding or peritonitis signs?"
    if "weakness" in c or "tingling" in c:
        return "Is the weakness ascending? Any respiratory compromise?"
    if "rash" in c or "purp" in c:
        return "Does the rash blanch with pressure? Is it spreading?"
    if "pregnancy" in c or "menstrual" in c:
        return "When was the last menstrual period? Any vaginal bleeding?"
    if "fever" in c:
        return "Any rash, neck stiffness, or altered mental status?"
    if "pain" in c:
        return "Can you describe the severity and character of the pain?"
    return "Can you provide more details about the symptoms and exam findings?"


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------
async def run_episode(ws, task_id: str, episode_idx: int, seed: int) -> float:
    """Run a single episode and return the reward."""
    print(f"\n--- Task: {task_id.upper()} | Episode {episode_idx + 1} | Seed {seed} ---")

    # Reset
    obs = await ws_reset(ws, task_id=task_id, seed=seed)
    complaint = obs.get("chief_complaint", "")
    vitals = obs.get("vitals")
    print(f"  Patient: {complaint[:80]}...")

    # Step 1: Assess
    obs = await ws_step(ws, {"action_type": "assess"})
    print(f"  → assess (step {obs.get('step_number', '?')})")

    # Step 2: Clarify
    question = pick_clarify_question(complaint)
    obs = await ws_step(ws, {"action_type": "clarify", "question": question})
    additional_info = obs.get("additional_info", "")
    print(f"  → clarify: '{question[:60]}...'")
    print(f"    response: '{(additional_info or 'N/A')[:80]}...'")

    # Step 3: Triage
    urgency = heuristic_urgency(vitals, complaint)
    specialty = heuristic_specialty(complaint, additional_info)
    reasoning = (
        f"Vitals and complaint indicate {specialty} at ESI {urgency}."
    )

    obs = await ws_step(ws, {
        "action_type": "triage",
        "urgency": urgency,
        "specialty": specialty,
        "reasoning": reasoning,
    })

    reward = float(obs.get("reward", 0.0))
    done = obs.get("done", False)
    breakdown = obs.get("reward_breakdown", {})

    print(f"  → triage: urgency={urgency}, specialty={specialty}")
    print(f"  ✓ Done={done} | Reward={reward:.2f}")
    if breakdown:
        for comp, val in breakdown.items():
            print(f"    {comp}: {val:+.2f}")

    return reward


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def async_main():
    print("=" * 60)
    print("Clinical Triage Gym — Baseline Evaluation")
    print("=" * 60)
    print(f"  Environment : {API_BASE_URL}")
    print(f"  WebSocket   : {_ws_url()}")
    print(f"  Model       : {MODEL_NAME}")
    print(f"  Episodes    : {NUM_EPISODES_PER_TASK} per task")
    print("=" * 60)

    tasks = ["easy", "medium", "hard"]
    all_results: Dict[str, List[float]] = {}

    headers = {}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"

    try:
        async with websockets.connect(
            _ws_url(),
            additional_headers=headers,
            ping_interval=30,
            ping_timeout=60,
            close_timeout=10,
        ) as ws:
            for task_id in tasks:
                scores = []
                for i in range(NUM_EPISODES_PER_TASK):
                    seed = 42 + i
                    try:
                        score = await run_episode(ws, task_id, i, seed)
                        scores.append(score)
                    except Exception as e:
                        print(f"\n  ✗ Error: {e}")
                        scores.append(0.0)
                    await asyncio.sleep(0.3)
                all_results[task_id] = scores

    except Exception as e:
        print(f"\n✗ Connection failed: {_ws_url()}: {e}")
        print("  Ensure the environment server is running.")
        sys.exit(1)

    # Summary
    print("\n\n" + "=" * 60)
    print("BASELINE EVALUATION RESULTS")
    print("=" * 60)
    print(f"{'Task':<12} {'Mean':>8} {'Min':>8} {'Max':>8} {'Scores'}")
    print("-" * 60)
    for task_id in tasks:
        scores = all_results[task_id]
        if scores:
            mean_s = sum(scores) / len(scores)
            scores_str = ", ".join(f"{s:.2f}" for s in scores)
            print(f"{task_id:<12} {mean_s:>8.2f} {min(scores):>8.2f} {max(scores):>8.2f} [{scores_str}]")
        else:
            print(f"{task_id:<12} {'N/A':>8} {'N/A':>8} {'N/A':>8} []")
    print("=" * 60)

    flat = [s for v in all_results.values() for s in v]
    if flat:
        print(f"\nOverall mean reward: {sum(flat)/len(flat):.3f}")


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
