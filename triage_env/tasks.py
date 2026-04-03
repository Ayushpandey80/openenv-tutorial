"""
Task definitions and grading logic for the Clinical Triage Gym.

Three difficulty tiers with deterministic, ESI-based graders that
produce scores in [0.0, 1.0].
"""

from typing import Any, Dict, List, Optional, Tuple

# Available specialties across all tasks
SPECIALTIES = [
    "emergency_medicine",
    "cardiology",
    "neurology",
    "orthopedics",
    "internal_medicine",
    "surgery",
    "pediatrics",
    "psychiatry",
    "pulmonology",
    "gastroenterology",
    "nephrology",
    "obstetrics_gynecology",
    "dermatology",
    "ophthalmology",
    "ent",
    "urology",
    "family_medicine",
    "geriatrics",
    "neurosurgery",
    "oncology",
]

# Task definitions
TASKS = {
    "easy": {
        "name": "Basic Triage",
        "description": (
            "Single clear symptom with obvious specialty routing and "
            "unambiguous urgency level. Tests fundamental triage skills."
        ),
        "scenario_filter": "easy",
        "expected_baseline": 0.85,
    },
    "medium": {
        "name": "Ambiguous Presentation",
        "description": (
            "Multiple symptoms with ambiguous urgency. May require "
            "clarifying questions before making a safe triage decision."
        ),
        "scenario_filter": "medium",
        "expected_baseline": 0.55,
    },
    "hard": {
        "name": "Complex Multi-System",
        "description": (
            "Conflicting clinical signals, comorbidities, and partial "
            "information. Dangerous misroutes are penalized heavily. "
            "Requires careful information gathering and clinical reasoning."
        ),
        "scenario_filter": "hard",
        "expected_baseline": 0.30,
    },
}


def compute_reward(
    action_history: List[Dict[str, Any]],
    ground_truth: Dict[str, Any],
    total_steps: int,
) -> Tuple[float, Dict[str, float]]:
    """Compute the triage reward based on the agent's final decision and trajectory.

    Args:
        action_history: List of all actions taken during the episode.
        ground_truth: The ground truth for the scenario.
        total_steps: Total number of steps taken.

    Returns:
        Tuple of (clamped_reward, breakdown_dict).
    """
    reward = 0.0
    breakdown: Dict[str, float] = {}

    # Extract final triage action (the last "triage" action)
    triage_action = None
    did_assess = False
    did_clarify = False

    for act in action_history:
        if act.get("action_type") == "assess":
            did_assess = True
        elif act.get("action_type") == "clarify":
            did_clarify = True
        elif act.get("action_type") == "triage":
            triage_action = act

    # If agent never made a triage decision, score is 0
    if triage_action is None:
        breakdown["no_decision"] = 0.0
        return 0.0, breakdown

    gt_urgency = ground_truth["urgency"]
    gt_specialty = ground_truth["specialty"]
    gt_acceptable = ground_truth.get("acceptable_specialties", [gt_specialty])
    gt_requires_clarification = ground_truth.get("requires_clarification", False)
    gt_optimal_steps = ground_truth.get("optimal_steps", 2)
    gt_danger_misroutes = ground_truth.get("danger_misroutes", [])

    agent_urgency = triage_action.get("urgency")
    agent_specialty = (triage_action.get("specialty") or "").lower().strip()

    # --- Component 1: Urgency match (0.30) ---
    if agent_urgency is not None and gt_urgency is not None:
        urgency_diff = abs(agent_urgency - gt_urgency)
        if urgency_diff == 0:
            breakdown["urgency"] = 0.30
        elif urgency_diff == 1:
            breakdown["urgency"] = 0.15
        else:
            breakdown["urgency"] = 0.0
    else:
        breakdown["urgency"] = 0.0
    reward += breakdown["urgency"]

    # --- Component 2: Specialty routing (0.30) ---
    if agent_specialty == gt_specialty:
        breakdown["routing"] = 0.30
    elif agent_specialty in gt_acceptable:
        breakdown["routing"] = 0.15
    else:
        breakdown["routing"] = 0.0
    reward += breakdown["routing"]

    # --- Component 3: Clarification awareness (0.15) ---
    if gt_requires_clarification and did_clarify:
        breakdown["clarification"] = 0.15
    elif not gt_requires_clarification:
        # No clarification needed — agent gets credit either way
        breakdown["clarification"] = 0.15
    else:
        breakdown["clarification"] = 0.0
    reward += breakdown["clarification"]

    # --- Component 4: Efficiency (0.10) ---
    if total_steps <= gt_optimal_steps:
        breakdown["efficiency"] = 0.10
    elif total_steps <= gt_optimal_steps + 1:
        breakdown["efficiency"] = 0.05
    else:
        breakdown["efficiency"] = 0.0
    reward += breakdown["efficiency"]

    # --- Component 5: Acknowledged complaint (0.15) ---
    if did_assess:
        breakdown["acknowledge"] = 0.15
    else:
        breakdown["acknowledge"] = 0.0
    reward += breakdown["acknowledge"]

    # --- Penalty: Dangerous misroute (-0.50) ---
    if agent_specialty in gt_danger_misroutes and gt_urgency <= 2:
        breakdown["danger_penalty"] = -0.50
        reward -= 0.50
    else:
        breakdown["danger_penalty"] = 0.0

    # Clamp to [0.0, 1.0]
    clamped = max(0.0, min(1.0, reward))
    return clamped, breakdown
