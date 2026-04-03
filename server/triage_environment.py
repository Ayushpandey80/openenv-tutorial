"""
Clinical Triage Environment — core environment logic.

Implements the OpenEnv Environment interface for medical triage simulation.
The agent receives patient intake data and must assess urgency, route to the
correct specialty, and optionally ask clarifying questions.
"""

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import Action, Observation, State

# Use relative imports when running as a package, absolute when standalone
try:
    from ..models import TriageAction, TriageObservation
    from ..tasks import SPECIALTIES, TASKS, compute_reward
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from models import TriageAction, TriageObservation
    from tasks import SPECIALTIES, TASKS, compute_reward


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MAX_STEPS = 5
MAX_CLARIFICATIONS = 2


def _load_scenarios() -> List[Dict[str, Any]]:
    """Load patient scenarios from JSON."""
    path = DATA_DIR / "scenarios.json"
    if not path.exists():
        raise FileNotFoundError(f"Scenarios file not found: {path}")
    with open(path) as f:
        return json.load(f)


class TriageEnvironment(Environment):
    """Clinical triage RL environment.

    The agent interacts through three action types:
    - "assess": Read and acknowledge patient presentation
    - "clarify": Ask a clarifying question (up to 2 per episode)
    - "triage": Make a final decision (urgency + specialty)

    Episodes end when the agent submits a "triage" action or reaches MAX_STEPS.
    """

    def __init__(self):
        super().__init__()
        self._scenarios = _load_scenarios()
        self._state = State(episode_id=None, step_count=0)
        self._current_scenario: Optional[Dict[str, Any]] = None
        self._action_history: List[Dict[str, Any]] = []
        self._clarifications_used = 0
        self._done = False
        self._task_id = ""

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: Optional[str] = None,
        **kwargs: Any,
    ) -> TriageObservation:
        """Reset the environment and present a new patient scenario.

        Args:
            seed: Random seed for scenario selection.
            episode_id: Optional custom episode identifier.
            task_id: Task difficulty filter: 'easy', 'medium', or 'hard'.
                     If None, picks from all scenarios.
        """
        if seed is not None:
            random.seed(seed)

        # Filter scenarios by task_id (difficulty)
        self._task_id = task_id or kwargs.get("task_id", "")
        if self._task_id and self._task_id in TASKS:
            filtered = [
                s for s in self._scenarios
                if s.get("difficulty") == TASKS[self._task_id]["scenario_filter"]
            ]
        else:
            filtered = self._scenarios

        if not filtered:
            filtered = self._scenarios

        self._current_scenario = random.choice(filtered)
        self._action_history = []
        self._clarifications_used = 0
        self._done = False

        self._state = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
        )

        patient = self._current_scenario["patient"]
        difficulty = self._current_scenario.get("difficulty", "unknown")

        return TriageObservation(
            done=False,
            reward=0.0,
            patient_id=self._current_scenario["id"],
            chief_complaint=patient["chief_complaint"],
            vitals=patient.get("vitals"),
            history=patient.get("history"),
            additional_info=None,
            available_specialties=SPECIALTIES,
            step_number=0,
            max_steps=MAX_STEPS,
            task_id=self._task_id or difficulty,
            task_difficulty=difficulty,
            feedback="Patient has arrived. Review the presentation and make a triage decision.",
        )

    def step(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> TriageObservation:
        """Execute a triage action.

        Args:
            action: TriageAction with action_type of 'assess', 'clarify', or 'triage'.
        """
        if self._done:
            return self._terminal_observation(
                "Episode already completed. Call reset() to start a new one."
            )

        self._state.step_count += 1

        # Parse the action
        if isinstance(action, TriageAction):
            act_data = action.model_dump()
        elif isinstance(action, dict):
            act_data = action
        else:
            act_data = action.model_dump() if hasattr(action, "model_dump") else {}

        action_type = act_data.get("action_type", "")
        self._action_history.append(act_data)

        patient = self._current_scenario["patient"]
        scenario = self._current_scenario

        # --- Handle "assess" ---
        if action_type == "assess":
            return TriageObservation(
                done=False,
                reward=0.0,
                patient_id=scenario["id"],
                chief_complaint=patient["chief_complaint"],
                vitals=patient.get("vitals"),
                history=patient.get("history"),
                additional_info=None,
                available_specialties=SPECIALTIES,
                step_number=self._state.step_count,
                max_steps=MAX_STEPS,
                task_id=self._task_id or scenario.get("difficulty", ""),
                task_difficulty=scenario.get("difficulty", ""),
                feedback="Assessment acknowledged. You may clarify or make your triage decision.",
            )

        # --- Handle "clarify" ---
        if action_type == "clarify":
            if self._clarifications_used >= MAX_CLARIFICATIONS:
                return TriageObservation(
                    done=False,
                    reward=0.0,
                    patient_id=scenario["id"],
                    chief_complaint=patient["chief_complaint"],
                    vitals=patient.get("vitals"),
                    history=patient.get("history"),
                    additional_info="No more clarifications allowed. Please make your triage decision.",
                    available_specialties=SPECIALTIES,
                    step_number=self._state.step_count,
                    max_steps=MAX_STEPS,
                    task_id=self._task_id or scenario.get("difficulty", ""),
                    task_difficulty=scenario.get("difficulty", ""),
                    feedback="Maximum clarifications reached.",
                )

            self._clarifications_used += 1
            question = (act_data.get("question") or "").lower().strip()
            response = self._answer_clarification(question, patient)

            return TriageObservation(
                done=False,
                reward=0.0,
                patient_id=scenario["id"],
                chief_complaint=patient["chief_complaint"],
                vitals=patient.get("vitals"),
                history=patient.get("history"),
                additional_info=response,
                available_specialties=SPECIALTIES,
                step_number=self._state.step_count,
                max_steps=MAX_STEPS,
                task_id=self._task_id or scenario.get("difficulty", ""),
                task_difficulty=scenario.get("difficulty", ""),
                feedback=f"Clarification response provided. ({MAX_CLARIFICATIONS - self._clarifications_used} remaining)",
            )

        # --- Handle "triage" (final decision) ---
        if action_type == "triage":
            self._done = True
            ground_truth = scenario["ground_truth"]
            reward, breakdown = compute_reward(
                self._action_history, ground_truth, self._state.step_count
            )

            return TriageObservation(
                done=True,
                reward=reward,
                patient_id=scenario["id"],
                chief_complaint=patient["chief_complaint"],
                vitals=patient.get("vitals"),
                history=patient.get("history"),
                additional_info=None,
                available_specialties=SPECIALTIES,
                step_number=self._state.step_count,
                max_steps=MAX_STEPS,
                task_id=self._task_id or scenario.get("difficulty", ""),
                task_difficulty=scenario.get("difficulty", ""),
                feedback=f"Triage decision recorded. Score: {reward:.2f}",
                reward_breakdown=breakdown,
            )

        # --- Handle max steps ---
        if self._state.step_count >= MAX_STEPS:
            self._done = True
            ground_truth = scenario["ground_truth"]
            reward, breakdown = compute_reward(
                self._action_history, ground_truth, self._state.step_count
            )
            breakdown["timeout_penalty"] = -0.20
            reward = max(0.0, reward - 0.20)

            return TriageObservation(
                done=True,
                reward=reward,
                patient_id=scenario["id"],
                chief_complaint=patient["chief_complaint"],
                vitals=patient.get("vitals"),
                history=patient.get("history"),
                additional_info=None,
                available_specialties=SPECIALTIES,
                step_number=self._state.step_count,
                max_steps=MAX_STEPS,
                task_id=self._task_id or scenario.get("difficulty", ""),
                task_difficulty=scenario.get("difficulty", ""),
                feedback="Episode timed out. Decision was not submitted in time.",
                reward_breakdown=breakdown,
            )

        # Unknown action type
        return TriageObservation(
            done=False,
            reward=0.0,
            patient_id=scenario["id"],
            chief_complaint=patient["chief_complaint"],
            vitals=patient.get("vitals"),
            history=patient.get("history"),
            additional_info=None,
            available_specialties=SPECIALTIES,
            step_number=self._state.step_count,
            max_steps=MAX_STEPS,
            task_id=self._task_id or scenario.get("difficulty", ""),
            task_difficulty=scenario.get("difficulty", ""),
            feedback=f"Unknown action_type '{action_type}'. Use 'assess', 'clarify', or 'triage'.",
        )

    def _answer_clarification(self, question: str, patient: Dict[str, Any]) -> str:
        """Match the agent's question to available clarification responses."""
        responses = patient.get("clarification_responses", {})
        if not question or not responses:
            return "No additional information available for that question."

        # Fuzzy keyword matching against available response keys
        best_match = None
        best_score = 0
        for key, value in responses.items():
            key_words = set(key.lower().replace("_", " ").split())
            question_words = set(question.lower().split())
            overlap = len(key_words & question_words)
            if overlap > best_score:
                best_score = overlap
                best_match = value

        if best_match and best_score > 0:
            return best_match

        # If no keyword match, return the first available response
        return next(iter(responses.values()))

    def _terminal_observation(self, msg: str) -> TriageObservation:
        """Return a terminal observation with a message."""
        return TriageObservation(
            done=True,
            reward=0.0,
            patient_id=self._current_scenario["id"] if self._current_scenario else "",
            chief_complaint="",
            available_specialties=SPECIALTIES,
            step_number=self._state.step_count,
            max_steps=MAX_STEPS,
            task_id=self._task_id,
            task_difficulty="",
            feedback=msg,
        )

    @property
    def state(self) -> State:
        return self._state
