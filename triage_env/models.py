# Copyright (c) 2024. Clinical Triage Gym.
# Licensed under the BSD 3-Clause License.

"""
Data models for the Clinical Triage Gym Environment.

Defines typed Pydantic models for agent actions and environment observations
in a clinical triage scenario based on the Emergency Severity Index (ESI).
"""

from typing import Any, Dict, List, Literal, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class TriageAction(Action):
    """Action taken by the triage agent.

    The agent can perform three types of actions during an episode:
    - "assess": Acknowledge reading the patient presentation (partial credit)
    - "clarify": Ask a clarifying question to gather more information
    - "triage": Make a final triage decision with urgency level and specialty routing

    Example:
        >>> # Acknowledge the patient presentation
        >>> TriageAction(action_type="assess")
        >>>
        >>> # Ask a clarifying question
        >>> TriageAction(action_type="clarify", question="Does the patient have a history of cardiac disease?")
        >>>
        >>> # Make a final triage decision
        >>> TriageAction(
        ...     action_type="triage",
        ...     urgency=2,
        ...     specialty="cardiology",
        ...     reasoning="Chest pain with elevated HR and mild SOB suggests acute coronary syndrome."
        ... )
    """

    action_type: Literal["assess", "clarify", "triage"] = Field(
        ..., description="Type of action: 'assess' to read patient info, 'clarify' to ask a question, 'triage' to make final decision"
    )
    question: Optional[str] = Field(
        default=None,
        description="Clarifying question to ask (required for 'clarify' action_type)",
    )
    urgency: Optional[int] = Field(
        default=None,
        ge=1,
        le=5,
        description="ESI urgency level 1-5 (1=resuscitation, 5=non-urgent). Required for 'triage' action_type.",
    )
    specialty: Optional[str] = Field(
        default=None,
        description="Target specialty for routing (e.g., 'cardiology', 'orthopedics'). Required for 'triage' action_type.",
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Clinical reasoning for the triage decision. Optional but encouraged.",
    )


class TriageObservation(Observation):
    """Observation returned to the triage agent.

    Contains patient information visible at the current step, plus
    episode metadata like step count and task difficulty.
    """

    patient_id: str = Field(
        default="", description="Unique identifier for the patient scenario"
    )
    chief_complaint: str = Field(
        default="", description="Patient's primary complaint"
    )
    vitals: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Patient vital signs: hr, bp, temp, spo2, rr",
    )
    history: Optional[str] = Field(
        default=None, description="Relevant past medical history"
    )
    additional_info: Optional[str] = Field(
        default=None,
        description="Response to a clarifying question asked by the agent",
    )
    available_specialties: List[str] = Field(
        default_factory=list,
        description="List of specialties the agent can route to",
    )
    step_number: int = Field(
        default=0, description="Current step number in the episode"
    )
    max_steps: int = Field(
        default=5, description="Maximum steps allowed in the episode"
    )
    task_id: str = Field(
        default="", description="Task identifier: 'easy', 'medium', or 'hard'"
    )
    task_difficulty: str = Field(
        default="", description="Human-readable difficulty label"
    )
    feedback: Optional[str] = Field(
        default=None,
        description="Feedback message from the environment (e.g., grading breakdown)",
    )
    reward_breakdown: Optional[Dict[str, float]] = Field(
        default=None,
        description="Detailed reward breakdown by component (only on terminal step)",
    )
