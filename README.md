# 🏥 Clinical Triage Gym

An OpenEnv-compatible RL environment for training AI agents on clinical triage — assessing patient urgency and routing to the correct medical specialty, scored against the **Emergency Severity Index (ESI)**.

## Quick Start

```bash
pip install openenv-core[core] faker
python -m uvicorn server.app:app --host 0.0.0.0 --port 8000
```

## Action Space

| Type | Fields | Description |
|------|--------|-------------|
| `assess` | — | Acknowledge patient presentation |
| `clarify` | `question` | Ask clarifying question (max 2) |
| `triage` | `urgency`, `specialty`, `reasoning` | Final decision |

## Observation Space

`chief_complaint`, `vitals`, `history`, `additional_info`, `available_specialties`, `step_number`, `max_steps`

## Tasks

| Task | Difficulty | Expected Baseline |
|------|-----------|-------------------|
| Basic Triage | Easy | ~0.85 |
| Ambiguous Presentation | Medium | ~0.55 |
| Complex Multi-System | Hard | ~0.30 |

## Reward Function

| Component | Weight |
|-----------|--------|
| Urgency match | 0.30 |
| Specialty routing | 0.30 |
| Clarification | 0.15 |
| Efficiency | 0.10 |
| Acknowledge | 0.15 |
| **Danger penalty** | **-0.50** |

## License

BSD 3-Clause
