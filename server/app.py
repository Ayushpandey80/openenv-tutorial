"""
FastAPI server for the Clinical Triage Gym.

Creates the OpenEnv-compatible HTTP/WebSocket server that hosts
the TriageEnvironment.
"""

import sys
from pathlib import Path

# Ensure the project root is importable (for Docker / standalone execution)
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from openenv.core.env_server.http_server import create_app
except ImportError:
    from openenv.core.env_server.http_server import create_app

try:
    from server.triage_environment import TriageEnvironment
except ImportError:
    from triage_environment import TriageEnvironment

try:
    from models import TriageAction, TriageObservation
except ImportError:
    try:
        from ..models import TriageAction, TriageObservation
    except ImportError:
        from triage_env.models import TriageAction, TriageObservation


# Create the FastAPI app using OpenEnv's create_app helper
app = create_app(
    TriageEnvironment,
    TriageAction,
    TriageObservation,
    env_name="triage_env"
)


def main():
    """Run the server locally for development."""
    import uvicorn
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )


if __name__ == "__main__":
    main()
