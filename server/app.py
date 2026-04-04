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

from fastapi.responses import HTMLResponse

@app.get("/")
def read_root():
    html_content = """
    <html>
        <head>
            <title>Clinical Triage Gym</title>
            <style>
                body { font-family: -apple-system, system-ui, sans-serif; text-align: center; padding: 50px; background-color: #f8fafc; }
                .card { background: white; padding: 40px; border-radius: 12px; box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1); max-width: 600px; margin: 0 auto; }
                h1 { color: #0f172a; }
                p { color: #475569; font-size: 1.1em; line-height: 1.6; }
                .status { color: #10b981; font-weight: bold; margin-top: 20px; padding: 10px 20px; background: #d1fae5; border-radius: 8px; display: inline-block; }
            </style>
        </head>
        <body>
            <div class="card">
                <h1>🏥 Clinical Triage Gym</h1>
                <p>Welcome to the Clinical Triage Gym OpenEnv Environment!</p>
                <div class="status">🟢 Server is Running</div>
                <p style="margin-top: 30px; font-size: 0.9em; color: #64748b;">
                    <em>This is an OpenEnv API backend. Evaluators and Agents should connect via WebSocket to <code>/ws</code> or HTTP to <code>/reset</code> and <code>/step</code>.</em>
                </p>
            </div>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)


def main():
    """Run the server locally for development."""
    import uvicorn
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=7860,
        reload=True,
    )


if __name__ == "__main__":
    main()
