# 🌍 my_env — OpenEnv Echo Environment

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compatible-blue?logo=python&logoColor=white)](https://github.com/meta-pytorch/OpenEnv)
[![Hugging Face Space](https://img.shields.io/badge/🤗%20HF%20Space-suppayp%2Fmy--env-yellow)](https://huggingface.co/spaces/suppayp/my-env)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-server-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-green)](./LICENSE)

A minimal, production-ready [OpenEnv](https://github.com/meta-pytorch/OpenEnv) environment that **echoes back messages** with metadata. Built using the Gymnasium-style `step()` / `reset()` / `state()` API, deployable locally or on Hugging Face Spaces.

---

## 🚀 Quick Start

### Connect to the Live Space

```bash
pip install openenv-core
```

```python
import asyncio
from my_env import MyEnv, MyAction

async def main():
    async with MyEnv(base_url="https://suppayp-my-env.hf.space") as env:
        result = await env.reset()
        print(result.observation.echoed_message)  # "My Env environment ready!"

        result = await env.step(MyAction(message="Hello, OpenEnv!"))
        print(result.observation.echoed_message)  # "Hello, OpenEnv!"
        print(result.observation.message_length)  # 16
        print(result.reward)                       # 1.6

asyncio.run(main())
```

**Prefer synchronous code?** Use the `.sync()` wrapper:

```python
from my_env import MyEnv, MyAction

with MyEnv(base_url="https://suppayp-my-env.hf.space").sync() as env:
    result = env.reset()
    result = env.step(MyAction(message="Hello!"))
    print(result.observation.echoed_message)
```

---

## 📦 Project Structure

```
my_env/
├── __init__.py               # Exports MyAction, MyObservation, MyEnv
├── models.py                 # Pydantic Action & Observation types
├── client.py                 # MyEnv(EnvClient) — WebSocket client
├── openenv.yaml              # Environment manifest
├── pyproject.toml            # Dependencies & package config
└── server/
    ├── app.py                # FastAPI server entrypoint
    ├── my_env_environment.py # Core environment logic
    └── Dockerfile            # Container for HF Spaces / Docker
```

---

## 🧠 How It Works

### Environment Logic

The `MyEnvironment` class implements three core methods:

| Method | Description |
|--------|-------------|
| `reset()` | Starts a new episode, returns a ready message |
| `step(action)` | Echoes the message back + computes reward |
| `state` | Returns episode ID and current step count |

**Reward function:** `reward = len(message) * 0.1` — longer messages get higher rewards.

### Data Models

**`MyAction`**
```python
class MyAction(Action):
    message: str  # The message to send to the environment
```

**`MyObservation`**
```python
class MyObservation(Observation):
    echoed_message: str   # The echoed message
    message_length: int   # Length of the message
```

---

## 🛠️ Local Development

### 1. Clone & Install

```bash
git clone https://github.com/suppayp/my-env.git
cd my-env
pip install -e .
```

### 2. Run the Server

```bash
# Using the package entry point (recommended)
server

# Or directly with uvicorn
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Test the Endpoints

```bash
# Health check
curl http://localhost:8000/health
# {"status":"healthy"}

# OpenAPI docs
open http://localhost:8000/docs

# Interactive Web UI
open http://localhost:8000/web
```

### 4. Connect a Client

```python
import asyncio
from my_env import MyEnv, MyAction

async def main():
    async with MyEnv(base_url="http://localhost:8000") as env:
        result = await env.reset()
        print(result.observation.echoed_message)

        result = await env.step(MyAction(message="Testing locally!"))
        print(result.observation)

asyncio.run(main())
```

---

## 🐳 Docker

```bash
# Build the image
docker build -t my-env:latest -f server/Dockerfile .

# Run the container
docker run -d -p 8000:8000 my-env:latest

# Connect from Python (same as above, just different base_url)
```

---

## ☁️ Hugging Face Spaces Deployment

This environment is live on Hugging Face Spaces:

**🔗 [https://huggingface.co/spaces/suppayp/my-env](https://huggingface.co/spaces/suppayp/my-env)**

| Endpoint | URL |
|----------|-----|
| Space Page | https://huggingface.co/spaces/suppayp/my-env |
| Health | https://suppayp-my-env.hf.space/health |
| API Docs | https://suppayp-my-env.hf.space/docs |
| Web UI | https://suppayp-my-env.hf.space/web |
| WebSocket | wss://suppayp-my-env.hf.space/ws |

### Re-deploy

```bash
pip install openenv-core
openenv push --repo-id suppayp/my-env
```

---

## 🔌 Available API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | `GET` | Health check — `{"status": "healthy"}` |
| `/reset` | `POST` | Reset the environment |
| `/step` | `POST` | Execute an action |
| `/state` | `GET` | Get current episode state |
| `/ws` | `WebSocket` | Persistent session (used by client) |
| `/docs` | `GET` | Auto-generated OpenAPI documentation |
| `/web` | `GET` | Interactive browser UI |

---

## 🤖 Use with RL Frameworks

### TRL (GRPO Training)

```python
from trl import GRPOTrainer
from my_env import MyEnv, MyAction

# Use the HF Space as your RL environment
env = MyEnv(base_url="https://suppayp-my-env.hf.space").sync()
```

See [TRL + OpenEnv docs](https://huggingface.co/docs/trl/openenv) for full integration guide.

---

## 📋 Requirements

- Python 3.10+
- `openenv-core >= 0.2.2`
- FastAPI, Uvicorn, Pydantic (installed automatically)
- Docker *(optional, for containerized deployment)*

---

## 🧩 Built With

- [OpenEnv](https://github.com/meta-pytorch/OpenEnv) — Agentic Execution Environment framework
- [FastAPI](https://fastapi.tiangolo.com/) — Web server
- [Pydantic](https://docs.pydantic.dev/) — Type-safe data models
- [Hugging Face Spaces](https://huggingface.co/spaces) — Cloud deployment

---

## 📄 License

BSD 3-Clause License — see [LICENSE](./LICENSE) for details.
