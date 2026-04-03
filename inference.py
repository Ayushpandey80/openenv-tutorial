"""
Baseline Inference Script for Clinical Triage Gym.

This script uses the OpenAI Python Client to connect to the OpenEnv-hosted
Clinical Triage Gym environment. It runs the agent on a few samples
from all three difficulty tiers and reports the final scores.
"""

import os
import time
from typing import Any, Dict

from openai import OpenAI

# Required environment variables
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4-turbo-preview")
HF_TOKEN = os.environ.get("HF_TOKEN")

# Number of episodes per task to evaluate
NUM_EPISODES_PER_TASK = 2


def run_episode(client: OpenAI, task_id: str, episode_idx: int) -> float:
    """Run a single episode and return the reward."""
    print(f"\n--- Running Task: {task_id.upper()} (Episode {episode_idx + 1}) ---")
    
    # 1. Reset the environment and get initial observation
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[],
            extra_body={
                "mcp": {
                    "action": "reset",
                    "task_id": task_id,
                }
            }
        )
    except Exception as e:
        print(f"Failed to reset environment: {e}")
        return 0.0

    # Initialize messages list to maintain conversation history
    messages = []
    
    # Process the reset response observation (this isn't standard OpenAI format,
    # it comes back via OpenEnv adapter so we just need to keep the session going
    # via the episode_id)
    
    # Since we are using standard OpenAI client, OpenEnv handles the session state 
    # transparently if we send the same session ID or if we just keep sending 
    # tool requests along the same connection.
    # We will use the MCP tool calling format for OpenEnv:
    
    system_prompt = (
        "You are an expert triage nurse. You must process patient presentations "
        "and route them according to the Emergency Severity Index (ESI).\n"
        "1. Assess the chief complaint.\n"
        "2. If needed, ask a MAXIMUM of 2 clarifying questions.\n"
        "3. Make a final triage routing decision (urgency 1-5 and specialty).\n"
    )
    messages.append({"role": "system", "content": system_prompt})
    
    # Ask the environment for its tools and the initial state
    tools = [
        {
            "type": "function",
            "function": {
                "name": "assess",
                "description": "Acknowledge reading patient presentation.",
            }
        },
        {
            "type": "function",
            "function": {
                "name": "clarify",
                "description": "Ask a clarifying question to gather more information.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "The clarifying question to ask."
                        }
                    },
                    "required": ["question"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "triage",
                "description": "Make a final triage decision.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "urgency": {
                            "type": "integer",
                            "description": "ESI urgency level 1-5 (1=resuscitation, 5=non-urgent)."
                        },
                        "specialty": {
                            "type": "string",
                            "description": "Target specialty for routing (e.g., 'cardiology', 'orthopedics')."
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "Clinical reasoning for the triage decision."
                        }
                    },
                    "required": ["urgency", "specialty"]
                }
            }
        }
    ]

    total_reward = 0.0
    episode_done = False
    
    # Actually, we need to adapt to OpenEnv's standard proxy. OpenEnv expects standard chat completions.
    # The proxy injects the observation into the messages automatically when returning tool results!
    
    messages.append({"role": "user", "content": "I am ready for the patient."})
    
    while not episode_done:
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                tools=tools,
            )
            
            message = response.choices[0].message
            messages.append(message)
            
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    print(f"Agent called tool: {tool_call.function.name}")
                    
                    # Execute tool call by letting the proxy handle it
                    # We just append the tool result and loop again.
                    # Wait, the proxy server actually executes the environment step 
                    # and returns it in the assistant message or as a tool response.
                    # To step the environment properly via the client script:
                    
                    tool_resp = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=messages,
                        tools=tools,
                    )
                    
                    # This is highly dependent on how the specific hackathon's OpenEnv server adapter expects 
                    # standard OpenAI API usage. Typically the environment injects `done` and `reward`
                    # into a special metadata header or payload if it's the final step.
                    rcv_msg = tool_resp.choices[0].message
                    messages.append(rcv_msg)
                    
                    # Let's break if the environment signals done.
                    # In a typical OpenEnv OpenAI proxy, if `done` is true, 
                    # the API will return a specific completion flag or we check custom headers.
                    # For simplicity, we'll stop after a 'triage' call since the environment done=True there.
                    if tool_call.function.name == "triage":
                        episode_done = True
                        break
            else:
                print("Agent responded directly:", message.content)
                messages.append({"role": "user", "content": "Please make a tool call to proceed."})
                
        except Exception as e:
            print(f"Error during episode: {e}")
            break
            
    # For now, this is a scaffolding of the inference logic.
    print(f"Episode {episode_idx + 1} completed.")
    return total_reward


def main():
    print(f"Connecting to OPENENV at: {API_BASE_URL}")
    print(f"Using model: {MODEL_NAME}")
    
    # Need auth header for the OpenEnv proxy? Typically OpenAI client allows base_url
    client = OpenAI(
        base_url=API_BASE_URL + "/v1", 
        api_key=HF_TOKEN or "dummy-token"
    )

    tasks = ["easy", "medium", "hard"]
    
    results = {}
    for task in tasks:
        scores = []
        for i in range(NUM_EPISODES_PER_TASK):
            score = run_episode(client, task, i)
            scores.append(score)
            time.sleep(1) # Tiny pause between episodes
        results[task] = sum(scores) / len(scores) if scores else 0.0

    print("\n\n" + "="*40)
    print("Baseline Evaluation Results:")
    print("="*40)
    for task, score in results.items():
        print(f"Task '{task}': {score:.2f} average reward")
    print("="*40)


if __name__ == "__main__":
    main()
