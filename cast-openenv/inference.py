"""
inference.py — CAST OpenEnv Submission
========================================
Adaptive Sign Language Interpretation using Reinforcement Learning

Follows OpenEnv submission spec:
  - Environment vars: API_BASE_URL, MODEL_NAME, HF_TOKEN
  - All LLM calls via OpenAI client
  - Stdout logs in START / STEP / END format
"""

import os
import sys
import json

from openai import OpenAI

# ─────────────────────────────────────────────────────────────
#  Required Environment Variables  (per OpenEnv spec)
# ─────────────────────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1/")
MODEL_NAME   = os.getenv("MODEL_NAME",   "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN")        # NO default — must be injected at runtime

# Optional — for from_docker_image() usage
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

# ─────────────────────────────────────────────────────────────
#  OpenAI Client  (all LLM calls use this)
# ─────────────────────────────────────────────────────────────

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN,
)

# ─────────────────────────────────────────────────────────────
#  CAST Environment  (inline — no import needed for HF Space)
# ─────────────────────────────────────────────────────────────

GESTURES   = ["hello", "stop", "help", "danger"]
NOISE_LVLS = ["low", "medium", "high"]
CONTEXTS   = ["classroom", "road", "home"]
ACTIONS    = {0: "show_text", 1: "ask_repeat", 2: "trigger_alert"}
EMERGENCY  = {"help", "danger"}

import random

class CASTEnv:
    def __init__(self, max_steps=10, seed=42):
        self.max_steps   = max_steps
        self._step_count = 0
        self._rng        = random.Random(seed)
        self._state      = None
        self.n_states    = len(GESTURES) * len(NOISE_LVLS) * len(CONTEXTS)
        self.n_actions   = len(ACTIONS)

    def reset(self):
        self._step_count = 0
        self._state = self._random_state()
        return dict(self._state)

    def step(self, action):
        reward           = self._compute_reward(action)
        self._step_count += 1
        done             = self._step_count >= self.max_steps
        self._state      = self._random_state()
        return dict(self._state), reward, done

    def _random_state(self):
        return {
            "gesture": self._rng.choice(GESTURES),
            "noise":   self._rng.choice(NOISE_LVLS),
            "context": self._rng.choice(CONTEXTS),
        }

    def _compute_reward(self, action):
        g, n = self._state["gesture"], self._state["noise"]
        if g in EMERGENCY:
            correct = 2
        elif n == "high":
            correct = 1
        else:
            correct = 0

        if action == correct:
            r = 1.0 + 0.5  # correct + fast
            if g in EMERGENCY and action == 2:
                r += 2.0   # emergency bonus
            return r
        return -1.0

    @staticmethod
    def action_name(action):
        return ACTIONS[action]


# ─────────────────────────────────────────────────────────────
#  LLM-based Agent  (uses OpenAI client)
# ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a sign language interpretation AI agent.

Given a state (gesture, noise level, context), choose the best action:
- show_text     (0): gesture is clear, show the interpreted text directly
- ask_repeat    (1): too much noise, ask the signer to repeat
- trigger_alert (2): emergency gesture detected (help/danger), fire alert immediately

Rules:
1. If gesture is "help" or "danger" → ALWAYS choose trigger_alert (2)
2. If noise is "high" and gesture is NOT emergency → choose ask_repeat (1)
3. Otherwise → choose show_text (0)

Respond ONLY with a JSON object: {"action": <0|1|2>, "reason": "<brief explanation>"}
"""

def llm_decide(state: dict) -> tuple[int, str]:
    """Ask the LLM to decide the best action given current state."""
    user_msg = (
        f"Current state:\n"
        f"  gesture : {state['gesture']}\n"
        f"  noise   : {state['noise']}\n"
        f"  context : {state['context']}\n\n"
        f"What action should the agent take?"
    )

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        max_tokens=100,
        temperature=0.1,
    )

    raw = response.choices[0].message.content.strip()

    # Parse JSON response
    try:
        # Handle markdown code blocks if model wraps output
        if "```" in raw:
            raw = raw.split("```")[1].strip()
            if raw.startswith("json"):
                raw = raw[4:].strip()
        parsed = json.loads(raw)
        action = int(parsed["action"])
        reason = parsed.get("reason", "")
    except Exception:
        # Fallback: extract action digit from raw text
        for char in raw:
            if char in "012":
                action = int(char)
                reason = raw
                break
        else:
            action = 0
            reason = "parse fallback"

    # Safety clamp
    action = max(0, min(2, action))
    return action, reason


# ─────────────────────────────────────────────────────────────
#  Main Agent Loop  (START / STEP / END format)
# ─────────────────────────────────────────────────────────────

def run_agent(episodes: int = 3, steps_per_episode: int = 5):
    """
    Runs the CAST RL agent with LLM decision-making.
    Logs in the required OpenEnv structured format.
    """
    print("START")
    sys.stdout.flush()

    env          = CASTEnv(max_steps=steps_per_episode)
    total_reward = 0.0
    step_count   = 0

    for ep in range(1, episodes + 1):
        state = env.reset()
        done  = False

        print(f"STEP: episode={ep} phase=reset state={json.dumps(state)}")
        sys.stdout.flush()

        while not done:
            step_count += 1

            # LLM decides action
            action, reason = llm_decide(state)
            action_name    = CASTEnv.action_name(action)

            print(f"STEP: episode={ep} step={step_count} "
                  f"state={json.dumps(state)} "
                  f"action={action_name} "
                  f"reason=\"{reason}\"")
            sys.stdout.flush()

            # Environment step
            next_state, reward, done = env.step(action)
            total_reward += reward

            print(f"STEP: episode={ep} step={step_count} "
                  f"reward={reward:+.1f} cumulative_reward={total_reward:+.1f} done={done}")
            sys.stdout.flush()

            state = next_state

    print(f"END: total_episodes={episodes} total_steps={step_count} "
          f"total_reward={total_reward:+.2f} "
          f"avg_reward_per_step={total_reward/max(step_count,1):+.3f}")
    sys.stdout.flush()


# ─────────────────────────────────────────────────────────────
#  Entry Point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Allow overriding episodes from CLI for testing
    episodes = int(os.getenv("CAST_EPISODES", "3"))
    steps    = int(os.getenv("CAST_STEPS",    "5"))

    run_agent(episodes=episodes, steps_per_episode=steps)
