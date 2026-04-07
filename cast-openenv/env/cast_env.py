"""
CAST Environment - OpenEnv Compatible
Adaptive Sign Language Interpretation using Reinforcement Learning

State Space  : gesture × noise × context
Action Space : show_text | ask_repeat | trigger_alert
"""

import random
import yaml
import os
from typing import Tuple, Dict, Any


# ─────────────────────────────────────────────
#  State & Action Definitions
# ─────────────────────────────────────────────

GESTURES   = ["hello", "stop", "help", "danger"]
NOISE_LVLS = ["low", "medium", "high"]
CONTEXTS   = ["classroom", "road", "home"]

ACTIONS = {
    0: "show_text",
    1: "ask_repeat",
    2: "trigger_alert",
}

# Emergency gestures that need special handling
EMERGENCY_GESTURES = {"help", "danger"}

# Correct action for each (gesture, noise, context) combination
# Logic:
#   - Emergency gesture + high noise → trigger_alert
#   - Emergency gesture + low/medium noise → trigger_alert
#   - Non-emergency + high noise → ask_repeat
#   - Non-emergency + low/medium noise → show_text
def _get_correct_action(gesture: str, noise: str, context: str) -> int:
    if gesture in EMERGENCY_GESTURES:
        return 2  # trigger_alert
    if noise == "high":
        return 1  # ask_repeat
    return 0      # show_text


# ─────────────────────────────────────────────
#  CAST Environment
# ─────────────────────────────────────────────

class CASTEnv:
    """
    OpenEnv-compatible environment for sign language RL.

    Usage:
        env = CASTEnv()
        state = env.reset()
        for _ in range(steps):
            action = agent.select(state)
            next_state, reward, done = env.step(action)
            state = next_state
    """

    def __init__(self, config_path: str = None, max_steps: int = 20, seed: int = None):
        # Load config if provided
        if config_path and os.path.exists(config_path):
            with open(config_path, "r") as f:
                cfg = yaml.safe_load(f)
            max_steps = cfg.get("env", {}).get("max_steps", max_steps)

        self.max_steps   = max_steps
        self._step_count = 0
        self._rng        = random.Random(seed)
        self._state      = None

        # Observation / action space sizes (for the agent)
        self.n_states  = len(GESTURES) * len(NOISE_LVLS) * len(CONTEXTS)
        self.n_actions = len(ACTIONS)

    # ── Core API ──────────────────────────────

    def reset(self) -> Dict[str, str]:
        """Initialize / restart the environment. Returns initial state."""
        self._step_count = 0
        self._state = self._random_state()
        return dict(self._state)

    def step(self, action: int) -> Tuple[Dict[str, str], float, bool]:
        """
        Apply action to the environment.

        Returns:
            next_state : dict  – new environment state
            reward     : float – reward for the taken action
            done       : bool  – episode finished?
        """
        assert action in ACTIONS, f"Invalid action: {action}"

        reward = self._compute_reward(action)
        self._step_count += 1
        done = self._step_count >= self.max_steps

        # Transition to a new random state
        self._state = self._random_state()
        return dict(self._state), reward, done

    def state(self) -> Dict[str, str]:
        """Return current state (read-only snapshot)."""
        return dict(self._state)

    def render(self) -> str:
        """Human-readable string of the current state."""
        s = self._state
        return (
            f"[CAST] gesture={s['gesture']:<7} | "
            f"noise={s['noise']:<7} | "
            f"context={s['context']}"
        )

    # ── Helpers ───────────────────────────────

    def _random_state(self) -> Dict[str, str]:
        return {
            "gesture": self._rng.choice(GESTURES),
            "noise":   self._rng.choice(NOISE_LVLS),
            "context": self._rng.choice(CONTEXTS),
        }

    def _compute_reward(self, action: int) -> float:
        g = self._state["gesture"]
        n = self._state["noise"]
        c = self._state["context"]
        correct = _get_correct_action(g, n, c)

        reward = 0.0

        if action == correct:
            # Base correct reward
            reward += 1.0
            # Emergency bonus
            if g in EMERGENCY_GESTURES and action == 2:
                reward += 2.0  # +2 for correctly handling emergency
            # Speed bonus (rewarded every step for correct fast response)
            reward += 0.5
        else:
            reward -= 1.0      # Wrong interpretation penalty

        return reward

    # ── State encoding (for Q-table indexing) ─

    def encode_state(self, state: Dict[str, str]) -> int:
        """Convert state dict → integer index for Q-table lookup."""
        g = GESTURES.index(state["gesture"])
        n = NOISE_LVLS.index(state["noise"])
        c = CONTEXTS.index(state["context"])
        return g * (len(NOISE_LVLS) * len(CONTEXTS)) + n * len(CONTEXTS) + c

    def decode_state(self, index: int) -> Dict[str, str]:
        """Convert integer index → state dict."""
        c = index % len(CONTEXTS)
        index //= len(CONTEXTS)
        n = index % len(NOISE_LVLS)
        g = index // len(NOISE_LVLS)
        return {
            "gesture": GESTURES[g],
            "noise":   NOISE_LVLS[n],
            "context": CONTEXTS[c],
        }

    @staticmethod
    def action_name(action: int) -> str:
        return ACTIONS[action]
