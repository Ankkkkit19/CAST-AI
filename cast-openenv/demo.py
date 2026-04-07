"""
demo.py — CAST RL Interactive Demo
=====================================
Runs pre-defined scenarios to showcase the trained agent's behavior.

Usage:
    python demo.py                   # uses saved Q-table
    python demo.py --random          # compare with random baseline
    python demo.py --interactive     # type your own gesture/noise/context

Scenarios demonstrated:
    1. Emergency gesture "help" + high noise → trigger_alert
    2. Clear gesture "hello" + low noise    → show_text
    3. Ambiguous: non-emergency + high noise → ask_repeat
"""

import os
import sys
import time
import yaml
import argparse

sys.path.insert(0, os.path.dirname(__file__))

from env.cast_env   import CASTEnv, GESTURES, NOISE_LVLS, CONTEXTS
from agent.rl_agent import QLearningAgent, RandomAgent


# ─────────────────────────────────────────────
#  Display Helpers
# ─────────────────────────────────────────────

COLORS = {
    "reset":  "\033[0m",
    "bold":   "\033[1m",
    "red":    "\033[91m",
    "green":  "\033[92m",
    "yellow": "\033[93m",
    "blue":   "\033[94m",
    "cyan":   "\033[96m",
    "purple": "\033[95m",
}

def c(text, color):
    return f"{COLORS.get(color,'')}{text}{COLORS['reset']}"

ACTION_DISPLAY = {
    "show_text":     c("📢  SHOW TEXT",       "green"),
    "ask_repeat":    c("🔄  ASK REPEAT",      "yellow"),
    "trigger_alert": c("🚨  TRIGGER ALERT",   "red"),
}

GESTURE_EMOJI = {
    "hello":  "👋",
    "stop":   "✋",
    "help":   "🆘",
    "danger": "⚠️ ",
}

NOISE_BAR = {
    "low":    "▓░░ LOW",
    "medium": "▓▓░ MEDIUM",
    "high":   "▓▓▓ HIGH",
}

def banner():
    print("\n" + c("═" * 62, "cyan"))
    print(c("  🤟  CAST RL — Sign Language Interpretation DEMO", "bold"))
    print(c("  Adaptive RL Agent in Action", "purple"))
    print(c("═" * 62, "cyan") + "\n")

def print_state(state: dict):
    g = state["gesture"]
    n = state["noise"]
    ctx = state["context"]
    print(f"  {c('INPUT STATE', 'bold')}")
    print(f"    Gesture  : {GESTURE_EMOJI.get(g, '❓')} {c(g.upper(), 'cyan')}")
    print(f"    Noise    : {NOISE_BAR[n]}")
    print(f"    Context  : 📍 {ctx.capitalize()}")

def print_action(action_name: str, reward: float):
    display = ACTION_DISPLAY.get(action_name, action_name)
    reward_color = "green" if reward > 0 else "red"
    print(f"\n  {c('AGENT DECISION', 'bold')}")
    print(f"    Action   : {display}")
    print(f"    Reward   : {c(f'{reward:+.1f}', reward_color)}")

def print_scenario_header(n: int, title: str):
    print("\n" + c(f"  ─── Scenario {n}: {title} ───", "blue"))
    print()

def animate_thinking(msg="  🧠 Agent deciding...", delay=0.6):
    print(msg, end="", flush=True)
    for _ in range(3):
        time.sleep(delay / 3)
        print(".", end="", flush=True)
    print()


# ─────────────────────────────────────────────
#  Demo Runner
# ─────────────────────────────────────────────

def run_scenario(agent, env, state_dict: dict, scenario_num: int, title: str):
    print_scenario_header(scenario_num, title)
    print_state(state_dict)
    animate_thinking()

    idx    = env.encode_state(state_dict)
    action = (
        agent.greedy_action(idx)
        if hasattr(agent, "greedy_action")
        else agent.select_action(idx)
    )
    action_name = env.action_name(action)

    # Simulate step to get reward
    env._state = state_dict        # inject specific state
    _, reward, _ = env.step(action)

    print_action(action_name, reward)

    # Explain WHY
    g = state_dict["gesture"]
    n = state_dict["noise"]
    explanation = ""
    if g in {"help", "danger"}:
        explanation = "Emergency gesture detected → alert triggered regardless of noise."
    elif n == "high":
        explanation = "High noise makes gesture ambiguous → agent requests clarification."
    else:
        explanation = "Clear signal in low noise → agent confident to show text."

    print(f"\n  {c('💡 Reasoning:', 'purple')} {explanation}")
    print()


def interactive_demo(agent, env):
    print(c("\n  🎮 INTERACTIVE MODE — Type your own scenario\n", "bold"))
    print("  Available values:")
    print(f"    Gestures : {', '.join(GESTURES)}")
    print(f"    Noise    : {', '.join(NOISE_LVLS)}")
    print(f"    Context  : {', '.join(CONTEXTS)}")
    print()

    while True:
        try:
            gesture = input(c("  Enter gesture  (or 'q' to quit): ", "cyan")).strip().lower()
            if gesture == "q":
                break
            if gesture not in GESTURES:
                print(f"  ⚠ Invalid. Choose from {GESTURES}")
                continue

            noise = input(c("  Enter noise level : ", "cyan")).strip().lower()
            if noise not in NOISE_LVLS:
                print(f"  ⚠ Invalid. Choose from {NOISE_LVLS}")
                continue

            context = input(c("  Enter context     : ", "cyan")).strip().lower()
            if context not in CONTEXTS:
                print(f"  ⚠ Invalid. Choose from {CONTEXTS}")
                continue

            state = {"gesture": gesture, "noise": noise, "context": context}
            run_scenario(agent, env, state, "?", "User Input")

        except (KeyboardInterrupt, EOFError):
            break

    print(c("\n  Demo ended. Thanks! 🤟\n", "green"))


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CAST RL Demo")
    parser.add_argument("--config",      default="config/openenv.yaml")
    parser.add_argument("--model",       default="models/q_table.json")
    parser.add_argument("--random",      action="store_true", help="Use random agent instead")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    env = CASTEnv(config_path=args.config, seed=42)

    # Load agent
    if args.random:
        print(c("  [Demo] Using RandomAgent (no learning)\n", "yellow"))
        agent = RandomAgent(env.n_states, env.n_actions)
    else:
        a_cfg = cfg.get("agent", {})
        agent = QLearningAgent(
            n_states      = env.n_states,
            n_actions     = env.n_actions,
            epsilon       = 0.0,          # greedy during demo
            epsilon_min   = 0.0,
            epsilon_decay = 1.0,
        )
        if os.path.exists(args.model):
            agent.load(args.model)
        else:
            print(c(f"  ⚠ No model found at '{args.model}'. Run train.py first!", "yellow"))
            print(c("  Running with untrained agent...\n", "yellow"))

    banner()

    if args.interactive:
        interactive_demo(agent, env)
        return

    # Pre-defined scenarios from roadmap
    scenarios = [
        ({"gesture": "help",   "noise": "high",   "context": "road"},      "Emergency + High Noise"),
        ({"gesture": "hello",  "noise": "low",    "context": "classroom"}, "Clear Greeting"),
        ({"gesture": "danger", "noise": "medium", "context": "home"},      "Danger at Home"),
        ({"gesture": "stop",   "noise": "high",   "context": "classroom"}, "Stop Signal + High Noise"),
    ]

    for i, (state, title) in enumerate(scenarios, 1):
        run_scenario(agent, env, state, i, title)
        time.sleep(0.3)

    print(c("═" * 62, "cyan"))
    print(c("  ✅ Demo Complete! CAST RL System working as expected.", "green"))
    print(c("═" * 62, "cyan") + "\n")


if __name__ == "__main__":
    main()
