"""
train.py — CAST RL Training Script
====================================
Trains a Q-Learning agent in the CAST sign-language environment.

Usage:
    python train.py                    # default config
    python train.py --episodes 2000    # custom episodes
    python train.py --agent random     # baseline run

Output:
    - Live reward logs every N episodes
    - Saved Q-table → models/q_table.json
    - Training summary
"""

import argparse
import os
import sys
import yaml
import json
import time
from collections import deque

# Make sure local modules are importable
sys.path.insert(0, os.path.dirname(__file__))

from env.cast_env   import CASTEnv
from agent.rl_agent import RandomAgent, QLearningAgent


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────

def load_config(path: str = "config/openenv.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def make_agent(cfg: dict, n_states: int, n_actions: int):
    agent_type = cfg.get("agent", {}).get("type", "q_learning")
    a = cfg.get("agent", {})
    if agent_type == "random":
        print("[Train] Using RandomAgent (baseline)")
        return RandomAgent(n_states, n_actions)
    else:
        print("[Train] Using QLearningAgent")
        return QLearningAgent(
            n_states      = n_states,
            n_actions     = n_actions,
            alpha         = a.get("alpha",         0.1),
            gamma         = a.get("gamma",         0.95),
            epsilon       = a.get("epsilon",       1.0),
            epsilon_min   = a.get("epsilon_min",   0.05),
            epsilon_decay = a.get("epsilon_decay", 0.995),
            seed          = a.get("seed",          42),
        )

def print_banner():
    print("=" * 60)
    print("  🤟 CAST RL — Sign Language Interpretation Agent")
    print("  Reinforcement Learning Training Session")
    print("=" * 60)

def print_progress(ep, total, rewards_window, epsilon=None):
    avg = sum(rewards_window) / max(len(rewards_window), 1)
    eps_str = f"  ε={epsilon:.4f}" if epsilon is not None else ""
    bar_len = 30
    filled  = int(bar_len * ep / total)
    bar     = "█" * filled + "░" * (bar_len - filled)
    print(f"  [{bar}] Ep {ep:>5}/{total}  AvgReward={avg:+.2f}{eps_str}")


# ─────────────────────────────────────────────
#  Main Training Loop
# ─────────────────────────────────────────────

def train(config_path: str = "config/openenv.yaml", episodes: int = None, agent_type: str = None):
    cfg = load_config(config_path)

    # Override from CLI if provided
    if episodes:
        cfg.setdefault("training", {})["episodes"] = episodes
    if agent_type:
        cfg.setdefault("agent", {})["type"] = agent_type

    t_cfg = cfg.get("training", {})
    n_episodes   = t_cfg.get("episodes",      1000)
    log_interval = t_cfg.get("log_interval",  100)
    save_path    = t_cfg.get("save_path",     "models/q_table.json")
    reward_win   = t_cfg.get("reward_window", 50)

    print_banner()
    print(f"\n[Config] Episodes={n_episodes}  LogEvery={log_interval}  SaveTo={save_path}\n")

    # Setup
    env   = CASTEnv(config_path=config_path, seed=cfg.get("env", {}).get("seed", 42))
    agent = make_agent(cfg, env.n_states, env.n_actions)

    rewards_window = deque(maxlen=reward_win)
    all_rewards    = []
    total_steps    = 0
    start_time     = time.time()

    # Training
    for ep in range(1, n_episodes + 1):
        state      = env.reset()
        state_idx  = env.encode_state(state)
        ep_reward  = 0.0
        done       = False

        while not done:
            action                       = agent.select_action(state_idx)
            next_state, reward, done     = env.step(action)
            next_idx                     = env.encode_state(next_state)

            agent.update(state_idx, action, reward, next_idx, done)

            state_idx  = next_idx
            ep_reward += reward
            total_steps += 1

        rewards_window.append(ep_reward)
        all_rewards.append(ep_reward)

        # Logging
        if ep % log_interval == 0 or ep == n_episodes:
            epsilon = getattr(agent, "epsilon", None)
            print_progress(ep, n_episodes, rewards_window, epsilon)

    # ── Save model ────────────────────────────
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    agent.save(save_path)

    # Save reward history for plotting
    history_path = save_path.replace(".json", "_history.json")
    with open(history_path, "w") as f:
        json.dump(all_rewards, f)

    # ── Final Summary ─────────────────────────
    elapsed = time.time() - start_time
    avg_last = sum(list(rewards_window)[-50:]) / min(50, len(rewards_window))
    print("\n" + "=" * 60)
    print("  ✅ Training Complete!")
    print(f"  Total Episodes  : {n_episodes}")
    print(f"  Total Steps     : {total_steps}")
    print(f"  Time Elapsed    : {elapsed:.1f}s")
    print(f"  Avg Reward (last {reward_win} eps): {avg_last:+.3f}")
    print(f"  Model Saved     : {save_path}")
    print("=" * 60)

    # Print learned policy
    if hasattr(agent, "best_action_summary"):
        print("\n📋 Learned Policy (State → Best Action):")
        print("-" * 50)
        for state_key, action_name in agent.best_action_summary(env).items():
            g, n, c = state_key.split("|")
            print(f"  gesture={g:<7} noise={n:<7} ctx={c:<12} → {action_name}")


# ─────────────────────────────────────────────
#  CLI Entry Point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CAST RL Agent")
    parser.add_argument("--config",   default="config/openenv.yaml", help="Path to config YAML")
    parser.add_argument("--episodes", type=int, default=None,        help="Number of training episodes")
    parser.add_argument("--agent",    default=None, choices=["random", "q_learning"],
                        help="Agent type override")
    args = parser.parse_args()

    train(
        config_path = args.config,
        episodes    = args.episodes,
        agent_type  = args.agent,
    )
