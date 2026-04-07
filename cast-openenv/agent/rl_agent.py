"""
CAST RL Agent
-------------
Implements Q-Learning with epsilon-greedy exploration.

Upgrade path:
  RandomAgent  →  QLearningAgent  →  DQNAgent (optional, see bottom)
"""

import numpy as np
import json
import os
import random
from abc import ABC, abstractmethod


# ─────────────────────────────────────────────
#  Base Agent Interface
# ─────────────────────────────────────────────

class BaseAgent(ABC):
    def __init__(self, n_states: int, n_actions: int):
        self.n_states  = n_states
        self.n_actions = n_actions

    @abstractmethod
    def select_action(self, state_idx: int) -> int:
        """Select action given encoded state index."""

    @abstractmethod
    def update(self, state_idx: int, action: int, reward: float,
               next_state_idx: int, done: bool):
        """Update internal model after receiving reward."""

    def save(self, path: str):
        pass

    def load(self, path: str):
        pass


# ─────────────────────────────────────────────
#  1. Random Agent  (baseline)
# ─────────────────────────────────────────────

class RandomAgent(BaseAgent):
    """Selects a uniformly random action. Used as baseline."""

    def select_action(self, state_idx: int) -> int:
        return random.randint(0, self.n_actions - 1)

    def update(self, *args, **kwargs):
        pass   # no learning


# ─────────────────────────────────────────────
#  2. Q-Learning Agent  (main)
# ─────────────────────────────────────────────

class QLearningAgent(BaseAgent):
    """
    Tabular Q-Learning with ε-greedy exploration.

    Q(s,a) ← Q(s,a) + α [ r + γ·max Q(s',a') − Q(s,a) ]
    """

    def __init__(
        self,
        n_states:    int,
        n_actions:   int,
        alpha:       float = 0.1,    # learning rate
        gamma:       float = 0.95,   # discount factor
        epsilon:     float = 1.0,    # exploration rate (starts high)
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        seed:        int   = 42,
    ):
        super().__init__(n_states, n_actions)
        self.alpha         = alpha
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay

        np.random.seed(seed)
        random.seed(seed)

        # Initialize Q-table with small random values to break ties
        self.q_table = np.random.uniform(
            low=-0.01, high=0.01, size=(n_states, n_actions)
        )

    # ── Action Selection ──────────────────────

    def select_action(self, state_idx: int) -> int:
        """ε-greedy: explore randomly or exploit Q-table."""
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)   # explore
        return int(np.argmax(self.q_table[state_idx]))     # exploit

    def greedy_action(self, state_idx: int) -> int:
        """Pure greedy (no exploration). Used at demo/eval time."""
        return int(np.argmax(self.q_table[state_idx]))

    # ── Learning Update ───────────────────────

    def update(
        self,
        state_idx:      int,
        action:         int,
        reward:         float,
        next_state_idx: int,
        done:           bool,
    ):
        """Bellman update for Q(state, action)."""
        target = reward
        if not done:
            target += self.gamma * np.max(self.q_table[next_state_idx])

        td_error = target - self.q_table[state_idx, action]
        self.q_table[state_idx, action] += self.alpha * td_error

        # Decay exploration rate
        self._decay_epsilon()

    def _decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon  = max(self.epsilon, self.epsilon_min)

    # ── Persistence ───────────────────────────

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data = {
            "q_table": self.q_table.tolist(),
            "epsilon": self.epsilon,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print("[Agent] Q-table saved -> " + path)

    def load(self, path: str):
        with open(path, "r") as f:
            data = json.load(f)
        self.q_table = np.array(data["q_table"])
        self.epsilon = data.get("epsilon", self.epsilon_min)
        print("[Agent] Q-table loaded <- " + path)

    # ── Diagnostics ───────────────────────────

    def best_action_summary(self, env) -> dict:
        """Return a dict mapping every state → best action name."""
        summary = {}
        for idx in range(self.n_states):
            state = env.decode_state(idx)
            best  = self.greedy_action(idx)
            key   = f"{state['gesture']}|{state['noise']}|{state['context']}"
            summary[key] = env.action_name(best)
        return summary


# ─────────────────────────────────────────────
#  3. Deep Q-Network Agent  (optional upgrade)
# ─────────────────────────────────────────────
#
#  Uncomment and install torch to enable DQN.
#  This is provided as an upgrade path from Q-Learning.
#
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from collections import deque
#
# class DQNetwork(nn.Module):
#     def __init__(self, n_states, n_actions):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(n_states, 64),
#             nn.ReLU(),
#             nn.Linear(64, 64),
#             nn.ReLU(),
#             nn.Linear(64, n_actions),
#         )
#     def forward(self, x):
#         return self.net(x)
#
# class DQNAgent(BaseAgent):
#     """Deep Q-Network with experience replay."""
#     def __init__(self, n_states, n_actions, lr=1e-3, gamma=0.95,
#                  epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995,
#                  buffer_size=10000, batch_size=64):
#         super().__init__(n_states, n_actions)
#         self.gamma         = gamma
#         self.epsilon       = epsilon
#         self.epsilon_min   = epsilon_min
#         self.epsilon_decay = epsilon_decay
#         self.batch_size    = batch_size
#
#         self.policy_net = DQNetwork(n_states, n_actions)
#         self.target_net = DQNetwork(n_states, n_actions)
#         self.target_net.load_state_dict(self.policy_net.state_dict())
#
#         self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
#         self.memory    = deque(maxlen=buffer_size)
#
#     def _one_hot(self, idx):
#         v = torch.zeros(self.n_states)
#         v[idx] = 1.0
#         return v
#
#     def select_action(self, state_idx):
#         if random.random() < self.epsilon:
#             return random.randint(0, self.n_actions - 1)
#         with torch.no_grad():
#             q = self.policy_net(self._one_hot(state_idx).unsqueeze(0))
#         return int(q.argmax().item())
#
#     def update(self, state_idx, action, reward, next_state_idx, done):
#         self.memory.append((state_idx, action, reward, next_state_idx, done))
#         if len(self.memory) < self.batch_size:
#             return
#         batch = random.sample(self.memory, self.batch_size)
#         states, actions, rewards, nexts, dones = zip(*batch)
#
#         s  = torch.stack([self._one_hot(i) for i in states])
#         ns = torch.stack([self._one_hot(i) for i in nexts])
#         a  = torch.tensor(actions)
#         r  = torch.tensor(rewards, dtype=torch.float32)
#         d  = torch.tensor(dones,   dtype=torch.float32)
#
#         q_vals  = self.policy_net(s).gather(1, a.unsqueeze(1)).squeeze()
#         with torch.no_grad():
#             next_q = self.target_net(ns).max(1).values
#         targets = r + self.gamma * next_q * (1 - d)
#
#         loss = nn.MSELoss()(q_vals, targets)
#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step()
#
#         if self.epsilon > self.epsilon_min:
#             self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
