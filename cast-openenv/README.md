# 🤟 CAST-OpenEnv — Adaptive Sign Language Interpretation using RL

> **Hackathon Project** | CAST × RL × OpenEnv  
> An AI agent that learns **how to behave** in uncertain sign language communication environments.

---

## 🎯 Problem Statement

Real-world sign language communication is noisy. Gestures can be ambiguous, lighting may be poor, and in emergencies, wrong interpretations can be life-threatening.

This project builds an **RL-based agent** that learns:
- When to **interpret** a gesture directly
- When to **ask for clarification**
- When to **trigger an emergency alert**

---

## 🧠 Approach

Instead of a static classifier, we frame the problem as a **Markov Decision Process (MDP)**:

| Component | Description |
|-----------|-------------|
| **State** | `gesture × noise_level × context` |
| **Action** | `show_text`, `ask_repeat`, `trigger_alert` |
| **Reward** | Shaped for correctness + emergency handling + speed |
| **Agent** | Q-Learning (upgradeable to DQN) |

The agent learns an optimal **policy** — a mapping from situation → best action — through repeated interaction with the environment.

---

## 🏗️ Architecture

```
cast-openenv/
│
├── env/
│   └── cast_env.py        ← OpenEnv-compatible environment
│
├── agent/
│   └── rl_agent.py        ← RandomAgent, QLearningAgent, DQNAgent (optional)
│
├── config/
│   └── openenv.yaml       ← All hyperparameters & settings
│
├── models/                ← Saved Q-tables (created at runtime)
│
├── train.py               ← Training pipeline
├── demo.py                ← Interactive demo
└── README.md
```

---

## 🌍 Environment Design

### State Space (27 unique states)

| Variable | Values |
|----------|--------|
| `gesture` | `hello`, `stop`, `help`, `danger` |
| `noise` | `low`, `medium`, `high` |
| `context` | `classroom`, `road`, `home` |

### Action Space

| ID | Action | When Used |
|----|--------|-----------|
| 0 | `show_text` | Clear gesture, low/medium noise |
| 1 | `ask_repeat` | Non-emergency + high noise |
| 2 | `trigger_alert` | Emergency gesture (`help`/`danger`) |

### Reward Function

| Event | Reward |
|-------|--------|
| Correct interpretation | `+1.0` |
| Wrong interpretation | `-1.0` |
| Emergency correctly handled | `+2.0` (bonus on top of +1) |
| Fast correct response | `+0.5` |

---

## ⚙️ Reinforcement Learning

### Q-Learning Update Rule

```
Q(s, a) ← Q(s, a) + α × [r + γ × max Q(s', a') − Q(s, a)]
```

| Parameter | Value | Meaning |
|-----------|-------|---------|
| α (alpha) | 0.1 | Learning rate |
| γ (gamma) | 0.95 | Discount factor (future rewards matter) |
| ε (epsilon) | 1.0 → 0.05 | Exploration decays over training |

### Training Flow

```
for each episode:
    state = env.reset()
    while not done:
        action = agent.select_action(state)   # ε-greedy
        next_state, reward, done = env.step(action)
        agent.update(state, action, reward, next_state, done)
        state = next_state
```

---

## 🚀 How to Run

### Prerequisites

```bash
pip install pyyaml numpy
```

### 1. Train the Agent

```bash
cd cast-openenv
python train.py
```

Optional flags:
```bash
python train.py --episodes 2000        # more training
python train.py --agent random         # baseline comparison
```

### 2. Run the Demo

```bash
python demo.py
```

Optional flags:
```bash
python demo.py --random                # compare with random agent
python demo.py --interactive           # type your own scenarios
```

---

## 🎭 Demo Scenarios

### Scenario 1 — Emergency + High Noise
```
Gesture  : 🆘 HELP
Noise    : ▓▓▓ HIGH
Context  : 📍 Road

Agent Decision → 🚨 TRIGGER ALERT (+3.5 reward)
Reasoning: Emergency gesture detected → alert triggered regardless of noise.
```

### Scenario 2 — Clear Communication
```
Gesture  : 👋 HELLO
Noise    : ▓░░ LOW
Context  : 📍 Classroom

Agent Decision → 📢 SHOW TEXT (+1.5 reward)
Reasoning: Clear signal in low noise → agent confident to show text.
```

### Scenario 3 — Ambiguous Input
```
Gesture  : ✋ STOP
Noise    : ▓▓▓ HIGH
Context  : 📍 Classroom

Agent Decision → 🔄 ASK REPEAT (+1.5 reward)
Reasoning: High noise makes gesture ambiguous → agent requests clarification.
```

---

## 📈 What the Agent Learns

Over 1000 episodes, the agent learns:

1. **Emergency always → Alert**: `help`/`danger` always triggers `trigger_alert`, regardless of noise or context.
2. **Noise matters**: For non-emergency gestures, high noise → `ask_repeat` instead of guessing.
3. **Clear = confident**: Low/medium noise + non-emergency → show text directly.

---

## 📊 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Average Reward** | Tracks learning improvement per episode |
| **Policy Accuracy** | % of states where agent picks optimal action |
| **Emergency Detection** | Correct `trigger_alert` on `help`/`danger` |
| **Convergence Speed** | Episodes needed to reach stable policy |

---

## 🔧 Configuration

All settings in `config/openenv.yaml`:

```yaml
agent:
  type: q_learning
  alpha: 0.1
  gamma: 0.95
  epsilon: 1.0
  epsilon_decay: 0.995

training:
  episodes: 1000
  save_path: models/q_table.json
```

---

## 🔮 Upgrade Path

| Level | Implementation |
|-------|----------------|
| ✅ Baseline | `RandomAgent` — no learning |
| ✅ Core | `QLearningAgent` — tabular Q-Learning |
| 🔲 Advanced | `DQNAgent` — Deep Q-Network (code provided, uncomment in `rl_agent.py`) |
| 🔲 Expert | Multi-agent, real gesture dataset integration |

---

## 💡 Key Insight

> "A good AI doesn't just recognize signs — it knows **when it's uncertain** and acts accordingly."

This system demonstrates that RL can learn **meta-behavior**: not just what a gesture means, but how confidently to act on that interpretation given real-world uncertainty.

---

## 👥 Team

CAST Hackathon | *Adaptive Sign Language Interpretation using Reinforcement Learning*
