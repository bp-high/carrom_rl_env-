---
title: Carrom RL Environment
emoji: 🎯
colorFrom: yellow
colorTo: red
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# OpenEnv Carrom

An [OpenEnv](https://github.com/meta-pytorch/OpenEnv) **physics-based RL environment** for training and evaluating AI agents on the board game Carrom. Features real Newtonian physics with Coulomb board friction, full ICF rule compliance, LLM-friendly text actions, and a Green Agent efficiency wrapper.

## Features

- **Coulomb board friction** — per-body `velocity_func` applies constant deceleration (not viscous drag), matching pieces on a boric-acid-powdered carrom surface
- **ICF-compliant rules** — due rule, queen cover, foul handling, color-based turn continuation
- **LLM-friendly** — text actions (`"aim at queen_0 with strong force"`) and rich board-state observations with rule reminders
- **Multi-agent** — single-agent API with automatic scripted opponent turns
- **Green Agent (evaluator)** — task suite + ICF-aware scoring for purple-agent benchmarking, à la [AgentBeats](https://rdi.berkeley.edu/agentx-agentbeats)
- **Deterministic** — seeded resets for reproducible experiments
- **OpenEnv standard** — `reset()` / `step()` / `state()` API with WebSocket support

## Installation

```bash
pip install -e .
```

Optional rendering:

```bash
pip install -e ".[render]"
```

## Quick Start

### As a client (connecting to a running Space)

```python
import asyncio
from client import CarromEnv
from carrom_env.models import Action

async def main():
    async with CarromEnv(base_url="https://your-space.hf.space") as env:
        result = await env.reset()
        print(result.observation.text_summary)

        result = await env.step(Action(placement_x=0.0, angle=0.1, force=0.6))
        print(f"Reward: {result.reward}, Done: {result.done}")

asyncio.run(main())
```

Synchronous usage:

```python
from client import CarromEnv
from carrom_env.models import Action

with CarromEnv(base_url="http://localhost:8000").sync() as env:
    result = env.reset()
    result = env.step(Action(placement_x=0.0, angle=0.0, force=0.6))
```

### Local development

```python
from carrom_env.env import CarromEnv
from carrom_env.models import Action

env = CarromEnv(seed=42)
obs = env.reset()

action = Action(placement_x=0.0, angle=0.0, force=0.6)
obs, reward, terminated, truncated, info = env.step(action)
```

### Text actions (for LLM agents)

```python
action = Action(action_type="text", text="aim at queen_0 with strong force from center")
obs, reward, terminated, truncated, info = env.step(action)
```

## Game Rules (ICF-Compliant)

This environment implements the key rules from the **International Carrom Federation (ICF)**.

### Board & Pieces

- **9 black coins**, **9 white coins**, **1 queen** (red) — 19 pieces total
- **Agent plays white**; opponent plays black
- Four corner pockets

### Shooting

- On each turn the player places their striker anywhere on their baseline and shoots
- Striker placement is automatically nudged away from any coin sitting on the baseline

### Scoring & Turn Continuation

- Pocket **your own colour** → +1 point, take another turn
- Pocket the **queen** → +3 points; you must then "cover" it (see below)
- Miss (no own coin pocketed) → turn passes to opponent

### Due Rule

- If you pocket your **opponent's colour**, that coin is returned to the board centre
- You score **nothing** for it and your turn **ends** — even if you also pocketed own coins on the same shot, turn continuation only applies to own-colour pockets

### Queen Cover Rule

- After pocketing the queen you must pocket **one of your own coins** on the same shot or on your next turn to "cover" it
- If you fail to cover, the queen is returned to the board centre and your queen points are reversed

### Foul

- Pocketing the **striker** is a foul
- One of your previously pocketed coins is returned to the board centre
- Your turn ends and passes to the opponent

### Win Condition

All coins cleared from the board → game ends; the player with the higher score wins.

### ICF Compliance Table

| Rule | Status | Notes |
|------|--------|-------|
| 9 black + 9 white + 1 queen | ✅ | Full piece complement |
| Agent = white, Opponent = black | ✅ | Enforced throughout |
| Score 1 pt per own coin | ✅ | |
| Queen = 3 pts | ✅ | Simplified from ICF face-value (1–9) |
| Due rule — opponent's coin returns to centre, no score, turn ends | ✅ | |
| Queen cover rule — cover on same/next shot or queen returns | ✅ | |
| Foul — striker pocketed returns own coin, ends turn | ✅ | |
| Turn continuation on own-colour pocket only | ✅ | Due coins do not extend turn |
| Baseline shooting with obstruction check | ✅ | Striker nudged clear of coins |
| Coulomb board friction (boric-acid surface, μ_k ≈ 0.04) | ✅ | `BOARD_DECEL = 2.5 units/s²` via `velocity_func` |
| Elastic rubber cushion walls | ✅ | `ELASTICITY = 0.92` |
| Pocket capture (no corner dead zones) | ✅ | `pocket_capture_radius = 0.09` decoupled from wall gap |
| Numbered coin scoring (ICF 1–9 per colour) | ❌ | Simplified to 1 pt per coin |
| Touch-coin / out-of-turn penalties | ❌ | Not applicable for AI agents |

## Physics Design

### Coulomb Board Friction

Real carrom boards are dusted with boric acid powder giving a kinetic friction coefficient of roughly μ_k ≈ 0.02–0.05. Unlike viscous drag (speed-proportional), sliding friction produces **constant deceleration** regardless of a piece's current speed.

This environment implements Coulomb friction via Pymunk's `body.velocity_func` callback on every piece and the striker:

```
a_friction = BOARD_DECEL   # 2.5 units/s² — equivalent to μ_k ≈ 0.04 on a normalised board
```

With `BOARD_DECEL = 2.5`:
- **Full-force shot** (v₀ ≈ 5 units/s): pieces settle in ~2 seconds after bouncing
- **Medium shot** (v₀ ≈ 2.5 units/s): pieces settle in ~1 second
- The simulation ends early once all pieces drop below `SETTLE_VELOCITY = 0.02 units/s`

### Contact Physics

Shape-to-shape contact friction (`FRICTION = 0.15`) handles the interaction between colliding pieces and between pieces and the rubber-cushioned walls. Collision restitution is `ELASTICITY = 0.92`, reflecting the near-elastic bounce of polished wooden pieces off a rubber cushion.

### Pocket Detection Geometry

Pocket capture uses **two separate radii** to handle a subtle geometry problem:

| Field | Value | Purpose |
|-------|-------|---------|
| `pocket_radius` | `0.06` | Visual pocket size; also the wall gap at each corner |
| `pocket_capture_radius` | `0.09` | Radius within which a piece is considered pocketed |

**Why they differ:** walls have a `wall_thickness = 0.02` and end at `pocket_radius` from each corner. Pymunk segments have rounded endcaps, so a piece (radius `0.03`) rolling along a wall is constrained to stay at distance `≥ 0.05` from the wall endcap. A piece can therefore come to rest at e.g. `(-0.44, -0.45)` — inside the pocket gap but at distance `≈ 0.078` from the corner, which was **outside the old `0.06` detection radius** (a "dead zone"). `pocket_capture_radius = pocket_radius + coin_radius = 0.09` fires as soon as the coin's edge reaches the pocket rim, eliminating the dead zone.

## Action Space

| Field | Type | Description |
|-------|------|-------------|
| `action_type` | `str` | `"numeric"` (default) or `"text"` for natural-language actions |
| `placement_x` | `float` | Striker placement along baseline `[-0.4, 0.4]`, 0 = center |
| `angle` | `float` | Shot angle in radians, 0 = straight ahead toward +y |
| `force` | `float` | Normalized shot force in `[0, 1]` |
| `text` | `str` | Natural-language shot description (when `action_type="text"`) |

## Observation

| Field | Type | Description |
|-------|------|-------------|
| `positions` | `List[List[float]]` | `[N, 2]` positions for striker + coins |
| `velocities` | `List[List[float]]` | `[N, 2]` velocities |
| `pocketed` | `List[bool]` | `[N]` pocketed flags |
| `agent_score` | `int` | Agent's current score |
| `opponent_score` | `int` | Opponent's current score |
| `current_player` | `str` | `"agent"` or `"opponent"` |
| `remaining_coins` | `int` | Coins still on the board |
| `coins` | `List[CoinInfo]` | Per-coin details with nearest pocket info |
| `text_summary` | `str` | Rich text board state for LLM prompting (includes rule reminders) |

## Reward Design

| Event | Reward | Description |
|-------|--------|-------------|
| Each agent turn | −0.01 | Small negative to encourage efficiency |
| Own coin potted | +1.0 | Per own-colour coin pocketed |
| Queen potted | +3.0 | Queen is worth 3× |
| Due coin (opponent's colour potted) | −0.3 | Coin returned to centre; teaches avoidance |
| Foul (striker pocketed) | −1.5 | Score −1 plus −0.5 extra penalty |
| Win (cleared board, agent leads) | +5.0 | Bonus for winning |
| Loss (cleared board, opponent leads) | −2.0 | Penalty for losing |
| Opponent scores | −0.5× | Partial penalty when opponent pots own coins |

## `info` Dict Keys

| Key | Type | Description |
|-----|------|-------------|
| `sim_steps` | `float` | Physics steps taken this turn |
| `energy` | `float` | Cumulative kinetic energy this turn |
| `coin_potted` | `float` | Own coins pocketed this turn |
| `due_coins` | `float` | Opponent's coins returned to centre (due rule) |
| `foul` | `float` | 1.0 if striker was pocketed |
| `queen_cover_pending` | `bool` | True if queen cover is still required |
| `placement_x_actual` | `float` | Actual striker x after obstruction nudge |

## Green Agent (Evaluator)

In the [AgentBeats / AgentX](https://rdi.berkeley.edu/agentx-agentbeats) taxonomy:

- 🟢 **Green Agent** — evaluator: defines tasks, environment, and scoring
- 🟣 **Purple Agent** — competitor: the AI being tested (any `Callable[[Observation], Action]`)
- 🔴 **Red Agent** — adversarial tester (not used here)

`GreenCarromAgent` is the green agent for this benchmark. It owns:

1. **A task suite** — curated seeded boards across `easy` / `standard` / `hard` tiers
2. **The environment** — wraps `CarromEnv` with full ICF rules
3. **Scoring** — ICF-aware metrics (reward, win rate, ICF compliance from dues/fouls) plus compute efficiency

```python
from carrom_env.green_agent import GreenCarromAgent, Task

def my_purple_agent(obs):
    return Action(placement_x=0.0, angle=0.1, force=0.6)

# Default suite: 3 easy + 3 standard + 3 hard tasks
evaluator = GreenCarromAgent()
report = evaluator.evaluate(my_purple_agent, verbose=True)
print(report.summary())
# {'n_tasks': 9, 'avg_reward': ..., 'win_rate': ..., 'icf_compliance': ...,
#  'efficiency_score': ..., ...}

# Or define a custom suite
tasks = [Task(task_id="focus", seed=0, max_turns=30, tier="standard")]
report = GreenCarromAgent(tasks=tasks).evaluate(my_purple_agent)
report.by_tier()  # per-tier breakdown
```

### Scoring metrics

| Metric | Type | Description |
|--------|------|-------------|
| `avg_reward` | game | Mean episode reward across the suite |
| `win_rate` | game | Fraction of tasks where agent beat the opponent |
| `avg_coins_potted` | game | Mean own-coins pocketed per task |
| `avg_dues` | ICF | Mean opponent-coin pockets per task (lower = better) |
| `avg_fouls` | ICF | Mean strikers pocketed per task (lower = better) |
| `icf_compliance` | ICF | `1 − (dues + fouls) / turns` — fraction of shots obeying ICF rules |
| `total_sim_steps` | compute | Total physics steps across all tasks |
| `efficiency_score` | compute | Coins potted per 1000 sim steps |

### What `max_turns` counts

`Task.max_turns` (and `MAX_STEPS` in the inference script) counts **combined agent + opponent turns** — the env's internal turn counter increments once for every played shot, on either side. A setting of `max_turns=200` therefore caps the episode at ~100 agent shots + ~100 heuristic-opponent shots. Set it to `400` if you want roughly 200 agent shots, or pass a custom `Task` with whatever cap you need.

## Benchmark Results

Full inference logs live in [`inference_runs/`](inference_runs/).

### MiniMaxAI/MiniMax-M2.5-fast (Nebius)

1 task × 200 turns (seed `0`), via `https://api.tokenfactory.us-central1.nebius.com/v1`.
Full log: [`inference_runs/minimax-m2.5-fast_200turns_inference.log`](inference_runs/minimax-m2.5-fast_200turns_inference.log)

| Purple Agent | Reward | Win% | Coins | Dues | ICF% | Efficiency |
|---|---:|---:|---:|---:|---:|---:|
| Random | −4.21 | 0% | 4.0 | 2.00 | 98% | 0.368 |
| Heuristic (ICF-aware) | −6.31 | 0% | 4.0 | 0.00 | 100% | 0.337 |
| **LLM · MiniMax-M2.5-fast** | **+4.07** | **100%** | **8.0** | 1.00 | 97% | **0.759** |

MiniMax-M2.5-fast wins the board at 8 coins potted with 1 due and 5 fouls, beating both baselines on reward and efficiency. The heuristic tanks to −6.31 because it's aggressive about shooting at white coins but hands the opponent easy board position on misses.

## Setup & Usage

### Prerequisites

- Python 3.10+
- Docker (for containerized deployment)
- `pip install openenv-core pymunk`

### Local Development

```bash
pip install -e ".[dev]"
PYTHONPATH=. uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
```

### Docker

```bash
docker build -t carrom-env:latest .
docker run -p 8000:8000 carrom-env:latest
```

### Baseline Inference

```bash
export API_BASE_URL="https://api-inference.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen3-4B"
export HF_TOKEN="hf_..."
python inference.py
```

## Project Structure

```
carrom_rl_env/
├── __init__.py              # Module exports
├── carrom_env/
│   ├── __init__.py          # Package exports
│   ├── env.py               # CarromEnv (physics + ICF game logic)
│   ├── models.py            # Action, Observation, State models
│   ├── constants.py         # Board config + physics constants (BOARD_DECEL, FRICTION, …)
│   └── green_agent.py       # Green Agent efficiency wrapper
├── client.py                # CarromEnv (EnvClient)
├── inference.py             # Baseline inference script
├── server/
│   ├── __init__.py
│   ├── carrom_environment.py # Server-side Environment wrapper
│   └── app.py               # FastAPI application
├── examples/
│   ├── train_stub.py        # Quick demo
│   ├── grpo_utils.py        # GRPO training utilities
│   └── grpo_carrom_tutorial.ipynb  # Training notebook
├── tests/
│   └── test_env_basic.py    # Test suite
├── openenv.yaml             # OpenEnv manifest
├── pyproject.toml           # Dependencies
├── Dockerfile               # Container image
└── README.md                # This file
```
