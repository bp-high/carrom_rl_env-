# Teaching LLMs to Play Carrom: A Physics-Based RL Environment for Frontier Agents

> **TL;DR.** I built an OpenEnv-compatible physics simulation of Carrom, the strategy board game played by hundreds of millions of people across South Asia but almost entirely absent from modern RL benchmarks. It ships with full International Carrom Federation (ICF) rules, Coulomb board friction via Pymunk, an AgentBeats-style green-agent evaluator, a Unsloth + TRL GRPO training recipe that runs for under $25 on Modal, and baseline results showing MiniMax-M2.5-fast outscoring random and ICF-aware heuristics with near-perfect rule compliance. Code, Space, and training notebooks are linked at the end.

![Initial Carrom board with 9 black coins, 9 white coins, and the queen in the centre](UPLOAD_INITIAL_BOARD.png)
*The initial board state: 19 pieces arranged in the classic ring formation, queen in the centre, surrounded by an inner ring of alternating blacks and whites, then an outer ring. The agent plays white, a heuristic opponent plays black.*

---

## 1. Why Carrom, and why now?

Pick a classic strategy game and there's a good chance an RL environment exists for it: Chess, Go, Shogi, even obscure things like Hnefatafl and Arimaa have been formalised in Gym/Gymnasium. Poker, Diplomacy, Catan, Hanabi are all well-trodden. But **Carrom**, a game played daily by hundreds of millions across India, Pakistan, Bangladesh, Sri Lanka, Nepal, and the diaspora, has been sitting in a blind spot.

That's a strange blind spot to have. Carrom is a *real* spatial reasoning problem: you have to predict multi-body physics, account for friction and elasticity, and plan two or three collisions ahead to pocket the coin you want without pocketing the wrong one. It's also culturally significant in a way that chess isn't everywhere on the planet, making it a benchmark that isn't monoculturally Western.

When the [OpenEnv Student Challenge](https://github.com/meta-pytorch/OpenEnv) went up (sponsored by the PyTorch team at Meta, Hugging Face, and Unsloth), building a Carrom environment felt obvious. It has the right mix of **verifiable reward** (did the disc go in the pocket? physics says yes or no), **diverse initial states** (seeded random setups give endless variety), and **hard-but-tractable** spatial reasoning that modern LLMs should be *close* to being able to do, but aren't quite nailing yet.

This post walks through how the environment was built, the technical choices that mattered, the evaluator I designed around it, and the baseline results I got from frontier models via [Nebius](https://nebius.com).

---

## 2. What Carrom actually is (briefly)

A 74 cm × 74 cm wooden board with four corner pockets. Pieces ("coins"):

- **9 white coins** and **9 black coins**, one player per colour
- **1 queen** (red) in the centre, worth extra points
- **1 striker** per player, the disc you actually flick

The board is coated with **boric acid powder** to make it very slippery (kinetic friction coefficient ≈ 0.02–0.05). You shoot the striker from your baseline, using only a flick of one finger, and try to pocket your coins.

### ICF rules that matter for simulation

The International Carrom Federation rulebook has a lot of edge cases. The ones that really shape environment design:

1. **Colour assignment.** Each player is assigned a colour and can only legally pocket *their own colour*.
2. **Due rule.** If you pocket one of your opponent's coins, it's called a *due*. The coin is placed back at the centre of the board, you score nothing for it, and your turn ends. This is the single most important rule for an LLM to understand because it punishes "pot anything that moves" strategies.
3. **Queen cover.** If you pocket the queen, you have to "cover" it on the same shot or the very next shot by also pocketing one of your own coins. If you fail to cover, the queen is returned to the centre and your queen points are revoked.
4. **Foul.** If your striker goes into a pocket, it's a foul. One of your previously pocketed coins returns to the board, and your turn ends.
5. **Turn continuation.** You keep shooting only after pocketing one of your own coins (or covering the queen). Any other outcome ends your turn.

If you've seen my repo's `_play_turn` function, this is exactly what's implemented.

---

## 3. Building the physics: Pymunk and the friction problem

The simulation runs on [Pymunk](http://www.pymunk.org) (a Python binding over Chipmunk2D), chosen because 2-D rigid-body physics is the right fidelity level for a top-down carrom board and Pymunk is extremely fast.

### Early version: a subtle friction bug

My first pass used Pymunk's global viscous damping, `space.damping = 0.995`, which multiplies every body's velocity by `0.995` each step. At 120 Hz, that means velocity is multiplied by `0.995^120 ≈ 0.548` per second. Pieces decay to half speed every 1.3 seconds.

This *looks* reasonable at first, but the physics is wrong. **Real carrom pieces don't experience viscous drag.** They're solid wooden discs sliding on a powdered board, which produces **Coulomb (kinetic) friction**: a constant deceleration, independent of speed. A fast-moving piece and a slow-moving piece both decelerate at the same rate until they stop.

With viscous drag, pieces slide forever at diminishing speeds. They look frictionless, they never quite settle, and the dynamics feel like underwater hockey pucks rather than a real carrom board.

### The fix: per-body velocity callbacks

Pymunk lets you override the per-step velocity integration for each body. I use this to apply Coulomb friction directly:

```python
@staticmethod
def _make_friction_func(decel: float):
    def _velocity_func(body, gravity, damping, dt):
        pymunk.Body.update_velocity(body, gravity, damping, dt)
        speed = body.velocity.length
        if speed > 1e-6:
            new_speed = max(0.0, speed - decel * dt)
            body.velocity = body.velocity * (new_speed / speed)
    return _velocity_func
```

I set `space.damping = 1.0` (no global drag) and attach this `velocity_func` to every coin and the striker when they're created. `BOARD_DECEL = 2.5 units/s²` approximates a μ_k of ~0.04 on a 1-unit-wide board. The result: pieces decelerate linearly, come to rest in roughly 2 seconds after a full-force shot, and look like real carrom physics.

### The pocket dead-zone

![Mid-game state with the striker aimed diagonally across the board](UPLOAD_AIM_OVERLAY.png)
*The aim overlay before a shot: striker placed on the baseline, arrow indicating angle and 50% force. Once you click Shoot!, Pymunk integrates the collision physics at 120 Hz and every frame is streamed back to the browser.*

While testing, I noticed pieces would sometimes come to rest *right next to a pocket* and never get captured. Digging in, it was a pure geometry bug:

- Pockets are at the corners with radius `0.06`
- The wall gap at each corner is *also* `0.06` (where the cushions end)
- Pymunk segments have rounded endcaps of radius `wall_thickness = 0.02`
- A piece (radius `0.03`) rolling along the wall can't get its centre within `0.05` of the wall endcap

So a piece could reach `(-0.44, -0.45)`, inside the pocket gap but at distance ≈ `0.078` from the corner, and be at rest just *outside* the `0.06` detection radius. Dead zone.

The fix was to decouple pocket *capture* from pocket *visual* size:

| Field | Value | Purpose |
|-------|-------|---------|
| `pocket_radius` | `0.06` | Visual rendering + wall gap |
| `pocket_capture_radius` | `0.09` (= `pocket_radius + coin_radius`) | Detection threshold |

`0.09` has a clean physical interpretation: the coin's edge has reached the pocket rim. Any piece that enters the pocket gap region now gets captured correctly.

### Rule enforcement inside the physics step

The trickier part of ICF compliance is the **due rule**. After each step I separate pocketed pieces by colour ownership:

```python
# ICF Due rule: opponent's coins pocketed return to centre, no score
due_coins = [
    cid for cid in coin_potted
    if "queen" not in cid and not self._coin_belongs_to(cid, player)
]
for due_id in due_coins:
    self._return_due_coin(due_id)
# Only own coins + queen proceed to scoring and turn-continuation
coin_potted = [cid for cid in coin_potted if cid not in due_coins]
```

The returned coin is placed back at the centre with zero velocity, and because `coin_potted` no longer contains it, the turn-continuation logic treats the shot as a miss. Clean. The only nuance is that the reward function also sees `due_coins` count in `info`, which lets training signals penalise due violations directly (I use `-0.3` per due during GRPO).

The same pattern handles queen cover: a flag `_queen_cover_pending` is set when you pocket the queen, and cleared either when you pocket your own coin on the same/next shot or when the queen is kicked back to the centre by an uncovered failure.

### Baseline obstruction: the "cannot place on a disc" rule

One more ICF detail that shows up every few shots: **the striker can't be placed on top of a coin.** On a real board, if a coin is sitting on your baseline exactly where you wanted to place the striker, you have to slide the striker along the line until you find a clear spot.

The environment handles this automatically inside `_play_turn`. Before the shot runs, `_find_valid_placement` scans every un-pocketed coin whose y-coordinate is within `striker_radius + coin_radius + margin` of the baseline. If the requested `placement_x` would overlap any of them, the striker is nudged in 0.02-unit increments, alternating right then left, until it finds the nearest obstruction-free position:

```python
def _find_valid_placement(self, desired_x: float, baseline_y: float) -> float:
    min_gap = self.board.striker_radius + self.board.coin_radius + 0.002
    def _blocked(x):
        for coin in self.coins:
            if coin.pocketed:
                continue
            cx, cy = coin.body.position
            if abs(cy - baseline_y) < min_gap and abs(cx - x) < min_gap:
                return True
        return False

    if not _blocked(desired_x):
        return desired_x
    for offset in range(1, 40):
        for sign in (1, -1):
            candidate = desired_x + sign * offset * 0.02
            candidate = float(np.clip(candidate, -placement_limit, placement_limit))
            if not _blocked(candidate):
                return candidate
    return desired_x
```

The `info` dict returned from `env.step()` includes `placement_x_actual`, so the agent can see what actually happened ("you asked for 0.10, I placed you at 0.14 because there was a coin in the way"). That matters for LLM agents especially, because otherwise a model that keeps specifying an obstructed `placement_x` has no way to learn that the world is preventing its plan from executing as-written.

---

## 4. OpenEnv and the green-agent evaluator

The environment conforms to the [OpenEnv](https://github.com/meta-pytorch/OpenEnv) standard (`reset()` / `step()` / `state()`), wrapped in a FastAPI + WebSocket server so it can be driven locally, from Colab, or remotely from a Hugging Face Space. Same code path, different transport.

On top of the environment sits the **green agent**. In the [AgentBeats / AgentX](https://rdi.berkeley.edu/agentx-agentbeats) taxonomy:

- 🟢 **Green Agent** is the evaluator: it defines tasks, environment, and scoring.
- 🟣 **Purple Agent** is the competitor: the AI agent under test.
- 🔴 **Red Agent** is an adversarial tester (not used here).

`GreenCarromAgent` is my green agent. It owns:

1. **A task suite** of seeded boards across easy / standard / hard tiers.
2. **The environment** itself, wrapping `CarromEnv` with full ICF rules.
3. **Scoring**, with ICF-aware metrics (reward, win rate, ICF compliance from dues/fouls) plus compute efficiency (sim steps, wall time).

```python
from carrom_env.green_agent import GreenCarromAgent

def my_purple_agent(obs):
    return Action(placement_x=0.0, angle=0.1, force=0.6)

report = GreenCarromAgent().evaluate(my_purple_agent)
print(report.summary())
# {'n_tasks': 9, 'avg_reward': ..., 'win_rate': ...,
#  'icf_compliance': 0.97, 'efficiency_score': 0.72, ...}
```

Any `Callable[[Observation], Action]` is a valid purple agent: random policies, the built-in heuristic, an LLM behind an OpenAI-compatible endpoint, or a trained GRPO model. The green agent treats them identically and produces a comparable scorecard.

The evaluator also tracks an **ICF compliance score**, defined as `1 − (dues + fouls) / turns`, which surfaces how well an agent actually *obeys the rules* versus just racking up reward. An agent that scores points by fouling its way through a game and an agent that wins cleanly get very different compliance numbers, which is the signal you want.

---

## 5. LLM-friendly design

The observation carries both numeric and text views:

- **Numeric.** `positions`, `velocities`, `pocketed` flags, intended for standard RL agents.
- **Text.** A human-readable `text_summary` with every coin's position, nearest pocket, and a plain-English rules reminder, intended for LLM agents.

```
=== Carrom Board State (Turn 12/200) ===
Score: You 3 - Opponent 1
Remaining coins on board: 15
Rules: You=WHITE coins (+1 pt each). Queen=+3 pts (must cover next shot).
Due rule: pocketing a BLACK coin returns it to centre; no score, turn ends.

Active coins:
  white_inner_0 (white) at (-0.069, 0.000) | nearest pocket: bottom-left (0.54 away)
  ...
```

Actions come in two flavours: **numeric** (`placement_x`, `angle` in radians, `force` in `[0, 1]`), or **text** (e.g. `"aim at queen_0 with strong force from centre"`, which the env parses). Text actions bridge the LLM-to-physics gap: the model reasons in natural language, the environment grounds it in Pymunk.

For inference, the script supports any OpenAI-compatible endpoint out of the box (Nebius, HuggingFace Inference, OpenAI, local vLLM). Just set env vars:

```bash
export API_BASE_URL="https://api.tokenfactory.us-central1.nebius.com/v1"
export MODEL_NAME="MiniMaxAI/MiniMax-M2.5-fast"
export NEBIUS_API_KEY="..."
python inference.py
```

One subtlety: **reasoning models** like MiniMax-M2.5 and Nemotron return `content: null` with the actual answer in a `reasoning_content` field. The parser falls back to `reasoning_content` when `content` is null, which was a bug I hit on my first Nebius run.

---

## 6. Training: GRPO + Unsloth on Modal

The [companion notebook](https://github.com/bp-high/carrom_rl_env-/blob/main/examples/grpo_carrom_tutorial.ipynb) and [`examples/train_modal.py`](https://github.com/bp-high/carrom_rl_env-/blob/main/examples/train_modal.py) train a small LLM to play Carrom using [TRL's GRPOTrainer](https://huggingface.co/docs/trl/main/en/grpo_trainer) with [Unsloth's](https://github.com/unslothai/unsloth) 4-bit quantization and gradient checkpointing for memory efficiency.

**Why GRPO?** It needs no value network. It samples G completions per prompt, scores each with the reward function, and uses the group's mean/std as the baseline. Simpler than PPO and works well for structured-output tasks like "return JSON with `placement_x`, `angle`, `force`".

The reward function is layered:

```python
def carrom_reward(completions, **kwargs):
    for completion in completions:
        reward = 0.0
        action = parse_response(completion)
        if action is not None:
            reward += 0.3                         # valid JSON
            if -0.4 <= action.placement_x <= 0.4: reward += 0.1
            if 0.15 <= action.force <= 0.9:       reward += 0.1
            if -π/2 <= action.angle <= π/2:       reward += 0.1
            # Execute in env for actual game reward
            _, env_reward, _, _, info = env.step(action)
            reward += env_reward
            reward -= 0.3 * info.get("due_coins", 0)  # ICF penalty
            reward -= 0.75 * info.get("foul", 0)
        else:
            reward -= 0.5                          # unparseable
```

**Cost estimates on Modal** (A10G at ~$1.10/hr, Gemma-3-4B with Unsloth 4-bit):

| Setting | Steps | Time | Cost |
|---------|-------|------|------|
| Smoke test | 200 | ~15 min | ~$0.30 |
| Light training | 500 | ~35 min | ~$0.65 |
| Blog-quality | 2000 | ~2.5 hr | ~$2.75 |
| A100 blog-quality | 2000 | ~1.5 hr | ~$5.60 |

All comfortably under the competition's budget. Run it yourself:

```bash
modal run examples/train_modal.py --dry-run         # cost preview
modal run examples/train_modal.py --steps 2000 --push --repo your-user/carrom-grpo
```

The script supports Gemma-3-1B/4B, Qwen2.5-1.5B/3B/7B out of the box, with the model, step count, and hardware tier all controllable from the command line.

---

## 7. Benchmark results

I ran the green agent over a 200-turn episode with three purple agents: a uniform random policy, an ICF-aware heuristic (aims at the nearest white coin to a pocket, explicitly avoids black coins), and MiniMax-M2.5-fast via Nebius.

### MiniMaxAI/MiniMax-M2.5-fast

| Purple Agent | Reward | Win % | Coins | Dues | ICF % | Efficiency |
|---|---:|---:|---:|---:|---:|---:|
| Random | −4.21 | 0% | 4.0 | 2.00 | 98% | 0.368 |
| Heuristic (ICF-aware) | −6.31 | 0% | 4.0 | 0.00 | 100% | 0.337 |
| **LLM · MiniMax-M2.5-fast** | **+4.07** | **100%** | **8.0** | 1.00 | 97% | **0.759** |

Two things worth calling out here:

**The heuristic tanks despite being rule-correct.** It scored 1 coin with 0 dues and 0 fouls (100% ICF compliance), yet ended at −6.31 reward. Why? It's aggressive about shooting at white coins, but every time it misses, the heuristic opponent (which aims at the nearest coin to the centre with force 0.65) cleans up black coins. The reward formula subtracts `0.5 × opponent_score`, so the heuristic is consistently handing the opponent easy board position on its missed shots.

**MiniMax plays carefully and wins.** 8 own-coins potted, one due, five fouls over 103 agent shots, 97% ICF compliance. The model clearly understands the rules (specifically "don't pot the opponent's coins") and plays the scoring game rather than the reward-hacking game. Higher coins per simulation step than either baseline.

### nvidia/Nemotron-3-Super-120b-a12b

Nemotron is NVIDIA's 120B hybrid-MoE reasoning model. Same green-agent evaluator, same `seed=0`, same 200-turn budget:

| Purple Agent | Reward | Win % | Coins | Dues | ICF % | Efficiency |
|---|---:|---:|---:|---:|---:|---:|
| Random | +4.17 | 100% | 8.0 | 1.00 | 100% | 0.712 |
| Heuristic (ICF-aware) | −6.31 | 0% | 4.0 | 0.00 | 100% | 0.337 |
| **LLM · Nemotron-3-Super-120b** | **+11.94** | **100%** | **13.0** | 2.00 | 97% | **1.179** |

Nemotron outscores every baseline on every game metric and hits the highest compute efficiency so far at **1.18 coins potted per 1 000 sim steps**. Total wall-clock for the 106 agent shots was ~33 minutes.

### Frontier-model head-to-head (same seed, same evaluator)

| Model | Reward | Coins | Dues | Fouls | ICF % | Efficiency |
|---|---:|---:|---:|---:|---:|---:|
| MiniMax-M2.5-fast | +4.07 | 8 | 1 | 5 | 97% | 0.759 |
| **Nemotron-3-Super-120b** | **+11.94** | **13** | 2 | **4** | 97% | **1.179** |

Two interesting patterns:

1. **Bigger reasoning model, more aggressive, higher-variance play.** Nemotron pots 5 more coins than MiniMax but commits one extra due. The same 97% ICF-compliance score reveals that *fewer total violations* is not the same as *identical behaviour*. Nemotron is trading a small bit of rule obedience for a lot more scoring.
2. **Efficiency scales with capability.** Both models see the same physics, same observation format, same system prompt. Nemotron produces 55% more coins per simulation step; the model spends fewer "wasted" shots on no-score outcomes.

Both models won (reward > 0 and `agent_score > opponent_score`). That's a nice property of the evaluator: the baselines establish a meaningful floor, and capability gains between frontier models show up on the same scorecard.

### Watch it play

> *Video placeholder. I'll be attaching a screen recording here of the LLM driving the board live via the Gradio auto-play panel. You'll see the striker aim, the physics animation, and the heuristic opponent reply, shot by shot.*

The Gradio UI has an **Auto-play with LLM** panel where you can paste any OpenAI-compatible endpoint, model name, and API key; click one button; and watch the model play in real time with full physics animation between shots. Perfect for recording demos.

---

## 8. Try it yourself

Everything is open-source.

- 🎯 **Live Space**: [huggingface.co/spaces/bpHigh/carrom_rl_env](https://huggingface.co/spaces/bpHigh/carrom_rl_env)
- 📦 **GitHub**: [github.com/bp-high/carrom_rl_env-](https://github.com/bp-high/carrom_rl_env-)
- 📓 **Colab notebook** (GRPO training): [`examples/grpo_carrom_tutorial.ipynb`](https://github.com/bp-high/carrom_rl_env-/blob/main/examples/grpo_carrom_tutorial.ipynb)
- ⚡ **Modal training script**: [`examples/train_modal.py`](https://github.com/bp-high/carrom_rl_env-/blob/main/examples/train_modal.py)

The Space's Gradio UI lets you play the board yourself or point it at any LLM endpoint for auto-play. For local benchmarking:

```bash
pip install -e .
NEBIUS_API_KEY=... API_BASE_URL=... MODEL_NAME=... python inference.py
```

---

## 9. What I'd build next

- **Multi-agent self-play**, replacing the scripted opponent with an older copy of the agent.
- **Curriculum learning**, starting from pre-set pocket-ready boards and grading up to full clears.
- **Vision observations**, rendered board images instead of structured text, for multimodal training.
- **Queen-focused tasks**, boards where the queen is the primary scoring target, testing cover-rule reasoning.
- **torchforge integration**, for distributed rollout collection on bigger models.

---

## 10. Acknowledgements

Built for the [OpenEnv Student Challenge](https://github.com/meta-pytorch/OpenEnv) sponsored by the PyTorch team at Meta, Hugging Face, and Unsloth. Huge thanks to the OpenEnv maintainers for the cleanest environment API I've seen. The green-agent framing is adapted from [Berkeley RDI's AgentBeats](https://rdi.berkeley.edu/agentx-agentbeats).

The game itself is a gift from my culture to everyone who has ever stretched across a wooden board, powder on their fingers, trying to pocket the queen without giving it back. I hope this makes some LLMs better at it.

---

*Questions or PRs welcome on [GitHub](https://github.com/bp-high/carrom_rl_env-). If you try your own model on this environment, I'd love to see the results.*
