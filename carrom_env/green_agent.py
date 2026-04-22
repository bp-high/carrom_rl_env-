"""Green Agent — the Carrom evaluator.

In the AgentBeats / AgentX taxonomy:

    🟢 Green Agent (evaluator) — defines tasks, environment, and scoring.
    🟣 Purple Agent (competitor) — the AI agent under test.
    🔴 Red Agent (adversarial) — finds weaknesses.

The ``GreenCarromAgent`` defined here *is the green agent* for this benchmark:

    1. **Tasks** — a curated suite of seeded board configurations across
       easy / standard / hard tiers.
    2. **Environment** — wraps :class:`CarromEnv` with full ICF rule compliance.
    3. **Scoring** — ICF-aware metrics (reward, win rate, ICF compliance from
       dues/fouls) plus compute efficiency (sim steps, wall time).

The purple agent is any ``Callable[[Observation], Action]``.

Example
-------
>>> from carrom_env.green_agent import GreenCarromAgent
>>> def my_policy(obs): return Action(placement_x=0.0, angle=0.0, force=0.5)
>>> evaluator = GreenCarromAgent()
>>> report = evaluator.evaluate(my_policy, verbose=True)
>>> print(report.summary())
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from carrom_env.env import CarromEnv
from carrom_env.models import Action, Observation


PolicyFn = Callable[[Observation], Action]
"""Purple-agent interface: observation in, action out."""


# ---------------------------------------------------------------------------
# Task & result types
# ---------------------------------------------------------------------------

@dataclass
class Task:
    """A single evaluation task — one seeded board configuration."""
    task_id:     str
    seed:        int
    max_turns:   int  = 30
    tier:        str  = "standard"   # easy | standard | hard
    description: str  = ""


@dataclass
class TaskResult:
    """Scorecard for a single purple-agent run on a single task."""
    task_id:            str
    tier:               str
    # Game outcome (ICF-aware)
    total_reward:       float
    coins_potted:       int
    due_coins:          int     # ICF rule violations — opponent's coins pocketed
    fouls:              int     # striker pocketed
    turns_played:       int
    agent_score:        int
    opponent_score:     int
    won:                bool
    # Compute metrics (secondary)
    total_sim_steps:    int
    total_wall_time_s:  float

    @property
    def icf_compliance(self) -> float:
        """Fraction of shots that obeyed ICF rules (no dues, no fouls). 1.0 = perfect."""
        if self.turns_played == 0:
            return 1.0
        violations = self.due_coins + self.fouls
        return max(0.0, 1.0 - violations / self.turns_played)


@dataclass
class EvalReport:
    """Aggregated evaluation across the task suite."""
    per_task: List[TaskResult] = field(default_factory=list)

    # ── Game metrics ────────────────────────────────────────────────
    @property
    def avg_reward(self) -> float:
        return _mean(t.total_reward for t in self.per_task)

    @property
    def win_rate(self) -> float:
        return _mean(1.0 if t.won else 0.0 for t in self.per_task)

    @property
    def avg_coins_potted(self) -> float:
        return _mean(float(t.coins_potted) for t in self.per_task)

    @property
    def avg_dues(self) -> float:
        return _mean(float(t.due_coins) for t in self.per_task)

    @property
    def avg_fouls(self) -> float:
        return _mean(float(t.fouls) for t in self.per_task)

    @property
    def icf_compliance(self) -> float:
        return _mean(t.icf_compliance for t in self.per_task) if self.per_task else 1.0

    # ── Compute metrics ─────────────────────────────────────────────
    @property
    def total_sim_steps(self) -> int:
        return sum(t.total_sim_steps for t in self.per_task)

    @property
    def total_wall_time_s(self) -> float:
        return sum(t.total_wall_time_s for t in self.per_task)

    @property
    def efficiency_score(self) -> float:
        """Coins potted per 1000 sim steps — higher = more compute-efficient."""
        total_coins = sum(t.coins_potted for t in self.per_task)
        return total_coins / max(self.total_sim_steps, 1) * 1000.0

    # ── Aggregates ──────────────────────────────────────────────────
    def summary(self) -> Dict[str, float]:
        return {
            "n_tasks":          len(self.per_task),
            "avg_reward":       round(self.avg_reward,      3),
            "win_rate":         round(self.win_rate,        3),
            "avg_coins_potted": round(self.avg_coins_potted, 2),
            "avg_dues":         round(self.avg_dues,        2),
            "avg_fouls":        round(self.avg_fouls,       2),
            "icf_compliance":   round(self.icf_compliance,  3),
            "total_sim_steps":  self.total_sim_steps,
            "total_wall_time_s": round(self.total_wall_time_s, 3),
            "efficiency_score": round(self.efficiency_score, 4),
        }

    def by_tier(self) -> Dict[str, Dict[str, float]]:
        """Break down summary by difficulty tier."""
        groups: Dict[str, List[TaskResult]] = {}
        for t in self.per_task:
            groups.setdefault(t.tier, []).append(t)
        out: Dict[str, Dict[str, float]] = {}
        for tier, results in groups.items():
            n = len(results)
            out[tier] = {
                "n":              n,
                "avg_reward":     round(sum(r.total_reward for r in results) / n,      3),
                "avg_coins":      round(sum(r.coins_potted for r in results) / n,      2),
                "win_rate":       round(sum(1 for r in results if r.won) / n,          3),
                "icf_compliance": round(sum(r.icf_compliance for r in results) / n,    3),
            }
        return out


def _mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


# ---------------------------------------------------------------------------
# The Green Agent itself
# ---------------------------------------------------------------------------

class GreenCarromAgent:
    """Green agent (evaluator) for the Carrom benchmark.

    Defines the task suite, owns the environment lifecycle, and scores a
    purple agent's performance under ICF rules.
    """

    def __init__(self, tasks: Optional[List[Task]] = None):
        self.tasks = tasks if tasks is not None else self.default_task_suite()

    # ── Task definition ────────────────────────────────────────────
    @staticmethod
    def default_task_suite(tasks_per_tier: int = 3) -> List[Task]:
        """Curated default benchmark: easy / standard / hard tiers.

        Higher tiers = longer horizon, harder to finish without ICF violations.
        """
        suite: List[Task] = []
        for i in range(tasks_per_tier):
            suite.append(Task(
                task_id=f"easy_{i}",     seed=100 + i, max_turns=20, tier="easy",
                description="Short game — tests basic pocketing",
            ))
        for i in range(tasks_per_tier):
            suite.append(Task(
                task_id=f"standard_{i}", seed=200 + i, max_turns=60, tier="standard",
                description="Default board, full episode",
            ))
        for i in range(tasks_per_tier):
            suite.append(Task(
                task_id=f"hard_{i}",     seed=300 + i, max_turns=100, tier="hard",
                description="Long horizon — tests ICF compliance over time",
            ))
        return suite

    # ── Evaluation ─────────────────────────────────────────────────
    def evaluate(self, policy_fn: PolicyFn, verbose: bool = False) -> EvalReport:
        """Run a purple agent against the full task suite and score it."""
        report = EvalReport()
        for task in self.tasks:
            report.per_task.append(self.evaluate_task(policy_fn, task, verbose=verbose))
        return report

    def evaluate_task(
        self, policy_fn: PolicyFn, task: Task, verbose: bool = False,
    ) -> TaskResult:
        """Run the purple agent on one task and return its scorecard."""
        env = CarromEnv(seed=task.seed, max_turns=task.max_turns)
        obs = env.reset(seed=task.seed)

        total_reward = 0.0
        coins_potted = 0
        due_coins    = 0
        fouls        = 0
        sim_steps    = 0
        turns_played = 0

        t0 = time.perf_counter()
        while turns_played < task.max_turns:
            action = policy_fn(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            coins_potted += int(info.get("coin_potted", 0))
            due_coins    += int(info.get("due_coins",  0))
            fouls        += int(info.get("foul",       0))
            sim_steps    += int(info.get("sim_steps",  0))
            turns_played  = int(info.get("turn_count", turns_played + 1))
            if terminated or truncated:
                break
        wall_time = time.perf_counter() - t0

        result = TaskResult(
            task_id=task.task_id,
            tier=task.tier,
            total_reward=total_reward,
            coins_potted=coins_potted,
            due_coins=due_coins,
            fouls=fouls,
            turns_played=turns_played,
            agent_score=env.agent_score,
            opponent_score=env.opponent_score,
            won=env.agent_score > env.opponent_score,
            total_sim_steps=sim_steps,
            total_wall_time_s=wall_time,
        )
        if verbose:
            print(f"  [{task.task_id:<12} {task.tier:<8}] "
                  f"reward={result.total_reward:+7.2f} "
                  f"coins={result.coins_potted:2d} "
                  f"dues={result.due_coins} fouls={result.fouls} "
                  f"won={'✓' if result.won else '✗'} "
                  f"icf={result.icf_compliance:.2f}")
        return result
