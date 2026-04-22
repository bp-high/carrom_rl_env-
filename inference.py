"""Baseline inference script for the Carrom OpenEnv environment.

Runs an LLM agent against the Carrom environment (ICF rules) and reports
game performance and Green Agent efficiency metrics.

Supports any OpenAI-compatible API endpoint.  Configure via environment
variables so the same script works with HuggingFace Inference, Nebius,
vLLM, OpenAI, or any other compatible provider:

    # HuggingFace Inference Router
    export API_BASE_URL="https://api-inference.huggingface.co/v1"
    export MODEL_NAME="Qwen/Qwen3-4B"
    export HF_TOKEN="hf_..."

    # Nebius
    export API_BASE_URL="https://api.studio.nebius.com/v1"
    export MODEL_NAME="nvidia/Llama-3.1-Nemotron-70B-Instruct-HF"
    export NEBIUS_API_KEY="ey..."

    # OpenAI
    export API_BASE_URL="https://api.openai.com/v1"
    export MODEL_NAME="gpt-4o-mini"
    export OPENAI_API_KEY="sk-..."

    # Local vLLM
    export API_BASE_URL="http://localhost:8000/v1"
    export MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
    # No key needed for local

Then run:
    python inference.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from typing import Optional

import requests

from carrom_env.env import CarromEnv
from carrom_env.models import Action, Observation
from carrom_env.green_agent import GreenCarromAgent, EvalReport, Task


# ---------------------------------------------------------------------------
# Configuration — all overridable via environment variables
# ---------------------------------------------------------------------------

API_BASE_URL = os.environ.get(
    "API_BASE_URL",
    "https://api-inference.huggingface.co/v1",
)
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3-4B")

# API key: checked in priority order — NEBIUS_API_KEY → OPENAI_API_KEY → HF_TOKEN
API_KEY: str = (
    os.environ.get("NEBIUS_API_KEY")
    or os.environ.get("OPENAI_API_KEY")
    or os.environ.get("HF_TOKEN")
    or ""
)

MAX_STEPS    = int(os.environ.get("MAX_STEPS",        "30"))
NUM_EPISODES = int(os.environ.get("NUM_EPISODES",      "3"))
TIMEOUT      = int(os.environ.get("TIMEOUT_MINUTES",  "20")) * 60

# ---------------------------------------------------------------------------
# System prompt (ICF rules)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert Carrom player following ICF (International Carrom Federation) rules.

Board layout
------------
- 1.0 × 1.0 square centred at (0, 0).  Pockets at the four corners (±0.5, ±0.5).
- Your striker starts on the BOTTOM baseline (y ≈ -0.42).
- You play WHITE coins.  The opponent plays BLACK coins.

Scoring & rules
---------------
- Pocket a WHITE coin  → +1 point, take another turn
- Pocket the QUEEN     → +3 points; you must then pocket a white coin on the
                         same shot OR your next turn to "cover" it
- Pocket a BLACK coin  → DUE: coin returns to board centre, your turn ENDS
- Pocket the STRIKER   → FOUL: one of your pocketed coins returns to board

Action format
-------------
Respond with ONLY a valid JSON object (no markdown, no explanation):
{
  "placement_x": <float, -0.4 to 0.4, 0 = centre>,
  "angle":       <float, radians, 0 = straight ahead toward +y>,
  "force":       <float, 0.0 to 1.0>
}

Strategy tips
-------------
- Prioritise white coins close to pockets for easy points
- Avoid shooting black coins — even if they are near a pocket
- Queen near centre: aim to pocket it AND a white coin in the same shot
- Adjust placement_x to get a direct line on your target
"""


# ---------------------------------------------------------------------------
# LLM interaction
# ---------------------------------------------------------------------------

def call_llm(observation_text: str) -> Optional[dict]:
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system",  "content": SYSTEM_PROMPT},
            {"role": "user",    "content": observation_text},
        ],
        # Generous budget to accommodate reasoning models (e.g. MiniMax-M2.5,
        # Nemotron) that emit long CoT before the final JSON answer.
        "max_tokens": int(os.environ.get("MAX_TOKENS", "2048")),
        "temperature": 0.3,
    }
    try:
        resp = requests.post(
            f"{API_BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        msg = resp.json()["choices"][0]["message"]
        # Reasoning models put their final answer in `content` and the trace in
        # `reasoning_content`.  Fall back to reasoning_content if content is
        # null (common when the JSON is inline inside the reasoning).
        text = msg.get("content") or msg.get("reasoning_content") or ""
        return _parse_json_action(text)
    except Exception as e:
        print(f"  [LLM error] {e}")
        return None


def _parse_json_action(text: str) -> Optional[dict]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$",          "", text)
    # Strip <think>…</think> blocks (some reasoning models)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    match = re.search(r"\{[^}]+\}", text)
    if match:
        try:
            data = json.loads(match.group())
            return {
                "placement_x": float(data.get("placement_x", 0.0)),
                "angle":       float(data.get("angle",       0.0)),
                "force":       float(data.get("force",       0.5)),
            }
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
    return None


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------

_llm_turn_counter = {"n": 0}


def llm_policy(obs: Observation) -> Action:
    _llm_turn_counter["n"] += 1
    parsed = call_llm(obs.text_summary)
    if parsed:
        action = Action(**parsed)
        print(f"  [shot {_llm_turn_counter['n']:>3}] "
              f"px={action.placement_x:+.2f} "
              f"angle={action.angle:+.2f} "
              f"force={action.force:.2f}   "
              f"(score {obs.agent_score}-{obs.opponent_score}, "
              f"coins left {obs.remaining_coins})", flush=True)
        return action
    import random
    print(f"  [shot {_llm_turn_counter['n']:>3}] PARSE FAIL → random fallback", flush=True)
    return Action(
        placement_x=random.uniform(-0.2, 0.2),
        angle=random.uniform(-0.5, 0.5),
        force=random.uniform(0.3, 0.8),
    )


def random_policy(obs: Observation) -> Action:
    import random
    return Action(
        placement_x=random.uniform(-0.35, 0.35),
        angle=random.uniform(-1.0, 1.0),
        force=random.uniform(0.2, 1.0),
    )


def heuristic_policy(obs: Observation) -> Action:
    """Aim at the nearest WHITE coin to a pocket; avoid black coins."""
    import math
    best_angle    = 0.0
    best_placement = 0.0
    best_score    = float("inf")
    baseline_y    = -0.5 + 0.08

    for coin in obs.coins:
        if coin.pocketed:
            continue
        # Skip black coins — pocketing them is a due under ICF rules
        if coin.color == "black":
            continue
        dx    = coin.x - 0.0
        dy    = coin.y - baseline_y
        angle = math.atan2(dx, dy)
        score = coin.pocket_distance
        if coin.color == "queen":
            score *= 0.5
        if score < best_score:
            best_score     = score
            best_angle     = angle
            best_placement = max(-0.35, min(0.35, coin.x * 0.5))

    return Action(placement_x=best_placement, angle=best_angle, force=0.6)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def build_task_suite(num_episodes: int, max_steps: int) -> list[Task]:
    """Flat task suite used for baseline comparisons: `num_episodes` tasks at
    the given horizon, each with a unique seed.  Keeps every policy evaluated
    on the *same* set of board states for fair comparison.
    """
    return [
        Task(task_id=f"ep_{i}", seed=i * 100, max_turns=max_steps, tier="standard")
        for i in range(num_episodes)
    ]


def run_baseline(
    policy_fn,
    policy_name: str,
    tasks: list[Task],
) -> EvalReport:
    """Run a purple agent (policy_fn) against the shared task suite
    using the green-agent evaluator, and print the scorecard.
    """
    print(f"\n--- Evaluating: {policy_name} ({len(tasks)} tasks) ---")
    evaluator = GreenCarromAgent(tasks=tasks)
    report    = evaluator.evaluate(policy_fn, verbose=True)

    s = report.summary()
    print(f"\n=== {policy_name} ({s['n_tasks']} tasks) ===")
    print(f"  Avg reward     : {s['avg_reward']:+.3f}")
    print(f"  Win rate       : {s['win_rate']:.2f}")
    print(f"  Avg coins      : {s['avg_coins_potted']:.1f}")
    print(f"  Avg dues       : {s['avg_dues']:.2f}   (ICF violations)")
    print(f"  Avg fouls      : {s['avg_fouls']:.2f}")
    print(f"  ICF compliance : {s['icf_compliance']:.3f}")
    print(f"  Sim steps      : {s['total_sim_steps']}")
    print(f"  Efficiency     : {s['efficiency_score']:.4f} coins/1k-steps")
    return report


def launch_web_server(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Start the FastAPI + Gradio server (foreground) and print the watch URL.

    Use this when you want to watch the LLM play on the board and screen-record
    it.  Configure the endpoint/model/key inside the "Auto-play with LLM" panel
    in the browser, then click "Auto-play" to stream animated shots.

    The environment variables ``API_BASE_URL``, ``MODEL_NAME``, and an API key
    (``NEBIUS_API_KEY`` / ``OPENAI_API_KEY`` / ``HF_TOKEN``) are inherited as
    defaults in the web form.
    """
    env = os.environ.copy()
    env["ENABLE_WEB_INTERFACE"] = "true"
    env.setdefault("PYTHONPATH", os.getcwd())

    url = f"http://localhost:{port}/web"
    print("=" * 70)
    print("Carrom server starting with web UI…")
    print(f"  Open:   {url}")
    print(f"  Inside the UI, configure model/endpoint, set number of shots,")
    print(f"  then click the \"🤖 Auto-play with LLM\" button to watch it play.")
    print(f"  Press Ctrl+C in this terminal to stop the server.")
    print("=" * 70)

    cmd = [
        sys.executable, "-m", "uvicorn",
        "server.app:app",
        "--host", host,
        "--port", str(port),
        "--ws-ping-interval", "60",
        "--ws-ping-timeout",  "60",
    ]
    # Foreground: user ctrl-c's to stop
    try:
        subprocess.run(cmd, env=env, check=False)
    except KeyboardInterrupt:
        print("\nServer stopped.")


def main():
    parser = argparse.ArgumentParser(
        description="Carrom inference — headless baselines or live web view."
    )
    parser.add_argument(
        "--web", action="store_true",
        help="Start the env server + Gradio web UI for auto-play watching "
             "(screen-record friendly). No headless baselines run in this mode.",
    )
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    if args.web:
        launch_web_server(host=args.host, port=args.port)
        return

    print(f"API endpoint : {API_BASE_URL}")
    print(f"Model        : {MODEL_NAME}")
    print(f"API key set  : {'yes' if API_KEY else 'no'}")

    # Shared task suite — every policy sees the same boards (deterministic)
    tasks = build_task_suite(NUM_EPISODES, MAX_STEPS)
    print(f"Task suite   : {len(tasks)} × {MAX_STEPS}-turn boards\n")

    start   = time.time()
    reports: dict[str, EvalReport] = {}

    print("=" * 60 + "\nPURPLE AGENT: Random\n" + "=" * 60)
    reports["random"] = run_baseline(random_policy, "Random", tasks)

    print("\n" + "=" * 60 + "\nPURPLE AGENT: Heuristic (ICF-aware)\n" + "=" * 60)
    reports["heuristic"] = run_baseline(heuristic_policy, "Heuristic", tasks)

    if API_KEY:
        elapsed = time.time() - start
        if elapsed < TIMEOUT - 120:
            print(f"\n{'=' * 60}\nPURPLE AGENT: LLM ({MODEL_NAME})\n{'=' * 60}")
            reports["llm"] = run_baseline(llm_policy, f"LLM ({MODEL_NAME})", tasks)
        else:
            print(f"\nSkipping LLM baseline — {elapsed:.0f}s elapsed.")
    else:
        print("\nSkipping LLM baseline — no API key (set NEBIUS_API_KEY / OPENAI_API_KEY / HF_TOKEN).")

    # ── Leaderboard ────────────────────────────────────────────────
    print("\n" + "=" * 78 + "\nLEADERBOARD\n" + "=" * 78)
    print(f"{'Purple Agent':<25} {'Reward':>8} {'Win%':>6} {'Coins':>6} {'Dues':>6} {'ICF%':>6} {'Eff':>8}")
    print("-" * 78)
    for name, report in reports.items():
        s = report.summary()
        print(f"{name:<25} {s['avg_reward']:>+8.2f} {s['win_rate']*100:>5.0f}% "
              f"{s['avg_coins_potted']:>6.1f} {s['avg_dues']:>6.2f} "
              f"{s['icf_compliance']*100:>5.0f}% {s['efficiency_score']:>8.3f}")

    print(f"\nTotal runtime: {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
