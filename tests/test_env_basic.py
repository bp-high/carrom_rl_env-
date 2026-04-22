import math

from carrom_env.env import CarromEnv
from carrom_env.models import Action, CoinInfo
from carrom_env.constants import MAX_COINS
from carrom_env.green_agent import GreenCarromAgent, Task


def test_reset_shapes():
    env = CarromEnv(seed=42)
    obs = env.reset()
    assert len(obs.positions) == 1 + MAX_COINS
    assert len(obs.velocities) == 1 + MAX_COINS
    assert len(obs.pocketed) == 1 + MAX_COINS


def test_step_determinism():
    env = CarromEnv(seed=123)
    env.reset()
    action = Action(placement_x=0.0, angle=0.0, force=0.5, spin=0.0)
    obs1, r1, t1, tr1, _ = env.step(action)

    env2 = CarromEnv(seed=123)
    env2.reset()
    obs2, r2, t2, tr2, _ = env2.step(action)

    assert r1 == r2
    assert t1 == t2
    assert tr1 == tr2
    assert obs1.remaining_coins == obs2.remaining_coins


def test_invalid_action_penalty():
    env = CarromEnv(seed=0)
    env.reset()
    action = Action(placement_x=0.0, angle=0.0, force=-1.0, spin=0.0)
    obs, reward, _, _, _ = env.step(action)
    assert math.isfinite(reward)
    assert obs.remaining_coins >= 0


def test_opponent_turn():
    env = CarromEnv(seed=1)
    env.reset()
    action = Action(placement_x=0.0, angle=0.0, force=0.1, spin=0.0)
    obs, _, _, _, info = env.step(action)
    assert info["current_player"] in {"agent", "opponent"}
    assert obs.current_player in {"agent", "opponent"}


def test_pot_detection():
    env = CarromEnv(seed=2)
    env.reset()
    # Move a coin into the corner pocket and take a zero-force shot.
    pocket = env._pocket_centers()[0]
    coin = next(c for c in env.coins if not c.pocketed)
    coin.body.position = pocket
    action = Action(placement_x=0.0, angle=0.0, force=0.0, spin=0.0)
    obs, _, _, _, _ = env.step(action)
    assert obs.remaining_coins <= MAX_COINS - 1


# --- New tests for v0.2.0 features ---


def test_text_action_aim_at_coin():
    """Text actions should parse and produce valid results."""
    env = CarromEnv(seed=10)
    env.reset()
    action = Action(action_type="text", text="aim at queen_0 with strong force from center")
    obs, reward, terminated, truncated, info = env.step(action)
    assert math.isfinite(reward)
    assert obs.remaining_coins >= 0


def test_text_action_key_value():
    """Key-value text actions should parse correctly."""
    env = CarromEnv(seed=11)
    env.reset()
    action = Action(action_type="text", text="placement_x=0.1 angle=0.3 force=0.7")
    obs, reward, terminated, truncated, info = env.step(action)
    assert math.isfinite(reward)


def test_text_action_pocket_target():
    """Aiming at a pocket should work."""
    env = CarromEnv(seed=12)
    env.reset()
    action = Action(action_type="text", text="shoot toward top-left pocket with medium force")
    obs, reward, terminated, truncated, info = env.step(action)
    assert math.isfinite(reward)


def test_observation_has_coin_info():
    """Observation should include per-coin details."""
    env = CarromEnv(seed=42)
    obs = env.reset()
    assert len(obs.coins) > 0
    assert isinstance(obs.coins[0], CoinInfo)
    assert obs.coins[0].nearest_pocket in {
        "bottom-left", "bottom-right", "top-left", "top-right"
    }
    assert obs.coins[0].pocket_distance > 0


def test_observation_text_summary_rich():
    """Text summary should contain board details for LLM readability."""
    env = CarromEnv(seed=42)
    obs = env.reset()
    assert "Carrom Board State" in obs.text_summary
    assert "Active coins:" in obs.text_summary
    assert "queen_0" in obs.text_summary
    assert "nearest pocket:" in obs.text_summary


def test_observation_turn_info():
    """Observation should include turn/max_turns."""
    env = CarromEnv(seed=42, max_turns=100)
    obs = env.reset()
    assert obs.turn_number == 0
    assert obs.max_turns == 100


def test_reward_win_bonus():
    """Agent should get a win bonus when clearing the board with a lead."""
    env = CarromEnv(seed=0)
    env.reset()
    # Pocket all coins manually
    for coin in env.coins:
        coin.pocketed = True
        if coin.body in env.space.bodies:
            env.space.remove(coin.body, coin.shape)
    # Artificially unpocket one coin to pot it via action
    last = env.coins[-1]
    last.pocketed = False
    pocket = env._pocket_centers()[0]
    last.body.position = pocket
    env.space.add(last.body, last.shape)
    env.agent_score = 10
    env.opponent_score = 2

    action = Action(placement_x=0.0, angle=0.0, force=0.0)
    obs, reward, terminated, truncated, info = env.step(action)
    # Should include win bonus
    assert reward > 1.0 or terminated


def test_green_agent_default_suite():
    """Green agent (evaluator) should have a tiered default task suite."""
    evaluator = GreenCarromAgent()
    assert len(evaluator.tasks) >= 3
    tiers = {t.tier for t in evaluator.tasks}
    assert {"easy", "standard", "hard"}.issubset(tiers)


def test_green_agent_evaluates_purple():
    """Green agent should score a purple-agent (policy fn) across its task suite."""
    def purple_agent(obs):
        return Action(placement_x=0.0, angle=0.0, force=0.5)

    # Compact suite for fast tests
    tasks = [
        Task(task_id="t_easy",     seed=0, max_turns=5, tier="easy"),
        Task(task_id="t_standard", seed=1, max_turns=5, tier="standard"),
    ]
    evaluator = GreenCarromAgent(tasks=tasks)
    report = evaluator.evaluate(purple_agent)
    summary = report.summary()

    assert summary["n_tasks"] == 2
    assert "avg_reward"      in summary
    assert "win_rate"        in summary
    assert "icf_compliance"  in summary
    assert "efficiency_score" in summary
    assert summary["total_sim_steps"] > 0


def test_green_agent_single_task():
    """evaluate_task should return a full TaskResult with ICF compliance."""
    def purple(obs):
        return Action(placement_x=0.0, angle=0.0, force=0.3)

    task      = Task(task_id="unit", seed=7, max_turns=3, tier="easy")
    evaluator = GreenCarromAgent(tasks=[task])
    result    = evaluator.evaluate_task(purple, task)

    assert result.task_id == "unit"
    assert result.tier    == "easy"
    assert 0.0 <= result.icf_compliance <= 1.0
    assert result.total_sim_steps > 0


def test_green_agent_by_tier():
    """by_tier() should group results by difficulty tier."""
    def purple(obs):
        return Action(placement_x=0.0, angle=0.0, force=0.5)

    tasks = [
        Task(task_id="e1", seed=0, max_turns=3, tier="easy"),
        Task(task_id="s1", seed=1, max_turns=3, tier="standard"),
    ]
    report = GreenCarromAgent(tasks=tasks).evaluate(purple)
    by_tier = report.by_tier()
    assert "easy"     in by_tier
    assert "standard" in by_tier
    assert by_tier["easy"]["n"]     == 1
    assert by_tier["standard"]["n"] == 1
