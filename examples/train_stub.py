"""Minimal usage stub for the Carrom environment + Green Agent evaluator."""

import random

from carrom_env.env import CarromEnv
from carrom_env.models import Action
from carrom_env.green_agent import GreenCarromAgent, Task


def random_action():
    return Action(
        placement_x=random.uniform(-0.2, 0.2),
        angle=random.uniform(-0.5, 0.5),
        force=random.uniform(0.2, 1.0),
        spin=0.0,
    )


def random_policy(obs):
    return random_action()


def main():
    # Method 1: Direct env usage
    env = CarromEnv(seed=0)
    obs = env.reset()
    total_reward = 0.0
    for _ in range(50):
        action = random_action()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    print("Direct env:", {"reward": total_reward, "turns": info.get("turn_count")})

    # Method 2: Green Agent (evaluator) — purple agent vs. task suite
    # The purple agent here is random_policy; green agent defines tasks & scores.
    evaluator = GreenCarromAgent(tasks=[
        Task(task_id="demo_easy",     seed=0, max_turns=15, tier="easy"),
        Task(task_id="demo_standard", seed=1, max_turns=30, tier="standard"),
    ])
    report = evaluator.evaluate(random_policy, verbose=True)
    print("\nGreen Agent scorecard:", report.summary())
    print("By tier:", report.by_tier())

    # Method 3: Text actions (for LLM agents)
    env = CarromEnv(seed=42)
    obs = env.reset()
    text_action = Action(
        action_type="text",
        text="aim at queen_0 with strong force from center",
    )
    obs, reward, _, _, info = env.step(text_action)
    print(f"\nText action: reward={reward:.3f}, coins_potted={info['coin_potted']}")


if __name__ == "__main__":
    main()
