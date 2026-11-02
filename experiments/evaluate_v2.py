
import csv
import os

from environment.grid_world import GridWorld
from environment.scenario_generator import ScenarioGenerator
from agents.q_learning import QLearningAgent


def evaluate_agent(episodes=500):

    # Generate unseen scenarios.
    generator = ScenarioGenerator(
        size=10,
        static_obstacle_count=12,
        moving_obstacle_count=2
    )

    # Load trained agent.
    agent = QLearningAgent()
    agent.load("models/q_table_v2.pkl")

    # Disable exploration.
    agent.epsilon = 0.0

    results = []

    successes = 0
    collisions = 0
    timeouts = 0

    total_rewards = []
    total_steps = []

    print("\nStarting Unseen Environment Evaluation...")
    print("-" * 55)

    for episode in range(1, episodes + 1):

        # Generate a new environment.
        scenario = generator.generate_scenario()

        env = GridWorld(
            size=10,
            max_steps=200,
            scenario=scenario
        )

        state = env.get_encoded_state()

        episode_reward = 0
        steps = 0

        while True:

            # Greedy action selection.
            action = agent.choose_action(
                state,
                training=False
            )

            # Execute action.
            _, reward, terminated, truncated, info = (
                env.step(action)
            )

            # Encode next state.
            next_state = env.get_encoded_state()

            # IMPORTANT:
            # No agent.learn() call during evaluation.
            state = next_state

            episode_reward += reward
            steps += 1

            if terminated or truncated:
                break

        # Record outcome.
        success = info["success"]
        collision = info["collision"]
        timeout = not success and not collision

        if success:
            successes += 1

        elif collision:
            collisions += 1

        else:
            timeouts += 1

        total_rewards.append(episode_reward)
        total_steps.append(steps)

        results.append({
            "episode": episode,
            "reward": episode_reward,
            "steps": steps,
            "success": int(success),
            "collision": int(collision),
            "timeout": int(timeout)
        })

        if episode % 50 == 0:

            recent = results[-50:]

            recent_successes = sum(
                item["success"] for item in recent
            )

            print(
                f"Episode: {episode}/{episodes} | "
                f"Recent Successes: {recent_successes}/50"
            )

    # Calculate metrics.
    success_rate = successes / episodes * 100
    collision_rate = collisions / episodes * 100
    timeout_rate = timeouts / episodes * 100

    average_reward = sum(total_rewards) / episodes
    average_steps = sum(total_steps) / episodes

    # Save evaluation results.
    os.makedirs("results", exist_ok=True)

    csv_path = "results/evaluation_results_v2.csv"

    with open(csv_path, "w", newline="") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "episode",
                "reward",
                "steps",
                "success",
                "collision",
                "timeout"
            ]
        )

        writer.writeheader()
        writer.writerows(results)

    print("\nUnseen Environment Evaluation Results")
    print("-" * 55)

    print("Total Episodes:", episodes)
    print("Successful Episodes:", successes)
    print("Collisions:", collisions)
    print("Timeouts:", timeouts)

    print(f"Success Rate: {success_rate:.2f}%")
    print(f"Collision Rate: {collision_rate:.2f}%")
    print(f"Timeout Rate: {timeout_rate:.2f}%")

    print(f"Average Reward: {average_reward:.2f}")
    print(f"Average Steps: {average_steps:.2f}")

    print("\nEvaluation Results Saved:", csv_path)
    print("Evaluation Completed!")


if __name__ == "__main__":

    evaluate_agent(episodes=500)
