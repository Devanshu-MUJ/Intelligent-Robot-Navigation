
import os
import csv

from environment.grid_world import GridWorld
from environment.scenario_generator import ScenarioGenerator
from agents.q_learning import QLearningAgent


def train_agent(episodes=5000):

    # Create scenario generator.
    generator = ScenarioGenerator(
        size=10,
        static_obstacle_count=12,
        moving_obstacle_count=2
    )

    # Create Q-Learning agent.
    agent = QLearningAgent(
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.999
    )

    # Store training results.
    results = []

    successes = 0
    collisions = 0
    timeouts = 0

    print("\nStarting Randomized Q-Learning Training...")
    print("-" * 55)

    for episode in range(1, episodes + 1):

        # Generate a new scenario for this episode.
        scenario = generator.generate_scenario()

        # Create a new environment.
        env = GridWorld(
            size=10,
            max_steps=200,
            scenario=scenario
        )

        # Get the encoded initial state.
        state = env.get_encoded_state()

        total_reward = 0
        episode_steps = 0

        success = False
        collision = False

        while True:

            # Select an action.
            action = agent.choose_action(
                state,
                training=True
            )

            # Perform the action.
            _, reward, terminated, truncated, info = (
                env.step(action)
            )

            # Encode the next state.
            next_state = env.get_encoded_state()

            # Update Q-values.
            # Bootstrap through time-limit truncation.
            agent.learn(
                state,
                action,
                reward,
                next_state,
                terminated
            )

            state = next_state

            total_reward += reward
            episode_steps += 1

            if terminated or truncated:

                success = info["success"]
                collision = info["collision"]

                break

        # Reduce exploration.
        agent.decay_epsilon()

        if success:
            successes += 1

        elif collision:
            collisions += 1

        else:
            timeouts += 1

        # Store episode information.
        results.append({
            "episode": episode,
            "reward": total_reward,
            "steps": episode_steps,
            "success": int(success),
            "collision": int(collision),
            "timeout": int(not success and not collision),
            "epsilon": agent.epsilon
        })

        # Display progress every 100 episodes.
        if episode % 100 == 0:

            recent = results[-100:]

            avg_reward = sum(
                item["reward"] for item in recent
            ) / len(recent)

            recent_successes = sum(
                item["success"] for item in recent
            )

            recent_collisions = sum(
                item["collision"] for item in recent
            )

            recent_timeouts = sum(
                item["timeout"] for item in recent
            )

            print(
                f"Episode: {episode}/{episodes} | "
                f"Avg Reward: {avg_reward:.2f} | "
                f"Success: {recent_successes}/100 | "
                f"Collisions: {recent_collisions}/100 | "
                f"Timeouts: {recent_timeouts}/100 | "
                f"Epsilon: {agent.epsilon:.3f} | "
                f"Q-table: {len(agent.q_table)}"
            )

    # Create folders.
    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # Save training history separately.
    csv_path = "results/training_results_v2.csv"

    with open(csv_path, "w", newline="") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "episode",
                "reward",
                "steps",
                "success",
                "collision",
                "timeout",
                "epsilon"
            ]
        )

        writer.writeheader()
        writer.writerows(results)

    # Save the new Q-table.
    model_path = "models/q_table_v2.pkl"

    agent.save(model_path)

    # Calculate final performance.
    success_rate = (successes / episodes) * 100
    collision_rate = (collisions / episodes) * 100
    timeout_rate = (timeouts / episodes) * 100

    average_reward = sum(
        item["reward"] for item in results
    ) / episodes

    average_steps = sum(
        item["steps"] for item in results
    ) / episodes

    print("\nTraining Completed!")
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

    print("Final Q-table Size:", len(agent.q_table))
    print("Model Saved:", model_path)
    print("Results Saved:", csv_path)

    return agent, results


if __name__ == "__main__":

    train_agent(episodes=5000)
