
from environment.grid_world import GridWorld
from agents.q_learning import QLearningAgent


def evaluate_agent(episodes=100):

    env = GridWorld()
    agent = QLearningAgent()

    # Load the trained Q-table.
    agent.load("models/q_table.pkl")

    total_rewards = []
    total_steps = []

    successes = 0
    collisions = 0
    timeouts = 0

    print("\nStarting Agent Evaluation...")
    print("-" * 40)

    for episode in range(1, episodes + 1):

        state = env.reset()

        episode_reward = 0
        steps = 0

        while True:

            # No exploration during evaluation.
            action = agent.choose_action(
                state,
                training=False
            )

            next_state, reward, terminated, truncated, info = (
                env.step(action)
            )

            state = next_state
            episode_reward += reward
            steps += 1

            if terminated or truncated:

                if info["success"]:
                    successes += 1

                elif info["collision"]:
                    collisions += 1

                else:
                    timeouts += 1

                break

        total_rewards.append(episode_reward)
        total_steps.append(steps)

    # Calculate performance metrics.
    success_rate = (successes / episodes) * 100

    average_reward = sum(total_rewards) / episodes
    average_steps = sum(total_steps) / episodes

    print("\nEvaluation Results")
    print("-" * 40)

    print("Total Episodes:", episodes)
    print("Successful Episodes:", successes)
    print("Collisions:", collisions)
    print("Timeouts:", timeouts)

    print(f"Success Rate: {success_rate:.2f}%")
    print(f"Average Reward: {average_reward:.2f}")
    print(f"Average Steps: {average_steps:.2f}")

    print("\nEvaluation completed!")


if __name__ == "__main__":
    evaluate_agent(episodes=100)
