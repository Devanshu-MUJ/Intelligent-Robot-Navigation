
import csv
import os
import matplotlib.pyplot as plt


def load_csv(filename):

    with open(filename, "r") as file:
        reader = csv.DictReader(file)
        return list(reader)


def calculate_rolling_average(values, window=100):

    averages = []

    for i in range(len(values)):

        start = max(0, i - window + 1)
        recent = values[start:i + 1]

        averages.append(
            sum(recent) / len(recent)
        )

    return averages


def plot_training_success():

    data = load_csv(
        "results/training_results_v2.csv"
    )

    episodes = [
        int(row["episode"]) for row in data
    ]

    successes = [
        int(row["success"]) for row in data
    ]

    success_rates = calculate_rolling_average(
        successes, window=100
    )

    success_rates = [
        value * 100 for value in success_rates
    ]

    plt.figure(figsize=(10, 5))

    plt.plot(episodes, success_rates)

    plt.title("Q-Learning Training Success Rate")
    plt.xlabel("Episodes")
    plt.ylabel("Success Rate (%)")

    plt.ylim(0, 100)
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        "results/graphs/training_success.png"
    )

    plt.show()


def plot_training_reward():

    data = load_csv(
        "results/training_results_v2.csv"
    )

    episodes = [
        int(row["episode"]) for row in data
    ]

    rewards = [
        float(row["reward"]) for row in data
    ]

    average_rewards = calculate_rolling_average(
        rewards, window=100
    )

    plt.figure(figsize=(10, 5))

    plt.plot(episodes, average_rewards)

    plt.title("Q-Learning Average Reward")
    plt.xlabel("Episodes")
    plt.ylabel("Average Reward")

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        "results/graphs/training_reward.png"
    )

    plt.show()


def plot_evaluation_results():

    data = load_csv(
        "results/evaluation_results_v2.csv"
    )

    successes = sum(
        int(row["success"]) for row in data
    )

    collisions = sum(
        int(row["collision"]) for row in data
    )

    timeouts = sum(
        int(row["timeout"]) for row in data
    )

    labels = [
        "Success",
        "Collision",
        "Timeout"
    ]

    values = [
        successes,
        collisions,
        timeouts
    ]

    plt.figure(figsize=(8, 5))

    plt.bar(labels, values)

    plt.title("Unseen Environment Evaluation")
    plt.xlabel("Outcome")
    plt.ylabel("Number of Episodes")

    plt.grid(axis="y")

    plt.tight_layout()

    plt.savefig(
        "results/graphs/evaluation_results.png"
    )

    plt.show()


def generate_all_graphs():

    os.makedirs(
        "results/graphs",
        exist_ok=True
    )

    print("\nGenerating Project Graphs...")
    print("-" * 40)

    plot_training_success()
    plot_training_reward()
    plot_evaluation_results()

    print("\nAll graphs generated successfully!")


if __name__ == "__main__":

    generate_all_graphs()
