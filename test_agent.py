
from environment.grid_world import GridWorld
from agents.q_learning import QLearningAgent

# Initialize environment and agent
env = GridWorld()
agent = QLearningAgent()

# Get initial state
state = env.reset()

print("Initial state:", state)

# Select an action
action = agent.choose_action(state)

print("Selected action:", action)

# Perform the action
next_state, reward, terminated, truncated, info = env.step(action)

# Update Q-table
done = terminated or truncated

agent.learn(
    state,
    action,
    reward,
    next_state,
    done
)

print("Reward received:", reward)
print("Updated Q-value:", agent.get_q_values(state))
print("Q-table size:", len(agent.q_table))

print("\nQ-Learning agent test completed!")
