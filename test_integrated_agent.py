
from environment.grid_world import GridWorld
from environment.scenario_generator import ScenarioGenerator
from agents.q_learning import QLearningAgent


# Generate a random scenario.
generator = ScenarioGenerator(
    size=10,
    static_obstacle_count=12,
    moving_obstacle_count=2
)

scenario = generator.generate_scenario()

# Create environment.
env = GridWorld(
    size=10,
    scenario=scenario
)

# Create Q-Learning agent.
agent = QLearningAgent()

# Get encoded initial state.
state = env.get_encoded_state()

print("\nIntegrated Agent Test")
print("-" * 40)

print("Initial Encoded State:", state)

# Select an action.
action = agent.choose_action(state)

print("Selected Action:", action)

# Execute action.
next_state, reward, terminated, truncated, info = (
    env.step(action)
)

# Get encoded next state.
next_state = env.get_encoded_state()

# Update Q-table.
agent.learn(
    state,
    action,
    reward,
    next_state,
    terminated or truncated
)

print("\nAfter One Action")
print("Next Encoded State:", next_state)
print("Reward:", reward)
print("Updated Q-values:", agent.get_q_values(state))
print("Q-table Size:", len(agent.q_table))

print("\nIntegrated Agent Test Completed!")
