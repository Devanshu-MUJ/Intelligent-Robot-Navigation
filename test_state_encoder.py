
from environment.grid_world import GridWorld
from environment.scenario_generator import ScenarioGenerator
from agents.state_encoder import StateEncoder


# Generate a random environment.
generator = ScenarioGenerator(
    size=10,
    static_obstacle_count=12,
    moving_obstacle_count=2
)

scenario = generator.generate_scenario()

env = GridWorld(
    size=10,
    scenario=scenario
)

# Create state encoder.
encoder = StateEncoder()

# Encode initial state.
encoded_state = encoder.encode(env)

print("\nState Encoder Test")
print("-" * 40)

print("Robot Position:", env.robot)
print("Goal Position:", env.goal)

print("\nEncoded State:")
print(encoded_state)

print("\nNumber of State Values:", len(encoded_state))

# Perform one action.
env.step(3)

# Encode the new state.
new_encoded_state = encoder.encode(env)

print("\nAfter Robot Moves Right:")
print("Robot Position:", env.robot)
print("New Encoded State:", new_encoded_state)

print("\nState Encoder Test Completed!")
