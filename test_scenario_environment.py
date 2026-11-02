
from environment.grid_world import GridWorld
from environment.scenario_generator import ScenarioGenerator


# Generate a random scenario.
generator = ScenarioGenerator(
    size=10,
    static_obstacle_count=12,
    moving_obstacle_count=2
)

scenario = generator.generate_scenario()

# Create GridWorld using the generated scenario.
env = GridWorld(
    size=10,
    scenario=scenario
)

print("\nGenerated Environment Test")
print("-" * 40)

print("Robot Start:", env.start)
print("Goal:", env.goal)

print("Static Obstacles:", env.static_obstacles)
print("Moving Obstacles:", env.get_moving_positions())

print("\nInitial State:", env.reset())

# Perform one action: move right.
next_state, reward, terminated, truncated, info = env.step(3)

print("\nAfter One Action")
print("Robot Position:", env.robot)
print("Reward:", reward)
print("Next State:", next_state)
print("Info:", info)

print("\nScenario Environment Test Completed!")
