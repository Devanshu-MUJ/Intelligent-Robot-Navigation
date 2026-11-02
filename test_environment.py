
from environment.grid_world import GridWorld

env = GridWorld()

state = env.reset()
print("Initial state:", state)
print("Robot position:", env.robot)
print("Goal position:", env.goal)

for step in range(5):
    action = 3  # Move right
    state, reward, terminated, truncated, info = env.step(action)

    print(
        f"Step {step + 1}: "
        f"Robot={env.robot}, "
        f"Reward={reward}, "
        f"State={state}"
    )

    if terminated or truncated:
        break

print("Environment test completed.")
