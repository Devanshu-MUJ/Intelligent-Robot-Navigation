
import pygame

from environment.grid_world import GridWorld
from agents.q_learning import QLearningAgent
from visualization.renderer import Renderer


def run_demo():

    env = GridWorld()
    agent = QLearningAgent()

    # Load the trained Q-table.
    agent.load("models/q_table.pkl")

    renderer = Renderer(size=env.size)

    state = env.reset()

    total_reward = 0
    step = 0

    running = True
    finished = False

    print("Starting robot navigation demo...")

    while running:

        # Handle window events.
        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                running = False

        if not running:
            break

        if not finished:

            # Select the best learned action.
            action = agent.choose_action(
                state,
                training=False
            )

            # Move the robot.
            next_state, reward, terminated, truncated, info = (
                env.step(action)
            )

            state = next_state
            total_reward += reward
            step += 1

            if terminated or truncated:
                finished = True

                if info["success"]:
                    print("Robot reached the goal!")

                elif info["collision"]:
                    print("Robot collided with an obstacle.")

                else:
                    print("Maximum steps reached.")

                print("Total reward:", total_reward)
                print("Total steps:", step)

        renderer.draw(
            env,
            episode=1,
            step=step,
            reward=total_reward
        )

        # Animation speed.
        renderer.clock.tick(4)

        if finished:
            pygame.time.wait(2000)
            running = False

    renderer.close()


if __name__ == "__main__":
    run_demo()
