
import pygame

from environment.grid_world import GridWorld
from agents.q_learning import QLearningAgent


# Grid settings
CELL_SIZE = 60
GRID_SIZE = 10
GRID_WIDTH = CELL_SIZE * GRID_SIZE

WINDOW_WIDTH = 700
WINDOW_HEIGHT = GRID_WIDTH + 80

# Colours
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
BLUE = (50, 120, 220)
GREEN = (50, 180, 80)
YELLOW = (240, 180, 30)
RED = (220, 60, 60)
DARK_GRAY = (70, 70, 70)


class NavigationApp:

    def __init__(self):

        pygame.init()

        self.screen = pygame.display.set_mode(
            (WINDOW_WIDTH, WINDOW_HEIGHT)
        )

        pygame.display.set_caption(
            "Intelligent Robot Navigation Using Q-Learning"
        )

        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 24)
        self.title_font = pygame.font.SysFont(None, 30)

        self.start = (0, 0)
        self.goal = (9, 9)

        self.static_obstacles = set()
        self.moving_obstacles = []

        self.env = None
        self.agent = None

        self.navigation_running = False
        self.navigation_done = False

        self.state = None
        self.status = "Design your environment."

        self.last_step_time = 0
        self.step_interval = 350

        self.button_rect = pygame.Rect(
            10, GRID_WIDTH + 15, 180, 45
        )

        self.running = True

    def handle_static_click(self, position):

        x, y = position

        row = y // CELL_SIZE
        col = x // CELL_SIZE

        cell = (row, col)

        if cell == self.start or cell == self.goal:
            return

        moving_positions = {
            (obs["row"], obs["col"])
            for obs in self.moving_obstacles
        }

        if cell in moving_positions:
            return

        if cell in self.static_obstacles:
            self.static_obstacles.remove(cell)
        else:
            self.static_obstacles.add(cell)

    def add_moving_obstacle(self, position):

        x, y = position

        row = y // CELL_SIZE
        col = x // CELL_SIZE

        cell = (row, col)

        if cell == self.start or cell == self.goal:
            return

        if cell in self.static_obstacles:
            return

        for obs in self.moving_obstacles:
            if (obs["row"], obs["col"]) == cell:
                return

        self.moving_obstacles.append({
            "row": row,
            "col": col,
            "direction": 1,
            "min_col": 0,
            "max_col": GRID_SIZE - 1
        })

    def set_goal(self, position):

        x, y = position

        row = y // CELL_SIZE
        col = x // CELL_SIZE

        cell = (row, col)

        if cell == self.start:
            return

        if cell in self.static_obstacles:
            return

        for obs in self.moving_obstacles:
            if (obs["row"], obs["col"]) == cell:
                return

        self.goal = cell

    def start_navigation(self):

        if self.start in self.static_obstacles:
            self.status = "Invalid starting position."
            return

        scenario = {
            "size": GRID_SIZE,
            "start": self.start,
            "goal": self.goal,
            "static_obstacles": list(self.static_obstacles),
            "moving_obstacles": [
                obs.copy() for obs in self.moving_obstacles
            ]
        }

        try:
            self.env = GridWorld(
                size=GRID_SIZE,
                max_steps=200,
                scenario=scenario
            )

            self.agent = QLearningAgent()

            self.agent.load("models/q_table_v2.pkl")

            # Disable exploration and learning
            self.agent.epsilon = 0.0

            self.state = self.env.get_encoded_state()

            self.navigation_running = True
            self.navigation_done = False

            self.last_step_time = pygame.time.get_ticks()

            self.status = "Robot navigation started!"

        except (FileNotFoundError, ValueError) as error:
            self.status = f"Could not start: {error}"

    def reset_navigation(self):

        self.env = None
        self.agent = None
        self.state = None

        self.navigation_running = False
        self.navigation_done = False

        self.status = "Design your environment."

    def step_navigation(self):

        if not self.navigation_running:
            return

        action = self.agent.choose_action(
            self.state,
            training=False
        )

        _, reward, terminated, truncated, info = (
            self.env.step(action)
        )

        self.state = self.env.get_encoded_state()

        if terminated or truncated:

            self.navigation_running = False
            self.navigation_done = True

            if info["success"]:
                self.status = (
                    f"SUCCESS! Goal reached in "
                    f"{self.env.steps} steps."
                )

            elif info["collision"]:
                self.status = "Navigation ended: collision."

            else:
                self.status = "Navigation ended: step limit reached."

    def draw_grid(self):

        self.screen.fill(WHITE)

        # Use environment data while navigating
        if self.env is not None:
            static_obstacles = self.env.static_obstacles
            moving_obstacles = self.env.moving_obstacles
            robot = self.env.robot
        else:
            static_obstacles = self.static_obstacles
            moving_obstacles = self.moving_obstacles
            robot = None

        moving_positions = {
            (obs["row"], obs["col"])
            for obs in moving_obstacles
        }

        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):

                cell = (row, col)

                rect = pygame.Rect(
                    col * CELL_SIZE,
                    row * CELL_SIZE,
                    CELL_SIZE,
                    CELL_SIZE
                )

                if cell in static_obstacles:
                    pygame.draw.rect(
                        self.screen, BLACK,
                        rect.inflate(-4, -4)
                    )

                elif cell in moving_positions:
                    pygame.draw.rect(
                        self.screen, YELLOW,
                        rect.inflate(-4, -4)
                    )

                elif cell == self.start:
                    pygame.draw.rect(
                        self.screen, BLUE,
                        rect.inflate(-4, -4)
                    )

                elif cell == self.goal:
                    pygame.draw.rect(
                        self.screen, GREEN,
                        rect.inflate(-4, -4)
                    )

                pygame.draw.rect(
                    self.screen, GRAY, rect, 1
                )

        # Draw robot over the grid
        if robot is not None:

            row, col = robot

            robot_rect = pygame.Rect(
                col * CELL_SIZE + 8,
                row * CELL_SIZE + 8,
                CELL_SIZE - 16,
                CELL_SIZE - 16
            )

            pygame.draw.rect(
                self.screen,
                RED,
                robot_rect,
                border_radius=10
            )

        # Start / reset button
        button_text = (
            "RESET / EDIT"
            if self.env is not None
            else "START NAVIGATION"
        )

        pygame.draw.rect(
            self.screen,
            DARK_GRAY,
            self.button_rect,
            border_radius=6
        )

        text_surface = self.font.render(
            button_text, True, WHITE
        )

        self.screen.blit(
            text_surface,
            (self.button_rect.x + 10,
             self.button_rect.y + 13)
        )

        status_surface = self.font.render(
            self.status[:65], True, BLACK
        )

        self.screen.blit(
            status_surface,
            (210, GRID_WIDTH + 27)
        )

        pygame.display.flip()

    def run(self):

        while self.running:

            for event in pygame.event.get():

                if event.type == pygame.QUIT:
                    self.running = False

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False

                elif event.type == pygame.MOUSEBUTTONDOWN:

                    x, y = event.pos

                    if event.button == 1:

                        if self.button_rect.collidepoint(event.pos):

                            if self.env is None:
                                self.start_navigation()
                            else:
                                self.reset_navigation()

                        elif (
                            self.env is None
                            and y < GRID_WIDTH
                        ):
                            self.handle_static_click(event.pos)

                    elif (
                        self.env is None
                        and y < GRID_WIDTH
                    ):

                        if event.button == 3:
                            self.add_moving_obstacle(event.pos)

                        elif event.button == 2:
                            self.set_goal(event.pos)

            current_time = pygame.time.get_ticks()

            if (
                self.navigation_running
                and current_time - self.last_step_time
                >= self.step_interval
            ):

                self.step_navigation()
                self.last_step_time = current_time

            self.draw_grid()
            self.clock.tick(60)

        pygame.quit()


if __name__ == "__main__":

    app = NavigationApp()
    app.run()
