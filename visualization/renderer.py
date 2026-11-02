
import pygame


class Renderer:

    def __init__(self, size=10, cell_size=60):

        pygame.init()

        self.size = size
        self.cell_size = cell_size

        self.margin = 30
        self.top = 80

        width = size * cell_size + 2 * self.margin
        height = size * cell_size + self.top + 30

        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Intelligent Robot Navigation")

        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 22)
        self.small_font = pygame.font.SysFont("Arial", 16)

        self.colors = {
            "background": (245, 247, 250),
            "grid": (190, 195, 205),
            "robot": (40, 110, 220),
            "goal": (40, 180, 90),
            "static": (70, 75, 85),
            "moving": (230, 135, 35),
            "text": (30, 35, 45)
        }

    def draw(self, env, episode=1, step=0, reward=0):

        self.screen.fill(self.colors["background"])

        # Display information.
        title = self.font.render(
            "Robot Navigation - Q-Learning",
            True,
            self.colors["text"]
        )

        self.screen.blit(title, (self.margin, 20))

        info = self.small_font.render(
            f"Episode: {episode}   Step: {step}   Reward: {reward}",
            True,
            self.colors["text"]
        )

        self.screen.blit(info, (self.margin, 52))

        # Draw grid cells.
        for row in range(self.size):
            for col in range(self.size):

                x = self.margin + col * self.cell_size
                y = self.top + row * self.cell_size

                rect = pygame.Rect(
                    x, y, self.cell_size, self.cell_size
                )

                pygame.draw.rect(
                    self.screen,
                    (255, 255, 255),
                    rect
                )

                pygame.draw.rect(
                    self.screen,
                    self.colors["grid"],
                    rect,
                    1
                )

        # Draw static obstacles.
        for row, col in env.static_obstacles:

            self.draw_cell(
                row, col, self.colors["static"]
            )

        # Draw moving obstacles.
        for row, col in env.get_moving_positions():

            self.draw_cell(
                row, col, self.colors["moving"]
            )

        # Draw goal.
        self.draw_cell(
            env.goal[0],
            env.goal[1],
            self.colors["goal"]
        )

        # Draw robot.
        self.draw_cell(
            env.robot[0],
            env.robot[1],
            self.colors["robot"]
        )

        pygame.display.flip()

    def draw_cell(self, row, col, color):

        x = self.margin + col * self.cell_size
        y = self.top + row * self.cell_size

        rect = pygame.Rect(
            x + 4,
            y + 4,
            self.cell_size - 8,
            self.cell_size - 8
        )

        pygame.draw.rect(
            self.screen,
            color,
            rect,
            border_radius=8
        )

    def close(self):
        pygame.quit()
