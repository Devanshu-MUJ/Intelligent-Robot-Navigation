
import random
from collections import deque


class ScenarioGenerator:
    """
    Generates random GridWorld scenarios containing:
    - Static obstacles
    - Horizontally moving obstacles
    - A reachable goal
    """

    def __init__(
        self,
        size=10,
        static_obstacle_count=12,
        moving_obstacle_count=2
    ):
        self.size = size
        self.static_obstacle_count = static_obstacle_count
        self.moving_obstacle_count = moving_obstacle_count

        self.start = (0, 0)
        self.goal = (size - 1, size - 1)

    def has_path(self, obstacles):
        """Check whether the goal is reachable using BFS."""

        queue = deque([self.start])
        visited = {self.start}

        directions = [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1)
        ]

        while queue:

            row, col = queue.popleft()

            if (row, col) == self.goal:
                return True

            for dr, dc in directions:

                new_position = (row + dr, col + dc)

                new_row, new_col = new_position

                inside_grid = (
                    0 <= new_row < self.size
                    and 0 <= new_col < self.size
                )

                if (
                    inside_grid
                    and new_position not in obstacles
                    and new_position not in visited
                ):
                    visited.add(new_position)
                    queue.append(new_position)

        return False

    def generate_scenario(self, max_attempts=1000):
        """Generate a valid random scenario."""

        for attempt in range(max_attempts):

            # Generate static obstacles.
            possible_cells = [
                (row, col)
                for row in range(1, self.size - 1)
                for col in range(1, self.size - 1)
            ]

            if self.static_obstacle_count > len(possible_cells):
                raise ValueError(
                    "Too many static obstacles for this grid."
                )

            static_obstacles = set(
                random.sample(
                    possible_cells,
                    self.static_obstacle_count
                )
            )

            # Reject layouts without a static route.
            if not self.has_path(static_obstacles):
                continue

            # Generate moving obstacles.
            moving_obstacles = []
            used_rows = set()
            valid_moving_layout = True

            for _ in range(self.moving_obstacle_count):

                possible_rows = [
                    row
                    for row in range(1, self.size - 1)
                    if row not in used_rows
                ]

                random.shuffle(possible_rows)

                obstacle_created = False

                for row in possible_rows:

                    min_col = random.randint(1, self.size - 4)
                    max_col = random.randint(
                        min_col + 1,
                        self.size - 2
                    )

                    # Ensure the entire moving lane is clear.
                    lane = {
                        (row, col)
                        for col in range(min_col, max_col + 1)
                    }

                    if lane.isdisjoint(static_obstacles):

                        obstacle = {
                            "row": row,
                            "col": random.randint(
                                min_col, max_col
                            ),
                            "direction": random.choice([-1, 1]),
                            "min_col": min_col,
                            "max_col": max_col
                        }

                        moving_obstacles.append(obstacle)
                        used_rows.add(row)

                        obstacle_created = True
                        break

                if not obstacle_created:
                    valid_moving_layout = False
                    break

            if not valid_moving_layout:
                continue

            return {
                "size": self.size,
                "start": self.start,
                "goal": self.goal,
                "static_obstacles": static_obstacles,
                "moving_obstacles": moving_obstacles
            }

        raise RuntimeError(
            "Could not generate a valid scenario. "
            "Try reducing the obstacle counts."
        )


# Test the scenario generator.
if __name__ == "__main__":

    generator = ScenarioGenerator(
        size=10,
        static_obstacle_count=12,
        moving_obstacle_count=2
    )

    scenario = generator.generate_scenario()

    print("\nScenario Generated Successfully!")
    print("-" * 40)

    print("Grid Size:", scenario["size"])
    print("Start:", scenario["start"])
    print("Goal:", scenario["goal"])

    print("\nStatic Obstacles:")
    print(sorted(scenario["static_obstacles"]))

    print("\nMoving Obstacles:")
    for obstacle in scenario["moving_obstacles"]:
        print(obstacle)

    print("\nStatic Path Available:",
          generator.has_path(scenario["static_obstacles"]))
