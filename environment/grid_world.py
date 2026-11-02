from agents.state_encoder import StateEncoder

class GridWorld:
    """
    Grid-based robot navigation environment.

    Actions:
    0 = Up
    1 = Down
    2 = Left
    3 = Right
    """

    def __init__(self, size=10, max_steps=150, scenario=None):

        self.size = size
        self.max_steps = max_steps

        # Use generated scenario if provided.
        if scenario is not None:

            if scenario["size"] != size:
                raise ValueError(
                    "Scenario size does not match GridWorld size."
                )

            self.start = scenario["start"]
            self.goal = scenario["goal"]

            self.static_obstacles = set(
                scenario["static_obstacles"]
            )

            self.initial_moving_obstacles = [
                obstacle.copy()
                for obstacle in scenario["moving_obstacles"]
            ]

        else:
            # Original default environment.
            self.start = (0, 0)
            self.goal = (size - 1, size - 1)

            self.static_obstacles = {
                (2, 2), (2, 3), (2, 4),
                (4, 5), (5, 5), (6, 5),
                (7, 2), (7, 3)
            }

            self.initial_moving_obstacles = [
                {
                    "row": 3,
                    "col": 1,
                    "direction": 1,
                    "min_col": 1,
                    "max_col": 5
                },
                {
                    "row": 6,
                    "col": 8,
                    "direction": -1,
                    "min_col": 6,
                    "max_col": 8
                }
            ]

        self.actions = {
            0: (-1, 0),
            1: (1, 0),
            2: (0, -1),
            3: (0, 1)
        }

        self.reset()
        self.state_encoder = StateEncoder()

    def get_encoded_state(self):
     return self.state_encoder.encode(self)

    def reset(self):

        self.robot = self.start
        self.steps = 0

        # Restore moving obstacles to their initial positions.
        self.moving_obstacles = [
            obstacle.copy()
            for obstacle in self.initial_moving_obstacles
        ]

        return self.get_state()

    def get_state(self):

        row, col = self.robot
        goal_row, goal_col = self.goal

        nearby = []

        # Check the four adjacent cells.
        for action in range(4):

            dr, dc = self.actions[action]
            cell = (row + dr, col + dc)

            blocked = (
                not (
                    0 <= cell[0] < self.size
                    and 0 <= cell[1] < self.size
                )
                or cell in self.static_obstacles
                or cell in self.get_moving_positions()
            )

            nearby.append(int(blocked))

        # Include moving obstacle positions and directions.
        obstacle_info = []

        for obstacle in self.moving_obstacles:

            obstacle_info.extend([
                obstacle["row"],
                obstacle["col"],
                obstacle["direction"]
            ])

        return (
            row,
            col,
            int(goal_row > row) - int(goal_row < row),
            int(goal_col > col) - int(goal_col < col),
            *nearby,
            *obstacle_info
        )

    def get_moving_positions(self):

        return {
            (obstacle["row"], obstacle["col"])
            for obstacle in self.moving_obstacles
        }

    
    def move_obstacles(self):

     occupied = {
         (obstacle["row"], obstacle["col"])
         for obstacle in self.moving_obstacles
     }

     for obstacle in self.moving_obstacles:

        current = (obstacle["row"], obstacle["col"])
        occupied.discard(current)

        next_col = obstacle["col"] + obstacle["direction"]
        next_position = (obstacle["row"], next_col)

        blocked = (
            next_col < obstacle["min_col"]
            or next_col > obstacle["max_col"]
            or next_position in self.static_obstacles
            or next_position == self.start
            or next_position == self.goal
            or next_position in occupied
        )

        if blocked:
            obstacle["direction"] *= -1

            next_col = obstacle["col"] + obstacle["direction"]
            next_position = (obstacle["row"], next_col)

            blocked = (
                next_col < obstacle["min_col"]
                or next_col > obstacle["max_col"]
                or next_position in self.static_obstacles
                or next_position == self.start
                or next_position == self.goal
                or next_position in occupied
            )

        if not blocked:
            obstacle["col"] = next_col

        occupied.add((obstacle["row"], obstacle["col"]))


    def step(self, action):

        if action not in self.actions:
            raise ValueError("Invalid action.")

        self.steps += 1

        dr, dc = self.actions[action]

        row, col = self.robot

        new_position = (
            row + dr,
            col + dc
        )

        reward = -1
        terminated = False
        truncated = False

        # Check boundaries and static obstacles.
        out_of_bounds = not (
            0 <= new_position[0] < self.size
            and 0 <= new_position[1] < self.size
        )

        if (
            out_of_bounds
            or new_position in self.static_obstacles
        ):

            reward = -5

        else:
            self.robot = new_position

        # Move dynamic obstacles.
        self.move_obstacles()

        # Check moving obstacle collision.
        if self.robot in self.get_moving_positions():

            reward = -20
            terminated = True

        # Check whether the robot reached the goal.
        elif self.robot == self.goal:

            reward = 100
            terminated = True

        # Check maximum step limit.
        if self.steps >= self.max_steps and not terminated:

            truncated = True

        info = {
            "success": self.robot == self.goal,
            "collision": terminated and reward == -20
        }

        return (
            self.get_state(),
            reward,
            terminated,
            truncated,
            info
        )
