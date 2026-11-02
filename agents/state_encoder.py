
class StateEncoder:
    """
    Converts the environment into a compact local state.

    State contains:
    1. Vertical direction towards the goal.
    2. Horizontal direction towards the goal.
    3. Information about the eight surrounding cells.

    Cell encoding:
    0 = Free
    1 = Static obstacle or outside grid
    2 = Moving obstacle travelling left
    3 = Moving obstacle travelling right
    """

    def encode(self, env):

        row, col = env.robot
        goal_row, goal_col = env.goal

        # Direction towards the goal.
        vertical_direction = (
            int(goal_row > row) - int(goal_row < row)
        )

        horizontal_direction = (
            int(goal_col > col) - int(goal_col < col)
        )

        # Eight surrounding cells, clockwise from top-left.
        directions = [
            (-1, -1),  # Top-left
            (-1, 0),   # Top
            (-1, 1),   # Top-right
            (0, 1),    # Right
            (1, 1),    # Bottom-right
            (1, 0),    # Bottom
            (1, -1),   # Bottom-left
            (0, -1)    # Left
        ]

        # Map moving obstacle positions to their directions.
        moving_info = {}

        for obstacle in env.moving_obstacles:

            position = (
                obstacle["row"],
                obstacle["col"]
            )

            moving_info[position] = obstacle["direction"]

        surroundings = []

        for dr, dc in directions:

            nearby_row = row + dr
            nearby_col = col + dc

            nearby_cell = (
                nearby_row,
                nearby_col
            )

            # Outside the grid.
            if not (
                0 <= nearby_row < env.size
                and 0 <= nearby_col < env.size
            ):

                surroundings.append(1)

            # Static obstacle.
            elif nearby_cell in env.static_obstacles:

                surroundings.append(1)

            # Moving obstacle.
            elif nearby_cell in moving_info:

                direction = moving_info[nearby_cell]

                if direction == -1:
                    surroundings.append(2)
                else:
                    surroundings.append(3)

            # Free cell.
            else:

                surroundings.append(0)

        # Final encoded state.
        encoded_state = (
            vertical_direction,
            horizontal_direction,
            *surroundings
        )

        return encoded_state
