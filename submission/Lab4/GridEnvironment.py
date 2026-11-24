class GridnEnvironment:
    def __init__(self, living_cost=0.0):
        self.living_cost = living_cost
        self.rows = 4
        self.cols = 4
        self.start_state = (0, 3)
        self.obstacle = (1, 1)
        self.terminal_states = {(3, 1): -1, (3, 3): 10}

        # 4x4 Grid Layout for Lab 3
        self.grid = [
            [0, 0, 0, "start"],
            [0, "X", 0, 0],
            [0, 0, 0, 0],
            [0, -1, 0, 10]
        ]

    def get_transition(self, intended_action):
        movements = {
            'N': [(0.8, 'N'), (0.1, 'E'), (0.1, 'W')],
            'S': [(0.8, 'S'), (0.1, 'E'), (0.1, 'W')],
            'E': [(0.8, 'E'), (0.1, 'N'), (0.1, 'S')],
            'W': [(0.8, 'W'), (0.1, 'N'), (0.1, 'S')]
        }
        return movements.get(intended_action, [])

    def get_next_state(self, current_state, direction):
        r, c = current_state

        if direction == 'N': r -= 1
        elif direction == 'S': r += 1
        elif direction == 'E': c += 1
        elif direction == 'W': c -= 1

        # Agent remains in current state if hitting a boundary of the obstacle
        if r<0 or r>= self.rows or c < 0 or c >= self.cols or (r, c) == self.obstacle:
            return current_state
        return (r, c)