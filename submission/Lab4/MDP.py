import random
from GridEnvironment import GridnEnvironment

class MDP:
    def __init__(self, env, discount=0.9, epsilon=0.001):
        self.env = env
        self.discount = discount
        self.epsilon = epsilon
        self.values = {(r,c): 0.0 for r in range(env.rows) for c in range(env.cols)}

        # Initialize terminal states with their absolute rewards
        for state, reward in self.env.terminal_states.items():
            self.values[state] = reward

    def value_iteration(self):
        directions = ['N', 'S', 'E', 'W']

        while True:
            delta = 0
            new_values = self.values.copy()

            for r in range(self.env.rows):
                for c in range(self.env.cols):
                    state = (r, c)

                    if state in self.env.terminal_states or state == self.env.obstacle:
                        continue

                    action_values = []
                    for action in directions:
                        transition = self.env.get_transition(action)
                        expected_utility = 0

                        for prob, actual_dir in transition:
                            next_state = self.env.get_next_state(state, actual_dir)
                            expected_utility += prob * self.values[next_state]

                        # Apply the Bellman Equation
                        action_values.append(self.env.living_cost + self.discount * expected_utility)

                    best_value = max(action_values)
                    delta = max(delta, abs(best_value - self.values[state]))
                    new_values[state] = best_value

            self.values = new_values
            if delta < self.epsilon:
                break

    def extract_policy(self):
        policy = {}
        directions = ['N', 'S', 'E', 'W']

        for r in range(self.env.rows):
            for c in range (self.env.cols):
                state = (r, c)
                if state in self.env.terminal_states or state == self.env.obstacle:
                    continue

                best_action = None
                max_utility = float('-inf')

                for action in directions:
                    transition = self.env.get_transition(action)
                    expected_utility = 0

                    for prob, actual_dir in transition:
                        next_state = self.env.get_next_state(state, actual_dir)
                        expected_utility += prob * self.values[next_state]

                    if expected_utility > max_utility:
                        max_utility = expected_utility
                        best_action = action
                policy[state] = best_action
        return policy

    def simulated_path(self, max_steps=100):
        policy = self.extract_policy()
        current_state = self.env.start_state
        path_log = [current_state]

        print(f"\n--- Starting Simulation ---")
        print(f"Living Cost: {self.env.living_cost}")
        print(f"Initial State: {current_state}")

        step = 1
        while current_state not in self.env.terminal_states and step <= max_steps:
            intended_action = policy[current_state]
            transition = self.env.get_transition(intended_action)

            # Unpack probabilities and directions to simulate the 80/10/10 slip
            probs = [t[0] for t in transition]
            dirs = [t[1] for t in transition]
            actual_dir = random.choices(dirs, weights=probs, k=1)[0]

            next_state = self.env.get_next_state(current_state, actual_dir)
            print(f"Step {step} | Intended: {intended_action} | Actual: {actual_dir} | State: {next_state}")

            current_state = next_state
            path_log.append(current_state)
            step += 1

        if current_state == (3, 3):
            print(f"Success: Reached recharge spot at {current_state}")
        elif current_state == (3, 1):
            print(f"Failed: Hit the trap at {current_state}")

        return path_log

if __name__ == "__main__":
    test_costs = [-0.04, -0.1, -2.0]

    for cost in test_costs:
        env = GridnEnvironment(living_cost=cost)
        agent = MDP(env, discount=0.9)
        agent.value_iteration()
        agent.simulated_path()
