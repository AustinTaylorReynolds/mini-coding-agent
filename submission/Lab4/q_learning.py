"""
Q-Learning Vacuum Robot Integration Module.
Wires together existing components to train, evaluate, and compare policies.
"""

import random
from GridEnvironment import GridnEnvironment
from q_table import ACTIONS, init_q_table, best_actions
from rewards import step_reward, is_terminal, terminal_payoff
from action_selection import choose_action, decay_epsilon
from q_update import q_update
from MDP import MDP


LIVING_COSTS = [-0.04, -0.1, -2.0]
SEED = 42


def env_step(env, state, action):
    """
    Execute one step in the environment: sample transition, move, compute reward and terminal value.
    """
    transitions = env.get_transition(action)
    probs = [t[0] for t in transitions]
    dirs = [t[1] for t in transitions]

    actual_dir = random.choices(dirs, weights=probs, k=1)[0]
    next_state = env.get_next_state(state, actual_dir)

    reward = step_reward(env, state, action, next_state)
    terminal_value = None
    if is_terminal(env, next_state):
        try:
            terminal_value = terminal_payoff(env, next_state)
        except KeyError:
            pass  # Treat as non-terminal if payoff key missing

    return reward, next_state, terminal_value


def train(env, episodes=5000, alpha=0.1, gamma=0.9, eps_start=1.0, eps_min=0.05,
          eps_decay=0.999, max_steps=100, seed=None):
    """
    Train the Q-table using model-free Q-learning with epsilon-greedy action selection.
    Returns the learned Q-table.
    """
    if seed is not None:
        random.seed(seed)

    q_table = init_q_table(env)
    epsilon = eps_start  # set once, so the decay below carries over from episode to episode

    for _ in range(episodes):
        state = env.start_state  # every episode begins at the start cell
        for _ in range(max_steps):  # step cap applies per episode, not to the whole run
            action = choose_action(q_table, state, epsilon)
            reward, next_state, terminal_value = env_step(env, state, action)

            q_update(
                q_table, state, action, reward, next_state, alpha, gamma, terminal_value
            )

            if is_terminal(env, next_state):
                break

            state = next_state

        epsilon = decay_epsilon(epsilon, eps_decay, eps_min)

    return q_table


def extract_policy(q_table, env):
    """
    Extract a deterministic policy from the Q-table by selecting the best action for each state.
    """
    policy = {}
    for row in range(env.rows):
        for col in range(env.cols):
            full_state = (row, col)
            if is_terminal(env, full_state):
                continue
            if full_state == env.obstacle or full_state in env.terminal_states:
                continue
            actions = best_actions(q_table, full_state)
            if actions:
                policy[full_state] = actions[0]
    return policy


def print_policy(policy, env):
    """
    Print the policy as a 4x4 grid with symbols for directions, obstacle, and terminals.
    """
    rows = env.rows
    cols = env.cols
    symbol_map = {'N': '^', 'S': 'v', 'E': '>', 'W': '<'}

    for r in range(rows):
        row_str = ""
        for c in range(cols):
            state = (r, c)
            if is_terminal(env, state):
                row_str += "T"
            elif env.obstacle and state == env.obstacle:
                row_str += "X"
            elif state in policy:
                action = policy[state]
                row_str += symbol_map.get(action, ".")
            else:
                row_str += "."
        print(row_str)


def simulate(policy, env, max_steps=100):
    """
    Simulate the agent following the policy and report the trajectory and outcome.
    Returns the list of visited states.
    """
    state = env.start_state
    visited = [state]

    for step in range(max_steps):
        action = policy.get(state, 'N')  # Default to North if unknown
        reward, next_state, terminal_value = env_step(env, state, action)

        print(f"Step {step}: Intended action: {action}, Resulting state: {next_state}")
        visited.append(next_state)
        state = next_state

        if is_terminal(env, state):
            break

    # Determine outcome
    if is_terminal(env, state):
        try:
            payoff = terminal_payoff(env, state)
            if payoff > 0:
                print(f"Reached recharge spot at {state} with payoff {payoff}.")
            else:
                print(f"Reached trap at {state} with payoff {payoff}.")
        except KeyError:
            print(f"Reached unknown terminal at {state}.")
    else:
        print(f"Stopped at max_steps ({max_steps}) at state {state}.")

    return visited


def compare_policies(q_policy, env, gamma):
    """
    Compare the learned Q-policy with the Value Iteration policy and report differences.
    This is an evaluation-only function; the learner never uses T and R directly.
    """
    mdp = MDP(env, discount=gamma)
    mdp.value_iteration()
    vi_policy = mdp.extract_policy()

    diff_states = []
    for state in q_policy:
        if state not in vi_policy:
            continue
        if q_policy[state] != vi_policy[state]:
            diff_states.append(state)

    print(f"States differing from value iteration: {diff_states}")
    return diff_states


if __name__ == "__main__":
    for cost in LIVING_COSTS:
        env = GridnEnvironment(living_cost=cost)
        q_table = train(env, seed=SEED)
        policy = extract_policy(q_table, env)
        print_policy(policy, env)
        compare_policies(policy, env, 0.9)
        simulate(policy, env)