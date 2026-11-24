"""Epsilon-greedy action selection module."""

import random


def choose_action(q_table, state, epsilon):
    """Select an action using epsilon-greedy policy."""
    actions = q_table[state]
    if not actions:
        return None

    # With probability epsilon, choose a random action (exploration)
    if random.random() < epsilon:
        return random.choice(list(actions.keys()))

    # Otherwise, choose the greedy action (exploitation)
    max_q = max(q_table[state][a] for a in actions)
    best_actions = [a for a in actions if q_table[state][a] == max_q]
    return random.choice(best_actions)


def decay_epsilon(epsilon, eps_decay, eps_min):
    """Decay epsilon towards a minimum value."""
    return max(eps_min, epsilon * eps_decay)


if __name__ == "__main__":
    # Test 1: epsilon=0, one clearly best action -> always returns it
    q_table_1 = {
        (0, 0): {"up": 1.0, "down": 0.5, "left": 0.3, "right": 0.2},
    }
    seed = 42
    random.seed(seed)
    count = 0
    for _ in range(1000):
        action = choose_action(q_table_1, (0, 0), epsilon=0)
        if action == "up":
            count += 1
    assert count == 1000, f"Test 1 failed: expected 1000 'up', got {count}"

    # Test 2: epsilon=1 -> returns all four actions uniformly
    q_table_2 = {
        (0, 0): {"up": 1.0, "down": 0.5, "left": 0.3, "right": 0.2},
    }
    random.seed(seed)
    seen_actions = set()
    for _ in range(1000):
        action = choose_action(q_table_2, (0, 0), epsilon=1)
        seen_actions.add(action)
    assert len(seen_actions) == 4, f"Test 2 failed: expected 4 unique actions, got {len(seen_actions)}"

    # Test 3: all-zero table with epsilon=0 -> random tie-break
    q_table_3 = {
        (0, 0): {"up": 0.0, "down": 0.0, "left": 0.0, "right": 0.0},
    }
    random.seed(seed)
    seen_actions = set()
    for _ in range(1000):
        action = choose_action(q_table_3, (0, 0), epsilon=0)
        seen_actions.add(action)
    assert len(seen_actions) > 1, f"Test 3 failed: expected >1 unique actions, got {len(seen_actions)}"

    # Test 4: decay_epsilon(1.0, 0.5, 0.05) -> 0.5
    result = decay_epsilon(1.0, 0.5, 0.05)
    assert result == 0.5, f"Test 4a failed: expected 0.5, got {result}"

    # Test 5: decay_epsilon(0.06, 0.5, 0.05) -> 0.05
    result = decay_epsilon(0.06, 0.5, 0.05)
    assert result == 0.05, f"Test 4b failed: expected 0.05, got {result}"

    print("action_selection: all checks passed")
