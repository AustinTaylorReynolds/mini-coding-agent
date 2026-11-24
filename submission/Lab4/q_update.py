"""Q-learning update rule module."""
import math


def q_update(q_table, state, action, reward, next_state, alpha, gamma, terminal_value=None) -> float:
    """Update a single Q-value in the table using the Bellman equation."""
    # Calculate future value: use terminal_value if provided, else max Q of next state
    if terminal_value is not None:
        future_value = terminal_value
    else:
        # Read next_state's entries; never write to them
        future_value = max(q_table[next_state].values())

    # Compute target for the update
    target = reward + gamma * future_value

    # Get current Q-value and update in place
    old_q = q_table[state][action]
    new_q = old_q + alpha * (target - old_q)
    q_table[state][action] = new_q

    return new_q


if __name__ == "__main__":
    def fresh_table():
        """Build a new hand-built table so every case starts from Q(s,a) = 0.0."""
        return {
            (0, 0): {"up": 0.0, "down": 0.0, "left": 0.0, "right": 0.0},
            (1, 1): {"up": 0.0, "down": 0.0, "left": 0.0, "right": 0.0},
        }

    state, action, next_state = (0, 0), "up", (1, 1)
    reward, alpha, gamma = -0.04, 0.1, 0.9

    # Case 1: non-terminal next state with best Q = 0.0 -> 0.0 + 0.1 * (-0.04 + 0.9 * 0.0 - 0.0) = -0.004
    q_table = fresh_table()
    new_q = q_update(q_table, state, action, reward, next_state, alpha, gamma)
    assert math.isclose(new_q, -0.004), f"Test 1 failed: {new_q} != -0.004"

    # Case 2: terminal_value = 10 -> 0.0 + 0.1 * (-0.04 + 0.9 * 10 - 0.0) = 0.896
    q_table = fresh_table()
    new_q = q_update(q_table, state, action, reward, next_state, alpha, gamma, 10)
    assert math.isclose(new_q, 0.896), f"Test 2 failed: {new_q} != 0.896"

    # The next state's entries must be unchanged afterwards
    assert q_table[next_state] == {"up": 0.0, "down": 0.0, "left": 0.0, "right": 0.0}, "Next state was modified"

    # Case 3: terminal_value works even when next_state is missing from the table
    q_table = fresh_table()
    del q_table[next_state]
    new_q = q_update(q_table, state, action, reward, next_state, alpha, gamma, 10)
    assert math.isclose(new_q, 0.896), f"Test 3 failed: {new_q} != 0.896"

    # The return value equals the stored value
    assert new_q == q_table[state][action], "Return value does not match stored Q-value"

    print("q_update: all checks passed")