"""Q-table module for a 4x4 grid environment."""

from typing import Dict, List, Tuple


ACTIONS = ['N', 'S', 'E', 'W']


def init_q_table(env) -> Dict[Tuple[int, int], Dict[str, float]]:
    """Initialize Q-table with zero values for all non-obstacle states.

    Creates a nested dictionary where each valid (row, col) state maps
    to a dictionary of actions with value 0.0.
    """
    q_table = {}
    for row in range(env.rows):
        for col in range(env.cols):
            if (row, col) != env.obstacle:
                q_table[(row, col)] = {action: 0.0 for action in ACTIONS}
    return q_table


def best_actions(q_table: Dict[Tuple[int, int], Dict[str, float]], state: Tuple[int, int]) -> List[str]:
    """Return list of actions with maximum Q-value for a given state.

    Returns all actions that achieve the maximum Q-value in ACTIONS order.
    """
    q_values = q_table[state]
    max_q = max(q_values.values())
    return [action for action, value in q_values.items() if value == max_q]


if __name__ == "__main__":
    from types import SimpleNamespace

    env = SimpleNamespace(rows=4, cols=4, obstacle=(1, 1))
    q_table = init_q_table(env)

    # Check number of states (16 total - 1 obstacle = 15)
    assert len(q_table) == 15, "Expected 15 states"

    # Check all Q-values are 0.0
    for state in q_table:
        for action in ACTIONS:
            assert q_table[state][action] == 0.0, f"Q-value should be 0.0 for {state}"

    # Check best_actions on fresh table returns all four actions
    fresh_state = (0, 0)
    fresh_actions = best_actions(q_table, fresh_state)
    assert fresh_actions == ACTIONS, f"Expected {ACTIONS}, got {fresh_actions}"

    # Modify one Q-value and check best_actions
    q_table[fresh_state]['N'] = 1.0
    modified_actions = best_actions(q_table, fresh_state)
    assert modified_actions == ['N'], f"Expected ['N'], got {modified_actions}"

    print("q_table: all checks passed")
