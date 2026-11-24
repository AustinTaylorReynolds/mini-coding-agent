"""Reward logic module for the environment."""


def step_reward(env, state, action, next_state) -> float:
    """Return the living cost for every step, including terminal steps."""
    return env.living_cost


def is_terminal(env, state) -> bool:
    """Check if the given state is a terminal state."""
    return state in env.terminal_states


def terminal_payoff(env, state) -> float:
    """Return the payoff for a terminal state.

    Raises KeyError if the state is not terminal.
    """
    return env.terminal_states[state]


if __name__ == "__main__":
    from types import SimpleNamespace

    # Stand-in environment object
    env = SimpleNamespace(
        living_cost=-0.04,
        terminal_states={(3, 1): -1, (3, 3): 10}
    )

    # Test step_reward: should return living_cost regardless of arguments
    assert step_reward(env, (0, 0), None, (0, 0)) == -0.04
    assert step_reward(env, (3, 1), None, (3, 1)) == -0.04

    # Test is_terminal: True for terminal states, False otherwise
    assert is_terminal(env, (3, 1)) is True
    assert is_terminal(env, (3, 3)) is True
    assert is_terminal(env, (0, 0)) is False

    # Test terminal_payoff: returns correct payoff or raises KeyError
    assert terminal_payoff(env, (3, 1)) == -1
    assert terminal_payoff(env, (3, 3)) == 10

    try:
        terminal_payoff(env, (0, 0))
        assert False, "Expected KeyError for non-terminal state"
    except KeyError:
        pass

    print("rewards: all checks passed")
