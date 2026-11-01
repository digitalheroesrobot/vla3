
"""
======================================================================
RL COURSE — PHASE 2.2
POLICY IMPROVEMENT AND GREEDY POLICIES
======================================================================

LESSON ROADMAP
--------------

STEP 1 — Evaluate the Current Policy
Purpose:
    Calculate V_pi(s) for the current policy.

What you learn:
    Policy evaluation measures how well the current strategy performs.

Implementation:
    Implement iterative policy evaluation using the Bellman expectation
    equation.

STEP 2 — Calculate Action Values
Purpose:
    Determine the expected return of each possible action.

What you learn:
    Q_pi(s,a) evaluates taking action a first and following pi afterward.

Implementation:
    Calculate Q-values using the transition model and V_pi.

STEP 3 — Greedy Policy Improvement
Purpose:
    Find actions with the highest Q-values.

What you learn:
    pi'(s) = argmax_a Q_pi(s,a)

    The policy improvement theorem guarantees that the improved
    policy is no worse than the original policy under standard
    finite discounted-MDP assumptions.

Implementation:
    Construct deterministic greedy policies.

STEP 4 — Stochastic Policy Improvement
Purpose:
    Handle multiple actions with equal or nearly equal Q-values.

What you learn:
    A policy may distribute probability among all maximizing actions.

Implementation:
    Implement tie-aware greedy and epsilon-greedy policies.

STEP 5 — Policy Comparison
Purpose:
    Measure whether policy improvement actually helps.

What you learn:
    V_pi'(s) >= V_pi(s) for every state under exact greedy improvement.

Implementation:
    Evaluate old and improved policies independently.

STEP 6 — Repeated Policy Improvement
Purpose:
    Understand how repeated evaluation and improvement lead toward
    an optimal policy.

What you learn:
    Policy Iteration alternates:
        Policy Evaluation
        Policy Improvement

Implementation:
    Implement a complete policy iteration algorithm.

STEP 7 — Numerical Verification
Purpose:
    Verify policy improvement mathematically and numerically.

What you learn:
    Improved policies should not reduce state values.
    A stable policy is greedy with respect to its own Q-values.

Implementation:
    Unit tests, Bellman residuals, and policy stability checks.

STEP 8 — Visualization
Purpose:
    See how the agent's decisions and state values change.

What you learn:
    Policy arrows represent selected actions.
    Value heatmaps represent expected discounted returns.

Implementation:
    Plot policy maps, value maps, Q-values, and improvement history.

LEARNING CYCLE
--------------
Understand -> Implement -> Visualize -> Verify -> Experiment

DEPENDENCIES
------------
pip install numpy matplotlib

RUN
---
python phase2/policy_improvement.py

======================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt


# ======================================================================
# STEP 1. MDP STRUCTURE
# ======================================================================

@dataclass
class MDP:

    P: np.ndarray
    R: np.ndarray
    policy: np.ndarray
    terminal: np.ndarray
    state_names: List[str]
    action_names: List[str]
    grid_shape: Tuple[int, int]
    obstacle_states: Tuple[int, ...]


# ======================================================================
# STEP 2. BUILD GRIDWORLD
# ======================================================================

def build_gridworld() -> MDP:

    rows = 4
    cols = 4

    num_states = rows * cols
    num_actions = 4

    directions = [
        (-1, 0),  # Up
        (0, 1),   # Right
        (1, 0),   # Down
        (0, -1),  # Left
    ]

    action_names = [
        "Up",
        "Right",
        "Down",
        "Left"
    ]

    goal = (3, 3)
    obstacle = (1, 1)

    goal_state = goal[0] * cols + goal[1]
    obstacle_state = obstacle[0] * cols + obstacle[1]

    terminal = np.zeros(num_states, dtype=bool)

    terminal[goal_state] = True
    terminal[obstacle_state] = True

    P = np.zeros(
        (num_actions, num_states, num_states)
    )

    R = np.zeros_like(P)

    def move(state, action):

        row, col = divmod(state, cols)

        dr, dc = directions[action]

        nr = row + dr
        nc = col + dc

        if not (0 <= nr < rows and 0 <= nc < cols):
            return state

        if (nr, nc) == obstacle:
            return state

        return nr * cols + nc

    for state in range(num_states):

        if terminal[state]:

            for action in range(num_actions):

                P[action, state, state] = 1.0
                R[action, state, state] = 0.0

            continue

        for action in range(num_actions):

            outcomes = [
                (action, 0.8),
                ((action - 1) % 4, 0.1),
                ((action + 1) % 4, 0.1),
            ]

            for actual_action, probability in outcomes:

                destination = move(
                    state,
                    actual_action
                )

                P[action, state, destination] += probability

                R[action, state, destination] = (
                    1.0 if destination == goal_state else -0.04
                )

    # Start with a uniform random policy.
    policy = np.full(
        (num_states, num_actions),
        1.0 / num_actions
    )

    return MDP(
        P=P,
        R=R,
        policy=policy,
        terminal=terminal,
        state_names=[
            f"({r},{c})"
            for r in range(rows)
            for c in range(cols)
        ],
        action_names=action_names,
        grid_shape=(rows, cols),
        obstacle_states=(obstacle_state,)
    )


# ======================================================================
# STEP 3. POLICY VALIDATION
# ======================================================================

def validate_policy(mdp: MDP, policy: np.ndarray):

    if policy.shape != (
        mdp.P.shape[1],
        mdp.P.shape[0]
    ):
        raise ValueError("Invalid policy shape.")

    if np.any(policy < -1e-12):
        raise ValueError("Negative policy probability.")

    if not np.allclose(
        np.sum(policy, axis=1),
        1.0
    ):
        raise ValueError(
            "Policy probabilities must sum to one."
        )


# ======================================================================
# STEP 4. POLICY EVALUATION
# ======================================================================

def policy_model(
    mdp: MDP,
    policy: np.ndarray
):

    num_states = mdp.P.shape[1]
    num_actions = mdp.P.shape[0]

    P_pi = np.zeros(
        (num_states, num_states)
    )

    r_pi = np.zeros(num_states)

    for state in range(num_states):

        if mdp.terminal[state]:
            continue

        for action in range(num_actions):

            probability = policy[state, action]

            P_pi[state] += (
                probability * mdp.P[action, state]
            )

            r_pi[state] += (
                probability
                * np.sum(
                    mdp.P[action, state]
                    * mdp.R[action, state]
                )
            )

    return P_pi, r_pi


def evaluate_policy(
    mdp: MDP,
    policy: np.ndarray,
    gamma: float = 0.90,
    theta: float = 1e-10,
    max_iterations: int = 10000
):

    validate_policy(mdp, policy)

    if not 0 <= gamma < 1:
        raise ValueError("gamma must be in [0, 1).")

    P_pi, r_pi = policy_model(
        mdp,
        policy
    )

    values = np.zeros(
        mdp.P.shape[1]
    )

    history = []

    for iteration in range(
        1,
        max_iterations + 1
    ):

        new_values = (
            r_pi
            + gamma * P_pi @ values
        )

        new_values[mdp.terminal] = 0.0

        delta = np.max(
            np.abs(new_values - values)
        )

        history.append(float(delta))

        values = new_values

        if delta < theta:
            return values, history, iteration

    return values, history, max_iterations


# ======================================================================
# STEP 5. ACTION-VALUE FUNCTION
# ======================================================================

def calculate_q_values(
    mdp: MDP,
    values: np.ndarray,
    gamma: float
):

    num_states = mdp.P.shape[1]
    num_actions = mdp.P.shape[0]

    q_values = np.zeros(
        (num_states, num_actions)
    )

    for state in range(num_states):

        if mdp.terminal[state]:
            continue

        for action in range(num_actions):

            q_values[state, action] = np.sum(
                mdp.P[action, state]
                * (
                    mdp.R[action, state]
                    + gamma * values
                )
            )

    return q_values


# ======================================================================
# STEP 6. GREEDY POLICY IMPROVEMENT
# ======================================================================

def greedy_policy_improvement(
    mdp: MDP,
    q_values: np.ndarray,
    tie_tolerance: float = 1e-10
):
    """
    Policy improvement:

        pi_new(s) = argmax_a Q_pi(s,a)

    If several actions share the maximum Q-value, distribute
    probability equally among those actions.

    The tie-aware rule prevents arbitrary action selection when
    multiple actions are equally good.
    """

    num_states, num_actions = q_values.shape

    new_policy = np.zeros(
        (num_states, num_actions)
    )

    for state in range(num_states):

        if mdp.terminal[state]:

            new_policy[state] = 1.0 / num_actions
            continue

        best_value = np.max(
            q_values[state]
        )

        best_actions = np.flatnonzero(
            np.abs(
                q_values[state] - best_value
            ) <= tie_tolerance
        )

        new_policy[state, best_actions] = (
            1.0 / len(best_actions)
        )

    return new_policy


# ======================================================================
# STEP 7. EPSILON-GREEDY POLICY
# ======================================================================

def epsilon_greedy_policy(
    mdp: MDP,
    q_values: np.ndarray,
    epsilon: float = 0.10,
    tie_tolerance: float = 1e-10
):
    """
    Epsilon-greedy policy:

        pi(a|s) =
            epsilon / |A|
            + (1-epsilon) / |A_best|, if a is greedy

    A greedy action receives additional probability.
    Every action retains some exploration probability when epsilon > 0.

    epsilon = 0:
        Pure greedy policy.

    epsilon = 1:
        Uniform random policy.
    """

    if not 0 <= epsilon <= 1:
        raise ValueError("epsilon must be in [0, 1].")

    num_states, num_actions = q_values.shape

    result = np.zeros(
        (num_states, num_actions)
    )

    for state in range(num_states):

        if mdp.terminal[state]:

            result[state] = 1.0 / num_actions
            continue

        best_value = np.max(
            q_values[state]
        )

        best_actions = np.flatnonzero(
            np.abs(
                q_values[state] - best_value
            ) <= tie_tolerance
        )

        result[state] = (
            epsilon / num_actions
        )

        result[state, best_actions] += (
            (1.0 - epsilon) / len(best_actions)
        )

    return result


# ======================================================================
# STEP 8. POLICY ITERATION
# ======================================================================

def policy_iteration(
    mdp: MDP,
    gamma: float = 0.90,
    max_policy_iterations: int = 100
):
    """
    Policy Iteration:

        1. Evaluate current policy.
        2. Calculate Q_pi(s,a).
        3. Improve policy greedily.
        4. Check policy stability.
        5. Repeat until stable.

    A stable policy is greedy with respect to its own value function.

    In a finite discounted MDP, exact policy iteration terminates
    after a finite number of policy improvements.

    Here policy evaluation is numerical, so stability is checked
    by comparing the policy arrays.
    """

    policy = mdp.policy.copy()

    evaluation_history = []
    policy_history = []

    for iteration in range(
        1,
        max_policy_iterations + 1
    ):

        values, history, evaluation_iterations = (
            evaluate_policy(
                mdp,
                policy,
                gamma
            )
        )

        evaluation_history.append(history)

        q_values = calculate_q_values(
            mdp,
            values,
            gamma
        )

        improved_policy = greedy_policy_improvement(
            mdp,
            q_values
        )

        policy_history.append(
            improved_policy.copy()
        )

        stable = np.allclose(
            policy,
            improved_policy,
            atol=1e-10
        )

        policy = improved_policy

        print(
            f"Policy iteration {iteration}: "
            f"evaluation steps={evaluation_iterations}, "
            f"stable={stable}"
        )

        if stable:

            return {
                "policy": policy,
                "values": values,
                "q_values": q_values,
                "evaluation_history": evaluation_history,
                "policy_history": policy_history,
                "iterations": iteration,
                "stable": True
            }

    return {
        "policy": policy,
        "values": values,
        "q_values": q_values,
        "evaluation_history": evaluation_history,
        "policy_history": policy_history,
        "iterations": max_policy_iterations,
        "stable": False
    }


# ======================================================================
# STEP 9. POLICY METRICS
# ======================================================================

def policy_difference(
    old_policy: np.ndarray,
    new_policy: np.ndarray
) -> float:

    return float(
        np.max(
            np.abs(old_policy - new_policy)
        )
    )


def policy_expected_value(
    mdp: MDP,
    policy: np.ndarray,
    values: np.ndarray,
    start_state: int = 0
) -> float:

    return float(values[start_state])


# ======================================================================
# STEP 10. VISUALIZATION
# ======================================================================

def values_to_grid(
    mdp: MDP,
    values: np.ndarray
):

    grid = values.reshape(
        mdp.grid_shape
    ).copy()

    for state in mdp.obstacle_states:

        row, col = divmod(
            state,
            mdp.grid_shape[1]
        )

        grid[row, col] = np.nan

    return grid


def visualize_values(
    mdp: MDP,
    values: np.ndarray,
    title: str
):

    grid = values_to_grid(
        mdp,
        values
    )

    fig, ax = plt.subplots(figsize=(7, 6))

    image = ax.imshow(
        grid,
        cmap="viridis"
    )

    fig.colorbar(
        image,
        ax=ax,
        label="V(s)"
    )

    rows, cols = mdp.grid_shape

    for row in range(rows):
        for col in range(cols):

            state = row * cols + col

            if state in mdp.obstacle_states:
                label = "WALL"
            elif mdp.terminal[state]:
                label = "GOAL"
            else:
                label = f"{values[state]:.3f}"

            ax.text(
                col,
                row,
                label,
                ha="center",
                va="center",
                color="white",
                fontweight="bold"
            )

    ax.set_title(title)
    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))

    plt.tight_layout()


def visualize_policy(
    mdp: MDP,
    policy: np.ndarray,
    title: str
):

    direction_symbols = [
        "↑",
        "→",
        "↓",
        "←"
    ]

    rows, cols = mdp.grid_shape

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(rows - 0.5, -0.5)

    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))

    ax.set_xticks(
        np.arange(-0.5, cols, 1),
        minor=True
    )

    ax.set_yticks(
        np.arange(-0.5, rows, 1),
        minor=True
    )

    ax.grid(
        which="minor",
        linewidth=1.5
    )

    for row in range(rows):
        for col in range(cols):

            state = row * cols + col

            if state in mdp.obstacle_states:

                ax.text(
                    col,
                    row,
                    "WALL",
                    ha="center",
                    va="center"
                )

            elif mdp.terminal[state]:

                ax.text(
                    col,
                    row,
                    "GOAL",
                    ha="center",
                    va="center"
                )

            else:

                probabilities = policy[state]

                symbols = []

                for action, probability in enumerate(
                    probabilities
                ):

                    if probability > 0.01:

                        symbols.append(
                            f"{direction_symbols[action]}"
                            f"{probability:.2f}"
                        )

                ax.text(
                    col,
                    row,
                    "\n".join(symbols),
                    ha="center",
                    va="center",
                    fontsize=9
                )

    ax.set_title(title)

    plt.tight_layout()


def visualize_policy_comparison(
    old_values: np.ndarray,
    new_values: np.ndarray
):

    states = np.arange(
        len(old_values)
    )

    plt.figure(figsize=(10, 5))

    plt.plot(
        states,
        old_values,
        marker="o",
        label="Original policy"
    )

    plt.plot(
        states,
        new_values,
        marker="s",
        label="Improved policy"
    )

    plt.xlabel("State")
    plt.ylabel("State Value")

    plt.title("Policy Improvement: State Value Comparison")

    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()


def visualize_policy_iteration(
    result: dict
):

    values_history = []

    for policy in result["policy_history"]:

        values, _, _ = evaluate_policy(
            GLOBAL_MDP,
            policy,
            GAMMA
        )

        values_history.append(
            values[0]
        )

    plt.figure(figsize=(8, 5))

    plt.plot(
        np.arange(1, len(values_history) + 1),
        values_history,
        marker="o"
    )

    plt.xlabel("Policy Improvement Iteration")
    plt.ylabel("Start State Value")

    plt.title("Policy Iteration Progress")

    plt.grid(True, alpha=0.3)

    plt.tight_layout()


def visualize_q_values(
    mdp: MDP,
    q_values: np.ndarray
):

    plt.figure(figsize=(9, 5))

    plt.imshow(
        q_values.T,
        aspect="auto",
        cmap="viridis"
    )

    plt.colorbar(
        label="Q(s,a)"
    )

    plt.yticks(
        range(mdp.P.shape[0]),
        mdp.action_names
    )

    plt.xticks(
        range(mdp.P.shape[1])
    )

    plt.xlabel("State")
    plt.ylabel("Action")

    plt.title("Action-Value Function")

    plt.tight_layout()


# Used by the policy-iteration visualization.
GLOBAL_MDP = None
GAMMA = 0.90


# ======================================================================
# STEP 11. UNIT TESTS
# ======================================================================

def run_tests():

    mdp = build_gridworld()

    gamma = 0.90

    original_policy = mdp.policy.copy()

    old_values, _, _ = evaluate_policy(
        mdp,
        original_policy,
        gamma
    )

    q_values = calculate_q_values(
        mdp,
        old_values,
        gamma
    )

    improved_policy = greedy_policy_improvement(
        mdp,
        q_values
    )

    new_values, _, _ = evaluate_policy(
        mdp,
        improved_policy,
        gamma
    )

    # TEST 1: Policy probability sums.
    validate_policy(
        mdp,
        improved_policy
    )

    print("[PASS] Improved policy is stochastic")

    # TEST 2: Improved policy does not reduce values.
    assert np.all(
        new_values >= old_values - 1e-8
    )

    print("[PASS] Policy improvement theorem")

    # TEST 3: Greedy actions maximize Q-values.
    for state in range(mdp.P.shape[1]):

        if mdp.terminal[state]:
            continue

        selected = np.flatnonzero(
            improved_policy[state] > 0
        )

        best = np.max(
            q_values[state]
        )

        assert np.allclose(
            q_values[state, selected],
            best
        )

    print("[PASS] Greedy policy selects maximum Q-values")

    # TEST 4: Epsilon-greedy probabilities.
    epsilon_policy = epsilon_greedy_policy(
        mdp,
        q_values,
        epsilon=0.20
    )

    validate_policy(
        mdp,
        epsilon_policy
    )

    print("[PASS] Epsilon-greedy policy probability sums")

    # TEST 5: Epsilon = 1 gives uniform policy.
    uniform_policy = epsilon_greedy_policy(
        mdp,
        q_values,
        epsilon=1.0
    )

    expected_uniform = np.full_like(
        uniform_policy,
        1.0 / mdp.P.shape[0]
    )

    assert np.allclose(
        uniform_policy,
        expected_uniform
    )

    print("[PASS] Epsilon=1 produces uniform policy")

    # TEST 6: Epsilon = 0 equals greedy policy.
    zero_epsilon_policy = epsilon_greedy_policy(
        mdp,
        q_values,
        epsilon=0.0
    )

    assert np.allclose(
        zero_epsilon_policy,
        improved_policy
    )

    print("[PASS] Epsilon=0 produces greedy policy")

    # TEST 7: Policy iteration stability.
    result = policy_iteration(
        mdp,
        gamma
    )

    assert result["stable"]

    print("[PASS] Policy iteration reaches stable policy")

    # TEST 8: Stable policy is greedy with respect to its own Q.
    stable_greedy = greedy_policy_improvement(
        mdp,
        result["q_values"]
    )

    assert np.allclose(
        stable_greedy,
        result["policy"]
    )

    print("[PASS] Final policy is greedy with respect to its Q-values")

    # TEST 9: Terminal values remain zero.
    assert np.allclose(
        result["values"][mdp.terminal],
        0.0
    )

    print("[PASS] Terminal values remain zero")

    print("\nAll 9 tests passed.")


# ======================================================================
# STEP 12. MAIN
# ======================================================================

def main():

    global GLOBAL_MDP, GAMMA

    GAMMA = 0.90

    mdp = build_gridworld()

    GLOBAL_MDP = mdp

    print("=" * 65)
    print("PHASE 2.2 — POLICY IMPROVEMENT")
    print("=" * 65)

    # --------------------------------------------------------------
    # A. Evaluate original policy.
    # --------------------------------------------------------------

    original_policy = mdp.policy.copy()

    old_values, old_history, old_iterations = evaluate_policy(
        mdp,
        original_policy,
        GAMMA
    )

    print("\n[1] ORIGINAL POLICY")

    print(
        "Start state value:",
        f"{old_values[0]:.6f}"
    )

    # --------------------------------------------------------------
    # B. Calculate Q-values.
    # --------------------------------------------------------------

    old_q_values = calculate_q_values(
        mdp,
        old_values,
        GAMMA
    )

    # --------------------------------------------------------------
    # C. Improve policy.
    # --------------------------------------------------------------

    improved_policy = greedy_policy_improvement(
        mdp,
        old_q_values
    )

    # --------------------------------------------------------------
    # D. Evaluate improved policy.
    # --------------------------------------------------------------

    new_values, new_history, new_iterations = evaluate_policy(
        mdp,
        improved_policy,
        GAMMA
    )

    print("\n[2] IMPROVED POLICY")

    print(
        "Start state value:",
        f"{new_values[0]:.6f}"
    )

    print(
        "Start state value improvement:",
        f"{new_values[0] - old_values[0]:.6f}"
    )

    print(
        "Maximum state value improvement:",
        f"{np.max(new_values - old_values):.6f}"
    )

    # --------------------------------------------------------------
    # E. Epsilon-greedy policy.
    # --------------------------------------------------------------

    epsilon_policy = epsilon_greedy_policy(
        mdp,
        old_q_values,
        epsilon=0.10
    )

    print("\n[3] EPSILON-GREEDY POLICY")

    print(
        "State 0 action probabilities:",
        epsilon_policy[0]
    )

    # --------------------------------------------------------------
    # F. Full policy iteration.
    # --------------------------------------------------------------

    print("\n[4] POLICY ITERATION")

    result = policy_iteration(
        mdp,
        gamma=GAMMA
    )

    print(
        "Stable:",
        result["stable"]
    )

    print(
        "Policy improvement iterations:",
        result["iterations"]
    )

    print(
        "Optimal start state value:",
        f"{result['values'][0]:.6f}"
    )

    # --------------------------------------------------------------
    # G. Unit tests.
    # --------------------------------------------------------------

    print("\n[5] UNIT TESTS")

    run_tests()

    # --------------------------------------------------------------
    # H. Visualizations.
    # --------------------------------------------------------------

    visualize_values(
        mdp,
        old_values,
        "Original Policy State Values"
    )

    visualize_values(
        mdp,
        new_values,
        "Improved Policy State Values"
    )

    visualize_policy(
        mdp,
        original_policy,
        "Original Random Policy"
    )

    visualize_policy(
        mdp,
        improved_policy,
        "Greedy Improved Policy"
    )

    visualize_policy_comparison(
        old_values,
        new_values
    )

    visualize_q_values(
        mdp,
        old_q_values
    )

    visualize_policy_iteration(
        result
    )

    plt.show()

    print("\nPhase 2.2 completed successfully.")


if __name__ == "__main__":
    main()
