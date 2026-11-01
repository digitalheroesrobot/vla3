
"""
======================================================================
RL COURSE — PHASE 2.3
VALUE ITERATION AND THE BELLMAN OPTIMALITY EQUATION
======================================================================

LESSON ROADMAP
--------------

STEP 1 — Bellman Optimality Equation
Purpose:
    Understand how to calculate the maximum achievable expected return.

What you learn:
    V*(s) = max_a sum_s' P(s'|s,a)
            [R(s,a,s') + gamma * V*(s')]

Implementation:
    Implement a Bellman optimality backup.

STEP 2 — Bellman Optimality Operator
Purpose:
    Understand how repeated optimality backups improve value estimates.

What you learn:
    V_new = T*(V_old)

    T* is a contraction for a finite discounted MDP when gamma < 1.

Implementation:
    Implement the Bellman optimality operator.

STEP 3 — Synchronous Value Iteration
Purpose:
    Calculate optimal state values using repeated updates.

What you learn:
    Every state is updated using the previous iteration's values.

Implementation:
    Implement synchronous Value Iteration and convergence tracking.

STEP 4 — Asynchronous Value Iteration
Purpose:
    Understand in-place state updates.

What you learn:
    Updated values can be reused immediately during the same sweep.

Implementation:
    Implement in-place Value Iteration and compare convergence.

STEP 5 — Optimal Q-Values
Purpose:
    Evaluate every action under the optimal value function.

What you learn:
    Q*(s,a) = sum_s' P(s'|s,a)
              [R(s,a,s') + gamma * V*(s')]

    V*(s) = max_a Q*(s,a)

Implementation:
    Calculate optimal Q-values.

STEP 6 — Optimal Policy Extraction
Purpose:
    Convert optimal values into an actionable policy.

What you learn:
    pi*(s) = argmax_a Q*(s,a)

    Multiple actions may be equally optimal.

Implementation:
    Extract deterministic or tie-aware stochastic greedy policies.

STEP 7 — Policy Iteration Comparison
Purpose:
    Compare two classical Dynamic Programming algorithms.

What you learn:
    Policy Iteration:
        Evaluate policy -> Improve policy -> Repeat.

    Value Iteration:
        Apply optimality backup -> Repeat.

Implementation:
    Compare final values and policies against Policy Iteration.

STEP 8 — Numerical Verification
Purpose:
    Verify that the implementation solves the Bellman optimality
    equation.

What you learn:
    The optimality residual approaches zero.
    The extracted greedy policy achieves the calculated optimal values.

Implementation:
    Run numerical checks and unit tests.

STEP 9 — Visualization
Purpose:
    Understand optimal values, actions, and convergence.

What you learn:
    Higher-value states generally offer better expected future returns.
    Policy arrows show the optimal actions.

Implementation:
    Plot value heatmaps, policy maps, convergence curves,
    Q-values, and algorithm comparisons.

LEARNING CYCLE
--------------
Understand -> Implement -> Visualize -> Verify -> Experiment

DEPENDENCIES
------------
pip install numpy matplotlib

RUN
---
python phase2/value_iteration.py

======================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


# ======================================================================
# STEP 1. MDP STRUCTURE
# ======================================================================

@dataclass
class MDP:

    P: np.ndarray
    R: np.ndarray
    terminal: np.ndarray

    state_names: List[str]
    action_names: List[str]

    grid_shape: Tuple[int, int]
    obstacle_states: Tuple[int, ...]

    @property
    def num_states(self):
        return self.P.shape[1]

    @property
    def num_actions(self):
        return self.P.shape[0]


# ======================================================================
# STEP 2. BUILD GRIDWORLD
# ======================================================================

def build_gridworld() -> MDP:
    """
    Construct a 4x4 stochastic GridWorld.

    Actions:
        0 = Up
        1 = Right
        2 = Down
        3 = Left

    Transition model:
        80% intended direction.
        10% first perpendicular direction.
        10% second perpendicular direction.

    Rewards:
        +1.0 when entering the goal.
        -0.04 for ordinary movement.

    Goal:
        (3, 3)

    Obstacle:
        (1, 1)

    Terminal-state convention:
        Goal and obstacle states are fixed at value zero.
        The reward for entering the goal is received on the
        transition into the goal.

    The obstacle is represented as an absorbing state with zero
    value, but normal transitions into the obstacle are blocked.
    """

    rows = 4
    cols = 4

    num_states = rows * cols
    num_actions = 4

    directions = [
        (-1, 0),
        (0, 1),
        (1, 0),
        (0, -1),
    ]

    action_names = [
        "Up",
        "Right",
        "Down",
        "Left",
    ]

    goal = (3, 3)
    obstacle = (1, 1)

    goal_state = goal[0] * cols + goal[1]
    obstacle_state = obstacle[0] * cols + obstacle[1]

    terminal = np.zeros(
        num_states,
        dtype=bool
    )

    terminal[goal_state] = True
    terminal[obstacle_state] = True

    P = np.zeros(
        (num_actions, num_states, num_states),
        dtype=float
    )

    R = np.zeros_like(P)

    def move(state: int, action: int) -> int:

        row, col = divmod(state, cols)

        dr, dc = directions[action]

        nr = row + dr
        nc = col + dc

        if not (0 <= nr < rows):
            return state

        if not (0 <= nc < cols):
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
                    1.0
                    if destination == goal_state
                    else -0.04
                )

    state_names = [
        f"({row},{col})"
        for row in range(rows)
        for col in range(cols)
    ]

    return MDP(
        P=P,
        R=R,
        terminal=terminal,
        state_names=state_names,
        action_names=action_names,
        grid_shape=(rows, cols),
        obstacle_states=(obstacle_state,)
    )


# ======================================================================
# STEP 3. MDP VALIDATION
# ======================================================================

def validate_mdp(mdp: MDP) -> None:

    if np.any(mdp.P < -1e-12):
        raise ValueError(
            "Transition probabilities cannot be negative."
        )

    probability_sums = np.sum(
        mdp.P,
        axis=2
    )

    if not np.allclose(
        probability_sums,
        1.0
    ):
        raise ValueError(
            "Transition probabilities must sum to one."
        )

    if mdp.R.shape != mdp.P.shape:
        raise ValueError(
            "Reward and transition tensors must have equal shapes."
        )

    if mdp.terminal.shape != (mdp.num_states,):
        raise ValueError(
            "Invalid terminal-state shape."
        )


# ======================================================================
# STEP 4. BELLMAN OPTIMALITY BACKUP
# ======================================================================

def calculate_q_values(
    mdp: MDP,
    values: np.ndarray,
    gamma: float
) -> np.ndarray:
    """
    Calculate Q(s,a) for every state-action pair.

    Q(s,a) = sum_s' P(s'|s,a)
             [R(s,a,s') + gamma * V(s')]

    The next-state value is evaluated using the supplied value vector.

    When values = V*, this produces Q*.
    """

    q_values = np.zeros(
        (mdp.num_states, mdp.num_actions),
        dtype=float
    )

    for state in range(mdp.num_states):

        if mdp.terminal[state]:
            continue

        for action in range(mdp.num_actions):

            q_values[state, action] = np.sum(
                mdp.P[action, state]
                * (
                    mdp.R[action, state]
                    + gamma * values
                )
            )

    return q_values


def bellman_optimality_backup(
    mdp: MDP,
    state: int,
    values: np.ndarray,
    gamma: float
) -> float:
    """
    Calculate one optimality backup.

    Bellman optimality equation:

        V*(s) = max_a sum_s' P(s'|s,a)
                [R(s,a,s') + gamma * V*(s')]

    The maximization is performed over the available actions.

    Unlike policy evaluation, there is no fixed policy probability
    pi(a|s) in this equation.

    The algorithm selects the action with the highest expected
    return under the current value estimate.
    """

    if mdp.terminal[state]:
        return 0.0

    q_values = np.zeros(
        mdp.num_actions
    )

    for action in range(mdp.num_actions):

        q_values[action] = np.sum(
            mdp.P[action, state]
            * (
                mdp.R[action, state]
                + gamma * values
            )
        )

    return float(
        np.max(q_values)
    )


def bellman_optimality_operator(
    mdp: MDP,
    values: np.ndarray,
    gamma: float
) -> np.ndarray:
    """
    Apply T* to every state.

        T*(V)(s) = max_a Q_V(s,a)

    Synchronous update:
        All new values are calculated from the same old value vector.
    """

    new_values = np.zeros_like(
        values,
        dtype=float
    )

    for state in range(mdp.num_states):

        new_values[state] = bellman_optimality_backup(
            mdp,
            state,
            values,
            gamma
        )

    return new_values


# ======================================================================
# STEP 5. SYNCHRONOUS VALUE ITERATION
# ======================================================================

def value_iteration(
    mdp: MDP,
    gamma: float = 0.90,
    theta: float = 1e-10,
    max_iterations: int = 10000,
    initial_values: np.ndarray | None = None
):
    """
    Synchronous Value Iteration.

    Algorithm:

        Initialize V(s).

        Repeat:

            V_new(s) = max_a sum_s' P(s'|s,a)
                       [R(s,a,s') + gamma * V_old(s')]

            delta = max_s |V_new(s) - V_old(s)|

            V_old = V_new

        Stop when delta < theta.

    Convergence:

        For a finite discounted MDP with gamma < 1:

            ||T*(V1) - T*(V2)||_infinity
                <= gamma * ||V1 - V2||_infinity

        Therefore T* is a contraction and has a unique fixed point V*.

    The fixed point satisfies the Bellman optimality equation.

    theta controls the numerical stopping criterion.
    """

    if not 0 <= gamma < 1:
        raise ValueError(
            "gamma must satisfy 0 <= gamma < 1."
        )

    if theta <= 0:
        raise ValueError(
            "theta must be positive."
        )

    if initial_values is None:

        values = np.zeros(
            mdp.num_states,
            dtype=float
        )

    else:

        values = np.asarray(
            initial_values,
            dtype=float
        ).copy()

        if values.shape != (mdp.num_states,):
            raise ValueError(
                "Invalid initial_values shape."
            )

    values[mdp.terminal] = 0.0

    history = []

    for iteration in range(
        1,
        max_iterations + 1
    ):

        new_values = bellman_optimality_operator(
            mdp,
            values,
            gamma
        )

        delta = float(
            np.max(
                np.abs(new_values - values)
            )
        )

        history.append(delta)

        values = new_values

        if delta < theta:

            return {
                "values": values,
                "history": history,
                "iterations": iteration,
                "converged": True
            }

    return {
        "values": values,
        "history": history,
        "iterations": max_iterations,
        "converged": False
    }


# ======================================================================
# STEP 6. ASYNCHRONOUS VALUE ITERATION
# ======================================================================

def asynchronous_value_iteration(
    mdp: MDP,
    gamma: float = 0.90,
    theta: float = 1e-10,
    max_iterations: int = 10000,
    initial_values: np.ndarray | None = None
):
    """
    In-place Value Iteration.

    Every state is updated immediately:

        V(s) <- T*(V)(s)

    A later state update can use values updated earlier in the same
    sweep.

    This differs from synchronous Value Iteration, which calculates
    a complete new vector from the previous vector.

    Update order can influence convergence speed.

    This implementation uses a fixed ascending state order and
    updates every state once per sweep.
    """

    if not 0 <= gamma < 1:
        raise ValueError(
            "gamma must satisfy 0 <= gamma < 1."
        )

    if theta <= 0:
        raise ValueError(
            "theta must be positive."
        )

    if initial_values is None:

        values = np.zeros(
            mdp.num_states,
            dtype=float
        )

    else:

        values = np.asarray(
            initial_values,
            dtype=float
        ).copy()

        if values.shape != (mdp.num_states,):
            raise ValueError(
                "Invalid initial_values shape."
            )

    values[mdp.terminal] = 0.0

    history = []

    for iteration in range(
        1,
        max_iterations + 1
    ):

        old_values = values.copy()

        for state in range(mdp.num_states):

            values[state] = bellman_optimality_backup(
                mdp,
                state,
                values,
                gamma
            )

        values[mdp.terminal] = 0.0

        delta = float(
            np.max(
                np.abs(values - old_values)
            )
        )

        history.append(delta)

        if delta < theta:

            return {
                "values": values,
                "history": history,
                "iterations": iteration,
                "converged": True
            }

    return {
        "values": values,
        "history": history,
        "iterations": max_iterations,
        "converged": False
    }


# ======================================================================
# STEP 7. OPTIMAL POLICY EXTRACTION
# ======================================================================

def extract_greedy_policy(
    mdp: MDP,
    q_values: np.ndarray,
    tie_tolerance: float = 1e-10
) -> np.ndarray:
    """
    Extract a policy greedy with respect to Q-values.

        pi*(a|s) = 1 for a maximizing Q-value.

    If multiple actions have equal maximum Q-values,
    distribute probability equally among them.

    The resulting policy is stochastic only where there are ties.

    This policy is optimal when q_values are calculated from V*.
    """

    policy = np.zeros(
        (mdp.num_states, mdp.num_actions),
        dtype=float
    )

    for state in range(mdp.num_states):

        if mdp.terminal[state]:

            policy[state] = 1.0 / mdp.num_actions
            continue

        best_value = np.max(
            q_values[state]
        )

        best_actions = np.flatnonzero(
            np.abs(
                q_values[state] - best_value
            ) <= tie_tolerance
        )

        policy[state, best_actions] = (
            1.0 / len(best_actions)
        )

    return policy


def extract_deterministic_policy(
    mdp: MDP,
    q_values: np.ndarray
) -> np.ndarray:
    """
    Return one maximizing action per state.

    np.argmax selects the first maximizing action when there is a tie.

    Terminal states receive action index -1 because no action is
    required after termination.
    """

    actions = np.full(
        mdp.num_states,
        -1,
        dtype=int
    )

    for state in range(mdp.num_states):

        if mdp.terminal[state]:
            continue

        actions[state] = int(
            np.argmax(q_values[state])
        )

    return actions


# ======================================================================
# STEP 8. POLICY EVALUATION FOR COMPARISON
# ======================================================================

def evaluate_policy_exact(
    mdp: MDP,
    policy: np.ndarray,
    gamma: float
) -> np.ndarray:
    """
    Exact policy evaluation.

        P_pi(s,s') = sum_a pi(a|s) P(s'|s,a)

        r_pi(s) = sum_a pi(a|s)
                  sum_s' P(s'|s,a) R(s,a,s')

        V_pi = r_pi + gamma P_pi V_pi

        (I - gamma P_pi) V_pi = r_pi

    Terminal values are fixed at zero.
    """

    if not 0 <= gamma < 1:
        raise ValueError(
            "gamma must satisfy 0 <= gamma < 1."
        )

    if policy.shape != (
        mdp.num_states,
        mdp.num_actions
    ):
        raise ValueError("Invalid policy shape.")

    if np.any(policy < -1e-12):
        raise ValueError(
            "Policy probabilities cannot be negative."
        )

    if not np.allclose(
        np.sum(policy, axis=1),
        1.0
    ):
        raise ValueError(
            "Policy probabilities must sum to one."
        )

    P_pi = np.zeros(
        (mdp.num_states, mdp.num_states)
    )

    r_pi = np.zeros(
        mdp.num_states
    )

    for state in range(mdp.num_states):

        if mdp.terminal[state]:
            continue

        for action in range(mdp.num_actions):

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

    nonterminal = np.where(
        ~mdp.terminal
    )[0]

    values = np.zeros(
        mdp.num_states
    )

    P_reduced = P_pi[
        np.ix_(nonterminal, nonterminal)
    ]

    A = (
        np.eye(len(nonterminal))
        - gamma * P_reduced
    )

    values[nonterminal] = np.linalg.solve(
        A,
        r_pi[nonterminal]
    )

    return values


# ======================================================================
# STEP 9. POLICY ITERATION REFERENCE
# ======================================================================

def policy_iteration(
    mdp: MDP,
    gamma: float = 0.90,
    max_iterations: int = 100
):
    """
    Reference implementation of Policy Iteration.

    Repeatedly:

        1. Evaluate current policy exactly.
        2. Calculate Q_pi.
        3. Improve policy greedily.
        4. Stop when policy is stable.

    This provides an independent algorithmic comparison with
    Value Iteration.
    """

    policy = np.full(
        (mdp.num_states, mdp.num_actions),
        1.0 / mdp.num_actions
    )

    history = []

    for iteration in range(
        1,
        max_iterations + 1
    ):

        values = evaluate_policy_exact(
            mdp,
            policy,
            gamma
        )

        q_values = calculate_q_values(
            mdp,
            values,
            gamma
        )

        new_policy = extract_greedy_policy(
            mdp,
            q_values
        )

        history.append(
            values[0]
        )

        stable = np.allclose(
            policy,
            new_policy,
            atol=1e-10
        )

        policy = new_policy

        if stable:

            return {
                "policy": policy,
                "values": values,
                "q_values": q_values,
                "history": history,
                "iterations": iteration,
                "stable": True
            }

    return {
        "policy": policy,
        "values": values,
        "q_values": q_values,
        "history": history,
        "iterations": max_iterations,
        "stable": False
    }


# ======================================================================
# STEP 10. BELLMAN OPTIMALITY RESIDUAL
# ======================================================================

def calculate_optimality_residual(
    mdp: MDP,
    values: np.ndarray,
    gamma: float
) -> np.ndarray:
    """
    Optimality residual:

        residual(s) = T*(V)(s) - V(s)

    The optimal value function satisfies:

        T*(V*) = V*

    Therefore:

        residual(s) = 0

    for every state, up to numerical precision.
    """

    return (
        bellman_optimality_operator(
            mdp,
            values,
            gamma
        )
        - values
    )


# ======================================================================
# STEP 11. POLICY VALUE CONSISTENCY
# ======================================================================

def calculate_policy_value_from_q(
    policy: np.ndarray,
    q_values: np.ndarray
) -> np.ndarray:

    return np.sum(
        policy * q_values,
        axis=1
    )


# ======================================================================
# STEP 12. VISUALIZATION HELPERS
# ======================================================================

def values_to_grid(
    mdp: MDP,
    values: np.ndarray
) -> np.ndarray:

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


def visualize_value_heatmap(
    mdp: MDP,
    values: np.ndarray,
    title: str
):

    grid = values_to_grid(
        mdp,
        values
    )

    fig, ax = plt.subplots(
        figsize=(7, 6)
    )

    image = ax.imshow(
        grid,
        cmap="viridis"
    )

    fig.colorbar(
        image,
        ax=ax,
        label="V*(s)"
    )

    rows, cols = mdp.grid_shape

    for row in range(rows):

        for col in range(cols):

            state = row * cols + col

            if state in mdp.obstacle_states:

                label = "WALL"

            elif mdp.terminal[state]:

                label = "GOAL\nV=0"

            else:

                label = f"{values[state]:.3f}"

            ax.text(
                col,
                row,
                label,
                ha="center",
                va="center",
                color="white",
                fontweight="bold",
                fontsize=9
            )

    ax.set_title(title)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")

    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))

    plt.tight_layout()


def visualize_policy(
    mdp: MDP,
    policy: np.ndarray,
    title: str
):

    symbols = [
        "↑",
        "→",
        "↓",
        "←"
    ]

    rows, cols = mdp.grid_shape

    fig, ax = plt.subplots(
        figsize=(7, 6)
    )

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

                label = "WALL"

            elif mdp.terminal[state]:

                label = "GOAL"

            else:

                label_parts = []

                for action, probability in enumerate(
                    policy[state]
                ):

                    if probability > 0.01:

                        label_parts.append(
                            f"{symbols[action]} "
                            f"{probability:.2f}"
                        )

                label = "\n".join(label_parts)

            ax.text(
                col,
                row,
                label,
                ha="center",
                va="center",
                fontsize=9
            )

    ax.set_title(title)

    plt.tight_layout()


def visualize_convergence(
    histories: Dict[str, List[float]]
):

    plt.figure(
        figsize=(9, 5)
    )

    for name, history in histories.items():

        plt.plot(
            np.arange(1, len(history) + 1),
            history,
            label=name
        )

    plt.yscale("log")

    plt.xlabel("Iteration")
    plt.ylabel("Maximum Value Change (log scale)")

    plt.title("Value Iteration Convergence")

    plt.grid(
        True,
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()


def visualize_q_values(
    mdp: MDP,
    q_values: np.ndarray
):

    plt.figure(
        figsize=(10, 5)
    )

    plt.imshow(
        q_values.T,
        aspect="auto",
        cmap="viridis"
    )

    plt.colorbar(
        label="Q*(s,a)"
    )

    plt.yticks(
        range(mdp.num_actions),
        mdp.action_names
    )

    plt.xticks(
        range(mdp.num_states)
    )

    plt.xlabel("State")
    plt.ylabel("Action")

    plt.title("Optimal Action-Value Function")

    plt.tight_layout()


def visualize_algorithm_comparison(
    value_iteration_values: np.ndarray,
    policy_iteration_values: np.ndarray
):

    states = np.arange(
        len(value_iteration_values)
    )

    plt.figure(
        figsize=(10, 5)
    )

    plt.plot(
        states,
        value_iteration_values,
        marker="o",
        label="Value Iteration"
    )

    plt.plot(
        states,
        policy_iteration_values,
        marker="s",
        linestyle="--",
        label="Policy Iteration"
    )

    plt.xlabel("State")
    plt.ylabel("Optimal State Value")

    plt.title("Value Iteration vs Policy Iteration")

    plt.grid(
        True,
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()


def visualize_residual(
    mdp: MDP,
    values: np.ndarray,
    gamma: float
):

    residual = calculate_optimality_residual(
        mdp,
        values,
        gamma
    )

    grid = values_to_grid(
        mdp,
        residual
    )

    fig, ax = plt.subplots(
        figsize=(7, 6)
    )

    image = ax.imshow(
        grid,
        cmap="coolwarm"
    )

    fig.colorbar(
        image,
        ax=ax,
        label="T*(V) - V"
    )

    rows, cols = mdp.grid_shape

    for row in range(rows):

        for col in range(cols):

            state = row * cols + col

            if state in mdp.obstacle_states:

                label = "WALL"

            else:

                label = f"{residual[state]:.2e}"

            ax.text(
                col,
                row,
                label,
                ha="center",
                va="center",
                fontsize=8
            )

    ax.set_title(
        f"Bellman Optimality Residual (gamma={gamma})"
    )

    ax.set_xlabel("Column")
    ax.set_ylabel("Row")

    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))

    plt.tight_layout()


def visualize_gamma_comparison(
    mdp: MDP,
    gamma_results: Dict[float, dict]
):

    plt.figure(
        figsize=(9, 5)
    )

    for gamma, result in gamma_results.items():

        plt.plot(
            np.arange(mdp.num_states),
            result["values"],
            marker="o",
            label=f"gamma={gamma}"
        )

    plt.xlabel("State")
    plt.ylabel("Optimal State Value")

    plt.title("Effect of Discount Factor on Optimal Values")

    plt.grid(
        True,
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()


# ======================================================================
# STEP 13. UNIT TESTS
# ======================================================================

def run_tests():

    mdp = build_gridworld()

    validate_mdp(mdp)

    gamma = 0.90

    synchronous = value_iteration(
        mdp,
        gamma=gamma
    )

    asynchronous = asynchronous_value_iteration(
        mdp,
        gamma=gamma
    )

    values = synchronous["values"]

    # --------------------------------------------------------------
    # TEST 1: Transition probabilities sum to one.
    # --------------------------------------------------------------

    assert np.allclose(
        np.sum(mdp.P, axis=2),
        1.0
    )

    print("[PASS] Transition probabilities sum to one")

    # --------------------------------------------------------------
    # TEST 2: Terminal values are zero.
    # --------------------------------------------------------------

    assert np.allclose(
        values[mdp.terminal],
        0.0
    )

    print("[PASS] Terminal values are zero")

    # --------------------------------------------------------------
    # TEST 3: One Bellman backup equals max Q.
    # --------------------------------------------------------------

    test_values = np.linspace(
        0.0,
        1.0,
        mdp.num_states
    )

    state = 0

    q = calculate_q_values(
        mdp,
        test_values,
        gamma
    )

    backup = bellman_optimality_backup(
        mdp,
        state,
        test_values,
        gamma
    )

    assert np.isclose(
        backup,
        np.max(q[state])
    )

    print("[PASS] Bellman optimality backup equals max Q")

    # --------------------------------------------------------------
    # TEST 4: Synchronous convergence.
    # --------------------------------------------------------------

    assert synchronous["converged"]

    print("[PASS] Synchronous Value Iteration converges")

    # --------------------------------------------------------------
    # TEST 5: Asynchronous convergence.
    # --------------------------------------------------------------

    assert asynchronous["converged"]

    print("[PASS] Asynchronous Value Iteration converges")

    # --------------------------------------------------------------
    # TEST 6: Both methods agree.
    # --------------------------------------------------------------

    assert np.allclose(
        synchronous["values"],
        asynchronous["values"],
        atol=1e-8
    )

    print("[PASS] Synchronous and asynchronous values agree")

    # --------------------------------------------------------------
    # TEST 7: Optimality residual.
    # --------------------------------------------------------------

    residual = calculate_optimality_residual(
        mdp,
        values,
        gamma
    )

    assert np.max(
        np.abs(residual)
    ) < 1e-8

    print("[PASS] Bellman optimality residual is near zero")

    # --------------------------------------------------------------
    # TEST 8: V*(s) equals max_a Q*(s,a).
    # --------------------------------------------------------------

    optimal_q = calculate_q_values(
        mdp,
        values,
        gamma
    )

    assert np.allclose(
        np.max(optimal_q, axis=1),
        values,
        atol=1e-8
    )

    print("[PASS] V*(s) equals max Q*(s,a)")

    # --------------------------------------------------------------
    # TEST 9: Greedy policy is valid.
    # --------------------------------------------------------------

    policy = extract_greedy_policy(
        mdp,
        optimal_q
    )

    assert np.allclose(
        np.sum(policy, axis=1),
        1.0
    )

    assert np.all(policy >= 0)

    print("[PASS] Extracted optimal policy is valid")

    # --------------------------------------------------------------
    # TEST 10: Policy evaluation reproduces V*.
    # --------------------------------------------------------------

    evaluated_values = evaluate_policy_exact(
        mdp,
        policy,
        gamma
    )

    assert np.allclose(
        evaluated_values,
        values,
        atol=1e-8
    )

    print("[PASS] Greedy policy evaluation reproduces V*")

    # --------------------------------------------------------------
    # TEST 11: Policy Iteration agrees.
    # --------------------------------------------------------------

    policy_result = policy_iteration(
        mdp,
        gamma
    )

    assert policy_result["stable"]

    assert np.allclose(
        policy_result["values"],
        values,
        atol=1e-8
    )

    print("[PASS] Policy Iteration agrees with Value Iteration")

    # --------------------------------------------------------------
    # TEST 12: Different initialization gives same solution.
    # --------------------------------------------------------------

    rng = np.random.default_rng(42)

    initial_values = rng.normal(
        size=mdp.num_states
    )

    random_initial_result = value_iteration(
        mdp,
        gamma=gamma,
        initial_values=initial_values
    )

    assert np.allclose(
        random_initial_result["values"],
        values,
        atol=1e-8
    )

    print("[PASS] Different initial values converge to same V*")

    # --------------------------------------------------------------
    # TEST 13: Invalid gamma is rejected.
    # --------------------------------------------------------------

    rejected = False

    try:

        value_iteration(
            mdp,
            gamma=1.0
        )

    except ValueError:

        rejected = True

    assert rejected

    print("[PASS] Invalid gamma is rejected")

    print("\nAll 13 tests passed.")


# ======================================================================
# STEP 14. MAIN
# ======================================================================

def main():

    print("=" * 70)
    print("PHASE 2.3 — VALUE ITERATION")
    print("=" * 70)

    mdp = build_gridworld()

    validate_mdp(mdp)

    gamma = 0.90

    # --------------------------------------------------------------
    # A. Synchronous Value Iteration.
    # --------------------------------------------------------------

    print("\n[1] SYNCHRONOUS VALUE ITERATION")

    sync_result = value_iteration(
        mdp,
        gamma=gamma
    )

    optimal_values = sync_result["values"]

    print(
        "Converged:",
        sync_result["converged"]
    )

    print(
        "Iterations:",
        sync_result["iterations"]
    )

    print(
        "Start state optimal value:",
        f"{optimal_values[0]:.8f}"
    )

    # --------------------------------------------------------------
    # B. Asynchronous Value Iteration.
    # --------------------------------------------------------------

    print("\n[2] ASYNCHRONOUS VALUE ITERATION")

    async_result = asynchronous_value_iteration(
        mdp,
        gamma=gamma
    )

    print(
        "Converged:",
        async_result["converged"]
    )

    print(
        "Iterations:",
        async_result["iterations"]
    )

    print(
        "Start state optimal value:",
        f"{async_result['values'][0]:.8f}"
    )

    print(
        "Maximum difference:",
        f"{np.max(np.abs(optimal_values - async_result['values'])):.3e}"
    )

    # --------------------------------------------------------------
    # C. Calculate optimal Q-values.
    # --------------------------------------------------------------

    print("\n[3] OPTIMAL Q-VALUES")

    optimal_q = calculate_q_values(
        mdp,
        optimal_values,
        gamma
    )

    for state in range(mdp.num_states):

        if mdp.terminal[state]:
            continue

        values_text = ", ".join(
            f"{mdp.action_names[action]}="
            f"{optimal_q[state, action]:.4f}"
            for action in range(mdp.num_actions)
        )

        print(
            f"State {state:2d}: {values_text}"
        )

    # --------------------------------------------------------------
    # D. Extract optimal policy.
    # --------------------------------------------------------------

    optimal_policy = extract_greedy_policy(
        mdp,
        optimal_q
    )

    deterministic_policy = extract_deterministic_policy(
        mdp,
        optimal_q
    )

    print("\n[4] OPTIMAL POLICY")

    for state in range(mdp.num_states):

        if mdp.terminal[state]:
            continue

        selected_actions = np.flatnonzero(
            optimal_policy[state] > 0
        )

        selected_names = [
            mdp.action_names[action]
            for action in selected_actions
        ]

        print(
            f"State {state:2d} "
            f"{mdp.state_names[state]}: "
            f"{selected_names}"
        )

    print(
        "\nDeterministic action indices:",
        deterministic_policy
    )

    # --------------------------------------------------------------
    # E. Compare with Policy Iteration.
    # --------------------------------------------------------------

    print("\n[5] POLICY ITERATION COMPARISON")

    policy_result = policy_iteration(
        mdp,
        gamma=gamma
    )

    print(
        "Policy Iteration stable:",
        policy_result["stable"]
    )

    print(
        "Policy Iteration rounds:",
        policy_result["iterations"]
    )

    print(
        "Maximum value difference:",
        f"{np.max(np.abs(optimal_values - policy_result['values'])):.3e}"
    )

    # --------------------------------------------------------------
    # F. Discount-factor experiment.
    # --------------------------------------------------------------

    print("\n[6] DISCOUNT FACTOR EXPERIMENT")

    gamma_results = {}

    for discount in [0.50, 0.90, 0.99]:

        result = value_iteration(
            mdp,
            gamma=discount
        )

        gamma_results[discount] = result

        print(
            f"gamma={discount:.2f}: "
            f"iterations={result['iterations']}, "
            f"V*(start)={result['values'][0]:.6f}"
        )

    # --------------------------------------------------------------
    # G. Unit tests.
    # --------------------------------------------------------------

    print("\n[7] UNIT TESTS")

    run_tests()

    # --------------------------------------------------------------
    # H. Visualizations.
    # --------------------------------------------------------------

    visualize_value_heatmap(
        mdp,
        optimal_values,
        f"Optimal State Values (gamma={gamma})"
    )

    visualize_policy(
        mdp,
        optimal_policy,
        "Optimal Greedy Policy"
    )

    visualize_convergence(
        {
            "Synchronous": sync_result["history"],
            "Asynchronous": async_result["history"],
        }
    )

    visualize_q_values(
        mdp,
        optimal_q
    )

    visualize_algorithm_comparison(
        optimal_values,
        policy_result["values"]
    )

    visualize_residual(
        mdp,
        optimal_values,
        gamma
    )

    visualize_gamma_comparison(
        mdp,
        gamma_results
    )

    plt.show()

    print("\nPhase 2.3 completed successfully.")


if __name__ == "__main__":
    main()
