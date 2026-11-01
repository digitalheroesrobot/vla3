
"""
======================================================================
RL COURSE — PHASE 2.1
Bellman Expectation Equation and Iterative Policy Evaluation
======================================================================

LESSON ROADMAP
--------------

STEP 1 — Bellman Expectation Equation
Purpose:
    Understand how a state's value is determined by immediate reward
    and the expected value of its possible next states.

What you learn:
    V_pi(s) = sum_a pi(a|s) sum_s' P(s'|s,a)
              [R(s,a,s') + gamma * V_pi(s')]

Implementation:
    Implement a Bellman backup for one state and for all states.

STEP 2 — Bellman Operator
Purpose:
    Understand the Bellman expectation operator T_pi.

What you learn:
    Repeatedly applying T_pi moves a value estimate toward V_pi.
    For a finite discounted MDP with gamma < 1, T_pi is a contraction.

Implementation:
    Implement synchronous and asynchronous Bellman updates.

STEP 3 — Iterative Policy Evaluation
Purpose:
    Calculate state values for a fixed policy.

What you learn:
    Initial value estimates can differ, but iterative evaluation
    converges to the same solution under the discounted-MDP conditions.

Implementation:
    Compare zero initialization, random initialization,
    synchronous updates, and in-place asynchronous updates.

STEP 4 — Exact Matrix Solution
Purpose:
    Establish an independent numerical reference.

What you learn:
    V = r_pi + gamma * P_pi * V
    (I - gamma * P_pi) V = r_pi

Implementation:
    Solve the linear system with numpy.linalg.solve.
    Do not explicitly calculate the matrix inverse.

STEP 5 — State Values and Action Values
Purpose:
    Understand the difference between V_pi(s) and Q_pi(s,a).

What you learn:
    V_pi(s) evaluates the policy.
    Q_pi(s,a) evaluates taking a particular action first,
    then following the policy.

Implementation:
    Calculate V and Q and verify their consistency.

STEP 6 — Convergence and Discount Factor
Purpose:
    Observe how gamma affects values and convergence speed.

What you learn:
    Higher gamma gives more weight to future rewards.
    The contraction factor is gamma, so convergence can become slower
    as gamma approaches 1.

Implementation:
    Track the maximum update error and compare convergence curves.

STEP 7 — Numerical Verification
Purpose:
    Verify the implementation rather than trusting the output.

What you learn:
    A correct iterative solution should agree with the exact solution.
    Its Bellman residual should approach zero.

Implementation:
    Run unit tests, numerical comparisons, and residual checks.

STEP 8 — Visualization
Purpose:
    Connect mathematical values to an intuitive environment.

What you learn:
    State values describe the expected discounted return from each
    location when following a fixed policy.

Implementation:
    Plot value heatmaps, convergence curves, residuals,
    and discount-factor comparisons.

LEARNING CYCLE
--------------
Understand (comments)
    -> Implement
    -> Visualize
    -> Verify
    -> Experiment

DEPENDENCIES
------------
pip install numpy matplotlib

RUN
---
python phase2/bellman_equation.py

======================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


# ======================================================================
# STEP 1. MDP DATA STRUCTURE
# ======================================================================

@dataclass
class MDP:
    """
    Finite MDP with transition-dependent rewards.

    P[a, s, s_next]:
        Probability of moving from s to s_next using action a.

    R[a, s, s_next]:
        Immediate reward for that transition.

    policy[s, a]:
        Probability of selecting action a in state s.

    terminal[s]:
        True if s is terminal.

    grid_shape:
        Optional dimensions used for visualization.
    """

    P: np.ndarray
    R: np.ndarray
    policy: np.ndarray
    terminal: np.ndarray
    state_names: List[str]
    action_names: List[str]
    grid_shape: Tuple[int, int] | None = None
    obstacle_states: Tuple[int, ...] = ()

    @property
    def num_actions(self) -> int:
        return self.P.shape[0]

    @property
    def num_states(self) -> int:
        return self.P.shape[1]


# ======================================================================
# STEP 2. BUILD A GRIDWORLD MDP
# ======================================================================

def build_gridworld() -> MDP:
    """
    Construct a 4x4 stochastic GridWorld.

    Actions:
        0 = Up
        1 = Right
        2 = Down
        3 = Left

    Movement:
        80% intended direction
        10% perpendicular direction 1
        10% perpendicular direction 2

    Rewards:
        +1.0 when entering the goal.
        -0.04 for an ordinary movement.

    Terminal:
        Goal = (3, 3)

    Obstacle:
        (1, 1)

    If an action would move into a wall or outside the grid,
    the agent remains in its current state.

    The goal is terminal. Its value is defined as zero because
    the reward for reaching it has already been received.

    This convention avoids counting the goal reward twice.
    """

    rows = 4
    cols = 4

    num_states = rows * cols
    num_actions = 4

    goal = (3, 3)
    obstacle = (1, 1)

    goal_state = goal[0] * cols + goal[1]
    obstacle_state = obstacle[0] * cols + obstacle[1]

    terminal = np.zeros(num_states, dtype=bool)
    terminal[goal_state] = True

    # An obstacle is not a valid decision state.
    # We mark it terminal so its value is fixed at zero.
    terminal[obstacle_state] = True

    action_names = ["Up", "Right", "Down", "Left"]

    # (row movement, column movement)
    directions = [
        (-1, 0),
        (0, 1),
        (1, 0),
        (0, -1),
    ]

    P = np.zeros(
        (num_actions, num_states, num_states),
        dtype=float
    )

    R = np.zeros_like(P)

    def state_to_position(state: int) -> Tuple[int, int]:
        return divmod(state, cols)

    def position_to_state(
        row: int,
        col: int
    ) -> int:
        return row * cols + col

    def next_state(
        state: int,
        action: int
    ) -> int:

        row, col = state_to_position(state)

        dr, dc = directions[action]

        next_row = row + dr
        next_col = col + dc

        # Boundary condition.
        if not (0 <= next_row < rows):
            return state

        if not (0 <= next_col < cols):
            return state

        # Obstacle condition.
        if (next_row, next_col) == obstacle:
            return state

        return position_to_state(next_row, next_col)

    for state in range(num_states):

        if terminal[state]:

            # Terminal states are absorbing.
            for action in range(num_actions):
                P[action, state, state] = 1.0
                R[action, state, state] = 0.0

            continue

        for action in range(num_actions):

            # Intended direction.
            possible_actions = [
                (action, 0.8),

                # Perpendicular directions.
                ((action - 1) % 4, 0.1),
                ((action + 1) % 4, 0.1),
            ]

            for actual_action, probability in possible_actions:

                destination = next_state(
                    state,
                    actual_action
                )

                P[action, state, destination] += probability

                if destination == goal_state:
                    reward = 1.0
                else:
                    reward = -0.04

                R[action, state, destination] = reward

    # Uniform random policy:
    # Every action has probability 0.25.
    policy = np.full(
        (num_states, num_actions),
        1.0 / num_actions,
        dtype=float
    )

    state_names = [
        f"({r},{c})"
        for r in range(rows)
        for c in range(cols)
    ]

    return MDP(
        P=P,
        R=R,
        policy=policy,
        terminal=terminal,
        state_names=state_names,
        action_names=action_names,
        grid_shape=(rows, cols),
        obstacle_states=(obstacle_state,)
    )


# ======================================================================
# STEP 3. POLICY VALIDATION
# ======================================================================

def validate_policy(mdp: MDP) -> None:
    """
    A stochastic policy must satisfy:

        pi(a|s) >= 0

        sum_a pi(a|s) = 1

    Terminal states also have valid probability distributions,
    although their values are fixed at zero.
    """

    if np.any(mdp.policy < -1e-12):
        raise ValueError("Policy contains negative probabilities.")

    row_sums = np.sum(mdp.policy, axis=1)

    if not np.allclose(row_sums, 1.0):
        raise ValueError(
            "Policy probabilities must sum to 1 for every state."
        )

    if np.any(mdp.P < -1e-12):
        raise ValueError("Transition probabilities cannot be negative.")

    transition_sums = np.sum(mdp.P, axis=2)

    if not np.allclose(transition_sums, 1.0):
        raise ValueError(
            "Transition probabilities must sum to 1."
        )


# ======================================================================
# STEP 4. EXPECTED REWARD AND TRANSITION MATRIX UNDER A POLICY
# ======================================================================

def calculate_policy_model(
    mdp: MDP
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert the action-dependent MDP into a policy-dependent model.

    P_pi[s, s_next]
        = sum_a pi(a|s) * P(s_next|s,a)

    r_pi[s]
        = sum_a pi(a|s)
          * sum_s_next P(s_next|s,a) * R(s,a,s_next)

    Therefore:

        V_pi = r_pi + gamma * P_pi * V_pi

    Terminal state values are fixed at zero.
    """

    num_states = mdp.num_states

    P_pi = np.zeros(
        (num_states, num_states),
        dtype=float
    )

    r_pi = np.zeros(num_states, dtype=float)

    for state in range(num_states):

        if mdp.terminal[state]:
            continue

        for action in range(mdp.num_actions):

            action_probability = mdp.policy[state, action]

            P_pi[state] += (
                action_probability
                * mdp.P[action, state]
            )

            expected_reward = np.sum(
                mdp.P[action, state]
                * mdp.R[action, state]
            )

            r_pi[state] += (
                action_probability
                * expected_reward
            )

    return P_pi, r_pi


# ======================================================================
# STEP 5. BELLMAN EXPECTATION BACKUP
# ======================================================================

def bellman_backup(
    mdp: MDP,
    state: int,
    values: np.ndarray,
    gamma: float
) -> float:
    """
    Calculate one Bellman expectation backup.

    Mathematical derivation:

        V_pi(s)
          = E_pi[R_(t+1) + gamma * V_pi(S_(t+1)) | S_t=s]

    Expand the expectation over actions:

        = sum_a pi(a|s)
            E[R_(t+1) + gamma * V_pi(S_(t+1)) | s,a]

    Expand over possible next states:

        = sum_a pi(a|s)
            sum_s' P(s'|s,a)
                [R(s,a,s') + gamma * V_pi(s')]

    This is the Bellman expectation equation.

    A backup uses the current value estimate of next states
    to produce an improved estimate for the selected state.
    """

    if mdp.terminal[state]:
        return 0.0

    result = 0.0

    for action in range(mdp.num_actions):

        action_probability = mdp.policy[state, action]

        for next_state in range(mdp.num_states):

            transition_probability = mdp.P[
                action,
                state,
                next_state
            ]

            reward = mdp.R[
                action,
                state,
                next_state
            ]

            result += (
                action_probability
                * transition_probability
                * (
                    reward
                    + gamma * values[next_state]
                )
            )

    return result


def bellman_operator(
    mdp: MDP,
    values: np.ndarray,
    gamma: float
) -> np.ndarray:
    """
    Apply the Bellman expectation operator to every state.

        T_pi(V) = r_pi + gamma * P_pi * V

    Terminal values remain zero.

    This implementation is synchronous:
    every new value is calculated from the same old value vector.
    """

    new_values = np.zeros_like(values, dtype=float)

    for state in range(mdp.num_states):

        new_values[state] = bellman_backup(
            mdp,
            state,
            values,
            gamma
        )

    return new_values


# ======================================================================
# STEP 6. SYNCHRONOUS ITERATIVE POLICY EVALUATION
# ======================================================================

def iterative_policy_evaluation(
    mdp: MDP,
    gamma: float = 0.90,
    theta: float = 1e-10,
    max_iterations: int = 10000,
    initial_values: np.ndarray | None = None
) -> Tuple[np.ndarray, List[float], int]:
    """
    Synchronous iterative policy evaluation.

    Algorithm:

        Initialize V(s), usually zero.

        Repeat:
            V_new = T_pi(V_old)

            delta = max_s |V_new(s) - V_old(s)|

            V_old = V_new

        Stop when delta < theta.

    Why does it converge?

    For a finite discounted MDP with gamma < 1:

        ||T_pi(V1) - T_pi(V2)||_infinity
          <= gamma * ||V1 - V2||_infinity

    This is the contraction property.

    Because gamma < 1, repeated Bellman updates converge
    to the unique fixed point V_pi.

    theta is the stopping tolerance, not the contraction factor.
    """

    if not (0.0 <= gamma < 1.0):
        raise ValueError("gamma must satisfy 0 <= gamma < 1.")

    if theta <= 0:
        raise ValueError("theta must be positive.")

    if initial_values is None:
        values = np.zeros(mdp.num_states, dtype=float)
    else:
        values = np.asarray(
            initial_values,
            dtype=float
        ).copy()

        if values.shape != (mdp.num_states,):
            raise ValueError("Invalid initial_values shape.")

    values[mdp.terminal] = 0.0

    convergence_history = []

    for iteration in range(1, max_iterations + 1):

        new_values = bellman_operator(
            mdp,
            values,
            gamma
        )

        delta = float(
            np.max(np.abs(new_values - values))
        )

        convergence_history.append(delta)

        values = new_values

        if delta < theta:
            return values, convergence_history, iteration

    return values, convergence_history, max_iterations


# ======================================================================
# STEP 7. ASYNCHRONOUS / IN-PLACE ITERATIVE POLICY EVALUATION
# ======================================================================

def asynchronous_policy_evaluation(
    mdp: MDP,
    gamma: float = 0.90,
    theta: float = 1e-10,
    max_iterations: int = 10000,
    initial_values: np.ndarray | None = None
) -> Tuple[np.ndarray, List[float], int]:
    """
    In-place asynchronous policy evaluation.

    Unlike synchronous evaluation, each updated value is immediately
    available to later state updates during the same sweep.

    Pseudocode:

        for each sweep:
            for each state s:
                V(s) = T_pi(V)(s)

    The order of state updates can influence convergence speed.

    Under standard finite discounted-MDP conditions, a fair update
    schedule that repeatedly updates all relevant states converges.

    Here, every state is updated once per sweep in a fixed order.
    """

    if not (0.0 <= gamma < 1.0):
        raise ValueError("gamma must satisfy 0 <= gamma < 1.")

    if theta <= 0:
        raise ValueError("theta must be positive.")

    if initial_values is None:
        values = np.zeros(mdp.num_states, dtype=float)
    else:
        values = np.asarray(
            initial_values,
            dtype=float
        ).copy()

        if values.shape != (mdp.num_states,):
            raise ValueError("Invalid initial_values shape.")

    values[mdp.terminal] = 0.0

    convergence_history = []

    for iteration in range(1, max_iterations + 1):

        old_values = values.copy()

        for state in range(mdp.num_states):

            values[state] = bellman_backup(
                mdp,
                state,
                values,
                gamma
            )

        values[mdp.terminal] = 0.0

        delta = float(
            np.max(np.abs(values - old_values))
        )

        convergence_history.append(delta)

        if delta < theta:
            return values, convergence_history, iteration

    return values, convergence_history, max_iterations


# ======================================================================
# STEP 8. EXACT MATRIX SOLUTION
# ======================================================================

def exact_policy_evaluation(
    mdp: MDP,
    gamma: float
) -> np.ndarray:
    """
    Solve the Bellman expectation equation directly.

        V = r_pi + gamma * P_pi * V

    Rearrange:

        V - gamma * P_pi * V = r_pi

        (I - gamma * P_pi) V = r_pi

    For nonterminal states:

        A = I - gamma * P_pi_nonterminal

        A * V = r_pi

    Solve using:

        numpy.linalg.solve(A, r_pi)

    Why exclude terminal states?

    Their values are fixed at zero. The reward received when entering
    a terminal state is already included in the preceding transition.

    This gives an independent numerical reference for the iterative
    algorithms.
    """

    if not (0.0 <= gamma < 1.0):
        raise ValueError("gamma must satisfy 0 <= gamma < 1.")

    P_pi, r_pi = calculate_policy_model(mdp)

    nonterminal = np.where(~mdp.terminal)[0]

    values = np.zeros(mdp.num_states, dtype=float)

    if len(nonterminal) == 0:
        return values

    P_reduced = P_pi[
        np.ix_(nonterminal, nonterminal)
    ]

    r_reduced = r_pi[nonterminal]

    identity = np.eye(len(nonterminal))

    A = identity - gamma * P_reduced

    values_reduced = np.linalg.solve(
        A,
        r_reduced
    )

    values[nonterminal] = values_reduced

    return values


# ======================================================================
# STEP 9. ACTION-VALUE FUNCTION
# ======================================================================

def calculate_q_values(
    mdp: MDP,
    values: np.ndarray,
    gamma: float
) -> np.ndarray:
    """
    Calculate Q_pi(s,a).

        Q_pi(s,a)
          = sum_s' P(s'|s,a)
              [R(s,a,s') + gamma * V_pi(s')]

    Important distinction:

        V_pi(s):
            Expected return when starting at s and following policy pi.

        Q_pi(s,a):
            Expected return when taking action a first,
            then following policy pi afterward.

    Their consistency relationship is:

        V_pi(s) = sum_a pi(a|s) * Q_pi(s,a)
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


def calculate_policy_value_from_q(
    mdp: MDP,
    q_values: np.ndarray
) -> np.ndarray:
    """
    Verify:

        V_pi(s) = sum_a pi(a|s) Q_pi(s,a)
    """

    return np.sum(
        mdp.policy * q_values,
        axis=1
    )


# ======================================================================
# STEP 10. BELLMAN RESIDUAL
# ======================================================================

def calculate_bellman_residual(
    mdp: MDP,
    values: np.ndarray,
    gamma: float
) -> np.ndarray:
    """
    Bellman residual:

        residual(s) = T_pi(V)(s) - V(s)

    A true fixed point satisfies:

        T_pi(V_pi) - V_pi = 0

    The infinity norm of the residual is:

        max_s |residual(s)|

    A small residual indicates that the Bellman equation
    is approximately satisfied.
    """

    updated = bellman_operator(
        mdp,
        values,
        gamma
    )

    return updated - values


# ======================================================================
# STEP 11. EXPERIMENTS
# ======================================================================

def compare_initializations(
    mdp: MDP,
    gamma: float
) -> Dict[str, dict]:

    rng = np.random.default_rng(42)

    initializations = {
        "Zero initialization": np.zeros(mdp.num_states),

        "Positive initialization": np.full(
            mdp.num_states,
            5.0
        ),

        "Random initialization": rng.normal(
            loc=0.0,
            scale=3.0,
            size=mdp.num_states
        ),
    }

    results = {}

    for name, initial_values in initializations.items():

        values, history, iterations = (
            iterative_policy_evaluation(
                mdp,
                gamma=gamma,
                initial_values=initial_values
            )
        )

        results[name] = {
            "values": values,
            "history": history,
            "iterations": iterations,
        }

    return results


def compare_gamma_values(
    mdp: MDP,
    gamma_values: List[float]
) -> Dict[float, dict]:

    results = {}

    for gamma in gamma_values:

        sync_values, sync_history, sync_iterations = (
            iterative_policy_evaluation(
                mdp,
                gamma=gamma
            )
        )

        async_values, async_history, async_iterations = (
            asynchronous_policy_evaluation(
                mdp,
                gamma=gamma
            )
        )

        exact_values = exact_policy_evaluation(
            mdp,
            gamma
        )

        results[gamma] = {
            "sync_values": sync_values,
            "sync_history": sync_history,
            "sync_iterations": sync_iterations,

            "async_values": async_values,
            "async_history": async_history,
            "async_iterations": async_iterations,

            "exact_values": exact_values,
        }

    return results


# ======================================================================
# STEP 12. VISUALIZATION FUNCTIONS
# ======================================================================

def values_to_grid(
    mdp: MDP,
    values: np.ndarray
) -> np.ndarray:

    if mdp.grid_shape is None:
        raise ValueError("GridWorld dimensions are not available.")

    grid = values.reshape(mdp.grid_shape).copy()

    for state in mdp.obstacle_states:
        row, col = divmod(state, mdp.grid_shape[1])
        grid[row, col] = np.nan

    return grid


def visualize_value_heatmap(
    mdp: MDP,
    values: np.ndarray,
    title: str
) -> None:

    grid = values_to_grid(mdp, values)

    fig, ax = plt.subplots(figsize=(7, 6))

    image = ax.imshow(
        grid,
        cmap="viridis",
        interpolation="nearest"
    )

    fig.colorbar(
        image,
        ax=ax,
        label="State Value V(s)"
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
                fontsize=9,
                fontweight="bold"
            )

    ax.set_title(title)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")

    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))

    plt.tight_layout()


def visualize_convergence(
    histories: Dict[str, List[float]],
    title: str
) -> None:

    plt.figure(figsize=(9, 5))

    for name, history in histories.items():

        plt.plot(
            range(1, len(history) + 1),
            history,
            label=name
        )

    plt.yscale("log")

    plt.xlabel("Iteration")
    plt.ylabel("Maximum Value Update (log scale)")
    plt.title(title)

    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()


def visualize_exact_vs_iterative(
    exact_values: np.ndarray,
    iterative_values: np.ndarray,
    asynchronous_values: np.ndarray
) -> None:

    x = np.arange(len(exact_values))

    width = 0.25

    plt.figure(figsize=(11, 5))

    plt.bar(
        x - width,
        exact_values,
        width,
        label="Exact"
    )

    plt.bar(
        x,
        iterative_values,
        width,
        label="Synchronous"
    )

    plt.bar(
        x + width,
        asynchronous_values,
        width,
        label="Asynchronous"
    )

    plt.xlabel("State Index")
    plt.ylabel("State Value")

    plt.title("Exact vs Iterative Policy Evaluation")

    plt.xticks(
        x,
        [str(i) for i in x]
    )

    plt.grid(axis="y", alpha=0.3)
    plt.legend()

    plt.tight_layout()


def visualize_bellman_residual(
    mdp: MDP,
    values: np.ndarray,
    gamma: float
) -> None:

    residual = calculate_bellman_residual(
        mdp,
        values,
        gamma
    )

    grid = values_to_grid(
        mdp,
        residual
    )

    fig, ax = plt.subplots(figsize=(7, 6))

    image = ax.imshow(
        grid,
        cmap="coolwarm",
        interpolation="nearest"
    )

    fig.colorbar(
        image,
        ax=ax,
        label="Bellman Residual"
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
                fontsize=8,
                color="black"
            )

    ax.set_title(
        f"Bellman Residual by State (gamma={gamma})"
    )

    ax.set_xlabel("Column")
    ax.set_ylabel("Row")

    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))

    plt.tight_layout()


def visualize_gamma_comparison(
    results: Dict[float, dict],
    mdp: MDP
) -> None:

    plt.figure(figsize=(9, 5))

    for gamma, result in results.items():

        values = result["exact_values"]

        plt.plot(
            np.arange(mdp.num_states),
            values,
            marker="o",
            label=f"gamma={gamma}"
        )

    plt.xlabel("State Index")
    plt.ylabel("State Value")

    plt.title("Effect of Discount Factor on State Values")

    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()


def visualize_policy_q_values(
    mdp: MDP,
    q_values: np.ndarray
) -> None:

    plt.figure(figsize=(9, 5))

    plt.imshow(
        q_values.T,
        cmap="viridis",
        aspect="auto"
    )

    plt.colorbar(label="Q(s,a)")

    plt.yticks(
        range(mdp.num_actions),
        mdp.action_names
    )

    plt.xticks(
        range(mdp.num_states),
        range(mdp.num_states)
    )

    plt.xlabel("State Index")
    plt.ylabel("Action")

    plt.title("Action-Value Function Q_pi(s,a)")

    plt.tight_layout()


# ======================================================================
# STEP 13. UNIT TESTS
# ======================================================================

def run_tests() -> None:

    mdp = build_gridworld()

    validate_policy(mdp)

    gamma = 0.90

    exact_values = exact_policy_evaluation(
        mdp,
        gamma
    )

    sync_values, sync_history, sync_iterations = (
        iterative_policy_evaluation(
            mdp,
            gamma=gamma
        )
    )

    async_values, async_history, async_iterations = (
        asynchronous_policy_evaluation(
            mdp,
            gamma=gamma
        )
    )

    # --------------------------------------------------------------
    # TEST 1: Policy probability sums.
    # --------------------------------------------------------------

    assert np.allclose(
        np.sum(mdp.policy, axis=1),
        1.0
    )

    print("[PASS] Policy probabilities sum to one")

    # --------------------------------------------------------------
    # TEST 2: Transition probability sums.
    # --------------------------------------------------------------

    assert np.allclose(
        np.sum(mdp.P, axis=2),
        1.0
    )

    print("[PASS] Transition probabilities sum to one")

    # --------------------------------------------------------------
    # TEST 3: Terminal values are zero.
    # --------------------------------------------------------------

    assert np.allclose(
        exact_values[mdp.terminal],
        0.0
    )

    print("[PASS] Terminal state values are zero")

    # --------------------------------------------------------------
    # TEST 4: One Bellman backup by hand.
    #
    # State 0 is the top-left corner.
    # Under the uniform policy:
    #   Up and Left stay in state 0.
    #   Right and Down move with stochastic outcomes.
    #
    # Verify the generic backup against an independently
    # calculated expression using P, R, policy, and values.
    # --------------------------------------------------------------

    test_values = np.arange(
        mdp.num_states,
        dtype=float
    ) / 10.0

    state = 0

    manual_backup = 0.0

    for action in range(mdp.num_actions):

        for next_state in range(mdp.num_states):

            manual_backup += (
                mdp.policy[state, action]
                * mdp.P[action, state, next_state]
                * (
                    mdp.R[action, state, next_state]
                    + gamma * test_values[next_state]
                )
            )

    implemented_backup = bellman_backup(
        mdp,
        state,
        test_values,
        gamma
    )

    assert np.isclose(
        implemented_backup,
        manual_backup,
        atol=1e-12
    )

    print("[PASS] Single-state Bellman backup")

    # --------------------------------------------------------------
    # TEST 5: Synchronous vs exact.
    # --------------------------------------------------------------

    assert np.allclose(
        sync_values,
        exact_values,
        atol=1e-8
    )

    print("[PASS] Synchronous evaluation matches exact solution")

    # --------------------------------------------------------------
    # TEST 6: Asynchronous vs exact.
    # --------------------------------------------------------------

    assert np.allclose(
        async_values,
        exact_values,
        atol=1e-8
    )

    print("[PASS] Asynchronous evaluation matches exact solution")

    # --------------------------------------------------------------
    # TEST 7: Both methods converge.
    # --------------------------------------------------------------

    assert sync_history[-1] < 1e-10
    assert async_history[-1] < 1e-10

    print("[PASS] Both iterative methods converge")

    # --------------------------------------------------------------
    # TEST 8: Bellman residual near zero.
    # --------------------------------------------------------------

    residual = calculate_bellman_residual(
        mdp,
        exact_values,
        gamma
    )

    assert np.max(np.abs(residual)) < 1e-10

    print("[PASS] Exact solution has near-zero Bellman residual")

    # --------------------------------------------------------------
    # TEST 9: Q-value and V-value consistency.
    # --------------------------------------------------------------

    q_values = calculate_q_values(
        mdp,
        exact_values,
        gamma
    )

    reconstructed_values = calculate_policy_value_from_q(
        mdp,
        q_values
    )

    assert np.allclose(
        reconstructed_values,
        exact_values,
        atol=1e-10
    )

    print("[PASS] V(s) equals policy-weighted Q(s,a)")

    # --------------------------------------------------------------
    # TEST 10: Initial values do not change the final solution.
    # --------------------------------------------------------------

    rng = np.random.default_rng(123)

    random_initial_values = rng.normal(
        size=mdp.num_states
    )

    random_values, _, _ = iterative_policy_evaluation(
        mdp,
        gamma=gamma,
        initial_values=random_initial_values
    )

    assert np.allclose(
        random_values,
        exact_values,
        atol=1e-8
    )

    print("[PASS] Different initial values converge to same solution")

    # --------------------------------------------------------------
    # TEST 11: Invalid gamma rejected.
    # --------------------------------------------------------------

    invalid_gamma_rejected = False

    try:
        iterative_policy_evaluation(
            mdp,
            gamma=1.0
        )
    except ValueError:
        invalid_gamma_rejected = True

    assert invalid_gamma_rejected

    print("[PASS] Invalid gamma is rejected")

    # --------------------------------------------------------------
    # TEST 12: Invalid policy rejected.
    # --------------------------------------------------------------

    invalid_mdp = build_gridworld()

    invalid_mdp.policy[0] = [0.5, 0.5, 0.5, 0.5]

    invalid_policy_rejected = False

    try:
        validate_policy(invalid_mdp)
    except ValueError:
        invalid_policy_rejected = True

    assert invalid_policy_rejected

    print("[PASS] Invalid policy is rejected")

    print("\nAll 12 tests passed.")


# ======================================================================
# STEP 14. MAIN
# ======================================================================

def main() -> None:

    print("=" * 68)
    print("PHASE 2.1 — BELLMAN EXPECTATION & POLICY EVALUATION")
    print("=" * 68)

    mdp = build_gridworld()

    validate_policy(mdp)

    gamma = 0.90

    # --------------------------------------------------------------
    # A. Exact matrix solution.
    # --------------------------------------------------------------

    exact_values = exact_policy_evaluation(
        mdp,
        gamma
    )

    # --------------------------------------------------------------
    # B. Synchronous iterative evaluation.
    # --------------------------------------------------------------

    sync_values, sync_history, sync_iterations = (
        iterative_policy_evaluation(
            mdp,
            gamma=gamma
        )
    )

    # --------------------------------------------------------------
    # C. Asynchronous iterative evaluation.
    # --------------------------------------------------------------

    async_values, async_history, async_iterations = (
        asynchronous_policy_evaluation(
            mdp,
            gamma=gamma
        )
    )

    print("\n[1] ITERATION RESULTS")

    print(
        f"Synchronous iterations : {sync_iterations}"
    )

    print(
        f"Asynchronous iterations: {async_iterations}"
    )

    print(
        "Maximum difference, synchronous vs exact:",
        np.max(np.abs(sync_values - exact_values))
    )

    print(
        "Maximum difference, asynchronous vs exact:",
        np.max(np.abs(async_values - exact_values))
    )

    # --------------------------------------------------------------
    # D. Print state values.
    # --------------------------------------------------------------

    print("\n[2] EXACT STATE VALUES")

    for state, value in enumerate(exact_values):

        print(
            f"State {state:2d} "
            f"{mdp.state_names[state]:>6s} "
            f"V(s)={value: .6f}"
        )

    # --------------------------------------------------------------
    # E. Calculate action values.
    # --------------------------------------------------------------

    q_values = calculate_q_values(
        mdp,
        exact_values,
        gamma
    )

    print("\n[3] ACTION VALUES")

    for state in range(mdp.num_states):

        if mdp.terminal[state]:
            continue

        q_text = ", ".join(
            f"{mdp.action_names[action]}="
            f"{q_values[state, action]:.4f}"
            for action in range(mdp.num_actions)
        )

        print(
            f"State {state:2d}: {q_text}"
        )

    # --------------------------------------------------------------
    # F. Bellman residual.
    # --------------------------------------------------------------

    residual = calculate_bellman_residual(
        mdp,
        exact_values,
        gamma
    )

    print("\n[4] BELLMAN RESIDUAL")

    print(
        "Maximum absolute residual:",
        f"{np.max(np.abs(residual)):.12e}"
    )

    # --------------------------------------------------------------
    # G. Different initial value estimates.
    # --------------------------------------------------------------

    initialization_results = compare_initializations(
        mdp,
        gamma
    )

    print("\n[5] EFFECT OF INITIAL VALUES")

    for name, result in initialization_results.items():

        print(
            f"{name}: "
            f"{result['iterations']} iterations, "
            f"max difference from exact = "
            f"{np.max(np.abs(result['values'] - exact_values)):.3e}"
        )

    # --------------------------------------------------------------
    # H. Discount-factor experiment.
    # --------------------------------------------------------------

    gamma_results = compare_gamma_values(
        mdp,
        [0.50, 0.90, 0.99]
    )

    print("\n[6] EFFECT OF DISCOUNT FACTOR")

    for discount, result in gamma_results.items():

        print(
            f"gamma={discount:.2f}: "
            f"synchronous iterations="
            f"{result['sync_iterations']}, "
            f"asynchronous iterations="
            f"{result['async_iterations']}, "
            f"start value="
            f"{result['exact_values'][0]:.6f}"
        )

    # --------------------------------------------------------------
    # I. Unit tests.
    # --------------------------------------------------------------

    print("\n[7] UNIT TESTS")

    run_tests()

    # --------------------------------------------------------------
    # J. VISUALIZATIONS
    #
    # All visualization functions are called directly here.
    # --------------------------------------------------------------

    visualize_value_heatmap(
        mdp,
        exact_values,
        f"GridWorld State Values (gamma={gamma})"
    )

    visualize_convergence(
        {
            "Synchronous": sync_history,
            "Asynchronous": async_history,
        },
        "Policy Evaluation Convergence"
    )

    visualize_exact_vs_iterative(
        exact_values,
        sync_values,
        async_values
    )

    visualize_bellman_residual(
        mdp,
        exact_values,
        gamma
    )

    visualize_gamma_comparison(
        gamma_results,
        mdp
    )

    visualize_policy_q_values(
        mdp,
        q_values
    )

    # Compare convergence from different initial values.
    visualize_convergence(
        {
            name: result["history"]
            for name, result in initialization_results.items()
        },
        "Convergence from Different Initial Value Estimates"
    )

    plt.show()

    print("\nLesson completed successfully.")


if __name__ == "__main__":
    main()
