
"""
======================================================================
RL COURSE — PHASE 2.4
POLICY ITERATION VS. VALUE ITERATION
======================================================================

LESSON ROADMAP
--------------

STEP 1 — Understand the Algorithms
Purpose:
    Compare Policy Iteration (PI) and Value Iteration (VI).

What you learn:
    PI alternates policy evaluation and policy improvement.
    VI repeatedly applies the Bellman optimality operator.

Implementation:
    Implement both algorithms using the same stochastic GridWorld.

STEP 2 — Policy Evaluation
Purpose:
    Calculate V_pi for a fixed policy.

What you learn:
    V_pi(s) = sum_a pi(a|s) sum_s' P(s'|s,a)
              [R(s,a,s') + gamma V_pi(s')]

Implementation:
    Implement iterative policy evaluation.

STEP 3 — Policy Iteration
Purpose:
    Find an optimal policy through repeated improvement.

What you learn:
    Policy evaluation + greedy policy improvement.

Implementation:
    Implement PI and record evaluation sweeps.

STEP 4 — Value Iteration
Purpose:
    Find optimal values without fully evaluating intermediate
    policies.

What you learn:
    V_new(s) = max_a sum_s' P(s'|s,a)
               [R(s,a,s') + gamma V_old(s')]

Implementation:
    Implement synchronous Value Iteration.

STEP 5 — Performance Measurement
Purpose:
    Compare computational costs.

What you learn:
    Iteration count alone does not measure execution speed.

Implementation:
    Measure runtime, sweeps, state backups, and action evaluations.

STEP 6 — Scaling Experiments
Purpose:
    Study the influence of state-space size and discount factor.

What you learn:
    Larger state spaces increase computational requirements.
    Gamma close to 1 generally slows convergence to tight accuracy.

Implementation:
    Test several GridWorld sizes and discount factors.

STEP 7 — Numerical Verification
Purpose:
    Verify that PI and VI find equivalent optimal solutions.

What you learn:
    Compare final values, policies, and Bellman residuals.

Implementation:
    Implement numerical tests.

STEP 8 — Visualization
Purpose:
    Understand algorithm convergence and performance.

What you learn:
    Interpret convergence curves, policy maps, and runtime charts.

Implementation:
    All visualization functions are included in this script
    and called directly from main().

IMPORTANT MATHEMATICS
---------------------

Policy Evaluation:

    V_pi(s) = sum_a pi(a|s) sum_s' P(s'|s,a)
              [R(s,a,s') + gamma V_pi(s')]

Policy Improvement:

    pi_new(s) = argmax_a Q_pi(s,a)

Value Iteration:

    V_(k+1)(s) = max_a sum_s' P(s'|s,a)
                 [R(s,a,s') + gamma V_k(s')]

Convergence:

    For a finite discounted MDP with gamma < 1:

    ||T*(V1) - T*(V2)||_infinity
        <= gamma ||V1 - V2||_infinity

Value Iteration stopping criterion:

    delta = ||V_(k+1) - V_k||_infinity

    If delta < theta:

    ||V_(k+1) - V*||_infinity
        <= gamma * theta / (1-gamma)

This is the standard contraction-based error bound.

COMPUTATIONAL COMPARISON
------------------------

Policy Iteration:
    - Alternates evaluation and improvement.
    - Often requires fewer policy improvement rounds.
    - Policy evaluation can be computationally expensive.

Value Iteration:
    - Performs optimality backups.
    - May require more sweeps.
    - Avoids fully evaluating intermediate policies.

Neither algorithm is universally faster.

DEPENDENCIES
------------

    pip install numpy matplotlib

RUN
---

    python phase2/dp_algorithm_comparison.py

======================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
import time

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

    rows: int
    cols: int

    goal_state: int
    obstacle_states: tuple

    @property
    def num_states(self):
        return self.rows * self.cols

    @property
    def num_actions(self):
        return self.P.shape[0]


ACTIONS = [
    (-1, 0),   # Up
    (0, 1),    # Right
    (1, 0),    # Down
    (0, -1),   # Left
]

ACTION_NAMES = [
    "Up", "Right", "Down", "Left"
]

ACTION_SYMBOLS = [
    "↑", "→", "↓", "←"
]


# ======================================================================
# STEP 2. BUILD GRIDWORLD
# ======================================================================

def build_gridworld(
    rows=5,
    cols=5,
    intended_probability=0.80,
    step_reward=-0.04,
    goal_reward=1.0
):
    """
    Build a stochastic GridWorld.

    Transition probabilities:

        Intended direction: p

        Perpendicular directions:
            (1-p)/2 each

    Invalid moves and obstacle collisions:
        Agent remains in its current state.

    Reward:
        Entering goal: goal_reward
        Other transitions: step_reward

    Goal and obstacle states are terminal.
    Their future values are fixed to zero.
    """

    if rows < 2 or cols < 2:
        raise ValueError("Grid dimensions must be at least 2.")

    if not 0 < intended_probability <= 1:
        raise ValueError(
            "intended_probability must be in (0, 1]."
        )

    num_states = rows * cols
    num_actions = 4

    goal = (rows - 1, cols - 1)
    goal_state = goal[0] * cols + goal[1]

    obstacle = (rows // 2, cols // 2)

    if obstacle == goal or obstacle == (0, 0):
        obstacle = None

    obstacle_states = ()

    if obstacle is not None:
        obstacle_states = (
            obstacle[0] * cols + obstacle[1],
        )

    terminal = np.zeros(
        num_states,
        dtype=bool
    )

    terminal[goal_state] = True

    for state in obstacle_states:
        terminal[state] = True

    P = np.zeros(
        (num_actions, num_states, num_states)
    )

    R = np.zeros_like(P)

    def move(state, action):

        row, col = divmod(state, cols)

        dr, dc = ACTIONS[action]

        nr = row + dr
        nc = col + dc

        if not (0 <= nr < rows):
            return state

        if not (0 <= nc < cols):
            return state

        if obstacle is not None and (nr, nc) == obstacle:
            return state

        return nr * cols + nc

    side_probability = (
        1.0 - intended_probability
    ) / 2.0

    for state in range(num_states):

        if terminal[state]:

            for action in range(num_actions):
                P[action, state, state] = 1.0

            continue

        for action in range(num_actions):

            outcomes = [
                (action, intended_probability),
                ((action - 1) % 4, side_probability),
                ((action + 1) % 4, side_probability)
            ]

            for actual_action, probability in outcomes:

                if probability == 0:
                    continue

                destination = move(
                    state,
                    actual_action
                )

                P[action, state, destination] += probability

                reward = (
                    goal_reward
                    if destination == goal_state
                    else step_reward
                )

                R[action, state, destination] = reward

    return MDP(
        P=P,
        R=R,
        terminal=terminal,
        rows=rows,
        cols=cols,
        goal_state=goal_state,
        obstacle_states=obstacle_states
    )


def validate_mdp(mdp):

    assert mdp.P.shape == (
        mdp.num_actions,
        mdp.num_states,
        mdp.num_states
    )

    assert mdp.R.shape == mdp.P.shape

    assert np.all(mdp.P >= 0)

    assert np.allclose(
        np.sum(mdp.P, axis=2),
        1.0
    )


# ======================================================================
# STEP 3. POLICY UTILITIES
# ======================================================================

def uniform_policy(mdp):

    return np.full(
        (mdp.num_states, mdp.num_actions),
        1.0 / mdp.num_actions
    )


def validate_policy(mdp, policy):

    if policy.shape != (
        mdp.num_states,
        mdp.num_actions
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
# STEP 4. BUILD POLICY MODEL
# ======================================================================

def build_policy_model(mdp, policy):
    """
    Construct the policy-induced transition model.

    P_pi(s,s') = sum_a pi(a|s) P(s'|s,a)

    r_pi(s) = sum_a pi(a|s)
              sum_s' P(s'|s,a) R(s,a,s')

    IMPORTANT DIMENSION CONVENTION:

        mdp.P:
            (num_actions, num_states, num_next_states)

        mdp.R:
            (num_actions, num_states, num_next_states)

        policy:
            (num_states, num_actions)

        P_pi:
            (num_states, num_next_states)

        expected_rewards:
            (num_states, num_actions)

        r_pi:
            (num_states,)

    The transpose below is essential because the original
    expected_rewards array has shape (actions, states).
    """

    validate_policy(mdp, policy)

    # P_pi shape: (num_states, num_next_states)
    P_pi = np.einsum(
        "sa,asn->sn",
        policy,
        mdp.P
    )

    # Original shape: (num_actions, num_states)
    # Transpose to: (num_states, num_actions)
    expected_rewards = np.sum(
        mdp.P * mdp.R,
        axis=2
    ).T

    # Both operands now have shape (num_states, num_actions).
    r_pi = np.sum(
        policy * expected_rewards,
        axis=1
    )

    # Terminal states have zero future value and reward.
    P_pi[mdp.terminal] = 0.0
    r_pi[mdp.terminal] = 0.0

    return P_pi, r_pi


# ======================================================================
# STEP 5. ITERATIVE POLICY EVALUATION
# ======================================================================

def evaluate_policy_iterative(
    mdp,
    policy,
    gamma,
    theta=1e-12,
    max_sweeps=100000
):
    """
    Iterative policy evaluation:

        V_new = r_pi + gamma * P_pi @ V_old

    Uses synchronous updates.
    """

    P_pi, r_pi = build_policy_model(
        mdp,
        policy
    )

    values = np.zeros(
        mdp.num_states
    )

    history = []

    for sweep in range(
        1,
        max_sweeps + 1
    ):

        new_values = (
            r_pi
            + gamma * P_pi @ values
        )

        new_values[mdp.terminal] = 0.0

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
                "sweeps": sweep,
                "converged": True
            }

    return {
        "values": values,
        "history": history,
        "sweeps": max_sweeps,
        "converged": False
    }


# ======================================================================
# STEP 6. ACTION-VALUE CALCULATION
# ======================================================================

def calculate_q_values(
    mdp,
    values,
    gamma
):
    """
    Q(s,a) = sum_s' P(s'|s,a)
             [R(s,a,s') + gamma V(s')]
    """

    q_values = np.einsum(
        "asn,asn->sa",
        mdp.P,
        mdp.R + gamma * values[None, None, :]
    )

    q_values[mdp.terminal] = 0.0

    return q_values


# ======================================================================
# STEP 7. GREEDY POLICY IMPROVEMENT
# ======================================================================

def greedy_policy(
    mdp,
    q_values,
    tolerance=1e-10
):
    """
    Select all actions tied for the maximum Q-value.

    If multiple actions have the same maximum value,
    distribute probability equally among them.
    """

    policy = np.zeros_like(q_values)

    for state in range(mdp.num_states):

        if mdp.terminal[state]:

            policy[state] = (
                1.0 / mdp.num_actions
            )

            continue

        best = np.max(
            q_values[state]
        )

        selected = np.flatnonzero(
            np.abs(
                q_values[state] - best
            ) <= tolerance
        )

        policy[state, selected] = (
            1.0 / len(selected)
        )

    return policy


# ======================================================================
# STEP 8. POLICY ITERATION
# ======================================================================

def policy_iteration(
    mdp,
    gamma=0.90,
    evaluation_theta=1e-12,
    max_policy_iterations=1000
):
    """
    Policy Iteration:

        1. Evaluate current policy.
        2. Calculate Q_pi(s,a).
        3. Improve policy greedily.
        4. Check policy stability.
        5. Repeat.

    Metrics:

        policy_iterations:
            Number of policy improvement rounds.

        evaluation_sweeps:
            Total evaluation sweeps.

        state_backups:
            Number of state updates during evaluation.

        action_evaluations:
            Number of state-action evaluations.
    """

    policy = uniform_policy(mdp)

    total_evaluation_sweeps = 0
    total_state_backups = 0
    total_action_evaluations = 0

    history = []

    for iteration in range(
        1,
        max_policy_iterations + 1
    ):

        evaluation = evaluate_policy_iterative(
            mdp,
            policy,
            gamma,
            theta=evaluation_theta
        )

        if not evaluation["converged"]:
            raise RuntimeError(
                "Policy evaluation did not converge."
            )

        values = evaluation["values"]

        sweeps = evaluation["sweeps"]

        total_evaluation_sweeps += sweeps

        total_state_backups += (
            sweeps * mdp.num_states
        )

        q_values = calculate_q_values(
            mdp,
            values,
            gamma
        )

        total_action_evaluations += (
            mdp.num_states * mdp.num_actions
        )

        improved_policy = greedy_policy(
            mdp,
            q_values
        )

        history.append({
            "start_value": float(values[0]),
            "evaluation_sweeps": sweeps,
            "policy_iteration": iteration
        })

        stable = np.allclose(
            policy,
            improved_policy,
            atol=1e-10
        )

        policy = improved_policy

        if stable:

            return {
                "policy": policy,
                "values": values,
                "q_values": q_values,
                "history": history,
                "policy_iterations": iteration,
                "evaluation_sweeps": total_evaluation_sweeps,
                "state_backups": total_state_backups,
                "action_evaluations": total_action_evaluations,
                "converged": True
            }

    return {
        "policy": policy,
        "values": values,
        "q_values": q_values,
        "history": history,
        "policy_iterations": max_policy_iterations,
        "evaluation_sweeps": total_evaluation_sweeps,
        "state_backups": total_state_backups,
        "action_evaluations": total_action_evaluations,
        "converged": False
    }


# ======================================================================
# STEP 9. VALUE ITERATION
# ======================================================================

def value_iteration(
    mdp,
    gamma=0.90,
    theta=1e-10,
    max_sweeps=100000
):
    """
    Value Iteration:

        V_new(s) = max_a Q_V(s,a)

    Stopping criterion:

        delta = max_s |V_new(s) - V_old(s)|

    When delta < theta:

        ||V_new - V*||_infinity
            <= gamma * theta / (1-gamma)

    This is the contraction-based value-error bound.
    """

    values = np.zeros(
        mdp.num_states
    )

    history = []

    for sweep in range(
        1,
        max_sweeps + 1
    ):

        q_values = calculate_q_values(
            mdp,
            values,
            gamma
        )

        new_values = np.max(
            q_values,
            axis=1
        )

        new_values[mdp.terminal] = 0.0

        delta = float(
            np.max(
                np.abs(new_values - values)
            )
        )

        history.append(delta)

        values = new_values

        if delta < theta:

            q_values = calculate_q_values(
                mdp,
                values,
                gamma
            )

            policy = greedy_policy(
                mdp,
                q_values
            )

            return {
                "values": values,
                "q_values": q_values,
                "policy": policy,
                "history": history,
                "sweeps": sweep,
                "action_evaluations": (
                    sweep
                    * mdp.num_states
                    * mdp.num_actions
                ),
                "state_backups": (
                    sweep * mdp.num_states
                ),
                "converged": True
            }

    q_values = calculate_q_values(
        mdp,
        values,
        gamma
    )

    return {
        "values": values,
        "q_values": q_values,
        "policy": greedy_policy(
            mdp,
            q_values
        ),
        "history": history,
        "sweeps": max_sweeps,
        "action_evaluations": (
            max_sweeps
            * mdp.num_states
            * mdp.num_actions
        ),
        "state_backups": (
            max_sweeps * mdp.num_states
        ),
        "converged": False
    }


# ======================================================================
# STEP 10. BELLMAN OPTIMALITY RESIDUAL
# ======================================================================

def bellman_optimality_residual(
    mdp,
    values,
    gamma
):

    q_values = calculate_q_values(
        mdp,
        values,
        gamma
    )

    backup = np.max(
        q_values,
        axis=1
    )

    backup[mdp.terminal] = 0.0

    return backup - values


# ======================================================================
# STEP 11. PERFORMANCE COMPARISON
# ======================================================================

def timed_run(function, *args, **kwargs):

    start = time.perf_counter()

    result = function(
        *args,
        **kwargs
    )

    elapsed = time.perf_counter() - start

    result["runtime_seconds"] = elapsed

    return result


def compare_algorithms(
    mdp,
    gamma=0.90,
    theta=1e-10
):

    pi_result = timed_run(
        policy_iteration,
        mdp,
        gamma,
        evaluation_theta=min(
            theta * 0.01,
            1e-12
        )
    )

    vi_result = timed_run(
        value_iteration,
        mdp,
        gamma,
        theta
    )

    difference = np.max(
        np.abs(
            pi_result["values"]
            - vi_result["values"]
        )
    )

    return {
        "policy_iteration": pi_result,
        "value_iteration": vi_result,
        "max_value_difference": float(difference)
    }


def print_comparison(
    comparison,
    mdp
):

    pi = comparison["policy_iteration"]
    vi = comparison["value_iteration"]

    print("\nALGORITHM COMPARISON")
    print("-" * 75)

    print(
        f"{'Metric':<30}"
        f"{'Policy Iteration':>20}"
        f"{'Value Iteration':>20}"
    )

    print("-" * 75)

    rows = [
        (
            "Converged",
            str(pi["converged"]),
            str(vi["converged"])
        ),
        (
            "Policy rounds / VI sweeps",
            str(pi["policy_iterations"]),
            str(vi["sweeps"])
        ),
        (
            "Evaluation sweeps",
            str(pi["evaluation_sweeps"]),
            "N/A"
        ),
        (
            "State backups",
            str(pi["state_backups"]),
            str(vi["state_backups"])
        ),
        (
            "Action evaluations",
            str(pi["action_evaluations"]),
            str(vi["action_evaluations"])
        ),
        (
            "Runtime (seconds)",
            f"{pi['runtime_seconds']:.6f}",
            f"{vi['runtime_seconds']:.6f}"
        ),
        (
            "Start-state value",
            f"{pi['values'][0]:.8f}",
            f"{vi['values'][0]:.8f}"
        )
    ]

    for name, pi_value, vi_value in rows:

        print(
            f"{name:<30}"
            f"{pi_value:>20}"
            f"{vi_value:>20}"
        )

    print("-" * 75)

    print(
        "Maximum value difference:",
        f"{comparison['max_value_difference']:.3e}"
    )


# ======================================================================
# STEP 12. SCALING EXPERIMENT
# ======================================================================

def run_size_experiment(
    sizes=(4, 5, 7, 9),
    gamma=0.90
):

    results = []

    for size in sizes:

        mdp = build_gridworld(
            rows=size,
            cols=size
        )

        comparison = compare_algorithms(
            mdp,
            gamma=gamma,
            theta=1e-9
        )

        pi = comparison["policy_iteration"]
        vi = comparison["value_iteration"]

        results.append({
            "size": size,
            "states": mdp.num_states,
            "pi_runtime": pi["runtime_seconds"],
            "vi_runtime": vi["runtime_seconds"],
            "pi_sweeps": pi["evaluation_sweeps"],
            "vi_sweeps": vi["sweeps"],
            "value_difference": comparison[
                "max_value_difference"
            ]
        })

        print(
            f"{size}x{size}: "
            f"PI={pi['runtime_seconds']:.5f}s, "
            f"VI={vi['runtime_seconds']:.5f}s, "
            f"difference={comparison['max_value_difference']:.2e}"
        )

    return results


# ======================================================================
# STEP 13. GAMMA EXPERIMENT
# ======================================================================

def run_gamma_experiment(
    mdp,
    gamma_values=(0.50, 0.80, 0.90, 0.99)
):

    results = []

    for gamma in gamma_values:

        result = timed_run(
            value_iteration,
            mdp,
            gamma=gamma,
            theta=1e-10
        )

        results.append({
            "gamma": gamma,
            "iterations": result["sweeps"],
            "runtime": result["runtime_seconds"],
            "start_value": result["values"][0],
            "values": result["values"]
        })

    return results


# ======================================================================
# STEP 14. VISUALIZATION
# ======================================================================

def visualize_convergence(
    pi_result,
    vi_result
):
    """
    Display the algorithms' convergence metrics separately.

    PI:
        Change in starting-state value between policy rounds.

    VI:
        Maximum value update per sweep.

    These are different quantities and should not be interpreted
    as identical convergence metrics.
    """

    pi_values = [
        item["start_value"]
        for item in pi_result["history"]
    ]

    pi_changes = [
        abs(pi_values[i] - pi_values[i - 1])
        for i in range(1, len(pi_values))
    ]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5)
    )

    if pi_changes:

        axes[0].plot(
            range(2, len(pi_values) + 1),
            np.maximum(pi_changes, 1e-16),
            marker="o"
        )

    axes[0].set_yscale("log")
    axes[0].set_title("Policy Iteration: Value Change")
    axes[0].set_xlabel("Policy Improvement Round")
    axes[0].set_ylabel("Absolute Start-Value Change")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(
        range(1, len(vi_result["history"]) + 1),
        np.maximum(vi_result["history"], 1e-16),
        marker="o"
    )

    axes[1].set_yscale("log")
    axes[1].set_title("Value Iteration: Bellman Update")
    axes[1].set_xlabel("Sweep")
    axes[1].set_ylabel("Maximum Value Update")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()


def visualize_value_comparison(
    pi_values,
    vi_values
):

    states = np.arange(
        len(pi_values)
    )

    plt.figure(figsize=(10, 5))

    plt.plot(
        states,
        pi_values,
        marker="o",
        label="Policy Iteration"
    )

    plt.plot(
        states,
        vi_values,
        marker="s",
        linestyle="--",
        label="Value Iteration"
    )

    plt.xlabel("State")
    plt.ylabel("State Value")

    plt.title("Optimal Value Comparison")

    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()


def visualize_runtime(
    scaling_results
):

    sizes = [
        item["states"]
        for item in scaling_results
    ]

    pi_times = [
        item["pi_runtime"]
        for item in scaling_results
    ]

    vi_times = [
        item["vi_runtime"]
        for item in scaling_results
    ]

    plt.figure(figsize=(9, 5))

    plt.plot(
        sizes,
        pi_times,
        marker="o",
        label="Policy Iteration"
    )

    plt.plot(
        sizes,
        vi_times,
        marker="s",
        label="Value Iteration"
    )

    plt.xlabel("Number of States")
    plt.ylabel("Runtime (seconds)")

    plt.title("Runtime vs. State Space Size")

    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()


def visualize_iteration_counts(
    scaling_results
):

    sizes = [
        item["states"]
        for item in scaling_results
    ]

    pi_sweeps = [
        item["pi_sweeps"]
        for item in scaling_results
    ]

    vi_sweeps = [
        item["vi_sweeps"]
        for item in scaling_results
    ]

    plt.figure(figsize=(9, 5))

    plt.plot(
        sizes,
        pi_sweeps,
        marker="o",
        label="PI evaluation sweeps"
    )

    plt.plot(
        sizes,
        vi_sweeps,
        marker="s",
        label="VI sweeps"
    )

    plt.xlabel("Number of States")
    plt.ylabel("Number of Sweeps")

    plt.title("Iteration Work vs. State Space Size")

    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()


def visualize_gamma(
    gamma_results
):

    gammas = [
        item["gamma"]
        for item in gamma_results
    ]

    values = [
        item["start_value"]
        for item in gamma_results
    ]

    iterations = [
        item["iterations"]
        for item in gamma_results
    ]

    fig, ax1 = plt.subplots(
        figsize=(9, 5)
    )

    ax1.plot(
        gammas,
        values,
        marker="o"
    )

    ax1.set_xlabel("Discount Factor (gamma)")
    ax1.set_ylabel("Optimal Start Value")

    ax2 = ax1.twinx()

    ax2.plot(
        gammas,
        iterations,
        marker="s",
        linestyle="--"
    )

    ax2.set_ylabel("Value Iteration Sweeps")

    ax1.set_title("Discount Factor Effects")

    ax1.grid(True, alpha=0.3)

    fig.tight_layout()


def visualize_policy(
    mdp,
    policy,
    title
):

    fig, ax = plt.subplots(
        figsize=(7, 6)
    )

    ax.set_xlim(-0.5, mdp.cols - 0.5)
    ax.set_ylim(mdp.rows - 0.5, -0.5)

    ax.set_xticks(range(mdp.cols))
    ax.set_yticks(range(mdp.rows))

    ax.set_xticks(
        np.arange(-0.5, mdp.cols, 1),
        minor=True
    )

    ax.set_yticks(
        np.arange(-0.5, mdp.rows, 1),
        minor=True
    )

    ax.grid(
        which="minor",
        linewidth=1.5
    )

    for row in range(mdp.rows):

        for col in range(mdp.cols):

            state = row * mdp.cols + col

            if state in mdp.obstacle_states:

                label = "WALL"

            elif state == mdp.goal_state:

                label = "GOAL"

            else:

                selected = np.flatnonzero(
                    policy[state] > 1e-8
                )

                label = "\n".join(
                    ACTION_SYMBOLS[action]
                    for action in selected
                )

            ax.text(
                col,
                row,
                label,
                ha="center",
                va="center",
                fontsize=13
            )

    ax.set_title(title)

    plt.tight_layout()


def visualize_value_grid(
    mdp,
    values,
    title
):

    grid = values.reshape(
        mdp.rows,
        mdp.cols
    ).copy()

    for state in mdp.obstacle_states:

        row, col = divmod(
            state,
            mdp.cols
        )

        grid[row, col] = np.nan

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
        label="V(s)"
    )

    for row in range(mdp.rows):

        for col in range(mdp.cols):

            state = row * mdp.cols + col

            if state in mdp.obstacle_states:
                label = "WALL"

            elif state == mdp.goal_state:
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
                fontsize=9
            )

    ax.set_title(title)

    plt.tight_layout()


# ======================================================================
# STEP 15. UNIT TESTS
# ======================================================================

def run_tests():

    mdp = build_gridworld(
        rows=4,
        cols=4
    )

    gamma = 0.90

    validate_mdp(mdp)

    # TEST 1: Uniform policy.
    policy = uniform_policy(mdp)

    validate_policy(
        mdp,
        policy
    )

    print("[PASS] Uniform policy is valid")

    # TEST 2: Policy model dimensions.
    P_pi, r_pi = build_policy_model(
        mdp,
        policy
    )

    assert P_pi.shape == (
        mdp.num_states,
        mdp.num_states
    )

    assert r_pi.shape == (
        mdp.num_states,
    )

    assert np.allclose(
        P_pi.sum(axis=1)[~mdp.terminal],
        1.0
    )

    print(
        "[PASS] Policy model dimensions and transition probabilities"
    )

    # TEST 3: Expected reward is finite.
    assert np.all(
        np.isfinite(r_pi)
    )

    print("[PASS] Expected reward vector is finite")

    # TEST 4: Policy evaluation.
    evaluation = evaluate_policy_iterative(
        mdp,
        policy,
        gamma
    )

    assert evaluation["converged"]

    print("[PASS] Policy evaluation converges")

    # TEST 5: Value Iteration.
    vi = value_iteration(
        mdp,
        gamma
    )

    assert vi["converged"]

    print("[PASS] Value Iteration converges")

    # TEST 6: Policy Iteration.
    pi = policy_iteration(
        mdp,
        gamma
    )

    assert pi["converged"]

    print("[PASS] Policy Iteration converges")

    # TEST 7: PI and VI optimal values agree.
    assert np.allclose(
        vi["values"],
        pi["values"],
        atol=1e-8
    )

    print("[PASS] PI and VI optimal values agree")

    # TEST 8: Terminal values are zero.
    assert np.allclose(
        vi["values"][mdp.terminal],
        0.0
    )

    print("[PASS] Terminal values are zero")

    # TEST 9: Greedy policy is valid.
    validate_policy(
        mdp,
        vi["policy"]
    )

    print("[PASS] Greedy optimal policy is valid")

    # TEST 10: Bellman optimality residual.
    residual = bellman_optimality_residual(
        mdp,
        vi["values"],
        gamma
    )

    assert np.max(
        np.abs(residual)
    ) < 1e-8

    print("[PASS] Bellman optimality residual is near zero")

    # TEST 11: Different initial values converge to same V*.
    rng = np.random.default_rng(42)

    values = rng.normal(
        size=mdp.num_states
    )

    values[mdp.terminal] = 0.0

    for _ in range(10000):

        q_values = calculate_q_values(
            mdp,
            values,
            gamma
        )

        new_values = np.max(
            q_values,
            axis=1
        )

        new_values[mdp.terminal] = 0.0

        delta = np.max(
            np.abs(new_values - values)
        )

        values = new_values

        if delta < 1e-10:
            break

    assert np.allclose(
        values,
        vi["values"],
        atol=1e-8
    )

    print("[PASS] Different initial values converge to same V*")

    # TEST 12: Evaluating the greedy policy reproduces V*.
    optimal_evaluation = evaluate_policy_iterative(
        mdp,
        vi["policy"],
        gamma
    )

    assert np.allclose(
        optimal_evaluation["values"],
        vi["values"],
        atol=1e-8
    )

    print("[PASS] Greedy policy evaluation reproduces V*")

    # TEST 13: Gamma = 0.
    gamma_zero = value_iteration(
        mdp,
        gamma=0.0
    )

    assert gamma_zero["converged"]

    print("[PASS] Value Iteration works with gamma=0")

    # TEST 14: Invalid transition model is rejected.
    invalid = build_gridworld()

    invalid.P[0, 0, 0] = -1.0

    rejected = False

    try:
        validate_mdp(invalid)
    except AssertionError:
        rejected = True

    assert rejected

    print("[PASS] Invalid transition probabilities rejected")

    # TEST 15: Deterministic transition model.
    deterministic_mdp = build_gridworld(
        rows=4,
        cols=4,
        intended_probability=1.0
    )

    validate_mdp(deterministic_mdp)

    assert np.allclose(
        deterministic_mdp.P.sum(axis=2),
        1.0
    )

    deterministic_vi = value_iteration(
        deterministic_mdp,
        gamma
    )

    assert deterministic_vi["converged"]

    print("[PASS] Deterministic transition model works")

    # TEST 16: Stochastic transition model.
    stochastic_mdp = build_gridworld(
        rows=4,
        cols=4,
        intended_probability=0.70
    )

    stochastic_vi = value_iteration(
        stochastic_mdp,
        gamma
    )

    assert stochastic_vi["converged"]

    print("[PASS] Stochastic transition model works")

    # TEST 17: PI and VI agree on deterministic model.
    deterministic_pi = policy_iteration(
        deterministic_mdp,
        gamma
    )

    assert np.allclose(
        deterministic_pi["values"],
        deterministic_vi["values"],
        atol=1e-8
    )

    print("[PASS] PI and VI agree on deterministic model")

    # TEST 18: PI and VI agree on stochastic model.
    stochastic_pi = policy_iteration(
        stochastic_mdp,
        gamma
    )

    assert np.allclose(
        stochastic_pi["values"],
        stochastic_vi["values"],
        atol=1e-8
    )

    print("[PASS] PI and VI agree on stochastic model")

    print("\nAll 18 tests passed.")


# ======================================================================
# STEP 16. MAIN
# ======================================================================

def main():

    print("=" * 75)
    print("PHASE 2.4 — POLICY ITERATION VS. VALUE ITERATION")
    print("=" * 75)

    # --------------------------------------------------------------
    # A. Main algorithm comparison.
    # --------------------------------------------------------------

    mdp = build_gridworld(
        rows=5,
        cols=5,
        intended_probability=0.80,
        step_reward=-0.04,
        goal_reward=1.0
    )

    validate_mdp(mdp)

    gamma = 0.90

    comparison = compare_algorithms(
        mdp,
        gamma=gamma,
        theta=1e-10
    )

    print_comparison(
        comparison,
        mdp
    )

    pi = comparison["policy_iteration"]
    vi = comparison["value_iteration"]

    # --------------------------------------------------------------
    # B. Scaling experiment.
    # --------------------------------------------------------------

    print("\n[SCALING EXPERIMENT]")

    scaling_results = run_size_experiment(
        sizes=(4, 5, 7, 9),
        gamma=0.90
    )

    # --------------------------------------------------------------
    # C. Discount factor experiment.
    # --------------------------------------------------------------

    print("\n[GAMMA EXPERIMENT]")

    gamma_results = run_gamma_experiment(
        mdp,
        gamma_values=(0.50, 0.80, 0.90, 0.99)
    )

    for result in gamma_results:

        print(
            f"gamma={result['gamma']:.2f}, "
            f"iterations={result['iterations']}, "
            f"runtime={result['runtime']:.6f}s, "
            f"V(start)={result['start_value']:.6f}"
        )

    # --------------------------------------------------------------
    # D. Unit tests.
    # --------------------------------------------------------------

    print("\n[UNIT TESTS]")

    run_tests()

    # --------------------------------------------------------------
    # E. Visualizations.
    # --------------------------------------------------------------

    visualize_convergence(
        pi,
        vi
    )

    visualize_value_comparison(
        pi["values"],
        vi["values"]
    )

    visualize_runtime(
        scaling_results
    )

    visualize_iteration_counts(
        scaling_results
    )

    visualize_gamma(
        gamma_results
    )

    visualize_policy(
        mdp,
        vi["policy"],
        "Optimal Policy — Value Iteration"
    )

    visualize_value_grid(
        mdp,
        vi["values"],
        "Optimal State Values"
    )

    plt.show()

    print("\nPhase 2.4 completed successfully.")


if __name__ == "__main__":
    main()
