
"""
======================================================================
RL COURSE — PHASE 2.5
GENERALIZED POLICY ITERATION AND ASYNCHRONOUS DYNAMIC PROGRAMMING
======================================================================

LESSON ROADMAP
--------------

STEP 1 — Generalized Policy Iteration (GPI)
Purpose:
    Understand how policy evaluation and policy improvement interact.

What you learn:
    Evaluation makes V more accurate for the current policy.
    Improvement makes the policy more greedy with respect to V.

Implementation:
    Alternate partial evaluation and greedy improvement.

STEP 2 — Modified Policy Iteration
Purpose:
    Avoid fully evaluating each intermediate policy.

What you learn:
    A fixed number of evaluation sweeps can be sufficient
    before improving a policy.

Implementation:
    Implement modified policy iteration with configurable
    evaluation sweeps.

STEP 3 — Asynchronous Policy Evaluation
Purpose:
    Update states in a flexible order rather than synchronously.

What you learn:
    States do not have to be updated simultaneously.
    Frequently visited or important states can be updated first.

Implementation:
    Implement in-place Gauss-Seidel-style policy evaluation.

STEP 4 — Asynchronous Value Iteration
Purpose:
    Update selected states instead of sweeping the entire MDP.

What you learn:
    An update schedule can influence convergence speed.
    All relevant states must continue receiving sufficient updates.

Implementation:
    Implement cyclic and randomized state update schedules.

STEP 5 — Prioritized Sweeping
Purpose:
    Focus computation on states with large Bellman errors.

What you learn:
    A large Bellman residual indicates that a state value
    is inconsistent with its Bellman backup.

Implementation:
    Maintain a priority queue and update states with large residuals.

STEP 6 — Algorithm Comparison
Purpose:
    Compare convergence and computational work.

What you learn:
    Different update schedules can reach the same solution
    with different amounts of computation.

Implementation:
    Compare synchronous VI, asynchronous VI, GPI,
    modified PI, and prioritized sweeping.

STEP 7 — Numerical Verification
Purpose:
    Verify algorithm correctness.

What you learn:
    Different valid DP update schedules should converge
    to the same optimal values for a finite discounted MDP.

Implementation:
    Unit tests compare values, policies, and Bellman residuals.

STEP 8 — Visualization
Purpose:
    Observe convergence and state-update behavior.

What you learn:
    Understand why selective updates can be useful.

Implementation:
    All visualization functions are included and called
    directly from main().

----------------------------------------------------------------------
THEORY AND MATHEMATICAL FOUNDATION
----------------------------------------------------------------------

1. GENERALIZED POLICY ITERATION

Two processes interact:

    Policy Evaluation:
        V -> V_pi

    Policy Improvement:
        pi -> greedy(V)

GPI does not require either process to finish completely
before the other begins.

2. BELLMAN EXPECTATION OPERATOR

    T_pi V(s) =
        sum_a pi(a|s) sum_s' P(s'|s,a)
        [R(s,a,s') + gamma V(s')]

3. BELLMAN OPTIMALITY OPERATOR

    T* V(s) =
        max_a sum_s' P(s'|s,a)
        [R(s,a,s') + gamma V(s')]

4. MODIFIED POLICY ITERATION

    Perform m policy-evaluation sweeps,
    then improve the policy.

    m = 1:
        Similar to Value Iteration when using greedy policies.

    m -> infinity:
        Approaches fully evaluated Policy Iteration.

5. ASYNCHRONOUS UPDATES

    Update selected states in-place:

        V(s) <- T_pi V(s)

    Other states remain unchanged during that update.

    Under suitable update-coverage conditions and gamma < 1,
    asynchronous dynamic programming converges.

6. PRIORITIZED SWEEPING

    Priority(s) = |T*V(s) - V(s)|

    Update states with the largest residual first.

    When a state's value changes, its predecessors may become
    more important and should be reconsidered.

7. COMPUTATIONAL COMPLEXITY

    Dense transition representation:

        One full Bellman sweep:
            O(S^2 A)

    Sparse transition representation:

        One full sweep:
            O(E A)

    where:
        S = number of states
        A = number of actions
        E = number of nonzero state transitions per action,
            approximately, depending on representation.

    Prioritized sweeping adds queue-management overhead.

    It can save work when only a small subset of states
    has significant Bellman errors.

----------------------------------------------------------------------
DEPENDENCIES
----------------------------------------------------------------------

    pip install numpy matplotlib

RUN
---

    python phase2/generalized_policy_iteration.py

======================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
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
    (-1, 0),
    (0, 1),
    (1, 0),
    (0, -1)
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

    if rows < 2 or cols < 2:
        raise ValueError("Grid dimensions must be at least 2.")

    if not 0 < intended_probability <= 1:
        raise ValueError(
            "intended_probability must be in (0, 1]."
        )

    num_states = rows * cols
    num_actions = 4

    goal = (rows - 1, cols - 1)

    goal_state = (
        goal[0] * cols + goal[1]
    )

    obstacle = (
        rows // 2,
        cols // 2
    )

    if obstacle in ((0, 0), goal):
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

        row, col = divmod(
            state,
            cols
        )

        dr, dc = ACTIONS[action]

        nr = row + dr
        nc = col + dc

        if not (
            0 <= nr < rows
            and 0 <= nc < cols
        ):
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


# ======================================================================
# STEP 3. VALIDATION AND POLICY UTILITIES
# ======================================================================

def validate_mdp(mdp):

    assert mdp.P.shape == (
        mdp.num_actions,
        mdp.num_states,
        mdp.num_states
    )

    assert mdp.R.shape == mdp.P.shape

    assert np.all(mdp.P >= 0)

    assert np.allclose(
        mdp.P.sum(axis=2),
        1.0
    )


def uniform_policy(mdp):

    return np.full(
        (mdp.num_states, mdp.num_actions),
        1.0 / mdp.num_actions
    )


def validate_policy(mdp, policy):

    assert policy.shape == (
        mdp.num_states,
        mdp.num_actions
    )

    assert np.all(policy >= -1e-12)

    assert np.allclose(
        policy.sum(axis=1),
        1.0
    )


# ======================================================================
# STEP 4. POLICY MODEL
# ======================================================================

def build_policy_model(
    mdp,
    policy
):

    validate_policy(
        mdp,
        policy
    )

    # P_pi shape: (states, next_states)
    P_pi = np.einsum(
        "sa,asn->sn",
        policy,
        mdp.P
    )

    # First shape: (actions, states)
    # Transposed shape: (states, actions)
    expected_rewards = np.sum(
        mdp.P * mdp.R,
        axis=2
    ).T

    r_pi = np.sum(
        policy * expected_rewards,
        axis=1
    )

    P_pi[mdp.terminal] = 0.0
    r_pi[mdp.terminal] = 0.0

    return P_pi, r_pi


# ======================================================================
# STEP 5. BELLMAN OPERATORS
# ======================================================================

def policy_backup(
    mdp,
    state,
    values,
    policy,
    gamma
):

    if mdp.terminal[state]:
        return 0.0

    expected_rewards = np.sum(
        mdp.P[:, state, :]
        * mdp.R[:, state, :],
        axis=1
    )

    expected_next_values = (
        mdp.P[:, state, :] @ values
    )

    action_values = (
        expected_rewards
        + gamma * expected_next_values
    )

    return float(
        policy[state] @ action_values
    )


def optimality_backup(
    mdp,
    state,
    values,
    gamma
):

    if mdp.terminal[state]:
        return 0.0

    expected_rewards = np.sum(
        mdp.P[:, state, :]
        * mdp.R[:, state, :],
        axis=1
    )

    expected_next_values = (
        mdp.P[:, state, :] @ values
    )

    action_values = (
        expected_rewards
        + gamma * expected_next_values
    )

    return float(
        np.max(action_values)
    )


def calculate_q_values(
    mdp,
    values,
    gamma
):

    q_values = np.einsum(
        "asn,asn->sa",
        mdp.P,
        mdp.R + gamma * values[None, None, :]
    )

    q_values[mdp.terminal] = 0.0

    return q_values


def greedy_policy(
    mdp,
    q_values,
    tolerance=1e-10
):

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


def policy_from_values(
    mdp,
    values,
    gamma
):

    q_values = calculate_q_values(
        mdp,
        values,
        gamma
    )

    return greedy_policy(
        mdp,
        q_values
    )


# ======================================================================
# STEP 6. EXACT POLICY EVALUATION
# ======================================================================

def evaluate_policy_exact(
    mdp,
    policy,
    gamma
):

    P_pi, r_pi = build_policy_model(
        mdp,
        policy
    )

    nonterminal = np.flatnonzero(
        ~mdp.terminal
    )

    values = np.zeros(
        mdp.num_states
    )

    if len(nonterminal) == 0:
        return values

    P_reduced = P_pi[
        np.ix_(nonterminal, nonterminal)
    ]

    system = (
        np.eye(len(nonterminal))
        - gamma * P_reduced
    )

    values[nonterminal] = np.linalg.solve(
        system,
        r_pi[nonterminal]
    )

    return values


# ======================================================================
# STEP 7. STANDARD VALUE ITERATION
# ======================================================================

def value_iteration(
    mdp,
    gamma=0.90,
    theta=1e-10,
    max_sweeps=100000
):

    values = np.zeros(
        mdp.num_states
    )

    history = []

    for sweep in range(
        1,
        max_sweeps + 1
    ):

        old_values = values.copy()

        new_values = np.zeros_like(values)

        for state in range(mdp.num_states):

            new_values[state] = optimality_backup(
                mdp,
                state,
                old_values,
                gamma
            )

        delta = float(
            np.max(
                np.abs(new_values - old_values)
            )
        )

        history.append(delta)

        values = new_values

        if delta < theta:

            policy = policy_from_values(
                mdp,
                values,
                gamma
            )

            return {
                "values": values,
                "policy": policy,
                "history": history,
                "sweeps": sweep,
                "converged": True
            }

    return {
        "values": values,
        "policy": policy_from_values(
            mdp,
            values,
            gamma
        ),
        "history": history,
        "sweeps": max_sweeps,
        "converged": False
    }


# ======================================================================
# STEP 8. GENERALIZED POLICY ITERATION
# ======================================================================

def generalized_policy_iteration(
    mdp,
    gamma=0.90,
    theta=1e-10,
    evaluation_sweeps=1,
    max_iterations=100000
):
    """
    GPI alternates:

        Partial policy evaluation.
        Greedy policy improvement.

    Unlike standard PI, policy evaluation is intentionally
    incomplete.

    The value function is updated in-place using the
    current policy's Bellman expectation operator.

    After each evaluation block, the policy is improved
    greedily with respect to the current value estimate.
    """

    if evaluation_sweeps < 1:
        raise ValueError(
            "evaluation_sweeps must be >= 1."
        )

    values = np.zeros(
        mdp.num_states
    )

    policy = uniform_policy(mdp)

    history = []

    total_evaluation_updates = 0
    total_policy_updates = 0

    for iteration in range(
        1,
        max_iterations + 1
    ):

        # ----------------------------------------------------------
        # A. Partial policy evaluation.
        # ----------------------------------------------------------

        for _ in range(evaluation_sweeps):

            old_values = values.copy()

            for state in range(mdp.num_states):

                values[state] = policy_backup(
                    mdp,
                    state,
                    values,
                    policy,
                    gamma
                )

            total_evaluation_updates += (
                mdp.num_states
            )

        # ----------------------------------------------------------
        # B. Policy improvement.
        # ----------------------------------------------------------

        q_values = calculate_q_values(
            mdp,
            values,
            gamma
        )

        improved_policy = greedy_policy(
            mdp,
            q_values
        )

        policy_change = np.max(
            np.abs(
                improved_policy - policy
            )
        )

        policy = improved_policy

        total_policy_updates += (
            mdp.num_states
        )

        # ----------------------------------------------------------
        # C. Measure optimality residual.
        # ----------------------------------------------------------

        backup_values = np.array([
            optimality_backup(
                mdp,
                state,
                values,
                gamma
            )
            for state in range(mdp.num_states)
        ])

        residual = np.max(
            np.abs(backup_values - values)
        )

        history.append({
            "residual": float(residual),
            "policy_change": float(policy_change),
            "start_value": float(values[0])
        })

        if residual < theta:

            # A final greedy policy is returned.
            policy = policy_from_values(
                mdp,
                values,
                gamma
            )

            return {
                "values": values,
                "policy": policy,
                "history": history,
                "iterations": iteration,
                "evaluation_updates": total_evaluation_updates,
                "policy_updates": total_policy_updates,
                "converged": True
            }

    return {
        "values": values,
        "policy": policy,
        "history": history,
        "iterations": max_iterations,
        "evaluation_updates": total_evaluation_updates,
        "policy_updates": total_policy_updates,
        "converged": False
    }


# ======================================================================
# STEP 9. MODIFIED POLICY ITERATION
# ======================================================================

def modified_policy_iteration(
    mdp,
    gamma=0.90,
    theta=1e-10,
    evaluation_sweeps=5,
    max_iterations=100000
):
    """
    Modified Policy Iteration:

        Repeat:
            1. Perform m policy-evaluation sweeps.
            2. Improve the policy greedily.

    The number of evaluation sweeps controls the algorithm.

        m = 1:
            Closely related to Value Iteration.

        Larger m:
            More complete policy evaluation.

        Very large m:
            Approaches conventional Policy Iteration.

    This implementation uses synchronous evaluation sweeps.
    """

    if evaluation_sweeps < 1:
        raise ValueError(
            "evaluation_sweeps must be >= 1."
        )

    values = np.zeros(
        mdp.num_states
    )

    policy = uniform_policy(mdp)

    history = []

    total_evaluation_updates = 0

    for iteration in range(
        1,
        max_iterations + 1
    ):

        P_pi, r_pi = build_policy_model(
            mdp,
            policy
        )

        # ----------------------------------------------------------
        # A. Perform m synchronous evaluation sweeps.
        # ----------------------------------------------------------

        for _ in range(evaluation_sweeps):

            new_values = (
                r_pi
                + gamma * P_pi @ values
            )

            new_values[mdp.terminal] = 0.0

            values = new_values

            total_evaluation_updates += (
                mdp.num_states
            )

        # ----------------------------------------------------------
        # B. Improve policy.
        # ----------------------------------------------------------

        q_values = calculate_q_values(
            mdp,
            values,
            gamma
        )

        policy = greedy_policy(
            mdp,
            q_values
        )

        # ----------------------------------------------------------
        # C. Optimality residual.
        # ----------------------------------------------------------

        backup_values = np.array([
            optimality_backup(
                mdp,
                state,
                values,
                gamma
            )
            for state in range(mdp.num_states)
        ])

        residual = np.max(
            np.abs(backup_values - values)
        )

        history.append(
            float(residual)
        )

        if residual < theta:

            return {
                "values": values,
                "policy": policy,
                "history": history,
                "iterations": iteration,
                "evaluation_updates": total_evaluation_updates,
                "converged": True
            }

    return {
        "values": values,
        "policy": policy,
        "history": history,
        "iterations": max_iterations,
        "evaluation_updates": total_evaluation_updates,
        "converged": False
    }


# ======================================================================
# STEP 10. ASYNCHRONOUS POLICY EVALUATION
# ======================================================================

def asynchronous_policy_evaluation(
    mdp,
    policy,
    gamma=0.90,
    theta=1e-10,
    max_updates=100000,
    schedule="cyclic"
):
    """
    In-place policy evaluation.

    Unlike synchronous evaluation:

        All states are not updated from the same old value vector.

    A state is updated immediately, so later states may use
    its newly updated value.

    This is similar to Gauss-Seidel iteration.

    Supported schedules:

        cyclic:
            Update states in numerical order.

        random:
            Select a nonterminal state randomly.

    Terminal states remain zero.
    """

    P_pi, r_pi = build_policy_model(
        mdp,
        policy
    )

    values = np.zeros(
        mdp.num_states
    )

    active_states = np.flatnonzero(
        ~mdp.terminal
    )

    if len(active_states) == 0:

        return {
            "values": values,
            "history": [],
            "updates": 0,
            "converged": True
        }

    rng = np.random.default_rng(42)

    history = []

    for update in range(
        1,
        max_updates + 1
    ):

        if schedule == "cyclic":

            state = active_states[
                (update - 1) % len(active_states)
            ]

        elif schedule == "random":

            state = int(
                rng.choice(active_states)
            )

        else:

            raise ValueError(
                "schedule must be cyclic or random."
            )

        old_value = values[state]

        values[state] = (
            r_pi[state]
            + gamma * P_pi[state] @ values
        )

        change = abs(
            values[state] - old_value
        )

        # Check a full Bellman residual periodically.
        if update % len(active_states) == 0:

            residual = 0.0

            for s in active_states:

                backup = (
                    r_pi[s]
                    + gamma * P_pi[s] @ values
                )

                residual = max(
                    residual,
                    abs(backup - values[s])
                )

            history.append(
                float(residual)
            )

            if residual < theta:

                return {
                    "values": values,
                    "history": history,
                    "updates": update,
                    "converged": True
                }

    return {
        "values": values,
        "history": history,
        "updates": max_updates,
        "converged": False
    }


# ======================================================================
# STEP 11. ASYNCHRONOUS VALUE ITERATION
# ======================================================================

def asynchronous_value_iteration(
    mdp,
    gamma=0.90,
    theta=1e-10,
    max_updates=100000,
    schedule="cyclic"
):
    """
    Update one selected state at a time:

        V(s) <- T*V(s)

    The update is in-place.

    Cyclic and randomized schedules are supported.

    For cyclic scheduling, every nonterminal state is updated
    once per cycle.

    The stopping condition uses the full Bellman optimality
    residual, not merely the last single-state update.
    """

    values = np.zeros(
        mdp.num_states
    )

    active_states = np.flatnonzero(
        ~mdp.terminal
    )

    if len(active_states) == 0:

        return {
            "values": values,
            "policy": uniform_policy(mdp),
            "history": [],
            "updates": 0,
            "converged": True
        }

    rng = np.random.default_rng(42)

    history = []

    for update in range(
        1,
        max_updates + 1
    ):

        if schedule == "cyclic":

            state = active_states[
                (update - 1) % len(active_states)
            ]

        elif schedule == "random":

            state = int(
                rng.choice(active_states)
            )

        else:

            raise ValueError(
                "schedule must be cyclic or random."
            )

        values[state] = optimality_backup(
            mdp,
            state,
            values,
            gamma
        )

        if update % len(active_states) == 0:

            residual = max(
                abs(
                    optimality_backup(
                        mdp,
                        s,
                        values,
                        gamma
                    ) - values[s]
                )
                for s in active_states
            )

            history.append(
                float(residual)
            )

            if residual < theta:

                policy = policy_from_values(
                    mdp,
                    values,
                    gamma
                )

                return {
                    "values": values,
                    "policy": policy,
                    "history": history,
                    "updates": update,
                    "converged": True
                }

    return {
        "values": values,
        "policy": policy_from_values(
            mdp,
            values,
            gamma
        ),
        "history": history,
        "updates": max_updates,
        "converged": False
    }


# ======================================================================
# STEP 12. PRIORITIZED SWEEPING
# ======================================================================

def build_predecessors(mdp):
    """
    predecessors[s] contains all states that can transition
    to state s with nonzero probability under at least one action.
    """

    predecessors = [
        set()
        for _ in range(mdp.num_states)
    ]

    for state in range(mdp.num_states):

        if mdp.terminal[state]:
            continue

        for action in range(mdp.num_actions):

            next_states = np.flatnonzero(
                mdp.P[action, state] > 0
            )

            for next_state in next_states:

                predecessors[next_state].add(
                    state
                )

    return predecessors


def prioritized_sweeping(
    mdp,
    gamma=0.90,
    theta=1e-10,
    max_updates=100000
):
    """
    Prioritized Sweeping:

        1. Calculate each state's Bellman optimality residual.
        2. Insert high-residual states into a max-priority queue.
        3. Update the highest-priority state.
        4. Recalculate priorities of its predecessors.
        5. Repeat.

    Python heapq is a min-heap, so priorities are stored
    as negative residual magnitudes.

    A version counter prevents stale queue entries from
    being processed.

    This is a simple tabular implementation. It is not an
    optimized sparse production implementation.
    """

    values = np.zeros(
        mdp.num_states
    )

    predecessors = build_predecessors(
        mdp
    )

    heap = []

    versions = np.zeros(
        mdp.num_states,
        dtype=int
    )

    def residual(state):

        return abs(
            optimality_backup(
                mdp,
                state,
                values,
                gamma
            ) - values[state]
        )

    # --------------------------------------------------------------
    # A. Initialize queue.
    # --------------------------------------------------------------

    for state in range(mdp.num_states):

        if mdp.terminal[state]:
            continue

        priority = residual(state)

        if priority > theta:

            versions[state] += 1

            heapq.heappush(
                heap,
                (
                    -priority,
                    int(versions[state]),
                    state
                )
            )

    history = []

    updates = 0

    # --------------------------------------------------------------
    # B. Repeated prioritized updates.
    # --------------------------------------------------------------

    while heap and updates < max_updates:

        negative_priority, version, state = heapq.heappop(
            heap
        )

        if version != versions[state]:
            continue

        current_priority = residual(state)

        if current_priority <= theta:
            continue

        # Update selected state.
        values[state] = optimality_backup(
            mdp,
            state,
            values,
            gamma
        )

        updates += 1

        # Recalculate priorities of predecessor states.
        for predecessor in predecessors[state]:

            if mdp.terminal[predecessor]:
                continue

            priority = residual(
                predecessor
            )

            if priority > theta:

                versions[predecessor] += 1

                heapq.heappush(
                    heap,
                    (
                        -priority,
                        int(versions[predecessor]),
                        predecessor
                    )
                )

        # Periodically evaluate the global residual.
        if updates % 20 == 0:

            global_residual = max(
                (
                    residual(s)
                    for s in range(mdp.num_states)
                    if not mdp.terminal[s]
                ),
                default=0.0
            )

            history.append(
                float(global_residual)
            )

            if global_residual <= theta:
                break

    # Final global residual verification.
    final_residual = max(
        (
            residual(s)
            for s in range(mdp.num_states)
            if not mdp.terminal[s]
        ),
        default=0.0
    )

    policy = policy_from_values(
        mdp,
        values,
        gamma
    )

    return {
        "values": values,
        "policy": policy,
        "history": history,
        "updates": updates,
        "final_residual": float(final_residual),
        "converged": final_residual <= theta
    }


# ======================================================================
# STEP 13. ALGORITHM PERFORMANCE
# ======================================================================

def timed_run(
    function,
    *args,
    **kwargs
):

    start = time.perf_counter()

    result = function(
        *args,
        **kwargs
    )

    result["runtime_seconds"] = (
        time.perf_counter() - start
    )

    return result


def compare_gpi_variants(
    mdp,
    gamma=0.90,
    theta=1e-9
):

    results = {}

    results["Value Iteration"] = timed_run(
        value_iteration,
        mdp,
        gamma,
        theta
    )

    results["GPI (1 sweep)"] = timed_run(
        generalized_policy_iteration,
        mdp,
        gamma,
        theta,
        evaluation_sweeps=1
    )

    results["GPI (5 sweeps)"] = timed_run(
        generalized_policy_iteration,
        mdp,
        gamma,
        theta,
        evaluation_sweeps=5
    )

    results["Modified PI (10 sweeps)"] = timed_run(
        modified_policy_iteration,
        mdp,
        gamma,
        theta,
        evaluation_sweeps=10
    )

    results["Async VI (cyclic)"] = timed_run(
        asynchronous_value_iteration,
        mdp,
        gamma,
        theta,
        schedule="cyclic"
    )

    results["Async VI (random)"] = timed_run(
        asynchronous_value_iteration,
        mdp,
        gamma,
        theta,
        schedule="random"
    )

    results["Prioritized Sweeping"] = timed_run(
        prioritized_sweeping,
        mdp,
        gamma,
        theta
    )

    # Independently evaluate each extracted policy.
    for name, result in results.items():

        if "policy" in result:

            result["evaluated_policy_values"] = (
                evaluate_policy_exact(
                    mdp,
                    result["policy"],
                    gamma
                )
            )

            result["evaluated_start_value"] = float(
                result["evaluated_policy_values"][0]
            )

    return results


def print_algorithm_comparison(results):

    print("\nGENERALIZED DP ALGORITHM COMPARISON")
    print("=" * 105)

    print(
        f"{'Algorithm':<28}"
        f"{'Converged':>12}"
        f"{'Iterations':>12}"
        f"{'Runtime':>13}"
        f"{'Start Value':>15}"
        f"{'Residual':>15}"
    )

    print("-" * 105)

    for name, result in results.items():

        if "sweeps" in result:
            iterations = result["sweeps"]

        elif "updates" in result:
            iterations = result["updates"]

        else:
            iterations = result["iterations"]

        if "final_residual" in result:

            final_residual = result["final_residual"]

        elif "history" in result and result["history"]:

            last = result["history"][-1]

            if isinstance(last, dict):
                final_residual = last["residual"]
            else:
                final_residual = last

        else:
            final_residual = float("nan")

        start_value = result.get(
            "evaluated_start_value",
            result["values"][0]
        )

        print(
            f"{name:<28}"
            f"{str(result['converged']):>12}"
            f"{iterations:>12}"
            f"{result['runtime_seconds']:>13.6f}"
            f"{start_value:>15.7f}"
            f"{final_residual:>15.3e}"
        )

    print("=" * 105)


# ======================================================================
# STEP 14. VISUALIZATION
# ======================================================================

def visualize_convergence(results):

    plt.figure(figsize=(11, 6))

    for name, result in results.items():

        history = result.get(
            "history",
            []
        )

        if not history:
            continue

        if isinstance(history[0], dict):

            residuals = [
                item["residual"]
                for item in history
            ]

        else:

            residuals = history

        if not residuals:
            continue

        plt.plot(
            range(1, len(residuals) + 1),
            np.maximum(residuals, 1e-16),
            label=name
        )

    plt.yscale("log")

    plt.xlabel("Recorded Iteration")
    plt.ylabel("Bellman Residual")

    plt.title("Generalized DP Convergence")

    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()


def visualize_runtime(results):

    names = list(results.keys())

    runtimes = [
        results[name]["runtime_seconds"]
        for name in names
    ]

    plt.figure(figsize=(12, 6))

    plt.bar(
        names,
        runtimes
    )

    plt.ylabel("Runtime (seconds)")
    plt.title("Runtime Comparison")

    plt.xticks(
        rotation=30,
        ha="right"
    )

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()


def visualize_start_values(results):

    names = list(results.keys())

    values = [
        results[name]["evaluated_start_value"]
        for name in names
    ]

    plt.figure(figsize=(12, 6))

    plt.bar(
        names,
        values
    )

    plt.ylabel("Evaluated Start-State Value")
    plt.title("Quality of Extracted Policies")

    plt.xticks(
        rotation=30,
        ha="right"
    )

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()


def visualize_policy(
    mdp,
    policy,
    title
):

    fig, ax = plt.subplots(
        figsize=(7, 6)
    )

    ax.set_xlim(
        -0.5,
        mdp.cols - 0.5
    )

    ax.set_ylim(
        mdp.rows - 0.5,
        -0.5
    )

    ax.set_xticks(
        range(mdp.cols)
    )

    ax.set_yticks(
        range(mdp.rows)
    )

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
                    ACTION_SYMBOLS[a]
                    for a in selected
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

    # TEST 1: Policy model dimensions.
    policy = uniform_policy(mdp)

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

    print("[PASS] Policy model dimensions")

    # TEST 2: Transition probabilities.
    assert np.allclose(
        P_pi.sum(axis=1)[~mdp.terminal],
        1.0
    )

    print("[PASS] Policy transition probabilities")

    # TEST 3: Exact policy evaluation.
    exact_values = evaluate_policy_exact(
        mdp,
        policy,
        gamma
    )

    assert np.all(
        np.isfinite(exact_values)
    )

    print("[PASS] Exact policy evaluation")

    # TEST 4: Standard Value Iteration.
    vi = value_iteration(
        mdp,
        gamma
    )

    assert vi["converged"]

    print("[PASS] Value Iteration")

    # TEST 5: GPI.
    gpi = generalized_policy_iteration(
        mdp,
        gamma,
        evaluation_sweeps=1
    )

    assert gpi["converged"]

    print("[PASS] Generalized Policy Iteration")

    # TEST 6: Modified Policy Iteration.
    mpi = modified_policy_iteration(
        mdp,
        gamma,
        evaluation_sweeps=5
    )

    assert mpi["converged"]

    print("[PASS] Modified Policy Iteration")

    # TEST 7: Asynchronous policy evaluation.
    async_eval = asynchronous_policy_evaluation(
        mdp,
        policy,
        gamma,
        schedule="cyclic"
    )

    assert async_eval["converged"]

    assert np.allclose(
        async_eval["values"],
        exact_values,
        atol=1e-8
    )

    print("[PASS] Asynchronous policy evaluation")

    # TEST 8: Asynchronous VI, cyclic.
    async_vi = asynchronous_value_iteration(
        mdp,
        gamma,
        schedule="cyclic"
    )

    assert async_vi["converged"]

    print("[PASS] Asynchronous Value Iteration — cyclic")

    # TEST 9: Asynchronous VI, random.
    random_vi = asynchronous_value_iteration(
        mdp,
        gamma,
        schedule="random"
    )

    assert random_vi["converged"]

    print("[PASS] Asynchronous Value Iteration — random")

    # TEST 10: Prioritized Sweeping.
    prioritized = prioritized_sweeping(
        mdp,
        gamma
    )

    assert prioritized["converged"]

    print("[PASS] Prioritized Sweeping")

    # TEST 11: GPI and VI values.
    assert np.allclose(
        gpi["values"],
        vi["values"],
        atol=1e-7
    )

    print("[PASS] GPI and VI values agree")

    # TEST 12: Modified PI and VI values.
    assert np.allclose(
        mpi["values"],
        vi["values"],
        atol=1e-7
    )

    print("[PASS] Modified PI and VI values agree")

    # TEST 13: Asynchronous VI and VI values.
    assert np.allclose(
        async_vi["values"],
        vi["values"],
        atol=1e-7
    )

    print("[PASS] Asynchronous VI and VI values agree")

    # TEST 14: Random asynchronous VI and VI values.
    assert np.allclose(
        random_vi["values"],
        vi["values"],
        atol=1e-7
    )

    print("[PASS] Random asynchronous VI and VI values agree")

    # TEST 15: Prioritized Sweeping and VI values.
    assert np.allclose(
        prioritized["values"],
        vi["values"],
        atol=1e-7
    )

    print("[PASS] Prioritized Sweeping and VI values agree")

    # TEST 16: Bellman residual.
    residual = max(
        abs(
            optimality_backup(
                mdp,
                state,
                vi["values"],
                gamma
            ) - vi["values"][state]
        )
        for state in range(mdp.num_states)
    )

    assert residual < 1e-8

    print("[PASS] Bellman optimality residual")

    # TEST 17: Terminal values.
    assert np.allclose(
        vi["values"][mdp.terminal],
        0.0
    )

    print("[PASS] Terminal values")

    # TEST 18: Different MPI evaluation depths.
    mpi_one = modified_policy_iteration(
        mdp,
        gamma,
        evaluation_sweeps=1
    )

    mpi_ten = modified_policy_iteration(
        mdp,
        gamma,
        evaluation_sweeps=10
    )

    assert mpi_one["converged"]
    assert mpi_ten["converged"]

    assert np.allclose(
        mpi_one["values"],
        vi["values"],
        atol=1e-7
    )

    assert np.allclose(
        mpi_ten["values"],
        vi["values"],
        atol=1e-7
    )

    print("[PASS] Modified PI evaluation-depth comparison")

    # TEST 19: Invalid schedule.
    rejected = False

    try:

        asynchronous_value_iteration(
            mdp,
            gamma,
            schedule="invalid"
        )

    except ValueError:
        rejected = True

    assert rejected

    print("[PASS] Invalid asynchronous schedule rejected")

    # TEST 20: Deterministic MDP.
    deterministic_mdp = build_gridworld(
        rows=4,
        cols=4,
        intended_probability=1.0
    )

    deterministic_vi = value_iteration(
        deterministic_mdp,
        gamma
    )

    deterministic_gpi = generalized_policy_iteration(
        deterministic_mdp,
        gamma
    )

    assert np.allclose(
        deterministic_vi["values"],
        deterministic_gpi["values"],
        atol=1e-7
    )

    print("[PASS] Deterministic MDP")

    print("\nAll 20 tests passed.")


# ======================================================================
# STEP 16. MAIN
# ======================================================================

def main():

    print("=" * 75)

    print(
        "PHASE 2.5 — GENERALIZED POLICY ITERATION "
        "AND ASYNCHRONOUS DP"
    )

    print("=" * 75)

    mdp = build_gridworld(
        rows=5,
        cols=5,
        intended_probability=0.80,
        step_reward=-0.04,
        goal_reward=1.0
    )

    gamma = 0.90
    theta = 1e-9

    # --------------------------------------------------------------
    # A. Run all algorithms.
    # --------------------------------------------------------------

    results = compare_gpi_variants(
        mdp,
        gamma,
        theta
    )

    print_algorithm_comparison(
        results
    )

    # --------------------------------------------------------------
    # B. Independent optimal-value reference.
    # --------------------------------------------------------------

    reference = value_iteration(
        mdp,
        gamma,
        theta=1e-12
    )

    print("\nREFERENCE OPTIMAL START VALUE:")
    print(
        f"{reference['values'][0]:.10f}"
    )

    # --------------------------------------------------------------
    # C. Unit tests.
    # --------------------------------------------------------------

    print("\nUNIT TESTS")

    run_tests()

    # --------------------------------------------------------------
    # D. Visualizations.
    # --------------------------------------------------------------

    visualize_convergence(
        results
    )

    visualize_runtime(
        results
    )

    visualize_start_values(
        results
    )

    visualize_policy(
        mdp,
        results["Value Iteration"]["policy"],
        "Optimal Policy — Standard VI"
    )

    visualize_policy(
        mdp,
        results["Prioritized Sweeping"]["policy"],
        "Policy — Prioritized Sweeping"
    )

    visualize_value_grid(
        mdp,
        reference["values"],
        "Optimal State Values"
    )

    plt.show()

    print("\nPhase 2.5 completed successfully.")


if __name__ == "__main__":
    main()
