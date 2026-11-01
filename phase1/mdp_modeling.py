
"""
===============================================================================
RL COURSE — PHASE 1.5
Complete MDP Modeling and Bellman Equation Preparation
===============================================================================

LESSON ROADMAP
--------------

STEP 1 — Define a Complete MDP
Purpose:
    Combine the components of an RL environment into a mathematical model.

What you learn:
    - MDP = (S, A, P, R, gamma)
    - Finite state and action spaces.
    - Transition probabilities.
    - Expected immediate rewards.
    - Terminal states.

Implementation:
    - MDP class
    - validate_mdp()

STEP 2 — Define a Policy
Purpose:
    Specify how an agent selects actions in each state.

What you learn:
    - Deterministic policies.
    - Stochastic policies.
    - Policy probability normalization.
    - State-dependent action selection.

Implementation:
    - Policy class
    - deterministic_policy()
    - stochastic_policy()

STEP 3 — Calculate State and Action Values
Purpose:
    Understand how to evaluate the quality of states and actions.

What you learn:
    - State-value function V_pi(s).
    - Action-value function Q_pi(s,a).
    - Expected immediate reward.
    - Discounted future return.

Implementation:
    - calculate_expected_reward()
    - calculate_action_value()

STEP 4 — Bellman Expectation Equation
Purpose:
    Express the relationship between current state value and future state values.

What you learn:
    - Recursive value calculation.
    - Expected future rewards.
    - Discount factor.
    - Terminal-state handling.

Implementation:
    - bellman_expectation_backup()
    - policy_evaluation()

STEP 5 — Bellman Equation Verification
Purpose:
    Check that the numerical solution satisfies the Bellman equation.

What you learn:
    - Linear system representation.
    - Matrix formulation.
    - Numerical solution.
    - Bellman residual.

Implementation:
    - solve_policy_exactly()
    - calculate_bellman_residual()
    - run_tests()

STEP 6 — Visualization and Experimentation
Purpose:
    Understand how policy and discount factor influence state values.

What you learn:
    - State-value heatmaps.
    - Action-value comparisons.
    - Discount-factor sensitivity.
    - Policy comparisons.

Implementation:
    - All visualization functions are included below.
    - main() executes the demonstrations and visualizations.

===============================================================================
MATHEMATICAL FOUNDATION
===============================================================================

1. FINITE MARKOV DECISION PROCESS

    M = (S, A, P, R, gamma)

    S:
        Finite state space.

    A:
        Finite action space.

    P(s'|s,a):
        Transition probability.

    R(s,a,s'):
        Expected immediate reward associated with a transition.

    gamma:
        Discount factor, 0 <= gamma < 1.

2. POLICY

    Deterministic:

        pi(s) = a

    Stochastic:

        pi(a|s) = P(A=a | S=s)

    Probability constraint:

        sum_a pi(a|s) = 1

3. RETURN

    G_t = R[t+1] + gamma R[t+2] + gamma^2 R[t+3] + ...

4. STATE-VALUE FUNCTION

    V_pi(s) = E_pi[G_t | S_t=s]

    It estimates the expected discounted return when the agent starts
    in state s and follows policy pi.

5. ACTION-VALUE FUNCTION

    Q_pi(s,a) = E_pi[G_t | S_t=s, A_t=a]

    It estimates the expected return after selecting action a in state s
    and following policy pi afterward.

6. BELLMAN EXPECTATION EQUATION

    V_pi(s) =
        sum_a pi(a|s)
        sum_s' P(s'|s,a)
        [R(s,a,s') + gamma V_pi(s')]

    Terminal states have:

        V_pi(terminal) = 0

    This assumes that the terminal transition reward is already included
    in R(s,a,s').

7. ACTION-VALUE BELLMAN EQUATION

    Q_pi(s,a) =
        sum_s' P(s'|s,a)
        [R(s,a,s') + gamma sum_a' pi(a'|s') Q_pi(s',a')]

8. MATRIX FORM

    For a fixed policy:

        V = r_pi + gamma P_pi V

    Therefore:

        (I - gamma P_pi)V = r_pi

        V = (I - gamma P_pi)^(-1) r_pi

    In practical numerical code, solve the linear system directly
    rather than explicitly calculating the matrix inverse.

9. BELLMAN RESIDUAL

    residual = T_pi(V) - V

    A correct Bellman solution should have a residual close to zero.

===============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# STEP 1 — COMPLETE MDP MODEL
# =============================================================================

class MDP:
    """
    Finite MDP with transition-dependent rewards.

    P[a, s, s_next]:
        P(s_next | s, a)

    R[a, s, s_next]:
        Immediate reward for the transition.

    terminal[s]:
        True if state s is terminal.

    Terminal states are modeled as absorbing states with zero reward
    for subsequent transitions.

    The reward on the transition INTO a terminal state is retained.
    """

    def __init__(
        self,
        transition_probabilities: np.ndarray,
        rewards: np.ndarray,
        terminal_states: List[int],
        gamma: float
    ):

        self.P = np.asarray(
            transition_probabilities,
            dtype=float
        )

        self.R = np.asarray(
            rewards,
            dtype=float
        )

        if self.P.ndim != 3:
            raise ValueError("P must be a 3D array")

        self.num_actions = self.P.shape[0]
        self.num_states = self.P.shape[1]

        if self.P.shape[2] != self.num_states:
            raise ValueError("P must have shape (A, S, S)")

        if self.R.shape != self.P.shape:
            raise ValueError("R and P must have identical shapes")

        if not np.all(np.isfinite(self.P)):
            raise ValueError("P must contain finite values")

        if not np.all(np.isfinite(self.R)):
            raise ValueError("R must contain finite values")

        if np.any(self.P < 0):
            raise ValueError("Transition probabilities cannot be negative")

        if not 0.0 <= gamma < 1.0:
            raise ValueError("gamma must satisfy 0 <= gamma < 1")

        row_sums = np.sum(self.P, axis=2)

        if not np.allclose(row_sums, 1.0):
            raise ValueError(
                "Every state-action transition distribution must sum to 1"
            )

        self.gamma = float(gamma)

        self.terminal = np.zeros(
            self.num_states,
            dtype=bool
        )

        for state in terminal_states:

            if not 0 <= state < self.num_states:
                raise ValueError("Invalid terminal state")

            self.terminal[state] = True

        self.validate_terminal_states()

    def validate_terminal_states(self):
        """
        Terminal states must be absorbing with zero subsequent reward.
        """
        for state in range(self.num_states):

            if not self.terminal[state]:
                continue

            for action in range(self.num_actions):

                expected = np.zeros(self.num_states)
                expected[state] = 1.0

                if not np.allclose(
                    self.P[action, state],
                    expected
                ):
                    raise ValueError(
                        "Terminal states must have self-loop transitions"
                    )

                if not np.allclose(
                    self.R[action, state],
                    0.0
                ):
                    raise ValueError(
                        "Terminal states must have zero subsequent reward"
                    )

    def validate_mdp(self) -> bool:

        return (
            self.P.ndim == 3
            and self.R.shape == self.P.shape
            and np.all(self.P >= 0)
            and np.all(np.isfinite(self.P))
            and np.all(np.isfinite(self.R))
            and np.allclose(
                np.sum(self.P, axis=2),
                1.0
            )
        )

    def expected_reward(
        self,
        state: int,
        action: int
    ) -> float:
        """
        E[R | s,a] = sum_s' P(s'|s,a) R(s,a,s')
        """
        return float(
            np.sum(
                self.P[action, state]
                * self.R[action, state]
            )
        )


# =============================================================================
# STEP 2 — POLICY REPRESENTATION
# =============================================================================

class Policy:
    """
    A tabular stochastic policy.

    probabilities[s, a] = pi(a|s)
    """

    def __init__(
        self,
        probabilities: np.ndarray
    ):

        self.probabilities = np.asarray(
            probabilities,
            dtype=float
        )

        if self.probabilities.ndim != 2:
            raise ValueError("Policy must be a 2D array")

        if not np.all(np.isfinite(self.probabilities)):
            raise ValueError("Policy probabilities must be finite")

        if np.any(self.probabilities < 0):
            raise ValueError("Policy probabilities cannot be negative")

        if not np.allclose(
            np.sum(self.probabilities, axis=1),
            1.0
        ):
            raise ValueError(
                "Policy probabilities must sum to one in every state"
            )

        self.num_states, self.num_actions = (
            self.probabilities.shape
        )

    def action_probabilities(
        self,
        state: int
    ) -> np.ndarray:

        if not 0 <= state < self.num_states:
            raise ValueError("Invalid state")

        return self.probabilities[state].copy()

    def sample_action(
        self,
        state: int,
        rng: np.random.Generator
    ) -> int:

        return int(
            rng.choice(
                self.num_actions,
                p=self.probabilities[state]
            )
        )

    @staticmethod
    def deterministic(
        actions: List[int],
        num_actions: int
    ) -> "Policy":

        num_states = len(actions)

        probabilities = np.zeros(
            (num_states, num_actions)
        )

        for state, action in enumerate(actions):

            if not 0 <= action < num_actions:
                raise ValueError("Invalid action")

            probabilities[state, action] = 1.0

        return Policy(probabilities)

    @staticmethod
    def uniform(
        num_states: int,
        num_actions: int
    ) -> "Policy":

        if num_states <= 0 or num_actions <= 0:
            raise ValueError("State and action counts must be positive")

        probabilities = np.full(
            (num_states, num_actions),
            1.0 / num_actions
        )

        return Policy(probabilities)


# =============================================================================
# STEP 3 — VALUE AND ACTION-VALUE FUNCTIONS
# =============================================================================

def calculate_action_value(
    mdp: MDP,
    policy: Policy,
    values: np.ndarray,
    state: int,
    action: int
) -> float:
    """
    Calculate one Bellman action-value backup.

    Q_pi(s,a) =
        sum_s' P(s'|s,a)
        [R(s,a,s') + gamma V_pi(s')]

    Terminal states have value zero.
    """
    if mdp.terminal[state]:
        return 0.0

    future_values = np.where(
        mdp.terminal,
        0.0,
        values
    )

    transition_values = (
        mdp.R[action, state]
        + mdp.gamma * future_values
    )

    return float(
        np.sum(
            mdp.P[action, state]
            * transition_values
        )
    )


def calculate_expected_reward(
    mdp: MDP,
    policy: Policy
) -> np.ndarray:
    """
    Calculate r_pi(s):

        r_pi(s) =
            sum_a pi(a|s)
            sum_s' P(s'|s,a) R(s,a,s')
    """
    rewards = np.zeros(mdp.num_states)

    for state in range(mdp.num_states):

        if mdp.terminal[state]:
            continue

        for action in range(mdp.num_actions):

            rewards[state] += (
                policy.probabilities[state, action]
                * mdp.expected_reward(state, action)
            )

    return rewards


# =============================================================================
# STEP 4 — BELLMAN EXPECTATION EQUATION
# =============================================================================

def bellman_expectation_backup(
    mdp: MDP,
    policy: Policy,
    values: np.ndarray
) -> np.ndarray:
    """
    Apply one Bellman expectation backup.

        V_new(s) = sum_a pi(a|s) Q(s,a)

    This performs one synchronous dynamic-programming update.
    """
    values = np.asarray(values, dtype=float)

    if values.shape != (mdp.num_states,):
        raise ValueError("Invalid value vector shape")

    new_values = np.zeros(mdp.num_states)

    for state in range(mdp.num_states):

        if mdp.terminal[state]:
            continue

        for action in range(mdp.num_actions):

            q_value = calculate_action_value(
                mdp,
                policy,
                values,
                state,
                action
            )

            new_values[state] += (
                policy.probabilities[state, action]
                * q_value
            )

    return new_values


def policy_evaluation(
    mdp: MDP,
    policy: Policy,
    tolerance: float = 1e-10,
    max_iterations: int = 100000
) -> Tuple[np.ndarray, int]:
    """
    Iteratively evaluate a fixed policy.

        V_(k+1) = T_pi(V_k)

    Stop when the maximum value change is below tolerance.
    """
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")

    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")

    values = np.zeros(mdp.num_states)

    for iteration in range(1, max_iterations + 1):

        new_values = bellman_expectation_backup(
            mdp,
            policy,
            values
        )

        error = np.max(
            np.abs(new_values - values)
        )

        values = new_values

        if error < tolerance:
            return values, iteration

    raise RuntimeError(
        "Policy evaluation did not converge"
    )


# =============================================================================
# STEP 5 — EXACT MATRIX SOLUTION
# =============================================================================

def build_policy_transition_matrix(
    mdp: MDP,
    policy: Policy
) -> np.ndarray:
    """
    Build:

        P_pi(s,s') = sum_a pi(a|s) P(s'|s,a)
    """
    matrix = np.zeros(
        (mdp.num_states, mdp.num_states)
    )

    for state in range(mdp.num_states):

        if mdp.terminal[state]:
            matrix[state, state] = 1.0
            continue

        for action in range(mdp.num_actions):

            matrix[state] += (
                policy.probabilities[state, action]
                * mdp.P[action, state]
            )

    return matrix


def solve_policy_exactly(
    mdp: MDP,
    policy: Policy
) -> np.ndarray:
    """
    Solve the Bellman expectation equation exactly up to
    floating-point numerical precision.

    Terminal values are fixed at zero.

        V = r_pi + gamma P_pi V

    For nonterminal states N:

        (I - gamma P_NN) V_N = r_N

    This reduced system avoids treating terminal-state self-loops
    as continuing reward-generating states.
    """
    P_pi = build_policy_transition_matrix(
        mdp,
        policy
    )

    rewards = calculate_expected_reward(
        mdp,
        policy
    )

    nonterminal_indices = np.where(
        ~mdp.terminal
    )[0]

    values = np.zeros(mdp.num_states)

    if len(nonterminal_indices) == 0:
        return values

    P_nonterminal = P_pi[
        np.ix_(
            nonterminal_indices,
            nonterminal_indices
        )
    ]

    matrix = (
        np.eye(len(nonterminal_indices))
        - mdp.gamma * P_nonterminal
    )

    rhs = rewards[nonterminal_indices]

    values[nonterminal_indices] = np.linalg.solve(
        matrix,
        rhs
    )

    return values


def calculate_bellman_residual(
    mdp: MDP,
    policy: Policy,
    values: np.ndarray
) -> np.ndarray:
    """
    residual = T_pi(V) - V
    """
    return (
        bellman_expectation_backup(
            mdp,
            policy,
            values
        )
        - values
    )


def calculate_q_table(
    mdp: MDP,
    policy: Policy,
    values: np.ndarray
) -> np.ndarray:
    """
    Calculate Q_pi(s,a) for all state-action pairs.
    """
    q_table = np.zeros(
        (mdp.num_states, mdp.num_actions)
    )

    for state in range(mdp.num_states):

        if mdp.terminal[state]:
            continue

        for action in range(mdp.num_actions):

            q_table[state, action] = calculate_action_value(
                mdp,
                policy,
                values,
                state,
                action
            )

    return q_table


# =============================================================================
# STEP 6 — EXAMPLE MDP
# =============================================================================

def build_example_mdp(
    gamma: float = 0.9
) -> Tuple[MDP, List[str]]:
    """
    Construct a four-state MDP.

    States:
        0 = Start
        1 = Safe
        2 = Risky
        3 = Goal (terminal)

    Actions:
        0 = Conservative
        1 = Aggressive

    Transition design:

    Start + Conservative:
        Safe: 100%

    Start + Aggressive:
        Safe: 50%
        Risky: 50%

    Safe + Conservative:
        Goal: 100%, reward +5

    Safe + Aggressive:
        Goal: 70%, reward +8
        Risky: 30%, reward -2

    Risky + Conservative:
        Goal: 60%, reward +4
        Risky: 40%, reward -1

    Risky + Aggressive:
        Goal: 30%, reward +10
        Risky: 70%, reward -3

    Goal:
        Terminal absorbing state.

    All unspecified transition rewards are zero.
    """
    num_states = 4
    num_actions = 2

    P = np.zeros(
        (num_actions, num_states, num_states)
    )

    R = np.zeros_like(P)

    # Start state: 0
    P[0, 0, 1] = 1.0

    P[1, 0, 1] = 0.5
    P[1, 0, 2] = 0.5

    # Safe state: 1
    P[0, 1, 3] = 1.0
    R[0, 1, 3] = 5.0

    P[1, 1, 3] = 0.7
    P[1, 1, 2] = 0.3

    R[1, 1, 3] = 8.0
    R[1, 1, 2] = -2.0

    # Risky state: 2
    P[0, 2, 3] = 0.6
    P[0, 2, 2] = 0.4

    R[0, 2, 3] = 4.0
    R[0, 2, 2] = -1.0

    P[1, 2, 3] = 0.3
    P[1, 2, 2] = 0.7

    R[1, 2, 3] = 10.0
    R[1, 2, 2] = -3.0

    # Terminal state: 3
    for action in range(num_actions):
        P[action, 3, 3] = 1.0

    mdp = MDP(
        transition_probabilities=P,
        rewards=R,
        terminal_states=[3],
        gamma=gamma
    )

    state_names = [
        "Start",
        "Safe",
        "Risky",
        "Goal"
    ]

    return mdp, state_names


def build_policies() -> Dict[str, Policy]:
    """
    Build three example policies.
    """
    # Conservative in every nonterminal state.
    conservative = Policy.deterministic(
        actions=[0, 0, 0, 0],
        num_actions=2
    )

    # Aggressive in every nonterminal state.
    aggressive = Policy.deterministic(
        actions=[1, 1, 1, 0],
        num_actions=2
    )

    # Equal probability for both actions.
    random_policy = Policy.uniform(
        num_states=4,
        num_actions=2
    )

    return {
        "Conservative": conservative,
        "Aggressive": aggressive,
        "Random": random_policy
    }


# =============================================================================
# STEP 7 — DISCOUNT FACTOR EXPERIMENT
# =============================================================================

def evaluate_gamma_sensitivity(
    gamma_values: np.ndarray,
    policy: Policy
) -> np.ndarray:
    """
    Evaluate a fixed policy under different discount factors.
    """
    values = []

    for gamma in gamma_values:

        mdp, _ = build_example_mdp(
            gamma=float(gamma)
        )

        value = solve_policy_exactly(
            mdp,
            policy
        )

        values.append(value)

    return np.asarray(values)


# =============================================================================
# STEP 8 — VISUALIZATION
# =============================================================================

def visualize_transition_matrices(mdp: MDP):
    """
    Display transition matrices for both actions.
    """
    for action in range(mdp.num_actions):

        plt.figure(figsize=(7, 5))

        plt.imshow(
            mdp.P[action],
            vmin=0,
            vmax=1,
            aspect="auto"
        )

        plt.colorbar(label="Probability")

        plt.xlabel("Next state")
        plt.ylabel("Current state")

        plt.title(
            f"Transition Matrix — Action {action}"
        )

        plt.xticks(range(mdp.num_states))
        plt.yticks(range(mdp.num_states))

        plt.tight_layout()


def visualize_policy_values(
    values_by_policy: Dict[str, np.ndarray],
    state_names: List[str]
):
    """
    Compare state values under different policies.
    """
    x = np.arange(len(state_names))
    width = 0.25

    plt.figure(figsize=(10, 5))

    names = list(values_by_policy.keys())

    for i, name in enumerate(names):

        offset = (
            i - (len(names) - 1) / 2
        ) * width

        plt.bar(
            x + offset,
            values_by_policy[name],
            width,
            label=name
        )

    plt.xticks(x, state_names)
    plt.ylabel("State value")
    plt.title("State Values Under Different Policies")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()


def visualize_q_values(
    q_table: np.ndarray,
    state_names: List[str]
):
    """
    Display the action-value table as a heatmap.
    """
    plt.figure(figsize=(7, 5))

    plt.imshow(
        q_table,
        aspect="auto"
    )

    plt.colorbar(label="Q value")

    plt.xlabel("Action")
    plt.ylabel("State")

    plt.xticks(
        [0, 1],
        ["Conservative", "Aggressive"]
    )

    plt.yticks(
        range(len(state_names)),
        state_names
    )

    plt.title("Action-Value Function Q_pi(s,a)")

    plt.tight_layout()


def visualize_gamma_sensitivity(
    gamma_values: np.ndarray,
    values: np.ndarray
):
    """
    Display how the discount factor changes state values.
    """
    plt.figure(figsize=(9, 5))

    for state_index in range(values.shape[1] - 1):

        plt.plot(
            gamma_values,
            values[:, state_index],
            marker="o",
            label=f"State {state_index}"
        )

    plt.xlabel("Discount factor gamma")
    plt.ylabel("State value")
    plt.title("Discount Factor Sensitivity")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()


def visualize_bellman_residual(
    residual: np.ndarray,
    state_names: List[str]
):
    """
    Plot the Bellman residual for each state.
    """
    plt.figure(figsize=(8, 4))

    plt.bar(
        np.arange(len(residual)),
        np.abs(residual)
    )

    plt.xticks(
        np.arange(len(state_names)),
        state_names
    )

    plt.ylabel("Absolute Bellman residual")
    plt.title("Bellman Equation Numerical Verification")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()


# =============================================================================
# STEP 9 — UNIT TESTS
# =============================================================================

def run_tests():

    print("\n========== UNIT TESTS ==========")

    mdp, _ = build_example_mdp()

    policies = build_policies()

    # Test 1: MDP validity
    assert mdp.validate_mdp()
    print("[PASS] MDP validation")

    # Test 2: Transition probability normalization
    assert np.allclose(
        np.sum(mdp.P, axis=2),
        1.0
    )
    print("[PASS] Transition normalization")

    # Test 3: Terminal-state identification
    assert mdp.terminal[3]
    assert not mdp.terminal[0]
    print("[PASS] Terminal-state identification")

    # Test 4: Terminal state self-loop
    assert np.allclose(
        mdp.P[:, 3, 3],
        1.0
    )
    print("[PASS] Terminal-state self-loop")

    # Test 5: Terminal rewards
    assert np.allclose(
        mdp.R[:, 3, :],
        0.0
    )
    print("[PASS] Terminal-state reward convention")

    # Test 6: Expected reward
    reward = mdp.expected_reward(
        state=1,
        action=0
    )

    assert np.isclose(reward, 5.0)
    print("[PASS] Expected reward")

    # Test 7: Policy normalization
    for policy in policies.values():

        assert np.allclose(
            np.sum(policy.probabilities, axis=1),
            1.0
        )

    print("[PASS] Policy normalization")

    # Test 8: Deterministic policy
    policy = policies["Conservative"]

    assert np.allclose(
        policy.probabilities[0],
        [1.0, 0.0]
    )
    print("[PASS] Deterministic policy")

    # Test 9: Uniform policy
    policy = policies["Random"]

    assert np.allclose(
        policy.probabilities[0],
        [0.5, 0.5]
    )
    print("[PASS] Uniform policy")

    # Test 10: Terminal value is zero
    values, _ = policy_evaluation(
        mdp,
        policies["Conservative"]
    )

    assert np.isclose(values[3], 0.0)
    print("[PASS] Terminal-state value")

    # Test 11: Exact solution versus iterative solution
    iterative, _ = policy_evaluation(
        mdp,
        policies["Conservative"]
    )

    exact = solve_policy_exactly(
        mdp,
        policies["Conservative"]
    )

    assert np.allclose(
        iterative,
        exact,
        atol=1e-8
    )
    print("[PASS] Iterative and exact policy evaluation")

    # Test 12: Bellman residual
    residual = calculate_bellman_residual(
        mdp,
        policies["Conservative"],
        exact
    )

    assert np.max(np.abs(residual)) < 1e-8
    print("[PASS] Bellman residual")

    # Test 13: Q table shape
    q_table = calculate_q_table(
        mdp,
        policies["Conservative"],
        exact
    )

    assert q_table.shape == (4, 2)
    print("[PASS] Q-table dimensions")

    # Test 14: Terminal Q values
    assert np.allclose(q_table[3], 0.0)
    print("[PASS] Terminal Q values")

    # Test 15: Policy transition matrix normalization
    P_pi = build_policy_transition_matrix(
        mdp,
        policies["Random"]
    )

    assert np.allclose(
        np.sum(P_pi, axis=1),
        1.0
    )
    print("[PASS] Policy transition matrix")

    # Test 16: Gamma zero
    mdp_zero, _ = build_example_mdp(gamma=0.0)

    values_zero = solve_policy_exactly(
        mdp_zero,
        policies["Conservative"]
    )

    assert np.allclose(
        values_zero,
        calculate_expected_reward(
            mdp_zero,
            policies["Conservative"]
        )
    )
    print("[PASS] Gamma zero")

    # Test 17: Policy sampling reproducibility
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)

    policy = policies["Random"]

    samples1 = [
        policy.sample_action(0, rng1)
        for _ in range(100)
    ]

    samples2 = [
        policy.sample_action(0, rng2)
        for _ in range(100)
    ]

    assert samples1 == samples2
    print("[PASS] Policy sampling reproducibility")

    # Test 18: Invalid discount factor
    try:
        build_example_mdp(gamma=1.2)
        raise AssertionError("Invalid gamma accepted")
    except ValueError:
        pass

    print("[PASS] Invalid discount factor validation")

    # Test 19: Invalid policy
    try:
        Policy(
            np.array([
                [0.2, 0.2],
                [0.5, 0.5]
            ])
        )

        raise AssertionError("Invalid policy accepted")

    except ValueError:
        pass

    print("[PASS] Invalid policy validation")

    # Test 20: Action-value Bellman consistency
    values = solve_policy_exactly(
        mdp,
        policies["Random"]
    )

    q_table = calculate_q_table(
        mdp,
        policies["Random"],
        values
    )

    reconstructed = (
        policies["Random"].probabilities * q_table
    ).sum(axis=1)

    assert np.allclose(
        reconstructed,
        values,
        atol=1e-8
    )

    print("[PASS] State value reconstructed from Q values")

    print("\nAll 20 tests passed.")


# =============================================================================
# STEP 10 — MAIN
# =============================================================================

def main():

    print("=" * 72)
    print("RL PHASE 1.5 — COMPLETE MDP MODELING")
    print("=" * 72)

    # -------------------------------------------------------------------------
    # Demo A: Construct MDP
    # -------------------------------------------------------------------------

    mdp, state_names = build_example_mdp(
        gamma=0.9
    )

    policies = build_policies()

    print("\n[DEMO A] MDP definition")

    print("States:", state_names)

    print("Actions:", [
        "Conservative",
        "Aggressive"
    ])

    print("Discount factor:", mdp.gamma)

    print("Transition tensor shape:", mdp.P.shape)

    # -------------------------------------------------------------------------
    # Demo B: Policy evaluation
    # -------------------------------------------------------------------------

    print("\n[DEMO B] Policy evaluation")

    values_by_policy = {}

    for name, policy in policies.items():

        values, iterations = policy_evaluation(
            mdp,
            policy
        )

        values_by_policy[name] = values

        print(f"\nPolicy: {name}")

        for state, value in enumerate(values):

            print(
                f"  {state_names[state]:10s}: "
                f"V(s) = {value:10.6f}"
            )

        print("  Iterations:", iterations)

    # -------------------------------------------------------------------------
    # Demo C: Exact versus iterative solution
    # -------------------------------------------------------------------------

    print("\n[DEMO C] Bellman equation verification")

    policy = policies["Random"]

    iterative_values, iterations = policy_evaluation(
        mdp,
        policy
    )

    exact_values = solve_policy_exactly(
        mdp,
        policy
    )

    residual = calculate_bellman_residual(
        mdp,
        policy,
        exact_values
    )

    print("Iterative solution:", iterative_values)

    print("Exact solution:    ", exact_values)

    print("Bellman residual:  ", residual)

    print(
        "Maximum residual:",
        np.max(np.abs(residual))
    )

    # -------------------------------------------------------------------------
    # Demo D: Action-value function
    # -------------------------------------------------------------------------

    print("\n[DEMO D] Action-value function")

    q_table = calculate_q_table(
        mdp,
        policy,
        exact_values
    )

    for state in range(mdp.num_states):

        print(
            f"{state_names[state]:10s} | "
            f"Conservative={q_table[state, 0]:9.4f} | "
            f"Aggressive={q_table[state, 1]:9.4f}"
        )

    # -------------------------------------------------------------------------
    # Demo E: Discount factor
    # -------------------------------------------------------------------------

    print("\n[DEMO E] Discount factor sensitivity")

    gamma_values = np.linspace(
        0.0,
        0.99,
        30
    )

    gamma_results = evaluate_gamma_sensitivity(
        gamma_values,
        policies["Random"]
    )

    print(
        "Start-state value at gamma=0:",
        gamma_results[0, 0]
    )

    print(
        "Start-state value at gamma=0.99:",
        gamma_results[-1, 0]
    )

    # -------------------------------------------------------------------------
    # Demo F: Unit tests
    # -------------------------------------------------------------------------

    run_tests()

    # -------------------------------------------------------------------------
    # Demo G: Integrated visualizations
    # -------------------------------------------------------------------------

    visualize_transition_matrices(mdp)

    visualize_policy_values(
        values_by_policy,
        state_names
    )

    visualize_q_values(
        q_table,
        state_names
    )

    visualize_gamma_sensitivity(
        gamma_values,
        gamma_results
    )

    visualize_bellman_residual(
        residual,
        state_names
    )

    plt.show()

    print("\nPhase 1 completed.")
    print("You are now ready for Dynamic Programming in Phase 2.")


if __name__ == "__main__":
    main()
