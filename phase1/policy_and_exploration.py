
"""
===============================================================================
RL COURSE — PHASE 1.3
Policy, State Space, Action Space, and Exploration
===============================================================================

LESSON ROADMAP
--------------

STEP 1 — Policy Fundamentals
Purpose:
    Understand how an RL agent selects actions.

What you learn:
    - Deterministic policy: pi(s) = a
    - Stochastic policy: pi(a|s) = P(A=a | S=s)
    - Difference between policy, value function, and Q-function.

Implementation:
    - deterministic_policy()
    - uniform_policy()
    - greedy_policy()
    - epsilon_greedy_policy()

STEP 2 — Action Probabilities
Purpose:
    Represent a policy as a probability distribution over actions.

What you learn:
    - Policy probabilities must be nonnegative.
    - The sum of action probabilities must equal 1.
    - Tied best actions must be handled correctly.
    - Policy entropy measures action-selection uncertainty.

Implementation:
    - epsilon_greedy_probs()
    - categorical_entropy()
    - sample_action()

STEP 3 — Exploration and Exploitation
Purpose:
    Understand why an agent must sometimes choose actions that do not
    currently appear to be the best.

What you learn:
    - Exploitation: choose the action with the highest estimated value.
    - Exploration: try other actions to gather information.
    - Epsilon controls the exploration probability.
    - A greedy agent can become trapped by an initially misleading reward.

Implementation:
    - Epsilon-greedy action selection.
    - Linear epsilon decay.
    - Multi-armed bandit experiment.

STEP 4 — Action-Value Learning
Purpose:
    Learn how an agent updates its estimates from observed rewards.

What you learn:
    - Q(a) is an estimate of the expected reward of action a.
    - Sample-average update:
          Q_new(a) = Q_old(a) + (R - Q_old(a)) / N(a)
    - Exploration and learning are separate concepts:
      epsilon-greedy selects actions; the update rule learns their values.

Implementation:
    - MultiArmedBandit
    - sample_average_update()
    - run_bandit_experiment()

STEP 5 — Numerical Verification
Purpose:
    Verify that policy calculations and learning updates are correct.

What you learn:
    - Probability normalization.
    - Correct greedy selection.
    - Epsilon-greedy behavior.
    - Entropy properties.
    - Reproducible random sampling.

Implementation:
    - run_tests()

STEP 6 — Visualization and Experimentation
Purpose:
    Observe the practical consequences of different exploration policies.

What you learn:
    - Average reward over time.
    - Percentage of optimal actions.
    - Effect of epsilon decay.
    - Action probability distributions.
    - Policy entropy.

Implementation:
    - All visualization functions are included below.
    - main() executes all demonstrations and visualizations.

===============================================================================
MATHEMATICAL FOUNDATION
===============================================================================

1. DETERMINISTIC POLICY

    pi(s) = a

    Given a state s, the policy always selects the same action a.

2. STOCHASTIC POLICY

    pi(a|s) = P(A_t = a | S_t = s)

    For every state s:

        sum_a pi(a|s) = 1

3. ACTION-VALUE FUNCTION

    Q(s,a) = E[G_t | S_t=s, A_t=a]

    Q(s,a) estimates the expected future return after selecting action a
    in state s.

    Important distinction:

        Policy:
            Determines which action to select.

        Value function V(s):
            Estimates the expected return from state s.

        Action-value function Q(s,a):
            Estimates the expected return from state s after action a.

4. GREEDY POLICY

    pi(s) = argmax_a Q(s,a)

    Always selects an action with the highest estimated action value.

5. EPSILON-GREEDY POLICY

    With probability epsilon:
        Explore by selecting a random action.

    With probability 1-epsilon:
        Exploit by selecting a greedy action.

    For n actions and a unique greedy action:

        pi(a|s) = 1-epsilon + epsilon/n, if a is greedy

        pi(a|s) = epsilon/n, otherwise

    If multiple actions tie for the maximum Q value, exploitation
    probability is distributed equally among those tied actions.

6. POLICY ENTROPY

    H(pi) = -sum_a pi(a|s) log(pi(a|s))

    High entropy:
        More uncertainty or randomness in action selection.

    Low entropy:
        More concentrated action-selection probabilities.

7. SAMPLE-AVERAGE ACTION-VALUE UPDATE

    Q_new(a) = Q_old(a) + (R - Q_old(a)) / N(a)

    This is an incremental calculation of the average observed reward.

8. EPSILON DECAY

    epsilon(t) = max(epsilon_min,
                     epsilon_start - decay_rate * t)

    Early training:
        More exploration.

    Later training:
        More exploitation.

9. MULTI-ARMED BANDIT

    A bandit has one decision state and multiple actions.

    Each action has an unknown reward distribution.

    The agent must discover which action has the highest expected reward.

    This experiment isolates exploration from state-transition complexity.
    It is not yet a full multi-state RL problem.

===============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# STEP 1 — POLICY FUNDAMENTALS
# =============================================================================

def deterministic_policy(action_values: np.ndarray) -> int:
    """
    Select the action with the highest estimated value.

    pi(s) = argmax_a Q(s,a)

    If several actions tie, NumPy returns the first maximum.
    """
    q = np.asarray(action_values, dtype=float)

    if q.ndim != 1 or q.size == 0:
        raise ValueError("action_values must be a non-empty 1D array")

    if not np.all(np.isfinite(q)):
        raise ValueError("action_values must contain finite numbers")

    return int(np.argmax(q))


def uniform_policy(num_actions: int) -> np.ndarray:
    """
    Return a uniform stochastic policy.

    pi(a|s) = 1 / number_of_actions
    """
    if num_actions <= 0:
        raise ValueError("num_actions must be positive")

    return np.full(num_actions, 1.0 / num_actions)


def greedy_policy(action_values: np.ndarray) -> np.ndarray:
    """
    Return a stochastic representation of a greedy policy.

    Probability is distributed equally among all actions tied
    for the maximum estimated action value.
    """
    q = np.asarray(action_values, dtype=float)

    if q.ndim != 1 or q.size == 0:
        raise ValueError("action_values must be a non-empty 1D array")

    if not np.all(np.isfinite(q)):
        raise ValueError("action_values must contain finite numbers")

    best_actions = np.isclose(q, np.max(q))
    probabilities = best_actions.astype(float)

    return probabilities / np.sum(probabilities)


# =============================================================================
# STEP 2 — EPSILON-GREEDY POLICY
# =============================================================================

def epsilon_greedy_probs(
    action_values: np.ndarray,
    epsilon: float
) -> np.ndarray:
    """
    Calculate epsilon-greedy action probabilities.

    The exploration component assigns epsilon / n to every action.

    The exploitation component distributes 1-epsilon equally
    among all greedy actions.

    This handles ties without arbitrarily favoring one action.
    """
    q = np.asarray(action_values, dtype=float)

    if q.ndim != 1 or q.size == 0:
        raise ValueError("action_values must be a non-empty 1D array")

    if not np.all(np.isfinite(q)):
        raise ValueError("action_values must contain finite numbers")

    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must be between 0 and 1")

    num_actions = len(q)

    exploration = np.full(
        num_actions,
        epsilon / num_actions
    )

    exploitation = greedy_policy(q) * (1.0 - epsilon)

    probabilities = exploration + exploitation

    return probabilities


def categorical_entropy(probabilities: np.ndarray) -> float:
    """
    Calculate Shannon entropy of a categorical policy.

        H(pi) = -sum p(a) log(p(a))

    Zero-probability terms contribute zero.
    """
    p = np.asarray(probabilities, dtype=float)

    if p.ndim != 1 or p.size == 0:
        raise ValueError("probabilities must be a non-empty 1D array")

    if np.any(p < 0):
        raise ValueError("probabilities cannot be negative")

    if not np.isclose(np.sum(p), 1.0):
        raise ValueError("probabilities must sum to 1")

    nonzero = p > 0

    return float(-np.sum(p[nonzero] * np.log(p[nonzero])))


def sample_action(
    probabilities: np.ndarray,
    rng: np.random.Generator
) -> int:
    """
    Sample an action according to a categorical distribution.
    """
    p = np.asarray(probabilities, dtype=float)

    if p.ndim != 1 or p.size == 0:
        raise ValueError("probabilities must be a non-empty 1D array")

    if np.any(p < 0) or not np.all(np.isfinite(p)):
        raise ValueError("probabilities must be finite and nonnegative")

    if not np.isclose(np.sum(p), 1.0):
        raise ValueError("probabilities must sum to 1")

    return int(rng.choice(len(p), p=p))


# =============================================================================
# STEP 3 — EPSILON SCHEDULE
# =============================================================================

def linear_epsilon_schedule(
    step: int,
    epsilon_start: float = 1.0,
    epsilon_min: float = 0.05,
    decay_steps: int = 1000
) -> float:
    """
    Linearly decrease epsilon from epsilon_start to epsilon_min.

    Before decay_steps:
        epsilon decreases linearly.

    After decay_steps:
        epsilon remains at epsilon_min.
    """
    if step < 0:
        raise ValueError("step cannot be negative")

    if decay_steps <= 0:
        raise ValueError("decay_steps must be positive")

    if not 0.0 <= epsilon_min <= epsilon_start <= 1.0:
        raise ValueError(
            "Require 0 <= epsilon_min <= epsilon_start <= 1"
        )

    progress = min(step / decay_steps, 1.0)

    return float(
        epsilon_start
        - progress * (epsilon_start - epsilon_min)
    )


# =============================================================================
# STEP 4 — MULTI-ARMED BANDIT ENVIRONMENT
# =============================================================================

@dataclass
class BanditResult:
    """
    Results collected during one bandit run.
    """
    rewards: np.ndarray
    actions: np.ndarray
    optimal_actions: np.ndarray
    q_values: np.ndarray
    action_counts: np.ndarray
    epsilons: np.ndarray


class MultiArmedBandit:
    """
    A stationary Gaussian multi-armed bandit.

    Each action a produces:

        R ~ Normal(true_mean[a], reward_std)

    The agent does not know true_mean.

    The agent must estimate action values from observed rewards.
    """

    def __init__(
        self,
        true_means: np.ndarray,
        reward_std: float = 1.0,
        seed: int = 42
    ):
        self.true_means = np.asarray(
            true_means,
            dtype=float
        )

        if self.true_means.ndim != 1 or self.true_means.size == 0:
            raise ValueError("true_means must be a non-empty 1D array")

        if not np.all(np.isfinite(self.true_means)):
            raise ValueError("true_means must contain finite values")

        if reward_std < 0:
            raise ValueError("reward_std cannot be negative")

        self.reward_std = float(reward_std)
        self.rng = np.random.default_rng(seed)

        self.num_actions = len(self.true_means)
        self.optimal_action = int(np.argmax(self.true_means))

    def step(self, action: int) -> float:
        """
        Execute one action and return its observed reward.
        """
        if not 0 <= action < self.num_actions:
            raise ValueError("Invalid action")

        return float(
            self.rng.normal(
                self.true_means[action],
                self.reward_std
            )
        )


# =============================================================================
# STEP 5 — ACTION-VALUE UPDATE
# =============================================================================

def sample_average_update(
    old_value: float,
    reward: float,
    count: int
) -> float:
    """
    Incremental sample-average update.

        Q_new = Q_old + (R - Q_old) / N

    count is the updated number of observations for this action.
    """
    if count <= 0:
        raise ValueError("count must be positive")

    return float(
        old_value + (reward - old_value) / count
    )


# =============================================================================
# STEP 6 — BANDIT EXPERIMENT
# =============================================================================

def run_bandit_experiment(
    true_means: np.ndarray,
    steps: int,
    epsilon_mode: str,
    seed: int = 42,
    epsilon: float = 0.1,
    epsilon_start: float = 1.0,
    epsilon_min: float = 0.05,
    decay_steps: int = 1000,
    optimistic_initial_value: float = 0.0
) -> BanditResult:
    """
    Run one bandit experiment.

    epsilon_mode:
        "greedy"  : epsilon = 0
        "fixed"   : constant epsilon
        "decay"   : linearly decreasing epsilon

    The action-value estimates are updated using sample averages.

    This experiment uses a stationary bandit. For nonstationary tasks,
    a constant step-size update is often more appropriate.
    """
    if steps <= 0:
        raise ValueError("steps must be positive")

    if epsilon_mode not in ("greedy", "fixed", "decay"):
        raise ValueError("Unknown epsilon_mode")

    env = MultiArmedBandit(
        true_means=true_means,
        seed=seed
    )

    rng = np.random.default_rng(seed + 1000)

    n = env.num_actions

    q_values = np.full(
        n,
        optimistic_initial_value,
        dtype=float
    )

    action_counts = np.zeros(n, dtype=int)

    rewards = np.zeros(steps)
    actions = np.zeros(steps, dtype=int)
    optimal_actions = np.zeros(steps, dtype=int)
    epsilons = np.zeros(steps)

    for t in range(steps):

        if epsilon_mode == "greedy":
            current_epsilon = 0.0

        elif epsilon_mode == "fixed":
            current_epsilon = epsilon

        else:
            current_epsilon = linear_epsilon_schedule(
                t,
                epsilon_start=epsilon_start,
                epsilon_min=epsilon_min,
                decay_steps=decay_steps
            )

        probabilities = epsilon_greedy_probs(
            q_values,
            current_epsilon
        )

        action = sample_action(
            probabilities,
            rng
        )

        reward = env.step(action)

        action_counts[action] += 1

        q_values[action] = sample_average_update(
            old_value=q_values[action],
            reward=reward,
            count=action_counts[action]
        )

        rewards[t] = reward
        actions[t] = action
        optimal_actions[t] = (
            1 if action == env.optimal_action else 0
        )
        epsilons[t] = current_epsilon

    return BanditResult(
        rewards=rewards,
        actions=actions,
        optimal_actions=optimal_actions,
        q_values=q_values.copy(),
        action_counts=action_counts.copy(),
        epsilons=epsilons
    )


def run_multiple_experiments(
    true_means: np.ndarray,
    steps: int,
    runs: int,
    modes: Dict[str, Dict]
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Average results over multiple independent runs.

    This reduces the effect of randomness in a single experiment.
    """
    results = {}

    for mode_name, config in modes.items():

        all_rewards = []
        all_optimal = []

        for run in range(runs):

            result = run_bandit_experiment(
                true_means=true_means,
                steps=steps,
                seed=1000 + run,
                **config
            )

            all_rewards.append(result.rewards)
            all_optimal.append(result.optimal_actions)

        results[mode_name] = {
            "average_reward": np.mean(
                all_rewards,
                axis=0
            ),
            "optimal_action_rate": np.mean(
                all_optimal,
                axis=0
            )
        }

    return results


# =============================================================================
# STEP 7 — VISUALIZATION
# =============================================================================

def moving_average(
    values: np.ndarray,
    window: int = 100
) -> np.ndarray:
    """
    Calculate a moving average for visualization.
    """
    values = np.asarray(values, dtype=float)

    if window <= 0:
        raise ValueError("window must be positive")

    if len(values) < window:
        return np.array([])

    kernel = np.ones(window) / window

    return np.convolve(
        values,
        kernel,
        mode="valid"
    )


def visualize_policy_probabilities():
    """
    Compare uniform, greedy, and epsilon-greedy policies.
    """
    q_values = np.array([1.0, 2.0, 4.0, 3.0])

    policies = {
        "Uniform": uniform_policy(4),
        "Greedy": greedy_policy(q_values),
        "Epsilon=0.1": epsilon_greedy_probs(q_values, 0.1),
        "Epsilon=0.5": epsilon_greedy_probs(q_values, 0.5)
    }

    names = list(policies.keys())
    x = np.arange(len(q_values))
    width = 0.18

    plt.figure(figsize=(10, 5))

    for i, name in enumerate(names):
        offset = (i - (len(names) - 1) / 2) * width

        plt.bar(
            x + offset,
            policies[name],
            width,
            label=name
        )

    plt.xticks(x, [f"Action {i}" for i in x])
    plt.ylabel("Action probability")
    plt.title("Comparison of Action-Selection Policies")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()


def visualize_epsilon_decay():
    """
    Visualize how the exploration probability changes over time.
    """
    steps = np.arange(2001)

    epsilons = [
        linear_epsilon_schedule(
            int(t),
            epsilon_start=1.0,
            epsilon_min=0.05,
            decay_steps=1500
        )
        for t in steps
    ]

    plt.figure(figsize=(9, 4))

    plt.plot(steps, epsilons)

    plt.xlabel("Training step")
    plt.ylabel("Epsilon")
    plt.title("Linear Epsilon Decay")
    plt.grid(alpha=0.3)
    plt.tight_layout()


def visualize_policy_entropy():
    """
    Show how epsilon changes the entropy of an epsilon-greedy policy.
    """
    q_values = np.array([1.0, 2.0, 5.0, 3.0])

    epsilon_values = np.linspace(0.0, 1.0, 101)

    entropies = []

    for epsilon in epsilon_values:
        probabilities = epsilon_greedy_probs(
            q_values,
            epsilon
        )

        entropies.append(
            categorical_entropy(probabilities)
        )

    plt.figure(figsize=(9, 4))

    plt.plot(epsilon_values, entropies)

    plt.xlabel("Epsilon")
    plt.ylabel("Policy entropy (nats)")
    plt.title("Exploration and Policy Entropy")
    plt.grid(alpha=0.3)
    plt.tight_layout()


def visualize_bandit_rewards(results):
    """
    Compare moving-average reward across exploration strategies.
    """
    plt.figure(figsize=(11, 5))

    for name, result in results.items():

        rewards = moving_average(
            result["average_reward"],
            window=100
        )

        plt.plot(
            np.arange(len(rewards)),
            rewards,
            label=name
        )

    plt.xlabel("Training step")
    plt.ylabel("Average reward (100-step moving average)")
    plt.title("Bandit: Average Reward Comparison")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()


def visualize_optimal_action_rate(results):
    """
    Compare how often each strategy selects the optimal action.
    """
    plt.figure(figsize=(11, 5))

    for name, result in results.items():

        rate = moving_average(
            result["optimal_action_rate"],
            window=100
        )

        plt.plot(
            np.arange(len(rate)),
            rate * 100,
            label=name
        )

    plt.xlabel("Training step")
    plt.ylabel("Optimal action selection (%)")
    plt.title("Bandit: Optimal Action Selection Rate")
    plt.ylim(0, 105)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()


def visualize_final_action_values(
    true_means: np.ndarray,
    result: BanditResult
):
    """
    Compare the true reward means and learned action-value estimates.
    """
    x = np.arange(len(true_means))
    width = 0.35

    plt.figure(figsize=(9, 5))

    plt.bar(
        x - width / 2,
        true_means,
        width,
        label="True reward mean"
    )

    plt.bar(
        x + width / 2,
        result.q_values,
        width,
        label="Estimated Q(a)"
    )

    plt.xlabel("Action")
    plt.ylabel("Expected reward")
    plt.title("True Action Values vs. Learned Estimates")
    plt.xticks(x, [f"Action {i}" for i in x])
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()


# =============================================================================
# STEP 8 — NUMERICAL VERIFICATION AND UNIT TESTS
# =============================================================================

def run_tests():
    """
    Verify the mathematical and implementation properties.
    """
    print("\n========== UNIT TESTS ==========")

    # Test 1: Deterministic policy
    q = np.array([1.0, 3.0, 2.0])
    assert deterministic_policy(q) == 1
    print("[PASS] Deterministic policy")

    # Test 2: Uniform policy
    p = uniform_policy(4)
    assert np.allclose(p, [0.25, 0.25, 0.25, 0.25])
    assert np.isclose(np.sum(p), 1.0)
    print("[PASS] Uniform policy")

    # Test 3: Greedy policy
    p = greedy_policy(q)
    assert np.allclose(p, [0.0, 1.0, 0.0])
    print("[PASS] Greedy policy")

    # Test 4: Greedy policy with ties
    p = greedy_policy(np.array([2.0, 5.0, 5.0, 1.0]))
    assert np.allclose(p, [0.0, 0.5, 0.5, 0.0])
    print("[PASS] Greedy policy with tied actions")

    # Test 5: Epsilon-greedy probabilities
    p = epsilon_greedy_probs(
        np.array([1.0, 3.0, 2.0]),
        epsilon=0.2
    )

    expected = np.array([
        0.2 / 3,
        0.8 + 0.2 / 3,
        0.2 / 3
    ])

    assert np.allclose(p, expected)
    print("[PASS] Epsilon-greedy probabilities")

    # Test 6: Probability normalization
    for epsilon in np.linspace(0.0, 1.0, 11):
        p = epsilon_greedy_probs(q, float(epsilon))

        assert np.all(p >= 0)
        assert np.isclose(np.sum(p), 1.0)

    print("[PASS] Probability normalization")

    # Test 7: Epsilon-greedy ties
    p = epsilon_greedy_probs(
        np.array([4.0, 4.0, 1.0]),
        epsilon=0.2
    )

    expected = np.array([
        0.4 + 0.2 / 3,
        0.4 + 0.2 / 3,
        0.2 / 3
    ])

    assert np.allclose(p, expected)
    print("[PASS] Epsilon-greedy tied actions")

    # Test 8: Entropy of uniform distribution
    p = uniform_policy(4)
    entropy = categorical_entropy(p)

    assert np.isclose(entropy, np.log(4))
    print("[PASS] Uniform policy entropy")

    # Test 9: Entropy of deterministic policy
    p = np.array([0.0, 1.0, 0.0])
    entropy = categorical_entropy(p)

    assert np.isclose(entropy, 0.0)
    print("[PASS] Deterministic policy entropy")

    # Test 10: Entropy nonnegative
    p = epsilon_greedy_probs(q, 0.3)
    assert categorical_entropy(p) >= 0
    print("[PASS] Nonnegative policy entropy")

    # Test 11: Epsilon schedule initial value
    assert np.isclose(
        linear_epsilon_schedule(0),
        1.0
    )
    print("[PASS] Epsilon schedule initial value")

    # Test 12: Epsilon schedule final value
    assert np.isclose(
        linear_epsilon_schedule(1000),
        0.05
    )
    print("[PASS] Epsilon schedule final value")

    # Test 13: Epsilon schedule remains at minimum
    assert np.isclose(
        linear_epsilon_schedule(2000),
        0.05
    )
    print("[PASS] Epsilon schedule minimum")

    # Test 14: Sample-average update
    value = sample_average_update(
        old_value=2.0,
        reward=6.0,
        count=2
    )

    assert np.isclose(value, 4.0)
    print("[PASS] Sample-average update")

    # Test 15: Sample-average update against direct average
    observations = [2.0, 4.0, 8.0, 6.0]
    estimate = 0.0

    for count, reward in enumerate(observations, start=1):
        estimate = sample_average_update(
            estimate,
            reward,
            count
        )

    assert np.isclose(
        estimate,
        np.mean(observations)
    )
    print("[PASS] Incremental average equals direct average")

    # Test 16: Reproducible action sampling
    p = np.array([0.2, 0.3, 0.5])

    rng1 = np.random.default_rng(123)
    rng2 = np.random.default_rng(123)

    actions1 = [
        sample_action(p, rng1)
        for _ in range(100)
    ]

    actions2 = [
        sample_action(p, rng2)
        for _ in range(100)
    ]

    assert actions1 == actions2
    print("[PASS] Reproducible action sampling")

    # Test 17: Bandit optimal action
    bandit = MultiArmedBandit(
        true_means=np.array([1.0, 5.0, 2.0]),
        seed=42
    )

    assert bandit.optimal_action == 1
    print("[PASS] Bandit optimal action")

    # Test 18: Zero-noise bandit reward
    bandit = MultiArmedBandit(
        true_means=np.array([1.0, 5.0, 2.0]),
        reward_std=0.0,
        seed=42
    )

    assert np.isclose(bandit.step(1), 5.0)
    print("[PASS] Deterministic bandit reward")

    # Test 19: Invalid epsilon
    try:
        epsilon_greedy_probs(q, 1.5)
        raise AssertionError("Invalid epsilon was accepted")
    except ValueError:
        pass

    print("[PASS] Invalid epsilon validation")

    # Test 20: Invalid probability distribution
    try:
        sample_action(
            np.array([0.2, 0.2, 0.2]),
            np.random.default_rng(42)
        )
        raise AssertionError("Invalid probabilities were accepted")
    except ValueError:
        pass

    print("[PASS] Invalid probability validation")

    print("\nAll 20 tests passed.")


# =============================================================================
# STEP 9 — DEMONSTRATION
# =============================================================================

def main():

    print("=" * 72)
    print("RL PHASE 1.3 — POLICY AND EXPLORATION")
    print("=" * 72)

    # -------------------------------------------------------------------------
    # Demo A: Policy probabilities
    # -------------------------------------------------------------------------

    q_values = np.array([1.0, 2.0, 4.0, 3.0])

    print("\n[DEMO A] Policy probabilities")

    print("Estimated action values:", q_values)

    for epsilon in [0.0, 0.1, 0.3, 0.5, 1.0]:

        probabilities = epsilon_greedy_probs(
            q_values,
            epsilon
        )

        entropy = categorical_entropy(probabilities)

        print(
            f"epsilon={epsilon:.1f} | "
            f"probabilities={np.round(probabilities, 3)} | "
            f"entropy={entropy:.4f}"
        )

    # -------------------------------------------------------------------------
    # Demo B: Multi-armed bandit
    # -------------------------------------------------------------------------

    print("\n[DEMO B] Multi-armed bandit")

    # The agent does not know these values.
    # They are used by the environment to generate rewards.
    true_means = np.array([
        1.0,
        1.5,
        2.0,
        1.2,
        2.5,
        1.8,
        0.5,
        1.6,
        2.2,
        1.1
    ])

    optimal_action = int(np.argmax(true_means))

    print("Number of actions:", len(true_means))
    print("True optimal action:", optimal_action)
    print("True optimal expected reward:", true_means[optimal_action])

    steps = 2000
    runs = 100

    modes = {
        "Pure greedy": {
            "epsilon_mode": "greedy"
        },
        "Fixed epsilon=0.1": {
            "epsilon_mode": "fixed",
            "epsilon": 0.1
        },
        "Fixed epsilon=0.3": {
            "epsilon_mode": "fixed",
            "epsilon": 0.3
        },
        "Decaying epsilon": {
            "epsilon_mode": "decay",
            "epsilon_start": 1.0,
            "epsilon_min": 0.05,
            "decay_steps": 1500
        }
    }

    print(
        f"\nRunning {runs} independent experiments, "
        f"{steps} steps each..."
    )

    results = run_multiple_experiments(
        true_means=true_means,
        steps=steps,
        runs=runs,
        modes=modes
    )

    print("\nAverage results over the final 500 steps:")

    for name, result in results.items():

        average_reward = np.mean(
            result["average_reward"][-500:]
        )

        optimal_rate = np.mean(
            result["optimal_action_rate"][-500:]
        )

        print(
            f"{name:22s} | "
            f"Average reward: {average_reward:7.4f} | "
            f"Optimal action: {optimal_rate * 100:6.2f}%"
        )

    # -------------------------------------------------------------------------
    # Demo C: A single learned bandit model
    # -------------------------------------------------------------------------

    print("\n[DEMO C] Learned action values from one experiment")

    single_result = run_bandit_experiment(
        true_means=true_means,
        steps=steps,
        epsilon_mode="decay",
        epsilon_start=1.0,
        epsilon_min=0.05,
        decay_steps=1500,
        seed=42
    )

    print("\nAction | True mean | Estimated Q | Number of selections")

    for action in range(len(true_means)):

        print(
            f"{action:6d} | "
            f"{true_means[action]:9.3f} | "
            f"{single_result.q_values[action]:11.3f} | "
            f"{single_result.action_counts[action]:20d}"
        )

    print(
        "\nLearned greedy action:",
        deterministic_policy(single_result.q_values)
    )

    print(
        "True optimal action:",
        optimal_action
    )

    # -------------------------------------------------------------------------
    # Demo D: Unit tests
    # -------------------------------------------------------------------------

    run_tests()

    # -------------------------------------------------------------------------
    # Demo E: Integrated visualizations
    # -------------------------------------------------------------------------

    visualize_policy_probabilities()

    visualize_epsilon_decay()

    visualize_policy_entropy()

    visualize_bandit_rewards(results)

    visualize_optimal_action_rate(results)

    visualize_final_action_values(
        true_means,
        single_result
    )

    plt.show()

    print("\nLesson completed.")


if __name__ == "__main__":
    main()
