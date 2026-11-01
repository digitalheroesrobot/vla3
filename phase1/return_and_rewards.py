
"""
====================================================================
RL_Course
Phase 1.2 — Return, Discount Factor, Reward Design
File: phase1/return_and_rewards.py
====================================================================

PURPOSE OF THIS LESSON
----------------------
Understand how an RL agent evaluates immediate and future rewards,
how reward design influences behavior, and how episode termination
affects the calculation of learning targets.

LEARNING ROADMAP
----------------

STEP 1 — Immediate Reward vs. Return
Purpose:
    Understand the difference between receiving one reward and
    evaluating the complete sequence of rewards.

What you learn:
    - Immediate reward r_t.
    - Episodic return G_t.
    - Discounted cumulative reward.

Implementation:
    Calculate the return of several reward sequences.

STEP 2 — Discount Factor
Purpose:
    Understand how an agent values future rewards relative to
    immediate rewards.

What you learn:
    - Discount factor gamma.
    - Short-term versus long-term preferences.
    - The effect of gamma on optimal decisions.
    - Why gamma = 0 and gamma close to 1 behave differently.

Implementation:
    Compare two strategies under different discount factors.

STEP 3 — Return-to-Go
Purpose:
    Calculate the future return starting from every time step.

What you learn:
    G_t = r_(t+1) + gamma * G_(t+1)

    Return-to-go is important in Monte Carlo learning and
    policy-gradient algorithms.

Implementation:
    Calculate return-to-go using backward recursion and verify
    it against direct calculation.

STEP 4 — Reward Design
Purpose:
    Understand how reward definitions influence agent behavior.

What you learn:
    - Sparse rewards.
    - Dense rewards.
    - Reward shaping.
    - Potential-based reward shaping.

    Potential-based shaping:

        F(s, s') = gamma * Phi(s') - Phi(s)

    Under the standard discounted-MDP assumptions, this shaping
    preserves the optimal policy while changing the learning signal.

Implementation:
    Compare original and shaped reward sequences.

STEP 5 — Termination vs. Truncation
Purpose:
    Understand two different reasons why an episode can end.

What you learn:
    terminated:
        The task reached a genuine terminal state.

    truncated:
        The simulation stopped because of an external limit,
        such as a maximum number of steps.

    These conditions affect value-function bootstrapping.

Implementation:
    Build a small episode simulator and distinguish both cases.

STEP 6 — TD Target
Purpose:
    Understand how an RL algorithm estimates a learning target
    from immediate reward and the next state's estimated value.

What you learn:

    y = r + gamma * V(s')

    For a genuine terminal state:

    y = r

    For a time-limit truncation, bootstrapping is normally retained
    if the underlying task continues beyond the time limit.

Implementation:
    Calculate and compare TD targets for different episode endings.

STEP 7 — Visualization and Verification
Purpose:
    Observe how mathematical choices affect learning objectives.

What you learn:
    - Discounted return curves.
    - Reward contribution over time.
    - Return-to-go.
    - Reward shaping.
    - TD target differences.

Implementation:
    Generate visualizations and execute automated tests.

LEARNING CYCLE
--------------
Understand (comments)
    -> Implement
    -> Visualize
    -> Verify
    -> Experiment
    -> Apply

REQUIREMENTS
------------
pip install numpy matplotlib

RUN
---
python phase1/return_and_rewards.py
====================================================================
"""

import numpy as np
import matplotlib.pyplot as plt


# ================================================================
# GLOBAL CONFIGURATION
# ================================================================

DEFAULT_GAMMA = 0.99

STEP_REWARD = -0.04
GOAL_REWARD = 1.0

RANDOM_SEED = 42


# ================================================================
# STEP 1 — IMMEDIATE REWARD VS. RETURN
# ================================================================

def calculate_return(rewards, gamma=DEFAULT_GAMMA):
    """
    Calculate discounted episodic return:

        G_0 = r_1 + gamma*r_2 + gamma^2*r_3 + ...

    The reward at time step t is discounted by gamma^t.

    gamma = 0:
        Only the first reward matters.

    gamma = 1:
        All rewards receive equal weight.

    This function assumes a finite reward sequence.
    """
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be between 0 and 1.")

    rewards = np.asarray(rewards, dtype=np.float64)

    discounts = gamma ** np.arange(len(rewards))

    return float(np.sum(discounts * rewards))


def calculate_total_reward(rewards):
    """
    Undiscounted cumulative reward:

        R_total = sum(r_t)

    This is different from discounted return when gamma < 1.
    """
    return float(np.sum(rewards))


def reward_contributions(rewards, gamma):
    """
    Calculate the discounted contribution of each reward:

        contribution_t = gamma^t * r_t

    Useful for understanding which rewards dominate the return.
    """
    rewards = np.asarray(rewards, dtype=np.float64)

    discounts = gamma ** np.arange(len(rewards))

    return discounts * rewards


# ================================================================
# STEP 2 — DISCOUNT FACTOR EXPERIMENT
# ================================================================

def compare_strategies(gamma):
    """
    Compare two strategies.

    Strategy A:
        Receive reward 3 immediately.
        Total reward sequence: [3, 0, 0]

    Strategy B:
        Receive reward 10 after two time steps.
        Total reward sequence: [0, 0, 10]

    Their discounted returns are:

        G_A = 3

        G_B = 10 * gamma^2

    Strategy B becomes preferable when:

        10 * gamma^2 > 3

        gamma > sqrt(0.3)

    This illustrates how gamma changes the preference
    between immediate and delayed rewards.
    """
    strategy_a = [3.0, 0.0, 0.0]
    strategy_b = [0.0, 0.0, 10.0]

    return {
        "strategy_a": calculate_return(
            strategy_a,
            gamma
        ),
        "strategy_b": calculate_return(
            strategy_b,
            gamma
        )
    }


def find_strategy_threshold():
    """
    Calculate the discount-factor threshold where both strategies
    have equal discounted return.
    """
    return float(np.sqrt(3.0 / 10.0))


# ================================================================
# STEP 3 — RETURN-TO-GO
# ================================================================

def calculate_returns_to_go(rewards, gamma):
    """
    Calculate return-to-go for every time step.

        G_t = r_(t+1) + gamma * G_(t+1)

    Backward recursion:

        G_last = r_last

        G_t = r_t + gamma * G_(t+1)

    The output at index t is the discounted return beginning
    with reward[t].

    This is more efficient than independently recalculating
    every future return.
    """
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be between 0 and 1.")

    rewards = np.asarray(rewards, dtype=np.float64)

    returns = np.zeros_like(rewards)

    running_return = 0.0

    for t in reversed(range(len(rewards))):

        running_return = (
            rewards[t] + gamma * running_return
        )

        returns[t] = running_return

    return returns


# ================================================================
# STEP 4 — REWARD DESIGN
# ================================================================

def sparse_reward(reached_goal):
    """
    Sparse reward:

        +1 when the goal is reached.
         0 otherwise.

    Advantage:
        Simple objective.

    Disadvantage:
        The agent may receive little feedback during exploration.
    """
    return 1.0 if reached_goal else 0.0


def dense_reward(reached_goal):
    """
    Dense reward:

        +1 when the goal is reached.
        -0.04 for each nonterminal transition.

    Advantage:
        Provides feedback at every step.

    Caution:
        Poorly designed dense rewards can create unintended behavior.
    """
    return GOAL_REWARD if reached_goal else STEP_REWARD


def potential_based_shaping(
    state_potential,
    next_state_potential,
    gamma
):
    """
    Potential-based reward shaping:

        F(s, s') = gamma * Phi(s') - Phi(s)

    The shaped reward is:

        R'(s, a, s') = R(s, a, s') + F(s, s')

    Under standard assumptions, including consistent terminal-state
    potentials, this transformation preserves optimal policies
    in a discounted MDP.

    The potential function Phi represents a heuristic estimate
    of how promising a state is.

    IMPORTANT:
        Arbitrary reward bonuses are not guaranteed to preserve
        the original optimal policy.
    """
    return (
        gamma * next_state_potential -
        state_potential
    )


def shaped_reward(
    original_reward,
    state_potential,
    next_state_potential,
    gamma
):
    """
    Calculate the complete shaped reward.
    """
    shaping = potential_based_shaping(
        state_potential,
        next_state_potential,
        gamma
    )

    return original_reward + shaping


def calculate_grid_potential(
    state,
    goal=(4, 4),
    scale=1.0
):
    """
    Simple heuristic potential for a GridWorld.

    Negative Manhattan distance:

        Phi(s) = -scale * distance(s, goal)

    States closer to the goal have higher potential.

    This is an illustrative heuristic, not necessarily
    the true optimal value function.
    """
    row, col = state
    goal_row, goal_col = goal

    distance = (
        abs(row - goal_row) +
        abs(col - goal_col)
    )

    return -scale * distance


def demonstrate_reward_shaping(
    states,
    rewards,
    gamma
):
    """
    Calculate shaped rewards for a complete trajectory.

    The final state is treated as terminal and assigned
    zero potential.

    For a terminal state, setting Phi(terminal)=0 is important
    for the usual episodic policy-invariance argument.
    """
    shaped = []

    for index, reward in enumerate(rewards):

        state = states[index]
        next_state = states[index + 1]

        phi_state = calculate_grid_potential(state)

        if index == len(rewards) - 1:
            phi_next = 0.0
        else:
            phi_next = calculate_grid_potential(next_state)

        new_reward = shaped_reward(
            reward,
            phi_state,
            phi_next,
            gamma
        )

        shaped.append(new_reward)

    return np.asarray(shaped)


# ================================================================
# STEP 5 — TERMINATION VS. TRUNCATION
# ================================================================

class EpisodeSimulator:
    """
    Minimal episodic simulator.

    This class demonstrates two different episode-ending conditions.

    terminated:
        The task itself has reached a terminal state.

    truncated:
        An external time limit has stopped the episode.

    For example:
        A robot successfully places an object:
            terminated = True

        A simulation reaches its 100-step time limit:
            truncated = True

    These flags must not be treated as interchangeable.
    """

    def __init__(self, max_steps=5):

        self.max_steps = max_steps
        self.current_step = 0

    def reset(self):

        self.current_step = 0

        return {
            "state": 0,
            "terminated": False,
            "truncated": False
        }

    def step(self, reached_goal=False):

        if reached_goal:
            return {
                "state": 1,
                "reward": 1.0,
                "terminated": True,
                "truncated": False
            }

        self.current_step += 1

        truncated = (
            self.current_step >= self.max_steps
        )

        return {
            "state": 0,
            "reward": -0.04,
            "terminated": False,
            "truncated": truncated
        }


# ================================================================
# STEP 6 — TD TARGET
# ================================================================

def calculate_td_target(
    reward,
    next_state_value,
    gamma,
    terminated,
    truncated=False
):
    """
    One-step TD target.

    For a continuing transition:

        y = r + gamma * V(s')

    For a genuine terminal transition:

        y = r

    For a time-limit truncation:
        Normally bootstrap from V(s') if the time limit is external
        to the underlying task and the final observation is valid.

    Therefore, the bootstrap mask is controlled by terminated,
    not automatically by terminated OR truncated.

    Some finite-horizon tasks have a true terminal time boundary.
    In those tasks, remaining time should be represented in the
    state and the terminal condition handled accordingly.
    """
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be between 0 and 1.")

    if terminated:
        return float(reward)

    return float(
        reward + gamma * next_state_value
    )


def calculate_td_target_batch(
    rewards,
    next_state_values,
    terminated_flags,
    gamma
):
    """
    Vectorized TD target:

        y = r + gamma * (1 - terminated) * V(s')

    The terminal mask prevents bootstrapping beyond genuine
    terminal states.

    Truncation flags are deliberately not used here because
    external time-limit truncation normally retains bootstrapping.
    """
    rewards = np.asarray(rewards, dtype=np.float64)

    next_state_values = np.asarray(
        next_state_values,
        dtype=np.float64
    )

    terminated_flags = np.asarray(
        terminated_flags,
        dtype=np.float64
    )

    return (
        rewards +
        gamma *
        (1.0 - terminated_flags) *
        next_state_values
    )


# ================================================================
# STEP 7 — VISUALIZATION FUNCTIONS
# ================================================================

def visualize_discount_factor():
    """
    Plot how the discount factor changes the preference
    between immediate and delayed rewards.
    """
    gammas = np.linspace(0.0, 1.0, 200)

    strategy_a = []
    strategy_b = []

    for gamma in gammas:

        result = compare_strategies(gamma)

        strategy_a.append(result["strategy_a"])
        strategy_b.append(result["strategy_b"])

    threshold = find_strategy_threshold()

    plt.figure(figsize=(8, 5))

    plt.plot(
        gammas,
        strategy_a,
        label="Strategy A: immediate reward"
    )

    plt.plot(
        gammas,
        strategy_b,
        label="Strategy B: delayed reward"
    )

    plt.axvline(
        threshold,
        linestyle="--",
        label=f"Preference threshold = {threshold:.3f}"
    )

    plt.xlabel("Discount factor gamma")
    plt.ylabel("Discounted return")
    plt.title("Effect of Discount Factor on Strategy Preference")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()


def visualize_reward_contributions():
    """
    Plot the contribution of each reward to discounted return.
    """
    rewards = [1.0, 1.0, 1.0, 1.0, 1.0]

    gammas = [0.5, 0.8, 0.99]

    steps = np.arange(len(rewards))

    plt.figure(figsize=(8, 5))

    width = 0.25

    for index, gamma in enumerate(gammas):

        contributions = reward_contributions(
            rewards,
            gamma
        )

        plt.bar(
            steps + index * width,
            contributions,
            width=width,
            label=f"gamma={gamma}"
        )

    plt.xlabel("Reward time step")
    plt.ylabel("Discounted contribution")
    plt.title("How Future Rewards Are Discounted")
    plt.xticks(
        steps + width,
        [f"t={t}" for t in steps]
    )

    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()


def visualize_return_to_go():
    """
    Display reward sequence and return-to-go.
    """
    rewards = np.array([
        0.0,
        -0.04,
        -0.04,
        -0.04,
        1.0
    ])

    gamma = 0.9

    returns = calculate_returns_to_go(
        rewards,
        gamma
    )

    steps = np.arange(len(rewards))

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(8, 7)
    )

    axes[0].bar(
        steps,
        rewards
    )

    axes[0].set_title("Immediate Rewards")
    axes[0].set_ylabel("Reward")
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].plot(
        steps,
        returns,
        marker="o"
    )

    axes[1].set_title(
        f"Return-to-Go (gamma={gamma})"
    )

    axes[1].set_xlabel("Time step")
    axes[1].set_ylabel("Future return")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()


def visualize_reward_shaping():
    """
    Compare original and shaped reward signals.
    """
    states = [
        (0, 0),
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 3),
        (2, 3),
        (3, 3),
        (4, 3),
        (4, 4)
    ]

    rewards = [
        STEP_REWARD,
        STEP_REWARD,
        STEP_REWARD,
        STEP_REWARD,
        STEP_REWARD,
        STEP_REWARD,
        STEP_REWARD,
        GOAL_REWARD
    ]

    gamma = 0.99

    shaped = demonstrate_reward_shaping(
        states,
        rewards,
        gamma
    )

    steps = np.arange(len(rewards))

    plt.figure(figsize=(9, 5))

    plt.plot(
        steps,
        rewards,
        marker="o",
        label="Original reward"
    )

    plt.plot(
        steps,
        shaped,
        marker="s",
        label="Potential-based shaped reward"
    )

    plt.xlabel("Time step")
    plt.ylabel("Reward")
    plt.title("Original Reward vs. Shaped Reward")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()


def visualize_td_targets():
    """
    Compare targets for continuing, terminal, and truncated
    transitions.
    """
    reward = 1.0
    next_value = 5.0
    gamma = 0.9

    cases = {
        "Continuing": calculate_td_target(
            reward,
            next_value,
            gamma,
            terminated=False,
            truncated=False
        ),

        "Terminated": calculate_td_target(
            reward,
            next_value,
            gamma,
            terminated=True,
            truncated=False
        ),

        "Truncated": calculate_td_target(
            reward,
            next_value,
            gamma,
            terminated=False,
            truncated=True
        )
    }

    plt.figure(figsize=(7, 4))

    plt.bar(
        list(cases.keys()),
        list(cases.values())
    )

    plt.ylabel("TD target")
    plt.title("Effect of Episode Ending on TD Target")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()


# ================================================================
# UNIT TESTS
# ================================================================

def run_tests():

    print("\n========== UNIT TESTS ==========\n")

    # Test 1: Undiscounted return
    rewards = [1.0, 2.0, 3.0]

    result = calculate_return(
        rewards,
        gamma=1.0
    )

    assert np.isclose(result, 6.0)

    print("[PASS] Undiscounted return")

    # Test 2: Discounted return
    result = calculate_return(
        [1.0, 1.0, 1.0],
        gamma=0.9
    )

    assert np.isclose(
        result,
        1.0 + 0.9 + 0.81
    )

    print("[PASS] Discounted return")

    # Test 3: Gamma zero
    result = calculate_return(
        [5.0, 10.0, 20.0],
        gamma=0.0
    )

    assert np.isclose(result, 5.0)

    print("[PASS] Gamma zero")

    # Test 4: Total reward
    result = calculate_total_reward(
        [1.0, 2.0, 3.0]
    )

    assert np.isclose(result, 6.0)

    print("[PASS] Total reward")

    # Test 5: Reward contributions
    contributions = reward_contributions(
        [1.0, 1.0, 1.0],
        0.5
    )

    assert np.allclose(
        contributions,
        [1.0, 0.5, 0.25]
    )

    print("[PASS] Reward contributions")

    # Test 6: Strategy preference
    result = compare_strategies(0.5)

    assert result["strategy_a"] > result["strategy_b"]

    print("[PASS] Immediate reward preference")

    # Test 7: Delayed reward preference
    result = compare_strategies(0.9)

    assert result["strategy_b"] > result["strategy_a"]

    print("[PASS] Delayed reward preference")

    # Test 8: Threshold
    threshold = find_strategy_threshold()

    result = compare_strategies(threshold)

    assert np.isclose(
        result["strategy_a"],
        result["strategy_b"]
    )

    print("[PASS] Strategy threshold")

    # Test 9: Return-to-go
    returns = calculate_returns_to_go(
        [1.0, 2.0, 3.0],
        gamma=0.5
    )

    assert np.allclose(
        returns,
        [2.75, 3.5, 3.0]
    )

    print("[PASS] Return-to-go")

    # Test 10: Return-to-go versus direct calculation
    rewards = [0.5, -0.2, 1.0, 2.0]

    returns = calculate_returns_to_go(
        rewards,
        gamma=0.9
    )

    for t in range(len(rewards)):

        direct = calculate_return(
            rewards[t:],
            gamma=0.9
        )

        assert np.isclose(
            returns[t],
            direct
        )

    print("[PASS] Return-to-go numerical verification")

    # Test 11: Potential-based shaping
    shaping = potential_based_shaping(
        state_potential=-3.0,
        next_state_potential=-2.0,
        gamma=0.9
    )

    assert np.isclose(
        shaping,
        1.2
    )

    print("[PASS] Potential-based reward shaping")

    # Test 12: Terminal TD target
    target = calculate_td_target(
        reward=1.0,
        next_state_value=5.0,
        gamma=0.9,
        terminated=True
    )

    assert np.isclose(target, 1.0)

    print("[PASS] Terminal TD target")

    # Test 13: Continuing TD target
    target = calculate_td_target(
        reward=1.0,
        next_state_value=5.0,
        gamma=0.9,
        terminated=False
    )

    assert np.isclose(target, 5.5)

    print("[PASS] Continuing TD target")

    # Test 14: Truncation retains bootstrapping
    target = calculate_td_target(
        reward=1.0,
        next_state_value=5.0,
        gamma=0.9,
        terminated=False,
        truncated=True
    )

    assert np.isclose(target, 5.5)

    print("[PASS] Truncation bootstrapping")

    # Test 15: Batch TD targets
    targets = calculate_td_target_batch(
        rewards=[1.0, 1.0],
        next_state_values=[5.0, 5.0],
        terminated_flags=[0.0, 1.0],
        gamma=0.9
    )

    assert np.allclose(
        targets,
        [5.5, 1.0]
    )

    print("[PASS] Batch TD targets")

    # Test 16: Episode simulator termination
    simulator = EpisodeSimulator(max_steps=5)

    simulator.reset()

    result = simulator.step(reached_goal=True)

    assert result["terminated"] is True
    assert result["truncated"] is False

    print("[PASS] Natural episode termination")

    # Test 17: Episode simulator truncation
    simulator.reset()

    for _ in range(5):
        result = simulator.step(reached_goal=False)

    assert result["terminated"] is False
    assert result["truncated"] is True

    print("[PASS] Time-limit truncation")

    print("\nAll 17 tests passed successfully.\n")


# ================================================================
# MAIN
# ================================================================

def main():

    print("\n============================================")
    print("RL_Course — Phase 1.2")
    print("Return, Discount Factor and Reward Design")
    print("============================================")

    # ------------------------------------------------------------
    # STEP 1 — Immediate reward versus return
    # ------------------------------------------------------------

    print("\n[STEP 1] Immediate Reward vs. Return")

    rewards = [1.0, 1.0, 1.0, 1.0]

    print("Rewards:", rewards)

    print(
        "Total reward:",
        calculate_total_reward(rewards)
    )

    print(
        "Discounted return (gamma=0.9):",
        calculate_return(rewards, gamma=0.9)
    )

    # ------------------------------------------------------------
    # STEP 2 — Discount factor
    # ------------------------------------------------------------

    print("\n[STEP 2] Discount Factor")

    for gamma in [0.0, 0.3, 0.5, 0.8, 0.9, 0.99, 1.0]:

        result = compare_strategies(gamma)

        preferred = (
            "Strategy A"
            if result["strategy_a"] > result["strategy_b"]
            else "Strategy B"
        )

        print(
            f"gamma={gamma:.2f} | "
            f"A={result['strategy_a']:.3f} | "
            f"B={result['strategy_b']:.3f} | "
            f"Preferred={preferred}"
        )

    print(
        "Preference threshold:",
        find_strategy_threshold()
    )

    # ------------------------------------------------------------
    # STEP 3 — Return-to-go
    # ------------------------------------------------------------

    print("\n[STEP 3] Return-to-Go")

    rewards = [0.0, -0.04, -0.04, -0.04, 1.0]

    returns = calculate_returns_to_go(
        rewards,
        gamma=0.9
    )

    for index, value in enumerate(returns):

        print(
            f"Time step {index}: "
            f"reward={rewards[index]:.2f}, "
            f"return-to-go={value:.4f}"
        )

    # ------------------------------------------------------------
    # STEP 4 — Reward design
    # ------------------------------------------------------------

    print("\n[STEP 4] Reward Design")

    print(
        "Sparse reward, intermediate step:",
        sparse_reward(False)
    )

    print(
        "Sparse reward, goal:",
        sparse_reward(True)
    )

    print(
        "Dense reward, intermediate step:",
        dense_reward(False)
    )

    print(
        "Dense reward, goal:",
        dense_reward(True)
    )

    states = [
        (0, 0),
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 3),
        (2, 3),
        (3, 3),
        (4, 3),
        (4, 4)
    ]

    original_rewards = [
        STEP_REWARD,
        STEP_REWARD,
        STEP_REWARD,
        STEP_REWARD,
        STEP_REWARD,
        STEP_REWARD,
        STEP_REWARD,
        GOAL_REWARD
    ]

    shaped = demonstrate_reward_shaping(
        states,
        original_rewards,
        gamma=0.99
    )

    print("Original rewards:", original_rewards)
    print("Shaped rewards:", shaped)

    # ------------------------------------------------------------
    # STEP 5 — Termination versus truncation
    # ------------------------------------------------------------

    print("\n[STEP 5] Termination vs. Truncation")

    simulator = EpisodeSimulator(max_steps=3)

    simulator.reset()

    for step in range(3):

        result = simulator.step(
            reached_goal=False
        )

        print(
            f"Step {step + 1}: "
            f"terminated={result['terminated']}, "
            f"truncated={result['truncated']}"
        )

    # ------------------------------------------------------------
    # STEP 6 — TD targets
    # ------------------------------------------------------------

    print("\n[STEP 6] TD Target")

    reward = 1.0
    next_value = 5.0
    gamma = 0.9

    continuing_target = calculate_td_target(
        reward,
        next_value,
        gamma,
        terminated=False
    )

    terminal_target = calculate_td_target(
        reward,
        next_value,
        gamma,
        terminated=True
    )

    truncated_target = calculate_td_target(
        reward,
        next_value,
        gamma,
        terminated=False,
        truncated=True
    )

    print("Continuing target:", continuing_target)
    print("Terminal target:", terminal_target)
    print("Truncated target:", truncated_target)

    # ------------------------------------------------------------
    # STEP 7 — Unit tests
    # ------------------------------------------------------------

    print("\n[STEP 7] Unit Tests")

    run_tests()

    # ------------------------------------------------------------
    # STEP 7 — Visualizations
    # ------------------------------------------------------------

    print("Generating visualizations...")

    visualize_discount_factor()
    visualize_reward_contributions()
    visualize_return_to_go()
    visualize_reward_shaping()
    visualize_td_targets()

    plt.show()

    print("\nPhase 1.2 completed successfully.")


if __name__ == "__main__":
    main()
