
"""
====================================================================
RL_Course — PHASE 3.6
N-STEP TEMPORAL-DIFFERENCE LEARNING
====================================================================

FILE:
    phase3/n_step_td.py

LESSON ROADMAP
--------------

STEP 1 — Random Walk Environment
Purpose:
    Create a simple episodic Markov decision process.
What you learn:
    State transitions, rewards and terminal states.
Implementation:
    RandomWalkEnvironment.

STEP 2 — Episode Collection
Purpose:
    Record state and reward trajectories.
What you learn:
    How multi-step algorithms use episode history.
Implementation:
    generate_episode().

STEP 3 — N-Step Return
Purpose:
    Calculate returns using multiple future rewards.
What you learn:
    Reward accumulation and bootstrapping.
Implementation:
    calculate_n_step_return().

STEP 4 — N-Step TD Prediction
Purpose:
    Update state values using n-step returns.
What you learn:
    TD(0), n-step TD and Monte Carlo.
Implementation:
    n_step_td_episode().

STEP 5 — Algorithm Comparison
Purpose:
    Compare different n-step horizons.
What you learn:
    Bias-variance tradeoff.
Implementation:
    train_agent() and compare_n_steps().

STEP 6 — Visualization
Purpose:
    Observe learning behavior.
What you learn:
    How n influences value estimates and convergence.
Implementation:
    Visualization functions called from main().

STEP 7 — Numerical Verification
Purpose:
    Verify mathematical correctness.
What you learn:
    Hand-calculated returns and terminal handling.
Implementation:
    run_tests().

MATHEMATICS
-----------

N-STEP RETURN:

G(t:t+n) =
    R(t+1) + gamma*R(t+2) + ... +
    gamma^(n-1)*R(t+n) + gamma^n*V(S(t+n))

N-STEP TD UPDATE:

V(S(t)) <- V(S(t)) +
           alpha * [G(t:t+n) - V(S(t))]

TD(0):
    n = 1

MONTE CARLO:
    Uses the complete remaining episode return.
    No bootstrapping.

BOOTSTRAPPING RULE:
    If the n-step horizon ends before the episode ends,
    bootstrap from the state at the end of the horizon.

    If the horizon reaches a true terminal state,
    do not bootstrap.

    If the horizon reaches an external time-limit
    truncation, bootstrap from the final valid observation.

REAL-WORLD APPLICATION
----------------------

Robot manipulation:
    N-step TD can propagate delayed feedback across
    several preceding robot states.

SAFETY:
    Learned values are not safety guarantees.
    Real robots need independent collision prevention,
    safety constraints and emergency stops.

DEPENDENCIES
------------
    pip install numpy matplotlib

RUN
---
    python phase3/n_step_td.py

====================================================================
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt


# ================================================================
# 1. CONFIGURATION
# ================================================================

SEED = 42

NUM_STATES = 7

LEFT_TERMINAL = 0
RIGHT_TERMINAL = 6

START_STATE = 3

TERMINAL_STATES = {
    LEFT_TERMINAL,
    RIGHT_TERMINAL,
}

GAMMA = 1.0
ALPHA = 0.1

NUM_EPISODES = 100
MAX_STEPS = 1000

N_VALUES = [1, 2, 4, 8, 1000]

NUM_RUNS = 100


# ================================================================
# 2. RANDOM WALK ENVIRONMENT
# ================================================================

@dataclass
class Episode:
    states: list[int]
    rewards: list[float]
    terminated: bool
    truncated: bool

    @property
    def length(self):
        return len(self.rewards)


class RandomWalkEnvironment:
    """
    Seven-state random walk.

    States:
        0  1  2  3  4  5  6

    Start:
        State 3.

    Actions:
        Left:  probability 0.5
        Right: probability 0.5

    Rewards:
        Enter left terminal:  0
        Enter right terminal: +1
        Other transitions:   0

    Terminal states have zero continuation value.
    The reward is received when entering the right terminal.
    """

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.state = START_STATE

    def reset(self):
        self.state = START_STATE
        return self.state

    def step(self):

        if self.state in TERMINAL_STATES:
            raise RuntimeError(
                "Cannot step from a terminal state."
            )

        if self.rng.random() < 0.5:
            next_state = self.state - 1
        else:
            next_state = self.state + 1

        reward = (
            1.0
            if next_state == RIGHT_TERMINAL
            else 0.0
        )

        self.state = next_state

        terminated = (
            next_state in TERMINAL_STATES
        )

        return next_state, reward, terminated


# ================================================================
# 3. EXACT STATE VALUES
# ================================================================

def exact_state_values():
    """
    For gamma=1:

        V(s) = s/6

    for nonterminal states.

    Terminal continuation values are zero.

        V = [0, 1/6, 2/6, 3/6, 4/6, 5/6, 0]
    """

    values = np.zeros(NUM_STATES)

    for state in range(1, NUM_STATES - 1):
        values[state] = state / 6.0

    return values


# ================================================================
# 4. EPISODE GENERATION
# ================================================================

def generate_episode(
    env,
    max_steps=MAX_STEPS,
):
    """
    Return:

        states  = [S0, S1, ..., ST]
        rewards = [R1, R2, ..., RT]

    len(states) = len(rewards) + 1

    terminated:
        True if a genuine terminal state was reached.

    truncated:
        True if the external step limit was reached.
    """

    state = env.reset()

    states = [state]
    rewards = []

    terminated = False

    for _ in range(max_steps):

        next_state, reward, terminated = env.step()

        states.append(next_state)
        rewards.append(reward)

        if terminated:
            break

    truncated = not terminated

    return Episode(
        states=states,
        rewards=rewards,
        terminated=terminated,
        truncated=truncated,
    )


# ================================================================
# 5. N-STEP RETURN
# ================================================================

def calculate_n_step_return(
    states,
    rewards,
    tau,
    n,
    gamma,
    values,
    terminated=True,
):
    """
    Calculate the n-step return beginning at tau.

    G(t:t+n) =
        sum[k=0 to horizon-1]
            gamma^k * R(t+k+1)

        + gamma^horizon * V(S(t+horizon))
          when bootstrapping is valid.

    IMPORTANT BOOTSTRAPPING RULE
    ----------------------------

    Case A:
        The n-step horizon ends BEFORE the episode ends.

        Bootstrap from S(t+n), even if the episode
        eventually terminates later.

    Case B:
        The n-step horizon reaches a TRUE terminal state.

        Do not bootstrap.

    Case C:
        The n-step horizon reaches an externally
        TRUNCATED episode boundary.

        Bootstrap from the final valid observation.
    """

    states = list(states)
    rewards = list(rewards)

    if int(n) != n or n <= 0:
        raise ValueError(
            "n must be a positive integer."
        )

    if not 0 <= gamma <= 1:
        raise ValueError(
            "gamma must be between 0 and 1."
        )

    if len(states) != len(rewards) + 1:
        raise ValueError(
            "states must contain exactly one more "
            "element than rewards."
        )

    if not 0 <= tau < len(rewards):
        raise ValueError(
            "Invalid tau."
        )

    values = np.asarray(values, dtype=float)

    if not np.all(np.isfinite(values)):
        raise ValueError(
            "values must be finite."
        )

    total_steps = len(rewards)

    remaining_steps = total_steps - tau

    horizon = min(
        int(n),
        remaining_steps,
    )

    # ------------------------------------------------------------
    # 1. Accumulate observed rewards.
    # ------------------------------------------------------------

    result = 0.0

    for k in range(horizon):

        result += (
            gamma ** k
        ) * rewards[tau + k]

    # ------------------------------------------------------------
    # 2. Determine whether bootstrapping is valid.
    # ------------------------------------------------------------

    # Condition A:
    # The n-step horizon ends before the episode ends.
    #
    # Example:
    # Episode ends at T=3.
    # tau=0, n=2.
    #
    # Horizon ends at time 2, before terminal time 3.
    # Therefore bootstrap from V(S2).

    horizon_ends_before_episode = (
        horizon == int(n)
        and tau + int(n) < total_steps
    )

    # Condition B:
    # The horizon reaches the external truncation boundary.
    #
    # The final observation is valid, so bootstrap.

    horizon_reaches_truncation = (
        not terminated
        and horizon == remaining_steps
    )

    should_bootstrap = (
        horizon_ends_before_episode
        or horizon_reaches_truncation
    )

    # ------------------------------------------------------------
    # 3. Add bootstrap value when valid.
    # ------------------------------------------------------------

    if should_bootstrap:

        bootstrap_state = states[
            tau + horizon
        ]

        result += (
            gamma ** horizon
        ) * values[bootstrap_state]

    return float(result)


# ================================================================
# 6. N-STEP TD EPISODE UPDATE
# ================================================================

def n_step_td_episode(
    values,
    episode,
    n,
    alpha,
    gamma,
):
    """
    Apply n-step TD updates to all nonterminal states
    in a recorded episode.
    """

    if not 0 < alpha <= 1:
        raise ValueError(
            "alpha must be in (0,1]."
        )

    if int(n) != n or n <= 0:
        raise ValueError(
            "n must be a positive integer."
        )

    if len(episode.states) != (
        len(episode.rewards) + 1
    ):
        raise ValueError(
            "Invalid episode structure."
        )

    errors = []

    for tau in range(episode.length):

        state = episode.states[tau]

        if state in TERMINAL_STATES:
            continue

        target = calculate_n_step_return(
            states=episode.states,
            rewards=episode.rewards,
            tau=tau,
            n=n,
            gamma=gamma,
            values=values,
            terminated=episode.terminated,
        )

        td_error = target - values[state]

        values[state] += alpha * td_error

        errors.append(float(td_error))

    return errors


# ================================================================
# 7. MONTE CARLO RETURN
# ================================================================

def calculate_monte_carlo_return(
    rewards,
    tau,
    gamma,
):
    """
    Complete discounted return.

    No bootstrapping.
    """

    if not 0 <= tau < len(rewards):
        raise ValueError(
            "Invalid tau."
        )

    result = 0.0

    for k, reward in enumerate(rewards[tau:]):

        result += (
            gamma ** k
        ) * reward

    return float(result)


# ================================================================
# 8. TRAINING RESULT
# ================================================================

@dataclass
class TrainingResult:

    values: np.ndarray

    rmse_history: np.ndarray

    episode_lengths: np.ndarray

    episode_returns: np.ndarray


# ================================================================
# 9. TRAIN ONE AGENT
# ================================================================

def train_agent(
    n,
    num_episodes=NUM_EPISODES,
    alpha=ALPHA,
    gamma=GAMMA,
    seed=SEED,
    max_steps=MAX_STEPS,
):
    """
    Train n-step TD prediction.

    n=1:
        TD(0).

    Intermediate n:
        N-step TD.

    n >= episode length:
        Complete-return behavior, equivalent to
        Monte Carlo for naturally terminated episodes.
    """

    if int(n) != n or n <= 0:
        raise ValueError(
            "n must be a positive integer."
        )

    if num_episodes <= 0:
        raise ValueError(
            "num_episodes must be positive."
        )

    if max_steps <= 0:
        raise ValueError(
            "max_steps must be positive."
        )

    if not 0 < alpha <= 1:
        raise ValueError(
            "alpha must be in (0,1]."
        )

    if not 0 <= gamma <= 1:
        raise ValueError(
            "gamma must be in [0,1]."
        )

    rng = np.random.default_rng(seed)

    env_seed = int(
        rng.integers(0, 2**31 - 1)
    )

    env = RandomWalkEnvironment(
        seed=env_seed
    )

    values = np.zeros(NUM_STATES)

    target_values = exact_state_values()

    nonterminal_states = np.arange(
        1,
        NUM_STATES - 1,
    )

    rmse_history = np.zeros(num_episodes)

    episode_lengths = np.zeros(
        num_episodes,
        dtype=int,
    )

    episode_returns = np.zeros(num_episodes)

    for episode_index in range(num_episodes):

        episode = generate_episode(
            env,
            max_steps=max_steps,
        )

        n_step_td_episode(
            values=values,
            episode=episode,
            n=n,
            alpha=alpha,
            gamma=gamma,
        )

        episode_lengths[episode_index] = (
            episode.length
        )

        episode_returns[episode_index] = sum(
            episode.rewards
        )

        rmse_history[episode_index] = np.sqrt(
            np.mean(
                (
                    values[nonterminal_states]
                    - target_values[nonterminal_states]
                ) ** 2
            )
        )

    return TrainingResult(
        values=values.copy(),
        rmse_history=rmse_history,
        episode_lengths=episode_lengths,
        episode_returns=episode_returns,
    )


# ================================================================
# 10. MULTI-RUN COMPARISON
# ================================================================

def compare_n_steps(
    n_values=N_VALUES,
    num_runs=NUM_RUNS,
    num_episodes=NUM_EPISODES,
    alpha=ALPHA,
    gamma=GAMMA,
    seed=SEED,
):
    """
    Compare n-step TD across independent random runs.
    """

    if num_runs <= 0:
        raise ValueError(
            "num_runs must be positive."
        )

    mean_rmse = {}
    std_rmse = {}
    mean_final_values = {}

    for n in n_values:

        all_rmse = []
        all_values = []

        for run_index in range(num_runs):

            result = train_agent(
                n=n,
                num_episodes=num_episodes,
                alpha=alpha,
                gamma=gamma,
                seed=seed + run_index,
            )

            all_rmse.append(
                result.rmse_history
            )

            all_values.append(
                result.values
            )

        all_rmse = np.asarray(all_rmse)

        all_values = np.asarray(all_values)

        mean_rmse[n] = np.mean(
            all_rmse,
            axis=0,
        )

        std_rmse[n] = np.std(
            all_rmse,
            axis=0,
        )

        mean_final_values[n] = np.mean(
            all_values,
            axis=0,
        )

    return {
        "mean_rmse": mean_rmse,
        "std_rmse": std_rmse,
        "mean_final_values": mean_final_values,
    }


# ================================================================
# 11. VISUALIZATION: RANDOM WALK
# ================================================================

def visualize_random_walk():

    fig, ax = plt.subplots(
        figsize=(10, 3)
    )

    ax.plot(
        range(NUM_STATES),
        np.zeros(NUM_STATES),
        "o-",
        markersize=12,
    )

    ax.set_xticks(
        range(NUM_STATES)
    )

    ax.set_yticks([])

    labels = [
        "Left\nTerminal",
        "1",
        "2",
        "Start\n3",
        "4",
        "5",
        "Right\nTerminal",
    ]

    for state, label in enumerate(labels):

        ax.text(
            state,
            0.08,
            label,
            ha="center",
            va="bottom",
        )

    ax.set_title(
        "Seven-State Random Walk"
    )

    ax.set_xlabel("State")

    ax.set_ylim(-0.2, 0.4)

    fig.tight_layout()


# ================================================================
# 12. VISUALIZATION: EXAMPLE EPISODE
# ================================================================

def visualize_episode(episode):

    fig, ax = plt.subplots(
        figsize=(10, 4)
    )

    ax.plot(
        range(len(episode.states)),
        episode.states,
        marker="o",
    )

    ax.set_title(
        "Example Random Walk Episode"
    )

    ax.set_xlabel("Time step")

    ax.set_ylabel("State")

    ax.set_yticks(
        range(NUM_STATES)
    )

    ax.grid(True)

    fig.tight_layout()


# ================================================================
# 13. VISUALIZATION: VALUE FUNCTIONS
# ================================================================

def visualize_value_functions(results):

    exact = exact_state_values()

    states = np.arange(NUM_STATES)

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.plot(
        states,
        exact,
        "k--",
        marker="o",
        label="Exact values",
    )

    for n, values in results[
        "mean_final_values"
    ].items():

        ax.plot(
            states,
            values,
            marker="o",
            label=f"n={n}",
        )

    ax.set_title(
        "Learned State-Value Functions"
    )

    ax.set_xlabel("State")

    ax.set_ylabel("Estimated V(s)")

    ax.set_xticks(states)

    ax.grid(True)

    ax.legend()

    fig.tight_layout()


# ================================================================
# 14. VISUALIZATION: LEARNING CURVES
# ================================================================

def visualize_learning_curves(results):

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    for n in results["mean_rmse"]:

        mean = results["mean_rmse"][n]

        std = results["std_rmse"][n]

        episodes = np.arange(
            1,
            len(mean) + 1,
        )

        ax.plot(
            episodes,
            mean,
            label=f"n={n}",
        )

        ax.fill_between(
            episodes,
            np.maximum(0, mean - std),
            mean + std,
            alpha=0.12,
        )

    ax.set_title(
        "N-Step TD Learning Curves"
    )

    ax.set_xlabel("Episode")

    ax.set_ylabel("RMSE")

    ax.grid(True)

    ax.legend()

    fig.tight_layout()


# ================================================================
# 15. VISUALIZATION: FINAL RMSE
# ================================================================

def visualize_final_rmse(results):

    n_values = list(
        results["mean_rmse"].keys()
    )

    final_rmse = [
        results["mean_rmse"][n][-1]
        for n in n_values
    ]

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    ax.bar(
        [str(n) for n in n_values],
        final_rmse,
    )

    ax.set_title(
        "Final RMSE by N-Step Horizon"
    )

    ax.set_xlabel("N-step horizon")

    ax.set_ylabel("Final average RMSE")

    ax.grid(axis="y")

    fig.tight_layout()


# ================================================================
# 16. VISUALIZATION: RETURN CONTRIBUTIONS
# ================================================================

def visualize_return_contributions():

    rewards = [
        0.0,
        0.0,
        0.0,
        1.0,
    ]

    gamma = 0.9

    values = np.zeros(NUM_STATES)

    values[4] = 0.7

    horizons = [1, 2, 3, 4]

    returns = []

    for n in horizons:

        returns.append(
            calculate_n_step_return(
                states=[3, 4, 5, 6, 5],
                rewards=rewards,
                tau=0,
                n=n,
                gamma=gamma,
                values=values,
                terminated=False,
            )
        )

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    ax.plot(
        horizons,
        returns,
        marker="o",
        linewidth=2,
    )

    ax.set_title(
        "N-Step Return vs. Horizon"
    )

    ax.set_xlabel("N-step horizon")

    ax.set_ylabel("Calculated return")

    ax.set_xticks(horizons)

    ax.grid(True)

    fig.tight_layout()


# ================================================================
# 17. UNIT TESTS
# ================================================================

def run_tests():

    print("\n========== UNIT TESTS ==========")

    # Test 1: Environment reset.
    env = RandomWalkEnvironment(seed=1)

    assert env.reset() == START_STATE

    print("[PASS] Environment reset")

    # Test 2: Episode structure.
    episode = generate_episode(
        env,
        max_steps=1000,
    )

    assert len(episode.states) == (
        len(episode.rewards) + 1
    )

    assert episode.length > 0

    assert (
        episode.terminated
        or episode.truncated
    )

    print("[PASS] Episode structure")

    # Test 3: Exact state values.
    exact = exact_state_values()

    expected = np.array([
        0.0,
        1 / 6,
        2 / 6,
        3 / 6,
        4 / 6,
        5 / 6,
        0.0,
    ])

    assert np.allclose(exact, expected)

    print("[PASS] Exact state values")

    # Test 4: One-step return.
    values = np.zeros(NUM_STATES)

    values[4] = 0.8

    result = calculate_n_step_return(
        states=[3, 4, 5],
        rewards=[0.5, 0.2],
        tau=0,
        n=1,
        gamma=0.9,
        values=values,
        terminated=False,
    )

    # 0.5 + 0.9*0.8 = 1.22

    assert np.isclose(result, 1.22)

    print("[PASS] One-step return")

    # Test 5: Two-step horizon ends BEFORE terminal.
    #
    # Episode eventually terminates at state 6.
    # However, tau=0 and n=2 end at state 5.
    #
    # Therefore bootstrap from V(5).
    #
    # G = 0.2 + 0.9*0.3 + 0.9^2*0.8
    #   = 0.2 + 0.27 + 0.648
    #   = 1.118

    values = np.zeros(NUM_STATES)

    values[5] = 0.8

    result = calculate_n_step_return(
        states=[3, 4, 5, 6],
        rewards=[0.2, 0.3, 1.0],
        tau=0,
        n=2,
        gamma=0.9,
        values=values,
        terminated=True,
    )

    assert np.isclose(
        result,
        1.118,
    )

    print(
        "[PASS] Two-step bootstrap before terminal"
    )

    # Test 6: Horizon reaches terminal.
    #
    # Start at state 4.
    # Rewards: [0, 1]
    #
    # G = 0 + 0.9*1 = 0.9
    #
    # No bootstrap from terminal state.

    values = np.ones(NUM_STATES) * 100.0

    result = calculate_n_step_return(
        states=[3, 4, 5, 6],
        rewards=[0.0, 0.0, 1.0],
        tau=1,
        n=5,
        gamma=0.9,
        values=values,
        terminated=True,
    )

    assert np.isclose(result, 0.9)

    print(
        "[PASS] Terminal return without bootstrap"
    )

    # Test 7: Truncated episode bootstraps.
    #
    # G = 0 + 0 + 0.9^2 * 0.8
    #   = 0.648

    values = np.zeros(NUM_STATES)

    values[5] = 0.8

    result = calculate_n_step_return(
        states=[3, 4, 5],
        rewards=[0.0, 0.0],
        tau=0,
        n=5,
        gamma=0.9,
        values=values,
        terminated=False,
    )

    assert np.isclose(result, 0.648)

    print("[PASS] Truncation bootstrap")

    # Test 8: Monte Carlo return.
    #
    # G = 0 + 0.9*0.5 + 0.9^2*1
    #   = 1.26

    result = calculate_monte_carlo_return(
        rewards=[0.0, 0.5, 1.0],
        tau=0,
        gamma=0.9,
    )

    assert np.isclose(result, 1.26)

    print("[PASS] Monte Carlo return")

    # Test 9: N-step episode update.

    values = np.zeros(NUM_STATES)

    episode = Episode(
        states=[3, 4, 5, 6],
        rewards=[0.0, 0.0, 1.0],
        terminated=True,
        truncated=False,
    )

    n_step_td_episode(
        values=values,
        episode=episode,
        n=1,
        alpha=0.5,
        gamma=1.0,
    )

    # Chronological updates:
    #
    # State 3 target = old V(4) = 0.
    # State 4 target = old V(5) = 0.
    # State 5 target = terminal reward 1.
    #
    # V(5) = 0 + 0.5*(1-0) = 0.5

    assert np.isclose(values[3], 0.0)

    assert np.isclose(values[5], 0.5)

    print("[PASS] N-step TD episode update")

    # Test 10: Terminal states are not updated.

    values = np.zeros(NUM_STATES)

    episode = Episode(
        states=[1, 0],
        rewards=[0.0],
        terminated=True,
        truncated=False,
    )

    n_step_td_episode(
        values,
        episode,
        n=1,
        alpha=0.5,
        gamma=1.0,
    )

    assert values[0] == 0.0

    print("[PASS] Terminal states remain unchanged")

    # Test 11: Invalid n.

    try:

        calculate_n_step_return(
            states=[3, 4],
            rewards=[0.0],
            tau=0,
            n=0,
            gamma=1.0,
            values=np.zeros(NUM_STATES),
        )

        raise AssertionError(
            "Expected ValueError."
        )

    except ValueError:
        pass

    print("[PASS] N-step validation")

    # Test 12: Training output.

    result = train_agent(
        n=2,
        num_episodes=20,
        seed=123,
    )

    assert result.values.shape == (
        NUM_STATES,
    )

    assert len(result.rmse_history) == 20

    assert np.all(
        np.isfinite(result.values)
    )

    assert np.all(
        np.isfinite(result.rmse_history)
    )

    print("[PASS] N-step training")

    # Test 13: Multiple horizons.

    for n in [1, 2, 4, 8, 1000]:

        result = train_agent(
            n=n,
            num_episodes=10,
            seed=100 + n,
        )

        assert np.all(
            np.isfinite(result.values)
        )

    print("[PASS] Multiple n-step horizons")

    print("\nAll 13 tests passed.")


# ================================================================
# 18. MAIN
# ================================================================

def main():

    print("\n==============================================")

    print("RL COURSE — PHASE 3.6")

    print("N-STEP TEMPORAL-DIFFERENCE LEARNING")

    print("==============================================")

    # Step 1: Numerical verification.
    run_tests()

    # Step 2: Environment visualization.
    visualize_random_walk()

    # Step 3: Example episode.
    example_env = RandomWalkEnvironment(
        seed=SEED
    )

    example_episode = generate_episode(
        example_env
    )

    visualize_episode(
        example_episode
    )

    # Step 4: Single-run experiments.
    print("\n========== SINGLE-RUN EXPERIMENT ==========")

    for n in N_VALUES:

        result = train_agent(
            n=n,
            num_episodes=NUM_EPISODES,
            seed=SEED,
        )

        print(f"\nn = {n}")

        print(
            "Estimated values:",
            np.round(result.values, 3),
        )

        print(
            "Final RMSE:",
            f"{result.rmse_history[-1]:.5f}",
        )

    # Step 5: Multi-run comparison.
    print("\n========== MULTI-RUN COMPARISON ==========")

    comparison = compare_n_steps(
        n_values=N_VALUES,
        num_runs=NUM_RUNS,
        num_episodes=NUM_EPISODES,
        alpha=ALPHA,
        gamma=GAMMA,
        seed=SEED,
    )

    for n in N_VALUES:

        final_rmse = comparison[
            "mean_rmse"
        ][n][-1]

        print(
            f"n={n:4d} | "
            f"average final RMSE={final_rmse:.5f}"
        )

    # Step 6: Visualizations.
    visualize_value_functions(comparison)

    visualize_learning_curves(comparison)

    visualize_final_rmse(comparison)

    visualize_return_contributions()

    # Step 7: Display all figures.
    plt.show()

    print("\n==============================================")

    print("PHASE 3.6 COMPLETED")

    print("==============================================")


if __name__ == "__main__":
    main()
