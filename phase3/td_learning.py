
"""
============================================================
RL COURSE — PHASE 3.2
TEMPORAL-DIFFERENCE LEARNING (TD(0))
============================================================

PURPOSE
-------
Learn how an agent estimates state values by updating after
each environment transition, without waiting for an episode
to finish and without knowing the transition model.

LESSON ROADMAP
--------------

STEP 1 — Understand TD Learning
Purpose:
    Understand the difference between Monte Carlo and TD.

What you learn:
    TD learns from a one-step reward and an estimate of
    the next state's value.

Implementation:
    Implement the TD(0) update rule.

STEP 2 — Build an Episodic Random Walk
Purpose:
    Create a small environment with an exact value solution.

What you learn:
    How states, actions, transitions, rewards, and terminal
    states define an episodic Markov Decision Process.

Implementation:
    Build a seven-state random walk.

STEP 3 — Implement TD(0) Prediction
Purpose:
    Estimate V_pi(s) from individual transitions.

What you learn:
    TD error, bootstrapping, learning rate, and terminal
    state handling.

Implementation:
    Update the value estimate after every transition.

STEP 4 — Compare TD(0) with Monte Carlo
Purpose:
    Understand the practical difference between the methods.

What you learn:
    TD uses bootstrapping.
    MC uses complete sampled returns.
    TD can learn before an episode ends.

Implementation:
    Train both methods on the same random-walk problem.

STEP 5 — Verify Against Exact Values
Purpose:
    Check whether the learned values approach the
    mathematically correct solution.

What you learn:
    Bellman expectation equations and numerical RMSE.

Implementation:
    Solve the exact linear system and compare estimates.

STEP 6 — Experiment and Visualize
Purpose:
    Understand how learning rate and episode count
    influence learning.

What you learn:
    Convergence behavior, estimation error, and sensitivity
    to the learning rate.

Implementation:
    Run multiple independent experiments and plot results.

CORE MATHEMATICS
----------------

TD error:

    delta_t = R_(t+1) + gamma * V(S_(t+1)) - V(S_t)

TD(0) update:

    V(S_t) <- V(S_t) + alpha * delta_t

For a terminal next state:

    V(S_terminal) = 0

Therefore:

    delta_t = R_(t+1) - V(S_t)

Monte Carlo update:

    V(S_t) <- V(S_t) + alpha * [G_t - V(S_t)]

IMPORTANT DISTINCTION
---------------------

Monte Carlo:
    Uses the actual sampled return G_t.
    Does not bootstrap.
    Standard episodic MC waits for episode completion.

TD(0):
    Uses a one-step target:
        R_(t+1) + gamma * V(S_(t+1))
    Bootstraps from its current estimate.
    Can update after every transition.

TD(0) is a model-free learning method.
It does not require P(s'|s,a) or a known reward model.

REAL-WORLD APPLICATION
----------------------

Warehouse Robot — Online Route Evaluation

Scenario:
    A mobile robot moves through a warehouse toward
    a delivery station.

Possible states:
    Robot position, orientation, battery category,
    and nearby obstacle information.

Possible actions:
    Move forward, turn left, turn right, stop.

Rewards:
    Small negative reward for movement time and energy.
    Positive reward for reaching the delivery station.
    Large negative reward for unsafe events.

How TD(0) applies:
    1. Observe the current robot state.
    2. Execute one movement.
    3. Observe the next state and reward.
    4. Immediately update the current state's value.
    5. Continue learning during subsequent movements.

This lesson demonstrates the value-learning mechanism
using a random walk. A later lesson will extend TD learning
to action values and robot control.

DEPENDENCIES
------------
pip install numpy matplotlib

RUN
---
python phase3/td_learning.py
============================================================
"""

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. CONFIGURATION
# ============================================================

SEED = 42

GAMMA = 1.0
ALPHA = 0.1

NUM_STATES = 7

LEFT_TERMINAL = 0
RIGHT_TERMINAL = 6

START_STATE = 3

NONTERMINAL_STATES = np.arange(1, 6)

NUM_EPISODES = 100

NUM_RUNS = 100

TRUE_VALUES = np.array([
    0.0,
    1.0 / 6.0,
    2.0 / 6.0,
    3.0 / 6.0,
    4.0 / 6.0,
    5.0 / 6.0,
    1.0,
])


# ============================================================
# 2. RANDOM WALK ENVIRONMENT
# ============================================================

class RandomWalk:
    """
    Seven-state episodic random walk.

    States:
        0: Left terminal.
        1-5: Nonterminal states.
        6: Right terminal.

    Start:
        State 3.

    Actions:
        0: Move left.
        1: Move right.

    Policy:
        Both actions have probability 0.5.

    Rewards:
        0 when moving left or remaining nonterminal.
        +1 when entering the right terminal.

    Terminal states:
        0 and 6.

    The transition model is deliberately simple so that
    exact state values can be calculated independently.
    """

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.state = START_STATE

    def reset(self):
        self.state = START_STATE
        return self.state

    def is_terminal(self, state):
        return state in (
            LEFT_TERMINAL,
            RIGHT_TERMINAL
        )

    def step(self, action):

        if self.is_terminal(self.state):
            raise RuntimeError(
                "Cannot step after terminal state."
            )

        if action not in (0, 1):
            raise ValueError("Action must be 0 or 1.")

        if action == 0:
            next_state = self.state - 1
        else:
            next_state = self.state + 1

        reward = (
            1.0
            if next_state == RIGHT_TERMINAL
            else 0.0
        )

        terminated = self.is_terminal(next_state)

        self.state = next_state

        return next_state, reward, terminated


# ============================================================
# 3. POLICY
# ============================================================

def random_walk_policy(rng):
    """
    Equiprobable random-walk policy.

        pi(left | s)  = 0.5
        pi(right | s) = 0.5
    """

    return int(rng.integers(0, 2))


# ============================================================
# 4. EXACT VALUE FUNCTION
# ============================================================

def exact_policy_evaluation():
    """
    Calculate the exact value function under the
    equiprobable random-walk policy.

    For each nonterminal state:

        V(s) = 0.5 * [R_left + gamma * V(s-1)]
             + 0.5 * [R_right + gamma * V(s+1)]

    Here gamma = 1.

    Boundary values:

        V(0) = 0
        V(6) = 1

    The exact solution is:

        V(s) = s / 6

    for s = 0, 1, ..., 6.

    We also solve the linear system numerically so that
    the theoretical result can be independently verified.
    """

    gamma = GAMMA

    n = len(NONTERMINAL_STATES)

    A = np.eye(n)
    b = np.zeros(n)

    for index, state in enumerate(
        NONTERMINAL_STATES
    ):

        # Left transition.
        left_state = state - 1

        if left_state in NONTERMINAL_STATES:

            left_index = int(
                np.where(
                    NONTERMINAL_STATES == left_state
                )[0][0]
            )

            A[index, left_index] -= 0.5 * gamma

        # Left terminal has value zero, so it contributes
        # nothing to the right-hand side.

        # Right transition.
        right_state = state + 1

        if right_state in NONTERMINAL_STATES:

            right_index = int(
                np.where(
                    NONTERMINAL_STATES == right_state
                )[0][0]
            )

            A[index, right_index] -= 0.5 * gamma

        elif right_state == RIGHT_TERMINAL:

            # Reward +1 is received when entering state 6.
            b[index] += 0.5 * 1.0

    values = np.zeros(NUM_STATES)

    values[LEFT_TERMINAL] = 0.0
    values[RIGHT_TERMINAL] = 1.0

    values[NONTERMINAL_STATES] = np.linalg.solve(
        A,
        b
    )

    return values


# ============================================================
# 5. TD(0) PREDICTION
# ============================================================

def td0_update(
    values,
    state,
    reward,
    next_state,
    terminated,
    alpha,
    gamma
):
    """
    Perform one TD(0) update.

    TD error:

        delta =
            reward
            + gamma * V(next_state)
            - V(state)

    If next_state is terminal:

        V(next_state) = 0

    Update:

        V(state) = V(state) + alpha * delta

    Only the current state's estimate is updated.
    """

    if terminated:
        next_value = 0.0
    else:
        next_value = values[next_state]

    td_target = reward + gamma * next_value

    td_error = td_target - values[state]

    values[state] += alpha * td_error

    return td_error


def train_td0(
    num_episodes=NUM_EPISODES,
    alpha=ALPHA,
    gamma=GAMMA,
    seed=SEED,
    initial_values=None
):
    """
    Train TD(0) prediction on the random-walk environment.

    Each state value is updated immediately after a transition.

    Returns:
        learned values,
        RMSE after each episode,
        number of transitions,
        final TD error.
    """

    env = RandomWalk(seed=seed)
    rng = np.random.default_rng(seed + 1)

    if initial_values is None:
        values = np.full(
            NUM_STATES,
            0.5,
            dtype=float
        )
    else:
        values = np.asarray(
            initial_values,
            dtype=float
        ).copy()

    # Terminal values are fixed.
    values[LEFT_TERMINAL] = 0.0
    values[RIGHT_TERMINAL] = 0.0

    true_values = exact_policy_evaluation()

    rmse_history = []
    transition_count = 0
    final_td_error = 0.0

    for _ in range(num_episodes):

        state = env.reset()

        terminated = False

        while not terminated:

            action = random_walk_policy(rng)

            next_state, reward, terminated = env.step(
                action
            )

            final_td_error = td0_update(
                values=values,
                state=state,
                reward=reward,
                next_state=next_state,
                terminated=terminated,
                alpha=alpha,
                gamma=gamma
            )

            state = next_state
            transition_count += 1

        rmse_history.append(
            calculate_rmse(
                values,
                true_values
            )
        )

    return {
        "values": values,
        "rmse": np.asarray(rmse_history),
        "transitions": transition_count,
        "final_td_error": final_td_error,
    }


# ============================================================
# 6. MONTE CARLO PREDICTION FOR COMPARISON
# ============================================================

def calculate_episode_returns(rewards, gamma):
    """
    Calculate complete-episode returns by backward recursion.
    """

    returns = np.zeros(
        len(rewards),
        dtype=float
    )

    running_return = 0.0

    for t in reversed(range(len(rewards))):

        running_return = (
            rewards[t] + gamma * running_return
        )

        returns[t] = running_return

    return returns


def train_monte_carlo(
    num_episodes=NUM_EPISODES,
    alpha=ALPHA,
    gamma=GAMMA,
    seed=SEED
):
    """
    Constant-step-size first-visit Monte Carlo prediction.

    MC target:

        G_t = R_(t+1) + gamma * R_(t+2) + ...

    Update:

        V(s) <- V(s) + alpha * [G_t - V(s)]

    Unlike TD(0), MC waits until the episode terminates
    before updating state values.

    First-visit MC updates each state once per episode.
    """

    env = RandomWalk(seed=seed)
    rng = np.random.default_rng(seed + 1)

    values = np.full(
        NUM_STATES,
        0.5,
        dtype=float
    )

    # Terminal values are not learned.
    values[LEFT_TERMINAL] = 0.0
    values[RIGHT_TERMINAL] = 0.0

    true_values = exact_policy_evaluation()

    rmse_history = []

    for _ in range(num_episodes):

        state = env.reset()

        trajectory = []
        terminated = False

        while not terminated:

            action = random_walk_policy(rng)

            next_state, reward, terminated = env.step(
                action
            )

            trajectory.append(
                (state, reward)
            )

            state = next_state

        rewards = [
            transition[1]
            for transition in trajectory
        ]

        returns = calculate_episode_returns(
            rewards,
            gamma
        )

        visited_states = set()

        for t, transition in enumerate(trajectory):

            state = transition[0]

            if state in visited_states:
                continue

            visited_states.add(state)

            values[state] += (
                alpha * (
                    returns[t] - values[state]
                )
            )

        rmse_history.append(
            calculate_rmse(
                values,
                true_values
            )
        )

    return {
        "values": values,
        "rmse": np.asarray(rmse_history),
    }


# ============================================================
# 7. NUMERICAL METRICS
# ============================================================

def calculate_rmse(
    estimated,
    reference
):
    """
    Root Mean Squared Error over nonterminal states.

        RMSE = sqrt(mean((V_estimated - V_true)^2))
    """

    estimated = np.asarray(estimated)
    reference = np.asarray(reference)

    estimated_nonterminal = estimated[
        NONTERMINAL_STATES
    ]

    reference_nonterminal = reference[
        NONTERMINAL_STATES
    ]

    return float(
        np.sqrt(
            np.mean(
                (
                    estimated_nonterminal
                    - reference_nonterminal
                ) ** 2
            )
        )
    )


def moving_average(values, window=10):

    values = np.asarray(
        values,
        dtype=float
    )

    if len(values) < window:
        return values

    return np.convolve(
        values,
        np.ones(window) / window,
        mode="valid"
    )


# ============================================================
# 8. MULTIPLE-RUN EXPERIMENT
# ============================================================

def run_multiple_experiments(
    num_runs=NUM_RUNS,
    num_episodes=NUM_EPISODES,
    alpha=ALPHA,
    gamma=GAMMA
):
    """
    Run independent TD(0) and MC experiments.

    Multiple runs reduce the influence of a single random
    trajectory.

    For each episode, calculate the average RMSE across runs.

    The two methods use the same environment and policy,
    but use independent random seeds.
    """

    td_errors = []
    mc_errors = []

    td_final_values = []
    mc_final_values = []

    for run in range(num_runs):

        td_result = train_td0(
            num_episodes=num_episodes,
            alpha=alpha,
            gamma=gamma,
            seed=SEED + 1000 * run
        )

        mc_result = train_monte_carlo(
            num_episodes=num_episodes,
            alpha=alpha,
            gamma=gamma,
            seed=SEED + 1000 * run + 500
        )

        td_errors.append(
            td_result["rmse"]
        )

        mc_errors.append(
            mc_result["rmse"]
        )

        td_final_values.append(
            td_result["values"]
        )

        mc_final_values.append(
            mc_result["values"]
        )

    return {
        "td_errors": np.asarray(td_errors),
        "mc_errors": np.asarray(mc_errors),
        "td_mean_error": np.mean(
            td_errors,
            axis=0
        ),
        "mc_mean_error": np.mean(
            mc_errors,
            axis=0
        ),
        "td_final_values": np.mean(
            td_final_values,
            axis=0
        ),
        "mc_final_values": np.mean(
            mc_final_values,
            axis=0
        ),
    }


# ============================================================
# 9. LEARNING-RATE EXPERIMENT
# ============================================================

def learning_rate_experiment(
    learning_rates=(0.01, 0.05, 0.1, 0.2, 0.5),
    num_runs=30,
    num_episodes=100,
):
    """
    Compare TD(0) learning rates.

    A small alpha:
        Updates are conservative and often slower.

    A large alpha:
        Updates respond more strongly to recent experience
        but can produce noisier estimates.

    Results are averaged across independent random seeds.
    """

    results = {}

    for alpha in learning_rates:

        run_errors = []

        for run in range(num_runs):

            result = train_td0(
                num_episodes=num_episodes,
                alpha=alpha,
                seed=SEED + run * 1000
            )

            run_errors.append(
                result["rmse"]
            )

        results[alpha] = np.mean(
            run_errors,
            axis=0
        )

    return results


# ============================================================
# 10. VISUALIZATIONS
# ============================================================

def visualize_value_comparison(
    true_values,
    td_values,
    mc_values
):

    states = np.arange(NUM_STATES)

    plt.figure(figsize=(9, 5))

    plt.plot(
        states,
        true_values,
        marker="o",
        label="Exact values"
    )

    plt.plot(
        states,
        td_values,
        marker="s",
        label="TD(0)"
    )

    plt.plot(
        states,
        mc_values,
        marker="^",
        label="Monte Carlo"
    )

    plt.xlabel("State")
    plt.ylabel("State value")

    plt.title(
        "Exact Values vs TD(0) vs Monte Carlo"
    )

    plt.xticks(states)
    plt.legend()
    plt.grid(True)

    plt.tight_layout()


def visualize_learning_curves(
    td_errors,
    mc_errors
):

    plt.figure(figsize=(9, 5))

    plt.plot(
        td_errors,
        label="TD(0)"
    )

    plt.plot(
        mc_errors,
        label="Monte Carlo"
    )

    plt.xlabel("Episode")
    plt.ylabel("RMSE")

    plt.title(
        "TD(0) vs Monte Carlo Learning Curves"
    )

    plt.legend()
    plt.grid(True)

    plt.tight_layout()


def visualize_multiple_runs(
    experiment_results
):

    td_mean = experiment_results["td_mean_error"]
    mc_mean = experiment_results["mc_mean_error"]

    td_std = np.std(
        experiment_results["td_errors"],
        axis=0
    )

    mc_std = np.std(
        experiment_results["mc_errors"],
        axis=0
    )

    episodes = np.arange(
        1,
        len(td_mean) + 1
    )

    plt.figure(figsize=(10, 5))

    plt.plot(
        episodes,
        td_mean,
        label="TD(0) mean RMSE"
    )

    plt.fill_between(
        episodes,
        np.maximum(0, td_mean - td_std),
        td_mean + td_std,
        alpha=0.2
    )

    plt.plot(
        episodes,
        mc_mean,
        label="MC mean RMSE"
    )

    plt.fill_between(
        episodes,
        np.maximum(0, mc_mean - mc_std),
        mc_mean + mc_std,
        alpha=0.2
    )

    plt.xlabel("Episode")
    plt.ylabel("RMSE")

    plt.title(
        "TD(0) vs MC Across Independent Runs"
    )

    plt.legend()
    plt.grid(True)

    plt.tight_layout()


def visualize_learning_rates(results):

    plt.figure(figsize=(10, 5))

    for alpha, errors in results.items():

        plt.plot(
            errors,
            label=f"alpha={alpha}"
        )

    plt.xlabel("Episode")
    plt.ylabel("RMSE")

    plt.title(
        "TD(0): Effect of Learning Rate"
    )

    plt.legend()
    plt.grid(True)

    plt.tight_layout()


def visualize_td_update():

    """
    Visualize one illustrative TD update.

    Example:
        V(s) = 0.4
        reward = 0
        V(s') = 0.7
        gamma = 1
        alpha = 0.1

    TD target = 0 + 1*0.7 = 0.7

    TD error = 0.7 - 0.4 = 0.3

    Updated value = 0.4 + 0.1*0.3 = 0.43
    """

    current_value = 0.4
    next_value = 0.7
    reward = 0.0

    gamma = 1.0
    alpha = 0.1

    td_target = reward + gamma * next_value

    td_error = td_target - current_value

    updated_value = (
        current_value + alpha * td_error
    )

    labels = [
        "Before update",
        "TD target",
        "After update"
    ]

    values = [
        current_value,
        td_target,
        updated_value
    ]

    plt.figure(figsize=(8, 5))

    bars = plt.bar(
        labels,
        values
    )

    plt.ylabel("Value")

    plt.title(
        "Illustrative TD(0) Update"
    )

    for bar, value in zip(bars, values):

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.3f}",
            ha="center",
            va="bottom"
        )

    plt.tight_layout()


# ============================================================
# 11. UNIT TESTS
# ============================================================

def run_tests():

    print("\n========== UNIT TESTS ==========")

    # Test 1: Environment starts at state 3.
    env = RandomWalk(seed=SEED)

    state = env.reset()

    assert state == START_STATE

    print("[PASS] Environment reset")

    # Test 2: Left transition.
    env.reset()

    next_state, reward, terminated = env.step(0)

    assert next_state == 2
    assert reward == 0.0
    assert not terminated

    print("[PASS] Left transition")

    # Test 3: Right terminal transition.
    env.state = 5

    next_state, reward, terminated = env.step(1)

    assert next_state == RIGHT_TERMINAL
    assert reward == 1.0
    assert terminated

    print("[PASS] Right terminal transition")

    # Test 4: Left terminal transition.
    env.state = 1

    next_state, reward, terminated = env.step(0)

    assert next_state == LEFT_TERMINAL
    assert reward == 0.0
    assert terminated

    print("[PASS] Left terminal transition")

    # Test 5: Terminal states reject additional actions.
    try:
        env.step(0)
        raise AssertionError(
            "Expected RuntimeError."
        )
    except RuntimeError:
        pass

    print("[PASS] Terminal state protection")

    # Test 6: TD(0) numerical update.
    values = np.array([
        0.0,
        0.5,
        0.7,
        0.5,
        0.5,
        0.5,
        0.0
    ])

    td_error = td0_update(
        values=values,
        state=2,
        reward=0.0,
        next_state=3,
        terminated=False,
        alpha=0.1,
        gamma=1.0
    )

    assert np.isclose(td_error, -0.2)
    assert np.isclose(values[2], 0.68)

    print("[PASS] TD(0) update")

    # Test 7: Terminal transition does not bootstrap.
    values = np.array([
        0.0,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        100.0
    ])

    td0_update(
        values=values,
        state=5,
        reward=1.0,
        next_state=6,
        terminated=True,
        alpha=0.1,
        gamma=1.0
    )

    assert np.isclose(
        values[5],
        0.55
    )

    print("[PASS] Terminal TD target")

    # Test 8: Terminal values are fixed in exact solution.
    true_values = exact_policy_evaluation()

    assert true_values[0] == 0.0
    assert true_values[6] == 1.0

    print("[PASS] Exact terminal values")

    # Test 9: Exact solution equals s/6.
    expected = np.arange(7) / 6.0

    assert np.allclose(
        true_values,
        expected,
        atol=1e-10
    )

    print("[PASS] Exact random-walk values")

    # Test 10: RMSE of exact values is zero.
    error = calculate_rmse(
        true_values,
        true_values
    )

    assert np.isclose(error, 0.0)

    print("[PASS] RMSE calculation")

    # Test 11: Monte Carlo returns.
    rewards = [0.0, 0.0, 1.0]

    returns = calculate_episode_returns(
        rewards,
        gamma=1.0
    )

    assert np.allclose(
        returns,
        [1.0, 1.0, 1.0]
    )

    print("[PASS] Monte Carlo return calculation")

    # Test 12: TD learning produces finite values.
    result = train_td0(
        num_episodes=100,
        alpha=0.1,
        seed=SEED
    )

    assert np.all(
        np.isfinite(result["values"])
    )

    print("[PASS] TD numerical stability")

    # Test 13: MC learning produces finite values.
    result = train_monte_carlo(
        num_episodes=100,
        alpha=0.1,
        seed=SEED
    )

    assert np.all(
        np.isfinite(result["values"])
    )

    print("[PASS] MC numerical stability")

    # Test 14: TD terminal values remain zero.
    result = train_td0(
        num_episodes=100,
        seed=SEED
    )

    assert result["values"][0] == 0.0
    assert result["values"][6] == 0.0

    print("[PASS] TD terminal value handling")

    # Test 15: Monte Carlo terminal values remain zero.
    result = train_monte_carlo(
        num_episodes=100,
        seed=SEED
    )

    assert result["values"][0] == 0.0
    assert result["values"][6] == 0.0

    print("[PASS] MC terminal value handling")

    print("\nAll 15 tests passed.")


# ============================================================
# 12. MAIN
# ============================================================

def main():

    print("\n========================================")
    print("PHASE 3.2 — TEMPORAL-DIFFERENCE LEARNING")
    print("========================================")

    # --------------------------------------------------------
    # STEP A — Exact reference
    # --------------------------------------------------------

    true_values = exact_policy_evaluation()

    print("\n[1] Exact state values")

    print(
        "Exact values:",
        np.round(true_values, 4)
    )

    # --------------------------------------------------------
    # STEP B — Single-run TD(0)
    # --------------------------------------------------------

    print("\n[2] TD(0) training")

    td_result = train_td0(
        num_episodes=NUM_EPISODES,
        alpha=ALPHA,
        gamma=GAMMA,
        seed=SEED
    )

    print(
        "TD(0) estimated values:",
        np.round(td_result["values"], 4)
    )

    print(
        "TD(0) final RMSE:",
        f"{td_result['rmse'][-1]:.6f}"
    )

    print(
        "Total transitions:",
        td_result["transitions"]
    )

    # --------------------------------------------------------
    # STEP C — Single-run MC
    # --------------------------------------------------------

    print("\n[3] Monte Carlo training")

    mc_result = train_monte_carlo(
        num_episodes=NUM_EPISODES,
        alpha=ALPHA,
        gamma=GAMMA,
        seed=SEED + 10
    )

    print(
        "MC estimated values:",
        np.round(mc_result["values"], 4)
    )

    print(
        "MC final RMSE:",
        f"{mc_result['rmse'][-1]:.6f}"
    )

    # --------------------------------------------------------
    # STEP D — Multiple independent runs
    # --------------------------------------------------------

    print("\n[4] Multiple-run comparison")

    experiment_results = run_multiple_experiments(
        num_runs=NUM_RUNS,
        num_episodes=NUM_EPISODES,
        alpha=ALPHA,
        gamma=GAMMA
    )

    print(
        "TD(0) mean final RMSE:",
        f"{experiment_results['td_mean_error'][-1]:.6f}"
    )

    print(
        "MC mean final RMSE:",
        f"{experiment_results['mc_mean_error'][-1]:.6f}"
    )

    # --------------------------------------------------------
    # STEP E — Learning-rate experiment
    # --------------------------------------------------------

    print("\n[5] Learning-rate experiment")

    learning_rate_results = learning_rate_experiment()

    for alpha, errors in learning_rate_results.items():

        print(
            f"alpha={alpha:.2f}, "
            f"final mean RMSE={errors[-1]:.6f}"
        )

    # --------------------------------------------------------
    # STEP F — Unit tests
    # --------------------------------------------------------

    run_tests()

    # --------------------------------------------------------
    # STEP G — Visualizations
    # --------------------------------------------------------

    print("\n[6] Generating visualizations...")

    visualize_value_comparison(
        true_values,
        experiment_results["td_final_values"],
        experiment_results["mc_final_values"]
    )

    visualize_learning_curves(
        td_result["rmse"],
        mc_result["rmse"]
    )

    visualize_multiple_runs(
        experiment_results
    )

    visualize_learning_rates(
        learning_rate_results
    )

    visualize_td_update()

    plt.show()

    print("\n========================================")
    print("PHASE 3.2 COMPLETED")
    print("========================================")


if __name__ == "__main__":
    main()