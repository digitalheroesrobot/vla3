
"""
====================================================================
RL COURSE — PHASE 3.10
FORWARD VIEW, BACKWARD VIEW, AND TRUE ONLINE TD(lambda)
====================================================================

FILE:
    phase3/td_lambda_views.py

LESSON ROADMAP
--------------

STEP 1 — Forward View
Purpose:
    Understand how future rewards contribute to current state values.

What you learn:
    n-step returns and the lambda-return.

Implementation:
    Calculate n-step returns and lambda-returns.

STEP 2 — Backward View
Purpose:
    Understand how TD errors update previously visited states.

What you learn:
    Accumulating eligibility traces and backward credit assignment.

Implementation:
    Calculate eligibility traces and batch TD(lambda) updates.

STEP 3 — Forward/Backward Equivalence
Purpose:
    Verify that the two views produce the same batch update.

What you learn:
    How the lambda-return relates to accumulated TD errors.

Implementation:
    Compare numerical parameter updates using identical
    initial values, episode data, and learning rates.

STEP 4 — Online TD(lambda)
Purpose:
    Understand why ordinary online updates can differ from
    the offline forward-view calculation.

What you learn:
    The effect of changing value estimates during an episode.

Implementation:
    Compare batch and sequential updates.

STEP 5 — True Online TD(lambda)
Purpose:
    Implement an online algorithm that matches the online
    forward-view formulation.

What you learn:
    Dutch traces and the correction term.

Implementation:
    Implement the true-online TD(lambda) update.

STEP 6 — Numerical Verification
Purpose:
    Verify the mathematical identities and update behavior.

What you learn:
    Lambda boundaries, terminal handling, and numerical errors.

Implementation:
    Unit tests and numerical comparisons.

STEP 7 — Visualization
Purpose:
    Make the learning process visible.

What you learn:
    Lambda-return behavior, eligibility decay, convergence,
    and forward/backward update differences.

Implementation:
    All plots are integrated into this script.

STEP 8 — Robotics Application
Purpose:
    Apply TD(lambda) to robot navigation and delayed rewards.

What you learn:
    How a robot can distribute feedback across recent decisions.

Implementation:
    Warehouse robot navigation example in comments.

--------------------------------------------------------------------
MATHEMATICAL FOUNDATIONS
--------------------------------------------------------------------

1. n-step return

    G_t^(n) =
        R_(t+1)
        + gamma R_(t+2)
        + ...
        + gamma^(n-1) R_(t+n)
        + gamma^n V(S_(t+n))

2. Lambda-return

    G_t^lambda =
        (1-lambda) *
        sum[n=1 to T-t-1] lambda^(n-1) G_t^(n)
        + lambda^(T-t-1) G_t^(T-t)

    This is a weighted combination of n-step returns.

3. Backward-view TD error

    delta_t =
        R_(t+1) + gamma V(S_(t+1)) - V(S_t)

4. Accumulating eligibility trace

    E_t(s) =
        gamma * lambda * E_(t-1)(s)
        + I(S_t = s)

5. Backward-view update

    V(s) <- V(s) + alpha * delta_t * E_t(s)

6. Offline forward/backward equivalence

    With a fixed value estimate during the episode:

    Sum_t alpha * [G_t^lambda - V(S_t)]

    equals the corresponding accumulated backward-view
    updates for each state.

    This equality assumes the same episode, initial value
    estimates, discount factor, lambda, and learning rate.

7. True Online TD(lambda)

    v_t = w^T x_t

    delta_t =
        R_(t+1) + gamma_t * w^T x_(t+1) - v_t

    e_t =
        gamma_t * lambda * e_(t-1)
        + (1 - alpha * gamma_t * lambda
           * e_(t-1)^T x_t) * x_t

    w_(t+1) =
        w_t
        + alpha * (delta_t + v_t - v_old) * e_t
        - alpha * (v_t - v_old) * x_t

    v_old <- w^T x_(t+1), using the pre-update next-state
    value calculated at this time step.

--------------------------------------------------------------------
EXPERIMENT
--------------------------------------------------------------------

Environment:
    Seven-state random walk.

    State 0: left terminal, reward -1
    State 6: right terminal, reward +1
    State 3: starting state

    Actions:
        LEFT
        RIGHT

    Gamma = 1

    Exact nonterminal values:
        V(s) = s / 6

--------------------------------------------------------------------
REAL-WORLD ROBOTICS APPLICATION
--------------------------------------------------------------------

Warehouse robot navigation:

    State:
        Robot location, orientation, obstacle information.

    Action:
        Forward, backward, turn left, turn right.

    Reward:
        -1 for each movement step.
        Large negative reward for a dangerous event.
        Positive reward for reaching the destination.

    Problem:
        The robot may only discover whether a route was good
        after several movements.

    Application:
        Eligibility traces distribute a delayed reward or
        penalty to preceding decisions.

    True Online TD(lambda) is useful when the robot receives
    observations and updates its value estimates continuously.

    Safety reminder:
        A learned value function is not a safety guarantee.
        Use collision detection, constrained actions,
        emergency stops, and independent safety validation.

--------------------------------------------------------------------
DEPENDENCIES
--------------------------------------------------------------------

    pip install numpy matplotlib

RUN
---

    python phase3/td_lambda_views.py

====================================================================
"""

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

NONTERMINAL_STATES = np.arange(1, 6)

GAMMA = 1.0
LAMBDA = 0.8
ALPHA = 0.1

NUM_EPISODES = 100

LAMBDA_VALUES = [
    0.0,
    0.3,
    0.6,
    0.9,
    1.0,
]


# ================================================================
# 2. RANDOM WALK ENVIRONMENT
# ================================================================

class RandomWalk:

    """
    Seven-state random walk.

    State 0:
        Left terminal, reward -1.

    State 6:
        Right terminal, reward +1.

    States 1 through 5:
        Nonterminal states.

    Start:
        State 3.

    Each action moves one state left or right
    with probability 0.5.
    """

    def __init__(self, seed=None):

        self.rng = np.random.default_rng(seed)

        self.state = START_STATE

    def reset(self):

        self.state = START_STATE

        return self.state

    def step(self, action):

        if self.state in (
            LEFT_TERMINAL,
            RIGHT_TERMINAL,
        ):
            raise RuntimeError(
                "Cannot step after termination."
            )

        if action not in (0, 1):
            raise ValueError(
                "Action must be 0 (left) or 1 (right)."
            )

        if action == 0:
            next_state = self.state - 1
        else:
            next_state = self.state + 1

        self.state = next_state

        terminated = next_state in (
            LEFT_TERMINAL,
            RIGHT_TERMINAL,
        )

        if next_state == LEFT_TERMINAL:
            reward = -1.0

        elif next_state == RIGHT_TERMINAL:
            reward = 1.0

        else:
            reward = 0.0

        return (
            next_state,
            reward,
            terminated,
        )


# ================================================================
# 3. EPISODE DATA STRUCTURE
# ================================================================

@dataclass
class Episode:

    states: list
    rewards: list
    terminated: bool

    @property
    def length(self):

        return len(self.rewards)


def generate_episode(
    seed=None,
    max_steps=1000,
):

    env = RandomWalk(seed=seed)

    state = env.reset()

    states = [state]
    rewards = []

    terminated = False

    for _ in range(max_steps):

        action = int(
            env.rng.integers(0, 2)
        )

        next_state, reward, terminated = env.step(
            action
        )

        rewards.append(reward)

        states.append(next_state)

        if terminated:
            break

    if not terminated:
        raise RuntimeError(
            "Episode did not terminate within max_steps."
        )

    return Episode(
        states=states,
        rewards=rewards,
        terminated=True,
    )


# ================================================================
# 4. EXACT STATE VALUES
# ================================================================

def exact_state_values():

    """
    For this symmetric random walk with terminal rewards
    -1 and +1 and gamma=1:

        V(s) = s/3 - 1

    for states 0 through 6.

    Therefore:

        [-1, -2/3, -1/3, 0, 1/3, 2/3, 1]

    This is equivalent to the probability of reaching
    the right terminal minus the probability of reaching
    the left terminal.
    """

    values = np.zeros(NUM_STATES)

    for state in range(NUM_STATES):

        values[state] = (
            2.0 * state / 6.0 - 1.0
        )

    return values


# ================================================================
# 5. N-STEP RETURNS
# ================================================================

def calculate_n_step_return(
    episode,
    tau,
    n,
    values,
    gamma=GAMMA,
):

    """
    Calculate the n-step return from time tau.

    If the n-step horizon reaches the terminal state,
    no bootstrap is used.

    If the horizon ends before termination,
    bootstrap from V(S_(tau+n)).
    """

    if n <= 0:
        raise ValueError("n must be positive.")

    if tau < 0 or tau >= episode.length:
        raise ValueError("Invalid tau.")

    total_steps = episode.length

    horizon = min(
        n,
        total_steps - tau,
    )

    result = 0.0

    for k in range(horizon):

        result += (
            gamma ** k
            * episode.rewards[tau + k]
        )

    horizon_state_index = tau + horizon

    if horizon_state_index < total_steps:

        bootstrap_state = episode.states[
            horizon_state_index
        ]

        result += (
            gamma ** horizon
            * values[bootstrap_state]
        )

    return float(result)


# ================================================================
# 6. LAMBDA-RETURN
# ================================================================

def calculate_lambda_returns(
    episode,
    values,
    gamma=GAMMA,
    lambda_value=LAMBDA,
):

    """
    Calculate finite-episode lambda-returns.

    Recursive form:

        G_lambda(t) =
            r_(t+1)
            + gamma * [
                (1-lambda) * V(S_(t+1))
                + lambda * G_lambda(t+1)
              ]

    At the final transition into a terminal state:

        G_lambda = reward

    The terminal state value is zero for bootstrapping.
    """

    if not 0.0 <= lambda_value <= 1.0:
        raise ValueError("lambda must be in [0, 1].")

    T = episode.length

    returns = np.zeros(T)

    next_lambda_return = 0.0

    for tau in reversed(range(T)):

        reward = episode.rewards[tau]

        next_state = episode.states[tau + 1]

        if tau == T - 1:

            current_return = reward

        else:

            current_return = (
                reward
                + gamma * (
                    (1.0 - lambda_value)
                    * values[next_state]
                    + lambda_value
                    * next_lambda_return
                )
            )

        returns[tau] = current_return

        next_lambda_return = current_return

    return returns


# ================================================================
# 7. FORWARD-VIEW BATCH UPDATE
# ================================================================

def forward_view_batch_update(
    episode,
    initial_values,
    alpha=ALPHA,
    gamma=GAMMA,
    lambda_value=LAMBDA,
):

    """
    Offline forward-view update.

    IMPORTANT:
        All lambda-returns are calculated using the same
        frozen initial value function.

        The updates are accumulated and applied together.

    This is the setting in which the offline
    forward/backward equivalence is demonstrated.
    """

    values = np.asarray(
        initial_values,
        dtype=float,
    ).copy()

    lambda_returns = calculate_lambda_returns(
        episode,
        values,
        gamma,
        lambda_value,
    )

    updates = np.zeros_like(values)

    for tau, state in enumerate(episode.states[:-1]):

        if state in (
            LEFT_TERMINAL,
            RIGHT_TERMINAL,
        ):
            continue

        error = (
            lambda_returns[tau]
            - values[state]
        )

        updates[state] += alpha * error

    updated_values = values + updates

    return {
        "values": updated_values,
        "updates": updates,
        "lambda_returns": lambda_returns,
    }


# ================================================================
# 8. BACKWARD-VIEW BATCH UPDATE
# ================================================================

def backward_view_batch_update(
    episode,
    initial_values,
    alpha=ALPHA,
    gamma=GAMMA,
    lambda_value=LAMBDA,
):

    """
    Offline backward-view TD(lambda).

    All TD errors use the frozen initial value function.

    Eligibility:

        E_t(s) =
            gamma * lambda * E_(t-1)(s)
            + I(S_t=s)

    Parameter update:

        Delta V(s) =
            alpha * sum_t delta_t * E_t(s)
    """

    values = np.asarray(
        initial_values,
        dtype=float,
    ).copy()

    eligibility = np.zeros_like(values)

    updates = np.zeros_like(values)

    td_errors = []

    for tau in range(episode.length):

        state = episode.states[tau]

        next_state = episode.states[tau + 1]

        reward = episode.rewards[tau]

        # Activate current state trace.
        eligibility[state] += 1.0

        # Terminal values are zero.
        delta = (
            reward
            + gamma * values[next_state]
            - values[state]
        )

        td_errors.append(delta)

        # Apply update to all eligible states.
        updates += (
            alpha
            * delta
            * eligibility
        )

        # Decay traces.
        eligibility *= (
            gamma * lambda_value
        )

    updated_values = values + updates

    return {
        "values": updated_values,
        "updates": updates,
        "td_errors": np.asarray(td_errors),
    }


# ================================================================
# 9. ONLINE FORWARD-VIEW REFERENCE
# ================================================================

def online_forward_view_update(
    episode,
    initial_values,
    alpha=ALPHA,
    gamma=GAMMA,
    lambda_value=LAMBDA,
):

    """
    Sequential online forward-view reference.

    At each time tau:

        1. Calculate the current lambda-return using
           the value estimates available at that time.

        2. Update V(S_tau).

        3. Continue to the next time step.

    This is not the same as the frozen-value offline
    forward-view batch update.
    """

    values = np.asarray(
        initial_values,
        dtype=float,
    ).copy()

    updates = []

    for tau in range(episode.length):

        current_state = episode.states[tau]

        if current_state in (
            LEFT_TERMINAL,
            RIGHT_TERMINAL,
        ):
            continue

        remaining_states = episode.states[tau:]

        remaining_rewards = episode.rewards[tau:]

        sub_episode = Episode(
            states=remaining_states,
            rewards=remaining_rewards,
            terminated=True,
        )

        lambda_returns = calculate_lambda_returns(
            sub_episode,
            values,
            gamma,
            lambda_value,
        )

        target = lambda_returns[0]

        error = target - values[current_state]

        update = alpha * error

        values[current_state] += update

        updates.append(update)

    return {
        "values": values,
        "updates": np.asarray(updates),
    }


# ================================================================
# 10. TRUE ONLINE TD(lambda)
# ================================================================

def true_online_td_lambda_episode(
    episode,
    initial_values,
    alpha=ALPHA,
    gamma=GAMMA,
    lambda_value=LAMBDA,
):

    """
    True Online TD(lambda), tabular feature representation.

    Each nonterminal state is represented by a one-hot vector.

    Dutch trace:

        e_t =
            gamma_t * lambda * e_(t-1)
            + (1 - alpha * gamma_t * lambda
               * e_(t-1)^T x_t) * x_t

    Correction:

        w <- w
             + alpha * (delta + v - v_old) * e
             - alpha * (v - v_old) * x

    Terminal transitions use gamma_t = 0.
    """

    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1].")

    if not 0.0 <= lambda_value <= 1.0:
        raise ValueError("lambda must be in [0, 1].")

    values = np.asarray(
        initial_values,
        dtype=float,
    ).copy()

    eligibility = np.zeros_like(values)

    v_old = 0.0

    td_errors = []

    for tau in range(episode.length):

        state = episode.states[tau]

        next_state = episode.states[tau + 1]

        reward = episode.rewards[tau]

        x = np.zeros(NUM_STATES)
        x[state] = 1.0

        terminal = (
            next_state in (
                LEFT_TERMINAL,
                RIGHT_TERMINAL,
            )
        )

        gamma_t = 0.0 if terminal else gamma

        x_next = np.zeros(NUM_STATES)

        if not terminal:
            x_next[next_state] = 1.0

        v = float(np.dot(values, x))

        v_next = float(
            np.dot(values, x_next)
        )

        delta = (
            reward
            + gamma_t * v_next
            - v
        )

        # Dutch trace correction.
        dutch_factor = (
            1.0
            - alpha
            * gamma_t
            * lambda_value
            * np.dot(eligibility, x)
        )

        eligibility = (
            gamma_t
            * lambda_value
            * eligibility
            + dutch_factor * x
        )

        # True-online weight correction.
        values += (
            alpha
            * (delta + v - v_old)
            * eligibility
            - alpha
            * (v - v_old)
            * x
        )

        td_errors.append(delta)

        # v_next is calculated using the weights before
        # the update above, as required by the algorithm.
        v_old = v_next

    return {
        "values": values,
        "td_errors": np.asarray(td_errors),
        "eligibility": eligibility,
    }


# ================================================================
# 11. CONVENTIONAL ONLINE TD(lambda)
# ================================================================

def conventional_online_td_lambda_episode(
    episode,
    initial_values,
    alpha=ALPHA,
    gamma=GAMMA,
    lambda_value=LAMBDA,
):

    """
    Conventional online backward-view TD(lambda).

    This implementation is included for comparison.

    It updates values after each TD error and uses
    ordinary accumulating traces.

    Unlike true-online TD(lambda), it does not include
    the Dutch trace and correction terms.
    """

    values = np.asarray(
        initial_values,
        dtype=float,
    ).copy()

    eligibility = np.zeros_like(values)

    td_errors = []

    for tau in range(episode.length):

        state = episode.states[tau]

        next_state = episode.states[tau + 1]

        reward = episode.rewards[tau]

        terminal = (
            next_state in (
                LEFT_TERMINAL,
                RIGHT_TERMINAL,
            )
        )

        gamma_t = 0.0 if terminal else gamma

        delta = (
            reward
            + gamma_t * values[next_state]
            - values[state]
        )

        eligibility[state] += 1.0

        values += (
            alpha
            * delta
            * eligibility
        )

        eligibility *= (
            gamma_t * lambda_value
        )

        td_errors.append(delta)

    return {
        "values": values,
        "td_errors": np.asarray(td_errors),
    }


# ================================================================
# 12. TRAIN TRUE ONLINE TD(lambda)
# ================================================================

def train_true_online_td_lambda(
    lambda_value,
    num_episodes=NUM_EPISODES,
    alpha=ALPHA,
    gamma=GAMMA,
    seed=SEED,
):

    rng = np.random.default_rng(seed)

    values = np.zeros(NUM_STATES)

    history = []

    for episode_index in range(num_episodes):

        episode = generate_episode(
            seed=int(rng.integers(0, 2**31 - 1))
        )

        result = true_online_td_lambda_episode(
            episode,
            values,
            alpha,
            gamma,
            lambda_value,
        )

        values = result["values"]

        history.append(values.copy())

    return {
        "values": values,
        "history": np.asarray(history),
    }


# ================================================================
# 13. RMSE
# ================================================================

def calculate_rmse(
    estimated_values,
    target_values,
):

    estimated_values = np.asarray(
        estimated_values,
        dtype=float,
    )

    target_values = np.asarray(
        target_values,
        dtype=float,
    )

    # Exclude terminal states.
    difference = (
        estimated_values[1:6]
        - target_values[1:6]
    )

    return float(
        np.sqrt(np.mean(difference ** 2))
    )


# ================================================================
# 14. LAMBDA EXPERIMENT
# ================================================================

def run_lambda_experiment(
    lambda_values=LAMBDA_VALUES,
    num_runs=10,
    num_episodes=NUM_EPISODES,
):

    exact = exact_state_values()

    results = {}

    for lambda_value in lambda_values:

        all_rmse = []

        for run_index in range(num_runs):

            result = train_true_online_td_lambda(
                lambda_value=lambda_value,
                num_episodes=num_episodes,
                seed=SEED + run_index * 100,
            )

            history = result["history"]

            rmse = [
                calculate_rmse(v, exact)
                for v in history
            ]

            all_rmse.append(rmse)

        all_rmse = np.asarray(
            all_rmse,
            dtype=float,
        )

        results[lambda_value] = {
            "mean": np.mean(all_rmse, axis=0),
            "std": np.std(all_rmse, axis=0),
        }

    return results


# ================================================================
# 15. VISUALIZATION — RANDOM WALK
# ================================================================

def visualize_random_walk():

    fig, ax = plt.subplots(figsize=(10, 2.5))

    for state in range(NUM_STATES):

        ax.scatter(
            state,
            0,
            s=700,
        )

        if state == LEFT_TERMINAL:
            label = "Terminal\n-1"

        elif state == RIGHT_TERMINAL:
            label = "Terminal\n+1"

        elif state == START_STATE:
            label = "Start"

        else:
            label = f"S{state}"

        ax.text(
            state,
            0,
            label,
            ha="center",
            va="center",
            fontsize=8,
        )

    ax.set_xlim(-0.5, 6.5)
    ax.set_ylim(-0.5, 0.5)

    ax.axis("off")

    ax.set_title("Seven-State Random Walk")

    fig.tight_layout()


# ================================================================
# 16. VISUALIZATION — LAMBDA RETURNS
# ================================================================

def visualize_lambda_returns(
    episode,
    values,
):

    lambda_values = [
        0.0,
        0.3,
        0.6,
        0.9,
        1.0,
    ]

    fig, ax = plt.subplots(figsize=(10, 5))

    for lambda_value in lambda_values:

        returns = calculate_lambda_returns(
            episode,
            values,
            GAMMA,
            lambda_value,
        )

        ax.plot(
            np.arange(len(returns)),
            returns,
            marker="o",
            label=f"lambda={lambda_value}",
        )

    ax.set_title("Lambda-Return Along One Episode")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Lambda-return")

    ax.legend()
    ax.grid(True)

    fig.tight_layout()


# ================================================================
# 17. VISUALIZATION — FORWARD/BACKWARD COMPARISON
# ================================================================

def visualize_forward_backward_comparison(
    forward_result,
    backward_result,
):

    states = np.arange(NUM_STATES)

    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(
        states - width / 2,
        forward_result["updates"],
        width,
        label="Forward view",
    )

    ax.bar(
        states + width / 2,
        backward_result["updates"],
        width,
        label="Backward view",
    )

    ax.set_title(
        "Offline Forward/Backward Update Comparison"
    )

    ax.set_xlabel("State")
    ax.set_ylabel("Value update")

    ax.legend()
    ax.grid(axis="y")

    fig.tight_layout()


# ================================================================
# 18. VISUALIZATION — UPDATE DIFFERENCE
# ================================================================

def visualize_update_difference(
    forward_result,
    backward_result,
):

    difference = (
        forward_result["updates"]
        - backward_result["updates"]
    )

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.bar(
        np.arange(NUM_STATES),
        difference,
    )

    ax.axhline(
        0,
        linewidth=1,
    )

    ax.set_title(
        "Forward/Backward Update Difference"
    )

    ax.set_xlabel("State")
    ax.set_ylabel("Difference")

    ax.grid(axis="y")

    fig.tight_layout()


# ================================================================
# 19. VISUALIZATION — LAMBDA CONVERGENCE
# ================================================================

def visualize_lambda_convergence(results):

    fig, ax = plt.subplots(figsize=(10, 5))

    for lambda_value, data in results.items():

        episodes = np.arange(
            1,
            len(data["mean"]) + 1,
        )

        ax.plot(
            episodes,
            data["mean"],
            label=f"lambda={lambda_value}",
        )

        ax.fill_between(
            episodes,
            data["mean"] - data["std"],
            data["mean"] + data["std"],
            alpha=0.12,
        )

    ax.set_title("True Online TD(lambda) Convergence")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Value-function RMSE")

    ax.legend()
    ax.grid(True)

    fig.tight_layout()


# ================================================================
# 20. VISUALIZATION — TRACE DECAY
# ================================================================

def visualize_trace_decay():

    gamma = 0.95

    lambdas = [
        0.0,
        0.3,
        0.6,
        0.9,
        1.0,
    ]

    steps = np.arange(21)

    fig, ax = plt.subplots(figsize=(9, 5))

    for lambda_value in lambdas:

        trace = (
            gamma * lambda_value
        ) ** steps

        ax.plot(
            steps,
            trace,
            marker="o",
            markersize=3,
            label=f"lambda={lambda_value}",
        )

    ax.set_title("Eligibility Trace Decay")
    ax.set_xlabel("Steps after activation")
    ax.set_ylabel("Trace strength")

    ax.legend()
    ax.grid(True)

    fig.tight_layout()


# ================================================================
# 21. UNIT TESTS
# ================================================================

def run_tests():

    print("\n========== UNIT TESTS ==========")

    # Test 1: Environment reset.
    env = RandomWalk(seed=SEED)

    assert env.reset() == START_STATE

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

    assert next_state == 6
    assert reward == 1.0
    assert terminated

    print("[PASS] Right terminal transition")

    # Test 4: Left terminal transition.
    env.reset()
    env.state = 1

    next_state, reward, terminated = env.step(0)

    assert next_state == 0
    assert reward == -1.0
    assert terminated

    print("[PASS] Left terminal transition")

    # Test 5: Terminal state protection.
    try:
        env.step(0)
        raise AssertionError("Expected RuntimeError.")
    except RuntimeError:
        pass

    print("[PASS] Terminal state protection")

    # Test 6: Exact state values.
    exact = exact_state_values()

    expected = np.array([
        -1.0,
        -2.0 / 3.0,
        -1.0 / 3.0,
        0.0,
        1.0 / 3.0,
        2.0 / 3.0,
        1.0,
    ])

    assert np.allclose(exact, expected)

    print("[PASS] Exact random-walk values")

    # Test 7: Episode structure.
    episode = generate_episode(seed=SEED)

    assert len(episode.states) == len(episode.rewards) + 1
    assert episode.terminated
    assert episode.states[-1] in (0, 6)

    print("[PASS] Episode structure")

    # Test 8: One-step return.
    episode = Episode(
        states=[3, 4, 5, 6],
        rewards=[0.0, 0.0, 1.0],
        terminated=True,
    )

    values = np.zeros(NUM_STATES)
    values[4] = 0.4
    values[5] = 0.8

    result = calculate_n_step_return(
        episode,
        tau=0,
        n=1,
        values=values,
        gamma=0.9,
    )

    assert np.isclose(result, 0.9 * 0.4)

    print("[PASS] One-step bootstrapped return")

    # Test 9: Terminal n-step return.
    result = calculate_n_step_return(
        episode,
        tau=0,
        n=3,
        values=values,
        gamma=0.9,
    )

    expected = 0.9 ** 2

    assert np.isclose(result, expected)

    print("[PASS] Terminal n-step return")

    # Test 10: Lambda=0 equals one-step return.
    lambda_zero = calculate_lambda_returns(
        episode,
        values,
        gamma=0.9,
        lambda_value=0.0,
    )

    for tau in range(episode.length):

        expected = calculate_n_step_return(
            episode,
            tau,
            1,
            values,
            0.9,
        )

        assert np.isclose(
            lambda_zero[tau],
            expected,
        )

    print("[PASS] Lambda-zero return")

    # Test 11: Lambda=1 equals full Monte Carlo return.
    lambda_one = calculate_lambda_returns(
        episode,
        values,
        gamma=0.9,
        lambda_value=1.0,
    )

    expected_returns = np.array([
        0.9 ** 2,
        0.9,
        1.0,
    ])

    assert np.allclose(
        lambda_one,
        expected_returns,
    )

    print("[PASS] Lambda-one return")

    # Test 12: Offline forward/backward equivalence.
    episode = Episode(
        states=[3, 4, 5, 4, 3, 2, 1, 0],
        rewards=[0, 0, 0, 0, 0, 0, -1],
        terminated=True,
    )

    initial_values = np.array([
        0.0,
        -0.2,
        -0.1,
        0.1,
        0.2,
        0.3,
        0.0,
    ])

    forward = forward_view_batch_update(
        episode,
        initial_values,
        alpha=0.1,
        gamma=0.9,
        lambda_value=0.8,
    )

    backward = backward_view_batch_update(
        episode,
        initial_values,
        alpha=0.1,
        gamma=0.9,
        lambda_value=0.8,
    )

    assert np.allclose(
        forward["updates"],
        backward["updates"],
        atol=1e-10,
    )

    print("[PASS] Offline forward/backward equivalence")

    # Test 13: Lambda boundary validation.
    try:
        calculate_lambda_returns(
            episode,
            initial_values,
            lambda_value=1.5,
        )
        raise AssertionError("Expected ValueError.")
    except ValueError:
        pass

    print("[PASS] Lambda boundary validation")

    # Test 14: Forward update is finite.
    assert np.all(
        np.isfinite(forward["values"])
    )

    print("[PASS] Forward-view numerical stability")

    # Test 15: Backward update is finite.
    assert np.all(
        np.isfinite(backward["values"])
    )

    print("[PASS] Backward-view numerical stability")

    # Test 16: True Online TD(lambda).
    episode = generate_episode(seed=SEED + 1)

    result = true_online_td_lambda_episode(
        episode,
        np.zeros(NUM_STATES),
        alpha=0.1,
        gamma=1.0,
        lambda_value=0.8,
    )

    assert np.all(
        np.isfinite(result["values"])
    )

    print("[PASS] True Online TD(lambda)")

    # Test 17: Conventional online TD(lambda).
    result = conventional_online_td_lambda_episode(
        episode,
        np.zeros(NUM_STATES),
        alpha=0.1,
        gamma=1.0,
        lambda_value=0.8,
    )

    assert np.all(
        np.isfinite(result["values"])
    )

    print("[PASS] Conventional online TD(lambda)")

    # Test 18: Lambda zero true-online behavior.
    result = true_online_td_lambda_episode(
        episode,
        np.zeros(NUM_STATES),
        alpha=0.1,
        gamma=1.0,
        lambda_value=0.0,
    )

    assert np.all(
        np.isfinite(result["values"])
    )

    print("[PASS] True-online lambda-zero behavior")

    # Test 19: True-online lambda one.
    result = true_online_td_lambda_episode(
        episode,
        np.zeros(NUM_STATES),
        alpha=0.1,
        gamma=1.0,
        lambda_value=1.0,
    )

    assert np.all(
        np.isfinite(result["values"])
    )

    print("[PASS] True-online lambda-one behavior")

    # Test 20: RMSE.
    exact = exact_state_values()

    assert np.isclose(
        calculate_rmse(exact, exact),
        0.0,
    )

    print("[PASS] RMSE calculation")

    # Test 21: Training output dimensions.
    result = train_true_online_td_lambda(
        lambda_value=0.8,
        num_episodes=10,
        seed=SEED,
    )

    assert result["history"].shape == (
        10,
        NUM_STATES,
    )

    print("[PASS] Training history dimensions")

    # Test 22: All trained values are finite.
    assert np.all(
        np.isfinite(result["values"])
    )

    print("[PASS] Finite trained values")

    print("\nAll 22 tests passed.")


# ================================================================
# 22. MAIN
# ================================================================

def main():

    print("\n==============================================")
    print("RL COURSE — PHASE 3.10")
    print("FORWARD VIEW AND BACKWARD VIEW")
    print("==============================================")

    # STEP 1: Unit tests.
    run_tests()

    # STEP 2: Generate a reproducible episode.
    episode = generate_episode(
        seed=SEED + 500,
    )

    initial_values = np.array([
        0.0,
        -0.2,
        -0.1,
        0.1,
        0.2,
        0.3,
        0.0,
    ])

    print("\n========== SAMPLE EPISODE ==========")

    print("States:", episode.states)
    print("Rewards:", episode.rewards)
    print("Episode length:", episode.length)

    # STEP 3: Calculate forward-view update.
    forward = forward_view_batch_update(
        episode,
        initial_values,
        alpha=0.1,
        gamma=0.9,
        lambda_value=0.8,
    )

    # STEP 4: Calculate backward-view update.
    backward = backward_view_batch_update(
        episode,
        initial_values,
        alpha=0.1,
        gamma=0.9,
        lambda_value=0.8,
    )

    print("\n========== OFFLINE EQUIVALENCE ==========")

    print(
        "Maximum update difference:",
        np.max(
            np.abs(
                forward["updates"]
                - backward["updates"]
            )
        ),
    )

    print(
        "Forward values:",
        np.round(forward["values"], 4),
    )

    print(
        "Backward values:",
        np.round(backward["values"], 4),
    )

    # STEP 5: Compare online algorithms.
    online_forward = online_forward_view_update(
        episode,
        initial_values,
        alpha=0.1,
        gamma=0.9,
        lambda_value=0.8,
    )

    conventional = conventional_online_td_lambda_episode(
        episode,
        initial_values,
        alpha=0.1,
        gamma=0.9,
        lambda_value=0.8,
    )

    true_online = true_online_td_lambda_episode(
        episode,
        initial_values,
        alpha=0.1,
        gamma=0.9,
        lambda_value=0.8,
    )

    print("\n========== ONLINE COMPARISON ==========")

    print(
        "Online forward-view values:",
        np.round(online_forward["values"], 4),
    )

    print(
        "Conventional online values:",
        np.round(conventional["values"], 4),
    )

    print(
        "True online values:",
        np.round(true_online["values"], 4),
    )

    # STEP 6: Lambda convergence experiment.
    print("\n========== LAMBDA EXPERIMENT ==========")

    experiment_results = run_lambda_experiment(
        lambda_values=LAMBDA_VALUES,
        num_runs=5,
        num_episodes=NUM_EPISODES,
    )

    exact = exact_state_values()

    for lambda_value, data in experiment_results.items():

        final_rmse = data["mean"][-1]

        print(
            f"lambda={lambda_value:.1f} | "
            f"final RMSE={final_rmse:.4f}"
        )

    # STEP 7: Visualizations.
    visualize_random_walk()

    visualize_lambda_returns(
        episode,
        initial_values,
    )

    visualize_forward_backward_comparison(
        forward,
        backward,
    )

    visualize_update_difference(
        forward,
        backward,
    )

    visualize_lambda_convergence(
        experiment_results,
    )

    visualize_trace_decay()

    # STEP 8: Display plots.
    plt.show()

    print("\n==============================================")
    print("PHASE 3.10 COMPLETED")
    print("==============================================")


if __name__ == "__main__":
    main()
