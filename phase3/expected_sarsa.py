
"""
============================================================
RL_Course
PHASE 3.5 — EXPECTED SARSA
============================================================

FILE:
    phase3/expected_sarsa.py

PURPOSE
-------
Implement Expected SARSA and compare it with SARSA and
Q-Learning using the same Cliff-Walking environment.

LESSON ROADMAP
--------------

STEP 1 — Q-table
Purpose:
    Represent estimated action values Q(s,a).
What you learn:
    How an agent stores action-value estimates.
Implementation:
    NumPy Q-table.

STEP 2 — Epsilon-Greedy Policy
Purpose:
    Balance exploration and exploitation.
What you learn:
    Action-selection probabilities and tied actions.
Implementation:
    epsilon_greedy_probabilities().

STEP 3 — Expected SARSA
Purpose:
    Calculate the expected next-state action value.
What you learn:
    Probability-weighted action values.
Implementation:
    expected_sarsa_update().

STEP 4 — Compare Three Algorithms
Purpose:
    Understand TD control algorithms.
What you learn:
    SARSA, Expected SARSA and Q-Learning.
Implementation:
    Shared environment and training interface.

STEP 5 — Training and Evaluation
Purpose:
    Learn and evaluate navigation policies.
What you learn:
    Exploration, convergence and greedy evaluation.
Implementation:
    train_agent() and evaluate_greedy_policy().

STEP 6 — Visualization
Purpose:
    Observe learned behavior.
What you learn:
    Learning curves, cliff falls, Q-values and routes.
Implementation:
    Visualization functions called by main().

STEP 7 — Numerical Verification
Purpose:
    Verify mathematical correctness.
What you learn:
    TD targets, terminal handling and policy expectations.
Implementation:
    Unit tests.

CORE MATHEMATICS
----------------

SARSA:

    Q(S,A) <- Q(S,A) + alpha *
        [R + gamma * Q(S',A') - Q(S,A)]

EXPECTED SARSA:

    Q(S,A) <- Q(S,A) + alpha *
        [R + gamma * sum_a pi(a|S')Q(S',a) - Q(S,A)]

Q-LEARNING:

    Q(S,A) <- Q(S,A) + alpha *
        [R + gamma * max_a Q(S',a) - Q(S,A)]

For a terminal transition:

    Target = R

Expected next-state value:

    E_pi[Q(S',A')] =
        sum_a pi(a|S') * Q(S',a)

For epsilon-greedy policy:

    Non-greedy action probability = epsilon / N

    Greedy action probability =
        epsilon / N
        + (1-epsilon) / number_of_greedy_actions

REAL-WORLD APPLICATION
----------------------

Warehouse robot navigation:

    State:
        Position, velocity, nearby obstacles.

    Actions:
        Forward, backward, left, right, stop.

    Reward:
        Delivery progress, movement cost, hazard penalty.

    Expected SARSA:
        Evaluates the expected next action value under
        the current action-selection policy.

SAFETY:
    Learned values are not safety guarantees.
    Real robots require independent collision prevention,
    emergency stops and safety constraints.

DEPENDENCIES
------------

    pip install numpy matplotlib

RUN
---

    python phase3/expected_sarsa.py

============================================================
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. CONFIGURATION
# ============================================================

SEED = 42

ROWS = 4
COLS = 12

START = (3, 0)
GOAL = (3, 11)

CLIFF_CELLS = {
    (3, col)
    for col in range(1, 11)
}

NUM_STATES = ROWS * COLS
NUM_ACTIONS = 4

ACTION_NAMES = [
    "UP",
    "RIGHT",
    "DOWN",
    "LEFT",
]

ACTION_DELTAS = [
    (-1, 0),
    (0, 1),
    (1, 0),
    (0, -1),
]

GAMMA = 1.0
ALPHA = 0.5

INITIAL_EPSILON = 1.0
MIN_EPSILON = 0.05
EPSILON_DECAY = 0.995

NUM_EPISODES = 500
MAX_STEPS = 500


# ============================================================
# 2. CLIFF-WALKING ENVIRONMENT
# ============================================================

@dataclass
class StepResult:

    next_state: int
    reward: float
    terminated: bool
    cliff_fall: bool


class CliffWalking:

    """
    4x12 Cliff-Walking environment.

    Start: (3,0)
    Goal:  (3,11)
    Cliff: (3,1) through (3,10)

    Rewards:
        Ordinary movement: -1
        Cliff:             -100
        Goal:              -1

    Cliff:
        Returns agent to START.
        Does not terminate episode.

    Goal:
        Terminates episode.
    """

    def __init__(self):

        self.state = self.position_to_state(START)

    @staticmethod
    def position_to_state(position):

        row, col = position

        return row * COLS + col

    @staticmethod
    def state_to_position(state):

        return divmod(int(state), COLS)

    def reset(self):

        self.state = self.position_to_state(START)

        return self.state

    def step(self, action):

        if action not in range(NUM_ACTIONS):

            raise ValueError("Invalid action.")

        row, col = self.state_to_position(
            self.state
        )

        dr, dc = ACTION_DELTAS[action]

        next_row = int(
            np.clip(
                row + dr,
                0,
                ROWS - 1,
            )
        )

        next_col = int(
            np.clip(
                col + dc,
                0,
                COLS - 1,
            )
        )

        next_position = (
            next_row,
            next_col,
        )

        # Cliff transition.
        if next_position in CLIFF_CELLS:

            self.state = self.position_to_state(
                START
            )

            return StepResult(
                next_state=self.state,
                reward=-100.0,
                terminated=False,
                cliff_fall=True,
            )

        self.state = self.position_to_state(
            next_position
        )

        # Goal transition.
        if next_position == GOAL:

            return StepResult(
                next_state=self.state,
                reward=-1.0,
                terminated=True,
                cliff_fall=False,
            )

        return StepResult(
            next_state=self.state,
            reward=-1.0,
            terminated=False,
            cliff_fall=False,
        )


# ============================================================
# 3. EPSILON-GREEDY POLICY
# ============================================================

def epsilon_greedy_probabilities(
    q_values,
    epsilon,
):
    """
    Calculate epsilon-greedy probabilities.

    Every action receives:

        epsilon / number_of_actions

    Greedy actions additionally receive:

        (1-epsilon) / number_of_greedy_actions

    Tied greedy actions share exploitation probability.
    """

    q_values = np.asarray(
        q_values,
        dtype=float,
    )

    if q_values.ndim != 1 or q_values.size == 0:

        raise ValueError(
            "q_values must be a non-empty 1D array."
        )

    if not np.all(np.isfinite(q_values)):

        raise ValueError(
            "q_values must contain finite numbers."
        )

    if not 0.0 <= epsilon <= 1.0:

        raise ValueError(
            "epsilon must be between 0 and 1."
        )

    greedy = np.isclose(
        q_values,
        np.max(q_values),
    )

    probabilities = np.full(
        len(q_values),
        epsilon / len(q_values),
    )

    probabilities[greedy] += (
        (1.0 - epsilon)
        / np.count_nonzero(greedy)
    )

    return probabilities


def select_action(
    q_table,
    state,
    epsilon,
    rng,
):

    probabilities = epsilon_greedy_probabilities(
        q_table[state],
        epsilon,
    )

    return int(
        rng.choice(
            NUM_ACTIONS,
            p=probabilities,
        )
    )


# ============================================================
# 4. SARSA UPDATE
# ============================================================

def sarsa_update(
    q_table,
    state,
    action,
    reward,
    next_state,
    next_action,
    terminated,
    alpha,
    gamma,
):
    """
    SARSA:

        Target = R + gamma * Q(S',A')

    Terminal:

        Target = R
    """

    if terminated:

        target = float(reward)

    else:

        if next_state is None or next_action is None:

            raise ValueError(
                "Nonterminal SARSA update requires "
                "next_state and next_action."
            )

        target = (
            reward
            + gamma * q_table[
                next_state,
                next_action,
            ]
        )

    td_error = (
        target - q_table[state, action]
    )

    q_table[state, action] += (
        alpha * td_error
    )

    return float(td_error)


# ============================================================
# 5. EXPECTED SARSA UPDATE
# ============================================================

def expected_sarsa_update(
    q_table,
    state,
    action,
    reward,
    next_state,
    next_epsilon,
    terminated,
    alpha,
    gamma,
):
    """
    Expected SARSA:

        Expected_next =
            sum_a pi(a|S') * Q(S',a)

        Target =
            R + gamma * Expected_next

        TD_error =
            Target - Q(S,A)

        Q(S,A) =
            Q(S,A) + alpha * TD_error

    Terminal:

        Target = R

    No terminal-state bootstrap is performed.

    Important:
        The expectation uses the complete action distribution.
        No sampled next action is needed for the TD target.
    """

    if terminated:

        target = float(reward)

    else:

        if next_state is None:

            raise ValueError(
                "Nonterminal Expected SARSA requires next_state."
            )

        probabilities = epsilon_greedy_probabilities(
            q_table[next_state],
            next_epsilon,
        )

        expected_next_value = float(
            np.dot(
                probabilities,
                q_table[next_state],
            )
        )

        target = (
            reward
            + gamma * expected_next_value
        )

    td_error = (
        target - q_table[state, action]
    )

    q_table[state, action] += (
        alpha * td_error
    )

    return float(td_error)


# ============================================================
# 6. Q-LEARNING UPDATE
# ============================================================

def q_learning_update(
    q_table,
    state,
    action,
    reward,
    next_state,
    terminated,
    alpha,
    gamma,
):
    """
    Q-learning:

        Target = R + gamma * max_a Q(S',a)

    Terminal:

        Target = R
    """

    if terminated:

        target = float(reward)

    else:

        if next_state is None:

            raise ValueError(
                "Nonterminal Q-learning requires next_state."
            )

        target = (
            reward
            + gamma * np.max(
                q_table[next_state]
            )
        )

    td_error = (
        target - q_table[state, action]
    )

    q_table[state, action] += (
        alpha * td_error
    )

    return float(td_error)


# ============================================================
# 7. TRAINING RESULT
# ============================================================

@dataclass
class TrainingResult:

    q_table: np.ndarray

    episode_returns: np.ndarray

    episode_lengths: np.ndarray

    cliff_falls: np.ndarray

    epsilon_history: np.ndarray


# ============================================================
# 8. TRAINING FUNCTION
# ============================================================

def train_agent(
    algorithm="expected_sarsa",
    num_episodes=NUM_EPISODES,
    alpha=ALPHA,
    gamma=GAMMA,
    initial_epsilon=INITIAL_EPSILON,
    min_epsilon=MIN_EPSILON,
    epsilon_decay=EPSILON_DECAY,
    seed=SEED,
    max_steps=MAX_STEPS,
):
    """
    Train one of three TD control algorithms.

    Supported algorithms:

        sarsa
        expected_sarsa
        q_learning

    All use epsilon-greedy behavior.

    SARSA:
        Uses the sampled next action.

    Expected SARSA:
        Uses the expected value under epsilon-greedy policy.

    Q-learning:
        Uses the maximum next-state action value.

    Terminal transition:
        No bootstrapping.

    External time-limit truncation:
        Bootstraps from the valid final observation.
    """

    if algorithm not in (
        "sarsa",
        "expected_sarsa",
        "q_learning",
    ):

        raise ValueError(
            "Unsupported algorithm."
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

    if not (
        0 <= min_epsilon <= initial_epsilon <= 1
    ):

        raise ValueError(
            "Invalid epsilon configuration."
        )

    if not 0 < epsilon_decay <= 1:

        raise ValueError(
            "epsilon_decay must be in (0,1]."
        )

    env = CliffWalking()

    rng = np.random.default_rng(seed)

    q_table = np.zeros(
        (NUM_STATES, NUM_ACTIONS),
        dtype=float,
    )

    returns = np.zeros(num_episodes)

    lengths = np.zeros(
        num_episodes,
        dtype=int,
    )

    cliff_falls = np.zeros(
        num_episodes,
        dtype=int,
    )

    epsilon_history = np.zeros(num_episodes)

    epsilon = initial_epsilon

    for episode in range(num_episodes):

        state = env.reset()

        action = select_action(
            q_table,
            state,
            epsilon,
            rng,
        )

        total_reward = 0.0

        total_falls = 0

        for step in range(max_steps):

            result = env.step(action)

            next_state = result.next_state

            reward = result.reward

            terminated = result.terminated

            total_reward += reward

            total_falls += int(
                result.cliff_fall
            )

            if algorithm == "sarsa":

                next_action = None

                if not terminated:

                    next_action = select_action(
                        q_table,
                        next_state,
                        epsilon,
                        rng,
                    )

                sarsa_update(
                    q_table=q_table,
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=(
                        None if terminated else next_state
                    ),
                    next_action=next_action,
                    terminated=terminated,
                    alpha=alpha,
                    gamma=gamma,
                )

            elif algorithm == "expected_sarsa":

                expected_sarsa_update(
                    q_table=q_table,
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=(
                        None if terminated else next_state
                    ),
                    next_epsilon=epsilon,
                    terminated=terminated,
                    alpha=alpha,
                    gamma=gamma,
                )

                next_action = None

                if not terminated:

                    next_action = select_action(
                        q_table,
                        next_state,
                        epsilon,
                        rng,
                    )

            else:

                q_learning_update(
                    q_table=q_table,
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=(
                        None if terminated else next_state
                    ),
                    terminated=terminated,
                    alpha=alpha,
                    gamma=gamma,
                )

                next_action = None

                if not terminated:

                    next_action = select_action(
                        q_table,
                        next_state,
                        epsilon,
                        rng,
                    )

            if terminated:

                lengths[episode] = step + 1

                break

            action = next_action

            state = next_state

            # The last transition has already bootstrapped
            # from the valid final observation.
            if step == max_steps - 1:

                lengths[episode] = max_steps

                break

        returns[episode] = total_reward

        cliff_falls[episode] = total_falls

        epsilon_history[episode] = epsilon

        epsilon = max(
            min_epsilon,
            epsilon * epsilon_decay,
        )

    return TrainingResult(
        q_table=q_table,
        episode_returns=returns,
        episode_lengths=lengths,
        cliff_falls=cliff_falls,
        epsilon_history=epsilon_history,
    )


# ============================================================
# 9. GREEDY POLICY EVALUATION
# ============================================================

def evaluate_greedy_policy(
    q_table,
    episodes=30,
    seed=SEED + 10000,
    max_steps=500,
):
    """
    Evaluate the learned greedy policy.

    epsilon = 0

    Tied greedy actions are sampled uniformly.
    """

    env = CliffWalking()

    rng = np.random.default_rng(seed)

    episode_returns = []

    episode_lengths = []

    successes = 0

    total_cliff_falls = 0

    for _ in range(episodes):

        state = env.reset()

        total_reward = 0.0

        for step in range(max_steps):

            action = select_action(
                q_table,
                state,
                epsilon=0.0,
                rng=rng,
            )

            result = env.step(action)

            total_reward += result.reward

            total_cliff_falls += int(
                result.cliff_fall
            )

            state = result.next_state

            if result.terminated:

                successes += 1

                episode_lengths.append(
                    step + 1
                )

                break

        else:

            episode_lengths.append(
                max_steps
            )

        episode_returns.append(
            total_reward
        )

    return {
        "average_return": float(
            np.mean(episode_returns)
        ),

        "average_length": float(
            np.mean(episode_lengths)
        ),

        "success_rate": (
            successes / episodes
        ),

        "total_cliff_falls": int(
            total_cliff_falls
        ),
    }


# ============================================================
# 10. POLICY AND VALUE FUNCTIONS
# ============================================================

def greedy_action(q_table, state):

    return int(
        np.argmax(q_table[state])
    )


def extract_greedy_path(
    q_table,
    max_steps=100,
):

    env = CliffWalking()

    state = env.reset()

    path = [START]

    falls = 0

    reached_goal = False

    for _ in range(max_steps):

        action = greedy_action(
            q_table,
            state,
        )

        result = env.step(action)

        falls += int(
            result.cliff_fall
        )

        state = result.next_state

        position = env.state_to_position(
            state
        )

        if not result.cliff_fall:

            path.append(position)

        if result.terminated:

            reached_goal = True

            break

    return path, falls, reached_goal


def state_value_view(q_table):

    return np.max(
        q_table,
        axis=1,
    ).reshape(
        ROWS,
        COLS,
    )


def moving_average(
    values,
    window=20,
):

    values = np.asarray(
        values,
        dtype=float,
    )

    if len(values) < window:

        return values.copy()

    return np.convolve(
        values,
        np.ones(window) / window,
        mode="valid",
    )


# ============================================================
# 11. VISUALIZATION HELPERS
# ============================================================

def draw_environment(ax):

    grid = np.zeros(
        (ROWS, COLS),
        dtype=int,
    )

    for row, col in CLIFF_CELLS:

        grid[row, col] = 1

    ax.imshow(
        grid,
        cmap="Greys",
        vmin=0,
        vmax=1,
    )

    ax.set_xticks(
        np.arange(COLS)
    )

    ax.set_yticks(
        np.arange(ROWS)
    )

    ax.set_xticks(
        np.arange(-0.5, COLS, 1),
        minor=True,
    )

    ax.set_yticks(
        np.arange(-0.5, ROWS, 1),
        minor=True,
    )

    ax.grid(
        which="minor",
        linewidth=0.7,
    )

    ax.tick_params(
        which="minor",
        bottom=False,
        left=False,
    )

    ax.text(
        START[1],
        START[0],
        "S",
        ha="center",
        va="center",
        fontweight="bold",
    )

    ax.text(
        GOAL[1],
        GOAL[0],
        "G",
        ha="center",
        va="center",
        fontweight="bold",
    )

    for col in range(1, 11):

        ax.text(
            col,
            3,
            "C",
            ha="center",
            va="center",
            fontsize=8,
        )


# ============================================================
# 12. VISUALIZATION FUNCTIONS
# ============================================================

def visualize_environment():

    fig, ax = plt.subplots(
        figsize=(12, 4)
    )

    draw_environment(ax)

    ax.set_title(
        "Cliff-Walking Environment"
    )

    ax.set_xlabel("Column")

    ax.set_ylabel("Row")

    fig.tight_layout()


def visualize_learning_curves(results):

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    for name, result in results.items():

        ax.plot(
            moving_average(
                result.episode_returns
            ),
            label=name,
        )

    ax.set_title(
        "Training Return Comparison"
    )

    ax.set_xlabel(
        "Episode (moving-average index)"
    )

    ax.set_ylabel(
        "Episode return"
    )

    ax.grid(True)

    ax.legend()

    fig.tight_layout()


def visualize_cliff_falls(results):

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    for name, result in results.items():

        ax.plot(
            moving_average(
                result.cliff_falls.astype(float)
            ),
            label=name,
        )

    ax.set_title(
        "Cliff Falls During Training"
    )

    ax.set_xlabel(
        "Episode (moving-average index)"
    )

    ax.set_ylabel(
        "Cliff falls per episode"
    )

    ax.grid(True)

    ax.legend()

    fig.tight_layout()


def visualize_epsilon(result):

    fig, ax = plt.subplots(
        figsize=(9, 4)
    )

    ax.plot(
        result.epsilon_history
    )

    ax.set_title(
        "Epsilon Decay"
    )

    ax.set_xlabel("Episode")

    ax.set_ylabel("Epsilon")

    ax.grid(True)

    fig.tight_layout()


def visualize_policy(
    q_table,
    title,
):

    arrows = (
        "↑",
        "→",
        "↓",
        "←",
    )

    fig, ax = plt.subplots(
        figsize=(12, 4)
    )

    draw_environment(ax)

    for row in range(ROWS):

        for col in range(COLS):

            if (
                (row, col) in CLIFF_CELLS
                or (row, col) in {
                    START,
                    GOAL,
                }
            ):

                continue

            state = (
                row * COLS + col
            )

            action = greedy_action(
                q_table,
                state,
            )

            ax.text(
                col,
                row,
                arrows[action],
                ha="center",
                va="center",
                fontsize=15,
            )

    ax.set_title(title)

    ax.set_xlabel("Column")

    ax.set_ylabel("Row")

    fig.tight_layout()


def visualize_value_heatmap(
    q_table,
    title,
):

    values = state_value_view(
        q_table
    )

    fig, ax = plt.subplots(
        figsize=(12, 4)
    )

    image = ax.imshow(
        values,
        cmap="viridis",
    )

    ax.set_title(title)

    ax.set_xlabel("Column")

    ax.set_ylabel("Row")

    ax.set_xticks(
        np.arange(COLS)
    )

    ax.set_yticks(
        np.arange(ROWS)
    )

    fig.colorbar(
        image,
        ax=ax,
        label="max Q(s,a)",
    )

    fig.tight_layout()


def visualize_greedy_paths(results):

    names = list(results.keys())

    fig, axes = plt.subplots(
        1,
        len(names),
        figsize=(5 * len(names), 4),
    )

    if len(names) == 1:

        axes = [axes]

    for ax, name in zip(
        axes,
        names,
    ):

        q_table = results[name].q_table

        path, _, success = extract_greedy_path(
            q_table
        )

        draw_environment(ax)

        if path:

            path_rows = [
                position[0]
                for position in path
            ]

            path_cols = [
                position[1]
                for position in path
            ]

            ax.plot(
                path_cols,
                path_rows,
                marker="o",
                linewidth=2,
            )

        status = (
            "Goal reached"
            if success
            else "Goal not reached"
        )

        ax.set_title(
            f"{name}: {status}"
        )

    fig.tight_layout()


def visualize_action_probabilities():

    q_values = np.array([
        0.2,
        0.8,
        0.8,
        0.1,
    ])

    epsilon_values = [
        0.0,
        0.1,
        0.3,
        0.7,
        1.0,
    ]

    probabilities = np.array([
        epsilon_greedy_probabilities(
            q_values,
            epsilon,
        )
        for epsilon in epsilon_values
    ])

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    x = np.arange(NUM_ACTIONS)

    width = 0.15

    for index, epsilon in enumerate(
        epsilon_values
    ):

        ax.bar(
            x + index * width,
            probabilities[index],
            width,
            label=f"epsilon={epsilon}",
        )

    ax.set_xticks(
        x + width * 2
    )

    ax.set_xticklabels(
        ACTION_NAMES
    )

    ax.set_ylabel(
        "Action probability"
    )

    ax.set_title(
        "Epsilon-Greedy Action Probabilities"
    )

    ax.legend()

    fig.tight_layout()


# ============================================================
# 13. UNIT TESTS
# ============================================================

def run_tests():

    print("\n========== UNIT TESTS ==========")

    # Test 1: environment reset.
    env = CliffWalking()

    assert env.reset() == (
        env.position_to_state(START)
    )

    print("[PASS] Environment reset")

    # Test 2: ordinary transition.
    env.reset()

    result = env.step(0)

    assert env.state_to_position(
        result.next_state
    ) == (2, 0)

    assert result.reward == -1.0

    assert not result.terminated

    print("[PASS] Ordinary transition")

    # Test 3: cliff transition.
    env.reset()

    result = env.step(1)

    assert result.cliff_fall

    assert result.reward == -100.0

    assert result.next_state == (
        env.position_to_state(START)
    )

    assert not result.terminated

    print("[PASS] Cliff penalty and reset")

    # Test 4: goal transition.
    env.state = env.position_to_state(
        (3, 10)
    )

    result = env.step(1)

    assert result.terminated

    assert result.reward == -1.0

    print("[PASS] Goal termination")

    # Test 5: epsilon-greedy normalization.
    probabilities = epsilon_greedy_probabilities(
        np.array([1.0, 2.0, 2.0, 0.0]),
        0.2,
    )

    assert np.isclose(
        np.sum(probabilities),
        1.0,
    )

    assert np.all(
        probabilities >= 0
    )

    print("[PASS] Epsilon-greedy normalization")

    # Test 6: tied greedy actions.
    assert np.isclose(
        probabilities[1],
        probabilities[2],
    )

    assert (
        probabilities[1]
        > probabilities[0]
    )

    print("[PASS] Tie-aware action probabilities")

    # Test 7: Expected SARSA expected value.
    q = np.zeros(
        (3, NUM_ACTIONS)
    )

    q[2] = [
        0.1,
        0.6,
        0.9,
        0.2,
    ]

    probabilities = epsilon_greedy_probabilities(
        q[2],
        0.2,
    )

    expected_value = np.dot(
        probabilities,
        q[2],
    )

    # Greedy action 2:
    # P(action 2) = 0.85
    # Other actions each have probability 0.05.
    #
    # Expected value:
    # 0.05*0.1 + 0.05*0.6
    # + 0.85*0.9 + 0.05*0.2
    #
    # = 0.005 + 0.030 + 0.765 + 0.010
    # = 0.810

    assert np.isclose(
        expected_value,
        0.81,
        atol=1e-12,
    )

    print("[PASS] Expected next-action value")

    # Test 8: Expected SARSA numerical update.
    q = np.zeros(
        (3, NUM_ACTIONS)
    )

    q[1, 0] = 0.4

    q[2] = [
        0.1,
        0.6,
        0.9,
        0.2,
    ]

    error = expected_sarsa_update(
        q_table=q,
        state=1,
        action=0,
        reward=0.0,
        next_state=2,
        next_epsilon=0.2,
        terminated=False,
        alpha=0.1,
        gamma=1.0,
    )

    # Expected target = 0.81
    # TD error = 0.81 - 0.4 = 0.41
    # New Q = 0.4 + 0.1*0.41 = 0.441

    assert np.isclose(
        error,
        0.41,
        atol=1e-12,
    )

    assert np.isclose(
        q[1, 0],
        0.441,
        atol=1e-12,
    )

    print("[PASS] Expected SARSA numerical update")

    # Test 9: terminal transition has no bootstrap.
    q = np.zeros(
        (3, NUM_ACTIONS)
    )

    q[1, 0] = 0.4

    q[2] = 1000.0

    expected_sarsa_update(
        q,
        1,
        0,
        reward=2.0,
        next_state=None,
        next_epsilon=0.2,
        terminated=True,
        alpha=0.1,
        gamma=0.99,
    )

    # Target = 2.0
    # Updated Q = 0.4 + 0.1*(2.0-0.4) = 0.56

    assert np.isclose(
        q[1, 0],
        0.56,
    )

    print("[PASS] Terminal no-bootstrap")

    # Test 10: expected value is computed from the
    # complete epsilon-greedy action distribution.
    q = np.zeros(
        (3, NUM_ACTIONS)
    )

    q[1, 0] = 0.4

    q[2] = [
        0.1,
        0.6,
        0.9,
        0.2,
    ]

    probabilities = epsilon_greedy_probabilities(
        q[2],
        0.2,
    )

    expected_action_value = np.dot(
        probabilities,
        q[2],
    )

    expected_target = (
        0.0 + 1.0 * expected_action_value
    )

    expected_updated_value = (
        0.4
        + 0.1 * (expected_target - 0.4)
    )

    expected_sarsa_update(
        q,
        1,
        0,
        0.0,
        2,
        0.2,
        False,
        0.1,
        1.0,
    )

    assert np.isclose(
        q[1, 0],
        expected_updated_value,
    )

    print("[PASS] Expected policy calculation")

    # Test 11: Q-learning uses maximum action value.
    q = np.zeros(
        (3, NUM_ACTIONS)
    )

    q[1, 0] = 0.4

    q[2] = [
        0.1,
        0.6,
        0.9,
        0.2,
    ]

    q_learning_update(
        q,
        1,
        0,
        0.0,
        2,
        False,
        0.1,
        1.0,
    )

    assert np.isclose(
        q[1, 0],
        0.45,
    )

    print("[PASS] Q-learning maximum target")

    # Test 12: SARSA uses sampled action value.
    q = np.zeros(
        (3, NUM_ACTIONS)
    )

    q[1, 0] = 0.4

    q[2] = [
        0.1,
        0.6,
        0.9,
        0.2,
    ]

    sarsa_update(
        q,
        1,
        0,
        0.0,
        2,
        1,
        False,
        0.1,
        1.0,
    )

    assert np.isclose(
        q[1, 0],
        0.42,
    )

    print("[PASS] SARSA sampled-action target")

    # Test 13: invalid epsilon.
    try:

        epsilon_greedy_probabilities(
            np.zeros(4),
            1.5,
        )

        raise AssertionError(
            "Expected ValueError."
        )

    except ValueError:

        pass

    print("[PASS] Epsilon validation")

    # Test 14: invalid algorithm.
    try:

        train_agent(
            algorithm="invalid",
            num_episodes=1,
        )

        raise AssertionError(
            "Expected ValueError."
        )

    except ValueError:

        pass

    print("[PASS] Algorithm validation")

    # Test 15: Expected SARSA training.
    result = train_agent(
        algorithm="expected_sarsa",
        num_episodes=30,
        seed=SEED,
        max_steps=200,
    )

    assert np.all(
        np.isfinite(result.q_table)
    )

    assert result.q_table.shape == (
        NUM_STATES,
        NUM_ACTIONS,
    )

    assert len(
        result.episode_returns
    ) == 30

    print("[PASS] Expected SARSA training")

    # Test 16: Q-learning training.
    result = train_agent(
        algorithm="q_learning",
        num_episodes=30,
        seed=SEED + 1,
        max_steps=200,
    )

    assert np.all(
        np.isfinite(result.q_table)
    )

    print("[PASS] Q-learning training")

    # Test 17: SARSA training.
    result = train_agent(
        algorithm="sarsa",
        num_episodes=30,
        seed=SEED + 2,
        max_steps=200,
    )

    assert np.all(
        np.isfinite(result.q_table)
    )

    print("[PASS] SARSA training")

    print("\nAll 17 tests passed.")


# ============================================================
# 14. MAIN
# ============================================================

def main():

    print("\n==============================================")

    print("RL COURSE — PHASE 3.5")

    print("EXPECTED SARSA")

    print("==============================================")

    # Step 1: visualize environment.
    visualize_environment()

    # Step 2: train all three algorithms.
    algorithms = [
        "sarsa",
        "expected_sarsa",
        "q_learning",
    ]

    results = {}

    for index, algorithm in enumerate(
        algorithms
    ):

        print(
            f"\nTraining {algorithm}..."
        )

        results[algorithm] = train_agent(
            algorithm=algorithm,
            seed=SEED + index,
        )

    # Step 3: evaluate greedy policies.
    print("\n========== GREEDY EVALUATION ==========")

    for index, algorithm in enumerate(
        algorithms
    ):

        result = results[algorithm]

        metrics = evaluate_greedy_policy(
            result.q_table,
            seed=SEED + 100 + index,
        )

        print(f"\n{algorithm}:")

        for key, value in metrics.items():

            print(
                f"  {key}: {value:.3f}"
            )

        print(
            "  Final 20-episode training return:",
            f"{np.mean(result.episode_returns[-20:]):.3f}",
        )

        print(
            "  Total training cliff falls:",
            int(np.sum(result.cliff_falls)),
        )

    # Step 4: numerical tests.
    print("\n========== NUMERICAL VERIFICATION ==========")

    run_tests()

    # Step 5: integrated visualizations.
    print("\n========== VISUALIZATIONS ==========")

    visualize_learning_curves(results)

    visualize_cliff_falls(results)

    visualize_epsilon(
        results["expected_sarsa"]
    )

    visualize_action_probabilities()

    for algorithm, result in results.items():

        visualize_policy(
            result.q_table,
            f"{algorithm} Greedy Policy",
        )

        visualize_value_heatmap(
            result.q_table,
            f"{algorithm} State-Value View",
        )

    visualize_greedy_paths(results)

    plt.show()

    print("\n==============================================")

    print("PHASE 3.5 COMPLETED")

    print("==============================================")


if __name__ == "__main__":

    main()
