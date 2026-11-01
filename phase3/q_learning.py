
"""
============================================================
RL COURSE — PHASE 3.4
Q-LEARNING: OFF-POLICY TEMPORAL-DIFFERENCE CONTROL
============================================================

PURPOSE
-------
Learn how an agent estimates optimal action values Q*(s,a)
and improves its policy through off-policy TD learning.

LESSON ROADMAP
--------------

STEP 1 — Understand Q-Learning
Purpose:
    Learn action values and optimal control.
What you learn:
    Q-learning uses the maximum next-state action value.
Implementation:
    Implement the tabular Q-learning update.

STEP 2 — Build Cliff-Walking
Purpose:
    Create a navigation environment with a hazardous route.
What you learn:
    State transitions, rewards, terminal states and hazards.
Implementation:
    Build a 4x12 GridWorld.

STEP 3 — Implement Epsilon-Greedy
Purpose:
    Balance exploration and exploitation.
What you learn:
    Random exploration and greedy action selection.
Implementation:
    Implement tie-aware epsilon-greedy probabilities.

STEP 4 — Implement Q-Learning
Purpose:
    Learn optimal action values.
What you learn:
    Off-policy temporal-difference learning.
Implementation:
    Use the maximum next-state Q-value in the TD target.

STEP 5 — Compare SARSA
Purpose:
    Understand on-policy and off-policy learning.
What you learn:
    Why Q-learning may prefer a shorter but riskier route.
Implementation:
    Train SARSA and Q-learning on the same environment.

STEP 6 — Verify and Visualize
Purpose:
    Verify mathematical correctness.
What you learn:
    Numerical updates, convergence, and policy behavior.
Implementation:
    Unit tests and integrated visualization functions.

CORE MATHEMATICS
----------------

Q-learning TD target:

    Y = R + gamma * max_a Q(S',a)

Q-learning TD error:

    delta = Y - Q(S,A)

Q-learning update:

    Q(S,A) <- Q(S,A) + alpha * delta

For a true terminal transition:

    Y = R

No terminal-state bootstrap is performed.

SARSA update:

    Q(S,A) <- Q(S,A) + alpha *
        [R + gamma * Q(S',A') - Q(S,A)]

IMPORTANT DIFFERENCE
--------------------

SARSA:
    Uses the next action actually selected by the behavior policy.
    On-policy learning.

Q-learning:
    Uses the maximum estimated next-state action value.
    Off-policy learning.

Bellman optimality equation:

    Q*(s,a) = sum_s' P(s'|s,a) *
        [R(s,a,s') + gamma * max_a' Q*(s',a')]

Q-learning approximates this optimal action-value function
from sampled experience without requiring a known transition model.

REAL-WORLD APPLICATION
----------------------

Warehouse robot navigation:

    State:
        Robot position, nearby obstacles, battery level.

    Actions:
        Forward, backward, left, right, stop.

    Reward:
        Movement cost, delivery reward, hazard penalty.

    Q-learning:
        Learns which actions maximize expected future reward.

SAFETY NOTE:
    Q-learning does not guarantee collision avoidance.
    A real robot requires independent safety constraints,
    obstacle detection, collision checking, and emergency stops.

DEPENDENCIES
------------
    pip install numpy matplotlib

RUN
---
    python phase3/q_learning.py
============================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List

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
    (3, col) for col in range(1, 11)
}

NUM_STATES = ROWS * COLS
NUM_ACTIONS = 4

# Action encoding:
# 0 = Up
# 1 = Right
# 2 = Down
# 3 = Left

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
# 2. ENVIRONMENT
# ============================================================

@dataclass
class StepResult:

    next_state: int
    reward: float
    terminated: bool
    cliff_fall: bool


class CliffWalking:

    """
    Cliff-Walking environment.

    Grid:

        .  .  .  .  .  .  .  .  .  .  .  .
        .  .  .  .  .  .  .  .  .  .  .  .
        .  .  .  .  .  .  .  .  .  .  .  .
        S  C  C  C  C  C  C  C  C  C  C  G

    S = Start
    C = Cliff
    G = Goal

    Rewards:

        Ordinary movement: -1
        Falling into cliff: -100
        Reaching goal: -1

    Cliff behavior:

        Agent returns to START.
        Episode does not terminate.

    Goal behavior:

        Episode terminates.
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

        row, col = self.state_to_position(self.state)

        dr, dc = ACTION_DELTAS[action]

        next_row = int(
            np.clip(row + dr, 0, ROWS - 1)
        )

        next_col = int(
            np.clip(col + dc, 0, COLS - 1)
        )

        next_position = (next_row, next_col)

        # Cliff transition.
        if next_position in CLIFF_CELLS:

            self.state = self.position_to_state(START)

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
    Calculate epsilon-greedy action probabilities.

    For greedy actions:

        P(a|s) =
            epsilon / number_of_actions
            + (1-epsilon) / number_of_greedy_actions

    For non-greedy actions:

        P(a|s) = epsilon / number_of_actions

    Tied greedy actions share exploitation probability equally.
    """

    q_values = np.asarray(
        q_values,
        dtype=float,
    )

    if q_values.ndim != 1 or q_values.size == 0:

        raise ValueError(
            "q_values must be a non-empty 1D array."
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
# 4. Q-LEARNING UPDATE
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
    Perform one Q-learning update.

    Mathematical derivation:

        Q_target =
            R + gamma * max_a Q(S',a)

        delta =
            Q_target - Q(S,A)

        Q(S,A) =
            Q(S,A) + alpha * delta

    Terminal transition:

        Q_target = R

    Why is this off-policy?

        The update does not use the next action selected
        by the behavior policy.

        It uses the maximum estimated action value instead.
    """

    if terminated:

        target = float(reward)

    else:

        if next_state is None:

            raise ValueError(
                "next_state required for nonterminal update."
            )

        target = (
            reward
            + gamma * np.max(q_table[next_state])
        )

    td_error = (
        target - q_table[state, action]
    )

    q_table[state, action] += (
        alpha * td_error
    )

    return float(td_error)


# ============================================================
# 5. SARSA UPDATE FOR COMPARISON
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
    SARSA uses the sampled next action.

        Target = R + gamma * Q(S',A')

    The difference from Q-learning is:

        SARSA:
            Q(S',A')

        Q-learning:
            max_a Q(S',a)
    """

    if terminated:

        target = float(reward)

    else:

        if next_state is None or next_action is None:

            raise ValueError(
                "SARSA requires next_state and next_action."
            )

        target = (
            reward
            + gamma * q_table[
                next_state,
                next_action
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
# 6. TRAINING RESULT
# ============================================================

@dataclass
class TrainingResult:

    q_table: np.ndarray

    episode_returns: np.ndarray

    episode_lengths: np.ndarray

    cliff_falls: np.ndarray

    epsilon_history: np.ndarray


# ============================================================
# 7. TRAINING FUNCTION
# ============================================================

def train_agent(
    algorithm="q_learning",
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
    Train Q-learning or SARSA.

    Q-learning:

        1. Select action using epsilon-greedy.
        2. Execute action.
        3. Observe reward and next state.
        4. Update using max_a Q(S',a).
        5. Continue.

    SARSA:

        1. Select action using epsilon-greedy.
        2. Execute action.
        3. Observe reward and next state.
        4. Select next action using epsilon-greedy.
        5. Update using Q(S',A').
        6. Continue using the sampled next action.

    External step-limit truncation:

        The final observation is valid.
        The update bootstraps from that observation.

    True goal termination:

        No bootstrapping.
    """

    if algorithm not in (
        "q_learning",
        "sarsa",
    ):

        raise ValueError(
            "algorithm must be q_learning or sarsa."
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

            truncated = (
                step == max_steps - 1
                and not terminated
            )

            if algorithm == "q_learning":

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

            else:

                if terminated:

                    next_action = None

                else:

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

            if terminated:

                lengths[episode] = step + 1

                break

            # For SARSA, continue with the sampled action.
            # For Q-learning, the action is selected independently
            # after the update.
            if algorithm == "sarsa":

                action = next_action

            else:

                action = select_action(
                    q_table,
                    next_state,
                    epsilon,
                    rng,
                )

            state = next_state

            if truncated:

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
# 8. GREEDY POLICY EVALUATION
# ============================================================

def evaluate_greedy_policy(
    q_table,
    episodes=30,
    seed=SEED + 10000,
    max_steps=500,
):
    """
    Evaluate the learned policy without exploration.

    epsilon = 0

    This separates policy evaluation from training behavior.
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

        "total_cliff_falls": (
            total_cliff_falls
        ),
    }


# ============================================================
# 9. POLICY AND ROUTE EXTRACTION
# ============================================================

def greedy_action(q_table, state):

    return int(
        np.argmax(q_table[state])
    )


def extract_greedy_path(
    q_table,
    max_steps=100,
):
    """
    Trace a greedy policy from START.

    Returns:
        path
        number of cliff falls
        whether goal was reached
    """

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


# ============================================================
# 10. METRICS
# ============================================================

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


def state_value_view(q_table):

    return np.max(
        q_table,
        axis=1,
    ).reshape(
        ROWS,
        COLS,
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
# 12. VISUALIZATIONS
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


def visualize_learning_curves(
    q_result,
    sarsa_result,
):
    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.plot(
        moving_average(
            q_result.episode_returns
        ),
        label="Q-learning",
    )

    ax.plot(
        moving_average(
            sarsa_result.episode_returns
        ),
        label="SARSA",
    )

    ax.set_title(
        "Q-learning vs SARSA: Training Return"
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


def visualize_cliff_falls(
    q_result,
    sarsa_result,
):
    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.plot(
        moving_average(
            q_result.cliff_falls.astype(float)
        ),
        label="Q-learning",
    )

    ax.plot(
        moving_average(
            sarsa_result.cliff_falls.astype(float)
        ),
        label="SARSA",
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


def visualize_greedy_paths(
    q_table,
    sarsa_q_table,
):
    q_path, _, q_success = extract_greedy_path(
        q_table
    )

    sarsa_path, _, sarsa_success = extract_greedy_path(
        sarsa_q_table
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(15, 4),
    )

    data = [
        (
            axes[0],
            q_path,
            "Q-learning",
            q_success,
        ),
        (
            axes[1],
            sarsa_path,
            "SARSA",
            sarsa_success,
        ),
    ]

    for ax, path, name, success in data:

        draw_environment(ax)

        if path:

            rows = [
                position[0]
                for position in path
            ]

            cols = [
                position[1]
                for position in path
            ]

            ax.plot(
                cols,
                rows,
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


def visualize_epsilon(
    result,
):
    fig, ax = plt.subplots(
        figsize=(9, 4)
    )

    ax.plot(
        result.epsilon_history
    )

    ax.set_title(
        "Epsilon Decay"
    )

    ax.set_xlabel(
        "Episode"
    )

    ax.set_ylabel(
        "Epsilon"
    )

    ax.grid(True)

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

    assert env.state_to_position(
        result.next_state
    ) == GOAL

    assert result.terminated

    assert result.reward == -1.0

    print("[PASS] Goal termination")

    # Test 5: epsilon-greedy probability normalization.
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

    # Test 7: numerical Q-learning update.
    q = np.zeros(
        (3, NUM_ACTIONS)
    )

    q[1, 0] = 0.4

    q[2] = [
        0.1,
        0.8,
        0.3,
        0.2,
    ]

    error = q_learning_update(
        q_table=q,
        state=1,
        action=0,
        reward=0.0,
        next_state=2,
        terminated=False,
        alpha=0.1,
        gamma=1.0,
    )

    # Target = 0 + 1 * 0.8 = 0.8
    # TD error = 0.8 - 0.4 = 0.4
    # Updated Q = 0.4 + 0.1 * 0.4 = 0.44

    assert np.isclose(
        error,
        0.4,
    )

    assert np.isclose(
        q[1, 0],
        0.44,
    )

    print("[PASS] Q-learning numerical update")

    # Test 8: terminal no-bootstrap.
    q = np.zeros(
        (3, NUM_ACTIONS)
    )

    q[1, 0] = 0.4

    q[2, :] = 1000.0

    q_learning_update(
        q,
        1,
        0,
        reward=2.0,
        next_state=None,
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

    # Test 9: SARSA uses the sampled next action.
    q = np.zeros(
        (3, NUM_ACTIONS)
    )

    q[1, 0] = 0.4

    q[2, 1] = 0.6

    q[2, 2] = 0.9

    sarsa_update(
        q,
        1,
        0,
        reward=0.0,
        next_state=2,
        next_action=1,
        terminated=False,
        alpha=0.1,
        gamma=1.0,
    )

    # SARSA uses 0.6, not max(0.6, 0.9).
    assert np.isclose(
        q[1, 0],
        0.42,
    )

    print("[PASS] SARSA sampled-action target")

    # Test 10: Q-learning and SARSA targets differ.
    q1 = np.zeros(
        (3, NUM_ACTIONS)
    )

    q2 = np.zeros(
        (3, NUM_ACTIONS)
    )

    q1[1, 0] = 0.4
    q2[1, 0] = 0.4

    q1[2, 1] = 0.6
    q1[2, 2] = 0.9

    q2[:] = q1

    q_learning_update(
        q1,
        1,
        0,
        0.0,
        2,
        False,
        0.1,
        1.0,
    )

    sarsa_update(
        q2,
        1,
        0,
        0.0,
        2,
        1,
        False,
        0.1,
        1.0,
    )

    assert not np.isclose(
        q1[1, 0],
        q2[1, 0],
    )

    print("[PASS] On-policy vs off-policy distinction")

    # Test 11: invalid epsilon.
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

    # Test 12: invalid algorithm.
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

    # Test 13: training produces finite Q values.
    result = train_agent(
        algorithm="q_learning",
        num_episodes=30,
        seed=SEED,
        max_steps=200,
    )

    assert np.all(
        np.isfinite(result.q_table)
    )

    assert len(
        result.episode_returns
    ) == 30

    print("[PASS] Q-learning numerical stability")

    # Test 14: SARSA training produces finite Q values.
    result = train_agent(
        algorithm="sarsa",
        num_episodes=30,
        seed=SEED,
        max_steps=200,
    )

    assert np.all(
        np.isfinite(result.q_table)
    )

    print("[PASS] SARSA numerical stability")

    # Test 15: Q-table shape.
    assert result.q_table.shape == (
        NUM_STATES,
        NUM_ACTIONS,
    )

    print("[PASS] Q-table dimensions")

    print("\nAll 15 tests passed.")


# ============================================================
# 14. MAIN
# ============================================================

def main():

    print("\n==============================================")

    print(
        "PHASE 3.4 — Q-LEARNING"
    )

    print(
        "OFF-POLICY TEMPORAL-DIFFERENCE CONTROL"
    )

    print("==============================================")

    # Step 1: environment visualization.
    visualize_environment()

    # Step 2: train Q-learning.
    print("\n[1] Training Q-learning...")

    q_result = train_agent(
        algorithm="q_learning",
        seed=SEED,
    )

    # Step 3: train SARSA.
    print("\n[2] Training SARSA...")

    sarsa_result = train_agent(
        algorithm="sarsa",
        seed=SEED + 1,
    )

    # Step 4: evaluate greedy policies.
    print("\n[3] Greedy policy evaluation")

    q_metrics = evaluate_greedy_policy(
        q_result.q_table,
        seed=SEED + 100,
    )

    sarsa_metrics = evaluate_greedy_policy(
        sarsa_result.q_table,
        seed=SEED + 200,
    )

    for name, metrics in (
        ("Q-learning", q_metrics),
        ("SARSA", sarsa_metrics),
    ):

        print(f"\n{name}:")

        for key, value in metrics.items():

            print(
                f"  {key}: {value:.3f}"
            )

    # Step 5: training statistics.
    print("\n[4] Training statistics")

    print(
        "Q-learning final 20-episode mean return:",
        f"{np.mean(q_result.episode_returns[-20:]):.3f}",
    )

    print(
        "SARSA final 20-episode mean return:",
        f"{np.mean(sarsa_result.episode_returns[-20:]):.3f}",
    )

    print(
        "Q-learning training cliff falls:",
        int(np.sum(q_result.cliff_falls)),
    )

    print(
        "SARSA training cliff falls:",
        int(np.sum(sarsa_result.cliff_falls)),
    )

    # Step 6: unit tests.
    print("\n[5] Unit tests")

    run_tests()

    # Step 7: integrated visualizations.
    print("\n[6] Visualizations")

    visualize_learning_curves(
        q_result,
        sarsa_result,
    )

    visualize_cliff_falls(
        q_result,
        sarsa_result,
    )

    visualize_policy(
        q_result.q_table,
        "Q-learning Greedy Policy",
    )

    visualize_policy(
        sarsa_result.q_table,
        "SARSA Greedy Policy",
    )

    visualize_value_heatmap(
        q_result.q_table,
        "Q-learning State-Value View",
    )

    visualize_value_heatmap(
        sarsa_result.q_table,
        "SARSA State-Value View",
    )

    visualize_greedy_paths(
        q_result.q_table,
        sarsa_result.q_table,
    )

    visualize_epsilon(
        q_result,
    )

    plt.show()

    print("\n==============================================")

    print(
        "PHASE 3.4 COMPLETED"
    )

    print("==============================================")


if __name__ == "__main__":

    main()
