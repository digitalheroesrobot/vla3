
"""
====================================================================
RL COURSE — PHASE 3.8
N-STEP SARSA AND N-STEP Q-LEARNING
====================================================================

FILE:
    phase3/n_step_control.py

LESSON ROADMAP
--------------

STEP 1 — One-Step TD Control
Purpose:
    Review SARSA and Q-learning.
What you learn:
    One-step action-value updates.
Implementation:
    sarsa_update(), q_learning_update().

STEP 2 — N-Step Returns
Purpose:
    Combine multiple rewards before bootstrapping.
What you learn:
    How n-step returns connect TD learning and Monte Carlo.
Implementation:
    calculate_n_step_return().

STEP 3 — N-Step SARSA
Purpose:
    Learn action values using sampled future actions.
What you learn:
    On-policy multi-step control.
Implementation:
    n_step_sarsa_update().

STEP 4 — N-Step Q-Learning
Purpose:
    Learn action values using greedy future action values.
What you learn:
    Off-policy multi-step control.
Implementation:
    n_step_q_learning_update().

STEP 5 — Episode Boundaries
Purpose:
    Correctly handle termination and truncation.
What you learn:
    When to bootstrap and when not to bootstrap.
Implementation:
    collect_episode(), calculate_n_step_return().

STEP 6 — Algorithm Comparison
Purpose:
    Compare SARSA and Q-learning with different n values.
What you learn:
    How the n-step horizon affects learning.
Implementation:
    train_agent(), run_experiments().

STEP 7 — Verification
Purpose:
    Verify equations and edge cases numerically.
What you learn:
    How to test RL update rules.
Implementation:
    run_tests().

STEP 8 — Visualization
Purpose:
    Visualize learning curves and learned policies.
What you learn:
    How to interpret control performance.
Implementation:
    visualize_*() functions called from main().

MATHEMATICS
-----------

N-STEP RETURN:

    G_(t:t+n) =
        R_(t+1)
        + gamma R_(t+2)
        + ...
        + gamma^(n-1) R_(t+n)
        + gamma^n Bootstrap

N-STEP SARSA BOOTSTRAP:

    Bootstrap = Q(S_(t+n), A_(t+n))

N-STEP Q-LEARNING BOOTSTRAP:

    Bootstrap = max_a Q(S_(t+n), a)

TRUE TERMINATION:
    If the n-step horizon reaches a true terminal state,
    do not bootstrap.

EXTERNAL TRUNCATION:
    If the n-step horizon reaches an external time limit,
    bootstrap from the final valid observation.

INTERMEDIATE HORIZON:
    If the n-step horizon ends before the episode ends,
    bootstrap from that intermediate state—even if the
    episode eventually terminates.

ENVIRONMENT
-----------

    Grid: 4 x 12
    Start: (3, 0)
    Goal:  (3, 11)
    Cliff: (3, 1) through (3, 10)

    Actions:
        0 = UP
        1 = RIGHT
        2 = DOWN
        3 = LEFT

    Rewards:
        Ordinary movement: -1
        Cliff: -100
        Goal: -1

REAL-WORLD APPLICATION
----------------------

    Warehouse robot navigation:
        State: position, velocity, obstacle information.
        Action: movement command.
        Reward: progress, energy cost, collision penalty.

    N-step SARSA uses sampled future actions.
    N-step Q-learning uses greedy future action values.

    Neither algorithm guarantees physical robot safety.
    A separate safety controller is necessary.

DEPENDENCIES
------------

    pip install numpy matplotlib

RUN
---

    python phase3/n_step_control.py

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

ROWS = 4
COLS = 12

START = (3, 0)
GOAL = (3, 11)

CLIFF_CELLS = {
    (3, col) for col in range(1, 11)
}

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

ACTIONS = [UP, RIGHT, DOWN, LEFT]

ACTION_NAMES = ["UP", "RIGHT", "DOWN", "LEFT"]
ACTION_SYMBOLS = ["^", ">", "v", "<"]

GAMMA = 1.0
ALPHA = 0.5

EPSILON_START = 0.10
EPSILON_END = 0.02
EPSILON_DECAY_EPISODES = 600

MAX_STEPS = 500
TRAINING_EPISODES = 800

N_VALUES = [1, 3, 5, 8]
NUM_RUNS = 3


# ================================================================
# 2. CLIFF-WALKING ENVIRONMENT
# ================================================================

class CliffWalking:
    """
    Cliff:
        Reward -100, return to start, episode continues.

    Goal:
        Reward -1, episode terminates.
    """

    def __init__(self):
        self.state = START
        self.terminated = False

    def reset(self):
        self.state = START
        self.terminated = False
        return self.state

    def step(self, action):

        if self.terminated:
            raise RuntimeError(
                "Episode terminated. Call reset()."
            )

        if action not in ACTIONS:
            raise ValueError("Invalid action.")

        row, col = self.state

        if action == UP:
            row -= 1
        elif action == RIGHT:
            col += 1
        elif action == DOWN:
            row += 1
        elif action == LEFT:
            col -= 1

        row = int(np.clip(row, 0, ROWS - 1))
        col = int(np.clip(col, 0, COLS - 1))

        next_state = (row, col)

        if next_state in CLIFF_CELLS:
            reward = -100.0
            next_state = START
            terminated = False

        elif next_state == GOAL:
            reward = -1.0
            terminated = True

        else:
            reward = -1.0
            terminated = False

        self.state = next_state
        self.terminated = terminated

        return next_state, reward, terminated


# ================================================================
# 3. STATE INDEXING
# ================================================================

def state_to_index(state):

    row, col = state

    if not (
        0 <= row < ROWS
        and 0 <= col < COLS
    ):
        raise ValueError("State outside environment.")

    return row * COLS + col


def index_to_state(index):

    if not 0 <= index < ROWS * COLS:
        raise ValueError("Invalid state index.")

    return index // COLS, index % COLS


# ================================================================
# 4. EPSILON-GREEDY POLICY
# ================================================================

def epsilon_greedy_probabilities(q_values, epsilon):

    q_values = np.asarray(q_values, dtype=float)

    if q_values.ndim != 1 or len(q_values) == 0:
        raise ValueError(
            "q_values must be a nonempty vector."
        )

    if not np.all(np.isfinite(q_values)):
        raise ValueError("q_values must be finite.")

    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must be in [0, 1].")

    probabilities = np.full(
        len(q_values),
        epsilon / len(q_values),
    )

    greedy_actions = np.flatnonzero(
        np.isclose(q_values, np.max(q_values))
    )

    probabilities[greedy_actions] += (
        (1.0 - epsilon) / len(greedy_actions)
    )

    return probabilities


def select_action(q_table, state, epsilon, rng):

    row, col = state

    probabilities = epsilon_greedy_probabilities(
        q_table[row, col],
        epsilon,
    )

    return int(
        rng.choice(ACTIONS, p=probabilities)
    )


def epsilon_schedule(episode):

    fraction = min(
        episode / EPSILON_DECAY_EPISODES,
        1.0,
    )

    return (
        EPSILON_START
        + fraction * (EPSILON_END - EPSILON_START)
    )


# ================================================================
# 5. ONE-STEP SARSA
# ================================================================

def sarsa_update(
    q_table,
    state,
    action,
    reward,
    next_state,
    next_action,
    terminated,
    alpha=ALPHA,
    gamma=GAMMA,
):

    row, col = state

    current_q = q_table[row, col, action]

    if terminated:
        target = reward

    else:

        if next_action is None:
            raise ValueError(
                "Nonterminal SARSA update requires next_action."
            )

        next_row, next_col = next_state

        target = (
            reward
            + gamma
            * q_table[next_row, next_col, next_action]
        )

    td_error = target - current_q

    q_table[row, col, action] += alpha * td_error

    return float(td_error)


# ================================================================
# 6. ONE-STEP Q-LEARNING
# ================================================================

def q_learning_update(
    q_table,
    state,
    action,
    reward,
    next_state,
    terminated,
    alpha=ALPHA,
    gamma=GAMMA,
):

    row, col = state

    current_q = q_table[row, col, action]

    if terminated:
        target = reward

    else:

        next_row, next_col = next_state

        target = (
            reward
            + gamma * np.max(q_table[next_row, next_col])
        )

    td_error = target - current_q

    q_table[row, col, action] += alpha * td_error

    return float(td_error)


# ================================================================
# 7. N-STEP RETURN
# ================================================================

def calculate_n_step_return(
    q_table,
    states,
    actions,
    rewards,
    tau,
    n,
    algorithm="sarsa",
    terminated=False,
    truncated=False,
    gamma=GAMMA,
):
    """
    Trajectory convention:

        states  = [S0, S1, ..., ST]
        actions = [A0, A1, ..., A(T-1)]
        rewards = [R1, R2, ..., RT]

    True terminal:
        len(actions) = len(rewards)

    External truncation:
        len(actions) = len(rewards) + 1

        The extra action is the sampled final SARSA
        bootstrap action.

    For Q-learning, the extra action is not required.

    Bootstrap rules:

        1. Horizon ends before episode end:
               Bootstrap.

        2. Horizon reaches true terminal:
               No bootstrap.

        3. Horizon reaches external truncation:
               Bootstrap.

    The discount exponent equals the number of rewards
    accumulated before the bootstrap state.
    """

    if algorithm not in ("sarsa", "q_learning"):
        raise ValueError("Invalid algorithm.")

    if not isinstance(n, (int, np.integer)) or n <= 0:
        raise ValueError("n must be a positive integer.")

    if not isinstance(tau, (int, np.integer)):
        raise ValueError("tau must be an integer.")

    if not 0 <= tau < len(rewards):
        raise ValueError("Invalid tau.")

    if terminated and truncated:
        raise ValueError(
            "An episode cannot be both terminated and truncated."
        )

    total_steps = len(rewards)

    if len(states) != total_steps + 1:
        raise ValueError(
            "states must have one more element than rewards."
        )

    if len(actions) not in (
        total_steps,
        total_steps + 1,
    ):
        raise ValueError(
            "actions length must equal rewards length "
            "or exceed it by one."
        )

    if algorithm == "sarsa" and truncated:
        if len(actions) != total_steps + 1:
            raise ValueError(
                "Truncated SARSA trajectory requires "
                "a final bootstrap action."
            )

    remaining_steps = total_steps - tau
    horizon = min(int(n), remaining_steps)

    # ------------------------------------------------------------
    # Calculate discounted observed rewards.
    #
    # G = R_(tau+1) + gamma*R_(tau+2) + ...
    # ------------------------------------------------------------

    G = 0.0

    for k in range(horizon):

        G += (
            gamma ** k
            * rewards[tau + k]
        )

    horizon_index = tau + horizon

    # ------------------------------------------------------------
    # Determine whether to bootstrap.
    #
    # Intermediate horizon:
    #     horizon_index < total_steps
    #
    # External truncation:
    #     horizon_index == total_steps and truncated
    #
    # True terminal:
    #     horizon_index == total_steps and terminated
    # ------------------------------------------------------------

    horizon_ends_before_episode = (
        horizon_index < total_steps
    )

    horizon_reaches_truncation = (
        horizon_index == total_steps
        and truncated
    )

    should_bootstrap = (
        horizon_ends_before_episode
        or horizon_reaches_truncation
    )

    if should_bootstrap:

        bootstrap_state = states[horizon_index]

        next_row, next_col = bootstrap_state

        if algorithm == "sarsa":

            if horizon_index >= len(actions):
                raise ValueError(
                    "SARSA requires the sampled bootstrap action."
                )

            bootstrap_action = actions[horizon_index]

            bootstrap_value = q_table[
                next_row,
                next_col,
                bootstrap_action,
            ]

        else:

            bootstrap_value = np.max(
                q_table[next_row, next_col]
            )

        G += (
            gamma ** horizon
            * bootstrap_value
        )

    return float(G)


# ================================================================
# 8. N-STEP SARSA UPDATE
# ================================================================

def n_step_sarsa_update(
    q_table,
    states,
    actions,
    rewards,
    tau,
    n,
    terminated=False,
    truncated=False,
    alpha=ALPHA,
    gamma=GAMMA,
):

    G = calculate_n_step_return(
        q_table=q_table,
        states=states,
        actions=actions,
        rewards=rewards,
        tau=tau,
        n=n,
        algorithm="sarsa",
        terminated=terminated,
        truncated=truncated,
        gamma=gamma,
    )

    row, col = states[tau]
    action = actions[tau]

    current_q = q_table[row, col, action]

    td_error = G - current_q

    q_table[row, col, action] += alpha * td_error

    return float(td_error)


# ================================================================
# 9. N-STEP Q-LEARNING UPDATE
# ================================================================

def n_step_q_learning_update(
    q_table,
    states,
    actions,
    rewards,
    tau,
    n,
    terminated=False,
    truncated=False,
    alpha=ALPHA,
    gamma=GAMMA,
):

    G = calculate_n_step_return(
        q_table=q_table,
        states=states,
        actions=actions,
        rewards=rewards,
        tau=tau,
        n=n,
        algorithm="q_learning",
        terminated=terminated,
        truncated=truncated,
        gamma=gamma,
    )

    row, col = states[tau]
    action = actions[tau]

    current_q = q_table[row, col, action]

    td_error = G - current_q

    q_table[row, col, action] += alpha * td_error

    return float(td_error)


# ================================================================
# 10. EPISODE COLLECTION
# ================================================================

def collect_episode(
    env,
    q_table,
    epsilon,
    rng,
    max_steps=MAX_STEPS,
):
    """
    Collect a complete trajectory.

    If a true terminal state is reached:
        No final bootstrap action is stored.

    If an external time limit is reached:
        Store one sampled action at the final valid state.
        This action is needed by SARSA bootstrapping.
    """

    states = [env.reset()]

    actions = [
        select_action(
            q_table,
            states[0],
            epsilon,
            rng,
        )
    ]

    rewards = []

    terminated = False
    truncated = False

    for step_index in range(max_steps):

        current_state = states[-1]
        current_action = actions[-1]

        next_state, reward, terminated = env.step(
            current_action
        )

        rewards.append(reward)
        states.append(next_state)

        if terminated:
            break

        next_action = select_action(
            q_table,
            next_state,
            epsilon,
            rng,
        )

        actions.append(next_action)

        if step_index == max_steps - 1:
            truncated = True
            break

    return {
        "states": states,
        "actions": actions,
        "rewards": rewards,
        "terminated": terminated,
        "truncated": truncated,
    }


# ================================================================
# 11. UPDATE A COMPLETE EPISODE
# ================================================================

def update_episode(
    q_table,
    episode,
    n,
    algorithm,
    alpha=ALPHA,
    gamma=GAMMA,
):

    states = episode["states"]
    actions = episode["actions"]
    rewards = episode["rewards"]

    terminated = episode["terminated"]
    truncated = episode["truncated"]

    errors = []

    for tau in range(len(rewards)):

        if algorithm == "sarsa":

            error = n_step_sarsa_update(
                q_table,
                states,
                actions,
                rewards,
                tau,
                n,
                terminated,
                truncated,
                alpha,
                gamma,
            )

        elif algorithm == "q_learning":

            error = n_step_q_learning_update(
                q_table,
                states,
                actions,
                rewards,
                tau,
                n,
                terminated,
                truncated,
                alpha,
                gamma,
            )

        else:
            raise ValueError("Unknown algorithm.")

        errors.append(error)

    return errors


# ================================================================
# 12. TRAINING
# ================================================================

@dataclass
class TrainingResult:

    algorithm: str
    n: int
    q_table: np.ndarray
    episode_rewards: np.ndarray
    episode_lengths: np.ndarray
    cliff_falls: np.ndarray
    terminated_count: int
    truncated_count: int


def count_cliff_falls(episode):

    return int(
        sum(
            reward == -100.0
            for reward in episode["rewards"]
        )
    )


def train_agent(
    algorithm,
    n,
    num_episodes=TRAINING_EPISODES,
    seed=SEED,
    alpha=ALPHA,
    gamma=GAMMA,
):

    if algorithm not in ("sarsa", "q_learning"):
        raise ValueError("Invalid algorithm.")

    if n <= 0:
        raise ValueError("n must be positive.")

    rng = np.random.default_rng(seed)

    env = CliffWalking()

    q_table = np.zeros(
        (ROWS, COLS, len(ACTIONS)),
        dtype=float,
    )

    episode_rewards = np.zeros(num_episodes)
    episode_lengths = np.zeros(num_episodes, dtype=int)
    cliff_falls = np.zeros(num_episodes, dtype=int)

    terminated_count = 0
    truncated_count = 0

    for episode_index in range(num_episodes):

        epsilon = epsilon_schedule(episode_index)

        episode = collect_episode(
            env,
            q_table,
            epsilon,
            rng,
        )

        update_episode(
            q_table,
            episode,
            n,
            algorithm,
            alpha,
            gamma,
        )

        episode_rewards[episode_index] = sum(
            episode["rewards"]
        )

        episode_lengths[episode_index] = len(
            episode["rewards"]
        )

        cliff_falls[episode_index] = (
            count_cliff_falls(episode)
        )

        terminated_count += int(
            episode["terminated"]
        )

        truncated_count += int(
            episode["truncated"]
        )

    return TrainingResult(
        algorithm=algorithm,
        n=n,
        q_table=q_table,
        episode_rewards=episode_rewards,
        episode_lengths=episode_lengths,
        cliff_falls=cliff_falls,
        terminated_count=terminated_count,
        truncated_count=truncated_count,
    )


# ================================================================
# 13. GREEDY POLICY AND EVALUATION
# ================================================================

def greedy_action(q_table, state):

    row, col = state

    return int(
        np.argmax(q_table[row, col])
    )


def evaluate_greedy_policy(
    q_table,
    num_episodes=30,
    max_steps=MAX_STEPS,
):

    env = CliffWalking()

    rewards = []
    lengths = []
    falls = []
    successes = []

    for _ in range(num_episodes):

        state = env.reset()

        total_reward = 0.0
        fall_count = 0
        terminated = False
        steps = 0

        while not terminated and steps < max_steps:

            action = greedy_action(
                q_table,
                state,
            )

            state, reward, terminated = env.step(
                action
            )

            total_reward += reward

            if reward == -100.0:
                fall_count += 1

            steps += 1

        rewards.append(total_reward)
        lengths.append(steps)
        falls.append(fall_count)
        successes.append(terminated)

    return {
        "mean_reward": float(np.mean(rewards)),
        "mean_length": float(np.mean(lengths)),
        "mean_cliff_falls": float(np.mean(falls)),
        "success_rate": float(np.mean(successes)),
    }


def extract_greedy_path(q_table, max_steps=100):

    env = CliffWalking()

    state = env.reset()

    path = [state]
    total_reward = 0.0
    success = False

    for _ in range(max_steps):

        action = greedy_action(
            q_table,
            state,
        )

        next_state, reward, terminated = env.step(
            action
        )

        total_reward += reward
        path.append(next_state)

        state = next_state

        if terminated:
            success = True
            break

    return {
        "path": path,
        "reward": total_reward,
        "success": success,
    }


# ================================================================
# 14. MOVING AVERAGE
# ================================================================

def moving_average(values, window=50):

    values = np.asarray(values, dtype=float)

    if len(values) < window:
        return values

    kernel = np.ones(window) / window

    return np.convolve(
        values,
        kernel,
        mode="valid",
    )


# ================================================================
# 15. COMPARISON EXPERIMENTS
# ================================================================

def run_experiments(
    num_runs=NUM_RUNS,
    num_episodes=TRAINING_EPISODES,
):

    configurations = [
        ("sarsa", 1),
        ("sarsa", 3),
        ("sarsa", 5),
        ("sarsa", 8),
        ("q_learning", 3),
        ("q_learning", 5),
        ("q_learning", 8),
    ]

    results = {}

    for algorithm, n in configurations:

        key = f"{algorithm}_n{n}"

        reward_runs = []
        length_runs = []
        fall_runs = []
        evaluation_runs = []

        print(f"\nTraining {key} ...")

        for run_index in range(num_runs):

            result = train_agent(
                algorithm,
                n,
                num_episodes,
                seed=SEED + run_index * 100,
            )

            reward_runs.append(result.episode_rewards)
            length_runs.append(result.episode_lengths)
            fall_runs.append(result.cliff_falls)

            evaluation_runs.append(
                evaluate_greedy_policy(
                    result.q_table,
                    num_episodes=20,
                )
            )

        results[key] = {
            "algorithm": algorithm,
            "n": n,
            "reward_mean": np.mean(
                reward_runs,
                axis=0,
            ),
            "reward_std": np.std(
                reward_runs,
                axis=0,
            ),
            "length_mean": np.mean(
                length_runs,
                axis=0,
            ),
            "falls_mean": np.mean(
                fall_runs,
                axis=0,
            ),
            "evaluation": evaluation_runs,
        }

        print(
            "  Average greedy reward:",
            np.mean([
                x["mean_reward"]
                for x in evaluation_runs
            ]),
        )

        print(
            "  Average greedy steps:",
            np.mean([
                x["mean_length"]
                for x in evaluation_runs
            ]),
        )

        print(
            "  Average greedy cliff falls:",
            np.mean([
                x["mean_cliff_falls"]
                for x in evaluation_runs
            ]),
        )

        print(
            "  Greedy success rate:",
            np.mean([
                x["success_rate"]
                for x in evaluation_runs
            ]),
        )

    return results


# ================================================================
# 16. VISUALIZATION — ENVIRONMENT
# ================================================================

def visualize_environment():

    fig, ax = plt.subplots(figsize=(12, 4))

    ax.imshow(
        np.zeros((ROWS, COLS)),
        cmap="Greys",
        vmin=0,
        vmax=1,
    )

    for row, col in CLIFF_CELLS:

        ax.add_patch(
            plt.Rectangle(
                (col - 0.5, row - 0.5),
                1,
                1,
                color="black",
            )
        )

    ax.text(
        START[1],
        START[0],
        "START",
        ha="center",
        va="center",
        color="blue",
        fontweight="bold",
    )

    ax.text(
        GOAL[1],
        GOAL[0],
        "GOAL",
        ha="center",
        va="center",
        color="green",
        fontweight="bold",
    )

    ax.set_xticks(range(COLS))
    ax.set_yticks(range(ROWS))
    ax.set_title("Cliff-Walking Environment")
    ax.grid(True)

    fig.tight_layout()


# ================================================================
# 17. VISUALIZATION — LEARNING CURVES
# ================================================================

def visualize_learning_curves(results):

    fig, ax = plt.subplots(figsize=(12, 6))

    for key, data in results.items():

        mean = moving_average(
            data["reward_mean"],
            50,
        )

        std = moving_average(
            data["reward_std"],
            50,
        )

        episodes = np.arange(len(mean)) + 49

        ax.plot(
            episodes,
            mean,
            label=key,
        )

        ax.fill_between(
            episodes,
            mean - std,
            mean + std,
            alpha=0.10,
        )

    ax.set_title("N-Step Control: Training Rewards")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total episode reward")
    ax.legend()
    ax.grid(True)

    fig.tight_layout()


# ================================================================
# 18. VISUALIZATION — EPISODE LENGTH
# ================================================================

def visualize_episode_lengths(results):

    fig, ax = plt.subplots(figsize=(12, 5))

    for key, data in results.items():

        lengths = moving_average(
            data["length_mean"],
            50,
        )

        ax.plot(
            lengths,
            label=key,
        )

    ax.set_title("Episode Length During Training")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Steps per episode")
    ax.legend()
    ax.grid(True)

    fig.tight_layout()


# ================================================================
# 19. VISUALIZATION — CLIFF FALLS
# ================================================================

def visualize_cliff_falls(results):

    fig, ax = plt.subplots(figsize=(12, 5))

    for key, data in results.items():

        falls = moving_average(
            data["falls_mean"],
            50,
        )

        ax.plot(
            falls,
            label=key,
        )

    ax.set_title("Cliff Falls During Training")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Cliff falls per episode")
    ax.legend()
    ax.grid(True)

    fig.tight_layout()


# ================================================================
# 20. VISUALIZATION — GREEDY POLICY
# ================================================================

def visualize_policy(
    q_table,
    title="Greedy Policy",
):

    fig, ax = plt.subplots(figsize=(12, 4))

    ax.set_xlim(-0.5, COLS - 0.5)
    ax.set_ylim(ROWS - 0.5, -0.5)

    for row in range(ROWS):

        for col in range(COLS):

            state = (row, col)

            if state in CLIFF_CELLS:

                ax.add_patch(
                    plt.Rectangle(
                        (col - 0.5, row - 0.5),
                        1,
                        1,
                        color="black",
                    )
                )

            elif state == START:

                ax.text(
                    col,
                    row,
                    "S",
                    ha="center",
                    va="center",
                    color="blue",
                    fontweight="bold",
                )

            elif state == GOAL:

                ax.text(
                    col,
                    row,
                    "G",
                    ha="center",
                    va="center",
                    color="green",
                    fontweight="bold",
                )

            else:

                action = greedy_action(
                    q_table,
                    state,
                )

                ax.text(
                    col,
                    row,
                    ACTION_SYMBOLS[action],
                    ha="center",
                    va="center",
                    fontsize=15,
                )

    ax.set_xticks(range(COLS))
    ax.set_yticks(range(ROWS))
    ax.grid(True)
    ax.set_title(title)

    fig.tight_layout()


# ================================================================
# 21. VISUALIZATION — Q-VALUE HEATMAP
# ================================================================

def visualize_q_values(
    q_table,
    title="Maximum Q-Value by State",
):

    values = np.max(q_table, axis=2)

    fig, ax = plt.subplots(figsize=(12, 4))

    image = ax.imshow(
        values,
        cmap="viridis",
    )

    for row in range(ROWS):

        for col in range(COLS):

            state = (row, col)

            if state in CLIFF_CELLS:
                label = "X"

            elif state == START:
                label = "S"

            elif state == GOAL:
                label = "G"

            else:
                label = f"{values[row, col]:.1f}"

            ax.text(
                col,
                row,
                label,
                ha="center",
                va="center",
                color="white",
                fontsize=8,
            )

    ax.set_title(title)
    ax.set_xticks(range(COLS))
    ax.set_yticks(range(ROWS))

    fig.colorbar(
        image,
        ax=ax,
        label="max Q(s,a)",
    )

    fig.tight_layout()


# ================================================================
# 22. VISUALIZATION — GREEDY PATH
# ================================================================

def visualize_greedy_path(
    q_table,
    title="Greedy Path",
):

    result = extract_greedy_path(q_table)

    path = result["path"]

    fig, ax = plt.subplots(figsize=(12, 4))

    ax.set_xlim(-0.5, COLS - 0.5)
    ax.set_ylim(ROWS - 0.5, -0.5)

    for row, col in CLIFF_CELLS:

        ax.add_patch(
            plt.Rectangle(
                (col - 0.5, row - 0.5),
                1,
                1,
                color="black",
            )
        )

    path_rows = [state[0] for state in path]
    path_cols = [state[1] for state in path]

    ax.plot(
        path_cols,
        path_rows,
        marker="o",
        linewidth=2,
    )

    ax.scatter(
        START[1],
        START[0],
        marker="s",
        s=100,
        label="Start",
    )

    ax.scatter(
        GOAL[1],
        GOAL[0],
        marker="*",
        s=200,
        label="Goal",
    )

    ax.set_xticks(range(COLS))
    ax.set_yticks(range(ROWS))
    ax.grid(True)

    ax.set_title(
        f"{title} | "
        f"Success={result['success']} | "
        f"Reward={result['reward']:.1f}"
    )

    ax.legend()

    fig.tight_layout()


# ================================================================
# 23. UNIT TESTS
# ================================================================

def run_tests():

    print("\n========== UNIT TESTS ==========")

    # Test 1: Environment reset.
    env = CliffWalking()

    assert env.reset() == START

    print("[PASS] Environment reset")

    # Test 2: Ordinary movement.
    state, reward, terminated = env.step(UP)

    assert state == (2, 0)
    assert reward == -1.0
    assert not terminated

    print("[PASS] Ordinary movement")

    # Test 3: Cliff transition.
    env.reset()

    state, reward, terminated = env.step(RIGHT)

    assert state == START
    assert reward == -100.0
    assert not terminated

    print("[PASS] Cliff penalty and reset")

    # Test 4: Goal transition.
    env.state = (3, 10)

    state, reward, terminated = env.step(RIGHT)

    assert state == GOAL
    assert reward == -1.0
    assert terminated

    print("[PASS] Goal termination")

    # Test 5: State indexing.
    state = (2, 5)

    assert index_to_state(
        state_to_index(state)
    ) == state

    print("[PASS] State indexing")

    # Test 6: Epsilon-greedy probabilities.
    probabilities = epsilon_greedy_probabilities(
        [1.0, 3.0, 2.0, 0.0],
        0.2,
    )

    assert np.isclose(np.sum(probabilities), 1.0)
    assert np.isclose(probabilities[1], 0.85)
    assert np.isclose(probabilities[0], 0.05)

    print("[PASS] Epsilon-greedy probabilities")

    # Test 7: One-step SARSA.
    q = np.zeros((ROWS, COLS, 4))

    q[1, 1, UP] = 0.5
    q[1, 2, RIGHT] = 0.8

    sarsa_update(
        q,
        (1, 1),
        UP,
        1.0,
        (1, 2),
        RIGHT,
        False,
        alpha=0.1,
        gamma=0.9,
    )

    assert np.isclose(q[1, 1, UP], 0.622)

    print("[PASS] One-step SARSA update")

    # Test 8: One-step Q-learning.
    q = np.zeros((ROWS, COLS, 4))

    q[1, 1, UP] = 0.5
    q[1, 2] = [0.2, 0.9, 0.3, 0.4]

    q_learning_update(
        q,
        (1, 1),
        UP,
        1.0,
        (1, 2),
        False,
        alpha=0.1,
        gamma=0.9,
    )

    assert np.isclose(q[1, 1, UP], 0.631)

    print("[PASS] One-step Q-learning update")

    # Test 9: Bootstrap before terminal transition.
    q = np.zeros((ROWS, COLS, 4))

    states = [
        (3, 0),
        (2, 0),
        (2, 1),
        GOAL,
    ]

    actions = [UP, RIGHT, RIGHT]
    rewards = [0.2, 0.3, 1.0]

    q[2, 1, RIGHT] = 0.8

    G = calculate_n_step_return(
        q,
        states,
        actions,
        rewards,
        tau=0,
        n=2,
        algorithm="sarsa",
        terminated=True,
        gamma=0.9,
    )

    # 0.2 + 0.9*0.3 + 0.9^2*0.8 = 1.118
    assert np.isclose(G, 1.118)

    print("[PASS] Bootstrap before terminal transition")

    # Test 10: No bootstrap at true terminal.
    q = np.zeros((ROWS, COLS, 4))

    q[3, 11] = [100, 100, 100, 100]

    G = calculate_n_step_return(
        q,
        states,
        actions,
        [0.0, 0.0, 1.0],
        tau=0,
        n=3,
        algorithm="sarsa",
        terminated=True,
        gamma=0.9,
    )

    # No bootstrap:
    # 0 + 0 + 0.9^2 * 1.0 = 0.81
    assert np.isclose(G, 0.81)

    print("[PASS] No bootstrap at true terminal")

    # Test 11: External truncation bootstrap.
    q = np.zeros((ROWS, COLS, 4))

    states = [
        (3, 0),
        (2, 0),
        (2, 1),
        (2, 2),
    ]

    actions = [
        UP,
        RIGHT,
        RIGHT,
        LEFT,
    ]

    q[2, 2, LEFT] = 0.8

    G = calculate_n_step_return(
        q,
        states,
        actions,
        [0.0, 0.0, 0.0],
        tau=0,
        n=3,
        algorithm="sarsa",
        terminated=False,
        truncated=True,
        gamma=0.9,
    )

    # Three observed rewards, then bootstrap:
    # 0 + 0 + 0 + 0.9^3 * 0.8 = 0.5832
    assert np.isclose(G, 0.5832)

    print("[PASS] External truncation bootstrap")

    # Test 12: Q-learning maximum bootstrap.
    q = np.zeros((ROWS, COLS, 4))

    q[2, 1] = [0.1, 0.8, 0.3, 0.2]

    states = [
        (3, 0),
        (2, 0),
        (2, 1),
    ]

    actions = [UP, RIGHT]

    G = calculate_n_step_return(
        q,
        states,
        actions,
        [0.0, 0.0],
        tau=0,
        n=2,
        algorithm="q_learning",
        terminated=False,
        truncated=True,
        gamma=0.9,
    )

    # gamma^2 * max(Q(2,1,:)) = 0.9^2 * 0.8
    assert np.isclose(G, 0.648)

    print("[PASS] Q-learning maximum bootstrap")

    # Test 13: One-step and n-step SARSA equivalence.
    #
    # This trajectory ends because of an external time limit.
    # Therefore it must bootstrap from the final valid state.
    #
    # G = 1.0 + 0.9 * 0.7 = 1.63
    #
    q = np.zeros((ROWS, COLS, 4))

    q[2, 0, RIGHT] = 0.7

    states = [
        (3, 0),
        (2, 0),
    ]

    actions = [
        UP,
        RIGHT,
    ]

    G = calculate_n_step_return(
        q,
        states,
        actions,
        [1.0],
        tau=0,
        n=1,
        algorithm="sarsa",
        terminated=False,
        truncated=True,
        gamma=0.9,
    )

    assert np.isclose(G, 1.63)

    print("[PASS] One-step and n-step SARSA equivalence")

    # Test 14: Terminal SARSA update does not bootstrap.
    q = np.zeros((ROWS, COLS, 4))

    q[3, 11] = 100.0

    sarsa_update(
        q,
        (3, 10),
        RIGHT,
        -1.0,
        GOAL,
        None,
        True,
        alpha=0.5,
        gamma=0.9,
    )

    assert np.isclose(q[3, 10, RIGHT], -0.5)

    print("[PASS] Terminal SARSA update")

    # Test 15: Invalid n is rejected.
    try:

        calculate_n_step_return(
            q,
            [(3, 0), (2, 0)],
            [UP],
            [1.0],
            tau=0,
            n=0,
        )

        raise AssertionError("Expected invalid n error.")

    except ValueError:
        pass

    print("[PASS] Invalid n rejection")

    # Test 16: SARSA training stability.
    result = train_agent(
        "sarsa",
        3,
        num_episodes=50,
        seed=123,
    )

    assert np.all(np.isfinite(result.q_table))
    assert len(result.episode_rewards) == 50

    print("[PASS] SARSA training stability")

    # Test 17: Q-learning training stability.
    result = train_agent(
        "q_learning",
        3,
        num_episodes=50,
        seed=124,
    )

    assert np.all(np.isfinite(result.q_table))

    print("[PASS] Q-learning training stability")

    # Test 18: Greedy evaluation.
    evaluation = evaluate_greedy_policy(
        result.q_table,
        num_episodes=5,
    )

    assert np.isfinite(evaluation["mean_reward"])

    assert 0.0 <= evaluation["success_rate"] <= 1.0

    print("[PASS] Greedy policy evaluation")

    # Test 19: Greedy path extraction.
    path_result = extract_greedy_path(
        result.q_table,
        max_steps=30,
    )

    assert len(path_result["path"]) >= 1
    assert isinstance(path_result["success"], bool)

    print("[PASS] Greedy path extraction")

    # Test 20: Episode trajectory structure.
    env = CliffWalking()
    rng = np.random.default_rng(456)

    q = np.zeros((ROWS, COLS, 4))

    episode = collect_episode(
        env,
        q,
        epsilon=0.2,
        rng=rng,
        max_steps=30,
    )

    assert len(episode["states"]) == (
        len(episode["rewards"]) + 1
    )

    assert len(episode["actions"]) in (
        len(episode["rewards"]),
        len(episode["rewards"]) + 1,
    )

    print("[PASS] Episode trajectory structure")

    print("\nAll 20 tests passed.")


# ================================================================
# 24. MAIN
# ================================================================

def main():

    print("\n==============================================")
    print("RL COURSE — PHASE 3.8")
    print("N-STEP SARSA AND N-STEP Q-LEARNING")
    print("==============================================")

    # Step 1: Run numerical tests.
    run_tests()

    # Step 2: Visualize the environment.
    visualize_environment()

    # Step 3: Compare algorithms.
    print("\n========== TRAINING COMPARISON ==========")

    results = run_experiments(
        num_runs=NUM_RUNS,
        num_episodes=TRAINING_EPISODES,
    )

    # Step 4: Visualize training performance.
    visualize_learning_curves(results)
    visualize_episode_lengths(results)
    visualize_cliff_falls(results)

    # Step 5: Train representative models.
    selected_models = [
        ("sarsa", 1),
        ("sarsa", 5),
        ("q_learning", 5),
    ]

    for algorithm, n in selected_models:

        model_name = f"{algorithm}_n{n}"

        representative = train_agent(
            algorithm,
            n,
            num_episodes=TRAINING_EPISODES,
            seed=SEED + 999,
        )

        evaluation = evaluate_greedy_policy(
            representative.q_table,
            num_episodes=50,
        )

        print(f"\n{model_name} representative evaluation:")

        print(
            f"  Mean reward: "
            f"{evaluation['mean_reward']:.2f}"
        )

        print(
            f"  Mean episode length: "
            f"{evaluation['mean_length']:.2f}"
        )

        print(
            f"  Mean cliff falls: "
            f"{evaluation['mean_cliff_falls']:.2f}"
        )

        print(
            f"  Success rate: "
            f"{evaluation['success_rate']:.2%}"
        )

        visualize_policy(
            representative.q_table,
            title=f"Greedy Policy: {model_name}",
        )

        visualize_q_values(
            representative.q_table,
            title=f"Q-Values: {model_name}",
        )

        visualize_greedy_path(
            representative.q_table,
            title=f"Greedy Path: {model_name}",
        )

    # Step 6: Display all visualizations.
    plt.show()

    print("\n==============================================")
    print("PHASE 3.8 COMPLETED")
    print("==============================================")


if __name__ == "__main__":
    main()
