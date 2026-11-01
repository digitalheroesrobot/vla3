
"""
====================================================================
RL COURSE — PHASE 3.9
ELIGIBILITY TRACES
====================================================================

LESSON ROADMAP
--------------

STEP 1 — Eligibility Traces
Purpose:
    Track recently visited state-action pairs.

What you learn:
    How eligibility traces assign credit to earlier decisions.

Implementation:
    Accumulating and replacing traces.

STEP 2 — CliffWalking Environment
Purpose:
    Create a navigation task for testing trace-based learning.

What you learn:
    State transitions, rewards, terminal states, cliff penalties.

Implementation:
    4 x 12 CliffWalking environment.

STEP 3 — SARSA(lambda)
Purpose:
    Extend one-step SARSA using eligibility traces.

What you learn:
    On-policy TD error and backward-view updates.

Implementation:
    Replacing traces and SARSA(lambda).

STEP 4 — Watkins's Q(lambda)
Purpose:
    Extend off-policy Q-learning using eligibility traces.

What you learn:
    Greedy targets and trace clearing after non-greedy actions.

Implementation:
    Watkins's Q(lambda).

STEP 5 — Lambda Experiments
Purpose:
    Compare short and long eligibility traces.

What you learn:
    lambda=0 corresponds to one-step TD control.
    Larger lambda propagates credit farther backward.

Implementation:
    Train multiple agents with different lambda values.

STEP 6 — Numerical Verification
Purpose:
    Verify mathematical updates and environment behavior.

What you learn:
    TD targets, trace decay, terminal handling, trace clearing.

Implementation:
    22 unit tests.

STEP 7 — Visualization
Purpose:
    Inspect learning and learned navigation policies.

What you learn:
    Learning curves, Q values, greedy paths, trace decay.

Implementation:
    All visualization functions are included in this file.

STEP 8 — Robotics Application
Purpose:
    Connect eligibility traces to robot navigation.

What you learn:
    Delayed reward assignment in sequential robot decisions.

Implementation:
    Warehouse robot navigation example in lesson comments.

CORE EQUATIONS
--------------

SARSA(lambda):

    delta =
        R + gamma * Q(S', A') - Q(S, A)

    E(S,A) = 1                 # replacing trace

    Q <- Q + alpha * delta * E

    E <- gamma * lambda * E

Watkins's Q(lambda):

    delta =
        R + gamma * max_a Q(S',a) - Q(S,A)

    Q <- Q + alpha * delta * E

    If next behavior action is non-greedy:
        E <- 0

SPECIAL CASES
-------------

    lambda = 0:
        One-step TD control.

    lambda approaching 1:
        Longer temporal credit assignment.

REAL-WORLD ROBOTICS APPLICATION
-------------------------------

Warehouse robot navigation:

    State:
        Robot position, orientation, obstacle information.

    Action:
        Forward, backward, left, right.

    Reward:
        -1 for ordinary movement.
        -100 for entering a dangerous region.
        Positive reward for reaching a destination.

Eligibility traces can assign delayed reward or penalty
to recent robot decisions.

IMPORTANT:
    Eligibility traces are not a safety guarantee.
    Real robots still require collision detection,
    emergency stops, action constraints, and validation.

DEPENDENCIES
------------

    pip install numpy matplotlib

RUN
---

    python phase3/eligibility.py
====================================================================
"""

from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt


# ================================================================
# 1. CONFIGURATION
# ================================================================

SEED = 42

ROWS = 4
COLS = 12

START_STATE = (3, 0)
GOAL_STATE = (3, 11)

CLIFF_CELLS = {
    (3, col)
    for col in range(1, 11)
}

# 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
ACTION_DELTAS = {
    0: (-1, 0),
    1: (0, 1),
    2: (1, 0),
    3: (0, -1),
}

ACTION_NAMES = [
    "UP",
    "RIGHT",
    "DOWN",
    "LEFT",
]

NUM_ACTIONS = 4
NUM_STATES = ROWS * COLS

GAMMA = 1.0
ALPHA = 0.5
EPSILON = 0.1

DEFAULT_LAMBDA = 0.8

NUM_EPISODES = 300
MAX_STEPS = 1000

LAMBDA_VALUES = [0.0, 0.3, 0.6, 0.9]

EXPERIMENT_RUNS = 5
EXPERIMENT_EPISODES = 150


# ================================================================
# 2. STATE REPRESENTATION
# ================================================================

def state_to_index(state):
    """Convert (row, col) into a single state index."""

    if not isinstance(state, tuple) or len(state) != 2:
        raise ValueError("State must be a (row, col) tuple.")

    row, col = state

    if not (0 <= row < ROWS):
        raise ValueError("Invalid row.")

    if not (0 <= col < COLS):
        raise ValueError("Invalid column.")

    return row * COLS + col


def index_to_state(index):
    """Convert a state index into (row, col)."""

    if not isinstance(index, (int, np.integer)):
        raise ValueError("Index must be an integer.")

    if not 0 <= index < NUM_STATES:
        raise ValueError("Invalid state index.")

    return int(index // COLS), int(index % COLS)


# ================================================================
# 3. CLIFFWALKING ENVIRONMENT
# ================================================================

class CliffWalking:
    """
    Standard 4 x 12 CliffWalking.

    Start: (3,0)
    Goal:  (3,11)
    Cliff: (3,1) through (3,10)

    Ordinary movement: -1
    Cliff: -100 and reset to start
    Goal: -1 and terminate

    Entering the cliff does NOT terminate the episode.
    """

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.state = START_STATE
        self.steps = 0

    def reset(self):
        self.state = START_STATE
        self.steps = 0
        return self.state

    def is_terminal(self, state):
        return state == GOAL_STATE

    def step(self, action):

        if self.is_terminal(self.state):
            raise RuntimeError(
                "Cannot step after reaching the goal."
            )

        if action not in ACTION_DELTAS:
            raise ValueError("Invalid action.")

        row, col = self.state
        dr, dc = ACTION_DELTAS[action]

        next_row = int(
            np.clip(row + dr, 0, ROWS - 1)
        )

        next_col = int(
            np.clip(col + dc, 0, COLS - 1)
        )

        next_state = (next_row, next_col)

        self.steps += 1

        if next_state in CLIFF_CELLS:
            self.state = START_STATE

            return (
                START_STATE,
                -100.0,
                False,
                {"cliff": True, "goal": False},
            )

        if next_state == GOAL_STATE:
            self.state = GOAL_STATE

            return (
                GOAL_STATE,
                -1.0,
                True,
                {"cliff": False, "goal": True},
            )

        self.state = next_state

        return (
            next_state,
            -1.0,
            False,
            {"cliff": False, "goal": False},
        )


# ================================================================
# 4. EPSILON-GREEDY POLICY
# ================================================================

def epsilon_greedy_probabilities(q_values, epsilon):

    q_values = np.asarray(q_values, dtype=float)

    if q_values.ndim != 1:
        raise ValueError("q_values must be one-dimensional.")

    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must be in [0, 1].")

    num_actions = len(q_values)

    if num_actions == 0:
        raise ValueError("At least one action is required.")

    probabilities = np.full(
        num_actions,
        epsilon / num_actions,
        dtype=float,
    )

    maximum = np.max(q_values)

    greedy_actions = np.flatnonzero(
        np.isclose(q_values, maximum)
    )

    probabilities[greedy_actions] += (
        (1.0 - epsilon) / len(greedy_actions)
    )

    return probabilities


def select_action(q_values, epsilon, rng):

    probabilities = epsilon_greedy_probabilities(
        q_values,
        epsilon,
    )

    return int(
        rng.choice(
            len(q_values),
            p=probabilities,
        )
    )


def greedy_action(q_values):
    return int(np.argmax(q_values))


def is_greedy_action(q_values, action):

    maximum = np.max(q_values)

    return bool(
        np.isclose(q_values[action], maximum)
    )


# ================================================================
# 5. ELIGIBILITY TRACE UTILITIES
# ================================================================

def update_eligibility_trace(
    eligibility,
    state_index,
    action,
    trace_type="replacing",
):
    """
    Accumulating trace:
        E(s,a) <- E(s,a) + 1

    Replacing trace:
        E(s,a) <- 1
    """

    if eligibility.ndim != 2:
        raise ValueError("Eligibility must be a 2D array.")

    if not 0 <= state_index < eligibility.shape[0]:
        raise ValueError("Invalid state index.")

    if not 0 <= action < eligibility.shape[1]:
        raise ValueError("Invalid action.")

    if trace_type == "accumulating":
        eligibility[state_index, action] += 1.0

    elif trace_type == "replacing":
        eligibility[state_index, action] = 1.0

    else:
        raise ValueError(
            "trace_type must be 'accumulating' or 'replacing'."
        )

    return eligibility


def decay_eligibility_traces(
    eligibility,
    gamma,
    lambda_value,
):
    """E <- gamma * lambda * E"""

    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1].")

    if not 0.0 <= lambda_value <= 1.0:
        raise ValueError("lambda must be in [0, 1].")

    eligibility *= gamma * lambda_value

    return eligibility


# ================================================================
# 6. SARSA(lambda)
# ================================================================

def sarsa_lambda_update(
    q_values,
    eligibility,
    state_index,
    action,
    reward,
    next_state_index,
    next_action,
    terminated,
    alpha=ALPHA,
    gamma=GAMMA,
    lambda_value=DEFAULT_LAMBDA,
    trace_type="replacing",
):
    """
    Backward-view SARSA(lambda).

    TD target:

        terminal:
            target = reward

        nonterminal:
            target = reward + gamma * Q(S',A')

    Update:

        delta = target - Q(S,A)

        E(S,A) = 1

        Q <- Q + alpha * delta * E

        E <- gamma * lambda * E
    """

    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1].")

    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1].")

    if not 0.0 <= lambda_value <= 1.0:
        raise ValueError("lambda must be in [0, 1].")

    current_q = q_values[state_index, action]

    if terminated:
        td_target = reward

    else:
        if next_action is None:
            raise ValueError(
                "A next action is required for nonterminal SARSA."
            )

        td_target = (
            reward
            + gamma
            * q_values[next_state_index, next_action]
        )

    td_error = td_target - current_q

    update_eligibility_trace(
        eligibility,
        state_index,
        action,
        trace_type,
    )

    q_values += (
        alpha
        * td_error
        * eligibility
    )

    decay_eligibility_traces(
        eligibility,
        gamma,
        lambda_value,
    )

    if terminated:
        eligibility.fill(0.0)

    return float(td_error)


# ================================================================
# 7. WATKINS'S Q(lambda)
# ================================================================

def watkins_q_lambda_update(
    q_values,
    eligibility,
    state_index,
    action,
    reward,
    next_state_index,
    next_action,
    terminated,
    alpha=ALPHA,
    gamma=GAMMA,
    lambda_value=DEFAULT_LAMBDA,
):
    """
    Watkins's Q(lambda).

    TD target:

        terminal:
            target = reward

        nonterminal:
            target = reward + gamma * max_a Q(S',a)

    Update:

        delta = target - Q(S,A)

        E(S,A) = 1

        Q <- Q + alpha * delta * E

    Trace handling:

        If the next behavior action is non-greedy,
        clear all eligibility traces.
    """

    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1].")

    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1].")

    if not 0.0 <= lambda_value <= 1.0:
        raise ValueError("lambda must be in [0, 1].")

    current_q = q_values[state_index, action]

    if terminated:
        td_target = reward

    else:
        td_target = (
            reward
            + gamma * np.max(q_values[next_state_index])
        )

    td_error = td_target - current_q

    # Replacing trace.
    eligibility[state_index, action] = 1.0

    # Update all eligible state-action pairs.
    q_values += (
        alpha
        * td_error
        * eligibility
    )

    # Decay traces.
    decay_eligibility_traces(
        eligibility,
        gamma,
        lambda_value,
    )

    if terminated:
        eligibility.fill(0.0)

    else:
        # Watkins trace rule:
        # Clear traces if the selected next action is non-greedy.
        if not is_greedy_action(
            q_values[next_state_index],
            next_action,
        ):
            eligibility.fill(0.0)

    return float(td_error)


# ================================================================
# 8. EPISODE RESULT
# ================================================================

@dataclass
class EpisodeResult:
    total_reward: float
    steps: int
    cliff_falls: int
    terminated: bool


# ================================================================
# 9. TRAINING
# ================================================================

def train_agent(
    algorithm="sarsa_lambda",
    lambda_value=DEFAULT_LAMBDA,
    num_episodes=NUM_EPISODES,
    alpha=ALPHA,
    gamma=GAMMA,
    epsilon=EPSILON,
    seed=SEED,
    max_steps=MAX_STEPS,
):
    """
    Train SARSA(lambda) or Watkins's Q(lambda).

    At an external time-limit truncation, the last valid
    next state/action is used for bootstrapping.

    At a true terminal goal transition, no bootstrapping
    is performed.
    """

    if algorithm not in (
        "sarsa_lambda",
        "watkins_q_lambda",
    ):
        raise ValueError("Unknown algorithm.")

    if num_episodes <= 0 or max_steps <= 0:
        raise ValueError("Episode and step counts must be positive.")

    env = CliffWalking(seed=seed)

    rng = np.random.default_rng(seed + 1)

    q_values = np.zeros(
        (NUM_STATES, NUM_ACTIONS),
        dtype=float,
    )

    episode_rewards = []
    episode_lengths = []
    cliff_falls_history = []
    td_error_history = []

    for episode in range(num_episodes):

        state = env.reset()

        state_index = state_to_index(state)

        eligibility = np.zeros_like(q_values)

        action = select_action(
            q_values[state_index],
            epsilon,
            rng,
        )

        total_reward = 0.0
        cliff_falls = 0
        terminated = False
        last_td_error = 0.0

        for step_number in range(max_steps):

            (
                next_state,
                reward,
                terminated,
                info,
            ) = env.step(action)

            next_state_index = state_to_index(
                next_state
            )

            total_reward += reward

            if info["cliff"]:
                cliff_falls += 1

            if terminated:
                next_action = None

            else:
                next_action = select_action(
                    q_values[next_state_index],
                    epsilon,
                    rng,
                )

            if algorithm == "sarsa_lambda":

                last_td_error = sarsa_lambda_update(
                    q_values,
                    eligibility,
                    state_index,
                    action,
                    reward,
                    next_state_index,
                    next_action,
                    terminated,
                    alpha,
                    gamma,
                    lambda_value,
                    "replacing",
                )

            else:

                last_td_error = watkins_q_lambda_update(
                    q_values,
                    eligibility,
                    state_index,
                    action,
                    reward,
                    next_state_index,
                    next_action,
                    terminated,
                    alpha,
                    gamma,
                    lambda_value,
                )

            if terminated:
                break

            # Truncation:
            # The final valid next state/action was used
            # in the update above. Now discard episode traces.
            if step_number == max_steps - 1:
                eligibility.fill(0.0)
                break

            state_index = next_state_index
            action = next_action

        episode_rewards.append(total_reward)
        episode_lengths.append(step_number + 1)
        cliff_falls_history.append(cliff_falls)
        td_error_history.append(last_td_error)

    return {
        "algorithm": algorithm,
        "lambda": lambda_value,
        "q_values": q_values,
        "episode_rewards": np.asarray(
            episode_rewards,
            dtype=float,
        ),
        "episode_lengths": np.asarray(
            episode_lengths,
            dtype=int,
        ),
        "cliff_falls": np.asarray(
            cliff_falls_history,
            dtype=int,
        ),
        "td_errors": np.asarray(
            td_error_history,
            dtype=float,
        ),
    }


# ================================================================
# 10. GREEDY POLICY EVALUATION
# ================================================================

def evaluate_greedy_policy(
    q_values,
    num_episodes=20,
    seed=SEED + 100,
    max_steps=MAX_STEPS,
):

    env = CliffWalking(seed=seed)

    rewards = []
    lengths = []
    cliff_falls = []

    for _ in range(num_episodes):

        state = env.reset()

        total_reward = 0.0
        falls = 0
        terminated = False

        for step_number in range(max_steps):

            state_index = state_to_index(state)

            action = greedy_action(
                q_values[state_index]
            )

            (
                next_state,
                reward,
                terminated,
                info,
            ) = env.step(action)

            total_reward += reward

            if info["cliff"]:
                falls += 1

            state = next_state

            if terminated:
                break

        rewards.append(total_reward)
        lengths.append(step_number + 1)
        cliff_falls.append(falls)

    return {
        "mean_reward": float(np.mean(rewards)),
        "mean_length": float(np.mean(lengths)),
        "mean_cliff_falls": float(np.mean(cliff_falls)),
        "rewards": np.asarray(rewards),
        "lengths": np.asarray(lengths),
        "cliff_falls": np.asarray(cliff_falls),
    }


# ================================================================
# 11. POLICY AND PATH EXTRACTION
# ================================================================

def extract_greedy_policy(q_values):

    policy = np.zeros(
        (ROWS, COLS),
        dtype=int,
    )

    for row in range(ROWS):
        for col in range(COLS):

            state = (row, col)

            if state in CLIFF_CELLS or state == GOAL_STATE:
                continue

            index = state_to_index(state)

            policy[row, col] = greedy_action(
                q_values[index]
            )

    return policy


def extract_greedy_path(q_values, max_steps=100):

    env = CliffWalking(seed=SEED)

    state = env.reset()

    path = [state]

    for _ in range(max_steps):

        index = state_to_index(state)

        action = greedy_action(
            q_values[index]
        )

        (
            next_state,
            reward,
            terminated,
            info,
        ) = env.step(action)

        path.append(next_state)

        state = next_state

        if terminated:
            return path, True

    return path, False


# ================================================================
# 12. MOVING AVERAGE
# ================================================================

def moving_average(values, window=20):

    values = np.asarray(
        values,
        dtype=float,
    )

    if len(values) == 0:
        return values.copy()

    window = max(
        1,
        min(window, len(values)),
    )

    kernel = np.ones(window) / window

    return np.convolve(
        values,
        kernel,
        mode="valid",
    )


# ================================================================
# 13. LAMBDA EXPERIMENT
# ================================================================

def run_lambda_experiment(
    lambda_values=LAMBDA_VALUES,
    num_runs=EXPERIMENT_RUNS,
    num_episodes=EXPERIMENT_EPISODES,
    algorithm="sarsa_lambda",
):

    results = {}

    for lambda_value in lambda_values:

        run_rewards = []
        run_lengths = []
        run_falls = []

        for run_index in range(num_runs):

            result = train_agent(
                algorithm=algorithm,
                lambda_value=lambda_value,
                num_episodes=num_episodes,
                seed=SEED + run_index * 1000,
            )

            run_rewards.append(
                result["episode_rewards"]
            )

            run_lengths.append(
                result["episode_lengths"]
            )

            run_falls.append(
                result["cliff_falls"]
            )

        rewards = np.asarray(run_rewards, dtype=float)
        lengths = np.asarray(run_lengths, dtype=float)
        falls = np.asarray(run_falls, dtype=float)

        results[lambda_value] = {
            "reward_mean": np.mean(rewards, axis=0),
            "reward_std": np.std(rewards, axis=0),
            "length_mean": np.mean(lengths, axis=0),
            "falls_mean": np.mean(falls, axis=0),
            "final_reward_mean": float(
                np.mean(rewards[:, -20:])
            ),
        }

    return results


# ================================================================
# 14. VISUALIZATION — ENVIRONMENT
# ================================================================

def visualize_environment():

    grid = np.zeros((ROWS, COLS), dtype=float)

    for row, col in CLIFF_CELLS:
        grid[row, col] = -1.0

    grid[START_STATE] = 0.5
    grid[GOAL_STATE] = 1.0

    fig, ax = plt.subplots(figsize=(12, 4))

    ax.imshow(
        grid,
        cmap="RdYlGn",
        vmin=-1,
        vmax=1,
    )

    ax.set_xticks(np.arange(COLS))
    ax.set_yticks(np.arange(ROWS))

    ax.set_xticks(
        np.arange(-0.5, COLS, 1),
        minor=True,
    )

    ax.set_yticks(
        np.arange(-0.5, ROWS, 1),
        minor=True,
    )

    ax.grid(which="minor", linewidth=1)

    ax.text(
        START_STATE[1],
        START_STATE[0],
        "START",
        ha="center",
        va="center",
        fontsize=8,
    )

    ax.text(
        GOAL_STATE[1],
        GOAL_STATE[0],
        "GOAL",
        ha="center",
        va="center",
        fontsize=8,
    )

    ax.set_title("CliffWalking Environment")

    fig.tight_layout()


# ================================================================
# 15. VISUALIZATION — LEARNING CURVE
# ================================================================

def visualize_learning_curve(training_result, window=20):

    rewards = training_result["episode_rewards"]

    smoothed = moving_average(rewards, window)

    episodes = np.arange(1, len(rewards) + 1)

    smoothed_episodes = np.arange(
        window,
        len(rewards) + 1,
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        episodes,
        rewards,
        alpha=0.25,
        label="Episode reward",
    )

    ax.plot(
        smoothed_episodes,
        smoothed,
        linewidth=2,
        label=f"Moving average ({window})",
    )

    ax.set_title(
        f"{training_result['algorithm']}, "
        f"lambda={training_result['lambda']}"
    )

    ax.set_xlabel("Episode")
    ax.set_ylabel("Total reward")

    ax.legend()
    ax.grid(True)

    fig.tight_layout()


# ================================================================
# 16. VISUALIZATION — LAMBDA COMPARISON
# ================================================================

def visualize_lambda_comparison(results):

    fig, ax = plt.subplots(figsize=(10, 5))

    for lambda_value, data in results.items():

        mean = data["reward_mean"]
        std = data["reward_std"]

        smooth_mean = moving_average(mean, 20)
        smooth_std = moving_average(std, 20)

        episodes = np.arange(
            20,
            len(mean) + 1,
        )

        ax.plot(
            episodes,
            smooth_mean,
            label=f"lambda={lambda_value}",
        )

        ax.fill_between(
            episodes,
            smooth_mean - smooth_std,
            smooth_mean + smooth_std,
            alpha=0.12,
        )

    ax.set_title("SARSA(lambda): Effect of Lambda")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Mean episode reward")

    ax.legend()
    ax.grid(True)

    fig.tight_layout()


# ================================================================
# 17. VISUALIZATION — EPISODE LENGTH
# ================================================================

def visualize_episode_lengths(training_result):

    lengths = training_result["episode_lengths"]

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        lengths,
        alpha=0.35,
        label="Episode length",
    )

    ax.plot(
        moving_average(lengths, 20),
        linewidth=2,
        label="Moving average",
    )

    ax.set_title("Episode Length During Training")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Steps")

    ax.legend()
    ax.grid(True)

    fig.tight_layout()


# ================================================================
# 18. VISUALIZATION — CLIFF FALLS
# ================================================================

def visualize_cliff_falls(training_result):

    falls = training_result["cliff_falls"]

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        falls,
        alpha=0.4,
        label="Cliff falls per episode",
    )

    ax.plot(
        moving_average(falls, 20),
        linewidth=2,
        label="Moving average",
    )

    ax.set_title("Cliff Falls During Training")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Number of cliff falls")

    ax.legend()
    ax.grid(True)

    fig.tight_layout()


# ================================================================
# 19. VISUALIZATION — GREEDY POLICY
# ================================================================

def visualize_policy(q_values):

    policy = extract_greedy_policy(q_values)

    fig, ax = plt.subplots(figsize=(12, 4))

    ax.set_xlim(-0.5, COLS - 0.5)
    ax.set_ylim(ROWS - 0.5, -0.5)

    ax.set_xticks(np.arange(COLS))
    ax.set_yticks(np.arange(ROWS))

    ax.grid(True)

    arrow_symbols = {
        0: "↑",
        1: "→",
        2: "↓",
        3: "←",
    }

    for row in range(ROWS):
        for col in range(COLS):

            state = (row, col)

            if state in CLIFF_CELLS:
                symbol = "X"

            elif state == START_STATE:
                symbol = "S"

            elif state == GOAL_STATE:
                symbol = "G"

            else:
                symbol = arrow_symbols[
                    policy[row, col]
                ]

            ax.text(
                col,
                row,
                symbol,
                ha="center",
                va="center",
                fontsize=12,
            )

    ax.set_title("Learned Greedy Policy")

    fig.tight_layout()


# ================================================================
# 20. VISUALIZATION — Q VALUE HEATMAP
# ================================================================

def visualize_q_values(q_values):

    state_values = np.max(
        q_values,
        axis=1,
    ).reshape(ROWS, COLS)

    fig, ax = plt.subplots(figsize=(12, 4))

    image = ax.imshow(
        state_values,
        cmap="viridis",
    )

    fig.colorbar(
        image,
        ax=ax,
        label="Maximum Q-value",
    )

    ax.set_xticks(np.arange(COLS))
    ax.set_yticks(np.arange(ROWS))

    ax.set_title("State Value Estimate: max Q(s,a)")

    fig.tight_layout()


# ================================================================
# 21. VISUALIZATION — GREEDY PATH
# ================================================================

def visualize_greedy_path(q_values):

    path, reached_goal = extract_greedy_path(q_values)

    fig, ax = plt.subplots(figsize=(12, 4))

    ax.set_xlim(-0.5, COLS - 0.5)
    ax.set_ylim(ROWS - 0.5, -0.5)

    ax.set_xticks(np.arange(COLS))
    ax.set_yticks(np.arange(ROWS))

    ax.grid(True)

    for row, col in CLIFF_CELLS:
        ax.text(
            col,
            row,
            "X",
            ha="center",
            va="center",
        )

    path_rows = [state[0] for state in path]
    path_cols = [state[1] for state in path]

    ax.plot(
        path_cols,
        path_rows,
        marker="o",
        linewidth=2,
        label="Greedy path",
    )

    ax.scatter(
        START_STATE[1],
        START_STATE[0],
        marker="s",
        s=100,
        label="Start",
    )

    ax.scatter(
        GOAL_STATE[1],
        GOAL_STATE[0],
        marker="*",
        s=180,
        label="Goal",
    )

    ax.set_title(
        "Greedy Path — "
        + ("Goal reached" if reached_goal else "Step limit reached")
    )

    ax.legend()

    fig.tight_layout()


# ================================================================
# 22. VISUALIZATION — TRACE DECAY
# ================================================================

def visualize_trace_decay():

    gamma = 0.95

    lambda_values = [
        0.0,
        0.3,
        0.6,
        0.9,
        1.0,
    ]

    steps = np.arange(0, 21)

    fig, ax = plt.subplots(figsize=(9, 5))

    for lambda_value in lambda_values:

        decay = (
            gamma * lambda_value
        ) ** steps

        ax.plot(
            steps,
            decay,
            marker="o",
            markersize=3,
            label=f"lambda={lambda_value}",
        )

    ax.set_title("Eligibility Trace Decay")
    ax.set_xlabel("Steps since activation")
    ax.set_ylabel("Trace strength")

    ax.legend()
    ax.grid(True)

    fig.tight_layout()


# ================================================================
# 23. UNIT TESTS
# ================================================================

def run_tests():

    print("\n========== UNIT TESTS ==========")

    # Test 1: State-index conversion.
    for row in range(ROWS):
        for col in range(COLS):

            state = (row, col)

            index = state_to_index(state)

            assert index_to_state(index) == state

    print("[PASS] State-index conversion")

    # Test 2: Environment reset.
    env = CliffWalking(seed=SEED)

    assert env.reset() == START_STATE

    print("[PASS] Environment reset")

    # Test 3: Safe rightward movement.
    #
    # (2,0) --RIGHT--> (2,1)
    #
    # The bottom row is the cliff row, so use row 2.

    env.reset()
    env.state = (2, 0)

    next_state, reward, terminated, info = env.step(1)

    assert next_state == (2, 1)
    assert reward == -1.0
    assert not terminated
    assert not info["cliff"]

    print("[PASS] Safe rightward transition")

    # Test 4: Cliff penalty and reset.
    #
    # (3,0) --RIGHT--> cliff --> reset to (3,0)

    env.reset()

    next_state, reward, terminated, info = env.step(1)

    assert next_state == START_STATE
    assert reward == -100.0
    assert not terminated
    assert info["cliff"]

    print("[PASS] Cliff penalty and reset")

    # Test 5: Goal transition.
    env.reset()
    env.state = (2, 11)

    next_state, reward, terminated, info = env.step(2)

    assert next_state == GOAL_STATE
    assert reward == -1.0
    assert terminated
    assert info["goal"]

    print("[PASS] Goal transition")

    # Test 6: Terminal state protection.
    try:
        env.step(0)
        raise AssertionError("Expected RuntimeError.")
    except RuntimeError:
        pass

    print("[PASS] Terminal state protection")

    # Test 7: Invalid action.
    env.reset()

    try:
        env.step(99)
        raise AssertionError("Expected ValueError.")
    except ValueError:
        pass

    print("[PASS] Invalid action rejection")

    # Test 8: Epsilon-greedy probabilities.
    probabilities = epsilon_greedy_probabilities(
        np.array([1.0, 2.0, 3.0, 0.0]),
        epsilon=0.2,
    )

    assert np.isclose(np.sum(probabilities), 1.0)
    assert np.isclose(probabilities[2], 0.85)
    assert np.isclose(probabilities[0], 0.05)

    print("[PASS] Epsilon-greedy probabilities")

    # Test 9: Tie-aware greedy probabilities.
    #
    # Four actions.
    # Two greedy actions.
    # epsilon = 0.2
    #
    # Exploration contribution per action:
    #     0.2 / 4 = 0.05
    #
    # Exploitation contribution per greedy action:
    #     0.8 / 2 = 0.40
    #
    # Therefore:
    #     [0.05, 0.45, 0.45, 0.05]
    #
    # IMPORTANT:
    # The correct greedy probability is 0.45, NOT 0.425.

    probabilities = epsilon_greedy_probabilities(
        np.array([1.0, 3.0, 3.0, 0.0]),
        epsilon=0.2,
    )

    assert np.isclose(probabilities[0], 0.05)
    assert np.isclose(probabilities[1], 0.45)
    assert np.isclose(probabilities[2], 0.45)
    assert np.isclose(probabilities[3], 0.05)
    assert np.isclose(np.sum(probabilities), 1.0)

    print("[PASS] Tie-aware greedy probabilities")

    # Test 10: Accumulating eligibility trace.
    eligibility = np.zeros((NUM_STATES, NUM_ACTIONS))

    eligibility[5, 1] = 0.5

    update_eligibility_trace(
        eligibility,
        5,
        1,
        "accumulating",
    )

    assert np.isclose(eligibility[5, 1], 1.5)

    print("[PASS] Accumulating eligibility trace")

    # Test 11: Replacing eligibility trace.
    eligibility = np.zeros((NUM_STATES, NUM_ACTIONS))

    eligibility[5, 1] = 0.5

    update_eligibility_trace(
        eligibility,
        5,
        1,
        "replacing",
    )

    assert np.isclose(eligibility[5, 1], 1.0)

    print("[PASS] Replacing eligibility trace")

    # Test 12: Eligibility trace decay.
    eligibility = np.ones((NUM_STATES, NUM_ACTIONS))

    decay_eligibility_traces(
        eligibility,
        gamma=0.9,
        lambda_value=0.8,
    )

    assert np.allclose(eligibility, 0.72)

    print("[PASS] Eligibility trace decay")

    # Test 13: SARSA(lambda) numerical update.
    #
    # Current Q = 0.2
    # Next Q = 0.4
    # Reward = 1
    # gamma = 0.9
    #
    # Target = 1 + 0.9*0.4 = 1.36
    # TD error = 1.36 - 0.2 = 1.16
    #
    # Current trace = 1
    # Previous trace = 0.5
    # alpha = 0.1
    #
    # Current Q:
    # 0.2 + 0.1*1.16 = 0.316
    #
    # Previous Q:
    # 0 + 0.1*1.16*0.5 = 0.058

    q_values = np.zeros((NUM_STATES, NUM_ACTIONS))
    eligibility = np.zeros_like(q_values)

    q_values[4, 0] = 0.2
    q_values[5, 1] = 0.4

    eligibility[3, 2] = 0.5

    td_error = sarsa_lambda_update(
        q_values=q_values,
        eligibility=eligibility,
        state_index=4,
        action=0,
        reward=1.0,
        next_state_index=5,
        next_action=1,
        terminated=False,
        alpha=0.1,
        gamma=0.9,
        lambda_value=0.8,
    )

    assert np.isclose(td_error, 1.16)
    assert np.isclose(q_values[4, 0], 0.316)
    assert np.isclose(q_values[3, 2], 0.058)

    print("[PASS] SARSA(lambda) numerical update")

    # Test 14: SARSA terminal update does not bootstrap.
    #
    # Q = 0.5
    # Reward = -1
    # Target = -1
    # TD error = -1.5
    # New Q = 0.5 + 0.1*(-1.5) = 0.35

    q_values = np.zeros((NUM_STATES, NUM_ACTIONS))
    eligibility = np.zeros_like(q_values)

    q_values[10, 2] = 0.5

    td_error = sarsa_lambda_update(
        q_values=q_values,
        eligibility=eligibility,
        state_index=10,
        action=2,
        reward=-1.0,
        next_state_index=11,
        next_action=None,
        terminated=True,
        alpha=0.1,
        gamma=0.9,
        lambda_value=0.8,
    )

    assert np.isclose(td_error, -1.5)
    assert np.isclose(q_values[10, 2], 0.35)
    assert np.allclose(eligibility, 0.0)

    print("[PASS] SARSA terminal update")

    # Test 15: Watkins retains traces after greedy action.
    q_values = np.zeros((NUM_STATES, NUM_ACTIONS))
    eligibility = np.zeros_like(q_values)

    q_values[5, 1] = 1.0
    eligibility[3, 2] = 0.5

    watkins_q_lambda_update(
        q_values=q_values,
        eligibility=eligibility,
        state_index=4,
        action=0,
        reward=0.0,
        next_state_index=5,
        next_action=1,
        terminated=False,
        alpha=0.1,
        gamma=0.9,
        lambda_value=0.8,
    )

    assert eligibility[3, 2] > 0.0

    print("[PASS] Watkins greedy trace retention")

    # Test 16: Watkins clears traces after non-greedy action.
    q_values = np.zeros((NUM_STATES, NUM_ACTIONS))
    eligibility = np.zeros_like(q_values)

    q_values[5, 1] = 1.0
    eligibility[3, 2] = 0.5

    watkins_q_lambda_update(
        q_values=q_values,
        eligibility=eligibility,
        state_index=4,
        action=0,
        reward=0.0,
        next_state_index=5,
        next_action=0,
        terminated=False,
        alpha=0.1,
        gamma=0.9,
        lambda_value=0.8,
    )

    assert np.allclose(eligibility, 0.0)

    print("[PASS] Watkins non-greedy trace clearing")

    # Test 17: Watkins terminal update clears traces.
    q_values = np.zeros((NUM_STATES, NUM_ACTIONS))
    eligibility = np.ones_like(q_values)

    watkins_q_lambda_update(
        q_values=q_values,
        eligibility=eligibility,
        state_index=10,
        action=0,
        reward=-1.0,
        next_state_index=NUM_STATES - 1,
        next_action=None,
        terminated=True,
        alpha=0.1,
        gamma=0.9,
        lambda_value=0.8,
    )

    assert np.allclose(eligibility, 0.0)

    print("[PASS] Watkins terminal trace clearing")

    # Test 18: Lambda zero removes previous traces.
    eligibility = np.ones((NUM_STATES, NUM_ACTIONS))

    decay_eligibility_traces(
        eligibility,
        gamma=0.9,
        lambda_value=0.0,
    )

    assert np.allclose(eligibility, 0.0)

    print("[PASS] Lambda-zero trace behavior")

    # Test 19: SARSA(lambda) training.
    result = train_agent(
        algorithm="sarsa_lambda",
        lambda_value=0.8,
        num_episodes=20,
        seed=SEED,
        max_steps=500,
    )

    assert np.all(np.isfinite(result["q_values"]))
    assert len(result["episode_rewards"]) == 20

    print("[PASS] SARSA(lambda) training")

    # Test 20: Watkins Q(lambda) training.
    result = train_agent(
        algorithm="watkins_q_lambda",
        lambda_value=0.8,
        num_episodes=20,
        seed=SEED + 1,
        max_steps=500,
    )

    assert np.all(np.isfinite(result["q_values"]))
    assert len(result["episode_rewards"]) == 20

    print("[PASS] Watkins Q(lambda) training")

    # Test 21: Greedy policy dimensions.
    policy = extract_greedy_policy(result["q_values"])

    assert policy.shape == (ROWS, COLS)

    print("[PASS] Greedy policy dimensions")

    # Test 22: Greedy evaluation.
    evaluation = evaluate_greedy_policy(
        result["q_values"],
        num_episodes=3,
        max_steps=100,
    )

    assert np.isfinite(evaluation["mean_reward"])

    print("[PASS] Greedy policy evaluation")

    print("\nAll 22 tests passed.")


# ================================================================
# 24. MAIN
# ================================================================

def main():

    print("\n==============================================")
    print("RL COURSE — PHASE 3.9")
    print("ELIGIBILITY TRACES")
    print("==============================================")

    # STEP 1: Run tests before training.
    run_tests()

    # STEP 2: Show environment.
    visualize_environment()

    # STEP 3: Train SARSA(lambda).
    print("\n========== SARSA(lambda) TRAINING ==========")

    sarsa_result = train_agent(
        algorithm="sarsa_lambda",
        lambda_value=DEFAULT_LAMBDA,
        num_episodes=NUM_EPISODES,
        seed=SEED,
    )

    print(
        "Final 20-episode average reward:",
        f"{np.mean(sarsa_result['episode_rewards'][-20:]):.3f}",
    )

    print(
        "Final 20-episode average length:",
        f"{np.mean(sarsa_result['episode_lengths'][-20:]):.1f}",
    )

    print(
        "Total cliff falls:",
        int(np.sum(sarsa_result["cliff_falls"])),
    )

    evaluation = evaluate_greedy_policy(
        sarsa_result["q_values"],
        num_episodes=20,
    )

    print(
        "Greedy evaluation average reward:",
        f"{evaluation['mean_reward']:.3f}",
    )

    print(
        "Greedy evaluation average steps:",
        f"{evaluation['mean_length']:.1f}",
    )

    # STEP 4: Train Watkins Q(lambda).
    print("\n========== WATKINS Q(lambda) TRAINING ==========")

    q_lambda_result = train_agent(
        algorithm="watkins_q_lambda",
        lambda_value=DEFAULT_LAMBDA,
        num_episodes=NUM_EPISODES,
        seed=SEED + 10,
    )

    print(
        "Final 20-episode average reward:",
        f"{np.mean(q_lambda_result['episode_rewards'][-20:]):.3f}",
    )

    q_lambda_evaluation = evaluate_greedy_policy(
        q_lambda_result["q_values"],
        num_episodes=20,
        seed=SEED + 200,
    )

    print(
        "Greedy evaluation average reward:",
        f"{q_lambda_evaluation['mean_reward']:.3f}",
    )

    # STEP 5: Compare lambda values.
    print("\n========== LAMBDA EXPERIMENT ==========")

    lambda_results = run_lambda_experiment(
        lambda_values=LAMBDA_VALUES,
        num_runs=EXPERIMENT_RUNS,
        num_episodes=EXPERIMENT_EPISODES,
        algorithm="sarsa_lambda",
    )

    for lambda_value, data in lambda_results.items():

        print(
            f"lambda={lambda_value:.1f} | "
            f"final average reward="
            f"{data['final_reward_mean']:.3f}"
        )

    # STEP 6: Generate visualizations.
    print("\n========== VISUALIZATIONS ==========")

    visualize_learning_curve(sarsa_result)

    visualize_episode_lengths(sarsa_result)

    visualize_cliff_falls(sarsa_result)

    visualize_policy(sarsa_result["q_values"])

    visualize_q_values(sarsa_result["q_values"])

    visualize_greedy_path(sarsa_result["q_values"])

    visualize_lambda_comparison(lambda_results)

    visualize_trace_decay()

    # STEP 7: Display all figures.
    plt.show()

    print("\n==============================================")
    print("PHASE 3.9 COMPLETED")
    print("==============================================")


if __name__ == "__main__":
    main()
