"""
============================================================
RL COURSE — PHASE 3.3
SARSA — ON-POLICY TEMPORAL-DIFFERENCE CONTROL
============================================================

PURPOSE
-------
Learn how an agent improves its behavior by learning action values
Q(s, a) directly from interaction, rather than only estimating V(s).

LESSON ROADMAP
--------------

STEP 1 — Understand action-value learning
Purpose:
    Move from TD(0) state-value prediction to control.
What you learn:
    Q(s,a) estimates the expected return after taking action a
    in state s and then following a policy.
Implementation:
    Implement tabular Q-values and epsilon-greedy action selection.

STEP 2 — Build the Cliff-Walking environment
Purpose:
    Create a navigation task where the shortest route is risky.
What you learn:
    The consequences of exploration, hazards, terminal states,
    and episodic rewards.
Implementation:
    Build a 4x12 grid with a start, goal, and cliff.

STEP 3 — Implement SARSA
Purpose:
    Learn action values using the agent's actual next action.
What you learn:
    SARSA is on-policy because its TD target uses A_(t+1)
    sampled from the current behavior policy.
Implementation:
    Q(S,A) <- Q(S,A) + alpha *
      [R + gamma * Q(S',A') - Q(S,A)]
    On a true terminal transition, the target is just R.

STEP 4 — Implement Q-learning as a comparison
Purpose:
    Contrast on-policy and off-policy TD control.
What you learn:
    Q-learning uses max_a Q(S',a), not the sampled next action.
Implementation:
    Q(S,A) <- Q(S,A) + alpha *
      [R + gamma * max_a Q(S',a) - Q(S,A)]

STEP 5 — Train and evaluate
Purpose:
    Compare learning behavior under the same environment.
What you learn:
    SARSA often learns a safer route because its target accounts
    for the exploratory actions it actually takes. Q-learning
    estimates the greedy target policy while behaving epsilon-greedily.
Implementation:
    Train both algorithms, then evaluate their greedy policies.

STEP 6 — Verify, visualize, and apply
Purpose:
    Check the update equations and inspect learned routes.
What you learn:
    Numerical tests, reward curves, policy maps, and route traces.
Implementation:
    Run unit tests and generate all plots from main().

CORE MATHEMATICS
----------------

SARSA transition tuple:
    (S_t, A_t, R_(t+1), S_(t+1), A_(t+1))

SARSA TD error:
    delta = R + gamma * Q(S', A') - Q(S, A)

SARSA update:
    Q(S,A) <- Q(S,A) + alpha * delta

Q-learning TD error:
    delta = R + gamma * max_a Q(S',a) - Q(S,A)

Q-learning update:
    Q(S,A) <- Q(S,A) + alpha * delta

For a true terminal transition:
    target = R
    No next-state bootstrap is performed.

For an external time-limit truncation with a valid next state:
    bootstrap from the next state. In this script, the episode
    step cap is treated as truncation, not true termination.

Epsilon-greedy policy:
    With probability epsilon, choose a random action.
    Otherwise, choose uniformly among the actions tied for max Q.

REAL-WORLD APPLICATION
----------------------

Warehouse robot navigation:
    State:
        Robot grid position, local obstacle status, and optionally
        battery / traffic context.
    Actions:
        Forward, backward, turn left, turn right, stop.
    Rewards:
        Small time/energy cost per move; delivery reward; hazard cost.
    SARSA:
        Learns action values while accounting for the exploratory
        behavior policy it actually executes.

IMPORTANT SAFETY NOTE:
    SARSA is not a safety guarantee. A real robot still needs
    collision checking, a safety controller, action masking,
    emergency stop logic, and validated operating constraints.
    Never use an unconstrained learned policy as the sole safety layer.

DEPENDENCIES
------------
    pip install numpy matplotlib

RUN
---
    python phase3/sarsa.py
============================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

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
CLIFF_CELLS = {(3, col) for col in range(1, 11)}

NUM_ACTIONS = 4
ACTION_NAMES = ("UP", "RIGHT", "DOWN", "LEFT")
ACTION_DELTAS = (
    (-1, 0),
    (0, 1),
    (1, 0),
    (0, -1),
)

GAMMA = 1.0
ALPHA = 0.5
INITIAL_EPSILON = 1.0
MIN_EPSILON = 0.05
EPSILON_DECAY = 0.995

NUM_EPISODES = 500
MAX_STEPS_PER_EPISODE = 500

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
    Classic deterministic cliff-walking task.

    Grid:
        S . . . . . . . . . . .
        . . . . . . . . . . . .
        . . . . . . . . . . . .
        S C C C C C C C C C C G

    The start is bottom-left, the goal bottom-right, and cells
    between them on the bottom row form the cliff.

    Falling into the cliff gives -100 and returns the agent to
    START. It does not terminate the episode.
    Reaching GOAL gives -1 and terminates the episode.
    Every ordinary movement gives -1.
    """

    def __init__(self):
        self.state = self.position_to_state(START)

    @staticmethod
    def position_to_state(position: Tuple[int, int]) -> int:
        row, col = position
        return row * COLS + col

    @staticmethod
    def state_to_position(state: int) -> Tuple[int, int]:
        return divmod(int(state), COLS)

    def reset(self) -> int:
        self.state = self.position_to_state(START)
        return self.state

    def step(self, action: int) -> StepResult:
        if not 0 <= int(action) < NUM_ACTIONS:
            raise ValueError(f"Invalid action: {action}")

        row, col = self.state_to_position(self.state)
        dr, dc = ACTION_DELTAS[int(action)]

        next_row = int(np.clip(row + dr, 0, ROWS - 1))
        next_col = int(np.clip(col + dc, 0, COLS - 1))
        next_position = (next_row, next_col)

        if next_position in CLIFF_CELLS:
            self.state = self.position_to_state(START)
            return StepResult(
                next_state=self.state,
                reward=-100.0,
                terminated=False,
                cliff_fall=True,
            )

        self.state = self.position_to_state(next_position)

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
    q_values: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    """
    Return epsilon-greedy action probabilities.

    Probability mass is distributed uniformly across tied greedy
    actions. The exploratory part is uniform over all actions.

    pi(a|s) = epsilon / |A| + (1-epsilon) / |A*|
    for greedy actions, and epsilon / |A| otherwise.
    """
    q_values = np.asarray(q_values, dtype=float)

    if q_values.ndim != 1 or q_values.size == 0:
        raise ValueError("q_values must be a non-empty 1D array.")
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must be in [0, 1].")

    greedy = np.isclose(q_values, np.max(q_values))
    probabilities = np.full(
        q_values.size,
        epsilon / q_values.size,
        dtype=float,
    )
    probabilities[greedy] += (
        (1.0 - epsilon) / np.count_nonzero(greedy)
    )
    return probabilities


def select_action(
    q_table: np.ndarray,
    state: int,
    epsilon: float,
    rng: np.random.Generator,
) -> int:
    probabilities = epsilon_greedy_probabilities(
        q_table[state],
        epsilon,
    )
    return int(rng.choice(NUM_ACTIONS, p=probabilities))


# ============================================================
# 4. SARSA AND Q-LEARNING UPDATES
# ============================================================


def sarsa_update(
    q_table: np.ndarray,
    state: int,
    action: int,
    reward: float,
    next_state: Optional[int],
    next_action: Optional[int],
    terminated: bool,
    alpha: float,
    gamma: float,
) -> float:
    """
    Perform one SARSA update.

    The next action is supplied by the caller and is sampled from
    the current epsilon-greedy behavior policy.

    True terminal:
        target = reward

    Nonterminal or externally truncated transition:
        target = reward + gamma * Q(next_state, next_action)
    """
    if terminated:
        target = float(reward)
    else:
        if next_state is None or next_action is None:
            raise ValueError(
                "Nonterminal SARSA update requires next_state and next_action."
            )
        target = (
            float(reward)
            + gamma * q_table[next_state, next_action]
        )

    td_error = target - q_table[state, action]
    q_table[state, action] += alpha * td_error
    return float(td_error)


def q_learning_update(
    q_table: np.ndarray,
    state: int,
    action: int,
    reward: float,
    next_state: Optional[int],
    terminated: bool,
    alpha: float,
    gamma: float,
) -> float:
    """
    Perform one Q-learning update.

    Q-learning is off-policy: the target uses max_a Q(s',a),
    regardless of which action the behavior policy might select.
    """
    if terminated:
        target = float(reward)
    else:
        if next_state is None:
            raise ValueError(
                "Nonterminal Q-learning update requires next_state."
            )
        target = (
            float(reward)
            + gamma * np.max(q_table[next_state])
        )

    td_error = target - q_table[state, action]
    q_table[state, action] += alpha * td_error
    return float(td_error)


# ============================================================
# 5. TRAINING
# ============================================================


@dataclass
class TrainingResult:
    q_table: np.ndarray
    episode_returns: np.ndarray
    episode_lengths: np.ndarray
    cliff_falls: np.ndarray
    epsilon_history: np.ndarray


def train_agent(
    algorithm: str,
    num_episodes: int = NUM_EPISODES,
    alpha: float = ALPHA,
    gamma: float = GAMMA,
    initial_epsilon: float = INITIAL_EPSILON,
    min_epsilon: float = MIN_EPSILON,
    epsilon_decay: float = EPSILON_DECAY,
    seed: int = SEED,
    max_steps: int = MAX_STEPS_PER_EPISODE,
) -> TrainingResult:
    """
    Train either SARSA or Q-learning.

    SARSA action sequence:
        S, A -> R, S' -> sample A' from epsilon-greedy(Q,S')
        -> update Q(S,A) using Q(S',A') -> continue with A'.

    Q-learning:
        Uses max_a Q(S',a) in its target, then independently
        selects the next behavior action.

    At an external max-step truncation, the final next observation
    is valid and the update bootstraps. A true goal termination
    does not bootstrap.
    """
    if algorithm not in {"sarsa", "q_learning"}:
        raise ValueError("algorithm must be 'sarsa' or 'q_learning'.")

    env = CliffWalking()
    rng = np.random.default_rng(seed)
    q_table = np.zeros((ROWS * COLS, NUM_ACTIONS), dtype=float)

    returns = np.zeros(num_episodes, dtype=float)
    lengths = np.zeros(num_episodes, dtype=int)
    falls = np.zeros(num_episodes, dtype=int)
    epsilons = np.zeros(num_episodes, dtype=float)

    epsilon = float(initial_epsilon)

    for episode in range(num_episodes):
        state = env.reset()
        action = select_action(q_table, state, epsilon, rng)

        total_reward = 0.0
        cliff_count = 0

        for step_index in range(max_steps):
            result = env.step(action)
            next_state = result.next_state
            reward = result.reward
            terminated = result.terminated

            total_reward += reward
            cliff_count += int(result.cliff_fall)

            # A cap reached without reaching the goal is truncation.
            truncated = (
                step_index == max_steps - 1 and not terminated
            )

            if terminated:
                if algorithm == "sarsa":
                    sarsa_update(
                        q_table, state, action, reward,
                        None, None, True, alpha, gamma
                    )
                else:
                    q_learning_update(
                        q_table, state, action, reward,
                        None, True, alpha, gamma
                    )
                lengths[episode] = step_index + 1
                break

            # For both ordinary transitions and external truncation,
            # the next state is valid and may be bootstrapped.
            next_action = select_action(
                q_table, next_state, epsilon, rng
            )

            if algorithm == "sarsa":
                sarsa_update(
                    q_table, state, action, reward,
                    next_state, next_action, False,
                    alpha, gamma
                )
            else:
                q_learning_update(
                    q_table, state, action, reward,
                    next_state, False, alpha, gamma
                )

            state = next_state
            action = next_action

            if truncated:
                lengths[episode] = max_steps
                break

        returns[episode] = total_reward
        falls[episode] = cliff_count
        epsilons[episode] = epsilon
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

    return TrainingResult(
        q_table=q_table,
        episode_returns=returns,
        episode_lengths=lengths,
        cliff_falls=falls,
        epsilon_history=epsilons,
    )


# ============================================================
# 6. POLICY EVALUATION AND ROUTE EXTRACTION
# ============================================================


def greedy_action(q_table: np.ndarray, state: int) -> int:
    """Deterministic greedy action; first action wins exact ties."""
    return int(np.argmax(q_table[state]))


def evaluate_greedy_policy(
    q_table: np.ndarray,
    episodes: int = 30,
    seed: int = SEED + 10000,
    max_steps: int = 500,
) -> Dict[str, float]:
    """
    Evaluate without exploration (epsilon=0).

    Reports average return, average episode length, success rate,
    and total cliff falls.
    """
    env = CliffWalking()
    rng = np.random.default_rng(seed)

    episode_returns = []
    episode_lengths = []
    successes = 0
    total_falls = 0

    for _ in range(episodes):
        state = env.reset()
        total_reward = 0.0

        for step in range(max_steps):
            action = select_action(q_table, state, 0.0, rng)
            result = env.step(action)
            total_reward += result.reward
            total_falls += int(result.cliff_fall)
            state = result.next_state

            if result.terminated:
                successes += 1
                episode_lengths.append(step + 1)
                break
        else:
            episode_lengths.append(max_steps)

        episode_returns.append(total_reward)

    return {
        "average_return": float(np.mean(episode_returns)),
        "average_length": float(np.mean(episode_lengths)),
        "success_rate": successes / episodes,
        "total_cliff_falls": float(total_falls),
    }


def extract_greedy_path(
    q_table: np.ndarray,
    max_steps: int = 100,
) -> Tuple[List[Tuple[int, int]], int]:
    """Trace the greedy policy from START; return path and cliff falls."""
    env = CliffWalking()
    state = env.reset()
    path = [START]
    falls = 0

    for _ in range(max_steps):
        action = greedy_action(q_table, state)
        result = env.step(action)
        falls += int(result.cliff_fall)
        state = result.next_state
        position = env.state_to_position(state)

        if not result.cliff_fall:
            path.append(position)

        if result.terminated:
            break

    return path, falls


# ============================================================
# 7. METRICS
# ============================================================


def moving_average(values: np.ndarray, window: int = 20) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) < window:
        return values.copy()
    return np.convolve(
        values,
        np.ones(window, dtype=float) / window,
        mode="valid",
    )


def state_value_view(q_table: np.ndarray) -> np.ndarray:
    """Convenient visualization: V_greedy(s) = max_a Q(s,a)."""
    return np.max(q_table, axis=1).reshape(ROWS, COLS)


# ============================================================
# 8. VISUALIZATIONS
# ============================================================


def draw_environment(ax):
    grid = np.zeros((ROWS, COLS), dtype=int)
    for row, col in CLIFF_CELLS:
        grid[row, col] = 1

    ax.imshow(grid, cmap="Greys", vmin=0, vmax=1)
    ax.set_xticks(np.arange(COLS))
    ax.set_yticks(np.arange(ROWS))
    ax.set_xticklabels(np.arange(COLS))
    ax.set_yticklabels(np.arange(ROWS))
    ax.set_xticks(np.arange(-0.5, COLS, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, ROWS, 1), minor=True)
    ax.grid(which="minor", linewidth=0.7)
    ax.tick_params(which="minor", bottom=False, left=False)

    ax.text(START[1], START[0], "S", ha="center", va="center",
            fontweight="bold")
    ax.text(GOAL[1], GOAL[0], "G", ha="center", va="center",
            fontweight="bold")
    for col in range(1, 11):
        ax.text(col, 3, "C", ha="center", va="center", fontsize=8)


def visualize_environment():
    fig, ax = plt.subplots(figsize=(12, 4))
    draw_environment(ax)
    ax.set_title("Cliff-Walking Environment (C = hazard)")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    fig.tight_layout()


def visualize_learning_curves(
    sarsa_result: TrainingResult,
    q_result: TrainingResult,
):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        moving_average(sarsa_result.episode_returns),
        label="SARSA: moving average return",
    )
    ax.plot(
        moving_average(q_result.episode_returns),
        label="Q-learning: moving average return",
    )
    ax.set_title("Training Returns: SARSA vs Q-learning")
    ax.set_xlabel("Episode (moving-average index)")
    ax.set_ylabel("Return")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()


def visualize_cliff_falls(
    sarsa_result: TrainingResult,
    q_result: TrainingResult,
):
    window = 20
    sarsa_falls = moving_average(
        sarsa_result.cliff_falls.astype(float), window
    )
    q_falls = moving_average(
        q_result.cliff_falls.astype(float), window
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(sarsa_falls, label="SARSA")
    ax.plot(q_falls, label="Q-learning")
    ax.set_title(f"Cliff Falls per Episode (moving average, {window})")
    ax.set_xlabel("Episode (moving-average index)")
    ax.set_ylabel("Cliff falls")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()


def visualize_policy(
    q_table: np.ndarray,
    title: str,
):
    arrows = ("↑", "→", "↓", "←")
    fig, ax = plt.subplots(figsize=(12, 4))
    draw_environment(ax)

    for row in range(ROWS):
        for col in range(COLS):
            if (row, col) in CLIFF_CELLS or (row, col) in {START, GOAL}:
                continue
            state = row * COLS + col
            action = greedy_action(q_table, state)
            ax.text(
                col, row, arrows[action],
                ha="center", va="center", fontsize=15,
            )

    ax.set_title(title)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    fig.tight_layout()


def visualize_value_heatmap(q_table: np.ndarray, title: str):
    values = state_value_view(q_table)
    fig, ax = plt.subplots(figsize=(12, 4))
    image = ax.imshow(values, cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_xticks(np.arange(COLS))
    ax.set_yticks(np.arange(ROWS))
    fig.colorbar(image, ax=ax, label="max_a Q(s,a)")
    fig.tight_layout()


def visualize_greedy_paths(
    sarsa_q: np.ndarray,
    q_learning_q: np.ndarray,
):
    sarsa_path, _ = extract_greedy_path(sarsa_q)
    q_path, _ = extract_greedy_path(q_learning_q)

    fig, axes = plt.subplots(1, 2, figsize=(15, 4))
    for ax, path, title in (
        (axes[0], sarsa_path, "SARSA greedy route"),
        (axes[1], q_path, "Q-learning greedy route"),
    ):
        draw_environment(ax)
        if path:
            rows = [p[0] for p in path]
            cols = [p[1] for p in path]
            ax.plot(cols, rows, marker="o", linewidth=2)
        ax.set_title(title)
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
    fig.tight_layout()


def visualize_epsilon(result: TrainingResult, title: str):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(result.epsilon_history)
    ax.set_title(title)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Epsilon")
    ax.grid(True)
    fig.tight_layout()


# ============================================================
# 9. UNIT TESTS
# ============================================================


def run_tests():
    print("\n========== UNIT TESTS ==========")

    # Test 1: reset state.
    env = CliffWalking()
    assert env.reset() == env.position_to_state(START)
    print("[PASS] Environment reset")

    # Test 2: ordinary movement and reward.
    env.reset()
    result = env.step(0)  # Up from start.
    assert env.state_to_position(result.next_state) == (2, 0)
    assert result.reward == -1.0
    assert not result.terminated
    print("[PASS] Ordinary transition")

    # Test 3: cliff penalty and reset.
    env.reset()
    result = env.step(1)  # Right from start enters cliff.
    assert result.cliff_fall
    assert result.reward == -100.0
    assert result.next_state == env.position_to_state(START)
    assert not result.terminated
    print("[PASS] Cliff penalty and reset")

    # Test 4: goal transition.
    env.state = env.position_to_state((3, 10))
    result = env.step(1)
    assert env.state_to_position(result.next_state) == GOAL
    assert result.reward == -1.0
    assert result.terminated
    print("[PASS] Goal termination")

    # Test 5: epsilon-greedy probabilities sum to one.
    probabilities = epsilon_greedy_probabilities(
        np.array([1.0, 2.0, 2.0, 0.0]), 0.2
    )
    assert np.isclose(np.sum(probabilities), 1.0)
    assert np.all(probabilities >= 0.0)
    print("[PASS] Epsilon-greedy probability normalization")

    # Test 6: tied greedy actions share exploitation mass.
    assert np.isclose(probabilities[1], probabilities[2])
    assert probabilities[1] > probabilities[0]
    print("[PASS] Tie-aware epsilon-greedy policy")

    # Test 7: SARSA exact numerical update.
    q = np.zeros((3, NUM_ACTIONS), dtype=float)
    q[1, 2] = 0.4
    q[2, 1] = 0.7
    error = sarsa_update(
        q, state=1, action=2, reward=0.0,
        next_state=2, next_action=1, terminated=False,
        alpha=0.1, gamma=1.0,
    )
    # target=0.7; delta=0.7-0.4=0.3; new Q=0.43
    assert np.isclose(error, 0.3)
    assert np.isclose(q[1, 2], 0.43)
    print("[PASS] SARSA numerical update")

    # Test 8: terminal SARSA update does not bootstrap.
    q = np.zeros((3, NUM_ACTIONS), dtype=float)
    q[1, 0] = 0.4
    q[2, :] = 1000.0
    error = sarsa_update(
        q, 1, 0, reward=2.0,
        next_state=None, next_action=None, terminated=True,
        alpha=0.1, gamma=0.99,
    )
    assert np.isclose(error, 1.6)
    assert np.isclose(q[1, 0], 0.56)
    print("[PASS] SARSA terminal no-bootstrap")

    # Test 9: Q-learning uses maximum next-state action value.
    q = np.zeros((3, NUM_ACTIONS), dtype=float)
    q[1, 0] = 0.4
    q[2] = np.array([0.1, 0.8, 0.3, 0.2])
    error = q_learning_update(
        q, 1, 0, reward=0.0, next_state=2,
        terminated=False, alpha=0.1, gamma=1.0,
    )
    assert np.isclose(error, 0.4)
    assert np.isclose(q[1, 0], 0.44)
    print("[PASS] Q-learning max target")

    # Test 10: SARSA requires a sampled next action when nonterminal.
    try:
        sarsa_update(
            np.zeros((3, NUM_ACTIONS)), 1, 0, 0.0,
            2, None, False, 0.1, 1.0,
        )
        raise AssertionError("Expected ValueError.")
    except ValueError:
        pass
    print("[PASS] SARSA next-action validation")

    # Test 11: invalid epsilon rejected.
    try:
        epsilon_greedy_probabilities(np.zeros(4), 1.5)
        raise AssertionError("Expected ValueError.")
    except ValueError:
        pass
    print("[PASS] Epsilon range validation")

    # Test 12: short training produces finite Q values.
    trained = train_agent(
        "sarsa", num_episodes=20, seed=SEED, max_steps=200
    )
    assert np.all(np.isfinite(trained.q_table))
    assert len(trained.episode_returns) == 20
    print("[PASS] SARSA training numerical stability")

    # Test 13: Q-learning training produces finite Q values.
    trained = train_agent(
        "q_learning", num_episodes=20, seed=SEED, max_steps=200
    )
    assert np.all(np.isfinite(trained.q_table))
    assert len(trained.episode_returns) == 20
    print("[PASS] Q-learning training numerical stability")

    # Test 14: state/action dimensions.
    assert trained.q_table.shape == (ROWS * COLS, NUM_ACTIONS)
    print("[PASS] Q-table dimensions")

    print("\nAll 14 tests passed.")


# ============================================================
# 10. MAIN
# ============================================================


def main():
    print("\n==============================================")
    print("PHASE 3.3 — SARSA: ON-POLICY TD CONTROL")
    print("==============================================")

    visualize_environment()

    print("\n[1] Training SARSA...")
    sarsa_result = train_agent(
        algorithm="sarsa",
        seed=SEED,
    )

    print("[2] Training Q-learning...")
    q_result = train_agent(
        algorithm="q_learning",
        seed=SEED + 1,
    )

    print("\n[3] Greedy policy evaluation (epsilon=0)")
    sarsa_eval = evaluate_greedy_policy(
        sarsa_result.q_table, seed=SEED + 100
    )
    q_eval = evaluate_greedy_policy(
        q_result.q_table, seed=SEED + 200
    )

    for name, metrics in (
        ("SARSA", sarsa_eval),
        ("Q-learning", q_eval),
    ):
        print(f"\n{name}:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.3f}")

    print("\n[4] Training summary")
    print(
        "SARSA final 20-episode mean return:",
        f"{np.mean(sarsa_result.episode_returns[-20:]):.3f}",
    )
    print(
        "Q-learning final 20-episode mean return:",
        f"{np.mean(q_result.episode_returns[-20:]):.3f}",
    )
    print(
        "SARSA total training cliff falls:",
        int(np.sum(sarsa_result.cliff_falls)),
    )
    print(
        "Q-learning total training cliff falls:",
        int(np.sum(q_result.cliff_falls)),
    )

    print("\n[5] Unit tests")
    run_tests()

    print("\n[6] Visualizations")
    visualize_learning_curves(sarsa_result, q_result)
    visualize_cliff_falls(sarsa_result, q_result)
    visualize_policy(sarsa_result.q_table, "SARSA Greedy Policy")
    visualize_policy(q_result.q_table, "Q-learning Greedy Policy")
    visualize_value_heatmap(
        sarsa_result.q_table, "SARSA State-Value View"
    )
    visualize_value_heatmap(
        q_result.q_table, "Q-learning State-Value View"
    )
    visualize_greedy_paths(
        sarsa_result.q_table, q_result.q_table
    )
    visualize_epsilon(
        sarsa_result, "SARSA Epsilon Decay"
    )

    plt.show()

    print("\n==============================================")
    print("PHASE 3.3 COMPLETED")
    print("==============================================")


if __name__ == "__main__":
    main()