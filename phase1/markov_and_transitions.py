
"""
===============================================================================
RL COURSE — PHASE 1.4
Markov Property, State Representation, and Transition Probabilities
===============================================================================

LESSON ROADMAP
--------------

STEP 1 — Understand the Markov Property
Purpose:
    Understand why the current state should contain enough information
    to predict the future.

What you learn:
    - Markov property.
    - Current state versus complete history.
    - Conditional independence of future and past given the present.

Implementation:
    - verify_markov_property()
    - demonstrate_markov_property()

STEP 2 — Represent the Environment State
Purpose:
    Understand how state design affects an RL agent.

What you learn:
    - Fully observable state.
    - Partial observation.
    - Hidden variables.
    - Position-only state versus position-and-velocity state.

Implementation:
    - MomentumEnvironment
    - simulate_momentum_environment()

STEP 3 — Understand State Transitions
Purpose:
    Represent environment dynamics mathematically.

What you learn:
    - Transition probability P(s'|s,a).
    - Deterministic transitions.
    - Stochastic transitions.
    - Transition probability matrix.
    - Probability normalization.

Implementation:
    - TransitionModel
    - transition_matrix()
    - validate_transition_model()

STEP 4 — Build a Stochastic GridWorld
Purpose:
    Apply transition probabilities to a familiar RL environment.

What you learn:
    - Action does not always produce the intended movement.
    - Slippery or noisy environments.
    - Boundary and obstacle handling.
    - Sampling the next state.

Implementation:
    - StochasticGridWorld
    - step()
    - get_transition_distribution()

STEP 5 — Numerical Verification
Purpose:
    Verify the transition model and Markov assumptions.

What you learn:
    - Transition probabilities sum to one.
    - Empirical probabilities approach theoretical probabilities.
    - A full state can be Markov even when a partial observation is not.

Implementation:
    - run_tests()

STEP 6 — Visualization and Experimentation
Purpose:
    See how state representation and stochastic transitions affect
    environment behavior.

What you learn:
    - Transition probability matrices.
    - Position-only versus full-state predictions.
    - Stochastic movement distributions.
    - Sample trajectories.

Implementation:
    - All visualization functions are included below.
    - main() executes all demonstrations and visualizations.

===============================================================================
MATHEMATICAL FOUNDATION
===============================================================================

1. MARKOV PROPERTY

    P(S[t+1] | S[t], S[t-1], ..., S[0], A[t])
        =
    P(S[t+1] | S[t], A[t])

    The next state depends on the current state and action, rather than
    requiring the entire previous history.

    This does NOT mean that the past is irrelevant.

    It means that the current state contains the relevant information
    from the past needed to predict the future.

2. MARKOV DECISION PROCESS

    M = (S, A, P, R, gamma)

    S:
        State space.

    A:
        Action space.

    P(s'|s,a):
        Probability of transitioning to state s' after action a in state s.

    R(s,a,s'):
        Reward associated with the transition.

    gamma:
        Discount factor.

3. TRANSITION PROBABILITY

    P(s'|s,a) >= 0

    For every state-action pair:

        sum_s' P(s'|s,a) = 1

4. TRANSITION MATRIX

    For a fixed action a:

        P_a[i,j] = P(S[t+1]=j | S[t]=i, A[t]=a)

    Each row of the transition matrix must sum to one.

5. STATE REPRESENTATION

    Suppose a moving object has:

        position[t+1] = position[t] + velocity[t]

        velocity[t+1] = velocity[t]

    If the agent observes position only, the next position depends
    on hidden velocity.

    Therefore, position alone is generally not Markov.

    A more complete state is:

        S[t] = (position[t], velocity[t])

    This state contains enough information to predict the next state.

6. PARTIAL OBSERVABILITY

    If an agent cannot observe the complete environment state, it may
    need to maintain a belief distribution:

        b(s) = P(S=s | observation history)

    Belief-state methods will be studied later.

7. EMPIRICAL TRANSITION PROBABILITY

    P_hat(s'|s,a) = N(s,a,s') / N(s,a)

    N(s,a,s'):
        Number of observed transitions from s to s' after action a.

    N(s,a):
        Total number of times action a was executed in state s.

===============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# STEP 1 — MARKOV PROPERTY
# =============================================================================

def verify_markov_property(
    conditional_distributions: List[np.ndarray],
    tolerance: float = 1e-10
) -> bool:
    """
    Check whether next-state distributions are equal across histories
    that share the same current state and action.

    Each array represents:

        P(S[t+1] | history, S[t], A[t])

    For a Markov state, these distributions should be identical
    regardless of the earlier history.
    """
    if len(conditional_distributions) < 2:
        raise ValueError("At least two distributions are required")

    reference = np.asarray(
        conditional_distributions[0],
        dtype=float
    )

    if reference.ndim != 1:
        raise ValueError("Distributions must be 1D arrays")

    if not np.all(np.isfinite(reference)):
        raise ValueError("Distributions must be finite")

    if np.any(reference < 0) or not np.isclose(np.sum(reference), 1.0):
        raise ValueError("Invalid probability distribution")

    for distribution in conditional_distributions[1:]:

        candidate = np.asarray(
            distribution,
            dtype=float
        )

        if candidate.shape != reference.shape:
            raise ValueError("Distribution dimensions must match")

        if not np.all(np.isfinite(candidate)):
            raise ValueError("Distributions must be finite")

        if np.any(candidate < 0):
            raise ValueError("Probabilities cannot be negative")

        if not np.isclose(np.sum(candidate), 1.0):
            raise ValueError("Each distribution must sum to one")

        if not np.allclose(
            reference,
            candidate,
            atol=tolerance,
            rtol=0.0
        ):
            return False

    return True


def demonstrate_markov_property():
    """
    Compare two cases:

    Case A:
        The state contains all relevant information.

    Case B:
        The state omits velocity, which influences the next position.
    """

    print("\n[MARKOV PROPERTY DEMONSTRATION]")

    # Full state: (position, velocity)
    #
    # Given position=5 and velocity=+1:
    # next position is always 6.
    #
    # The earlier history does not change this prediction.

    full_state_history_1 = np.array([0.0, 0.0, 1.0])
    full_state_history_2 = np.array([3.0, 4.0, 1.0])

    print("Full state history 1:", full_state_history_1)
    print("Full state history 2:", full_state_history_2)

    # Both histories have the same current full state:
    # position=5, velocity=+1.
    #
    # Both produce the same next position distribution.

    full_state_distributions = [
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    ]

    full_markov = verify_markov_property(
        full_state_distributions
    )

    # Partial observation:
    # The agent sees position=5 but does not know velocity.
    #
    # History 1: velocity=+1 -> next position=6.
    # History 2: velocity=-1 -> next position=4.

    partial_state_distributions = [
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    ]

    partial_markov = verify_markov_property(
        partial_state_distributions
    )

    print("Full state Markov property:", full_markov)
    print("Position-only Markov property:", partial_markov)

    return full_markov, partial_markov


# =============================================================================
# STEP 2 — ENVIRONMENT WITH HIDDEN VELOCITY
# =============================================================================

@dataclass
class MomentumState:
    position: int
    velocity: int


class MomentumEnvironment:
    """
    One-dimensional environment with constant velocity.

    Full state:
        S = (position, velocity)

    Observation:
        O = position

    Dynamics:
        position[t+1] = position[t] + velocity[t]

        velocity[t+1] = velocity[t]

    This is a simple example of partial observability.

    The full state is Markov, but position-only observation is not.
    """

    def __init__(
        self,
        initial_position: int = 5,
        initial_velocity: int = 1
    ):
        self.state = MomentumState(
            position=initial_position,
            velocity=initial_velocity
        )

    def reset(
        self,
        position: int = 5,
        velocity: int = 1
    ):
        self.state = MomentumState(
            position=position,
            velocity=velocity
        )

        return self.observe()

    def observe(self) -> int:
        """
        Return only the observable position.
        """
        return self.state.position

    def get_full_state(self) -> Tuple[int, int]:
        """
        Return position and velocity.
        """
        return (
            self.state.position,
            self.state.velocity
        )

    def step(self) -> Tuple[int, Tuple[int, int]]:
        """
        Advance the environment by one time step.
        """
        self.state.position += self.state.velocity

        return (
            self.observe(),
            self.get_full_state()
        )


def simulate_momentum_environment(
    initial_position: int,
    initial_velocity: int,
    steps: int
):
    """
    Generate a trajectory with observable position and hidden velocity.
    """
    env = MomentumEnvironment(
        initial_position,
        initial_velocity
    )

    positions = [env.observe()]
    velocities = [env.state.velocity]

    for _ in range(steps):
        env.step()

        positions.append(env.observe())
        velocities.append(env.state.velocity)

    return (
        np.asarray(positions),
        np.asarray(velocities)
    )


# =============================================================================
# STEP 3 — GENERAL TRANSITION MODEL
# =============================================================================

class TransitionModel:
    """
    General finite-state transition model.

    Tensor shape:

        (num_actions, num_states, num_states)

    Element:

        P[a, s, s_next]

    Represents:

        P(S[t+1]=s_next | S[t]=s, A[t]=a)
    """

    def __init__(self, probabilities: np.ndarray):

        self.probabilities = np.asarray(
            probabilities,
            dtype=float
        )

        if self.probabilities.ndim != 3:
            raise ValueError(
                "Transition probabilities must be a 3D array"
            )

        if not np.all(np.isfinite(self.probabilities)):
            raise ValueError("Transition probabilities must be finite")

        if np.any(self.probabilities < 0):
            raise ValueError("Probabilities cannot be negative")

        row_sums = np.sum(
            self.probabilities,
            axis=2
        )

        if not np.allclose(row_sums, 1.0):
            raise ValueError(
                "Each state-action transition distribution must sum to 1"
            )

        self.num_actions = self.probabilities.shape[0]
        self.num_states = self.probabilities.shape[1]

        if self.probabilities.shape[2] != self.num_states:
            raise ValueError("Transition matrix must be square")

    def get_distribution(
        self,
        state: int,
        action: int
    ) -> np.ndarray:

        self._validate_indices(state, action)

        return self.probabilities[action, state].copy()

    def sample_next_state(
        self,
        state: int,
        action: int,
        rng: np.random.Generator
    ) -> int:

        probabilities = self.get_distribution(
            state,
            action
        )

        return int(
            rng.choice(
                self.num_states,
                p=probabilities
            )
        )

    def transition_matrix(self, action: int) -> np.ndarray:

        if not 0 <= action < self.num_actions:
            raise ValueError("Invalid action")

        return self.probabilities[action].copy()

    def _validate_indices(self, state: int, action: int):

        if not 0 <= state < self.num_states:
            raise ValueError("Invalid state")

        if not 0 <= action < self.num_actions:
            raise ValueError("Invalid action")


def validate_transition_model(
    probabilities: np.ndarray
) -> bool:
    """
    Verify transition probability constraints.
    """
    p = np.asarray(probabilities, dtype=float)

    if p.ndim != 3:
        return False

    if not np.all(np.isfinite(p)):
        return False

    if np.any(p < 0):
        return False

    if not np.allclose(
        np.sum(p, axis=2),
        1.0
    ):
        return False

    if p.shape[1] != p.shape[2]:
        return False

    return True


# =============================================================================
# STEP 4 — STOCHASTIC GRIDWORLD
# =============================================================================

class StochasticGridWorld:
    """
    A 4x4 GridWorld with stochastic movement.

    Actions:
        0 = Up
        1 = Down
        2 = Left
        3 = Right

    Movement:
        80% intended direction.
        10% one perpendicular direction.
        10% the other perpendicular direction.

    If movement hits a boundary or obstacle, the agent stays in place.

    This creates a stochastic transition model.
    """

    ACTIONS = {
        0: (-1, 0),
        1: (1, 0),
        2: (0, -1),
        3: (0, 1)
    }

    ACTION_NAMES = [
        "Up",
        "Down",
        "Left",
        "Right"
    ]

    def __init__(
        self,
        rows: int = 4,
        cols: int = 4,
        seed: int = 42
    ):

        self.rows = rows
        self.cols = cols

        self.start = (0, 0)
        self.goal = (3, 3)

        self.obstacles = {
            (1, 1),
            (2, 1)
        }

        self.rng = np.random.default_rng(seed)

        self.state = self.start

        self.num_states = rows * cols
        self.num_actions = 4

    def state_to_index(
        self,
        state: Tuple[int, int]
    ) -> int:

        row, col = state

        return row * self.cols + col

    def index_to_state(
        self,
        index: int
    ) -> Tuple[int, int]:

        if not 0 <= index < self.num_states:
            raise ValueError("Invalid state index")

        return (
            index // self.cols,
            index % self.cols
        )

    def is_valid(
        self,
        state: Tuple[int, int]
    ) -> bool:

        row, col = state

        return (
            0 <= row < self.rows
            and 0 <= col < self.cols
            and state not in self.obstacles
        )

    def move(
        self,
        state: Tuple[int, int],
        direction: int
    ) -> Tuple[int, int]:

        dr, dc = self.ACTIONS[direction]

        candidate = (
            state[0] + dr,
            state[1] + dc
        )

        if self.is_valid(candidate):
            return candidate

        return state

    def get_transition_distribution(
        self,
        state: Tuple[int, int],
        action: int
    ) -> Dict[Tuple[int, int], float]:
        """
        Return the theoretical next-state probability distribution.

        Direction probabilities:
            Intended direction: 0.8
            Perpendicular direction 1: 0.1
            Perpendicular direction 2: 0.1

        If multiple directions lead to the same state, their
        probabilities are combined.
        """
        if not self.is_valid(state):
            raise ValueError("Current state must be valid")

        if not 0 <= action < self.num_actions:
            raise ValueError("Invalid action")

        perpendicular = {
            0: (2, 3),
            1: (2, 3),
            2: (0, 1),
            3: (0, 1)
        }

        directions = [
            (action, 0.8),
            (perpendicular[action][0], 0.1),
            (perpendicular[action][1], 0.1)
        ]

        distribution = {}

        for direction, probability in directions:

            next_state = self.move(
                state,
                direction
            )

            distribution[next_state] = (
                distribution.get(next_state, 0.0)
                + probability
            )

        return distribution

    def step(
        self,
        action: int
    ) -> Tuple[Tuple[int, int], float, bool]:

        distribution = self.get_transition_distribution(
            self.state,
            action
        )

        next_states = list(distribution.keys())

        probabilities = np.array(
            list(distribution.values())
        )

        selected = int(
            self.rng.choice(
                len(next_states),
                p=probabilities
            )
        )

        self.state = next_states[selected]

        terminated = self.state == self.goal

        reward = 1.0 if terminated else -0.04

        return self.state, reward, terminated

    def reset(self):
        self.state = self.start
        return self.state

    def build_transition_model(self) -> TransitionModel:
        """
        Build a complete transition tensor for valid GridWorld states.

        Obstacle cells are represented as self-loop states so that
        the tensor has a regular rectangular state representation.
        The agent is never allowed to enter these obstacle states.
        """
        probabilities = np.zeros(
            (
                self.num_actions,
                self.num_states,
                self.num_states
            ),
            dtype=float
        )

        for state_index in range(self.num_states):

            state = self.index_to_state(state_index)

            if not self.is_valid(state):

                # Obstacle states are unreachable from valid states.
                # A self-loop makes their transition rows well-defined.
                for action in range(self.num_actions):
                    probabilities[
                        action,
                        state_index,
                        state_index
                    ] = 1.0

                continue

            for action in range(self.num_actions):

                distribution = self.get_transition_distribution(
                    state,
                    action
                )

                for next_state, probability in distribution.items():

                    next_index = self.state_to_index(next_state)

                    probabilities[
                        action,
                        state_index,
                        next_index
                    ] += probability

        return TransitionModel(probabilities)


# =============================================================================
# STEP 5 — EMPIRICAL TRANSITION ESTIMATION
# =============================================================================

def estimate_transition_distribution(
    samples: List[int],
    num_states: int
) -> np.ndarray:
    """
    Estimate transition probabilities from observed next-state samples.

        P_hat(s') = count(s') / total_samples
    """
    if len(samples) == 0:
        raise ValueError("At least one sample is required")

    counts = np.zeros(num_states, dtype=int)

    for state in samples:

        if not 0 <= state < num_states:
            raise ValueError("Invalid state sample")

        counts[state] += 1

    return counts / len(samples)


def sample_transition_distribution(
    model: TransitionModel,
    state: int,
    action: int,
    num_samples: int,
    seed: int = 42
) -> np.ndarray:
    """
    Generate samples and estimate the transition distribution.
    """
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")

    rng = np.random.default_rng(seed)

    samples = [
        model.sample_next_state(state, action, rng)
        for _ in range(num_samples)
    ]

    return estimate_transition_distribution(
        samples,
        model.num_states
    )


# =============================================================================
# STEP 6 — VISUALIZATION
# =============================================================================

def visualize_markov_comparison():
    """
    Compare the next-position distributions under two hidden velocities.

    Same observed position:
        position = 5

    Different hidden velocities:
        velocity = +1
        velocity = -1

    Different next-state distributions show why position-only
    observation is not Markov.
    """
    distributions = np.array([
        [0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1, 0, 0]
    ])

    plt.figure(figsize=(9, 4))

    x = np.arange(7)
    width = 0.35

    plt.bar(
        x - width / 2,
        distributions[0],
        width,
        label="Hidden velocity = +1"
    )

    plt.bar(
        x + width / 2,
        distributions[1],
        width,
        label="Hidden velocity = -1"
    )

    plt.xlabel("Next position")
    plt.ylabel("Probability")
    plt.title("Same Position Observation, Different Future")
    plt.xticks(x)
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()


def visualize_momentum_trajectories():
    """
    Show position trajectories with different hidden velocities.
    """
    position_forward, _ = simulate_momentum_environment(
        initial_position=5,
        initial_velocity=1,
        steps=5
    )

    position_backward, _ = simulate_momentum_environment(
        initial_position=5,
        initial_velocity=-1,
        steps=5
    )

    plt.figure(figsize=(9, 4))

    plt.plot(
        position_forward,
        marker="o",
        label="Velocity = +1"
    )

    plt.plot(
        position_backward,
        marker="s",
        label="Velocity = -1"
    )

    plt.xlabel("Time step")
    plt.ylabel("Position")
    plt.title("Position Trajectories with Hidden Velocity")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()


def visualize_transition_matrix(
    model: TransitionModel,
    action: int
):
    """
    Display P(s'|s,a) as a heatmap.
    """
    matrix = model.transition_matrix(action)

    plt.figure(figsize=(9, 7))

    plt.imshow(
        matrix,
        aspect="auto",
        interpolation="nearest"
    )

    plt.colorbar(label="Transition probability")

    plt.xlabel("Next state index")
    plt.ylabel("Current state index")

    plt.title(
        f"Transition Probability Matrix — "
        f"Action: {StochasticGridWorld.ACTION_NAMES[action]}"
    )

    plt.tight_layout()


def visualize_gridworld(
    env: StochasticGridWorld,
    trajectory: List[Tuple[int, int]]
):
    """
    Display the GridWorld and one sampled trajectory.
    """
    grid = np.zeros(
        (env.rows, env.cols)
    )

    for row, col in env.obstacles:
        grid[row, col] = -1

    grid[env.goal] = 2
    grid[env.start] = 1

    plt.figure(figsize=(6, 6))

    plt.imshow(
        grid,
        cmap="coolwarm",
        vmin=-1,
        vmax=2
    )

    for row in range(env.rows):
        for col in range(env.cols):
            label = ""

            if (row, col) == env.start:
                label = "START"
            elif (row, col) == env.goal:
                label = "GOAL"
            elif (row, col) in env.obstacles:
                label = "BLOCK"

            plt.text(
                col,
                row,
                label,
                ha="center",
                va="center"
            )

    trajectory_array = np.asarray(trajectory)

    plt.plot(
        trajectory_array[:, 1],
        trajectory_array[:, 0],
        marker="o",
        linewidth=2,
        label="Agent trajectory"
    )

    plt.xticks(range(env.cols))
    plt.yticks(range(env.rows))
    plt.grid(alpha=0.3)
    plt.title("Stochastic GridWorld Trajectory")
    plt.legend()
    plt.tight_layout()


def visualize_empirical_vs_theoretical(
    theoretical: np.ndarray,
    empirical: np.ndarray
):
    """
    Compare theoretical and sampled transition probabilities.
    """
    x = np.arange(len(theoretical))
    width = 0.35

    plt.figure(figsize=(10, 4))

    plt.bar(
        x - width / 2,
        theoretical,
        width,
        label="Theoretical"
    )

    plt.bar(
        x + width / 2,
        empirical,
        width,
        label="Empirical"
    )

    plt.xlabel("Next state index")
    plt.ylabel("Probability")
    plt.title("Theoretical vs. Empirical Transition Probability")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()


# =============================================================================
# STEP 7 — NUMERICAL VERIFICATION AND UNIT TESTS
# =============================================================================

def run_tests():

    print("\n========== UNIT TESTS ==========")

    # Test 1: Markov state has identical future distributions
    distributions = [
        np.array([0.2, 0.8]),
        np.array([0.2, 0.8])
    ]

    assert verify_markov_property(distributions)
    print("[PASS] Markov property verification")

    # Test 2: Non-Markov observation
    distributions = [
        np.array([1.0, 0.0]),
        np.array([0.0, 1.0])
    ]

    assert not verify_markov_property(distributions)
    print("[PASS] Non-Markov observation detection")

    # Test 3: Momentum full state
    env = MomentumEnvironment(5, 1)

    assert env.get_full_state() == (5, 1)

    observation, full_state = env.step()

    assert observation == 6
    assert full_state == (6, 1)

    print("[PASS] Momentum state transition")

    # Test 4: Momentum with negative velocity
    env = MomentumEnvironment(5, -1)

    observation, full_state = env.step()

    assert observation == 4
    assert full_state == (4, -1)

    print("[PASS] Negative velocity transition")

    # Test 5: Valid transition model
    probabilities = np.array([
        [
            [0.8, 0.2],
            [0.3, 0.7]
        ]
    ])

    assert validate_transition_model(probabilities)

    model = TransitionModel(probabilities)

    assert model.num_states == 2
    assert model.num_actions == 1

    print("[PASS] Transition model validation")

    # Test 6: Transition distribution sums to one
    distribution = model.get_distribution(0, 0)

    assert np.isclose(np.sum(distribution), 1.0)

    print("[PASS] Transition distribution normalization")

    # Test 7: Invalid transition probabilities
    invalid = np.array([
        [
            [0.5, 0.3],
            [0.3, 0.7]
        ]
    ])

    assert not validate_transition_model(invalid)

    print("[PASS] Invalid transition probability detection")

    # Test 8: Stochastic GridWorld movement distribution
    grid = StochasticGridWorld()

    distribution = grid.get_transition_distribution(
        (1, 2),
        3
    )

    assert np.isclose(
        sum(distribution.values()),
        1.0
    )

    print("[PASS] GridWorld transition distribution")

    # Test 9: Boundary handling
    next_state = grid.move((0, 0), 0)

    assert next_state == (0, 0)

    print("[PASS] Boundary handling")

    # Test 10: Obstacle handling
    next_state = grid.move((0, 1), 1)

    assert next_state == (0, 1)

    print("[PASS] Obstacle handling")

    # Test 11: GridWorld transition tensor
    grid_model = grid.build_transition_model()

    assert grid_model.num_actions == 4
    assert grid_model.num_states == 16

    print("[PASS] GridWorld transition model")

    # Test 12: Every transition row sums to one
    sums = np.sum(
        grid_model.probabilities,
        axis=2
    )

    assert np.allclose(sums, 1.0)

    print("[PASS] All GridWorld transition rows normalized")

    # Test 13: Empirical transition estimation
    samples = [0, 0, 1, 1, 1]

    estimated = estimate_transition_distribution(
        samples,
        num_states=2
    )

    assert np.allclose(
        estimated,
        [0.4, 0.6]
    )

    print("[PASS] Empirical transition estimation")

    # Test 14: Empirical distribution approaches theoretical
    probabilities = np.array([
        [
            [0.7, 0.3],
            [0.2, 0.8]
        ]
    ])

    model = TransitionModel(probabilities)

    empirical = sample_transition_distribution(
        model,
        state=0,
        action=0,
        num_samples=50000,
        seed=123
    )

    assert np.allclose(
        empirical,
        [0.7, 0.3],
        atol=0.015
    )

    print("[PASS] Empirical transition probability")

    # Test 15: Deterministic transition sampling
    probabilities = np.array([
        [
            [1.0, 0.0],
            [0.0, 1.0]
        ]
    ])

    model = TransitionModel(probabilities)

    rng = np.random.default_rng(42)

    for _ in range(100):
        assert model.sample_next_state(
            0,
            0,
            rng
        ) == 0

    print("[PASS] Deterministic transition sampling")

    # Test 16: Invalid action detection
    try:
        grid.get_transition_distribution(
            (0, 0),
            10
        )

        raise AssertionError("Invalid action accepted")

    except ValueError:
        pass

    print("[PASS] Invalid action validation")

    # Test 17: Invalid state detection
    try:
        grid.get_transition_distribution(
            (-1, 0),
            0
        )

        raise AssertionError("Invalid state accepted")

    except ValueError:
        pass

    print("[PASS] Invalid state validation")

    print("\nAll 17 tests passed.")


# =============================================================================
# STEP 8 — MAIN DEMONSTRATION
# =============================================================================

def main():

    print("=" * 72)
    print("RL PHASE 1.4 — MARKOV PROPERTY AND STATE TRANSITIONS")
    print("=" * 72)

    # -------------------------------------------------------------------------
    # Demo A: Markov property
    # -------------------------------------------------------------------------

    full_markov, partial_markov = demonstrate_markov_property()

    print("\nInterpretation:")

    print(
        "Full state (position, velocity) is Markov:",
        full_markov
    )

    print(
        "Position-only observation is Markov:",
        partial_markov
    )

    # -------------------------------------------------------------------------
    # Demo B: Transition model
    # -------------------------------------------------------------------------

    print("\n[DEMO B] Transition probability model")

    probabilities = np.array([
        [
            [0.8, 0.2, 0.0],
            [0.1, 0.7, 0.2],
            [0.0, 0.3, 0.7]
        ],

        [
            [0.6, 0.4, 0.0],
            [0.2, 0.6, 0.2],
            [0.0, 0.1, 0.9]
        ]
    ])

    model = TransitionModel(probabilities)

    print("Number of actions:", model.num_actions)
    print("Number of states:", model.num_states)

    for action in range(model.num_actions):

        print(f"\nAction {action} transition matrix:")

        print(
            model.transition_matrix(action)
        )

    # -------------------------------------------------------------------------
    # Demo C: Stochastic GridWorld
    # -------------------------------------------------------------------------

    print("\n[DEMO C] Stochastic GridWorld")

    grid = StochasticGridWorld(seed=42)

    current_state = (0, 0)
    selected_action = 3  # Right

    distribution = grid.get_transition_distribution(
        current_state,
        selected_action
    )

    print("Current state:", current_state)
    print("Action:", grid.ACTION_NAMES[selected_action])
    print("Possible next states:")

    for next_state, probability in distribution.items():

        print(
            f"  {next_state}: {probability:.3f}"
        )

    # -------------------------------------------------------------------------
    # Demo D: Empirical versus theoretical probabilities
    # -------------------------------------------------------------------------

    print("\n[DEMO D] Empirical transition estimation")

    grid_model = grid.build_transition_model()

    state_index = grid.state_to_index((0, 0))

    theoretical = grid_model.get_distribution(
        state=state_index,
        action=3
    )

    empirical = sample_transition_distribution(
        grid_model,
        state=state_index,
        action=3,
        num_samples=50000,
        seed=42
    )

    print("Theoretical:", np.round(theoretical, 4))
    print("Empirical:  ", np.round(empirical, 4))

    # -------------------------------------------------------------------------
    # Demo E: Sample trajectory
    # -------------------------------------------------------------------------

    print("\n[DEMO E] Stochastic GridWorld trajectory")

    trajectory = [grid.reset()]

    for _ in range(20):

        # A simple fixed action policy:
        # move right until the last column, then move down.
        row, col = grid.state

        if col < grid.cols - 1:
            action = 3
        else:
            action = 1

        next_state, reward, terminated = grid.step(action)

        trajectory.append(next_state)

        if terminated:
            break

    print("Trajectory:", trajectory)

    # -------------------------------------------------------------------------
    # Demo F: Unit tests
    # -------------------------------------------------------------------------

    run_tests()

    # -------------------------------------------------------------------------
    # Demo G: Integrated visualizations
    # -------------------------------------------------------------------------

    visualize_markov_comparison()

    visualize_momentum_trajectories()

    visualize_transition_matrix(
        grid_model,
        action=3
    )

    visualize_gridworld(
        grid,
        trajectory
    )

    visualize_empirical_vs_theoretical(
        theoretical,
        empirical
    )

    plt.show()

    print("\nLesson completed.")


if __name__ == "__main__":
    main()
