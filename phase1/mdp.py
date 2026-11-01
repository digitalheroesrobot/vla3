
"""
====================================================================
RL_Course
Phase 1.1 — Agent–Environment Interaction and MDP
File: phase1/mdp.py
====================================================================

PURPOSE OF THIS LESSON
----------------------
Understand the fundamental structure of Reinforcement Learning by
building a GridWorld environment and an agent that interacts with it.

LEARNING ROADMAP
----------------

STEP 1 — RL Agent–Environment Interaction
Purpose:
    Understand the basic feedback loop through which an RL agent
    learns from its environment.

What you learn:
    - Agent and environment.
    - Observation/state.
    - Action.
    - Reward.
    - State transition.
    - Episode termination.

Implementation:
    Build an interaction loop in which an agent observes a state,
    chooses an action, and receives the next state and reward.

STEP 2 — Markov Decision Process (MDP)
Purpose:
    Understand the mathematical model underlying most RL algorithms.

What you learn:
    An MDP is commonly represented as:

        M = (S, A, P, R, gamma)

        S     : State space
        A     : Action space
        P     : Transition probability
        R     : Reward function
        gamma : Discount factor

    The transition probability is:

        P(s' | s, a)

    It describes the probability of reaching state s' after
    taking action a in state s.

    The expected return is:

        G_t = r_(t+1) + gamma*r_(t+2)
              + gamma^2*r_(t+3) + ...

Implementation:
    Define the state space, action space, transition function,
    reward function, and discount factor.

STEP 3 — Build a GridWorld Environment
Purpose:
    Convert the mathematical MDP definition into an executable
    environment.

What you learn:
    - State representation using grid coordinates.
    - Discrete action representation.
    - Obstacles and boundaries.
    - Goal states.
    - Deterministic state transitions.

Implementation:
    Create a 5x5 grid with a start position, obstacles, and a goal.

STEP 4 — Episode Simulation
Purpose:
    Understand how an agent interacts with an environment over time.

What you learn:
    - One interaction step.
    - Episode.
    - Immediate reward.
    - Cumulative reward.
    - Terminated versus truncated episodes.

Implementation:
    Run complete episodes and record their trajectories.

STEP 5 — Markov Property
Purpose:
    Understand why an MDP can represent a sequential decision problem
    using the current state rather than the entire interaction history.

What you learn:

        P(S_(t+1) | S_t, A_t, history)
        =
        P(S_(t+1) | S_t, A_t)

    The current state must contain sufficient information to
    determine the future transition distribution.

Implementation:
    Verify the transition distribution for a state-action pair.

STEP 6 — Compare Different Agents
Purpose:
    Understand that an agent's behavior depends on its policy.

What you learn:
    - Random policy.
    - Goal-directed policy.
    - Policy evaluation through episode results.
    - Success rate and cumulative reward.

Implementation:
    Compare a random agent with a shortest-path baseline.

    IMPORTANT:
    The shortest-path agent is NOT a learned RL agent.
    It is an informed baseline used to demonstrate how policies
    influence behavior. Learning algorithms come in later lessons.

STEP 7 — Visualization and Verification
Purpose:
    Make the interaction process observable and verify correctness.

What you learn:
    - Agent trajectories.
    - Reward sequences.
    - Episode outcomes.
    - Environment state transitions.

Implementation:
    Display the GridWorld, agent trajectories, reward comparisons,
    and transition sequences. Run automated tests.

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
python phase1/mdp.py
====================================================================
"""

from collections import deque

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


# ================================================================
# GLOBAL CONFIGURATION
# ================================================================

GRID_SIZE = 5

START_STATE = (0, 0)
GOAL_STATE = (4, 4)

OBSTACLES = {
    (1, 1),
    (2, 1),
    (3, 1)
}

STEP_REWARD = -0.04
GOAL_REWARD = 1.0

DISCOUNT_FACTOR = 0.99

MAX_STEPS = 100

# Actions:
# 0 = Up
# 1 = Down
# 2 = Left
# 3 = Right

ACTIONS = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, -1),
    3: (0, 1)
}

ACTION_NAMES = {
    0: "UP",
    1: "DOWN",
    2: "LEFT",
    3: "RIGHT"
}


# ================================================================
# STEP 1 — AGENT–ENVIRONMENT INTERACTION
# ================================================================

class Agent:
    """
    Base class for an RL agent.

    The agent receives the current state and chooses an action.

    In a typical RL interaction:

        state = environment.reset()

        while not done:
            action = agent.select_action(state)

            next_state, reward, terminated, truncated, info = (
                environment.step(action)
            )

            agent receives feedback
            state = next_state

    The agent and environment are separate components.
    This separation allows the same agent interface to work
    with different environments.
    """

    def select_action(self, state):
        raise NotImplementedError


class RandomAgent(Agent):
    """
    Random policy.

    Every available action has equal selection probability.

    This agent does not learn. It provides a baseline for comparison.
    """

    def __init__(self, seed=42):
        self.rng = np.random.default_rng(seed)

    def select_action(self, state):
        return int(self.rng.integers(0, len(ACTIONS)))


# ================================================================
# STEP 2 — MARKOV DECISION PROCESS DEFINITION
# ================================================================

class GridWorld:
    """
    A deterministic GridWorld MDP.

    MDP components:

        State space S:
            All grid positions except obstacle cells.

        Action space A:
            UP, DOWN, LEFT, RIGHT.

        Transition model P(s' | s, a):
            The agent moves in the selected direction.
            If the move hits a wall or obstacle, the agent stays
            in the same position.

        Reward function R(s, a, s'):
            +1.0 when the goal is reached.
            -0.04 for every other transition.

        Discount factor gamma:
            0.99

    The environment is fully observable: the current grid position
    contains all information needed for this example.
    """

    def __init__(
        self,
        grid_size=GRID_SIZE,
        start_state=START_STATE,
        goal_state=GOAL_STATE,
        obstacles=OBSTACLES,
        max_steps=MAX_STEPS,
        step_reward=STEP_REWARD,
        goal_reward=GOAL_REWARD,
        gamma=DISCOUNT_FACTOR
    ):

        self.grid_size = grid_size
        self.start_state = tuple(start_state)
        self.goal_state = tuple(goal_state)
        self.obstacles = set(obstacles)

        self.max_steps = max_steps
        self.step_reward = step_reward
        self.goal_reward = goal_reward
        self.gamma = gamma

        self.state = self.start_state

        self.elapsed_steps = 0
        self.terminated = False
        self.truncated = False

        self._validate_configuration()

    def _validate_configuration(self):

        assert self.grid_size > 1

        assert self._inside_grid(self.start_state)
        assert self._inside_grid(self.goal_state)

        assert self.start_state not in self.obstacles
        assert self.goal_state not in self.obstacles

        assert self.start_state != self.goal_state

        assert all(
            self._inside_grid(obstacle)
            for obstacle in self.obstacles
        )

        assert self.max_steps > 0
        assert 0.0 <= self.gamma <= 1.0

    def _inside_grid(self, state):

        row, col = state

        return (
            0 <= row < self.grid_size
            and
            0 <= col < self.grid_size
        )

    def get_states(self):
        """
        Return all valid states.

        Obstacles are excluded from the state space.
        """
        states = []

        for row in range(self.grid_size):
            for col in range(self.grid_size):

                state = (row, col)

                if state not in self.obstacles:
                    states.append(state)

        return states

    def get_actions(self):
        """
        Return all available action IDs.
        """
        return list(ACTIONS.keys())

    def get_next_state(self, state, action):
        """
        Deterministic transition function.

        P(s' | s, a) = 1 for the resulting next state.

        Invalid movement:
            If the next position is outside the grid or is an
            obstacle, the agent remains in its current position.
        """
        if action not in ACTIONS:
            raise ValueError(f"Invalid action: {action}")

        row, col = state

        delta_row, delta_col = ACTIONS[action]

        next_state = (
            row + delta_row,
            col + delta_col
        )

        if not self._inside_grid(next_state):
            return state

        if next_state in self.obstacles:
            return state

        return next_state

    def transition_probability(
        self,
        state,
        action,
        next_state
    ):
        """
        Calculate P(next_state | state, action).

        Since this GridWorld is deterministic:

            Probability = 1 if next_state is the actual result.
            Probability = 0 otherwise.
        """
        expected_state = self.get_next_state(
            state,
            action
        )

        return float(expected_state == tuple(next_state))

    def reward_function(self, state, action, next_state):
        """
        Immediate reward:

            R(s, a, s') = +1.0 if s' is the goal
                          -0.04 otherwise
        """
        if next_state == self.goal_state:
            return self.goal_reward

        return self.step_reward

    def reset(self, seed=None):
        """
        Start a new episode.

        Returns:
            initial_state, info
        """
        if seed is not None:
            self._reset_rng_seed = seed

        self.state = self.start_state

        self.elapsed_steps = 0

        self.terminated = False
        self.truncated = False

        info = {
            "episode_step": 0,
            "is_start_state": True
        }

        return self.state, info

    def step(self, action):
        """
        Execute one environment transition.

        Returns:

            next_state
            reward
            terminated
            truncated
            info

        terminated:
            The agent reached the goal.

        truncated:
            The maximum episode length was reached.

        These two conditions are different:
            Termination is part of the task definition.
            Truncation is an imposed simulation limit.
        """

        if self.terminated or self.truncated:
            raise RuntimeError(
                "Episode finished. Call reset() before step()."
            )

        current_state = self.state

        next_state = self.get_next_state(
            current_state,
            action
        )

        reward = self.reward_function(
            current_state,
            action,
            next_state
        )

        self.state = next_state

        self.elapsed_steps += 1

        self.terminated = (
            next_state == self.goal_state
        )

        self.truncated = (
            self.elapsed_steps >= self.max_steps
            and not self.terminated
        )

        info = {
            "episode_step": self.elapsed_steps,
            "current_state": current_state,
            "next_state": next_state,
            "action_name": ACTION_NAMES[action]
        }

        return (
            next_state,
            reward,
            self.terminated,
            self.truncated,
            info
        )


# ================================================================
# STEP 3 — MDP MATHEMATICAL UTILITIES
# ================================================================

def calculate_discounted_return(
    rewards,
    gamma=DISCOUNT_FACTOR
):
    """
    Calculate discounted return:

        G_0 = r_1 + gamma*r_2 + gamma^2*r_3 + ...

    Earlier rewards receive greater weight than distant rewards
    when gamma is between 0 and 1.
    """
    total_return = 0.0

    for time_step, reward in enumerate(rewards):

        total_return += (
            gamma ** time_step
        ) * reward

    return float(total_return)


def calculate_episode_statistics(episode):
    """
    Summarize an episode.

    Total reward:
        Sum of all immediate rewards.

    Discounted return:
        Sum of discounted immediate rewards.

    Success:
        True if the agent reached the goal.
    """
    rewards = episode["rewards"]

    return {
        "steps": len(rewards),
        "total_reward": float(np.sum(rewards)),
        "discounted_return": calculate_discounted_return(
            rewards
        ),
        "success": episode["terminated"]
    }


# ================================================================
# STEP 4 — EPISODE SIMULATION
# ================================================================

def run_episode(
    environment,
    agent,
    seed=None,
    verbose=False
):
    """
    Run one complete episode.

    The environment and agent interact repeatedly:

        1. Agent observes current state.
        2. Agent selects an action.
        3. Environment executes the action.
        4. Environment returns next state and reward.
        5. Agent receives the new observation.
        6. Repeat until termination or truncation.

    The complete trajectory is recorded for analysis.
    """

    state, reset_info = environment.reset(seed=seed)

    states = [state]
    actions = []
    rewards = []

    terminated = False
    truncated = False

    if verbose:
        print("\nEpisode starts")
        print("Initial state:", state)

    while not (terminated or truncated):

        action = agent.select_action(state)

        (
            next_state,
            reward,
            terminated,
            truncated,
            info
        ) = environment.step(action)

        actions.append(action)
        rewards.append(reward)
        states.append(next_state)

        if verbose:
            print(
                f"Step {info['episode_step']:02d}: "
                f"State={state}, "
                f"Action={info['action_name']}, "
                f"Next={next_state}, "
                f"Reward={reward:.2f}"
            )

        state = next_state

    episode = {
        "states": states,
        "actions": actions,
        "rewards": rewards,
        "terminated": terminated,
        "truncated": truncated
    }

    episode["statistics"] = calculate_episode_statistics(
        episode
    )

    if verbose:
        print("\nEpisode finished")
        print("Steps:", episode["statistics"]["steps"])
        print(
            "Total reward:",
            episode["statistics"]["total_reward"]
        )
        print(
            "Discounted return:",
            episode["statistics"]["discounted_return"]
        )
        print("Success:", episode["statistics"]["success"])

    return episode


# ================================================================
# STEP 5 — MARKOV PROPERTY VERIFICATION
# ================================================================

def verify_markov_property(environment, state, action):
    """
    Verify the transition distribution for a state-action pair.

    In this deterministic environment:

        P(s' | s, a) = 1

    for exactly one next state.

    The transition model depends on the current state and action,
    not on the path used to arrive at that state.

    Important:
        A transition test alone does not prove that an arbitrary
        real-world problem is Markov. It verifies that this
        particular environment has been implemented with a
        Markov transition model.
    """
    probabilities = []

    for next_state in environment.get_states():

        probability = environment.transition_probability(
            state,
            action,
            next_state
        )

        probabilities.append(probability)

    probabilities = np.asarray(probabilities)

    return {
        "probabilities": probabilities,
        "sum": float(np.sum(probabilities)),
        "nonzero_count": int(
            np.count_nonzero(probabilities)
        )
    }


# ================================================================
# STEP 6 — GOAL-DIRECTED BASELINE AGENT
# ================================================================

class ShortestPathAgent(Agent):
    """
    Goal-directed baseline using Breadth-First Search (BFS).

    This agent knows the environment layout and calculates a
    shortest path to the goal.

    It is included as a reference policy, not as an RL algorithm.

    Later, Q-learning, SARSA, and Deep RL agents will learn
    policies through interaction rather than being given
    a shortest-path solution.
    """

    def __init__(self, environment):
        self.environment = environment

    def select_action(self, state):

        start = tuple(state)
        goal = self.environment.goal_state

        if start == goal:
            return 0

        queue = deque([start])

        parent = {
            start: None
        }

        parent_action = {}

        found = False

        while queue:

            current = queue.popleft()

            if current == goal:
                found = True
                break

            for action in self.environment.get_actions():

                next_state = self.environment.get_next_state(
                    current,
                    action
                )

                if next_state == current:
                    continue

                if next_state in parent:
                    continue

                parent[next_state] = current
                parent_action[next_state] = action

                queue.append(next_state)

        if not found:
            raise RuntimeError("Goal is unreachable.")

        # Reconstruct the first action of the shortest path.
        current = goal

        while parent[current] != start:
            current = parent[current]

        return parent_action[current]


# ================================================================
# EXPERIMENT — RUN MULTIPLE EPISODES
# ================================================================

def evaluate_agent(
    environment,
    agent,
    episodes=30,
    seed_start=100
):
    """
    Evaluate an agent over multiple episodes.

    This produces:
        - Episode rewards.
        - Episode lengths.
        - Success rate.
        - Average reward.
    """
    results = []

    for episode_index in range(episodes):

        episode = run_episode(
            environment,
            agent,
            seed=seed_start + episode_index
        )

        results.append(episode)

    success_rate = np.mean([
        episode["terminated"]
        for episode in results
    ])

    average_reward = np.mean([
        episode["statistics"]["total_reward"]
        for episode in results
    ])

    average_steps = np.mean([
        episode["statistics"]["steps"]
        for episode in results
    ])

    return {
        "episodes": results,
        "success_rate": float(success_rate),
        "average_reward": float(average_reward),
        "average_steps": float(average_steps)
    }


# ================================================================
# STEP 7 — VISUALIZATION FUNCTIONS
# ================================================================

def visualize_environment(
    environment,
    episode=None,
    title="GridWorld Environment"
):
    """
    Visualize the environment and an optional agent trajectory.

    Cell meanings:
        White: Free cell
        Gray: Obstacle
        Green: Goal
        Blue: Start
    """

    size = environment.grid_size

    grid = np.zeros((size, size))

    for row, col in environment.obstacles:
        grid[row, col] = 1

    grid[environment.goal_state] = 2
    grid[environment.start_state] = 3

    cmap = ListedColormap([
        "white",
        "gray",
        "lightgreen",
        "lightblue"
    ])

    plt.figure(figsize=(6, 6))

    plt.imshow(
        grid,
        cmap=cmap,
        vmin=0,
        vmax=3
    )

    ax = plt.gca()

    ax.set_xticks(np.arange(-0.5, size, 1))
    ax.set_yticks(np.arange(-0.5, size, 1))

    ax.set_xticklabels([])
    ax.set_yticklabels([])

    ax.grid(
        color="black",
        linewidth=1
    )

    ax.set_title(title)

    for row, col in environment.obstacles:
        ax.text(
            col,
            row,
            "X",
            ha="center",
            va="center",
            fontsize=16
        )

    start_row, start_col = environment.start_state
    goal_row, goal_col = environment.goal_state

    ax.text(
        start_col,
        start_row,
        "S",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold"
    )

    ax.text(
        goal_col,
        goal_row,
        "G",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold"
    )

    if episode is not None:

        states = episode["states"]

        rows = [state[0] for state in states]
        cols = [state[1] for state in states]

        ax.plot(
            cols,
            rows,
            "o-",
            linewidth=2,
            markersize=4,
            label="Agent trajectory"
        )

        ax.legend(loc="upper left")

    plt.tight_layout()


def visualize_agent_comparison(
    random_results,
    shortest_results
):
    """
    Compare episode rewards for two different policies.
    """

    random_rewards = [
        episode["statistics"]["total_reward"]
        for episode in random_results["episodes"]
    ]

    shortest_rewards = [
        episode["statistics"]["total_reward"]
        for episode in shortest_results["episodes"]
    ]

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(9, 8)
    )

    axes[0].plot(
        random_rewards,
        label="Random agent"
    )

    axes[0].plot(
        shortest_rewards,
        label="Shortest-path agent"
    )

    axes[0].set_title("Episode Reward Comparison")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Total reward")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    labels = [
        "Random agent",
        "Shortest-path agent"
    ]

    success_rates = [
        random_results["success_rate"],
        shortest_results["success_rate"]
    ]

    axes[1].bar(
        labels,
        success_rates
    )

    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("Success rate")
    axes[1].set_title("Goal Reaching Success Rate")
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()


def visualize_episode_rewards(episode):
    """
    Show immediate reward at each time step and cumulative reward.

    Immediate reward:
        Feedback from one transition.

    Cumulative reward:
        Sum of all rewards received so far.
    """
    rewards = np.asarray(episode["rewards"])

    cumulative_rewards = np.cumsum(rewards)

    steps = np.arange(1, len(rewards) + 1)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(8, 7)
    )

    axes[0].bar(
        steps,
        rewards
    )

    axes[0].set_title("Immediate Reward per Step")
    axes[0].set_xlabel("Time step")
    axes[0].set_ylabel("Reward")
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].plot(
        steps,
        cumulative_rewards,
        marker="o"
    )

    axes[1].set_title("Cumulative Reward")
    axes[1].set_xlabel("Time step")
    axes[1].set_ylabel("Cumulative reward")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()


def visualize_transition_sequence(episode):
    """
    Visualize the sequence:

        State -> Action -> Reward -> Next State

    This is the basic data structure used to represent
    experience in many RL algorithms.
    """
    states = episode["states"]
    actions = episode["actions"]
    rewards = episode["rewards"]

    count = min(len(actions), 12)

    fig, ax = plt.subplots(
        figsize=(12, 4)
    )

    ax.axis("off")

    lines = []

    for i in range(count):

        lines.append(
            f"t={i}: {states[i]} "
            f"--{ACTION_NAMES[actions[i]]}--> "
            f"{states[i + 1]} "
            f"| reward={rewards[i]:.2f}"
        )

    text_block = "\n".join(lines)

    ax.text(
        0.02,
        0.95,
        text_block,
        transform=ax.transAxes,
        verticalalignment="top",
        family="monospace",
        fontsize=10
    )

    ax.set_title(
        "Agent–Environment Transition Sequence"
    )

    plt.tight_layout()


# ================================================================
# UNIT TESTS
# ================================================================

def run_tests():

    print("\n========== UNIT TESTS ==========\n")

    environment = GridWorld()

    # Test 1: State space
    states = environment.get_states()

    assert len(states) == (
        GRID_SIZE * GRID_SIZE - len(OBSTACLES)
    )

    print("[PASS] State space construction")

    # Test 2: Action space
    assert len(environment.get_actions()) == 4

    print("[PASS] Action space construction")

    # Test 3: Reset
    state, info = environment.reset()

    assert state == START_STATE
    assert info["is_start_state"] is True

    print("[PASS] Environment reset")

    # Test 4: Normal movement
    next_state = environment.get_next_state(
        (0, 0),
        3
    )

    assert next_state == (0, 1)

    print("[PASS] Normal state transition")

    # Test 5: Boundary handling
    next_state = environment.get_next_state(
        (0, 0),
        0
    )

    assert next_state == (0, 0)

    print("[PASS] Boundary handling")

    # Test 6: Obstacle handling
    next_state = environment.get_next_state(
        (0, 1),
        1
    )

    assert next_state == (0, 1)

    print("[PASS] Obstacle handling")

    # Test 7: Transition probability
    result = verify_markov_property(
        environment,
        (0, 0),
        3
    )

    assert np.isclose(result["sum"], 1.0)
    assert result["nonzero_count"] == 1

    print("[PASS] Transition probability")

    # Test 8: Reward function
    reward = environment.reward_function(
        (3, 4),
        1,
        (4, 4)
    )

    assert np.isclose(reward, GOAL_REWARD)

    print("[PASS] Reward function")

    # Test 9: Discounted return
    rewards = [1.0, 1.0, 1.0]

    discounted = calculate_discounted_return(
        rewards,
        gamma=0.9
    )

    expected = 1.0 + 0.9 + 0.81

    assert np.isclose(discounted, expected)

    print("[PASS] Discounted return")

    # Test 10: Goal-directed agent
    agent = ShortestPathAgent(environment)

    episode = run_episode(
        environment,
        agent
    )

    assert episode["terminated"] is True
    assert episode["states"][-1] == GOAL_STATE

    print("[PASS] Goal-directed agent reaches goal")

    # Test 11: Episode length
    assert episode["statistics"]["steps"] > 0
    assert episode["statistics"]["steps"] < MAX_STEPS

    print("[PASS] Episode termination")

    # Test 12: Terminal state protection
    environment.reset()

    environment.state = GOAL_STATE
    environment.terminated = True

    try:
        environment.step(0)
        raise AssertionError(
            "Expected RuntimeError after termination."
        )

    except RuntimeError:
        pass

    print("[PASS] Terminal state protection")

    # Test 13: Discount factor
    assert 0.0 <= environment.gamma <= 1.0

    print("[PASS] Discount factor")

    # Test 14: Agent action validity
    random_agent = RandomAgent(seed=42)

    action = random_agent.select_action(START_STATE)

    assert action in environment.get_actions()

    print("[PASS] Random agent action validity")

    print("\nAll 14 tests passed successfully.\n")


# ================================================================
# MAIN
# ================================================================

def main():

    print("\n============================================")
    print("RL_Course — Phase 1.1")
    print("Agent–Environment Interaction and MDP")
    print("============================================")

    environment = GridWorld()

    # ------------------------------------------------------------
    # STEP 1 — Display MDP components
    # ------------------------------------------------------------

    print("\n[STEP 1] MDP Definition")

    print("State space size:", len(environment.get_states()))
    print("Action space:", ACTION_NAMES)
    print("Start state:", START_STATE)
    print("Goal state:", GOAL_STATE)
    print("Obstacles:", OBSTACLES)
    print("Discount factor:", environment.gamma)

    # ------------------------------------------------------------
    # STEP 2 — Transition probability
    # ------------------------------------------------------------

    print("\n[STEP 2] Transition Probability")

    state = (0, 0)
    action = 3

    next_state = environment.get_next_state(
        state,
        action
    )

    probability = environment.transition_probability(
        state,
        action,
        next_state
    )

    print("Current state:", state)
    print("Action:", ACTION_NAMES[action])
    print("Next state:", next_state)
    print("Transition probability:", probability)

    # ------------------------------------------------------------
    # STEP 3 — Run one goal-directed episode
    # ------------------------------------------------------------

    print("\n[STEP 3] Goal-Directed Agent Episode")

    shortest_agent = ShortestPathAgent(environment)

    shortest_episode = run_episode(
        environment,
        shortest_agent,
        verbose=True
    )

    # ------------------------------------------------------------
    # STEP 4 — Run random agent episodes
    # ------------------------------------------------------------

    print("\n[STEP 4] Random Agent Evaluation")

    random_agent = RandomAgent(seed=42)

    random_results = evaluate_agent(
        environment,
        random_agent,
        episodes=30
    )

    print(
        "Random agent success rate:",
        f"{random_results['success_rate']:.2%}"
    )

    print(
        "Random agent average reward:",
        f"{random_results['average_reward']:.3f}"
    )

    print(
        "Random agent average steps:",
        f"{random_results['average_steps']:.2f}"
    )

    # ------------------------------------------------------------
    # STEP 5 — Evaluate shortest-path baseline
    # ------------------------------------------------------------

    print("\n[STEP 5] Shortest-Path Agent Evaluation")

    shortest_results = evaluate_agent(
        environment,
        shortest_agent,
        episodes=30
    )

    print(
        "Shortest-path success rate:",
        f"{shortest_results['success_rate']:.2%}"
    )

    print(
        "Shortest-path average reward:",
        f"{shortest_results['average_reward']:.3f}"
    )

    print(
        "Shortest-path average steps:",
        f"{shortest_results['average_steps']:.2f}"
    )

    # ------------------------------------------------------------
    # STEP 6 — Markov property
    # ------------------------------------------------------------

    print("\n[STEP 6] Markov Property Verification")

    markov_result = verify_markov_property(
        environment,
        (0, 0),
        3
    )

    print(
        "Sum of transition probabilities:",
        markov_result["sum"]
    )

    print(
        "Number of possible next states:",
        markov_result["nonzero_count"]
    )

    # ------------------------------------------------------------
    # STEP 7 — Automated tests and visualizations
    # ------------------------------------------------------------

    print("\n[STEP 7] Unit Tests")

    run_tests()

    print("Generating visualizations...")

    visualize_environment(
        environment,
        shortest_episode,
        title="GridWorld — Shortest-Path Trajectory"
    )

    visualize_agent_comparison(
        random_results,
        shortest_results
    )

    visualize_episode_rewards(
        shortest_episode
    )

    visualize_transition_sequence(
        shortest_episode
    )

    plt.show()

    print("\nPhase 1.1 completed successfully.")


if __name__ == "__main__":
    main()
