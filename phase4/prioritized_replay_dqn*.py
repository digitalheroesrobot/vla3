
# ============================================================
# File: phase4/prioritized_replay_dqn.py
#
# PHASE 4.5 — PRIORITIZED EXPERIENCE REPLAY (PER)
#
# PURPOSE
#   Improve Dueling Double DQN training by sampling important
#   experiences more frequently than ordinary uniform replay.
#
# WHAT YOU LEARN
#
# STEP 1 — TD ERROR
#   delta = target_Q - current_Q
#
# STEP 2 — PRIORITY
#   p_i = |delta_i| + epsilon
#
# STEP 3 — PRIORITIZED SAMPLING
#   P(i) = p_i^alpha / sum_j(p_j^alpha)
#
# STEP 4 — IMPORTANCE-SAMPLING WEIGHTS
#   w_i = (N * P(i))^(-beta)
#
#   Normalize:
#   w_i = w_i / max_j(w_j)
#
# STEP 5 — WEIGHTED LOSS
#   L = mean(w_i * TD_loss_i)
#
# STEP 6 — PRIORITY UPDATE
#   After learning, update each sampled transition's priority
#   using its newly calculated absolute TD error.
#
# STEP 7 — BETA ANNEALING
#   Increase beta gradually toward 1.0 to reduce sampling bias.
#
# NETWORK
#   Dueling Double DQN:
#       Online network selects next action.
#       Target network evaluates selected action.
#
# ROBOTICS APPLICATION
#   A robot's near-collision experiences, successful turns,
#   and difficult navigation situations can be sampled more
#   frequently than routine transitions.
#
# IMPORTANT
#   Large TD error does not always mean an experience is useful.
#   PER can repeatedly sample noisy or unusual transitions.
#   Importance-sampling weights help correct the resulting bias.
#
# PYGAME CONTROLS
#   Left click : Set target
#   SPACE      : Pause / resume
#   R          : Reset rocket
#   + / -      : Simulation speed
#   ESC        : Exit
#
# LEARNING CYCLE
#   Understand -> Implement -> Visualize -> Verify
#       -> Experiment -> Apply
#
# INSTALL
#   pip install numpy pygame torch
#
# RUN
#   python phase4/prioritized_replay_dqn.py
# ============================================================

from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass

import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.optim as optim


# ============================================================
# STEP 0 — CONFIGURATION
# ============================================================

SEED = 42

WORLD_WIDTH = 700
WORLD_HEIGHT = 560

WINDOW_WIDTH = 1020
WINDOW_HEIGHT = 650

OBSTACLE = np.array([350.0, 280.0], dtype=np.float32)
OBSTACLE_RADIUS = 55.0

ROCKET_RADIUS = 11.0
SAFETY_MARGIN = 20.0
SAFE_DISTANCE = OBSTACLE_RADIUS + ROCKET_RADIUS + SAFETY_MARGIN

MOVE_STEP = 8.0
MAX_EPISODE_STEPS = 170

STATE_SIZE = 10
NUM_ACTIONS = 9
HIDDEN_SIZE = 128

GAMMA = 0.98
LEARNING_RATE = 0.0005

REPLAY_CAPACITY = 50_000
BATCH_SIZE = 64
WARMUP_STEPS = 1_000

TRAIN_EPISODES = 350
TARGET_UPDATE_INTERVAL = 300

EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY_STEPS = 35_000

# PER parameters.
PER_ALPHA = 0.6
PER_BETA_START = 0.4
PER_BETA_END = 1.0
PER_BETA_STEPS = 50_000

PER_EPSILON = 1e-5

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

ACTION_VECTORS = np.array(
    [
        [1.0, 0.0],       # RIGHT
        [1.0, -1.0],      # UP-RIGHT
        [0.0, -1.0],      # UP
        [-1.0, -1.0],     # UP-LEFT
        [-1.0, 0.0],      # LEFT
        [-1.0, 1.0],      # DOWN-LEFT
        [0.0, 1.0],       # DOWN
        [1.0, 1.0],       # DOWN-RIGHT
        [0.0, 0.0],       # STAY
    ],
    dtype=np.float32,
)

for i in range(8):
    ACTION_VECTORS[i] /= np.linalg.norm(ACTION_VECTORS[i])

ACTION_NAMES = [
    "RIGHT",
    "UP-RIGHT",
    "UP",
    "UP-LEFT",
    "LEFT",
    "DOWN-LEFT",
    "DOWN",
    "DOWN-RIGHT",
    "STAY",
]


# ============================================================
# STEP 1 — REPRODUCIBILITY
# ============================================================

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# STEP 2 — GEOMETRY AND SAFETY
# ============================================================

def distance(a, b):
    return float(np.linalg.norm(a - b))


def is_inside_world(position):
    return (
        ROCKET_RADIUS <= position[0] <= WORLD_WIDTH - ROCKET_RADIUS
        and ROCKET_RADIUS <= position[1] <= WORLD_HEIGHT - ROCKET_RADIUS
    )


def is_safe_position(position):
    return distance(position, OBSTACLE) >= SAFE_DISTANCE


def candidate_position(position, action):
    return position + ACTION_VECTORS[action] * MOVE_STEP


def calculate_action_mask(position):
    """
    Return True for actions that pass the immediate safety check.

    The safety mask is applied during both training and
    interactive evaluation.

    This is a one-step filter, not a complete path planner.
    """
    mask = np.zeros(NUM_ACTIONS, dtype=bool)

    for action in range(8):
        next_position = candidate_position(position, action)

        mask[action] = (
            is_inside_world(next_position)
            and is_safe_position(next_position)
        )

    mask[8] = True
    return mask


def random_valid_position():
    for _ in range(10_000):
        p = np.array(
            [
                random.uniform(35, WORLD_WIDTH - 35),
                random.uniform(35, WORLD_HEIGHT - 35),
            ],
            dtype=np.float32,
        )

        if is_safe_position(p):
            return p

    raise RuntimeError("Cannot generate a safe position.")


# ============================================================
# STEP 3 — NAVIGATION ENVIRONMENT
# ============================================================

@dataclass
class StepResult:
    state: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    info: dict


class RocketNavigationEnv:

    def __init__(self):
        self.position = np.array([80.0, 280.0], dtype=np.float32)
        self.target = np.array([620.0, 280.0], dtype=np.float32)

        self.previous_action = 8
        self.steps = 0
        self.total_reward = 0.0
        self.success = False
        self.collision = False

    def set_scenario(self, start, target):
        start = np.array(start, dtype=np.float32)
        target = np.array(target, dtype=np.float32)

        if not is_inside_world(start) or not is_inside_world(target):
            raise ValueError("Start or target is outside the world.")

        if not is_safe_position(start) or not is_safe_position(target):
            raise ValueError("Start or target violates safety distance.")

        self.position = start.copy()
        self.target = target.copy()

        self.previous_action = 8
        self.steps = 0
        self.total_reward = 0.0
        self.success = False
        self.collision = False

        return self.get_state()

    def reset(self):
        start = random_valid_position()
        target = random_valid_position()

        while distance(start, target) < 250:
            target = random_valid_position()

        return self.set_scenario(start, target)

    def get_state(self):
        target_delta = self.target - self.position
        obstacle_delta = OBSTACLE - self.position

        target_distance = np.linalg.norm(target_delta)
        obstacle_distance = np.linalg.norm(obstacle_delta)

        clearance = obstacle_distance - SAFE_DISTANCE

        previous_vector = ACTION_VECTORS[self.previous_action]

        return np.array(
            [
                target_delta[0] / WORLD_WIDTH,
                target_delta[1] / WORLD_HEIGHT,
                target_distance / math.hypot(
                    WORLD_WIDTH,
                    WORLD_HEIGHT,
                ),

                obstacle_delta[0] / WORLD_WIDTH,
                obstacle_delta[1] / WORLD_HEIGHT,
                clearance / math.hypot(
                    WORLD_WIDTH,
                    WORLD_HEIGHT,
                ),

                self.position[0] / WORLD_WIDTH,
                self.position[1] / WORLD_HEIGHT,

                previous_vector[0],
                previous_vector[1],
            ],
            dtype=np.float32,
        )

    def action_mask(self):
        return calculate_action_mask(self.position)

    def step(self, action):
        if not 0 <= action < NUM_ACTIONS:
            raise ValueError("Invalid action.")

        old_distance = distance(self.position, self.target)
        next_position = candidate_position(self.position, action)

        reward = -0.04
        terminated = False
        truncated = False

        if not is_inside_world(next_position):
            reward -= 5.0
            terminated = True
            self.collision = True

        elif not is_safe_position(next_position):
            reward -= 15.0
            terminated = True
            self.collision = True

        else:
            self.position = next_position

            new_distance = distance(self.position, self.target)

            # Reward progress toward the target.
            reward += 0.12 * (old_distance - new_distance)

            # Discourage unnecessary direct reversals.
            old_vector = ACTION_VECTORS[self.previous_action]
            new_vector = ACTION_VECTORS[action]

            if (
                self.previous_action != 8
                and action != 8
                and np.dot(old_vector, new_vector) < -0.5
            ):
                reward -= 0.08

            self.previous_action = action

            if new_distance <= 18.0:
                reward += 15.0
                terminated = True
                self.success = True

        self.steps += 1

        if self.steps >= MAX_EPISODE_STEPS and not terminated:
            truncated = True

        self.total_reward += reward

        return StepResult(
            state=self.get_state(),
            reward=float(reward),
            terminated=terminated,
            truncated=truncated,
            info={
                "success": self.success,
                "collision": self.collision,
                "distance": distance(self.position, self.target),
            },
        )


# ============================================================
# STEP 4 — DUELING DQN NETWORK
# ============================================================

class DuelingDQN(nn.Module):
    """
    Shared feature representation:
        state -> feature layers

    Two output streams:
        V(s): state value
        A(s,a): action advantage

    Combination:
        Q(s,a) = V(s) + A(s,a) - mean_a A(s,a)
    """

    def __init__(self):
        super().__init__()

        self.feature_layer = nn.Sequential(
            nn.Linear(STATE_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
        )

        self.value_stream = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, 64),
            nn.ReLU(),
            nn.Linear(64, NUM_ACTIONS),
        )

    def forward(self, state):
        features = self.feature_layer(state)

        value = self.value_stream(features)
        advantage = self.advantage_stream(features)

        return (
            value
            + advantage
            - advantage.mean(dim=1, keepdim=True)
        )


# ============================================================
# STEP 5 — PRIORITIZED EXPERIENCE REPLAY
# ============================================================

@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    terminated: bool
    next_mask: np.ndarray


class PrioritizedReplayBuffer:
    """
    Proportional Prioritized Experience Replay.

    Priority:
        p_i = |TD_error_i| + epsilon

    Sampling probability:
        P(i) = p_i^alpha / sum_j(p_j^alpha)

    Importance-sampling weight:
        w_i = (N * P(i))^(-beta)

    Normalization:
        w_i = w_i / max_j(w_j)

    The buffer stores:
        - transitions
        - priority values

    A newly inserted transition receives the current maximum
    priority so that it has a good chance of being sampled.
    """

    def __init__(
        self,
        capacity,
        alpha=PER_ALPHA,
        epsilon=PER_EPSILON,
    ):
        self.capacity = capacity
        self.alpha = alpha
        self.epsilon = epsilon

        self.memory = []
        self.priorities = np.zeros(
            capacity,
            dtype=np.float32,
        )

        self.position = 0
        self.max_priority = 1.0

    def add(self, transition):
        if len(self.memory) < self.capacity:
            self.memory.append(transition)
        else:
            self.memory[self.position] = transition

        # New transitions get maximum current priority.
        self.priorities[self.position] = self.max_priority

        self.position = (self.position + 1) % self.capacity

    def probabilities(self):
        size = len(self.memory)

        if size == 0:
            raise ValueError("Cannot sample an empty buffer.")

        scaled_priorities = (
            self.priorities[:size] ** self.alpha
        )

        total = float(np.sum(scaled_priorities))

        if not np.isfinite(total) or total <= 0:
            return np.full(
                size,
                1.0 / size,
                dtype=np.float64,
            )

        return scaled_priorities / total

    def sample(self, batch_size, beta):
        size = len(self.memory)

        if size < batch_size:
            raise ValueError("Not enough transitions to sample.")

        probabilities = self.probabilities()

        indices = np.random.choice(
            size,
            size=batch_size,
            replace=False,
            p=probabilities,
        )

        selected_probabilities = probabilities[indices]

        # Importance-sampling correction.
        weights = (
            size * selected_probabilities
        ) ** (-beta)

        weights /= np.max(weights)

        batch = [self.memory[i] for i in indices]

        states = np.stack([t.state for t in batch])
        actions = np.array(
            [t.action for t in batch],
            dtype=np.int64,
        )
        rewards = np.array(
            [t.reward for t in batch],
            dtype=np.float32,
        )
        next_states = np.stack(
            [t.next_state for t in batch]
        )
        terminated = np.array(
            [t.terminated for t in batch],
            dtype=np.float32,
        )
        next_masks = np.stack(
            [t.next_mask for t in batch]
        )

        return (
            states,
            actions,
            rewards,
            next_states,
            terminated,
            next_masks,
            indices,
            weights.astype(np.float32),
        )

    def update_priorities(self, indices, td_errors):
        """
        Update sampled priorities using the latest TD errors.

        A small epsilon prevents zero-probability transitions.
        """
        td_errors = np.asarray(
            td_errors,
            dtype=np.float32,
        )

        new_priorities = (
            np.abs(td_errors) + self.epsilon
        )

        for index, priority in zip(indices, new_priorities):
            priority = float(priority)

            if not np.isfinite(priority):
                priority = self.max_priority

            self.priorities[index] = priority

            self.max_priority = max(
                self.max_priority,
                priority,
            )

    def __len__(self):
        return len(self.memory)

    def priority_statistics(self):
        if len(self.memory) == 0:
            return {
                "mean": 0.0,
                "max": 0.0,
                "min": 0.0,
            }

        p = self.priorities[:len(self.memory)]

        return {
            "mean": float(np.mean(p)),
            "max": float(np.max(p)),
            "min": float(np.min(p)),
        }


# ============================================================
# STEP 6 — EPSILON AND BETA SCHEDULES
# ============================================================

def epsilon_by_step(step):
    fraction = min(step / EPSILON_DECAY_STEPS, 1.0)

    return (
        EPSILON_START
        + fraction * (EPSILON_END - EPSILON_START)
    )


def beta_by_step(step):
    """
    Gradually increase beta from 0.4 to 1.0.

    Early training:
        Partial importance-sampling correction.

    Later training:
        Stronger correction, approaching full correction.
    """
    fraction = min(step / PER_BETA_STEPS, 1.0)

    return (
        PER_BETA_START
        + fraction * (PER_BETA_END - PER_BETA_START)
    )


# ============================================================
# STEP 7 — ACTION SELECTION
# ============================================================

def select_action(q_values, mask, epsilon):
    valid_actions = np.flatnonzero(mask)

    if len(valid_actions) == 0:
        return 8

    if random.random() < epsilon:
        return int(random.choice(valid_actions))

    valid_q = q_values[valid_actions]

    return int(valid_actions[np.argmax(valid_q)])


def choose_stable_action(
    q_values,
    mask,
    previous_action,
    hysteresis=0.12,
):
    valid_actions = np.flatnonzero(mask)

    if len(valid_actions) == 0:
        return 8

    best_action = int(
        valid_actions[
            np.argmax(q_values[valid_actions])
        ]
    )

    if (
        previous_action in valid_actions
        and previous_action != 8
        and best_action != 8
    ):
        if (
            q_values[previous_action]
            >= q_values[best_action] - hysteresis
        ):
            return int(previous_action)

    return best_action


# ============================================================
# STEP 8 — DUELING DOUBLE DQN + PER AGENT
# ============================================================

class PERDuelingDoubleDQNAgent:

    def __init__(self):
        self.policy_net = DuelingDQN().to(DEVICE)
        self.target_net = DuelingDQN().to(DEVICE)

        self.target_net.load_state_dict(
            self.policy_net.state_dict()
        )

        self.target_net.eval()

        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=LEARNING_RATE,
        )

        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=REPLAY_CAPACITY,
        )

        self.training_steps = 0
        self.loss_history = []
        self.td_error_history = []

    def predict(self, state):
        self.policy_net.eval()

        with torch.no_grad():
            state_t = torch.tensor(
                state,
                dtype=torch.float32,
                device=DEVICE,
            ).unsqueeze(0)

            q_values = self.policy_net(state_t)

        self.policy_net.train()

        return q_values.squeeze(0).cpu().numpy()

    def remember(
        self,
        state,
        action,
        reward,
        next_state,
        terminated,
        next_mask,
    ):
        transition = Transition(
            state=state.copy(),
            action=action,
            reward=reward,
            next_state=next_state.copy(),
            terminated=terminated,
            next_mask=next_mask.copy(),
        )

        self.replay_buffer.add(transition)

    def train_step(self, beta):
        if len(self.replay_buffer) < max(
            BATCH_SIZE,
            WARMUP_STEPS,
        ):
            return None

        (
            states,
            actions,
            rewards,
            next_states,
            terminated,
            next_masks,
            indices,
            importance_weights,
        ) = self.replay_buffer.sample(
            BATCH_SIZE,
            beta,
        )

        states_t = torch.tensor(
            states,
            dtype=torch.float32,
            device=DEVICE,
        )

        actions_t = torch.tensor(
            actions,
            dtype=torch.int64,
            device=DEVICE,
        )

        rewards_t = torch.tensor(
            rewards,
            dtype=torch.float32,
            device=DEVICE,
        )

        next_states_t = torch.tensor(
            next_states,
            dtype=torch.float32,
            device=DEVICE,
        )

        terminated_t = torch.tensor(
            terminated,
            dtype=torch.float32,
            device=DEVICE,
        )

        next_masks_t = torch.tensor(
            next_masks,
            dtype=torch.bool,
            device=DEVICE,
        )

        weights_t = torch.tensor(
            importance_weights,
            dtype=torch.float32,
            device=DEVICE,
        )

        # ----------------------------------------------------
        # 1. Current Q(s,a) from online network.
        # ----------------------------------------------------

        current_q_all = self.policy_net(states_t)

        current_q = current_q_all.gather(
            1,
            actions_t.unsqueeze(1),
        ).squeeze(1)

        # ----------------------------------------------------
        # 2. Double DQN next-state calculation.
        #
        # Online network selects the best valid next action.
        # Target network evaluates that action.
        # ----------------------------------------------------

        with torch.no_grad():
            online_next_q = self.policy_net(next_states_t)

            online_next_q = online_next_q.masked_fill(
                ~next_masks_t,
                -1e9,
            )

            next_actions = online_next_q.argmax(
                dim=1,
                keepdim=True,
            )

            target_next_q = self.target_net(next_states_t)

            selected_next_q = target_next_q.gather(
                1,
                next_actions,
            ).squeeze(1)

            targets = (
                rewards_t
                + GAMMA
                * (1.0 - terminated_t)
                * selected_next_q
            )

        # ----------------------------------------------------
        # 3. TD error.
        #
        # delta = target - prediction
        #
        # The TD error determines the updated priority.
        # ----------------------------------------------------

        td_errors = targets - current_q

        # ----------------------------------------------------
        # 4. PER-weighted loss.
        #
        # Each sampled transition has an importance weight.
        #
        # L = mean(w_i * SmoothL1(delta_i))
        # ----------------------------------------------------

        individual_losses = nn.functional.smooth_l1_loss(
            current_q,
            targets,
            reduction="none",
        )

        weighted_loss = (
            weights_t * individual_losses
        ).mean()

        self.optimizer.zero_grad()
        weighted_loss.backward()

        nn.utils.clip_grad_norm_(
            self.policy_net.parameters(),
            max_norm=5.0,
        )

        self.optimizer.step()

        # ----------------------------------------------------
        # 5. Update replay priorities.
        #
        # Use the TD errors calculated before this optimizer
        # update. They are detached from the computation graph.
        # ----------------------------------------------------

        self.replay_buffer.update_priorities(
            indices,
            td_errors.detach().cpu().numpy(),
        )

        self.training_steps += 1

        # ----------------------------------------------------
        # 6. Target network synchronization.
        # ----------------------------------------------------

        if self.training_steps % TARGET_UPDATE_INTERVAL == 0:
            self.update_target_network()

        loss_value = float(weighted_loss.item())

        self.loss_history.append(loss_value)

        self.td_error_history.append(
            float(td_errors.detach().abs().mean().item())
        )

        return loss_value

    def update_target_network(self):
        self.target_net.load_state_dict(
            self.policy_net.state_dict()
        )

    def save(self, filename="per_dueling_double_dqn.pth"):
        torch.save(
            {
                "policy_net": self.policy_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "training_steps": self.training_steps,
            },
            filename,
        )

    def load(self, filename="per_dueling_double_dqn.pth"):
        checkpoint = torch.load(
            filename,
            map_location=DEVICE,
        )

        self.policy_net.load_state_dict(
            checkpoint["policy_net"]
        )

        self.target_net.load_state_dict(
            checkpoint["target_net"]
        )

        self.optimizer.load_state_dict(
            checkpoint["optimizer"]
        )

        self.training_steps = checkpoint.get(
            "training_steps",
            0,
        )


# ============================================================
# STEP 9 — TRAINING
# ============================================================

def train_agent(
    agent,
    env,
    episodes=TRAIN_EPISODES,
):
    global_step = 0

    episode_rewards = []
    success_history = []
    collision_history = []
    episode_losses = []

    print("\n" + "=" * 70)
    print("PHASE 4.5 — PRIORITIZED EXPERIENCE REPLAY")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Architecture: Dueling Double DQN")
    print(f"Replay: Proportional PER")
    print(f"Alpha: {PER_ALPHA}")
    print(f"Beta start: {PER_BETA_START}")
    print(f"Beta end: {PER_BETA_END}")
    print(f"Episodes: {episodes}")
    print("=" * 70)

    for episode in range(episodes):
        state = env.reset()

        total_reward = 0.0
        losses = []

        for _ in range(MAX_EPISODE_STEPS):
            mask = env.action_mask()

            q_values = agent.predict(state)

            epsilon = epsilon_by_step(global_step)

            action = select_action(
                q_values,
                mask,
                epsilon,
            )

            result = env.step(action)

            next_mask = env.action_mask()

            agent.remember(
                state=state,
                action=action,
                reward=result.reward,
                next_state=result.state,
                terminated=result.terminated,
                next_mask=next_mask,
            )

            beta = beta_by_step(global_step)

            loss = agent.train_step(beta)

            if loss is not None:
                losses.append(loss)

            state = result.state
            total_reward += result.reward
            global_step += 1

            if result.terminated or result.truncated:
                break

        episode_rewards.append(total_reward)
        success_history.append(int(env.success))
        collision_history.append(int(env.collision))

        episode_losses.append(
            float(np.mean(losses)) if losses else 0.0
        )

        if (episode + 1) % 25 == 0 or episode == 0:
            start = max(0, episode - 24)

            avg_reward = float(
                np.mean(episode_rewards[start:episode + 1])
            )

            success_rate = float(
                np.mean(success_history[start:episode + 1])
            )

            collision_rate = float(
                np.mean(collision_history[start:episode + 1])
            )

            print(
                f"Episode {episode + 1:4d}/{episodes} | "
                f"Reward {avg_reward:8.2f} | "
                f"Success {success_rate * 100:5.1f}% | "
                f"Collision {collision_rate * 100:5.1f}% | "
                f"Epsilon {epsilon_by_step(global_step):.3f} | "
                f"Beta {beta_by_step(global_step):.3f}"
            )

    agent.update_target_network()

    print("\nTraining finished.")
    print(f"Environment steps: {global_step}")
    print(f"Training updates: {agent.training_steps}")

    return {
        "episode_rewards": episode_rewards,
        "success_history": success_history,
        "collision_history": collision_history,
        "episode_losses": episode_losses,
    }


# ============================================================
# STEP 10 — PYGAME ROCKET VISUALIZATION
# ============================================================

class RocketGUI:

    def __init__(self, env, agent, results):
        pygame.init()

        pygame.display.set_caption(
            "Phase 4.5 — PER Dueling Double DQN Rocket"
        )

        self.screen = pygame.display.set_mode(
            (WINDOW_WIDTH, WINDOW_HEIGHT)
        )

        self.clock = pygame.time.Clock()

        self.font = pygame.font.SysFont("consolas", 16)
        self.small_font = pygame.font.SysFont("consolas", 13)
        self.title_font = pygame.font.SysFont(
            "consolas",
            20,
            bold=True,
        )

        self.env = env
        self.agent = agent
        self.results = results

        self.paused = False
        self.running = True
        self.speed = 1

        self.current_action = 8
        self.previous_action = 8

        self.q_values = np.zeros(
            NUM_ACTIONS,
            dtype=np.float32,
        )

        self.mask = np.ones(
            NUM_ACTIONS,
            dtype=bool,
        )

        self.status = "READY"

        self.reset_robot()

    def reset_robot(self):
        self.env.set_scenario(
            np.array([80.0, 280.0], dtype=np.float32),
            np.array([620.0, 280.0], dtype=np.float32),
        )

        self.current_action = 8
        self.previous_action = 8

        self.status = "READY"

    def set_target(self, mouse_position):
        x, y = mouse_position

        if not (0 <= x <= WORLD_WIDTH and 0 <= y <= WORLD_HEIGHT):
            return

        target = np.array([x, y], dtype=np.float32)

        if not is_inside_world(target):
            self.status = "INVALID TARGET"
            return

        if not is_safe_position(target):
            self.status = "TARGET TOO CLOSE TO OBSTACLE"
            return

        if distance(target, self.env.position) < 20:
            self.status = "TARGET TOO CLOSE TO ROCKET"
            return

        self.env.target = target
        self.env.steps = 0
        self.env.total_reward = 0.0
        self.env.success = False
        self.env.collision = False
        self.env.previous_action = 8

        self.current_action = 8
        self.previous_action = 8

        self.status = "NEW TARGET SET"

    def draw_text(
        self,
        message,
        x,
        y,
        color=(230, 230, 230),
        font=None,
    ):
        if font is None:
            font = self.font

        surface = font.render(str(message), True, color)

        # Pygame blit requires integer destination coordinates.
        # Convert NumPy floats and Python floats to integers.
        x = int(x)
        y = int(y)

        self.screen.blit(surface, (x, y))

    def draw_rocket(self, x, y):
        """
        Rocket-shaped robot.
        Nose direction follows the selected action.
        """
        vector = ACTION_VECTORS[self.current_action]

        if np.linalg.norm(vector) < 0.01:
            vector = ACTION_VECTORS[self.previous_action]

        if np.linalg.norm(vector) < 0.01:
            vector = np.array([1.0, 0.0])

        angle = math.atan2(vector[1], vector[0])

        local_points = [
            (20, 0),
            (5, -9),
            (-9, -7),
            (-13, -4),
            (-13, 4),
            (-9, 7),
            (5, 9),
        ]

        def rotate(point):
            lx, ly = point

            rx = lx * math.cos(angle) - ly * math.sin(angle)
            ry = lx * math.sin(angle) + ly * math.cos(angle)

            return int(x + rx), int(y + ry)

        rocket_points = [rotate(p) for p in local_points]

        flame_length = random.randint(9, 16)

        flame_points = [
            rotate((-13, -4)),
            rotate((-13 - flame_length, 0)),
            rotate((-13, 4)),
        ]

        pygame.draw.polygon(
            self.screen,
            (255, 145, 35),
            flame_points,
        )

        pygame.draw.polygon(
            self.screen,
            (235, 240, 250),
            rocket_points,
        )

        pygame.draw.polygon(
            self.screen,
            (70, 150, 230),
            rocket_points,
            2,
        )

        pygame.draw.circle(
            self.screen,
            (50, 150, 245),
            rotate((7, 0)),
            4,
        )

        pygame.draw.polygon(
            self.screen,
            (220, 65, 65),
            [
                rotate((-9, -6)),
                rotate((-17, -12)),
                rotate((-12, -2)),
            ],
        )

        pygame.draw.polygon(
            self.screen,
            (220, 65, 65),
            [
                rotate((-9, 6)),
                rotate((-17, 12)),
                rotate((-12, 2)),
            ],
        )

    def draw_world(self):
        self.screen.fill((18, 22, 30))

        pygame.draw.rect(
            self.screen,
            (28, 34, 44),
            (0, 0, WORLD_WIDTH, WORLD_HEIGHT),
        )

        for x in range(0, WORLD_WIDTH, 40):
            pygame.draw.line(
                self.screen,
                (39, 46, 58),
                (x, 0),
                (x, WORLD_HEIGHT),
                1,
            )

        for y in range(0, WORLD_HEIGHT, 40):
            pygame.draw.line(
                self.screen,
                (39, 46, 58),
                (0, y),
                (WORLD_WIDTH, y),
                1,
            )

        center = tuple(OBSTACLE.astype(int))

        pygame.draw.circle(
            self.screen,
            (130, 75, 55),
            center,
            int(SAFE_DISTANCE),
            2,
        )

        pygame.draw.circle(
            self.screen,
            (170, 55, 55),
            center,
            int(OBSTACLE_RADIUS),
        )

        self.draw_text(
            "OBSTACLE",
            OBSTACLE[0] - 40,
            OBSTACLE[1] - 12,
            (255, 220, 220),
            self.small_font,
        )

        tx, ty = self.env.target.astype(int)

        pygame.draw.circle(
            self.screen,
            (60, 230, 130),
            (tx, ty),
            13,
            2,
        )

        pygame.draw.line(
            self.screen,
            (60, 230, 130),
            (tx - 18, ty),
            (tx + 18, ty),
            2,
        )

        pygame.draw.line(
            self.screen,
            (60, 230, 130),
            (tx, ty - 18),
            (tx, ty + 18),
            2,
        )

        self.draw_text(
            "TARGET",
            tx + 15,
            ty - 25,
            (100, 255, 160),
            self.small_font,
        )

        px, py = self.env.position.astype(int)

        pygame.draw.circle(
            self.screen,
            (90, 130, 210),
            (px, py),
            19,
            1,
        )

        self.draw_rocket(px, py)

    def draw_q_values(self):
        x0 = 720
        y0 = 275

        self.draw_text(
            "PER DQN ACTION Q-VALUES",
            x0,
            y0,
            (120, 210, 255),
            self.title_font,
        )

        valid_values = self.q_values[self.mask]

        if len(valid_values):
            low = float(np.min(valid_values))
            high = float(np.max(valid_values))
        else:
            low, high = -1.0, 1.0

        span = max(high - low, 1e-6)

        for i, name in enumerate(ACTION_NAMES):
            y = y0 + 33 + i * 26

            valid = bool(self.mask[i])
            selected = i == self.current_action

            color = (
                (100, 255, 150)
                if selected
                else (210, 210, 210)
            )

            if not valid:
                color = (100, 100, 100)

            self.draw_text(
                f"{name:10s}",
                x0,
                y,
                color,
                self.small_font,
            )

            self.draw_text(
                f"{self.q_values[i]:8.3f}",
                x0 + 105,
                y,
                color,
                self.small_font,
            )

            if valid:
                width = int(
                    90 * (self.q_values[i] - low) / span
                )

                pygame.draw.rect(
                    self.screen,
                    (60, 120, 190),
                    (x0 + 205, y + 3, width, 10),
                )

            else:
                pygame.draw.line(
                    self.screen,
                    (100, 100, 100),
                    (x0 + 205, y + 8),
                    (x0 + 270, y + 8),
                    2,
                )

    def draw_reward_curve(self):
        x0 = 720
        y0 = 555
        width = 260
        height = 55

        self.draw_text(
            "TRAINING REWARD",
            x0,
            y0 - 20,
            (120, 210, 255),
            self.small_font,
        )

        pygame.draw.rect(
            self.screen,
            (50, 60, 75),
            (x0, y0, width, height),
            1,
        )

        rewards = self.results["episode_rewards"]

        if len(rewards) < 2:
            return

        window = 15

        smoothed = [
            np.mean(rewards[max(0, i - window + 1):i + 1])
            for i in range(len(rewards))
        ]

        low = float(min(smoothed))
        high = float(max(smoothed))
        span = max(high - low, 1e-6)

        points = []

        for i, value in enumerate(smoothed):
            px = x0 + int(
                i / max(len(smoothed) - 1, 1) * width
            )

            py = y0 + height - int(
                (value - low) / span * (height - 6)
            )

            points.append((px, py))

        if len(points) > 1:
            pygame.draw.lines(
                self.screen,
                (70, 220, 150),
                False,
                points,
                2,
            )

    def draw_panel(self):
        pygame.draw.rect(
            self.screen,
            (22, 27, 37),
            (
                WORLD_WIDTH,
                0,
                WINDOW_WIDTH - WORLD_WIDTH,
                WINDOW_HEIGHT,
            ),
        )

        self.draw_text(
            "PRIORITIZED REPLAY",
            720,
            18,
            (120, 210, 255),
            self.title_font,
        )

        self.draw_text(
            f"Device: {DEVICE}",
            720,
            50,
            font=self.small_font,
        )

        self.draw_text(
            f"Position: ({self.env.position[0]:.1f}, "
            f"{self.env.position[1]:.1f})",
            720,
            78,
            font=self.small_font,
        )

        self.draw_text(
            f"Target distance: "
            f"{distance(self.env.position, self.env.target):.1f}",
            720,
            103,
            font=self.small_font,
        )

        self.draw_text(
            f"Action: {ACTION_NAMES[self.current_action]}",
            720,
            128,
            (120, 255, 160),
            self.small_font,
        )

        self.draw_text(
            f"Steps: {self.env.steps}",
            720,
            153,
            font=self.small_font,
        )

        self.draw_text(
            f"Reward: {self.env.total_reward:.2f}",
            720,
            178,
            font=self.small_font,
        )

        self.draw_text(
            f"Replay size: {len(self.agent.replay_buffer)}",
            720,
            203,
            font=self.small_font,
        )

        self.draw_text(
            f"Training updates: {self.agent.training_steps}",
            720,
            228,
            font=self.small_font,
        )

        stats = self.agent.replay_buffer.priority_statistics()

        self.draw_text(
            f"Mean priority: {stats['mean']:.4f}",
            720,
            250,
            font=self.small_font,
        )

        self.draw_q_values()
        self.draw_reward_curve()

        self.draw_text(
            "CLICK: TARGET  SPACE: PAUSE  R: RESET",
            720,
            625,
            font=self.small_font,
        )

    def simulation_step(self):
        self.mask = self.env.action_mask()

        self.q_values = self.agent.predict(
            self.env.get_state()
        )

        self.current_action = choose_stable_action(
            self.q_values,
            self.mask,
            self.previous_action,
        )

        result = self.env.step(self.current_action)

        self.previous_action = self.current_action

        if result.terminated:
            if self.env.success:
                self.status = "TARGET REACHED"
            elif self.env.collision:
                self.status = "COLLISION"
            else:
                self.status = "TERMINATED"

        elif result.truncated:
            self.status = "STEP LIMIT REACHED"

    def run(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False

                    elif event.key == pygame.K_SPACE:
                        self.paused = not self.paused

                    elif event.key == pygame.K_r:
                        self.reset_robot()

                    elif event.key in (
                        pygame.K_PLUS,
                        pygame.K_EQUALS,
                        pygame.K_KP_PLUS,
                    ):
                        self.speed = min(self.speed + 1, 8)

                    elif event.key in (
                        pygame.K_MINUS,
                        pygame.K_KP_MINUS,
                    ):
                        self.speed = max(self.speed - 1, 1)

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        self.set_target(event.pos)

            if not self.paused:
                for _ in range(self.speed):
                    if (
                        self.env.success
                        or self.env.collision
                        or self.env.steps >= MAX_EPISODE_STEPS
                    ):
                        break

                    self.simulation_step()

            self.draw_world()
            self.draw_panel()

            if self.paused:
                self.draw_text(
                    "PAUSED",
                    300,
                    25,
                    (255, 220, 80),
                    self.title_font,
                )

            if self.status != "READY":
                self.draw_text(
                    self.status,
                    20,
                    WORLD_HEIGHT - 28,
                    (255, 220, 100),
                    self.small_font,
                )

            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()


# ============================================================
# STEP 11 — NUMERICAL VERIFICATION
# ============================================================

def run_tests():
    print("\n" + "=" * 70)
    print("PHASE 4.5 UNIT TESTS")
    print("=" * 70)

    passed = 0

    def check(name, condition):
        nonlocal passed

        assert condition, f"FAILED: {name}"

        passed += 1
        print(f"[PASS] {name}")

    # Test 1: Initial priority.
    replay = PrioritizedReplayBuffer(capacity=10)

    transition = Transition(
        state=np.zeros(STATE_SIZE, dtype=np.float32),
        action=0,
        reward=1.0,
        next_state=np.ones(STATE_SIZE, dtype=np.float32),
        terminated=False,
        next_mask=np.ones(NUM_ACTIONS, dtype=bool),
    )

    replay.add(transition)

    check(
        "New transition receives initial maximum priority",
        np.isclose(replay.priorities[0], 1.0),
    )

    # Test 2: Probability normalization.
    probabilities = replay.probabilities()

    check(
        "Sampling probabilities sum to one",
        np.isclose(np.sum(probabilities), 1.0),
    )

    # Test 3: Uniform priorities produce uniform probabilities.
    uniform_replay = PrioritizedReplayBuffer(
        capacity=10,
        alpha=0.6,
    )

    for _ in range(5):
        uniform_replay.add(transition)

    uniform_probabilities = uniform_replay.probabilities()

    check(
        "Equal priorities produce uniform sampling",
        np.allclose(
            uniform_probabilities,
            np.full(5, 0.2),
        ),
    )

    # Test 4: Higher priority produces higher probability.
    uniform_replay.update_priorities(
        indices=np.array([0, 1, 2, 3, 4]),
        td_errors=np.array([1, 1, 1, 1, 10]),
    )

    prioritized_probabilities = uniform_replay.probabilities()

    check(
        "Higher TD error produces higher sampling probability",
        prioritized_probabilities[4]
        > prioritized_probabilities[0],
    )

    # Test 5: Priority epsilon avoids zero priority.
    uniform_replay.update_priorities(
        indices=np.array([0]),
        td_errors=np.array([0.0]),
    )

    check(
        "Zero TD error retains positive priority",
        uniform_replay.priorities[0] > 0,
    )

    # Test 6: Sampling returns expected shapes.
    sampled = uniform_replay.sample(
        batch_size=3,
        beta=0.4,
    )

    states, actions, rewards, next_states, terminated, masks, indices, weights = sampled

    check(
        "Sampled state shape is (3, 10)",
        states.shape == (3, STATE_SIZE),
    )

    # Test 7: Importance weights are normalized.
    check(
        "Importance weights are positive and at most one",
        np.all(weights > 0)
        and np.all(weights <= 1.0 + 1e-6),
    )

    # Test 8: Sample indices are valid.
    check(
        "Sample indices are valid",
        np.all(indices >= 0)
        and np.all(indices < len(uniform_replay)),
    )

    # Test 9: Priority updates work.
    old_priority = uniform_replay.priorities[indices[0]]

    uniform_replay.update_priorities(
        indices=np.array([indices[0]]),
        td_errors=np.array([5.0]),
    )

    check(
        "Priority update changes sampled transition",
        uniform_replay.priorities[indices[0]] > old_priority,
    )

    # Test 10: Beta schedule starts correctly.
    check(
        "Beta schedule starts at 0.4",
        np.isclose(beta_by_step(0), PER_BETA_START),
    )

    # Test 11: Beta schedule reaches one.
    check(
        "Beta schedule reaches one",
        np.isclose(
            beta_by_step(PER_BETA_STEPS),
            PER_BETA_END,
        ),
    )

    # Test 12: Dueling network output.
    network = DuelingDQN().to(DEVICE)

    x = torch.zeros(
        (2, STATE_SIZE),
        dtype=torch.float32,
        device=DEVICE,
    )

    output = network(x)

    check(
        "Dueling network output shape is (2, 9)",
        tuple(output.shape) == (2, NUM_ACTIONS),
    )

    # Test 13: Environment state.
    env = RocketNavigationEnv()
    state = env.reset()

    check(
        "Environment state shape is (10,)",
        state.shape == (STATE_SIZE,),
    )

    # Test 14: Action mask.
    mask = env.action_mask()

    check(
        "Safety mask contains nine actions",
        mask.shape == (NUM_ACTIONS,),
    )

    # Test 15: Stay action.
    check(
        "Stay action remains available",
        bool(mask[8]),
    )

    # Test 16: Action vector normalization.
    norms = np.linalg.norm(ACTION_VECTORS[:8], axis=1)

    check(
        "Directional action vectors are normalized",
        np.allclose(norms, np.ones(8)),
    )

    # Test 17: Stable action selection.
    stable_action = choose_stable_action(
        q_values=np.array(
            [1.0, 0.95, 0.1, 0, 0, 0, 0, 0, -1],
            dtype=np.float32,
        ),
        mask=np.ones(NUM_ACTIONS, dtype=bool),
        previous_action=0,
    )

    check(
        "Hysteresis retains near-optimal previous action",
        stable_action == 0,
    )

    print("-" * 70)
    print(f"Tests passed: {passed}/17")
    print("=" * 70)


# ============================================================
# STEP 12 — MAIN
# ============================================================

def main():
    set_seed()

    run_tests()

    env = RocketNavigationEnv()
    agent = PERDuelingDoubleDQNAgent()

    results = train_agent(
        agent,
        env,
        episodes=TRAIN_EPISODES,
    )

    agent.save("per_dueling_double_dqn.pth")

    print("\nSaved model: per_dueling_double_dqn.pth")

    gui = RocketGUI(
        env=env,
        agent=agent,
        results=results,
    )

    gui.run()


if __name__ == "__main__":
    main()
