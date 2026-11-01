
# ============================================================
# File: phase4/noisy_network_dqn.py
#
# PHASE 4.6 — NOISY NETWORKS FOR DEEP RL
#
# PURPOSE
#   Replace epsilon-greedy exploration with learnable
#   parameter-space noise.
#
# WHAT YOU LEARN
#
# STEP 1 — NOISY LINEAR LAYER
#   y = (W + sigma_W * epsilon_W)x
#       + (b + sigma_b * epsilon_b)
#
# STEP 2 — FACTORIZED GAUSSIAN NOISE
#   f(x) = sign(x) * sqrt(|x|)
#
#   epsilon_W[i,j] = f(epsilon_i) * f(epsilon_j)
#
# STEP 3 — LEARNABLE NOISE SCALE
#   sigma_W and sigma_b are trainable parameters.
#
# STEP 4 — DUELING NOISY DQN
#   Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
#
# STEP 5 — DOUBLE DQN
#   Online network selects next action.
#   Target network evaluates selected action.
#
# STEP 6 — PRIORITIZED EXPERIENCE REPLAY
#   Sample transitions according to TD-error priorities.
#
# STEP 7 — NOISY EXPLORATION
#   Remove epsilon-greedy randomness.
#   Reset network noise to generate exploratory behavior.
#
# STEP 8 — VISUALIZATION
#   Interactive Pygame rocket navigation.
#
# ROBOTICS APPLICATION
#   A mobile robot often encounters unfamiliar situations.
#   Parameter-space noise can encourage coherent exploratory
#   behavior rather than independently random actions at
#   every time step.
#
# IMPORTANT
#   Noisy Networks do not guarantee safe exploration.
#   The geometric safety filter in this demonstration is
#   separate from the learned policy.
#
# PYGAME CONTROLS
#   Left click : Set target
#   SPACE      : Pause / resume
#   R          : Reset rocket
#   + / -      : Simulation speed
#   N          : Reset network noise
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
#   python phase4/noisy_network_dqn.py
# ============================================================

from __future__ import annotations

import math
import random
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
LEARNING_RATE = 0.0003

REPLAY_CAPACITY = 30_000
BATCH_SIZE = 64
WARMUP_STEPS = 800

TRAIN_EPISODES = 350
TARGET_UPDATE_INTERVAL = 300

# Noisy Network parameters.
NOISE_SIGMA = 0.5

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
        [1.0, 0.0],
        [1.0, -1.0],
        [0.0, -1.0],
        [-1.0, -1.0],
        [-1.0, 0.0],
        [-1.0, 1.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [0.0, 0.0],
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

    raise RuntimeError("Could not generate safe position.")


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
        self.position = np.array(
            [80.0, 280.0],
            dtype=np.float32,
        )

        self.target = np.array(
            [620.0, 280.0],
            dtype=np.float32,
        )

        self.previous_action = 8
        self.steps = 0
        self.total_reward = 0.0

        self.success = False
        self.collision = False

    def set_scenario(self, start, target):
        start = np.array(start, dtype=np.float32)
        target = np.array(target, dtype=np.float32)

        if not is_inside_world(start):
            raise ValueError("Invalid start position.")

        if not is_inside_world(target):
            raise ValueError("Invalid target position.")

        if not is_safe_position(start):
            raise ValueError("Start violates safety distance.")

        if not is_safe_position(target):
            raise ValueError("Target violates safety distance.")

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

        old_distance = distance(
            self.position,
            self.target,
        )

        next_position = candidate_position(
            self.position,
            action,
        )

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

            new_distance = distance(
                self.position,
                self.target,
            )

            reward += 0.12 * (
                old_distance - new_distance
            )

            old_vector = ACTION_VECTORS[self.previous_action]
            new_vector = ACTION_VECTORS[action]

            # Penalize unnecessary direct reversals.
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
                "distance": distance(
                    self.position,
                    self.target,
                ),
            },
        )


# ============================================================
# STEP 4 — NOISY LINEAR LAYER
# ============================================================

class NoisyLinear(nn.Module):
    """
    Factorized Gaussian Noisy Linear Layer.

    Standard linear layer:
        y = Wx + b

    Noisy linear layer:
        y = (W + sigma_W * epsilon_W)x
            + (b + sigma_b * epsilon_b)

    sigma parameters are learned by gradient descent.

    Factorized noise:
        f(x) = sign(x) * sqrt(|x|)

        epsilon_W[i,j] = f(epsilon_out[i])
                         * f(epsilon_in[j])

    This reduces the cost of sampling a full noise matrix.
    """

    def __init__(
        self,
        in_features,
        out_features,
        sigma_init=NOISE_SIGMA,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.weight_mu = nn.Parameter(
            torch.empty(out_features, in_features)
        )

        self.weight_sigma = nn.Parameter(
            torch.empty(out_features, in_features)
        )

        self.bias_mu = nn.Parameter(
            torch.empty(out_features)
        )

        self.bias_sigma = nn.Parameter(
            torch.empty(out_features)
        )

        self.register_buffer(
            "weight_epsilon",
            torch.empty(out_features, in_features),
        )

        self.register_buffer(
            "bias_epsilon",
            torch.empty(out_features),
        )

        self.reset_parameters(sigma_init)
        self.reset_noise()

    def reset_parameters(self, sigma_init):
        bound = 1.0 / math.sqrt(self.in_features)

        nn.init.uniform_(
            self.weight_mu,
            -bound,
            bound,
        )

        nn.init.uniform_(
            self.bias_mu,
            -bound,
            bound,
        )

        nn.init.constant_(
            self.weight_sigma,
            sigma_init / math.sqrt(self.in_features),
        )

        nn.init.constant_(
            self.bias_sigma,
            sigma_init / math.sqrt(self.out_features),
        )

    @staticmethod
    def scale_noise(size, device):
        x = torch.randn(
            size,
            device=device,
        )

        return x.sign() * x.abs().sqrt()

    def reset_noise(self):
        epsilon_in = self.scale_noise(
            self.in_features,
            self.weight_mu.device,
        )

        epsilon_out = self.scale_noise(
            self.out_features,
            self.weight_mu.device,
        )

        self.weight_epsilon.copy_(
            epsilon_out.outer(epsilon_in)
        )

        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x):
        if self.training:
            weight = (
                self.weight_mu
                + self.weight_sigma * self.weight_epsilon
            )

            bias = (
                self.bias_mu
                + self.bias_sigma * self.bias_epsilon
            )

        else:
            # Deterministic evaluation uses mean parameters.
            weight = self.weight_mu
            bias = self.bias_mu

        return nn.functional.linear(
            x,
            weight,
            bias,
        )


# ============================================================
# STEP 5 — DUELING NOISY DQN
# ============================================================

class NoisyDuelingDQN(nn.Module):

    def __init__(self):
        super().__init__()

        self.feature_layer = nn.Sequential(
            nn.Linear(STATE_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
        )

        self.value_stream = nn.Sequential(
            NoisyLinear(HIDDEN_SIZE, 64),
            nn.ReLU(),
            NoisyLinear(64, 1),
        )

        self.advantage_stream = nn.Sequential(
            NoisyLinear(HIDDEN_SIZE, 64),
            nn.ReLU(),
            NoisyLinear(64, NUM_ACTIONS),
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

    def reset_noise(self):
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()


# ============================================================
# STEP 6 — PRIORITIZED EXPERIENCE REPLAY
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

    def __init__(
        self,
        capacity,
        alpha=PER_ALPHA,
    ):
        self.capacity = capacity
        self.alpha = alpha

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

        self.priorities[self.position] = self.max_priority

        self.position = (
            self.position + 1
        ) % self.capacity

    def probabilities(self):
        size = len(self.memory)

        if size == 0:
            raise ValueError("Replay buffer is empty.")

        scaled = (
            self.priorities[:size] ** self.alpha
        )

        total = float(np.sum(scaled))

        if not np.isfinite(total) or total <= 0:
            return np.full(size, 1.0 / size)

        return scaled / total

    def sample(self, batch_size, beta):
        size = len(self.memory)

        if size < batch_size:
            raise ValueError("Insufficient replay samples.")

        probabilities = self.probabilities()

        indices = np.random.choice(
            size,
            batch_size,
            replace=False,
            p=probabilities,
        )

        selected_probabilities = probabilities[indices]

        weights = (
            size * selected_probabilities
        ) ** (-beta)

        weights /= weights.max()

        batch = [self.memory[i] for i in indices]

        return (
            np.stack([t.state for t in batch]),
            np.array(
                [t.action for t in batch],
                dtype=np.int64,
            ),
            np.array(
                [t.reward for t in batch],
                dtype=np.float32,
            ),
            np.stack([t.next_state for t in batch]),
            np.array(
                [t.terminated for t in batch],
                dtype=np.float32,
            ),
            np.stack([t.next_mask for t in batch]),
            indices,
            weights.astype(np.float32),
        )

    def update_priorities(self, indices, td_errors):
        for index, error in zip(indices, td_errors):
            priority = abs(float(error)) + PER_EPSILON

            if not np.isfinite(priority):
                priority = self.max_priority

            self.priorities[index] = priority

            self.max_priority = max(
                self.max_priority,
                priority,
            )

    def __len__(self):
        return len(self.memory)

    def mean_priority(self):
        if not self.memory:
            return 0.0

        return float(
            np.mean(self.priorities[:len(self.memory)])
        )


# ============================================================
# STEP 7 — BETA SCHEDULE
# ============================================================

def beta_by_step(step):
    fraction = min(
        step / PER_BETA_STEPS,
        1.0,
    )

    return (
        PER_BETA_START
        + fraction * (
            PER_BETA_END - PER_BETA_START
        )
    )


# ============================================================
# STEP 8 — NOISY NETWORK AGENT
# ============================================================

class NoisyDQNAgent:

    def __init__(self):
        self.policy_net = NoisyDuelingDQN().to(DEVICE)
        self.target_net = NoisyDuelingDQN().to(DEVICE)

        self.target_net.load_state_dict(
            self.policy_net.state_dict()
        )

        self.target_net.eval()

        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=LEARNING_RATE,
        )

        self.replay_buffer = PrioritizedReplayBuffer(
            REPLAY_CAPACITY,
        )

        self.training_steps = 0
        self.loss_history = []
        self.td_error_history = []

    def reset_noise(self):
        self.policy_net.reset_noise()
        self.target_net.reset_noise()

    def predict(self, state):
        self.policy_net.eval()

        with torch.no_grad():
            state_t = torch.as_tensor(
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
        self.replay_buffer.add(
            Transition(
                state=state.copy(),
                action=action,
                reward=reward,
                next_state=next_state.copy(),
                terminated=terminated,
                next_mask=next_mask.copy(),
            )
        )

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
            weights,
        ) = self.replay_buffer.sample(
            BATCH_SIZE,
            beta,
        )

        states_t = torch.as_tensor(
            states,
            dtype=torch.float32,
            device=DEVICE,
        )

        actions_t = torch.as_tensor(
            actions,
            dtype=torch.int64,
            device=DEVICE,
        )

        rewards_t = torch.as_tensor(
            rewards,
            dtype=torch.float32,
            device=DEVICE,
        )

        next_states_t = torch.as_tensor(
            next_states,
            dtype=torch.float32,
            device=DEVICE,
        )

        terminated_t = torch.as_tensor(
            terminated,
            dtype=torch.float32,
            device=DEVICE,
        )

        next_masks_t = torch.as_tensor(
            next_masks,
            dtype=torch.bool,
            device=DEVICE,
        )

        weights_t = torch.as_tensor(
            weights,
            dtype=torch.float32,
            device=DEVICE,
        )

        # Reset online network noise for this training batch.
        self.policy_net.reset_noise()

        current_q_all = self.policy_net(states_t)

        current_q = current_q_all.gather(
            1,
            actions_t.unsqueeze(1),
        ).squeeze(1)

        # Double DQN target:
        # online network chooses; target network evaluates.
        with torch.no_grad():
            self.policy_net.reset_noise()
            self.target_net.reset_noise()

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

        td_errors = targets - current_q

        individual_loss = nn.functional.smooth_l1_loss(
            current_q,
            targets,
            reduction="none",
        )

        loss = (
            weights_t * individual_loss
        ).mean()

        self.optimizer.zero_grad()
        loss.backward()

        nn.utils.clip_grad_norm_(
            self.policy_net.parameters(),
            5.0,
        )

        self.optimizer.step()

        self.replay_buffer.update_priorities(
            indices,
            td_errors.detach().cpu().numpy(),
        )

        self.training_steps += 1

        if (
            self.training_steps
            % TARGET_UPDATE_INTERVAL
            == 0
        ):
            self.update_target_network()

        loss_value = float(loss.item())

        self.loss_history.append(loss_value)

        self.td_error_history.append(
            float(td_errors.detach().abs().mean().item())
        )

        return loss_value

    def update_target_network(self):
        self.target_net.load_state_dict(
            self.policy_net.state_dict()
        )

    def save(self, filename="noisy_dqn.pth"):
        torch.save(
            {
                "policy_net": self.policy_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "training_steps": self.training_steps,
            },
            filename,
        )


# ============================================================
# STEP 9 — ACTION SELECTION
# ============================================================

def choose_action(q_values, mask):
    valid_actions = np.flatnonzero(mask)

    if len(valid_actions) == 0:
        return 8

    return int(
        valid_actions[
            np.argmax(q_values[valid_actions])
        ]
    )


def choose_stable_action(
    q_values,
    mask,
    previous_action,
    hysteresis=0.12,
):
    valid_actions = np.flatnonzero(mask)

    if len(valid_actions) == 0:
        return 8

    best_action = choose_action(q_values, mask)

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
# STEP 10 — TRAINING
# ============================================================

def train_agent(agent, env, episodes=TRAIN_EPISODES):
    global_step = 0

    rewards_history = []
    success_history = []
    collision_history = []
    losses_history = []

    print("\n" + "=" * 70)
    print("PHASE 4.6 — NOISY NETWORK DQN")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print("Exploration: Factorized Gaussian Noisy Networks")
    print("Architecture: Dueling Double DQN")
    print("Replay: Prioritized Experience Replay")
    print(f"Episodes: {episodes}")
    print("=" * 70)

    for episode in range(episodes):
        state = env.reset()

        # New parameter noise at the start of each episode.
        agent.reset_noise()

        total_reward = 0.0
        episode_losses = []

        for _ in range(MAX_EPISODE_STEPS):
            mask = env.action_mask()

            # No epsilon-greedy random action.
            # Exploration comes from noisy parameters.
            q_values = agent.predict(state)

            action = choose_action(
                q_values,
                mask,
            )

            result = env.step(action)

            next_mask = env.action_mask()

            agent.remember(
                state,
                action,
                result.reward,
                result.state,
                result.terminated,
                next_mask,
            )

            loss = agent.train_step(
                beta_by_step(global_step)
            )

            if loss is not None:
                episode_losses.append(loss)

            state = result.state
            total_reward += result.reward
            global_step += 1

            if result.terminated or result.truncated:
                break

        rewards_history.append(total_reward)
        success_history.append(int(env.success))
        collision_history.append(int(env.collision))

        losses_history.append(
            float(np.mean(episode_losses))
            if episode_losses
            else 0.0
        )

        if (episode + 1) % 25 == 0 or episode == 0:
            start = max(0, episode - 24)

            avg_reward = np.mean(
                rewards_history[start:episode + 1]
            )

            success_rate = np.mean(
                success_history[start:episode + 1]
            )

            collision_rate = np.mean(
                collision_history[start:episode + 1]
            )

            print(
                f"Episode {episode + 1:4d}/{episodes} | "
                f"Reward {avg_reward:8.2f} | "
                f"Success {success_rate * 100:5.1f}% | "
                f"Collision {collision_rate * 100:5.1f}% | "
                f"Beta {beta_by_step(global_step):.3f} | "
                f"Priority {agent.replay_buffer.mean_priority():.4f}"
            )

    agent.update_target_network()

    print("\nTraining complete.")
    print(f"Environment steps: {global_step}")
    print(f"Training updates: {agent.training_steps}")

    return {
        "episode_rewards": rewards_history,
        "success_history": success_history,
        "collision_history": collision_history,
        "episode_losses": losses_history,
    }


# ============================================================
# STEP 11 — PYGAME ROCKET VISUALIZATION
# ============================================================

class RocketGUI:

    def __init__(self, env, agent, results):
        pygame.init()

        pygame.display.set_caption(
            "Phase 4.6 — Noisy Networks Rocket"
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
            [80.0, 280.0],
            [620.0, 280.0],
        )

        self.current_action = 8
        self.previous_action = 8

        self.agent.reset_noise()

        self.status = "READY"

    def set_target(self, mouse_position):
        x, y = mouse_position

        if not (
            0 <= x <= WORLD_WIDTH
            and 0 <= y <= WORLD_HEIGHT
        ):
            return

        target = np.array(
            [x, y],
            dtype=np.float32,
        )

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

        self.previous_action = 8
        self.current_action = 8

        self.agent.reset_noise()

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

        surface = font.render(
            str(message),
            True,
            color,
        )

        self.screen.blit(surface, (x, y))

    def draw_rocket(self, x, y):
        vector = ACTION_VECTORS[self.current_action]

        if np.linalg.norm(vector) < 0.01:
            vector = ACTION_VECTORS[self.previous_action]

        if np.linalg.norm(vector) < 0.01:
            vector = np.array([1.0, 0.0])

        angle = math.atan2(
            vector[1],
            vector[0],
        )

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

            rx = (
                lx * math.cos(angle)
                - ly * math.sin(angle)
            )

            ry = (
                lx * math.sin(angle)
                + ly * math.cos(angle)
            )

            return int(x + rx), int(y + ry)

        rocket_points = [
            rotate(point)
            for point in local_points
        ]

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
            )

        for y in range(0, WORLD_HEIGHT, 40):
            pygame.draw.line(
                self.screen,
                (39, 46, 58),
                (0, y),
                (WORLD_WIDTH, y),
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
        y0 = 270

        self.draw_text(
            "NOISY DQN ACTION VALUES",
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
            y = y0 + 32 + i * 25

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
                    90
                    * (self.q_values[i] - low)
                    / span
                )

                pygame.draw.rect(
                    self.screen,
                    (60, 120, 190),
                    (x0 + 205, y + 3, width, 10),
                )

    def draw_reward_curve(self):
        x0 = 720
        y0 = 550

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
            np.mean(
                rewards[max(0, i - window + 1):i + 1]
            )
            for i in range(len(rewards))
        ]

        low = float(min(smoothed))
        high = float(max(smoothed))
        span = max(high - low, 1e-6)

        points = []

        for i, value in enumerate(smoothed):
            px = x0 + int(
                i / max(len(smoothed) - 1, 1)
                * width
            )

            py = y0 + height - int(
                (value - low) / span
                * (height - 6)
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
            "NOISY NETWORK DQN",
            720,
            18,
            (120, 210, 255),
            self.title_font,
        )

        self.draw_text(
            f"Device: {DEVICE}",
            720,
            48,
            font=self.small_font,
        )

        self.draw_text(
            f"Position: ({self.env.position[0]:.1f}, "
            f"{self.env.position[1]:.1f})",
            720,
            73,
            font=self.small_font,
        )

        self.draw_text(
            f"Target distance: "
            f"{distance(self.env.position, self.env.target):.1f}",
            720,
            98,
            font=self.small_font,
        )

        self.draw_text(
            f"Action: {ACTION_NAMES[self.current_action]}",
            720,
            123,
            (120, 255, 160),
            self.small_font,
        )

        self.draw_text(
            f"Steps: {self.env.steps}",
            720,
            148,
            font=self.small_font,
        )

        self.draw_text(
            f"Reward: {self.env.total_reward:.2f}",
            720,
            173,
            font=self.small_font,
        )

        self.draw_text(
            f"Replay size: {len(self.agent.replay_buffer)}",
            720,
            198,
            font=self.small_font,
        )

        self.draw_text(
            f"Training updates: {self.agent.training_steps}",
            720,
            223,
            font=self.small_font,
        )

        self.draw_text(
            f"Mean priority: "
            f"{self.agent.replay_buffer.mean_priority():.4f}",
            720,
            248,
            font=self.small_font,
        )

        self.draw_q_values()
        self.draw_reward_curve()

        self.draw_text(
            "CLICK: TARGET | SPACE: PAUSE | R: RESET",
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

                    elif event.key == pygame.K_n:
                        self.agent.reset_noise()
                        self.status = "NETWORK NOISE RESET"

                    elif event.key in (
                        pygame.K_PLUS,
                        pygame.K_EQUALS,
                        pygame.K_KP_PLUS,
                    ):
                        self.speed = min(
                            self.speed + 1,
                            8,
                        )

                    elif event.key in (
                        pygame.K_MINUS,
                        pygame.K_KP_MINUS,
                    ):
                        self.speed = max(
                            self.speed - 1,
                            1,
                        )

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
# STEP 12 — NUMERICAL VERIFICATION
# ============================================================

def run_tests():
    print("\n" + "=" * 70)
    print("PHASE 4.6 UNIT TESTS")
    print("=" * 70)

    passed = 0

    def check(name, condition):
        nonlocal passed

        assert condition, f"FAILED: {name}"

        passed += 1
        print(f"[PASS] {name}")

    # Test 1 — Layer output dimensions.
    layer = NoisyLinear(10, 32)

    x = torch.randn(4, 10)

    y = layer(x)

    check(
        "NoisyLinear output shape",
        tuple(y.shape) == (4, 32),
    )

    # Test 2 — Noise reset changes noise parameters.
    layer.reset_noise()
    epsilon_before = layer.weight_epsilon.clone()

    layer.reset_noise()
    epsilon_after = layer.weight_epsilon.clone()

    check(
        "Reset noise changes sampled noise",
        not torch.equal(
            epsilon_before,
            epsilon_after,
        ),
    )

    # Test 3 — Noise scale is trainable.
    check(
        "Weight noise scale is trainable",
        layer.weight_sigma.requires_grad,
    )

    # Test 4 — Bias noise scale is trainable.
    check(
        "Bias noise scale is trainable",
        layer.bias_sigma.requires_grad,
    )

    # Test 5 — Gradients propagate.
    layer.zero_grad()

    output = layer(torch.randn(4, 10))
    loss = output.square().mean()
    loss.backward()

    check(
        "Gradient reaches noise scale",
        layer.weight_sigma.grad is not None,
    )

    # Test 6 — Deterministic evaluation.
    layer.eval()

    x_eval = torch.randn(3, 10)

    output1 = layer(x_eval)
    layer.reset_noise()
    output2 = layer(x_eval)

    check(
        "Evaluation output ignores sampled noise",
        torch.allclose(output1, output2),
    )

    # Test 7 — Dueling network output dimensions.
    network = NoisyDuelingDQN().to(DEVICE)

    state = torch.randn(
        2,
        STATE_SIZE,
        device=DEVICE,
    )

    q = network(state)

    check(
        "Noisy Dueling DQN output shape",
        tuple(q.shape) == (2, NUM_ACTIONS),
    )

    # Test 8 — Dueling aggregation identity.
    value = torch.tensor([[2.0]])
    advantage = torch.tensor(
        [[1.0, 2.0, 3.0]]
    )

    q_values = (
        value
        + advantage
        - advantage.mean(dim=1, keepdim=True)
    )

    check(
        "Dueling aggregation preserves mean value",
        torch.isclose(
            q_values.mean(),
            value.squeeze(),
        ).item(),
    )

    # Test 9 — Beta schedule start.
    check(
        "PER beta begins at 0.4",
        np.isclose(
            beta_by_step(0),
            PER_BETA_START,
        ),
    )

    # Test 10 — Beta schedule end.
    check(
        "PER beta reaches 1.0",
        np.isclose(
            beta_by_step(PER_BETA_STEPS),
            PER_BETA_END,
        ),
    )

    # Test 11 — Replay sampling.
    replay = PrioritizedReplayBuffer(20)

    transition = Transition(
        state=np.zeros(STATE_SIZE, dtype=np.float32),
        action=0,
        reward=1.0,
        next_state=np.ones(STATE_SIZE, dtype=np.float32),
        terminated=False,
        next_mask=np.ones(NUM_ACTIONS, dtype=bool),
    )

    for _ in range(10):
        replay.add(transition)

    sample = replay.sample(4, beta=0.4)

    check(
        "PER returns correct batch dimensions",
        sample[0].shape == (4, STATE_SIZE),
    )

    # Test 12 — Probability normalization.
    check(
        "PER probabilities sum to one",
        np.isclose(
            replay.probabilities().sum(),
            1.0,
        ),
    )

    # Test 13 — Priority update.
    replay.update_priorities(
        np.array([0, 1]),
        np.array([2.0, 5.0]),
    )

    check(
        "PER updates transition priorities",
        replay.priorities[1] > replay.priorities[0],
    )

    # Test 14 — Environment state dimensions.
    env = RocketNavigationEnv()

    state = env.reset()

    check(
        "Robot state dimension is ten",
        state.shape == (STATE_SIZE,),
    )

    # Test 15 — Safety mask.
    mask = env.action_mask()

    check(
        "Safety mask contains nine actions",
        mask.shape == (NUM_ACTIONS,),
    )

    # Test 16 — Stay action.
    check(
        "Stay action is available",
        bool(mask[8]),
    )

    # Test 17 — Action vectors.
    norms = np.linalg.norm(
        ACTION_VECTORS[:8],
        axis=1,
    )

    check(
        "Directional actions are normalized",
        np.allclose(norms, np.ones(8)),
    )

    # Test 18 — Action selection respects mask.
    test_mask = np.array(
        [False, True, False, False, False,
         False, False, False, True],
        dtype=bool,
    )

    selected = choose_action(
        np.array(
            [100, 1, 100, 100, 100, 100, 100, 100, 0],
            dtype=np.float32,
        ),
        test_mask,
    )

    check(
        "Action selection respects safety mask",
        selected == 1,
    )

    print("-" * 70)
    print(f"Tests passed: {passed}/18")
    print("=" * 70)


# ============================================================
# STEP 13 — MAIN
# ============================================================

def main():
    set_seed()

    run_tests()

    env = RocketNavigationEnv()
    agent = NoisyDQNAgent()

    results = train_agent(
        agent,
        env,
        episodes=TRAIN_EPISODES,
    )

    agent.save("noisy_dqn.pth")

    print("\nModel saved: noisy_dqn.pth")

    gui = RocketGUI(
        env,
        agent,
        results,
    )

    gui.run()


if __name__ == "__main__":
    main()
