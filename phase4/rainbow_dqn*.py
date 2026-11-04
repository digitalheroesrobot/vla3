
# ============================================================
# File: phase4/rainbow_dqn.py
#
# PHASE 4.7 — RAINBOW DEEP Q-NETWORK
#
# PURPOSE
#   Combine six major Deep RL improvements into one agent.
#
# WHAT YOU LEARN
#
# STEP 1 — DOUBLE DQN
#   Online network selects the next action.
#   Target network evaluates that action.
#
# STEP 2 — DUELING NETWORK
#   Q(s,a) = V(s) + A(s,a) - mean_a(A(s,a))
#
# STEP 3 — PRIORITIZED EXPERIENCE REPLAY
#   P(i) = p_i^alpha / sum_j(p_j^alpha)
#
# STEP 4 — NOISY NETWORKS
#   Learnable parameter-space exploration.
#
# STEP 5 — N-STEP RETURNS
#   R_n = r_0 + gamma*r_1 + ... + gamma^(n-1)*r_(n-1)
#
#   Target = R_n + gamma^n * V(s_n)
#   Bootstrap is suppressed only for true termination.
#
# STEP 6 — DISTRIBUTIONAL RL (C51)
#   Learn a probability distribution over possible returns.
#
#   Z(s,a) = distribution over return atoms
#   Q(s,a) = sum_i z_i * P(z_i | s,a)
#
# STEP 7 — RAINBOW TRAINING
#   Combine all six components.
#
# STEP 8 — PYGAME ROBOT SIMULATION
#   Interactive rocket navigation with target selection.
#
# ROBOTICS APPLICATION
#   A mobile robot must learn navigation strategies while
#   handling uncertainty, delayed rewards, and obstacles.
#   Distributional value learning can represent uncertainty
#   in possible future returns, while n-step learning
#   propagates navigation rewards more quickly.
#
# IMPORTANT
#   This is a compact educational Rainbow implementation.
#   It uses a simple proportional PER array, not a SumTree.
#   It is not a certified real-world robot controller.
#
# PYGAME CONTROLS
#   Left click : Change target
#   SPACE      : Pause / resume
#   R          : Reset rocket
#   + / -      : Simulation speed
#   N          : Resample network noise
#   ESC        : Exit
#
# INSTALL
#   pip install numpy pygame torch
#
# RUN
#   python phase4/rainbow_dqn.py
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

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

WORLD_WIDTH = 700
WORLD_HEIGHT = 560

WINDOW_WIDTH = 1020
WINDOW_HEIGHT = 650

OBSTACLE = np.array([350.0, 280.0], dtype=np.float32)
OBSTACLE_RADIUS = 55.0

ROCKET_RADIUS = 11.0
SAFE_DISTANCE = OBSTACLE_RADIUS + ROCKET_RADIUS + 20.0

MOVE_STEP = 8.0
MAX_EPISODE_STEPS = 170

STATE_SIZE = 10
NUM_ACTIONS = 9

HIDDEN_SIZE = 128

# C51 distributional parameters.
NUM_ATOMS = 51
V_MIN = -20.0
V_MAX = 20.0

SUPPORT = torch.linspace(
    V_MIN,
    V_MAX,
    NUM_ATOMS,
    device=DEVICE,
)

DELTA_Z = (V_MAX - V_MIN) / (NUM_ATOMS - 1)

# Learning parameters.
GAMMA = 0.99
N_STEP = 3

LEARNING_RATE = 0.0001

BATCH_SIZE = 64
REPLAY_CAPACITY = 30_000
WARMUP_STEPS = 800

TRAIN_EPISODES = 350
TARGET_UPDATE_INTERVAL = 300

# PER parameters.
PER_ALPHA = 0.6
PER_BETA_START = 0.4
PER_BETA_END = 1.0
PER_BETA_STEPS = 50_000
PER_EPSILON = 1e-5

# Noisy network parameters.
NOISE_SIGMA = 0.5

# Action hysteresis.
ACTION_HYSTERESIS = 0.12

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


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# STEP 1 — NAVIGATION GEOMETRY
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
        position = np.array(
            [
                random.uniform(35, WORLD_WIDTH - 35),
                random.uniform(35, WORLD_HEIGHT - 35),
            ],
            dtype=np.float32,
        )

        if is_safe_position(position):
            return position

    raise RuntimeError("Could not generate a safe position.")


# ============================================================
# STEP 2 — ROCKET NAVIGATION ENVIRONMENT
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
        start = np.asarray(start, dtype=np.float32)
        target = np.asarray(target, dtype=np.float32)

        if not is_inside_world(start) or not is_safe_position(start):
            raise ValueError("Invalid start position.")

        if not is_inside_world(target) or not is_safe_position(target):
            raise ValueError("Invalid target position.")

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

        previous = ACTION_VECTORS[self.previous_action]

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
                previous[0],
                previous[1],
            ],
            dtype=np.float32,
        )

    def action_mask(self):
        return calculate_action_mask(self.position)

    def step(self, action):
        old_distance = distance(self.position, self.target)

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

            if self.previous_action != 8 and action != 8:
                old_vector = ACTION_VECTORS[self.previous_action]
                new_vector = ACTION_VECTORS[action]

                if np.dot(old_vector, new_vector) < -0.5:
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
# STEP 3 — FACTORIZED GAUSSIAN NOISY LINEAR
# ============================================================

class NoisyLinear(nn.Module):
    """
    y = (W + sigma_W * epsilon_W)x
        + (b + sigma_b * epsilon_b)

    sigma is learned by gradient descent.

    Factorized Gaussian noise:
        f(x) = sign(x) * sqrt(abs(x))

        epsilon_W[i,j] = f(epsilon_out[i])
                         * f(epsilon_in[j])
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

        nn.init.uniform_(self.weight_mu, -bound, bound)
        nn.init.uniform_(self.bias_mu, -bound, bound)

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
        x = torch.randn(size, device=device)
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
            weight = self.weight_mu
            bias = self.bias_mu

        return nn.functional.linear(x, weight, bias)


# ============================================================
# STEP 4 — DUELING DISTRIBUTIONAL NETWORK
# ============================================================

class RainbowNetwork(nn.Module):
    """
    C51 output:
        [batch, actions, atoms]

    Dueling distributional aggregation:
        logits = value + advantage - mean_action(advantage)

    The expected Q-value is:
        Q(s,a) = sum_z z * softmax(logits)_z
    """

    def __init__(self):
        super().__init__()

        self.feature_layer = nn.Sequential(
            nn.Linear(STATE_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
        )

        self.value_stream = nn.Sequential(
            NoisyLinear(HIDDEN_SIZE, 64),
            nn.ReLU(),
            NoisyLinear(64, NUM_ATOMS),
        )

        self.advantage_stream = nn.Sequential(
            NoisyLinear(HIDDEN_SIZE, 64),
            nn.ReLU(),
            NoisyLinear(64, NUM_ACTIONS * NUM_ATOMS),
        )

    def forward(self, state):
        features = self.feature_layer(state)

        value = self.value_stream(features).view(
            -1,
            1,
            NUM_ATOMS,
        )

        advantage = self.advantage_stream(features).view(
            -1,
            NUM_ACTIONS,
            NUM_ATOMS,
        )

        logits = (
            value
            + advantage
            - advantage.mean(dim=1, keepdim=True)
        )

        return logits

    def probabilities(self, state):
        return torch.softmax(self.forward(state), dim=-1)

    def expected_q(self, state):
        probabilities = self.probabilities(state)

        return (
            probabilities
            * SUPPORT.view(1, 1, -1)
        ).sum(dim=-1)

    def reset_noise(self):
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()


# ============================================================
# STEP 5 — N-STEP TRANSITION ACCUMULATOR
# ============================================================

@dataclass
class RawTransition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    terminated: bool
    truncated: bool
    next_mask: np.ndarray


@dataclass
class ReplayTransition:
    state: np.ndarray
    action: int
    reward_n: float
    next_state: np.ndarray
    terminated: bool
    gamma_n: float
    next_mask: np.ndarray


class NStepAccumulator:
    """
    Accumulate up to N_STEP rewards.

    For a full n-step transition:
        R_n = sum(k=0 to n-1) gamma^k * r_k

    If true termination occurs:
        Do not bootstrap.

    If time-limit truncation occurs:
        Flush pending transitions and bootstrap from the
        final valid observation.
    """

    def __init__(self, n_step=N_STEP, gamma=GAMMA):
        self.n_step = n_step
        self.gamma = gamma
        self.queue = deque()

    def _build_transition(self, count):
        items = list(self.queue)[:count]

        first = items[0]
        last = items[-1]

        reward_n = sum(
            (self.gamma ** i) * item.reward
            for i, item in enumerate(items)
        )

        return ReplayTransition(
            state=first.state.copy(),
            action=first.action,
            reward_n=float(reward_n),
            next_state=last.next_state.copy(),
            terminated=last.terminated,
            gamma_n=self.gamma ** count,
            next_mask=last.next_mask.copy(),
        )

    def append(self, transition):
        self.queue.append(transition)

        emitted = []

        if len(self.queue) >= self.n_step:
            emitted.append(
                self._build_transition(self.n_step)
            )
            self.queue.popleft()

        if transition.terminated or transition.truncated:
            while self.queue:
                count = min(
                    self.n_step,
                    len(self.queue),
                )

                emitted.append(
                    self._build_transition(count)
                )

                self.queue.popleft()

        return emitted


# ============================================================
# STEP 6 — PRIORITIZED EXPERIENCE REPLAY
# ============================================================

class PrioritizedReplayBuffer:

    def __init__(self, capacity, alpha=PER_ALPHA):
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

        total = float(scaled.sum())

        if not np.isfinite(total) or total <= 0:
            return np.full(size, 1.0 / size)

        return scaled / total

    def sample(self, batch_size, beta):
        size = len(self.memory)

        if size < batch_size:
            raise ValueError("Insufficient replay samples.")

        probabilities = self.probabilities()

        # Sampling with replacement matches the usual
        # proportional PER importance-sampling formula.
        indices = np.random.choice(
            size,
            batch_size,
            replace=True,
            p=probabilities,
        )

        selected_probabilities = probabilities[indices]

        weights = (
            size * selected_probabilities
        ) ** (-beta)

        weights /= weights.max()

        batch = [self.memory[i] for i in indices]

        return {
            "transitions": batch,
            "indices": indices,
            "weights": weights.astype(np.float32),
        }

    def update_priorities(self, indices, priorities):
        for index, priority in zip(indices, priorities):
            value = abs(float(priority)) + PER_EPSILON

            if not np.isfinite(value):
                value = self.max_priority

            self.priorities[index] = value

            self.max_priority = max(
                self.max_priority,
                value,
            )

    def __len__(self):
        return len(self.memory)

    def mean_priority(self):
        if not self.memory:
            return 0.0

        return float(
            self.priorities[:len(self.memory)].mean()
        )


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
# STEP 7 — C51 DISTRIBUTION PROJECTION
# ============================================================

def project_distribution(
    next_probabilities,
    rewards,
    terminated,
    gamma_n,
):
    """
    C51 categorical Bellman projection.

    Each source atom contributes probability mass to the two
    neighboring destination atoms. Destination indices must be
    offset by the batch row; otherwise every batch item writes into
    the first row and later rows lose their probability mass.
    """
    if next_probabilities.ndim != 2:
        raise ValueError(
            "next_probabilities must have shape [batch, NUM_ATOMS]."
        )

    batch_size, atom_count = next_probabilities.shape
    if atom_count != NUM_ATOMS:
        raise ValueError(
            f"Expected {NUM_ATOMS} atoms, received {atom_count}."
        )

    rewards = rewards.reshape(batch_size, 1)
    terminated = terminated.reshape(batch_size, 1).to(
        dtype=next_probabilities.dtype
    )
    gamma_n = gamma_n.reshape(batch_size, 1)

    support = SUPPORT.to(
        device=next_probabilities.device,
        dtype=next_probabilities.dtype,
    ).unsqueeze(0)

    transformed = (
        rewards
        + gamma_n * (1.0 - terminated) * support
    ).clamp(V_MIN, V_MAX)

    delta_z = (V_MAX - V_MIN) / (NUM_ATOMS - 1)
    b = (transformed - V_MIN) / delta_z

    lower = b.floor().long().clamp(0, NUM_ATOMS - 1)
    upper = b.ceil().long().clamp(0, NUM_ATOMS - 1)

    lower_weight = upper.to(b.dtype) - b
    upper_weight = b - lower.to(b.dtype)

    same = lower == upper
    lower_weight = torch.where(
        same, torch.ones_like(lower_weight), lower_weight
    )
    upper_weight = torch.where(
        same, torch.zeros_like(upper_weight), upper_weight
    )

    # Flattened destination indices need a per-batch-row offset.
    batch_offsets = (
        torch.arange(batch_size, device=next_probabilities.device)
        .unsqueeze(1)
        * NUM_ATOMS
    )
    lower_indices = (lower + batch_offsets).reshape(-1)
    upper_indices = (upper + batch_offsets).reshape(-1)

    projected = torch.zeros_like(next_probabilities)
    projected_flat = projected.reshape(-1)

    projected_flat.index_add_(
        0,
        lower_indices,
        (next_probabilities * lower_weight).reshape(-1),
    )
    projected_flat.index_add_(
        0,
        upper_indices,
        (next_probabilities * upper_weight).reshape(-1),
    )

    return projected



# ============================================================
# STEP 8 — RAINBOW DQN AGENT
# ============================================================

class RainbowAgent:

    def __init__(self):
        self.online = RainbowNetwork().to(DEVICE)
        self.target = RainbowNetwork().to(DEVICE)

        self.target.load_state_dict(
            self.online.state_dict()
        )

        self.optimizer = optim.Adam(
            self.online.parameters(),
            lr=LEARNING_RATE,
        )

        self.replay = PrioritizedReplayBuffer(
            REPLAY_CAPACITY,
        )

        self.n_step_accumulator = NStepAccumulator()

        self.training_steps = 0
        self.loss_history = []
        self.td_error_history = []

        # Noisy layers must be active during training action
        # selection. Do not switch this network to eval mode.
        self.online.train()
        self.target.train()

    def reset_noise(self):
        self.online.reset_noise()
        self.target.reset_noise()

    def select_action(self, state, mask):
        """
        Noisy action selection.

        The online network remains in training mode so
        NoisyLinear uses sampled parameters.

        No epsilon-greedy random action is used.
        """

        valid = np.flatnonzero(mask)

        if len(valid) == 0:
            return 8

        state_t = torch.as_tensor(
            state,
            dtype=torch.float32,
            device=DEVICE,
        ).unsqueeze(0)

        with torch.no_grad():
            q_values = self.online.expected_q(state_t)

        q_values = q_values.squeeze(0).cpu().numpy()

        q_values[~mask] = -np.inf

        return int(np.argmax(q_values))

    def expected_q_values(self, state, noisy=True):
        state_t = torch.as_tensor(
            state,
            dtype=torch.float32,
            device=DEVICE,
        ).unsqueeze(0)

        if noisy:
            self.online.train()
        else:
            self.online.eval()

        with torch.no_grad():
            values = self.online.expected_q(state_t)

        self.online.train()

        return values.squeeze(0).cpu().numpy()

    def remember_raw(
        self,
        state,
        action,
        reward,
        next_state,
        terminated,
        truncated,
        next_mask,
    ):
        raw = RawTransition(
            state=state.copy(),
            action=int(action),
            reward=float(reward),
            next_state=next_state.copy(),
            terminated=bool(terminated),
            truncated=bool(truncated),
            next_mask=next_mask.copy(),
        )

        completed = self.n_step_accumulator.append(raw)

        for transition in completed:
            self.replay.add(transition)

    def train_step(self, beta):
        if len(self.replay) < max(
            BATCH_SIZE,
            min(WARMUP_STEPS, REPLAY_CAPACITY),
        ):
            return None

        sampled = self.replay.sample(
            BATCH_SIZE,
            beta,
        )

        transitions = sampled["transitions"]
        indices = sampled["indices"]
        weights = sampled["weights"]

        states = np.stack(
            [t.state for t in transitions]
        )

        actions = np.array(
            [t.action for t in transitions],
            dtype=np.int64,
        )

        rewards = np.array(
            [t.reward_n for t in transitions],
            dtype=np.float32,
        )

        next_states = np.stack(
            [t.next_state for t in transitions]
        )

        terminated = np.array(
            [t.terminated for t in transitions],
            dtype=np.float32,
        )

        gamma_n = np.array(
            [t.gamma_n for t in transitions],
            dtype=np.float32,
        )

        next_masks = np.stack(
            [t.next_mask for t in transitions]
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

        gamma_n_t = torch.as_tensor(
            gamma_n,
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

        # ----------------------------------------------------
        # Current-state distribution.
        # ----------------------------------------------------

        self.online.train()

        # Sample noise once per optimization step. The same online-network
        # noise is used for the current-state prediction and Double-DQN
        # next-action selection. Do not reset it again before backward(),
        # because the current forward graph retains references to these
        # NoisyLinear noise buffers.
        self.online.reset_noise()

        current_logits = self.online(states_t)

        batch_indices = torch.arange(
            BATCH_SIZE,
            device=DEVICE,
        )

        selected_logits = current_logits[
            batch_indices,
            actions_t,
        ]

        log_probabilities = torch.log_softmax(
            selected_logits,
            dim=-1,
        )

        # ----------------------------------------------------
        # Double DQN action selection and C51 projection.
        # ----------------------------------------------------

        with torch.no_grad():
            # Target-network noise is safe to resample here: target outputs
            # are computed without gradients. Keep online noise unchanged
            # until loss.backward() has completed.
            self.target.reset_noise()

            online_next_probabilities = self.online.probabilities(
                next_states_t
            )

            online_next_q = (
                online_next_probabilities
                * SUPPORT.view(1, 1, -1)
            ).sum(dim=-1)

            online_next_q = online_next_q.masked_fill(
                ~next_masks_t,
                -1e9,
            )

            next_actions = online_next_q.argmax(
                dim=1
            )

            target_next_probabilities = self.target.probabilities(
                next_states_t
            )

            selected_next_probabilities = (
                target_next_probabilities[
                    batch_indices,
                    next_actions,
                ]
            )

            projected = project_distribution(
                selected_next_probabilities,
                rewards_t,
                terminated_t,
                gamma_n_t,
            )

        # ----------------------------------------------------
        # Distributional cross-entropy loss.
        # ----------------------------------------------------

        per_sample_loss = -(
            projected * log_probabilities
        ).sum(dim=1)

        loss = (
            weights_t * per_sample_loss
        ).mean()

        # ----------------------------------------------------
        # PER priority based on expected-value TD error.
        # ----------------------------------------------------

        with torch.no_grad():
            current_expected_q = (
                torch.softmax(
                    selected_logits,
                    dim=-1,
                ) * SUPPORT
            ).sum(dim=1)

            projected_expected_q = (
                projected * SUPPORT
            ).sum(dim=1)

            td_errors = (
                projected_expected_q
                - current_expected_q
            ).abs()

        self.optimizer.zero_grad()
        loss.backward()

        nn.utils.clip_grad_norm_(
            self.online.parameters(),
            10.0,
        )

        self.optimizer.step()

        self.replay.update_priorities(
            indices,
            td_errors.cpu().numpy(),
        )

        self.training_steps += 1

        if (
            self.training_steps
            % TARGET_UPDATE_INTERVAL
            == 0
        ):
            self.update_target()

        loss_value = float(loss.item())

        self.loss_history.append(loss_value)

        self.td_error_history.append(
            float(td_errors.mean().item())
        )

        return loss_value

    def update_target(self):
        self.target.load_state_dict(
            self.online.state_dict()
        )

        self.target.train()

    def save(self, filename="rainbow_dqn.pth"):
        torch.save(
            {
                "online": self.online.state_dict(),
                "target": self.target.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "training_steps": self.training_steps,
            },
            filename,
        )


# ============================================================
# STEP 9 — STABLE ACTION SELECTION FOR THE GUI
# ============================================================

def choose_stable_action(
    q_values,
    mask,
    previous_action,
    hysteresis=ACTION_HYSTERESIS,
):
    valid = np.flatnonzero(mask)

    if len(valid) == 0:
        return 8

    best_action = int(
        valid[np.argmax(q_values[valid])]
    )

    if (
        previous_action in valid
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
# STEP 10 — RAINBOW TRAINING LOOP
# ============================================================

def train_agent(agent, env, episodes=TRAIN_EPISODES):
    global_step = 0

    episode_rewards = []
    success_history = []
    collision_history = []
    episode_losses = []

    print("\n" + "=" * 72)
    print("PHASE 4.7 — RAINBOW DQN")
    print("=" * 72)
    print(f"Device: {DEVICE}")
    print("Double DQN: enabled")
    print("Dueling network: enabled")
    print("Prioritized replay: enabled")
    print("Noisy networks: enabled")
    print(f"N-step return: {N_STEP}")
    print(f"C51 atoms: {NUM_ATOMS}")
    print(f"C51 support: [{V_MIN}, {V_MAX}]")
    print("=" * 72)

    for episode in range(episodes):
        state = env.reset()

        agent.reset_noise()

        total_reward = 0.0
        losses = []

        for _ in range(MAX_EPISODE_STEPS):
            mask = env.action_mask()

            action = agent.select_action(
                state,
                mask,
            )

            result = env.step(action)

            next_mask = env.action_mask()

            agent.remember_raw(
                state,
                action,
                result.reward,
                result.state,
                result.terminated,
                result.truncated,
                next_mask,
            )

            loss = agent.train_step(
                beta_by_step(global_step)
            )

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
            float(np.mean(losses))
            if losses
            else 0.0
        )

        if episode == 0 or (episode + 1) % 25 == 0:
            start = max(0, episode - 24)

            avg_reward = np.mean(
                episode_rewards[start:episode + 1]
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
                f"Loss {episode_losses[-1]:.4f} | "
                f"Beta {beta_by_step(global_step):.3f}"
            )

    agent.update_target()

    print("\nTraining completed.")
    print(f"Environment steps: {global_step}")
    print(f"Training updates: {agent.training_steps}")

    return {
        "episode_rewards": episode_rewards,
        "success_history": success_history,
        "collision_history": collision_history,
        "episode_losses": episode_losses,
    }


# ============================================================
# STEP 11 — PYGAME VISUALIZATION
# ============================================================

class RocketGUI:

    def __init__(self, env, agent, results):
        pygame.init()

        pygame.display.set_caption(
            "Phase 4.7 — Rainbow DQN Rocket"
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

        self.running = True
        self.paused = False
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

        if not (0 <= x <= WORLD_WIDTH and 0 <= y <= WORLD_HEIGHT):
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

        # Pygame requires integer destination coordinates for blit().
        # World positions (for example OBSTACLE) are stored as NumPy floats,
        # so normalize every text position here rather than fixing each call.
        try:
            x_float = float(x)
            y_float = float(y)
        except (TypeError, ValueError, OverflowError):
            return

        # Avoid passing NaN or infinity to Pygame if a simulation value
        # becomes temporarily invalid.
        if not (math.isfinite(x_float) and math.isfinite(y_float)):
            return

        self.screen.blit(
            surface,
            (int(round(x_float)), int(round(y_float))),
        )

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

        body = [
            (20, 0),
            (5, -9),
            (-9, -7),
            (-13, -4),
            (-13, 4),
            (-9, 7),
            (5, 9),
        ]

        flame_length = random.randint(9, 16)

        flame = [
            rotate((-13, -4)),
            rotate((-13 - flame_length, 0)),
            rotate((-13, 4)),
        ]

        pygame.draw.polygon(
            self.screen,
            (255, 145, 35),
            flame,
        )

        points = [rotate(p) for p in body]

        pygame.draw.polygon(
            self.screen,
            (235, 240, 250),
            points,
        )

        pygame.draw.polygon(
            self.screen,
            (70, 150, 230),
            points,
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
        y0 = 275

        self.draw_text(
            "RAINBOW EXPECTED Q VALUES",
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

        smoothed = [
            np.mean(
                rewards[max(0, i - 14):i + 1]
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
            "RAINBOW DQN",
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
            f"Replay: {len(self.agent.replay)}",
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
            f"Mean priority: {self.agent.replay.mean_priority():.4f}",
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
        state = self.env.get_state()
        self.mask = self.env.action_mask()

        self.q_values = self.agent.expected_q_values(
            state,
            noisy=True,
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
# STEP 12 — UNIT TESTS
# ============================================================

def run_tests():
    print("\n" + "=" * 72)
    print("PHASE 4.7 UNIT TESTS")
    print("=" * 72)

    passed = 0

    def check(name, condition):
        nonlocal passed

        assert bool(condition), f"FAILED: {name}"

        passed += 1
        print(f"[PASS] {name}")

    # Test 1 — Network output dimensions.
    network = RainbowNetwork().to(DEVICE)

    state = torch.randn(
        4,
        STATE_SIZE,
        device=DEVICE,
    )

    logits = network(state)

    check(
        "C51 output shape",
        tuple(logits.shape)
        == (4, NUM_ACTIONS, NUM_ATOMS),
    )

    # Test 2 — Probability normalization.
    probabilities = torch.softmax(
        logits,
        dim=-1,
    )

    check(
        "C51 probabilities sum to one",
        torch.allclose(
            probabilities.sum(dim=-1),
            torch.ones(
                4,
                NUM_ACTIONS,
                device=DEVICE,
            ),
            atol=1e-5,
        ),
    )

    # Test 3 — Expected Q dimensions.
    q_values = network.expected_q(state)

    check(
        "Expected Q-value shape",
        tuple(q_values.shape) == (4, NUM_ACTIONS),
    )

    # Test 4 — Noisy layer output.
    noisy = NoisyLinear(10, 32)

    noisy_output = noisy(torch.randn(3, 10))

    check(
        "NoisyLinear output shape",
        tuple(noisy_output.shape) == (3, 32),
    )

    # Test 5 — Noise parameters are trainable.
    check(
        "Noisy weight scale is trainable",
        noisy.weight_sigma.requires_grad,
    )

    # Test 6 — Deterministic evaluation.
    noisy.eval()

    test_input = torch.randn(3, 10)

    output_a = noisy(test_input)
    noisy.reset_noise()
    output_b = noisy(test_input)

    check(
        "Evaluation ignores sampled noise",
        torch.allclose(output_a, output_b),
    )

    # Test 7 — PER probabilities.
    replay = PrioritizedReplayBuffer(20)

    transition = ReplayTransition(
        state=np.zeros(STATE_SIZE, dtype=np.float32),
        action=0,
        reward_n=1.0,
        next_state=np.ones(STATE_SIZE, dtype=np.float32),
        terminated=False,
        gamma_n=GAMMA,
        next_mask=np.ones(NUM_ACTIONS, dtype=bool),
    )

    for _ in range(10):
        replay.add(transition)

    check(
        "PER probabilities sum to one",
        np.isclose(
            replay.probabilities().sum(),
            1.0,
        ),
    )

    # Test 8 — PER priority update.
    old_priority = replay.priorities[0]

    replay.update_priorities(
        np.array([0]),
        np.array([20.0]),
    )

    check(
        "PER priority increases",
        replay.priorities[0] > old_priority,
    )

    # Test 9 — PER batch dimensions.
    sample = replay.sample(4, beta=0.4)

    check(
        "PER returns requested batch size",
        len(sample["transitions"]) == 4,
    )

    # Test 10 — N-step full return.
    accumulator = NStepAccumulator(
        n_step=3,
        gamma=0.9,
    )

    raw_items = [
        RawTransition(
            state=np.full(STATE_SIZE, i, dtype=np.float32),
            action=0,
            reward=1.0,
            next_state=np.full(
                STATE_SIZE,
                i + 1,
                dtype=np.float32,
            ),
            terminated=False,
            truncated=False,
            next_mask=np.ones(NUM_ACTIONS, dtype=bool),
        )
        for i in range(3)
    ]

    emitted = []

    for item in raw_items:
        emitted.extend(accumulator.append(item))

    expected_return = 1.0 + 0.9 + 0.9 ** 2

    check(
        "Three-step reward calculation",
        np.isclose(
            emitted[0].reward_n,
            expected_return,
        ),
    )

    # Test 11 — N-step terminal flush.
    terminal_accumulator = NStepAccumulator(
        n_step=3,
        gamma=0.9,
    )

    terminal_items = [
        RawTransition(
            state=np.zeros(STATE_SIZE, dtype=np.float32),
            action=0,
            reward=1.0,
            next_state=np.ones(STATE_SIZE, dtype=np.float32),
            terminated=(i == 1),
            truncated=False,
            next_mask=np.ones(NUM_ACTIONS, dtype=bool),
        )
        for i in range(2)
    ]

    terminal_emitted = []

    for item in terminal_items:
        terminal_emitted.extend(
            terminal_accumulator.append(item)
        )

    check(
        "N-step accumulator flushes at termination",
        len(terminal_emitted) == 2,
    )

    # Test 12 — Terminal transitions do not bootstrap.
    check(
        "Terminal transition has termination flag",
        terminal_emitted[0].terminated,
    )

    # Test 13 — C51 projection mass.
    next_probs = torch.full(
        (2, NUM_ATOMS),
        1.0 / NUM_ATOMS,
        device=DEVICE,
    )

    rewards = torch.tensor(
        [0.0, 1.0],
        device=DEVICE,
    )

    terminal_flags = torch.tensor(
        [0.0, 1.0],
        device=DEVICE,
    )

    gamma_n = torch.tensor(
        [GAMMA ** N_STEP, GAMMA ** N_STEP],
        device=DEVICE,
    )

    projected = project_distribution(
        next_probs,
        rewards,
        terminal_flags,
        gamma_n,
    )

    check(
        "C51 projected mass is preserved",
        torch.allclose(
            projected.sum(dim=1),
            torch.ones(2, device=DEVICE),
            atol=1e-5,
        ),
    )

    # Test 14 — Terminal projection is centered at reward.
    terminal_projection = project_distribution(
        next_probs[1:2],
        rewards[1:2],
        terminal_flags[1:2],
        gamma_n[1:2],
    )

    terminal_mean = (
        terminal_projection * SUPPORT
    ).sum(dim=1)

    check(
        "Terminal projection mean equals reward",
        torch.allclose(
            terminal_mean,
            rewards[1:2],
            atol=1e-4,
        ),
    )

    # Test 15 — Action mask.
    env = RocketNavigationEnv()

    env.reset()

    mask = env.action_mask()

    check(
        "Action mask has nine entries",
        mask.shape == (NUM_ACTIONS,),
    )

    # Test 16 — Stay is always available.
    check(
        "Stay action is available",
        bool(mask[8]),
    )

    # Test 17 — Direction vectors normalized.
    norms = np.linalg.norm(
        ACTION_VECTORS[:8],
        axis=1,
    )

    check(
        "Eight directional vectors normalized",
        np.allclose(norms, np.ones(8)),
    )

    # Test 18 — Safety filter blocks invalid actions.
    test_mask = np.array(
        [False, True, False, False, False,
         False, False, False, True],
        dtype=bool,
    )

    test_q = np.array(
        [100, 1, 100, 100, 100, 100, 100, 100, 0],
        dtype=np.float32,
    )

    selected = choose_stable_action(
        test_q,
        test_mask,
        previous_action=8,
        hysteresis=0.0,
    )

    check(
        "Safety filter blocks invalid best action",
        selected == 1,
    )

    print("-" * 72)
    print(f"Tests passed: {passed}/18")
    print("=" * 72)


# ============================================================
# STEP 13 — MAIN
# ============================================================

def main():
    set_seed()

    run_tests()

    env = RocketNavigationEnv()
    agent = RainbowAgent()

    results = train_agent(
        agent,
        env,
        episodes=TRAIN_EPISODES,
    )

    agent.save("rainbow_dqn.pth")

    print("\nModel saved: rainbow_dqn.pth")

    gui = RocketGUI(
        env,
        agent,
        results,
    )

    gui.run()


if __name__ == "__main__":
    main()