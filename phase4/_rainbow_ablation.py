
# ============================================================
# File: phase4/rainbow_ablation.py
#
# PHASE 4.8 — RAINBOW DQN ABLATION STUDY
#
# PURPOSE
#   Measure the contribution of individual Rainbow components.
#
# WHAT YOU LEARN
#   1. Baseline DQN
#   2. Double DQN
#   3. Dueling Network
#   4. Prioritized Experience Replay (PER)
#   5. Noisy Networks
#   6. N-step returns
#   7. Distributional RL (C51)
#   8. Ablation experiments and statistical comparison
#   9. Pygame visualization of a trained rocket agent
#
# EXPERIMENT DESIGN
#   Every experiment uses the same environment.
#   Each variant has the same episode budget.
#   Random seeds are controlled.
#
# IMPORTANT
#   This is a compact educational ablation framework.
#   It is not a reproduction of the original Rainbow paper.
#   Results from one seed are not statistically conclusive.
#
# C51 PROJECTION
#   This version fixes batch-aware probability projection:
#   each sample receives its own atom-index offset.
#
# ROBOTICS APPLICATION
#   Compare which learning mechanisms help a mobile robot
#   navigate around an obstacle while reaching a target.
#
# PYGAME CONTROLS
#   Left click : Change target
#   SPACE      : Pause / resume
#   R          : Reset rocket
#   +/-        : Simulation speed
#   ESC        : Exit
#
# INSTALL
#   pip install numpy pygame torch matplotlib
#
# RUN
#   python phase4/rainbow_ablation.py
# ============================================================

from __future__ import annotations

import copy
import math
import random
import time
from collections import deque
from dataclasses import dataclass

import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# ============================================================
# STEP 0 — CONFIGURATION
# ============================================================

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WORLD_W = 700
WORLD_H = 560

OBSTACLE = np.array([350.0, 280.0], dtype=np.float32)
OBSTACLE_RADIUS = 55.0
ROCKET_RADIUS = 11.0
SAFE_DISTANCE = OBSTACLE_RADIUS + ROCKET_RADIUS + 20.0

STATE_SIZE = 8
NUM_ACTIONS = 9

GAMMA = 0.99
LEARNING_RATE = 0.0003

HIDDEN = 128
BATCH_SIZE = 64
REPLAY_CAPACITY = 15_000
WARMUP = 500

EPISODES = 100
MAX_STEPS = 150

TARGET_UPDATE = 250

N_STEP = 3

NUM_ATOMS = 51
V_MIN = -20.0
V_MAX = 20.0

PER_ALPHA = 0.6
PER_BETA = 0.4
PER_EPS = 1e-5

NOISE_SIGMA = 0.5

WINDOW_W = 1020
WINDOW_H = 650

ACTION_VECTORS = np.array(
    [
        [1, 0],
        [1, -1],
        [0, -1],
        [-1, -1],
        [-1, 0],
        [-1, 1],
        [0, 1],
        [1, 1],
        [0, 0],
    ],
    dtype=np.float32,
)

for i in range(8):
    ACTION_VECTORS[i] /= np.linalg.norm(ACTION_VECTORS[i])

ACTION_NAMES = [
    "RIGHT", "UP-RIGHT", "UP", "UP-LEFT",
    "LEFT", "DOWN-LEFT", "DOWN", "DOWN-RIGHT", "STAY",
]


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ============================================================
# STEP 1 — ENVIRONMENT
# ============================================================

def dist(a, b):
    return float(np.linalg.norm(a - b))


def safe_position(p):
    inside = (
        ROCKET_RADIUS <= p[0] <= WORLD_W - ROCKET_RADIUS
        and ROCKET_RADIUS <= p[1] <= WORLD_H - ROCKET_RADIUS
    )
    return inside and dist(p, OBSTACLE) >= SAFE_DISTANCE


def valid_random_position():
    for _ in range(5000):
        p = np.array(
            [
                random.uniform(35, WORLD_W - 35),
                random.uniform(35, WORLD_H - 35),
            ],
            dtype=np.float32,
        )
        if safe_position(p):
            return p
    raise RuntimeError("Unable to generate safe position.")


class RocketEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        self.position = valid_random_position()
        self.target = valid_random_position()

        while dist(self.position, self.target) < 230:
            self.target = valid_random_position()

        self.steps = 0
        self.total_reward = 0.0
        self.success = False
        self.collision = False
        self.previous_action = 8

        return self.state()

    def set_target(self, target):
        target = np.asarray(target, dtype=np.float32)
        if not safe_position(target):
            return False
        if dist(target, self.position) < 20:
            return False

        self.target = target
        self.steps = 0
        self.total_reward = 0.0
        self.success = False
        self.collision = False
        self.previous_action = 8
        return True

    def state(self):
        target_delta = self.target - self.position
        obstacle_delta = OBSTACLE - self.position

        return np.array(
            [
                target_delta[0] / WORLD_W,
                target_delta[1] / WORLD_H,
                np.linalg.norm(target_delta) / 900.0,
                obstacle_delta[0] / WORLD_W,
                obstacle_delta[1] / WORLD_H,
                (np.linalg.norm(obstacle_delta) - SAFE_DISTANCE) / 900.0,
                self.position[0] / WORLD_W,
                self.position[1] / WORLD_H,
            ],
            dtype=np.float32,
        )

    def action_mask(self):
        mask = np.zeros(NUM_ACTIONS, dtype=bool)

        for a in range(8):
            p = self.position + ACTION_VECTORS[a] * 8.0
            mask[a] = safe_position(p)

        mask[8] = True
        return mask

    def step(self, action):
        old_distance = dist(self.position, self.target)

        next_position = (
            self.position + ACTION_VECTORS[action] * 8.0
        )

        reward = -0.04
        terminated = False
        truncated = False

        if not safe_position(next_position):
            reward -= 15.0
            self.collision = True
            terminated = True
        else:
            self.position = next_position
            new_distance = dist(self.position, self.target)

            reward += 0.12 * (old_distance - new_distance)

            if self.previous_action != 8 and action != 8:
                old_v = ACTION_VECTORS[self.previous_action]
                new_v = ACTION_VECTORS[action]

                if np.dot(old_v, new_v) < -0.5:
                    reward -= 0.08

            self.previous_action = action

            if new_distance <= 18:
                reward += 15.0
                self.success = True
                terminated = True

        self.steps += 1

        if self.steps >= MAX_STEPS and not terminated:
            truncated = True

        self.total_reward += reward

        return (
            self.state(),
            float(reward),
            terminated,
            truncated,
        )


# ============================================================
# STEP 2 — NOISY LINEAR LAYER
# ============================================================

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.weight_mu = nn.Parameter(
            torch.empty(out_features, in_features)
        )
        self.weight_sigma = nn.Parameter(
            torch.full(
                (out_features, in_features),
                NOISE_SIGMA / math.sqrt(in_features),
            )
        )

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(
            torch.full(
                (out_features,),
                NOISE_SIGMA / math.sqrt(out_features),
            )
        )

        self.register_buffer(
            "weight_epsilon",
            torch.zeros(out_features, in_features),
        )
        self.register_buffer(
            "bias_epsilon",
            torch.zeros(out_features),
        )

        bound = 1 / math.sqrt(in_features)
        nn.init.uniform_(self.weight_mu, -bound, bound)
        nn.init.uniform_(self.bias_mu, -bound, bound)

        self.reset_noise()

    @staticmethod
    def noise(size, device):
        x = torch.randn(size, device=device)
        return x.sign() * torch.sqrt(torch.abs(x))

    def reset_noise(self):
        e_in = self.noise(
            self.in_features,
            self.weight_mu.device,
        )
        e_out = self.noise(
            self.out_features,
            self.weight_mu.device,
        )

        self.weight_epsilon.copy_(e_out.outer(e_in))
        self.bias_epsilon.copy_(e_out)

    def forward(self, x):
        if self.training:
            w = self.weight_mu + self.weight_sigma * self.weight_epsilon
            b = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            w = self.weight_mu
            b = self.bias_mu

        return nn.functional.linear(x, w, b)


# ============================================================
# STEP 3 — CONFIGURABLE NETWORK
# ============================================================

@dataclass
class Experiment:
    name: str
    double: bool = False
    dueling: bool = False
    per: bool = False
    noisy: bool = False
    n_step: int = 1
    c51: bool = False


class QNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.features = nn.Sequential(
            nn.Linear(STATE_SIZE, HIDDEN),
            nn.ReLU(),
        )

        layer = NoisyLinear if config.noisy else nn.Linear

        output_size = NUM_ATOMS if config.c51 else 1

        if config.dueling:
            self.value = nn.Sequential(
                layer(HIDDEN, 64),
                nn.ReLU(),
                layer(64, output_size),
            )

            self.advantage = nn.Sequential(
                layer(HIDDEN, 64),
                nn.ReLU(),
                layer(64, NUM_ACTIONS * output_size),
            )
        else:
            self.head = nn.Sequential(
                layer(HIDDEN, 128),
                nn.ReLU(),
                layer(128, NUM_ACTIONS * output_size),
            )

    def forward(self, x):
        features = self.features(x)

        if self.config.dueling:
            value = self.value(features)
            advantage = self.advantage(features)

            if self.config.c51:
                value = value.view(-1, 1, NUM_ATOMS)
                advantage = advantage.view(
                    -1, NUM_ACTIONS, NUM_ATOMS
                )
                return (
                    value
                    + advantage
                    - advantage.mean(dim=1, keepdim=True)
                )

            value = value.view(-1, 1)
            advantage = advantage.view(-1, NUM_ACTIONS)

            return value + advantage - advantage.mean(
                dim=1,
                keepdim=True,
            )

        if self.config.c51:
            return self.head(features).view(
                -1,
                NUM_ACTIONS,
                NUM_ATOMS,
            )

        return self.head(features).view(-1, NUM_ACTIONS)

    def q_values(self, x):
        output = self.forward(x)

        if self.config.c51:
            support = torch.linspace(
                V_MIN,
                V_MAX,
                NUM_ATOMS,
                device=x.device,
            )
            return (
                torch.softmax(output, dim=-1)
                * support
            ).sum(dim=-1)

        return output

    def reset_noise(self):
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()


# ============================================================
# STEP 4 — C51 PROJECTION
# ============================================================

def c51_projection(next_prob, rewards, dones, gamma_n):
    """
    Correct batch-aware categorical projection.

    Every sample gets an offset:
        offset = batch_index * NUM_ATOMS

    Without this offset, samples can write their probability
    mass into the wrong rows of the flattened output tensor.
    """

    batch_size = next_prob.shape[0]

    support = torch.linspace(
        V_MIN,
        V_MAX,
        NUM_ATOMS,
        device=next_prob.device,
    )

    delta_z = (V_MAX - V_MIN) / (NUM_ATOMS - 1)

    target_atoms = (
        rewards[:, None]
        + gamma_n[:, None]
        * (1.0 - dones[:, None])
        * support[None, :]
    ).clamp(V_MIN, V_MAX)

    b = (target_atoms - V_MIN) / delta_z

    lower = b.floor().long()
    upper = b.ceil().long()

    lower = lower.clamp(0, NUM_ATOMS - 1)
    upper = upper.clamp(0, NUM_ATOMS - 1)

    lower_weight = upper.float() - b
    upper_weight = b - lower.float()

    same = lower == upper

    lower_weight = torch.where(
        same,
        torch.ones_like(lower_weight),
        lower_weight,
    )

    upper_weight = torch.where(
        same,
        torch.zeros_like(upper_weight),
        upper_weight,
    )

    offsets = (
        torch.arange(batch_size, device=next_prob.device)
        * NUM_ATOMS
    ).unsqueeze(1)

    lower_index = (lower + offsets).reshape(-1)
    upper_index = (upper + offsets).reshape(-1)

    projected = torch.zeros(
        batch_size * NUM_ATOMS,
        device=next_prob.device,
    )

    projected.index_add_(
        0,
        lower_index,
        (next_prob * lower_weight).reshape(-1),
    )

    projected.index_add_(
        0,
        upper_index,
        (next_prob * upper_weight).reshape(-1),
    )

    return projected.view(batch_size, NUM_ATOMS)


# ============================================================
# STEP 5 — REPLAY BUFFER
# ============================================================

@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    gamma_n: float
    next_mask: np.ndarray


class ReplayBuffer:
    def __init__(self, capacity, prioritized):
        self.capacity = capacity
        self.prioritized = prioritized
        self.memory = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.max_priority = 1.0

    def add(self, item):
        if len(self.memory) < self.capacity:
            self.memory.append(item)
        else:
            self.memory[self.position] = item

        self.priorities[self.position] = self.max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        size = len(self.memory)

        if self.prioritized:
            p = self.priorities[:size] ** PER_ALPHA
            p = p / p.sum()

            indices = np.random.choice(
                size,
                batch_size,
                replace=True,
                p=p,
            )

            weights = (size * p[indices]) ** (-PER_BETA)
            weights /= weights.max()
        else:
            indices = np.random.choice(
                size,
                batch_size,
                replace=False,
            )
            weights = np.ones(batch_size)

        return (
            [self.memory[i] for i in indices],
            indices,
            weights.astype(np.float32),
        )

    def update_priorities(self, indices, errors):
        for i, error in zip(indices, errors):
            p = abs(float(error)) + PER_EPS
            self.priorities[i] = p
            self.max_priority = max(self.max_priority, p)

    def __len__(self):
        return len(self.memory)


# ============================================================
# STEP 6 — N-STEP ACCUMULATOR
# ============================================================

class NStepBuffer:
    def __init__(self, n, gamma):
        self.n = n
        self.gamma = gamma
        self.queue = deque()

    def append(self, transition):
        self.queue.append(transition)
        output = []

        if len(self.queue) >= self.n:
            output.append(self._build(self.n))
            self.queue.popleft()

        if transition.done:
            while self.queue:
                output.append(
                    self._build(min(self.n, len(self.queue)))
                )
                self.queue.popleft()

        return output

    def _build(self, count):
        items = list(self.queue)[:count]

        first = items[0]
        last = items[-1]

        reward = sum(
            (self.gamma ** i) * item[2]
            for i, item in enumerate(items)
        )

        return Transition(
            state=first[0].copy(),
            action=first[1],
            reward=float(reward),
            next_state=last[3].copy(),
            done=last[4],
            gamma_n=self.gamma ** count,
            next_mask=last[5].copy(),
        )


# ============================================================
# STEP 7 — DQN AGENT
# ============================================================

class Agent:
    def __init__(self, config):
        self.config = config

        self.online = QNetwork(config).to(DEVICE)
        self.target = QNetwork(config).to(DEVICE)

        self.target.load_state_dict(
            self.online.state_dict()
        )

        self.optimizer = optim.Adam(
            self.online.parameters(),
            lr=LEARNING_RATE,
        )

        self.replay = ReplayBuffer(
            REPLAY_CAPACITY,
            config.per,
        )

        self.nstep = NStepBuffer(
            config.n_step,
            GAMMA,
        )

        self.updates = 0
        self.losses = []

        self.online.train()
        self.target.train()

    def reset_noise(self):
        self.online.reset_noise()
        self.target.reset_noise()

    def select_action(self, state, mask, epsilon=0.05):
        state_t = torch.tensor(
            state,
            dtype=torch.float32,
            device=DEVICE,
        ).unsqueeze(0)

        if self.config.noisy:
            self.online.train()
            self.online.reset_noise()

            with torch.no_grad():
                q = self.online.q_values(state_t)[0].cpu().numpy()
        else:
            if random.random() < epsilon:
                valid = np.flatnonzero(mask)
                return int(random.choice(valid))

            with torch.no_grad():
                q = self.online.q_values(state_t)[0].cpu().numpy()

        q[~mask] = -1e9
        return int(np.argmax(q))

    def remember(self, state, action, reward, next_state,
                 terminated, truncated, next_mask):

        # Time-limit truncation is treated as an episode boundary
        # for flushing n-step transitions, but not as an MDP
        # terminal state. Therefore it remains bootstrappable.
        episode_end = terminated or truncated

        raw = (
            state.copy(),
            int(action),
            float(reward),
            next_state.copy(),
            bool(terminated),
            next_mask.copy(),
        )

        for item in self.nstep.append(raw):
            self.replay.add(item)

        # A truncation must flush the pending queue too.
        if truncated and not terminated and self.nstep.queue:
            while self.nstep.queue:
                self.replay.add(
                    self.nstep._build(
                        min(self.nstep.n, len(self.nstep.queue))
                    )
                )
                self.nstep.queue.popleft()

    def train_step(self):
        if len(self.replay) < max(BATCH_SIZE, WARMUP):
            return None

        batch, indices, weights = self.replay.sample(BATCH_SIZE)

        states = torch.tensor(
            np.stack([t.state for t in batch]),
            dtype=torch.float32,
            device=DEVICE,
        )

        actions = torch.tensor(
            [t.action for t in batch],
            dtype=torch.long,
            device=DEVICE,
        )

        rewards = torch.tensor(
            [t.reward for t in batch],
            dtype=torch.float32,
            device=DEVICE,
        )

        next_states = torch.tensor(
            np.stack([t.next_state for t in batch]),
            dtype=torch.float32,
            device=DEVICE,
        )

        dones = torch.tensor(
            [t.done for t in batch],
            dtype=torch.float32,
            device=DEVICE,
        )

        gamma_n = torch.tensor(
            [t.gamma_n for t in batch],
            dtype=torch.float32,
            device=DEVICE,
        )

        next_masks = torch.tensor(
            np.stack([t.next_mask for t in batch]),
            dtype=torch.bool,
            device=DEVICE,
        )

        weights_t = torch.tensor(
            weights,
            dtype=torch.float32,
            device=DEVICE,
        )

        self.online.train()
        self.online.reset_noise()

        output = self.online(states)

        if self.config.c51:
            selected_logits = output[
                torch.arange(BATCH_SIZE, device=DEVICE),
                actions,
            ]

            log_prob = torch.log_softmax(
                selected_logits,
                dim=-1,
            )

            with torch.no_grad():
                self.online.reset_noise()
                self.target.reset_noise()

                online_next = self.online.q_values(next_states)
                online_next = online_next.masked_fill(
                    ~next_masks,
                    -1e9,
                )

                if self.config.double:
                    next_actions = online_next.argmax(dim=1)
                else:
                    target_next = self.target.q_values(next_states)
                    target_next = target_next.masked_fill(
                        ~next_masks,
                        -1e9,
                    )
                    next_actions = target_next.argmax(dim=1)

                target_probs = torch.softmax(
                    self.target(next_states),
                    dim=-1,
                )

                next_probs = target_probs[
                    torch.arange(BATCH_SIZE, device=DEVICE),
                    next_actions,
                ]

                projected = c51_projection(
                    next_probs,
                    rewards,
                    dones,
                    gamma_n,
                )

            per_sample_loss = -(
                projected * log_prob
            ).sum(dim=1)

            loss = (weights_t * per_sample_loss).mean()

            with torch.no_grad():
                support = torch.linspace(
                    V_MIN,
                    V_MAX,
                    NUM_ATOMS,
                    device=DEVICE,
                )

                current_q = (
                    torch.softmax(selected_logits, dim=-1)
                    * support
                ).sum(dim=1)

                target_q = (projected * support).sum(dim=1)

                errors = (target_q - current_q).abs()

        else:
            current_q = output.gather(
                1,
                actions.unsqueeze(1),
            ).squeeze(1)

            with torch.no_grad():
                online_next = self.online.q_values(next_states)
                online_next = online_next.masked_fill(
                    ~next_masks,
                    -1e9,
                )

                if self.config.double:
                    next_actions = online_next.argmax(dim=1)

                    next_q = self.target.q_values(next_states).gather(
                        1,
                        next_actions.unsqueeze(1),
                    ).squeeze(1)
                else:
                    target_next = self.target.q_values(next_states)
                    target_next = target_next.masked_fill(
                        ~next_masks,
                        -1e9,
                    )
                    next_q = target_next.max(dim=1).values

                target_q = (
                    rewards
                    + gamma_n * (1.0 - dones) * next_q
                )

            errors = (target_q - current_q).abs()

            per_sample_loss = nn.functional.smooth_l1_loss(
                current_q,
                target_q,
                reduction="none",
            )

            loss = (weights_t * per_sample_loss).mean()

        self.optimizer.zero_grad()
        loss.backward()

        nn.utils.clip_grad_norm_(
            self.online.parameters(),
            10.0,
        )

        self.optimizer.step()

        if self.config.per:
            self.replay.update_priorities(
                indices,
                errors.detach().cpu().numpy(),
            )

        self.updates += 1
        self.losses.append(float(loss.item()))

        if self.updates % TARGET_UPDATE == 0:
            self.target.load_state_dict(
                self.online.state_dict()
            )

        return float(loss.item())


# ============================================================
# STEP 8 — EXPERIMENT DEFINITIONS
# ============================================================

EXPERIMENTS = [
    Experiment(
        name="DQN baseline",
    ),
    Experiment(
        name="Double DQN",
        double=True,
    ),
    Experiment(
        name="Double + Dueling",
        double=True,
        dueling=True,
    ),
    Experiment(
        name="Double + Dueling + PER",
        double=True,
        dueling=True,
        per=True,
    ),
    Experiment(
        name="Rainbow without C51",
        double=True,
        dueling=True,
        per=True,
        noisy=True,
        n_step=3,
    ),
    Experiment(
        name="Rainbow full",
        double=True,
        dueling=True,
        per=True,
        noisy=True,
        n_step=3,
        c51=True,
    ),
]


# ============================================================
# STEP 9 — TRAINING AND EVALUATION
# ============================================================

def run_experiment(config, seed):
    seed_everything(seed)

    env = RocketEnv()
    agent = Agent(config)

    rewards = []
    successes = []
    collisions = []

    started = time.time()

    for episode in range(EPISODES):
        state = env.reset()
        agent.reset_noise()

        episode_reward = 0.0

        for _ in range(MAX_STEPS):
            mask = env.action_mask()

            action = agent.select_action(
                state,
                mask,
                epsilon=max(
                    0.03,
                    0.25 * (1 - episode / EPISODES),
                ),
            )

            next_state, reward, terminated, truncated = env.step(
                action
            )

            next_mask = env.action_mask()

            agent.remember(
                state,
                action,
                reward,
                next_state,
                terminated,
                truncated,
                next_mask,
            )

            agent.train_step()

            state = next_state
            episode_reward += reward

            if terminated or truncated:
                break

        rewards.append(episode_reward)
        successes.append(int(env.success))
        collisions.append(int(env.collision))

        if (episode + 1) % 25 == 0:
            start = max(0, episode - 24)

            print(
                f"{config.name:29s} "
                f"Episode {episode + 1:3d}/{EPISODES} | "
                f"Reward {np.mean(rewards[start:]):8.2f} | "
                f"Success {100*np.mean(successes[start:]):5.1f}%"
            )

    elapsed = time.time() - started

    return {
        "config": config,
        "agent": agent,
        "rewards": rewards,
        "successes": successes,
        "collisions": collisions,
        "elapsed": elapsed,
    }


def summarize(result):
    last = min(25, len(result["rewards"]))

    return {
        "name": result["config"].name,
        "avg_reward": float(np.mean(result["rewards"][-last:])),
        "success_rate": float(np.mean(result["successes"][-last:])),
        "collision_rate": float(np.mean(result["collisions"][-last:])),
        "seconds": result["elapsed"],
    }


# ============================================================
# STEP 10 — RESULTS VISUALIZATION
# ============================================================

def plot_results(results):
    plt.figure(figsize=(12, 6))

    for result in results:
        rewards = result["rewards"]

        window = min(15, len(rewards))

        smooth = np.convolve(
            rewards,
            np.ones(window) / window,
            mode="valid",
        )

        plt.plot(
            smooth,
            label=result["config"].name,
        )

    plt.title("Rainbow DQN Ablation — Episode Reward")
    plt.xlabel("Episode")
    plt.ylabel("Smoothed reward")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    summaries = [summarize(r) for r in results]

    names = [s["name"] for s in summaries]
    success = [100 * s["success_rate"] for s in summaries]

    plt.figure(figsize=(12, 5))
    plt.bar(names, success)
    plt.title("Success Rate — Final 25 Episodes")
    plt.ylabel("Success rate (%)")
    plt.xticks(rotation=25, ha="right")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.show()

    collision = [100 * s["collision_rate"] for s in summaries]

    plt.figure(figsize=(12, 5))
    plt.bar(names, collision)
    plt.title("Collision Rate — Final 25 Episodes")
    plt.ylabel("Collision rate (%)")
    plt.xticks(rotation=25, ha="right")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.show()

    print("\n" + "=" * 100)
    print("EXPERIMENT SUMMARY — FINAL 25 EPISODES")
    print("=" * 100)

    print(
        f"{'Configuration':30s} "
        f"{'Reward':>12s} "
        f"{'Success':>12s} "
        f"{'Collision':>12s} "
        f"{'Seconds':>12s}"
    )

    for s in summaries:
        print(
            f"{s['name']:30s} "
            f"{s['avg_reward']:12.2f} "
            f"{100*s['success_rate']:11.1f}% "
            f"{100*s['collision_rate']:11.1f}% "
            f"{s['seconds']:12.1f}"
        )

    print("=" * 100)


# ============================================================
# STEP 11 — PYGAME ROCKET VISUALIZATION
# ============================================================

class RocketGUI:
    def __init__(self, env, agent, config):
        pygame.init()

        pygame.display.set_caption(
            f"Phase 4.8 — {config.name}"
        )

        self.screen = pygame.display.set_mode(
            (WINDOW_W, WINDOW_H)
        )

        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 15)
        self.title_font = pygame.font.SysFont(
            "consolas",
            20,
            bold=True,
        )

        self.env = env
        self.agent = agent
        self.config = config

        self.paused = False
        self.running = True
        self.speed = 1

        self.action = 8
        self.previous_action = 8

        self.reset()

    def reset(self):
        self.env.position = np.array(
            [80.0, 280.0],
            dtype=np.float32,
        )

        self.env.target = np.array(
            [620.0, 280.0],
            dtype=np.float32,
        )

        self.env.steps = 0
        self.env.total_reward = 0.0
        self.env.success = False
        self.env.collision = False
        self.env.previous_action = 8

        self.action = 8
        self.previous_action = 8

    def text(self, value, x, y, color=(230, 230, 230), font=None):
        if font is None:
            font = self.font

        self.screen.blit(
            font.render(str(value), True, color),
            (x, y),
        )

    def draw_rocket(self):
        x, y = self.env.position.astype(int)

        vector = ACTION_VECTORS[self.action]

        if np.linalg.norm(vector) < 0.01:
            vector = ACTION_VECTORS[self.previous_action]

        angle = math.atan2(vector[1], vector[0])

        def rotate(point):
            lx, ly = point

            return (
                int(x + lx * math.cos(angle) - ly * math.sin(angle)),
                int(y + lx * math.sin(angle) + ly * math.cos(angle)),
            )

        body = [
            (20, 0),
            (5, -9),
            (-9, -7),
            (-13, -4),
            (-13, 4),
            (-9, 7),
            (5, 9),
        ]

        pygame.draw.polygon(
            self.screen,
            (255, 140, 30),
            [
                rotate((-13, -4)),
                rotate((-27, 0)),
                rotate((-13, 4)),
            ],
        )

        pygame.draw.polygon(
            self.screen,
            (230, 240, 250),
            [rotate(p) for p in body],
        )

        pygame.draw.polygon(
            self.screen,
            (70, 160, 230),
            [rotate(p) for p in body],
            2,
        )

        pygame.draw.circle(
            self.screen,
            (50, 140, 245),
            rotate((7, 0)),
            4,
        )

    def draw_world(self):
        self.screen.fill((18, 22, 30))

        pygame.draw.rect(
            self.screen,
            (28, 34, 44),
            (0, 0, WORLD_W, WORLD_H),
        )

        for x in range(0, WORLD_W, 40):
            pygame.draw.line(
                self.screen,
                (40, 47, 59),
                (x, 0),
                (x, WORLD_H),
            )

        for y in range(0, WORLD_H, 40):
            pygame.draw.line(
                self.screen,
                (40, 47, 59),
                (0, y),
                (WORLD_W, y),
            )

        pygame.draw.circle(
            self.screen,
            (130, 75, 55),
            tuple(OBSTACLE.astype(int)),
            int(SAFE_DISTANCE),
            2,
        )

        pygame.draw.circle(
            self.screen,
            (165, 55, 55),
            tuple(OBSTACLE.astype(int)),
            int(OBSTACLE_RADIUS),
        )

        tx, ty = self.env.target.astype(int)

        pygame.draw.circle(
            self.screen,
            (70, 240, 130),
            (tx, ty),
            14,
            2,
        )

        pygame.draw.line(
            self.screen,
            (70, 240, 130),
            (tx - 18, ty),
            (tx + 18, ty),
            2,
        )

        pygame.draw.line(
            self.screen,
            (70, 240, 130),
            (tx, ty - 18),
            (tx, ty + 18),
            2,
        )

        self.draw_rocket()

    def draw_panel(self):
        pygame.draw.rect(
            self.screen,
            (22, 27, 37),
            (WORLD_W, 0, WINDOW_W - WORLD_W, WINDOW_H),
        )

        self.text(
            "ABLATION STUDY",
            720,
            20,
            (120, 210, 255),
            self.title_font,
        )

        self.text(
            self.config.name,
            720,
            55,
            (100, 255, 160),
        )

        self.text(
            f"Position: {self.env.position.round(1)}",
            720,
            95,
        )

        self.text(
            f"Target: {self.env.target.round(1)}",
            720,
            125,
        )

        self.text(
            f"Distance: {dist(self.env.position, self.env.target):.1f}",
            720,
            155,
        )

        self.text(
            f"Action: {ACTION_NAMES[self.action]}",
            720,
            185,
        )

        self.text(
            f"Steps: {self.env.steps}",
            720,
            215,
        )

        self.text(
            f"Reward: {self.env.total_reward:.2f}",
            720,
            245,
        )

        self.text(
            f"Training updates: {self.agent.updates}",
            720,
            275,
        )

        self.text(
            f"Replay samples: {len(self.agent.replay)}",
            720,
            305,
        )

        self.text(
            f"Double DQN: {self.config.double}",
            720,
            350,
        )

        self.text(
            f"Dueling: {self.config.dueling}",
            720,
            375,
        )

        self.text(
            f"PER: {self.config.per}",
            720,
            400,
        )

        self.text(
            f"Noisy: {self.config.noisy}",
            720,
            425,
        )

        self.text(
            f"N-step: {self.config.n_step}",
            720,
            450,
        )

        self.text(
            f"C51: {self.config.c51}",
            720,
            475,
        )

        self.text(
            "CLICK: TARGET | SPACE: PAUSE | R: RESET",
            720,
            620,
        )

    def step(self):
        state = self.env.state()
        mask = self.env.action_mask()

        q = self.agent.online.q_values(
            torch.tensor(
                state,
                dtype=torch.float32,
                device=DEVICE,
            ).unsqueeze(0)
        ).detach().cpu().numpy()[0]

        q[~mask] = -1e9

        candidate = int(np.argmax(q))

        # Hysteresis: avoid switching directions for tiny
        # differences between action values.
        if (
            self.previous_action in np.flatnonzero(mask)
            and candidate != self.previous_action
            and q[self.previous_action] >= q[candidate] - 0.12
        ):
            candidate = self.previous_action

        self.action = candidate

        next_state, reward, terminated, truncated = self.env.step(
            self.action
        )

        self.previous_action = self.action

        if terminated or truncated:
            self.paused = True

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
                        self.reset()
                        self.paused = False

                    elif event.key in (
                        pygame.K_PLUS,
                        pygame.K_EQUALS,
                        pygame.K_KP_PLUS,
                    ):
                        self.speed = min(8, self.speed + 1)

                    elif event.key in (
                        pygame.K_MINUS,
                        pygame.K_KP_MINUS,
                    ):
                        self.speed = max(1, self.speed - 1)

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        x, y = event.pos

                        if 0 <= x <= WORLD_W and 0 <= y <= WORLD_H:
                            if self.env.set_target([x, y]):
                                self.previous_action = 8
                                self.action = 8
                                self.paused = False

            if not self.paused:
                for _ in range(self.speed):
                    self.step()

                    if self.paused:
                        break

            self.draw_world()
            self.draw_panel()

            if self.paused:
                self.text(
                    "PAUSED",
                    300,
                    25,
                    (255, 220, 80),
                    self.title_font,
                )

            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()


# ============================================================
# STEP 12 — UNIT TESTS
# ============================================================

def run_tests():
    print("\nPHASE 4.8 — UNIT TESTS")

    passed = 0

    def check(name, condition):
        nonlocal passed
        assert bool(condition), f"FAILED: {name}"
        passed += 1
        print(f"[PASS] {name}")

    # Test 1: environment state dimensions.
    env = RocketEnv()

    check(
        "Environment state shape",
        env.state().shape == (STATE_SIZE,),
    )

    # Test 2: action mask dimensions.
    check(
        "Action mask shape",
        env.action_mask().shape == (NUM_ACTIONS,),
    )

    # Test 3: stay action always valid.
    check(
        "Stay action is available",
        env.action_mask()[8],
    )

    # Test 4: all directional actions are normalized.
    norms = np.linalg.norm(ACTION_VECTORS[:8], axis=1)

    check(
        "Movement vectors normalized",
        np.allclose(norms, 1.0),
    )

    # Test 5: C51 projection preserves mass.
    batch = 4

    probs = torch.full(
        (batch, NUM_ATOMS),
        1.0 / NUM_ATOMS,
        device=DEVICE,
    )

    rewards = torch.tensor(
        [0.0, 1.0, -1.0, 2.0],
        device=DEVICE,
    )

    dones = torch.tensor(
        [0.0, 1.0, 0.0, 1.0],
        device=DEVICE,
    )

    gamma_n = torch.full(
        (batch,),
        GAMMA,
        device=DEVICE,
    )

    projected = c51_projection(
        probs,
        rewards,
        dones,
        gamma_n,
    )

    check(
        "C51 projection preserves probability mass",
        torch.allclose(
            projected.sum(dim=1),
            torch.ones(batch, device=DEVICE),
            atol=1e-5,
        ),
    )

    # Test 6: each C51 row remains independent.
    # Use a different terminal reward per sample.
    one_atom = torch.zeros(
        (batch, NUM_ATOMS),
        device=DEVICE,
    )

    center = NUM_ATOMS // 2
    one_atom[:, center] = 1.0

    different_rewards = torch.tensor(
        [-5.0, -1.0, 3.0, 8.0],
        device=DEVICE,
    )

    terminal = torch.ones(
        batch,
        device=DEVICE,
    )

    independent = c51_projection(
        one_atom,
        different_rewards,
        terminal,
        gamma_n,
    )

    support = torch.linspace(
        V_MIN,
        V_MAX,
        NUM_ATOMS,
        device=DEVICE,
    )

    projected_means = (
        independent * support
    ).sum(dim=1)

    check(
        "C51 samples project independently",
        torch.allclose(
            projected_means,
            different_rewards,
            atol=1e-4,
        ),
    )

    # Test 7: DQN output dimensions.
    dqn = QNetwork(
        Experiment("test")
    ).to(DEVICE)

    x = torch.randn(
        3,
        STATE_SIZE,
        device=DEVICE,
    )

    check(
        "Scalar DQN output dimensions",
        tuple(dqn(x).shape) == (3, NUM_ACTIONS),
    )

    # Test 8: Dueling output dimensions.
    dueling = QNetwork(
        Experiment("dueling", dueling=True)
    ).to(DEVICE)

    check(
        "Dueling output dimensions",
        tuple(dueling(x).shape) == (3, NUM_ACTIONS),
    )

    # Test 9: C51 output dimensions.
    c51 = QNetwork(
        Experiment("c51", c51=True)
    ).to(DEVICE)

    check(
        "C51 network output dimensions",
        tuple(c51(x).shape)
        == (3, NUM_ACTIONS, NUM_ATOMS),
    )

    # Test 10: noisy layer changes output while training.
    noisy = NoisyLinear(8, 16).to(DEVICE)
    noisy.train()

    input_tensor = torch.randn(2, 8, device=DEVICE)

    noisy.reset_noise()
    output_a = noisy(input_tensor)

    noisy.reset_noise()
    output_b = noisy(input_tensor)

    check(
        "Noisy layer can change output",
        not torch.allclose(output_a, output_b),
    )

    # Test 11: evaluation ignores noise.
    noisy.eval()
    output_a = noisy(input_tensor)

    noisy.reset_noise()
    output_b = noisy(input_tensor)

    check(
        "Noisy layer evaluation is deterministic",
        torch.allclose(output_a, output_b),
    )

    # Test 12: replay buffer capacity.
    replay = ReplayBuffer(3, prioritized=False)

    dummy = Transition(
        state=np.zeros(STATE_SIZE, dtype=np.float32),
        action=8,
        reward=1.0,
        next_state=np.ones(STATE_SIZE, dtype=np.float32),
        done=False,
        gamma_n=GAMMA,
        next_mask=np.ones(NUM_ACTIONS, dtype=bool),
    )

    for _ in range(5):
        replay.add(dummy)

    check(
        "Replay capacity is enforced",
        len(replay) == 3,
    )

    # Test 13: prioritized probabilities.
    per = ReplayBuffer(5, prioritized=True)

    for _ in range(5):
        per.add(dummy)

    sample, indices, weights = per.sample(4)

    check(
        "PER returns requested batch size",
        len(sample) == 4,
    )

    # Test 14: n-step return.
    nstep = NStepBuffer(3, 0.9)

    items = [
        (
            np.zeros(STATE_SIZE, dtype=np.float32),
            0,
            1.0,
            np.ones(STATE_SIZE, dtype=np.float32),
            False,
            np.ones(NUM_ACTIONS, dtype=bool),
        )
        for _ in range(3)
    ]

    output = []

    for item in items:
        output.extend(nstep.append(item))

    check(
        "N-step discounted return",
        np.isclose(output[0].reward, 2.71),
    )

    print(f"\n{passed}/14 tests passed.")


# ============================================================
# STEP 13 — MAIN
# ============================================================

def main():
    seed_everything(SEED)

    run_tests()

    print("\n" + "=" * 80)
    print("RAINBOW DQN ABLATION EXPERIMENTS")
    print("=" * 80)

    results = []

    # Change this to multiple seeds for a more reliable study.
    seeds = [SEED]

    for config in EXPERIMENTS:
        seed_results = []

        for seed in seeds:
            print(f"\nStarting: {config.name} | seed={seed}")

            result = run_experiment(
                config,
                seed,
            )

            seed_results.append(result)

        results.append(seed_results[0])

    plot_results(results)

    # Save the full Rainbow model.
    full_result = results[-1]

    torch.save(
        {
            "online": full_result["agent"].online.state_dict(),
            "target": full_result["agent"].target.state_dict(),
            "config": full_result["config"].__dict__,
        },
        "rainbow_ablation_full.pth",
    )

    print("\nSaved: rainbow_ablation_full.pth")

    # Visualize the trained full Rainbow model.
    gui_env = RocketEnv()

    gui = RocketGUI(
        gui_env,
        full_result["agent"],
        full_result["config"],
    )

    gui.run()


if __name__ == "__main__":
    main()
