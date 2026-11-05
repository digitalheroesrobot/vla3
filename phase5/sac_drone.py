
"""
File: phase5/sac_drone.py

PHASE 5.14 — SOFT ACTOR-CRITIC (SAC)
====================================

PURPOSE
-------
Implement Soft Actor-Critic for continuous-action quadcopter
control and compare it with TD3.

WHAT YOU LEARN
--------------

STEP 1 — STOCHASTIC GAUSSIAN ACTOR

    A policy produces:

        mean(s)
        log_std(s)

    Sample:

        z = mean + std * epsilon
        epsilon ~ Normal(0, I)

STEP 2 — TANH ACTION SQUASHING

        action = tanh(z)

    This constrains each action component to [-1, 1].

STEP 3 — CORRECTED LOG PROBABILITY

    log pi(a|s) =
        log Normal(z|mean,std)
        - sum(log(1 - tanh(z)^2))

    The correction is necessary because tanh transforms
    the probability distribution.

STEP 4 — TWIN CRITICS

        Q1(s,a)
        Q2(s,a)

    Use the smaller critic estimate.

STEP 5 — MAXIMUM ENTROPY OBJECTIVE

    Maximize:

        E[ Q(s,a) - alpha*log_pi(a|s) ]

    Entropy encourages exploration.

STEP 6 — AUTOMATIC TEMPERATURE ADJUSTMENT

    Learn alpha using:

        log_alpha

    This adapts the entropy strength during training.

STEP 7 — SOFT TARGET UPDATE

        target = tau*online + (1-tau)*target

STEP 8 — ROBOTICS APPLICATION

    A simplified 2D quadcopter tracks a continuously moving
    target while avoiding an obstacle.

    The observation includes target velocity.

IMPORTANT
---------
This is an educational 2D simulation, not a flight-certified
controller.

SAC does not automatically guarantee obstacle avoidance.
"""

from __future__ import annotations

import copy
import math
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import matplotlib.pyplot as plt


# ============================================================
# 0. CONFIGURATION
# ============================================================

SEED = 514

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

OBS_DIM = 10
ACTION_DIM = 2

WORLD_W = 900
WORLD_H = 650

DT = 0.08

MAX_SPEED = 150.0
MAX_ACCEL = 125.0
DRONE_RADIUS = 12.0

OBSTACLE = np.array(
    [450.0, 330.0],
    dtype=np.float32,
)

OBSTACLE_RADIUS = 62.0
SAFE_RADIUS = 112.0

TARGET_A = 145.0
TARGET_B = 95.0
TARGET_OMEGA = 0.55

GAMMA = 0.99
TAU = 0.005

ACTOR_LR = 3e-4
CRITIC_LR = 3e-4
ALPHA_LR = 3e-4

REPLAY_CAPACITY = 100_000
BATCH_SIZE = 128

INITIAL_RANDOM_STEPS = 2_000
TRAINING_STEPS = 35_000

MAX_EPISODE_STEPS = 400

# For a 2D continuous action, a common initial target entropy
# is -ACTION_DIM. This is a heuristic, not a universal optimum.
TARGET_ENTROPY = -float(ACTION_DIM)

GRADIENT_CLIP = 1.0

EVALUATION_INTERVAL = 5_000
EVALUATION_EPISODES = 4

LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0

EPSILON = 1e-6


# ============================================================
# 1. REPRODUCIBILITY
# ============================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 2. MOVING TARGET
# ============================================================

class MovingTarget:

    def __init__(self, center):

        self.center = np.asarray(
            center,
            dtype=np.float32,
        )

        self.phase = 0.0

    def position(self, phase=None):

        if phase is None:
            phase = self.phase

        return self.center + np.array(
            [
                TARGET_A * math.cos(phase),
                TARGET_B * math.sin(phase),
            ],
            dtype=np.float32,
        )

    def velocity(self, phase=None):

        if phase is None:
            phase = self.phase

        return np.array(
            [
                -TARGET_A * TARGET_OMEGA * math.sin(phase),
                TARGET_B * TARGET_OMEGA * math.cos(phase),
            ],
            dtype=np.float32,
        )

    def advance(self):

        self.phase += TARGET_OMEGA * DT


# ============================================================
# 3. DRONE DYNAMICS
# ============================================================

def clip_drone_state(state):

    state = np.asarray(
        state,
        dtype=np.float32,
    ).copy()

    speed = np.linalg.norm(state[2:4])

    if speed > MAX_SPEED:

        state[2:4] *= MAX_SPEED / (speed + 1e-9)

    state[0] = np.clip(
        state[0],
        DRONE_RADIUS,
        WORLD_W - DRONE_RADIUS,
    )

    state[1] = np.clip(
        state[1],
        DRONE_RADIUS,
        WORLD_H - DRONE_RADIUS,
    )

    return state


def drone_dynamics(state, action):

    action = np.clip(
        np.asarray(action, dtype=np.float32),
        -1.0,
        1.0,
    )

    velocity = state[2:4]

    acceleration = (
        MAX_ACCEL * 0.78 * action
        - 0.13 * velocity
        + np.array(
            [3.0, -2.0],
            dtype=np.float32,
        )
    )

    next_velocity = (
        velocity + acceleration * DT
    )

    next_position = (
        state[:2] + next_velocity * DT
    )

    return clip_drone_state(
        np.concatenate(
            [next_position, next_velocity]
        )
    )


# ============================================================
# 4. DRONE ENVIRONMENT
# ============================================================

class DroneEnv:

    def __init__(self, seed=None):

        self.rng = np.random.default_rng(seed)

        self.target = MovingTarget(
            [700.0, 310.0]
        )

        self.state = None
        self.steps = 0

        self.reset()

    def observation(self):

        position = self.state[:2]
        velocity = self.state[2:4]

        target_position = self.target.position()
        target_velocity = self.target.velocity()

        return np.concatenate(
            [
                (target_position - position) / 400.0,
                target_velocity / 200.0,
                velocity / MAX_SPEED,
                (OBSTACLE - position) / 400.0,
                [
                    math.sin(self.target.phase),
                    math.cos(self.target.phase),
                ],
            ]
        ).astype(np.float32)

    def reset(self):

        self.state = np.array(
            [
                self.rng.uniform(80, 140),
                self.rng.uniform(440, 540),
                0.0,
                0.0,
            ],
            dtype=np.float32,
        )

        self.target.phase = self.rng.uniform(
            0.0,
            2.0 * math.pi,
        )

        self.steps = 0

        return self.observation()

    def step(self, action):

        action = np.clip(
            np.asarray(action, dtype=np.float32),
            -1.0,
            1.0,
        )

        old_position = self.state[:2].copy()

        # Compare both positions against the same future target
        # position. This avoids confusing target movement with
        # progress made by the drone.

        future_phase = (
            self.target.phase + TARGET_OMEGA * DT
        )

        future_target = self.target.position(
            future_phase
        )

        distance_before = np.linalg.norm(
            old_position - future_target
        )

        self.state = drone_dynamics(
            self.state,
            action,
        )

        self.target.advance()

        distance_after = np.linalg.norm(
            self.state[:2] - self.target.position()
        )

        obstacle_distance = np.linalg.norm(
            self.state[:2] - OBSTACLE
        )

        boundary_distance = min(
            self.state[0],
            WORLD_W - self.state[0],
            self.state[1],
            WORLD_H - self.state[1],
        )

        collision = (
            obstacle_distance < OBSTACLE_RADIUS
        )

        reward = (
            0.08 * (distance_before - distance_after)
            - 0.010 * distance_after
            - 0.10 * np.sum(action ** 2)
        )

        reward -= (
            0.006
            * max(
                0.0,
                SAFE_RADIUS - obstacle_distance,
            ) ** 2
        )

        reward -= (
            0.015
            * max(
                0.0,
                55.0 - boundary_distance,
            ) ** 2
        )

        if collision:
            reward -= 100.0

        self.steps += 1

        terminated = bool(collision)

        truncated = (
            self.steps >= MAX_EPISODE_STEPS
        )

        return (
            self.observation(),
            float(reward),
            terminated,
            truncated,
            {
                "distance": float(distance_after),
                "collision": collision,
            },
        )


# ============================================================
# 5. REPLAY BUFFER
# ============================================================

class ReplayBuffer:

    def __init__(self, capacity=REPLAY_CAPACITY):

        self.buffer = deque(
            maxlen=capacity
        )

    def add(
        self,
        state,
        action,
        reward,
        next_state,
        terminated,
    ):

        self.buffer.append(
            (
                np.asarray(
                    state,
                    dtype=np.float32,
                ).copy(),

                np.asarray(
                    action,
                    dtype=np.float32,
                ).copy(),

                float(reward),

                np.asarray(
                    next_state,
                    dtype=np.float32,
                ).copy(),

                float(terminated),
            )
        )

    def sample(self, batch_size):

        batch = random.sample(
            self.buffer,
            batch_size,
        )

        arrays = (
            np.stack([x[0] for x in batch]),
            np.stack([x[1] for x in batch]),
            np.asarray(
                [x[2] for x in batch],
                dtype=np.float32,
            ).reshape(-1, 1),
            np.stack([x[3] for x in batch]),
            np.asarray(
                [x[4] for x in batch],
                dtype=np.float32,
            ).reshape(-1, 1),
        )

        return tuple(
            torch.as_tensor(
                array,
                dtype=torch.float32,
                device=DEVICE,
            )
            for array in arrays
        )

    def __len__(self):

        return len(self.buffer)


# ============================================================
# 6. GAUSSIAN ACTOR
# ============================================================

class GaussianActor(nn.Module):

    def __init__(self):

        super().__init__()

        self.backbone = nn.Sequential(
            nn.Linear(OBS_DIM, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        self.mean_layer = nn.Linear(
            256,
            ACTION_DIM,
        )

        self.log_std_layer = nn.Linear(
            256,
            ACTION_DIM,
        )

    def forward(self, states):

        features = self.backbone(states)

        mean = self.mean_layer(features)

        log_std = self.log_std_layer(
            features
        )

        log_std = torch.clamp(
            log_std,
            LOG_STD_MIN,
            LOG_STD_MAX,
        )

        return mean, log_std

    def sample(self, states, deterministic=False):

        mean, log_std = self(states)

        std = log_std.exp()

        if deterministic:

            raw_action = mean

        else:

            # SAC uses reparameterized sampling:
            #
            # raw_action = mean + std * epsilon
            #
            # This allows gradients to pass through the sample
            # during actor optimization.

            distribution = Normal(
                mean,
                std,
            )

            raw_action = distribution.rsample()

        action = torch.tanh(
            raw_action
        )

        if deterministic:

            return action, None

        distribution = Normal(
            mean,
            std,
        )

        raw_log_prob = distribution.log_prob(
            raw_action
        )

        # Tanh change-of-variables correction:
        #
        # log(1 - tanh(z)^2)
        #
        # Stable equivalent:
        #
        # 2 * (log(2) - z - softplus(-2z))

        log_tanh_jacobian = 2.0 * (
            math.log(2.0)
            - raw_action
            - F.softplus(-2.0 * raw_action)
        )

        log_prob = (
            raw_log_prob - log_tanh_jacobian
        ).sum(
            dim=-1,
            keepdim=True,
        )

        return action, log_prob


# ============================================================
# 7. CRITIC NETWORK
# ============================================================

class Critic(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                OBS_DIM + ACTION_DIM,
                256,
            ),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, states, actions):

        inputs = torch.cat(
            [states, actions],
            dim=-1,
        )

        return self.network(inputs)


# ============================================================
# 8. SOFT ACTOR-CRITIC AGENT
# ============================================================

class SACAgent:

    def __init__(self):

        self.actor = GaussianActor().to(DEVICE)

        self.critic1 = Critic().to(DEVICE)
        self.critic2 = Critic().to(DEVICE)

        self.target_critic1 = copy.deepcopy(
            self.critic1
        ).to(DEVICE)

        self.target_critic2 = copy.deepcopy(
            self.critic2
        ).to(DEVICE)

        # Target critics are not optimized directly.

        for parameter in self.target_critic1.parameters():
            parameter.requires_grad_(False)

        for parameter in self.target_critic2.parameters():
            parameter.requires_grad_(False)

        self.actor_optimizer = optim.Adam(
            self.actor.parameters(),
            lr=ACTOR_LR,
        )

        self.critic_optimizer = optim.Adam(
            list(self.critic1.parameters())
            + list(self.critic2.parameters()),
            lr=CRITIC_LR,
        )

        # Learn log(alpha), rather than alpha directly,
        # to keep alpha positive after exponentiation.

        self.log_alpha = torch.tensor(
            0.0,
            dtype=torch.float32,
            device=DEVICE,
            requires_grad=True,
        )

        self.alpha_optimizer = optim.Adam(
            [self.log_alpha],
            lr=ALPHA_LR,
        )

        self.target_entropy = TARGET_ENTROPY

        self.replay = ReplayBuffer()

        self.updates = 0

    @property
    def alpha(self):

        return self.log_alpha.exp().detach()

    def select_action(
        self,
        observation,
        deterministic=False,
    ):

        state = torch.as_tensor(
            observation,
            dtype=torch.float32,
            device=DEVICE,
        ).unsqueeze(0)

        with torch.no_grad():

            action, _ = self.actor.sample(
                state,
                deterministic=deterministic,
            )

        return action.squeeze(0).cpu().numpy().astype(
            np.float32
        )

    def store(
        self,
        state,
        action,
        reward,
        next_state,
        terminated,
    ):

        self.replay.add(
            state,
            action,
            reward,
            next_state,
            terminated,
        )

    def update(self):

        if len(self.replay) < BATCH_SIZE:

            return None

        (
            states,
            actions,
            rewards,
            next_states,
            terminated,
        ) = self.replay.sample(
            BATCH_SIZE
        )

        # ----------------------------------------------------
        # STEP 1 — SOFT BELLMAN TARGET
        # ----------------------------------------------------

        with torch.no_grad():

            next_actions, next_log_prob = (
                self.actor.sample(
                    next_states
                )
            )

            target_q1 = self.target_critic1(
                next_states,
                next_actions,
            )

            target_q2 = self.target_critic2(
                next_states,
                next_actions,
            )

            target_min_q = torch.minimum(
                target_q1,
                target_q2,
            )

            # Entropy-regularized target:
            #
            # y = r + gamma*(1-done)*
            #     (min(Q1,Q2) - alpha*log_pi)

            soft_next_q = (
                target_min_q
                - self.alpha * next_log_prob
            )

            targets = (
                rewards
                + GAMMA
                * (1.0 - terminated)
                * soft_next_q
            )

        # ----------------------------------------------------
        # STEP 2 — CRITIC UPDATE
        # ----------------------------------------------------

        current_q1 = self.critic1(
            states,
            actions,
        )

        current_q2 = self.critic2(
            states,
            actions,
        )

        critic1_loss = F.mse_loss(
            current_q1,
            targets,
        )

        critic2_loss = F.mse_loss(
            current_q2,
            targets,
        )

        critic_loss = (
            critic1_loss + critic2_loss
        )

        self.critic_optimizer.zero_grad()

        critic_loss.backward()

        nn.utils.clip_grad_norm_(
            list(self.critic1.parameters())
            + list(self.critic2.parameters()),
            GRADIENT_CLIP,
        )

        self.critic_optimizer.step()

        # ----------------------------------------------------
        # STEP 3 — ACTOR UPDATE
        # ----------------------------------------------------

        sampled_actions, log_prob = (
            self.actor.sample(states)
        )

        # Freeze critic parameters while computing the
        # actor gradient. Gradients still flow through the
        # critic with respect to the sampled actions.

        for parameter in self.critic1.parameters():
            parameter.requires_grad_(False)

        for parameter in self.critic2.parameters():
            parameter.requires_grad_(False)

        actor_q1 = self.critic1(
            states,
            sampled_actions,
        )

        actor_q2 = self.critic2(
            states,
            sampled_actions,
        )

        actor_min_q = torch.minimum(
            actor_q1,
            actor_q2,
        )

        # SAC actor objective:
        #
        # minimize E[alpha*log_pi(a|s) - min(Q1,Q2)]

        actor_loss = (
            self.alpha * log_prob
            - actor_min_q
        ).mean()

        self.actor_optimizer.zero_grad()

        actor_loss.backward()

        nn.utils.clip_grad_norm_(
            self.actor.parameters(),
            GRADIENT_CLIP,
        )

        self.actor_optimizer.step()

        for parameter in self.critic1.parameters():
            parameter.requires_grad_(True)

        for parameter in self.critic2.parameters():
            parameter.requires_grad_(True)

        # ----------------------------------------------------
        # STEP 4 — AUTOMATIC TEMPERATURE UPDATE
        # ----------------------------------------------------

        # Detach log probability here. The alpha update
        # should optimize alpha, not the actor.

        alpha_loss = -(
            self.log_alpha
            * (
                log_prob.detach()
                + self.target_entropy
            )
        ).mean()

        self.alpha_optimizer.zero_grad()

        alpha_loss.backward()

        self.alpha_optimizer.step()

        # ----------------------------------------------------
        # STEP 5 — SOFT TARGET UPDATES
        # ----------------------------------------------------

        self.soft_update(
            self.critic1,
            self.target_critic1,
        )

        self.soft_update(
            self.critic2,
            self.target_critic2,
        )

        self.updates += 1

        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "alpha": self.alpha.item(),
            "entropy": (-log_prob.detach()).mean().item(),
            "q1_mean": current_q1.detach().mean().item(),
            "q2_mean": current_q2.detach().mean().item(),
        }

    @staticmethod
    def soft_update(source, target):

        with torch.no_grad():

            for source_parameter, target_parameter in zip(
                source.parameters(),
                target.parameters(),
            ):

                target_parameter.mul_(
                    1.0 - TAU
                )

                target_parameter.add_(
                    TAU * source_parameter
                )


# ============================================================
# 9. TRAINING AND EVALUATION
# ============================================================

def evaluate(agent, episodes=EVALUATION_EPISODES):

    returns = []
    distances = []
    collisions = []

    for episode in range(episodes):

        env = DroneEnv(
            seed=30_000 + episode
        )

        observation = env.reset()

        total_reward = 0.0
        episode_distances = []
        collision_count = 0

        for _ in range(MAX_EPISODE_STEPS):

            action = agent.select_action(
                observation,
                deterministic=True,
            )

            (
                observation,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(action)

            total_reward += reward

            episode_distances.append(
                info["distance"]
            )

            collision_count += int(
                info["collision"]
            )

            if terminated or truncated:
                break

        returns.append(total_reward)

        distances.append(
            float(np.mean(episode_distances))
        )

        collisions.append(
            collision_count
        )

    return {
        "return": float(np.mean(returns)),
        "distance": float(np.mean(distances)),
        "collisions": int(np.sum(collisions)),
    }


def train_agent(agent):

    env = DroneEnv(
        seed=SEED + 1
    )

    history = {
        "episode_return": [],
        "episode_distance": [],
        "actor_loss": [],
        "critic_loss": [],
        "alpha": [],
        "entropy": [],
        "evaluation_step": [],
        "evaluation_return": [],
        "evaluation_distance": [],
        "evaluation_collisions": [],
    }

    observation = env.reset()

    episode_return = 0.0
    episode_distances = []

    for step in range(TRAINING_STEPS):

        if step < INITIAL_RANDOM_STEPS:

            action = np.random.uniform(
                -1.0,
                1.0,
                size=ACTION_DIM,
            ).astype(np.float32)

        else:

            action = agent.select_action(
                observation,
                deterministic=False,
            )

        (
            next_observation,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(action)

        agent.store(
            observation,
            action,
            reward,
            next_observation,
            terminated,
        )

        observation = next_observation

        episode_return += reward

        episode_distances.append(
            info["distance"]
        )

        metrics = agent.update()

        if metrics is not None:

            history["actor_loss"].append(
                metrics["actor_loss"]
            )

            history["critic_loss"].append(
                metrics["critic_loss"]
            )

            history["alpha"].append(
                metrics["alpha"]
            )

            history["entropy"].append(
                metrics["entropy"]
            )

        if terminated or truncated:

            history["episode_return"].append(
                episode_return
            )

            history["episode_distance"].append(
                float(np.mean(episode_distances))
            )

            episode_number = len(
                history["episode_return"]
            )

            if episode_number % 5 == 0:

                alpha_value = (
                    agent.alpha.item()
                    if len(history["alpha"]) > 0
                    else float("nan")
                )

                print(
                    f"[SAC]"
                    f" Step={step + 1:6d}"
                    f" Episodes={episode_number:4d}"
                    f" Return={episode_return:8.2f}"
                    f" Distance={np.mean(episode_distances):7.2f}"
                    f" Alpha={alpha_value:.4f}"
                )

            observation = env.reset()

            episode_return = 0.0
            episode_distances = []

        if (
            step > INITIAL_RANDOM_STEPS
            and (step + 1) % EVALUATION_INTERVAL == 0
        ):

            result = evaluate(agent)

            history["evaluation_step"].append(
                step + 1
            )

            history["evaluation_return"].append(
                result["return"]
            )

            history["evaluation_distance"].append(
                result["distance"]
            )

            history["evaluation_collisions"].append(
                result["collisions"]
            )

            print(
                f"\n[SAC EVALUATION]"
                f" Step={step + 1}"
                f" Return={result['return']:.2f}"
                f" Distance={result['distance']:.2f}"
                f" Collisions={result['collisions']}"
                f" Alpha={agent.alpha.item():.4f}\n"
            )

    return history


# ============================================================
# 10. VISUALIZATION
# ============================================================

def moving_average(values, window=20):

    values = np.asarray(
        values,
        dtype=np.float32,
    )

    if len(values) < window:
        return values

    return np.convolve(
        values,
        np.ones(window, dtype=np.float32) / window,
        mode="valid",
    )


def plot_training(history):

    plots = [
        (
            "episode_return",
            "SAC Episode Return",
        ),
        (
            "episode_distance",
            "SAC Mean Target Distance",
        ),
        (
            "critic_loss",
            "SAC Critic Loss",
        ),
        (
            "actor_loss",
            "SAC Actor Loss",
        ),
        (
            "alpha",
            "Automatic Entropy Temperature",
        ),
        (
            "entropy",
            "Policy Entropy",
        ),
    ]

    for key, title in plots:

        values = history[key]

        if len(values) == 0:
            continue

        plt.figure(figsize=(10, 4))

        plt.plot(
            moving_average(values),
        )

        plt.title(title)
        plt.xlabel("Episode or gradient update")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    if history["evaluation_step"]:

        plt.figure(figsize=(10, 4))

        plt.plot(
            history["evaluation_step"],
            history["evaluation_distance"],
            marker="o",
        )

        plt.title(
            "SAC Deterministic Evaluation Distance"
        )

        plt.xlabel("Training steps")
        plt.ylabel("Mean target distance")
        plt.grid(True)
        plt.tight_layout()
        plt.show()


# ============================================================
# 11. PYGAME QUADCOPTER VISUALIZATION
# ============================================================

def run_pygame(agent):

    try:

        import pygame

    except ImportError:

        print(
            "Install pygame: pip install pygame"
        )

        return

    pygame.init()

    screen = pygame.display.set_mode(
        (WORLD_W, WORLD_H)
    )

    pygame.display.set_caption(
        "Phase 5.14 — SAC Moving-Target Quadcopter"
    )

    clock = pygame.time.Clock()

    font = pygame.font.SysFont(
        "consolas",
        15,
    )

    env = DroneEnv(
        seed=SEED + 100
    )

    observation = env.reset()

    paused = False
    speed = 1

    trail = []

    episode_reward = 0.0
    last_distance = 0.0

    def clamp_center(position):

        return np.array(
            [
                np.clip(
                    position[0],
                    TARGET_A + 25,
                    WORLD_W - TARGET_A - 25,
                ),
                np.clip(
                    position[1],
                    TARGET_B + 25,
                    WORLD_H - TARGET_B - 25,
                ),
            ],
            dtype=np.float32,
        )

    def draw_text(message, x, y):

        screen.blit(
            font.render(
                message,
                True,
                (240, 240, 240),
            ),
            (x, y),
        )

    def draw_target():

        points = []

        for index in range(100):

            phase = (
                env.target.phase
                + 2.0 * math.pi * index / 100
            )

            points.append(
                env.target.position(
                    phase
                ).astype(int)
            )

        pygame.draw.lines(
            screen,
            (50, 100, 70),
            True,
            points,
            2,
        )

        position = env.target.position()
        velocity = env.target.velocity()

        pygame.draw.circle(
            screen,
            (70, 245, 105),
            position.astype(int),
            11,
        )

        pygame.draw.line(
            screen,
            (70, 255, 115),
            position.astype(int),
            (
                position + velocity * 0.18
            ).astype(int),
            3,
        )

    def draw_drone():

        center = env.state[:2]
        velocity = env.state[2:4]

        magnitude = np.linalg.norm(
            velocity
        )

        if magnitude > 1e-5:

            direction = velocity / magnitude

        else:

            direction = np.array(
                [1.0, 0.0]
            )

        side = np.array(
            [-direction[1], direction[0]]
        )

        for forward in (-1, 1):

            for lateral in (-1, 1):

                rotor = (
                    center
                    + direction * forward * 17
                    + side * lateral * 17
                )

                pygame.draw.line(
                    screen,
                    (230, 235, 245),
                    center.astype(int),
                    rotor.astype(int),
                    5,
                )

                pygame.draw.circle(
                    screen,
                    (80, 190, 245),
                    rotor.astype(int),
                    7,
                    2,
                )

        pygame.draw.circle(
            screen,
            (250, 250, 250),
            center.astype(int),
            8,
        )

        pygame.draw.line(
            screen,
            (90, 160, 255),
            center.astype(int),
            (
                center + velocity * 0.4
            ).astype(int),
            3,
        )

    running = True

    while running:

        clock.tick(60)

        for event in pygame.event.get():

            if event.type == pygame.QUIT:

                running = False

            elif event.type == pygame.KEYDOWN:

                if event.key == pygame.K_ESCAPE:

                    running = False

                elif event.key == pygame.K_SPACE:

                    paused = not paused

                elif event.key == pygame.K_r:

                    observation = env.reset()

                    trail.clear()

                    episode_reward = 0.0

                elif event.key in (
                    pygame.K_PLUS,
                    pygame.K_EQUALS,
                    pygame.K_KP_PLUS,
                ):

                    speed = min(
                        5,
                        speed + 1,
                    )

                elif event.key in (
                    pygame.K_MINUS,
                    pygame.K_KP_MINUS,
                ):

                    speed = max(
                        1,
                        speed - 1,
                    )

            elif event.type == pygame.MOUSEBUTTONDOWN:

                if event.button == 1:

                    env.target.center = clamp_center(
                        np.asarray(
                            event.pos,
                            dtype=np.float32,
                        )
                    )

        if not paused:

            for _ in range(speed):

                action = agent.select_action(
                    observation,
                    deterministic=True,
                )

                (
                    observation,
                    reward,
                    terminated,
                    truncated,
                    info,
                ) = env.step(action)

                episode_reward += reward

                last_distance = info["distance"]

                trail.append(
                    env.state[:2].copy()
                )

                if len(trail) > 250:

                    trail.pop(0)

                if terminated or truncated:

                    observation = env.reset()

                    trail.clear()

                    episode_reward = 0.0

        screen.fill(
            (17, 23, 34)
        )

        for x in range(0, WORLD_W, 50):

            pygame.draw.line(
                screen,
                (30, 38, 50),
                (x, 0),
                (x, WORLD_H),
            )

        for y in range(0, WORLD_H, 50):

            pygame.draw.line(
                screen,
                (30, 38, 50),
                (0, y),
                (WORLD_W, y),
            )

        pygame.draw.circle(
            screen,
            (105, 105, 115),
            OBSTACLE.astype(int),
            OBSTACLE_RADIUS,
        )

        pygame.draw.circle(
            screen,
            (190, 95, 95),
            OBSTACLE.astype(int),
            SAFE_RADIUS,
            1,
        )

        draw_target()

        if len(trail) > 1:

            pygame.draw.lines(
                screen,
                (75, 135, 235),
                False,
                [
                    point.astype(int)
                    for point in trail
                ],
                2,
            )

        draw_drone()

        draw_text(
            "PHASE 5.14 — SOFT ACTOR-CRITIC",
            15,
            12,
        )

        draw_text(
            f"Target distance: {last_distance:.2f}",
            15,
            40,
        )

        draw_text(
            f"Episode reward: {episode_reward:.2f}",
            15,
            65,
        )

        draw_text(
            f"Entropy temperature alpha: {agent.alpha.item():.4f}",
            15,
            90,
        )

        draw_text(
            f"Simulation speed: {speed}x",
            15,
            115,
        )

        draw_text(
            "Click: change moving-target trajectory center",
            15,
            WORLD_H - 100,
        )

        draw_text(
            "SPACE pause | R reset | +/- speed | ESC exit",
            15,
            WORLD_H - 70,
        )

        if paused:

            draw_text(
                "PAUSED",
                WORLD_W - 100,
                15,
            )

        pygame.display.flip()

    pygame.quit()


# ============================================================
# 12. NUMERICAL TESTS
# ============================================================

def run_tests():

    print("\nRunning SAC tests...")

    passed = 0

    def check(condition, description):

        nonlocal passed

        assert condition, description

        passed += 1

        print("[PASS]", description)

    # Test 1: Gaussian actor output shapes.

    actor = GaussianActor().to(DEVICE)

    states = torch.randn(
        16,
        OBS_DIM,
        device=DEVICE,
    )

    mean, log_std = actor(states)

    check(
        mean.shape == (16, ACTION_DIM)
        and log_std.shape == (16, ACTION_DIM),
        "Gaussian actor output shapes",
    )

    # Test 2: Standard deviation is positive.

    std = log_std.exp()

    check(
        bool(torch.all(std > 0)),
        "Gaussian standard deviation positivity",
    )

    # Test 3: Log standard deviation bounds.

    check(
        bool(
            torch.all(log_std >= LOG_STD_MIN)
            and torch.all(log_std <= LOG_STD_MAX)
        ),
        "Log standard deviation bounds",
    )

    # Test 4: Tanh action bounds.

    actions, log_prob = actor.sample(states)

    check(
        bool(torch.all(actions.abs() <= 1.0)),
        "Tanh action bounds",
    )

    # Test 5: Log probability shape.

    check(
        log_prob.shape == (16, 1),
        "Log probability shape",
    )

    # Test 6: Log probability finite.

    check(
        bool(torch.isfinite(log_prob).all()),
        "Log probability numerical stability",
    )

    # Test 7: Deterministic action.

    deterministic_actions, deterministic_log_prob = (
        actor.sample(
            states,
            deterministic=True,
        )
    )

    check(
        deterministic_actions.shape == (
            16,
            ACTION_DIM,
        )
        and deterministic_log_prob is None,
        "Deterministic actor output",
    )

    # Test 8: Twin critic shapes.

    critic1 = Critic().to(DEVICE)
    critic2 = Critic().to(DEVICE)

    q1 = critic1(states, actions)
    q2 = critic2(states, actions)

    check(
        q1.shape == (16, 1)
        and q2.shape == (16, 1),
        "Twin critic output shapes",
    )

    # Test 9: Clipped Double Q.

    q1_test = torch.tensor(
        [[5.0], [2.0]],
        device=DEVICE,
    )

    q2_test = torch.tensor(
        [[3.0], [4.0]],
        device=DEVICE,
    )

    minimum = torch.minimum(
        q1_test,
        q2_test,
    )

    check(
        torch.allclose(
            minimum,
            torch.tensor(
                [[3.0], [2.0]],
                device=DEVICE,
            ),
        ),
        "Clipped Double Q calculation",
    )

    # Test 10: Terminal Bellman target.

    reward = torch.tensor(
        [[2.0]],
        device=DEVICE,
    )

    terminal = torch.tensor(
        [[1.0]],
        device=DEVICE,
    )

    soft_q = torch.tensor(
        [[5.0]],
        device=DEVICE,
    )

    target = (
        reward
        + GAMMA
        * (1.0 - terminal)
        * soft_q
    )

    check(
        torch.allclose(
            target,
            reward,
        ),
        "Terminal Bellman target",
    )

    # Test 11: Nonterminal Bellman target.

    nonterminal = torch.tensor(
        [[0.0]],
        device=DEVICE,
    )

    target = (
        reward
        + GAMMA
        * (1.0 - nonterminal)
        * soft_q
    )

    check(
        torch.allclose(
            target,
            torch.tensor(
                [[2.0 + GAMMA * 5.0]],
                device=DEVICE,
            ),
        ),
        "Nonterminal Bellman target",
    )

    # Test 12: Entropy-adjusted Q target.

    alpha = torch.tensor(
        0.2,
        device=DEVICE,
    )

    q_value = torch.tensor(
        [[5.0]],
        device=DEVICE,
    )

    log_probability = torch.tensor(
        [[-2.0]],
        device=DEVICE,
    )

    soft_value = (
        q_value - alpha * log_probability
    )

    check(
        torch.allclose(
            soft_value,
            torch.tensor(
                [[5.4]],
                device=DEVICE,
            ),
        ),
        "Entropy-adjusted target",
    )

    # Test 13: Alpha positivity.

    log_alpha = torch.tensor(
        -2.0,
        device=DEVICE,
    )

    check(
        bool(torch.exp(log_alpha) > 0),
        "Positive entropy temperature",
    )

    # Test 14: Soft update.

    source = nn.Linear(3, 2)
    target_network = nn.Linear(3, 2)

    with torch.no_grad():

        for parameter in source.parameters():
            parameter.fill_(2.0)

        for parameter in target_network.parameters():
            parameter.fill_(0.0)

        for source_parameter, target_parameter in zip(
            source.parameters(),
            target_network.parameters(),
        ):

            target_parameter.mul_(0.75)

            target_parameter.add_(
                0.25 * source_parameter
            )

    check(
        all(
            torch.allclose(
                parameter,
                torch.full_like(
                    parameter,
                    0.5,
                ),
            )
            for parameter in target_network.parameters()
        ),
        "Soft target update",
    )

    # Test 15: Actor gradient.

    actor.zero_grad()

    sample_actions, sample_log_prob = actor.sample(
        states
    )

    actor_loss = (
        0.2 * sample_log_prob
        - critic1(
            states,
            sample_actions,
        )
    ).mean()

    actor_loss.backward()

    actor_gradients = [
        parameter.grad
        for parameter in actor.parameters()
        if parameter.grad is not None
    ]

    check(
        len(actor_gradients) > 0
        and all(
            bool(torch.isfinite(gradient).all())
            for gradient in actor_gradients
        ),
        "Actor gradient through reparameterized sample",
    )

    # Test 16: Replay buffer.

    replay = ReplayBuffer(100)

    for index in range(20):

        replay.add(
            np.zeros(OBS_DIM),
            np.zeros(ACTION_DIM),
            float(index),
            np.ones(OBS_DIM),
            False,
        )

    check(
        len(replay) == 20,
        "Replay buffer storage",
    )

    # Test 17: Replay batch shapes.

    batch = replay.sample(8)

    check(
        batch[0].shape == (8, OBS_DIM)
        and batch[1].shape == (8, ACTION_DIM)
        and batch[2].shape == (8, 1)
        and batch[3].shape == (8, OBS_DIM)
        and batch[4].shape == (8, 1),
        "Replay batch shapes",
    )

    # Test 18: Environment observation.

    env = DroneEnv(seed=514)

    observation = env.reset()

    check(
        observation.shape == (OBS_DIM,)
        and np.isfinite(observation).all(),
        "Environment observation",
    )

    # Test 19: Environment transition.

    (
        next_observation,
        reward,
        terminated,
        truncated,
        info,
    ) = env.step(
        np.zeros(
            ACTION_DIM,
            dtype=np.float32,
        )
    )

    check(
        next_observation.shape == (OBS_DIM,)
        and np.isfinite(reward)
        and "distance" in info,
        "Environment transition",
    )

    # Test 20: Moving target velocity.

    check(
        env.target.velocity().shape == (2,)
        and np.isfinite(env.target.velocity()).all(),
        "Moving target velocity",
    )

    # Test 21: Agent construction.

    agent = SACAgent()

    check(
        agent.alpha.item() > 0,
        "SAC agent initialization",
    )

    # Test 22: Deterministic action selection.

    action = agent.select_action(
        observation,
        deterministic=True,
    )

    check(
        action.shape == (ACTION_DIM,)
        and np.isfinite(action).all()
        and np.all(np.abs(action) <= 1.0),
        "Deterministic action selection",
    )

    print(
        f"\n{passed}/22 tests passed."
    )

    return passed == 22


# ============================================================
# 13. MAIN
# ============================================================

def main():

    print("=" * 72)

    print(
        "PHASE 5.14 — SOFT ACTOR-CRITIC"
    )

    print("=" * 72)

    print("Device:", DEVICE)

    print("Target entropy:", TARGET_ENTROPY)

    if not run_tests():

        raise RuntimeError(
            "SAC tests failed."
        )

    print("\nStarting SAC training...")

    set_seed(SEED)

    agent = SACAgent()

    history = train_agent(agent)

    print("\nFinal evaluation:")

    print(
        evaluate(
            agent,
            episodes=8,
        )
    )

    print("\nPlotting training results...")

    plot_training(history)

    print("\nStarting interactive drone simulation...")

    run_pygame(agent)

    print("\nPhase 5.14 completed.")


if __name__ == "__main__":

    main()
