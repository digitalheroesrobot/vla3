
"""
File: phase5/deterministic_policy_gradient.py

PHASE 5.11 — DETERMINISTIC POLICY GRADIENT (DPG)
================================================

PURPOSE
-------
Learn how deterministic continuous-action policies are optimized
using a differentiable critic.

WHAT YOU LEARN
--------------

STEP 1 — Deterministic Actor
    The actor maps a state directly to an action:

        a = mu(s | theta)

    Unlike a Gaussian policy, it does not sample an action
    from a probability distribution.

STEP 2 — Critic
    The critic estimates:

        Q(s, a | phi)

STEP 3 — Deterministic Policy Gradient

    The objective is:

        J(theta) = E[Q(s, mu(s | theta))]

    Its gradient is:

        grad_theta J =
            E[
                grad_theta mu(s | theta)
                * grad_a Q(s,a)|a=mu(s)
            ]

    In implementation, PyTorch's automatic differentiation
    computes the chain rule through actor and critic.

STEP 4 — Replay Buffer
    Store transitions:

        (s, a, r, s_next, terminated)

    Random mini-batches reduce temporal correlation.

STEP 5 — Target Networks
    The critic target is:

        y = r + gamma * (1-terminated)
                 * Q_target(s_next, mu_target(s_next))

    Target networks are slowly updated:

        theta_target =
            tau*theta + (1-tau)*theta_target

STEP 6 — Exploration
    The deterministic actor is supplemented by noise:

        a_explore = clip(mu(s) + noise, -1, 1)

    Exploration noise is used when collecting experience.
    It is not part of the deterministic actor itself.

STEP 7 — ROBOTICS APPLICATION
    A quadcopter uses continuous horizontal and vertical
    acceleration commands to track a moving target.

    Target velocity is part of the observation.

IMPORTANT DISTINCTIONS
---------------------

This is a DPG/DDPG-style educational implementation.

It includes:
    - deterministic actor
    - critic
    - replay buffer
    - target actor
    - target critic
    - soft target updates
    - exploration noise

It does not yet include:
    - TD3 twin critics
    - TD3 target policy smoothing
    - TD3 delayed actor updates
    - SAC entropy regularization

The environment is a simplified 2D simulation, not a
flight-certified quadcopter controller.
"""

from __future__ import annotations

import copy
import math
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# ============================================================
# 0. CONFIGURATION
# ============================================================

SEED = 511

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

ACTOR_LR = 1e-4
CRITIC_LR = 1e-3

REPLAY_CAPACITY = 100_000
BATCH_SIZE = 128

INITIAL_RANDOM_STEPS = 2_000
TRAINING_STEPS = 35_000

EXPLORATION_STD = 0.20

MAX_EPISODE_STEPS = 400


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

        self.phase += (
            TARGET_OMEGA * DT
        )


# ============================================================
# 3. DRONE DYNAMICS
# ============================================================

def clip_drone_state(state):

    state = np.asarray(
        state,
        dtype=np.float32,
    ).copy()

    speed = np.linalg.norm(
        state[2:4]
    )

    if speed > MAX_SPEED:

        state[2:4] *= (
            MAX_SPEED / (speed + 1e-9)
        )

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
        action,
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
            [
                next_position,
                next_velocity,
            ]
        )
    )


# ============================================================
# 4. DRONE ENVIRONMENT
# ============================================================

class DroneEnv:

    def __init__(self, seed=None):

        self.rng = np.random.default_rng(
            seed
        )

        self.target = MovingTarget(
            [700.0, 310.0]
        )

        self.state = None
        self.steps = 0

        self.reset()

    def observation(self):

        position = self.state[:2]
        velocity = self.state[2:4]

        target_position = (
            self.target.position()
        )

        target_velocity = (
            self.target.velocity()
        )

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
            np.asarray(
                action,
                dtype=np.float32,
            ),
            -1.0,
            1.0,
        )

        old_position = (
            self.state[:2].copy()
        )

        # Compare both distances against the same future target
        # position. This avoids confusing target movement with
        # progress made by the drone.

        future_phase = (
            self.target.phase
            + TARGET_OMEGA * DT
        )

        future_target = (
            self.target.position(
                future_phase
            )
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
            self.state[:2]
            - self.target.position()
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
            0.08 * (
                distance_before - distance_after
            )
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

        terminated = bool(
            collision
        )

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
# 5. DETERMINISTIC ACTOR
# ============================================================

class Actor(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(OBS_DIM, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, ACTION_DIM),
            nn.Tanh(),
        )

    def forward(self, states):

        return self.network(
            states
        )


# ============================================================
# 6. CRITIC NETWORK
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
            [
                states,
                actions,
            ],
            dim=-1,
        )

        return self.network(
            inputs
        )


# ============================================================
# 7. REPLAY BUFFER
# ============================================================

class ReplayBuffer:

    def __init__(
        self,
        capacity=REPLAY_CAPACITY,
    ):

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

        states = np.stack(
            [item[0] for item in batch]
        )

        actions = np.stack(
            [item[1] for item in batch]
        )

        rewards = np.asarray(
            [item[2] for item in batch],
            dtype=np.float32,
        ).reshape(-1, 1)

        next_states = np.stack(
            [item[3] for item in batch]
        )

        terminated = np.asarray(
            [item[4] for item in batch],
            dtype=np.float32,
        ).reshape(-1, 1)

        return (
            torch.as_tensor(
                states,
                dtype=torch.float32,
                device=DEVICE,
            ),

            torch.as_tensor(
                actions,
                dtype=torch.float32,
                device=DEVICE,
            ),

            torch.as_tensor(
                rewards,
                dtype=torch.float32,
                device=DEVICE,
            ),

            torch.as_tensor(
                next_states,
                dtype=torch.float32,
                device=DEVICE,
            ),

            torch.as_tensor(
                terminated,
                dtype=torch.float32,
                device=DEVICE,
            ),
        )

    def __len__(self):

        return len(self.buffer)


# ============================================================
# 8. DPG / DDPG AGENT
# ============================================================

class DDPGAgent:

    def __init__(self):

        self.actor = Actor().to(
            DEVICE
        )

        self.critic = Critic().to(
            DEVICE
        )

        self.target_actor = copy.deepcopy(
            self.actor
        ).to(DEVICE)

        self.target_critic = copy.deepcopy(
            self.critic
        ).to(DEVICE)

        self.actor_optimizer = optim.Adam(
            self.actor.parameters(),
            lr=ACTOR_LR,
        )

        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=CRITIC_LR,
        )

        self.replay = ReplayBuffer()

        self.training_updates = 0

    def select_action(
        self,
        observation,
        noise_std=0.0,
    ):

        state = torch.as_tensor(
            observation,
            dtype=torch.float32,
            device=DEVICE,
        ).unsqueeze(0)

        with torch.no_grad():

            action = self.actor(
                state
            )

        action = (
            action.squeeze(0)
            .cpu()
            .numpy()
        )

        if noise_std > 0:

            action += np.random.normal(
                0.0,
                noise_std,
                size=ACTION_DIM,
            )

        return np.clip(
            action,
            -1.0,
            1.0,
        ).astype(np.float32)

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

    def soft_update(
        self,
        source,
        target,
    ):

        with torch.no_grad():

            for source_parameter, target_parameter in zip(
                source.parameters(),
                target.parameters(),
            ):

                target_parameter.data.mul_(
                    1.0 - TAU
                )

                target_parameter.data.add_(
                    TAU * source_parameter.data
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
        # STEP 1: CRITIC UPDATE
        # ----------------------------------------------------

        with torch.no_grad():

            next_actions = (
                self.target_actor(
                    next_states
                )
            )

            next_q = (
                self.target_critic(
                    next_states,
                    next_actions,
                )
            )

            # True terminal states do not bootstrap.
            # Time-limit truncations do bootstrap because their
            # final state is still a valid continuation state.

            targets = (
                rewards
                + GAMMA
                * (1.0 - terminated)
                * next_q
            )

        current_q = self.critic(
            states,
            actions,
        )

        critic_loss = nn.functional.mse_loss(
            current_q,
            targets,
        )

        self.critic_optimizer.zero_grad()

        critic_loss.backward()

        nn.utils.clip_grad_norm_(
            self.critic.parameters(),
            1.0,
        )

        self.critic_optimizer.step()

        # ----------------------------------------------------
        # STEP 2: ACTOR UPDATE
        # ----------------------------------------------------

        # The actor maximizes Q(s, actor(s)).
        #
        # Freeze critic parameters during actor optimization.
        # Gradients must still flow through the critic's action
        # input to the actor.

        for parameter in self.critic.parameters():

            parameter.requires_grad_(False)

        predicted_actions = self.actor(
            states
        )

        actor_loss = -self.critic(
            states,
            predicted_actions,
        ).mean()

        self.actor_optimizer.zero_grad()

        actor_loss.backward()

        nn.utils.clip_grad_norm_(
            self.actor.parameters(),
            1.0,
        )

        self.actor_optimizer.step()

        for parameter in self.critic.parameters():

            parameter.requires_grad_(True)

        # ----------------------------------------------------
        # STEP 3: SOFT TARGET UPDATE
        # ----------------------------------------------------

        self.soft_update(
            self.actor,
            self.target_actor,
        )

        self.soft_update(
            self.critic,
            self.target_critic,
        )

        self.training_updates += 1

        return {
            "actor_loss": float(
                actor_loss.detach().item()
            ),

            "critic_loss": float(
                critic_loss.detach().item()
            ),

            "q_mean": float(
                current_q.detach().mean().item()
            ),
        }


# ============================================================
# 9. TRAINING
# ============================================================

def train():

    env = DroneEnv(
        seed=SEED + 1
    )

    agent = DDPGAgent()

    history = {
        "episode_return": [],
        "episode_distance": [],
        "actor_loss": [],
        "critic_loss": [],
        "q_mean": [],
    }

    observation = env.reset()

    episode_return = 0.0
    episode_distances = []

    for step in range(
        TRAINING_STEPS
    ):

        if step < INITIAL_RANDOM_STEPS:

            action = np.random.uniform(
                -1.0,
                1.0,
                size=ACTION_DIM,
            ).astype(np.float32)

        else:

            action = agent.select_action(
                observation,
                noise_std=EXPLORATION_STD,
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

            history["q_mean"].append(
                metrics["q_mean"]
            )

        if terminated or truncated:

            history["episode_return"].append(
                episode_return
            )

            history["episode_distance"].append(
                float(
                    np.mean(
                        episode_distances
                    )
                )
            )

            if len(history["episode_return"]) % 5 == 0:

                print(
                    f"Step {step + 1:6d}"
                    f" | Episodes={len(history['episode_return']):4d}"
                    f" | Return={episode_return:8.2f}"
                    f" | Mean distance={np.mean(episode_distances):7.2f}"
                    f" | Replay={len(agent.replay):6d}"
                )

            observation = env.reset()

            episode_return = 0.0

            episode_distances = []

    return agent, history


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
        np.ones(
            window,
            dtype=np.float32,
        ) / window,
        mode="valid",
    )


def plot_history(history):

    plots = [
        (
            "episode_return",
            "Episode Return",
        ),

        (
            "episode_distance",
            "Mean Target Distance",
        ),

        (
            "actor_loss",
            "Actor Loss",
        ),

        (
            "critic_loss",
            "Critic Loss",
        ),

        (
            "q_mean",
            "Mean Critic Q Value",
        ),
    ]

    for key, title in plots:

        values = history[key]

        if len(values) == 0:

            continue

        plt.figure(
            figsize=(10, 4)
        )

        plt.plot(
            values,
            alpha=0.3,
            label="Raw",
        )

        plt.plot(
            moving_average(values),
            label="Moving average",
        )

        plt.title(title)

        plt.xlabel("Episode or update")

        plt.grid(True)

        plt.legend()

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
        "Phase 5.11 — Deterministic Policy Gradient Drone"
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

    def draw_drone():

        center = env.state[:2]

        velocity = env.state[2:4]

        magnitude = np.linalg.norm(
            velocity
        )

        if magnitude > 1e-5:

            direction = (
                velocity / magnitude
            )

        else:

            direction = np.array(
                [1.0, 0.0]
            )

        side = np.array(
            [
                -direction[1],
                direction[0],
            ]
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

        pygame.draw.circle(
            screen,
            (255, 185, 55),
            (
                center + direction * 22
            ).astype(int),
            4,
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

    def draw_target():

        points = []

        for index in range(100):

            phase = (
                env.target.phase
                + 2 * math.pi * index / 100
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
                ):

                    speed = min(
                        5,
                        speed + 1,
                    )

                elif event.key == pygame.K_MINUS:

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
                    noise_std=0.0,
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

        for x in range(
            0,
            WORLD_W,
            50,
        ):

            pygame.draw.line(
                screen,
                (30, 38, 50),
                (x, 0),
                (x, WORLD_H),
            )

        for y in range(
            0,
            WORLD_H,
            50,
        ):

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
            "PHASE 5.11 — DETERMINISTIC POLICY GRADIENT",
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
            f"Replay transitions: {len(agent.replay)}",
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

    print(
        "\nRunning DPG tests..."
    )

    passed = 0

    def check(condition, description):

        nonlocal passed

        assert condition, description

        passed += 1

        print(
            "[PASS]",
            description,
        )

    # Test 1: Actor output shape.

    actor = Actor().to(
        DEVICE
    )

    states = torch.randn(
        16,
        OBS_DIM,
        device=DEVICE,
    )

    actions = actor(
        states
    )

    check(
        actions.shape == (16, ACTION_DIM),
        "Actor output dimensions",
    )

    # Test 2: Actor output bounds.

    check(
        bool(
            torch.all(
                actions.abs() <= 1.0
            )
        ),
        "Actor actions stay within [-1, 1]",
    )

    # Test 3: Critic output shape.

    critic = Critic().to(
        DEVICE
    )

    q_values = critic(
        states,
        actions,
    )

    check(
        q_values.shape == (16, 1),
        "Critic output dimensions",
    )

    # Test 4: Critic finite outputs.

    check(
        bool(
            torch.isfinite(
                q_values
            ).all()
        ),
        "Critic outputs are finite",
    )

    # Test 5: Actor gradient through critic.

    actor.zero_grad()

    loss = -critic(
        states,
        actor(states),
    ).mean()

    loss.backward()

    actor_gradients = [
        p.grad
        for p in actor.parameters()
        if p.grad is not None
    ]

    check(
        len(actor_gradients) > 0
        and all(
            torch.isfinite(g).all()
            for g in actor_gradients
        ),
        "Actor receives finite gradients through critic",
    )

    # Test 6: Critic gradients.

    critic.zero_grad()

    predicted = critic(
        states,
        actions.detach(),
    )

    critic_loss = predicted.pow(2).mean()

    critic_loss.backward()

    critic_gradients = [
        p.grad
        for p in critic.parameters()
        if p.grad is not None
    ]

    check(
        len(critic_gradients) > 0
        and all(
            torch.isfinite(g).all()
            for g in critic_gradients
        ),
        "Critic gradients are finite",
    )

    # Test 7: Bellman target for nonterminal state.

    reward = torch.tensor(
        [[2.0]]
    )

    next_q = torch.tensor(
        [[5.0]]
    )

    terminal = torch.tensor(
        [[0.0]]
    )

    target = (
        reward
        + GAMMA
        * (1.0 - terminal)
        * next_q
    )

    expected = 2.0 + GAMMA * 5.0

    check(
        torch.allclose(
            target,
            torch.tensor(
                [[expected]]
            ),
        ),
        "Nonterminal Bellman target",
    )

    # Test 8: Bellman target for terminal state.

    terminal_target = (
        reward
        + GAMMA
        * (1.0 - torch.ones_like(terminal))
        * next_q
    )

    check(
        torch.allclose(
            terminal_target,
            reward,
        ),
        "Terminal state suppresses bootstrap",
    )

    # Test 9: Soft update.

    source = nn.Linear(
        3,
        2,
    )

    target_network = nn.Linear(
        3,
        2,
    )

    with torch.no_grad():

        for parameter in source.parameters():

            parameter.fill_(2.0)

        for parameter in target_network.parameters():

            parameter.fill_(0.0)

    tau = 0.25

    with torch.no_grad():

        for source_parameter, target_parameter in zip(
            source.parameters(),
            target_network.parameters(),
        ):

            target_parameter.mul_(
                1.0 - tau
            )

            target_parameter.add_(
                tau * source_parameter
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
        "Soft target update arithmetic",
    )

    # Test 10: Replay buffer.

    replay = ReplayBuffer(
        capacity=100
    )

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
        "Replay buffer stores transitions",
    )

    # Test 11: Replay batch dimensions.

    batch = replay.sample(
        8
    )

    check(
        batch[0].shape == (8, OBS_DIM)
        and batch[1].shape == (8, ACTION_DIM)
        and batch[2].shape == (8, 1)
        and batch[3].shape == (8, OBS_DIM)
        and batch[4].shape == (8, 1),
        "Replay mini-batch dimensions",
    )

    # Test 12: Exploration action bounds.

    env = DroneEnv()

    observation = env.reset()

    agent = DDPGAgent()

    exploratory_action = agent.select_action(
        observation,
        noise_std=0.5,
    )

    check(
        exploratory_action.shape == (ACTION_DIM,)
        and np.all(
            exploratory_action >= -1.0
        )
        and np.all(
            exploratory_action <= 1.0
        ),
        "Exploration action clipping",
    )

    # Test 13: Observation dimensions.

    check(
        observation.shape == (OBS_DIM,),
        "Drone observation dimensions",
    )

    # Test 14: Environment transition.

    next_observation, reward, terminated, truncated, info = (
        env.step(
            np.zeros(
                ACTION_DIM,
                dtype=np.float32,
            )
        )
    )

    check(
        next_observation.shape == (OBS_DIM,)
        and np.isfinite(reward),
        "Drone environment transition",
    )

    # Test 15: Target velocity dimensions.

    check(
        env.target.velocity().shape == (2,),
        "Moving target velocity dimensions",
    )

    # Test 16: Target prediction consistency.

    phase = 0.7

    future_position = env.target.position(
        phase + TARGET_OMEGA * DT
    )

    check(
        future_position.shape == (2,)
        and np.isfinite(future_position).all(),
        "Moving target future position",
    )

    print(
        f"\n{passed}/16 tests passed."
    )

    return passed == 16


# ============================================================
# 13. MAIN
# ============================================================

def main():

    print("=" * 72)

    print(
        "PHASE 5.11 — DETERMINISTIC POLICY GRADIENT"
    )

    print("=" * 72)

    print(
        "Device:",
        DEVICE,
    )

    if not run_tests():

        raise RuntimeError(
            "DPG tests failed."
        )

    print(
        "\nTraining deterministic actor-critic..."
    )

    agent, history = train()

    print(
        "\nPlotting training results..."
    )

    plot_history(
        history
    )

    print(
        "\nStarting interactive quadcopter..."
    )

    run_pygame(
        agent
    )

    print(
        "\nPhase 5.11 completed."
    )


if __name__ == "__main__":

    main()
