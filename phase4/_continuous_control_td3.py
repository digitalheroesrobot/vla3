
# ============================================================
# File: phase4/continuous_control_td3.py
#
# PHASE 4.9 — CONTINUOUS CONTROL WITH DDPG AND TD3
#
# PURPOSE
#   Replace discrete movement decisions with continuous control.
#
# WHAT YOU LEARN
#   Step 1: Continuous action spaces
#   Step 2: Actor network
#   Step 3: Critic network
#   Step 4: Deterministic policy gradient
#   Step 5: DDPG algorithm
#   Step 6: TD3 improvements
#   Step 7: Replay buffer and target networks
#   Step 8: Training a continuous-control rocket
#   Step 9: Pygame visualization
#   Step 10: Numerical verification and tests
#
# KEY FORMULAS
#
#   Actor:
#       a = mu(s | theta)
#
#   Critic:
#       Q(s, a | phi)
#
#   DDPG target:
#       y = r + gamma * Q_target(s', mu_target(s'))
#
#   Deterministic policy gradient:
#       grad J(theta)
#       = E[ grad_a Q(s,a) * grad_theta mu(s) ]
#
#   TD3 target:
#       a' = mu_target(s') + clipped_noise
#
#       y = r + gamma * (1-done)
#               * min(Q1_target(s',a'), Q2_target(s',a'))
#
#   TD3 improvements:
#       1. Twin critics reduce overestimation.
#       2. Delayed actor updates improve stability.
#       3. Target policy smoothing reduces exploitation
#          of narrow critic errors.
#
# ROBOTICS APPLICATION
#   A mobile robot uses continuous steering and thrust
#   to approach a target while avoiding an obstacle.
#
# PYGAME CONTROLS
#   Left click : Change target
#   SPACE      : Pause / resume
#   R          : Reset simulation
#   +/-        : Change simulation speed
#   ESC        : Exit
#
# INSTALL
#   pip install numpy pygame torch matplotlib
#
# RUN
#   python phase4/continuous_control_td3.py
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

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

WORLD_W = 700
WORLD_H = 560

WINDOW_W = 1020
WINDOW_H = 650

OBSTACLE = np.array(
    [350.0, 280.0],
    dtype=np.float32,
)

OBSTACLE_RADIUS = 55.0
ROCKET_RADIUS = 12.0
SAFE_DISTANCE = OBSTACLE_RADIUS + ROCKET_RADIUS + 22.0

STATE_SIZE = 10
ACTION_SIZE = 2

# Continuous action:
#
# action[0] = steering command in [-1, 1]
# action[1] = thrust command in [-1, 1]
#
# Negative thrust means reverse thrust.
# Negative steering means turn left.
# Positive steering means turn right.

GAMMA = 0.99
TAU = 0.005

ACTOR_LR = 1e-4
CRITIC_LR = 3e-4

BUFFER_CAPACITY = 100_000
BATCH_SIZE = 128
WARMUP_STEPS = 1_000

TRAINING_EPISODES = 180
MAX_STEPS = 250

POLICY_DELAY = 2

TARGET_NOISE = 0.2
NOISE_CLIP = 0.5

EXPLORATION_NOISE = 0.20

MAX_SPEED = 5.0
MAX_ANGULAR_SPEED = 0.09

DT = 1.0

HIDDEN = 256

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# STEP 1 — MATHEMATICAL HELPERS
# ============================================================

def distance(a, b):
    return float(np.linalg.norm(a - b))


def clamp_action(action):
    return np.clip(
        action,
        -1.0,
        1.0,
    ).astype(np.float32)


def soft_update(source, target, tau=TAU):
    """
    Polyak averaging:

        target = tau * source + (1-tau) * target

    A small tau produces slowly changing target networks.
    """

    with torch.no_grad():
        for source_param, target_param in zip(
            source.parameters(),
            target.parameters(),
        ):
            target_param.data.mul_(1.0 - tau)
            target_param.data.add_(
                tau * source_param.data
            )


def hard_update(source, target):
    target.load_state_dict(source.state_dict())


# ============================================================
# STEP 2 — CONTINUOUS ROCKET ENVIRONMENT
# ============================================================

def safe_position(position):
    x, y = position

    inside_world = (
        ROCKET_RADIUS <= x <= WORLD_W - ROCKET_RADIUS
        and ROCKET_RADIUS <= y <= WORLD_H - ROCKET_RADIUS
    )

    outside_obstacle = (
        distance(position, OBSTACLE) >= SAFE_DISTANCE
    )

    return inside_world and outside_obstacle


class ContinuousRocketEnv:

    def __init__(self):
        self.reset()

    def random_safe_position(self):
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

        raise RuntimeError("Cannot find safe position.")

    def reset(self):
        self.position = self.random_safe_position()

        self.target = self.random_safe_position()

        while distance(self.position, self.target) < 230:
            self.target = self.random_safe_position()

        self.velocity = np.zeros(
            2,
            dtype=np.float32,
        )

        self.heading = random.uniform(
            -math.pi,
            math.pi,
        )

        self.steps = 0

        self.total_reward = 0.0

        self.success = False
        self.collision = False

        self.previous_distance = distance(
            self.position,
            self.target,
        )

        return self.get_state()

    def set_target(self, target):
        target = np.asarray(
            target,
            dtype=np.float32,
        )

        if not safe_position(target):
            return False

        if distance(target, self.position) < 25:
            return False

        self.target = target

        self.steps = 0
        self.total_reward = 0.0

        self.success = False
        self.collision = False

        self.previous_distance = distance(
            self.position,
            self.target,
        )

        return True

    def get_state(self):
        target_delta = self.target - self.position

        obstacle_delta = OBSTACLE - self.position

        target_distance = np.linalg.norm(target_delta)

        target_angle = math.atan2(
            target_delta[1],
            target_delta[0],
        )

        angle_error = (
            target_angle - self.heading + math.pi
        ) % (2 * math.pi) - math.pi

        return np.array(
            [
                target_delta[0] / WORLD_W,
                target_delta[1] / WORLD_H,

                target_distance / 900.0,

                math.sin(angle_error),
                math.cos(angle_error),

                self.velocity[0] / MAX_SPEED,
                self.velocity[1] / MAX_SPEED,

                obstacle_delta[0] / WORLD_W,
                obstacle_delta[1] / WORLD_H,

                (
                    np.linalg.norm(obstacle_delta)
                    - SAFE_DISTANCE
                ) / 900.0,
            ],
            dtype=np.float32,
        )

    def step(self, action):

        action = clamp_action(action)

        steering = float(action[0])
        thrust = float(action[1])

        old_position = self.position.copy()

        old_distance = distance(
            self.position,
            self.target,
        )

        # ----------------------------------------------------
        # Continuous vehicle dynamics
        # ----------------------------------------------------

        self.heading += (
            steering * MAX_ANGULAR_SPEED * DT
        )

        forward = np.array(
            [
                math.cos(self.heading),
                math.sin(self.heading),
            ],
            dtype=np.float32,
        )

        acceleration = forward * thrust * 0.22

        self.velocity += acceleration

        # Damping prevents unlimited acceleration.
        self.velocity *= 0.96

        speed = np.linalg.norm(self.velocity)

        if speed > MAX_SPEED:
            self.velocity *= MAX_SPEED / speed

        candidate_position = (
            self.position + self.velocity * DT
        )

        reward = -0.03

        terminated = False
        truncated = False

        # ----------------------------------------------------
        # Safety check
        # ----------------------------------------------------

        if not safe_position(candidate_position):

            reward -= 20.0

            self.collision = True
            terminated = True

        else:

            self.position = candidate_position

            new_distance = distance(
                self.position,
                self.target,
            )

            # Reward progress toward target.
            progress = old_distance - new_distance

            reward += 0.20 * progress

            # Small velocity penalty discourages unnecessary
            # oscillation and excessive movement.
            reward -= 0.015 * np.linalg.norm(self.velocity)

            # Reward approaching the target.
            if new_distance < 35:
                reward += 0.3

            if new_distance < 20:
                reward += 25.0

                self.success = True
                terminated = True

        self.steps += 1

        if self.steps >= MAX_STEPS and not terminated:
            truncated = True

        self.total_reward += reward

        self.previous_distance = distance(
            self.position,
            self.target,
        )

        return (
            self.get_state(),
            float(reward),
            terminated,
            truncated,
        )


# ============================================================
# STEP 3 — ACTOR NETWORK
# ============================================================

class Actor(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(STATE_SIZE, HIDDEN),
            nn.ReLU(),

            nn.Linear(HIDDEN, HIDDEN),
            nn.ReLU(),

            nn.Linear(HIDDEN, ACTION_SIZE),
            nn.Tanh(),
        )

    def forward(self, state):
        """
        Output range:

            steering in [-1, 1]
            thrust in [-1, 1]

        Tanh constrains the continuous action.
        """

        return self.network(state)


# ============================================================
# STEP 4 — CRITIC NETWORK
# ============================================================

class Critic(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                STATE_SIZE + ACTION_SIZE,
                HIDDEN,
            ),
            nn.ReLU(),

            nn.Linear(HIDDEN, HIDDEN),
            nn.ReLU(),

            nn.Linear(HIDDEN, 1),
        )

    def forward(self, state, action):
        x = torch.cat(
            [state, action],
            dim=1,
        )

        return self.network(x)


# ============================================================
# STEP 5 — REPLAY BUFFER
# ============================================================

@dataclass
class Transition:
    state: np.ndarray
    action: np.ndarray
    reward: float
    next_state: np.ndarray
    done: float


class ReplayBuffer:

    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def add(
        self,
        state,
        action,
        reward,
        next_state,
        terminated,
    ):
        self.memory.append(
            Transition(
                state=state.copy(),
                action=action.copy(),
                reward=float(reward),
                next_state=next_state.copy(),
                done=float(terminated),
            )
        )

    def sample(self, batch_size):

        batch = random.sample(
            self.memory,
            batch_size,
        )

        states = np.stack(
            [t.state for t in batch]
        )

        actions = np.stack(
            [t.action for t in batch]
        )

        rewards = np.array(
            [t.reward for t in batch],
            dtype=np.float32,
        ).reshape(-1, 1)

        next_states = np.stack(
            [t.next_state for t in batch]
        )

        dones = np.array(
            [t.done for t in batch],
            dtype=np.float32,
        ).reshape(-1, 1)

        return (
            torch.tensor(
                states,
                dtype=torch.float32,
                device=DEVICE,
            ),
            torch.tensor(
                actions,
                dtype=torch.float32,
                device=DEVICE,
            ),
            torch.tensor(
                rewards,
                dtype=torch.float32,
                device=DEVICE,
            ),
            torch.tensor(
                next_states,
                dtype=torch.float32,
                device=DEVICE,
            ),
            torch.tensor(
                dones,
                dtype=torch.float32,
                device=DEVICE,
            ),
        )

    def __len__(self):
        return len(self.memory)


# ============================================================
# STEP 6 — DDPG AGENT
# ============================================================

class DDPGAgent:

    def __init__(self):

        self.actor = Actor().to(DEVICE)
        self.actor_target = Actor().to(DEVICE)

        self.critic = Critic().to(DEVICE)
        self.critic_target = Critic().to(DEVICE)

        hard_update(
            self.actor,
            self.actor_target,
        )

        hard_update(
            self.critic,
            self.critic_target,
        )

        self.actor_optimizer = optim.Adam(
            self.actor.parameters(),
            lr=ACTOR_LR,
        )

        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=CRITIC_LR,
        )

        self.replay = ReplayBuffer(
            BUFFER_CAPACITY,
        )

        self.losses = []

        self.training_steps = 0

    def select_action(
        self,
        state,
        exploration_noise=0.0,
    ):

        state_t = torch.tensor(
            state,
            dtype=torch.float32,
            device=DEVICE,
        ).unsqueeze(0)

        self.actor.eval()

        with torch.no_grad():
            action = self.actor(state_t).cpu().numpy()[0]

        self.actor.train()

        if exploration_noise > 0:
            action += np.random.normal(
                0,
                exploration_noise,
                size=ACTION_SIZE,
            )

        return clamp_action(action)

    def train_step(self):

        if len(self.replay) < max(
            BATCH_SIZE,
            WARMUP_STEPS,
        ):
            return None

        (
            states,
            actions,
            rewards,
            next_states,
            dones,
        ) = self.replay.sample(BATCH_SIZE)

        # ----------------------------------------------------
        # Critic update
        #
        # y = r + gamma * (1-done)
        #                 * Q_target(s', actor_target(s'))
        # ----------------------------------------------------

        with torch.no_grad():

            next_actions = self.actor_target(
                next_states
            )

            target_q = self.critic_target(
                next_states,
                next_actions,
            )

            y = (
                rewards
                + GAMMA * (1.0 - dones) * target_q
            )

        current_q = self.critic(
            states,
            actions,
        )

        critic_loss = nn.functional.mse_loss(
            current_q,
            y,
        )

        self.critic_optimizer.zero_grad()

        critic_loss.backward()

        nn.utils.clip_grad_norm_(
            self.critic.parameters(),
            10.0,
        )

        self.critic_optimizer.step()

        # ----------------------------------------------------
        # Actor update
        #
        # Maximize:
        #       Q(s, actor(s))
        #
        # Minimize negative Q.
        # ----------------------------------------------------

        actor_loss = -self.critic(
            states,
            self.actor(states),
        ).mean()

        self.actor_optimizer.zero_grad()

        actor_loss.backward()

        nn.utils.clip_grad_norm_(
            self.actor.parameters(),
            10.0,
        )

        self.actor_optimizer.step()

        soft_update(
            self.actor,
            self.actor_target,
        )

        soft_update(
            self.critic,
            self.critic_target,
        )

        self.training_steps += 1

        self.losses.append(
            float(critic_loss.item())
        )

        return float(critic_loss.item())


# ============================================================
# STEP 7 — TD3 AGENT
# ============================================================

class TD3Agent(DDPGAgent):

    def __init__(self):
        super().__init__()

        # TD3 uses two independent critics.
        self.critic2 = Critic().to(DEVICE)

        self.critic2_target = Critic().to(DEVICE)

        hard_update(
            self.critic2,
            self.critic2_target,
        )

        self.critic2_optimizer = optim.Adam(
            self.critic2.parameters(),
            lr=CRITIC_LR,
        )

        self.actor_updates = 0

        self.critic2_losses = []

    def train_step(self):

        if len(self.replay) < max(
            BATCH_SIZE,
            WARMUP_STEPS,
        ):
            return None

        (
            states,
            actions,
            rewards,
            next_states,
            dones,
        ) = self.replay.sample(BATCH_SIZE)

        # ----------------------------------------------------
        # TD3 IMPROVEMENT 1:
        # TARGET POLICY SMOOTHING
        #
        # Add clipped noise to target actions.
        #
        # This prevents the critic from exploiting narrow,
        # unrealistic peaks in its learned Q-function.
        # ----------------------------------------------------

        with torch.no_grad():

            next_actions = self.actor_target(
                next_states
            )

            noise = torch.randn_like(
                next_actions
            ) * TARGET_NOISE

            noise = noise.clamp(
                -NOISE_CLIP,
                NOISE_CLIP,
            )

            next_actions = (
                next_actions + noise
            ).clamp(-1.0, 1.0)

            target_q1 = self.critic_target(
                next_states,
                next_actions,
            )

            target_q2 = self.critic2_target(
                next_states,
                next_actions,
            )

            # ------------------------------------------------
            # TD3 IMPROVEMENT 2:
            # TWIN CRITIC MINIMUM
            #
            # Use the smaller Q estimate.
            # This reduces positive overestimation bias.
            # ------------------------------------------------

            target_q = torch.minimum(
                target_q1,
                target_q2,
            )

            y = (
                rewards
                + GAMMA * (1.0 - dones) * target_q
            )

        current_q1 = self.critic(
            states,
            actions,
        )

        current_q2 = self.critic2(
            states,
            actions,
        )

        loss1 = nn.functional.mse_loss(
            current_q1,
            y,
        )

        loss2 = nn.functional.mse_loss(
            current_q2,
            y,
        )

        self.critic_optimizer.zero_grad()
        loss1.backward()

        nn.utils.clip_grad_norm_(
            self.critic.parameters(),
            10.0,
        )

        self.critic_optimizer.step()

        self.critic2_optimizer.zero_grad()
        loss2.backward()

        nn.utils.clip_grad_norm_(
            self.critic2.parameters(),
            10.0,
        )

        self.critic2_optimizer.step()

        self.training_steps += 1

        # ----------------------------------------------------
        # TD3 IMPROVEMENT 3:
        # DELAYED POLICY UPDATE
        #
        # Update actor and target networks only once every
        # POLICY_DELAY critic updates.
        # ----------------------------------------------------

        if self.training_steps % POLICY_DELAY == 0:

            actor_loss = -self.critic(
                states,
                self.actor(states),
            ).mean()

            self.actor_optimizer.zero_grad()

            actor_loss.backward()

            nn.utils.clip_grad_norm_(
                self.actor.parameters(),
                10.0,
            )

            self.actor_optimizer.step()

            soft_update(
                self.actor,
                self.actor_target,
            )

            soft_update(
                self.critic,
                self.critic_target,
            )

            soft_update(
                self.critic2,
                self.critic2_target,
            )

            self.actor_updates += 1

        combined_loss = (
            loss1.item() + loss2.item()
        ) / 2.0

        self.losses.append(
            float(combined_loss)
        )

        self.critic2_losses.append(
            float(loss2.item())
        )

        return float(combined_loss)


# ============================================================
# STEP 8 — TRAINING
# ============================================================

def train_agent(agent, episodes, name):

    env = ContinuousRocketEnv()

    rewards = []
    successes = []
    collisions = []

    started = time.time()

    for episode in range(episodes):

        state = env.reset()

        episode_reward = 0.0

        for step in range(MAX_STEPS):

            noise = max(
                0.04,
                EXPLORATION_NOISE
                * (1.0 - episode / episodes),
            )

            action = agent.select_action(
                state,
                exploration_noise=noise,
            )

            next_state, reward, terminated, truncated = (
                env.step(action)
            )

            # For a time-limit truncation, preserve bootstrapping:
            # it is not an MDP terminal state.
            agent.replay.add(
                state,
                action,
                reward,
                next_state,
                terminated,
            )

            agent.train_step()

            state = next_state

            episode_reward += reward

            if terminated or truncated:
                break

        rewards.append(episode_reward)

        successes.append(
            int(env.success)
        )

        collisions.append(
            int(env.collision)
        )

        if (episode + 1) % 20 == 0:

            recent = min(20, len(rewards))

            print(
                f"{name:8s} "
                f"Episode {episode + 1:3d}/{episodes} | "
                f"Reward {np.mean(rewards[-recent:]):8.2f} | "
                f"Success {100*np.mean(successes[-recent:]):5.1f}% | "
                f"Collision {100*np.mean(collisions[-recent:]):5.1f}%"
            )

    elapsed = time.time() - started

    return {
        "agent": agent,
        "rewards": rewards,
        "successes": successes,
        "collisions": collisions,
        "seconds": elapsed,
    }


# ============================================================
# STEP 9 — TRAINING VISUALIZATION
# ============================================================

def plot_training(ddpg_result, td3_result):

    plt.figure(figsize=(12, 5))

    for name, result in [
        ("DDPG", ddpg_result),
        ("TD3", td3_result),
    ]:

        rewards = result["rewards"]

        window = 15

        smooth = np.convolve(
            rewards,
            np.ones(window) / window,
            mode="valid",
        )

        plt.plot(
            smooth,
            label=name,
        )

    plt.title("DDPG vs TD3 — Episode Reward")
    plt.xlabel("Episode")
    plt.ylabel("Smoothed reward")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 5))

    for name, result in [
        ("DDPG", ddpg_result),
        ("TD3", td3_result),
    ]:

        plt.plot(
            result["agent"].losses,
            label=name,
        )

    plt.title("Critic Training Loss")
    plt.xlabel("Training update")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("\nFINAL PERFORMANCE")

    for name, result in [
        ("DDPG", ddpg_result),
        ("TD3", td3_result),
    ]:

        recent = min(25, len(result["rewards"]))

        print(
            f"{name}: "
            f"Reward={np.mean(result['rewards'][-recent:]):.2f}, "
            f"Success={100*np.mean(result['successes'][-recent:]):.1f}%, "
            f"Collision={100*np.mean(result['collisions'][-recent:]):.1f}%, "
            f"Time={result['seconds']:.1f}s"
        )


# ============================================================
# STEP 10 — PYGAME ROCKET SIMULATOR
# ============================================================

class RocketGUI:

    def __init__(self, env, agent, title):

        pygame.init()

        pygame.display.set_caption(
            f"Phase 4.9 — {title}"
        )

        self.screen = pygame.display.set_mode(
            (WINDOW_W, WINDOW_H)
        )

        self.clock = pygame.time.Clock()

        self.font = pygame.font.SysFont(
            "consolas",
            15,
        )

        self.title_font = pygame.font.SysFont(
            "consolas",
            20,
            bold=True,
        )

        self.env = env
        self.agent = agent
        self.title = title

        self.paused = False
        self.running = True
        self.speed = 1

        self.last_action = np.zeros(
            ACTION_SIZE,
            dtype=np.float32,
        )

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

        self.env.velocity = np.zeros(
            2,
            dtype=np.float32,
        )

        self.env.heading = 0.0

        self.env.steps = 0
        self.env.total_reward = 0.0

        self.env.success = False
        self.env.collision = False

        self.last_action[:] = 0.0

    def draw_text(
        self,
        text,
        x,
        y,
        color=(230, 230, 230),
        font=None,
    ):

        if font is None:
            font = self.font

        surface = font.render(
            str(text),
            True,
            color,
        )

        self.screen.blit(
            surface,
            (x, y),
        )

    def draw_rocket(self):

        x, y = self.env.position.astype(int)

        angle = self.env.heading

        def rotate(point):

            lx, ly = point

            return (
                int(
                    x
                    + lx * math.cos(angle)
                    - ly * math.sin(angle)
                ),
                int(
                    y
                    + lx * math.sin(angle)
                    + ly * math.cos(angle)
                ),
            )

        body = [
            (23, 0),
            (5, -9),
            (-11, -7),
            (-15, -4),
            (-15, 4),
            (-11, 7),
            (5, 9),
        ]

        thrust = max(
            0.0,
            float(self.last_action[1]),
        )

        flame_length = int(
            8 + 22 * thrust
        )

        if thrust > 0.05:

            pygame.draw.polygon(
                self.screen,
                (255, 125, 30),
                [
                    rotate((-15, -4)),
                    rotate((-15 - flame_length, 0)),
                    rotate((-15, 4)),
                ],
            )

        pygame.draw.polygon(
            self.screen,
            (230, 240, 250),
            [
                rotate(p)
                for p in body
            ],
        )

        pygame.draw.polygon(
            self.screen,
            (70, 160, 230),
            [
                rotate(p)
                for p in body
            ],
            2,
        )

        pygame.draw.circle(
            self.screen,
            (40, 140, 245),
            rotate((8, 0)),
            4,
        )

    def draw_world(self):

        self.screen.fill(
            (18, 22, 30)
        )

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
            (135, 75, 55),
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
            (
                WORLD_W,
                0,
                WINDOW_W - WORLD_W,
                WINDOW_H,
            ),
        )

        self.draw_text(
            "CONTINUOUS CONTROL",
            720,
            20,
            (120, 210, 255),
            self.title_font,
        )

        self.draw_text(
            self.title,
            720,
            55,
            (100, 255, 160),
        )

        self.draw_text(
            f"Position: {self.env.position.round(1)}",
            720,
            100,
        )

        self.draw_text(
            f"Velocity: {self.env.velocity.round(2)}",
            720,
            130,
        )

        self.draw_text(
            f"Heading: {math.degrees(self.env.heading):.1f} deg",
            720,
            160,
        )

        self.draw_text(
            f"Distance: {distance(self.env.position, self.env.target):.1f}",
            720,
            190,
        )

        self.draw_text(
            f"Steering: {self.last_action[0]:+.3f}",
            720,
            235,
        )

        self.draw_text(
            f"Thrust:   {self.last_action[1]:+.3f}",
            720,
            265,
        )

        self.draw_text(
            f"Steps: {self.env.steps}",
            720,
            310,
        )

        self.draw_text(
            f"Reward: {self.env.total_reward:.2f}",
            720,
            340,
        )

        self.draw_text(
            f"Speed multiplier: {self.speed}",
            720,
            370,
        )

        self.draw_text(
            "ACTION SPACE",
            720,
            420,
            (120, 210, 255),
        )

        self.draw_text(
            "Steering: [-1, +1]",
            720,
            450,
        )

        self.draw_text(
            "Thrust:   [-1, +1]",
            720,
            475,
        )

        self.draw_text(
            "CLICK: TARGET",
            720,
            550,
        )

        self.draw_text(
            "SPACE: PAUSE | R: RESET",
            720,
            575,
        )

        self.draw_text(
            "ESC: EXIT | +/-: SPEED",
            720,
            600,
        )

    def step(self):

        state = self.env.get_state()

        action = self.agent.select_action(
            state,
            exploration_noise=0.0,
        )

        self.last_action = action.copy()

        _, _, terminated, truncated = self.env.step(
            action
        )

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
                        self.speed = min(
                            8,
                            self.speed + 1,
                        )

                    elif event.key in (
                        pygame.K_MINUS,
                        pygame.K_KP_MINUS,
                    ):
                        self.speed = max(
                            1,
                            self.speed - 1,
                        )

                elif event.type == pygame.MOUSEBUTTONDOWN:

                    if event.button == 1:

                        x, y = event.pos

                        if (
                            0 <= x <= WORLD_W
                            and 0 <= y <= WORLD_H
                        ):

                            if self.env.set_target(
                                [x, y]
                            ):
                                self.last_action[:] = 0.0
                                self.paused = False

            if not self.paused:

                for _ in range(self.speed):

                    self.step()

                    if self.paused:
                        break

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

            pygame.display.flip()

            self.clock.tick(60)

        pygame.quit()


# ============================================================
# STEP 11 — UNIT TESTS
# ============================================================

def run_tests():

    print("\nPHASE 4.9 — UNIT TESTS")

    passed = 0

    def check(name, condition):

        nonlocal passed

        assert bool(condition), (
            f"FAILED: {name}"
        )

        passed += 1

        print(
            f"[PASS] {name}"
        )

    # Test 1: state dimensions.
    env = ContinuousRocketEnv()

    check(
        "State dimension",
        env.get_state().shape == (STATE_SIZE,),
    )

    # Test 2: action dimensions.
    check(
        "Action dimension",
        clamp_action([0.5, -0.5]).shape
        == (ACTION_SIZE,),
    )

    # Test 3: action clipping.
    action = clamp_action([2.0, -3.0])

    check(
        "Action clipping",
        np.allclose(action, [1.0, -1.0]),
    )

    # Test 4: actor output range.
    actor = Actor().to(DEVICE)

    states = torch.randn(
        8,
        STATE_SIZE,
        device=DEVICE,
    )

    actions = actor(states)

    check(
        "Actor output range",
        bool(
            torch.all(actions >= -1.0)
            and torch.all(actions <= 1.0)
        ),
    )

    # Test 5: actor output dimensions.
    check(
        "Actor output dimensions",
        tuple(actions.shape) == (8, ACTION_SIZE),
    )

    # Test 6: critic output dimensions.
    critic = Critic().to(DEVICE)

    q = critic(
        states,
        actions,
    )

    check(
        "Critic output dimensions",
        tuple(q.shape) == (8, 1),
    )

    # Test 7: critic differentiates with respect to action.
    test_actions = torch.randn(
        8,
        ACTION_SIZE,
        device=DEVICE,
        requires_grad=True,
    )

    test_q = critic(
        states,
        test_actions,
    )

    action_grad = torch.autograd.grad(
        test_q.sum(),
        test_actions,
    )[0]

    check(
        "Critic action gradient exists",
        action_grad.shape == test_actions.shape
        and torch.isfinite(action_grad).all(),
    )

    # Test 8: soft update.
    source = nn.Linear(2, 2)
    target = nn.Linear(2, 2)

    with torch.no_grad():

        for p in source.parameters():
            p.fill_(1.0)

        for p in target.parameters():
            p.fill_(0.0)

    soft_update(
        source,
        target,
        tau=0.1,
    )

    check(
        "Soft update",
        torch.allclose(
            target.weight,
            torch.full_like(target.weight, 0.1),
        ),
    )

    # Test 9: replay buffer.
    replay = ReplayBuffer(10)

    state = np.zeros(
        STATE_SIZE,
        dtype=np.float32,
    )

    next_state = np.ones(
        STATE_SIZE,
        dtype=np.float32,
    )

    replay.add(
        state,
        np.array([0.1, 0.5], dtype=np.float32),
        1.0,
        next_state,
        False,
    )

    check(
        "Replay insertion",
        len(replay) == 1,
    )

    # Test 10: replay sample shapes.
    for _ in range(BATCH_SIZE):

        replay.add(
            state,
            np.array([0.1, 0.5], dtype=np.float32),
            1.0,
            next_state,
            False,
        )

    sample = replay.sample(BATCH_SIZE)

    check(
        "Replay batch state dimensions",
        tuple(sample[0].shape)
        == (BATCH_SIZE, STATE_SIZE),
    )

    check(
        "Replay batch action dimensions",
        tuple(sample[1].shape)
        == (BATCH_SIZE, ACTION_SIZE),
    )

    # Test 11: twin critics.
    agent = TD3Agent()

    with torch.no_grad():

        q1 = agent.critic(
            states,
            actions,
        )

        q2 = agent.critic2(
            states,
            actions,
        )

    check(
        "Twin critics have valid outputs",
        q1.shape == q2.shape == (8, 1),
    )

    # Test 12: TD3 target uses smaller Q.
    test_q1 = torch.tensor(
        [[5.0], [2.0]],
    )

    test_q2 = torch.tensor(
        [[3.0], [4.0]],
    )

    minimum = torch.minimum(
        test_q1,
        test_q2,
    )

    check(
        "Twin critic minimum",
        torch.allclose(
            minimum,
            torch.tensor([[3.0], [2.0]]),
        ),
    )

    # Test 13: target noise clipping.
    noise = torch.tensor(
        [[2.0, -2.0]],
    ).clamp(
        -NOISE_CLIP,
        NOISE_CLIP,
    )

    check(
        "Target noise clipping",
        torch.allclose(
            noise,
            torch.tensor([[0.5, -0.5]]),
        ),
    )

    # Test 14: environment returns valid tuple.
    result = env.step(
        np.array([0.0, 0.0], dtype=np.float32)
    )

    check(
        "Environment transition tuple",
        len(result) == 4,
    )

    # Test 15: environment action accepts continuous values.
    result = env.step(
        np.array([0.25, 0.5], dtype=np.float32)
    )

    check(
        "Continuous action accepted",
        len(result[0]) == STATE_SIZE,
    )

    print(
        f"\n{passed}/15 tests passed."
    )


# ============================================================
# STEP 12 — MAIN
# ============================================================

def main():

    print("=" * 75)
    print("PHASE 4.9 — CONTINUOUS CONTROL")
    print("=" * 75)

    print("Device:", DEVICE)

    run_tests()

    # --------------------------------------------------------
    # Train DDPG
    # --------------------------------------------------------

    print("\nTraining DDPG...")

    ddpg_agent = DDPGAgent()

    ddpg_result = train_agent(
        ddpg_agent,
        TRAINING_EPISODES,
        "DDPG",
    )

    # --------------------------------------------------------
    # Train TD3
    # --------------------------------------------------------

    print("\nTraining TD3...")

    td3_agent = TD3Agent()

    td3_result = train_agent(
        td3_agent,
        TRAINING_EPISODES,
        "TD3",
    )

    # --------------------------------------------------------
    # Compare results
    # --------------------------------------------------------

    plot_training(
        ddpg_result,
        td3_result,
    )

    # --------------------------------------------------------
    # Save models
    # --------------------------------------------------------

    torch.save(
        ddpg_agent.actor.state_dict(),
        "ddpg_actor.pth",
    )

    torch.save(
        td3_agent.actor.state_dict(),
        "td3_actor.pth",
    )

    print("\nSaved:")
    print("  ddpg_actor.pth")
    print("  td3_actor.pth")

    # --------------------------------------------------------
    # Visualize trained TD3 rocket.
    # --------------------------------------------------------

    print("\nStarting Pygame TD3 simulation...")

    gui_env = ContinuousRocketEnv()

    gui = RocketGUI(
        gui_env,
        td3_agent,
        "TD3",
    )

    gui.run()


if __name__ == "__main__":
    main()
