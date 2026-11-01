
# ============================================================
# File: phase4/moving_target_comparison.py
#
# PHASE 4.11 — MOVING-TARGET TRACKING
# Compare DQN, TD3 and SAC.
#
# PURPOSE
#   Train a rocket robot to follow a continuously moving target.
#
# WHAT YOU LEARN
#   Step 1: Dynamic target trajectories
#   Step 2: Relative position and target velocity in state
#   Step 3: DQN with discrete control
#   Step 4: TD3 with deterministic continuous control
#   Step 5: SAC with stochastic continuous control
#   Step 6: Compare algorithms in one environment
#   Step 7: Visualize target tracking with Pygame
#   Step 8: Verify algorithms and environment behavior
#
# STATE REPRESENTATION (12 dimensions)
#   0-1:  Target relative position (x, y)
#   2:    Distance to target
#   3-4:  Direction error (sin, cos)
#   5-6:  Robot velocity (x, y)
#   7-8:  Obstacle relative position (x, y)
#   9:    Obstacle clearance
#   10-11: Target velocity (x, y)
#
# WHY TARGET VELOCITY?
#   A moving target makes position-only observations less
#   informative. Including target velocity helps the robot
#   estimate where the target is heading.
#
# DQN ACTIONS
#   9 discrete combinations of steering and thrust.
#
# TD3 / SAC ACTIONS
#   Continuous steering and thrust, each in [-1, 1].
#
# MOVING TARGET
#   The target follows a smooth elliptical trajectory.
#   Clicking a safe location changes the trajectory center.
#
# PYGAME CONTROLS
#   Left click : Change moving-target trajectory center
#   SPACE      : Pause / resume
#   R          : Reset rocket and target
#   +/-        : Change simulation speed
#   1          : Display DQN
#   2          : Display TD3
#   3          : Display SAC
#   ESC        : Exit
#
# INSTALL
#   pip install numpy pygame torch matplotlib
#
# RUN
#   python phase4/moving_target_comparison.py
#
# NOTE
#   This is an educational 2D simulator, not a validated
#   physical robot model or safety controller.
# ============================================================

from __future__ import annotations

import math
import random
import time

from collections import deque
from dataclasses import dataclass

import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt


# ============================================================
# STEP 0 — CONFIGURATION
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

WORLD_W = 700
WORLD_H = 560

WINDOW_W = 1020
WINDOW_H = 650

STATE_SIZE = 12
ACTION_SIZE = 2

HIDDEN = 192

GAMMA = 0.99
TAU = 0.005

ACTOR_LR = 3e-4
CRITIC_LR = 3e-4
ALPHA_LR = 3e-4

REPLAY_CAPACITY = 80_000
BATCH_SIZE = 96
WARMUP_STEPS = 500

TRAINING_EPISODES = 100
MAX_STEPS = 220

MAX_SPEED = 5.0
MAX_ANGULAR_SPEED = 0.09

OBSTACLE = np.array(
    [350.0, 280.0],
    dtype=np.float32,
)

OBSTACLE_RADIUS = 55.0
ROCKET_RADIUS = 12.0

SAFE_DISTANCE = (
    OBSTACLE_RADIUS
    + ROCKET_RADIUS
    + 22.0
)

# Target trajectory:
#
# target_x = center_x + AX * sin(phase)
# target_y = center_y + AY * sin(2*phase)
#
# The target moves smoothly in both dimensions.
#
# The center is kept in a safe region by set_target_center().

TARGET_AX = 48.0
TARGET_AY = 55.0

TARGET_OMEGA = 0.035

INITIAL_TARGET_CENTER = np.array(
    [550.0, 150.0],
    dtype=np.float32,
)

# DQN actions:
#   steering: -1, 0, +1
#   thrust:   -1, 0, +1
#
# 3 x 3 = 9 discrete actions.

DQN_ACTIONS = np.array(
    [
        [-1.0, -1.0],
        [-1.0,  0.0],
        [-1.0,  1.0],

        [ 0.0, -1.0],
        [ 0.0,  0.0],
        [ 0.0,  1.0],

        [ 1.0, -1.0],
        [ 1.0,  0.0],
        [ 1.0,  1.0],
    ],
    dtype=np.float32,
)


# ============================================================
# STEP 1 — GENERAL HELPERS
# ============================================================

def distance(a, b):
    return float(
        np.linalg.norm(a - b)
    )


def clamp_action(action):
    return np.clip(
        action,
        -1.0,
        1.0,
    ).astype(np.float32)


def soft_update(source, target, tau=TAU):

    with torch.no_grad():

        for source_p, target_p in zip(
            source.parameters(),
            target.parameters(),
        ):

            target_p.data.mul_(1.0 - tau)

            target_p.data.add_(
                tau * source_p.data
            )


def hard_update(source, target):
    target.load_state_dict(
        source.state_dict()
    )


def tensor(array):
    return torch.as_tensor(
        array,
        dtype=torch.float32,
        device=DEVICE,
    )


# ============================================================
# STEP 2 — MOVING-TARGET ENVIRONMENT
# ============================================================

def safe_position(position):

    x, y = position

    inside = (
        ROCKET_RADIUS <= x <= WORLD_W - ROCKET_RADIUS
        and ROCKET_RADIUS <= y <= WORLD_H - ROCKET_RADIUS
    )

    outside_obstacle = (
        distance(position, OBSTACLE) >= SAFE_DISTANCE
    )

    return inside and outside_obstacle


class MovingTargetEnv:

    def __init__(self):
        self.reset()

    def target_position_at(self, phase):

        cx, cy = self.target_center

        return np.array(
            [
                cx + TARGET_AX * math.sin(phase),
                cy + TARGET_AY * math.sin(2.0 * phase),
            ],
            dtype=np.float32,
        )

    def target_velocity_at(self, phase):

        # Analytical derivative of target trajectory:
        #
        # dx/dt = AX * omega * cos(phase)
        # dy/dt = 2*AY*omega * cos(2*phase)

        return np.array(
            [
                TARGET_AX
                * TARGET_OMEGA
                * math.cos(phase),

                2.0
                * TARGET_AY
                * TARGET_OMEGA
                * math.cos(2.0 * phase),
            ],
            dtype=np.float32,
        )

    def reset(self):

        self.position = np.array(
            [80.0, 280.0],
            dtype=np.float32,
        )

        self.velocity = np.zeros(
            2,
            dtype=np.float32,
        )

        self.heading = 0.0

        self.target_center = (
            INITIAL_TARGET_CENTER.copy()
        )

        self.phase = random.uniform(
            0.0,
            2.0 * math.pi,
        )

        self.target = self.target_position_at(
            self.phase
        )

        self.target_velocity = (
            self.target_velocity_at(
                self.phase
            )
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

    def set_target_center(self, center):

        center = np.asarray(
            center,
            dtype=np.float32,
        )

        # Keep the complete trajectory inside the world.
        margin_x = TARGET_AX + 25.0
        margin_y = TARGET_AY + 25.0

        if not (
            margin_x <= center[0] <= WORLD_W - margin_x
            and margin_y <= center[1] <= WORLD_H - margin_y
        ):
            return False

        # Ensure the target trajectory remains away from
        # the obstacle for every phase.
        #
        # A conservative bounding-circle test is used here.
        trajectory_radius = math.sqrt(
            TARGET_AX ** 2
            + TARGET_AY ** 2
        )

        if (
            distance(center, OBSTACLE)
            < SAFE_DISTANCE + trajectory_radius
        ):
            return False

        self.target_center = center.copy()

        # Keep the target moving continuously from its
        # current phase rather than freezing it.
        self.target = self.target_position_at(
            self.phase
        )

        self.target_velocity = (
            self.target_velocity_at(
                self.phase
            )
        )

        self.previous_distance = distance(
            self.position,
            self.target,
        )

        return True

    def get_state(self):

        relative = (
            self.target - self.position
        )

        obstacle_relative = (
            OBSTACLE - self.position
        )

        target_angle = math.atan2(
            relative[1],
            relative[0],
        )

        angle_error = (
            target_angle
            - self.heading
            + math.pi
        ) % (2.0 * math.pi) - math.pi

        target_distance = np.linalg.norm(
            relative
        )

        return np.array(
            [
                relative[0] / WORLD_W,
                relative[1] / WORLD_H,

                target_distance / 900.0,

                math.sin(angle_error),
                math.cos(angle_error),

                self.velocity[0] / MAX_SPEED,
                self.velocity[1] / MAX_SPEED,

                obstacle_relative[0] / WORLD_W,
                obstacle_relative[1] / WORLD_H,

                (
                    np.linalg.norm(obstacle_relative)
                    - SAFE_DISTANCE
                ) / 900.0,

                self.target_velocity[0] / 5.0,
                self.target_velocity[1] / 5.0,
            ],
            dtype=np.float32,
        )

    def step(self, action):

        action = clamp_action(action)

        steering = float(action[0])
        thrust = float(action[1])

        old_distance = distance(
            self.position,
            self.target,
        )

        # ----------------------------------------------------
        # 1. Update robot dynamics.
        # ----------------------------------------------------

        self.heading += (
            steering * MAX_ANGULAR_SPEED
        )

        forward = np.array(
            [
                math.cos(self.heading),
                math.sin(self.heading),
            ],
            dtype=np.float32,
        )

        acceleration = (
            forward * thrust * 0.22
        )

        self.velocity += acceleration

        self.velocity *= 0.96

        speed = np.linalg.norm(
            self.velocity
        )

        if speed > MAX_SPEED:

            self.velocity *= (
                MAX_SPEED / speed
            )

        candidate = (
            self.position + self.velocity
        )

        reward = -0.03

        terminated = False
        truncated = False

        if not safe_position(candidate):

            reward -= 20.0

            self.collision = True
            terminated = True

        else:

            self.position = candidate

        # ----------------------------------------------------
        # 2. Move the target.
        #
        # The target moves once per environment step.
        # ----------------------------------------------------

        self.phase += TARGET_OMEGA

        self.target = self.target_position_at(
            self.phase
        )

        self.target_velocity = (
            self.target_velocity_at(
                self.phase
            )
        )

        # ----------------------------------------------------
        # 3. Reward tracking progress.
        #
        # Distance after the target has moved is important:
        # the robot must follow the moving target, not its
        # previous position.
        # ----------------------------------------------------

        if not terminated:

            new_distance = distance(
                self.position,
                self.target,
            )

            progress = (
                old_distance - new_distance
            )

            reward += 0.25 * progress

            # Penalize excessive speed.
            reward -= (
                0.012
                * np.linalg.norm(self.velocity)
            )

            # Reward maintaining a close tracking distance.
            if new_distance < 55:
                reward += 0.5

            # Tracking success is measured by proximity.
            if new_distance < 28:

                reward += 12.0

                self.success = True
                terminated = True

        self.steps += 1

        if (
            self.steps >= MAX_STEPS
            and not terminated
        ):
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
# STEP 3 — SHARED REPLAY BUFFER
# ============================================================

@dataclass
class Transition:

    state: np.ndarray
    action: np.ndarray
    reward: float
    next_state: np.ndarray
    terminated: float


class ReplayBuffer:

    def __init__(self, capacity):

        self.memory = deque(
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

        self.memory.append(
            Transition(
                state.copy(),
                np.asarray(
                    action,
                    dtype=np.float32,
                ).copy(),
                float(reward),
                next_state.copy(),
                float(terminated),
            )
        )

    def sample(self, batch_size):

        batch = random.sample(
            self.memory,
            batch_size,
        )

        states = np.stack(
            [item.state for item in batch]
        )

        actions = np.stack(
            [item.action for item in batch]
        )

        rewards = np.asarray(
            [item.reward for item in batch],
            dtype=np.float32,
        ).reshape(-1, 1)

        next_states = np.stack(
            [item.next_state for item in batch]
        )

        dones = np.asarray(
            [item.terminated for item in batch],
            dtype=np.float32,
        ).reshape(-1, 1)

        return (
            tensor(states),
            tensor(actions),
            tensor(rewards),
            tensor(next_states),
            tensor(dones),
        )

    def __len__(self):
        return len(self.memory)


# ============================================================
# STEP 4 — DQN
# ============================================================

class DQNNetwork(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                STATE_SIZE,
                HIDDEN,
            ),

            nn.ReLU(),

            nn.Linear(
                HIDDEN,
                HIDDEN,
            ),

            nn.ReLU(),

            nn.Linear(
                HIDDEN,
                len(DQN_ACTIONS),
            ),
        )

    def forward(self, state):
        return self.network(state)


class DQNAgent:

    def __init__(self):

        self.online = DQNNetwork().to(
            DEVICE
        )

        self.target = DQNNetwork().to(
            DEVICE
        )

        hard_update(
            self.online,
            self.target,
        )

        self.optimizer = optim.Adam(
            self.online.parameters(),
            lr=3e-4,
        )

        self.replay = ReplayBuffer(
            REPLAY_CAPACITY
        )

        self.steps = 0

        self.losses = []

    def select_action(
        self,
        state,
        epsilon=0.0,
    ):

        if random.random() < epsilon:

            return random.randrange(
                len(DQN_ACTIONS)
            )

        with torch.no_grad():

            q = self.online(
                tensor(state).unsqueeze(0)
            )

        return int(
            q.argmax(dim=1).item()
        )

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
        ) = self.replay.sample(
            BATCH_SIZE
        )

        actions = actions.long().reshape(
            -1,
            1,
        )

        current_q = self.online(
            states
        ).gather(
            1,
            actions,
        )

        with torch.no_grad():

            next_q = self.target(
                next_states
            ).max(
                dim=1,
                keepdim=True,
            )[0]

            target_q = (
                rewards
                + GAMMA
                * (1.0 - dones)
                * next_q
            )

        loss = F.smooth_l1_loss(
            current_q,
            target_q,
        )

        self.optimizer.zero_grad()

        loss.backward()

        nn.utils.clip_grad_norm_(
            self.online.parameters(),
            10.0,
        )

        self.optimizer.step()

        self.steps += 1

        if self.steps % 250 == 0:

            soft_update(
                self.online,
                self.target,
                tau=0.05,
            )

        self.losses.append(
            float(loss.item())
        )

        return float(loss.item())


# ============================================================
# STEP 5 — ACTOR AND CRITIC NETWORKS
# ============================================================

class DeterministicActor(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                STATE_SIZE,
                HIDDEN,
            ),

            nn.ReLU(),

            nn.Linear(
                HIDDEN,
                HIDDEN,
            ),

            nn.ReLU(),

            nn.Linear(
                HIDDEN,
                ACTION_SIZE,
            ),

            nn.Tanh(),
        )

    def forward(self, state):
        return self.network(state)


class GaussianActor(nn.Module):

    def __init__(self):

        super().__init__()

        self.backbone = nn.Sequential(

            nn.Linear(
                STATE_SIZE,
                HIDDEN,
            ),

            nn.ReLU(),

            nn.Linear(
                HIDDEN,
                HIDDEN,
            ),

            nn.ReLU(),
        )

        self.mean = nn.Linear(
            HIDDEN,
            ACTION_SIZE,
        )

        self.log_std = nn.Linear(
            HIDDEN,
            ACTION_SIZE,
        )

    def forward(self, state):

        features = self.backbone(state)

        mean = self.mean(features)

        log_std = self.log_std(features).clamp(
            -5.0,
            2.0,
        )

        return mean, log_std

    def sample(
        self,
        state,
        deterministic=False,
    ):

        mean, log_std = self.forward(state)

        distribution = torch.distributions.Normal(
            mean,
            log_std.exp(),
        )

        if deterministic:
            pre_tanh = mean
        else:
            pre_tanh = distribution.rsample()

        action = torch.tanh(
            pre_tanh
        )

        # Correct log probability for tanh transformation.
        correction = 2.0 * (
            math.log(2.0)
            - pre_tanh
            - F.softplus(-2.0 * pre_tanh)
        )

        log_prob = (
            distribution.log_prob(pre_tanh)
            - correction
        ).sum(
            dim=-1,
            keepdim=True,
        )

        return (
            action,
            log_prob,
            torch.tanh(mean),
        )


class Critic(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                STATE_SIZE + ACTION_SIZE,
                HIDDEN,
            ),

            nn.ReLU(),

            nn.Linear(
                HIDDEN,
                HIDDEN,
            ),

            nn.ReLU(),

            nn.Linear(
                HIDDEN,
                1,
            ),
        )

    def forward(
        self,
        state,
        action,
    ):

        return self.network(
            torch.cat(
                [
                    state,
                    action,
                ],
                dim=1,
            )
        )


# ============================================================
# STEP 6 — TD3
# ============================================================

class TD3Agent:

    def __init__(self):

        self.actor = DeterministicActor().to(
            DEVICE
        )

        self.actor_target = DeterministicActor().to(
            DEVICE
        )

        hard_update(
            self.actor,
            self.actor_target,
        )

        self.q1 = Critic().to(DEVICE)
        self.q2 = Critic().to(DEVICE)

        self.q1_target = Critic().to(DEVICE)
        self.q2_target = Critic().to(DEVICE)

        hard_update(self.q1, self.q1_target)
        hard_update(self.q2, self.q2_target)

        self.actor_optimizer = optim.Adam(
            self.actor.parameters(),
            lr=ACTOR_LR,
        )

        self.q1_optimizer = optim.Adam(
            self.q1.parameters(),
            lr=CRITIC_LR,
        )

        self.q2_optimizer = optim.Adam(
            self.q2.parameters(),
            lr=CRITIC_LR,
        )

        self.replay = ReplayBuffer(
            REPLAY_CAPACITY
        )

        self.steps = 0

        self.losses = []

    def select_action(
        self,
        state,
        noise=0.0,
    ):

        with torch.no_grad():

            action = self.actor(
                tensor(state).unsqueeze(0)
            ).squeeze(0).cpu().numpy()

        if noise > 0:

            action += np.random.normal(
                0,
                noise,
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
        ) = self.replay.sample(
            BATCH_SIZE
        )

        with torch.no_grad():

            next_actions = self.actor_target(
                next_states
            )

            noise = (
                torch.randn_like(next_actions)
                * 0.2
            ).clamp(
                -0.5,
                0.5,
            )

            next_actions = (
                next_actions + noise
            ).clamp(
                -1.0,
                1.0,
            )

            target_q = torch.minimum(
                self.q1_target(
                    next_states,
                    next_actions,
                ),
                self.q2_target(
                    next_states,
                    next_actions,
                ),
            )

            y = (
                rewards
                + GAMMA
                * (1.0 - dones)
                * target_q
            )

        current_q1 = self.q1(
            states,
            actions,
        )

        current_q2 = self.q2(
            states,
            actions,
        )

        loss1 = F.mse_loss(
            current_q1,
            y,
        )

        loss2 = F.mse_loss(
            current_q2,
            y,
        )

        self.q1_optimizer.zero_grad()
        loss1.backward()

        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        loss2.backward()

        self.q2_optimizer.step()

        self.steps += 1

        if self.steps % 2 == 0:

            actor_loss = -self.q1(
                states,
                self.actor(states),
            ).mean()

            self.actor_optimizer.zero_grad()

            actor_loss.backward()

            self.actor_optimizer.step()

            soft_update(
                self.actor,
                self.actor_target,
            )

            soft_update(
                self.q1,
                self.q1_target,
            )

            soft_update(
                self.q2,
                self.q2_target,
            )

        result = (
            loss1.item() + loss2.item()
        ) / 2.0

        self.losses.append(
            float(result)
        )

        return float(result)


# ============================================================
# STEP 7 — SAC
# ============================================================

class SACAgent:

    def __init__(self):

        self.actor = GaussianActor().to(
            DEVICE
        )

        self.q1 = Critic().to(DEVICE)
        self.q2 = Critic().to(DEVICE)

        self.q1_target = Critic().to(DEVICE)
        self.q2_target = Critic().to(DEVICE)

        hard_update(self.q1, self.q1_target)
        hard_update(self.q2, self.q2_target)

        self.actor_optimizer = optim.Adam(
            self.actor.parameters(),
            lr=ACTOR_LR,
        )

        self.q1_optimizer = optim.Adam(
            self.q1.parameters(),
            lr=CRITIC_LR,
        )

        self.q2_optimizer = optim.Adam(
            self.q2.parameters(),
            lr=CRITIC_LR,
        )

        self.log_alpha = torch.tensor(
            math.log(0.2),
            dtype=torch.float32,
            device=DEVICE,
            requires_grad=True,
        )

        self.alpha_optimizer = optim.Adam(
            [self.log_alpha],
            lr=ALPHA_LR,
        )

        self.target_entropy = -float(
            ACTION_SIZE
        )

        self.replay = ReplayBuffer(
            REPLAY_CAPACITY
        )

        self.steps = 0
        self.losses = []
        self.alphas = []

    @property
    def alpha(self):
        return self.log_alpha.exp().detach()

    def select_action(
        self,
        state,
        deterministic=False,
    ):

        with torch.no_grad():

            action, _, mean_action = (
                self.actor.sample(
                    tensor(state).unsqueeze(0),
                    deterministic=deterministic,
                )
            )

        result = (
            mean_action if deterministic
            else action
        )

        return (
            result.squeeze(0)
            .cpu()
            .numpy()
            .astype(np.float32)
        )

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
        ) = self.replay.sample(
            BATCH_SIZE
        )

        # Soft Bellman target.
        with torch.no_grad():

            next_actions, next_log_prob, _ = (
                self.actor.sample(
                    next_states
                )
            )

            next_q = torch.minimum(
                self.q1_target(
                    next_states,
                    next_actions,
                ),
                self.q2_target(
                    next_states,
                    next_actions,
                ),
            )

            soft_next_q = (
                next_q
                - self.alpha * next_log_prob
            )

            y = (
                rewards
                + GAMMA
                * (1.0 - dones)
                * soft_next_q
            )

        # Twin critic update.
        q1_value = self.q1(
            states,
            actions,
        )

        q2_value = self.q2(
            states,
            actions,
        )

        loss1 = F.mse_loss(
            q1_value,
            y,
        )

        loss2 = F.mse_loss(
            q2_value,
            y,
        )

        self.q1_optimizer.zero_grad()
        loss1.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        loss2.backward()
        self.q2_optimizer.step()

        # Stochastic actor update.
        new_actions, log_prob, _ = (
            self.actor.sample(states)
        )

        min_q = torch.minimum(
            self.q1(states, new_actions),
            self.q2(states, new_actions),
        )

        actor_loss = (
            self.alpha * log_prob - min_q
        ).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Automatic entropy temperature.
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

        soft_update(
            self.q1,
            self.q1_target,
        )

        soft_update(
            self.q2,
            self.q2_target,
        )

        self.steps += 1

        result = (
            loss1.item() + loss2.item()
        ) / 2.0

        self.losses.append(
            float(result)
        )

        self.alphas.append(
            float(self.alpha.item())
        )

        return float(result)


# ============================================================
# STEP 8 — COMMON TRAINING INTERFACE
# ============================================================

def choose_action(
    name,
    agent,
    state,
    episode,
    total_steps,
):

    if name == "DQN":

        epsilon = max(
            0.05,
            0.9 * (
                1.0
                - episode / TRAINING_EPISODES
            ),
        )

        action_index = agent.select_action(
            state,
            epsilon,
        )

        return DQN_ACTIONS[
            action_index
        ].copy()

    if name == "TD3":

        noise = max(
            0.04,
            0.20 * (
                1.0
                - episode / TRAINING_EPISODES
            ),
        )

        return agent.select_action(
            state,
            noise,
        )

    if name == "SAC":

        return agent.select_action(
            state,
            deterministic=False,
        )

    raise ValueError(name)


def train_one(name, agent):

    env = MovingTargetEnv()

    rewards = []
    successes = []
    collisions = []

    total_steps = 0

    start = time.time()

    for episode in range(
        TRAINING_EPISODES
    ):

        state = env.reset()

        episode_reward = 0.0

        for step in range(MAX_STEPS):

            if total_steps < WARMUP_STEPS:

                if name == "DQN":

                    action_index = random.randrange(
                        len(DQN_ACTIONS)
                    )

                    action = DQN_ACTIONS[
                        action_index
                    ].copy()

                else:

                    action = np.random.uniform(
                        -1.0,
                        1.0,
                        ACTION_SIZE,
                    ).astype(np.float32)

            else:

                action = choose_action(
                    name,
                    agent,
                    state,
                    episode,
                    total_steps,
                )

            next_state, reward, terminated, truncated = (
                env.step(action)
            )

            if name == "DQN":

                # Store discrete action index.
                action_index = int(
                    np.argmin(
                        np.sum(
                            (DQN_ACTIONS - action) ** 2,
                            axis=1,
                        )
                    )
                )

                replay_action = np.array(
                    [action_index],
                    dtype=np.float32,
                )

            else:

                replay_action = action

            agent.replay.add(
                state,
                replay_action,
                reward,
                next_state,
                terminated,
            )

            if total_steps >= WARMUP_STEPS:
                agent.train_step()

            state = next_state

            episode_reward += reward
            total_steps += 1

            if terminated or truncated:
                break

        rewards.append(
            episode_reward
        )

        successes.append(
            int(env.success)
        )

        collisions.append(
            int(env.collision)
        )

        if (episode + 1) % 20 == 0:

            window = min(
                20,
                len(rewards),
            )

            print(
                f"{name:4s} | "
                f"Episode {episode+1:3d}/{TRAINING_EPISODES} | "
                f"Reward {np.mean(rewards[-window:]):8.2f} | "
                f"Success {100*np.mean(successes[-window:]):5.1f}% | "
                f"Collision {100*np.mean(collisions[-window:]):5.1f}%"
            )

    return {
        "agent": agent,
        "rewards": rewards,
        "successes": successes,
        "collisions": collisions,
        "seconds": time.time() - start,
    }


# ============================================================
# STEP 9 — BENCHMARK VISUALIZATION
# ============================================================

def smooth(values, window=12):

    values = np.asarray(
        values,
        dtype=np.float32,
    )

    if len(values) < window:
        return values

    return np.convolve(
        values,
        np.ones(window) / window,
        mode="valid",
    )


def plot_comparison(results):

    plt.figure(figsize=(12, 5))

    for name, result in results.items():

        plt.plot(
            smooth(result["rewards"]),
            label=name,
        )

    plt.title(
        "Moving Target Tracking — Episode Reward"
    )

    plt.xlabel("Episode")
    plt.ylabel("Smoothed reward")

    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 5))

    for name, result in results.items():

        plt.plot(
            smooth(
                result["successes"],
                window=10,
            ),
            label=name,
        )

    plt.title(
        "Moving Target Tracking — Success"
    )

    plt.xlabel("Episode")
    plt.ylabel("Smoothed success rate")

    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 5))

    for name, result in results.items():

        plt.plot(
            result["agent"].losses,
            label=name,
        )

    plt.title(
        "Critic / Q-Network Training Loss"
    )

    plt.xlabel("Training update")
    plt.ylabel("Loss")

    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("\nFINAL BENCHMARK")

    for name, result in results.items():

        recent = min(
            20,
            len(result["rewards"]),
        )

        print(
            f"{name:4s} | "
            f"Reward={np.mean(result['rewards'][-recent:]):.2f} | "
            f"Success={100*np.mean(result['successes'][-recent:]):.1f}% | "
            f"Collision={100*np.mean(result['collisions'][-recent:]):.1f}% | "
            f"Time={result['seconds']:.1f}s"
        )


# ============================================================
# STEP 10 — PYGAME MOVING-TARGET VISUALIZATION
# ============================================================

class RocketGUI:

    def __init__(
        self,
        environments,
        agents,
    ):

        pygame.init()

        pygame.display.set_caption(
            "Phase 4.11 — Moving Target Tracking"
        )

        self.screen = pygame.display.set_mode(
            (
                WINDOW_W,
                WINDOW_H,
            )
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

        self.environments = environments
        self.agents = agents

        self.selected = "SAC"

        self.env = self.environments[
            self.selected
        ]

        self.agent = self.agents[
            self.selected
        ]

        self.running = True
        self.paused = False
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

        self.env.velocity = np.zeros(
            2,
            dtype=np.float32,
        )

        self.env.heading = 0.0

        self.env.phase = 0.0

        self.env.target = (
            self.env.target_position_at(
                self.env.phase
            )
        )

        self.env.target_velocity = (
            self.env.target_velocity_at(
                self.env.phase
            )
        )

        self.env.steps = 0
        self.env.total_reward = 0.0

        self.env.success = False
        self.env.collision = False

        self.last_action[:] = 0.0

    def select_algorithm(self, name):

        self.selected = name

        self.env = self.environments[name]
        self.agent = self.agents[name]

        self.reset()

        self.paused = False

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

        self.screen.blit(
            surface,
            (x, y),
        )

    def rotate_point(
        self,
        lx,
        ly,
    ):

        x, y = self.env.position

        angle = self.env.heading

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

    def draw_rocket(self):

        thrust = max(
            0.0,
            float(self.last_action[1]),
        )

        if thrust > 0.05:

            flame = int(
                8 + 24 * thrust
            )

            pygame.draw.polygon(
                self.screen,
                (255, 125, 25),
                [
                    self.rotate_point(-15, -4),
                    self.rotate_point(-15-flame, 0),
                    self.rotate_point(-15, 4),
                ],
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

        pygame.draw.polygon(
            self.screen,
            (230, 240, 250),
            [
                self.rotate_point(x, y)
                for x, y in body
            ],
        )

        pygame.draw.polygon(
            self.screen,
            (70, 160, 230),
            [
                self.rotate_point(x, y)
                for x, y in body
            ],
            2,
        )

        pygame.draw.circle(
            self.screen,
            (40, 140, 245),
            self.rotate_point(8, 0),
            4,
        )

    def draw_world(self):

        self.screen.fill(
            (18, 22, 30)
        )

        pygame.draw.rect(
            self.screen,
            (28, 34, 44),
            (
                0,
                0,
                WORLD_W,
                WORLD_H,
            ),
        )

        for x in range(
            0,
            WORLD_W,
            40,
        ):

            pygame.draw.line(
                self.screen,
                (40, 47, 59),
                (x, 0),
                (x, WORLD_H),
            )

        for y in range(
            0,
            WORLD_H,
            40,
        ):

            pygame.draw.line(
                self.screen,
                (40, 47, 59),
                (0, y),
                (WORLD_W, y),
            )

        # Obstacle and safety boundary.
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

        # Draw approximate target trajectory.
        trajectory = []

        for i in range(100):

            phase = (
                self.env.phase
                + 2.0 * math.pi * i / 100.0
            )

            p = self.env.target_position_at(
                phase
            )

            trajectory.append(
                tuple(p.astype(int))
            )

        pygame.draw.lines(
            self.screen,
            (65, 95, 75),
            True,
            trajectory,
            1,
        )

        # Target velocity indicator.
        target = self.env.target

        velocity_end = (
            target
            + self.env.target_velocity * 12.0
        )

        pygame.draw.line(
            self.screen,
            (255, 220, 80),
            tuple(target.astype(int)),
            tuple(velocity_end.astype(int)),
            3,
        )

        # Moving target.
        pygame.draw.circle(
            self.screen,
            (70, 240, 130),
            tuple(target.astype(int)),
            13,
        )

        pygame.draw.circle(
            self.screen,
            (220, 255, 220),
            tuple(target.astype(int)),
            19,
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
            "MOVING TARGET TRACKING",
            720,
            20,
            (120, 210, 255),
            self.title_font,
        )

        self.draw_text(
            f"Algorithm: {self.selected}",
            720,
            65,
            (100, 255, 160),
        )

        self.draw_text(
            f"Robot: {self.env.position.round(1)}",
            720,
            115,
        )

        self.draw_text(
            f"Target: {self.env.target.round(1)}",
            720,
            145,
        )

        self.draw_text(
            f"Target velocity: {self.env.target_velocity.round(2)}",
            720,
            175,
        )

        self.draw_text(
            f"Distance: {distance(self.env.position, self.env.target):.1f}",
            720,
            205,
        )

        self.draw_text(
            f"Robot velocity: {self.env.velocity.round(2)}",
            720,
            235,
        )

        self.draw_text(
            f"Steering: {self.last_action[0]:+.3f}",
            720,
            285,
        )

        self.draw_text(
            f"Thrust: {self.last_action[1]:+.3f}",
            720,
            315,
        )

        self.draw_text(
            f"Steps: {self.env.steps}",
            720,
            355,
        )

        self.draw_text(
            f"Reward: {self.env.total_reward:.2f}",
            720,
            385,
        )

        self.draw_text(
            f"Simulation speed: {self.speed}",
            720,
            415,
        )

        self.draw_text(
            "ALGORITHM SELECTION",
            720,
            465,
            (120, 210, 255),
        )

        self.draw_text(
            "1: DQN",
            720,
            490,
        )

        self.draw_text(
            "2: TD3",
            720,
            515,
        )

        self.draw_text(
            "3: SAC",
            720,
            540,
        )

        self.draw_text(
            "CLICK: MOVE TRAJECTORY CENTER",
            720,
            570,
        )

        self.draw_text(
            "SPACE: PAUSE | R: RESET | ESC: EXIT",
            720,
            600,
        )

    def step(self):

        state = self.env.get_state()

        if self.selected == "DQN":

            index = self.agent.select_action(
                state,
                epsilon=0.0,
            )

            action = DQN_ACTIONS[index].copy()

        elif self.selected == "TD3":

            action = self.agent.select_action(
                state,
                noise=0.0,
            )

        else:

            action = self.agent.select_action(
                state,
                deterministic=True,
            )

        self.last_action = action.copy()

        _, _, terminated, truncated = (
            self.env.step(action)
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

                    elif event.key == pygame.K_1:
                        self.select_algorithm("DQN")

                    elif event.key == pygame.K_2:
                        self.select_algorithm("TD3")

                    elif event.key == pygame.K_3:
                        self.select_algorithm("SAC")

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

                            changed = (
                                self.env.set_target_center(
                                    [x, y]
                                )
                            )

                            if changed:

                                self.paused = False

                                self.last_action[:] = 0.0

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

    print("\nPHASE 4.11 — UNIT TESTS")

    passed = 0

    def check(name, condition):

        nonlocal passed

        assert bool(condition), (
            f"FAILED: {name}"
        )

        passed += 1

        print(
            "[PASS]",
            name,
        )

    env = MovingTargetEnv()

    # Test 1: State dimension.
    check(
        "State dimension is 12",
        env.get_state().shape == (12,),
    )

    # Test 2: Target position changes over time.
    old_target = env.target.copy()

    env.step(
        np.array(
            [0.0, 0.0],
            dtype=np.float32,
        )
    )

    check(
        "Target moves",
        not np.allclose(
            old_target,
            env.target,
        ),
    )

    # Test 3: Target velocity is available.
    check(
        "Target velocity dimension",
        env.target_velocity.shape == (2,),
    )

    # Test 4: Target velocity is finite.
    check(
        "Target velocity finite",
        np.isfinite(
            env.target_velocity
        ).all(),
    )

    # Test 5: State includes target velocity.
    check(
        "Target velocity encoded in state",
        np.allclose(
            env.get_state()[10:12],
            env.target_velocity / 5.0,
        ),
    )

    # Test 6: Trajectory position remains finite.
    p = env.target_position_at(1.2)

    check(
        "Trajectory position finite",
        np.isfinite(p).all(),
    )

    # Test 7: DQN action count.
    check(
        "DQN has nine discrete actions",
        len(DQN_ACTIONS) == 9,
    )

    # Test 8: DQN network output.
    dqn = DQNAgent()

    q = dqn.online(
        tensor(
            np.zeros(
                (4, STATE_SIZE),
                dtype=np.float32,
            )
        )
    )

    check(
        "DQN output dimensions",
        tuple(q.shape) == (4, 9),
    )

    # Test 9: TD3 actor action range.
    td3 = TD3Agent()

    td3_action = td3.select_action(
        env.get_state()
    )

    check(
        "TD3 action range",
        td3_action.shape == (2,)
        and np.all(np.abs(td3_action) <= 1.0),
    )

    # Test 10: SAC stochastic actor action range.
    sac = SACAgent()

    sac_action = sac.select_action(
        env.get_state(),
        deterministic=False,
    )

    check(
        "SAC action range",
        sac_action.shape == (2,)
        and np.all(np.abs(sac_action) <= 1.0),
    )

    # Test 11: SAC deterministic action range.
    sac_deterministic = sac.select_action(
        env.get_state(),
        deterministic=True,
    )

    check(
        "SAC deterministic action range",
        np.all(
            np.abs(sac_deterministic) <= 1.0
        ),
    )

    # Test 12: Replay buffer insertion.
    replay = ReplayBuffer(100)

    replay.add(
        np.zeros(STATE_SIZE, dtype=np.float32),
        np.zeros(ACTION_SIZE, dtype=np.float32),
        1.0,
        np.ones(STATE_SIZE, dtype=np.float32),
        False,
    )

    check(
        "Replay insertion",
        len(replay) == 1,
    )

    # Test 13: Replay batch dimensions.
    for _ in range(BATCH_SIZE):

        replay.add(
            np.zeros(STATE_SIZE, dtype=np.float32),
            np.zeros(ACTION_SIZE, dtype=np.float32),
            1.0,
            np.ones(STATE_SIZE, dtype=np.float32),
            False,
        )

    batch = replay.sample(
        BATCH_SIZE
    )

    check(
        "Replay batch state dimensions",
        tuple(batch[0].shape)
        == (BATCH_SIZE, STATE_SIZE),
    )

    # Test 14: Safe default trajectory center.
    check(
        "Default target trajectory is safe",
        all(
            safe_position(
                env.target_position_at(
                    2.0 * math.pi * i / 100
                )
            )
            for i in range(100)
        ),
    )

    # Test 15: Moving target center update.
    previous_center = env.target_center.copy()

    changed = env.set_target_center(
        [550.0, 155.0]
    )

    check(
        "Valid trajectory center update",
        changed
        and not np.allclose(
            previous_center,
            env.target_center,
        ),
    )

    # Test 16: Invalid target center rejected.
    previous_center = env.target_center.copy()

    changed = env.set_target_center(
        [350.0, 280.0]
    )

    check(
        "Unsafe target center rejected",
        not changed
        and np.allclose(
            previous_center,
            env.target_center,
        ),
    )

    # Test 17: Environment reward finite.
    transition = env.step(
        np.array(
            [0.0, 0.5],
            dtype=np.float32,
        )
    )

    check(
        "Environment reward finite",
        np.isfinite(transition[1]),
    )

    print(
        f"\n{passed}/17 tests passed."
    )


# ============================================================
# STEP 12 — MAIN
# ============================================================

def main():

    print("=" * 72)
    print("PHASE 4.11 — MOVING-TARGET TRACKING")
    print("=" * 72)

    print(
        "Device:",
        DEVICE,
    )

    run_tests()

    results = {}

    # --------------------------------------------------------
    # DQN
    # --------------------------------------------------------

    print("\nTraining DQN...")

    results["DQN"] = train_one(
        "DQN",
        DQNAgent(),
    )

    # --------------------------------------------------------
    # TD3
    # --------------------------------------------------------

    print("\nTraining TD3...")

    results["TD3"] = train_one(
        "TD3",
        TD3Agent(),
    )

    # --------------------------------------------------------
    # SAC
    # --------------------------------------------------------

    print("\nTraining SAC...")

    results["SAC"] = train_one(
        "SAC",
        SACAgent(),
    )

    # --------------------------------------------------------
    # Compare results.
    # --------------------------------------------------------

    plot_comparison(results)

    # --------------------------------------------------------
    # Save models.
    # --------------------------------------------------------

    torch.save(
        results["DQN"]["agent"].online.state_dict(),
        "moving_dqn.pth",
    )

    torch.save(
        results["TD3"]["agent"].actor.state_dict(),
        "moving_td3_actor.pth",
    )

    torch.save(
        results["SAC"]["agent"].actor.state_dict(),
        "moving_sac_actor.pth",
    )

    print("\nSaved model files:")
    print("  moving_dqn.pth")
    print("  moving_td3_actor.pth")
    print("  moving_sac_actor.pth")

    # --------------------------------------------------------
    # Create independent environments for visualization.
    # --------------------------------------------------------

    environments = {
        name: MovingTargetEnv()
        for name in (
            "DQN",
            "TD3",
            "SAC",
        )
    }

    agents = {
        name: results[name]["agent"]
        for name in (
            "DQN",
            "TD3",
            "SAC",
        )
    }

    # --------------------------------------------------------
    # Launch Pygame.
    # --------------------------------------------------------

    print(
        "\nLaunching moving-target rocket simulator..."
    )

    gui = RocketGUI(
        environments,
        agents,
    )

    gui.run()


if __name__ == "__main__":
    main()