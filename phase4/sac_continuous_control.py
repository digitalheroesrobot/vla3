
# ============================================================
# File: phase4/_sac_continuous_control.py
#
# PHASE 4.10 — IMPROVED SOFT ACTOR-CRITIC (SAC)
#
# PURPOSE
#   Learn continuous control for a quadcopter tracking a
#   continuously moving target while avoiding an obstacle.
#
# WHAT YOU LEARN
#   Step 1: Model continuous vehicle dynamics.
#   Step 2: Represent a moving target using position and velocity.
#   Step 3: Implement a tanh-squashed Gaussian actor.
#   Step 4: Implement twin critics and target critics.
#   Step 5: Implement entropy-regularized SAC updates.
#   Step 6: Implement automatic entropy-temperature tuning.
#   Step 7: Use a replay buffer with safe batch sampling.
#   Step 8: Shape rewards for target tracking and obstacle clearance.
#   Step 9: Use a moving-target speed curriculum.
#   Step 10: Evaluate the deterministic policy separately.
#   Step 11: Visualize the quadcopter and moving target.
#
# IMPORTANT IMPROVEMENTS
#   - The target trajectory no longer intersects the obstacle.
#   - The target starts slowly and gradually accelerates.
#   - Initial drone positions are selected from a consistent region.
#   - Obstacle clearance is included in the reward.
#   - Target tracking uses dense progress rewards.
#   - Target orbit centers are randomized during training.
#   - Evaluation is deterministic and separate from training.
#
# SAC FORMULAS
#
#   Gaussian policy:
#       u = mu(s) + sigma(s) * epsilon
#
#   Squashed action:
#       a = tanh(u)
#
#   Soft Bellman target:
#       y = r + gamma*(1-done) *
#           [min(Q1_target,Q2_target) - alpha*log_pi]
#
#   Actor loss:
#       L_pi = E[alpha*log_pi - min(Q1,Q2)]
#
#   Temperature:
#       alpha = exp(log_alpha)
#
# ROBOTICS APPLICATION
#   Moving-target tracking with continuous steering and thrust.
#
# PYGAME CONTROLS
#   Left click : Change target trajectory center
#   SPACE      : Pause / resume
#   R          : Reset simulation
#   +/-        : Change simulation speed
#   ESC        : Exit
#
# INSTALL
#   pip install numpy pygame torch matplotlib
#
# RUN
#   python _sac_continuous_control.py
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

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

WORLD_W = 700
WORLD_H = 560

WINDOW_W = 1020
WINDOW_H = 650

STATE_SIZE = 12
ACTION_SIZE = 2

HIDDEN = 256

GAMMA = 0.99
TAU = 0.005

ACTOR_LR = 3e-4
CRITIC_LR = 3e-4
ALPHA_LR = 1e-4

REPLAY_CAPACITY = 150_000
BATCH_SIZE = 256
WARMUP_STEPS = 2_000

TRAINING_EPISODES = 300
MAX_STEPS = 300

INITIAL_ALPHA = 0.15
TARGET_ENTROPY = -float(ACTION_SIZE)

LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0

MAX_SPEED = 5.0
MAX_ANGULAR_SPEED = 0.09
MAX_ACCELERATION = 0.22

DRONE_RADIUS = 12.0

OBSTACLE = np.array(
    [350.0, 280.0],
    dtype=np.float32,
)

OBSTACLE_RADIUS = 55.0

SAFE_DISTANCE = (
    OBSTACLE_RADIUS
    + DRONE_RADIUS
    + 22.0
)

SUCCESS_DISTANCE = 30.0

# A safe target orbit.
#
# The target remains on the right side of the obstacle.
# Its trajectory does not cross the obstacle safety region.

TARGET_ORBIT_RADIUS_X = 75.0
TARGET_ORBIT_RADIUS_Y = 45.0

TARGET_OMEGA_START = 0.008
TARGET_OMEGA_END = 0.022

# Randomized training target centers.
TARGET_CENTER_MIN = np.array(
    [520.0, 110.0],
    dtype=np.float32,
)

TARGET_CENTER_MAX = np.array(
    [580.0, 170.0],
    dtype=np.float32,
)

# Consistent initial spawn region.
SPAWN_X_MIN = 55.0
SPAWN_X_MAX = 220.0

SPAWN_Y_MIN = 65.0
SPAWN_Y_MAX = 495.0

# Reward configuration.
PROGRESS_WEIGHT = 0.35
DISTANCE_WEIGHT = 0.08
OBSTACLE_WEIGHT = 0.20
VELOCITY_WEIGHT = 0.008
ACTION_WEIGHT = 0.015

SUCCESS_BONUS = 35.0
COLLISION_PENALTY = 35.0


# ============================================================
# STEP 1 — HELPERS
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

    target.load_state_dict(
        source.state_dict()
    )


def obstacle_clearance(position):

    return (
        distance(position, OBSTACLE)
        - SAFE_DISTANCE
    )


def obstacle_potential(position):

    # Negative potential becomes more severe near obstacles.
    #
    # This gives the agent a smooth signal before collision.

    clearance = obstacle_clearance(position)

    return -math.exp(
        -max(clearance, 0.0) / 65.0
    )


# ============================================================
# STEP 2 — ENVIRONMENT
# ============================================================

def safe_position(position):

    x, y = position

    inside_world = (
        DRONE_RADIUS <= x <= WORLD_W - DRONE_RADIUS
        and DRONE_RADIUS <= y <= WORLD_H - DRONE_RADIUS
    )

    outside_obstacle = (
        distance(position, OBSTACLE) >= SAFE_DISTANCE
    )

    return inside_world and outside_obstacle


class DroneEnv:

    def __init__(self):

        self.omega = TARGET_OMEGA_START

        self.reset()

    def random_spawn(self):

        for _ in range(5000):

            position = np.array(
                [
                    random.uniform(
                        SPAWN_X_MIN,
                        SPAWN_X_MAX,
                    ),
                    random.uniform(
                        SPAWN_Y_MIN,
                        SPAWN_Y_MAX,
                    ),
                ],
                dtype=np.float32,
            )

            if safe_position(position):
                return position

        raise RuntimeError(
            "Could not generate a safe spawn position."
        )

    def random_target_center(self):

        return np.array(
            [
                random.uniform(
                    TARGET_CENTER_MIN[0],
                    TARGET_CENTER_MAX[0],
                ),
                random.uniform(
                    TARGET_CENTER_MIN[1],
                    TARGET_CENTER_MAX[1],
                ),
            ],
            dtype=np.float32,
        )

    def reset(self):

        self.position = self.random_spawn()

        self.velocity = np.zeros(
            2,
            dtype=np.float32,
        )

        # Initial heading points approximately toward
        # the target region.
        initial_target = np.array(
            [550.0, 140.0],
            dtype=np.float32,
        )

        delta = initial_target - self.position

        self.heading = math.atan2(
            delta[1],
            delta[0],
        )

        self.target_center = self.random_target_center()

        self.target_phase = random.uniform(
            0.0,
            2.0 * math.pi,
        )

        self.target = np.zeros(
            2,
            dtype=np.float32,
        )

        self.target_velocity = np.zeros(
            2,
            dtype=np.float32,
        )

        self.update_target()

        self.steps = 0
        self.total_reward = 0.0

        self.success = False
        self.collision = False

        return self.get_state()

    def update_target(self):

        phase = self.target_phase

        self.target = self.target_center + np.array(
            [
                TARGET_ORBIT_RADIUS_X * math.cos(phase),
                TARGET_ORBIT_RADIUS_Y * math.sin(phase),
            ],
            dtype=np.float32,
        )

        self.target_velocity = np.array(
            [
                -TARGET_ORBIT_RADIUS_X
                * self.omega
                * math.sin(phase),

                TARGET_ORBIT_RADIUS_Y
                * self.omega
                * math.cos(phase),
            ],
            dtype=np.float32,
        )

    def advance_target(self):

        self.target_phase += self.omega

        self.update_target()

    def set_target_center(self, center):

        center = np.asarray(
            center,
            dtype=np.float32,
        )

        center = np.clip(
            center,
            TARGET_CENTER_MIN,
            TARGET_CENTER_MAX,
        )

        phase = self.target_phase

        candidate = center + np.array(
            [
                TARGET_ORBIT_RADIUS_X * math.cos(phase),
                TARGET_ORBIT_RADIUS_Y * math.sin(phase),
            ],
            dtype=np.float32,
        )

        if not safe_position(candidate):
            return False

        self.target_center = center

        self.update_target()

        return True

    def get_state(self):

        target_delta = (
            self.target - self.position
        )

        target_angle = math.atan2(
            target_delta[1],
            target_delta[0],
        )

        angle_error = (
            target_angle
            - self.heading
            + math.pi
        ) % (2.0 * math.pi) - math.pi

        target_distance = np.linalg.norm(
            target_delta
        )

        obstacle_delta = (
            OBSTACLE - self.position
        )

        clearance = obstacle_clearance(
            self.position
        )

        return np.array(
            [
                target_delta[0] / WORLD_W,
                target_delta[1] / WORLD_H,

                self.target_velocity[0] / MAX_SPEED,
                self.target_velocity[1] / MAX_SPEED,

                target_distance / 900.0,

                math.sin(angle_error),
                math.cos(angle_error),

                self.velocity[0] / MAX_SPEED,
                self.velocity[1] / MAX_SPEED,

                obstacle_delta[0] / WORLD_W,
                obstacle_delta[1] / WORLD_H,

                clearance / 900.0,
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

        old_obstacle_potential = obstacle_potential(
            self.position
        )

        # ----------------------------------------------------
        # Continuous vehicle dynamics
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
            forward
            * thrust
            * MAX_ACCELERATION
        )

        self.velocity += acceleration

        # Physical damping.
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

        # Advance target before measuring new distance.
        self.advance_target()

        new_distance = distance(
            candidate,
            self.target,
        )

        reward = -0.02

        terminated = False
        truncated = False

        # ----------------------------------------------------
        # Safety
        # ----------------------------------------------------

        if not safe_position(candidate):

            reward -= COLLISION_PENALTY

            self.collision = True

            terminated = True

        else:

            self.position = candidate

            # ------------------------------------------------
            # Tracking reward
            # ------------------------------------------------

            progress = (
                old_distance - new_distance
            )

            reward += (
                PROGRESS_WEIGHT
                * progress
            )

            # Dense distance reward.
            reward -= (
                DISTANCE_WEIGHT
                * new_distance / 100.0
            )

            # ------------------------------------------------
            # Obstacle-clearance reward shaping
            # ------------------------------------------------

            new_obstacle_potential = obstacle_potential(
                self.position
            )

            reward += (
                OBSTACLE_WEIGHT
                * (
                    new_obstacle_potential
                    - old_obstacle_potential
                )
            )

            # ------------------------------------------------
            # Velocity tracking
            # ------------------------------------------------

            velocity_error = np.linalg.norm(
                self.target_velocity
                - self.velocity
            )

            reward -= (
                VELOCITY_WEIGHT
                * velocity_error
            )

            # Avoid excessive control.
            reward -= (
                ACTION_WEIGHT
                * float(np.sum(action ** 2))
            )

            # ------------------------------------------------
            # Success
            # ------------------------------------------------

            if new_distance < SUCCESS_DISTANCE:

                reward += SUCCESS_BONUS

                self.success = True

                terminated = True

        self.steps += 1

        if (
            self.steps >= MAX_STEPS
            and not terminated
        ):

            truncated = True

        self.total_reward += reward

        return (
            self.get_state(),
            float(reward),
            terminated,
            truncated,
        )


# ============================================================
# STEP 3 — GAUSSIAN ACTOR
# ============================================================

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

        self.mean_layer = nn.Linear(
            HIDDEN,
            ACTION_SIZE,
        )

        self.log_std_layer = nn.Linear(
            HIDDEN,
            ACTION_SIZE,
        )

    def forward(self, state):

        features = self.backbone(state)

        mean = self.mean_layer(features)

        log_std = self.log_std_layer(features)

        log_std = torch.clamp(
            log_std,
            LOG_STD_MIN,
            LOG_STD_MAX,
        )

        return mean, log_std

    def sample(
        self,
        state,
        deterministic=False,
    ):

        mean, log_std = self.forward(state)

        std = log_std.exp()

        distribution = torch.distributions.Normal(
            mean,
            std,
        )

        if deterministic:

            pre_tanh = mean

        else:

            pre_tanh = distribution.rsample()

        action = torch.tanh(
            pre_tanh
        )

        log_prob = distribution.log_prob(
            pre_tanh
        )

        # Numerically stable tanh Jacobian correction.
        correction = 2.0 * (
            math.log(2.0)
            - pre_tanh
            - F.softplus(-2.0 * pre_tanh)
        )

        log_prob = (
            log_prob - correction
        ).sum(
            dim=-1,
            keepdim=True,
        )

        return (
            action,
            log_prob,
            torch.tanh(mean),
        )


# ============================================================
# STEP 4 — TWIN CRITICS
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
                state=np.asarray(
                    state,
                    dtype=np.float32,
                ).copy(),

                action=np.asarray(
                    action,
                    dtype=np.float32,
                ).copy(),

                reward=float(reward),

                next_state=np.asarray(
                    next_state,
                    dtype=np.float32,
                ).copy(),

                terminated=float(terminated),
            )
        )

    def sample(self, batch_size):

        if batch_size <= 0:

            raise ValueError(
                "batch_size must be positive."
            )

        if len(self.memory) < batch_size:

            raise ValueError(
                f"Insufficient replay data: "
                f"requested {batch_size}, "
                f"available {len(self.memory)}."
            )

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

        def tensor(array):

            return torch.as_tensor(
                array,
                dtype=torch.float32,
                device=DEVICE,
            )

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
# STEP 6 — SAC AGENT
# ============================================================

class SACAgent:

    def __init__(self):

        self.actor = GaussianActor().to(
            DEVICE
        )

        self.critic1 = Critic().to(
            DEVICE
        )

        self.critic2 = Critic().to(
            DEVICE
        )

        self.target1 = Critic().to(
            DEVICE
        )

        self.target2 = Critic().to(
            DEVICE
        )

        hard_update(
            self.critic1,
            self.target1,
        )

        hard_update(
            self.critic2,
            self.target2,
        )

        self.actor_optimizer = optim.Adam(
            self.actor.parameters(),
            lr=ACTOR_LR,
        )

        self.critic1_optimizer = optim.Adam(
            self.critic1.parameters(),
            lr=CRITIC_LR,
        )

        self.critic2_optimizer = optim.Adam(
            self.critic2.parameters(),
            lr=CRITIC_LR,
        )

        self.log_alpha = torch.tensor(
            math.log(INITIAL_ALPHA),
            dtype=torch.float32,
            device=DEVICE,
            requires_grad=True,
        )

        self.alpha_optimizer = optim.Adam(
            [self.log_alpha],
            lr=ALPHA_LR,
        )

        self.replay = ReplayBuffer(
            REPLAY_CAPACITY
        )

        self.critic_losses = []
        self.actor_losses = []
        self.alpha_values = []

        self.training_steps = 0

    @property
    def alpha(self):

        return self.log_alpha.exp().detach()

    def select_action(
        self,
        state,
        deterministic=False,
    ):

        state_tensor = torch.as_tensor(
            state,
            dtype=torch.float32,
            device=DEVICE,
        ).unsqueeze(0)

        self.actor.eval()

        with torch.no_grad():

            action, _, mean_action = self.actor.sample(
                state_tensor,
                deterministic=deterministic,
            )

        self.actor.train()

        result = (
            mean_action
            if deterministic
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

        # ----------------------------------------------------
        # 6.1 Soft Bellman target
        # ----------------------------------------------------

        with torch.no_grad():

            next_actions, next_log_prob, _ = (
                self.actor.sample(
                    next_states
                )
            )

            target_q1 = self.target1(
                next_states,
                next_actions,
            )

            target_q2 = self.target2(
                next_states,
                next_actions,
            )

            target_min = torch.minimum(
                target_q1,
                target_q2,
            )

            soft_target_q = (
                target_min
                - self.alpha * next_log_prob
            )

            target = (
                rewards
                + GAMMA
                * (1.0 - dones)
                * soft_target_q
            )

        # ----------------------------------------------------
        # 6.2 Critic updates
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
            target,
        )

        critic2_loss = F.mse_loss(
            current_q2,
            target,
        )

        self.critic1_optimizer.zero_grad()

        critic1_loss.backward()

        nn.utils.clip_grad_norm_(
            self.critic1.parameters(),
            10.0,
        )

        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()

        critic2_loss.backward()

        nn.utils.clip_grad_norm_(
            self.critic2.parameters(),
            10.0,
        )

        self.critic2_optimizer.step()

        # ----------------------------------------------------
        # 6.3 Actor update
        # ----------------------------------------------------

        new_actions, log_prob, _ = (
            self.actor.sample(states)
        )

        q1_pi = self.critic1(
            states,
            new_actions,
        )

        q2_pi = self.critic2(
            states,
            new_actions,
        )

        min_q_pi = torch.minimum(
            q1_pi,
            q2_pi,
        )

        actor_loss = (
            self.alpha * log_prob
            - min_q_pi
        ).mean()

        self.actor_optimizer.zero_grad()

        actor_loss.backward()

        nn.utils.clip_grad_norm_(
            self.actor.parameters(),
            10.0,
        )

        self.actor_optimizer.step()

        # ----------------------------------------------------
        # 6.4 Automatic entropy temperature
        # ----------------------------------------------------

        alpha_loss = -(
            self.log_alpha
            * (
                log_prob.detach()
                + TARGET_ENTROPY
            )
        ).mean()

        self.alpha_optimizer.zero_grad()

        alpha_loss.backward()

        self.alpha_optimizer.step()

        # Prevent extreme temperature values.
        with torch.no_grad():

            self.log_alpha.clamp_(
                min=-8.0,
                max=2.0,
            )

        # ----------------------------------------------------
        # 6.5 Target critic updates
        # ----------------------------------------------------

        soft_update(
            self.critic1,
            self.target1,
        )

        soft_update(
            self.critic2,
            self.target2,
        )

        self.training_steps += 1

        combined_critic_loss = (
            critic1_loss.item()
            + critic2_loss.item()
        ) / 2.0

        self.critic_losses.append(
            combined_critic_loss
        )

        self.actor_losses.append(
            float(actor_loss.item())
        )

        self.alpha_values.append(
            float(self.alpha.item())
        )

        return {
            "critic_loss": combined_critic_loss,
            "actor_loss": float(actor_loss.item()),
            "alpha": float(self.alpha.item()),
            "entropy": float(
                -log_prob.detach().mean().item()
            ),
        }


# ============================================================
# STEP 7 — MOVING-TARGET CURRICULUM
# ============================================================

def curriculum_omega(episode):

    progress = min(
        1.0,
        episode / max(1, TRAINING_EPISODES - 1),
    )

    return (
        TARGET_OMEGA_START
        + progress
        * (
            TARGET_OMEGA_END
            - TARGET_OMEGA_START
        )
    )


# ============================================================
# STEP 8 — TRAINING
# ============================================================

def train_agent(
    agent,
    episodes=TRAINING_EPISODES,
):

    env = DroneEnv()

    episode_rewards = []
    episode_successes = []
    episode_collisions = []

    total_steps = 0

    start_time = time.time()

    for episode in range(episodes):

        # Increase target speed gradually.
        env.omega = curriculum_omega(episode)

        state = env.reset()

        episode_reward = 0.0

        for step in range(MAX_STEPS):

            # ------------------------------------------------
            # Exploration
            # ------------------------------------------------

            if total_steps < WARMUP_STEPS:

                action = np.random.uniform(
                    -1.0,
                    1.0,
                    size=ACTION_SIZE,
                ).astype(np.float32)

            else:

                action = agent.select_action(
                    state,
                    deterministic=False,
                )

            (
                next_state,
                reward,
                terminated,
                truncated,
            ) = env.step(action)

            agent.replay.add(
                state,
                action,
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

        episode_rewards.append(
            episode_reward
        )

        episode_successes.append(
            int(env.success)
        )

        episode_collisions.append(
            int(env.collision)
        )

        if (episode + 1) % 20 == 0:

            recent = min(
                20,
                len(episode_rewards),
            )

            print(
                f"Episode {episode + 1:3d}/{episodes} | "
                f"Reward={np.mean(episode_rewards[-recent:]):8.2f} | "
                f"Success={100*np.mean(episode_successes[-recent:]):5.1f}% | "
                f"Collision={100*np.mean(episode_collisions[-recent:]):5.1f}% | "
                f"Omega={env.omega:.4f} | "
                f"Alpha={agent.alpha.item():.4f}"
            )

    elapsed = time.time() - start_time

    print(
        f"\nTraining completed in {elapsed:.1f} seconds."
    )

    return {
        "rewards": episode_rewards,
        "successes": episode_successes,
        "collisions": episode_collisions,
        "elapsed": elapsed,
    }


# ============================================================
# STEP 9 — DETERMINISTIC EVALUATION
# ============================================================

def evaluate_agent(
    agent,
    episodes=30,
):

    print("\nDETERMINISTIC EVALUATION")

    env = DroneEnv()

    rewards = []
    successes = []
    collisions = []

    distances = []

    for episode in range(episodes):

        # Evaluate at the final target speed.
        env.omega = TARGET_OMEGA_END

        state = env.reset()

        episode_reward = 0.0
        final_distance = float("inf")

        for _ in range(MAX_STEPS):

            action = agent.select_action(
                state,
                deterministic=True,
            )

            (
                next_state,
                reward,
                terminated,
                truncated,
            ) = env.step(action)

            state = next_state

            episode_reward += reward

            final_distance = distance(
                env.position,
                env.target,
            )

            if terminated or truncated:
                break

        rewards.append(episode_reward)

        successes.append(
            int(env.success)
        )

        collisions.append(
            int(env.collision)
        )

        distances.append(
            final_distance
        )

    print(
        f"Evaluation episodes: {episodes}"
    )

    print(
        f"Mean reward: {np.mean(rewards):.2f}"
    )

    print(
        f"Success rate: {100*np.mean(successes):.1f}%"
    )

    print(
        f"Collision rate: {100*np.mean(collisions):.1f}%"
    )

    print(
        f"Mean final distance: {np.mean(distances):.2f}"
    )

    return {
        "rewards": rewards,
        "successes": successes,
        "collisions": collisions,
        "distances": distances,
    }


# ============================================================
# STEP 10 — TRAINING VISUALIZATION
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
        np.ones(window) / window,
        mode="valid",
    )


def plot_training(
    agent,
    result,
    evaluation,
):

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(13, 9),
    )

    axes[0, 0].plot(
        moving_average(result["rewards"]),
    )

    axes[0, 0].set_title(
        "Training reward"
    )

    axes[0, 0].set_xlabel(
        "Episode"
    )

    axes[0, 0].grid(True)

    axes[0, 1].plot(
        moving_average(result["successes"]),
    )

    axes[0, 1].set_title(
        "Training success rate"
    )

    axes[0, 1].set_xlabel(
        "Episode"
    )

    axes[0, 1].set_ylim(
        -0.05,
        1.05,
    )

    axes[0, 1].grid(True)

    axes[1, 0].plot(
        agent.critic_losses,
        label="Critic",
    )

    axes[1, 0].plot(
        agent.actor_losses,
        label="Actor",
    )

    axes[1, 0].set_title(
        "Training losses"
    )

    axes[1, 0].set_xlabel(
        "Update"
    )

    axes[1, 0].legend()
    axes[1, 0].grid(True)

    axes[1, 1].plot(
        agent.alpha_values,
    )

    axes[1, 1].set_title(
        "Entropy temperature"
    )

    axes[1, 1].set_xlabel(
        "Update"
    )

    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.show()

    print("\nFINAL TRAINING PERFORMANCE")

    print(
        "Last 25 episode success rate:",
        f"{100*np.mean(result['successes'][-25:]):.1f}%",
    )

    print(
        "Last 25 episode collision rate:",
        f"{100*np.mean(result['collisions'][-25:]):.1f}%",
    )

    print("\nFINAL DETERMINISTIC EVALUATION")

    print(
        "Success rate:",
        f"{100*np.mean(evaluation['successes']):.1f}%",
    )

    print(
        "Collision rate:",
        f"{100*np.mean(evaluation['collisions']):.1f}%",
    )


# ============================================================
# STEP 11 — PYGAME QUADCOPTER VISUALIZATION
# ============================================================

class DroneGUI:

    def __init__(
        self,
        env,
        agent,
    ):

        pygame.init()

        pygame.display.set_caption(
            "Phase 4.10 — Improved SAC Quadcopter"
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
            [90.0, 280.0],
            dtype=np.float32,
        )

        self.env.velocity = np.zeros(
            2,
            dtype=np.float32,
        )

        self.env.heading = 0.0

        self.env.target_center = np.array(
            [550.0, 140.0],
            dtype=np.float32,
        )

        self.env.target_phase = 0.0
        self.env.omega = TARGET_OMEGA_END

        self.env.update_target()

        self.env.steps = 0
        self.env.total_reward = 0.0

        self.env.success = False
        self.env.collision = False

        self.last_action[:] = 0.0

    def text(
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
        local_x,
        local_y,
    ):

        x, y = self.env.position

        angle = self.env.heading

        return (
            int(
                x
                + local_x * math.cos(angle)
                - local_y * math.sin(angle)
            ),
            int(
                y
                + local_x * math.sin(angle)
                + local_y * math.cos(angle)
            ),
        )

    def draw_drone(self):

        center = self.rotate_point(
            0,
            0,
        )

        rotor_positions = [
            (13, 13),
            (13, -13),
            (-13, 13),
            (-13, -13),
        ]

        for rx, ry in rotor_positions:

            rotor = self.rotate_point(
                rx,
                ry,
            )

            pygame.draw.line(
                self.screen,
                (190, 210, 230),
                center,
                rotor,
                4,
            )

            pygame.draw.circle(
                self.screen,
                (80, 180, 240),
                rotor,
                7,
                2,
            )

        pygame.draw.circle(
            self.screen,
            (235, 240, 250),
            center,
            8,
        )

        pygame.draw.circle(
            self.screen,
            (60, 150, 240),
            center,
            4,
        )

        pygame.draw.line(
            self.screen,
            (255, 210, 80),
            center,
            self.rotate_point(19, 0),
            3,
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

        # Obstacle safety boundary.
        pygame.draw.circle(
            self.screen,
            (220, 130, 70),
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

        # Target trajectory.
        trajectory = []

        for i in range(120):

            phase = (
                self.env.target_phase
                + 2.0 * math.pi * i / 120
            )

            point = self.env.target_center + np.array(
                [
                    TARGET_ORBIT_RADIUS_X * math.cos(phase),
                    TARGET_ORBIT_RADIUS_Y * math.sin(phase),
                ]
            )

            trajectory.append(
                tuple(point.astype(int))
            )

        pygame.draw.lines(
            self.screen,
            (55, 105, 75),
            True,
            trajectory,
            1,
        )

        # Target trajectory center.
        pygame.draw.circle(
            self.screen,
            (240, 200, 80),
            tuple(self.env.target_center.astype(int)),
            7,
            2,
        )

        # Moving target.
        tx, ty = self.env.target.astype(int)

        pygame.draw.circle(
            self.screen,
            (70, 240, 130),
            (tx, ty),
            16,
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

        velocity_end = (
            self.env.target
            + self.env.target_velocity * 3.0
        )

        pygame.draw.line(
            self.screen,
            (100, 230, 160),
            tuple(self.env.target.astype(int)),
            tuple(velocity_end.astype(int)),
            2,
        )

        self.draw_drone()

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

        self.text(
            "IMPROVED SAC",
            720,
            20,
            (120, 210, 255),
            self.title_font,
        )

        self.text(
            f"Position: {self.env.position.round(1)}",
            720,
            85,
        )

        self.text(
            f"Velocity: {self.env.velocity.round(2)}",
            720,
            115,
        )

        self.text(
            f"Target: {self.env.target.round(1)}",
            720,
            145,
        )

        self.text(
            f"Target velocity: {self.env.target_velocity.round(2)}",
            720,
            175,
        )

        self.text(
            f"Distance: {distance(self.env.position, self.env.target):.1f}",
            720,
            205,
        )

        self.text(
            f"Clearance: {obstacle_clearance(self.env.position):.1f}",
            720,
            235,
        )

        self.text(
            f"Steering: {self.last_action[0]:+.3f}",
            720,
            280,
        )

        self.text(
            f"Thrust: {self.last_action[1]:+.3f}",
            720,
            310,
        )

        self.text(
            f"Alpha: {self.agent.alpha.item():.4f}",
            720,
            355,
        )

        self.text(
            f"Steps: {self.env.steps}",
            720,
            385,
        )

        self.text(
            f"Reward: {self.env.total_reward:.2f}",
            720,
            415,
        )

        self.text(
            f"Simulation speed: {self.speed}",
            720,
            445,
        )

        self.text(
            "LEFT CLICK: CHANGE TARGET CENTER",
            720,
            500,
        )

        self.text(
            "SPACE: PAUSE / RESUME",
            720,
            535,
        )

        self.text(
            "R: RESET | +/-: SPEED",
            720,
            565,
        )

        self.text(
            "ESC: EXIT",
            720,
            595,
        )

    def step(self):

        state = self.env.get_state()

        action = self.agent.select_action(
            state,
            deterministic=True,
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

                            if self.env.set_target_center(
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

    print("\nPHASE 4.10 — UNIT TESTS")

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

    env = DroneEnv()

    # Test 1.
    check(
        "State dimension",
        env.get_state().shape == (STATE_SIZE,),
    )

    # Test 2.
    actor = GaussianActor().to(DEVICE)

    states = torch.randn(
        8,
        STATE_SIZE,
        device=DEVICE,
    )

    mean, log_std = actor(states)

    check(
        "Actor mean dimensions",
        tuple(mean.shape) == (8, ACTION_SIZE),
    )

    # Test 3.
    check(
        "Log standard deviation bounds",
        bool(
            torch.all(log_std >= LOG_STD_MIN)
            and torch.all(log_std <= LOG_STD_MAX)
        ),
    )

    # Test 4.
    actions, log_prob, mean_action = actor.sample(
        states
    )

    check(
        "Squashed action range",
        bool(
            torch.all(actions >= -1.0)
            and torch.all(actions <= 1.0)
        ),
    )

    # Test 5.
    check(
        "Log probability dimensions",
        tuple(log_prob.shape) == (8, 1),
    )

    # Test 6.
    check(
        "Mean action range",
        bool(
            torch.all(mean_action >= -1.0)
            and torch.all(mean_action <= 1.0)
        ),
    )

    # Test 7.
    gradient = torch.autograd.grad(
        actions.sum() + log_prob.sum(),
        list(actor.parameters()),
        allow_unused=True,
    )

    has_gradient = any(
        g is not None
        and torch.isfinite(g).all()
        and torch.any(g != 0)
        for g in gradient
    )

    check(
        "Reparameterized gradient",
        has_gradient,
    )

    # Test 8.
    critic1 = Critic().to(DEVICE)
    critic2 = Critic().to(DEVICE)

    q = critic1(
        states,
        actions,
    )

    check(
        "Critic output dimensions",
        tuple(q.shape) == (8, 1),
    )

    # Test 9.
    different = any(
        not torch.equal(a, b)
        for a, b in zip(
            critic1.parameters(),
            critic2.parameters(),
        )
    )

    check(
        "Independent critics",
        different,
    )

    # Test 10.
    replay = ReplayBuffer(1000)

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

    # Test 11.
    insufficient_data_detected = False

    try:
        replay.sample(BATCH_SIZE)

    except ValueError:
        insufficient_data_detected = True

    check(
        "Replay rejects insufficient batch",
        insufficient_data_detected,
    )

    # Test 12.
    for _ in range(BATCH_SIZE - len(replay)):

        replay.add(
            np.zeros(STATE_SIZE, dtype=np.float32),
            np.zeros(ACTION_SIZE, dtype=np.float32),
            1.0,
            np.ones(STATE_SIZE, dtype=np.float32),
            False,
        )

    batch = replay.sample(BATCH_SIZE)

    check(
        "Replay state batch",
        tuple(batch[0].shape) == (
            BATCH_SIZE,
            STATE_SIZE,
        ),
    )

    # Test 13.
    check(
        "Replay action batch",
        tuple(batch[1].shape) == (
            BATCH_SIZE,
            ACTION_SIZE,
        ),
    )

    # Test 14.
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
        "Soft target update",
        torch.allclose(
            target.weight,
            torch.full_like(
                target.weight,
                0.1,
            ),
        ),
    )

    # Test 15.
    agent = SACAgent()

    check(
        "Positive entropy temperature",
        agent.alpha.item() > 0,
    )

    # Test 16.
    transition = env.step(
        np.array(
            [0.25, 0.5],
            dtype=np.float32,
        )
    )

    check(
        "Continuous environment action",
        len(transition) == 4
        and transition[0].shape == (STATE_SIZE,),
    )

    # Test 17.
    check(
        "Finite environment reward",
        np.isfinite(transition[1]),
    )

    # Test 18.
    check(
        "Target has nonzero velocity",
        np.linalg.norm(env.target_velocity) > 0,
    )

    # Test 19.
    old_center = env.target_center.copy()

    changed = env.set_target_center(
        [550.0, 140.0]
    )

    check(
        "Moving target center can be changed",
        changed
        and not np.allclose(
            old_center,
            env.target_center,
        ),
    )

    # Test 20.
    check(
        "Initial spawn is safe",
        safe_position(env.position),
    )

    # Test 21.
    check(
        "Target orbit remains safe",
        safe_position(env.target),
    )

    # Test 22.
    check(
        "Training waits for replay warm-up",
        agent.train_step() is None,
    )

    # Test 23.
    action = agent.select_action(
        env.get_state(),
        deterministic=True,
    )

    check(
        "Deterministic action range",
        action.shape == (ACTION_SIZE,)
        and np.all(np.abs(action) <= 1.0),
    )

    print(
        f"\n{passed}/23 tests passed."
    )


# ============================================================
# STEP 13 — MAIN
# ============================================================

def main():

    print("=" * 75)
    print("PHASE 4.10 — IMPROVED SOFT ACTOR-CRITIC")
    print("=" * 75)

    print(
        "Device:",
        DEVICE,
    )

    print(
        "State size:",
        STATE_SIZE,
    )

    print(
        "Action size:",
        ACTION_SIZE,
    )

    run_tests()

    # --------------------------------------------------------
    # Train.
    # --------------------------------------------------------

    print("\nTraining SAC...")

    agent = SACAgent()

    result = train_agent(
        agent,
        TRAINING_EPISODES,
    )

    # --------------------------------------------------------
    # Evaluate independently.
    # --------------------------------------------------------

    evaluation = evaluate_agent(
        agent,
        episodes=30,
    )

    # --------------------------------------------------------
    # Plot.
    # --------------------------------------------------------

    plot_training(
        agent,
        result,
        evaluation,
    )

    # --------------------------------------------------------
    # Save.
    # --------------------------------------------------------

    torch.save(
        {
            "actor": agent.actor.state_dict(),
            "critic1": agent.critic1.state_dict(),
            "critic2": agent.critic2.state_dict(),
            "log_alpha": agent.log_alpha.detach().cpu(),
            "state_size": STATE_SIZE,
            "action_size": ACTION_SIZE,
        },
        "sac_continuous_control.pth",
    )

    print(
        "\nSaved model: sac_continuous_control.pth"
    )

    # --------------------------------------------------------
    # Interactive simulation.
    # --------------------------------------------------------

    print(
        "\nLaunching SAC quadcopter simulator..."
    )

    gui_env = DroneEnv()

    gui_env.omega = TARGET_OMEGA_END

    gui = DroneGUI(
        gui_env,
        agent,
    )

    gui.run()


if __name__ == "__main__":
    main()
