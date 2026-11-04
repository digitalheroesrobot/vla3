
# ============================================================
# File: phase4/continuous_control_td3.py
#
# PHASE 4.9 — CONTINUOUS CONTROL WITH DDPG AND TD3
# IMPROVED VERSION — OBSTACLE AVOIDANCE
#
# PURPOSE
#   Learn continuous steering and thrust control.
#
# IMPROVEMENTS
#   1. Better obstacle-aware state representation.
#   2. Continuous obstacle proximity penalty.
#   3. Stronger collision and success rewards.
#   4. Repeated obstacle-crossing training scenarios.
#   5. More training episodes.
#   6. TD3 twin critics.
#   7. Delayed actor updates.
#   8. Target policy smoothing.
#   9. Deterministic evaluation.
#  10. Pygame visualization.
#
# STATE:
#   0  target_dx
#   1  target_dy
#   2  target_distance
#   3  sin(target_angle_error)
#   4  cos(target_angle_error)
#   5  velocity_x
#   6  velocity_y
#   7  obstacle_dx
#   8  obstacle_dy
#   9  obstacle_clearance
#   10 sin(obstacle_angle_error)
#   11 cos(obstacle_angle_error)
#
# ACTION:
#   action[0] = steering [-1, 1]
#   action[1] = thrust   [-1, 1]
#
# INSTALL:
#   pip install numpy pygame torch matplotlib
#
# RUN:
#   python phase4/continuous_control_td3.py
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

SAFE_DISTANCE = (
    OBSTACLE_RADIUS
    + ROCKET_RADIUS
    + 22.0
)

# Reward obstacle proximity when clearance
# falls below this distance.
OBSTACLE_WARNING_DISTANCE = 145.0

STATE_SIZE = 12
ACTION_SIZE = 2

GAMMA = 0.99
TAU = 0.005

ACTOR_LR = 1e-4
CRITIC_LR = 3e-4

BUFFER_CAPACITY = 200_000

BATCH_SIZE = 128
WARMUP_STEPS = 1500

TRAINING_EPISODES = 300
MAX_STEPS = 350

POLICY_DELAY = 2

TARGET_NOISE = 0.2
NOISE_CLIP = 0.5

EXPLORATION_NOISE = 0.25

MAX_SPEED = 5.0
MAX_ANGULAR_SPEED = 0.09

DT = 1.0

HIDDEN = 256

SUCCESS_DISTANCE = 22.0

# Every third episode uses the same obstacle-crossing task.
CHALLENGE_INTERVAL = 3

EVALUATION_EPISODES = 20

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# STEP 1 — MATHEMATICAL HELPERS
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


def wrap_angle(angle):

    return (
        angle + math.pi
    ) % (2.0 * math.pi) - math.pi


def soft_update(source, target, tau=TAU):

    with torch.no_grad():

        for source_param, target_param in zip(
            source.parameters(),
            target.parameters(),
        ):

            target_param.data.mul_(
                1.0 - tau
            )

            target_param.data.add_(
                tau * source_param.data
            )


def hard_update(source, target):

    target.load_state_dict(
        source.state_dict()
    )


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

    return (
        inside_world
        and outside_obstacle
    )


class ContinuousRocketEnv:

    def __init__(self):

        self.reset()

    def random_safe_position(self):

        for _ in range(5000):

            position = np.array(
                [
                    random.uniform(35, WORLD_W - 35),
                    random.uniform(35, WORLD_H - 35),
                ],
                dtype=np.float32,
            )

            if safe_position(position):

                return position

        raise RuntimeError(
            "Could not find a safe position."
        )

    def reset(
        self,
        start=None,
        target=None,
        heading=None,
    ):

        if start is None:

            self.position = self.random_safe_position()

        else:

            self.position = np.asarray(
                start,
                dtype=np.float32,
            ).copy()

        if target is None:

            self.target = self.random_safe_position()

            attempts = 0

            while (
                distance(self.position, self.target) < 230
                and attempts < 100
            ):

                self.target = self.random_safe_position()

                attempts += 1

        else:

            self.target = np.asarray(
                target,
                dtype=np.float32,
            ).copy()

        if not safe_position(self.position):

            raise ValueError(
                "Initial rocket position is unsafe."
            )

        if not safe_position(self.target):

            raise ValueError(
                "Target position is unsafe."
            )

        if heading is None:

            self.heading = random.uniform(
                -math.pi,
                math.pi,
            )

        else:

            self.heading = float(heading)

        self.velocity = np.zeros(
            2,
            dtype=np.float32,
        )

        self.steps = 0

        self.total_reward = 0.0

        self.success = False
        self.collision = False

        self.minimum_obstacle_clearance = float("inf")

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

        if distance(
            target,
            self.position,
        ) < 25:

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

    def get_obstacle_clearance(self):

        obstacle_distance = distance(
            self.position,
            OBSTACLE,
        )

        return (
            obstacle_distance
            - SAFE_DISTANCE
        )

    def get_state(self):

        target_delta = (
            self.target - self.position
        )

        obstacle_delta = (
            OBSTACLE - self.position
        )

        target_distance = np.linalg.norm(
            target_delta
        )

        obstacle_distance = np.linalg.norm(
            obstacle_delta
        )

        target_angle = math.atan2(
            target_delta[1],
            target_delta[0],
        )

        obstacle_angle = math.atan2(
            obstacle_delta[1],
            obstacle_delta[0],
        )

        target_angle_error = wrap_angle(
            target_angle - self.heading
        )

        obstacle_angle_error = wrap_angle(
            obstacle_angle - self.heading
        )

        obstacle_clearance = (
            obstacle_distance - SAFE_DISTANCE
        )

        return np.array(
            [
                target_delta[0] / WORLD_W,
                target_delta[1] / WORLD_H,

                target_distance / 900.0,

                math.sin(target_angle_error),
                math.cos(target_angle_error),

                self.velocity[0] / MAX_SPEED,
                self.velocity[1] / MAX_SPEED,

                obstacle_delta[0] / WORLD_W,
                obstacle_delta[1] / WORLD_H,

                obstacle_clearance / 200.0,

                math.sin(obstacle_angle_error),
                math.cos(obstacle_angle_error),
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
        # VEHICLE DYNAMICS
        # ----------------------------------------------------

        self.heading = wrap_angle(
            self.heading
            + steering
            * MAX_ANGULAR_SPEED
            * DT
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

        candidate_position = (
            self.position
            + self.velocity * DT
        )

        reward = 0.0

        terminated = False
        truncated = False

        # ----------------------------------------------------
        # COLLISION CHECK
        # ----------------------------------------------------

        if not safe_position(candidate_position):

            reward -= 100.0

            self.collision = True

            terminated = True

        else:

            self.position = candidate_position

            new_distance = distance(
                self.position,
                self.target,
            )

            # ------------------------------------------------
            # 1. TARGET PROGRESS
            # ------------------------------------------------

            progress = (
                old_distance - new_distance
            )

            reward += (
                0.15 * progress
            )

            # ------------------------------------------------
            # 2. TARGET ALIGNMENT
            # ------------------------------------------------

            target_delta = (
                self.target - self.position
            )

            target_angle = math.atan2(
                target_delta[1],
                target_delta[0],
            )

            angle_error = wrap_angle(
                target_angle - self.heading
            )

            reward += (
                0.08 * math.cos(angle_error)
            )

            # ------------------------------------------------
            # 3. OBSTACLE PROXIMITY PENALTY
            #
            # This is the main obstacle-avoidance change.
            #
            # The rocket receives a penalty before collision.
            # ------------------------------------------------

            clearance = self.get_obstacle_clearance()

            self.minimum_obstacle_clearance = min(
                self.minimum_obstacle_clearance,
                clearance,
            )

            danger = max(
                0.0,
                OBSTACLE_WARNING_DISTANCE - clearance,
            )

            reward -= (
                0.035 * danger
            )

            # ------------------------------------------------
            # 4. TARGET PROXIMITY REWARD
            # ------------------------------------------------

            reward += (
                0.25
                * math.exp(-new_distance / 100.0)
            )

            # ------------------------------------------------
            # 5. SPEED PENALTY
            # ------------------------------------------------

            reward -= (
                0.01
                * np.linalg.norm(self.velocity)
            )

            # ------------------------------------------------
            # 6. ACTION SMOOTHNESS
            # ------------------------------------------------

            reward -= (
                0.025 * steering ** 2
                + 0.008 * thrust ** 2
            )

            # ------------------------------------------------
            # 7. SUCCESS
            # ------------------------------------------------

            if new_distance < SUCCESS_DISTANCE:

                reward += 100.0

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
# STEP 3 — ACTOR NETWORK
# ============================================================

class Actor(nn.Module):

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

                done=float(terminated),
            )
        )

    def sample(self, batch_size):

        if len(self.memory) < batch_size:

            raise ValueError(
                f"Replay buffer has {len(self.memory)} "
                f"transitions; {batch_size} required."
            )

        batch = random.sample(
            list(self.memory),
            batch_size,
        )

        states = np.stack(
            [t.state for t in batch]
        ).astype(np.float32)

        actions = np.stack(
            [t.action for t in batch]
        ).astype(np.float32)

        rewards = np.array(
            [t.reward for t in batch],
            dtype=np.float32,
        ).reshape(-1, 1)

        next_states = np.stack(
            [t.next_state for t in batch]
        ).astype(np.float32)

        dones = np.array(
            [t.done for t in batch],
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
            BUFFER_CAPACITY
        )

        self.losses = []
        self.actor_losses = []

        self.training_steps = 0

    def select_action(
        self,
        state,
        exploration_noise=0.0,
    ):

        state_t = torch.as_tensor(
            state,
            dtype=torch.float32,
            device=DEVICE,
        ).unsqueeze(0)

        self.actor.eval()

        with torch.no_grad():

            action = (
                self.actor(state_t)
                .cpu()
                .numpy()[0]
            )

        self.actor.train()

        if exploration_noise > 0:

            action += np.random.normal(
                0.0,
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
        ) = self.replay.sample(
            BATCH_SIZE
        )

        # ----------------------------------------------------
        # DDPG TARGET
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
                + GAMMA
                * (1.0 - dones)
                * target_q
            )

        # ----------------------------------------------------
        # CRITIC UPDATE
        # ----------------------------------------------------

        current_q = self.critic(
            states,
            actions,
        )

        critic_loss = nn.functional.mse_loss(
            current_q,
            y,
        )

        self.critic_optimizer.zero_grad(
            set_to_none=True
        )

        critic_loss.backward()

        nn.utils.clip_grad_norm_(
            self.critic.parameters(),
            10.0,
        )

        self.critic_optimizer.step()

        # ----------------------------------------------------
        # ACTOR UPDATE
        # ----------------------------------------------------

        actor_loss = -self.critic(
            states,
            self.actor(states),
        ).mean()

        self.actor_optimizer.zero_grad(
            set_to_none=True
        )

        actor_loss.backward()

        nn.utils.clip_grad_norm_(
            self.actor.parameters(),
            10.0,
        )

        self.actor_optimizer.step()

        # ----------------------------------------------------
        # TARGET UPDATES
        # ----------------------------------------------------

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

        self.actor_losses.append(
            float(actor_loss.item())
        )

        return float(critic_loss.item())


# ============================================================
# STEP 7 — TD3 AGENT
# ============================================================

class TD3Agent(DDPGAgent):

    def __init__(self):

        super().__init__()

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

        self.critic2_losses = []

        self.actor_updates = 0

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
        # TARGET POLICY SMOOTHING
        # ----------------------------------------------------

        with torch.no_grad():

            next_actions = self.actor_target(
                next_states
            )

            noise = (
                torch.randn_like(next_actions)
                * TARGET_NOISE
            )

            noise = noise.clamp(
                -NOISE_CLIP,
                NOISE_CLIP,
            )

            next_actions = (
                next_actions + noise
            ).clamp(
                -1.0,
                1.0,
            )

            # ------------------------------------------------
            # TWIN CRITIC MINIMUM
            # ------------------------------------------------

            target_q1 = self.critic_target(
                next_states,
                next_actions,
            )

            target_q2 = self.critic2_target(
                next_states,
                next_actions,
            )

            target_q = torch.minimum(
                target_q1,
                target_q2,
            )

            y = (
                rewards
                + GAMMA
                * (1.0 - dones)
                * target_q
            )

        # ----------------------------------------------------
        # CRITIC 1
        # ----------------------------------------------------

        current_q1 = self.critic(
            states,
            actions,
        )

        loss1 = nn.functional.mse_loss(
            current_q1,
            y,
        )

        self.critic_optimizer.zero_grad(
            set_to_none=True
        )

        loss1.backward()

        nn.utils.clip_grad_norm_(
            self.critic.parameters(),
            10.0,
        )

        self.critic_optimizer.step()

        # ----------------------------------------------------
        # CRITIC 2
        # ----------------------------------------------------

        current_q2 = self.critic2(
            states,
            actions,
        )

        loss2 = nn.functional.mse_loss(
            current_q2,
            y,
        )

        self.critic2_optimizer.zero_grad(
            set_to_none=True
        )

        loss2.backward()

        nn.utils.clip_grad_norm_(
            self.critic2.parameters(),
            10.0,
        )

        self.critic2_optimizer.step()

        self.training_steps += 1

        # ----------------------------------------------------
        # DELAYED ACTOR UPDATE
        # ----------------------------------------------------

        if self.training_steps % POLICY_DELAY == 0:

            actor_loss = -self.critic(
                states,
                self.actor(states),
            ).mean()

            self.actor_optimizer.zero_grad(
                set_to_none=True
            )

            actor_loss.backward()

            nn.utils.clip_grad_norm_(
                self.actor.parameters(),
                10.0,
            )

            self.actor_optimizer.step()

            # Update target networks alongside actor.
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

            self.actor_losses.append(
                float(actor_loss.item())
            )

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
# STEP 8 — TRAINING SCENARIOS
# ============================================================

def reset_training_environment(
    env,
    episode,
):
    """
    Mix random navigation with repeated obstacle-crossing
    scenarios.

    The repeated challenge matches the Pygame demonstration.
    """

    if episode % CHALLENGE_INTERVAL == 0:

        return env.reset(
            start=[80.0, 280.0],
            target=[620.0, 280.0],
            heading=0.0,
        )

    return env.reset()


# ============================================================
# STEP 9 — TRAINING
# ============================================================

def train_agent(
    agent,
    episodes,
    name,
):

    env = ContinuousRocketEnv()

    rewards = []
    successes = []
    collisions = []

    minimum_clearances = []

    started = time.time()

    for episode in range(episodes):

        state = reset_training_environment(
            env,
            episode,
        )

        episode_reward = 0.0

        for step in range(MAX_STEPS):

            # Gradually decrease exploration.
            progress = episode / max(
                episodes - 1,
                1,
            )

            noise = max(
                0.04,
                EXPLORATION_NOISE
                * (1.0 - 0.75 * progress),
            )

            action = agent.select_action(
                state,
                exploration_noise=noise,
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

            agent.train_step()

            state = next_state

            episode_reward += reward

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

        minimum_clearances.append(
            env.minimum_obstacle_clearance
        )

        if (episode + 1) % 20 == 0:

            recent = min(
                20,
                len(rewards),
            )

            print(
                f"{name:8s} "
                f"Episode {episode + 1:3d}/{episodes} | "
                f"Reward {np.mean(rewards[-recent:]):8.2f} | "
                f"Success "
                f"{100*np.mean(successes[-recent:]):5.1f}% | "
                f"Collision "
                f"{100*np.mean(collisions[-recent:]):5.1f}% | "
                f"Clearance "
                f"{np.mean(minimum_clearances[-recent:]):6.1f}"
            )

    elapsed = time.time() - started

    return {
        "agent": agent,
        "rewards": rewards,
        "successes": successes,
        "collisions": collisions,
        "minimum_clearances": minimum_clearances,
        "seconds": elapsed,
    }


# ============================================================
# STEP 10 — EVALUATION
# ============================================================

def evaluate_agent(
    agent,
    episodes=EVALUATION_EPISODES,
):
    """
    Evaluate without exploration noise.

    Uses the same obstacle-crossing scenario as Pygame.
    """

    env = ContinuousRocketEnv()

    results = []

    print("\nTD3 DETERMINISTIC EVALUATION")
    print("=" * 65)

    for episode in range(episodes):

        state = env.reset(
            start=[80.0, 280.0],
            target=[620.0, 280.0],
            heading=0.0,
        )

        episode_reward = 0.0

        for step in range(MAX_STEPS):

            action = agent.select_action(
                state,
                exploration_noise=0.0,
            )

            (
                next_state,
                reward,
                terminated,
                truncated,
            ) = env.step(action)

            state = next_state

            episode_reward += reward

            if terminated or truncated:
                break

        results.append(
            {
                "success": int(env.success),
                "collision": int(env.collision),
                "reward": episode_reward,
                "steps": env.steps,
                "clearance": env.minimum_obstacle_clearance,
            }
        )

        print(
            f"Episode {episode + 1:02d}/{episodes} | "
            f"Success={env.success} | "
            f"Collision={env.collision} | "
            f"Steps={env.steps:3d} | "
            f"Reward={episode_reward:8.2f} | "
            f"Min clearance="
            f"{env.minimum_obstacle_clearance:7.2f}"
        )

    success_rate = np.mean(
        [r["success"] for r in results]
    ) * 100.0

    collision_rate = np.mean(
        [r["collision"] for r in results]
    ) * 100.0

    average_steps = np.mean(
        [r["steps"] for r in results]
    )

    print("\nEVALUATION SUMMARY")
    print("-" * 65)

    print(
        f"Success rate:   {success_rate:.1f}%"
    )

    print(
        f"Collision rate: {collision_rate:.1f}%"
    )

    print(
        f"Average steps:  {average_steps:.1f}"
    )

    return results


# ============================================================
# STEP 11 — TRAINING VISUALIZATION
# ============================================================

def moving_average(
    values,
    window=15,
):

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
    ddpg_result,
    td3_result,
):

    # --------------------------------------------------------
    # Episode reward
    # --------------------------------------------------------

    plt.figure(figsize=(12, 5))

    for name, result in [
        ("DDPG", ddpg_result),
        ("TD3", td3_result),
    ]:

        plt.plot(
            moving_average(
                result["rewards"],
                15,
            ),
            label=name,
        )

    plt.title(
        "DDPG vs TD3 — Episode Reward"
    )

    plt.xlabel("Episode")
    plt.ylabel("Smoothed reward")

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.show()

    # --------------------------------------------------------
    # Success rate
    # --------------------------------------------------------

    plt.figure(figsize=(12, 5))

    for name, result in [
        ("DDPG", ddpg_result),
        ("TD3", td3_result),
    ]:

        success_rate = moving_average(
            result["successes"],
            15,
        )

        plt.plot(
            success_rate * 100.0,
            label=name,
        )

    plt.title(
        "Success Rate"
    )

    plt.xlabel("Episode")
    plt.ylabel("Success rate (%)")

    plt.ylim(0, 100)

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.show()

    # --------------------------------------------------------
    # Collision rate
    # --------------------------------------------------------

    plt.figure(figsize=(12, 5))

    for name, result in [
        ("DDPG", ddpg_result),
        ("TD3", td3_result),
    ]:

        collision_rate = moving_average(
            result["collisions"],
            15,
        )

        plt.plot(
            collision_rate * 100.0,
            label=name,
        )

    plt.title(
        "Collision Rate"
    )

    plt.xlabel("Episode")
    plt.ylabel("Collision rate (%)")

    plt.ylim(0, 100)

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.show()

    # --------------------------------------------------------
    # Critic losses
    # --------------------------------------------------------

    plt.figure(figsize=(12, 5))

    for name, result in [
        ("DDPG", ddpg_result),
        ("TD3", td3_result),
    ]:

        plt.plot(
            result["agent"].losses,
            label=name,
        )

    plt.title(
        "Critic Training Loss"
    )

    plt.xlabel("Training update")
    plt.ylabel("Loss")

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.show()


# ============================================================
# STEP 12 — PYGAME SIMULATOR
# ============================================================

class RocketGUI:

    def __init__(
        self,
        env,
        agent,
        title,
    ):

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

        self.path = deque(maxlen=500)

        self.reset()

    def reset(self):

        self.env.reset(
            start=[80.0, 280.0],
            target=[620.0, 280.0],
            heading=0.0,
        )

        self.last_action[:] = 0.0

        self.path.clear()

        self.path.append(
            self.env.position.copy()
        )

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
                rotate(point)
                for point in body
            ],
        )

        pygame.draw.polygon(
            self.screen,
            (70, 160, 230),
            [
                rotate(point)
                for point in body
            ],
            2,
        )

        pygame.draw.circle(
            self.screen,
            (40, 140, 245),
            rotate((8, 0)),
            4,
        )

    def draw_path(self):

        if len(self.path) < 2:
            return

        points = [
            tuple(
                np.asarray(p).astype(int)
            )
            for p in self.path
        ]

        pygame.draw.lines(
            self.screen,
            (90, 150, 230),
            False,
            points,
            2,
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

        # Grid.
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

        # Warning zone.
        pygame.draw.circle(
            self.screen,
            (115, 75, 55),
            tuple(OBSTACLE.astype(int)),
            int(
                SAFE_DISTANCE
                + OBSTACLE_WARNING_DISTANCE
            ),
            1,
        )

        # Safe boundary.
        pygame.draw.circle(
            self.screen,
            (180, 105, 60),
            tuple(OBSTACLE.astype(int)),
            int(SAFE_DISTANCE),
            2,
        )

        # Obstacle.
        pygame.draw.circle(
            self.screen,
            (165, 55, 55),
            tuple(OBSTACLE.astype(int)),
            int(OBSTACLE_RADIUS),
        )

        # Target.
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

        self.draw_path()

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
            f"Heading: "
            f"{math.degrees(self.env.heading):.1f} deg",
            720,
            160,
        )

        self.draw_text(
            f"Target distance: "
            f"{distance(self.env.position, self.env.target):.1f}",
            720,
            190,
        )

        self.draw_text(
            f"Obstacle clearance: "
            f"{self.env.get_obstacle_clearance():.1f}",
            720,
            220,
        )

        self.draw_text(
            f"Steering: {self.last_action[0]:+.3f}",
            720,
            270,
        )

        self.draw_text(
            f"Thrust:   {self.last_action[1]:+.3f}",
            720,
            300,
        )

        self.draw_text(
            f"Steps: {self.env.steps}",
            720,
            345,
        )

        self.draw_text(
            f"Reward: {self.env.total_reward:.2f}",
            720,
            375,
        )

        self.draw_text(
            f"Speed multiplier: {self.speed}",
            720,
            405,
        )

        if self.env.success:

            status = "SUCCESS"

            color = (80, 255, 130)

        elif self.env.collision:

            status = "COLLISION"

            color = (255, 90, 90)

        else:

            status = "RUNNING"

            color = (220, 220, 220)

        self.draw_text(
            f"Status: {status}",
            720,
            445,
            color,
        )

        self.draw_text(
            "ACTION SPACE",
            720,
            490,
            (120, 210, 255),
        )

        self.draw_text(
            "Steering: [-1, +1]",
            720,
            520,
        )

        self.draw_text(
            "Thrust:   [-1, +1]",
            720,
            545,
        )

        self.draw_text(
            "CLICK: TARGET",
            720,
            580,
        )

        self.draw_text(
            "SPACE: PAUSE | R: RESET",
            720,
            605,
        )

        self.draw_text(
            "ESC: EXIT | +/-: SPEED",
            720,
            628,
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

        self.path.append(
            self.env.position.copy()
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

                                self.path.clear()

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
# STEP 13 — UNIT TESTS
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

    env = ContinuousRocketEnv()

    # 1
    check(
        "State dimension",
        env.get_state().shape == (STATE_SIZE,),
    )

    # 2
    check(
        "Action dimension",
        clamp_action([0.5, -0.5]).shape
        == (ACTION_SIZE,),
    )

    # 3
    check(
        "Action clipping",
        np.allclose(
            clamp_action([2.0, -3.0]),
            [1.0, -1.0],
        ),
    )

    # 4
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

    # 5
    check(
        "Actor output dimensions",
        tuple(actions.shape) == (8, ACTION_SIZE),
    )

    # 6
    critic = Critic().to(DEVICE)

    q = critic(
        states,
        actions,
    )

    check(
        "Critic output dimensions",
        tuple(q.shape) == (8, 1),
    )

    # 7
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
        and bool(torch.isfinite(action_grad).all()),
    )

    # 8
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
            torch.full_like(
                target.weight,
                0.1,
            ),
        ),
    )

    # 9
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

    # 10
    replay_test = ReplayBuffer(
        max(BATCH_SIZE * 2, 256)
    )

    for _ in range(BATCH_SIZE):

        replay_test.add(
            state,
            np.array(
                [0.1, 0.5],
                dtype=np.float32,
            ),
            1.0,
            next_state,
            False,
        )

    sample = replay_test.sample(
        BATCH_SIZE
    )

    check(
        "Replay batch state dimensions",
        tuple(sample[0].shape)
        == (BATCH_SIZE, STATE_SIZE),
    )

    # 11
    check(
        "Replay batch action dimensions",
        tuple(sample[1].shape)
        == (BATCH_SIZE, ACTION_SIZE),
    )

    # 12
    check(
        "Replay batch reward dimensions",
        tuple(sample[2].shape)
        == (BATCH_SIZE, 1),
    )

    # 13
    check(
        "Replay batch next-state dimensions",
        tuple(sample[3].shape)
        == (BATCH_SIZE, STATE_SIZE),
    )

    # 14
    check(
        "Replay batch done dimensions",
        tuple(sample[4].shape)
        == (BATCH_SIZE, 1),
    )

    # 15
    caught_error = False

    try:

        replay.sample(BATCH_SIZE)

    except ValueError:

        caught_error = True

    check(
        "Replay insufficient-sample validation",
        caught_error,
    )

    # 16
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
        "Twin critic dimensions",
        q1.shape == q2.shape == (8, 1),
    )

    # 17
    minimum = torch.minimum(
        torch.tensor([[5.0], [2.0]]),
        torch.tensor([[3.0], [4.0]]),
    )

    check(
        "Twin critic minimum",
        torch.allclose(
            minimum,
            torch.tensor([[3.0], [2.0]]),
        ),
    )

    # 18
    noise = torch.tensor(
        [[2.0, -2.0]]
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

    # 19
    result = env.step(
        np.array(
            [0.0, 0.0],
            dtype=np.float32,
        )
    )

    check(
        "Environment transition tuple",
        len(result) == 4,
    )

    # 20
    check(
        "Environment state remains finite",
        np.isfinite(result[0]).all(),
    )

    # 21
    challenge_env = ContinuousRocketEnv()

    challenge_state = challenge_env.reset(
        start=[80.0, 280.0],
        target=[620.0, 280.0],
        heading=0.0,
    )

    check(
        "Challenge scenario initialization",
        challenge_state.shape == (STATE_SIZE,),
    )

    # 22
    check(
        "Obstacle clearance is positive initially",
        challenge_env.get_obstacle_clearance() > 0,
    )

    # 23
    check(
        "Obstacle clearance is finite",
        np.isfinite(
            challenge_env.get_obstacle_clearance()
        ),
    )

    # 24
    check(
        "Success threshold is positive",
        SUCCESS_DISTANCE > 0,
    )

    print(
        f"\n{passed}/24 tests passed."
    )


# ============================================================
# STEP 14 — MAIN
# ============================================================

def main():

    print("=" * 75)

    print(
        "PHASE 4.9 — IMPROVED CONTINUOUS CONTROL"
    )

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

    print(
        "Training episodes:",
        TRAINING_EPISODES,
    )

    run_tests()

    # --------------------------------------------------------
    # TRAIN DDPG
    # --------------------------------------------------------

    print("\n" + "=" * 65)

    print("TRAINING DDPG")

    print("=" * 65)

    ddpg_agent = DDPGAgent()

    ddpg_result = train_agent(
        ddpg_agent,
        TRAINING_EPISODES,
        "DDPG",
    )

    # --------------------------------------------------------
    # TRAIN TD3
    # --------------------------------------------------------

    print("\n" + "=" * 65)

    print("TRAINING TD3")

    print("=" * 65)

    td3_agent = TD3Agent()

    td3_result = train_agent(
        td3_agent,
        TRAINING_EPISODES,
        "TD3",
    )

    # --------------------------------------------------------
    # TRAINING PLOTS
    # --------------------------------------------------------

    plot_training(
        ddpg_result,
        td3_result,
    )

    # --------------------------------------------------------
    # DETERMINISTIC TD3 EVALUATION
    # --------------------------------------------------------

    evaluation_results = evaluate_agent(
        td3_agent,
        EVALUATION_EPISODES,
    )

    # --------------------------------------------------------
    # SAVE MODELS
    # --------------------------------------------------------

    torch.save(
        ddpg_agent.actor.state_dict(),
        "ddpg_actor.pth",
    )

    torch.save(
        td3_agent.actor.state_dict(),
        "td3_actor.pth",
    )

    torch.save(
        {
            "actor": td3_agent.actor.state_dict(),
            "actor_target": td3_agent.actor_target.state_dict(),

            "critic1": td3_agent.critic.state_dict(),
            "critic2": td3_agent.critic2.state_dict(),

            "critic1_target": td3_agent.critic_target.state_dict(),
            "critic2_target": td3_agent.critic2_target.state_dict(),

            "state_size": STATE_SIZE,
            "action_size": ACTION_SIZE,
        },
        "td3_full_checkpoint.pth",
    )

    print("\nSaved models:")

    print("  ddpg_actor.pth")
    print("  td3_actor.pth")
    print("  td3_full_checkpoint.pth")

    # --------------------------------------------------------
    # PYGAME SIMULATION
    # --------------------------------------------------------

    print(
        "\nStarting improved TD3 Pygame simulation..."
    )

    gui_env = ContinuousRocketEnv()

    gui = RocketGUI(
        gui_env,
        td3_agent,
        "TD3 — Obstacle Avoidance",
    )

    gui.run()


if __name__ == "__main__":

    main()
