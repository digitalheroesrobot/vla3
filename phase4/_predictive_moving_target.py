
"""
RL COURSE — PHASE 4.12
============================================================
Filename:
    phase4/predictive_moving_target.py

Lesson:
    Predictive Moving-Target Tracking

PURPOSE
-------
Learn how to track a moving target by predicting its future
position instead of merely chasing its current position.

WHAT YOU LEARN
--------------
STEP 1 — Relative motion
    Represent the target position and velocity relative to
    the rocket.

STEP 2 — Future-position prediction
    Predict target position using:
        p_future = p_target + v_target * prediction_horizon

STEP 3 — Reactive vs predictive pursuit
    Compare a controller that chases the current target with
    one that aims at the predicted future target position.

STEP 4 — Continuous control
    Use two continuous actions:
        action[0] = horizontal acceleration
        action[1] = vertical acceleration

STEP 5 — TD3
    Train a Twin Delayed Deep Deterministic Policy Gradient
    agent for continuous target tracking.

STEP 6 — Pygame visualization
    Interactively observe the rocket, moving target, predicted
    target position, obstacle, velocity vectors and controller.

STEP 7 — Verification
    Test prediction, state dimensions, action bounds, replay
    buffer, reward behavior and TD3 network outputs.

MATHEMATICAL IDEAS
------------------
Target prediction:
    p_hat(t + H) = p_target(t) + H * v_target(t)

Relative position:
    r = p_target - p_robot

Relative velocity:
    v_rel = v_target - v_robot

Tracking objective:
    Minimize ||p_robot - p_target||.

TD3 objective:
    Use two critics to reduce overestimation:
        y = r + gamma * min(Q1_target, Q2_target)

    Delayed actor update:
        maximize Q1(s, actor(s))

ROBOTICS APPLICATION
--------------------
The same principles are useful for:
    - Mobile robots tracking moving objects
    - Autonomous vehicle target following
    - Drone target interception
    - Robotic pursuit and rendezvous
    - Moving-object manipulation

IMPORTANT
---------
This is a simulation and learning exercise, not a validated
real-world flight controller.

CONTROLS
--------
1       Reactive pursuit
2       Predictive pursuit
3       TD3 learned controller
SPACE   Pause / resume
R       Reset rocket
+ / -   Simulation speed
UP/DOWN Adjust prediction horizon
Mouse   Change moving-target trajectory center
ESC     Exit
============================================================
"""

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


# ============================================================
# 1. CONFIGURATION
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WIDTH = 720
HEIGHT = 580
PANEL_WIDTH = 300

WORLD_WIDTH = WIDTH
WORLD_HEIGHT = HEIGHT

DT = 0.10
ROBOT_RADIUS = 13
MAX_SPEED = 115.0
MAX_ACCELERATION = 100.0

OBSTACLE = np.array([350.0, 285.0], dtype=np.float32)
OBSTACLE_RADIUS = 52.0
SAFETY_MARGIN = 35.0

TARGET_AMPLITUDE = np.array([90.0, 65.0], dtype=np.float32)
TARGET_OMEGA = 0.65

STATE_DIM = 9
ACTION_DIM = 2

GAMMA = 0.99
TAU = 0.005
ACTOR_LR = 1e-4
CRITIC_LR = 3e-4

BUFFER_CAPACITY = 100_000
BATCH_SIZE = 128
WARMUP_STEPS = 500
POLICY_DELAY = 2
POLICY_NOISE = 0.15
NOISE_CLIP = 0.30

TRAIN_EPISODES = 120
MAX_STEPS_PER_EPISODE = 450

PREDICTION_HORIZON = 0.75


# ============================================================
# 2. COMMON UTILITIES
# ============================================================

def clamp_vector(vector, max_norm):
    vector = np.asarray(vector, dtype=np.float32)
    norm = np.linalg.norm(vector)

    if norm > max_norm:
        vector = vector * (max_norm / (norm + 1e-8))

    return vector


def normalize_vector(vector, scale):
    return np.asarray(vector, dtype=np.float32) / float(scale)


def safe_unit(vector):
    vector = np.asarray(vector, dtype=np.float32)
    norm = np.linalg.norm(vector)

    if norm < 1e-8:
        return np.zeros_like(vector)

    return vector / norm


def distance(a, b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


# ============================================================
# 3. MOVING TARGET MODEL
# ============================================================

class MovingTarget:

    def __init__(self, center=None):
        self.center = np.array(
            center if center is not None else [550.0, 145.0],
            dtype=np.float32
        )

        self.phase = 0.0

    def position_at(self, phase):
        return self.center + TARGET_AMPLITUDE * np.array(
            [
                math.sin(phase),
                math.sin(phase * 1.35 + 0.7)
            ],
            dtype=np.float32
        )

    def velocity_at(self, phase):
        return TARGET_AMPLITUDE * np.array(
            [
                TARGET_OMEGA * math.cos(phase),
                TARGET_OMEGA * 1.35 *
                math.cos(phase * 1.35 + 0.7)
            ],
            dtype=np.float32
        )

    @property
    def position(self):
        return self.position_at(self.phase)

    @property
    def velocity(self):
        return self.velocity_at(self.phase)

    def step(self):
        self.phase += TARGET_OMEGA * DT

    def set_center(self, position):
        self.center = np.asarray(position, dtype=np.float32)


def predict_target_position(position, velocity, horizon):
    """
    Constant-velocity prediction.

    p_future = p_current + velocity * horizon

    This is a first-order prediction. It works best when target
    velocity does not change dramatically during the horizon.
    """

    return (
        np.asarray(position, dtype=np.float32)
        + np.asarray(velocity, dtype=np.float32) * horizon
    )


# ============================================================
# 4. ROBOT ENVIRONMENT
# ============================================================

class MovingTargetEnv:

    def __init__(self):
        self.target = MovingTarget()
        self.max_steps = MAX_STEPS_PER_EPISODE
        self.step_count = 0

        self.position = np.array([100.0, 440.0], dtype=np.float32)
        self.velocity = np.zeros(2, dtype=np.float32)

        self.previous_distance = 0.0

    def reset(self):
        self.position = np.array([100.0, 440.0], dtype=np.float32)
        self.velocity = np.zeros(2, dtype=np.float32)

        self.target.phase = random.uniform(0.0, 2.0 * math.pi)

        self.step_count = 0
        self.previous_distance = distance(
            self.position,
            self.target.position
        )

        return self.get_state()

    def get_state(self):
        """
        Nine-dimensional observation:

        0,1: target relative position
        2,3: target velocity
        4,5: robot velocity
        6,7: obstacle relative position
        8:   target distance

        Relative position and velocity allow the agent to
        reason about where the target is and how it is moving.
        """

        relative_position = self.target.position - self.position
        relative_velocity = self.target.velocity - self.velocity
        obstacle_relative = OBSTACLE - self.position

        state = np.concatenate(
            [
                normalize_vector(relative_position, 600.0),
                normalize_vector(self.target.velocity, 150.0),
                normalize_vector(self.velocity, MAX_SPEED),
                normalize_vector(obstacle_relative, 600.0),
                np.array(
                    [distance(self.position, self.target.position) / 600.0],
                    dtype=np.float32
                )
            ]
        )

        return state.astype(np.float32)

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)

        old_position = self.position.copy()

        # Move target first, then compare robot progress against
        # the target at the same new time instant.
        next_phase = self.target.phase + TARGET_OMEGA * DT

        target_at_next_time = self.target.position_at(next_phase)

        distance_before_robot_move = distance(
            self.position,
            target_at_next_time
        )

        acceleration = action * MAX_ACCELERATION

        self.velocity += acceleration * DT
        self.velocity = clamp_vector(self.velocity, MAX_SPEED)

        self.position += self.velocity * DT

        self.target.step()

        distance_after_robot_move = distance(
            self.position,
            self.target.position
        )

        progress = (
            distance_before_robot_move -
            distance_after_robot_move
        )

        # Reward progress toward the moving target.
        reward = 0.08 * progress

        # Encourage close tracking.
        reward += 0.3 * math.exp(
            -distance_after_robot_move / 75.0
        )

        # Penalize unnecessary acceleration.
        reward -= 0.015 * float(np.sum(action ** 2))

        # Collision detection.
        obstacle_distance = distance(
            self.position,
            OBSTACLE
        )

        terminated = False

        if obstacle_distance < (
            OBSTACLE_RADIUS + ROBOT_RADIUS
        ):
            reward -= 100.0
            terminated = True

        # Keep the rocket inside the world.
        if (
            self.position[0] < ROBOT_RADIUS
            or self.position[0] > WORLD_WIDTH - ROBOT_RADIUS
            or self.position[1] < ROBOT_RADIUS
            or self.position[1] > WORLD_HEIGHT - ROBOT_RADIUS
        ):
            reward -= 100.0
            terminated = True

        self.step_count += 1

        truncated = self.step_count >= self.max_steps

        # A truncation is not a true terminal state.
        next_state = self.get_state()

        info = {
            "distance": distance_after_robot_move,
            "progress": progress,
            "collision": obstacle_distance < (
                OBSTACLE_RADIUS + ROBOT_RADIUS
            )
        }

        return next_state, float(reward), terminated, truncated, info


# ============================================================
# 5. ACTION SAFETY FILTER
# ============================================================

def safety_filter(position, velocity, action):
    """
    Simple reactive obstacle safety filter.

    It removes the acceleration component that pushes the rocket
    toward the obstacle when the rocket is inside the warning zone.

    This is a simplified simulation safety measure, not a formal
    control-barrier-function guarantee.
    """

    action = np.asarray(action, dtype=np.float32).copy()

    relative = position - OBSTACLE
    obstacle_distance = np.linalg.norm(relative)

    warning_distance = (
        OBSTACLE_RADIUS + ROBOT_RADIUS + SAFETY_MARGIN
    )

    if obstacle_distance < warning_distance:
        outward = safe_unit(relative)

        acceleration = action * MAX_ACCELERATION
        inward_component = float(np.dot(acceleration, outward))

        # Negative outward component means acceleration toward obstacle.
        if inward_component < 0.0:
            acceleration -= inward_component * outward
            action = acceleration / MAX_ACCELERATION

    return np.clip(action, -1.0, 1.0)


# ============================================================
# 6. REACTIVE AND PREDICTIVE CONTROLLERS
# ============================================================

def compute_pursuit_action(
    robot_position,
    robot_velocity,
    target_position,
    target_velocity,
    prediction_horizon,
    predictive=True
):
    """
    Reactive controller:
        aim at the current target position.

    Predictive controller:
        aim at the estimated future target position.

    A velocity-damping term reduces overshoot and oscillation.
    """

    if predictive:
        aim_position = predict_target_position(
            target_position,
            target_velocity,
            prediction_horizon
        )
    else:
        aim_position = np.asarray(
            target_position,
            dtype=np.float32
        )

    desired_direction = aim_position - robot_position

    desired_velocity = (
        safe_unit(desired_direction) * MAX_SPEED * 0.75
    )

    # Proportional velocity controller.
    acceleration = (
        2.0 * (desired_velocity - robot_velocity)
    )

    acceleration = clamp_vector(
        acceleration,
        MAX_ACCELERATION
    )

    return acceleration / MAX_ACCELERATION, aim_position


# ============================================================
# 7. TD3 REPLAY BUFFER
# ============================================================

class ReplayBuffer:

    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append(
            (
                np.asarray(state, dtype=np.float32),
                np.asarray(action, dtype=np.float32),
                float(reward),
                np.asarray(next_state, dtype=np.float32),
                float(done)
            )
        )

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.tensor(np.array(states), dtype=torch.float32, device=DEVICE),
            torch.tensor(np.array(actions), dtype=torch.float32, device=DEVICE),
            torch.tensor(np.array(rewards), dtype=torch.float32, device=DEVICE).unsqueeze(1),
            torch.tensor(np.array(next_states), dtype=torch.float32, device=DEVICE),
            torch.tensor(np.array(dones), dtype=torch.float32, device=DEVICE).unsqueeze(1)
        )

    def __len__(self):
        return len(self.buffer)


# ============================================================
# 8. TD3 NETWORKS
# ============================================================

class Actor(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(STATE_DIM, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, ACTION_DIM),
            nn.Tanh()
        )

    def forward(self, state):
        return self.network(state)


class Critic(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(STATE_DIM + ACTION_DIM, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, action):
        return self.network(
            torch.cat([state, action], dim=1)
        )


# ============================================================
# 9. TD3 AGENT
# ============================================================

class TD3Agent:

    def __init__(self):
        self.actor = Actor().to(DEVICE)
        self.actor_target = Actor().to(DEVICE)

        self.critic1 = Critic().to(DEVICE)
        self.critic2 = Critic().to(DEVICE)

        self.critic1_target = Critic().to(DEVICE)
        self.critic2_target = Critic().to(DEVICE)

        self.actor_target.load_state_dict(
            self.actor.state_dict()
        )

        self.critic1_target.load_state_dict(
            self.critic1.state_dict()
        )

        self.critic2_target.load_state_dict(
            self.critic2.state_dict()
        )

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=ACTOR_LR
        )

        self.critic_optimizer = torch.optim.Adam(
            list(self.critic1.parameters()) +
            list(self.critic2.parameters()),
            lr=CRITIC_LR
        )

        self.training_steps = 0

    def select_action(self, state, noise=0.0):
        state_tensor = torch.tensor(
            state,
            dtype=torch.float32,
            device=DEVICE
        ).unsqueeze(0)

        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy()[0]

        if noise > 0:
            action += np.random.normal(
                0.0,
                noise,
                size=ACTION_DIM
            )

        return np.clip(action, -1.0, 1.0).astype(np.float32)

    def soft_update(self, source, target):
        for source_parameter, target_parameter in zip(
            source.parameters(),
            target.parameters()
        ):
            target_parameter.data.copy_(
                TAU * source_parameter.data
                + (1.0 - TAU) * target_parameter.data
            )

    def train_step(self, replay):
        if len(replay) < BATCH_SIZE:
            return None

        (
            states,
            actions,
            rewards,
            next_states,
            dones
        ) = replay.sample(BATCH_SIZE)

        with torch.no_grad():

            next_actions = self.actor_target(next_states)

            noise = torch.randn_like(next_actions) * POLICY_NOISE
            noise = noise.clamp(-NOISE_CLIP, NOISE_CLIP)

            next_actions = (
                next_actions + noise
            ).clamp(-1.0, 1.0)

            target_q1 = self.critic1_target(
                next_states,
                next_actions
            )

            target_q2 = self.critic2_target(
                next_states,
                next_actions
            )

            target_q = torch.minimum(
                target_q1,
                target_q2
            )

            target = rewards + GAMMA * (1.0 - dones) * target_q

        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)

        critic_loss = (
            F.mse_loss(current_q1, target)
            + F.mse_loss(current_q2, target)
        )

        self.critic_optimizer.zero_grad()
        critic_loss.backward()

        nn.utils.clip_grad_norm_(
            list(self.critic1.parameters()) +
            list(self.critic2.parameters()),
            1.0
        )

        self.critic_optimizer.step()

        self.training_steps += 1

        actor_loss_value = None

        # TD3 delays actor and target-network updates.
        if self.training_steps % POLICY_DELAY == 0:

            actor_loss = -self.critic1(
                states,
                self.actor(states)
            ).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()

            nn.utils.clip_grad_norm_(
                self.actor.parameters(),
                1.0
            )

            self.actor_optimizer.step()

            self.soft_update(
                self.actor,
                self.actor_target
            )

            self.soft_update(
                self.critic1,
                self.critic1_target
            )

            self.soft_update(
                self.critic2,
                self.critic2_target
            )

            actor_loss_value = float(actor_loss.item())

        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": actor_loss_value
        }


# ============================================================
# 10. TRAIN TD3
# ============================================================

def train_td3():
    print("\nTraining TD3 moving-target tracking agent...")
    print(f"Device: {DEVICE}")

    env = MovingTargetEnv()
    agent = TD3Agent()
    replay = ReplayBuffer(BUFFER_CAPACITY)

    episode_rewards = []
    episode_distances = []

    total_steps = 0

    for episode in range(TRAIN_EPISODES):

        state = env.reset()

        total_reward = 0.0
        distances = []

        for step in range(MAX_STEPS_PER_EPISODE):

            if total_steps < WARMUP_STEPS:
                action = np.random.uniform(
                    -1.0,
                    1.0,
                    size=ACTION_DIM
                ).astype(np.float32)
            else:
                exploration = max(
                    0.05,
                    0.25 * (1.0 - episode / TRAIN_EPISODES)
                )

                action = agent.select_action(
                    state,
                    noise=exploration
                )

            action = safety_filter(
                env.position,
                env.velocity,
                action
            )

            next_state, reward, terminated, truncated, info = env.step(
                action
            )

            # True terminal states stop bootstrapping.
            # Time-limit truncation does not.
            replay.add(
                state,
                action,
                reward,
                next_state,
                terminated
            )

            if total_steps >= WARMUP_STEPS:
                agent.train_step(replay)

            state = next_state
            total_reward += reward
            distances.append(info["distance"])

            total_steps += 1

            if terminated or truncated:
                break

        episode_rewards.append(total_reward)
        episode_distances.append(float(np.mean(distances)))

        if (episode + 1) % 10 == 0:
            recent_reward = np.mean(episode_rewards[-10:])
            recent_distance = np.mean(episode_distances[-10:])

            print(
                f"Episode {episode + 1:3d}/{TRAIN_EPISODES} | "
                f"Reward: {recent_reward:8.2f} | "
                f"Mean distance: {recent_distance:7.2f} | "
                f"Replay: {len(replay)}"
            )

    return agent, episode_rewards, episode_distances


# ============================================================
# 11. PYGAME VISUALIZATION
# ============================================================

class RocketGUI:

    def __init__(self, agent, rewards, distances):
        pygame.init()

        self.screen = pygame.display.set_mode(
            (WIDTH + PANEL_WIDTH, HEIGHT)
        )

        pygame.display.set_caption(
            "Phase 4.12 — Predictive Moving-Target Tracking"
        )

        self.clock = pygame.time.Clock()

        self.font = pygame.font.SysFont("consolas", 16)
        self.small_font = pygame.font.SysFont("consolas", 13)
        self.title_font = pygame.font.SysFont("consolas", 20, bold=True)

        self.agent = agent
        self.rewards = rewards
        self.distances = distances

        self.env = MovingTargetEnv()
        self.env.reset()

        self.mode = 2
        self.paused = False
        self.speed = 1
        self.horizon = PREDICTION_HORIZON

        self.last_action = np.zeros(2, dtype=np.float32)
        self.aim_position = self.env.target.position.copy()

        self.running = True

    def reset(self):
        self.env.reset()

        self.last_action[:] = 0.0
        self.aim_position = self.env.target.position.copy()

    def select_action(self):
        target_position = self.env.target.position
        target_velocity = self.env.target.velocity

        if self.mode == 1:
            action, aim = compute_pursuit_action(
                self.env.position,
                self.env.velocity,
                target_position,
                target_velocity,
                self.horizon,
                predictive=False
            )

        elif self.mode == 2:
            action, aim = compute_pursuit_action(
                self.env.position,
                self.env.velocity,
                target_position,
                target_velocity,
                self.horizon,
                predictive=True
            )

        else:
            state = self.env.get_state()

            action = self.agent.select_action(
                state,
                noise=0.0
            )

            # Display predicted target position for comparison.
            aim = predict_target_position(
                target_position,
                target_velocity,
                self.horizon
            )

        action = safety_filter(
            self.env.position,
            self.env.velocity,
            action
        )

        self.aim_position = aim
        self.last_action = action

        return action

    def change_target_center(self, position):
        x, y = position

        # Keep the trajectory center away from the screen boundary
        # and the obstacle. The complete trajectory is also checked.
        if not (
            150 < x < WIDTH - 130
            and 100 < y < HEIGHT - 100
        ):
            return

        candidate_center = np.array(
            [x, y],
            dtype=np.float32
        )

        # Conservative trajectory-clearance test.
        minimum_clearance = (
            np.linalg.norm(candidate_center - OBSTACLE)
            - np.linalg.norm(TARGET_AMPLITUDE)
        )

        if minimum_clearance < (
            OBSTACLE_RADIUS + ROBOT_RADIUS + 25
        ):
            return

        self.env.target.set_center(candidate_center)

    def draw_arrow(self, start, end, color, width=2):
        start = tuple(np.asarray(start, dtype=int))
        end = tuple(np.asarray(end, dtype=int))

        pygame.draw.line(
            self.screen,
            color,
            start,
            end,
            width
        )

        direction = safe_unit(
            np.array(end, dtype=np.float32)
            - np.array(start, dtype=np.float32)
        )

        if np.linalg.norm(direction) < 1e-6:
            return

        perpendicular = np.array(
            [-direction[1], direction[0]]
        )

        tip = np.array(end, dtype=np.float32)

        p1 = tip - direction * 11 + perpendicular * 5
        p2 = tip - direction * 11 - perpendicular * 5

        pygame.draw.polygon(
            self.screen,
            color,
            [end, p1, p2]
        )

    def draw_rocket(self):
        position = self.env.position
        velocity = self.env.velocity

        direction = safe_unit(velocity)

        if np.linalg.norm(direction) < 0.1:
            direction = safe_unit(
                self.aim_position - position
            )

        if np.linalg.norm(direction) < 0.1:
            direction = np.array([1.0, 0.0])

        perpendicular = np.array(
            [-direction[1], direction[0]]
        )

        nose = position + direction * 22
        rear = position - direction * 15

        left_fin = rear + perpendicular * 13
        right_fin = rear - perpendicular * 13

        body_left = position + perpendicular * 8
        body_right = position - perpendicular * 8

        # Rocket exhaust.
        exhaust_length = random.randint(10, 19)

        exhaust_tip = rear - direction * exhaust_length

        pygame.draw.line(
            self.screen,
            (255, 130, 30),
            tuple(rear.astype(int)),
            tuple(exhaust_tip.astype(int)),
            5
        )

        pygame.draw.polygon(
            self.screen,
            (200, 210, 225),
            [
                nose.astype(int),
                body_left.astype(int),
                left_fin.astype(int),
                rear.astype(int),
                right_fin.astype(int),
                body_right.astype(int)
            ]
        )

        pygame.draw.polygon(
            self.screen,
            (60, 130, 220),
            [
                body_left.astype(int),
                left_fin.astype(int),
                rear.astype(int)
            ]
        )

        pygame.draw.polygon(
            self.screen,
            (60, 130, 220),
            [
                body_right.astype(int),
                right_fin.astype(int),
                rear.astype(int)
            ]
        )

        pygame.draw.circle(
            self.screen,
            (30, 45, 65),
            tuple(position.astype(int)),
            4
        )

    def draw_target_path(self):
        points = []

        for i in range(100):
            phase = self.env.target.phase + i * 0.08

            point = self.env.target.position_at(phase)
            points.append(tuple(point.astype(int)))

        if len(points) > 1:
            pygame.draw.lines(
                self.screen,
                (65, 95, 75),
                False,
                points,
                1
            )

    def draw_world(self):
        self.screen.fill((15, 20, 30))

        # Grid.
        for x in range(0, WIDTH, 40):
            pygame.draw.line(
                self.screen,
                (28, 35, 48),
                (x, 0),
                (x, HEIGHT)
            )

        for y in range(0, HEIGHT, 40):
            pygame.draw.line(
                self.screen,
                (28, 35, 48),
                (0, y),
                (WIDTH, y)
            )

        self.draw_target_path()

        # Obstacle and safety boundary.
        pygame.draw.circle(
            self.screen,
            (110, 55, 55),
            OBSTACLE.astype(int),
            int(OBSTACLE_RADIUS + SAFETY_MARGIN),
            2
        )

        pygame.draw.circle(
            self.screen,
            (170, 65, 65),
            OBSTACLE.astype(int),
            int(OBSTACLE_RADIUS)
        )

        pygame.draw.circle(
            self.screen,
            (220, 115, 105),
            OBSTACLE.astype(int),
            8
        )

        # Predicted future target position.
        pygame.draw.circle(
            self.screen,
            (70, 150, 230),
            self.aim_position.astype(int),
            9,
            2
        )

        # Current target.
        target = self.env.target.position

        pygame.draw.circle(
            self.screen,
            (70, 230, 120),
            target.astype(int),
            10
        )

        pygame.draw.circle(
            self.screen,
            (180, 255, 190),
            target.astype(int),
            18,
            1
        )

        # Target velocity vector.
        target_velocity_end = (
            target + self.env.target.velocity * 0.35
        )

        self.draw_arrow(
            target,
            target_velocity_end,
            (100, 240, 130),
            2
        )

        # Rocket velocity vector.
        self.draw_arrow(
            self.env.position,
            self.env.position + self.env.velocity * 0.35,
            (255, 210, 90),
            2
        )

        self.draw_rocket()

        pygame.draw.rect(
            self.screen,
            (40, 45, 60),
            (0, 0, WIDTH, HEIGHT),
            2
        )

    def draw_panel(self):
        x = WIDTH

        pygame.draw.rect(
            self.screen,
            (23, 28, 40),
            (x, 0, PANEL_WIDTH, HEIGHT)
        )

        pygame.draw.line(
            self.screen,
            (70, 80, 100),
            (x, 0),
            (x, HEIGHT),
            2
        )

        lines = [
            ("PHASE 4.12", (255, 220, 100)),
            ("Predictive Tracking", (220, 230, 245)),
            ("", (200, 200, 200)),
            ("CONTROLLER", (150, 190, 255)),
            ("1 Reactive pursuit", (220, 220, 220)),
            ("2 Predictive pursuit", (220, 220, 220)),
            ("3 TD3 learned", (220, 220, 220)),
            ("", (200, 200, 200)),
            (f"Active mode: {self.mode}", (100, 240, 150)),
            ("", (200, 200, 200)),
            ("TARGET TRACKING", (150, 190, 255)),
            (
                f"Distance: {distance(self.env.position, self.env.target.position):.1f}",
                (230, 230, 230)
            ),
            (
                f"Target vx: {self.env.target.velocity[0]:.1f}",
                (180, 240, 180)
            ),
            (
                f"Target vy: {self.env.target.velocity[1]:.1f}",
                (180, 240, 180)
            ),
            (
                f"Prediction H: {self.horizon:.2f}s",
                (120, 190, 255)
            ),
            ("", (200, 200, 200)),
            ("ROCKET", (150, 190, 255)),
            (
                f"Velocity: {np.linalg.norm(self.env.velocity):.1f}",
                (230, 230, 230)
            ),
            (
                f"Action X: {self.last_action[0]:+.2f}",
                (230, 230, 230)
            ),
            (
                f"Action Y: {self.last_action[1]:+.2f}",
                (230, 230, 230)
            ),
            ("", (200, 200, 200)),
            ("SIMULATION", (150, 190, 255)),
            (
                f"Speed: {self.speed}x",
                (230, 230, 230)
            ),
            (
                f"Steps: {self.env.step_count}",
                (230, 230, 230)
            ),
            (
                "PAUSED" if self.paused else "RUNNING",
                (255, 190, 100)
            ),
            ("", (200, 200, 200)),
            ("CONTROLS", (150, 190, 255)),
            ("Mouse: move target center", (190, 190, 190)),
            ("SPACE: pause / resume", (190, 190, 190)),
            ("R: reset", (190, 190, 190)),
            ("UP/DOWN: prediction time", (190, 190, 190)),
            ("+/-: simulation speed", (190, 190, 190)),
            ("ESC: exit", (190, 190, 190))
        ]

        y = 15

        for text, color in lines:
            if text:
                surface = self.small_font.render(
                    text,
                    True,
                    color
                )

                self.screen.blit(surface, (x + 15, y))

            y += 17

    def handle_events(self):
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

                elif event.key == pygame.K_1:
                    self.mode = 1

                elif event.key == pygame.K_2:
                    self.mode = 2

                elif event.key == pygame.K_3:
                    self.mode = 3

                elif event.key in (
                    pygame.K_PLUS,
                    pygame.K_EQUALS,
                    pygame.K_KP_PLUS
                ):
                    self.speed = min(5, self.speed + 1)

                elif event.key in (
                    pygame.K_MINUS,
                    pygame.K_KP_MINUS
                ):
                    self.speed = max(1, self.speed - 1)

                elif event.key == pygame.K_UP:
                    self.horizon = min(
                        2.0,
                        self.horizon + 0.1
                    )

                elif event.key == pygame.K_DOWN:
                    self.horizon = max(
                        0.0,
                        self.horizon - 0.1
                    )

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    if event.pos[0] < WIDTH:
                        self.change_target_center(event.pos)

    def run(self):
        while self.running:
            self.handle_events()

            if not self.paused:
                for _ in range(self.speed):
                    action = self.select_action()

                    _, _, terminated, truncated, _ = self.env.step(
                        action
                    )

                    if terminated or truncated:
                        self.reset()
                        break

            self.draw_world()
            self.draw_panel()

            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()


# ============================================================
# 12. VISUALIZATION OF TRAINING RESULTS
# ============================================================

def visualize_training(rewards, distances):
    """
    Plot training reward and average tracking distance.

    Matplotlib is optional. The Pygame simulation remains
    available if Matplotlib is not installed.
    """

    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(10, 7))

        axes[0].plot(rewards, alpha=0.35, label="Episode reward")

        if len(rewards) >= 10:
            moving_average = np.convolve(
                rewards,
                np.ones(10) / 10,
                mode="valid"
            )

            axes[0].plot(
                np.arange(9, len(rewards)),
                moving_average,
                label="10-episode average"
            )

        axes[0].set_title("TD3 Training Reward")
        axes[0].set_xlabel("Episode")
        axes[0].set_ylabel("Reward")
        axes[0].legend()
        axes[0].grid(True)

        axes[1].plot(
            distances,
            label="Mean tracking distance"
        )

        axes[1].set_title("Tracking Distance During Training")
        axes[1].set_xlabel("Episode")
        axes[1].set_ylabel("Distance")
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        plt.show()

    except ImportError:
        print("Matplotlib not installed; skipping training plots.")


# ============================================================
# 13. UNIT TESTS
# ============================================================

def run_tests():
    print("\nRunning Phase 4.12 tests...")

    # Test 1: Target position is two-dimensional.
    target = MovingTarget()
    assert target.position.shape == (2,)
    print("Test 1 passed: target position dimension")

    # Test 2: Target velocity is two-dimensional.
    assert target.velocity.shape == (2,)
    print("Test 2 passed: target velocity dimension")

    # Test 3: Constant-velocity prediction.
    predicted = predict_target_position(
        np.array([10.0, 20.0]),
        np.array([2.0, -4.0]),
        0.5
    )

    assert np.allclose(predicted, [11.0, 18.0])
    print("Test 3 passed: target prediction")

    # Test 4: Zero-horizon prediction.
    predicted = predict_target_position(
        np.array([10.0, 20.0]),
        np.array([2.0, -4.0]),
        0.0
    )

    assert np.allclose(predicted, [10.0, 20.0])
    print("Test 4 passed: zero prediction horizon")

    # Test 5: Reactive pursuit aims at current target.
    action, aim = compute_pursuit_action(
        np.array([100.0, 100.0]),
        np.zeros(2),
        np.array([200.0, 100.0]),
        np.array([0.0, 20.0]),
        1.0,
        predictive=False
    )

    assert np.allclose(aim, [200.0, 100.0])
    print("Test 5 passed: reactive pursuit")

    # Test 6: Predictive pursuit aims ahead.
    action, aim = compute_pursuit_action(
        np.array([100.0, 100.0]),
        np.zeros(2),
        np.array([200.0, 100.0]),
        np.array([0.0, 20.0]),
        1.0,
        predictive=True
    )

    assert np.allclose(aim, [200.0, 120.0])
    print("Test 6 passed: predictive pursuit")

    # Test 7: Controller action is bounded.
    assert np.all(action >= -1.0)
    assert np.all(action <= 1.0)
    print("Test 7 passed: action bounds")

    # Test 8: Environment state dimension.
    env = MovingTargetEnv()
    state = env.reset()

    assert state.shape == (STATE_DIM,)
    print("Test 8 passed: state dimension")

    # Test 9: Environment action shape and return values.
    result = env.step(np.zeros(ACTION_DIM))

    assert len(result) == 5
    assert result[0].shape == (STATE_DIM,)
    print("Test 9 passed: environment step")

    # Test 10: Environment reward is finite.
    assert np.isfinite(result[1])
    print("Test 10 passed: finite reward")

    # Test 11: Replay buffer stores transitions.
    replay = ReplayBuffer(100)

    replay.add(
        np.zeros(STATE_DIM),
        np.zeros(ACTION_DIM),
        1.0,
        np.ones(STATE_DIM),
        False
    )

    assert len(replay) == 1
    print("Test 11 passed: replay buffer")

    # Test 12: Replay batch shapes.
    for _ in range(BATCH_SIZE):
        replay.add(
            np.zeros(STATE_DIM),
            np.zeros(ACTION_DIM),
            1.0,
            np.ones(STATE_DIM),
            False
        )

    batch = replay.sample(BATCH_SIZE)

    assert batch[0].shape == (BATCH_SIZE, STATE_DIM)
    assert batch[1].shape == (BATCH_SIZE, ACTION_DIM)
    assert batch[2].shape == (BATCH_SIZE, 1)
    assert batch[3].shape == (BATCH_SIZE, STATE_DIM)
    assert batch[4].shape == (BATCH_SIZE, 1)

    print("Test 12 passed: replay batch dimensions")

    # Test 13: Actor output shape and range.
    actor = Actor().to(DEVICE)

    sample_state = torch.zeros(
        (4, STATE_DIM),
        device=DEVICE
    )

    with torch.no_grad():
        actor_output = actor(sample_state)

    assert actor_output.shape == (4, ACTION_DIM)
    assert torch.all(actor_output <= 1.0)
    assert torch.all(actor_output >= -1.0)

    print("Test 13 passed: actor output")

    # Test 14: Critic output shape.
    critic = Critic().to(DEVICE)

    with torch.no_grad():
        q_value = critic(
            sample_state,
            torch.zeros((4, ACTION_DIM), device=DEVICE)
        )

    assert q_value.shape == (4, 1)
    print("Test 14 passed: critic output")

    # Test 15: TD3 agent action.
    agent = TD3Agent()

    action = agent.select_action(
        np.zeros(STATE_DIM, dtype=np.float32)
    )

    assert action.shape == (ACTION_DIM,)
    assert np.all(action >= -1.0)
    assert np.all(action <= 1.0)

    print("Test 15 passed: TD3 action")

    # Test 16: Safety filter preserves action bounds.
    filtered = safety_filter(
        np.array([300.0, 285.0]),
        np.zeros(2),
        np.array([-1.0, 0.5])
    )

    assert np.all(filtered >= -1.0)
    assert np.all(filtered <= 1.0)

    print("Test 16 passed: safety filter")

    # Test 17: Target center can be changed.
    target = MovingTarget()
    target.set_center([500.0, 150.0])

    assert np.allclose(target.center, [500.0, 150.0])
    print("Test 17 passed: target center update")

    print("\nAll 17 tests passed.\n")


# ============================================================
# 14. MAIN
# ============================================================

def main():
    print("=" * 60)
    print("PHASE 4.12 — PREDICTIVE MOVING-TARGET TRACKING")
    print("=" * 60)

    run_tests()

    agent, rewards, distances = train_td3()

    visualize_training(rewards, distances)

    print("\nStarting interactive Pygame simulation...")
    print("Select controller 1, 2 or 3 to compare performance.")

    gui = RocketGUI(
        agent,
        rewards,
        distances
    )

    gui.run()


if __name__ == "__main__":
    main()
