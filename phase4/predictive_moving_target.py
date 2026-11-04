
"""
RL COURSE — PHASE 4.12
============================================================
Filename:
    _predictive_moving_target.py

Lesson:
    Improved Predictive Moving-Target Tracking with TD3

IMPROVEMENTS
------------
1. State includes relative target velocity.
2. Reward encourages both proximity and velocity matching.
3. Stronger and more useful distance-based reward shaping.
4. Longer training: 250 episodes x 500 steps.
5. Longer exploration warm-up.
6. More stable TD3 actor updates.
7. Evaluation of reactive, predictive and TD3 controllers.
8. Improved Pygame arrow rendering.

CONTROLS
--------
1       Reactive pursuit
2       Predictive pursuit
3       TD3 learned controller
SPACE   Pause / resume
R       Reset rocket
+ / -   Simulation speed
UP/DOWN Adjust prediction horizon
Mouse   Change target trajectory center
ESC     Exit

IMPORTANT
---------
This is a simulation and learning exercise, not a validated
real-world flight controller.
============================================================
"""

from __future__ import annotations

import math
import random
from collections import deque

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

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

WIDTH = 720
HEIGHT = 580
PANEL_WIDTH = 300

WORLD_WIDTH = WIDTH
WORLD_HEIGHT = HEIGHT

DT = 0.10

ROBOT_RADIUS = 13

MAX_SPEED = 115.0
MAX_ACCELERATION = 100.0

OBSTACLE = np.array(
    [350.0, 285.0],
    dtype=np.float32
)

OBSTACLE_RADIUS = 52.0
SAFETY_MARGIN = 35.0

TARGET_AMPLITUDE = np.array(
    [90.0, 65.0],
    dtype=np.float32
)

TARGET_OMEGA = 0.65

STATE_DIM = 9
ACTION_DIM = 2

GAMMA = 0.99
TAU = 0.005

ACTOR_LR = 1e-4
CRITIC_LR = 3e-4

BUFFER_CAPACITY = 200_000
BATCH_SIZE = 128

WARMUP_STEPS = 2_000

POLICY_DELAY = 2
POLICY_NOISE = 0.15
NOISE_CLIP = 0.30

# Increased training budget.
TRAIN_EPISODES = 250
MAX_STEPS_PER_EPISODE = 500

PREDICTION_HORIZON = 0.75

# Reward parameters.
DISTANCE_SCALE = 600.0
TRACKING_DISTANCE_SCALE = 100.0


# ============================================================
# 2. COMMON UTILITIES
# ============================================================

def clamp_vector(vector, max_norm):

    vector = np.asarray(
        vector,
        dtype=np.float32
    )

    norm = np.linalg.norm(vector)

    if norm > max_norm:
        vector = vector * (
            max_norm / (norm + 1e-8)
        )

    return vector


def normalize_vector(vector, scale):

    return (
        np.asarray(vector, dtype=np.float32)
        / float(scale)
    )


def safe_unit(vector):

    vector = np.asarray(
        vector,
        dtype=np.float32
    )

    norm = np.linalg.norm(vector)

    if norm < 1e-8:
        return np.zeros_like(vector)

    return vector / norm


def distance(a, b):

    return float(
        np.linalg.norm(
            np.asarray(a) - np.asarray(b)
        )
    )


# ============================================================
# 3. MOVING TARGET MODEL
# ============================================================

class MovingTarget:

    def __init__(self, center=None):

        self.center = np.array(
            center if center is not None
            else [550.0, 145.0],
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

        self.center = np.asarray(
            position,
            dtype=np.float32
        )


def predict_target_position(
    position,
    velocity,
    horizon
):

    return (
        np.asarray(position, dtype=np.float32)
        + np.asarray(velocity, dtype=np.float32)
        * float(horizon)
    )


# ============================================================
# 4. ENVIRONMENT
# ============================================================

class MovingTargetEnv:

    def __init__(self):

        self.target = MovingTarget()

        self.max_steps = MAX_STEPS_PER_EPISODE

        self.step_count = 0

        self.position = np.array(
            [100.0, 440.0],
            dtype=np.float32
        )

        self.velocity = np.zeros(
            2,
            dtype=np.float32
        )

        self.previous_distance = 0.0

    def reset(self):

        self.position = np.array(
            [100.0, 440.0],
            dtype=np.float32
        )

        self.velocity = np.zeros(
            2,
            dtype=np.float32
        )

        self.target.phase = random.uniform(
            0.0,
            2.0 * math.pi
        )

        self.step_count = 0

        self.previous_distance = distance(
            self.position,
            self.target.position
        )

        return self.get_state()

    def get_state(self):
        """
        Nine-dimensional observation:

        0,1: Relative target position
        2,3: Relative target velocity
        4,5: Robot velocity
        6,7: Obstacle relative position
        8:   Target distance

        Relative target velocity is:

            target_velocity - robot_velocity

        This is important because the agent needs to learn
        how to match the target's movement, not just position.
        """

        relative_position = (
            self.target.position - self.position
        )

        relative_velocity = (
            self.target.velocity - self.velocity
        )

        obstacle_relative = (
            OBSTACLE - self.position
        )

        target_distance = distance(
            self.position,
            self.target.position
        )

        state = np.concatenate(
            [
                normalize_vector(
                    relative_position,
                    DISTANCE_SCALE
                ),

                normalize_vector(
                    relative_velocity,
                    MAX_SPEED * 2.0
                ),

                normalize_vector(
                    self.velocity,
                    MAX_SPEED
                ),

                normalize_vector(
                    obstacle_relative,
                    DISTANCE_SCALE
                ),

                np.array(
                    [
                        target_distance / DISTANCE_SCALE
                    ],
                    dtype=np.float32
                )
            ]
        )

        return state.astype(np.float32)

    def step(self, action):

        action = np.asarray(
            action,
            dtype=np.float32
        )

        if action.shape != (ACTION_DIM,):
            raise ValueError(
                f"Expected action shape {(ACTION_DIM,)}, "
                f"received {action.shape}"
            )

        action = np.clip(
            action,
            -1.0,
            1.0
        )

        # Target position at the next time step.
        next_phase = (
            self.target.phase
            + TARGET_OMEGA * DT
        )

        target_next = self.target.position_at(
            next_phase
        )

        distance_before = distance(
            self.position,
            target_next
        )

        # Robot dynamics.
        acceleration = (
            action * MAX_ACCELERATION
        )

        self.velocity += acceleration * DT

        self.velocity = clamp_vector(
            self.velocity,
            MAX_SPEED
        )

        self.position += self.velocity * DT

        # Update target.
        self.target.step()

        distance_after = distance(
            self.position,
            self.target.position
        )

        relative_velocity = (
            self.target.velocity - self.velocity
        )

        relative_speed = np.linalg.norm(
            relative_velocity
        )

        # ----------------------------------------------------
        # IMPROVED REWARD
        # ----------------------------------------------------

        # 1. Encourage the rocket to stay near the target.
        proximity_reward = 1.0 * math.exp(
            -distance_after / TRACKING_DISTANCE_SCALE
        )

        # 2. Penalize large tracking distance.
        distance_penalty = (
            0.8 * distance_after / DISTANCE_SCALE
        )

        # 3. Encourage velocity matching.
        velocity_penalty = (
            0.45 * relative_speed / (MAX_SPEED * 2.0)
        )

        # 4. Small progress reward.
        progress_reward = (
            0.35
            * (distance_before - distance_after)
            / (MAX_SPEED * DT)
        )

        # 5. Avoid unnecessary acceleration.
        action_penalty = (
            0.025 * float(np.sum(action ** 2))
        )

        reward = (
            proximity_reward
            - distance_penalty
            - velocity_penalty
            + progress_reward
            - action_penalty
        )

        # ----------------------------------------------------
        # COLLISION DETECTION
        # ----------------------------------------------------

        obstacle_distance = distance(
            self.position,
            OBSTACLE
        )

        terminated = False

        collision = (
            obstacle_distance
            < OBSTACLE_RADIUS + ROBOT_RADIUS
        )

        if collision:

            reward -= 100.0

            terminated = True

        # ----------------------------------------------------
        # WORLD BOUNDARIES
        # ----------------------------------------------------

        outside_world = (
            self.position[0] < ROBOT_RADIUS
            or self.position[0] > WORLD_WIDTH - ROBOT_RADIUS
            or self.position[1] < ROBOT_RADIUS
            or self.position[1] > WORLD_HEIGHT - ROBOT_RADIUS
        )

        if outside_world:

            reward -= 100.0

            terminated = True

        self.step_count += 1

        truncated = (
            self.step_count >= self.max_steps
        )

        next_state = self.get_state()

        info = {
            "distance": distance_after,
            "relative_speed": float(relative_speed),
            "progress": distance_before - distance_after,
            "collision": collision
        }

        return (
            next_state,
            float(reward),
            terminated,
            truncated,
            info
        )


# ============================================================
# 5. ACTION SAFETY FILTER
# ============================================================

def safety_filter(
    position,
    velocity,
    action
):

    action = np.asarray(
        action,
        dtype=np.float32
    ).copy()

    relative = (
        np.asarray(position, dtype=np.float32)
        - OBSTACLE
    )

    obstacle_distance = np.linalg.norm(
        relative
    )

    warning_distance = (
        OBSTACLE_RADIUS
        + ROBOT_RADIUS
        + SAFETY_MARGIN
    )

    if obstacle_distance < warning_distance:

        outward = safe_unit(relative)

        acceleration = (
            action * MAX_ACCELERATION
        )

        inward_component = float(
            np.dot(acceleration, outward)
        )

        if inward_component < 0.0:

            acceleration -= (
                inward_component * outward
            )

            action = (
                acceleration / MAX_ACCELERATION
            )

    return np.clip(
        action,
        -1.0,
        1.0
    )


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

    desired_direction = (
        aim_position - robot_position
    )

    desired_velocity = (
        safe_unit(desired_direction)
        * MAX_SPEED
        * 0.75
    )

    # Velocity feedback controller.
    acceleration = (
        2.0
        * (desired_velocity - robot_velocity)
    )

    acceleration = clamp_vector(
        acceleration,
        MAX_ACCELERATION
    )

    action = (
        acceleration / MAX_ACCELERATION
    )

    return action.astype(np.float32), aim_position


# ============================================================
# 7. REPLAY BUFFER
# ============================================================

class ReplayBuffer:

    def __init__(self, capacity):

        if capacity <= 0:
            raise ValueError(
                "Replay buffer capacity must be positive."
            )

        self.capacity = int(capacity)

        self.buffer = deque(
            maxlen=self.capacity
        )

    def add(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):

        state = np.asarray(
            state,
            dtype=np.float32
        )

        action = np.asarray(
            action,
            dtype=np.float32
        )

        next_state = np.asarray(
            next_state,
            dtype=np.float32
        )

        if state.shape != (STATE_DIM,):
            raise ValueError(
                f"Invalid state shape: {state.shape}"
            )

        if action.shape != (ACTION_DIM,):
            raise ValueError(
                f"Invalid action shape: {action.shape}"
            )

        if next_state.shape != (STATE_DIM,):
            raise ValueError(
                f"Invalid next_state shape: {next_state.shape}"
            )

        if not all(
            np.all(np.isfinite(x))
            for x in (state, action, next_state)
        ):
            raise ValueError(
                "Replay transition contains NaN or Inf."
            )

        if not np.isfinite(reward):
            raise ValueError(
                "Reward contains NaN or Inf."
            )

        self.buffer.append(
            (
                state.copy(),
                action.copy(),
                float(reward),
                next_state.copy(),
                float(done)
            )
        )

    def sample(self, batch_size):

        if batch_size <= 0:
            raise ValueError(
                "Batch size must be greater than zero."
            )

        if len(self.buffer) < batch_size:
            raise ValueError(
                f"Not enough transitions. "
                f"Available: {len(self.buffer)}, "
                f"required: {batch_size}."
            )

        batch = random.sample(
            self.buffer,
            batch_size
        )

        states, actions, rewards, next_states, dones = zip(
            *batch
        )

        states = torch.as_tensor(
            np.stack(states),
            dtype=torch.float32,
            device=DEVICE
        )

        actions = torch.as_tensor(
            np.stack(actions),
            dtype=torch.float32,
            device=DEVICE
        )

        rewards = torch.as_tensor(
            np.asarray(rewards),
            dtype=torch.float32,
            device=DEVICE
        ).unsqueeze(1)

        next_states = torch.as_tensor(
            np.stack(next_states),
            dtype=torch.float32,
            device=DEVICE
        )

        dones = torch.as_tensor(
            np.asarray(dones),
            dtype=torch.float32,
            device=DEVICE
        ).unsqueeze(1)

        return (
            states,
            actions,
            rewards,
            next_states,
            dones
        )

    def __len__(self):

        return len(self.buffer)


# ============================================================
# 8. ACTOR AND CRITIC NETWORKS
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
            nn.Linear(
                STATE_DIM + ACTION_DIM,
                256
            ),
            nn.ReLU(),

            nn.Linear(256, 256),
            nn.ReLU(),

            nn.Linear(256, 1)
        )

    def forward(self, state, action):

        return self.network(
            torch.cat(
                [state, action],
                dim=1
            )
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
            list(self.critic1.parameters())
            + list(self.critic2.parameters()),
            lr=CRITIC_LR
        )

        self.training_steps = 0

    def select_action(
        self,
        state,
        noise=0.0
    ):

        state_tensor = torch.as_tensor(
            state,
            dtype=torch.float32,
            device=DEVICE
        ).unsqueeze(0)

        with torch.no_grad():

            action = self.actor(
                state_tensor
            ).cpu().numpy()[0]

        if noise > 0.0:

            action += np.random.normal(
                0.0,
                noise,
                size=ACTION_DIM
            )

        return np.clip(
            action,
            -1.0,
            1.0
        ).astype(np.float32)

    @staticmethod
    def set_gradients_enabled(network, enabled):

        for parameter in network.parameters():
            parameter.requires_grad_(enabled)

    def soft_update(
        self,
        source,
        target
    ):

        with torch.no_grad():

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

        # ----------------------------------------------------
        # 1. TARGET POLICY SMOOTHING
        # ----------------------------------------------------

        with torch.no_grad():

            next_actions = self.actor_target(
                next_states
            )

            noise = (
                torch.randn_like(next_actions)
                * POLICY_NOISE
            )

            noise = noise.clamp(
                -NOISE_CLIP,
                NOISE_CLIP
            )

            next_actions = (
                next_actions + noise
            ).clamp(
                -1.0,
                1.0
            )

            # ------------------------------------------------
            # 2. TWIN TARGET CRITICS
            # ------------------------------------------------

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

            # ------------------------------------------------
            # 3. BELLMAN TARGET
            # ------------------------------------------------

            target = (
                rewards
                + GAMMA * (1.0 - dones) * target_q
            )

        # ----------------------------------------------------
        # 4. CURRENT CRITIC VALUES
        # ----------------------------------------------------

        current_q1 = self.critic1(
            states,
            actions
        )

        current_q2 = self.critic2(
            states,
            actions
        )

        critic_loss = (
            F.mse_loss(current_q1, target)
            + F.mse_loss(current_q2, target)
        )

        # ----------------------------------------------------
        # 5. UPDATE CRITICS
        # ----------------------------------------------------

        self.critic_optimizer.zero_grad(
            set_to_none=True
        )

        critic_loss.backward()

        nn.utils.clip_grad_norm_(
            list(self.critic1.parameters())
            + list(self.critic2.parameters()),
            1.0
        )

        self.critic_optimizer.step()

        self.training_steps += 1

        actor_loss_value = None

        # ----------------------------------------------------
        # 6. DELAYED ACTOR UPDATE
        # ----------------------------------------------------

        if self.training_steps % POLICY_DELAY == 0:

            # Freeze critic weights while calculating
            # the actor's policy gradient.
            self.set_gradients_enabled(
                self.critic1,
                False
            )

            actor_loss = -self.critic1(
                states,
                self.actor(states)
            ).mean()

            self.actor_optimizer.zero_grad(
                set_to_none=True
            )

            actor_loss.backward()

            nn.utils.clip_grad_norm_(
                self.actor.parameters(),
                1.0
            )

            self.actor_optimizer.step()

            self.set_gradients_enabled(
                self.critic1,
                True
            )

            actor_loss_value = float(
                actor_loss.item()
            )

            # ------------------------------------------------
            # 7. SOFT TARGET UPDATES
            # ------------------------------------------------

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

        return {
            "critic_loss": float(
                critic_loss.item()
            ),
            "actor_loss": actor_loss_value
        }


# ============================================================
# 10. TRAINING
# ============================================================

def train_td3():

    print(
        "\nTraining improved TD3 tracking agent..."
    )

    print(f"Device: {DEVICE}")

    print(
        f"Episodes: {TRAIN_EPISODES}"
    )

    print(
        f"Maximum environment steps: "
        f"{TRAIN_EPISODES * MAX_STEPS_PER_EPISODE:,}"
    )

    env = MovingTargetEnv()

    agent = TD3Agent()

    replay = ReplayBuffer(
        BUFFER_CAPACITY
    )

    episode_rewards = []

    episode_distances = []

    episode_relative_speeds = []

    total_steps = 0

    for episode in range(TRAIN_EPISODES):

        state = env.reset()

        total_reward = 0.0

        distances = []

        relative_speeds = []

        for step in range(MAX_STEPS_PER_EPISODE):

            # ------------------------------------------------
            # EXPLORATION
            # ------------------------------------------------

            if total_steps < WARMUP_STEPS:

                action = np.random.uniform(
                    -1.0,
                    1.0,
                    size=ACTION_DIM
                ).astype(np.float32)

            else:

                # Gradually reduce exploration.
                progress = min(
                    1.0,
                    episode / TRAIN_EPISODES
                )

                exploration = (
                    0.22 * (1.0 - progress)
                    + 0.06 * progress
                )

                action = agent.select_action(
                    state,
                    noise=exploration
                )

            # ------------------------------------------------
            # SAFETY FILTER
            # ------------------------------------------------

            action = safety_filter(
                env.position,
                env.velocity,
                action
            )

            # ------------------------------------------------
            # ENVIRONMENT TRANSITION
            # ------------------------------------------------

            (
                next_state,
                reward,
                terminated,
                truncated,
                info
            ) = env.step(action)

            # A time-limit truncation is not a terminal
            # MDP state, so preserve bootstrapping.
            replay.add(
                state,
                action,
                reward,
                next_state,
                terminated
            )

            # ------------------------------------------------
            # TD3 UPDATE
            # ------------------------------------------------

            if total_steps >= WARMUP_STEPS:

                agent.train_step(replay)

            state = next_state

            total_reward += reward

            distances.append(
                info["distance"]
            )

            relative_speeds.append(
                info["relative_speed"]
            )

            total_steps += 1

            if terminated or truncated:
                break

        episode_rewards.append(
            total_reward
        )

        episode_distances.append(
            float(np.mean(distances))
        )

        episode_relative_speeds.append(
            float(np.mean(relative_speeds))
        )

        if (episode + 1) % 10 == 0:

            recent_reward = np.mean(
                episode_rewards[-10:]
            )

            recent_distance = np.mean(
                episode_distances[-10:]
            )

            recent_relative_speed = np.mean(
                episode_relative_speeds[-10:]
            )

            print(
                f"Episode {episode + 1:3d}/{TRAIN_EPISODES} | "
                f"Reward: {recent_reward:8.2f} | "
                f"Distance: {recent_distance:7.2f} | "
                f"Relative speed: {recent_relative_speed:6.2f} | "
                f"Replay: {len(replay):6d}"
            )

    return (
        agent,
        episode_rewards,
        episode_distances,
        episode_relative_speeds
    )


# ============================================================
# 11. CONTROLLER EVALUATION
# ============================================================

def evaluate_controller(
    agent,
    mode,
    episodes=5,
    steps_per_episode=400
):

    env = MovingTargetEnv()

    all_distances = []

    all_relative_speeds = []

    collisions = 0

    total_steps = 0

    for episode in range(episodes):

        state = env.reset()

        for _ in range(steps_per_episode):

            if mode == 1:

                action, _ = compute_pursuit_action(
                    env.position,
                    env.velocity,
                    env.target.position,
                    env.target.velocity,
                    PREDICTION_HORIZON,
                    predictive=False
                )

            elif mode == 2:

                action, _ = compute_pursuit_action(
                    env.position,
                    env.velocity,
                    env.target.position,
                    env.target.velocity,
                    PREDICTION_HORIZON,
                    predictive=True
                )

            else:

                action = agent.select_action(
                    state,
                    noise=0.0
                )

            action = safety_filter(
                env.position,
                env.velocity,
                action
            )

            (
                state,
                reward,
                terminated,
                truncated,
                info
            ) = env.step(action)

            all_distances.append(
                info["distance"]
            )

            all_relative_speeds.append(
                info["relative_speed"]
            )

            total_steps += 1

            if info["collision"]:
                collisions += 1

            if terminated or truncated:
                break

    return {
        "mean_distance": float(
            np.mean(all_distances)
        ),
        "mean_relative_speed": float(
            np.mean(all_relative_speeds)
        ),
        "collision_rate": (
            collisions / max(episodes, 1)
        ),
        "steps": total_steps
    }


def compare_controllers(agent):

    print("\n" + "=" * 65)

    print("CONTROLLER PERFORMANCE EVALUATION")

    print("=" * 65)

    results = {}

    for mode, name in [
        (1, "Reactive"),
        (2, "Predictive"),
        (3, "TD3")
    ]:

        result = evaluate_controller(
            agent,
            mode,
            episodes=5,
            steps_per_episode=400
        )

        results[name] = result

        print(
            f"{name:12s} | "
            f"Distance: {result['mean_distance']:8.2f} | "
            f"Relative speed: "
            f"{result['mean_relative_speed']:7.2f} | "
            f"Collision rate: "
            f"{result['collision_rate']:.2f}"
        )

    print("=" * 65)

    return results


# ============================================================
# 12. PYGAME GUI
# ============================================================

class RocketGUI:

    def __init__(
        self,
        agent,
        rewards,
        distances,
        relative_speeds
    ):

        pygame.init()

        self.screen = pygame.display.set_mode(
            (WIDTH + PANEL_WIDTH, HEIGHT)
        )

        pygame.display.set_caption(
            "Phase 4.12 — Improved Predictive Tracking"
        )

        self.clock = pygame.time.Clock()

        self.font = pygame.font.SysFont(
            "consolas",
            16
        )

        self.agent = agent

        self.rewards = rewards

        self.distances = distances

        self.relative_speeds = relative_speeds

        self.env = MovingTargetEnv()

        self.env.reset()

        self.mode = 2

        self.paused = False

        self.speed = 1

        self.horizon = PREDICTION_HORIZON

        self.last_action = np.zeros(
            2,
            dtype=np.float32
        )

        self.aim_position = (
            self.env.target.position.copy()
        )

        self.running = True

    def reset(self):

        self.env.reset()

        self.last_action[:] = 0.0

        self.aim_position = (
            self.env.target.position.copy()
        )

    def select_action(self):

        target_position = (
            self.env.target.position
        )

        target_velocity = (
            self.env.target.velocity
        )

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

        if not (
            150 < x < WIDTH - 130
            and 100 < y < HEIGHT - 100
        ):
            return

        candidate = np.array(
            [x, y],
            dtype=np.float32
        )

        clearance = (
            np.linalg.norm(candidate - OBSTACLE)
            - np.linalg.norm(TARGET_AMPLITUDE)
        )

        if clearance < (
            OBSTACLE_RADIUS + ROBOT_RADIUS + 25
        ):
            return

        self.env.target.set_center(candidate)

    def draw_arrow(
        self,
        start,
        end,
        color,
        width=2
    ):

        start = np.asarray(
            start,
            dtype=np.float64
        )

        end = np.asarray(
            end,
            dtype=np.float64
        )

        if start.shape != (2,) or end.shape != (2,):
            return

        if not (
            np.all(np.isfinite(start))
            and np.all(np.isfinite(end))
        ):
            return

        start_point = (
            int(round(start[0])),
            int(round(start[1]))
        )

        end_point = (
            int(round(end[0])),
            int(round(end[1]))
        )

        pygame.draw.line(
            self.screen,
            color,
            start_point,
            end_point,
            width
        )

        vector = end - start

        norm = np.linalg.norm(vector)

        if norm < 1e-6:
            return

        direction = vector / norm

        perpendicular = np.array(
            [-direction[1], direction[0]]
        )

        p1 = (
            end
            - direction * 11
            + perpendicular * 5
        )

        p2 = (
            end
            - direction * 11
            - perpendicular * 5
        )

        pygame.draw.polygon(
            self.screen,
            color,
            [
                end_point,
                (
                    int(round(p1[0])),
                    int(round(p1[1]))
                ),
                (
                    int(round(p2[0])),
                    int(round(p2[1]))
                )
            ]
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

            direction = np.array(
                [1.0, 0.0],
                dtype=np.float32
            )

        perpendicular = np.array(
            [-direction[1], direction[0]]
        )

        nose = position + direction * 22

        rear = position - direction * 15

        left_fin = rear + perpendicular * 13

        right_fin = rear - perpendicular * 13

        body_left = position + perpendicular * 8

        body_right = position - perpendicular * 8

        exhaust_length = random.randint(
            10,
            19
        )

        exhaust_tip = (
            rear - direction * exhaust_length
        )

        def point(value):

            return (
                int(round(value[0])),
                int(round(value[1]))
            )

        pygame.draw.line(
            self.screen,
            (255, 130, 30),
            point(rear),
            point(exhaust_tip),
            5
        )

        pygame.draw.polygon(
            self.screen,
            (200, 210, 225),
            [
                point(nose),
                point(body_left),
                point(left_fin),
                point(rear),
                point(right_fin),
                point(body_right)
            ]
        )

        pygame.draw.circle(
            self.screen,
            (30, 45, 65),
            point(position),
            4
        )

    def draw_target_path(self):

        points = []

        for i in range(100):

            phase = (
                self.env.target.phase + i * 0.08
            )

            position = (
                self.env.target.position_at(phase)
            )

            points.append(
                (
                    int(position[0]),
                    int(position[1])
                )
            )

        if len(points) > 1:

            pygame.draw.lines(
                self.screen,
                (65, 95, 75),
                False,
                points,
                1
            )

    def draw_world(self):

        self.screen.fill(
            (15, 20, 30)
        )

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

        pygame.draw.circle(
            self.screen,
            (110, 55, 55),
            tuple(OBSTACLE.astype(int)),
            int(OBSTACLE_RADIUS + SAFETY_MARGIN),
            2
        )

        pygame.draw.circle(
            self.screen,
            (170, 65, 65),
            tuple(OBSTACLE.astype(int)),
            int(OBSTACLE_RADIUS)
        )

        pygame.draw.circle(
            self.screen,
            (220, 115, 105),
            tuple(OBSTACLE.astype(int)),
            8
        )

        # Predicted target location.
        pygame.draw.circle(
            self.screen,
            (70, 150, 230),
            tuple(self.aim_position.astype(int)),
            9,
            2
        )

        target = self.env.target.position

        pygame.draw.circle(
            self.screen,
            (70, 230, 120),
            tuple(target.astype(int)),
            10
        )

        pygame.draw.circle(
            self.screen,
            (180, 255, 190),
            tuple(target.astype(int)),
            18,
            1
        )

        self.draw_arrow(
            target,
            target + self.env.target.velocity * 0.35,
            (100, 240, 130),
            2
        )

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

        target_distance = distance(
            self.env.position,
            self.env.target.position
        )

        relative_speed = np.linalg.norm(
            self.env.target.velocity - self.env.velocity
        )

        lines = [
            ("PHASE 4.12", (255, 220, 100)),
            ("Improved TD3 Tracking", (220, 230, 245)),
            ("", (200, 200, 200)),

            ("CONTROLLER", (150, 190, 255)),
            ("1 Reactive pursuit", (220, 220, 220)),
            ("2 Predictive pursuit", (220, 220, 220)),
            ("3 TD3 learned", (220, 220, 220)),
            ("", (200, 200, 200)),

            (f"Active mode: {self.mode}", (100, 240, 150)),
            ("", (200, 200, 200)),

            ("TARGET TRACKING", (150, 190, 255)),
            (f"Distance: {target_distance:.1f}", (230, 230, 230)),
            (f"Relative speed: {relative_speed:.1f}", (230, 230, 230)),
            (f"Target vx: {self.env.target.velocity[0]:.1f}", (180, 240, 180)),
            (f"Target vy: {self.env.target.velocity[1]:.1f}", (180, 240, 180)),
            (f"Prediction H: {self.horizon:.2f}s", (120, 190, 255)),
            ("", (200, 200, 200)),

            ("ROCKET", (150, 190, 255)),
            (f"Velocity: {np.linalg.norm(self.env.velocity):.1f}", (230, 230, 230)),
            (f"Action X: {self.last_action[0]:+.2f}", (230, 230, 230)),
            (f"Action Y: {self.last_action[1]:+.2f}", (230, 230, 230)),
            ("", (200, 200, 200)),

            ("SIMULATION", (150, 190, 255)),
            (f"Speed: {self.speed}x", (230, 230, 230)),
            (f"Steps: {self.env.step_count}", (230, 230, 230)),
            ("PAUSED" if self.paused else "RUNNING", (255, 190, 100)),
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

                surface = self.font.render(
                    text,
                    True,
                    color
                )

                self.screen.blit(
                    surface,
                    (x + 15, y)
                )

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

                    self.speed = min(
                        5,
                        self.speed + 1
                    )

                elif event.key in (
                    pygame.K_MINUS,
                    pygame.K_KP_MINUS
                ):

                    self.speed = max(
                        1,
                        self.speed - 1
                    )

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

                        self.change_target_center(
                            event.pos
                        )

    def run(self):

        while self.running:

            self.handle_events()

            if not self.paused:

                for _ in range(self.speed):

                    action = self.select_action()

                    (
                        _,
                        _,
                        terminated,
                        truncated,
                        _
                    ) = self.env.step(action)

                    if terminated or truncated:

                        self.reset()

                        break

            self.draw_world()

            self.draw_panel()

            pygame.display.flip()

            self.clock.tick(60)

        pygame.quit()


# ============================================================
# 13. TRAINING VISUALIZATION
# ============================================================

def visualize_training(
    rewards,
    distances,
    relative_speeds
):

    try:

        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(
            3,
            1,
            figsize=(10, 10)
        )

        axes[0].plot(
            rewards,
            alpha=0.35,
            label="Episode reward"
        )

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

        axes[0].set_title(
            "TD3 Training Reward"
        )

        axes[0].set_xlabel("Episode")

        axes[0].set_ylabel("Reward")

        axes[0].legend()

        axes[0].grid(True)

        axes[1].plot(
            distances,
            label="Mean tracking distance"
        )

        axes[1].set_title(
            "Tracking Distance"
        )

        axes[1].set_xlabel("Episode")

        axes[1].set_ylabel("Distance")

        axes[1].legend()

        axes[1].grid(True)

        axes[2].plot(
            relative_speeds,
            label="Mean relative speed"
        )

        axes[2].set_title(
            "Relative Velocity Error"
        )

        axes[2].set_xlabel("Episode")

        axes[2].set_ylabel("Relative speed")

        axes[2].legend()

        axes[2].grid(True)

        plt.tight_layout()

        plt.show()

    except ImportError:

        print(
            "Matplotlib not installed; "
            "skipping training plots."
        )


# ============================================================
# 14. UNIT TESTS
# ============================================================

def run_tests():

    print("\nRunning Phase 4.12 tests...\n")

    # Test 1
    target = MovingTarget()

    assert target.position.shape == (2,)

    print("Test 1 passed: target position")

    # Test 2
    assert target.velocity.shape == (2,)

    print("Test 2 passed: target velocity")

    # Test 3
    predicted = predict_target_position(
        np.array([10.0, 20.0]),
        np.array([2.0, -4.0]),
        0.5
    )

    assert np.allclose(
        predicted,
        [11.0, 18.0]
    )

    print("Test 3 passed: prediction")

    # Test 4
    predicted = predict_target_position(
        np.array([10.0, 20.0]),
        np.array([2.0, -4.0]),
        0.0
    )

    assert np.allclose(
        predicted,
        [10.0, 20.0]
    )

    print("Test 4 passed: zero prediction horizon")

    # Test 5
    action, aim = compute_pursuit_action(
        np.array([100.0, 100.0]),
        np.zeros(2),
        np.array([200.0, 100.0]),
        np.array([0.0, 20.0]),
        1.0,
        predictive=False
    )

    assert np.allclose(
        aim,
        [200.0, 100.0]
    )

    print("Test 5 passed: reactive pursuit")

    # Test 6
    action, aim = compute_pursuit_action(
        np.array([100.0, 100.0]),
        np.zeros(2),
        np.array([200.0, 100.0]),
        np.array([0.0, 20.0]),
        1.0,
        predictive=True
    )

    assert np.allclose(
        aim,
        [200.0, 120.0]
    )

    print("Test 6 passed: predictive pursuit")

    # Test 7
    assert np.all(action >= -1.0)
    assert np.all(action <= 1.0)

    print("Test 7 passed: action bounds")

    # Test 8
    env = MovingTargetEnv()

    state = env.reset()

    assert state.shape == (STATE_DIM,)

    print("Test 8 passed: state dimension")

    # Test 9
    result = env.step(
        np.zeros(ACTION_DIM)
    )

    assert len(result) == 5

    assert result[0].shape == (STATE_DIM,)

    print("Test 9 passed: environment step")

    # Test 10
    assert np.isfinite(result[1])

    print("Test 10 passed: finite reward")

    # Test 11
    replay = ReplayBuffer(
        BATCH_SIZE * 2
    )

    replay.add(
        np.zeros(STATE_DIM),
        np.zeros(ACTION_DIM),
        1.0,
        np.ones(STATE_DIM),
        False
    )

    assert len(replay) == 1

    print("Test 11 passed: replay buffer")

    # Test 12
    for _ in range(BATCH_SIZE):

        replay.add(
            np.zeros(STATE_DIM),
            np.zeros(ACTION_DIM),
            1.0,
            np.ones(STATE_DIM),
            False
        )

    batch = replay.sample(
        BATCH_SIZE
    )

    assert batch[0].shape == (BATCH_SIZE, STATE_DIM)

    assert batch[1].shape == (BATCH_SIZE, ACTION_DIM)

    assert batch[2].shape == (BATCH_SIZE, 1)

    assert batch[3].shape == (BATCH_SIZE, STATE_DIM)

    assert batch[4].shape == (BATCH_SIZE, 1)

    print("Test 12 passed: replay batch dimensions")

    # Test 13
    actor = Actor().to(DEVICE)

    sample_state = torch.zeros(
        (4, STATE_DIM),
        device=DEVICE
    )

    with torch.no_grad():

        actor_output = actor(
            sample_state
        )

    assert actor_output.shape == (4, ACTION_DIM)

    assert torch.all(actor_output <= 1.0)

    assert torch.all(actor_output >= -1.0)

    print("Test 13 passed: actor output")

    # Test 14
    critic = Critic().to(DEVICE)

    with torch.no_grad():

        q_value = critic(
            sample_state,
            torch.zeros(
                (4, ACTION_DIM),
                device=DEVICE
            )
        )

    assert q_value.shape == (4, 1)

    print("Test 14 passed: critic output")

    # Test 15
    agent = TD3Agent()

    action = agent.select_action(
        np.zeros(STATE_DIM, dtype=np.float32)
    )

    assert action.shape == (ACTION_DIM,)

    assert np.all(action >= -1.0)

    assert np.all(action <= 1.0)

    print("Test 15 passed: TD3 action")

    # Test 16
    filtered = safety_filter(
        np.array([300.0, 285.0]),
        np.zeros(2),
        np.array([-1.0, 0.5])
    )

    assert np.all(filtered >= -1.0)

    assert np.all(filtered <= 1.0)

    print("Test 16 passed: safety filter")

    # Test 17
    target = MovingTarget()

    target.set_center(
        [500.0, 150.0]
    )

    assert np.allclose(
        target.center,
        [500.0, 150.0]
    )

    print("Test 17 passed: target center")

    # Test 18: TD3 update
    test_agent = TD3Agent()

    first_update = test_agent.train_step(
        replay
    )

    assert first_update is not None

    assert np.isfinite(
        first_update["critic_loss"]
    )

    assert first_update["actor_loss"] is None

    second_update = test_agent.train_step(
        replay
    )

    assert second_update is not None

    assert second_update["actor_loss"] is not None

    assert np.isfinite(
        second_update["actor_loss"]
    )

    assert test_agent.training_steps == 2

    print("Test 18 passed: TD3 updates")

    # Test 19
    small_replay = ReplayBuffer(10)

    small_replay.add(
        np.zeros(STATE_DIM),
        np.zeros(ACTION_DIM),
        1.0,
        np.ones(STATE_DIM),
        False
    )

    try:

        small_replay.sample(BATCH_SIZE)

        raise AssertionError(
            "Expected insufficient-sample ValueError."
        )

    except ValueError as error:

        assert "Not enough transitions" in str(error)

    print("Test 19 passed: insufficient sampling protection")

    # Test 20
    capacity_test = ReplayBuffer(5)

    for i in range(8):

        capacity_test.add(
            np.ones(STATE_DIM) * i,
            np.zeros(ACTION_DIM),
            float(i),
            np.ones(STATE_DIM) * (i + 1),
            False
        )

    assert len(capacity_test) == 5

    print("Test 20 passed: replay capacity")

    # Test 21: relative velocity observation
    env = MovingTargetEnv()

    env.reset()

    state = env.get_state()

    expected_relative_velocity = (
        env.target.velocity - env.velocity
    ) / (MAX_SPEED * 2.0)

    assert np.allclose(
        state[2:4],
        expected_relative_velocity
    )

    print("Test 21 passed: relative velocity observation")

    # Test 22: finite state
    assert np.all(np.isfinite(state))

    print("Test 22 passed: finite observation")

    print("\n" + "=" * 55)

    print("ALL 22 TESTS PASSED SUCCESSFULLY!")

    print("=" * 55)


# ============================================================
# 15. MAIN
# ============================================================

def main():

    print("=" * 65)

    print(
        "PHASE 4.12 — IMPROVED PREDICTIVE MOVING-TARGET TRACKING"
    )

    print("=" * 65)

    print(f"PyTorch device: {DEVICE}")

    print(f"State dimension: {STATE_DIM}")

    print(f"Action dimension: {ACTION_DIM}")

    print(f"TD3 batch size: {BATCH_SIZE}")

    print(f"Replay capacity: {BUFFER_CAPACITY}")

    print(f"Training episodes: {TRAIN_EPISODES}")

    print(
        f"Maximum training steps: "
        f"{TRAIN_EPISODES * MAX_STEPS_PER_EPISODE:,}"
    )

    # Step 1: Unit tests.
    run_tests()

    # Step 2: Train TD3.
    (
        agent,
        rewards,
        distances,
        relative_speeds
    ) = train_td3()

    # Step 3: Compare controllers.
    compare_controllers(agent)

    # Step 4: Plot training results.
    visualize_training(
        rewards,
        distances,
        relative_speeds
    )

    # Step 5: Interactive simulation.
    print(
        "\nStarting interactive Pygame simulation..."
    )

    print(
        "Select controller 1, 2 or 3 to compare performance."
    )

    gui = RocketGUI(
        agent,
        rewards,
        distances,
        relative_speeds
    )

    gui.run()


if __name__ == "__main__":

    main()
