
"""
RL COURSE — PHASE 4.13
============================================================
Filename:
    phase4/mpc_rl_hybrid.py

LESSON:
    Model Predictive Control (MPC) and RL Hybrid Control

PURPOSE
-------
Combine model-based planning with a learned reinforcement
learning policy to track a moving target safely.

WHAT YOU LEARN
--------------
STEP 1 — Predictive dynamics
    Predict future rocket position and velocity.

STEP 2 — Moving-target prediction
    Use the target's trajectory model to predict future
    target positions.

STEP 3 — Model Predictive Control
    Evaluate candidate action sequences over a finite horizon.
    Select the first action from the sequence with the lowest
    predicted cost.

STEP 4 — TD3
    Use a learned continuous-control policy as an RL baseline.

STEP 5 — Hybrid control
    Combine MPC and TD3 actions:
        a_hybrid = w * a_MPC + (1-w) * a_TD3

STEP 6 — Safety and collision avoidance
    Penalize predicted obstacle proximity and apply a
    reactive safety filter.

STEP 7 — Interactive Pygame visualization
    Compare TD3, MPC and Hybrid control with a moving target.

MATHEMATICAL FOUNDATION
-----------------------

Robot dynamics:
    v(t+1) = clip(v(t) + a(t)*dt)
    p(t+1) = p(t) + v(t+1)*dt

Target prediction:
    p_target(t+H) = trajectory_model(t+H)

MPC optimization:
    U* = argmin_U J(x,U)

    J = sum(
        tracking_cost
        + velocity_cost
        + obstacle_cost
        + control_cost
    )

Only the first action of the optimized sequence is executed.
At the next time step, MPC predicts again and replans.

TD3:
    Uses twin critics, delayed actor updates and target-policy
    smoothing to learn continuous actions.

Hybrid controller:
    a = w * a_MPC + (1-w) * a_TD3

ROBOTICS APPLICATION
--------------------
- Autonomous mobile robot navigation
- Moving-object tracking
- Drone rendezvous
- Obstacle-aware target following
- Robot manipulation with moving objects

IMPORTANT
---------
MPC in this lesson uses a simplified known simulation model.
The TD3 agent is trained in simulation.
Neither is a validated real-world safety controller.

CONTROLS
--------
1       TD3
2       MPC
3       Hybrid MPC + TD3
SPACE   Pause / resume
R       Reset rocket
+ / -   Simulation speed
UP/DOWN Change MPC prediction horizon
LEFT/RIGHT Change hybrid RL weight
Mouse   Change moving-target trajectory center
ESC     Exit
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

SEED = 43

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

WIDTH = 720
HEIGHT = 580
PANEL_WIDTH = 320

DT = 0.10

ROBOT_RADIUS = 13
MAX_SPEED = 115.0
MAX_ACCELERATION = 100.0

OBSTACLE = np.array([350.0, 285.0], dtype=np.float32)
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

BUFFER_CAPACITY = 100_000
BATCH_SIZE = 128
WARMUP_STEPS = 500

POLICY_DELAY = 2
POLICY_NOISE = 0.15
NOISE_CLIP = 0.30

TRAIN_EPISODES = 100
MAX_STEPS_PER_EPISODE = 400

# MPC parameters.
MPC_HORIZON = 12
MPC_CANDIDATES = 96

MPC_TRACKING_WEIGHT = 1.0
MPC_VELOCITY_WEIGHT = 0.025
MPC_CONTROL_WEIGHT = 0.035
MPC_OBSTACLE_WEIGHT = 12.0

# Hybrid controller:
# 0.0 = pure TD3
# 1.0 = pure MPC
HYBRID_MPC_WEIGHT = 0.65


# ============================================================
# 2. COMMON UTILITIES
# ============================================================

def clamp_vector(vector, maximum):
    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))

    if norm > maximum:
        vector = vector * (maximum / (norm + 1e-8))

    return vector


def safe_unit(vector):
    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))

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
# 3. MOVING TARGET
# ============================================================

class MovingTarget:

    def __init__(self, center=None):
        if center is None:
            center = [550.0, 145.0]

        self.center = np.asarray(
            center,
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

    def set_center(self, center):
        self.center = np.asarray(
            center,
            dtype=np.float32
        )


# ============================================================
# 4. ROBOT DYNAMICS MODEL
# ============================================================

def predict_robot_state(position, velocity, action, dt=DT):
    """
    One-step robot dynamics.

    The action is normalized to [-1,1].

    The simulation uses acceleration control:
        a = action * MAX_ACCELERATION

    Velocity is clipped to MAX_SPEED.
    """

    action = np.clip(
        np.asarray(action, dtype=np.float32),
        -1.0,
        1.0
    )

    acceleration = action * MAX_ACCELERATION

    next_velocity = velocity + acceleration * dt

    next_velocity = clamp_vector(
        next_velocity,
        MAX_SPEED
    )

    next_position = position + next_velocity * dt

    return next_position, next_velocity


# ============================================================
# 5. ENVIRONMENT
# ============================================================

class MovingTargetEnv:

    def __init__(self):
        self.target = MovingTarget()
        self.max_steps = MAX_STEPS_PER_EPISODE

        self.position = np.array(
            [100.0, 440.0],
            dtype=np.float32
        )

        self.velocity = np.zeros(
            2,
            dtype=np.float32
        )

        self.step_count = 0

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

        return self.get_state()

    def get_state(self):
        relative_target = (
            self.target.position - self.position
        )

        relative_obstacle = (
            OBSTACLE - self.position
        )

        state = np.concatenate(
            [
                relative_target / 600.0,
                self.target.velocity / 150.0,
                self.velocity / MAX_SPEED,
                relative_obstacle / 600.0,
                np.array(
                    [
                        distance(
                            self.position,
                            self.target.position
                        ) / 600.0
                    ],
                    dtype=np.float32
                )
            ]
        )

        return state.astype(np.float32)

    def step(self, action):
        action = np.clip(
            np.asarray(action, dtype=np.float32),
            -1.0,
            1.0
        )

        # Predict the target's next position.
        next_phase = (
            self.target.phase + TARGET_OMEGA * DT
        )

        target_at_next_time = (
            self.target.position_at(next_phase)
        )

        # Compare robot progress against the target at the
        # same time instant, avoiding reward distortion from
        # target movement alone.
        distance_before = distance(
            self.position,
            target_at_next_time
        )

        self.position, self.velocity = predict_robot_state(
            self.position,
            self.velocity,
            action
        )

        self.target.step()

        distance_after = distance(
            self.position,
            self.target.position
        )

        progress = distance_before - distance_after

        reward = 0.08 * progress

        reward += 0.3 * math.exp(
            -distance_after / 75.0
        )

        reward -= 0.015 * float(
            np.sum(action ** 2)
        )

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

        if (
            self.position[0] < ROBOT_RADIUS
            or self.position[0] > WIDTH - ROBOT_RADIUS
            or self.position[1] < ROBOT_RADIUS
            or self.position[1] > HEIGHT - ROBOT_RADIUS
        ):
            reward -= 100.0
            terminated = True

        self.step_count += 1

        truncated = (
            self.step_count >= self.max_steps
        )

        next_state = self.get_state()

        info = {
            "distance": distance_after,
            "progress": progress,
            "collision": obstacle_distance < (
                OBSTACLE_RADIUS + ROBOT_RADIUS
            )
        }

        return (
            next_state,
            float(reward),
            terminated,
            truncated,
            info
        )


# ============================================================
# 6. MPC COST FUNCTION
# ============================================================

def mpc_step_cost(
    position,
    velocity,
    target_position,
    action
):
    """
    Evaluate one predicted state.

    Cost components:
        1. Distance to moving target.
        2. Excessive robot velocity.
        3. Excessive control effort.
        4. Obstacle proximity.

    Obstacle cost rises sharply near the obstacle.
    """

    tracking_cost = (
        distance(position, target_position) ** 2
    )

    velocity_cost = float(
        np.sum(velocity ** 2)
    )

    control_cost = float(
        np.sum(action ** 2)
    )

    obstacle_distance = distance(
        position,
        OBSTACLE
    )

    safe_distance = (
        OBSTACLE_RADIUS
        + ROBOT_RADIUS
        + SAFETY_MARGIN
    )

    penetration = max(
        0.0,
        safe_distance - obstacle_distance
    )

    obstacle_cost = penetration ** 2

    total = (
        MPC_TRACKING_WEIGHT * tracking_cost
        + MPC_VELOCITY_WEIGHT * velocity_cost
        + MPC_CONTROL_WEIGHT * control_cost
        + MPC_OBSTACLE_WEIGHT * obstacle_cost
    )

    # Strong penalty for predicted collision.
    if obstacle_distance < (
        OBSTACLE_RADIUS + ROBOT_RADIUS
    ):
        total += 100_000.0

    # Strong penalty for leaving the world.
    if (
        position[0] < ROBOT_RADIUS
        or position[0] > WIDTH - ROBOT_RADIUS
        or position[1] < ROBOT_RADIUS
        or position[1] > HEIGHT - ROBOT_RADIUS
    ):
        total += 100_000.0

    return float(total)


# ============================================================
# 7. MODEL PREDICTIVE CONTROL
# ============================================================

class MPCController:

    def __init__(
        self,
        horizon=MPC_HORIZON,
        candidates=MPC_CANDIDATES,
        seed=SEED
    ):
        self.horizon = horizon
        self.candidates = candidates

        self.rng = np.random.default_rng(seed)

        self.last_best_sequence = None
        self.last_cost = None

    def evaluate_sequence(
        self,
        position,
        velocity,
        target,
        sequence
    ):
        """
        Simulate a complete candidate action sequence.

        Only the first action will be applied to the real
        environment. The remaining actions are predictions.
        """

        predicted_position = position.copy()
        predicted_velocity = velocity.copy()

        phase = target.phase

        total_cost = 0.0

        for index, action in enumerate(sequence):

            predicted_position, predicted_velocity = (
                predict_robot_state(
                    predicted_position,
                    predicted_velocity,
                    action
                )
            )

            phase += TARGET_OMEGA * DT

            future_target = target.position_at(
                phase
            )

            cost = mpc_step_cost(
                predicted_position,
                predicted_velocity,
                future_target,
                action
            )

            # Slightly discount distant predicted costs.
            total_cost += (
                (0.98 ** index) * cost
            )

        return total_cost

    def control(
        self,
        position,
        velocity,
        target
    ):
        """
        Random-shooting MPC.

        Generate candidate action sequences, predict their
        consequences and choose the lowest-cost sequence.

        This is a practical introductory MPC implementation.
        It is not a globally optimal solver.
        """

        best_cost = float("inf")
        best_sequence = None

        # Candidate 1: maintain current velocity.
        maintain = np.zeros(
            (self.horizon, ACTION_DIM),
            dtype=np.float32
        )

        # Candidate 2: direct pursuit of current target.
        direct_action = safe_unit(
            target.position - position
        )

        direct_sequence = np.tile(
            direct_action,
            (self.horizon, 1)
        ).astype(np.float32)

        # Candidate 3: velocity damping.
        damping_action = np.clip(
            -velocity / MAX_ACCELERATION,
            -1.0,
            1.0
        )

        damping_sequence = np.tile(
            damping_action,
            (self.horizon, 1)
        ).astype(np.float32)

        sequences = [
            maintain,
            direct_sequence,
            damping_sequence
        ]

        # Random shooting candidates.
        random_sequences = self.rng.uniform(
            -1.0,
            1.0,
            size=(
                self.candidates,
                self.horizon,
                ACTION_DIM
            )
        ).astype(np.float32)

        sequences.extend(random_sequences)

        # Include the previous best sequence shifted by one
        # step. This warm-starts the next optimization.
        if self.last_best_sequence is not None:
            warm_start = np.vstack(
                [
                    self.last_best_sequence[1:],
                    self.last_best_sequence[-1]
                ]
            )

            sequences.append(warm_start)

        for sequence in sequences:
            cost = self.evaluate_sequence(
                position,
                velocity,
                target,
                sequence
            )

            if cost < best_cost:
                best_cost = cost
                best_sequence = sequence.copy()

        self.last_best_sequence = best_sequence
        self.last_cost = best_cost

        return np.clip(
            best_sequence[0],
            -1.0,
            1.0
        )


# ============================================================
# 8. SAFETY FILTER
# ============================================================

def safety_filter(position, velocity, action):
    """
    Simplified reactive obstacle filter.

    When close to an obstacle, remove the acceleration
    component that pushes the robot toward it.

    This is not a formal safety guarantee.
    """

    action = np.asarray(
        action,
        dtype=np.float32
    ).copy()

    relative = position - OBSTACLE

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

    return np.clip(action, -1.0, 1.0)


# ============================================================
# 9. TD3 REPLAY BUFFER
# ============================================================

class ReplayBuffer:

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(
        self,
        state,
        action,
        reward,
        next_state,
        terminated
    ):
        self.buffer.append(
            (
                np.asarray(state, dtype=np.float32),
                np.asarray(action, dtype=np.float32),
                float(reward),
                np.asarray(next_state, dtype=np.float32),
                float(terminated)
            )
        )

    def sample(self, batch_size):
        batch = random.sample(
            self.buffer,
            batch_size
        )

        states, actions, rewards, next_states, dones = zip(
            *batch
        )

        return (
            torch.tensor(
                np.array(states),
                dtype=torch.float32,
                device=DEVICE
            ),
            torch.tensor(
                np.array(actions),
                dtype=torch.float32,
                device=DEVICE
            ),
            torch.tensor(
                np.array(rewards),
                dtype=torch.float32,
                device=DEVICE
            ).unsqueeze(1),
            torch.tensor(
                np.array(next_states),
                dtype=torch.float32,
                device=DEVICE
            ),
            torch.tensor(
                np.array(dones),
                dtype=torch.float32,
                device=DEVICE
            ).unsqueeze(1)
        )

    def __len__(self):
        return len(self.buffer)


# ============================================================
# 10. TD3 NETWORKS
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
# 11. TD3 AGENT
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

    def select_action(self, state, noise=0.0):
        state_tensor = torch.tensor(
            state,
            dtype=torch.float32,
            device=DEVICE
        ).unsqueeze(0)

        with torch.no_grad():
            action = self.actor(
                state_tensor
            ).cpu().numpy()[0]

        if noise > 0:
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
            ).clamp(-1.0, 1.0)

            q1_target = self.critic1_target(
                next_states,
                next_actions
            )

            q2_target = self.critic2_target(
                next_states,
                next_actions
            )

            target_q = torch.minimum(
                q1_target,
                q2_target
            )

            target = (
                rewards
                + GAMMA * (1.0 - dones) * target_q
            )

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

        self.critic_optimizer.zero_grad()
        critic_loss.backward()

        nn.utils.clip_grad_norm_(
            list(self.critic1.parameters())
            + list(self.critic2.parameters()),
            1.0
        )

        self.critic_optimizer.step()

        self.training_steps += 1

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

        return float(critic_loss.item())


# ============================================================
# 12. HYBRID CONTROLLER
# ============================================================

def hybrid_action(
    mpc_action,
    td3_action,
    mpc_weight
):
    """
    Blend MPC and TD3 actions.

    weight = 1.0:
        Pure MPC.

    weight = 0.0:
        Pure TD3.

    Intermediate values:
        Hybrid controller.
    """

    weight = float(
        np.clip(mpc_weight, 0.0, 1.0)
    )

    action = (
        weight * np.asarray(mpc_action)
        + (1.0 - weight) * np.asarray(td3_action)
    )

    return np.clip(
        action,
        -1.0,
        1.0
    ).astype(np.float32)


# ============================================================
# 13. TRAIN TD3
# ============================================================

def train_td3():
    print("\nTraining TD3 agent...")
    print(f"Device: {DEVICE}")

    env = MovingTargetEnv()
    agent = TD3Agent()
    replay = ReplayBuffer(BUFFER_CAPACITY)

    rewards_history = []
    distance_history = []

    total_steps = 0

    for episode in range(TRAIN_EPISODES):

        state = env.reset()

        episode_reward = 0.0
        episode_distances = []

        for _ in range(MAX_STEPS_PER_EPISODE):

            if total_steps < WARMUP_STEPS:
                action = np.random.uniform(
                    -1.0,
                    1.0,
                    ACTION_DIM
                ).astype(np.float32)

            else:
                exploration = max(
                    0.05,
                    0.25 * (
                        1.0 - episode / TRAIN_EPISODES
                    )
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

            (
                next_state,
                reward,
                terminated,
                truncated,
                info
            ) = env.step(action)

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

            episode_reward += reward
            episode_distances.append(
                info["distance"]
            )

            total_steps += 1

            if terminated or truncated:
                break

        rewards_history.append(
            episode_reward
        )

        distance_history.append(
            float(np.mean(episode_distances))
        )

        if (episode + 1) % 10 == 0:
            print(
                f"Episode {episode + 1:3d}/"
                f"{TRAIN_EPISODES} | "
                f"Reward: "
                f"{np.mean(rewards_history[-10:]):8.2f} | "
                f"Distance: "
                f"{np.mean(distance_history[-10:]):7.2f}"
            )

    return agent, rewards_history, distance_history


# ============================================================
# 14. TRAINING VISUALIZATION
# ============================================================

def visualize_training(rewards, distances):
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(
            2,
            1,
            figsize=(10, 7)
        )

        axes[0].plot(
            rewards,
            alpha=0.4,
            label="Episode reward"
        )

        if len(rewards) >= 10:
            average = np.convolve(
                rewards,
                np.ones(10) / 10,
                mode="valid"
            )

            axes[0].plot(
                np.arange(9, len(rewards)),
                average,
                label="10-episode average"
            )

        axes[0].set_title("TD3 Training Reward")
        axes[0].set_xlabel("Episode")
        axes[0].set_ylabel("Reward")
        axes[0].grid(True)
        axes[0].legend()

        axes[1].plot(
            distances,
            label="Mean tracking distance"
        )

        axes[1].set_title("Tracking Distance")
        axes[1].set_xlabel("Episode")
        axes[1].set_ylabel("Distance")
        axes[1].grid(True)
        axes[1].legend()

        plt.tight_layout()
        plt.show()

    except ImportError:
        print("Matplotlib unavailable; skipping plots.")


# ============================================================
# 15. PYGAME ROCKET SIMULATOR
# ============================================================

class RocketGUI:

    def __init__(
        self,
        agent,
        rewards,
        distances
    ):
        pygame.init()

        self.screen = pygame.display.set_mode(
            (WIDTH + PANEL_WIDTH, HEIGHT)
        )

        pygame.display.set_caption(
            "Phase 4.13 — MPC + TD3 Hybrid Control"
        )

        self.clock = pygame.time.Clock()

        self.font = pygame.font.SysFont(
            "consolas",
            16
        )

        self.small_font = pygame.font.SysFont(
            "consolas",
            13
        )

        self.agent = agent
        self.rewards = rewards
        self.distances = distances

        self.env = MovingTargetEnv()
        self.env.reset()

        self.mpc = MPCController()

        # 1 = TD3
        # 2 = MPC
        # 3 = Hybrid
        self.mode = 3

        self.paused = False
        self.speed = 1

        self.mpc_weight = HYBRID_MPC_WEIGHT

        self.horizon = MPC_HORIZON

        self.last_action = np.zeros(
            ACTION_DIM,
            dtype=np.float32
        )

        self.last_mpc_cost = 0.0

        self.running = True

    def reset(self):
        self.env.reset()

        self.mpc.last_best_sequence = None
        self.mpc.last_cost = None

        self.last_action[:] = 0.0

    def select_action(self):
        state = self.env.get_state()

        td3 = self.agent.select_action(
            state,
            noise=0.0
        )

        # Update MPC horizon from interactive controls.
        self.mpc.horizon = self.horizon

        mpc = self.mpc.control(
            self.env.position,
            self.env.velocity,
            self.env.target
        )

        if self.mode == 1:
            action = td3

        elif self.mode == 2:
            action = mpc

        else:
            action = hybrid_action(
                mpc,
                td3,
                self.mpc_weight
            )

        action = safety_filter(
            self.env.position,
            self.env.velocity,
            action
        )

        self.last_action = action

        self.last_mpc_cost = (
            self.mpc.last_cost
            if self.mpc.last_cost is not None
            else 0.0
        )

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

        # Conservative check for the entire target orbit.
        orbit_radius = float(
            np.linalg.norm(TARGET_AMPLITUDE)
        )

        minimum_clearance = (
            distance(candidate, OBSTACLE)
            - orbit_radius
        )

        if minimum_clearance < (
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
            dtype=np.float32
        )

        end = np.asarray(
            end,
            dtype=np.float32
        )

        pygame.draw.line(
            self.screen,
            color,
            tuple(start.astype(int)),
            tuple(end.astype(int)),
            width
        )

        direction = safe_unit(end - start)

        if np.linalg.norm(direction) < 1e-6:
            return

        perpendicular = np.array(
            [-direction[1], direction[0]]
        )

        tip = end

        p1 = (
            tip
            - direction * 10
            + perpendicular * 5
        )

        p2 = (
            tip
            - direction * 10
            - perpendicular * 5
        )

        pygame.draw.polygon(
            self.screen,
            color,
            [
                tip.astype(int),
                p1.astype(int),
                p2.astype(int)
            ]
        )

    def draw_rocket(self):
        position = self.env.position
        velocity = self.env.velocity

        direction = safe_unit(velocity)

        if np.linalg.norm(direction) < 0.1:
            direction = safe_unit(
                self.env.target.position - position
            )

        if np.linalg.norm(direction) < 0.1:
            direction = np.array(
                [1.0, 0.0]
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

        exhaust_length = random.randint(10, 18)

        exhaust_end = (
            rear - direction * exhaust_length
        )

        pygame.draw.line(
            self.screen,
            (255, 125, 30),
            tuple(rear.astype(int)),
            tuple(exhaust_end.astype(int)),
            5
        )

        pygame.draw.polygon(
            self.screen,
            (205, 215, 230),
            [
                nose.astype(int),
                body_left.astype(int),
                left_fin.astype(int),
                rear.astype(int),
                right_fin.astype(int),
                body_right.astype(int)
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

        for index in range(100):
            phase = (
                self.env.target.phase
                + index * 0.08
            )

            point = self.env.target.position_at(
                phase
            )

            points.append(
                tuple(point.astype(int))
            )

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

        # Obstacle safety zone.
        pygame.draw.circle(
            self.screen,
            (110, 55, 55),
            OBSTACLE.astype(int),
            int(
                OBSTACLE_RADIUS + SAFETY_MARGIN
            ),
            2
        )

        pygame.draw.circle(
            self.screen,
            (170, 65, 65),
            OBSTACLE.astype(int),
            int(OBSTACLE_RADIUS)
        )

        # Moving target.
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

        self.draw_arrow(
            target,
            target + self.env.target.velocity * 0.35,
            (100, 240, 130)
        )

        # Rocket velocity.
        self.draw_arrow(
            self.env.position,
            self.env.position + self.env.velocity * 0.35,
            (255, 210, 90)
        )

        # First MPC predicted action position.
        predicted_position, _ = predict_robot_state(
            self.env.position,
            self.env.velocity,
            self.last_action
        )

        pygame.draw.circle(
            self.screen,
            (100, 150, 250),
            predicted_position.astype(int),
            5,
            1
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

        mode_names = {
            1: "TD3",
            2: "MPC",
            3: "HYBRID MPC + TD3"
        }

        target_distance = distance(
            self.env.position,
            self.env.target.position
        )

        lines = [
            ("PHASE 4.13", (255, 220, 100)),
            ("MPC + RL Hybrid Control", (220, 230, 245)),
            ("", (200, 200, 200)),
            ("CONTROLLER", (150, 190, 255)),
            ("1  TD3 learned policy", (220, 220, 220)),
            ("2  Model Predictive Control", (220, 220, 220)),
            ("3  Hybrid MPC + TD3", (220, 220, 220)),
            ("", (200, 200, 200)),
            (
                f"Mode: {mode_names[self.mode]}",
                (100, 240, 150)
            ),
            ("", (200, 200, 200)),
            ("TRACKING", (150, 190, 255)),
            (
                f"Distance: {target_distance:.1f}",
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
            ("", (200, 200, 200)),
            ("MPC", (150, 190, 255)),
            (
                f"Horizon: {self.horizon} steps",
                (220, 220, 220)
            ),
            (
                f"Prediction time: {self.horizon * DT:.1f}s",
                (220, 220, 220)
            ),
            (
                f"Hybrid MPC weight: {self.mpc_weight:.2f}",
                (220, 220, 220)
            ),
            (
                f"Predicted cost: {self.last_mpc_cost:.0f}",
                (220, 220, 220)
            ),
            ("", (200, 200, 200)),
            ("ROCKET", (150, 190, 255)),
            (
                f"Speed: {np.linalg.norm(self.env.velocity):.1f}",
                (220, 220, 220)
            ),
            (
                f"Action X: {self.last_action[0]:+.2f}",
                (220, 220, 220)
            ),
            (
                f"Action Y: {self.last_action[1]:+.2f}",
                (220, 220, 220)
            ),
            ("", (200, 200, 200)),
            ("SIMULATION", (150, 190, 255)),
            (
                f"Simulation speed: {self.speed}x",
                (220, 220, 220)
            ),
            (
                "PAUSED" if self.paused else "RUNNING",
                (255, 190, 100)
            ),
            ("", (200, 200, 200)),
            ("CONTROLS", (150, 190, 255)),
            ("Mouse: move target center", (190, 190, 190)),
            ("SPACE: pause", (190, 190, 190)),
            ("R: reset", (190, 190, 190)),
            ("UP/DOWN: MPC horizon", (190, 190, 190)),
            ("LEFT/RIGHT: MPC weight", (190, 190, 190)),
            ("+/-: simulation speed", (190, 190, 190)),
            ("ESC: exit", (190, 190, 190))
        ]

        y = 12

        for text, color in lines:
            if text:
                surface = self.small_font.render(
                    text,
                    True,
                    color
                )

                self.screen.blit(
                    surface,
                    (x + 12, y)
                )

            y += 15

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
                        25,
                        self.horizon + 1
                    )

                elif event.key == pygame.K_DOWN:
                    self.horizon = max(
                        3,
                        self.horizon - 1
                    )

                elif event.key == pygame.K_RIGHT:
                    self.mpc_weight = min(
                        1.0,
                        self.mpc_weight + 0.05
                    )

                elif event.key == pygame.K_LEFT:
                    self.mpc_weight = max(
                        0.0,
                        self.mpc_weight - 0.05
                    )

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if (
                    event.button == 1
                    and event.pos[0] < WIDTH
                ):
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
# 16. UNIT TESTS
# ============================================================

def run_tests():
    print("\nRunning Phase 4.13 tests...\n")

    # Test 1: Robot dynamics return 2D position and velocity.
    p, v = predict_robot_state(
        np.array([100.0, 100.0]),
        np.zeros(2),
        np.array([0.5, 0.0])
    )

    assert p.shape == (2,)
    assert v.shape == (2,)
    print("Test 1 passed: robot dynamics")

    # Test 2: Zero action preserves velocity.
    p, v = predict_robot_state(
        np.array([100.0, 100.0]),
        np.array([10.0, 5.0]),
        np.zeros(2)
    )

    assert np.allclose(v, [10.0, 5.0])
    print("Test 2 passed: zero-action velocity")

    # Test 3: Action bounds are enforced.
    _, v = predict_robot_state(
        np.array([100.0, 100.0]),
        np.zeros(2),
        np.array([5.0, -5.0])
    )

    assert np.linalg.norm(v) <= MAX_SPEED + 1e-5
    print("Test 3 passed: maximum velocity")

    # Test 4: Moving target has 2D position.
    target = MovingTarget()

    assert target.position.shape == (2,)
    print("Test 4 passed: moving target position")

    # Test 5: Target velocity has 2D shape.
    assert target.velocity.shape == (2,)
    print("Test 5 passed: moving target velocity")

    # Test 6: Target prediction changes with phase.
    initial = target.position.copy()

    target.step()

    assert not np.allclose(
        initial,
        target.position
    )

    print("Test 6 passed: moving target update")

    # Test 7: MPC cost is finite in a safe location.
    cost = mpc_step_cost(
        np.array([100.0, 100.0]),
        np.zeros(2),
        np.array([200.0, 200.0]),
        np.zeros(2)
    )

    assert np.isfinite(cost)
    print("Test 7 passed: MPC cost")

    # Test 8: Obstacle collision has a high cost.
    collision_cost = mpc_step_cost(
        OBSTACLE.copy(),
        np.zeros(2),
        np.array([500.0, 500.0]),
        np.zeros(2)
    )

    assert collision_cost > 100_000.0
    print("Test 8 passed: collision penalty")

    # Test 9: MPC produces bounded actions.
    mpc = MPCController(
        horizon=5,
        candidates=10
    )

    action = mpc.control(
        np.array([100.0, 440.0]),
        np.zeros(2),
        target
    )

    assert action.shape == (ACTION_DIM,)
    assert np.all(action >= -1.0)
    assert np.all(action <= 1.0)

    print("Test 9 passed: MPC action bounds")

    # Test 10: MPC remembers its best sequence.
    assert mpc.last_best_sequence.shape == (
        5,
        ACTION_DIM
    )

    print("Test 10 passed: MPC warm start")

    # Test 11: Environment observation dimensions.
    env = MovingTargetEnv()

    state = env.reset()

    assert state.shape == (STATE_DIM,)
    print("Test 11 passed: observation dimensions")

    # Test 12: Environment step structure.
    result = env.step(np.zeros(ACTION_DIM))

    assert len(result) == 5
    assert result[0].shape == (STATE_DIM,)
    assert np.isfinite(result[1])

    print("Test 12 passed: environment step")

    # Test 13: Replay buffer.
    replay = ReplayBuffer(100)

    replay.add(
        np.zeros(STATE_DIM),
        np.zeros(ACTION_DIM),
        1.0,
        np.ones(STATE_DIM),
        False
    )

    assert len(replay) == 1
    print("Test 13 passed: replay buffer")

    # Test 14: Replay batch dimensions.
    for _ in range(BATCH_SIZE):
        replay.add(
            np.zeros(STATE_DIM),
            np.zeros(ACTION_DIM),
            1.0,
            np.ones(STATE_DIM),
            False
        )

    batch = replay.sample(BATCH_SIZE)

    assert batch[0].shape == (
        BATCH_SIZE,
        STATE_DIM
    )

    assert batch[1].shape == (
        BATCH_SIZE,
        ACTION_DIM
    )

    assert batch[2].shape == (
        BATCH_SIZE,
        1
    )

    print("Test 14 passed: replay batch")

    # Test 15: TD3 actor output.
    actor = Actor().to(DEVICE)

    with torch.no_grad():
        output = actor(
            torch.zeros(
                (4, STATE_DIM),
                device=DEVICE
            )
        )

    assert output.shape == (4, ACTION_DIM)
    assert torch.all(output <= 1.0)
    assert torch.all(output >= -1.0)

    print("Test 15 passed: actor output")

    # Test 16: TD3 critic output.
    critic = Critic().to(DEVICE)

    with torch.no_grad():
        q = critic(
            torch.zeros(
                (4, STATE_DIM),
                device=DEVICE
            ),
            torch.zeros(
                (4, ACTION_DIM),
                device=DEVICE
            )
        )

    assert q.shape == (4, 1)
    print("Test 16 passed: critic output")

    # Test 17: Hybrid action endpoints.
    mpc_action = np.array([1.0, 0.0])
    td3_action = np.array([0.0, 1.0])

    assert np.allclose(
        hybrid_action(mpc_action, td3_action, 1.0),
        mpc_action
    )

    assert np.allclose(
        hybrid_action(mpc_action, td3_action, 0.0),
        td3_action
    )

    print("Test 17 passed: hybrid action endpoints")

    # Test 18: Hybrid action midpoint.
    midpoint = hybrid_action(
        mpc_action,
        td3_action,
        0.5
    )

    assert np.allclose(
        midpoint,
        [0.5, 0.5]
    )

    print("Test 18 passed: hybrid action blending")

    # Test 19: Safety filter preserves action bounds.
    filtered = safety_filter(
        np.array([300.0, 285.0]),
        np.zeros(2),
        np.array([-1.0, 0.5])
    )

    assert np.all(filtered >= -1.0)
    assert np.all(filtered <= 1.0)

    print("Test 19 passed: safety filter")

    # Test 20: TD3 agent returns valid actions.
    agent = TD3Agent()

    action = agent.select_action(
        np.zeros(STATE_DIM, dtype=np.float32)
    )

    assert action.shape == (ACTION_DIM,)
    assert np.all(action >= -1.0)
    assert np.all(action <= 1.0)

    print("Test 20 passed: TD3 action")

    print("\nAll 20 tests passed.\n")


# ============================================================
# 17. MAIN
# ============================================================

def main():
    print("=" * 65)
    print("PHASE 4.13 — MPC AND RL HYBRID CONTROL")
    print("=" * 65)

    run_tests()

    agent, rewards, distances = train_td3()

    visualize_training(
        rewards,
        distances
    )

    print("\nStarting Pygame simulation...")
    print("Press 1 for TD3, 2 for MPC, 3 for Hybrid.")

    gui = RocketGUI(
        agent,
        rewards,
        distances
    )

    gui.run()


if __name__ == "__main__":
    main()
