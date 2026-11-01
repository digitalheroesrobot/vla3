
"""
RL COURSE — PHASE 4.14
============================================================
Filename:
    phase4/safe_rl_cbf.py

LESSON:
    Safe Reinforcement Learning and Control Barrier Functions

PURPOSE
-------
Learn how to constrain an RL agent's actions to reduce the
risk of collisions and boundary violations.

WHAT YOU LEARN
--------------
STEP 1 — Safety constraints
    Define safe operating regions for a robot.

STEP 2 — Control Barrier Functions
    Represent obstacle clearance and boundary clearance
    mathematically.

STEP 3 — Safety filter
    Modify an RL action when it violates a safety constraint.

STEP 4 — TD3
    Train a continuous-action RL agent.

STEP 5 — Safe RL
    Apply the CBF safety filter during training and evaluation.

STEP 6 — Comparison
    Compare raw TD3 and CBF-filtered TD3.

STEP 7 — Pygame
    Visualize the rocket, moving target, obstacle, safety
    boundaries, and filtered actions.

MATHEMATICAL FOUNDATION
-----------------------

Obstacle barrier:
    h(x) = ||p - p_obstacle||^2 - R_safe^2

A safe state satisfies:
    h(x) >= 0

Continuous-time CBF condition:
    dh/dt + alpha*h >= 0

For a moving robot:
    dh/dt = 2*(p-p_obstacle).(v)

For this lesson, we use a discrete acceleration-level
approximation:

    r . (v + a*dt) >= -k*h

where:
    r = p - p_obstacle
    a = MAX_ACCELERATION * action

This produces a linear inequality in the normalized action.

Boundary constraints use analogous one-dimensional
barrier inequalities.

The safety filter projects the requested action onto
the intersection of the linear safety constraints.

IMPORTANT LIMITATIONS
---------------------
This is an educational, approximate CBF implementation.

It does not guarantee safety when:
    - The robot is already outside the safe set.
    - Constraints are mutually infeasible.
    - The dynamics model is inaccurate.
    - The robot has insufficient acceleration authority.
    - There are unmodeled delays or disturbances.

A production robot requires a validated safety architecture.

ROBOTICS APPLICATION
--------------------
- Autonomous mobile robot collision avoidance
- Drone obstacle avoidance
- Safe manipulation
- Human-robot interaction
- Autonomous vehicle motion constraints

CONTROLS
--------
1       Raw TD3
2       Safe TD3 with CBF
SPACE   Pause / resume
R       Reset
+ / -   Simulation speed
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

SEED = 44

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

WIDTH = 720
HEIGHT = 580
PANEL_WIDTH = 310

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

# The CBF obstacle radius includes the robot radius and
# an additional clearance margin.
SAFE_OBSTACLE_RADIUS = (
    OBSTACLE_RADIUS
    + ROBOT_RADIUS
    + SAFETY_MARGIN
)

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

# CBF parameters.
CBF_ALPHA = 2.0
CBF_PROJECTION_ITERATIONS = 30
CBF_TOLERANCE = 1e-5


# ============================================================
# 2. COMMON UTILITIES
# ============================================================

def clamp_vector(vector, maximum):
    vector = np.asarray(
        vector,
        dtype=np.float32
    )

    norm = float(np.linalg.norm(vector))

    if norm > maximum:
        vector = vector * (
            maximum / (norm + 1e-8)
        )

    return vector


def safe_unit(vector):
    vector = np.asarray(
        vector,
        dtype=np.float32
    )

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
# 4. ROBOT DYNAMICS
# ============================================================

def predict_robot_state(
    position,
    velocity,
    action
):
    action = np.clip(
        np.asarray(action, dtype=np.float32),
        -1.0,
        1.0
    )

    acceleration = (
        action * MAX_ACCELERATION
    )

    next_velocity = (
        velocity + acceleration * DT
    )

    next_velocity = clamp_vector(
        next_velocity,
        MAX_SPEED
    )

    next_position = (
        position + next_velocity * DT
    )

    return next_position, next_velocity


# ============================================================
# 5. CONTROL BARRIER FUNCTION CONSTRAINTS
# ============================================================

def obstacle_barrier(position):
    """
    h_obstacle = ||p - obstacle||^2 - R_safe^2

    Safe region:
        h_obstacle >= 0

    The safe radius includes:
        obstacle radius
        robot radius
        safety margin
    """

    relative = (
        np.asarray(position, dtype=np.float32)
        - OBSTACLE
    )

    return float(
        np.dot(relative, relative)
        - SAFE_OBSTACLE_RADIUS ** 2
    )


def boundary_barriers(position):
    """
    Four boundary barrier functions.

    Left:
        h_left = x - robot_radius

    Right:
        h_right = width - robot_radius - x

    Top:
        h_top = y - robot_radius

    Bottom:
        h_bottom = height - robot_radius - y
    """

    x, y = position

    return {
        "left": float(x - ROBOT_RADIUS),
        "right": float(WIDTH - ROBOT_RADIUS - x),
        "top": float(y - ROBOT_RADIUS),
        "bottom": float(HEIGHT - ROBOT_RADIUS - y)
    }


def build_cbf_constraints(position, velocity):
    """
    Construct linear inequalities:

        c_i . action >= b_i

    The normalized action is in [-1,1]^2.

    Obstacle:
        r . (v + A*dt*u) >= -alpha*h

    Rearranging:
        (A*dt*r) . u >= -alpha*h - r.v

    Boundaries are derived from the corresponding
    one-dimensional barrier functions.
    """

    position = np.asarray(
        position,
        dtype=np.float32
    )

    velocity = np.asarray(
        velocity,
        dtype=np.float32
    )

    constraints = []

    # --------------------------------------------------------
    # Obstacle constraint
    # --------------------------------------------------------

    relative = position - OBSTACLE

    h = obstacle_barrier(position)

    coefficient = (
        MAX_ACCELERATION * DT * relative
    )

    rhs = (
        -CBF_ALPHA * h
        - float(np.dot(relative, velocity))
    )

    constraints.append(
        (coefficient.astype(np.float32), float(rhs), "obstacle")
    )

    # --------------------------------------------------------
    # Left boundary
    # --------------------------------------------------------

    h_left = position[0] - ROBOT_RADIUS

    coefficient = np.array(
        [MAX_ACCELERATION * DT, 0.0],
        dtype=np.float32
    )

    rhs = (
        -CBF_ALPHA * h_left
        - velocity[0]
    )

    constraints.append(
        (coefficient, float(rhs), "left")
    )

    # --------------------------------------------------------
    # Right boundary
    # --------------------------------------------------------

    h_right = (
        WIDTH - ROBOT_RADIUS - position[0]
    )

    coefficient = np.array(
        [-MAX_ACCELERATION * DT, 0.0],
        dtype=np.float32
    )

    rhs = (
        -CBF_ALPHA * h_right
        + velocity[0]
    )

    constraints.append(
        (coefficient, float(rhs), "right")
    )

    # --------------------------------------------------------
    # Top boundary
    # --------------------------------------------------------

    h_top = position[1] - ROBOT_RADIUS

    coefficient = np.array(
        [0.0, MAX_ACCELERATION * DT],
        dtype=np.float32
    )

    rhs = (
        -CBF_ALPHA * h_top
        - velocity[1]
    )

    constraints.append(
        (coefficient, float(rhs), "top")
    )

    # --------------------------------------------------------
    # Bottom boundary
    # --------------------------------------------------------

    h_bottom = (
        HEIGHT - ROBOT_RADIUS - position[1]
    )

    coefficient = np.array(
        [0.0, -MAX_ACCELERATION * DT],
        dtype=np.float32
    )

    rhs = (
        -CBF_ALPHA * h_bottom
        + velocity[1]
    )

    constraints.append(
        (coefficient, float(rhs), "bottom")
    )

    return constraints


# ============================================================
# 6. CBF ACTION PROJECTION
# ============================================================

def project_action_to_safety(
    requested_action,
    position,
    velocity,
    iterations=CBF_PROJECTION_ITERATIONS
):
    """
    Find an action close to the requested RL action while
    satisfying linear CBF inequalities.

    Optimization objective:
        minimize ||u - u_requested||^2

    Subject to:
        c_i . u >= b_i
        -1 <= u <= 1

    We use iterative projection onto half-spaces.

    This is an approximate projection algorithm. If the
    constraints are infeasible, the result may not satisfy
    every constraint.
    """

    requested_action = np.clip(
        np.asarray(
            requested_action,
            dtype=np.float32
        ),
        -1.0,
        1.0
    )

    action = requested_action.copy()

    constraints = build_cbf_constraints(
        position,
        velocity
    )

    for _ in range(iterations):

        changed = False

        for coefficient, rhs, _name in constraints:

            value = float(
                np.dot(coefficient, action)
            )

            violation = rhs - value

            norm_squared = float(
                np.dot(coefficient, coefficient)
            )

            if violation > CBF_TOLERANCE:

                if norm_squared < 1e-12:
                    continue

                action += (
                    violation / norm_squared
                ) * coefficient

                action = np.clip(
                    action,
                    -1.0,
                    1.0
                )

                changed = True

        if not changed:
            break

    return action.astype(np.float32)


def check_cbf_constraints(
    action,
    position,
    velocity,
    tolerance=1e-3
):
    constraints = build_cbf_constraints(
        position,
        velocity
    )

    results = {}

    for coefficient, rhs, name in constraints:
        value = float(
            np.dot(coefficient, action)
        )

        results[name] = value >= rhs - tolerance

    return results


def cbf_action_filter(
    requested_action,
    position,
    velocity
):
    filtered_action = project_action_to_safety(
        requested_action,
        position,
        velocity
    )

    results = check_cbf_constraints(
        filtered_action,
        position,
        velocity
    )

    all_satisfied = all(results.values())

    changed = not np.allclose(
        requested_action,
        filtered_action,
        atol=1e-4
    )

    return (
        filtered_action,
        {
            "constraints": results,
            "all_satisfied": all_satisfied,
            "changed": changed
        }
    )


# ============================================================
# 7. ENVIRONMENT
# ============================================================

class MovingTargetEnv:

    def __init__(self):
        self.target = MovingTarget()

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

        next_phase = (
            self.target.phase + TARGET_OMEGA * DT
        )

        target_at_next_time = (
            self.target.position_at(next_phase)
        )

        distance_before = distance(
            self.position,
            target_at_next_time
        )

        self.position, self.velocity = (
            predict_robot_state(
                self.position,
                self.velocity,
                action
            )
        )

        self.target.step()

        distance_after = distance(
            self.position,
            self.target.position
        )

        progress = (
            distance_before - distance_after
        )

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

        collision = (
            obstacle_distance
            < OBSTACLE_RADIUS + ROBOT_RADIUS
        )

        outside = (
            self.position[0] < ROBOT_RADIUS
            or self.position[0] > WIDTH - ROBOT_RADIUS
            or self.position[1] < ROBOT_RADIUS
            or self.position[1] > HEIGHT - ROBOT_RADIUS
        )

        if collision:
            reward -= 100.0
            terminated = True

        if outside:
            reward -= 100.0
            terminated = True

        self.step_count += 1

        truncated = (
            self.step_count >= MAX_STEPS_PER_EPISODE
        )

        info = {
            "distance": distance_after,
            "progress": progress,
            "collision": collision,
            "outside": outside
        }

        return (
            self.get_state(),
            float(reward),
            terminated,
            truncated,
            info
        )


# ============================================================
# 8. TD3 REPLAY BUFFER
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
# 9. TD3 NETWORKS
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
# 10. TD3 AGENT
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
# 11. TRAINING
# ============================================================

def train_td3():
    print("\nTraining Safe TD3...")
    print(f"Device: {DEVICE}")

    env = MovingTargetEnv()
    agent = TD3Agent()

    replay = ReplayBuffer(
        BUFFER_CAPACITY
    )

    rewards_history = []
    distances_history = []

    safety_modifications = 0
    total_steps = 0

    for episode in range(TRAIN_EPISODES):

        state = env.reset()

        episode_reward = 0.0
        episode_distances = []

        for _ in range(MAX_STEPS_PER_EPISODE):

            if total_steps < WARMUP_STEPS:
                requested_action = np.random.uniform(
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

                requested_action = agent.select_action(
                    state,
                    noise=exploration
                )

            # Apply CBF during training.
            safe_action, safety_info = cbf_action_filter(
                requested_action,
                env.position,
                env.velocity
            )

            if safety_info["changed"]:
                safety_modifications += 1

            next_state, reward, terminated, truncated, info = (
                env.step(safe_action)
            )

            # Store the action actually executed by the environment.
            replay.add(
                state,
                safe_action,
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

        distances_history.append(
            float(np.mean(episode_distances))
        )

        if (episode + 1) % 10 == 0:
            print(
                f"Episode {episode + 1:3d}/"
                f"{TRAIN_EPISODES} | "
                f"Reward: "
                f"{np.mean(rewards_history[-10:]):8.2f} | "
                f"Distance: "
                f"{np.mean(distances_history[-10:]):7.2f} | "
                f"Safety interventions: "
                f"{safety_modifications}"
            )

    return (
        agent,
        rewards_history,
        distances_history
    )


# ============================================================
# 12. TRAINING VISUALIZATION
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

        axes[0].set_title("Safe TD3 Training Reward")
        axes[0].set_xlabel("Episode")
        axes[0].set_ylabel("Reward")
        axes[0].legend()
        axes[0].grid(True)

        axes[1].plot(
            distances,
            label="Mean tracking distance"
        )

        axes[1].set_title("Moving-Target Tracking Distance")
        axes[1].set_xlabel("Episode")
        axes[1].set_ylabel("Distance")
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        plt.show()

    except ImportError:
        print("Matplotlib unavailable; skipping plots.")


# ============================================================
# 13. PYGAME VISUALIZATION
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
            "Phase 4.14 — Safe RL with CBF"
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

        # 1 = Raw TD3
        # 2 = Safe TD3
        self.mode = 2

        self.paused = False
        self.speed = 1

        self.last_requested_action = np.zeros(
            ACTION_DIM,
            dtype=np.float32
        )

        self.last_safe_action = np.zeros(
            ACTION_DIM,
            dtype=np.float32
        )

        self.last_safety_info = {
            "constraints": {},
            "all_satisfied": True,
            "changed": False
        }

        self.running = True

        self.interventions = 0
        self.total_actions = 0

    def reset(self):
        self.env.reset()

        self.last_requested_action[:] = 0.0
        self.last_safe_action[:] = 0.0

        self.interventions = 0
        self.total_actions = 0

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

        self.env.target.set_center(
            candidate
        )

    def select_action(self):
        state = self.env.get_state()

        requested = self.agent.select_action(
            state,
            noise=0.0
        )

        if self.mode == 1:
            action = requested

            safety_info = {
                "changed": False,
                "all_satisfied": False,
                "constraints": {}
            }

        else:
            action, safety_info = cbf_action_filter(
                requested,
                self.env.position,
                self.env.velocity
            )

        self.last_requested_action = requested.copy()
        self.last_safe_action = action.copy()
        self.last_safety_info = safety_info

        self.total_actions += 1

        if safety_info["changed"]:
            self.interventions += 1

        return action

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

        direction = safe_unit(
            end - start
        )

        if np.linalg.norm(direction) < 1e-6:
            return

        perpendicular = np.array(
            [-direction[1], direction[0]]
        )

        p1 = (
            end
            - direction * 10
            + perpendicular * 5
        )

        p2 = (
            end
            - direction * 10
            - perpendicular * 5
        )

        pygame.draw.polygon(
            self.screen,
            color,
            [
                end.astype(int),
                p1.astype(int),
                p2.astype(int)
            ]
        )

    def draw_rocket(self):
        position = self.env.position
        velocity = self.env.velocity

        direction = safe_unit(
            velocity
        )

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

        exhaust_length = random.randint(
            10,
            18
        )

        pygame.draw.line(
            self.screen,
            (255, 125, 30),
            tuple(rear.astype(int)),
            tuple(
                (
                    rear - direction * exhaust_length
                ).astype(int)
            ),
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

        # World boundary.
        pygame.draw.rect(
            self.screen,
            (90, 110, 140),
            (0, 0, WIDTH, HEIGHT),
            2
        )

        # Obstacle safe zone.
        pygame.draw.circle(
            self.screen,
            (115, 75, 55),
            OBSTACLE.astype(int),
            int(SAFE_OBSTACLE_RADIUS),
            2
        )

        # Physical obstacle.
        pygame.draw.circle(
            self.screen,
            (170, 65, 65),
            OBSTACLE.astype(int),
            int(OBSTACLE_RADIUS)
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

        self.draw_arrow(
            target,
            target + self.env.target.velocity * 0.35,
            (100, 240, 130)
        )

        # Requested action vector.
        self.draw_arrow(
            self.env.position,
            self.env.position
            + self.last_requested_action * 45,
            (230, 100, 100),
            2
        )

        # Filtered action vector.
        self.draw_arrow(
            self.env.position,
            self.env.position
            + self.last_safe_action * 65,
            (100, 170, 255),
            3
        )

        # Rocket velocity.
        self.draw_arrow(
            self.env.position,
            self.env.position + self.env.velocity * 0.35,
            (255, 210, 90),
            2
        )

        self.draw_rocket()

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

        mode_name = (
            "RAW TD3"
            if self.mode == 1
            else "SAFE TD3 + CBF"
        )

        intervention_rate = (
            100.0 * self.interventions
            / max(1, self.total_actions)
        )

        lines = [
            ("PHASE 4.14", (255, 220, 100)),
            ("Safe RL + Control Barrier", (220, 230, 245)),
            ("Functions", (220, 230, 245)),
            ("", (200, 200, 200)),
            ("CONTROLLER", (150, 190, 255)),
            ("1  Raw TD3", (220, 220, 220)),
            ("2  Safe TD3 + CBF", (220, 220, 220)),
            ("", (200, 200, 200)),
            (f"Mode: {mode_name}", (100, 240, 150)),
            ("", (200, 200, 200)),
            ("TRACKING", (150, 190, 255)),
            (f"Distance: {target_distance:.1f}", (230, 230, 230)),
            (f"Target vx: {self.env.target.velocity[0]:.1f}", (180, 240, 180)),
            (f"Target vy: {self.env.target.velocity[1]:.1f}", (180, 240, 180)),
            ("", (200, 200, 200)),
            ("CBF SAFETY FILTER", (150, 190, 255)),
            (f"Obstacle h: {obstacle_barrier(self.env.position):.0f}", (220, 220, 220)),
            (f"Safety modified: {self.last_safety_info['changed']}", (220, 220, 220)),
            (f"Constraints pass: {self.last_safety_info['all_satisfied']}", (220, 220, 220)),
            (f"Interventions: {self.interventions}", (220, 220, 220)),
            (f"Intervention rate: {intervention_rate:.1f}%", (220, 220, 220)),
            ("", (200, 200, 200)),
            ("ACTIONS", (150, 190, 255)),
            (f"Requested X: {self.last_requested_action[0]:+.2f}", (230, 140, 140)),
            (f"Requested Y: {self.last_requested_action[1]:+.2f}", (230, 140, 140)),
            (f"Executed X: {self.last_safe_action[0]:+.2f}", (130, 190, 255)),
            (f"Executed Y: {self.last_safe_action[1]:+.2f}", (130, 190, 255)),
            ("", (200, 200, 200)),
            ("SIMULATION", (150, 190, 255)),
            (f"Speed: {self.speed}x", (220, 220, 220)),
            ("PAUSED" if self.paused else "RUNNING", (255, 190, 100)),
            ("", (200, 200, 200)),
            ("CONTROLS", (150, 190, 255)),
            ("Mouse: move target center", (190, 190, 190)),
            ("SPACE: pause / resume", (190, 190, 190)),
            ("R: reset", (190, 190, 190)),
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
# 14. UNIT TESTS
# ============================================================

def run_tests():
    print("\nRunning Phase 4.14 tests...\n")

    # Test 1: Obstacle barrier is positive in a safe location.
    safe_position = np.array(
        [100.0, 100.0],
        dtype=np.float32
    )

    assert obstacle_barrier(safe_position) > 0.0
    print("Test 1 passed: obstacle safe region")

    # Test 2: Obstacle barrier is negative inside the safe radius.
    unsafe_position = OBSTACLE + np.array(
        [10.0, 0.0],
        dtype=np.float32
    )

    assert obstacle_barrier(unsafe_position) < 0.0
    print("Test 2 passed: obstacle unsafe region")

    # Test 3: Boundary barriers are positive at the center.
    barriers = boundary_barriers(
        np.array([360.0, 290.0])
    )

    assert all(value > 0 for value in barriers.values())
    print("Test 3 passed: boundary barriers")

    # Test 4: Boundary barrier detects an outside position.
    barriers = boundary_barriers(
        np.array([-10.0, 100.0])
    )

    assert barriers["left"] < 0.0
    print("Test 4 passed: outside boundary detection")

    # Test 5: Constraint builder returns five constraints.
    constraints = build_cbf_constraints(
        np.array([100.0, 100.0]),
        np.zeros(2)
    )

    assert len(constraints) == 5
    print("Test 5 passed: CBF constraint construction")

    # Test 6: All constraint coefficients are two-dimensional.
    for coefficient, rhs, name in constraints:
        assert coefficient.shape == (ACTION_DIM,)
        assert np.isfinite(rhs)

    print("Test 6 passed: CBF constraint dimensions")

    # Test 7: Action projection preserves bounds.
    requested = np.array(
        [1.0, -1.0],
        dtype=np.float32
    )

    filtered = project_action_to_safety(
        requested,
        np.array([100.0, 100.0]),
        np.zeros(2)
    )

    assert np.all(filtered >= -1.0)
    assert np.all(filtered <= 1.0)
    print("Test 7 passed: projected action bounds")

    # Test 8: Safe action is unchanged in a distant safe state
    # when the requested action already satisfies all constraints.
    requested = np.array(
        [0.0, 0.0],
        dtype=np.float32
    )

    filtered = project_action_to_safety(
        requested,
        np.array([100.0, 100.0]),
        np.zeros(2)
    )

    assert np.allclose(
        filtered,
        requested,
        atol=1e-4
    )

    print("Test 8 passed: no unnecessary intervention")

    # Test 9: Safe action satisfies CBF constraints in a
    # representative near-obstacle state with feasible braking.
    position = np.array(
        [OBSTACLE[0] - SAFE_OBSTACLE_RADIUS - 8.0,
         OBSTACLE[1]],
        dtype=np.float32
    )

    velocity = np.array(
        [12.0, 0.0],
        dtype=np.float32
    )

    requested = np.array(
        [1.0, 0.0],
        dtype=np.float32
    )

    filtered = project_action_to_safety(
        requested,
        position,
        velocity
    )

    results = check_cbf_constraints(
        filtered,
        position,
        velocity
    )

    assert results["obstacle"]
    print("Test 9 passed: feasible obstacle constraint")

    # Test 10: Environment state dimension.
    env = MovingTargetEnv()

    state = env.reset()

    assert state.shape == (STATE_DIM,)
    print("Test 10 passed: environment observation")

    # Test 11: Environment step structure.
    result = env.step(
        np.zeros(ACTION_DIM)
    )

    assert len(result) == 5
    assert result[0].shape == (STATE_DIM,)
    assert np.isfinite(result[1])

    print("Test 11 passed: environment step")

    # Test 12: Replay buffer.
    replay = ReplayBuffer(100)

    replay.add(
        np.zeros(STATE_DIM),
        np.zeros(ACTION_DIM),
        1.0,
        np.ones(STATE_DIM),
        False
    )

    assert len(replay) == 1
    print("Test 12 passed: replay buffer")

    # Test 13: Replay batch shapes.
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

    print("Test 13 passed: replay batch dimensions")

    # Test 14: Actor output.
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

    print("Test 14 passed: actor output")

    # Test 15: Critic output.
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
    print("Test 15 passed: critic output")

    # Test 16: TD3 action.
    agent = TD3Agent()

    action = agent.select_action(
        np.zeros(STATE_DIM, dtype=np.float32)
    )

    assert action.shape == (ACTION_DIM,)
    assert np.all(action >= -1.0)
    assert np.all(action <= 1.0)

    print("Test 16 passed: TD3 action")

    # Test 17: CBF action filter result structure.
    action, info = cbf_action_filter(
        np.array([1.0, 1.0]),
        np.array([100.0, 100.0]),
        np.zeros(2)
    )

    assert action.shape == (ACTION_DIM,)
    assert "changed" in info
    assert "all_satisfied" in info

    print("Test 17 passed: CBF filter interface")

    print("\nAll 17 tests passed.\n")


# ============================================================
# 15. MAIN
# ============================================================

def main():
    print("=" * 65)
    print("PHASE 4.14 — SAFE RL AND CONTROL BARRIER FUNCTIONS")
    print("=" * 65)

    run_tests()

    agent, rewards, distances = train_td3()

    visualize_training(
        rewards,
        distances
    )

    print("\nStarting Pygame simulation.")
    print("Press 1 for Raw TD3 or 2 for Safe TD3 + CBF.")

    gui = RocketGUI(
        agent,
        rewards,
        distances
    )

    gui.run()


if __name__ == "__main__":
    main()
