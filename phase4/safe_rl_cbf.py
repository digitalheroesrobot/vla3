
"""
RL COURSE — PHASE 4.14
============================================================
Filename: phase4/safe_rl_cbf.py

PURPOSE
-------
Train a drone to track a moving target while avoiding
obstacles and remaining inside the operating area.

WHAT YOU LEARN
--------------
STEP 1 — Moving-target interception
    Use target position and velocity to predict where the
    target will be rather than chasing its current position.

STEP 2 — PD demonstration controller
    Generate useful initial tracking experiences.

STEP 3 — Reward shaping
    Reward distance reduction, velocity matching, and
    successful interception.

STEP 4 — TD3
    Learn continuous drone acceleration commands.

STEP 5 — CBF safety filter
    Project actions onto approximate safety constraints.

STEP 6 — Evaluation
    Compare raw TD3 and CBF-filtered TD3.

STEP 7 — Visualization
    Display the quadcopter, moving target, obstacle,
    requested action, and executed action.

IMPORTANT
---------
This is an educational 2D acceleration-level drone model.
It is not a flight controller for a physical drone.

The CBF projection is approximate and does not guarantee
safety when constraints are infeasible.
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

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

WIDTH = 720
HEIGHT = 580
PANEL_WIDTH = 310

DT = 0.10

DRONE_RADIUS = 13
MAX_SPEED = 115.0
MAX_ACCELERATION = 100.0

OBSTACLE = np.array(
    [350.0, 285.0],
    dtype=np.float32
)

OBSTACLE_RADIUS = 52.0
SAFETY_MARGIN = 35.0

SAFE_OBSTACLE_RADIUS = (
    OBSTACLE_RADIUS
    + DRONE_RADIUS
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

BUFFER_CAPACITY = 200_000
BATCH_SIZE = 256

# More experience before learning.
WARMUP_STEPS = 2500

POLICY_DELAY = 2
POLICY_NOISE = 0.12
NOISE_CLIP = 0.25

# Longer training.
TRAIN_EPISODES = 400
MAX_STEPS_PER_EPISODE = 400

# Demonstration controller schedule.
EXPERT_STEPS = 6000
EXPERT_MIX_PROBABILITY = 0.65

# PD tracking parameters.
PD_KP = 1.8
PD_KD = 2.2
LOOKAHEAD_TIME = 0.45
DESIRED_SPEED = 95.0

# Reward parameters.
DISTANCE_REWARD = 0.020
PROGRESS_REWARD = 0.12
VELOCITY_REWARD = 0.006
ACTION_PENALTY = 0.012
SUCCESS_RADIUS = 25.0
SUCCESS_REWARD = 20.0
COLLISION_PENALTY = 100.0
BOUNDARY_PENALTY = 100.0

# CBF.
CBF_ALPHA = 2.0
CBF_ITERATIONS = 60
CBF_TOLERANCE = 1e-5


# ============================================================
# 2. UTILITIES
# ============================================================

def clamp_vector(vector, maximum):
    vector = np.asarray(vector, dtype=np.float32)

    norm = float(np.linalg.norm(vector))

    if norm > maximum:
        vector = vector * maximum / (norm + 1e-8)

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


def validate_action(action):
    action = np.asarray(action, dtype=np.float32)

    if action.shape != (ACTION_DIM,):
        raise ValueError(
            f"Invalid action shape: {action.shape}"
        )

    if not np.all(np.isfinite(action)):
        raise ValueError("Non-finite action.")

    return np.clip(action, -1.0, 1.0)


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
# 4. DRONE DYNAMICS
# ============================================================

def predict_drone_state(position, velocity, action):
    action = validate_action(action)

    acceleration = action * MAX_ACCELERATION

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
# 5. PD DEMONSTRATION CONTROLLER
# ============================================================

def pd_tracking_controller(position, velocity, target):
    """
    The controller predicts the target's future position.

    Desired velocity:
        v_desired = Kp * (p_target_future - p_drone)

    Acceleration:
        a = Kd * (v_desired - v_drone)

    The output is normalized to [-1, 1].

    This controller supplies useful initial replay data.
    It is a demonstration policy, not the learned policy.
    """

    future_target = (
        target.position
        + target.velocity * LOOKAHEAD_TIME
    )

    position_error = future_target - position

    desired_velocity = clamp_vector(
        PD_KP * position_error,
        DESIRED_SPEED
    )

    acceleration = (
        PD_KD * (desired_velocity - velocity)
    )

    acceleration = clamp_vector(
        acceleration,
        MAX_ACCELERATION
    )

    return np.clip(
        acceleration / MAX_ACCELERATION,
        -1.0,
        1.0
    ).astype(np.float32)


# ============================================================
# 6. CONTROL BARRIER FUNCTIONS
# ============================================================

def obstacle_barrier(position):
    relative = (
        np.asarray(position, dtype=np.float32)
        - OBSTACLE
    )

    return float(
        np.dot(relative, relative)
        - SAFE_OBSTACLE_RADIUS ** 2
    )


def build_cbf_constraints(position, velocity):
    position = np.asarray(position, dtype=np.float32)
    velocity = np.asarray(velocity, dtype=np.float32)

    constraints = []

    # Obstacle.
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
        (coefficient, float(rhs), "obstacle")
    )

    # Left.
    h = position[0] - DRONE_RADIUS

    constraints.append(
        (
            np.array(
                [MAX_ACCELERATION * DT, 0.0],
                dtype=np.float32
            ),
            float(-CBF_ALPHA * h - velocity[0]),
            "left"
        )
    )

    # Right.
    h = WIDTH - DRONE_RADIUS - position[0]

    constraints.append(
        (
            np.array(
                [-MAX_ACCELERATION * DT, 0.0],
                dtype=np.float32
            ),
            float(-CBF_ALPHA * h + velocity[0]),
            "right"
        )
    )

    # Top.
    h = position[1] - DRONE_RADIUS

    constraints.append(
        (
            np.array(
                [0.0, MAX_ACCELERATION * DT],
                dtype=np.float32
            ),
            float(-CBF_ALPHA * h - velocity[1]),
            "top"
        )
    )

    # Bottom.
    h = HEIGHT - DRONE_RADIUS - position[1]

    constraints.append(
        (
            np.array(
                [0.0, -MAX_ACCELERATION * DT],
                dtype=np.float32
            ),
            float(-CBF_ALPHA * h + velocity[1]),
            "bottom"
        )
    )

    return constraints


def project_action_to_safety(
    requested_action,
    position,
    velocity
):
    requested_action = validate_action(
        requested_action
    )

    action = requested_action.copy()

    constraints = build_cbf_constraints(
        position,
        velocity
    )

    for _ in range(CBF_ITERATIONS):
        changed = False

        for coefficient, rhs, _name in constraints:

            violation = (
                rhs - float(np.dot(coefficient, action))
            )

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


def check_cbf_constraints(action, position, velocity):
    results = {}

    for coefficient, rhs, name in build_cbf_constraints(
        position,
        velocity
    ):
        results[name] = (
            float(np.dot(coefficient, action))
            >= rhs - 1e-3
        )

    return results


def cbf_action_filter(action, position, velocity):
    filtered = project_action_to_safety(
        action,
        position,
        velocity
    )

    results = check_cbf_constraints(
        filtered,
        position,
        velocity
    )

    return filtered, {
        "changed": not np.allclose(
            action,
            filtered,
            atol=1e-4
        ),
        "all_satisfied": all(results.values()),
        "constraints": results
    }


# ============================================================
# 7. ENVIRONMENT
# ============================================================

class MovingTargetEnv:

    def __init__(self):
        self.target = MovingTarget()
        self.reset()

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
        ).astype(np.float32)

        assert state.shape == (STATE_DIM,)
        assert np.all(np.isfinite(state))

        return state

    def step(self, action):
        action = validate_action(action)

        old_position = self.position.copy()
        old_velocity = self.velocity.copy()

        # Predict the moving target at the next time step.
        next_phase = (
            self.target.phase + TARGET_OMEGA * DT
        )

        future_target = self.target.position_at(
            next_phase
        )

        old_distance = distance(
            old_position,
            future_target
        )

        self.position, self.velocity = (
            predict_drone_state(
                self.position,
                self.velocity,
                action
            )
        )

        self.target.step()

        new_distance = distance(
            self.position,
            self.target.position
        )

        progress = old_distance - new_distance

        relative_velocity = (
            self.target.velocity - self.velocity
        )

        velocity_error = float(
            np.linalg.norm(relative_velocity)
        )

        # Dense tracking reward.
        reward = (
            PROGRESS_REWARD * progress
            - DISTANCE_REWARD * new_distance
            - VELOCITY_REWARD * velocity_error
            - ACTION_PENALTY * float(np.sum(action ** 2))
        )

        collision = (
            distance(self.position, OBSTACLE)
            < OBSTACLE_RADIUS + DRONE_RADIUS
        )

        outside = (
            self.position[0] < DRONE_RADIUS
            or self.position[0] > WIDTH - DRONE_RADIUS
            or self.position[1] < DRONE_RADIUS
            or self.position[1] > HEIGHT - DRONE_RADIUS
        )

        success = (
            new_distance <= SUCCESS_RADIUS
        )

        terminated = False

        if collision:
            reward -= COLLISION_PENALTY
            terminated = True

        elif outside:
            reward -= BOUNDARY_PENALTY
            terminated = True

        elif success:
            reward += SUCCESS_REWARD
            terminated = True

        self.step_count += 1

        truncated = (
            self.step_count >= MAX_STEPS_PER_EPISODE
        )

        info = {
            "distance": new_distance,
            "progress": progress,
            "velocity_error": velocity_error,
            "collision": collision,
            "outside": outside,
            "success": success
        }

        return (
            self.get_state(),
            float(reward),
            terminated,
            truncated,
            info
        )


# ============================================================
# 8. REPLAY BUFFER
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
        if len(self.buffer) < batch_size:
            raise ValueError(
                f"Replay buffer has {len(self.buffer)} "
                f"samples; {batch_size} required."
            )

        batch = random.sample(
            self.buffer,
            batch_size
        )

        states, actions, rewards, next_states, dones = zip(
            *batch
        )

        return (
            torch.as_tensor(
                np.asarray(states),
                dtype=torch.float32,
                device=DEVICE
            ),
            torch.as_tensor(
                np.asarray(actions),
                dtype=torch.float32,
                device=DEVICE
            ),
            torch.as_tensor(
                np.asarray(rewards),
                dtype=torch.float32,
                device=DEVICE
            ).unsqueeze(1),
            torch.as_tensor(
                np.asarray(next_states),
                dtype=torch.float32,
                device=DEVICE
            ),
            torch.as_tensor(
                np.asarray(dones),
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
            torch.cat([state, action], dim=1)
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
        state = torch.as_tensor(
            state,
            dtype=torch.float32,
            device=DEVICE
        ).unsqueeze(0)

        with torch.no_grad():
            action = self.actor(
                state
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

        with torch.no_grad():
            next_actions = self.actor_target(
                next_states
            )

            noise = torch.randn_like(
                next_actions
            ) * POLICY_NOISE

            noise = noise.clamp(
                -NOISE_CLIP,
                NOISE_CLIP
            )

            next_actions = (
                next_actions + noise
            ).clamp(-1.0, 1.0)

            q1 = self.critic1_target(
                next_states,
                next_actions
            )

            q2 = self.critic2_target(
                next_states,
                next_actions
            )

            target_q = torch.minimum(q1, q2)

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
    print("\nTraining improved Safe TD3...")
    print(f"Device: {DEVICE}")

    env = MovingTargetEnv()
    agent = TD3Agent()
    replay = ReplayBuffer(BUFFER_CAPACITY)

    rewards_history = []
    distances_history = []
    success_history = []

    total_steps = 0
    interventions = 0

    for episode in range(TRAIN_EPISODES):

        state = env.reset()

        episode_reward = 0.0
        episode_distances = []
        episode_success = False

        for _ in range(MAX_STEPS_PER_EPISODE):

            # ------------------------------------------------
            # Demonstration-guided exploration
            # ------------------------------------------------

            expert_action = pd_tracking_controller(
                env.position,
                env.velocity,
                env.target
            )

            if total_steps < EXPERT_STEPS:
                # Initially use the PD controller with noise.
                requested_action = (
                    expert_action
                    + np.random.normal(
                        0.0,
                        0.12,
                        ACTION_DIM
                    )
                )

            else:
                learned_action = agent.select_action(
                    state,
                    noise=0.15
                )

                # Gradually reduce expert assistance.
                if random.random() < EXPERT_MIX_PROBABILITY:
                    requested_action = (
                        0.7 * expert_action
                        + 0.3 * learned_action
                    )
                else:
                    requested_action = learned_action

            requested_action = validate_action(
                requested_action
            )

            # CBF filter during training.
            executed_action, safety_info = cbf_action_filter(
                requested_action,
                env.position,
                env.velocity
            )

            if safety_info["changed"]:
                interventions += 1

            (
                next_state,
                reward,
                terminated,
                truncated,
                info
            ) = env.step(executed_action)

            # Store the actual action executed.
            replay.add(
                state,
                executed_action,
                reward,
                next_state,
                terminated
            )

            # Begin learning after enough experience.
            if (
                total_steps >= WARMUP_STEPS
                and len(replay) >= BATCH_SIZE
            ):
                agent.train_step(replay)

            state = next_state

            episode_reward += reward
            episode_distances.append(
                info["distance"]
            )

            if info["success"]:
                episode_success = True

            total_steps += 1

            if terminated or truncated:
                break

        rewards_history.append(episode_reward)

        distances_history.append(
            float(np.mean(episode_distances))
        )

        success_history.append(
            float(episode_success)
        )

        if (episode + 1) % 10 == 0:
            recent_rewards = np.mean(
                rewards_history[-10:]
            )

            recent_distances = np.mean(
                distances_history[-10:]
            )

            recent_success = np.mean(
                success_history[-10:]
            ) * 100.0

            print(
                f"Episode {episode + 1:3d}/"
                f"{TRAIN_EPISODES} | "
                f"Reward: {recent_rewards:8.2f} | "
                f"Distance: {recent_distances:7.2f} | "
                f"Success: {recent_success:5.1f}% | "
                f"CBF interventions: {interventions}"
            )

    return (
        agent,
        rewards_history,
        distances_history,
        success_history
    )


# ============================================================
# 12. TRAINING VISUALIZATION
# ============================================================

def visualize_training(
    rewards,
    distances,
    successes
):
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(
            3,
            1,
            figsize=(10, 9)
        )

        axes[0].plot(
            rewards,
            alpha=0.4,
            label="Episode reward"
        )

        if len(rewards) >= 10:
            smooth = np.convolve(
                rewards,
                np.ones(10) / 10,
                mode="valid"
            )

            axes[0].plot(
                np.arange(9, len(rewards)),
                smooth,
                label="10-episode average"
            )

        axes[0].set_title("TD3 Training Reward")
        axes[0].set_ylabel("Reward")
        axes[0].legend()
        axes[0].grid(True)

        axes[1].plot(
            distances,
            label="Mean target distance"
        )

        axes[1].set_title("Tracking Performance")
        axes[1].set_ylabel("Pixels")
        axes[1].legend()
        axes[1].grid(True)

        window = 10

        success_rate = np.convolve(
            successes,
            np.ones(window) / window,
            mode="valid"
        ) * 100.0

        axes[2].plot(
            np.arange(window, len(successes) + 1),
            success_rate,
            label="10-episode success rate"
        )

        axes[2].set_title("Interception Success Rate")
        axes[2].set_xlabel("Episode")
        axes[2].set_ylabel("Success (%)")
        axes[2].set_ylim(0, 100)
        axes[2].legend()
        axes[2].grid(True)

        plt.tight_layout()
        plt.show()

    except ImportError:
        print("Matplotlib unavailable; skipping plots.")


# ============================================================
# 13. PYGAME DRONE VISUALIZATION
# ============================================================

class DroneGUI:

    def __init__(self, agent):
        pygame.init()

        self.screen = pygame.display.set_mode(
            (WIDTH + PANEL_WIDTH, HEIGHT)
        )

        pygame.display.set_caption(
            "Phase 4.14 — Improved Safe TD3 Drone"
        )

        self.clock = pygame.time.Clock()

        self.font = pygame.font.SysFont(
            "consolas",
            15
        )

        self.agent = agent
        self.env = MovingTargetEnv()
        self.env.reset()

        self.mode = 2
        self.paused = False
        self.speed = 1
        self.running = True

        self.requested_action = np.zeros(
            ACTION_DIM,
            dtype=np.float32
        )

        self.executed_action = np.zeros(
            ACTION_DIM,
            dtype=np.float32
        )

        self.safety_info = {
            "changed": False,
            "all_satisfied": True
        }

        self.interventions = 0
        self.total_actions = 0

    def reset(self):
        self.env.reset()

        self.requested_action[:] = 0.0
        self.executed_action[:] = 0.0

        self.interventions = 0
        self.total_actions = 0

    def change_target_center(self, position):
        x, y = position

        if not (
            150 < x < WIDTH - 130
            and 100 < y < HEIGHT - 100
        ):
            return

        candidate = np.asarray(
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
            OBSTACLE_RADIUS + DRONE_RADIUS + 25
        ):
            return

        self.env.target.set_center(candidate)

    def select_action(self):
        state = self.env.get_state()

        requested = self.agent.select_action(
            state,
            noise=0.0
        )

        if self.mode == 1:
            executed = requested

            info = {
                "changed": False,
                "all_satisfied": False
            }

        else:
            executed, info = cbf_action_filter(
                requested,
                self.env.position,
                self.env.velocity
            )

        self.requested_action = requested.copy()
        self.executed_action = executed.copy()
        self.safety_info = info

        self.total_actions += 1

        if info["changed"]:
            self.interventions += 1

        return executed

    def draw_arrow(self, start, end, color, width=2):
        start = np.asarray(start, dtype=np.float32)
        end = np.asarray(end, dtype=np.float32)

        pygame.draw.line(
            self.screen,
            color,
            start.astype(int),
            end.astype(int),
            width
        )

        direction = safe_unit(end - start)

        if np.linalg.norm(direction) < 1e-6:
            return

        perpendicular = np.array(
            [-direction[1], direction[0]]
        )

        p1 = end - direction * 9 + perpendicular * 4
        p2 = end - direction * 9 - perpendicular * 4

        pygame.draw.polygon(
            self.screen,
            color,
            [
                end.astype(int),
                p1.astype(int),
                p2.astype(int)
            ]
        )

    def draw_drone(self):
        p = self.env.position

        velocity_direction = safe_unit(
            self.env.velocity
        )

        if np.linalg.norm(velocity_direction) < 0.1:
            velocity_direction = safe_unit(
                self.env.target.position - p
            )

        if np.linalg.norm(velocity_direction) < 0.1:
            velocity_direction = np.array(
                [1.0, 0.0]
            )

        perpendicular = np.array(
            [-velocity_direction[1], velocity_direction[0]]
        )

        offsets = [
            velocity_direction * 20 + perpendicular * 20,
            velocity_direction * 20 - perpendicular * 20,
            -velocity_direction * 20 + perpendicular * 20,
            -velocity_direction * 20 - perpendicular * 20
        ]

        for offset in offsets:
            motor = p + offset

            pygame.draw.line(
                self.screen,
                (195, 210, 230),
                p.astype(int),
                motor.astype(int),
                4
            )

            pygame.draw.circle(
                self.screen,
                (90, 160, 225),
                motor.astype(int),
                7
            )

            pygame.draw.circle(
                self.screen,
                (190, 220, 255),
                motor.astype(int),
                10,
                1
            )

        pygame.draw.circle(
            self.screen,
            (230, 235, 245),
            p.astype(int),
            11
        )

        pygame.draw.circle(
            self.screen,
            (35, 45, 65),
            p.astype(int),
            5
        )

        front = p + velocity_direction * 14

        pygame.draw.circle(
            self.screen,
            (255, 185, 70),
            front.astype(int),
            4
        )

    def draw_target_path(self):
        points = []

        for i in range(100):
            phase = self.env.target.phase + i * 0.08

            points.append(
                self.env.target.position_at(
                    phase
                ).astype(int)
            )

        pygame.draw.lines(
            self.screen,
            (65, 95, 75),
            False,
            points,
            1
        )

    def draw_world(self):
        self.screen.fill((15, 20, 30))

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

        pygame.draw.rect(
            self.screen,
            (90, 110, 140),
            (0, 0, WIDTH, HEIGHT),
            2
        )

        pygame.draw.circle(
            self.screen,
            (115, 75, 55),
            OBSTACLE.astype(int),
            int(SAFE_OBSTACLE_RADIUS),
            2
        )

        pygame.draw.circle(
            self.screen,
            (170, 65, 65),
            OBSTACLE.astype(int),
            int(OBSTACLE_RADIUS)
        )

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

        self.draw_arrow(
            self.env.position,
            self.env.position + self.requested_action * 45,
            (230, 100, 100),
            2
        )

        self.draw_arrow(
            self.env.position,
            self.env.position + self.executed_action * 65,
            (100, 170, 255),
            3
        )

        self.draw_arrow(
            self.env.position,
            self.env.position + self.env.velocity * 0.35,
            (255, 210, 90),
            2
        )

        self.draw_drone()

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

        d = distance(
            self.env.position,
            self.env.target.position
        )

        mode_name = (
            "RAW TD3"
            if self.mode == 1
            else "SAFE TD3 + CBF"
        )

        rate = (
            100.0 * self.interventions
            / max(1, self.total_actions)
        )

        lines = [
            ("PHASE 4.14", (255, 220, 100)),
            ("IMPROVED SAFE TD3", (220, 230, 245)),
            ("", (200, 200, 200)),
            ("CONTROLLER", (150, 190, 255)),
            ("1  Raw TD3", (220, 220, 220)),
            ("2  Safe TD3 + CBF", (220, 220, 220)),
            (f"Mode: {mode_name}", (100, 240, 150)),
            ("", (200, 200, 200)),
            ("TRACKING", (150, 190, 255)),
            (f"Distance: {d:.1f}", (230, 230, 230)),
            (f"Target vx: {self.env.target.velocity[0]:.1f}", (180, 240, 180)),
            (f"Target vy: {self.env.target.velocity[1]:.1f}", (180, 240, 180)),
            ("", (200, 200, 200)),
            ("SAFETY", (150, 190, 255)),
            (f"Obstacle h: {obstacle_barrier(self.env.position):.0f}", (220, 220, 220)),
            (f"Modified: {self.safety_info['changed']}", (220, 220, 220)),
            (f"Constraints: {self.safety_info['all_satisfied']}", (220, 220, 220)),
            (f"Interventions: {self.interventions}", (220, 220, 220)),
            (f"Intervention rate: {rate:.1f}%", (220, 220, 220)),
            ("", (200, 200, 200)),
            ("ACTIONS", (150, 190, 255)),
            (f"Requested X: {self.requested_action[0]:+.2f}", (230, 140, 140)),
            (f"Requested Y: {self.requested_action[1]:+.2f}", (230, 140, 140)),
            (f"Executed X: {self.executed_action[0]:+.2f}", (130, 190, 255)),
            (f"Executed Y: {self.executed_action[1]:+.2f}", (130, 190, 255)),
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
                surface = self.font.render(
                    text,
                    True,
                    color
                )

                self.screen.blit(
                    surface,
                    (x + 10, y)
                )

            y += 16

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

                if event.button == 1 and event.pos[0] < WIDTH:
                    self.change_target_center(event.pos)

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
# 14. TESTS
# ============================================================

def run_tests():
    print("\nRunning improved Phase 4.14 tests...\n")

    tests = 0

    def passed(name):
        nonlocal tests
        tests += 1
        print(f"Test {tests} passed: {name}")

    env = MovingTargetEnv()

    state = env.reset()

    assert state.shape == (STATE_DIM,)
    passed("environment state dimension")

    assert np.all(np.isfinite(state))
    passed("finite environment state")

    action = pd_tracking_controller(
        env.position,
        env.velocity,
        env.target
    )

    assert action.shape == (ACTION_DIM,)
    assert np.all(np.abs(action) <= 1.0)
    passed("PD controller action")

    filtered, info = cbf_action_filter(
        action,
        env.position,
        env.velocity
    )

    assert filtered.shape == (ACTION_DIM,)
    assert np.all(np.abs(filtered) <= 1.0)
    passed("CBF action projection")

    constraints = build_cbf_constraints(
        env.position,
        env.velocity
    )

    assert len(constraints) == 5
    passed("five CBF constraints")

    next_state, reward, terminated, truncated, info = env.step(
        filtered
    )

    assert next_state.shape == (STATE_DIM,)
    assert np.isfinite(reward)
    passed("environment transition")

    replay = ReplayBuffer(1000)

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
    passed("replay buffer batch dimensions")

    agent = TD3Agent()

    output = agent.select_action(
        np.zeros(STATE_DIM, dtype=np.float32)
    )

    assert output.shape == (ACTION_DIM,)
    assert np.all(np.abs(output) <= 1.0)
    passed("TD3 actor output")

    loss = agent.train_step(replay)

    assert loss is not None
    assert np.isfinite(loss)
    passed("TD3 training update")

    # Verify target actually moves.
    previous = env.target.position.copy()
    env.target.step()

    assert not np.allclose(
        previous,
        env.target.position
    )

    passed("moving target")

    print(f"\nAll {tests} tests passed.\n")


# ============================================================
# 15. MAIN
# ============================================================

def main():
    print("=" * 65)
    print("PHASE 4.14 — IMPROVED SAFE TD3 DRONE TRACKING")
    print("=" * 65)

    run_tests()

    (
        agent,
        rewards,
        distances,
        successes
    ) = train_td3()

    visualize_training(
        rewards,
        distances,
        successes
    )

    print("\nStarting interactive drone simulation.")

    print("1 = Raw TD3")
    print("2 = Safe TD3 + CBF")
    print("SPACE = Pause / resume")
    print("R = Reset")
    print("+/- = Simulation speed")
    print("Mouse = Change moving target center")
    print("ESC = Exit")

    gui = DroneGUI(agent)
    gui.run()


if __name__ == "__main__":
    main()
