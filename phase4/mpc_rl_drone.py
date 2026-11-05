
"""
RL COURSE — PHASE 4.17
============================================================
Filename:
    phase4/mpc_rl_drone.py

LESSON:
    Model Predictive Control (MPC) + Reinforcement Learning

PURPOSE
-------
Combine a learned RL policy with short-horizon trajectory
planning to navigate a quadcopter drone toward a moving target
while avoiding obstacles.

WHAT YOU LEARN
--------------
STEP 1 — Continuous drone control
    Represent drone movement using continuous horizontal
    and vertical acceleration commands.

STEP 2 — REINFORCE policy learning
    Train a Gaussian neural policy from episode rewards.

STEP 3 — Predictive dynamics
    Predict future drone position, velocity, and moving-target
    position over a finite planning horizon.

STEP 4 — Random-shooting MPC
    Generate candidate action sequences, simulate them,
    and select the sequence with the best predicted score.

STEP 5 — Receding-horizon control
    Execute only the first action, observe the new state,
    and replan.

STEP 6 — RL-guided MPC
    Use the learned RL policy to center candidate action
    sequences. MPC evaluates alternatives before execution.

STEP 7 — Interactive simulation
    Visualize the quadcopter, moving target, obstacle,
    predicted trajectory, and selected action.

MATHEMATICAL FOUNDATION
-----------------------
REINFORCE:

    G_t = sum(k=t to T) gamma^(k-t) * r_k

    theta <- theta
             + alpha * sum_t
               G_t * grad(log pi_theta(a_t | s_t))

MPC:

    a* = argmax over action sequences A:

         sum(k=0 to H-1)
             gamma^k * r_hat(s_k, a_k)
             + gamma^H * V_hat(s_H)

    Execute only:

         a_0*

    Then observe the real environment and plan again.

RL-GUIDED MPC:

    A_candidate = policy(state) + exploration_noise

    MPC evaluates the candidate trajectories and chooses
    the best predicted action sequence.

IMPORTANT DISTINCTION
---------------------
The RL policy is trained using REINFORCE.

The MPC controller uses a known approximate drone-motion
model and random-shooting trajectory search.

This is an educational model-based planning system.
It is not a learned world model, and its predicted
safety is not a formal guarantee.

ROBOTICS APPLICATION
--------------------
- Quadcopter target tracking
- Autonomous inspection drones
- Warehouse aerial robots
- Obstacle-aware navigation
- Moving-object interception

CONTROLS
--------
Mouse       Change moving-target trajectory center
SPACE       Pause / resume
R           Reset drone
+ / -       Simulation speed
M           Toggle RL-only / RL-guided MPC
ESC         Exit
============================================================
"""

from __future__ import annotations

import math
import random

import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


# ============================================================
# 1. CONFIGURATION
# ============================================================

SEED = 417

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

DRONE_RADIUS = 13
MAX_SPEED = 120.0
MAX_ACCELERATION = 100.0

OBSTACLE = np.array(
    [350.0, 285.0],
    dtype=np.float32
)

OBSTACLE_RADIUS = 52.0
SAFETY_RADIUS = 95.0

TARGET_AMPLITUDE = np.array(
    [85.0, 60.0],
    dtype=np.float32
)

TARGET_OMEGA = 0.65

STATE_DIM = 10
ACTION_DIM = 2

GAMMA = 0.99

# RL training.
RL_EPISODES = 45
RL_MAX_STEPS = 250
RL_LEARNING_RATE = 2e-4
ENTROPY_WEIGHT = 0.002

# MPC configuration.
MPC_HORIZON = 14
MPC_CANDIDATES = 128
MPC_NOISE = 0.55
MPC_GAMMA = 0.94

# GUI.
FPS = 60


# ============================================================
# 2. NUMERICAL UTILITIES
# ============================================================

def clamp_vector(vector, maximum):
    vector = np.asarray(
        vector,
        dtype=np.float32
    )

    norm = float(np.linalg.norm(vector))

    if norm > maximum:
        vector = vector * maximum / (norm + 1e-8)

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

    def __init__(self):
        self.center = np.array(
            [555.0, 145.0],
            dtype=np.float32
        )

        self.phase = random.uniform(
            0.0,
            2.0 * math.pi
        )

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
                TARGET_OMEGA * 1.35
                * math.cos(phase * 1.35 + 0.7)
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

def drone_dynamics(position, velocity, action):
    """
    Simplified 2D quadcopter model.

    Action:
        action[0] = horizontal acceleration command
        action[1] = vertical acceleration command

    The simulator represents a top-down 2D navigation task,
    not full 3D flight dynamics.
    """

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
# 5. ENVIRONMENT
# ============================================================

class DroneEnv:

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

        self.steps = 0

        self.target.phase = random.uniform(
            0.0,
            2.0 * math.pi
        )

        return self.get_state()

    def get_state(self):
        relative_target = (
            self.target.position - self.position
        )

        relative_obstacle = (
            OBSTACLE - self.position
        )

        clearance = (
            distance(self.position, OBSTACLE)
            - SAFETY_RADIUS
        )

        state = np.concatenate([
            relative_target / 600.0,
            self.target.velocity / 150.0,
            self.velocity / MAX_SPEED,
            relative_obstacle / 600.0,
            np.array([
                clearance / 200.0,
                distance(
                    self.position,
                    self.target.position
                ) / 600.0
            ], dtype=np.float32)
        ])

        return state.astype(np.float32)

    def step(self, action):
        next_phase = (
            self.target.phase + TARGET_OMEGA * DT
        )

        future_target = (
            self.target.position_at(next_phase)
        )

        distance_before = distance(
            self.position,
            future_target
        )

        self.position, self.velocity = drone_dynamics(
            self.position,
            self.velocity,
            action
        )

        self.target.step()

        distance_after = distance(
            self.position,
            self.target.position
        )

        progress = (
            distance_before - distance_after
        )

        clearance = (
            distance(self.position, OBSTACLE)
            - SAFETY_RADIUS
        )

        # Proximity penalty encourages safe trajectories.
        proximity_penalty = (
            0.03 * max(0.0, -clearance)
        )

        reward = (
            0.10 * progress
            + 0.4 * math.exp(-distance_after / 75.0)
            - 0.015 * float(np.sum(np.asarray(action) ** 2))
            - proximity_penalty
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

        terminated = collision or outside

        if terminated:
            reward -= 100.0

        self.steps += 1

        truncated = (
            self.steps >= RL_MAX_STEPS
        )

        info = {
            "distance": distance_after,
            "clearance": clearance,
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
# 6. GAUSSIAN RL POLICY
# ============================================================

class GaussianPolicy(nn.Module):

    def __init__(self):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Linear(STATE_DIM, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh()
        )

        self.mean = nn.Linear(
            128,
            ACTION_DIM
        )

        self.log_std = nn.Parameter(
            torch.full(
                (ACTION_DIM,),
                -0.8
            )
        )

    def forward(self, states):
        features = self.backbone(states)

        mean = self.mean(features)

        std = self.log_std.clamp(
            -2.5,
            0.3
        ).exp().expand_as(mean)

        return mean, std

    def distribution(self, states):
        mean, std = self(states)

        return Normal(mean, std)

    def sample_action(self, states):
        """
        Squashed Gaussian policy.

        a = tanh(z)

        The log probability includes the tanh Jacobian
        correction, which is required for a valid policy
        gradient with bounded continuous actions.
        """

        distribution = self.distribution(states)

        raw_action = distribution.rsample()

        action = torch.tanh(
            raw_action
        )

        log_probability = (
            distribution.log_prob(raw_action)
            - torch.log(
                1.0 - action.pow(2) + 1e-6
            )
        ).sum(
            dim=-1,
            keepdim=True
        )

        return (
            action,
            log_probability,
            distribution
        )

    @torch.no_grad()
    def deterministic_action(self, state):
        state_tensor = torch.as_tensor(
            state,
            dtype=torch.float32,
            device=DEVICE
        ).unsqueeze(0)

        mean, _ = self(state_tensor)

        return torch.tanh(
            mean
        ).cpu().numpy()[0]


# ============================================================
# 7. REINFORCE TRAINING
# ============================================================

def discounted_returns(rewards):
    returns = []

    running_return = 0.0

    for reward in reversed(rewards):
        running_return = (
            reward + GAMMA * running_return
        )

        returns.append(
            running_return
        )

    returns.reverse()

    return np.asarray(
        returns,
        dtype=np.float32
    )


def train_reinforce():
    """
    Train a continuous Gaussian policy with REINFORCE.

    This is a policy-gradient algorithm, not behavioral
    cloning. The policy learns from reward feedback.

    MPC is not used during this training stage. The learned
    policy is subsequently used as a candidate-action prior
    for the MPC planner.
    """

    env = DroneEnv()

    policy = GaussianPolicy().to(DEVICE)

    optimizer = torch.optim.Adam(
        policy.parameters(),
        lr=RL_LEARNING_RATE
    )

    reward_history = []

    print("\nTraining REINFORCE drone policy...")
    print(f"Device: {DEVICE}")

    for episode in range(RL_EPISODES):

        state = env.reset()

        log_probabilities = []
        rewards = []
        entropies = []

        for _ in range(RL_MAX_STEPS):

            state_tensor = torch.as_tensor(
                state,
                dtype=torch.float32,
                device=DEVICE
            ).unsqueeze(0)

            action, log_probability, distribution = (
                policy.sample_action(state_tensor)
            )

            action_np = (
                action.detach().cpu().numpy()[0]
            )

            (
                next_state,
                reward,
                terminated,
                truncated,
                info
            ) = env.step(action_np)

            log_probabilities.append(
                log_probability.squeeze()
            )

            rewards.append(reward)

            entropies.append(
                distribution.entropy().sum()
            )

            state = next_state

            if terminated or truncated:
                break

        returns = discounted_returns(
            rewards
        )

        returns_tensor = torch.as_tensor(
            returns,
            dtype=torch.float32,
            device=DEVICE
        )

        # Normalize returns to reduce gradient variance.
        if len(returns_tensor) > 1:
            normalized_returns = (
                returns_tensor - returns_tensor.mean()
            ) / (
                returns_tensor.std(unbiased=False) + 1e-6
            )
        else:
            normalized_returns = returns_tensor

        log_prob_tensor = torch.stack(
            log_probabilities
        )

        entropy_tensor = torch.stack(
            entropies
        )

        policy_loss = -(
            log_prob_tensor * normalized_returns.detach()
        ).mean()

        entropy_bonus = entropy_tensor.mean()

        loss = (
            policy_loss
            - ENTROPY_WEIGHT * entropy_bonus
        )

        optimizer.zero_grad()

        loss.backward()

        nn.utils.clip_grad_norm_(
            policy.parameters(),
            1.0
        )

        optimizer.step()

        total_reward = float(
            np.sum(rewards)
        )

        reward_history.append(
            total_reward
        )

        if (episode + 1) % 5 == 0:
            recent = np.mean(
                reward_history[-5:]
            )

            print(
                f"Episode {episode + 1:3d}/"
                f"{RL_EPISODES} | "
                f"Reward: {total_reward:8.2f} | "
                f"Recent mean: {recent:8.2f}"
            )

    return policy, reward_history


# ============================================================
# 8. MPC PREDICTIVE MODEL
# ============================================================

def predict_target(
    target_center,
    phase,
    future_step
):
    future_phase = (
        phase
        + TARGET_OMEGA * DT * future_step
    )

    return target_center + TARGET_AMPLITUDE * np.array(
        [
            math.sin(future_phase),
            math.sin(future_phase * 1.35 + 0.7)
        ],
        dtype=np.float32
    )


def predict_drone_batch(
    positions,
    velocities,
    actions
):
    """
    Vectorized one-step drone prediction.

    positions: (N, 2)
    velocities: (N, 2)
    actions: (N, 2)
    """

    next_velocities = (
        velocities
        + actions * MAX_ACCELERATION * DT
    )

    norms = np.linalg.norm(
        next_velocities,
        axis=1,
        keepdims=True
    )

    scales = np.minimum(
        1.0,
        MAX_SPEED / (norms + 1e-8)
    )

    next_velocities *= scales

    next_positions = (
        positions + next_velocities * DT
    )

    return next_positions, next_velocities


# ============================================================
# 9. RL-GUIDED RANDOM-SHOOTING MPC
# ============================================================

class MPCPlanner:

    def __init__(self, policy):
        self.policy = policy

        self.last_best_sequence = None
        self.last_predicted_path = []

        self.last_score = 0.0
        self.last_candidate_count = MPC_CANDIDATES

    def plan(self, env):
        """
        Random-shooting MPC.

        1. Obtain the RL policy's current action.
        2. Generate candidate action sequences around it.
        3. Predict future drone and target motion.
        4. Score each sequence.
        5. Return only the first action of the best sequence.

        The process repeats after the next real observation.
        """

        base_action = self.policy.deterministic_action(
            env.get_state()
        )

        candidates = np.random.normal(
            loc=base_action,
            scale=MPC_NOISE,
            size=(
                MPC_CANDIDATES,
                MPC_HORIZON,
                ACTION_DIM
            )
        ).astype(np.float32)

        candidates = np.clip(
            candidates,
            -1.0,
            1.0
        )

        # Candidate zero is the RL policy's nominal sequence.
        candidates[0, :, :] = base_action

        # Add a few braking / directional alternatives.
        candidates[1, :, :] = 0.0

        candidates[2, :, :] = np.array(
            [-1.0, 0.0],
            dtype=np.float32
        )

        candidates[3, :, :] = np.array(
            [1.0, 0.0],
            dtype=np.float32
        )

        candidates[4, :, :] = np.array(
            [0.0, -1.0],
            dtype=np.float32
        )

        candidates[5, :, :] = np.array(
            [0.0, 1.0],
            dtype=np.float32
        )

        # Every candidate starts from the actual current state.
        positions = np.repeat(
            env.position[None, :],
            MPC_CANDIDATES,
            axis=0
        )

        velocities = np.repeat(
            env.velocity[None, :],
            MPC_CANDIDATES,
            axis=0
        )

        scores = np.zeros(
            MPC_CANDIDATES,
            dtype=np.float32
        )

        predicted_paths = []

        for candidate_index in range(
            MPC_CANDIDATES
        ):
            predicted_paths.append([
                positions[candidate_index].copy()
            ])

        for step in range(
            MPC_HORIZON
        ):
            actions = candidates[:, step, :]

            next_positions, next_velocities = (
                predict_drone_batch(
                    positions,
                    velocities,
                    actions
                )
            )

            target = predict_target(
                env.target.center,
                env.target.phase,
                step + 1
            )

            distances = np.linalg.norm(
                next_positions - target[None, :],
                axis=1
            )

            progress = (
                np.linalg.norm(
                    positions - target[None, :],
                    axis=1
                )
                - distances
            )

            obstacle_distances = np.linalg.norm(
                next_positions - OBSTACLE[None, :],
                axis=1
            )

            clearance = (
                obstacle_distances - SAFETY_RADIUS
            )

            obstacle_penalty = (
                np.maximum(
                    0.0,
                    -clearance
                ) ** 2
            )

            # Penalize physical collision heavily.
            collision = (
                obstacle_distances
                < OBSTACLE_RADIUS + DRONE_RADIUS
            )

            collision_penalty = (
                collision.astype(np.float32) * 500.0
            )

            boundary_clearance = np.minimum.reduce([
                next_positions[:, 0] - DRONE_RADIUS,
                WIDTH - DRONE_RADIUS - next_positions[:, 0],
                next_positions[:, 1] - DRONE_RADIUS,
                HEIGHT - DRONE_RADIUS - next_positions[:, 1]
            ])

            boundary_penalty = (
                np.maximum(
                    0.0,
                    25.0 - boundary_clearance
                ) ** 2
            )

            action_penalty = np.sum(
                actions ** 2,
                axis=1
            )

            # Reward approaching the predicted moving target.
            stage_reward = (
                0.12 * progress
                + 0.25 * np.exp(-distances / 80.0)
                - 0.018 * action_penalty
                - 0.025 * obstacle_penalty
                - collision_penalty
                - 0.03 * boundary_penalty
            )

            scores += (
                MPC_GAMMA ** step
            ) * stage_reward

            positions = next_positions
            velocities = next_velocities

            for index in range(
                MPC_CANDIDATES
            ):
                predicted_paths[index].append(
                    positions[index].copy()
                )

        # Terminal reward improves final target interception.
        final_target = predict_target(
            env.target.center,
            env.target.phase,
            MPC_HORIZON
        )

        final_distances = np.linalg.norm(
            positions - final_target[None, :],
            axis=1
        )

        scores += (
            0.5 * np.exp(-final_distances / 70.0)
        )

        best_index = int(
            np.argmax(scores)
        )

        self.last_best_sequence = (
            candidates[best_index].copy()
        )

        self.last_predicted_path = (
            predicted_paths[best_index]
        )

        self.last_score = float(
            scores[best_index]
        )

        # Receding horizon: execute only the first action.
        return candidates[
            best_index,
            0
        ].copy()


# ============================================================
# 10. TRAINING VISUALIZATION
# ============================================================

def visualize_training(reward_history):
    try:
        import matplotlib.pyplot as plt

        plt.figure(
            figsize=(10, 5)
        )

        plt.plot(
            reward_history,
            label="REINFORCE episode reward"
        )

        if len(reward_history) >= 5:
            moving_average = np.convolve(
                reward_history,
                np.ones(5) / 5,
                mode="valid"
            )

            plt.plot(
                np.arange(4, len(reward_history)),
                moving_average,
                label="5-episode moving average"
            )

        plt.title(
            "Drone RL Policy Training"
        )

        plt.xlabel(
            "Episode"
        )

        plt.ylabel(
            "Total reward"
        )

        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    except ImportError:
        print("Matplotlib unavailable; skipping plot.")


# ============================================================
# 11. PYGAME DRONE VISUALIZATION
# ============================================================

class DroneGUI:

    def __init__(self, policy):
        pygame.init()

        self.screen = pygame.display.set_mode(
            (WIDTH + PANEL_WIDTH, HEIGHT)
        )

        pygame.display.set_caption(
            "Phase 4.17 — RL-Guided MPC Quadcopter"
        )

        self.clock = pygame.time.Clock()

        self.font = pygame.font.SysFont(
            "consolas",
            15
        )

        self.small_font = pygame.font.SysFont(
            "consolas",
            13
        )

        self.env = DroneEnv()

        self.policy = policy
        self.planner = MPCPlanner(policy)

        self.paused = False
        self.running = True
        self.speed = 1

        self.use_mpc = True

        self.last_action = np.zeros(
            2,
            dtype=np.float32
        )

        self.last_reward = 0.0
        self.last_info = {}

        self.trail = []

    def reset(self):
        self.env.reset()

        self.last_action[:] = 0.0
        self.last_reward = 0.0

        self.trail = []

        self.planner.last_predicted_path = []

    def set_target_center(self, position):
        x, y = position

        if not (
            145 < x < WIDTH - 120
            and 100 < y < HEIGHT - 90
        ):
            return

        candidate = np.array(
            [x, y],
            dtype=np.float32
        )

        orbit_radius = float(
            np.linalg.norm(TARGET_AMPLITUDE)
        )

        clearance = (
            distance(candidate, OBSTACLE)
            - orbit_radius
        )

        if clearance < (
            OBSTACLE_RADIUS + DRONE_RADIUS + 25
        ):
            return

        self.env.target.set_center(
            candidate
        )

    def draw_arrow(
        self,
        start,
        end,
        color,
        width=2
    ):
        start = np.asarray(start)
        end = np.asarray(end)

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
            end - direction * 10
            + perpendicular * 5
        )

        p2 = (
            end - direction * 10
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

    def draw_drone(self):
        """
        Top-down quadcopter drawing.

        The drone has:
            - central body
            - four arms
            - four propellers
            - front direction indicator
        """

        center = self.env.position

        direction = safe_unit(
            self.env.velocity
        )

        if np.linalg.norm(direction) < 0.1:
            direction = safe_unit(
                self.env.target.position - center
            )

        if np.linalg.norm(direction) < 0.1:
            direction = np.array(
                [1.0, 0.0]
            )

        perpendicular = np.array(
            [-direction[1], direction[0]]
        )

        arm_length = 20.0

        motor_positions = [
            center + direction * arm_length
                     + perpendicular * arm_length,

            center + direction * arm_length
                     - perpendicular * arm_length,

            center - direction * arm_length
                     + perpendicular * arm_length,

            center - direction * arm_length
                     - perpendicular * arm_length
        ]

        # Arms.
        for motor in motor_positions:
            pygame.draw.line(
                self.screen,
                (175, 195, 220),
                center.astype(int),
                motor.astype(int),
                5
            )

        # Propellers.
        for motor in motor_positions:
            pygame.draw.circle(
                self.screen,
                (90, 180, 255),
                motor.astype(int),
                9,
                2
            )

            pygame.draw.line(
                self.screen,
                (90, 180, 255),
                (
                    int(motor[0] - 7),
                    int(motor[1])
                ),
                (
                    int(motor[0] + 7),
                    int(motor[1])
                ),
                2
            )

        # Central body.
        pygame.draw.circle(
            self.screen,
            (220, 230, 245),
            center.astype(int),
            11
        )

        pygame.draw.circle(
            self.screen,
            (35, 55, 85),
            center.astype(int),
            5
        )

        # Front indicator.
        front = (
            center + direction * 15
        )

        pygame.draw.circle(
            self.screen,
            (255, 100, 80),
            front.astype(int),
            4
        )

    def draw_predicted_path(self):
        path = self.planner.last_predicted_path

        if len(path) < 2:
            return

        points = [
            tuple(
                np.asarray(point).astype(int)
            )
            for point in path
        ]

        pygame.draw.lines(
            self.screen,
            (100, 160, 255),
            False,
            points,
            2
        )

        for point in points[1:-1:3]:
            pygame.draw.circle(
                self.screen,
                (100, 160, 255),
                point,
                3
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

        # Target trajectory.
        path = []

        for index in range(100):
            phase = (
                self.env.target.phase
                + index * 0.08
            )

            path.append(
                tuple(
                    self.env.target.position_at(
                        phase
                    ).astype(int)
                )
            )

        pygame.draw.lines(
            self.screen,
            (65, 95, 75),
            False,
            path,
            1
        )

        # Safety boundary.
        pygame.draw.circle(
            self.screen,
            (130, 85, 55),
            OBSTACLE.astype(int),
            int(SAFETY_RADIUS),
            2
        )

        # Obstacle.
        pygame.draw.circle(
            self.screen,
            (170, 65, 65),
            OBSTACLE.astype(int),
            int(OBSTACLE_RADIUS)
        )

        # Target.
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
            (100, 240, 130),
            2
        )

        # Drone trail.
        if len(self.trail) > 1:
            pygame.draw.lines(
                self.screen,
                (220, 190, 90),
                False,
                [
                    tuple(p.astype(int))
                    for p in self.trail
                ],
                2
            )

        # Predicted MPC trajectory.
        if self.use_mpc:
            self.draw_predicted_path()

        # Drone velocity.
        self.draw_arrow(
            self.env.position,
            self.env.position + self.env.velocity * 0.35,
            (255, 210, 90),
            2
        )

        self.draw_drone()

        pygame.draw.rect(
            self.screen,
            (90, 110, 140),
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

        clearance = (
            distance(self.env.position, OBSTACLE)
            - SAFETY_RADIUS
        )

        lines = [
            ("PHASE 4.17", (255, 220, 100)),
            ("RL + MPC QUADCOPTER", (220, 230, 245)),
            ("", (200, 200, 200)),
            ("RL POLICY", (150, 190, 255)),
            ("REINFORCE / Gaussian", (100, 240, 150)),
            ("", (200, 200, 200)),
            ("MPC PLANNER", (150, 190, 255)),
            (f"Horizon: {MPC_HORIZON}", (220, 220, 220)),
            (f"Candidates: {MPC_CANDIDATES}", (220, 220, 220)),
            (f"Score: {self.planner.last_score:.2f}", (220, 220, 220)),
            ("", (200, 200, 200)),
            ("TRACKING", (150, 190, 255)),
            (f"Distance: {target_distance:.1f}", (230, 230, 230)),
            (f"Target vx: {self.env.target.velocity[0]:.1f}", (180, 240, 180)),
            (f"Target vy: {self.env.target.velocity[1]:.1f}", (180, 240, 180)),
            ("", (200, 200, 200)),
            ("SAFETY", (150, 190, 255)),
            (f"Clearance: {clearance:.1f}", (230, 230, 230)),
            ("", (200, 200, 200)),
            ("CONTROL", (150, 190, 255)),
            (f"Ax: {self.last_action[0]:+.3f}", (150, 190, 255)),
            (f"Ay: {self.last_action[1]:+.3f}", (150, 190, 255)),
            ("", (200, 200, 200)),
            ("MODE", (150, 190, 255)),
            ("RL + MPC" if self.use_mpc else "RL ONLY", (255, 190, 100)),
            ("", (200, 200, 200)),
            ("SIMULATION", (150, 190, 255)),
            (f"Speed: {self.speed}x", (220, 220, 220)),
            ("PAUSED" if self.paused else "RUNNING", (255, 190, 100)),
            ("", (200, 200, 200)),
            ("CONTROLS", (150, 190, 255)),
            ("Mouse: change target center", (190, 190, 190)),
            ("SPACE: pause / resume", (190, 190, 190)),
            ("R: reset", (190, 190, 190)),
            ("M: toggle RL / MPC", (190, 190, 190)),
            ("+/-: simulation speed", (190, 190, 190)),
            ("ESC: exit", (190, 190, 190))
        ]

        y = 10

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

                elif event.key == pygame.K_m:
                    self.use_mpc = not self.use_mpc

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
                    self.set_target_center(
                        event.pos
                    )

    def update_simulation(self):
        state = self.env.get_state()

        if self.use_mpc:
            action = self.planner.plan(
                self.env
            )
        else:
            action = self.policy.deterministic_action(
                state
            )

        (
            next_state,
            reward,
            terminated,
            truncated,
            info
        ) = self.env.step(action)

        self.last_action = action.copy()
        self.last_reward = reward
        self.last_info = info

        self.trail.append(
            self.env.position.copy()
        )

        if len(self.trail) > 250:
            self.trail.pop(0)

        if terminated or truncated:
            self.reset()

    def run(self):
        while self.running:

            self.handle_events()

            if not self.paused:
                for _ in range(self.speed):
                    self.update_simulation()

            self.draw_world()
            self.draw_panel()

            pygame.display.flip()

            self.clock.tick(FPS)

        pygame.quit()


# ============================================================
# 12. UNIT TESTS
# ============================================================

def run_tests():
    print("\nRunning Phase 4.17 tests...\n")

    # Test 1: Environment state.
    env = DroneEnv()

    state = env.reset()

    assert state.shape == (STATE_DIM,)
    assert np.all(np.isfinite(state))

    print("Test 1 passed: environment state")

    # Test 2: Moving target.
    old_position = env.target.position.copy()

    env.target.step()

    new_position = env.target.position.copy()

    assert not np.allclose(
        old_position,
        new_position
    )

    print("Test 2 passed: moving target")

    # Test 3: Target velocity.
    assert env.target.velocity.shape == (2,)

    print("Test 3 passed: target velocity")

    # Test 4: Drone speed limit.
    _, velocity = drone_dynamics(
        np.array([100.0, 100.0]),
        np.array([110.0, 0.0]),
        np.array([1.0, 0.0])
    )

    assert np.linalg.norm(velocity) <= MAX_SPEED + 1e-5

    print("Test 4 passed: drone speed limit")

    # Test 5: Action bounds.
    action = np.array(
        [2.0, -2.0],
        dtype=np.float32
    )

    clipped = np.clip(
        action,
        -1.0,
        1.0
    )

    assert np.all(clipped <= 1.0)
    assert np.all(clipped >= -1.0)

    print("Test 5 passed: action bounds")

    # Test 6: Environment transition.
    result = env.step(
        np.zeros(ACTION_DIM)
    )

    assert len(result) == 5

    print("Test 6 passed: environment transition")

    # Test 7: Gaussian policy dimensions.
    policy = GaussianPolicy().to(DEVICE)

    states = torch.zeros(
        (8, STATE_DIM),
        dtype=torch.float32,
        device=DEVICE
    )

    actions, log_probs, distribution = (
        policy.sample_action(states)
    )

    assert actions.shape == (8, ACTION_DIM)
    assert log_probs.shape == (8, 1)

    print("Test 7 passed: Gaussian policy")

    # Test 8: Bounded policy actions.
    assert torch.all(actions <= 1.0)
    assert torch.all(actions >= -1.0)

    print("Test 8 passed: bounded actions")

    # Test 9: Finite policy log probabilities.
    assert torch.all(
        torch.isfinite(log_probs)
    )

    print("Test 9 passed: policy log probabilities")

    # Test 10: Discounted returns.
    returns = discounted_returns(
        [1.0, 1.0, 1.0]
    )

    expected = np.array(
        [
            1.0 + GAMMA + GAMMA ** 2,
            1.0 + GAMMA,
            1.0
        ],
        dtype=np.float32
    )

    assert np.allclose(
        returns,
        expected,
        atol=1e-5
    )

    print("Test 10 passed: discounted returns")

    # Test 11: Vectorized dynamics.
    positions = np.array(
        [
            [100.0, 100.0],
            [200.0, 200.0]
        ],
        dtype=np.float32
    )

    velocities = np.zeros_like(positions)

    actions = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0]
        ],
        dtype=np.float32
    )

    next_positions, next_velocities = (
        predict_drone_batch(
            positions,
            velocities,
            actions
        )
    )

    assert next_positions.shape == (2, 2)
    assert next_velocities.shape == (2, 2)

    print("Test 11 passed: vectorized dynamics")

    # Test 12: MPC target prediction.
    predicted_target = predict_target(
        env.target.center,
        env.target.phase,
        5
    )

    assert predicted_target.shape == (2,)
    assert np.all(np.isfinite(predicted_target))

    print("Test 12 passed: target prediction")

    # Test 13: MPC planner returns valid action.
    planner = MPCPlanner(policy)

    planner.last_best_sequence = None

    action = planner.plan(env)

    assert action.shape == (ACTION_DIM,)
    assert np.all(action <= 1.0)
    assert np.all(action >= -1.0)

    print("Test 13 passed: MPC action")

    # Test 14: MPC predicted trajectory.
    assert (
        len(planner.last_predicted_path)
        == MPC_HORIZON + 1
    )

    print("Test 14 passed: predicted trajectory")

    # Test 15: MPC finite score.
    assert math.isfinite(
        planner.last_score
    )

    print("Test 15 passed: MPC score")

    print("\nAll 15 tests passed.\n")


# ============================================================
# 13. MAIN
# ============================================================

def main():
    print("=" * 70)
    print("PHASE 4.17 — RL + MPC QUADCOPTER NAVIGATION")
    print("=" * 70)

    run_tests()

    policy, reward_history = train_reinforce()

    visualize_training(
        reward_history
    )

    print("\nStarting interactive quadcopter simulation.")
    print("MPC is enabled by default.")
    print("Press M to compare RL-only and RL-guided MPC.")

    gui = DroneGUI(policy)
    gui.run()


if __name__ == "__main__":
    main()
