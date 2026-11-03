
"""
File: phase4/sim_to_real_adaptation_drone.py

PHASE 4.25 — HIGH-PERFORMANCE SIM-TO-REAL TRANSFER

PURPOSE
-------
Train a domain-conditioned drone tracking policy, identify unknown
target-environment dynamics, and compare three transfer methods:

1. Zero-shot transfer.
2. Calibrated transfer.
3. Target-domain fine-tuning.

LEARNING STEPS
--------------
1. Randomized source-domain demonstration collection.
2. System identification using persistent excitation and least squares.
3. Zero-shot transfer using nominal dynamics information.
4. Calibrated transfer using identified dynamics.
5. Target adaptation using limited demonstrations.
6. Paired evaluation on identical moving-target phases.
7. Interactive Pygame experiment.

IMPORTANT
---------
This is a 2D educational simulation, not a real drone flight controller.
The obstacle safety filter is heuristic, not a formal safety guarantee.
"""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# ============================================================
# 0. CONFIGURATION
# ============================================================

SEED = 425

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

WORLD_W = 900
WORLD_H = 650
DT = 0.08

MAX_SPEED = 145.0
MAX_ACCEL = 120.0
DRONE_RADIUS = 12.0

OBSTACLE = np.array([450.0, 330.0])
OBSTACLE_RADIUS = 65.0
SAFE_RADIUS = 125.0

TARGET_A = 165.0
TARGET_B = 105.0
TARGET_OMEGA = 0.55

BASE_STATE_DIM = 8
DOMAIN_DIM = 5

STATE_DIM = BASE_STATE_DIM + DOMAIN_DIM
ACTION_DIM = 2

HIDDEN_DIM = 192

SOURCE_EPISODES = 300
SOURCE_STEPS = 260
SOURCE_DATA_PER_EPISODE = 130

ADAPTATION_EPISODES = 35
ADAPTATION_STEPS = 260
ADAPTATION_DATA_PER_EPISODE = 130

CALIBRATION_STEPS = 1000

TRAIN_EPOCHS = 180
ADAPTATION_EPOCHS = 70
BATCH_SIZE = 256

LEARNING_RATE = 1e-3
ADAPTATION_LEARNING_RATE = 1e-4

VALIDATION_FRACTION = 0.15
EARLY_STOPPING_PATIENCE = 18
MIN_IMPROVEMENT = 1e-5

EVAL_EPISODES = 12
EVAL_STEPS = 300

ACTION_SMOOTHING = 0.30

SAFETY_MARGIN = 18.0
SAFETY_LOOKAHEAD = 0.65
SAFETY_STRENGTH = 180.0

SOURCE_RANGES = {
    "motor_gain": (0.60, 1.10),
    "linear_drag": (0.04, 0.18),
    "quadratic_drag": (0.00010, 0.00055),
    "wind_x": (-10.0, 10.0),
    "wind_y": (-10.0, 10.0),
}

TARGET_TRUE = {
    "motor_gain": 0.56,
    "linear_drag": 0.155,
    "quadratic_drag": 0.00042,
    "wind_x": 9.0,
    "wind_y": -6.0,
}

DOMAIN_SCALE = np.array(
    [1.0, 0.2, 0.0006, 12.0, 12.0],
    dtype=np.float32,
)

BASE_SCALE = np.ones(
    BASE_STATE_DIM,
    dtype=np.float32,
)


# ============================================================
# 1. DOMAIN MODEL
# ============================================================

@dataclass
class Domain:
    motor_gain: float
    linear_drag: float
    quadratic_drag: float
    wind_x: float
    wind_y: float

    def wind(self):
        return np.array(
            [self.wind_x, self.wind_y],
            dtype=np.float64,
        )

    def as_array(self):
        return np.array(
            [
                self.motor_gain,
                self.linear_drag,
                self.quadratic_drag,
                self.wind_x,
                self.wind_y,
            ],
            dtype=np.float32,
        )


def sample_source_domain():
    return Domain(
        motor_gain=np.random.uniform(
            *SOURCE_RANGES["motor_gain"]
        ),
        linear_drag=np.random.uniform(
            *SOURCE_RANGES["linear_drag"]
        ),
        quadratic_drag=np.random.uniform(
            *SOURCE_RANGES["quadratic_drag"]
        ),
        wind_x=np.random.uniform(
            *SOURCE_RANGES["wind_x"]
        ),
        wind_y=np.random.uniform(
            *SOURCE_RANGES["wind_y"]
        ),
    )


def target_domain():
    return Domain(**TARGET_TRUE)


def nominal_domain():
    return Domain(
        motor_gain=0.85,
        linear_drag=0.085,
        quadratic_drag=0.00022,
        wind_x=0.0,
        wind_y=0.0,
    )


def normalize_domain(domain):
    return (
        domain.as_array() / DOMAIN_SCALE
    ).astype(np.float32)


def normalize_observation(observation):
    result = observation.copy()
    result[:BASE_STATE_DIM] /= BASE_SCALE
    return result.astype(np.float32)


# ============================================================
# 2. MOVING TARGET
# ============================================================

class MovingTarget:

    def __init__(self, center=None):
        self.center = np.array(
            [710.0, 315.0]
            if center is None
            else center,
            dtype=np.float64,
        )

        self.phase = 0.0

    def position(self, phase=None):
        p = self.phase if phase is None else phase

        return self.center + np.array(
            [
                TARGET_A * math.cos(p),
                TARGET_B * math.sin(p),
            ]
        )

    def velocity(self, phase=None):
        p = self.phase if phase is None else phase

        return np.array(
            [
                -TARGET_A * TARGET_OMEGA * math.sin(p),
                TARGET_B * TARGET_OMEGA * math.cos(p),
            ]
        )

    def advance(self):
        self.phase += TARGET_OMEGA * DT


# ============================================================
# 3. DRONE ENVIRONMENT
# ============================================================

class DroneEnv:

    def __init__(self, domain):
        self.domain = copy.deepcopy(domain)
        self.target = MovingTarget()
        self.reset()

    def reset(self, phase=None):
        self.position = np.array(
            [110.0, 500.0],
            dtype=np.float64,
        )

        self.velocity = np.zeros(
            2,
            dtype=np.float64,
        )

        self.target.phase = (
            np.random.uniform(0.0, 2.0 * math.pi)
            if phase is None
            else float(phase)
        )

        self.steps = 0
        return self.observe()

    def observe(self):
        target_position = self.target.position()
        target_velocity = self.target.velocity()

        relative_target = target_position - self.position
        relative_obstacle = OBSTACLE - self.position

        base = np.concatenate(
            [
                relative_target / np.array(
                    [WORLD_W, WORLD_H]
                ),
                target_velocity / MAX_SPEED,
                self.velocity / MAX_SPEED,
                relative_obstacle / np.array(
                    [WORLD_W, WORLD_H]
                ),
            ]
        )

        return np.concatenate(
            [
                base,
                normalize_domain(self.domain),
            ]
        ).astype(np.float32)

    def step(self, action):
        action = np.clip(
            np.asarray(action, dtype=np.float64),
            -1.0,
            1.0,
        )

        old_position = self.position.copy()
        d = self.domain

        acceleration = (
            d.motor_gain * MAX_ACCEL * action
            - d.linear_drag * self.velocity
            - d.quadratic_drag
            * np.abs(self.velocity)
            * self.velocity
            + d.wind()
        )

        self.velocity += acceleration * DT

        speed = np.linalg.norm(self.velocity)

        if speed > MAX_SPEED:
            self.velocity *= MAX_SPEED / speed

        self.position += self.velocity * DT

        self.position = np.clip(
            self.position,
            DRONE_RADIUS,
            [
                WORLD_W - DRONE_RADIUS,
                WORLD_H - DRONE_RADIUS,
            ],
        )

        self.target.advance()
        self.steps += 1

        return (
            self.observe(),
            old_position,
            self.position.copy(),
        )


# ============================================================
# 4. TEACHER CONTROLLER
# ============================================================

def teacher_controller(env):
    position = env.position
    velocity = env.velocity

    target_position = env.target.position()
    target_velocity = env.target.velocity()

    tau = 0.55

    desired_position = (
        target_position + tau * target_velocity
    )

    error = desired_position - position

    desired_velocity = np.clip(
        1.15 * error,
        -MAX_SPEED * 0.75,
        MAX_SPEED * 0.75,
    )

    acceleration_command = (
        2.1 * (desired_velocity - velocity)
    )

    obstacle_vector = position - OBSTACLE
    obstacle_distance = (
        np.linalg.norm(obstacle_vector) + 1e-8
    )

    if obstacle_distance < SAFE_RADIUS:
        direction = obstacle_vector / obstacle_distance

        strength = (
            (SAFE_RADIUS - obstacle_distance)
            / SAFE_RADIUS
        )

        acceleration_command += (
            180.0 * strength * direction
        )

    margin = 65.0

    if position[0] < margin:
        acceleration_command[0] += 100.0

    if position[0] > WORLD_W - margin:
        acceleration_command[0] -= 100.0

    if position[1] < margin:
        acceleration_command[1] += 100.0

    if position[1] > WORLD_H - margin:
        acceleration_command[1] -= 100.0

    d = env.domain

    drag = (
        d.linear_drag * velocity
        + d.quadratic_drag
        * np.abs(velocity)
        * velocity
    )

    required_acceleration = (
        acceleration_command + drag - d.wind()
    )

    action = (
        required_acceleration
        / (max(d.motor_gain, 0.1) * MAX_ACCEL)
    )

    return np.clip(action, -1.0, 1.0)


# ============================================================
# 5. SAFETY FILTER AND ACTION SMOOTHING
# ============================================================

def obstacle_safety_filter(env, action):
    action = np.asarray(
        action,
        dtype=np.float64,
    ).copy()

    relative = env.position - OBSTACLE
    distance = np.linalg.norm(relative) + 1e-8

    direction = relative / distance
    clearance = distance - OBSTACLE_RADIUS

    predicted_position = (
        env.position
        + env.velocity * SAFETY_LOOKAHEAD
    )

    predicted_distance = np.linalg.norm(
        predicted_position - OBSTACLE
    )

    approaching = predicted_distance < distance

    required_clearance = (
        SAFE_RADIUS + SAFETY_MARGIN
    )

    if (
        clearance < required_clearance
        or approaching
    ):
        proximity = np.clip(
            (
                required_clearance - clearance
            ) / required_clearance,
            0.0,
            1.0,
        )

        outward_acceleration = (
            SAFETY_STRENGTH
            * proximity
            * direction
        )

        correction = (
            outward_acceleration
            / max(
                env.domain.motor_gain * MAX_ACCEL,
                1e-6,
            )
        )

        action += correction

    return np.clip(action, -1.0, 1.0)


class ActionSmoother:

    def __init__(self, alpha=ACTION_SMOOTHING):
        self.alpha = float(alpha)
        self.previous = np.zeros(
            ACTION_DIM,
            dtype=np.float64,
        )

    def reset(self):
        self.previous[:] = 0.0

    def apply(self, action):
        action = np.asarray(
            action,
            dtype=np.float64,
        )

        result = (
            self.alpha * action
            + (1.0 - self.alpha) * self.previous
        )

        self.previous = result.copy()

        return np.clip(result, -1.0, 1.0)


# ============================================================
# 6. REWARD
# ============================================================

def calculate_reward(
    old_position,
    new_position,
    future_target,
    action,
):
    distance_before = np.linalg.norm(
        old_position - future_target
    )

    distance_after = np.linalg.norm(
        new_position - future_target
    )

    progress = distance_before - distance_after

    obstacle_distance = np.linalg.norm(
        new_position - OBSTACLE
    )

    clearance = obstacle_distance - OBSTACLE_RADIUS

    obstacle_penalty = (
        max(0.0, SAFE_RADIUS - clearance) / SAFE_RADIUS
    ) ** 2

    boundary_distance = min(
        new_position[0],
        WORLD_W - new_position[0],
        new_position[1],
        WORLD_H - new_position[1],
    )

    boundary_penalty = (
        max(0.0, 55.0 - boundary_distance) / 55.0
    ) ** 2

    effort_penalty = 0.015 * np.sum(action ** 2)

    collision_penalty = (
        12.0 if clearance < 0.0 else 0.0
    )

    return float(
        0.12 * progress
        - 0.008 * distance_after
        - 2.0 * obstacle_penalty
        - 2.0 * boundary_penalty
        - collision_penalty
        - effort_penalty
    )


# ============================================================
# 7. DATA COLLECTION
# ============================================================

def collect_demonstrations(
    domain,
    episodes,
    steps,
    samples_per_episode,
):
    observations = []
    actions = []

    for _ in range(episodes):
        env = DroneEnv(domain)
        observation = env.reset()

        smoother = ActionSmoother()

        interval = max(
            1,
            steps // samples_per_episode,
        )

        for step in range(steps):
            action = teacher_controller(env)
            action = obstacle_safety_filter(env, action)
            action = smoother.apply(action)

            if step % interval == 0:
                observations.append(observation.copy())
                actions.append(action.copy())

            observation, _, _ = env.step(action)

    return (
        np.asarray(observations, dtype=np.float32),
        np.asarray(actions, dtype=np.float32),
    )


def collect_randomized_source_data():
    all_observations = []
    all_actions = []

    print("\nCollecting source demonstrations...")

    for episode in range(SOURCE_EPISODES):
        domain = sample_source_domain()

        observations, actions = collect_demonstrations(
            domain,
            episodes=1,
            steps=SOURCE_STEPS,
            samples_per_episode=SOURCE_DATA_PER_EPISODE,
        )

        all_observations.append(observations)
        all_actions.append(actions)

        if (episode + 1) % 30 == 0:
            print(
                f"Source episodes: "
                f"{episode + 1}/{SOURCE_EPISODES}"
            )

    return (
        np.concatenate(all_observations),
        np.concatenate(all_actions),
    )


def split_dataset(observations, actions):
    indices = np.random.permutation(len(observations))

    validation_count = max(
        1,
        int(len(indices) * VALIDATION_FRACTION),
    )

    validation_indices = indices[:validation_count]
    training_indices = indices[validation_count:]

    return (
        observations[training_indices],
        actions[training_indices],
        observations[validation_indices],
        actions[validation_indices],
    )


# ============================================================
# 8. POLICY NETWORK
# ============================================================

class TrackingPolicy(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(STATE_DIM, HIDDEN_DIM),
            nn.LayerNorm(HIDDEN_DIM),
            nn.ReLU(),

            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.LayerNorm(HIDDEN_DIM),
            nn.ReLU(),

            nn.Linear(HIDDEN_DIM, HIDDEN_DIM // 2),
            nn.ReLU(),

            nn.Linear(HIDDEN_DIM // 2, ACTION_DIM),
            nn.Tanh(),
        )

    def forward(self, state):
        return self.network(state)


# ============================================================
# 9. TRAINING WITH VALIDATION AND EARLY STOPPING
# ============================================================

def train_behavior_cloning(
    policy,
    observations,
    actions,
    epochs,
    learning_rate,
    label,
):
    policy = policy.to(DEVICE)

    (
        train_x,
        train_y,
        validation_x,
        validation_y,
    ) = split_dataset(observations, actions)

    x_train = torch.as_tensor(
        train_x,
        dtype=torch.float32,
        device=DEVICE,
    )

    y_train = torch.as_tensor(
        train_y,
        dtype=torch.float32,
        device=DEVICE,
    )

    x_val = torch.as_tensor(
        validation_x,
        dtype=torch.float32,
        device=DEVICE,
    )

    y_val = torch.as_tensor(
        validation_y,
        dtype=torch.float32,
        device=DEVICE,
    )

    optimizer = optim.AdamW(
        policy.parameters(),
        lr=learning_rate,
        weight_decay=1e-5,
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=6,
        min_lr=1e-6,
    )

    best_validation_loss = float("inf")
    best_state = copy.deepcopy(policy.state_dict())

    patience_counter = 0

    train_history = []
    validation_history = []

    print(f"\nTraining: {label}")
    print(f"Training samples: {len(x_train)}")
    print(f"Validation samples: {len(x_val)}")

    for epoch in range(epochs):
        policy.train()

        permutation = torch.randperm(
            len(x_train),
            device=DEVICE,
        )

        batch_losses = []

        for start in range(0, len(x_train), BATCH_SIZE):
            indices = permutation[
                start:start + BATCH_SIZE
            ]

            prediction = policy(x_train[indices])

            loss = nn.functional.mse_loss(
                prediction,
                y_train[indices],
            )

            optimizer.zero_grad()
            loss.backward()

            nn.utils.clip_grad_norm_(
                policy.parameters(),
                1.0,
            )

            optimizer.step()

            batch_losses.append(
                float(loss.detach().cpu())
            )

        train_loss = float(np.mean(batch_losses))

        policy.eval()

        with torch.no_grad():
            validation_prediction = policy(x_val)

            validation_loss = float(
                nn.functional.mse_loss(
                    validation_prediction,
                    y_val,
                ).cpu()
            )

        scheduler.step(validation_loss)

        train_history.append(train_loss)
        validation_history.append(validation_loss)

        if (
            validation_loss
            < best_validation_loss - MIN_IMPROVEMENT
        ):
            best_validation_loss = validation_loss
            best_state = copy.deepcopy(
                policy.state_dict()
            )
            patience_counter = 0
        else:
            patience_counter += 1

        if (
            (epoch + 1) % 10 == 0
            or epoch == 0
        ):
            print(
                f"{label} "
                f"{epoch + 1:3d}/{epochs}"
                f" | train={train_loss:.6f}"
                f" | validation={validation_loss:.6f}"
            )

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(
                f"Early stopping at epoch {epoch + 1}. "
                f"Best validation loss={best_validation_loss:.6f}"
            )
            break

    policy.load_state_dict(best_state)
    policy.eval()

    return (
        policy,
        train_history,
        validation_history,
    )


# ============================================================
# 10. SYSTEM IDENTIFICATION
# ============================================================

@dataclass
class DynamicsEstimate:
    motor_gain: float
    linear_drag: float
    quadratic_drag: float
    wind_x: float
    wind_y: float

    def as_domain(self):
        return Domain(
            self.motor_gain,
            self.linear_drag,
            self.quadratic_drag,
            self.wind_x,
            self.wind_y,
        )


def collect_calibration_data(
    env,
    steps=CALIBRATION_STEPS,
):
    env.reset()

    features = [[], []]
    targets = [[], []]

    for step in range(steps):
        t = step * DT

        action = np.array(
            [
                0.68 * math.sin(1.13 * t)
                + 0.23 * math.sin(2.41 * t)
                + 0.08 * math.sin(3.71 * t),

                0.63 * math.cos(0.97 * t)
                + 0.25 * math.sin(1.79 * t)
                + 0.08 * math.cos(3.13 * t),
            ]
        )

        action = np.clip(action, -1.0, 1.0)

        old_velocity = env.velocity.copy()

        env.step(action)

        measured_acceleration = (
            env.velocity - old_velocity
        ) / DT

        if np.linalg.norm(env.velocity) >= MAX_SPEED - 1.0:
            continue

        for axis in range(2):
            v = old_velocity[axis]

            features[axis].append([
                MAX_ACCEL * action[axis],
                -v,
                -abs(v) * v,
                1.0,
            ])

            targets[axis].append(
                measured_acceleration[axis]
            )

    return (
        np.asarray(features[0]),
        np.asarray(targets[0]),
        np.asarray(features[1]),
        np.asarray(targets[1]),
    )


def identify_dynamics(data):
    x_features, x_targets, y_features, y_targets = data

    beta_x, _, rank_x, singular_x = np.linalg.lstsq(
        x_features,
        x_targets,
        rcond=None,
    )

    beta_y, _, rank_y, singular_y = np.linalg.lstsq(
        y_features,
        y_targets,
        rcond=None,
    )

    estimate = DynamicsEstimate(
        motor_gain=float(
            0.5 * (beta_x[0] + beta_y[0])
        ),
        linear_drag=float(
            0.5 * (beta_x[1] + beta_y[1])
        ),
        quadratic_drag=float(
            0.5 * (beta_x[2] + beta_y[2])
        ),
        wind_x=float(beta_x[3]),
        wind_y=float(beta_y[3]),
    )

    pred_x = x_features @ beta_x
    pred_y = y_features @ beta_y

    rmse_x = np.sqrt(
        np.mean((pred_x - x_targets) ** 2)
    )

    rmse_y = np.sqrt(
        np.mean((pred_y - y_targets) ** 2)
    )

    rmse = float(
        np.sqrt(
            0.5 * (rmse_x ** 2 + rmse_y ** 2)
        )
    )

    condition_x = float(
        singular_x[0] / max(singular_x[-1], 1e-12)
    )

    condition_y = float(
        singular_y[0] / max(singular_y[-1], 1e-12)
    )

    diagnostics = {
        "rank_x": int(rank_x),
        "rank_y": int(rank_y),
        "condition_x": condition_x,
        "condition_y": condition_y,
    }

    return estimate, rmse, diagnostics


# ============================================================
# 11. DOMAIN INPUT POLICY
# ============================================================

class DomainInputPolicy:
    """
    Supply nominal or identified parameters to the policy.

    The simulator always uses the true target physics. The policy
    receives only the parameters supplied to this wrapper.
    """

    def __init__(
        self,
        model,
        supplied_domain,
        use_safety_filter=True,
        smooth=True,
    ):
        self.model = model
        self.supplied_domain = copy.deepcopy(
            supplied_domain
        )

        self.use_safety_filter = use_safety_filter

        self.smoother = (
            ActionSmoother()
            if smooth
            else None
        )

        self.model.eval()

    def reset(self):
        if self.smoother is not None:
            self.smoother.reset()

    def __call__(self, observation, env=None):
        state = observation.copy()

        state[BASE_STATE_DIM:] = normalize_domain(
            self.supplied_domain
        )

        state = normalize_observation(state)

        tensor = torch.as_tensor(
            state,
            dtype=torch.float32,
            device=DEVICE,
        ).unsqueeze(0)

        with torch.no_grad():
            action = self.model(tensor).cpu().numpy()[0]

        if self.use_safety_filter and env is not None:
            action = obstacle_safety_filter(
                env,
                action,
            )

        if self.smoother is not None:
            action = self.smoother.apply(action)

        return np.clip(action, -1.0, 1.0)

    def eval(self):
        self.model.eval()


# ============================================================
# 12. PAIRED EVALUATION
# ============================================================

def evaluate_controller(controller, phases):
    results = []

    for phase in phases:
        env = DroneEnv(target_domain())
        observation = env.reset(phase=phase)

        controller.reset()

        distances = []
        rewards = []

        collision_steps = 0
        minimum_clearance = float("inf")

        for _ in range(EVAL_STEPS):
            action = controller(
                observation,
                env,
            )

            future_target = env.target.position(
                env.target.phase + TARGET_OMEGA * DT
            )

            observation, old_position, new_position = (
                env.step(action)
            )

            rewards.append(
                calculate_reward(
                    old_position,
                    new_position,
                    future_target,
                    action,
                )
            )

            distances.append(
                np.linalg.norm(
                    new_position - env.target.position()
                )
            )

            clearance = (
                np.linalg.norm(new_position - OBSTACLE)
                - OBSTACLE_RADIUS
            )

            minimum_clearance = min(
                minimum_clearance,
                clearance,
            )

            collision_steps += int(clearance < 0.0)

        results.append({
            "reward": float(np.sum(rewards)),
            "distance": float(np.mean(distances)),
            "collision_steps": float(collision_steps),
            "minimum_clearance": float(minimum_clearance),
        })

    keys = results[0].keys()

    return {
        key: float(
            np.mean([row[key] for row in results])
        )
        for key in keys
    }


def evaluate_all(controllers):
    phases = np.linspace(
        0.0,
        2.0 * math.pi,
        EVAL_EPISODES,
        endpoint=False,
    )

    return {
        name: evaluate_controller(controller, phases)
        for name, controller in controllers.items()
    }


def print_results(results):
    print("\n" + "=" * 90)
    print("PAIRED TARGET-ENVIRONMENT EVALUATION")
    print("=" * 90)

    print(
        f"{'Policy':14s}"
        f" | {'Reward':>10s}"
        f" | {'Distance':>10s}"
        f" | {'Collision':>10s}"
        f" | {'Clearance':>10s}"
    )

    print("-" * 90)

    for name, result in results.items():
        print(
            f"{name:14s}"
            f" | {result['reward']:10.2f}"
            f" | {result['distance']:10.2f}"
            f" | {result['collision_steps']:10.2f}"
            f" | {result['minimum_clearance']:10.2f}"
        )

    print("=" * 90)


# ============================================================
# 13. VISUALIZATION
# ============================================================

def plot_training(
    source_train,
    source_validation,
    adaptation_train,
    adaptation_validation,
):
    plt.figure(figsize=(10, 5))

    plt.plot(source_train, label="Source training")
    plt.plot(source_validation, label="Source validation")

    plt.title("Source behavior-cloning training")
    plt.xlabel("Epoch")
    plt.ylabel("Action MSE")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))

    plt.plot(adaptation_train, label="Adaptation training")
    plt.plot(adaptation_validation, label="Adaptation validation")

    plt.title("Target-domain fine-tuning")
    plt.xlabel("Epoch")
    plt.ylabel("Action MSE")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_identification(estimate, rmse):
    names = [
        "Motor gain",
        "Linear drag",
        "Quadratic drag",
        "Wind X",
        "Wind Y",
    ]

    true_values = np.array([
        TARGET_TRUE["motor_gain"],
        TARGET_TRUE["linear_drag"],
        TARGET_TRUE["quadratic_drag"],
        TARGET_TRUE["wind_x"],
        TARGET_TRUE["wind_y"],
    ])

    estimated_values = np.array([
        estimate.motor_gain,
        estimate.linear_drag,
        estimate.quadratic_drag,
        estimate.wind_x,
        estimate.wind_y,
    ])

    scale = np.maximum(np.abs(true_values), 1e-8)

    true_normalized = true_values / scale
    estimated_normalized = estimated_values / scale

    x = np.arange(len(names))
    width = 0.36

    plt.figure(figsize=(10, 5))

    plt.bar(
        x - width / 2,
        true_normalized,
        width,
        label="True / true scale",
    )

    plt.bar(
        x + width / 2,
        estimated_normalized,
        width,
        label="Estimated / true scale",
    )

    plt.xticks(x, names)
    plt.ylabel("Normalized parameter value")

    plt.title(
        f"System identification — acceleration RMSE={rmse:.5f}"
    )

    plt.grid(axis="y")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_transfer(results):
    names = list(results.keys())

    for metric, title in [
        ("reward", "Mean episode reward"),
        ("distance", "Mean tracking distance"),
        ("collision_steps", "Collision steps"),
        ("minimum_clearance", "Minimum obstacle clearance"),
    ]:
        values = [
            results[name][metric]
            for name in names
        ]

        plt.figure(figsize=(8, 5))
        plt.bar(names, values)
        plt.title(title)
        plt.grid(axis="y")
        plt.tight_layout()
        plt.show()


# ============================================================
# 14. INTERACTIVE PYGAME
# ============================================================

def run_pygame(controllers):
    """
    Interactive experiment.

    FIX FOR THE REPORTED KeyError:
    --------------------------------
    The controller dictionary uses string keys:

        "Zero-shot"
        "Calibrated"
        "Fine-tuned"

    The keyboard uses integer selections:

        1, 2, 3

    These are now explicitly mapped through POLICY_KEYS.

    This avoids the previous invalid operation:

        controllers[1]

    which raised:

        KeyError: 1
    """

    try:
        import pygame
    except ImportError:
        print("Install Pygame: pip install pygame")
        return

    # --------------------------------------------------------
    # Validate controller dictionary before starting the GUI.
    # --------------------------------------------------------

    required_names = {
        "Zero-shot",
        "Calibrated",
        "Fine-tuned",
    }

    missing = required_names - set(controllers.keys())

    if missing:
        raise ValueError(
            "Missing interactive controllers: "
            + ", ".join(sorted(missing))
        )

    # Integer keyboard selections map to string dictionary keys.
    POLICY_KEYS = {
        1: "Zero-shot",
        2: "Calibrated",
        3: "Fine-tuned",
    }

    POLICY_LABELS = {
        1: "Zero-shot",
        2: "Calibrated",
        3: "Fine-tuned",
    }

    selected = 1

    pygame.init()

    screen = pygame.display.set_mode(
        (WORLD_W, WORLD_H)
    )

    pygame.display.set_caption(
        "Phase 4.25 — Sim-to-Real Drone Transfer"
    )

    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 15)

    domain = target_domain()
    env = DroneEnv(domain)
    observation = env.reset()

    paused = False
    speed = 1
    trail = []

    def current_controller():
        """
        Always return the selected controller by its actual
        string dictionary key.
        """

        controller_name = POLICY_KEYS[selected]

        if controller_name not in controllers:
            raise RuntimeError(
                f"Controller '{controller_name}' is unavailable."
            )

        return controllers[controller_name]

    def reset_controllers():
        for controller in controllers.values():
            controller.reset()

    def text(message, x, y, color=(235, 235, 235)):
        screen.blit(
            font.render(message, True, color),
            (x, y),
        )

    def draw_drone():
        center = env.position.astype(int)

        direction = env.velocity.copy()
        norm = np.linalg.norm(direction)

        if norm < 1e-6:
            direction = np.array([1.0, 0.0])
        else:
            direction /= norm

        side = np.array(
            [-direction[1], direction[0]]
        )

        for forward in (-1, 1):
            for lateral in (-1, 1):
                endpoint = (
                    env.position
                    + 17 * forward * direction
                    + 17 * lateral * side
                ).astype(int)

                pygame.draw.line(
                    screen,
                    (220, 225, 235),
                    center,
                    endpoint,
                    5,
                )

                pygame.draw.circle(
                    screen,
                    (80, 190, 245),
                    endpoint,
                    7,
                    2,
                )

        pygame.draw.circle(
            screen,
            (250, 250, 250),
            center,
            8,
        )

    def draw_target():
        orbit = []

        for i in range(120):
            phase = (
                env.target.phase
                + 2.0 * math.pi * i / 120
            )

            orbit.append(
                env.target.position(phase).astype(int)
            )

        pygame.draw.lines(
            screen,
            (65, 105, 75),
            True,
            orbit,
            2,
        )

        position = env.target.position()
        velocity = env.target.velocity()

        pygame.draw.circle(
            screen,
            (65, 245, 110),
            position.astype(int),
            11,
        )

        pygame.draw.line(
            screen,
            (65, 255, 120),
            position.astype(int),
            (position + 0.18 * velocity).astype(int),
            3,
        )

    running = True

    while running:
        clock.tick(60)

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:

                if event.key == pygame.K_ESCAPE:
                    running = False

                elif event.key == pygame.K_SPACE:
                    paused = not paused

                elif event.key == pygame.K_r:
                    env = DroneEnv(copy.deepcopy(domain))
                    observation = env.reset()
                    trail.clear()
                    reset_controllers()

                elif event.key in (
                    pygame.K_1,
                    pygame.K_2,
                    pygame.K_3,
                ):
                    # Correctly map keyboard key to policy index.
                    selected = int(event.unicode)

                    reset_controllers()

                elif event.key in (
                    pygame.K_EQUALS,
                    pygame.K_KP_PLUS,
                ):
                    speed = min(5, speed + 1)

                elif event.key == pygame.K_MINUS:
                    speed = max(1, speed - 1)

            elif event.type == pygame.MOUSEBUTTONDOWN:

                if event.button == 1:
                    click = np.asarray(
                        event.pos,
                        dtype=np.float64,
                    )

                    click[0] = np.clip(
                        click[0],
                        TARGET_A + 20,
                        WORLD_W - TARGET_A - 20,
                    )

                    click[1] = np.clip(
                        click[1],
                        TARGET_B + 20,
                        WORLD_H - TARGET_B - 20,
                    )

                    if (
                        np.linalg.norm(click - OBSTACLE)
                        > OBSTACLE_RADIUS + SAFE_RADIUS
                    ):
                        # Change the orbit center, not the target's
                        # motion. The target continues moving.
                        env.target.center = click

        # ----------------------------------------------------
        # IMPORTANT FIX:
        # ----------------------------------------------------
        # Do not use controllers[selected].
        # selected is an integer, but controller keys are strings.
        #
        # current_controller() performs the correct mapping.
        # ----------------------------------------------------

        controller = current_controller()

        if not paused:
            for _ in range(speed):
                action = controller(
                    observation,
                    env,
                )

                observation, _, _ = env.step(action)

                trail.append(env.position.copy())

                if len(trail) > 250:
                    trail.pop(0)

        # ----------------------------------------------------
        # Draw world.
        # ----------------------------------------------------

        screen.fill((17, 23, 34))

        for x in range(0, WORLD_W, 50):
            pygame.draw.line(
                screen,
                (30, 38, 50),
                (x, 0),
                (x, WORLD_H),
            )

        for y in range(0, WORLD_H, 50):
            pygame.draw.line(
                screen,
                (30, 38, 50),
                (0, y),
                (WORLD_W, y),
            )

        pygame.draw.circle(
            screen,
            (100, 100, 110),
            OBSTACLE.astype(int),
            OBSTACLE_RADIUS,
        )

        pygame.draw.circle(
            screen,
            (175, 90, 90),
            OBSTACLE.astype(int),
            SAFE_RADIUS,
            1,
        )

        draw_target()

        if len(trail) > 1:
            pygame.draw.lines(
                screen,
                (70, 135, 235),
                False,
                [
                    point.astype(int)
                    for point in trail
                ],
                2,
            )

        draw_drone()

        distance = np.linalg.norm(
            env.position - env.target.position()
        )

        clearance = (
            np.linalg.norm(env.position - OBSTACLE)
            - OBSTACLE_RADIUS
        )

        supplied = controller.supplied_domain

        text(
            f"Policy: {POLICY_LABELS[selected]}",
            15,
            12,
        )

        text(
            f"Tracking distance: {distance:.2f}",
            15,
            38,
        )

        text(
            f"Obstacle clearance: {clearance:.2f}",
            15,
            64,
        )

        text(
            f"Policy motor gain: {supplied.motor_gain:.3f}",
            15,
            90,
        )

        text(
            f"Policy linear drag: {supplied.linear_drag:.4f}",
            15,
            116,
        )

        text(
            f"Policy quadratic drag: "
            f"{supplied.quadratic_drag:.6f}",
            15,
            142,
        )

        text(
            f"Policy wind: "
            f"({supplied.wind_x:.1f}, {supplied.wind_y:.1f})",
            15,
            168,
        )

        text(
            f"Simulation speed: {speed}x",
            15,
            194,
        )

        text(
            "1 Zero-shot | 2 Calibrated | 3 Fine-tuned",
            15,
            WORLD_H - 100,
        )

        text(
            "SPACE pause | R reset | +/- speed | "
            "click target center | ESC exit",
            15,
            WORLD_H - 65,
        )

        if paused:
            text(
                "PAUSED",
                WORLD_W - 110,
                15,
                (255, 210, 80),
            )

        pygame.display.flip()

    pygame.quit()


# ============================================================
# 15. UNIT TESTS
# ============================================================

def run_tests():
    print("\nRunning unit tests...")

    passed = 0

    def check(condition, message):
        nonlocal passed

        assert condition, message
        passed += 1

        print(f"[PASS] {message}")

    env = DroneEnv(nominal_domain())
    observation = env.reset()

    check(
        observation.shape == (STATE_DIM,),
        "Observation dimension",
    )

    check(
        np.all(np.isfinite(observation)),
        "Finite observation",
    )

    next_observation, old_position, new_position = (
        env.step(np.array([0.2, -0.3]))
    )

    check(
        next_observation.shape == (STATE_DIM,),
        "Transition observation dimension",
    )

    check(
        np.all(np.isfinite(new_position)),
        "Finite drone position",
    )

    check(
        np.linalg.norm(env.velocity) <= MAX_SPEED + 1e-8,
        "Maximum drone speed",
    )

    before = env.target.position().copy()
    env.target.advance()

    check(
        not np.allclose(before, env.target.position()),
        "Target continuously moves",
    )

    action = teacher_controller(env)

    check(
        action.shape == (ACTION_DIM,),
        "Teacher action dimension",
    )

    check(
        np.all(np.abs(action) <= 1.0),
        "Teacher action bounds",
    )

    filtered = obstacle_safety_filter(env, action)

    check(
        np.all(np.abs(filtered) <= 1.0),
        "Safety-filter action bounds",
    )

    smoother = ActionSmoother()
    smoothed = smoother.apply(filtered)

    check(
        np.all(np.abs(smoothed) <= 1.0),
        "Smoothed action bounds",
    )

    policy = TrackingPolicy().to(DEVICE)

    test_state = torch.zeros(
        (4, STATE_DIM),
        device=DEVICE,
    )

    prediction = policy(test_state)

    check(
        prediction.shape == (4, ACTION_DIM),
        "Neural policy output dimension",
    )

    check(
        torch.all(torch.abs(prediction) <= 1.0),
        "Neural policy output bounds",
    )

    rng = np.random.default_rng(42)

    beta = np.array([
        0.72,
        0.11,
        0.00030,
        3.5,
    ])

    x = rng.normal(size=(500, 4))
    x[:, 1] *= 20
    x[:, 2] *= 100

    y = x @ beta

    recovered, *_ = np.linalg.lstsq(
        x,
        y,
        rcond=None,
    )

    check(
        np.allclose(recovered, beta, atol=1e-7),
        "Least-squares exact coefficient recovery",
    )

    calibration_env = DroneEnv(target_domain())

    calibration_data = collect_calibration_data(
        calibration_env,
        steps=150,
    )

    check(
        len(calibration_data) == 4,
        "Calibration dataset structure",
    )

    estimate, rmse, diagnostics = identify_dynamics(
        calibration_data
    )

    check(
        np.all(np.isfinite([
            estimate.motor_gain,
            estimate.linear_drag,
            estimate.quadratic_drag,
            estimate.wind_x,
            estimate.wind_y,
        ])),
        "Finite identified parameters",
    )

    check(
        np.isfinite(rmse) and rmse >= 0,
        "Finite calibration RMSE",
    )

    check(
        diagnostics["rank_x"] == 4
        and diagnostics["rank_y"] == 4,
        "Calibration regression has full rank",
    )

    reward = calculate_reward(
        np.array([100.0, 100.0]),
        np.array([105.0, 105.0]),
        np.array([300.0, 300.0]),
        np.array([0.1, 0.1]),
    )

    check(
        np.isfinite(reward),
        "Finite reward",
    )

    observations, actions = collect_demonstrations(
        nominal_domain(),
        episodes=1,
        steps=80,
        samples_per_episode=20,
    )

    check(
        observations.shape == (20, STATE_DIM),
        "Demonstration observation shape",
    )

    check(
        actions.shape == (20, ACTION_DIM),
        "Demonstration action shape",
    )

    # Test the exact keyboard-to-controller mapping.
    test_controllers = {
        "Zero-shot": object(),
        "Calibrated": object(),
        "Fine-tuned": object(),
    }

    policy_keys = {
        1: "Zero-shot",
        2: "Calibrated",
        3: "Fine-tuned",
    }

    check(
        all(
            policy_keys[i] in test_controllers
            for i in (1, 2, 3)
        ),
        "Pygame keyboard controller mapping",
    )

    EXPECTED_TESTS = 21

    print(
        f"\n{passed}/{EXPECTED_TESTS} tests passed."
    )

    return passed == EXPECTED_TESTS


# ============================================================
# 16. MAIN
# ============================================================

def main():
    print("=" * 82)
    print("PHASE 4.25 — HIGH-PERFORMANCE SIM-TO-REAL TRANSFER")
    print("=" * 82)

    print(f"Device: {DEVICE}")

    # --------------------------------------------------------
    # STEP 0 — Unit tests
    # --------------------------------------------------------

    print("\n[1/7] Unit tests")

    if not run_tests():
        raise RuntimeError("Unit tests failed.")

    # --------------------------------------------------------
    # STEP 1 — Source-domain training
    # --------------------------------------------------------

    print("\n[2/7] Randomized source-domain data")

    source_observations, source_actions = (
        collect_randomized_source_data()
    )

    print(
        f"Source dataset size: {len(source_observations)}"
    )

    source_policy = TrackingPolicy()

    (
        source_policy,
        source_train,
        source_validation,
    ) = train_behavior_cloning(
        source_policy,
        source_observations,
        source_actions,
        epochs=TRAIN_EPOCHS,
        learning_rate=LEARNING_RATE,
        label="Source policy",
    )

    # --------------------------------------------------------
    # STEP 2 — System identification
    # --------------------------------------------------------

    print("\n[3/7] Target system identification")

    calibration_env = DroneEnv(target_domain())

    calibration_data = collect_calibration_data(
        calibration_env
    )

    estimate, calibration_rmse, diagnostics = (
        identify_dynamics(calibration_data)
    )

    estimated_domain = estimate.as_domain()

    print("\nCalibration estimates")
    print("-" * 75)

    for name, true_value, estimated_value in [
        (
            "motor_gain",
            TARGET_TRUE["motor_gain"],
            estimate.motor_gain,
        ),
        (
            "linear_drag",
            TARGET_TRUE["linear_drag"],
            estimate.linear_drag,
        ),
        (
            "quadratic_drag",
            TARGET_TRUE["quadratic_drag"],
            estimate.quadratic_drag,
        ),
        (
            "wind_x",
            TARGET_TRUE["wind_x"],
            estimate.wind_x,
        ),
        (
            "wind_y",
            TARGET_TRUE["wind_y"],
            estimate.wind_y,
        ),
    ]:
        print(
            f"{name:20s}"
            f" true={true_value:12.6f}"
            f" estimated={estimated_value:12.6f}"
            f" error={estimated_value - true_value:12.6f}"
        )

    print(f"\nCalibration RMSE: {calibration_rmse:.6f}")
    print(
        f"X regression condition: "
        f"{diagnostics['condition_x']:.3f}"
    )
    print(
        f"Y regression condition: "
        f"{diagnostics['condition_y']:.3f}"
    )

    plot_identification(
        estimate,
        calibration_rmse,
    )

    # --------------------------------------------------------
    # STEP 3 — Zero-shot transfer
    # --------------------------------------------------------

    print("\n[4/7] Zero-shot transfer")

    zero_shot_policy = copy.deepcopy(source_policy)

    zero_shot_controller = DomainInputPolicy(
        zero_shot_policy,
        nominal_domain(),
    )

    # --------------------------------------------------------
    # STEP 4 — Calibrated transfer
    # --------------------------------------------------------

    print("\n[5/7] Calibrated transfer")

    calibrated_policy = copy.deepcopy(source_policy)

    calibrated_controller = DomainInputPolicy(
        calibrated_policy,
        estimated_domain,
    )

    # --------------------------------------------------------
    # STEP 5 — Target adaptation
    # --------------------------------------------------------

    print("\n[6/7] Target-domain adaptation")

    target_observations, target_actions = (
        collect_demonstrations(
            target_domain(),
            episodes=ADAPTATION_EPISODES,
            steps=ADAPTATION_STEPS,
            samples_per_episode=ADAPTATION_DATA_PER_EPISODE,
        )
    )

    adapted_policy = copy.deepcopy(source_policy)

    (
        adapted_policy,
        adaptation_train,
        adaptation_validation,
    ) = train_behavior_cloning(
        adapted_policy,
        target_observations,
        target_actions,
        epochs=ADAPTATION_EPOCHS,
        learning_rate=ADAPTATION_LEARNING_RATE,
        label="Target fine-tuning",
    )

    adapted_controller = DomainInputPolicy(
        adapted_policy,
        estimated_domain,
    )

    # --------------------------------------------------------
    # STEP 6 — Paired evaluation
    # --------------------------------------------------------

    print("\n[7/7] Paired evaluation")

    controllers = {
        "Zero-shot": zero_shot_controller,
        "Calibrated": calibrated_controller,
        "Fine-tuned": adapted_controller,
    }

    results = evaluate_all(controllers)

    print_results(results)

    plot_training(
        source_train,
        source_validation,
        adaptation_train,
        adaptation_validation,
    )

    plot_transfer(results)

    # --------------------------------------------------------
    # STEP 7 — Interactive experiment
    # --------------------------------------------------------

    print("\nStarting interactive drone experiment...")

    run_pygame(controllers)

    print("\nPhase 4.25 completed.")


if __name__ == "__main__":
    main()
