"""
File: phase4/ensemble_mpc_drone.py

PHASE 4.19 — IMPROVED ENSEMBLE DYNAMICS + CEM-MPC
=================================================

IMPROVEMENTS
------------
1. Cross-Entropy Method (CEM) action-sequence optimization.
2. Warm-start MPC using the previous optimal sequence.
3. Position + velocity tracking objective.
4. Stronger terminal-state objective.
5. Obstacle clearance penalty.
6. Action smoothness penalty.
7. Improved dynamics training with more data.
8. Validation-based early stopping.
9. Bootstrapped ensemble training.
10. Single-model, Ensemble, and Oracle MPC comparisons.

CONTROLS
--------
1 : PD
2 : Single-model MPC
3 : Ensemble MPC
4 : Oracle MPC
SPACE : Pause/resume
R : Reset
+/- : Simulation speed
Left click : Move target center
ESC : Exit

INSTALL
-------
pip install numpy torch matplotlib pygame

RUN
---
python phase4/ensemble_mpc_drone.py

NOTE
----
Educational 2D simulator only. Not flight-certified software.
Ensemble disagreement is not a calibrated failure probability.
"""

from __future__ import annotations

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

SEED = 419

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

MAX_SPEED = 150.0
MAX_ACCEL = 115.0
DRONE_RADIUS = 13.0

OBSTACLE = np.array([450.0, 330.0], dtype=np.float32)

OBSTACLE_RADIUS = 65.0
SAFE_RADIUS = 105.0

TARGET_A = 170.0
TARGET_B = 105.0
TARGET_OMEGA = 0.55

ENSEMBLE_SIZE = 5
HIDDEN = 128

EPOCHS = 120
BATCH_SIZE = 256
LEARNING_RATE = 7e-4
EARLY_STOPPING_PATIENCE = 18

DATA_EPISODES = 120
DATA_STEPS = 180

# MPC
MPC_HORIZON = 16
CEM_CANDIDATES = 72
CEM_ELITES = 12
CEM_ITERATIONS = 3

CEM_INITIAL_STD = 0.55
CEM_MIN_STD = 0.08
CEM_SMOOTHING = 0.65

# Objective weights
POSITION_WEIGHT = 0.035
VELOCITY_WEIGHT = 0.008
TERMINAL_POSITION_WEIGHT = 0.16
TERMINAL_VELOCITY_WEIGHT = 0.035

OBSTACLE_WEIGHT = 0.055
CONTROL_WEIGHT = 0.010
SMOOTHNESS_WEIGHT = 0.035

UNCERTAINTY_WEIGHT = 2.0
UNCERTAINTY_SCALE = 22.0

BENCHMARK_EPISODES = 3
BENCHMARK_STEPS = 220


# ============================================================
# 1. MOVING TARGET
# ============================================================

@dataclass
class MovingTarget:

    center: np.ndarray
    phase: float = 0.0

    def position(self, phase=None):

        p = self.phase if phase is None else phase

        return self.center + np.array(
            [
                TARGET_A * math.cos(p),
                TARGET_B * math.sin(p),
            ],
            dtype=np.float32,
        )

    def velocity(self, phase=None):

        p = self.phase if phase is None else phase

        return np.array(
            [
                -TARGET_A * TARGET_OMEGA * math.sin(p),
                TARGET_B * TARGET_OMEGA * math.cos(p),
            ],
            dtype=np.float32,
        )

    def advance(self):

        self.phase += TARGET_OMEGA * DT


# ============================================================
# 2. DRONE ENVIRONMENT
# ============================================================

class DroneEnv:

    def __init__(self):

        self.target = MovingTarget(
            np.array([700.0, 320.0], dtype=np.float32)
        )

        self.reset()

    def reset(self):

        self.position = np.array(
            [115.0, 500.0],
            dtype=np.float32,
        )

        self.velocity = np.zeros(2, dtype=np.float32)

        self.target.phase = float(
            np.random.uniform(0.0, 2.0 * math.pi)
        )

        return self.drone_state()

    def drone_state(self):

        return np.concatenate(
            [self.position, self.velocity]
        ).astype(np.float32)

    def observation(self):

        return np.concatenate(
            [
                self.position,
                self.velocity,
                self.target.position(),
                self.target.velocity(),
            ]
        ).astype(np.float32)

    def step_drone(self, action):

        action = np.clip(
            np.asarray(action, dtype=np.float32),
            -1.0,
            1.0,
        )

        old_state = self.drone_state()

        self.velocity += action * MAX_ACCEL * DT

        speed = np.linalg.norm(self.velocity)

        if speed > MAX_SPEED:
            self.velocity *= MAX_SPEED / (speed + 1e-8)

        self.position += self.velocity * DT

        self.position[0] = np.clip(
            self.position[0],
            DRONE_RADIUS,
            WORLD_W - DRONE_RADIUS,
        )

        self.position[1] = np.clip(
            self.position[1],
            DRONE_RADIUS,
            WORLD_H - DRONE_RADIUS,
        )

        return self.drone_state() - old_state

    def step(self, action):

        self.step_drone(action)
        self.target.advance()

        return self.observation()


# ============================================================
# 3. CONTROLLERS
# ============================================================

def pd_controller(state, target_position, target_velocity):

    position = state[:2]
    velocity = state[2:4]

    error = target_position - position

    desired_velocity = target_velocity + 0.65 * error

    desired_velocity = np.clip(
        desired_velocity,
        -MAX_SPEED,
        MAX_SPEED,
    )

    acceleration = 2.0 * (desired_velocity - velocity)

    return np.clip(
        acceleration / MAX_ACCEL,
        -1.0,
        1.0,
    ).astype(np.float32)


def safety_filter(action, position):

    delta = position - OBSTACLE
    distance = np.linalg.norm(delta) + 1e-8

    if distance < SAFE_RADIUS:

        direction = delta / distance

        correction = (
            1.4
            * (SAFE_RADIUS - distance)
            / SAFE_RADIUS
        )

        action = action + direction * correction

    return np.clip(
        action,
        -1.0,
        1.0,
    ).astype(np.float32)


# ============================================================
# 4. DATA COLLECTION
# ============================================================

def collect_data():

    inputs = []
    targets = []

    env = DroneEnv()

    print("Collecting dynamics data...")

    for episode in range(DATA_EPISODES):

        env.reset()

        for step in range(DATA_STEPS):

            state = env.drone_state()

            pd_action = pd_controller(
                state,
                env.target.position(),
                env.target.velocity(),
            )

            # More varied action distribution.
            if np.random.random() < 0.30:

                action = np.random.uniform(
                    -1.0,
                    1.0,
                    size=2,
                )

            else:

                action = pd_action + np.random.normal(
                    0.0,
                    0.28,
                    size=2,
                )

            action = np.clip(
                action,
                -1.0,
                1.0,
            ).astype(np.float32)

            # Include obstacle-related states in training.
            if np.linalg.norm(state[:2] - OBSTACLE) < 150:

                action = safety_filter(
                    action,
                    state[:2],
                )

            model_input = np.concatenate(
                [state, action]
            ).astype(np.float32)

            delta_state = env.step_drone(action)

            inputs.append(model_input)
            targets.append(delta_state.astype(np.float32))

            env.target.advance()

        if (episode + 1) % 20 == 0:

            print(
                f"  Data episode {episode + 1}/{DATA_EPISODES}"
            )

    return (
        np.asarray(inputs, dtype=np.float32),
        np.asarray(targets, dtype=np.float32),
    )


# ============================================================
# 5. NEURAL DYNAMICS MODEL
# ============================================================

class DynamicsModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(6, HIDDEN),
            nn.ReLU(),

            nn.Linear(HIDDEN, HIDDEN),
            nn.ReLU(),

            nn.Linear(HIDDEN, HIDDEN),
            nn.ReLU(),

            nn.Linear(HIDDEN, 4),
        )

    def forward(self, x):

        return self.net(x)


@dataclass
class Normalizer:

    x_mean: torch.Tensor
    x_std: torch.Tensor
    y_mean: torch.Tensor
    y_std: torch.Tensor


def make_normalizer(x, y):

    return Normalizer(
        x.mean(dim=0),
        x.std(dim=0).clamp_min(1e-5),
        y.mean(dim=0),
        y.std(dim=0).clamp_min(1e-5),
    )


# ============================================================
# 6. ENSEMBLE TRAINING
# ============================================================

class DynamicsEnsemble:

    def __init__(self, size=ENSEMBLE_SIZE):

        self.models = [
            DynamicsModel().to(DEVICE)
            for _ in range(size)
        ]

        self.normalizer = None

        self.train_losses = []
        self.validation_losses = []

        self.validation_x = None
        self.validation_y = None

    def train(self, x_np, y_np):

        if len(x_np) != len(y_np):
            raise ValueError("Input/target length mismatch.")

        x = torch.tensor(
            x_np,
            dtype=torch.float32,
            device=DEVICE,
        )

        y = torch.tensor(
            y_np,
            dtype=torch.float32,
            device=DEVICE,
        )

        indices = torch.randperm(
            len(x),
            device=DEVICE,
        )

        split = int(0.85 * len(x))

        train_idx = indices[:split]
        val_idx = indices[split:]

        x_train = x[train_idx]
        y_train = y[train_idx]

        x_val = x[val_idx]
        y_val = y[val_idx]

        self.validation_x = x_val.cpu().numpy()
        self.validation_y = y_val.cpu().numpy()

        self.normalizer = make_normalizer(
            x_train,
            y_train,
        )

        n = self.normalizer

        x_train_n = (x_train - n.x_mean) / n.x_std
        y_train_n = (y_train - n.y_mean) / n.y_std

        x_val_n = (x_val - n.x_mean) / n.x_std
        y_val_n = (y_val - n.y_mean) / n.y_std

        loss_fn = nn.MSELoss()

        self.train_losses = []
        self.validation_losses = []

        print(
            f"Training {len(self.models)} dynamics models..."
        )

        for model_index, model in enumerate(self.models):

            optimizer = optim.AdamW(
                model.parameters(),
                lr=LEARNING_RATE,
                weight_decay=1e-5,
            )

            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.5,
                patience=7,
            )

            best_loss = float("inf")
            best_weights = None
            patience = 0

            model_train_history = []
            model_val_history = []

            for epoch in range(EPOCHS):

                # Bootstrap sample for ensemble diversity.
                bootstrap_idx = torch.randint(
                    0,
                    len(x_train),
                    (len(x_train),),
                    device=DEVICE,
                )

                model.train()

                batch_losses = []

                for start in range(
                    0,
                    len(bootstrap_idx),
                    BATCH_SIZE,
                ):

                    idx = bootstrap_idx[
                        start:start + BATCH_SIZE
                    ]

                    prediction = model(x_train_n[idx])

                    loss = loss_fn(
                        prediction,
                        y_train_n[idx],
                    )

                    optimizer.zero_grad()
                    loss.backward()

                    nn.utils.clip_grad_norm_(
                        model.parameters(),
                        5.0,
                    )

                    optimizer.step()

                    batch_losses.append(loss.item())

                model.eval()

                with torch.no_grad():

                    val_prediction = model(x_val_n)

                    val_loss = loss_fn(
                        val_prediction,
                        y_val_n,
                    ).item()

                train_loss = float(np.mean(batch_losses))

                model_train_history.append(train_loss)
                model_val_history.append(val_loss)

                scheduler.step(val_loss)

                if val_loss < best_loss:

                    best_loss = val_loss

                    best_weights = {
                        key: value.detach().clone()
                        for key, value in model.state_dict().items()
                    }

                    patience = 0

                else:

                    patience += 1

                if patience >= EARLY_STOPPING_PATIENCE:
                    break

            if best_weights is not None:
                model.load_state_dict(best_weights)

            self.train_losses.append(
                float(np.mean(model_train_history))
            )

            self.validation_losses.append(best_loss)

            print(
                f"  Model {model_index + 1}/{len(self.models)} | "
                f"epochs={len(model_val_history)} | "
                f"best validation={best_loss:.6f}"
            )

    def predict(self, model_input):

        if self.normalizer is None:
            raise RuntimeError("Ensemble has not been trained.")

        x = torch.as_tensor(
            model_input,
            dtype=torch.float32,
            device=DEVICE,
        )

        x_norm = (
            x - self.normalizer.x_mean
        ) / self.normalizer.x_std

        predictions = []

        for model in self.models:

            model.eval()

            with torch.no_grad():

                y_norm = model(x_norm)

                y = (
                    y_norm * self.normalizer.y_std
                    + self.normalizer.y_mean
                )

            predictions.append(y.cpu().numpy())

        return np.stack(predictions, axis=0)


# ============================================================
# 7. UNCERTAINTY AND MODEL EVALUATION
# ============================================================

def ensemble_uncertainty(predictions):

    mean_prediction = np.mean(
        predictions,
        axis=0,
    )

    position_std = np.std(
        predictions[..., :2],
        axis=0,
    )

    uncertainty = np.sqrt(
        np.sum(position_std ** 2, axis=-1)
    )

    return mean_prediction, uncertainty


def evaluate_model_error(ensemble, x_np, y_np):

    predictions = ensemble.predict(x_np)

    mean_prediction, uncertainty = ensemble_uncertainty(
        predictions
    )

    errors = mean_prediction - y_np

    rmse = float(
        np.sqrt(np.mean(errors ** 2))
    )

    component_rmse = np.sqrt(
        np.mean(errors ** 2, axis=0)
    )

    position_error = np.linalg.norm(
        errors[:, :2],
        axis=1,
    )

    return {
        "rmse": rmse,
        "component_rmse": component_rmse,
        "uncertainty": uncertainty,
        "position_error": position_error,
    }


# ============================================================
# 8. TARGET PREDICTION
# ============================================================

def target_at(target, future_step):

    future_phase = (
        target.phase
        + TARGET_OMEGA * DT * future_step
    )

    return target.position(future_phase)


def target_velocity_at(target, future_step):

    future_phase = (
        target.phase
        + TARGET_OMEGA * DT * future_step
    )

    return target.velocity(future_phase)


# ============================================================
# 9. ORACLE TRANSITION
# ============================================================

def oracle_transition(state, action):

    velocity = (
        state[2:4] + action * MAX_ACCEL * DT
    )

    speed = np.linalg.norm(velocity)

    if speed > MAX_SPEED:
        velocity *= MAX_SPEED / (speed + 1e-8)

    position = state[:2] + velocity * DT

    position[0] = np.clip(
        position[0],
        DRONE_RADIUS,
        WORLD_W - DRONE_RADIUS,
    )

    position[1] = np.clip(
        position[1],
        DRONE_RADIUS,
        WORLD_H - DRONE_RADIUS,
    )

    return np.concatenate(
        [position, velocity]
    ).astype(np.float32)


# ============================================================
# 10. NOMINAL ACTION SEQUENCE
# ============================================================

def make_nominal_sequence(state, target):

    nominal_state = state.copy()
    sequence = []

    for t in range(1, MPC_HORIZON + 1):

        action = pd_controller(
            nominal_state,
            target_at(target, t),
            target_velocity_at(target, t),
        )

        sequence.append(action)

        nominal_state = oracle_transition(
            nominal_state,
            action,
        )

    return np.asarray(sequence, dtype=np.float32)


# ============================================================
# 11. MPC OBJECTIVE
# ============================================================

def obstacle_penalty(position):

    distance = np.linalg.norm(
        position - OBSTACLE,
        axis=-1,
    )

    violation = np.maximum(
        0.0,
        SAFE_RADIUS - distance,
    )

    return violation ** 2


def evaluate_candidates(
    initial_state,
    candidates,
    target,
    ensemble,
    mode,
):
    """
    candidates: (candidate_count, horizon, 2)

    Returns:
        candidate costs
        best predicted path
        best predicted uncertainty
    """

    initial_state = np.asarray(
        initial_state,
        dtype=np.float32,
    )

    candidates = np.asarray(
        candidates,
        dtype=np.float32,
    )

    candidate_count, horizon, action_dim = candidates.shape

    if initial_state.shape != (4,):
        raise ValueError("initial_state must have shape (4,).")

    if action_dim != 2:
        raise ValueError("Action dimension must be 2.")

    if mode == "Oracle MPC":

        member_count = 1
        models_to_use = None

    elif mode == "Single-model MPC":

        if ensemble is None or ensemble.normalizer is None:
            raise RuntimeError("Trained ensemble required.")

        member_count = 1
        models_to_use = ensemble.models[:1]

    elif mode == "Ensemble MPC":

        if ensemble is None or ensemble.normalizer is None:
            raise RuntimeError("Trained ensemble required.")

        member_count = len(ensemble.models)
        models_to_use = ensemble.models

    else:

        raise ValueError(f"Unknown mode: {mode}")

    states = np.repeat(
        initial_state[None, None, :],
        member_count,
        axis=0,
    )

    states = np.repeat(
        states,
        candidate_count,
        axis=1,
    ).astype(np.float32)

    costs = np.zeros(
        candidate_count,
        dtype=np.float64,
    )

    mean_paths = []
    uncertainty_paths = []

    normalizer = (
        ensemble.normalizer
        if ensemble is not None
        else None
    )

    for t in range(horizon):

        actions = candidates[:, t, :]

        if mode == "Oracle MPC":

            next_states = []

            for candidate_index in range(candidate_count):

                next_states.append(
                    oracle_transition(
                        states[0, candidate_index],
                        actions[candidate_index],
                    )
                )

            states = np.asarray(
                next_states,
                dtype=np.float32,
            )[None, :, :]

        else:

            model_inputs = np.concatenate(
                [
                    states,

                    np.broadcast_to(
                        actions[None, :, :],
                        (member_count, candidate_count, 2),
                    ),
                ],
                axis=2,
            )

            member_deltas = []

            for member_index, model in enumerate(models_to_use):

                x = torch.as_tensor(
                    model_inputs[member_index],
                    dtype=torch.float32,
                    device=DEVICE,
                )

                x_norm = (
                    x - normalizer.x_mean
                ) / normalizer.x_std

                model.eval()

                with torch.no_grad():

                    y_norm = model(x_norm)

                    delta = (
                        y_norm * normalizer.y_std
                        + normalizer.y_mean
                    )

                member_deltas.append(
                    delta.cpu().numpy()
                )

            member_deltas = np.stack(
                member_deltas,
                axis=0,
            )

            states = (
                states + member_deltas
            ).astype(np.float32)

        mean_state = np.mean(
            states,
            axis=0,
        )

        mean_position = mean_state[:, :2]
        mean_velocity = mean_state[:, 2:4]

        if member_count > 1:

            position_std = np.std(
                states[:, :, :2],
                axis=0,
            )

            uncertainty = np.sqrt(
                np.sum(position_std ** 2, axis=1)
            )

        else:

            uncertainty = np.zeros(
                candidate_count,
                dtype=np.float32,
            )

        desired_position = target_at(
            target,
            t + 1,
        )

        desired_velocity = target_velocity_at(
            target,
            t + 1,
        )

        position_error = np.linalg.norm(
            mean_position - desired_position[None, :],
            axis=1,
        )

        velocity_error = np.linalg.norm(
            mean_velocity - desired_velocity[None, :],
            axis=1,
        )

        costs += POSITION_WEIGHT * position_error
        costs += VELOCITY_WEIGHT * velocity_error

        costs += (
            OBSTACLE_WEIGHT
            * obstacle_penalty(mean_position)
        )

        costs += (
            CONTROL_WEIGHT
            * np.sum(actions ** 2, axis=1)
        )

        if t > 0:

            previous_actions = candidates[:, t - 1, :]

            costs += (
                SMOOTHNESS_WEIGHT
                * np.sum(
                    (actions - previous_actions) ** 2,
                    axis=1,
                )
            )

        if mode == "Ensemble MPC":

            costs += (
                UNCERTAINTY_WEIGHT
                * (uncertainty / UNCERTAINTY_SCALE) ** 2
            )

        mean_paths.append(
            mean_position.copy()
        )

        uncertainty_paths.append(
            uncertainty.copy()
        )

    # Terminal objective.
    final_target = target_at(
        target,
        horizon,
    )

    final_target_velocity = target_velocity_at(
        target,
        horizon,
    )

    terminal_position_error = np.linalg.norm(
        mean_state[:, :2] - final_target[None, :],
        axis=1,
    )

    terminal_velocity_error = np.linalg.norm(
        mean_state[:, 2:4] - final_target_velocity[None, :],
        axis=1,
    )

    costs += (
        TERMINAL_POSITION_WEIGHT
        * terminal_position_error ** 2
    )

    costs += (
        TERMINAL_VELOCITY_WEIGHT
        * terminal_velocity_error ** 2
    )

    costs += (
        OBSTACLE_WEIGHT
        * obstacle_penalty(mean_state[:, :2])
    )

    mean_paths = np.stack(
        mean_paths,
        axis=1,
    )

    uncertainty_paths = np.stack(
        uncertainty_paths,
        axis=1,
    )

    best_index = int(np.argmin(costs))

    return (
        costs,
        mean_paths[best_index],
        uncertainty_paths[best_index],
    )


# ============================================================
# 12. CEM-MPC PLANNER
# ============================================================

class CEMPlanner:

    def __init__(self):

        self.previous_sequence = None

    def reset(self):

        self.previous_sequence = None

    def plan(self, state, target, ensemble, mode):

        nominal = make_nominal_sequence(
            state,
            target,
        )

        if self.previous_sequence is None:

            mean = nominal.copy()

        else:

            # Shift the previous sequence one step forward.
            shifted = np.vstack(
                [
                    self.previous_sequence[1:],
                    self.previous_sequence[-1:],
                ]
            )

            # Blend shifted solution with fresh PD reference.
            mean = (
                0.75 * shifted
                + 0.25 * nominal
            )

        mean = np.clip(
            mean,
            -1.0,
            1.0,
        )

        std = np.full(
            (MPC_HORIZON, 2),
            CEM_INITIAL_STD,
            dtype=np.float32,
        )

        best_cost = float("inf")
        best_sequence = mean.copy()
        best_path = np.zeros(
            (MPC_HORIZON, 2),
            dtype=np.float32,
        )

        best_uncertainty = np.zeros(
            MPC_HORIZON,
            dtype=np.float32,
        )

        for iteration in range(CEM_ITERATIONS):

            candidates = np.random.normal(
                loc=mean,
                scale=std,
                size=(
                    CEM_CANDIDATES,
                    MPC_HORIZON,
                    2,
                ),
            ).astype(np.float32)

            candidates = np.clip(
                candidates,
                -1.0,
                1.0,
            )

            # Keep the current mean and PD sequence in the population.
            candidates[0] = mean
            candidates[1] = nominal

            costs, paths, uncertainties = evaluate_candidates(
                state,
                candidates,
                target,
                ensemble,
                mode,
            )

            elite_indices = np.argsort(
                costs
            )[:CEM_ELITES]

            elites = candidates[
                elite_indices
            ]

            elite_mean = np.mean(
                elites,
                axis=0,
            )

            elite_std = np.std(
                elites,
                axis=0,
            )

            mean = (
                CEM_SMOOTHING * mean
                + (1.0 - CEM_SMOOTHING) * elite_mean
            )

            std = (
                CEM_SMOOTHING * std
                + (1.0 - CEM_SMOOTHING) * elite_std
            )

            std = np.maximum(
                std,
                CEM_MIN_STD,
            )

            current_best = int(
                elite_indices[0]
            )

            if costs[current_best] < best_cost:

                best_cost = float(
                    costs[current_best]
                )

                best_sequence = candidates[
                    current_best
                ].copy()

                best_path = paths.copy()
                best_uncertainty = uncertainties.copy()

        self.previous_sequence = best_sequence.copy()

        return (
            best_sequence[0].copy(),
            best_sequence.copy(),
            best_path,
            best_uncertainty,
            best_cost,
        )


# ============================================================
# 13. BENCHMARK
# ============================================================

def run_benchmark(
    ensemble,
    mode,
    episodes=BENCHMARK_EPISODES,
    steps=BENCHMARK_STEPS,
):

    env = DroneEnv()

    episode_distances = []
    episode_clearances = []
    episode_efforts = []

    for episode in range(episodes):

        env.reset()

        planner = CEMPlanner()

        distances = []
        clearances = []
        efforts = []

        for _ in range(steps):

            state = env.drone_state()

            if mode == "PD":

                action = pd_controller(
                    state,
                    env.target.position(),
                    env.target.velocity(),
                )

            else:

                action, _, _, _, _ = planner.plan(
                    state,
                    env.target,
                    ensemble,
                    mode,
                )

            action = safety_filter(
                action,
                state[:2],
            )

            env.step(action)

            distances.append(
                np.linalg.norm(
                    env.position - env.target.position()
                )
            )

            clearances.append(
                np.linalg.norm(
                    env.position - OBSTACLE
                ) - OBSTACLE_RADIUS
            )

            efforts.append(
                np.linalg.norm(action)
            )

        episode_distances.append(
            np.mean(distances)
        )

        episode_clearances.append(
            np.min(clearances)
        )

        episode_efforts.append(
            np.mean(efforts)
        )

        print(
            f"  Episode {episode + 1}/{episodes}: "
            f"distance={episode_distances[-1]:.2f} | "
            f"clearance={episode_clearances[-1]:.2f}"
        )

    return {
        "mean_distance": float(np.mean(episode_distances)),
        "distance_std": float(np.std(episode_distances)),
        "mean_min_clearance": float(np.mean(episode_clearances)),
        "mean_action_effort": float(np.mean(episode_efforts)),
    }


# ============================================================
# 14. PYGAME
# ============================================================

def run_pygame(ensemble):

    try:
        import pygame

    except ImportError:

        print("Install pygame: pip install pygame")
        return

    pygame.init()

    screen = pygame.display.set_mode(
        (WORLD_W, WORLD_H)
    )

    pygame.display.set_caption(
        "Phase 4.19 — Improved CEM Ensemble MPC"
    )

    clock = pygame.time.Clock()

    font = pygame.font.SysFont(
        "consolas",
        17,
    )

    env = DroneEnv()
    env.reset()

    planner = CEMPlanner()

    modes = [
        "PD",
        "Single-model MPC",
        "Ensemble MPC",
        "Oracle MPC",
    ]

    mode_index = 2

    paused = False
    speed = 1

    trail = []
    planned_path = []

    planned_uncertainty = np.zeros(0)

    last_action = np.zeros(
        2,
        dtype=np.float32,
    )

    def draw_text(text, x, y, color=(235, 235, 235)):

        screen.blit(
            font.render(
                text,
                True,
                color,
            ),
            (x, y),
        )

    def draw_target():

        points = []

        for i in range(100):

            phase = (
                env.target.phase
                + i * 2.0 * math.pi / 100
            )

            points.append(
                env.target.position(phase).astype(int)
            )

        pygame.draw.lines(
            screen,
            (70, 105, 75),
            True,
            points,
            2,
        )

        position = env.target.position()

        pygame.draw.circle(
            screen,
            (70, 235, 115),
            position.astype(int),
            12,
        )

        pygame.draw.circle(
            screen,
            (70, 235, 115),
            position.astype(int),
            23,
            1,
        )

    def draw_drone(position, velocity):

        center = np.asarray(position)

        direction = velocity / (
            np.linalg.norm(velocity) + 1e-6
        )

        if np.linalg.norm(velocity) < 1.0:

            direction = np.array(
                [1.0, 0.0]
            )

        side = np.array(
            [-direction[1], direction[0]]
        )

        for sign in (-1, 1):

            for lateral in (-1, 1):

                endpoint = (
                    center
                    + direction * sign * 17
                    + side * lateral * 17
                )

                pygame.draw.line(
                    screen,
                    (210, 220, 235),
                    center.astype(int),
                    endpoint.astype(int),
                    5,
                )

                pygame.draw.circle(
                    screen,
                    (90, 195, 245),
                    endpoint.astype(int),
                    7,
                    2,
                )

        pygame.draw.circle(
            screen,
            (245, 245, 245),
            center.astype(int),
            8,
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

                    env.reset()
                    planner.reset()

                    trail.clear()
                    planned_path = []
                    planned_uncertainty = np.zeros(0)

                elif event.key in (
                    pygame.K_EQUALS,
                    pygame.K_PLUS,
                ):

                    speed = min(5, speed + 1)

                elif event.key == pygame.K_MINUS:

                    speed = max(1, speed - 1)

                elif event.key == pygame.K_1:

                    mode_index = 0
                    planner.reset()

                elif event.key == pygame.K_2:

                    mode_index = 1
                    planner.reset()

                elif event.key == pygame.K_3:

                    mode_index = 2
                    planner.reset()

                elif event.key == pygame.K_4:

                    mode_index = 3
                    planner.reset()

            elif event.type == pygame.MOUSEBUTTONDOWN:

                if event.button == 1:

                    click = np.array(
                        event.pos,
                        dtype=np.float32,
                    )

                    if np.linalg.norm(
                        click - OBSTACLE
                    ) > SAFE_RADIUS + 15:

                        env.target.center = click
                        planner.reset()

        mode = modes[mode_index]

        if not paused:

            for _ in range(speed):

                state = env.drone_state()

                if mode == "PD":

                    action = pd_controller(
                        state,
                        env.target.position(),
                        env.target.velocity(),
                    )

                    planned_path = []
                    planned_uncertainty = np.zeros(0)

                else:

                    (
                        action,
                        _sequence,
                        planned_path,
                        planned_uncertainty,
                        _cost,
                    ) = planner.plan(
                        state,
                        env.target,
                        ensemble,
                        mode,
                    )

                action = safety_filter(
                    action,
                    state[:2],
                )

                last_action = action.copy()

                env.step(action)

                trail.append(
                    env.position.copy()
                )

                if len(trail) > 160:
                    trail.pop(0)

        # Background
        screen.fill((18, 24, 34))

        for x in range(0, WORLD_W, 50):

            pygame.draw.line(
                screen,
                (30, 39, 50),
                (x, 0),
                (x, WORLD_H),
            )

        for y in range(0, WORLD_H, 50):

            pygame.draw.line(
                screen,
                (30, 39, 50),
                (0, y),
                (WORLD_W, y),
            )

        # Obstacle
        pygame.draw.circle(
            screen,
            (95, 95, 105),
            OBSTACLE.astype(int),
            OBSTACLE_RADIUS,
        )

        pygame.draw.circle(
            screen,
            (210, 100, 85),
            OBSTACLE.astype(int),
            SAFE_RADIUS,
            2,
        )

        draw_target()

        # Trail
        if len(trail) > 1:

            pygame.draw.lines(
                screen,
                (85, 140, 235),
                False,
                [p.astype(int) for p in trail],
                2,
            )

        # Predicted path
        if len(planned_path) > 1:

            pygame.draw.lines(
                screen,
                (255, 190, 65),
                False,
                [p.astype(int) for p in planned_path],
                3,
            )

            for point, uncertainty in zip(
                planned_path,
                planned_uncertainty,
            ):

                radius = int(
                    np.clip(
                        uncertainty * 1.5,
                        2,
                        35,
                    )
                )

                pygame.draw.circle(
                    screen,
                    (230, 175, 90),
                    point.astype(int),
                    radius,
                    1,
                )

        draw_drone(
            env.position,
            env.velocity,
        )

        target_distance = np.linalg.norm(
            env.position - env.target.position()
        )

        obstacle_clearance = (
            np.linalg.norm(
                env.position - OBSTACLE
            ) - OBSTACLE_RADIUS
        )

        draw_text(
            f"Mode: {mode}",
            15,
            12,
        )

        draw_text(
            f"Target distance: {target_distance:.1f}px",
            15,
            39,
        )

        draw_text(
            f"Obstacle clearance: {obstacle_clearance:.1f}px",
            15,
            66,
        )

        draw_text(
            f"Action: [{last_action[0]:+.2f}, "
            f"{last_action[1]:+.2f}]",
            15,
            93,
        )

        draw_text(
            f"Simulation speed: {speed}x",
            15,
            120,
        )

        draw_text(
            "Gold circles: ensemble prediction disagreement",
            15,
            147,
            (245, 195, 110),
        )

        draw_text(
            "1 PD | 2 Single MPC | 3 Ensemble MPC | 4 Oracle MPC",
            15,
            WORLD_H - 75,
        )

        draw_text(
            "SPACE pause | R reset | +/- speed | "
            "click target center | ESC exit",
            15,
            WORLD_H - 45,
        )

        if paused:

            draw_text(
                "PAUSED",
                WORLD_W - 115,
                15,
                (255, 210, 80),
            )

        pygame.display.flip()

    pygame.quit()


# ============================================================
# 15. VISUALIZATION
# ============================================================

def visualize_training(ensemble):

    plt.figure(figsize=(8, 4))

    plt.plot(
        ensemble.validation_losses,
        marker="o",
        label="Best validation loss per model",
    )

    plt.xlabel("Ensemble member")
    plt.ylabel("Normalized MSE")
    plt.title("Dynamics Ensemble Validation")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def visualize_uncertainty(evaluation):

    errors = evaluation["position_error"]
    uncertainty = evaluation["uncertainty"]

    correlation = np.corrcoef(
        uncertainty,
        errors,
    )[0, 1]

    plt.figure(figsize=(7, 5))

    plt.scatter(
        uncertainty,
        errors,
        alpha=0.35,
    )

    plt.xlabel("Ensemble disagreement (pixels)")
    plt.ylabel("Actual position prediction error (pixels)")

    plt.title(
        f"Uncertainty vs Prediction Error "
        f"(correlation={correlation:.3f})"
    )

    plt.grid(True)
    plt.tight_layout()
    plt.show()


def visualize_benchmark(results):

    names = list(results.keys())

    metrics = [
        ("mean_distance", "Mean target distance (pixels)"),
        (
            "mean_min_clearance",
            "Mean minimum obstacle clearance (pixels)",
        ),
        ("mean_action_effort", "Mean action effort"),
    ]

    for key, title in metrics:

        values = [
            results[name][key]
            for name in names
        ]

        plt.figure(figsize=(8, 4))

        plt.bar(names, values)

        plt.title(title)
        plt.ylabel(title)

        plt.xticks(rotation=12)
        plt.grid(axis="y")
        plt.tight_layout()
        plt.show()


# ============================================================
# 16. UNIT TESTS
# ============================================================


def run_tests(ensemble):

    print("\nRunning unit tests...")

    passed = 0
    total = 21

    def check(condition, description):

        nonlocal passed

        assert condition, description

        passed += 1

        print(f"[PASS] {description}")

    target = MovingTarget(
        np.array([400.0, 300.0], dtype=np.float32)
    )

    check(
        target.position().shape == (2,),
        "Target position shape",
    )

    check(
        target.velocity().shape == (2,),
        "Target velocity shape",
    )

    before = target.position().copy()
    target.advance()

    check(
        not np.allclose(before, target.position()),
        "Target phase advances",
    )

    env = DroneEnv()

    check(
        env.drone_state().shape == (4,),
        "Drone state shape",
    )

    delta = env.step_drone(
        np.zeros(2, dtype=np.float32)
    )

    check(
        delta.shape == (4,),
        "Drone transition shape",
    )

    env.step_drone(
        np.array([5.0, -5.0], dtype=np.float32)
    )

    check(
        np.linalg.norm(env.velocity) <= MAX_SPEED + 1e-4,
        "Maximum drone speed",
    )

    check(
        np.all(np.isfinite(env.drone_state())),
        "Finite drone state",
    )

    action = pd_controller(
        env.drone_state(),
        env.target.position(),
        env.target.velocity(),
    )

    check(
        action.shape == (2,),
        "PD action shape",
    )

    check(
        np.all(np.abs(action) <= 1.0),
        "PD action bounds",
    )

    corrected = safety_filter(
        np.zeros(2, dtype=np.float32),
        OBSTACLE + np.array(
            [20.0, 0.0],
            dtype=np.float32,
        ),
    )

    check(
        np.all(np.abs(corrected) <= 1.0),
        "Safety filter action bounds",
    )

    nominal = make_nominal_sequence(
        env.drone_state(),
        env.target,
    )

    check(
        nominal.shape == (MPC_HORIZON, 2),
        "Nominal sequence shape",
    )

    check(
        np.all(np.abs(nominal) <= 1.0),
        "Nominal sequence bounds",
    )

    sample = np.concatenate(
        [
            env.drone_state(),
            np.zeros(2, dtype=np.float32),
        ]
    )

    predictions = ensemble.predict(sample)

    check(
        predictions.shape == (len(ensemble.models), 4),
        "Ensemble prediction shape",
    )

    mean_prediction, uncertainty = ensemble_uncertainty(
        predictions
    )

    check(
        mean_prediction.shape == (4,)
        and np.isfinite(uncertainty)
        and uncertainty >= 0.0,
        "Ensemble uncertainty is valid",
    )

    candidates = np.repeat(
        nominal[None, :, :],
        4,
        axis=0,
    )

    for mode in (
        "Single-model MPC",
        "Ensemble MPC",
        "Oracle MPC",
    ):

        costs, path, uncertainty_path = evaluate_candidates(
            env.drone_state(),
            candidates,
            env.target,
            ensemble,
            mode,
        )

        check(
            costs.shape == (4,)
            and path.shape == (MPC_HORIZON, 2)
            and uncertainty_path.shape == (MPC_HORIZON,)
            and np.all(np.isfinite(costs)),
            f"{mode} candidate evaluation",
        )

    planner = CEMPlanner()

    action, sequence, path, uncertainty_path, cost = planner.plan(
        env.drone_state(),
        env.target,
        ensemble,
        "Single-model MPC",
    )

    check(
        action.shape == (2,)
        and sequence.shape == (MPC_HORIZON, 2)
        and path.shape == (MPC_HORIZON, 2)
        and uncertainty_path.shape == (MPC_HORIZON,)
        and np.isfinite(cost),
        "CEM planner output shapes and values",
    )

    check(
        np.all(np.abs(action) <= 1.0),
        "CEM action bounds",
    )

    check(
        planner.previous_sequence is not None,
        "CEM warm-start sequence is stored",
    )

    check(
        ensemble.validation_x is not None
        and ensemble.validation_y is not None,
        "Validation dataset retained",
    )

    print(f"\n{passed}/{total} tests passed.")

    return passed == total



# ============================================================
# 17. MAIN
# ============================================================

def main():

    print("=" * 72)

    print(
        "PHASE 4.19 — IMPROVED ENSEMBLE DYNAMICS + CEM-MPC"
    )

    print("=" * 72)

    print("Device:", DEVICE)

    # 1. Data
    print("\n[1/7] Collecting dynamics data...")

    x_data, y_data = collect_data()

    print("Transitions:", len(x_data))
    print("Input shape:", x_data.shape)
    print("Target shape:", y_data.shape)

    # 2. Training
    print("\n[2/7] Training dynamics ensemble...")

    ensemble = DynamicsEnsemble(
        ENSEMBLE_SIZE
    )

    ensemble.train(
        x_data,
        y_data,
    )

    # 3. Model evaluation
    print("\n[3/7] Evaluating dynamics model...")

    evaluation = evaluate_model_error(
        ensemble,
        ensemble.validation_x,
        ensemble.validation_y,
    )

    print(
        f"Validation RMSE: {evaluation['rmse']:.6f}"
    )

    print(
        "Component RMSE [dx, dy, dvx, dvy]:",
        np.array2string(
            evaluation["component_rmse"],
            precision=6,
        ),
    )

    print(
        "Mean ensemble disagreement:",
        f"{np.mean(evaluation['uncertainty']):.4f} pixels",
    )

    # 4. Tests
    print("\n[4/7] Running unit tests...")

    if not run_tests(ensemble):

        raise RuntimeError(
            "Unit tests failed."
        )

    # 5. Benchmark
    print("\n[5/7] Benchmarking controllers...")

    results = {}

    for mode in (
        "PD",
        "Single-model MPC",
        "Ensemble MPC",
        "Oracle MPC",
    ):

        print(f"\nBenchmark: {mode}")

        results[mode] = run_benchmark(
            ensemble,
            mode,
        )

    print("\n========== BENCHMARK RESULTS ==========")

    for mode, metrics in results.items():

        print(f"\n{mode}")

        for key, value in metrics.items():

            print(f"  {key}: {value:.4f}")

    # 6. Plots
    print("\n[6/7] Displaying evaluation plots...")

    visualize_training(ensemble)
    visualize_uncertainty(evaluation)
    visualize_benchmark(results)

    # 7. Interactive simulator
    print("\n[7/7] Starting interactive simulator...")

    run_pygame(ensemble)


if __name__ == "__main__":

    main()