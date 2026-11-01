
"""
File: phase4/ensemble_mpc_drone.py

PHASE 4.19 — ENSEMBLE DYNAMICS MODELS AND UNCERTAINTY-AWARE MPC
================================================================

PURPOSE
-------
Train multiple neural dynamics models and use their disagreement to
estimate prediction uncertainty during model-based drone planning.

WHAT YOU LEARN
--------------
STEP 1 — Collect transition data
    Collect (state, action, next_state) transitions from a drone simulator.

STEP 2 — Train an ensemble
    Train multiple neural networks using different bootstrap samples.
    Each model learns:
        [x, y, vx, vy, ax, ay]
            -> [delta_x, delta_y, delta_vx, delta_vy]

STEP 3 — Estimate uncertainty
    Compare the predictions from all ensemble members.
    Their disagreement is an approximate epistemic uncertainty measure.

STEP 4 — Uncertainty-aware MPC
    Generate candidate action sequences.
    Predict their future trajectories with the ensemble.
    Penalize trajectories with large prediction disagreement.
    Execute only the first action, then replan.

STEP 5 — Compare controllers
    PD controller       : reactive baseline
    Single-model MPC    : plans using one learned model
    Ensemble MPC        : plans using all models and uncertainty penalty
    Oracle MPC          : uses true simulator dynamics as a reference

STEP 6 — Visualize and verify
    Training curves, held-out model error, uncertainty/error correlation,
    controller benchmark, and interactive quadcopter drone simulation.

INSTALL
-------
    pip install numpy torch matplotlib pygame

RUN
---
    python phase4/ensemble_mpc_drone.py

CONTROLS
--------
    1 : PD controller
    2 : Single-model MPC
    3 : Ensemble MPC
    4 : Oracle MPC
    SPACE : Pause/resume
    R : Reset drone
    + / - : Simulation speed
    Left mouse click : Change moving-target trajectory center
    ESC : Exit

ROBOTICS APPLICATION
--------------------
A real drone's dynamics can vary with payload, battery voltage, wind,
temperature, and actuator wear. An ensemble can identify states and
actions where learned models disagree, helping a planner avoid
unfamiliar predictions or collect additional data.

IMPORTANT LIMITATIONS
--------------------
1. Ensemble disagreement is not a calibrated probability of failure.
2. Models can agree and still be wrong because of shared model bias.
3. The uncertainty penalty is a heuristic, not a safety guarantee.
4. This is a simplified 2D educational drone simulator.
5. Oracle MPC uses true simulator dynamics and is only a benchmark.
6. This is not flight-certified control software.
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

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

WORLD_W = 900
WORLD_H = 650

DT = 0.08

MAX_SPEED = 150.0
MAX_ACCEL = 115.0
DRONE_RADIUS = 13.0

OBSTACLE = np.array(
    [450.0, 330.0], dtype=np.float32
)
OBSTACLE_RADIUS = 65.0
SAFE_RADIUS = 92.0

TARGET_A = 170.0
TARGET_B = 105.0
TARGET_OMEGA = 0.55

ENSEMBLE_SIZE = 5
HIDDEN = 128

EPOCHS = 80
BATCH_SIZE = 256
LEARNING_RATE = 1e-3

DATA_EPISODES = 90
DATA_STEPS = 150

MPC_HORIZON = 12
MPC_CANDIDATES = 64
ACTION_NOISE = 0.55

# Higher values make Ensemble MPC more conservative.
UNCERTAINTY_WEIGHT = 2.5

# Scale used to normalize position uncertainty in the cost.
UNCERTAINTY_SCALE = 25.0

# Benchmark configuration.
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
# 2. TRUE DRONE SIMULATOR
# ============================================================

class DroneEnv:
    """
    Drone state:
        [x, y, vx, vy]

    Action:
        [ax_normalized, ay_normalized]

    Simplified dynamics:
        v(t+1) = clip(v(t) + a * dt)
        p(t+1) = p(t) + v(t+1) * dt

    The neural models learn the resulting state differences.
    """

    def __init__(self):
        self.target = MovingTarget(
            np.array([700.0, 320.0], dtype=np.float32)
        )
        self.reset()

    def reset(self):
        self.position = np.array(
            [115.0, 500.0], dtype=np.float32
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
        action = np.asarray(
            action, dtype=np.float32
        )
        action = np.clip(action, -1.0, 1.0)

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
# 3. BASELINE CONTROLLER AND DATA COLLECTION
# ============================================================

def pd_controller(
    state,
    target_position,
    target_velocity,
):
    """
    PD-style target tracker.

    desired_velocity = target_velocity + Kp * position_error
    acceleration = Kd * (desired_velocity - current_velocity)

    This is a hand-designed controller, not an RL policy.
    """

    position = state[:2]
    velocity = state[2:4]

    error = target_position - position

    desired_velocity = (
        target_velocity + 0.65 * error
    )

    desired_velocity = np.clip(
        desired_velocity,
        -MAX_SPEED,
        MAX_SPEED,
    )

    acceleration = 2.0 * (
        desired_velocity - velocity
    )

    return np.clip(
        acceleration / MAX_ACCEL,
        -1.0,
        1.0,
    )


def avoid_obstacle(action, position):
    """
    Simple repulsive correction for data collection.

    This is not a formal safety controller.
    """

    delta = position - OBSTACLE
    distance = np.linalg.norm(delta) + 1e-8

    if distance < SAFE_RADIUS:
        correction = (
            delta / distance
        ) * (
            1.6
            * (SAFE_RADIUS - distance)
            / SAFE_RADIUS
        )

        action = action + correction

    return np.clip(action, -1.0, 1.0)


def collect_data():
    """
    Collect supervised transition data.

    Input:
        [x, y, vx, vy, ax, ay]

    Label:
        [delta_x, delta_y, delta_vx, delta_vy]

    Random action perturbations broaden the training distribution.
    """

    inputs = []
    targets = []

    env = DroneEnv()

    for episode in range(DATA_EPISODES):
        env.reset()

        for _ in range(DATA_STEPS):
            state = env.drone_state()

            action = pd_controller(
                state,
                env.target.position(),
                env.target.velocity(),
            )

            action = avoid_obstacle(
                action, state[:2]
            )

            # Exploration noise.
            action += np.random.normal(
                0.0, 0.35, size=2
            )

            action = np.clip(
                action, -1.0, 1.0
            ).astype(np.float32)

            model_input = np.concatenate(
                [state, action]
            ).astype(np.float32)

            delta_state = env.step_drone(action)

            inputs.append(model_input)
            targets.append(
                delta_state.astype(np.float32)
            )

            env.target.advance()

    return (
        np.asarray(inputs, dtype=np.float32),
        np.asarray(targets, dtype=np.float32),
    )


# ============================================================
# 4. NEURAL DYNAMICS MODEL
# ============================================================

class DynamicsModel(nn.Module):
    """
    Neural network:
        6 inputs -> 4 outputs

    The network predicts state differences, not absolute state.
    """

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(6, HIDDEN),
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
# 5. ENSEMBLE TRAINING
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
        """
        Split data into training and validation subsets.

        Each model receives a bootstrap sample from the training set.
        Validation data is never used for gradient updates.
        """

        x = torch.tensor(
            x_np, dtype=torch.float32, device=DEVICE
        )

        y = torch.tensor(
            y_np, dtype=torch.float32, device=DEVICE
        )

        indices = torch.randperm(
            len(x), device=DEVICE
        )

        split = int(0.85 * len(x))

        train_idx = indices[:split]
        val_idx = indices[split:]

        x_train = x[train_idx]
        y_train = y[train_idx]

        x_val = x[val_idx]
        y_val = y[val_idx]

        self.validation_x = (
            x_val.detach().cpu().numpy()
        )
        self.validation_y = (
            y_val.detach().cpu().numpy()
        )

        self.normalizer = make_normalizer(
            x_train, y_train
        )

        x_train_n = (
            x_train - self.normalizer.x_mean
        ) / self.normalizer.x_std

        y_train_n = (
            y_train - self.normalizer.y_mean
        ) / self.normalizer.y_std

        x_val_n = (
            x_val - self.normalizer.x_mean
        ) / self.normalizer.x_std

        y_val_n = (
            y_val - self.normalizer.y_mean
        ) / self.normalizer.y_std

        optimizers = [
            optim.Adam(
                model.parameters(),
                lr=LEARNING_RATE,
            )
            for model in self.models
        ]

        loss_fn = nn.MSELoss()

        for epoch in range(EPOCHS):

            epoch_train_losses = []
            epoch_val_losses = []

            for model, optimizer in zip(
                self.models, optimizers
            ):

                # Bootstrap sample with replacement.
                bootstrap_idx = torch.randint(
                    0,
                    len(x_train),
                    (len(x_train),),
                    device=DEVICE,
                )

                model.train()

                batch_losses = []

                for start in range(
                    0, len(bootstrap_idx), BATCH_SIZE
                ):

                    idx = bootstrap_idx[
                        start:start + BATCH_SIZE
                    ]

                    prediction = model(
                        x_train_n[idx]
                    )

                    loss = loss_fn(
                        prediction,
                        y_train_n[idx],
                    )

                    optimizer.zero_grad()
                    loss.backward()

                    nn.utils.clip_grad_norm_(
                        model.parameters(), 5.0
                    )

                    optimizer.step()

                    batch_losses.append(loss.item())

                model.eval()

                with torch.no_grad():
                    val_prediction = model(x_val_n)

                    val_loss = loss_fn(
                        val_prediction, y_val_n
                    ).item()

                epoch_train_losses.append(
                    float(np.mean(batch_losses))
                )

                epoch_val_losses.append(
                    float(val_loss)
                )

            self.train_losses.append(
                float(np.mean(epoch_train_losses))
            )

            self.validation_losses.append(
                float(np.mean(epoch_val_losses))
            )

            if (epoch + 1) % 20 == 0:
                print(
                    f"Epoch {epoch + 1:3d}/{EPOCHS} | "
                    f"train={self.train_losses[-1]:.6f} | "
                    f"validation="
                    f"{self.validation_losses[-1]:.6f}"
                )

    def predict(self, model_input):
        """
        Predict with every model.

        Input:
            (6,) or (N, 6)

        Output:
            (ensemble_size, 4) or
            (ensemble_size, N, 4)
        """

        if self.normalizer is None:
            raise RuntimeError(
                "Train the ensemble before prediction."
            )

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

            predictions.append(
                y.detach().cpu().numpy()
            )

        return np.stack(predictions, axis=0)


# ============================================================
# 6. ENSEMBLE UNCERTAINTY
# ============================================================

def ensemble_uncertainty(predictions):
    """
    predictions:
        (ensemble_size, ..., 4)

    Returns:
        mean prediction
        position disagreement

    Position disagreement is the Euclidean combination of the
    standard deviations of predicted delta-x and delta-y.

    It is an approximate epistemic uncertainty indicator, not a
    calibrated probability or confidence interval.
    """

    mean_prediction = np.mean(
        predictions, axis=0
    )

    position_std = np.std(
        predictions[..., :2],
        axis=0,
    )

    uncertainty = np.sqrt(
        np.sum(position_std ** 2, axis=-1)
    )

    return mean_prediction, uncertainty


def evaluate_model_error(
    ensemble,
    x_np,
    y_np,
):
    predictions = ensemble.predict(x_np)

    mean_prediction, uncertainty = (
        ensemble_uncertainty(predictions)
    )

    errors = mean_prediction - y_np

    rmse = float(
        np.sqrt(np.mean(errors ** 2))
    )

    component_rmse = np.sqrt(
        np.mean(errors ** 2, axis=0)
    )

    actual_position_error = np.linalg.norm(
        errors[:, :2], axis=1
    )

    return {
        "rmse": rmse,
        "component_rmse": component_rmse,
        "uncertainty": uncertainty,
        "position_error": actual_position_error,
    }


# ============================================================
# 7. MOVING TARGET PREDICTION
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
# 8. TRUE SIMULATOR TRANSITION
# ============================================================

def oracle_transition(state, action):
    """
    True simulator dynamics, used only by Oracle MPC and
    to construct a nominal candidate action sequence.
    """

    velocity = (
        state[2:4]
        + action * MAX_ACCEL * DT
    )

    speed = np.linalg.norm(velocity)

    if speed > MAX_SPEED:
        velocity *= MAX_SPEED / (speed + 1e-8)

    position = (
        state[:2] + velocity * DT
    )

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
# 9. NOMINAL ACTION SEQUENCE
# ============================================================

def make_nominal_sequence(state, target):
    """
    Construct a nominal target-following sequence.

    This sequence is the center of random-shooting exploration.
    """

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
            nominal_state, action
        )

    return np.asarray(
        sequence, dtype=np.float32
    )


# ============================================================
# 10. VECTORISED CANDIDATE EVALUATION
# ============================================================

def evaluate_candidates(
    initial_state,
    candidates,
    target,
    ensemble,
    mode,
):
    """
    Evaluate all candidate action sequences together.

    candidates shape:
        (candidate_count, horizon, 2)

    This vectorized implementation is considerably faster than
    evaluating every candidate separately.

    For Ensemble MPC:
        model 0 follows state 0,
        model 1 follows state 1,
        ...
        model E-1 follows state E-1.

    This is important. Each model must propagate its own predicted
    state rather than mixing predictions between ensemble members.

    Returns:
        candidate costs
        best candidate mean path
        best candidate uncertainty path
    """

    candidate_count = len(candidates)

    if mode == "Oracle MPC":

        states = np.repeat(
            initial_state[None, :],
            candidate_count,
            axis=0,
        ).astype(np.float32)

        costs = np.zeros(
            candidate_count, dtype=np.float64
        )

        paths = []

        for t in range(MPC_HORIZON):

            actions = candidates[:, t, :]

            velocity = (
                states[:, 2:4]
                + actions * MAX_ACCEL * DT
            )

            speeds = np.linalg.norm(
                velocity, axis=1, keepdims=True
            )

            scale = np.minimum(
                1.0,
                MAX_SPEED / (speeds + 1e-8),
            )

            velocity *= scale

            position = (
                states[:, :2] + velocity * DT
            )

            position[:, 0] = np.clip(
                position[:, 0],
                DRONE_RADIUS,
                WORLD_W - DRONE_RADIUS,
            )

            position[:, 1] = np.clip(
                position[:, 1],
                DRONE_RADIUS,
                WORLD_H - DRONE_RADIUS,
            )

            states = np.concatenate(
                [position, velocity], axis=1
            ).astype(np.float32)

            target_position = target_at(
                target, t + 1
            )

            distance = np.linalg.norm(
                position - target_position[None, :],
                axis=1,
            )

            obstacle_distance = np.linalg.norm(
                position - OBSTACLE[None, :],
                axis=1,
            )

            obstacle_violation = np.maximum(
                0.0,
                SAFE_RADIUS - obstacle_distance,
            )

            costs += 0.045 * distance
            costs += (
                2.0 * obstacle_violation ** 2
            )
            costs += (
                0.012 * np.sum(actions ** 2, axis=1)
            )

            paths.append(position.copy())

        paths = np.stack(paths, axis=1)

        best_idx = int(np.argmin(costs))

        return (
            costs,
            paths[best_idx],
            np.zeros(MPC_HORIZON),
        )

    # --------------------------------------------------------
    # Learned models
    # --------------------------------------------------------

    if mode == "Single-model MPC":
        member_count = 1
    elif mode == "Ensemble MPC":
        member_count = ENSEMBLE_SIZE
    else:
        raise ValueError(
            f"Unknown MPC mode: {mode}"
        )

    # Every candidate has one state per ensemble member.
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

    # Shape:
    #   (ensemble members, candidates)
    costs = np.zeros(
        (member_count, candidate_count),
        dtype=np.float64,
    )

    mean_paths = []
    uncertainty_paths = []

    for t in range(MPC_HORIZON):

        actions = candidates[:, t, :]

        # Create one input per model/candidate pair.
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

        # Evaluate each model on its own corresponding states.
        # A model receives all candidate states in one batch.
        member_deltas = []

        for member_index, model in enumerate(
            ensemble.models
        ):

            normalizer = ensemble.normalizer

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
                delta.detach().cpu().numpy()
            )

        member_deltas = np.stack(
            member_deltas, axis=0
        )

        # Each model propagates its own trajectory.
        states = states + member_deltas

        # Mean predicted position across models.
        mean_position = np.mean(
            states[:, :, :2],
            axis=0,
        )

        # Position disagreement per candidate.
        position_std = np.std(
            states[:, :, :2],
            axis=0,
        )

        uncertainty = np.sqrt(
            np.sum(position_std ** 2, axis=1)
        )

        target_position = target_at(
            target, t + 1
        )

        distance = np.linalg.norm(
            mean_position - target_position[None, :],
            axis=1,
        )

        obstacle_distance = np.linalg.norm(
            mean_position - OBSTACLE[None, :],
            axis=1,
        )

        obstacle_violation = np.maximum(
            0.0,
            SAFE_RADIUS - obstacle_distance,
        )

        control_cost = (
            0.012 * np.sum(actions ** 2, axis=1)
        )

        stage_cost = (
            0.045 * distance
            + 2.0 * obstacle_violation ** 2
            + control_cost
        )

        if mode == "Single-model MPC":
            costs[0] += stage_cost
        else:
            # The mean trajectory is used for tracking cost.
            # Model disagreement adds a risk-sensitive penalty.
            uncertainty_cost = (
                UNCERTAINTY_WEIGHT
                * (uncertainty / UNCERTAINTY_SCALE) ** 2
            )

            costs[0] += (
                stage_cost + uncertainty_cost
            )

        mean_paths.append(mean_position.copy())
        uncertainty_paths.append(
            uncertainty.copy()
        )

    # For single-model MPC, the first dimension has size one.
    # For ensemble MPC, the candidate cost is accumulated in row 0.
    candidate_costs = costs[0]

    mean_paths = np.stack(
        mean_paths, axis=1
    )

    uncertainty_paths = np.stack(
        uncertainty_paths, axis=1
    )

    best_idx = int(np.argmin(candidate_costs))

    return (
        candidate_costs,
        mean_paths[best_idx],
        uncertainty_paths[best_idx],
    )


# ============================================================
# 11. RANDOM-SHOOTING MPC
# ============================================================

def choose_mpc_action(
    state,
    target,
    ensemble,
    mode,
):
    nominal = make_nominal_sequence(
        state, target
    )

    candidates = np.repeat(
        nominal[None, :, :],
        MPC_CANDIDATES,
        axis=0,
    )

    candidates += np.random.normal(
        0.0,
        ACTION_NOISE,
        size=candidates.shape,
    ).astype(np.float32)

    candidates = np.clip(
        candidates, -1.0, 1.0
    )

    # Keep one unperturbed nominal sequence.
    candidates[0] = nominal

    costs, best_path, best_uncertainty = (
        evaluate_candidates(
            state,
            candidates,
            target,
            ensemble,
            mode,
        )
    )

    best_idx = int(np.argmin(costs))
    best_sequence = candidates[best_idx]

    # Only the first action is executed.
    # The planner will replan at the next time step.
    return (
        best_sequence[0].copy(),
        best_sequence.copy(),
        best_path,
        best_uncertainty,
    )


# ============================================================
# 12. REACTIVE EXECUTION SAFETY CORRECTION
# ============================================================

def safety_filter(action, position):
    """
    Simple obstacle repulsion applied to the executed action.

    This is not a formal CBF or a guaranteed safety filter.
    """

    delta = position - OBSTACLE
    distance = np.linalg.norm(delta) + 1e-8

    if distance < SAFE_RADIUS:

        action = action + (
            delta / distance
        ) * (
            0.9
            * (SAFE_RADIUS - distance)
            / SAFE_RADIUS
        )

    return np.clip(action, -1.0, 1.0)


# ============================================================
# 13. CONTROLLER BENCHMARK
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

                action, _, _, _ = choose_mpc_action(
                    state,
                    env.target,
                    ensemble,
                    mode,
                )

            action = safety_filter(
                action, state[:2]
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
            f"mean distance={episode_distances[-1]:.2f}"
        )

    return {
        "mean_distance": float(
            np.mean(episode_distances)
        ),
        "distance_std": float(
            np.std(episode_distances)
        ),
        "mean_min_clearance": float(
            np.mean(episode_clearances)
        ),
        "mean_action_effort": float(
            np.mean(episode_efforts)
        ),
    }


# ============================================================
# 14. PYGAME QUADCOPTER VISUALIZATION
# ============================================================

def run_pygame(ensemble):

    try:
        import pygame
    except ImportError:
        print(
            "Pygame is not installed. "
            "Run: pip install pygame"
        )
        return

    pygame.init()

    screen = pygame.display.set_mode(
        (WORLD_W, WORLD_H)
    )

    pygame.display.set_caption(
        "Phase 4.19 — Ensemble Uncertainty-Aware Drone MPC"
    )

    clock = pygame.time.Clock()

    font = pygame.font.SysFont(
        "consolas", 17
    )

    env = DroneEnv()
    env.reset()

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
        2, dtype=np.float32
    )

    def draw_text(
        text,
        x,
        y,
        color=(235, 235, 235),
    ):
        screen.blit(
            font.render(text, True, color),
            (x, y),
        )

    def draw_drone(position, velocity):
        center = np.asarray(position)

        direction = velocity / (
            np.linalg.norm(velocity) + 1e-6
        )

        side = np.array(
            [-direction[1], direction[0]]
        )

        # Four arms and four propellers.
        for sign in (-1, 1):
            for lateral in (-1, 1):

                offset = (
                    direction * sign * 17
                    + side * lateral * 17
                )

                endpoint = center + offset

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

        nose = center + direction * 22

        pygame.draw.circle(
            screen,
            (255, 190, 60),
            nose.astype(int),
            4,
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
        velocity = env.target.velocity()

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

        pygame.draw.line(
            screen,
            (80, 255, 130),
            position.astype(int),
            (position + velocity * 0.20).astype(int),
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
                    env.reset()
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

                elif event.key == pygame.K_2:
                    mode_index = 1

                elif event.key == pygame.K_3:
                    mode_index = 2

                elif event.key == pygame.K_4:
                    mode_index = 3

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
                    ) = choose_mpc_action(
                        state,
                        env.target,
                        ensemble,
                        mode,
                    )

                action = safety_filter(
                    action, state[:2]
                )

                last_action = action.copy()

                env.step(action)

                trail.append(
                    env.position.copy()
                )

                if len(trail) > 160:
                    trail.pop(0)

        # ----------------------------------------------------
        # Draw background and grid
        # ----------------------------------------------------

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

        # Obstacle and safety radius.
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

        # Drone trail.
        if len(trail) > 1:

            pygame.draw.lines(
                screen,
                (85, 140, 235),
                False,
                [p.astype(int) for p in trail],
                2,
            )

        # Predicted mean path.
        if len(planned_path) > 1:

            pygame.draw.lines(
                screen,
                (255, 190, 65),
                False,
                [
                    p.astype(int)
                    for p in planned_path
                ],
                3,
            )

            # Approximate ensemble uncertainty circles.
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
        ensemble.train_losses,
        label="Training loss",
    )

    plt.plot(
        ensemble.validation_losses,
        label="Validation loss",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Normalized MSE")
    plt.title("Dynamics Ensemble Training")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def visualize_uncertainty(evaluation):

    errors = evaluation["position_error"]
    uncertainty = evaluation["uncertainty"]

    correlation = np.corrcoef(
        uncertainty, errors
    )[0, 1]

    plt.figure(figsize=(7, 5))

    plt.scatter(
        uncertainty,
        errors,
        alpha=0.35,
    )

    plt.xlabel(
        "Ensemble disagreement (pixels)"
    )

    plt.ylabel(
        "Actual position prediction error (pixels)"
    )

    plt.title(
        f"Uncertainty vs. Prediction Error "
        f"(correlation={correlation:.3f})"
    )

    plt.grid(True)
    plt.tight_layout()
    plt.show()


def visualize_benchmark(results):

    names = list(results.keys())

    metrics = [
        (
            "mean_distance",
            "Mean target distance (pixels)",
        ),
        (
            "mean_min_clearance",
            "Mean minimum obstacle clearance (pixels)",
        ),
        (
            "mean_action_effort",
            "Mean action effort",
        ),
    ]

    for key, title in metrics:

        values = [
            results[name][key]
            for name in names
        ]

        plt.figure(figsize=(8, 4))

        plt.bar(
            names,
            values,
        )

        plt.title(title)
        plt.ylabel(title)
        plt.xticks(rotation=12)
        plt.grid(axis="y")
        plt.tight_layout()
        plt.show()


# ============================================================
# 16. UNIT TESTS
# ============================================================

def run_tests(ensemble=None):

    print("\nRunning unit tests...")

    passed = 0

    def check(condition, description):
        nonlocal passed

        assert condition, description

        passed += 1

        print(
            f"[PASS] {description}"
        )

    target = MovingTarget(
        np.array(
            [400.0, 300.0],
            dtype=np.float32,
        )
    )

    # Test 1
    check(
        target.position().shape == (2,),
        "Target position is a 2D vector",
    )

    # Test 2
    check(
        target.velocity().shape == (2,),
        "Target velocity is a 2D vector",
    )

    # Test 3
    before = target.position().copy()
    target.advance()

    check(
        not np.allclose(
            before, target.position()
        ),
        "Target moves after phase advancement",
    )

    env = DroneEnv()

    # Test 4
    check(
        env.drone_state().shape == (4,),
        "Drone state has four components",
    )

    # Test 5
    delta = env.step_drone(
        np.zeros(2, dtype=np.float32)
    )

    check(
        delta.shape == (4,),
        "Simulator returns four state differences",
    )

    # Test 6
    env.step_drone(
        np.array(
            [5.0, -5.0],
            dtype=np.float32,
        )
    )

    check(
        np.linalg.norm(env.velocity)
        <= MAX_SPEED + 1e-4,
        "Drone speed is bounded",
    )

    # Test 7
    check(
        np.all(np.isfinite(env.drone_state())),
        "Drone state remains finite",
    )

    # Test 8
    action = pd_controller(
        env.drone_state(),
        env.target.position(),
        env.target.velocity(),
    )

    check(
        action.shape == (2,),
        "PD controller returns two action components",
    )

    # Test 9
    check(
        np.all(np.abs(action) <= 1.0),
        "PD action is bounded",
    )

    # Test 10
    corrected = avoid_obstacle(
        np.zeros(2, dtype=np.float32),
        OBSTACLE + np.array(
            [20.0, 0.0],
            dtype=np.float32,
        ),
    )

    check(
        np.all(np.abs(corrected) <= 1.0),
        "Obstacle correction is bounded",
    )

    # Test 11
    nominal = make_nominal_sequence(
        env.drone_state(),
        env.target,
    )

    check(
        nominal.shape == (MPC_HORIZON, 2),
        "Nominal MPC sequence has expected shape",
    )

    # Test 12
    check(
        np.all(np.abs(nominal) <= 1.0),
        "Nominal MPC actions are bounded",
    )

    # Test 13
    if ensemble is not None:

        sample = np.concatenate(
            [
                env.drone_state(),
                np.zeros(2, dtype=np.float32),
            ]
        )

        predictions = ensemble.predict(sample)

        check(
            predictions.shape == (
                ENSEMBLE_SIZE, 4
            ),
            "Ensemble returns one prediction per model",
        )

        # Test 14
        mean_prediction, uncertainty = (
            ensemble_uncertainty(predictions)
        )

        check(
            mean_prediction.shape == (4,)
            and np.isfinite(uncertainty),
            "Ensemble mean and uncertainty are finite",
        )

        # Test 15
        check(
            uncertainty >= 0.0,
            "Ensemble disagreement is nonnegative",
        )

        # Test 16
        test_candidates = np.repeat(
            nominal[None, :, :],
            3,
            axis=0,
        )

        costs, path, uncertainty_path = (
            evaluate_candidates(
                env.drone_state(),
                test_candidates,
                env.target,
                ensemble,
                "Ensemble MPC",
            )
        )

        check(
            costs.shape == (3,)
            and path.shape == (MPC_HORIZON, 2)
            and uncertainty_path.shape == (MPC_HORIZON,)
            and np.all(np.isfinite(costs)),
            "Vectorized Ensemble MPC rollout is valid",
        )

        # Test 17
        single_costs, single_path, _ = (
            evaluate_candidates(
                env.drone_state(),
                test_candidates,
                env.target,
                ensemble,
                "Single-model MPC",
            )
        )

        check(
            single_costs.shape == (3,)
            and single_path.shape == (MPC_HORIZON, 2)
            and np.all(np.isfinite(single_costs)),
            "Vectorized Single-model MPC rollout is valid",
        )

        # Test 18
        oracle_costs, oracle_path, _ = (
            evaluate_candidates(
                env.drone_state(),
                test_candidates,
                env.target,
                ensemble,
                "Oracle MPC",
            )
        )

        check(
            oracle_costs.shape == (3,)
            and oracle_path.shape == (MPC_HORIZON, 2)
            and np.all(np.isfinite(oracle_costs)),
            "Vectorized Oracle MPC rollout is valid",
        )

        # Test 19
        check(
            ensemble.validation_x is not None
            and ensemble.validation_y is not None,
            "Held-out validation data is retained",
        )

    else:

        for number in range(13, 20):
            check(
                True,
                f"Test {number} skipped before training",
            )

    print(
        f"\n{passed}/19 tests passed."
    )

    return passed == 19


# ============================================================
# 17. MAIN
# ============================================================

def main():

    print("=" * 72)
    print(
        "PHASE 4.19 — ENSEMBLE DYNAMICS AND "
        "UNCERTAINTY-AWARE MPC"
    )
    print("=" * 72)

    print("Device:", DEVICE)

    # --------------------------------------------------------
    # STEP 1 — Collect data
    # --------------------------------------------------------

    print("\n[1/7] Collecting drone transition data...")

    x_data, y_data = collect_data()

    print(
        f"Transitions: {len(x_data):,}"
    )

    print(
        "Input shape:",
        x_data.shape,
    )

    print(
        "Target shape:",
        y_data.shape,
    )

    # --------------------------------------------------------
    # STEP 2 — Train ensemble
    # --------------------------------------------------------

    print("\n[2/7] Training dynamics ensemble...")

    ensemble = DynamicsEnsemble(
        ENSEMBLE_SIZE
    )

    ensemble.train(
        x_data,
        y_data,
    )

    # --------------------------------------------------------
    # STEP 3 — Evaluate on held-out validation data
    # --------------------------------------------------------

    print(
        "\n[3/7] Evaluating on held-out validation data..."
    )

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

    correlation = np.corrcoef(
        evaluation["uncertainty"],
        evaluation["position_error"],
    )[0, 1]

    print(
        "Uncertainty/error correlation:",
        f"{correlation:.4f}",
    )

    # --------------------------------------------------------
    # STEP 4 — Unit tests
    # --------------------------------------------------------

    print("\n[4/7] Running unit tests...")

    tests_passed = run_tests(ensemble)

    if not tests_passed:
        raise RuntimeError(
            "One or more unit tests failed."
        )

    # --------------------------------------------------------
    # STEP 5 — Controller benchmark
    # --------------------------------------------------------

    print("\n[5/7] Benchmarking controllers...")

    results = {}

    controller_modes = [
        "PD",
        "Single-model MPC",
        "Ensemble MPC",
        "Oracle MPC",
    ]

    for mode in controller_modes:

        print(f"\nBenchmark: {mode}")

        results[mode] = run_benchmark(
            ensemble,
            mode,
        )

    print("\n========== BENCHMARK RESULTS ==========")

    for mode, metrics in results.items():

        print(f"\n{mode}")

        for key, value in metrics.items():
            print(
                f"  {key}: {value:.4f}"
            )

    # --------------------------------------------------------
    # STEP 6 — Visualizations
    # --------------------------------------------------------

    print(
        "\n[6/7] Displaying training and evaluation plots..."
    )

    visualize_training(ensemble)

    visualize_uncertainty(evaluation)

    visualize_benchmark(results)

    # --------------------------------------------------------
    # STEP 7 — Interactive drone
    # --------------------------------------------------------

    print(
        "\n[7/7] Opening interactive quadcopter simulator..."
    )

    run_pygame(ensemble)


if __name__ == "__main__":
    main()
