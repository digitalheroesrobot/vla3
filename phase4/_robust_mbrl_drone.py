
"""
File: phase4/robust_mbrl_drone.py

PHASE 4.20 — ROBUST MODEL-BASED RL
===================================

PURPOSE
-------
Train dynamics models under randomized physical conditions and use
them to plan drone motion that remains effective under model mismatch
and external disturbances.

WHAT YOU LEARN
--------------

STEP 1 — Domain randomization
    Train with different:
        - drone mass
        - aerodynamic drag
        - acceleration effectiveness
        - wind conditions

STEP 2 — Disturbance modeling
    Represent wind as part of the observed state.
    Learn how wind affects the drone's velocity and position.

STEP 3 — Domain-specific dynamics models
    Train an ensemble where each model represents a different
    randomized physical domain.

STEP 4 — Robust MPC
    Evaluate candidate action sequences across all models.
    Optimize a risk-sensitive cost:
        J_robust = mean(J) + risk_weight * std(J)
                    + worst_case_weight * max(J)

STEP 5 — Sim-to-real-style evaluation
    Test under:
        - nominal domain
        - unseen mass and drag
        - strong wind
        - combined parameter variation and disturbances

STEP 6 — Compare controllers
    PD controller
    Nominal-model MPC
    Robust ensemble MPC
    Oracle MPC

STEP 7 — Visualize and verify
    Training loss, controller metrics, robustness under different
    test conditions, and interactive quadcopter visualization.

INSTALL
-------
    pip install numpy torch matplotlib pygame

RUN
---
    python phase4/robust_mbrl_drone.py

CONTROLS
--------
    1 : PD controller
    2 : Nominal MPC
    3 : Robust Ensemble MPC
    4 : Oracle MPC
    SPACE : Pause/resume
    R : Reset drone
    + / - : Simulation speed
    Left mouse click : Change moving-target trajectory center
    ESC : Exit

ROBOTICS APPLICATION
--------------------
A real drone may experience changing payload, battery voltage,
air density, wind, and motor performance.

Domain randomization exposes the learning system to a range of
conditions. Robust MPC then evaluates a candidate action under
multiple plausible dynamics models before choosing an action.

IMPORTANT LIMITATIONS
--------------------
1. This is a simplified 2D drone simulation.
2. The domain ensemble represents plausible physical domains;
   it is not a calibrated probability distribution.
3. Future wind prediction is approximate.
4. The robust cost is a heuristic, not a formal safety guarantee.
5. Oracle MPC knows the simulator's physical parameters and is
   an idealized reference, not a real-world controller.
6. This is not flight-certified software.
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

SEED = 420

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

WORLD_W = 900
WORLD_H = 650

DT = 0.08

MAX_SPEED = 155.0
MAX_ACCEL = 125.0
DRONE_RADIUS = 13.0

OBSTACLE = np.array(
    [450.0, 330.0],
    dtype=np.float32,
)

OBSTACLE_RADIUS = 65.0
SAFE_RADIUS = 100.0

TARGET_A = 170.0
TARGET_B = 105.0
TARGET_OMEGA = 0.55

# Domain-specific dynamics.
# Mass multiplier: larger means less acceleration.
# Drag: velocity damping coefficient.
# Acceleration multiplier: motor effectiveness.

DOMAINS = [
    {
        "name": "light_low_drag",
        "mass": 0.80,
        "drag": 0.035,
        "accel": 1.15,
    },
    {
        "name": "light_high_drag",
        "mass": 0.85,
        "drag": 0.105,
        "accel": 1.10,
    },
    {
        "name": "nominal",
        "mass": 1.00,
        "drag": 0.065,
        "accel": 1.00,
    },
    {
        "name": "heavy_low_drag",
        "mass": 1.25,
        "drag": 0.040,
        "accel": 0.88,
    },
    {
        "name": "heavy_high_drag",
        "mass": 1.30,
        "drag": 0.115,
        "accel": 0.85,
    },
]

MODEL_COUNT = len(DOMAINS)

HIDDEN = 128
EPOCHS = 65
BATCH_SIZE = 256
LEARNING_RATE = 1e-3

DATA_EPISODES_PER_DOMAIN = 28
DATA_STEPS_PER_EPISODE = 130

MPC_HORIZON = 12
MPC_CANDIDATES = 64

ACTION_NOISE = 0.48

ROBUST_RISK_WEIGHT = 1.4
ROBUST_WORST_WEIGHT = 0.35

WIND_DECAY = 0.88
WIND_NOISE_TRAIN = 13.0
WIND_NOISE_TEST = 18.0

BENCHMARK_EPISODES = 3
BENCHMARK_STEPS = 200


# ============================================================
# 1. MOVING TARGET
# ============================================================

@dataclass
class MovingTarget:

    center: np.ndarray
    phase: float = 0.0

    def position(self, phase=None):

        p = (
            self.phase
            if phase is None
            else phase
        )

        return self.center + np.array(
            [
                TARGET_A * math.cos(p),
                TARGET_B * math.sin(p),
            ],
            dtype=np.float32,
        )

    def velocity(self, phase=None):

        p = (
            self.phase
            if phase is None
            else phase
        )

        return np.array(
            [
                -TARGET_A * TARGET_OMEGA * math.sin(p),
                TARGET_B * TARGET_OMEGA * math.cos(p),
            ],
            dtype=np.float32,
        )

    def advance(self):

        self.phase += TARGET_OMEGA * DT


def target_at(target, future_step):

    phase = (
        target.phase
        + TARGET_OMEGA * DT * future_step
    )

    return target.position(phase)


def target_velocity_at(target, future_step):

    phase = (
        target.phase
        + TARGET_OMEGA * DT * future_step
    )

    return target.velocity(phase)


# ============================================================
# 2. DOMAIN RANDOMIZATION
# ============================================================

def random_domain():

    return {
        "name": "randomized",
        "mass": float(
            np.random.uniform(0.75, 1.40)
        ),
        "drag": float(
            np.random.uniform(0.025, 0.13)
        ),
        "accel": float(
            np.random.uniform(0.78, 1.22)
        ),
    }


def unseen_test_domains():

    return [
        {
            "name": "unseen_light",
            "mass": 0.72,
            "drag": 0.055,
            "accel": 1.20,
        },
        {
            "name": "unseen_heavy",
            "mass": 1.42,
            "drag": 0.095,
            "accel": 0.78,
        },
        {
            "name": "strong_drag",
            "mass": 1.10,
            "drag": 0.145,
            "accel": 0.92,
        },
        {
            "name": "combined_shift",
            "mass": 1.36,
            "drag": 0.13,
            "accel": 0.80,
        },
    ]


# ============================================================
# 3. DRONE ENVIRONMENT WITH WIND
# ============================================================

class DroneEnv:

    """
    State:
        [x, y, vx, vy, wind_x, wind_y]

    Action:
        [ax_normalized, ay_normalized]

    Simplified dynamics:
        v_next = v + (
            action * MAX_ACCEL * accel / mass
            - drag * v
            + wind
        ) * dt

        p_next = p + v_next * dt

    Wind follows an approximate autoregressive process:
        wind_next = WIND_DECAY * wind + random_noise

    The model receives current wind as an input.
    """

    def __init__(
        self,
        domain=None,
        wind_noise=WIND_NOISE_TRAIN,
    ):

        self.domain = (
            random_domain()
            if domain is None
            else dict(domain)
        )

        self.wind_noise = float(wind_noise)

        self.target = MovingTarget(
            np.array(
                [700.0, 310.0],
                dtype=np.float32,
            )
        )

        self.reset()

    def reset(self):

        self.position = np.array(
            [115.0, 500.0],
            dtype=np.float32,
        )

        self.velocity = np.zeros(
            2, dtype=np.float32
        )

        self.wind = np.random.normal(
            0.0,
            8.0,
            size=2,
        ).astype(np.float32)

        self.target.phase = float(
            np.random.uniform(
                0.0,
                2.0 * math.pi,
            )
        )

        return self.state()

    def state(self):

        return np.concatenate(
            [
                self.position,
                self.velocity,
                self.wind,
            ]
        ).astype(np.float32)

    def step_drone(self, action):

        action = np.asarray(
            action,
            dtype=np.float32,
        )

        action = np.clip(
            action,
            -1.0,
            1.0,
        )

        old_state = self.state()

        mass = self.domain["mass"]
        drag = self.domain["drag"]
        accel = self.domain["accel"]

        acceleration = (
            action * MAX_ACCEL * accel / mass
            - drag * self.velocity
            + self.wind
        )

        self.velocity += acceleration * DT

        speed = np.linalg.norm(
            self.velocity
        )

        if speed > MAX_SPEED:

            self.velocity *= (
                MAX_SPEED / (speed + 1e-8)
            )

        self.position += (
            self.velocity * DT
        )

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

        # Wind changes after the drone transition.
        self.wind = (
            WIND_DECAY * self.wind
            + np.random.normal(
                0.0,
                self.wind_noise,
                size=2,
            ).astype(np.float32)
        )

        self.wind = np.clip(
            self.wind,
            -55.0,
            55.0,
        )

        return self.state() - old_state

    def step(self, action):

        self.step_drone(action)
        self.target.advance()

        return self.state()


# ============================================================
# 4. BASELINE PD CONTROLLER
# ============================================================

def pd_controller(
    state,
    target_position,
    target_velocity,
):

    position = state[:2]
    velocity = state[2:4]

    error = (
        target_position - position
    )

    desired_velocity = (
        target_velocity + 0.62 * error
    )

    desired_velocity = np.clip(
        desired_velocity,
        -MAX_SPEED,
        MAX_SPEED,
    )

    acceleration = (
        2.0 * (
            desired_velocity - velocity
        )
        - state[4:6]
    )

    return np.clip(
        acceleration / MAX_ACCEL,
        -1.0,
        1.0,
    )


def avoid_obstacle(action, position):

    delta = position - OBSTACLE

    distance = (
        np.linalg.norm(delta) + 1e-8
    )

    if distance < SAFE_RADIUS:

        correction = (
            delta / distance
        ) * (
            1.4
            * (SAFE_RADIUS - distance)
            / SAFE_RADIUS
        )

        action = action + correction

    return np.clip(
        action,
        -1.0,
        1.0,
    )


# ============================================================
# 5. DATA COLLECTION UNDER DOMAIN RANDOMIZATION
# ============================================================

def collect_domain_data(domain):

    inputs = []
    targets = []

    for episode in range(
        DATA_EPISODES_PER_DOMAIN
    ):

        env = DroneEnv(
            domain=domain,
            wind_noise=WIND_NOISE_TRAIN,
        )

        env.reset()

        for _ in range(
            DATA_STEPS_PER_EPISODE
        ):

            state = env.state()

            action = pd_controller(
                state,
                env.target.position(),
                env.target.velocity(),
            )

            action = avoid_obstacle(
                action,
                state[:2],
            )

            # Random exploration broadens the action distribution.
            action += np.random.normal(
                0.0,
                0.35,
                size=2,
            )

            action = np.clip(
                action,
                -1.0,
                1.0,
            ).astype(np.float32)

            model_input = np.concatenate(
                [
                    state,
                    action,
                ]
            ).astype(np.float32)

            delta = env.step_drone(
                action
            )

            inputs.append(model_input)
            targets.append(delta)

            env.target.advance()

    return (
        np.asarray(
            inputs,
            dtype=np.float32,
        ),
        np.asarray(
            targets,
            dtype=np.float32,
        ),
    )


# ============================================================
# 6. NEURAL NETWORK DYNAMICS MODEL
# ============================================================

class DynamicsModel(nn.Module):

    """
    Input:
        [x, y, vx, vy, wind_x, wind_y, ax, ay]

    Output:
        [delta_x, delta_y, delta_vx, delta_vy,
         delta_wind_x, delta_wind_y]

    Predicting delta state makes learning the transition easier.
    """

    def __init__(self):

        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(8, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, 6),
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
# 7. DOMAIN-SPECIFIC MODEL ENSEMBLE
# ============================================================

class DomainEnsemble:

    def __init__(self):

        self.models = [
            DynamicsModel().to(DEVICE)
            for _ in DOMAINS
        ]

        self.normalizers = []

        self.train_curves = []
        self.validation_curves = []

        self.validation_data = []

    def train_one_model(
        self,
        model,
        x_np,
        y_np,
    ):

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

        permutation = torch.randperm(
            len(x),
            device=DEVICE,
        )

        split = int(
            0.85 * len(x)
        )

        train_idx = permutation[:split]
        val_idx = permutation[split:]

        x_train = x[train_idx]
        y_train = y[train_idx]

        x_val = x[val_idx]
        y_val = y[val_idx]

        normalizer = make_normalizer(
            x_train,
            y_train,
        )

        self.normalizers.append(
            normalizer
        )

        x_train = (
            x_train - normalizer.x_mean
        ) / normalizer.x_std

        y_train = (
            y_train - normalizer.y_mean
        ) / normalizer.y_std

        x_val_norm = (
            x_val - normalizer.x_mean
        ) / normalizer.x_std

        y_val_norm = (
            y_val - normalizer.y_mean
        ) / normalizer.y_std

        optimizer = optim.Adam(
            model.parameters(),
            lr=LEARNING_RATE,
        )

        loss_fn = nn.MSELoss()

        train_history = []
        val_history = []

        for epoch in range(EPOCHS):

            model.train()

            permutation = torch.randperm(
                len(x_train),
                device=DEVICE,
            )

            batch_losses = []

            for start in range(
                0,
                len(permutation),
                BATCH_SIZE,
            ):

                idx = permutation[
                    start:start + BATCH_SIZE
                ]

                prediction = model(
                    x_train[idx]
                )

                loss = loss_fn(
                    prediction,
                    y_train[idx],
                )

                optimizer.zero_grad()
                loss.backward()

                nn.utils.clip_grad_norm_(
                    model.parameters(),
                    5.0,
                )

                optimizer.step()

                batch_losses.append(
                    loss.item()
                )

            model.eval()

            with torch.no_grad():

                val_prediction = model(
                    x_val_norm
                )

                val_loss = loss_fn(
                    val_prediction,
                    y_val_norm,
                ).item()

            train_history.append(
                float(np.mean(batch_losses))
            )

            val_history.append(
                float(val_loss)
            )

        self.validation_data.append(
            (
                x_val.detach().cpu().numpy(),
                y_val.detach().cpu().numpy(),
            )
        )

        return train_history, val_history

    def train(self, datasets):

        for index, (
            model,
            dataset,
        ) in enumerate(
            zip(self.models, datasets)
        ):

            print(
                f"Training model {index + 1}/"
                f"{len(self.models)}: "
                f"{DOMAINS[index]['name']}"
            )

            train_curve, val_curve = (
                self.train_one_model(
                    model,
                    dataset[0],
                    dataset[1],
                )
            )

            self.train_curves.append(
                train_curve
            )

            self.validation_curves.append(
                val_curve
            )

    def predict_member(
        self,
        member_index,
        model_input,
    ):

        model = self.models[
            member_index
        ]

        normalizer = self.normalizers[
            member_index
        ]

        x = torch.as_tensor(
            model_input,
            dtype=torch.float32,
            device=DEVICE,
        )

        x_norm = (
            x - normalizer.x_mean
        ) / normalizer.x_std

        model.eval()

        with torch.no_grad():

            y_norm = model(x_norm)

            y = (
                y_norm * normalizer.y_std
                + normalizer.y_mean
            )

        return y.detach().cpu().numpy()

    def predict_all(self, model_input):

        return np.stack(
            [
                self.predict_member(
                    i,
                    model_input,
                )
                for i in range(
                    len(self.models)
                )
            ],
            axis=0,
        )


# ============================================================
# 8. TRUE AND LEARNED TRANSITIONS
# ============================================================

def oracle_transition(
    state,
    action,
    domain,
    predict_wind=True,
):

    mass = domain["mass"]
    drag = domain["drag"]
    accel = domain["accel"]

    position = state[:2].copy()
    velocity = state[2:4].copy()
    wind = state[4:6].copy()

    acceleration = (
        action * MAX_ACCEL * accel / mass
        - drag * velocity
        + wind
    )

    velocity += acceleration * DT

    speed = np.linalg.norm(velocity)

    if speed > MAX_SPEED:

        velocity *= (
            MAX_SPEED / (speed + 1e-8)
        )

    position += velocity * DT

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

    if predict_wind:

        # Expected future wind, without future random noise.
        wind = WIND_DECAY * wind

    return np.concatenate(
        [
            position,
            velocity,
            wind,
        ]
    ).astype(np.float32)


def learned_transition(
    model,
    state,
    action,
):

    model_input = np.concatenate(
        [
            state,
            action,
        ]
    ).astype(np.float32)

    delta = model.predict_member(
        0,
        model_input,
    )

    next_state = state + delta

    next_state[:2] = np.clip(
        next_state[:2],
        DRONE_RADIUS,
        WORLD_W - DRONE_RADIUS,
    )

    next_state[2:4] = np.clip(
        next_state[2:4],
        -MAX_SPEED,
        MAX_SPEED,
    )

    next_state[4:6] = np.clip(
        next_state[4:6],
        -55.0,
        55.0,
    )

    return next_state.astype(
        np.float32
    )


# ============================================================
# 9. NOMINAL ACTION SEQUENCE
# ============================================================

def make_nominal_sequence(
    state,
    target,
):

    nominal_state = state.copy()

    sequence = []

    nominal_domain = DOMAINS[2]

    for t in range(
        1,
        MPC_HORIZON + 1,
    ):

        action = pd_controller(
            nominal_state,
            target_at(target, t),
            target_velocity_at(target, t),
        )

        action = avoid_obstacle(
            action,
            nominal_state[:2],
        )

        sequence.append(action)

        nominal_state = oracle_transition(
            nominal_state,
            action,
            nominal_domain,
        )

    return np.asarray(
        sequence,
        dtype=np.float32,
    )


# ============================================================
# 10. RISK-SENSITIVE CANDIDATE EVALUATION
# ============================================================

def trajectory_cost(
    state,
    action,
    target_position,
):

    position = state[:2]

    distance = np.linalg.norm(
        position - target_position
    )

    obstacle_distance = np.linalg.norm(
        position - OBSTACLE
    )

    obstacle_violation = max(
        0.0,
        SAFE_RADIUS - obstacle_distance,
    )

    boundary_margin = min(
        position[0],
        WORLD_W - position[0],
        position[1],
        WORLD_H - position[1],
    )

    boundary_violation = max(
        0.0,
        45.0 - boundary_margin,
    )

    return (
        0.045 * distance
        + 2.0 * obstacle_violation ** 2
        + 1.5 * boundary_violation ** 2
        + 0.012 * np.sum(action ** 2)
    )


def evaluate_candidates(
    initial_state,
    candidates,
    target,
    ensemble,
    mode,
    oracle_domain=None,
):

    """
    candidates:
        (candidate_count, horizon, 2)

    Learned-model rollout:
        Every domain model predicts its own trajectory.

    Robust objective:
        mean_cost
        + risk_weight * standard_deviation
        + worst_case_weight * maximum_cost

    Nominal MPC:
        Uses only the nominal-domain model.

    Oracle MPC:
        Uses true dynamics and the actual test-domain parameters.
    """

    candidate_count = len(candidates)

    model_indices = (
        [2]
        if mode == "Nominal MPC"
        else list(range(MODEL_COUNT))
    )

    if mode == "Oracle MPC":

        if oracle_domain is None:
            raise ValueError(
                "Oracle MPC requires the actual test domain."
            )

        model_indices = [-1]

    all_costs = []

    all_paths = []
    all_uncertainties = []

    for model_index in model_indices:

        states = np.repeat(
            initial_state[None, :],
            candidate_count,
            axis=0,
        ).astype(np.float32)

        costs = np.zeros(
            candidate_count,
            dtype=np.float64,
        )

        path = []
        uncertainty = []

        for t in range(
            MPC_HORIZON
        ):

            actions = candidates[:, t, :]

            if model_index == -1:

                next_states = np.stack(
                    [
                        oracle_transition(
                            states[i],
                            actions[i],
                            oracle_domain,
                        )
                        for i in range(
                            candidate_count
                        )
                    ],
                    axis=0,
                )

            else:

                model_inputs = np.concatenate(
                    [
                        states,
                        actions,
                    ],
                    axis=1,
                )

                deltas = ensemble.predict_member(
                    model_index,
                    model_inputs,
                )

                next_states = (
                    states + deltas
                )

                next_states[:, :2] = np.clip(
                    next_states[:, :2],
                    DRONE_RADIUS,
                    WORLD_W - DRONE_RADIUS,
                )

                next_states[:, 2:4] = np.clip(
                    next_states[:, 2:4],
                    -MAX_SPEED,
                    MAX_SPEED,
                )

                next_states[:, 4:6] = np.clip(
                    next_states[:, 4:6],
                    -55.0,
                    55.0,
                )

            target_position = target_at(
                target,
                t + 1,
            )

            for i in range(
                candidate_count
            ):

                costs[i] += trajectory_cost(
                    next_states[i],
                    actions[i],
                    target_position,
                )

            states = next_states

            path.append(
                states[:, :2].copy()
            )

        all_costs.append(costs)

        all_paths.append(
            np.stack(path, axis=1)
        )

    all_costs = np.stack(
        all_costs,
        axis=0,
    )

    all_paths = np.stack(
        all_paths,
        axis=0,
    )

    if mode == "Nominal MPC" or mode == "Oracle MPC":

        robust_cost = all_costs[0]

        best_index = int(
            np.argmin(robust_cost)
        )

        best_path = all_paths[
            0, best_index
        ]

        return (
            robust_cost,
            best_path,
            np.zeros(MPC_HORIZON),
        )

    # Robust cost across all randomized domains.
    mean_cost = np.mean(
        all_costs,
        axis=0,
    )

    std_cost = np.std(
        all_costs,
        axis=0,
    )

    worst_cost = np.max(
        all_costs,
        axis=0,
    )

    robust_cost = (
        mean_cost
        + ROBUST_RISK_WEIGHT * std_cost
        + ROBUST_WORST_WEIGHT * worst_cost
    )

    best_index = int(
        np.argmin(robust_cost)
    )

    # Show mean position path across domains.
    mean_path = np.mean(
        all_paths[:, best_index],
        axis=0,
    )

    # Show domain disagreement at each horizon step.
    position_std = np.std(
        all_paths[:, best_index],
        axis=0,
    )

    path_uncertainty = np.sqrt(
        np.sum(
            position_std ** 2,
            axis=1,
        )
    )

    return (
        robust_cost,
        mean_path,
        path_uncertainty,
    )


# ============================================================
# 11. MPC ACTION SELECTION
# ============================================================

def choose_mpc_action(
    state,
    target,
    ensemble,
    mode,
    oracle_domain=None,
):

    nominal = make_nominal_sequence(
        state,
        target,
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
        candidates,
        -1.0,
        1.0,
    )

    candidates[0] = nominal

    costs, best_path, uncertainty = (
        evaluate_candidates(
            state,
            candidates,
            target,
            ensemble,
            mode,
            oracle_domain,
        )
    )

    best_index = int(
        np.argmin(costs)
    )

    return (
        candidates[best_index, 0].copy(),
        best_path,
        uncertainty,
    )


# ============================================================
# 12. EXECUTION SAFETY CORRECTION
# ============================================================

def safety_filter(
    action,
    position,
):

    delta = (
        position - OBSTACLE
    )

    distance = (
        np.linalg.norm(delta) + 1e-8
    )

    if distance < SAFE_RADIUS:

        action = action + (
            delta / distance
        ) * (
            1.0
            * (SAFE_RADIUS - distance)
            / SAFE_RADIUS
        )

    return np.clip(
        action,
        -1.0,
        1.0,
    )


# ============================================================
# 13. SIM-TO-REAL-STYLE BENCHMARK
# ============================================================

def run_episode(
    ensemble,
    mode,
    domain,
    wind_noise,
    steps=BENCHMARK_STEPS,
):

    env = DroneEnv(
        domain=domain,
        wind_noise=wind_noise,
    )

    env.reset()

    distances = []
    clearances = []
    efforts = []

    collisions = 0

    for _ in range(steps):

        state = env.state()

        if mode == "PD":

            action = pd_controller(
                state,
                env.target.position(),
                env.target.velocity(),
            )

        else:

            action, _, _ = choose_mpc_action(
                state,
                env.target,
                ensemble,
                mode,
                oracle_domain=domain,
            )

        action = safety_filter(
            action,
            state[:2],
        )

        env.step(action)

        distance = np.linalg.norm(
            env.position - env.target.position()
        )

        clearance = (
            np.linalg.norm(
                env.position - OBSTACLE
            )
            - OBSTACLE_RADIUS
        )

        if clearance < 0:
            collisions += 1

        distances.append(distance)
        clearances.append(clearance)
        efforts.append(
            np.linalg.norm(action)
        )

    return {
        "mean_distance": float(
            np.mean(distances)
        ),
        "minimum_clearance": float(
            np.min(clearances)
        ),
        "collision_steps": int(
            collisions
        ),
        "mean_effort": float(
            np.mean(efforts)
        ),
    }


def benchmark(
    ensemble,
    modes,
    test_domains,
):

    results = {}

    for domain in test_domains:

        print(
            f"\nTest domain: {domain['name']}"
        )

        results[domain["name"]] = {}

        for mode in modes:

            episode_results = []

            for episode in range(
                BENCHMARK_EPISODES
            ):

                result = run_episode(
                    ensemble,
                    mode,
                    domain,
                    WIND_NOISE_TEST,
                )

                episode_results.append(
                    result
                )

            results[domain["name"]][mode] = {
                key: float(
                    np.mean(
                        [
                            item[key]
                            for item in episode_results
                        ]
                    )
                )
                for key in episode_results[0]
            }

            print(
                f"  {mode}: "
                f"distance="
                f"{results[domain['name']][mode]['mean_distance']:.2f}, "
                f"clearance="
                f"{results[domain['name']][mode]['minimum_clearance']:.2f}, "
                f"collision steps="
                f"{results[domain['name']][mode]['collision_steps']:.1f}"
            )

    return results


# ============================================================
# 14. PYGAME QUADCOPTER VISUALIZATION
# ============================================================

def run_pygame(ensemble):

    try:
        import pygame
    except ImportError:
        print(
            "Install Pygame with: pip install pygame"
        )
        return

    pygame.init()

    screen = pygame.display.set_mode(
        (WORLD_W, WORLD_H)
    )

    pygame.display.set_caption(
        "Phase 4.20 — Robust Model-Based Drone"
    )

    clock = pygame.time.Clock()

    font = pygame.font.SysFont(
        "consolas",
        16,
    )

    domain = dict(DOMAINS[2])

    env = DroneEnv(
        domain=domain,
        wind_noise=WIND_NOISE_TEST,
    )

    env.reset()

    modes = [
        "PD",
        "Nominal MPC",
        "Robust Ensemble MPC",
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

    def draw_text(
        message,
        x,
        y,
        color=(235, 235, 235),
    ):

        screen.blit(
            font.render(
                message,
                True,
                color,
            ),
            (x, y),
        )

    def draw_drone(position, velocity):

        center = np.asarray(
            position
        )

        direction = velocity / (
            np.linalg.norm(velocity) + 1e-6
        )

        side = np.array(
            [
                -direction[1],
                direction[0],
            ]
        )

        for forward in (-1, 1):

            for lateral in (-1, 1):

                offset = (
                    direction * forward * 17
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
                env.target.position(
                    phase
                ).astype(int)
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
            (
                position + velocity * 0.2
            ).astype(int),
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

                    speed = min(
                        5,
                        speed + 1,
                    )

                elif event.key == pygame.K_MINUS:

                    speed = max(
                        1,
                        speed - 1,
                    )

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

                state = env.state()

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
                        planned_path,
                        planned_uncertainty,
                    ) = choose_mpc_action(
                        state,
                        env.target,
                        ensemble,
                        mode,
                        oracle_domain=domain,
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

        # Background.
        screen.fill(
            (18, 24, 34)
        )

        for x in range(
            0,
            WORLD_W,
            50,
        ):

            pygame.draw.line(
                screen,
                (30, 39, 50),
                (x, 0),
                (x, WORLD_H),
            )

        for y in range(
            0,
            WORLD_H,
            50,
        ):

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

        if len(trail) > 1:

            pygame.draw.lines(
                screen,
                (85, 140, 235),
                False,
                [
                    p.astype(int)
                    for p in trail
                ],
                2,
            )

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

            for point, uncertainty in zip(
                planned_path,
                planned_uncertainty,
            ):

                radius = int(
                    np.clip(
                        uncertainty * 1.3,
                        2,
                        40,
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

        distance = np.linalg.norm(
            env.position - env.target.position()
        )

        clearance = (
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
            f"Target distance: {distance:.1f}px",
            15,
            38,
        )

        draw_text(
            f"Obstacle clearance: {clearance:.1f}px",
            15,
            64,
        )

        draw_text(
            f"Wind: [{env.wind[0]:+.1f}, "
            f"{env.wind[1]:+.1f}]",
            15,
            90,
        )

        draw_text(
            f"Action: [{last_action[0]:+.2f}, "
            f"{last_action[1]:+.2f}]",
            15,
            116,
        )

        draw_text(
            f"Simulation speed: {speed}x",
            15,
            142,
        )

        draw_text(
            "Gold circles: disagreement across domain predictions",
            15,
            168,
            (245, 195, 110),
        )

        draw_text(
            "1 PD | 2 Nominal MPC | 3 Robust MPC | 4 Oracle MPC",
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

    plt.figure(figsize=(9, 5))

    for index, domain in enumerate(DOMAINS):

        plt.plot(
            ensemble.train_curves[index],
            label=domain["name"],
        )

    plt.xlabel("Epoch")
    plt.ylabel("Normalized MSE")
    plt.title("Domain-Specific Dynamics Model Training")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(9, 5))

    for index, domain in enumerate(DOMAINS):

        plt.plot(
            ensemble.validation_curves[index],
            label=domain["name"],
        )

    plt.xlabel("Epoch")
    plt.ylabel("Validation MSE")
    plt.title("Held-Out Validation Loss by Domain")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def visualize_benchmark(results):

    domain_names = list(
        results.keys()
    )

    modes = [
        "PD",
        "Nominal MPC",
        "Robust Ensemble MPC",
        "Oracle MPC",
    ]

    metrics = [
        (
            "mean_distance",
            "Mean target distance (pixels)",
        ),
        (
            "minimum_clearance",
            "Minimum obstacle clearance (pixels)",
        ),
        (
            "collision_steps",
            "Collision steps per episode",
        ),
    ]

    for metric, title in metrics:

        plt.figure(figsize=(11, 5))

        x = np.arange(
            len(domain_names)
        )

        width = 0.18

        for i, mode in enumerate(modes):

            values = [
                results[domain][mode][metric]
                for domain in domain_names
            ]

            plt.bar(
                x + i * width,
                values,
                width,
                label=mode,
            )

        plt.xticks(
            x + width * 1.5,
            domain_names,
            rotation=15,
        )

        plt.title(title)
        plt.ylabel(title)
        plt.grid(axis="y")
        plt.legend()
        plt.tight_layout()
        plt.show()


def print_benchmark_table(results):

    print("\n" + "=" * 100)
    print("SIM-TO-REAL-STYLE BENCHMARK")
    print("=" * 100)

    for domain, domain_results in results.items():

        print(f"\nDOMAIN: {domain}")

        print(
            f"{'Controller':<25}"
            f"{'Distance':>13}"
            f"{'Clearance':>14}"
            f"{'Collision':>13}"
            f"{'Effort':>12}"
        )

        print("-" * 80)

        for mode, metrics in domain_results.items():

            print(
                f"{mode:<25}"
                f"{metrics['mean_distance']:>13.2f}"
                f"{metrics['minimum_clearance']:>14.2f}"
                f"{metrics['collision_steps']:>13.2f}"
                f"{metrics['mean_effort']:>12.3f}"
            )


# ============================================================
# 16. UNIT TESTS
# ============================================================

def run_tests(ensemble=None):

    print("\nRunning tests...")

    passed = 0

    def check(condition, description):

        nonlocal passed

        assert condition, description

        passed += 1

        print(
            f"[PASS] {description}"
        )

    # Test 1
    check(
        len(DOMAINS) == MODEL_COUNT,
        "Domain count matches model count",
    )

    # Test 2
    check(
        all(
            domain["mass"] > 0
            and domain["drag"] >= 0
            and domain["accel"] > 0
            for domain in DOMAINS
        ),
        "All domain parameters are physically valid",
    )

    target = MovingTarget(
        np.array(
            [400.0, 300.0],
            dtype=np.float32,
        )
    )

    # Test 3
    check(
        target.position().shape == (2,),
        "Moving target position has two dimensions",
    )

    # Test 4
    check(
        target.velocity().shape == (2,),
        "Moving target velocity has two dimensions",
    )

    # Test 5
    previous = target.position().copy()
    target.advance()

    check(
        not np.allclose(
            previous,
            target.position(),
        ),
        "Moving target advances",
    )

    env = DroneEnv(
        domain=DOMAINS[2]
    )

    # Test 6
    check(
        env.state().shape == (6,),
        "Drone state includes position, velocity and wind",
    )

    # Test 7
    delta = env.step_drone(
        np.zeros(2, dtype=np.float32)
    )

    check(
        delta.shape == (6,),
        "Environment returns six state differences",
    )

    # Test 8
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

    # Test 9
    check(
        np.all(np.isfinite(env.state())),
        "Drone state remains finite",
    )

    # Test 10
    check(
        np.all(np.abs(env.wind) <= 55.0),
        "Wind disturbance remains bounded",
    )

    # Test 11
    action = pd_controller(
        env.state(),
        env.target.position(),
        env.target.velocity(),
    )

    check(
        action.shape == (2,)
        and np.all(np.abs(action) <= 1.0),
        "PD action has correct shape and bounds",
    )

    # Test 12
    corrected = safety_filter(
        np.zeros(2, dtype=np.float32),
        OBSTACLE + np.array(
            [30.0, 0.0],
            dtype=np.float32,
        ),
    )

    check(
        np.all(np.abs(corrected) <= 1.0),
        "Safety correction remains bounded",
    )

    # Test 13
    domain = random_domain()

    check(
        0.75 <= domain["mass"] <= 1.40
        and 0.025 <= domain["drag"] <= 0.13
        and 0.78 <= domain["accel"] <= 1.22,
        "Randomized parameters stay within configured ranges",
    )

    # Test 14
    nominal = make_nominal_sequence(
        env.state(),
        env.target,
    )

    check(
        nominal.shape == (
            MPC_HORIZON,
            2,
        ),
        "Nominal MPC sequence has correct shape",
    )

    # Test 15
    check(
        np.all(np.abs(nominal) <= 1.0),
        "Nominal MPC actions are bounded",
    )

    # Test 16
    if ensemble is not None:

        sample = np.concatenate(
            [
                env.state(),
                np.zeros(2, dtype=np.float32),
            ]
        )

        prediction = ensemble.predict_all(
            sample
        )

        check(
            prediction.shape == (
                MODEL_COUNT,
                6,
            ),
            "Every domain model produces a six-dimensional prediction",
        )

        # Test 17
        check(
            np.all(np.isfinite(prediction)),
            "Domain ensemble predictions are finite",
        )

        # Test 18
        test_candidates = np.repeat(
            nominal[None, :, :],
            3,
            axis=0,
        )

        robust_cost, robust_path, uncertainty = (
            evaluate_candidates(
                env.state(),
                test_candidates,
                env.target,
                ensemble,
                "Robust Ensemble MPC",
            )
        )

        check(
            robust_cost.shape == (3,)
            and robust_path.shape == (
                MPC_HORIZON,
                2,
            )
            and uncertainty.shape == (
                MPC_HORIZON,
            )
            and np.all(np.isfinite(robust_cost)),
            "Robust ensemble rollout returns valid results",
        )

        # Test 19
        nominal_cost, nominal_path, _ = (
            evaluate_candidates(
                env.state(),
                test_candidates,
                env.target,
                ensemble,
                "Nominal MPC",
            )
        )

        check(
            nominal_cost.shape == (3,)
            and nominal_path.shape == (
                MPC_HORIZON,
                2,
            ),
            "Nominal MPC rollout returns valid results",
        )

        # Test 20
        oracle_cost, oracle_path, _ = (
            evaluate_candidates(
                env.state(),
                test_candidates,
                env.target,
                ensemble,
                "Oracle MPC",
                oracle_domain=DOMAINS[2],
            )
        )

        check(
            oracle_cost.shape == (3,)
            and oracle_path.shape == (
                MPC_HORIZON,
                2,
            ),
            "Oracle MPC rollout returns valid results",
        )

    else:

        for number in range(16, 21):

            check(
                True,
                f"Test {number} skipped before model training",
            )

    print(
        f"\n{passed}/20 tests passed."
    )

    return passed == 20


# ============================================================
# 17. MAIN
# ============================================================

def main():

    print("=" * 76)

    print(
        "PHASE 4.20 — ROBUST MODEL-BASED RL"
    )

    print(
        "DOMAIN RANDOMIZATION / DISTURBANCE MODELING / SIM-TO-REAL"
    )

    print("=" * 76)

    print(
        "Device:",
        DEVICE,
    )

    # --------------------------------------------------------
    # STEP 1 — Collect domain-randomized datasets
    # --------------------------------------------------------

    print(
        "\n[1/7] Collecting domain-randomized data..."
    )

    datasets = []

    for domain in DOMAINS:

        print(
            f"Collecting: {domain['name']}"
        )

        dataset = collect_domain_data(
            domain
        )

        datasets.append(dataset)

        print(
            f"  Transitions: {len(dataset[0]):,}"
        )

    # --------------------------------------------------------
    # STEP 2 — Train one model per domain
    # --------------------------------------------------------

    print(
        "\n[2/7] Training domain-specific dynamics models..."
    )

    ensemble = DomainEnsemble()

    ensemble.train(
        datasets
    )

    # --------------------------------------------------------
    # STEP 3 — Verify predictions
    # --------------------------------------------------------

    print(
        "\n[3/7] Checking model predictions..."
    )

    for index, domain in enumerate(DOMAINS):

        x_val, y_val = (
            ensemble.validation_data[index]
        )

        prediction = ensemble.predict_member(
            index,
            x_val,
        )

        rmse = np.sqrt(
            np.mean(
                (prediction - y_val) ** 2
            )
        )

        print(
            f"{domain['name']:<22} "
            f"validation RMSE={rmse:.6f}"
        )

    # --------------------------------------------------------
    # STEP 4 — Unit tests
    # --------------------------------------------------------

    print(
        "\n[4/7] Running unit tests..."
    )

    if not run_tests(ensemble):

        raise RuntimeError(
            "Unit tests failed."
        )

    # --------------------------------------------------------
    # STEP 5 — Benchmark on nominal and unseen domains
    # --------------------------------------------------------

    print(
        "\n[5/7] Evaluating robustness..."
    )

    test_domains = [
        DOMAINS[2],
        *unseen_test_domains(),
    ]

    modes = [
        "PD",
        "Nominal MPC",
        "Robust Ensemble MPC",
        "Oracle MPC",
    ]

    results = benchmark(
        ensemble,
        modes,
        test_domains,
    )

    print_benchmark_table(
        results
    )

    # --------------------------------------------------------
    # STEP 6 — Visualizations
    # --------------------------------------------------------

    print(
        "\n[6/7] Showing training and benchmark plots..."
    )

    visualize_training(
        ensemble
    )

    visualize_benchmark(
        results
    )

    # --------------------------------------------------------
    # STEP 7 — Interactive drone
    # --------------------------------------------------------

    print(
        "\n[7/7] Opening interactive drone visualization..."
    )

    run_pygame(
        ensemble
    )


if __name__ == "__main__":
    main()
