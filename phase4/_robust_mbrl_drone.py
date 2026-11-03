"""
File: phase4/robust_mbrl_drone.py

PHASE 4.20 — ROBUST MODEL-BASED RL (REVISED AND CORRECTED)

PURPOSE
-------
Train an ensemble of drone dynamics models across continuously randomized
physics, then use robust model-predictive control (MPC) to intercept a
moving target while avoiding an obstacle.

WHAT YOU LEARN
--------------

STEP 1 — Diverse dynamics data
    Randomize mass, drag, acceleration effectiveness, wind, initial state,
    and action sequences. Mix tracking, random, and temporally smooth actions.

STEP 2 — Episode-level validation
    Keep complete episodes out of training.

STEP 3 — Bootstrapped dynamics ensemble
    Train independent models on bootstrapped episode samples.
    Bootstrap duplicates are preserved.

STEP 4 — Robust MPC
    Evaluate candidate action sequences across:
      - multiple randomized physical domains
      - multiple independently propagated learned ensemble members

STEP 5 — CEM action-sequence optimization
    Iteratively sample temporally correlated action sequences.

STEP 6 — Moving-target interception
    Include target position and velocity in every MPC prediction step.

STEP 7 — Predictive obstacle avoidance
    Include obstacle clearance in planning and apply a heuristic
    velocity-aware safety filter.

STEP 8 — Paired evaluation
    Compare PD, nominal MPC, robust ensemble MPC, and oracle MPC.

STEP 9 — Interactive visualization
    Display the drone, moving target, obstacle, planned trajectory,
    and ensemble disagreement.

IMPORTANT LIMITATIONS
---------------------
This is a simplified 2D simulation, not flight-certified software.
Ensemble disagreement is not a calibrated probability distribution.
The obstacle filter does not guarantee collision avoidance.

INSTALL
-------
pip install numpy torch matplotlib pygame

RUN
---
python robust_mbrl_drone.py

CONTROLS
--------
1: PD
2: Nominal MPC
3: Robust Ensemble MPC
4: Oracle MPC
SPACE: Pause
R: Reset
+/-: Simulation speed
Mouse click: Move target trajectory center
ESC: Exit
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

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WORLD_W = 900
WORLD_H = 650
DT = 0.08

MAX_SPEED = 155.0
MAX_ACCEL = 125.0
DRONE_RADIUS = 13.0

OBSTACLE = np.array([450.0, 330.0], dtype=np.float32)
OBSTACLE_RADIUS = 65.0
SAFE_RADIUS = 115.0

TARGET_A = 145.0
TARGET_B = 85.0
TARGET_OMEGA = 0.42

WIND_DECAY = 0.88
WIND_NOISE_TRAIN = 13.0
WIND_NOISE_TEST = 18.0

MASS_RANGE = (0.70, 1.50)
DRAG_RANGE = (0.02, 0.16)
ACCEL_RANGE = (0.72, 1.28)

# Data and model training.
DATA_EPISODES = 120
STEPS_PER_EPISODE = 250

MODEL_COUNT = 5
HIDDEN = 192
EPOCHS = 45
BATCH_SIZE = 512
LEARNING_RATE = 7e-4
WEIGHT_DECAY = 1e-5

# MPC/CEM.
MPC_HORIZON = 18

CEM_CANDIDATES = 96
CEM_ELITES = 12
CEM_ITERATIONS = 4

CEM_INITIAL_STD = 0.55
CEM_MIN_STD = 0.10
CEM_SMOOTHING = 0.65

# Robust objective:
# J = mean(J) + risk_weight*std(J) + worst_weight*max(J)

ROBUST_RISK_WEIGHT = 0.55
ROBUST_WORST_WEIGHT = 0.25

# Cost weights.
W_DISTANCE = 0.045
W_VELOCITY = 0.018
W_OBSTACLE = 0.12
W_BOUNDARY = 0.08
W_ACTION = 0.018
W_ACTION_CHANGE = 0.025
W_TERMINAL_DISTANCE = 0.16
W_TERMINAL_VELOCITY = 0.06

BENCHMARK_EPISODES = 6
BENCHMARK_STEPS = 200

INPUT_SIZE = 11
OUTPUT_SIZE = 6

MODES = [
    "PD",
    "Nominal MPC",
    "Robust Ensemble MPC",
    "Oracle MPC",
]

NOMINAL_DOMAIN = {
    "name": "nominal",
    "mass": 1.0,
    "drag": 0.065,
    "accel": 1.0,
}

TRAIN_DOMAINS = [
    NOMINAL_DOMAIN,
    {"name": "light", "mass": 0.78, "drag": 0.045, "accel": 1.18},
    {"name": "heavy", "mass": 1.40, "drag": 0.105, "accel": 0.78},
    {"name": "strong_drag", "mass": 1.10, "drag": 0.15, "accel": 0.92},
    {"name": "combined_shift", "mass": 1.35, "drag": 0.135, "accel": 0.80},
]


def random_domain(name="randomized"):
    return {
        "name": name,
        "mass": float(np.random.uniform(*MASS_RANGE)),
        "drag": float(np.random.uniform(*DRAG_RANGE)),
        "accel": float(np.random.uniform(*ACCEL_RANGE)),
    }


def unseen_test_domains():
    return [
        {"name": "unseen_light", "mass": 0.72, "drag": 0.055, "accel": 1.20},
        {"name": "unseen_heavy", "mass": 1.46, "drag": 0.095, "accel": 0.75},
        {"name": "unseen_drag", "mass": 1.10, "drag": 0.155, "accel": 0.90},
        {"name": "unseen_combined", "mass": 1.40, "drag": 0.145, "accel": 0.76},
    ]


def domain_vector(domain):
    return np.array(
        [
            domain["mass"],
            domain["drag"],
            domain["accel"],
        ],
        dtype=np.float32,
    )


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

        self.phase = (
            self.phase + TARGET_OMEGA * DT
        ) % (2.0 * math.pi)


def target_at(target, step):

    phase = target.phase + TARGET_OMEGA * DT * step

    return target.position(phase)


def target_velocity_at(target, step):

    phase = target.phase + TARGET_OMEGA * DT * step

    return target.velocity(phase)


# ============================================================
# 2. DRONE ENVIRONMENT
# ============================================================

class DroneEnv:

    """
    State:
        [x, y, vx, vy, wind_x, wind_y]

    Action:
        [ax_normalized, ay_normalized]

    Physical acceleration:
        a = action * MAX_ACCEL * accel / mass - drag*v + wind

    Integration:
        v_next = v + a*DT
        p_next = p + v_next*DT

    Wind:
        wind_next = WIND_DECAY*wind + Gaussian disturbance
    """

    def __init__(self, domain=None, wind_noise=WIND_NOISE_TRAIN):

        self.domain = dict(domain or random_domain())

        self.wind_noise = float(wind_noise)

        self.target = MovingTarget(
            np.array([710.0, 300.0], dtype=np.float32)
        )

        self.reset()

    def reset(self, seed=None):

        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng()

        self.position = np.array(
            [125.0, 500.0],
            dtype=np.float32,
        )

        self.velocity = np.zeros(2, dtype=np.float32)

        self.wind = self.rng.normal(
            0.0,
            8.0,
            size=2,
        ).astype(np.float32)

        self.target.phase = float(
            self.rng.uniform(0.0, 2.0 * math.pi)
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

        action = np.clip(
            np.asarray(action, dtype=np.float32),
            -1.0,
            1.0,
        )

        old = self.state()

        acceleration = (
            action
            * MAX_ACCEL
            * self.domain["accel"]
            / self.domain["mass"]
            - self.domain["drag"] * self.velocity
            + self.wind
        )

        self.velocity += acceleration * DT

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

        self.wind = (
            WIND_DECAY * self.wind
            + self.rng.normal(
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

        return self.state() - old

    def step(self, action):

        self.step_drone(action)

        self.target.advance()

        return self.state()


# ============================================================
# 3. CONTROLLERS AND SAFETY FILTER
# ============================================================

def pd_controller(state, target_position, target_velocity):

    position = state[:2]
    velocity = state[2:4]

    error = target_position - position

    desired_velocity = target_velocity + 0.55 * error

    speed = np.linalg.norm(desired_velocity)

    if speed > MAX_SPEED:

        desired_velocity *= MAX_SPEED / (speed + 1e-8)

    acceleration = (
        2.3 * (desired_velocity - velocity)
        - state[4:6]
    )

    return np.clip(
        acceleration / MAX_ACCEL,
        -1.0,
        1.0,
    ).astype(np.float32)


def predictive_safety_filter(action, state):

    """
    Heuristic velocity-aware obstacle correction.

    This is not a formal control barrier function.
    """

    action = np.asarray(
        action,
        dtype=np.float32,
    ).copy()

    delta = state[:2] - OBSTACLE

    distance = np.linalg.norm(delta) + 1e-8

    normal = delta / distance

    clearance = distance - SAFE_RADIUS

    closing_speed = -float(
        np.dot(state[2:4], normal)
    )

    if clearance < 75.0:

        urgency = np.clip(
            (75.0 - clearance) / 75.0,
            0.0,
            1.0,
        )

        braking = max(0.0, closing_speed) / MAX_SPEED

        correction = urgency * (
            0.45 + 0.8 * braking
        )

        action += normal * correction

        inward_component = float(
            np.dot(action, normal)
        )

        if inward_component < 0.0:

            action -= normal * inward_component

    return np.clip(
        action,
        -1.0,
        1.0,
    ).astype(np.float32)


# ============================================================
# 4. DATA COLLECTION
# ============================================================

def smooth_random_action(
    previous,
    rho=0.82,
    scale=0.45,
):

    noise = np.random.normal(
        0.0,
        scale,
        size=2,
    )

    return np.clip(
        rho * previous + (1.0 - rho) * noise,
        -1.0,
        1.0,
    )


def collect_data():

    inputs = []
    targets = []
    episode_ids = []

    total_steps = DATA_EPISODES * STEPS_PER_EPISODE

    print(
        f"Collecting {DATA_EPISODES} episodes "
        f"({total_steps:,} transitions)."
    )

    for episode in range(DATA_EPISODES):

        domain = random_domain()

        env = DroneEnv(
            domain,
            WIND_NOISE_TRAIN,
        )

        state = env.reset(
            seed=SEED + 1000 + episode
        )

        env.target.center = np.array(
            [
                np.random.uniform(650.0, 760.0),
                np.random.uniform(250.0, 370.0),
            ],
            dtype=np.float32,
        )

        smooth_action = np.zeros(
            2,
            dtype=np.float32,
        )

        for step in range(STEPS_PER_EPISODE):

            teacher = pd_controller(
                state,
                env.target.position(),
                env.target.velocity(),
            )

            mode = np.random.rand()

            if mode < 0.42:

                action = (
                    teacher
                    + np.random.normal(0.0, 0.28, size=2)
                )

            elif mode < 0.68:

                action = np.random.uniform(
                    -1.0,
                    1.0,
                    size=2,
                )

            elif mode < 0.88:

                smooth_action = smooth_random_action(
                    smooth_action
                )

                action = smooth_action

            else:

                action = (
                    -np.sign(state[2:4])
                    * np.random.uniform(0.25, 1.0, 2)
                )

            action = np.clip(
                action,
                -1.0,
                1.0,
            ).astype(np.float32)

            action = predictive_safety_filter(
                action,
                state,
            )

            model_input = np.concatenate(
                [
                    state,
                    action,
                    domain_vector(domain),
                ]
            ).astype(np.float32)

            delta = env.step_drone(action)

            inputs.append(model_input)
            targets.append(delta)
            episode_ids.append(episode)

            env.target.advance()

            state = env.state()

    return (
        np.asarray(inputs, dtype=np.float32),
        np.asarray(targets, dtype=np.float32),
        np.asarray(episode_ids, dtype=np.int32),
    )


# ============================================================
# 5. BOOTSTRAPPED DYNAMICS ENSEMBLE
# ============================================================

class DynamicsModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(INPUT_SIZE, HIDDEN),
            nn.SiLU(),

            nn.Linear(HIDDEN, HIDDEN),
            nn.SiLU(),

            nn.Linear(HIDDEN, HIDDEN),
            nn.SiLU(),

            nn.Linear(HIDDEN, OUTPUT_SIZE),
        )

    def forward(self, x):

        return self.net(x)


class DynamicsEnsemble:

    def __init__(self):

        self.models = [
            DynamicsModel().to(DEVICE)
            for _ in range(MODEL_COUNT)
        ]

        self.x_mean = None
        self.x_std = None

        self.y_mean = None
        self.y_std = None

        self.train_curves = []
        self.val_curves = []

        self.validation = None

    def normalize_x(self, x):

        return (x - self.x_mean) / self.x_std

    def normalize_y(self, y):

        return (y - self.y_mean) / self.y_std

    def denormalize_y(self, y):

        return y * self.y_std + self.y_mean

    def train(self, x_np, y_np, episode_ids):

        x = torch.as_tensor(
            x_np,
            dtype=torch.float32,
            device=DEVICE,
        )

        y = torch.as_tensor(
            y_np,
            dtype=torch.float32,
            device=DEVICE,
        )

        groups = np.unique(episode_ids)

        rng = np.random.default_rng(SEED + 77)

        shuffled = rng.permutation(groups)

        split = max(
            1,
            int(0.85 * len(shuffled)),
        )

        train_groups = set(
            shuffled[:split].tolist()
        )

        val_groups = set(
            shuffled[split:].tolist()
        )

        train_mask_np = np.isin(
            episode_ids,
            list(train_groups),
        )

        val_mask_np = np.isin(
            episode_ids,
            list(val_groups),
        )

        train_idx = torch.as_tensor(
            np.flatnonzero(train_mask_np),
            dtype=torch.long,
            device=DEVICE,
        )

        val_idx = torch.as_tensor(
            np.flatnonzero(val_mask_np),
            dtype=torch.long,
            device=DEVICE,
        )

        if len(train_idx) == 0 or len(val_idx) == 0:

            raise RuntimeError(
                "Episode-level train/validation split is empty."
            )

        x_train_raw = x[train_idx]
        y_train_raw = y[train_idx]

        x_val_raw = x[val_idx]
        y_val_raw = y[val_idx]

        self.x_mean = x_train_raw.mean(0)

        self.x_std = x_train_raw.std(0).clamp_min(1e-4)

        self.y_mean = y_train_raw.mean(0)

        self.y_std = y_train_raw.std(0).clamp_min(1e-4)

        x_train = self.normalize_x(x_train_raw)
        y_train = self.normalize_y(y_train_raw)

        x_val = self.normalize_x(x_val_raw)
        y_val = self.normalize_y(y_val_raw)

        self.validation = (
            x_val_raw.detach().cpu().numpy(),
            y_val_raw.detach().cpu().numpy(),
        )

        train_episode_ids = episode_ids[train_mask_np]

        train_episode_groups = np.unique(
            train_episode_ids
        )

        for member, model in enumerate(self.models):

            print(
                f"\nTraining ensemble member "
                f"{member + 1}/{MODEL_COUNT}"
            )

            optimizer = optim.AdamW(
                model.parameters(),
                lr=LEARNING_RATE,
                weight_decay=WEIGHT_DECAY,
            )

            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=EPOCHS,
            )

            # ------------------------------------------------
            # TRUE EPISODE BOOTSTRAPPING
            # ------------------------------------------------
            # Sampling episodes with replacement must preserve
            # duplicate episode selections.
            #
            # np.isin() would remove duplicate bootstrap samples.
            # We explicitly concatenate each sampled episode's rows.

            sampled_groups = rng.choice(
                train_episode_groups,
                size=len(train_episode_groups),
                replace=True,
            )

            bootstrap_rows = np.concatenate(
                [
                    np.flatnonzero(
                        train_episode_ids == group
                    )
                    for group in sampled_groups
                ]
            )

            bootstrap_idx = torch.as_tensor(
                bootstrap_rows,
                dtype=torch.long,
                device=DEVICE,
            )

            member_x = x_train[bootstrap_idx]
            member_y = y_train[bootstrap_idx]

            train_history = []
            val_history = []

            for epoch in range(EPOCHS):

                model.train()

                order = torch.randperm(
                    len(member_x),
                    device=DEVICE,
                )

                batch_losses = []

                for start in range(
                    0,
                    len(order),
                    BATCH_SIZE,
                ):

                    idx = order[
                        start:start + BATCH_SIZE
                    ]

                    prediction = model(member_x[idx])

                    loss = nn.functional.smooth_l1_loss(
                        prediction,
                        member_y[idx],
                    )

                    optimizer.zero_grad(
                        set_to_none=True
                    )

                    loss.backward()

                    nn.utils.clip_grad_norm_(
                        model.parameters(),
                        5.0,
                    )

                    optimizer.step()

                    batch_losses.append(
                        float(loss.item())
                    )

                model.eval()

                with torch.no_grad():

                    val_loss = nn.functional.smooth_l1_loss(
                        model(x_val),
                        y_val,
                    ).item()

                train_history.append(
                    float(np.mean(batch_losses))
                )

                val_history.append(
                    float(val_loss)
                )

                scheduler.step()

                if (epoch + 1) % 15 == 0:

                    print(
                        f"  epoch {epoch + 1:3d}/{EPOCHS} "
                        f"train={train_history[-1]:.4f} "
                        f"val={val_loss:.4f}"
                    )

            self.train_curves.append(train_history)
            self.val_curves.append(val_history)

    # --------------------------------------------------------
    # CORRECTED ENSEMBLE PREDICTION
    # --------------------------------------------------------

    def predict_all(self, model_input):

        """
        Return:

            Single input:
                (MODEL_COUNT, OUTPUT_SIZE)

            Batch input:
                (MODEL_COUNT, BATCH_SIZE, OUTPUT_SIZE)

        Supports one-dimensional and two-dimensional inputs.
        """

        array = np.asarray(
            model_input,
            dtype=np.float32,
        )

        single = array.ndim == 1

        if single:

            array = array.reshape(1, -1)

        if array.ndim != 2:

            raise ValueError(
                "predict_all expects a 1D input or a 2D batch. "
                f"Received shape: {array.shape}"
            )

        if array.shape[-1] != INPUT_SIZE:

            raise ValueError(
                f"Expected input dimension {INPUT_SIZE}, "
                f"received {array.shape[-1]}"
            )

        x = torch.as_tensor(
            array,
            dtype=torch.float32,
            device=DEVICE,
        )

        x = self.normalize_x(x)

        predictions = []

        for model in self.models:

            model.eval()

            with torch.no_grad():

                y = self.denormalize_y(
                    model(x)
                )

            predictions.append(
                y.cpu().numpy()
            )

        result = np.stack(
            predictions,
            axis=0,
        )

        # Shape:
        # single input -> (MODEL_COUNT, 1, OUTPUT_SIZE)
        # batch input  -> (MODEL_COUNT, BATCH_SIZE, OUTPUT_SIZE)

        if single:

            return result[:, 0, :]

        return result


# ============================================================
# 6. TRANSITION MODELS
# ============================================================

def oracle_transition_batch(
    states,
    actions,
    domain,
):

    states = np.asarray(
        states,
        dtype=np.float32,
    )

    actions = np.clip(
        np.asarray(actions, dtype=np.float32),
        -1.0,
        1.0,
    )

    position = states[:, :2].copy()
    velocity = states[:, 2:4].copy()
    wind = states[:, 4:6].copy()

    acceleration = (
        actions
        * MAX_ACCEL
        * domain["accel"]
        / domain["mass"]
        - domain["drag"] * velocity
        + wind
    )

    velocity += acceleration * DT

    speeds = np.linalg.norm(
        velocity,
        axis=1,
        keepdims=True,
    )

    velocity *= np.minimum(
        1.0,
        MAX_SPEED / (speeds + 1e-8),
    )

    position += velocity * DT

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

    wind *= WIND_DECAY

    return np.concatenate(
        [
            position,
            velocity,
            wind,
        ],
        axis=1,
    ).astype(np.float32)


def learned_transition_ensemble(
    ensemble,
    states,
    actions,
    domain,
):

    """
    Predict next states for all ensemble members.

    Input:
        states:  (N, 6)
        actions: (N, 2)

    Output:
        (MODEL_COUNT, N, 6)
    """

    states = np.asarray(
        states,
        dtype=np.float32,
    )

    actions = np.asarray(
        actions,
        dtype=np.float32,
    )

    params = np.repeat(
        domain_vector(domain)[None, :],
        len(states),
        axis=0,
    )

    model_input = np.concatenate(
        [
            states,
            actions,
            params,
        ],
        axis=1,
    )

    deltas = ensemble.predict_all(model_input)

    next_states = states[None, :, :] + deltas

    next_states[:, :, 0] = np.clip(
        next_states[:, :, 0],
        DRONE_RADIUS,
        WORLD_W - DRONE_RADIUS,
    )

    next_states[:, :, 1] = np.clip(
        next_states[:, :, 1],
        DRONE_RADIUS,
        WORLD_H - DRONE_RADIUS,
    )

    velocity = next_states[:, :, 2:4]

    speeds = np.linalg.norm(
        velocity,
        axis=2,
        keepdims=True,
    )

    velocity *= np.minimum(
        1.0,
        MAX_SPEED / (speeds + 1e-8),
    )

    next_states[:, :, 4:6] = np.clip(
        next_states[:, :, 4:6],
        -55.0,
        55.0,
    )

    return next_states.astype(np.float32)


def learned_transition_independent_members(
    ensemble,
    states,
    actions,
    domain,
):

    """
    Independently propagate each ensemble member.

    Input:
        states:  (MODEL_COUNT, N, 6)
        actions: (N, 2)

    Output:
        (MODEL_COUNT, N, 6)

    IMPORTANT:
        Each member receives its own previous predicted state.
        This prevents ensemble rollouts from collapsing onto member 0.
    """

    states = np.asarray(
        states,
        dtype=np.float32,
    )

    actions = np.asarray(
        actions,
        dtype=np.float32,
    )

    member_count, batch_size, state_dim = states.shape

    if member_count != MODEL_COUNT:

        raise ValueError(
            f"Expected {MODEL_COUNT} ensemble members, "
            f"received {member_count}"
        )

    if state_dim != 6:

        raise ValueError(
            f"Expected state dimension 6, got {state_dim}"
        )

    if actions.shape != (batch_size, 2):

        raise ValueError(
            "Actions must have shape "
            f"({batch_size}, 2), got {actions.shape}"
        )

    flat_states = states.reshape(
        member_count * batch_size,
        6,
    )

    flat_actions = np.tile(
        actions,
        (member_count, 1),
    )

    flat_params = np.repeat(
        domain_vector(domain)[None, :],
        member_count * batch_size,
        axis=0,
    )

    model_input = np.concatenate(
        [
            flat_states,
            flat_actions,
            flat_params,
        ],
        axis=1,
    )

    # All models predict all flattened states.
    all_deltas = ensemble.predict_all(model_input)

    next_states = np.empty_like(states)

    # Select each model's prediction for its own rollout states.
    for member in range(member_count):

        start = member * batch_size
        end = start + batch_size

        member_delta = all_deltas[
            member,
            start:end,
        ]

        next_states[member] = (
            states[member] + member_delta
        )

    next_states[:, :, 0] = np.clip(
        next_states[:, :, 0],
        DRONE_RADIUS,
        WORLD_W - DRONE_RADIUS,
    )

    next_states[:, :, 1] = np.clip(
        next_states[:, :, 1],
        DRONE_RADIUS,
        WORLD_H - DRONE_RADIUS,
    )

    velocity = next_states[:, :, 2:4]

    speeds = np.linalg.norm(
        velocity,
        axis=2,
        keepdims=True,
    )

    velocity *= np.minimum(
        1.0,
        MAX_SPEED / (speeds + 1e-8),
    )

    next_states[:, :, 4:6] = np.clip(
        next_states[:, :, 4:6],
        -55.0,
        55.0,
    )

    return next_states.astype(np.float32)


# ============================================================
# 7. MPC COST
# ============================================================

def state_cost(
    state,
    target_position,
    target_velocity,
    action,
    previous_action,
):

    position = state[:2]
    velocity = state[2:4]

    distance = np.linalg.norm(
        position - target_position
    )

    velocity_error = np.linalg.norm(
        velocity - target_velocity
    )

    obstacle_distance = np.linalg.norm(
        position - OBSTACLE
    )

    safety_clearance = (
        obstacle_distance - SAFE_RADIUS
    )

    obstacle_penalty = max(
        0.0,
        35.0 - safety_clearance,
    ) ** 2

    boundary_margin = min(
        position[0] - DRONE_RADIUS,
        WORLD_W - DRONE_RADIUS - position[0],
        position[1] - DRONE_RADIUS,
        WORLD_H - DRONE_RADIUS - position[1],
    )

    boundary_penalty = max(
        0.0,
        45.0 - boundary_margin,
    ) ** 2

    effort = float(
        np.sum(action ** 2)
    )

    action_change = float(
        np.sum((action - previous_action) ** 2)
    )

    return (
        W_DISTANCE * distance
        + W_VELOCITY * velocity_error
        + W_OBSTACLE * obstacle_penalty
        + W_BOUNDARY * boundary_penalty
        + W_ACTION * effort
        + W_ACTION_CHANGE * action_change
    )


def make_nominal_sequence(
    state,
    target,
    domain=NOMINAL_DOMAIN,
):

    current = state.copy()

    sequence = []

    for t in range(
        1,
        MPC_HORIZON + 1,
    ):

        action = pd_controller(
            current,
            target_at(target, t),
            target_velocity_at(target, t),
        )

        action = predictive_safety_filter(
            action,
            current,
        )

        sequence.append(action)

        current = oracle_transition_batch(
            current[None, :],
            action[None, :],
            domain,
        )[0]

    return np.asarray(
        sequence,
        dtype=np.float32,
    )


# ============================================================
# 8. ROBUST CANDIDATE EVALUATION
# ============================================================

def evaluate_candidate_batch(
    initial_state,
    candidates,
    target,
    ensemble,
    mode,
    oracle_domain=None,
):

    """
    Evaluate candidate sequences across domains and models.

    Robust MPC:
        - Maintains independent rollout state for every model.
        - Evaluates all training domains.
        - Aggregates mean, standard deviation, and worst-case cost.
    """

    candidates = np.asarray(
        candidates,
        dtype=np.float32,
    )

    count = len(candidates)

    if mode == "Oracle MPC":

        if oracle_domain is None:

            raise ValueError(
                "Oracle MPC requires oracle_domain."
            )

        domains = [oracle_domain]

    elif mode == "Nominal MPC":

        domains = [NOMINAL_DOMAIN]

    else:

        domains = TRAIN_DOMAINS

    rollout_costs = []
    rollout_paths = []

    for domain in domains:

        if mode == "Oracle MPC":

            member_count = 1

        elif mode == "Nominal MPC":

            member_count = 1

        else:

            member_count = MODEL_COUNT

        states = np.repeat(
            initial_state[None, None, :],
            member_count * count,
            axis=0,
        ).reshape(
            member_count,
            count,
            6,
        ).astype(np.float32)

        costs = np.zeros(
            (member_count, count),
            dtype=np.float64,
        )

        paths = np.zeros(
            (
                member_count,
                count,
                MPC_HORIZON,
                2,
            ),
            dtype=np.float32,
        )

        previous_actions = np.zeros(
            (
                member_count,
                count,
                2,
            ),
            dtype=np.float32,
        )

        for t in range(MPC_HORIZON):

            actions = candidates[:, t, :]

            if mode == "Oracle MPC":

                next_states = oracle_transition_batch(
                    states[0],
                    actions,
                    domain,
                )[None, :, :]

            elif mode == "Nominal MPC":

                predictions = learned_transition_ensemble(
                    ensemble,
                    states[0],
                    actions,
                    domain,
                )

                next_states = np.mean(
                    predictions,
                    axis=0,
                    keepdims=True,
                )

            else:

                # Corrected:
                # Every ensemble member continues from its own state.
                next_states = learned_transition_independent_members(
                    ensemble,
                    states,
                    actions,
                    domain,
                )

            tp = target_at(
                target,
                t + 1,
            )

            tv = target_velocity_at(
                target,
                t + 1,
            )

            for member in range(member_count):

                for candidate_index in range(count):

                    costs[member, candidate_index] += state_cost(
                        next_states[member, candidate_index],
                        tp,
                        tv,
                        actions[candidate_index],
                        previous_actions[member, candidate_index],
                    )

            states = next_states

            paths[:, :, t, :] = states[:, :, :2]

            previous_actions[:] = actions[None, :, :]

        terminal_position = target_at(
            target,
            MPC_HORIZON,
        )

        terminal_velocity = target_velocity_at(
            target,
            MPC_HORIZON,
        )

        terminal_distance = np.linalg.norm(
            states[:, :, :2]
            - terminal_position[None, None, :],
            axis=2,
        )

        terminal_velocity_error = np.linalg.norm(
            states[:, :, 2:4]
            - terminal_velocity[None, None, :],
            axis=2,
        )

        costs += (
            W_TERMINAL_DISTANCE * terminal_distance
            + W_TERMINAL_VELOCITY * terminal_velocity_error
        )

        rollout_costs.append(
            costs.reshape(-1, count)
        )

        rollout_paths.append(
            paths.reshape(
                -1,
                count,
                MPC_HORIZON,
                2,
            )
        )

    all_costs = np.concatenate(
        rollout_costs,
        axis=0,
    )

    all_paths = np.concatenate(
        rollout_paths,
        axis=0,
    )

    if mode in (
        "Nominal MPC",
        "Oracle MPC",
    ):

        objective = all_costs.mean(axis=0)

    else:

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

        objective = (
            mean_cost
            + ROBUST_RISK_WEIGHT * std_cost
            + ROBUST_WORST_WEIGHT * worst_cost
        )

    best = int(
        np.argmin(objective)
    )

    best_path = np.mean(
        all_paths[:, best],
        axis=0,
    )

    path_disagreement = np.std(
        all_paths[:, best],
        axis=0,
    )

    uncertainty = np.linalg.norm(
        path_disagreement,
        axis=1,
    )

    return (
        objective,
        best_path,
        uncertainty,
    )


# ============================================================
# 9. CEM MPC PLANNER
# ============================================================

class CEMPlanner:

    def __init__(self):

        self.warm_mean = None

        self.last_path = np.zeros(
            (MPC_HORIZON, 2),
            dtype=np.float32,
        )

        self.last_uncertainty = np.zeros(
            MPC_HORIZON,
            dtype=np.float32,
        )

    def reset(self):

        self.warm_mean = None

        self.last_path = np.zeros(
            (MPC_HORIZON, 2),
            dtype=np.float32,
        )

        self.last_uncertainty = np.zeros(
            MPC_HORIZON,
            dtype=np.float32,
        )

    def initial_distribution(
        self,
        state,
        target,
    ):

        if self.warm_mean is None:

            mean = make_nominal_sequence(
                state,
                target,
            )

        else:

            mean = np.vstack(
                [
                    self.warm_mean[1:],
                    self.warm_mean[-1:],
                ]
            ).astype(np.float32)

        std = np.full(
            (MPC_HORIZON, 2),
            CEM_INITIAL_STD,
            dtype=np.float32,
        )

        return mean, std

    @staticmethod
    def correlated_noise():

        noise = np.random.normal(
            0.0,
            1.0,
            size=(
                CEM_CANDIDATES,
                MPC_HORIZON,
                2,
            ),
        ).astype(np.float32)

        for t in range(1, MPC_HORIZON):

            noise[:, t] = (
                0.72 * noise[:, t - 1]
                + math.sqrt(1.0 - 0.72 ** 2) * noise[:, t]
            )

        return noise

    def plan(
        self,
        state,
        target,
        ensemble,
        mode,
        oracle_domain=None,
    ):

        mean, std = self.initial_distribution(
            state,
            target,
        )

        best_sequence = mean.copy()

        best_path = self.last_path.copy()

        best_uncertainty = self.last_uncertainty.copy()

        best_cost = float("inf")

        for _ in range(CEM_ITERATIONS):

            noise = self.correlated_noise()

            candidates = (
                mean[None, :, :]
                + noise * std[None, :, :]
            )

            candidates = np.clip(
                candidates,
                -1.0,
                1.0,
            )

            candidates[0] = np.clip(
                mean,
                -1.0,
                1.0,
            )

            costs, path, uncertainty = evaluate_candidate_batch(
                state,
                candidates,
                target,
                ensemble,
                mode,
                oracle_domain,
            )

            order = np.argsort(costs)

            elite_indices = order[:CEM_ELITES]

            elites = candidates[elite_indices]

            if costs[order[0]] < best_cost:

                best_cost = float(
                    costs[order[0]]
                )

                best_sequence = candidates[
                    order[0]
                ].copy()

                best_path = path.copy()

                best_uncertainty = uncertainty.copy()

            new_mean = np.mean(
                elites,
                axis=0,
            )

            new_std = (
                np.std(elites, axis=0)
                + CEM_MIN_STD
            )

            mean = (
                CEM_SMOOTHING * mean
                + (1.0 - CEM_SMOOTHING) * new_mean
            )

            std = (
                CEM_SMOOTHING * std
                + (1.0 - CEM_SMOOTHING) * new_std
            )

            std = np.clip(
                std,
                CEM_MIN_STD,
                0.75,
            )

        self.warm_mean = best_sequence.copy()

        self.last_path = best_path.copy()

        self.last_uncertainty = best_uncertainty.copy()

        return (
            best_sequence[0].copy(),
            best_path.copy(),
            best_uncertainty.copy(),
        )


# ============================================================
# 10. BENCHMARK
# ============================================================

def run_episode(
    ensemble,
    mode,
    domain,
    seed,
    steps=BENCHMARK_STEPS,
):

    env = DroneEnv(
        domain,
        WIND_NOISE_TEST,
    )

    env.reset(seed=seed)

    planner = CEMPlanner()

    distances = []
    physical_clearances = []
    safety_clearances = []
    efforts = []

    physical_collision_steps = 0
    safety_violation_steps = 0

    for _ in range(steps):

        state = env.state()

        if mode == "PD":

            action = pd_controller(
                state,
                env.target.position(),
                env.target.velocity(),
            )

        else:

            action, _, _ = planner.plan(
                state,
                env.target,
                ensemble,
                mode,
                oracle_domain=domain,
            )

        action = predictive_safety_filter(
            action,
            state,
        )

        env.step(action)

        distance = np.linalg.norm(
            env.position - env.target.position()
        )

        obstacle_distance = np.linalg.norm(
            env.position - OBSTACLE
        )

        physical_clearance = (
            obstacle_distance - OBSTACLE_RADIUS
        )

        safety_clearance = (
            obstacle_distance - SAFE_RADIUS
        )

        physical_collision_steps += int(
            physical_clearance < 0.0
        )

        safety_violation_steps += int(
            safety_clearance < 0.0
        )

        distances.append(distance)

        physical_clearances.append(
            physical_clearance
        )

        safety_clearances.append(
            safety_clearance
        )

        efforts.append(
            np.linalg.norm(action)
        )

    return {
        "mean_distance": float(np.mean(distances)),
        "terminal_distance": float(distances[-1]),
        "minimum_physical_clearance": float(
            np.min(physical_clearances)
        ),
        "minimum_safety_clearance": float(
            np.min(safety_clearances)
        ),
        "physical_collision_steps": float(
            physical_collision_steps
        ),
        "safety_violation_steps": float(
            safety_violation_steps
        ),
        "mean_effort": float(np.mean(efforts)),
    }


def benchmark(ensemble):

    test_domains = [
        NOMINAL_DOMAIN,
        *unseen_test_domains(),
    ]

    results = {}

    for domain in test_domains:

        print(
            f"\nTest domain: {domain['name']}"
        )

        results[domain["name"]] = {}

        for mode in MODES:

            episode_results = []

            for episode in range(BENCHMARK_EPISODES):

                seed = 10000 + episode

                episode_results.append(
                    run_episode(
                        ensemble,
                        mode,
                        domain,
                        seed=seed,
                    )
                )

            results[domain["name"]][mode] = {
                key: float(
                    np.mean(
                        [
                            r[key]
                            for r in episode_results
                        ]
                    )
                )
                for key in episode_results[0]
            }

            m = results[domain["name"]][mode]

            print(
                f"  {mode:<24} "
                f"distance={m['mean_distance']:.1f} "
                f"terminal={m['terminal_distance']:.1f} "
                f"physical_clearance="
                f"{m['minimum_physical_clearance']:.1f} "
                f"safety_clearance="
                f"{m['minimum_safety_clearance']:.1f} "
                f"collisions="
                f"{m['physical_collision_steps']:.1f}"
            )

    return results


def print_benchmark_table(results):

    print("\n" + "=" * 125)

    print("PAIRED ROBUST-MBRL BENCHMARK")

    print("=" * 125)

    for domain, values in results.items():

        print(f"\nDOMAIN: {domain}")

        print(
            f"{'Controller':<24}"
            f"{'Mean dist':>11}"
            f"{'Final dist':>12}"
            f"{'Physical clr':>14}"
            f"{'Safety clr':>12}"
            f"{'Collision':>11}"
            f"{'Safety viol.':>13}"
            f"{'Effort':>9}"
        )

        print("-" * 116)

        for mode, m in values.items():

            print(
                f"{mode:<24}"
                f"{m['mean_distance']:>11.1f}"
                f"{m['terminal_distance']:>12.1f}"
                f"{m['minimum_physical_clearance']:>14.1f}"
                f"{m['minimum_safety_clearance']:>12.1f}"
                f"{m['physical_collision_steps']:>11.1f}"
                f"{m['safety_violation_steps']:>13.1f}"
                f"{m['mean_effort']:>9.3f}"
            )


# ============================================================
# 11. VALIDATION REPORT
# ============================================================

def validation_report(ensemble):

    x_val, y_val = ensemble.validation

    limit = min(
        10000,
        len(x_val),
    )

    predictions = ensemble.predict_all(
        x_val[:limit]
    )

    mean_prediction = np.mean(
        predictions,
        axis=0,
    )

    dimension_names = [
        "delta_x",
        "delta_y",
        "delta_vx",
        "delta_vy",
        "delta_wind_x",
        "delta_wind_y",
    ]

    print("\nHeld-out episode validation:")

    print(
        "Ensemble-mean RMSE by output dimension:"
    )

    errors = (
        mean_prediction - y_val[:limit]
    )

    for i, name in enumerate(dimension_names):

        rmse = float(
            np.sqrt(
                np.mean(errors[:, i] ** 2)
            )
        )

        baseline = float(
            np.sqrt(
                np.mean(y_val[:limit, i] ** 2)
            )
        )

        print(
            f"  {name:<14} "
            f"RMSE={rmse:9.4f} "
            f"zero-predictor baseline={baseline:9.4f}"
        )

    overall_rmse = float(
        np.sqrt(
            np.mean(errors ** 2)
        )
    )

    print(
        f"Overall ensemble-mean RMSE: {overall_rmse:.5f}"
    )

    member_rmses = []

    for i in range(MODEL_COUNT):

        member_error = (
            predictions[i] - y_val[:limit]
        )

        member_rmse = float(
            np.sqrt(
                np.mean(member_error ** 2)
            )
        )

        member_rmses.append(member_rmse)

        print(
            f"  Member {i + 1} RMSE: {member_rmse:.5f}"
        )

    return overall_rmse, member_rmses


# ============================================================
# 12. VISUALIZATION
# ============================================================

def visualize_training(ensemble):

    plt.figure(figsize=(10, 5))

    for i, curve in enumerate(
        ensemble.train_curves
    ):

        plt.plot(
            curve,
            label=f"Member {i + 1}",
        )

    plt.title(
        "Dynamics Ensemble — Training Loss"
    )

    plt.xlabel("Epoch")

    plt.ylabel(
        "Normalized Smooth-L1 Loss"
    )

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    plt.show()

    plt.figure(figsize=(10, 5))

    for i, curve in enumerate(
        ensemble.val_curves
    ):

        plt.plot(
            curve,
            label=f"Member {i + 1}",
        )

    plt.title(
        "Dynamics Ensemble — Episode-Held-Out Validation Loss"
    )

    plt.xlabel("Epoch")

    plt.ylabel(
        "Normalized Smooth-L1 Loss"
    )

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    plt.show()


def visualize_benchmark(results):

    metrics = [
        (
            "mean_distance",
            "Mean target distance",
        ),
        (
            "minimum_physical_clearance",
            "Minimum physical-obstacle clearance",
        ),
        (
            "safety_violation_steps",
            "Safety-boundary violation steps",
        ),
    ]

    modes = list(
        next(iter(results.values())).keys()
    )

    domains = list(results.keys())

    for metric, title in metrics:

        plt.figure(figsize=(13, 5))

        x = np.arange(len(domains))

        width = 0.18

        for i, mode in enumerate(modes):

            values = [
                results[d][mode][metric]
                for d in domains
            ]

            plt.bar(
                x + i * width,
                values,
                width,
                label=mode,
            )

        plt.xticks(
            x + width * 1.5,
            domains,
            rotation=15,
        )

        plt.title(title)

        plt.ylabel(title)

        plt.grid(axis="y")

        plt.legend()

        plt.tight_layout()

        plt.show()


# ============================================================
# 13. UNIT TESTS
# ============================================================

def run_tests(ensemble):

    print("\nRunning unit tests...")

    passed = 0
    total = 0

    failures = []

    def check(condition, description):

        nonlocal passed, total

        total += 1

        if bool(condition):

            passed += 1

            print(
                f"[PASS] {description}"
            )

        else:

            failures.append(description)

            print(
                f"[FAIL] {description}"
            )

    env = DroneEnv(NOMINAL_DOMAIN)

    state = env.reset(seed=123)

    check(
        state.shape == (6,),
        "State has six dimensions",
    )

    check(
        np.all(np.isfinite(state)),
        "Initial state is finite",
    )

    delta = env.step_drone(
        np.zeros(2, dtype=np.float32)
    )

    check(
        delta.shape == (6,),
        "Transition delta has six dimensions",
    )

    check(
        np.all(np.isfinite(delta)),
        "Transition delta is finite",
    )

    check(
        np.linalg.norm(env.velocity) <= MAX_SPEED + 1e-4,
        "Drone speed is bounded",
    )

    check(
        np.all(np.abs(env.wind) <= 55.0),
        "Wind is bounded",
    )

    previous_target = (
        env.target.position().copy()
    )

    env.target.advance()

    check(
        not np.allclose(
            previous_target,
            env.target.position(),
        ),
        "Moving target advances",
    )

    action = pd_controller(
        env.state(),
        env.target.position(),
        env.target.velocity(),
    )

    check(
        action.shape == (2,)
        and np.all(np.abs(action) <= 1.0),
        "PD action is bounded",
    )

    safe_action = predictive_safety_filter(
        action,
        env.state(),
    )

    check(
        safe_action.shape == (2,)
        and np.all(np.abs(safe_action) <= 1.0),
        "Safety-filter output is bounded",
    )

    params = domain_vector(
        random_domain()
    )

    check(
        params.shape == (3,)
        and np.all(np.isfinite(params)),
        "Randomized domain vector is valid",
    )

    sample_input = np.concatenate(
        [
            env.state(),
            np.zeros(2),
            domain_vector(NOMINAL_DOMAIN),
        ]
    ).astype(np.float32)

    # --------------------------------------------------------
    # SINGLE-INPUT ENSEMBLE TEST
    # --------------------------------------------------------

    predictions = ensemble.predict_all(
        sample_input
    )

    check(
        predictions.shape == (MODEL_COUNT, OUTPUT_SIZE),
        "Single-input ensemble prediction has correct shape",
    )

    check(
        np.all(np.isfinite(predictions)),
        "Single-input predictions are finite",
    )

    # --------------------------------------------------------
    # BATCH ENSEMBLE TEST
    # --------------------------------------------------------

    batch_inputs = np.repeat(
        sample_input[None, :],
        4,
        axis=0,
    )

    batch_predictions = ensemble.predict_all(
        batch_inputs
    )

    check(
        batch_predictions.shape == (
            MODEL_COUNT,
            4,
            OUTPUT_SIZE,
        ),
        "Batched ensemble prediction has correct shape",
    )

    # --------------------------------------------------------
    # VALIDATION-BATCH SHAPE TEST
    # --------------------------------------------------------

    large_batch = np.repeat(
        sample_input[None, :],
        100,
        axis=0,
    )

    large_predictions = ensemble.predict_all(
        large_batch
    )

    check(
        large_predictions.shape == (
            MODEL_COUNT,
            100,
            OUTPUT_SIZE,
        ),
        "Large validation batch has correct shape",
    )

    # --------------------------------------------------------
    # NOMINAL SEQUENCE TEST
    # --------------------------------------------------------

    nominal = make_nominal_sequence(
        env.state(),
        env.target,
    )

    check(
        nominal.shape == (
            MPC_HORIZON,
            2,
        ),
        "Nominal action sequence has correct shape",
    )

    check(
        np.all(np.abs(nominal) <= 1.0),
        "Nominal action sequence is bounded",
    )

    # --------------------------------------------------------
    # ROBUST MPC TEST
    # --------------------------------------------------------

    candidates = np.repeat(
        nominal[None, :, :],
        3,
        axis=0,
    )

    costs, path, uncertainty = evaluate_candidate_batch(
        env.state(),
        candidates,
        env.target,
        ensemble,
        "Robust Ensemble MPC",
    )

    check(
        costs.shape == (3,),
        "Robust MPC returns one cost per candidate",
    )

    check(
        path.shape == (
            MPC_HORIZON,
            2,
        ),
        "Robust MPC returns a full predicted path",
    )

    check(
        uncertainty.shape == (
            MPC_HORIZON,
        ),
        "Robust MPC returns horizon uncertainty",
    )

    check(
        np.all(np.isfinite(costs))
        and np.all(np.isfinite(path))
        and np.all(np.isfinite(uncertainty)),
        "Robust MPC outputs are finite",
    )

    # --------------------------------------------------------
    # INDEPENDENT MEMBER ROLLOUT TEST
    # --------------------------------------------------------

    initial_states = np.repeat(
        env.state()[None, None, :],
        MODEL_COUNT * 3,
        axis=0,
    ).reshape(
        MODEL_COUNT,
        3,
        6,
    )

    test_actions = np.zeros(
        (3, 2),
        dtype=np.float32,
    )

    independent_predictions = (
        learned_transition_independent_members(
            ensemble,
            initial_states,
            test_actions,
            NOMINAL_DOMAIN,
        )
    )

    check(
        independent_predictions.shape == (
            MODEL_COUNT,
            3,
            6,
        ),
        "Independent ensemble rollouts have correct shape",
    )

    check(
        np.all(np.isfinite(independent_predictions)),
        "Independent ensemble rollouts are finite",
    )

    # --------------------------------------------------------
    # CEM TEST
    # --------------------------------------------------------

    planner = CEMPlanner()

    planned_action, planned_path, planned_uncertainty = (
        planner.plan(
            env.state(),
            env.target,
            ensemble,
            "Robust Ensemble MPC",
        )
    )

    check(
        planned_action.shape == (2,)
        and np.all(np.abs(planned_action) <= 1.0),
        "CEM planner returns a bounded action",
    )

    check(
        planned_path.shape == (
            MPC_HORIZON,
            2,
        ),
        "CEM planner returns a predicted trajectory",
    )

    check(
        planned_uncertainty.shape == (
            MPC_HORIZON,
        ),
        "CEM planner returns uncertainty values",
    )

    # --------------------------------------------------------
    # REPRODUCIBILITY TEST
    # --------------------------------------------------------

    env_a = DroneEnv(
        NOMINAL_DOMAIN,
        WIND_NOISE_TEST,
    )

    env_b = DroneEnv(
        NOMINAL_DOMAIN,
        WIND_NOISE_TEST,
    )

    state_a = env_a.reset(seed=999)
    state_b = env_b.reset(seed=999)

    check(
        np.allclose(state_a, state_b),
        "Same reset seed produces identical initial drone state",
    )

    check(
        np.isclose(
            env_a.target.phase,
            env_b.target.phase,
        ),
        "Same reset seed produces identical target phase",
    )

    print(
        f"\n{passed}/{total} tests passed."
    )

    if failures:

        print("Failed tests:")

        for failure in failures:

            print(" -", failure)

        return False

    print(
        "[SUCCESS] All unit tests passed."
    )

    return True


# ============================================================
# 14. PYGAME VISUALIZATION
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
        "Phase 4.20 — Robust MBRL Quadcopter"
    )

    clock = pygame.time.Clock()

    font = pygame.font.SysFont(
        "consolas",
        16,
    )

    domain = dict(NOMINAL_DOMAIN)

    env = DroneEnv(
        domain,
        WIND_NOISE_TEST,
    )

    env.reset(seed=2026)

    planner = CEMPlanner()

    mode_index = 2

    paused = False

    speed = 1

    trail = []

    planned_path = []

    uncertainty = np.zeros(0)

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

    def draw_quadcopter(position, velocity):

        center = np.asarray(
            position,
            dtype=np.float32,
        )

        speed_now = np.linalg.norm(
            velocity
        )

        if speed_now > 1.0:

            direction = velocity / speed_now

        else:

            direction = np.array(
                [1.0, 0.0],
                dtype=np.float32,
            )

        side = np.array(
            [-direction[1], direction[0]],
            dtype=np.float32,
        )

        for forward in (-1.0, 1.0):

            for lateral in (-1.0, 1.0):

                endpoint = (
                    center
                    + forward * direction * 18.0
                    + lateral * side * 18.0
                )

                pygame.draw.line(
                    screen,
                    (215, 225, 238),
                    center.astype(int),
                    endpoint.astype(int),
                    5,
                )

                pygame.draw.circle(
                    screen,
                    (90, 195, 245),
                    endpoint.astype(int),
                    8,
                    2,
                )

        pygame.draw.circle(
            screen,
            (245, 245, 245),
            center.astype(int),
            9,
        )

        pygame.draw.circle(
            screen,
            (255, 190, 60),
            (center + direction * 23.0).astype(int),
            4,
        )

    def draw_target():

        points = []

        for i in range(120):

            phase = (
                env.target.phase
                + i * 2.0 * math.pi / 120
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
            (position + velocity * 0.2).astype(int),
            3,
        )

        pygame.draw.circle(
            screen,
            (80, 255, 130),
            env.target.center.astype(int),
            5,
            1,
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

                    planner.reset()

                    planned_path = []

                    uncertainty = np.zeros(0)

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

                    if (
                        np.linalg.norm(click - OBSTACLE)
                        > SAFE_RADIUS + 30
                        and 180 < click[0] < WORLD_W - 180
                        and 130 < click[1] < WORLD_H - 130
                    ):

                        env.target.center = click

                        planner.reset()

        mode = MODES[mode_index]

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

                    uncertainty = np.zeros(0)

                else:

                    action, planned_path, uncertainty = planner.plan(
                        state,
                        env.target,
                        ensemble,
                        mode,
                        oracle_domain=domain,
                    )

                action = predictive_safety_filter(
                    action,
                    state,
                )

                last_action = action.copy()

                env.step(action)

                trail.append(
                    env.position.copy()
                )

                if len(trail) > 180:

                    trail.pop(0)

        screen.fill(
            (18, 24, 34)
        )

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

            for point, spread in zip(
                planned_path,
                uncertainty,
            ):

                radius = int(
                    np.clip(
                        spread * 1.2,
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

        draw_quadcopter(
            env.position,
            env.velocity,
        )

        target_distance = np.linalg.norm(
            env.position - env.target.position()
        )

        obstacle_distance = np.linalg.norm(
            env.position - OBSTACLE
        )

        physical_clearance = (
            obstacle_distance - OBSTACLE_RADIUS
        )

        safety_clearance = (
            obstacle_distance - SAFE_RADIUS
        )

        draw_text(
            f"Mode: {mode}",
            15,
            12,
        )

        draw_text(
            f"Target distance: {target_distance:.1f}px",
            15,
            38,
        )

        draw_text(
            f"Physical clearance: {physical_clearance:.1f}px",
            15,
            64,
        )

        draw_text(
            f"Safety clearance: {safety_clearance:.1f}px",
            15,
            90,
        )

        draw_text(
            f"Wind: [{env.wind[0]:+.1f}, {env.wind[1]:+.1f}]",
            15,
            116,
        )

        draw_text(
            f"Action: [{last_action[0]:+.2f}, {last_action[1]:+.2f}]",
            15,
            142,
        )

        draw_text(
            f"Simulation speed: {speed}x",
            15,
            168,
        )

        draw_text(
            "Gold rings: ensemble trajectory disagreement",
            15,
            194,
            (245, 195, 110),
        )

        draw_text(
            "1 PD | 2 Nominal MPC | 3 Robust MPC | 4 Oracle MPC",
            15,
            WORLD_H - 75,
        )

        draw_text(
            "SPACE pause | R reset | +/- speed | click orbit center | ESC exit",
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
# 15. MAIN
# ============================================================

def main():

    print("=" * 80)

    print(
        "PHASE 4.20 — ROBUST MODEL-BASED RL"
    )

    print("=" * 80)

    print(
        "Device:",
        DEVICE,
    )

    print(
        f"Training episodes: {DATA_EPISODES} | "
        f"Transitions: "
        f"{DATA_EPISODES * STEPS_PER_EPISODE:,}"
    )

    print(
        f"Ensemble members: {MODEL_COUNT} | "
        f"CEM candidates: {CEM_CANDIDATES} | "
        f"Horizon: {MPC_HORIZON}"
    )

    # STEP 1
    print(
        "\n[1/7] Collecting randomized dynamics data..."
    )

    x, y, episode_ids = collect_data()

    print(
        f"Input shape: {x.shape}"
    )

    print(
        f"Target shape: {y.shape}"
    )

    print(
        f"Unique episodes: "
        f"{len(np.unique(episode_ids))}"
    )

    # STEP 2
    print(
        "\n[2/7] Training bootstrapped dynamics ensemble..."
    )

    ensemble = DynamicsEnsemble()

    ensemble.train(
        x,
        y,
        episode_ids,
    )

    # STEP 3
    print(
        "\n[3/7] Evaluating episode-held-out predictions..."
    )

    validation_report(ensemble)

    # STEP 4
    print(
        "\n[4/7] Running unit tests..."
    )

    if not run_tests(ensemble):

        raise RuntimeError(
            "One or more unit tests failed."
        )

    # STEP 5
    print(
        "\n[5/7] Running paired robustness benchmark..."
    )

    results = benchmark(ensemble)

    print_benchmark_table(results)

    # STEP 6
    print(
        "\n[6/7] Showing training and benchmark plots..."
    )

    visualize_training(ensemble)

    visualize_benchmark(results)

    # STEP 7
    print(
        "\n[7/7] Opening interactive quadcopter visualization..."
    )

    run_pygame(ensemble)


if __name__ == "__main__":

    main()