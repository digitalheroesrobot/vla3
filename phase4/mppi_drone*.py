"""
File: phase4/mppi_drone.py

PHASE 4.27 — HIGH-PERFORMANCE MODEL PREDICTIVE PATH
INTEGRAL CONTROL (MPPI)

FIX
---
1. Fixed missing environment-reset unit test.
2. Corrected test counting: 24 checks are now executed.
3. Preserved original MPPI architecture and simulation settings.

CONTROLLERS
-----------
1. PD
2. Random Shooting MPC
3. Learned MPPI
4. Oracle MPPI

CONTROLS
--------
1-4       Select controller
SPACE     Pause/resume
R         Reset
+/-       Simulation speed
UP/DOWN   Change planning quality
Mouse     Move target trajectory center
ESC       Exit

Educational 2D simulation. Not a flight-certified controller.
"""

from __future__ import annotations

import math
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# ============================================================
# 0. CONFIGURATION
# ============================================================

SEED = 427

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

if DEVICE.type == "cpu":
    torch.set_num_threads(2)

WORLD_W = 900
WORLD_H = 650

DT = 0.08

MAX_SPEED = 145.0
MAX_ACCEL = 120.0

DRONE_RADIUS = 12.0

OBSTACLE = np.array(
    [450.0, 330.0],
    dtype=np.float64,
)

OBSTACLE_RADIUS = 65.0
SAFE_RADIUS = 115.0

TARGET_A = 165.0
TARGET_B = 105.0
TARGET_OMEGA = 0.55

MOTOR_GAIN = 0.72
LINEAR_DRAG = 0.115
QUADRATIC_DRAG = 0.00032

WIND = np.array(
    [5.0, -3.5],
    dtype=np.float64,
)

STATE_DIM = 4
INPUT_DIM = 6

DATA_STEPS = 6500
TRAIN_EPOCHS = 120
BATCH_SIZE = 256
LEARNING_RATE = 0.001

HORIZON = 16
SAMPLE_COUNT = 180
MPPI_ITERATIONS = 3

INTERACTIVE_HORIZON = 16
INTERACTIVE_SAMPLES = 100
INTERACTIVE_ITERATIONS = 2

NOISE_STD = np.array(
    [0.34, 0.34],
    dtype=np.float64,
)

LAMBDA = 10.0

ACTION_SMOOTHING = 0.30
NOMINAL_BLEND = 0.85

BENCHMARK_EPISODES = 3
BENCHMARK_STEPS = 180

RENDER_FPS = 60
TRAIL_LENGTH = 350


# ============================================================
# 1. MOVING TARGET
# ============================================================

class MovingTarget:

    def __init__(self, center):

        self.center = np.asarray(
            center,
            dtype=np.float64,
        )

        self.phase = 0.0

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
            ]
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
            ]
        )

    def advance(self):

        self.phase += TARGET_OMEGA * DT


# ============================================================
# 2. DRONE DYNAMICS
# ============================================================

def clip_state(state):

    state = np.asarray(
        state,
        dtype=np.float64,
    ).copy()

    speed = np.linalg.norm(
        state[2:4]
    )

    if speed > MAX_SPEED:

        state[2:4] *= (
            MAX_SPEED / (speed + 1e-12)
        )

    state[0] = np.clip(
        state[0],
        DRONE_RADIUS,
        WORLD_W - DRONE_RADIUS,
    )

    state[1] = np.clip(
        state[1],
        DRONE_RADIUS,
        WORLD_H - DRONE_RADIUS,
    )

    return state


def clip_states_batch(states):

    states = np.asarray(
        states,
        dtype=np.float64,
    ).copy()

    speeds = np.linalg.norm(
        states[:, 2:4],
        axis=1,
        keepdims=True,
    )

    scale = np.minimum(
        1.0,
        MAX_SPEED / (speeds + 1e-12),
    )

    states[:, 2:4] *= scale

    states[:, 0] = np.clip(
        states[:, 0],
        DRONE_RADIUS,
        WORLD_W - DRONE_RADIUS,
    )

    states[:, 1] = np.clip(
        states[:, 1],
        DRONE_RADIUS,
        WORLD_H - DRONE_RADIUS,
    )

    return states


def true_transition(state, action):

    state = np.asarray(
        state,
        dtype=np.float64,
    )

    action = np.clip(
        np.asarray(action, dtype=np.float64),
        -1.0,
        1.0,
    )

    velocity = state[2:4]

    acceleration = (
        MOTOR_GAIN * MAX_ACCEL * action
        - LINEAR_DRAG * velocity
        - QUADRATIC_DRAG
        * np.abs(velocity)
        * velocity
        + WIND
    )

    next_velocity = (
        velocity + acceleration * DT
    )

    next_position = (
        state[:2] + next_velocity * DT
    )

    return clip_state(
        np.concatenate(
            [
                next_position,
                next_velocity,
            ]
        )
    )


def true_transition_batch(states, actions):

    actions = np.clip(
        actions,
        -1.0,
        1.0,
    )

    velocity = states[:, 2:4]

    acceleration = (
        MOTOR_GAIN * MAX_ACCEL * actions
        - LINEAR_DRAG * velocity
        - QUADRATIC_DRAG
        * np.abs(velocity)
        * velocity
        + WIND
    )

    next_velocity = (
        velocity + acceleration * DT
    )

    next_position = (
        states[:, :2]
        + next_velocity * DT
    )

    next_states = np.concatenate(
        [
            next_position,
            next_velocity,
        ],
        axis=1,
    )

    return clip_states_batch(
        next_states
    )


# ============================================================
# 3. ENVIRONMENT
# ============================================================

class DroneEnv:

    def __init__(self):

        self.target = MovingTarget(
            [710.0, 315.0]
        )

        self.reset()

    def reset(self):

        self.state = np.array(
            [
                110.0,
                500.0,
                0.0,
                0.0,
            ],
            dtype=np.float64,
        )

        self.target.phase = np.random.uniform(
            0.0,
            2.0 * math.pi,
        )

        return self.state.copy()

    @property
    def position(self):

        return self.state[:2].copy()

    @property
    def velocity(self):

        return self.state[2:4].copy()

    def step(self, action):

        self.state = true_transition(
            self.state,
            action,
        )

        self.target.advance()

        return self.state.copy()


# ============================================================
# 4. PD CONTROLLER
# ============================================================

def pd_controller(
    state,
    target_position,
    target_velocity,
):

    position = state[:2]
    velocity = state[2:4]

    desired_velocity = (
        0.75 * (target_position - position)
        + target_velocity
    )

    speed = np.linalg.norm(
        desired_velocity
    )

    if speed > MAX_SPEED:

        desired_velocity *= (
            MAX_SPEED / (speed + 1e-12)
        )

    desired_acceleration = (
        1.5 * (desired_velocity - velocity)
    )

    action = (
        desired_acceleration
        + LINEAR_DRAG * velocity
        + QUADRATIC_DRAG
        * np.abs(velocity)
        * velocity
        - WIND
    ) / (
        MOTOR_GAIN * MAX_ACCEL
    )

    return np.clip(
        action,
        -1.0,
        1.0,
    )


# ============================================================
# 5. DATA COLLECTION
# ============================================================

def collect_dataset(steps=DATA_STEPS):

    env = DroneEnv()

    inputs = []
    targets = []

    state = env.reset()

    for step in range(steps):

        action_pd = pd_controller(
            state,
            env.target.position(),
            env.target.velocity(),
        )

        mode = np.random.rand()

        if mode < 0.25:

            action = np.random.uniform(
                -1.0,
                1.0,
                size=2,
            )

        elif mode < 0.75:

            action = (
                action_pd
                + np.random.normal(
                    0.0,
                    0.28,
                    size=2,
                )
            )

        else:

            action = action_pd

        action = np.clip(
            action,
            -1.0,
            1.0,
        )

        next_state = env.step(
            action
        )

        inputs.append(
            np.concatenate(
                [
                    state,
                    action,
                ]
            )
        )

        targets.append(
            next_state - state
        )

        state = next_state

        if (step + 1) % 350 == 0:

            state = env.reset()

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
# 6. NORMALIZATION
# ============================================================

class Normalizer:

    def __init__(self, values):

        self.mean = values.mean(
            axis=0
        )

        self.std = np.maximum(
            values.std(axis=0),
            1e-6,
        )

    def transform(self, values):

        return (
            values - self.mean
        ) / self.std

    def inverse(self, values):

        return (
            values * self.std
            + self.mean
        )


# ============================================================
# 7. LEARNED DYNAMICS
# ============================================================

class DynamicsNetwork(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(INPUT_DIM, 128),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.ReLU(),

            nn.Linear(128, STATE_DIM),
        )

    def forward(self, x):

        return self.network(x)


class LearnedDynamics:

    def __init__(self):

        self.model = DynamicsNetwork().to(
            DEVICE
        )

        self.input_normalizer = None
        self.output_normalizer = None

        self.loss_history = []

    def fit(
        self,
        inputs,
        targets,
        epochs=TRAIN_EPOCHS,
    ):

        self.input_normalizer = Normalizer(
            inputs
        )

        self.output_normalizer = Normalizer(
            targets
        )

        x = self.input_normalizer.transform(
            inputs
        )

        y = self.output_normalizer.transform(
            targets
        )

        x_tensor = torch.as_tensor(
            x,
            dtype=torch.float32,
            device=DEVICE,
        )

        y_tensor = torch.as_tensor(
            y,
            dtype=torch.float32,
            device=DEVICE,
        )

        dataset = torch.utils.data.TensorDataset(
            x_tensor,
            y_tensor,
        )

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
        )

        optimizer = optim.Adam(
            self.model.parameters(),
            lr=LEARNING_RATE,
        )

        loss_fn = nn.MSELoss()

        for epoch in range(epochs):

            self.model.train()

            losses = []

            for batch_x, batch_y in loader:

                prediction = self.model(
                    batch_x
                )

                loss = loss_fn(
                    prediction,
                    batch_y,
                )

                optimizer.zero_grad(
                    set_to_none=True
                )

                loss.backward()

                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    2.0,
                )

                optimizer.step()

                losses.append(
                    loss.item()
                )

            average_loss = float(
                np.mean(losses)
            )

            self.loss_history.append(
                average_loss
            )

            if (
                (epoch + 1) % 20 == 0
                or epoch == 0
            ):

                print(
                    f"Epoch {epoch+1:3d}/{epochs}"
                    f" | normalized MSE={average_loss:.6f}"
                )

        self.model.eval()

    @torch.inference_mode()
    def predict_delta_batch(
        self,
        states,
        actions,
    ):

        model_inputs = np.concatenate(
            [
                states,
                actions,
            ],
            axis=1,
        ).astype(
            np.float32
        )

        normalized = (
            self.input_normalizer.transform(
                model_inputs
            )
        )

        tensor = torch.as_tensor(
            normalized,
            dtype=torch.float32,
            device=DEVICE,
        )

        prediction = self.model(
            tensor
        ).cpu().numpy()

        return self.output_normalizer.inverse(
            prediction
        )

    def predict_next(self, state, action):

        delta = self.predict_delta_batch(
            state[None, :],
            action[None, :],
        )[0]

        return clip_state(
            state + delta
        )


# ============================================================
# 8. IMPROVED COST FUNCTION
# ============================================================

def stage_cost_batch(
    states,
    target_position,
    target_velocity,
    actions,
    previous_actions=None,
):

    position_error = (
        states[:, :2] - target_position
    )

    distance = np.linalg.norm(
        position_error,
        axis=1,
    )

    velocity_error = (
        states[:, 2:4] - target_velocity
    )

    velocity_cost = np.sum(
        velocity_error**2,
        axis=1,
    )

    obstacle_distance = np.linalg.norm(
        states[:, :2] - OBSTACLE,
        axis=1,
    )

    safe_violation = np.maximum(
        0.0,
        SAFE_RADIUS - obstacle_distance,
    )

    obstacle_cost = (
        0.075 * safe_violation**2
    )

    boundary_distance = np.minimum.reduce(
        [
            states[:, 0],
            WORLD_W - states[:, 0],
            states[:, 1],
            WORLD_H - states[:, 1],
        ]
    )

    boundary_violation = np.maximum(
        0.0,
        65.0 - boundary_distance,
    )

    boundary_cost = (
        0.12 * boundary_violation**2
    )

    effort_cost = (
        0.18 * np.sum(
            actions**2,
            axis=1,
        )
    )

    if previous_actions is not None:

        smooth_cost = (
            0.16 * np.sum(
                (actions - previous_actions)**2,
                axis=1,
            )
        )

    else:

        smooth_cost = 0.0

    return (
        0.035 * distance
        + 0.0007 * velocity_cost
        + obstacle_cost
        + boundary_cost
        + effort_cost
        + smooth_cost
    )


def terminal_cost_batch(
    states,
    target_position,
    target_velocity,
):

    distance = np.linalg.norm(
        states[:, :2] - target_position,
        axis=1,
    )

    velocity_error = np.linalg.norm(
        states[:, 2:4] - target_velocity,
        axis=1,
    )

    return (
        0.12 * distance
        + 0.001 * velocity_error**2
    )


# ============================================================
# 9. BATCHED TRAJECTORY EVALUATION
# ============================================================

def evaluate_action_sequences(
    initial_state,
    action_sequences,
    target,
    phase,
    model,
):

    candidate_count = len(
        action_sequences
    )

    states = np.repeat(
        initial_state[None, :],
        candidate_count,
        axis=0,
    )

    costs = np.zeros(
        candidate_count,
        dtype=np.float64,
    )

    previous_actions = np.repeat(
        np.zeros((1, 2)),
        candidate_count,
        axis=0,
    )

    horizon = action_sequences.shape[1]

    for step in range(horizon):

        actions = action_sequences[
            :,
            step,
            :,
        ]

        if model == "oracle":

            next_states = true_transition_batch(
                states,
                actions,
            )

        elif isinstance(model, LearnedDynamics):

            deltas = model.predict_delta_batch(
                states,
                actions,
            )

            next_states = clip_states_batch(
                states + deltas
            )

        else:

            raise ValueError(
                "Unsupported dynamics model"
            )

        future_phase = (
            phase
            + TARGET_OMEGA * DT * (step + 1)
        )

        future_target = target.position(
            future_phase
        )

        future_velocity = target.velocity(
            future_phase
        )

        costs += stage_cost_batch(
            next_states,
            future_target,
            future_velocity,
            actions,
            previous_actions,
        )

        states = next_states
        previous_actions = actions

    final_phase = (
        phase
        + TARGET_OMEGA * DT * horizon
    )

    costs += terminal_cost_batch(
        states,
        target.position(final_phase),
        target.velocity(final_phase),
    )

    return costs


# ============================================================
# 10. NOMINAL ACTION SEQUENCE
# ============================================================

def make_nominal_sequence(
    state,
    target,
    phase,
    horizon,
):

    nominal = np.zeros(
        (horizon, 2),
        dtype=np.float64,
    )

    estimated_state = state.copy()

    for step in range(horizon):

        future_phase = (
            phase + TARGET_OMEGA * DT * step
        )

        future_position = target.position(
            future_phase
        )

        future_velocity = target.velocity(
            future_phase
        )

        action = pd_controller(
            estimated_state,
            future_position,
            future_velocity,
        )

        nominal[step] = action

        estimated_state = true_transition(
            estimated_state,
            action,
        )

    return nominal


# ============================================================
# 11. STABLE MPPI WEIGHTS
# ============================================================

def calculate_mppi_weights(
    costs,
    temperature=LAMBDA,
):

    minimum_cost = np.min(
        costs
    )

    logits = (
        -(costs - minimum_cost)
        / max(temperature, 1e-6)
    )

    logits -= np.max(
        logits
    )

    weights = np.exp(
        np.clip(logits, -80.0, 0.0)
    )

    weights /= (
        np.sum(weights) + 1e-12
    )

    return weights


# ============================================================
# 12. RANDOM SHOOTING MPC
# ============================================================

def random_shooting_action(
    state,
    target,
    phase,
    model,
    horizon=HORIZON,
    samples=SAMPLE_COUNT,
):

    nominal = make_nominal_sequence(
        state,
        target,
        phase,
        horizon,
    )

    candidates = (
        nominal[None, :, :]
        + np.random.normal(
            0.0,
            NOISE_STD,
            size=(samples, horizon, 2),
        )
    )

    candidates = np.clip(
        candidates,
        -1.0,
        1.0,
    )

    candidates[0] = nominal

    costs = evaluate_action_sequences(
        state,
        candidates,
        target,
        phase,
        model,
    )

    best = int(
        np.argmin(costs)
    )

    return (
        candidates[best, 0].copy(),
        candidates[best].copy(),
        float(costs[best]),
    )


# ============================================================
# 13. IMPROVED MPPI CONTROLLER
# ============================================================

class MPPIController:

    def __init__(
        self,
        model,
        horizon=HORIZON,
        sample_count=SAMPLE_COUNT,
        iterations=MPPI_ITERATIONS,
        temperature=LAMBDA,
    ):

        self.model = model

        self.horizon = horizon
        self.sample_count = sample_count
        self.iterations = iterations
        self.temperature = temperature

        self.nominal = np.zeros(
            (horizon, 2),
            dtype=np.float64,
        )

        self.initialized = False

        self.last_weights = None
        self.last_costs = None

        self.last_action = np.zeros(
            2,
            dtype=np.float64,
        )

    def reset(self):

        self.nominal.fill(0.0)

        self.initialized = False

        self.last_weights = None
        self.last_costs = None

        self.last_action.fill(0.0)

    def plan(
        self,
        state,
        target,
        phase,
    ):

        if not self.initialized:

            self.nominal = make_nominal_sequence(
                state,
                target,
                phase,
                self.horizon,
            )

            self.initialized = True

        nominal = self.nominal.copy()

        for _ in range(self.iterations):

            noise = np.random.normal(
                0.0,
                NOISE_STD,
                size=(
                    self.sample_count,
                    self.horizon,
                    2,
                ),
            )

            candidates = (
                nominal[None, :, :]
                + noise
            )

            candidates = np.clip(
                candidates,
                -1.0,
                1.0,
            )

            candidates[0] = nominal

            costs = evaluate_action_sequences(
                state,
                candidates,
                target,
                phase,
                self.model,
            )

            weights = calculate_mppi_weights(
                costs,
                self.temperature,
            )

            weighted_sequence = np.sum(
                weights[:, None, None]
                * candidates,
                axis=0,
            )

            nominal = (
                NOMINAL_BLEND * weighted_sequence
                + (1.0 - NOMINAL_BLEND) * nominal
            )

            nominal = np.clip(
                nominal,
                -1.0,
                1.0,
            )

            self.last_weights = weights
            self.last_costs = costs

        raw_action = nominal[0].copy()

        action = (
            ACTION_SMOOTHING * self.last_action
            + (1.0 - ACTION_SMOOTHING) * raw_action
        )

        action = np.clip(
            action,
            -1.0,
            1.0,
        )

        self.last_action = action.copy()

        self.nominal[:-1] = nominal[1:]
        self.nominal[-1] = nominal[-1]

        return (
            action.copy(),
            nominal.copy(),
            float(np.min(costs)),
        )


# ============================================================
# 14. BENCHMARKING
# ============================================================

def run_episode(
    mode,
    learned_model=None,
    steps=BENCHMARK_STEPS,
):

    env = DroneEnv()
    state = env.reset()

    if mode == "Learned MPPI":

        controller = MPPIController(
            learned_model
        )

    elif mode == "Oracle MPPI":

        controller = MPPIController(
            "oracle"
        )

    distances = []
    collision_steps = 0

    for _ in range(steps):

        target_position = env.target.position()
        target_velocity = env.target.velocity()

        if mode == "PD":

            action = pd_controller(
                state,
                target_position,
                target_velocity,
            )

        elif mode == "Random MPC":

            action, _, _ = random_shooting_action(
                state,
                env.target,
                env.target.phase,
                learned_model,
            )

        elif mode in (
            "Learned MPPI",
            "Oracle MPPI",
        ):

            action, _, _ = controller.plan(
                state,
                env.target,
                env.target.phase,
            )

        else:

            raise ValueError(mode)

        state = env.step(action)

        distances.append(
            np.linalg.norm(
                state[:2] - env.target.position()
            )
        )

        clearance = (
            np.linalg.norm(
                state[:2] - OBSTACLE
            )
            - OBSTACLE_RADIUS
        )

        if clearance < 0:
            collision_steps += 1

    return {
        "mean_distance": float(
            np.mean(distances)
        ),

        "collision_steps": collision_steps,

        "distances": np.asarray(
            distances
        ),
    }


def benchmark(learned_model):

    modes = [
        "PD",
        "Random MPC",
        "Learned MPPI",
        "Oracle MPPI",
    ]

    results = {}

    for mode in modes:

        print(
            f"\nBenchmarking {mode}..."
        )

        episode_results = []

        for episode in range(BENCHMARK_EPISODES):

            result = run_episode(
                mode,
                learned_model,
            )

            episode_results.append(result)

        results[mode] = {
            "mean_distance": float(
                np.mean(
                    [
                        item["mean_distance"]
                        for item in episode_results
                    ]
                )
            ),

            "collision_steps": float(
                np.mean(
                    [
                        item["collision_steps"]
                        for item in episode_results
                    ]
                )
            ),
        }

        print(
            f"{mode:15s}"
            f" | distance={results[mode]['mean_distance']:8.2f}"
            f" | collision steps={results[mode]['collision_steps']:7.2f}"
        )

    return results


# ============================================================
# 15. VISUALIZATION — TRAINING
# ============================================================

def plot_training_loss(model):

    plt.figure(figsize=(9, 5))

    plt.plot(
        model.loss_history,
        linewidth=2,
    )

    plt.title(
        "Learned Dynamics Model Training Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Normalized MSE")

    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_benchmark(results):

    names = list(results.keys())

    distances = [
        results[name]["mean_distance"]
        for name in names
    ]

    collisions = [
        results[name]["collision_steps"]
        for name in names
    ]

    plt.figure(figsize=(9, 5))

    plt.bar(names, distances)

    plt.title(
        "Moving Target Tracking Performance"
    )

    plt.ylabel("Mean tracking distance")

    plt.grid(
        axis="y",
        alpha=0.3,
    )

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(9, 5))

    plt.bar(names, collisions)

    plt.title(
        "Obstacle Collision Performance"
    )

    plt.ylabel("Average collision steps")

    plt.grid(
        axis="y",
        alpha=0.3,
    )

    plt.tight_layout()
    plt.show()


# ============================================================
# 16. INTERACTIVE PYGAME
# ============================================================

def run_pygame(learned_model):

    try:
        import pygame

    except ImportError:

        print(
            "Install pygame: pip install pygame"
        )

        return

    pygame.init()

    screen = pygame.display.set_mode(
        (WORLD_W, WORLD_H)
    )

    pygame.display.set_caption(
        "PHASE 4.27 | High Performance MPPI Drone"
    )

    clock = pygame.time.Clock()

    font = pygame.font.SysFont(
        "consolas",
        16,
    )

    small_font = pygame.font.SysFont(
        "consolas",
        13,
    )

    env = DroneEnv()
    state = env.reset()

    controllers = {
        1: "PD",
        2: "Random MPC",
        3: "Learned MPPI",
        4: "Oracle MPPI",
    }

    selected = 3

    learned_mppi = MPPIController(
        learned_model,
        horizon=INTERACTIVE_HORIZON,
        sample_count=INTERACTIVE_SAMPLES,
        iterations=INTERACTIVE_ITERATIONS,
    )

    oracle_mppi = MPPIController(
        "oracle",
        horizon=INTERACTIVE_HORIZON,
        sample_count=INTERACTIVE_SAMPLES,
        iterations=INTERACTIVE_ITERATIONS,
    )

    paused = False
    simulation_speed = 1

    trail = []
    predicted_path = []

    current_weights = None

    current_action = np.zeros(
        2,
        dtype=np.float64,
    )

    current_cost = 0.0

    frame_times = []
    fps_value = 0.0
    planning_time_ms = 0.0

    quality_levels = [
        (60, 1),
        (100, 2),
        (150, 2),
        (180, 3),
    ]

    quality_index = 1

    def update_quality():

        samples, iterations = quality_levels[
            quality_index
        ]

        learned_mppi.sample_count = samples
        learned_mppi.iterations = iterations

        oracle_mppi.sample_count = samples
        oracle_mppi.iterations = iterations

    def draw_text(
        message,
        x,
        y,
        color=(235, 235, 235),
        use_small=False,
    ):

        selected_font = (
            small_font if use_small else font
        )

        surface = selected_font.render(
            message,
            True,
            color,
        )

        screen.blit(
            surface,
            (x, y),
        )

    def draw_background():

        screen.fill(
            (14, 19, 29)
        )

        for x in range(0, WORLD_W, 50):

            pygame.draw.line(
                screen,
                (28, 36, 49),
                (x, 0),
                (x, WORLD_H),
            )

        for y in range(0, WORLD_H, 50):

            pygame.draw.line(
                screen,
                (28, 36, 49),
                (0, y),
                (WORLD_W, y),
            )

    def draw_obstacle():

        center = OBSTACLE.astype(int)

        pygame.draw.circle(
            screen,
            (75, 79, 91),
            center,
            int(OBSTACLE_RADIUS),
        )

        pygame.draw.circle(
            screen,
            (110, 120, 135),
            center,
            int(OBSTACLE_RADIUS),
            2,
        )

        pygame.draw.circle(
            screen,
            (180, 95, 85),
            center,
            int(SAFE_RADIUS),
            1,
        )

        pygame.draw.circle(
            screen,
            (200, 105, 90),
            center,
            4,
        )

    def draw_target():

        points = []

        for index in range(120):

            phase = (
                env.target.phase
                + 2.0 * math.pi * index / 120
            )

            points.append(
                env.target.position(
                    phase
                ).astype(int)
            )

        pygame.draw.lines(
            screen,
            (48, 95, 65),
            True,
            points,
            2,
        )

        position = env.target.position()
        velocity = env.target.velocity()

        future_position = env.target.position(
            env.target.phase + TARGET_OMEGA * DT * 5
        )

        pygame.draw.line(
            screen,
            (90, 170, 105),
            position.astype(int),
            future_position.astype(int),
            2,
        )

        pygame.draw.circle(
            screen,
            (65, 235, 105),
            position.astype(int),
            12,
        )

        pygame.draw.circle(
            screen,
            (200, 255, 210),
            position.astype(int),
            5,
        )

        pygame.draw.line(
            screen,
            (90, 255, 125),
            position.astype(int),
            (
                position + velocity * 0.16
            ).astype(int),
            3,
        )

    def draw_drone():

        center = state[:2]
        velocity = state[2:4]

        speed = np.linalg.norm(
            velocity
        )

        if speed > 1e-6:
            direction = velocity / speed
        else:
            direction = np.array([1.0, 0.0])

        side = np.array(
            [
                -direction[1],
                direction[0],
            ]
        )

        center_int = center.astype(int)

        pygame.draw.circle(
            screen,
            (8, 12, 19),
            (center_int[0] + 3, center_int[1] + 4),
            18,
        )

        for forward in (-1, 1):

            for lateral in (-1, 1):

                end = (
                    center
                    + 16 * forward * direction
                    + 16 * lateral * side
                )

                pygame.draw.line(
                    screen,
                    (200, 210, 225),
                    center_int,
                    end.astype(int),
                    5,
                )

                pygame.draw.circle(
                    screen,
                    (90, 180, 240),
                    end.astype(int),
                    8,
                    2,
                )

        pygame.draw.circle(
            screen,
            (220, 225, 235),
            center_int,
            9,
        )

        pygame.draw.circle(
            screen,
            (245, 190, 65),
            (
                center + direction * 23
            ).astype(int),
            5,
        )

    def draw_trail():

        if len(trail) > 1:

            pygame.draw.lines(
                screen,
                (65, 130, 230),
                False,
                [
                    point.astype(int)
                    for point in trail
                ],
                3,
            )

    def draw_prediction():

        if len(predicted_path) > 1:

            points = [
                point.astype(int)
                for point in predicted_path
            ]

            pygame.draw.lines(
                screen,
                (255, 195, 65),
                False,
                points,
                3,
            )

            for index in range(
                0,
                len(points),
                3,
            ):

                pygame.draw.circle(
                    screen,
                    (255, 215, 110),
                    points[index],
                    3,
                )

    def draw_hud():

        distance = np.linalg.norm(
            state[:2] - env.target.position()
        )

        clearance = (
            np.linalg.norm(
                state[:2] - OBSTACLE
            )
            - OBSTACLE_RADIUS
        )

        speed = np.linalg.norm(
            state[2:4]
        )

        panel = pygame.Surface(
            (380, 230),
            pygame.SRCALPHA,
        )

        panel.fill(
            (10, 15, 25, 220)
        )

        screen.blit(
            panel,
            (8, 8),
        )

        draw_text(
            "PHASE 4.27 | MPPI CONTROL",
            20,
            16,
            (110, 200, 255),
        )

        draw_text(
            f"Controller: {controllers[selected]}",
            20,
            45,
            (245, 245, 245),
        )

        draw_text(
            f"Tracking distance: {distance:7.2f}",
            20,
            73,
        )

        draw_text(
            f"Drone speed:       {speed:7.2f}",
            20,
            99,
        )

        clearance_color = (
            (255, 90, 90)
            if clearance < 0
            else (100, 230, 140)
        )

        draw_text(
            f"Obstacle clearance: {clearance:7.2f}",
            20,
            125,
            clearance_color,
        )

        draw_text(
            f"FPS: {fps_value:5.1f}",
            20,
            151,
            (120, 210, 255),
        )

        draw_text(
            f"Planning time: {planning_time_ms:6.1f} ms",
            20,
            177,
            (255, 210, 100),
        )

        draw_text(
            f"Simulation speed: {simulation_speed}x",
            20,
            203,
        )

        draw_text(
            "1 PD | 2 Random MPC | 3 Learned MPPI | 4 Oracle MPPI",
            15,
            WORLD_H - 85,
            (230, 230, 240),
            True,
        )

        draw_text(
            "SPACE Pause | R Reset | +/- Speed | UP/DOWN Quality",
            15,
            WORLD_H - 60,
            (190, 200, 215),
            True,
        )

        draw_text(
            "Mouse click: move target center | ESC: Exit",
            15,
            WORLD_H - 35,
            (190, 200, 215),
            True,
        )

        if current_weights is not None:

            effective_samples = (
                1.0
                / (
                    np.sum(
                        current_weights**2
                    )
                    + 1e-12
                )
            )

            draw_text(
                f"Effective samples: {effective_samples:.1f}",
                WORLD_W - 260,
                20,
                (255, 210, 100),
            )

        if paused:

            draw_text(
                "PAUSED",
                WORLD_W - 150,
                55,
                (255, 210, 80),
            )

    running = True

    while running:

        frame_start = time.perf_counter()

        clock.tick(
            RENDER_FPS
        )

        frame_times.append(
            time.perf_counter() - frame_start
        )

        if len(frame_times) > 30:
            frame_times.pop(0)

        average_frame_time = np.mean(
            frame_times
        )

        fps_value = (
            1.0 / max(average_frame_time, 1e-6)
        )

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:

                if event.key == pygame.K_ESCAPE:
                    running = False

                elif event.key == pygame.K_SPACE:
                    paused = not paused

                elif event.key == pygame.K_r:

                    state = env.reset()

                    learned_mppi.reset()
                    oracle_mppi.reset()

                    trail.clear()
                    predicted_path.clear()

                    current_weights = None
                    current_action.fill(0.0)

                elif event.key == pygame.K_1:
                    selected = 1

                elif event.key == pygame.K_2:
                    selected = 2

                elif event.key == pygame.K_3:
                    selected = 3

                elif event.key == pygame.K_4:
                    selected = 4

                elif event.key in (
                    pygame.K_EQUALS,
                    pygame.K_PLUS,
                ):

                    simulation_speed = min(
                        5,
                        simulation_speed + 1,
                    )

                elif event.key == pygame.K_MINUS:

                    simulation_speed = max(
                        1,
                        simulation_speed - 1,
                    )

                elif event.key == pygame.K_UP:

                    quality_index = min(
                        len(quality_levels) - 1,
                        quality_index + 1,
                    )

                    update_quality()

                elif event.key == pygame.K_DOWN:

                    quality_index = max(
                        0,
                        quality_index - 1,
                    )

                    update_quality()

            elif event.type == pygame.MOUSEBUTTONDOWN:

                if event.button == 1:

                    click = np.asarray(
                        event.pos,
                        dtype=np.float64,
                    )

                    if (
                        np.linalg.norm(
                            click - OBSTACLE
                        ) > SAFE_RADIUS
                    ):

                        env.target.center = np.clip(
                            click,
                            [
                                TARGET_A + 20,
                                TARGET_B + 20,
                            ],
                            [
                                WORLD_W - TARGET_A - 20,
                                WORLD_H - TARGET_B - 20,
                            ],
                        )

                        learned_mppi.reset()
                        oracle_mppi.reset()

        mode = controllers[selected]

        if not paused:

            for _ in range(simulation_speed):

                planning_start = time.perf_counter()

                if mode == "PD":

                    current_action = pd_controller(
                        state,
                        env.target.position(),
                        env.target.velocity(),
                    )

                    predicted_path = []
                    current_weights = None

                elif mode == "Random MPC":

                    current_action, sequence, current_cost = (
                        random_shooting_action(
                            state,
                            env.target,
                            env.target.phase,
                            learned_model,
                            horizon=INTERACTIVE_HORIZON,
                            samples=INTERACTIVE_SAMPLES,
                        )
                    )

                    predicted_path = []
                    current_weights = None

                else:

                    controller = (
                        learned_mppi
                        if mode == "Learned MPPI"
                        else oracle_mppi
                    )

                    current_action, sequence, current_cost = (
                        controller.plan(
                            state,
                            env.target,
                            env.target.phase,
                        )
                    )

                    current_weights = (
                        controller.last_weights
                    )

                    predicted_state = state.copy()

                    predicted_path = [
                        predicted_state[:2].copy()
                    ]

                    if mode == "Oracle MPPI":

                        for future_action in sequence:

                            predicted_state = true_transition(
                                predicted_state,
                                future_action,
                            )

                            predicted_path.append(
                                predicted_state[:2].copy()
                            )

                    else:

                        predicted_states = [
                            predicted_state.copy()
                        ]

                        states_batch = predicted_state[None, :]

                        for future_action in sequence:

                            delta = learned_model.predict_delta_batch(
                                states_batch,
                                future_action[None, :],
                            )

                            predicted_state = clip_state(
                                predicted_state + delta[0]
                            )

                            predicted_states.append(
                                predicted_state.copy()
                            )

                            states_batch = predicted_state[None, :]

                        predicted_path = [
                            item[:2].copy()
                            for item in predicted_states
                        ]

                planning_time_ms = (
                    (time.perf_counter() - planning_start)
                    * 1000.0
                )

                state = env.step(
                    current_action
                )

                trail.append(
                    state[:2].copy()
                )

                if len(trail) > TRAIL_LENGTH:
                    trail.pop(0)

        draw_background()
        draw_obstacle()
        draw_target()
        draw_trail()
        draw_prediction()
        draw_drone()
        draw_hud()

        pygame.display.flip()

    pygame.quit()


# ============================================================
# 17. UNIT TESTS — FIXED: EXACTLY 24 CHECKS
# ============================================================

def run_tests():

    print("\nRunning unit tests...")

    passed = 0
    total = 24

    def check(condition, description):

        nonlocal passed

        if not bool(condition):

            raise AssertionError(
                f"[FAIL] {description}"
            )

        passed += 1

        print(
            f"[PASS] {description}"
        )

    # 1.
    state = np.array(
        [100.0, 200.0, 10.0, -5.0]
    )

    action = np.array(
        [0.2, -0.1]
    )

    next_state = true_transition(
        state,
        action,
    )

    check(
        next_state.shape == (4,)
        and np.all(np.isfinite(next_state)),
        "True dynamics return valid state",
    )

    # 2.
    clipped = true_transition(
        state,
        np.array([5.0, -5.0]),
    )

    bounded = true_transition(
        state,
        np.array([1.0, -1.0]),
    )

    check(
        np.allclose(clipped, bounded),
        "Action clipping works",
    )

    # 3.
    fast_state = np.array(
        [100.0, 100.0, 500.0, 500.0]
    )

    bounded_state = clip_state(
        fast_state
    )

    check(
        np.linalg.norm(
            bounded_state[2:4]
        ) <= MAX_SPEED + 1e-8,
        "Speed limit works",
    )

    # 4.
    check(
        DRONE_RADIUS <= bounded_state[0]
        <= WORLD_W - DRONE_RADIUS,
        "Horizontal boundary works",
    )

    # 5.
    target = MovingTarget(
        [700.0, 300.0]
    )

    before = target.position().copy()

    target.advance()

    check(
        not np.allclose(
            before,
            target.position(),
        ),
        "Moving target advances",
    )

    # 6.
    inputs, targets = collect_dataset(
        steps=100
    )

    check(
        inputs.shape == (100, 6)
        and targets.shape == (100, 4),
        "Dataset dimensions are correct",
    )

    # 7.
    check(
        np.all(np.isfinite(inputs))
        and np.all(np.isfinite(targets)),
        "Dataset is finite",
    )

    # 8.
    normalizer = Normalizer(
        inputs
    )

    reconstructed = normalizer.inverse(
        normalizer.transform(inputs)
    )

    check(
        np.allclose(
            reconstructed,
            inputs,
            atol=1e-5,
        ),
        "Normalization round trip works",
    )

    # 9.
    network = DynamicsNetwork().to(
        DEVICE
    )

    output = network(
        torch.zeros(
            (10, INPUT_DIM),
            device=DEVICE,
        )
    )

    check(
        output.shape == (10, 4),
        "Neural network output dimensions",
    )

    # 10.
    check(
        torch.all(torch.isfinite(output)),
        "Neural network output is finite",
    )

    # 11. FIXED: Environment reset is now counted.
    env = DroneEnv()

    reset_state = env.reset()

    check(
        reset_state.shape == (4,)
        and np.all(np.isfinite(reset_state))
        and np.allclose(
            reset_state,
            [110.0, 500.0, 0.0, 0.0],
        ),
        "Environment reset works",
    )

    # 12.
    nominal = make_nominal_sequence(
        env.state,
        env.target,
        env.target.phase,
        HORIZON,
    )

    check(
        nominal.shape == (HORIZON, 2),
        "Nominal sequence dimensions",
    )

    # 13.
    check(
        np.all(nominal >= -1.0)
        and np.all(nominal <= 1.0),
        "Nominal sequence is bounded",
    )

    # 14.
    sequences = np.repeat(
        nominal[None, :, :],
        5,
        axis=0,
    )

    oracle_costs = evaluate_action_sequences(
        env.state,
        sequences,
        env.target,
        env.target.phase,
        "oracle",
    )

    check(
        oracle_costs.shape == (5,)
        and np.all(np.isfinite(oracle_costs)),
        "Oracle batch rollout works",
    )

    # 15.
    states_batch = np.repeat(
        env.state[None, :],
        10,
        axis=0,
    )

    actions_batch = np.zeros(
        (10, 2)
    )

    batch_result = true_transition_batch(
        states_batch,
        actions_batch,
    )

    single_result = true_transition(
        env.state,
        np.zeros(2),
    )

    check(
        np.allclose(
            batch_result[0],
            single_result,
        ),
        "Vectorized and single dynamics agree",
    )

    # 16.
    costs = np.array(
        [10.0, 11.0, 12.0, 20.0]
    )

    weights = calculate_mppi_weights(
        costs
    )

    check(
        np.isclose(weights.sum(), 1.0),
        "MPPI weights sum to one",
    )

    # 17.
    check(
        weights[0] > weights[1] > weights[2] > weights[3],
        "Lower-cost sequences receive higher weights",
    )

    # 18.
    candidates = np.array(
        [
            [[0.0, 0.0]],
            [[1.0, 1.0]],
        ]
    )

    test_weights = np.array(
        [0.75, 0.25]
    )

    updated = np.sum(
        test_weights[:, None, None] * candidates,
        axis=0,
    )

    check(
        np.allclose(
            updated,
            [[0.25, 0.25]],
        ),
        "Weighted sequence update works",
    )

    # 19.
    controller = MPPIController(
        "oracle",
        horizon=5,
        sample_count=12,
        iterations=2,
    )

    action, sequence, cost = controller.plan(
        env.state,
        env.target,
        env.target.phase,
    )

    check(
        action.shape == (2,)
        and sequence.shape == (5, 2)
        and np.isfinite(cost),
        "MPPI planner interface works",
    )

    # 20.
    check(
        np.all(action >= -1.0)
        and np.all(action <= 1.0),
        "MPPI action bounds work",
    )

    # 21.
    check(
        controller.last_weights is not None
        and np.isclose(
            controller.last_weights.sum(),
            1.0,
        ),
        "MPPI weights are normalized",
    )

    # 22.
    controller.reset()

    check(
        not controller.initialized
        and controller.last_weights is None,
        "MPPI reset works",
    )

    # 23.
    controller.plan(
        env.state,
        env.target,
        env.target.phase,
    )

    check(
        controller.initialized,
        "MPPI initializes correctly",
    )

    # 24.
    learned_model = LearnedDynamics()

    learned_model.input_normalizer = Normalizer(
        np.random.normal(
            0,
            1,
            (10, 6),
        ).astype(np.float32)
    )

    learned_model.output_normalizer = Normalizer(
        np.random.normal(
            0,
            1,
            (10, 4),
        ).astype(np.float32)
    )

    prediction = learned_model.predict_delta_batch(
        states_batch,
        actions_batch,
    )

    check(
        prediction.shape == (10, 4)
        and np.all(np.isfinite(prediction)),
        "Batched learned dynamics prediction works",
    )

    print(
        f"\n{passed}/{total} tests passed."
    )

    if passed != total:
        raise RuntimeError(
            f"Expected {total} passing tests, got {passed}."
        )

    return True


# ============================================================
# 18. MAIN
# ============================================================

def main():

    print("=" * 72)

    print(
        "PHASE 4.27 — HIGH-PERFORMANCE MPPI"
    )

    print("=" * 72)

    print(
        f"Device: {DEVICE}"
    )

    print(
        f"CUDA available: {torch.cuda.is_available()}"
    )

    # 1. Tests.
    if not run_tests():

        raise RuntimeError(
            "Unit tests failed."
        )

    # 2. Dataset.
    print(
        "\n[1/6] Collecting dynamics dataset..."
    )

    inputs, targets = collect_dataset()

    print(
        f"Transitions collected: {len(inputs)}"
    )

    # 3. Model training.
    print(
        "\n[2/6] Training learned dynamics..."
    )

    learned_model = LearnedDynamics()

    learned_model.fit(
        inputs,
        targets,
    )

    plot_training_loss(
        learned_model
    )

    # 4. Benchmark.
    print(
        "\n[3/6] Benchmarking controllers..."
    )

    results = benchmark(
        learned_model
    )

    plot_benchmark(
        results
    )

    # 5. MPPI configuration.
    print(
        "\n[4/6] MPPI configuration"
    )

    print(
        f"Benchmark horizon: {HORIZON}"
    )

    print(
        f"Benchmark samples: {SAMPLE_COUNT}"
    )

    print(
        f"Benchmark iterations: {MPPI_ITERATIONS}"
    )

    print(
        f"Interactive samples: {INTERACTIVE_SAMPLES}"
    )

    print(
        f"Interactive iterations: {INTERACTIVE_ITERATIONS}"
    )

    print(
        f"Lambda: {LAMBDA}"
    )

    print(
        f"Action noise: {NOISE_STD}"
    )

    # 6. Pygame.
    print(
        "\n[5/6] Starting interactive simulation..."
    )

    print("\nCONTROLS")
    print("1 = PD")
    print("2 = Random MPC")
    print("3 = Learned MPPI")
    print("4 = Oracle MPPI")
    print("SPACE = Pause")
    print("R = Reset")
    print("+/- = Simulation speed")
    print("UP/DOWN = MPPI planning quality")
    print("Mouse click = Move target center")
    print("ESC = Exit")

    run_pygame(
        learned_model
    )

    print(
        "\n[6/6] PHASE 4.27 COMPLETED."
    )


if __name__ == "__main__":

    main()