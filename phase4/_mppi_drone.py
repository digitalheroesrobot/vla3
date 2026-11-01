
"""
File: phase4/mppi_drone.py

PHASE 4.27 — MODEL PREDICTIVE PATH INTEGRAL CONTROL (MPPI)

PURPOSE
-------
Use MPPI to plan quadcopter actions for tracking a continuously
moving target while avoiding obstacles.

WHAT YOU LEARN
--------------

STEP 1 — Dynamics Dataset
    Collect drone state/action/next-state transitions.

STEP 2 — Learned Dynamics
    Train a neural network to predict drone state changes.

STEP 3 — MPPI Sampling
    Sample many candidate future action sequences.

STEP 4 — Cost Evaluation
    Simulate candidate sequences and calculate their costs.

STEP 5 — Path Integral Update
    Give lower-cost sequences higher weights and update the
    nominal action sequence using a weighted average.

STEP 6 — Receding-Horizon Control
    Execute only the first action, then replan.

STEP 7 — Controller Comparison
    Compare PD, random-shooting MPC, learned MPPI and Oracle MPPI.

MATHEMATICAL FOUNDATIONS
------------------------

For a sampled action sequence:

    U_k = [u_0, u_1, ..., u_(H-1)]

Its trajectory cost is:

    S_k = phi(x_H) + sum_t q(x_t, u_t)

MPPI weights:

    w_k = exp(-(S_k - S_min) / lambda)
          --------------------------------
          sum_j exp(-(S_j - S_min) / lambda)

Weighted action update:

    U_new = sum_k w_k * U_k

The subtraction of S_min is a numerical stabilization:
it does not change the normalized weights.

In practice, action noise is sampled around a nominal sequence:

    U_k = U_nominal + epsilon_k

Then:

    U_nominal <- weighted_mean(U_k)

Only the first action is executed. The sequence is shifted
forward and optimized again at the next time step.

IMPORTANT
---------
This educational implementation uses a simplified MPPI update.
It uses weighted sampled action sequences without the full
importance-sampling correction term found in some MPPI variants.

The learned model is trained from simulated transitions.
Prediction errors can accumulate over long planning horizons.

Obstacle avoidance uses soft cost penalties. It is not a
formal safety guarantee.

ROBOTICS APPLICATION
--------------------
MPPI is useful for:
    - drone trajectory tracking
    - autonomous navigation
    - obstacle-aware motion planning
    - mobile robot control
    - manipulation trajectory optimization

The simulator is 2D and is not a flight-certified controller.
"""

from __future__ import annotations

import copy
import math
import random

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

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

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
SAFE_RADIUS = 110.0

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

# MPPI parameters.
HORIZON = 16
SAMPLE_COUNT = 180
MPPI_ITERATIONS = 3

NOISE_STD = np.array(
    [0.30, 0.30],
    dtype=np.float64,
)

LAMBDA = 8.0

# Benchmark.
BENCHMARK_EPISODES = 3
BENCHMARK_STEPS = 180


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

        self.phase += (
            TARGET_OMEGA * DT
        )


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
            MAX_SPEED / (speed + 1e-9)
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


def true_transition(
    state,
    action,
):

    """
    Ground-truth 2D drone dynamics.

    state = [x, y, vx, vy]
    action = [ux, uy]
    """

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

    def observation(self):

        relative_target = (
            self.target.position()
            - self.position
        )

        target_velocity = (
            self.target.velocity()
        )

        relative_obstacle = (
            OBSTACLE - self.position
        )

        return np.concatenate(
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
        ).astype(
            np.float32
        )


# ============================================================
# 4. DATA COLLECTION
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
            MAX_SPEED / (speed + 1e-9)
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


def collect_dataset(
    steps=DATA_STEPS,
):

    env = DroneEnv()

    inputs = []
    targets = []

    state = env.reset()

    for step in range(steps):

        pd_action = pd_controller(
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
                pd_action
                + np.random.normal(
                    0.0,
                    0.28,
                    size=2,
                )
            )

        else:

            action = pd_action

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
# 5. NORMALIZATION
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
# 6. LEARNED DYNAMICS MODEL
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

            losses = []

            for batch_x, batch_y in loader:

                prediction = self.model(
                    batch_x
                )

                loss = loss_fn(
                    prediction,
                    batch_y,
                )

                optimizer.zero_grad()

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
                    f"Model epoch {epoch+1:3d}/{epochs}"
                    f" | normalized MSE={average_loss:.6f}"
                )

    def predict_delta_batch(
        self,
        states,
        actions,
    ):

        """
        Vectorized model prediction.

        states:  (N, 4)
        actions: (N, 2)
        """

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

        with torch.no_grad():

            prediction = (
                self.model(tensor)
                .cpu()
                .numpy()
            )

        return self.output_normalizer.inverse(
            prediction
        )

    def predict_next(
        self,
        state,
        action,
    ):

        delta = self.predict_delta_batch(
            state[None, :],
            action[None, :],
        )[0]

        return clip_state(
            state + delta
        )


# ============================================================
# 7. MPC STAGE COST
# ============================================================

def stage_cost(
    state,
    target_position,
    action,
):

    distance = np.linalg.norm(
        state[:2] - target_position
    )

    obstacle_distance = np.linalg.norm(
        state[:2] - OBSTACLE
    )

    obstacle_violation = max(
        0.0,
        SAFE_RADIUS - obstacle_distance,
    )

    boundary_distance = min(
        state[0],
        WORLD_W - state[0],
        state[1],
        WORLD_H - state[1],
    )

    boundary_violation = max(
        0.0,
        55.0 - boundary_distance,
    )

    return (
        0.025 * distance
        + 0.055 * obstacle_violation**2
        + 0.06 * boundary_violation**2
        + 0.25 * np.sum(action**2)
    )


def terminal_cost(
    state,
    target_position,
):

    distance = np.linalg.norm(
        state[:2] - target_position
    )

    return 0.05 * distance


# ============================================================
# 8. BATCHED CANDIDATE ROLLOUT
# ============================================================

def evaluate_action_sequences(
    initial_state,
    action_sequences,
    target,
    phase,
    model,
):

    """
    Evaluate many candidate sequences.

    action_sequences:
        (N, H, 2)

    Each candidate starts from the same initial state.

    Learned-model predictions are vectorized across candidates.
    """

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

    for step in range(
        action_sequences.shape[1]
    ):

        actions = action_sequences[
            :,
            step,
            :,
        ]

        future_phase = (
            phase
            + TARGET_OMEGA * DT * (step + 1)
        )

        future_target = target.position(
            future_phase
        )

        if model == "oracle":

            # Ground-truth dynamics for each candidate.
            next_states = np.asarray(
                [
                    true_transition(
                        states[index],
                        actions[index],
                    )
                    for index in range(candidate_count)
                ]
            )

        elif isinstance(
            model,
            LearnedDynamics,
        ):

            deltas = model.predict_delta_batch(
                states,
                actions,
            )

            next_states = np.asarray(
                [
                    clip_state(
                        states[index] + deltas[index]
                    )
                    for index in range(candidate_count)
                ]
            )

        else:

            raise ValueError(
                "Unsupported dynamics model"
            )

        for index in range(candidate_count):

            costs[index] += stage_cost(
                next_states[index],
                future_target,
                actions[index],
            )

        states = next_states

    final_phase = (
        phase
        + TARGET_OMEGA
        * DT
        * action_sequences.shape[1]
    )

    final_target = target.position(
        final_phase
    )

    for index in range(candidate_count):

        costs[index] += terminal_cost(
            states[index],
            final_target,
        )

    return costs


# ============================================================
# 9. NOMINAL ACTION SEQUENCE
# ============================================================

def make_nominal_sequence(
    state,
    target,
    phase,
    horizon,
):

    """
    Construct a nominal sequence using a PD controller.

    This seed is common to learned MPPI and Oracle MPPI.
    It does not use the learned or true model for its rollout.
    """

    nominal = np.zeros(
        (horizon, 2),
        dtype=np.float64,
    )

    estimated_state = state.copy()

    for step in range(horizon):

        future_phase = (
            phase
            + TARGET_OMEGA * DT * step
        )

        action = pd_controller(
            estimated_state,
            target.position(future_phase),
            target.velocity(future_phase),
        )

        nominal[step] = action

        # Approximate kinematic seed propagation.
        estimated_state[2:4] += (
            MAX_ACCEL * action * DT
        )

        speed = np.linalg.norm(
            estimated_state[2:4]
        )

        if speed > MAX_SPEED:

            estimated_state[2:4] *= (
                MAX_SPEED / (speed + 1e-9)
            )

        estimated_state[:2] += (
            estimated_state[2:4] * DT
        )

        estimated_state = clip_state(
            estimated_state
        )

    return nominal


# ============================================================
# 10. RANDOM-SHOOTING MPC
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
            size=(
                samples,
                horizon,
                2,
            ),
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
# 11. MPPI CONTROLLER
# ============================================================

class MPPIController:

    """
    Simplified MPPI.

    At every iteration:

        1. Sample action sequences around nominal.
        2. Evaluate all trajectories.
        3. Convert costs into normalized exponential weights.
        4. Update nominal sequence using weighted averaging.

    Stable weights:

        w = softmax(-(cost - min(cost)) / lambda)

    Receding horizon:
        Execute the first action only.
        Shift the optimized sequence and replan.
    """

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

        self.last_weights = None
        self.last_costs = None

    def reset(self):

        self.nominal.fill(
            0.0
        )

        self.last_weights = None
        self.last_costs = None

    def plan(
        self,
        state,
        target,
        phase,
    ):

        if not np.any(self.nominal):

            self.nominal = make_nominal_sequence(
                state,
                target,
                phase,
                self.horizon,
            )

        nominal = self.nominal.copy()

        for _ in range(
            self.iterations
        ):

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

            # Preserve the current nominal sequence as one candidate.
            candidates[0] = nominal

            costs = evaluate_action_sequences(
                state,
                candidates,
                target,
                phase,
                self.model,
            )

            # Numerically stable path-integral weights.
            minimum_cost = np.min(
                costs
            )

            logits = (
                -(costs - minimum_cost)
                / max(
                    self.temperature,
                    1e-6,
                )
            )

            logits -= np.max(
                logits
            )

            weights = np.exp(
                logits
            )

            weights /= (
                np.sum(weights) + 1e-12
            )

            # Weighted average of action sequences.
            nominal = np.sum(
                weights[:, None, None]
                * candidates,
                axis=0,
            )

            nominal = np.clip(
                nominal,
                -1.0,
                1.0,
            )

            self.last_weights = weights
            self.last_costs = costs

        action = nominal[0].copy()

        # Receding-horizon warm start:
        # shift the plan one step into the future.
        self.nominal[:-1] = nominal[1:]

        self.nominal[-1] = nominal[-1]

        return (
            action,
            nominal.copy(),
            float(
                np.min(costs)
            ),
        )


# ============================================================
# 12. BENCHMARKING
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
    collisions = []

    for _ in range(steps):

        target_position = (
            env.target.position()
        )

        target_velocity = (
            env.target.velocity()
        )

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

            raise ValueError(
                mode
            )

        state = env.step(
            action
        )

        future_target = (
            env.target.position()
        )

        distances.append(
            np.linalg.norm(
                state[:2] - future_target
            )
        )

        clearance = (
            np.linalg.norm(
                state[:2] - OBSTACLE
            )
            - OBSTACLE_RADIUS
        )

        collisions.append(
            clearance < 0.0
        )

    return {
        "mean_distance": float(
            np.mean(distances)
        ),

        "collision_steps": int(
            np.sum(collisions)
        ),

        "distances": np.asarray(
            distances
        ),
    }


def benchmark(
    learned_model,
):

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

        for episode in range(
            BENCHMARK_EPISODES
        ):

            result = run_episode(
                mode,
                learned_model,
            )

            episode_results.append(
                result
            )

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
# 13. VISUALIZATION
# ============================================================

def plot_training_loss(model):

    plt.figure(
        figsize=(9, 5)
    )

    plt.plot(
        model.loss_history
    )

    plt.title(
        "Learned Dynamics Model Training Loss"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Normalized MSE"
    )

    plt.grid(True)

    plt.tight_layout()

    plt.show()


def plot_benchmark(results):

    names = list(
        results.keys()
    )

    distances = [
        results[name]["mean_distance"]
        for name in names
    ]

    collisions = [
        results[name]["collision_steps"]
        for name in names
    ]

    plt.figure(
        figsize=(9, 5)
    )

    plt.bar(
        names,
        distances,
    )

    plt.title(
        "Moving-Target Tracking Distance"
    )

    plt.ylabel(
        "Mean distance"
    )

    plt.grid(
        axis="y"
    )

    plt.tight_layout()

    plt.show()

    plt.figure(
        figsize=(9, 5)
    )

    plt.bar(
        names,
        collisions,
    )

    plt.title(
        "Obstacle Collision Steps"
    )

    plt.ylabel(
        "Average collision steps per episode"
    )

    plt.grid(
        axis="y"
    )

    plt.tight_layout()

    plt.show()


# ============================================================
# 14. INTERACTIVE PYGAME
# ============================================================

def run_pygame(
    learned_model,
):

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
        "Phase 4.27 — MPPI Drone Control"
    )

    clock = pygame.time.Clock()

    font = pygame.font.SysFont(
        "consolas",
        15,
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
        learned_model
    )

    oracle_mppi = MPPIController(
        "oracle"
    )

    paused = False
    speed = 1

    trail = []
    predicted_path = []

    current_weights = None

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

    def draw_drone():

        center = state[:2]

        velocity = state[2:4]

        norm = np.linalg.norm(
            velocity
        )

        direction = (
            velocity / norm
            if norm > 1e-6
            else np.array([1.0, 0.0])
        )

        side = np.array(
            [
                -direction[1],
                direction[0],
            ]
        )

        for forward in (-1, 1):

            for lateral in (-1, 1):

                end = (
                    center
                    + 17 * forward * direction
                    + 17 * lateral * side
                )

                pygame.draw.line(
                    screen,
                    (220, 225, 235),
                    center.astype(int),
                    end.astype(int),
                    5,
                )

                pygame.draw.circle(
                    screen,
                    (90, 190, 245),
                    end.astype(int),
                    7,
                    2,
                )

        pygame.draw.circle(
            screen,
            (250, 250, 250),
            center.astype(int),
            8,
        )

        pygame.draw.circle(
            screen,
            (255, 190, 60),
            (
                center + direction * 22
            ).astype(int),
            4,
        )

    def draw_target():

        points = []

        for index in range(100):

            phase = (
                env.target.phase
                + 2.0 * math.pi * index / 100
            )

            points.append(
                env.target.position(
                    phase
                ).astype(int)
            )

        pygame.draw.lines(
            screen,
            (65, 105, 75),
            True,
            points,
            2,
        )

        position = (
            env.target.position()
        )

        velocity = (
            env.target.velocity()
        )

        pygame.draw.circle(
            screen,
            (70, 240, 110),
            position.astype(int),
            11,
        )

        pygame.draw.line(
            screen,
            (70, 255, 120),
            position.astype(int),
            (
                position + 0.18 * velocity
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

                    state = env.reset()

                    learned_mppi.reset()
                    oracle_mppi.reset()

                    trail.clear()
                    predicted_path.clear()

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

                    speed = min(
                        5,
                        speed + 1,
                    )

                elif event.key == pygame.K_MINUS:

                    speed = max(
                        1,
                        speed - 1,
                    )

            elif event.type == pygame.MOUSEBUTTONDOWN:

                if event.button == 1:

                    click = np.asarray(
                        event.pos,
                        dtype=np.float64,
                    )

                    if (
                        np.linalg.norm(
                            click - OBSTACLE
                        )
                        > SAFE_RADIUS
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

        mode = controllers[
            selected
        ]

        if not paused:

            for _ in range(speed):

                if mode == "PD":

                    action = pd_controller(
                        state,
                        env.target.position(),
                        env.target.velocity(),
                    )

                    predicted_path = []
                    current_weights = None

                elif mode == "Random MPC":

                    action, sequence, _ = (
                        random_shooting_action(
                            state,
                            env.target,
                            env.target.phase,
                            learned_model,
                            samples=100,
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

                    action, sequence, _ = (
                        controller.plan(
                            state,
                            env.target,
                            env.target.phase,
                        )
                    )

                    current_weights = (
                        controller.last_weights
                    )

                    predicted_state = (
                        state.copy()
                    )

                    predicted_path = []

                    for future_action in sequence:

                        if mode == "Oracle MPPI":

                            predicted_state = (
                                true_transition(
                                    predicted_state,
                                    future_action,
                                )
                            )

                        else:

                            predicted_state = (
                                learned_model.predict_next(
                                    predicted_state,
                                    future_action,
                                )
                            )

                        predicted_path.append(
                            predicted_state[:2].copy()
                        )

                state = env.step(
                    action
                )

                trail.append(
                    state[:2].copy()
                )

                if len(trail) > 200:

                    trail.pop(0)

        screen.fill(
            (17, 23, 34)
        )

        for x in range(
            0,
            WORLD_W,
            50,
        ):

            pygame.draw.line(
                screen,
                (30, 38, 50),
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

        if len(predicted_path) > 1:

            pygame.draw.lines(
                screen,
                (245, 190, 60),
                False,
                [
                    point.astype(int)
                    for point in predicted_path
                ],
                2,
            )

        draw_drone()

        distance = np.linalg.norm(
            state[:2] - env.target.position()
        )

        clearance = (
            np.linalg.norm(
                state[:2] - OBSTACLE
            )
            - OBSTACLE_RADIUS
        )

        draw_text(
            f"Controller: {mode}",
            15,
            12,
        )

        draw_text(
            f"Target distance: {distance:.2f}",
            15,
            38,
        )

        draw_text(
            f"Obstacle clearance: {clearance:.2f}",
            15,
            64,
        )

        draw_text(
            f"Drone velocity: ({state[2]:.1f}, {state[3]:.1f})",
            15,
            90,
        )

        draw_text(
            f"Simulation speed: {speed}x",
            15,
            116,
        )

        draw_text(
            "Yellow line: planned future trajectory",
            15,
            142,
            (255, 205, 95),
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
                f"Effective weighted samples: {effective_samples:.1f}",
                15,
                168,
                (255, 205, 95),
            )

        draw_text(
            "1 PD | 2 Random MPC | 3 Learned MPPI | 4 Oracle MPPI",
            15,
            WORLD_H - 100,
        )

        draw_text(
            "SPACE pause | R reset | +/- speed | "
            "click target center | ESC exit",
            15,
            WORLD_H - 65,
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
# 15. UNIT TESTS
# ============================================================

def run_tests():

    print(
        "\nRunning unit tests..."
    )

    passed = 0

    def check(condition, description):

        nonlocal passed

        assert condition, description

        passed += 1

        print(
            f"[PASS] {description}"
        )

    # Test 1 — State transition.
    state = np.array(
        [100.0, 200.0, 10.0, -5.0]
    )

    next_state = true_transition(
        state,
        np.array([0.2, -0.1]),
    )

    check(
        next_state.shape == (4,)
        and np.all(np.isfinite(next_state)),
        "True dynamics return a finite four-dimensional state",
    )

    # Test 2 — Action clipping.
    clipped = true_transition(
        state,
        np.array([5.0, -5.0]),
    )

    bounded = true_transition(
        state,
        np.array([1.0, -1.0]),
    )

    check(
        np.allclose(
            clipped,
            bounded,
        ),
        "Actions are clipped to valid bounds",
    )

    # Test 3 — Speed clipping.
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
        "Speed constraint is enforced",
    )

    # Test 4 — Position clipping.
    check(
        DRONE_RADIUS <= bounded_state[0]
        <= WORLD_W - DRONE_RADIUS
        and DRONE_RADIUS <= bounded_state[1]
        <= WORLD_H - DRONE_RADIUS,
        "World boundary constraint is enforced",
    )

    # Test 5 — Moving target.
    target = MovingTarget(
        [700.0, 300.0]
    )

    before = target.position().copy()

    target.advance()

    after = target.position()

    check(
        not np.allclose(before, after),
        "Target position changes over time",
    )

    # Test 6 — Dataset dimensions.
    inputs, targets = collect_dataset(
        steps=100
    )

    check(
        inputs.shape == (100, 6)
        and targets.shape == (100, 4),
        "Transition dataset has correct dimensions",
    )

    # Test 7 — Dataset finite.
    check(
        np.all(np.isfinite(inputs))
        and np.all(np.isfinite(targets)),
        "Dataset contains finite values",
    )

    # Test 8 — Normalizer round trip.
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
        "Normalizer round trip is consistent",
    )

    # Test 9 — Neural network output shape.
    network = DynamicsNetwork().to(
        DEVICE
    )

    test_tensor = torch.zeros(
        (10, INPUT_DIM),
        device=DEVICE,
    )

    output = network(
        test_tensor
    )

    check(
        output.shape == (10, 4),
        "Dynamics network output shape is correct",
    )

    # Test 10 — Neural output finite.
    check(
        torch.all(
            torch.isfinite(output)
        ),
        "Dynamics network output is finite",
    )

    # Test 11 — Stage cost.
    cost = stage_cost(
        state,
        np.array([300.0, 300.0]),
        np.array([0.1, 0.2]),
    )

    check(
        np.isfinite(cost),
        "MPPI stage cost is finite",
    )

    # Test 12 — Nominal sequence.
    nominal = make_nominal_sequence(
        state,
        target,
        target.phase,
        HORIZON,
    )

    check(
        nominal.shape == (HORIZON, 2),
        "Nominal action sequence has correct shape",
    )

    # Test 13 — Nominal action bounds.
    check(
        np.all(nominal <= 1.0)
        and np.all(nominal >= -1.0),
        "Nominal action sequence is bounded",
    )

    # Test 14 — Sequence rollout.
    env = DroneEnv()

    env.reset()

    test_sequences = np.repeat(
        nominal[None, :, :],
        5,
        axis=0,
    )

    rollout_costs = evaluate_action_sequences(
        env.state,
        test_sequences,
        env.target,
        env.target.phase,
        "oracle",
    )

    check(
        rollout_costs.shape == (5,)
        and np.all(np.isfinite(rollout_costs)),
        "Batched Oracle rollout returns finite candidate costs",
    )

    # Test 15 — Weight normalization.
    test_costs = np.array(
        [10.0, 11.0, 12.0, 20.0]
    )

    logits = -(
        test_costs - test_costs.min()
    ) / LAMBDA

    logits -= logits.max()

    weights = np.exp(
        logits
    )

    weights /= weights.sum()

    check(
        np.isclose(
            weights.sum(),
            1.0,
        ),
        "MPPI weights sum to one",
    )

    # Test 16 — Lower cost receives higher weight.
    check(
        weights[0] > weights[1] > weights[2] > weights[3],
        "Lower-cost sequences receive higher weights",
    )

    # Test 17 — Weighted update.
    candidates = np.array(
        [
            [[0.0, 0.0]],
            [[1.0, 1.0]],
        ]
    )

    sample_weights = np.array(
        [0.75, 0.25]
    )

    updated = np.sum(
        sample_weights[:, None, None]
        * candidates,
        axis=0,
    )

    check(
        np.allclose(
            updated,
            [[0.25, 0.25]],
        ),
        "Weighted action-sequence update is correct",
    )

    # Test 18 — MPPI controller interface.
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
        "MPPI planner returns valid action, sequence and cost",
    )

    # Test 19 — MPPI action bounds.
    check(
        np.all(action >= -1.0)
        and np.all(action <= 1.0),
        "MPPI action is bounded",
    )

    # Test 20 — MPPI weights.
    check(
        controller.last_weights is not None
        and np.isclose(
            controller.last_weights.sum(),
            1.0,
        ),
        "MPPI produces normalized sample weights",
    )

    print(
        f"\n{passed}/20 tests passed."
    )

    return passed == 20


# ============================================================
# 16. MAIN
# ============================================================

def main():

    print(
        "=" * 72
    )

    print(
        "PHASE 4.27 — MODEL PREDICTIVE PATH INTEGRAL CONTROL"
    )

    print(
        "=" * 72
    )

    print(
        f"Device: {DEVICE}"
    )

    # Step 1 — Unit tests.
    if not run_tests():

        raise RuntimeError(
            "Unit tests failed."
        )

    # Step 2 — Collect dynamics dataset.
    print(
        "\n[1/6] Collecting transition dataset..."
    )

    inputs, targets = collect_dataset()

    print(
        f"Collected {len(inputs)} transitions."
    )

    # Step 3 — Train learned dynamics.
    print(
        "\n[2/6] Training learned dynamics model..."
    )

    learned_model = LearnedDynamics()

    learned_model.fit(
        inputs,
        targets,
    )

    plot_training_loss(
        learned_model
    )

    # Step 4 — Benchmark controllers.
    print(
        "\n[3/6] Benchmarking controllers..."
    )

    results = benchmark(
        learned_model
    )

    plot_benchmark(
        results
    )

    # Step 5 — Report MPPI parameters.
    print(
        "\n[4/6] MPPI configuration"
    )

    print(
        f"Horizon: {HORIZON}"
    )

    print(
        f"Samples: {SAMPLE_COUNT}"
    )

    print(
        f"Optimization iterations: {MPPI_ITERATIONS}"
    )

    print(
        f"Temperature lambda: {LAMBDA}"
    )

    print(
        f"Action noise std: {NOISE_STD}"
    )

    # Step 6 — Interactive visualization.
    print(
        "\n[5/6] Starting interactive simulation..."
    )

    print(
        "1 = PD"
    )

    print(
        "2 = Random shooting MPC"
    )

    print(
        "3 = Learned-model MPPI"
    )

    print(
        "4 = Oracle MPPI"
    )

    print(
        "SPACE = pause"
    )

    print(
        "R = reset"
    )

    print(
        "+/- = simulation speed"
    )

    print(
        "Mouse click = move target trajectory center"
    )

    print(
        "ESC = exit"
    )

    run_pygame(
        learned_model
    )

    print(
        "\n[6/6] Phase 4.27 completed."
    )


if __name__ == "__main__":

    main()
