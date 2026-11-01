
"""
File: phase4/neural_dynamics_mpc_drone.py

PHASE 4.26 — NEURAL NETWORK SYSTEM IDENTIFICATION

PURPOSE
-------
Learn drone dynamics from collected state-action transitions,
then use the learned model to plan movement toward a moving target.

WHAT YOU LEARN
--------------

STEP 1 — Data Collection
    Collect drone state, action and next-state transitions.

STEP 2 — Linear Dynamics Baseline
    Fit a linear regression model to the transition data.

STEP 3 — Neural Dynamics Model
    Train an MLP to predict the next drone state.

STEP 4 — Prediction Evaluation
    Compare one-step and multi-step prediction errors.

STEP 5 — Learned-Model MPC
    Generate candidate action sequences and evaluate them
    using the learned dynamics model.

STEP 6 — Oracle Comparison
    Compare learned-model planning against planning with
    the simulator's true dynamics.

MATHEMATICAL FOUNDATIONS
------------------------

Drone state:
    s = [x, y, vx, vy]

Control:
    u = [ux, uy]

Unknown dynamics:
    a = k*A*u - c*v - q*abs(v)*v + wind

State transition:
    v_next = v + a*dt
    p_next = p + v_next*dt

The learned model predicts the state difference:

    delta_s = f_theta(s, u)

Training objective:
    L(theta) = mean(||f_theta(s,u) - delta_s||^2)

Linear baseline:
    delta_s = X*W + b

Multi-step prediction:
    s_(t+1) = s_t + f_theta(s_t,u_t)

MPC:
    u* = argmin_(u_0,...,u_(H-1)) J

Only the first action is executed.
The model is queried again at the next time step.

ROBOTICS APPLICATION
--------------------
A drone may have unknown motor response, aerodynamic drag,
payload and environmental disturbances.

A learned dynamics model can support:
    - trajectory prediction
    - model predictive control
    - motion planning
    - sim-to-real adaptation
    - model-based reinforcement learning

IMPORTANT LIMITATIONS
---------------------
This is a simplified 2D educational model.

The learned model is only reliable within the distribution
of states and actions represented in its training data.

Long-horizon predictions can accumulate model errors.

The Oracle MPC is a simulator reference, not a real-world
controller.

Obstacle penalties in MPC are soft penalties, not a formal
collision-avoidance guarantee.
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

SEED = 426

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

# Hidden physical parameters.
MOTOR_GAIN = 0.72
LINEAR_DRAG = 0.115
QUADRATIC_DRAG = 0.00032

WIND = np.array(
    [5.0, -3.5],
    dtype=np.float64,
)

STATE_DIM = 4
MODEL_INPUT_DIM = 6
MODEL_OUTPUT_DIM = 4

HIDDEN_DIM = 128

DATA_STEPS = 9000
TRAIN_EPOCHS = 180
BATCH_SIZE = 256
LEARNING_RATE = 0.001

MPC_HORIZON = 14
MPC_CANDIDATES = 160

SIMULATION_EPISODES = 4
SIMULATION_STEPS = 220


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

@dataclass
class DroneState:

    x: float
    y: float
    vx: float
    vy: float

    def array(self):

        return np.array(
            [
                self.x,
                self.y,
                self.vx,
                self.vy,
            ],
            dtype=np.float64,
        )


def clip_state(state):

    state = np.asarray(
        state,
        dtype=np.float64,
    ).copy()

    velocity = state[2:4]

    speed = np.linalg.norm(
        velocity
    )

    if speed > MAX_SPEED:

        velocity *= (
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
    Ground-truth simulator dynamics.

    This function is used by the environment and Oracle MPC.
    The learned model does not receive the hidden parameters.
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

    next_state = np.concatenate(
        [
            next_position,
            next_velocity,
        ]
    )

    return clip_state(
        next_state
    )


# ============================================================
# 3. DRONE ENVIRONMENT
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

        old_state = self.state.copy()

        self.state = true_transition(
            self.state,
            action,
        )

        self.target.advance()

        return (
            self.state.copy(),
            self.target.position(),
            self.state - old_state,
        )

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
# 4. REFERENCE CONTROLLER FOR DATA COLLECTION
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

    # Approximate inverse dynamics for data collection.
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
# 5. COLLECT TRANSITION DATA
# ============================================================

def collect_dataset(
    steps=DATA_STEPS,
):

    """
    Dataset:

        X = [x, y, vx, vy, ux, uy]

        Y = [delta_x, delta_y, delta_vx, delta_vy]

    Collection uses a mixture of:
        - PD tracking
        - random actions
        - noisy PD actions

    This improves coverage compared with collecting only
    one deterministic trajectory.
    """

    env = DroneEnv()

    inputs = []
    targets = []

    state = env.reset()

    for step in range(steps):

        target_position = (
            env.target.position()
        )

        target_velocity = (
            env.target.velocity()
        )

        pd_action = pd_controller(
            state,
            target_position,
            target_velocity,
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

        next_state, _, _ = env.step(
            action
        )

        delta_state = (
            next_state - state
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
            delta_state
        )

        state = next_state

        # Periodically restart to improve initial-state coverage.
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
# 6. DATA NORMALIZATION
# ============================================================

@dataclass
class Normalizer:

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, values):

        mean = values.mean(
            axis=0
        )

        std = values.std(
            axis=0
        )

        std = np.maximum(
            std,
            1e-6,
        )

        return cls(
            mean=mean,
            std=std,
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
# 7. LINEAR BASELINE
# ============================================================

class LinearDynamicsModel:

    """
    Baseline:

        delta_state = X @ W + b

    The linear model cannot represent arbitrary nonlinear
    dynamics, but it provides a useful reference.
    """

    def __init__(self):

        self.weights = None

    def fit(
        self,
        inputs,
        targets,
    ):

        design = np.concatenate(
            [
                inputs,
                np.ones(
                    (len(inputs), 1)
                ),
            ],
            axis=1,
        )

        self.weights, *_ = np.linalg.lstsq(
            design,
            targets,
            rcond=None,
        )

    def predict(self, inputs):

        design = np.concatenate(
            [
                inputs,
                np.ones(
                    (len(inputs), 1)
                ),
            ],
            axis=1,
        )

        return design @ self.weights


# ============================================================
# 8. NEURAL DYNAMICS MODEL
# ============================================================

class DynamicsNetwork(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                MODEL_INPUT_DIM,
                HIDDEN_DIM,
            ),

            nn.ReLU(),

            nn.Linear(
                HIDDEN_DIM,
                HIDDEN_DIM,
            ),

            nn.ReLU(),

            nn.Linear(
                HIDDEN_DIM,
                HIDDEN_DIM,
            ),

            nn.ReLU(),

            nn.Linear(
                HIDDEN_DIM,
                MODEL_OUTPUT_DIM,
            ),
        )

    def forward(self, x):

        return self.network(x)


class LearnedDynamicsModel:

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

        input_train = inputs
        target_train = targets

        self.input_normalizer = (
            Normalizer.fit(input_train)
        )

        self.output_normalizer = (
            Normalizer.fit(target_train)
        )

        x = self.input_normalizer.transform(
            input_train
        )

        y = self.output_normalizer.transform(
            target_train
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

        loss_function = nn.MSELoss()

        for epoch in range(epochs):

            epoch_losses = []

            for batch_x, batch_y in loader:

                prediction = self.model(
                    batch_x
                )

                loss = loss_function(
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

                epoch_losses.append(
                    loss.item()
                )

            average_loss = float(
                np.mean(epoch_losses)
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

    def predict_delta(
        self,
        state,
        action,
    ):

        model_input = np.concatenate(
            [
                state,
                action,
            ]
        ).astype(
            np.float32
        )

        normalized_input = (
            self.input_normalizer.transform(
                model_input
            )
        )

        tensor = torch.as_tensor(
            normalized_input,
            dtype=torch.float32,
            device=DEVICE,
        ).unsqueeze(0)

        with torch.no_grad():

            normalized_prediction = (
                self.model(tensor)
                .cpu()
                .numpy()[0]
            )

        return self.output_normalizer.inverse(
            normalized_prediction
        )

    def predict_next(
        self,
        state,
        action,
    ):

        delta = self.predict_delta(
            state,
            action,
        )

        return clip_state(
            state + delta
        )


# ============================================================
# 9. MODEL EVALUATION
# ============================================================

def split_dataset(
    inputs,
    targets,
    train_fraction=0.8,
):

    indices = np.random.permutation(
        len(inputs)
    )

    split = int(
        train_fraction * len(inputs)
    )

    train_indices = indices[:split]
    test_indices = indices[split:]

    return (
        inputs[train_indices],
        targets[train_indices],
        inputs[test_indices],
        targets[test_indices],
    )


def evaluate_one_step(
    model,
    test_inputs,
    test_targets,
):

    if isinstance(
        model,
        LearnedDynamicsModel,
    ):

        predictions = []

        for row in test_inputs:

            predictions.append(
                model.predict_delta(
                    row[:4],
                    row[4:6],
                )
            )

        predictions = np.asarray(
            predictions
        )

    else:

        predictions = model.predict(
            test_inputs
        )

    errors = (
        predictions - test_targets
    )

    component_rmse = np.sqrt(
        np.mean(
            errors**2,
            axis=0,
        )
    )

    total_rmse = float(
        np.sqrt(
            np.mean(
                errors**2
            )
        )
    )

    return (
        total_rmse,
        component_rmse,
        predictions,
    )


def rollout_prediction_error(
    model,
    episodes=4,
    steps=100,
):

    """
    Multi-step rollout test.

    The model receives its own predicted state after the first
    transition. This reveals accumulated model error.

    Target motion is not predicted by the learned drone model:
    it is known analytically and advances independently.
    """

    episode_errors = []

    for _ in range(episodes):

        env = DroneEnv()

        true_state = env.reset()

        predicted_state = (
            true_state.copy()
        )

        errors = []

        for step in range(steps):

            target_position = (
                env.target.position()
            )

            target_velocity = (
                env.target.velocity()
            )

            action = pd_controller(
                true_state,
                target_position,
                target_velocity,
            )

            next_true_state, _, _ = env.step(
                action
            )

            if isinstance(
                model,
                LearnedDynamicsModel,
            ):

                predicted_state = (
                    model.predict_next(
                        predicted_state,
                        action,
                    )
                )

            elif isinstance(
                model,
                LinearDynamicsModel,
            ):

                delta = model.predict(
                    np.concatenate(
                        [
                            predicted_state,
                            action,
                        ]
                    )[None, :]
                )[0]

                predicted_state = clip_state(
                    predicted_state + delta
                )

            elif model == "oracle":

                predicted_state = true_transition(
                    predicted_state,
                    action,
                )

            errors.append(
                np.linalg.norm(
                    predicted_state - next_true_state
                )
            )

            true_state = (
                next_true_state.copy()
            )

        episode_errors.append(
            np.asarray(errors)
        )

    return np.mean(
        np.asarray(episode_errors),
        axis=0,
    )


# ============================================================
# 10. MPC COST
# ============================================================

def tracking_cost(
    state,
    target_position,
    action,
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

    boundary_distance = min(
        position[0],
        WORLD_W - position[0],
        position[1],
        WORLD_H - position[1],
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


# ============================================================
# 11. NOMINAL ACTION SEQUENCE
# ============================================================

def make_nominal_sequence(
    initial_state,
    target,
    phase,
):

    """
    Use a PD controller to create a nominal action sequence.

    This sequence does not use the learned model or Oracle
    dynamics. It is a common starting point for all planners.
    """

    state = initial_state.copy()

    actions = []

    for step in range(MPC_HORIZON):

        future_phase = (
            phase
            + TARGET_OMEGA * DT * step
        )

        target_position = (
            target.position(
                future_phase
            )
        )

        target_velocity = (
            target.velocity(
                future_phase
            )
        )

        action = pd_controller(
            state,
            target_position,
            target_velocity,
        )

        actions.append(
            action
        )

        # Use a neutral kinematic approximation only for
        # creating the nominal seed. Candidate scoring uses
        # the selected dynamics model.
        state = state.copy()

        state[2:4] += (
            MAX_ACCEL * action * DT
        )

        state[2:4] *= 0.98

        speed = np.linalg.norm(
            state[2:4]
        )

        if speed > MAX_SPEED:

            state[2:4] *= (
                MAX_SPEED / (speed + 1e-9)
            )

        state[:2] += (
            state[2:4] * DT
        )

        state = clip_state(
            state
        )

    return np.asarray(
        actions
    )


# ============================================================
# 12. MODEL-BASED MPC
# ============================================================

def rollout_candidate(
    initial_state,
    actions,
    target,
    phase,
    model,
):

    state = initial_state.copy()

    total_cost = 0.0

    for step, action in enumerate(actions):

        future_phase = (
            phase
            + TARGET_OMEGA * DT * (step + 1)
        )

        future_target = (
            target.position(
                future_phase
            )
        )

        if model == "oracle":

            next_state = true_transition(
                state,
                action,
            )

        elif isinstance(
            model,
            LearnedDynamicsModel,
        ):

            next_state = model.predict_next(
                state,
                action,
            )

        elif isinstance(
            model,
            LinearDynamicsModel,
        ):

            delta = model.predict(
                np.concatenate(
                    [
                        state,
                        action,
                    ]
                )[None, :]
            )[0]

            next_state = clip_state(
                state + delta
            )

        else:

            raise ValueError(
                "Unknown dynamics model"
            )

        total_cost += tracking_cost(
            next_state,
            future_target,
            action,
        )

        state = next_state

    return total_cost


def choose_mpc_action(
    state,
    target,
    phase,
    model,
    candidates=MPC_CANDIDATES,
):

    nominal = make_nominal_sequence(
        state,
        target,
        phase,
    )

    action_sequences = np.repeat(
        nominal[None, :, :],
        candidates,
        axis=0,
    )

    # Candidate perturbations around the nominal sequence.
    noise = np.random.normal(
        0.0,
        0.28,
        size=action_sequences.shape,
    )

    action_sequences += noise

    action_sequences = np.clip(
        action_sequences,
        -1.0,
        1.0,
    )

    # Keep the unperturbed nominal sequence as candidate 0.
    action_sequences[0] = nominal

    costs = np.zeros(
        candidates,
        dtype=np.float64,
    )

    for index in range(candidates):

        costs[index] = rollout_candidate(
            state,
            action_sequences[index],
            target,
            phase,
            model,
        )

    best_index = int(
        np.argmin(costs)
    )

    return (
        action_sequences[best_index, 0].copy(),
        action_sequences[best_index].copy(),
        float(costs[best_index]),
    )


# ============================================================
# 13. BENCHMARK CONTROLLERS
# ============================================================

def run_controller_episode(
    mode,
    learned_model=None,
    linear_model=None,
    steps=SIMULATION_STEPS,
):

    env = DroneEnv()

    state = env.reset()

    distances = []
    collisions = []
    prediction_costs = []

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

        elif mode == "Learned MPC":

            action, _, cost = choose_mpc_action(
                state,
                env.target,
                env.target.phase,
                learned_model,
            )

            prediction_costs.append(
                cost
            )

        elif mode == "Linear MPC":

            action, _, cost = choose_mpc_action(
                state,
                env.target,
                env.target.phase,
                linear_model,
            )

            prediction_costs.append(
                cost
            )

        elif mode == "Oracle MPC":

            action, _, cost = choose_mpc_action(
                state,
                env.target,
                env.target.phase,
                "oracle",
            )

            prediction_costs.append(
                cost
            )

        else:

            raise ValueError(
                f"Unknown mode: {mode}"
            )

        old_position = state[:2].copy()

        state, _, _ = env.step(
            action
        )

        # Compare against the same future target position.
        future_target = (
            env.target.position()
        )

        distance = np.linalg.norm(
            state[:2] - future_target
        )

        distances.append(
            distance
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

        "prediction_costs": np.asarray(
            prediction_costs
        ),
    }


def benchmark_controllers(
    learned_model,
    linear_model,
):

    modes = [
        "PD",
        "Linear MPC",
        "Learned MPC",
        "Oracle MPC",
    ]

    results = {}

    for mode in modes:

        print(
            f"\nBenchmarking {mode}..."
        )

        episodes = []

        for episode in range(
            SIMULATION_EPISODES
        ):

            result = run_controller_episode(
                mode,
                learned_model,
                linear_model,
            )

            episodes.append(
                result
            )

        results[mode] = {
            "mean_distance": float(
                np.mean(
                    [
                        item["mean_distance"]
                        for item in episodes
                    ]
                )
            ),

            "collision_steps": float(
                np.mean(
                    [
                        item["collision_steps"]
                        for item in episodes
                    ]
                )
            ),
        }

        print(
            f"{mode:15s}"
            f" | mean distance={results[mode]['mean_distance']:8.2f}"
            f" | collision steps={results[mode]['collision_steps']:7.2f}"
        )

    return results


# ============================================================
# 14. VISUALIZATION
# ============================================================

def plot_training_loss(model):

    plt.figure(
        figsize=(9, 5)
    )

    plt.plot(
        model.loss_history
    )

    plt.title(
        "Neural Dynamics Model Training Loss"
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


def plot_one_step_errors(
    linear_rmse,
    neural_rmse,
    linear_components,
    neural_components,
):

    names = [
        "x",
        "y",
        "vx",
        "vy",
    ]

    x = np.arange(
        len(names)
    )

    width = 0.35

    plt.figure(
        figsize=(9, 5)
    )

    plt.bar(
        x - width / 2,
        linear_components,
        width,
        label="Linear model",
    )

    plt.bar(
        x + width / 2,
        neural_components,
        width,
        label="Neural model",
    )

    plt.xticks(
        x,
        names,
    )

    plt.title(
        "One-Step Prediction RMSE by State Component"
    )

    plt.ylabel(
        "RMSE"
    )

    plt.grid(
        axis="y"
    )

    plt.legend()

    plt.tight_layout()

    plt.show()

    print(
        f"Linear overall RMSE: {linear_rmse:.6f}"
    )

    print(
        f"Neural overall RMSE: {neural_rmse:.6f}"
    )


def plot_rollout_errors(
    linear_errors,
    neural_errors,
    oracle_errors,
):

    plt.figure(
        figsize=(10, 5)
    )

    plt.plot(
        linear_errors,
        label="Linear model",
    )

    plt.plot(
        neural_errors,
        label="Neural model",
    )

    plt.plot(
        oracle_errors,
        label="Oracle model",
    )

    plt.title(
        "Multi-Step Model Prediction Error"
    )

    plt.xlabel(
        "Rollout step"
    )

    plt.ylabel(
        "State prediction error"
    )

    plt.grid(True)

    plt.legend()

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
# 15. INTERACTIVE PYGAME
# ============================================================

def run_pygame(
    learned_model,
    linear_model,
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
        "Phase 4.26 — Neural Dynamics MPC Drone"
    )

    clock = pygame.time.Clock()

    font = pygame.font.SysFont(
        "consolas",
        15,
    )

    env = DroneEnv()

    state = env.reset()

    modes = {
        1: (
            "PD",
            None,
        ),

        2: (
            "Linear MPC",
            linear_model,
        ),

        3: (
            "Neural MPC",
            learned_model,
        ),

        4: (
            "Oracle MPC",
            "oracle",
        ),
    }

    selected = 3

    paused = False
    speed = 1

    trail = []
    predicted_path = []

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

        mode_name, model = modes[
            selected
        ]

        if not paused:

            for _ in range(speed):

                if mode_name == "PD":

                    action = pd_controller(
                        state,
                        env.target.position(),
                        env.target.velocity(),
                    )

                    predicted_path = []

                else:

                    action, sequence, _ = (
                        choose_mpc_action(
                            state,
                            env.target,
                            env.target.phase,
                            model,
                            candidates=100,
                        )
                    )

                    # Draw the selected model's predicted path.
                    predicted_path = []

                    predicted_state = (
                        state.copy()
                    )

                    for step, future_action in enumerate(
                        sequence
                    ):

                        if model == "oracle":

                            predicted_state = (
                                true_transition(
                                    predicted_state,
                                    future_action,
                                )
                            )

                        elif isinstance(
                            model,
                            LearnedDynamicsModel,
                        ):

                            predicted_state = (
                                model.predict_next(
                                    predicted_state,
                                    future_action,
                                )
                            )

                        else:

                            delta = model.predict(
                                np.concatenate(
                                    [
                                        predicted_state,
                                        future_action,
                                    ]
                                )[None, :]
                            )[0]

                            predicted_state = clip_state(
                                predicted_state + delta
                            )

                        predicted_path.append(
                            predicted_state[:2].copy()
                        )

                state, _, _ = env.step(
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
            f"Controller: {mode_name}",
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
            "Yellow line: predicted drone trajectory",
            15,
            142,
            (255, 205, 95),
        )

        draw_text(
            "1 PD | 2 Linear MPC | 3 Neural MPC | 4 Oracle MPC",
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
# 16. UNIT TESTS
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

    # Test 1 — State dimension.
    state = np.array(
        [100.0, 200.0, 10.0, -5.0]
    )

    check(
        state.shape == (STATE_DIM,),
        "Drone state has four dimensions",
    )

    # Test 2 — Dynamics finite.
    next_state = true_transition(
        state,
        np.array([0.2, -0.1]),
    )

    check(
        np.all(np.isfinite(next_state)),
        "True dynamics produce finite state",
    )

    # Test 3 — Action clipping.
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
        "Dynamics clip actions to valid bounds",
    )

    # Test 4 — Speed bound.
    fast_state = np.array(
        [200.0, 200.0, 500.0, 500.0]
    )

    bounded_state = clip_state(
        fast_state
    )

    check(
        np.linalg.norm(
            bounded_state[2:4]
        ) <= MAX_SPEED + 1e-8,
        "State clipping enforces speed limit",
    )

    # Test 5 — Position bounds.
    check(
        DRONE_RADIUS <= bounded_state[0]
        <= WORLD_W - DRONE_RADIUS
        and DRONE_RADIUS <= bounded_state[1]
        <= WORLD_H - DRONE_RADIUS,
        "State clipping enforces position bounds",
    )

    # Test 6 — Moving target.
    target = MovingTarget(
        [700.0, 300.0]
    )

    p1 = target.position().copy()

    target.advance()

    p2 = target.position()

    check(
        not np.allclose(p1, p2),
        "Target moves over time",
    )

    # Test 7 — Dataset dimensions.
    x, y = collect_dataset(
        steps=100
    )

    check(
        x.shape == (100, MODEL_INPUT_DIM)
        and y.shape == (100, MODEL_OUTPUT_DIM),
        "Dataset has correct input and target dimensions",
    )

    # Test 8 — Dataset finite.
    check(
        np.all(np.isfinite(x))
        and np.all(np.isfinite(y)),
        "Dataset contains finite values",
    )

    # Test 9 — Normalizer.
    normalizer = Normalizer.fit(x)

    normalized = normalizer.transform(
        x
    )

    reconstructed = normalizer.inverse(
        normalized
    )

    check(
        np.allclose(
            reconstructed,
            x,
            atol=1e-5,
        ),
        "Normalizer transform and inverse are consistent",
    )

    # Test 10 — Linear regression.
    rng = np.random.default_rng(42)

    test_x = rng.normal(
        size=(300, 6)
    )

    test_w = rng.normal(
        size=(6, 4)
    )

    test_y = (
        test_x @ test_w
        + 0.3
    )

    linear = LinearDynamicsModel()

    linear.fit(
        test_x,
        test_y,
    )

    check(
        np.allclose(
            linear.predict(test_x),
            test_y,
            atol=1e-5,
        ),
        "Linear model recovers synthetic linear mapping",
    )

    # Test 11 — Neural network output shape.
    network = DynamicsNetwork().to(
        DEVICE
    )

    test_tensor = torch.zeros(
        (8, MODEL_INPUT_DIM),
        device=DEVICE,
    )

    output = network(
        test_tensor
    )

    check(
        output.shape == (
            8,
            MODEL_OUTPUT_DIM,
        ),
        "Neural dynamics network output shape is correct",
    )

    # Test 12 — Neural network finite output.
    check(
        torch.all(
            torch.isfinite(output)
        ),
        "Neural dynamics network output is finite",
    )

    # Test 13 — MPC cost finite.
    cost = tracking_cost(
        state,
        np.array([300.0, 300.0]),
        np.array([0.1, 0.2]),
    )

    check(
        np.isfinite(cost),
        "MPC stage cost is finite",
    )

    # Test 14 — Nominal sequence shape.
    nominal = make_nominal_sequence(
        state,
        target,
        target.phase,
    )

    check(
        nominal.shape == (
            MPC_HORIZON,
            ACTION_DIM,
        ),
        "Nominal MPC sequence has correct shape",
    )

    # Test 15 — Candidate action bounds.
    check(
        np.all(nominal <= 1.0)
        and np.all(nominal >= -1.0),
        "Nominal MPC actions are bounded",
    )

    # Test 16 — Oracle rollout finite.
    oracle_cost = rollout_candidate(
        state,
        nominal,
        target,
        target.phase,
        "oracle",
    )

    check(
        np.isfinite(oracle_cost),
        "Oracle MPC rollout cost is finite",
    )

    # Test 17 — Target velocity finite.
    check(
        np.all(
            np.isfinite(
                target.velocity()
            )
        ),
        "Target velocity is finite",
    )

    # Test 18 — Environment observation shape.
    env = DroneEnv()

    env.reset()

    observation = env.observation()

    check(
        observation.shape == (8,),
        "Environment observation has eight dimensions",
    )

    # Test 19 — Environment transition.
    next_state, _, _ = env.step(
        np.array([0.1, 0.2])
    )

    check(
        next_state.shape == (4,),
        "Environment returns valid next state",
    )

    print(
        f"\n{passed}/19 tests passed."
    )

    return passed == 19


# ============================================================
# 17. MAIN
# ============================================================

def main():

    print(
        "=" * 72
    )

    print(
        "PHASE 4.26 — NEURAL NETWORK SYSTEM IDENTIFICATION"
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

    # Step 2 — Collect transition data.
    print(
        "\n[1/7] Collecting drone transition data..."
    )

    inputs, targets = collect_dataset()

    print(
        f"Collected {len(inputs)} transitions."
    )

    # Step 3 — Train/test split.
    print(
        "\n[2/7] Splitting dataset..."
    )

    (
        x_train,
        y_train,
        x_test,
        y_test,
    ) = split_dataset(
        inputs,
        targets,
    )

    print(
        f"Training samples: {len(x_train)}"
    )

    print(
        f"Test samples: {len(x_test)}"
    )

    # Step 4 — Fit linear baseline.
    print(
        "\n[3/7] Training linear dynamics baseline..."
    )

    linear_model = LinearDynamicsModel()

    linear_model.fit(
        x_train,
        y_train,
    )

    # Step 5 — Train neural dynamics model.
    print(
        "\n[4/7] Training neural dynamics model..."
    )

    learned_model = LearnedDynamicsModel()

    learned_model.fit(
        x_train,
        y_train,
    )

    plot_training_loss(
        learned_model
    )

    # Step 6 — Evaluate prediction accuracy.
    print(
        "\n[5/7] Evaluating one-step prediction..."
    )

    (
        linear_rmse,
        linear_components,
        _,
    ) = evaluate_one_step(
        linear_model,
        x_test,
        y_test,
    )

    (
        neural_rmse,
        neural_components,
        _,
    ) = evaluate_one_step(
        learned_model,
        x_test,
        y_test,
    )

    plot_one_step_errors(
        linear_rmse,
        neural_rmse,
        linear_components,
        neural_components,
    )

    print(
        "\nEvaluating multi-step prediction..."
    )

    linear_rollout = rollout_prediction_error(
        linear_model
    )

    neural_rollout = rollout_prediction_error(
        learned_model
    )

    oracle_rollout = rollout_prediction_error(
        "oracle"
    )

    plot_rollout_errors(
        linear_rollout,
        neural_rollout,
        oracle_rollout,
    )

    print(
        "\nFinal rollout errors:"
    )

    print(
        f"Linear model: {linear_rollout[-1]:.4f}"
    )

    print(
        f"Neural model: {neural_rollout[-1]:.4f}"
    )

    print(
        f"Oracle model: {oracle_rollout[-1]:.4f}"
    )

    # Step 7 — Benchmark controllers.
    print(
        "\n[6/7] Benchmarking controllers..."
    )

    results = benchmark_controllers(
        learned_model,
        linear_model,
    )

    plot_benchmark(
        results
    )

    # Interactive visualization.
    print(
        "\n[7/7] Starting interactive drone simulation..."
    )

    print(
        "Controls:"
    )

    print(
        "1 = PD controller"
    )

    print(
        "2 = Linear-model MPC"
    )

    print(
        "3 = Neural-model MPC"
    )

    print(
        "4 = Oracle MPC"
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
        "Mouse click = change moving-target center"
    )

    print(
        "ESC = exit"
    )

    run_pygame(
        learned_model,
        linear_model,
    )


if __name__ == "__main__":

    main()
