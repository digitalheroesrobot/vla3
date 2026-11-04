
"""
File: phase4/differentiable_mpc_drone.py

PHASE 4.28 — DIFFERENTIABLE MODEL PREDICTIVE CONTROL (DMPC)

PURPOSE
-------
Optimize a drone's future action sequence by differentiating
the predicted trajectory cost with respect to every future action.

WHAT YOU LEARN
--------------

STEP 1 — Collect dynamics data
    Gather state/action/next-state transitions from the simulator.

STEP 2 — Train a differentiable dynamics model
    Learn the state transition function using PyTorch.

STEP 3 — Build a differentiable rollout
    Predict future drone states while preserving gradients.

STEP 4 — Define a trajectory objective
    Penalize target distance, obstacle proximity, boundary risk,
    excessive acceleration and abrupt action changes.

STEP 5 — Optimize actions using backpropagation
    Use Adam to minimize the predicted trajectory cost.

STEP 6 — Receding-horizon control
    Execute the first optimized action and replan at every step.

STEP 7 — Compare planning methods
    Compare PD, MPPI, learned-model DMPC and oracle DMPC.

MATHEMATICAL FOUNDATION
-----------------------

Let the learned dynamics be:

    x[t+1] = f_theta(x[t], u[t])

For a horizon H, optimize:

    U = [u[0], u[1], ..., u[H-1]]

The objective is:

    J(U) =
        sum(t=0 to H-1) cost(x[t], u[t])
        + terminal_cost(x[H])

The optimal action sequence is:

    U* = argmin_U J(U)

Gradient-based optimization:

    U <- U - learning_rate * gradient_U J(U)

The gradient is computed by backpropagation through the
differentiable dynamics rollout.

The chain rule propagates the effect of future actions:

    dJ/du[t] =
        dJ/dx[H] * dx[H]/du[t]
        + intermediate cost derivatives

Unlike MPPI, DMPC does not need to sample hundreds of
candidate action sequences for every optimization iteration.

IMPORTANT LIMITATIONS
---------------------
- The learned model can be inaccurate.
- Model prediction errors accumulate over long horizons.
- Differentiable planning can exploit model errors.
- Soft obstacle penalties are not formal safety guarantees.
- The oracle model is a simulation reference, not a real-world
  flight controller.
- The example uses simplified 2D quadcopter dynamics.

ROBOTICS APPLICATION
--------------------
Differentiable MPC is relevant to:
    - drone trajectory optimization
    - autonomous mobile robots
    - robot manipulation
    - model-based reinforcement learning
    - differentiable planning and control

The Pygame simulation uses a quadcopter silhouette and a
continuously moving target.
"""

from __future__ import annotations

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

SEED = 428

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
    dtype=np.float32,
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
    dtype=np.float32,
)

STATE_DIM = 4
ACTION_DIM = 2
MODEL_INPUT_DIM = 6

DATA_STEPS = 5500
TRAIN_EPOCHS = 100
BATCH_SIZE = 256
MODEL_LR = 0.001

HORIZON = 16
DMPC_ITERATIONS = 35
DMPC_LR = 0.045

MPPI_SAMPLES = 140
MPPI_ITERATIONS = 3
MPPI_TEMPERATURE = 8.0

ACTION_NOISE = np.array(
    [0.30, 0.30],
    dtype=np.float32,
)

BENCHMARK_EPISODES = 2
BENCHMARK_STEPS = 150


# ============================================================
# 1. MOVING TARGET
# ============================================================

class MovingTarget:

    def __init__(self, center):

        self.center = np.asarray(
            center,
            dtype=np.float32,
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


# ============================================================
# 2. DRONE DYNAMICS
# ============================================================

def clip_state(state):

    state = np.asarray(
        state,
        dtype=np.float32,
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

    state = np.asarray(
        state,
        dtype=np.float32,
    )

    action = np.clip(
        np.asarray(
            action,
            dtype=np.float32,
        ),
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
            dtype=np.float32,
        )

        self.target.phase = np.random.uniform(
            0.0,
            2.0 * math.pi,
        )

        return self.state.copy()

    def step(self, action):

        self.state = true_transition(
            self.state,
            action,
        )

        self.target.advance()

        return self.state.copy()


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

        nominal_action = pd_controller(
            state,
            env.target.position(),
            env.target.velocity(),
        )

        random_value = np.random.rand()

        if random_value < 0.25:

            action = np.random.uniform(
                -1.0,
                1.0,
                size=2,
            )

        elif random_value < 0.75:

            action = (
                nominal_action
                + np.random.normal(
                    0.0,
                    0.28,
                    size=2,
                )
            )

        else:

            action = nominal_action

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

        # Learn delta-state rather than absolute next state.
        targets.append(
            next_state - state
        )

        state = next_state

        if (step + 1) % 350 == 0:

            state = env.reset()

    return (
        np.asarray(inputs, dtype=np.float32),
        np.asarray(targets, dtype=np.float32),
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
# 6. DIFFERENTIABLE LEARNED DYNAMICS
# ============================================================

class DynamicsNetwork(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(MODEL_INPUT_DIM, 128),
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

        normalized_inputs = (
            self.input_normalizer.transform(
                inputs
            )
        )

        normalized_targets = (
            self.output_normalizer.transform(
                targets
            )
        )

        x = torch.as_tensor(
            normalized_inputs,
            dtype=torch.float32,
            device=DEVICE,
        )

        y = torch.as_tensor(
            normalized_targets,
            dtype=torch.float32,
            device=DEVICE,
        )

        dataset = torch.utils.data.TensorDataset(
            x,
            y,
        )

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
        )

        optimizer = optim.Adam(
            self.model.parameters(),
            lr=MODEL_LR,
        )

        loss_fn = nn.MSELoss()

        for epoch in range(epochs):

            epoch_losses = []

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

                epoch_losses.append(
                    loss.item()
                )

            mean_loss = float(
                np.mean(epoch_losses)
            )

            self.loss_history.append(
                mean_loss
            )

            if (
                (epoch + 1) % 20 == 0
                or epoch == 0
            ):

                print(
                    f"Epoch {epoch+1:3d}/{epochs}"
                    f" | normalized MSE={mean_loss:.6f}"
                )

    def predict_delta_torch(
        self,
        states,
        actions,
    ):

        """
        Differentiable batch prediction.

        states:  (N, 4)
        actions: (N, 2)

        Gradients flow through actions and states.
        """

        model_input = torch.cat(
            [
                states,
                actions,
            ],
            dim=-1,
        )

        input_mean = torch.as_tensor(
            self.input_normalizer.mean,
            dtype=model_input.dtype,
            device=model_input.device,
        )

        input_std = torch.as_tensor(
            self.input_normalizer.std,
            dtype=model_input.dtype,
            device=model_input.device,
        )

        output_mean = torch.as_tensor(
            self.output_normalizer.mean,
            dtype=model_input.dtype,
            device=model_input.device,
        )

        output_std = torch.as_tensor(
            self.output_normalizer.std,
            dtype=model_input.dtype,
            device=model_input.device,
        )

        normalized_input = (
            model_input - input_mean
        ) / input_std

        normalized_delta = self.model(
            normalized_input
        )

        return (
            normalized_delta * output_std
            + output_mean
        )

    def predict_delta_numpy(
        self,
        states,
        actions,
    ):

        model_input = np.concatenate(
            [
                states,
                actions,
            ],
            axis=-1,
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
        )

        with torch.no_grad():

            normalized_delta = (
                self.model(tensor)
                .cpu()
                .numpy()
            )

        return self.output_normalizer.inverse(
            normalized_delta
        )

    def predict_next(
        self,
        state,
        action,
    ):

        delta = self.predict_delta_numpy(
            state[None, :],
            action[None, :],
        )[0]

        return clip_state(
            state + delta
        )


# ============================================================
# 7. DIFFERENTIABLE ORACLE DYNAMICS
# ============================================================

def oracle_transition_torch(
    states,
    actions,
):

    """
    Differentiable version of the known simulator dynamics.

    This is used as an oracle reference.

    We intentionally do not use hard position/speed clipping
    inside the differentiable rollout. Instead, boundary
    penalties are included in the objective.

    This avoids zero or discontinuous gradients at clipping
    boundaries.
    """

    velocity = states[..., 2:4]

    acceleration = (
        MOTOR_GAIN * MAX_ACCEL * actions
        - LINEAR_DRAG * velocity
        - QUADRATIC_DRAG
        * torch.abs(velocity)
        * velocity
        + torch.as_tensor(
            WIND,
            dtype=states.dtype,
            device=states.device,
        )
    )

    next_velocity = (
        velocity + acceleration * DT
    )

    next_position = (
        states[..., :2]
        + next_velocity * DT
    )

    return torch.cat(
        [
            next_position,
            next_velocity,
        ],
        dim=-1,
    )


# ============================================================
# 8. DIFFERENTIABLE TRAJECTORY COST
# ============================================================

def differentiable_stage_cost(
    states,
    actions,
    target_positions,
):

    """
    Vectorized differentiable cost.

    states:
        (H, 4)

    actions:
        (H, 2)

    target_positions:
        (H, 2)
    """

    position = states[..., :2]

    target_distance = torch.linalg.vector_norm(
        position - target_positions,
        dim=-1,
    )

    obstacle = torch.as_tensor(
        OBSTACLE,
        dtype=states.dtype,
        device=states.device,
    )

    obstacle_distance = torch.linalg.vector_norm(
        position - obstacle,
        dim=-1,
    )

    obstacle_violation = torch.relu(
        SAFE_RADIUS - obstacle_distance
    )

    x = position[..., 0]
    y = position[..., 1]

    boundary_distances = torch.stack(
        [
            x,
            WORLD_W - x,
            y,
            WORLD_H - y,
        ],
        dim=-1,
    )

    minimum_boundary_distance = (
        torch.min(
            boundary_distances,
            dim=-1,
        ).values
    )

    boundary_violation = torch.relu(
        55.0 - minimum_boundary_distance
    )

    action_effort = torch.sum(
        actions**2,
        dim=-1,
    )

    return (
        0.025 * target_distance
        + 0.055 * obstacle_violation**2
        + 0.06 * boundary_violation**2
        + 0.25 * action_effort
    )


def terminal_cost(
    final_state,
    final_target,
):

    distance = torch.linalg.vector_norm(
        final_state[:2] - final_target
    )

    return 0.05 * distance


def build_future_targets(
    target,
    phase,
    horizon,
):

    positions = []
    velocities = []

    for step in range(
        1,
        horizon + 1,
    ):

        future_phase = (
            phase
            + TARGET_OMEGA * DT * step
        )

        positions.append(
            target.position(
                future_phase
            )
        )

        velocities.append(
            target.velocity(
                future_phase
            )
        )

    return (
        np.asarray(
            positions,
            dtype=np.float32,
        ),

        np.asarray(
            velocities,
            dtype=np.float32,
        ),
    )


# ============================================================
# 9. DIFFERENTIABLE ROLLOUT
# ============================================================

def differentiable_rollout(
    initial_state,
    actions,
    target_positions,
    dynamics,
):

    """
    Unroll a differentiable model over the planning horizon.

    actions:
        (H, 2)

    Returns:
        predicted_states: (H, 4)
        total_cost: scalar tensor
    """

    state = torch.as_tensor(
        initial_state,
        dtype=torch.float32,
        device=DEVICE,
    )

    predicted_states = []

    for step in range(
        actions.shape[0]
    ):

        action = actions[step]

        if dynamics == "oracle":

            next_state = oracle_transition_torch(
                state,
                action,
            )

        else:

            delta = dynamics.predict_delta_torch(
                state.unsqueeze(0),
                action.unsqueeze(0),
            )[0]

            next_state = (
                state + delta
            )

        predicted_states.append(
            next_state
        )

        state = next_state

    predicted_states = torch.stack(
        predicted_states,
        dim=0,
    )

    target_tensor = torch.as_tensor(
        target_positions,
        dtype=torch.float32,
        device=DEVICE,
    )

    stage_costs = differentiable_stage_cost(
        predicted_states,
        actions,
        target_tensor,
    )

    final_target = target_tensor[-1]

    total_cost = (
        torch.sum(stage_costs)
        + terminal_cost(
            predicted_states[-1],
            final_target,
        )
    )

    return (
        predicted_states,
        total_cost,
    )


# ============================================================
# 10. DIFFERENTIABLE MPC CONTROLLER
# ============================================================

class DifferentiableMPC:

    def __init__(
        self,
        dynamics,
        horizon=HORIZON,
        iterations=DMPC_ITERATIONS,
        learning_rate=DMPC_LR,
    ):

        self.dynamics = dynamics

        self.horizon = horizon
        self.iterations = iterations
        self.learning_rate = learning_rate

        self.nominal = np.zeros(
            (horizon, ACTION_DIM),
            dtype=np.float32,
        )

        self.last_cost_history = []
        self.last_prediction = None

    def reset(self):

        self.nominal.fill(
            0.0
        )

        self.last_cost_history = []
        self.last_prediction = None

    def make_initial_sequence(
        self,
        state,
        target,
        phase,
    ):

        actions = []

        estimated_state = state.copy()

        for step in range(
            self.horizon
        ):

            future_phase = (
                phase
                + TARGET_OMEGA * DT * step
            )

            action = pd_controller(
                estimated_state,
                target.position(
                    future_phase
                ),
                target.velocity(
                    future_phase
                ),
            )

            actions.append(
                action
            )

            # Kinematic seed propagation.
            # This is only an initialization heuristic.
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

        return np.asarray(
            actions,
            dtype=np.float32,
        )

    def plan(
        self,
        state,
        target,
        phase,
    ):

        if not np.any(self.nominal):

            self.nominal = self.make_initial_sequence(
                state,
                target,
                phase,
            )

        future_positions, _ = build_future_targets(
            target,
            phase,
            self.horizon,
        )

        # U is the optimization variable.
        # Adam updates U, not the learned dynamics parameters.
        action_variable = torch.nn.Parameter(
            torch.as_tensor(
                self.nominal,
                dtype=torch.float32,
                device=DEVICE,
            ).clone()
        )

        optimizer = optim.Adam(
            [action_variable],
            lr=self.learning_rate,
        )

        self.last_cost_history = []

        for _ in range(
            self.iterations
        ):

            optimizer.zero_grad()

            bounded_actions = torch.tanh(
                action_variable
            )

            predicted_states, cost = (
                differentiable_rollout(
                    state,
                    bounded_actions,
                    future_positions,
                    self.dynamics,
                )
            )

            cost.backward()

            torch.nn.utils.clip_grad_norm_(
                [action_variable],
                10.0,
            )

            optimizer.step()

            self.last_cost_history.append(
                float(cost.detach().cpu())
            )

        with torch.no_grad():

            final_actions = torch.tanh(
                action_variable
            )

            final_states, final_cost = (
                differentiable_rollout(
                    state,
                    final_actions,
                    future_positions,
                    self.dynamics,
                )
            )

            final_actions_np = (
                final_actions.cpu().numpy()
            )

            self.last_prediction = (
                final_states.cpu().numpy()
            )

            result_cost = float(
                final_cost.cpu()
            )

        # Receding-horizon warm start.
        self.nominal[:-1] = (
            final_actions_np[1:]
        )

        self.nominal[-1] = (
            final_actions_np[-1]
        )

        return (
            final_actions_np[0].copy(),
            final_actions_np.copy(),
            result_cost,
        )


# ============================================================
# 11. MPPI BASELINE
# ============================================================

def numpy_stage_cost(
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


def evaluate_numpy_sequence(
    initial_state,
    actions,
    target,
    phase,
    dynamics,
):

    state = initial_state.copy()

    total_cost = 0.0

    for step, action in enumerate(actions):

        future_phase = (
            phase
            + TARGET_OMEGA * DT * (step + 1)
        )

        future_target = target.position(
            future_phase
        )

        if dynamics == "oracle":

            state = true_transition(
                state,
                action,
            )

        else:

            state = dynamics.predict_next(
                state,
                action,
            )

        total_cost += numpy_stage_cost(
            state,
            future_target,
            action,
        )

    final_phase = (
        phase
        + TARGET_OMEGA * DT * len(actions)
    )

    final_target = target.position(
        final_phase
    )

    total_cost += (
        0.05
        * np.linalg.norm(
            state[:2] - final_target
        )
    )

    return total_cost


class MPPIController:

    def __init__(
        self,
        dynamics,
        horizon=HORIZON,
    ):

        self.dynamics = dynamics
        self.horizon = horizon

        self.nominal = np.zeros(
            (horizon, 2),
            dtype=np.float32,
        )

    def reset(self):

        self.nominal.fill(
            0.0
        )

    def plan(
        self,
        state,
        target,
        phase,
    ):

        if not np.any(self.nominal):

            self.nominal = np.asarray(
                [
                    pd_controller(
                        state,
                        target.position(
                            phase + TARGET_OMEGA * DT * step
                        ),
                        target.velocity(
                            phase + TARGET_OMEGA * DT * step
                        ),
                    )
                    for step in range(self.horizon)
                ],
                dtype=np.float32,
            )

        noise = np.random.normal(
            0.0,
            ACTION_NOISE,
            size=(
                MPPI_SAMPLES,
                self.horizon,
                2,
            ),
        ).astype(
            np.float32
        )

        candidates = np.clip(
            self.nominal[None, :, :] + noise,
            -1.0,
            1.0,
        )

        candidates[0] = self.nominal

        costs = np.asarray(
            [
                evaluate_numpy_sequence(
                    state,
                    candidate,
                    target,
                    phase,
                    self.dynamics,
                )
                for candidate in candidates
            ]
        )

        minimum = np.min(
            costs
        )

        logits = (
            -(costs - minimum)
            / MPPI_TEMPERATURE
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

        for _ in range(
            MPPI_ITERATIONS - 1
        ):

            candidates = np.clip(
                self.nominal[None, :, :]
                + np.random.normal(
                    0.0,
                    ACTION_NOISE,
                    size=(
                        MPPI_SAMPLES,
                        self.horizon,
                        2,
                    ),
                ),
                -1.0,
                1.0,
            )

            candidates[0] = self.nominal

            costs = np.asarray(
                [
                    evaluate_numpy_sequence(
                        state,
                        candidate,
                        target,
                        phase,
                        self.dynamics,
                    )
                    for candidate in candidates
                ]
            )

            minimum = np.min(
                costs
            )

            logits = (
                -(costs - minimum)
                / MPPI_TEMPERATURE
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

        updated = np.sum(
            weights[:, None, None]
            * candidates,
            axis=0,
        )

        self.nominal = updated.astype(
            np.float32
        )

        action = self.nominal[0].copy()

        self.nominal[:-1] = (
            self.nominal[1:]
        )

        self.nominal[-1] = (
            self.nominal[-2]
        )

        return (
            action,
            updated.copy(),
            float(np.min(costs)),
        )


# ============================================================
# 12. BENCHMARKING
# ============================================================

def run_episode(
    mode,
    learned_model,
    steps=BENCHMARK_STEPS,
):

    env = DroneEnv()

    state = env.reset()

    if mode == "MPPI":

        controller = MPPIController(
            learned_model
        )

    elif mode == "DMPC":

        controller = DifferentiableMPC(
            learned_model
        )

    elif mode == "Oracle DMPC":

        controller = DifferentiableMPC(
            "oracle"
        )

    distances = []
    collision_steps = 0

    for _ in range(steps):

        if mode == "PD":

            action = pd_controller(
                state,
                env.target.position(),
                env.target.velocity(),
            )

        else:

            action, _, _ = controller.plan(
                state,
                env.target,
                env.target.phase,
            )

        state = env.step(
            action
        )

        # Compare the drone against the target's new position.
        distance = np.linalg.norm(
            state[:2] - env.target.position()
        )

        distances.append(
            distance
        )

        physical_clearance = (
            np.linalg.norm(
                state[:2] - OBSTACLE
            )
            - OBSTACLE_RADIUS
        )

        if physical_clearance < 0.0:

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
        "MPPI",
        "DMPC",
        "Oracle DMPC",
    ]

    results = {}

    for mode in modes:

        print(
            f"\nBenchmarking {mode}..."
        )

        episode_results = []

        for _ in range(
            BENCHMARK_EPISODES
        ):

            episode_results.append(
                run_episode(
                    mode,
                    learned_model,
                )
            )

        results[mode] = {
            "mean_distance": float(
                np.mean(
                    [
                        result["mean_distance"]
                        for result in episode_results
                    ]
                )
            ),

            "collision_steps": float(
                np.mean(
                    [
                        result["collision_steps"]
                        for result in episode_results
                    ]
                )
            ),
        }

        print(
            f"{mode:14s}"
            f" | mean distance={results[mode]['mean_distance']:8.2f}"
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
        "Learned Dynamics Training Loss"
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
        "Moving-Target Tracking Performance"
    )

    plt.ylabel(
        "Mean target distance"
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
        "Obstacle Collision Comparison"
    )

    plt.ylabel(
        "Average collision steps"
    )

    plt.grid(
        axis="y"
    )

    plt.tight_layout()

    plt.show()


def plot_gradient_optimization(
    controller,
):

    if not controller.last_cost_history:

        return

    plt.figure(
        figsize=(9, 5)
    )

    plt.plot(
        controller.last_cost_history
    )

    plt.title(
        "Differentiable MPC Optimization"
    )

    plt.xlabel(
        "Gradient optimization iteration"
    )

    plt.ylabel(
        "Predicted trajectory cost"
    )

    plt.grid(True)

    plt.tight_layout()

    plt.show()


# ============================================================
# 14. PYGAME INTERACTIVE DRONE
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
        "Phase 4.28 — Differentiable MPC Drone"
    )

    clock = pygame.time.Clock()

    font = pygame.font.SysFont(
        "consolas",
        15,
    )

    env = DroneEnv()

    state = env.reset()

    modes = {
        1: "PD",
        2: "MPPI",
        3: "DMPC",
        4: "Oracle DMPC",
    }

    selected = 3

    controllers = {
        "MPPI": MPPIController(
            learned_model
        ),

        "DMPC": DifferentiableMPC(
            learned_model
        ),

        "Oracle DMPC": DifferentiableMPC(
            "oracle"
        ),
    }

    paused = False
    speed = 1

    trail = []
    predicted_path = []

    running = True

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

        speed_value = np.linalg.norm(
            velocity
        )

        direction = (
            velocity / speed_value
            if speed_value > 1e-6
            else np.array(
                [1.0, 0.0]
            )
        )

        side = np.array(
            [
                -direction[1],
                direction[0],
            ]
        )

        for forward in (-1, 1):

            for lateral in (-1, 1):

                rotor = (
                    center
                    + 17 * forward * direction
                    + 17 * lateral * side
                )

                pygame.draw.line(
                    screen,
                    (220, 225, 235),
                    center.astype(int),
                    rotor.astype(int),
                    5,
                )

                pygame.draw.circle(
                    screen,
                    (90, 190, 245),
                    rotor.astype(int),
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

        orbit = []

        for index in range(100):

            phase = (
                env.target.phase
                + 2.0 * math.pi * index / 100
            )

            orbit.append(
                env.target.position(
                    phase
                ).astype(int)
            )

        pygame.draw.lines(
            screen,
            (65, 105, 75),
            True,
            orbit,
            2,
        )

        target_position = (
            env.target.position()
        )

        target_velocity = (
            env.target.velocity()
        )

        pygame.draw.circle(
            screen,
            (70, 240, 110),
            target_position.astype(int),
            11,
        )

        pygame.draw.line(
            screen,
            (70, 255, 120),
            target_position.astype(int),
            (
                target_position
                + 0.18 * target_velocity
            ).astype(int),
            3,
        )

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

                    for controller in controllers.values():

                        controller.reset()

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
                        dtype=np.float32,
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

        mode = modes[selected]

        if not paused:

            for _ in range(speed):

                if mode == "PD":

                    action = pd_controller(
                        state,
                        env.target.position(),
                        env.target.velocity(),
                    )

                    predicted_path = []

                else:

                    controller = controllers[
                        mode
                    ]

                    action, sequence, _ = (
                        controller.plan(
                            state,
                            env.target,
                            env.target.phase,
                        )
                    )

                    if mode in (
                        "DMPC",
                        "Oracle DMPC",
                    ):

                        predicted = (
                            controller.last_prediction
                        )

                        if predicted is not None:

                            predicted_path = (
                                predicted[:, :2].copy()
                            )

                    else:

                        predicted_path = []

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

        target_distance = np.linalg.norm(
            state[:2] - env.target.position()
        )

        obstacle_clearance = (
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
            f"Target distance: {target_distance:.2f}",
            15,
            38,
        )

        draw_text(
            f"Obstacle clearance: {obstacle_clearance:.2f}",
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
            "Yellow: predicted trajectory",
            15,
            142,
            (255, 205, 95),
        )

        draw_text(
            "1 PD | 2 MPPI | 3 Differentiable MPC | 4 Oracle DMPC",
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

    # Test 1 — True dynamics.
    state = np.array(
        [100.0, 200.0, 10.0, -5.0],
        dtype=np.float32,
    )

    next_state = true_transition(
        state,
        np.array(
            [0.2, -0.1],
            dtype=np.float32,
        ),
    )

    check(
        next_state.shape == (4,)
        and np.all(np.isfinite(next_state)),
        "True dynamics return a finite state",
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
        "Actions are clipped",
    )

    # Test 3 — Speed constraint.
    bounded_state = clip_state(
        np.array(
            [100.0, 100.0, 500.0, 500.0]
        )
    )

    check(
        np.linalg.norm(
            bounded_state[2:4]
        ) <= MAX_SPEED + 1e-5,
        "Speed constraint is enforced",
    )

    # Test 4 — Position constraints.
    check(
        DRONE_RADIUS <= bounded_state[0]
        <= WORLD_W - DRONE_RADIUS
        and DRONE_RADIUS <= bounded_state[1]
        <= WORLD_H - DRONE_RADIUS,
        "World boundary constraints are enforced",
    )

    # Test 5 — Moving target.
    target = MovingTarget(
        [700.0, 300.0]
    )

    before = target.position().copy()

    target.advance()

    after = target.position()

    check(
        not np.allclose(
            before,
            after,
        ),
        "Moving target advances",
    )

    # Test 6 — Dataset dimensions.
    inputs, targets = collect_dataset(
        steps=100
    )

    check(
        inputs.shape == (100, 6)
        and targets.shape == (100, 4),
        "Dataset dimensions are correct",
    )

    # Test 7 — Dataset finite.
    check(
        np.all(np.isfinite(inputs))
        and np.all(np.isfinite(targets)),
        "Dataset values are finite",
    )

    # Test 8 — Neural network output.
    network = DynamicsNetwork().to(
        DEVICE
    )

    test_input = torch.zeros(
        (8, MODEL_INPUT_DIM),
        device=DEVICE,
    )

    output = network(
        test_input
    )

    check(
        output.shape == (8, STATE_DIM),
        "Dynamics network output shape is correct",
    )

    # Test 9 — Neural output finite.
    check(
        torch.all(
            torch.isfinite(output)
        ),
        "Dynamics network output is finite",
    )

    # Test 10 — Oracle differentiability.
    torch_state = torch.tensor(
        state,
        dtype=torch.float32,
        device=DEVICE,
        requires_grad=True,
    )

    torch_action = torch.tensor(
        [0.2, -0.1],
        dtype=torch.float32,
        device=DEVICE,
        requires_grad=True,
    )

    oracle_next = oracle_transition_torch(
        torch_state,
        torch_action,
    )

    oracle_next.sum().backward()

    check(
        torch_action.grad is not None
        and torch.all(
            torch.isfinite(torch_action.grad)
        ),
        "Oracle dynamics propagate action gradients",
    )

    # Test 11 — Differentiable stage cost.
    actions = torch.zeros(
        (5, 2),
        dtype=torch.float32,
        device=DEVICE,
        requires_grad=True,
    )

    states = torch.tensor(
        np.tile(
            state,
            (5, 1),
        ),
        dtype=torch.float32,
        device=DEVICE,
    )

    target_positions = torch.tensor(
        np.tile(
            [300.0, 300.0],
            (5, 1),
        ),
        dtype=torch.float32,
        device=DEVICE,
    )

    costs = differentiable_stage_cost(
        states,
        actions,
        target_positions,
    )

    costs.sum().backward()

    check(
        actions.grad is not None
        and torch.all(
            torch.isfinite(actions.grad)
        ),
        "Stage cost propagates gradients to actions",
    )

    # Test 12 — Differentiable rollout.
    rollout_actions = torch.zeros(
        (5, 2),
        dtype=torch.float32,
        device=DEVICE,
        requires_grad=True,
    )

    rollout_states, rollout_cost = (
        differentiable_rollout(
            state,
            rollout_actions,
            target_positions.detach().cpu().numpy(),
            "oracle",
        )
    )

    check(
        rollout_states.shape == (5, 4)
        and rollout_cost.ndim == 0
        and torch.isfinite(rollout_cost),
        "Differentiable rollout returns correct shapes",
    )

    # Test 13 — Full rollout gradient.
    rollout_cost.backward()

    check(
        rollout_actions.grad is not None
        and torch.all(
            torch.isfinite(
                rollout_actions.grad
            )
        ),
        "Trajectory cost gradients reach all future actions",
    )

    # Test 14 — Normalizer.
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
        "Normalization round trip is correct",
    )

    # Test 15 — Target prediction.
    future_positions, future_velocities = (
        build_future_targets(
            target,
            target.phase,
            5,
        )
    )

    check(
        future_positions.shape == (5, 2)
        and future_velocities.shape == (5, 2),
        "Future target trajectory has correct dimensions",
    )

    # Test 16 — DMPC output interface.
    controller = DifferentiableMPC(
        "oracle",
        horizon=5,
        iterations=3,
    )

    action, sequence, cost = controller.plan(
        state,
        target,
        target.phase,
    )

    check(
        action.shape == (2,)
        and sequence.shape == (5, 2)
        and np.isfinite(cost),
        "DMPC returns action, sequence and finite cost",
    )

    # Test 17 — DMPC action limits.
    check(
        np.all(action >= -1.0)
        and np.all(action <= 1.0),
        "DMPC actions remain within bounds",
    )

    # Test 18 — MPPI interface.
    mppi = MPPIController(
        "oracle",
        horizon=5,
    )

    mppi_action, mppi_sequence, mppi_cost = (
        mppi.plan(
            state,
            target,
            target.phase,
        )
    )

    check(
        mppi_action.shape == (2,)
        and mppi_sequence.shape == (5, 2)
        and np.isfinite(mppi_cost),
        "MPPI baseline returns valid outputs",
    )

    # Test 19 — MPPI action limits.
    check(
        np.all(mppi_action >= -1.0)
        and np.all(mppi_action <= 1.0),
        "MPPI actions remain within bounds",
    )

    # Test 20 — Model prediction interface.
    model = LearnedDynamics()

    model.input_normalizer = Normalizer(
        inputs
    )

    model.output_normalizer = Normalizer(
        targets
    )

    prediction = model.predict_delta_numpy(
        state[None, :],
        np.array(
            [[0.1, 0.2]],
            dtype=np.float32,
        ),
    )

    check(
        prediction.shape == (1, 4)
        and np.all(np.isfinite(prediction)),
        "Learned dynamics prediction interface works",
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
        "PHASE 4.28 — DIFFERENTIABLE MODEL PREDICTIVE CONTROL"
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
        "\n[1/6] Collecting drone dynamics data..."
    )

    inputs, targets = collect_dataset()

    print(
        f"Collected {len(inputs)} transitions."
    )

    # Step 3 — Train learned dynamics.
    print(
        "\n[2/6] Training differentiable dynamics model..."
    )

    learned_model = LearnedDynamics()

    learned_model.fit(
        inputs,
        targets,
    )

    plot_training_loss(
        learned_model
    )

    # Step 4 — Benchmark.
    print(
        "\n[3/6] Comparing controllers..."
    )

    results = benchmark(
        learned_model
    )

    plot_benchmark(
        results
    )

    # Step 5 — Demonstrate gradient optimization.
    print(
        "\n[4/6] Demonstrating differentiable MPC optimization..."
    )

    demo_env = DroneEnv()

    demo_state = demo_env.reset()

    demo_controller = DifferentiableMPC(
        learned_model
    )

    demo_controller.plan(
        demo_state,
        demo_env.target,
        demo_env.target.phase,
    )

    plot_gradient_optimization(
        demo_controller
    )

    # Step 6 — Interactive simulation.
    print(
        "\n[5/6] Starting interactive drone simulation..."
    )

    print(
        "1 = PD"
    )

    print(
        "2 = MPPI"
    )

    print(
        "3 = Learned-model DMPC"
    )

    print(
        "4 = Oracle DMPC"
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
        "Click = move target trajectory center"
    )

    print(
        "ESC = exit"
    )

    run_pygame(
        learned_model
    )

    print(
        "\n[6/6] Phase 4.28 completed."
    )


if __name__ == "__main__":

    main()
