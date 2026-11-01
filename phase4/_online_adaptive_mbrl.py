
"""
File: phase4/online_adaptive_mbrl_drone.py

PHASE 4.21 — SIM-TO-REAL TRANSFER
=================================

ONLINE SYSTEM IDENTIFICATION
ADAPTIVE DYNAMICS
RESIDUAL DYNAMICS LEARNING
ADAPTIVE MODEL PREDICTIVE CONTROL

PURPOSE
-------
Build a drone controller that can estimate unknown physical parameters
and improve its dynamics predictions while operating.

WHAT YOU LEARN
--------------

STEP 1 — Unknown physical parameters
    The drone's motor effectiveness and aerodynamic drag are unknown.

STEP 2 — Online system identification
    Estimate motor gain and linear drag using Recursive Least Squares
    (RLS), updating the estimates after every observed transition.

STEP 3 — Adaptive dynamics
    Replace fixed physics parameters with online estimates.

STEP 4 — Residual dynamics learning
    Learn the remaining prediction error, including nonlinear drag
    and an unmodeled acceleration bias.

STEP 5 — Adaptive MPC
    Generate candidate action sequences using the learned dynamics.
    Execute only the first action, observe the result, and replan.

STEP 6 — Sim-to-real-style evaluation
    Test on multiple physical domains, including unfamiliar domains.

STEP 7 — Interactive visualization
    Display the quadcopter, moving target, obstacle, predicted path,
    parameter estimates, and residual-model learning progress.

MATHEMATICAL MODEL
------------------

Actual simplified drone dynamics:

    a = g * MAX_ACCEL * u
        - drag * v
        + wind
        - qdrag * abs(v) * v
        + bias

where:
    g       = motor gain
    drag    = linear aerodynamic drag
    qdrag   = unmodeled nonlinear drag coefficient
    bias    = unmodeled acceleration bias
    u       = normalized control action
    v       = velocity
    wind    = measured wind disturbance

The controller initially does not know g, drag, qdrag, or bias.

SYSTEM IDENTIFICATION
--------------------

From an observed transition:

    y = (v_next - v) / dt - wind

The linear model is:

    y = g * MAX_ACCEL * u - drag * v

Define:

    phi = [MAX_ACCEL*u, -v]
    theta = [g, drag]

Then:

    y = phi.T @ theta

RLS updates:

    K = P @ phi / (lambda + phi.T @ P @ phi)

    theta = theta + K * (y - phi.T @ theta)

    P = (P - K @ phi.T @ P) / lambda

The two velocity axes provide separate observations of the same
motor gain and drag parameters.

RESIDUAL DYNAMICS LEARNING
--------------------------

The adaptive linear model cannot represent every physical effect.

We therefore estimate the remaining acceleration error:

    residual = observed_acceleration
                - predicted_linear_acceleration

A second RLS model learns this residual from nonlinear features:

    [1, vx, vy, wind_x, wind_y, ax, ay,
     abs(vx)*vx, abs(vy)*vy]

This is an online residual model, not a deep neural network.
It is intentionally lightweight so that adaptation can occur
during operation.

MPC OBJECTIVE
-------------

For each candidate action sequence:

    J = tracking_cost
        + obstacle_penalty
        + boundary_penalty
        + control_effort_penalty

The planner:
    1. predicts the moving target;
    2. predicts drone motion with the current learned model;
    3. evaluates candidate action sequences;
    4. selects the lowest-cost sequence;
    5. executes only its first action;
    6. observes the real transition and updates its model.

INSTALL
-------
    pip install numpy matplotlib pygame

RUN
---
    python phase4/online_adaptive_mbrl_drone.py

CONTROLS
--------
    1 : Fixed-parameter MPC
    2 : Adaptive MPC
    3 : Adaptive MPC + residual learning
    4 : Oracle MPC
    SPACE : Pause/resume
    R : Reset drone and adaptation estimates
    + / - : Simulation speed
    Left mouse click : Change moving-target trajectory center
    ESC : Exit

ROBOTICS APPLICATION
--------------------
A real drone's effective dynamics may change because of:
    - payload changes;
    - battery voltage;
    - motor wear;
    - propeller replacement;
    - aerodynamic disturbances;
    - wind and air-density changes.

Online system identification allows a controller to adapt its
physical model from observed behavior.

IMPORTANT LIMITATIONS
--------------------
1. This is a simplified 2D quadcopter simulation.
2. RLS estimates motor gain and drag, not mass and motor efficiency
   separately. These parameters are not independently identifiable
   from the chosen equation.
3. Parameter identification requires sufficiently varied actions
   and velocities. Poor excitation can make estimates unreliable.
4. Residual learning is a compact linear model over nonlinear features.
5. MPC uses a finite candidate set and is not globally optimal.
6. The obstacle correction is an educational heuristic, not a
   formal collision-avoidance guarantee.
7. This is not flight-certified software.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 0. CONFIGURATION
# ============================================================

SEED = 421

random.seed(SEED)
np.random.seed(SEED)

WORLD_W = 900
WORLD_H = 650

DT = 0.08

MAX_SPEED = 155.0
MAX_ACCEL = 125.0
DRONE_RADIUS = 13.0

OBSTACLE = np.array(
    [450.0, 330.0],
    dtype=np.float64,
)

OBSTACLE_RADIUS = 65.0
SAFE_RADIUS = 105.0

TARGET_A = 170.0
TARGET_B = 105.0
TARGET_OMEGA = 0.55

MPC_HORIZON = 10
MPC_CANDIDATES = 44

ACTION_NOISE = 0.43

BENCHMARK_EPISODES = 2
BENCHMARK_STEPS = 120

RLS_FORGETTING = 0.995
RESIDUAL_FORGETTING = 0.995

PARAMETER_MIN = np.array(
    [0.35, 0.0],
    dtype=np.float64,
)

PARAMETER_MAX = np.array(
    [1.65, 0.35],
    dtype=np.float64,
)

WIND_DECAY = 0.88
WIND_NOISE = 16.0

# Actual physical domains.
#
# motor_gain = motor effectiveness / effective mass
#
# qdrag and bias are deliberately absent from the linear
# identification equation. They are learned by residual RLS.

DOMAINS = [
    {
        "name": "nominal",
        "motor_gain": 1.00,
        "drag": 0.065,
        "qdrag": 0.00018,
        "bias": np.array(
            [0.0, 0.0],
            dtype=np.float64,
        ),
    },
    {
        "name": "heavy_payload",
        "motor_gain": 0.76,
        "drag": 0.085,
        "qdrag": 0.00024,
        "bias": np.array(
            [1.0, -1.0],
            dtype=np.float64,
        ),
    },
    {
        "name": "light_payload",
        "motor_gain": 1.28,
        "drag": 0.045,
        "qdrag": 0.00012,
        "bias": np.array(
            [-1.0, 1.0],
            dtype=np.float64,
        ),
    },
    {
        "name": "motor_degradation",
        "motor_gain": 0.62,
        "drag": 0.105,
        "qdrag": 0.00030,
        "bias": np.array(
            [2.0, -1.5],
            dtype=np.float64,
        ),
    },
    {
        "name": "unseen_combined",
        "motor_gain": 1.38,
        "drag": 0.125,
        "qdrag": 0.00035,
        "bias": np.array(
            [-2.0, 2.5],
            dtype=np.float64,
        ),
    },
]


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
            dtype=np.float64,
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
            dtype=np.float64,
        )

    def advance(self):

        self.phase += TARGET_OMEGA * DT


def future_target_position(target, step):

    phase = (
        target.phase
        + TARGET_OMEGA * DT * step
    )

    return target.position(phase)


def future_target_velocity(target, step):

    phase = (
        target.phase
        + TARGET_OMEGA * DT * step
    )

    return target.velocity(phase)


# ============================================================
# 2. TRUE DRONE ENVIRONMENT
# ============================================================

class DroneEnv:

    """
    State:
        [x, y, vx, vy, wind_x, wind_y]

    Action:
        [ax, ay] normalized to [-1, 1]

    Actual dynamics:

        acceleration =
            motor_gain * MAX_ACCEL * action
            - drag * velocity
            + wind
            - qdrag * abs(velocity) * velocity
            + bias

    The controller does not receive the true domain parameters.
    """

    def __init__(
        self,
        domain=None,
        wind_noise=WIND_NOISE,
    ):

        self.domain = (
            dict(DOMAINS[0])
            if domain is None
            else dict(domain)
        )

        self.wind_noise = float(
            wind_noise
        )

        self.target = MovingTarget(
            np.array(
                [700.0, 310.0],
                dtype=np.float64,
            )
        )

        self.reset()

    def reset(self):

        self.position = np.array(
            [115.0, 500.0],
            dtype=np.float64,
        )

        self.velocity = np.zeros(
            2,
            dtype=np.float64,
        )

        self.wind = np.random.normal(
            0.0,
            8.0,
            size=2,
        )

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
        ).astype(np.float64)

    def step_drone(self, action):

        action = np.asarray(
            action,
            dtype=np.float64,
        )

        action = np.clip(
            action,
            -1.0,
            1.0,
        )

        old_state = self.state()

        gain = self.domain["motor_gain"]
        drag = self.domain["drag"]
        qdrag = self.domain["qdrag"]
        bias = self.domain["bias"]

        acceleration = (
            gain * MAX_ACCEL * action
            - drag * self.velocity
            + self.wind
            - qdrag
            * np.abs(self.velocity)
            * self.velocity
            + bias
        )

        self.velocity += (
            acceleration * DT
        )

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

        # The wind evolves after the observed drone transition.
        self.wind = (
            WIND_DECAY * self.wind
            + np.random.normal(
                0.0,
                self.wind_noise,
                size=2,
            )
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
# 3. RECURSIVE LEAST SQUARES — SYSTEM IDENTIFICATION
# ============================================================

class RLSIdentifier:

    """
    Estimate:

        theta = [motor_gain, linear_drag]

    Observation equation for each axis:

        y = (delta_velocity / DT) - wind

        phi = [MAX_ACCEL * action, -velocity]

        y = phi.T @ theta

    Both axes share the same physical parameters.

    Each axis contributes one scalar RLS update.
    """

    def __init__(self):

        self.theta = np.array(
            [0.95, 0.06],
            dtype=np.float64,
        )

        self.P = (
            np.eye(2, dtype=np.float64)
            * 100.0
        )

        self.forgetting = (
            RLS_FORGETTING
        )

        self.observation_count = 0

        self.history = []

    def update_axis(
        self,
        phi,
        observation,
    ):

        phi = np.asarray(
            phi,
            dtype=np.float64,
        )

        denominator = (
            self.forgetting
            + phi @ self.P @ phi
        )

        gain = (
            self.P @ phi
            / denominator
        )

        prediction = (
            phi @ self.theta
        )

        error = (
            observation - prediction
        )

        self.theta += (
            gain * error
        )

        self.P = (
            self.P
            - np.outer(
                gain,
                phi @ self.P,
            )
        ) / self.forgetting

        self.theta = np.clip(
            self.theta,
            PARAMETER_MIN,
            PARAMETER_MAX,
        )

        self.P = (
            self.P + self.P.T
        ) / 2.0

        self.observation_count += 1

        return float(error)

    def update(
        self,
        previous_state,
        action,
        next_state,
    ):

        velocity = previous_state[2:4]
        wind = previous_state[4:6]

        next_velocity = next_state[2:4]

        delta_velocity = (
            next_velocity - velocity
        )

        errors = []

        for axis in range(2):

            phi = np.array(
                [
                    MAX_ACCEL * action[axis],
                    -velocity[axis],
                ],
                dtype=np.float64,
            )

            observation = (
                delta_velocity[axis] / DT
                - wind[axis]
            )

            error = self.update_axis(
                phi,
                observation,
            )

            errors.append(error)

        self.history.append(
            self.theta.copy()
        )

        return np.asarray(
            errors,
            dtype=np.float64,
        )

    def reset(self):

        self.__init__()


# ============================================================
# 4. RESIDUAL DYNAMICS MODEL — ONLINE RLS
# ============================================================

class ResidualRLS:

    """
    Learn acceleration that the linear physical model cannot explain.

    Residual features:

        phi = [
            1,
            vx,
            vy,
            wind_x,
            wind_y,
            action_x,
            action_y,
            abs(vx)*vx,
            abs(vy)*vy
        ]

    Output:
        [residual_acceleration_x,
         residual_acceleration_y]

    The model is updated after every observed transition.
    """

    FEATURE_COUNT = 9

    def __init__(self):

        self.W = np.zeros(
            (
                self.FEATURE_COUNT,
                2,
            ),
            dtype=np.float64,
        )

        self.P = (
            np.eye(
                self.FEATURE_COUNT,
                dtype=np.float64,
            )
            * 100.0
        )

        self.forgetting = (
            RESIDUAL_FORGETTING
        )

        self.update_count = 0
        self.error_history = []

    def features(
        self,
        state,
        action,
    ):

        vx, vy = state[2:4]
        wx, wy = state[4:6]

        ax, ay = action

        return np.array(
            [
                1.0,
                vx,
                vy,
                wx,
                wy,
                ax,
                ay,
                abs(vx) * vx,
                abs(vy) * vy,
            ],
            dtype=np.float64,
        )

    def predict(
        self,
        state,
        action,
    ):

        phi = self.features(
            state,
            action,
        )

        return (
            phi @ self.W
        )

    def update(
        self,
        state,
        action,
        observed_acceleration,
        predicted_linear_acceleration,
    ):

        phi = self.features(
            state,
            action,
        )

        residual = (
            observed_acceleration
            - predicted_linear_acceleration
        )

        denominator = (
            self.forgetting
            + phi @ self.P @ phi
        )

        gain = (
            self.P @ phi
            / denominator
        )

        prediction = (
            phi @ self.W
        )

        error = (
            residual - prediction
        )

        self.W += np.outer(
            gain,
            error,
        )

        self.P = (
            self.P
            - np.outer(
                gain,
                phi @ self.P,
            )
        ) / self.forgetting

        self.P = (
            self.P + self.P.T
        ) / 2.0

        self.update_count += 1

        self.error_history.append(
            float(
                np.linalg.norm(error)
            )
        )

        return error

    def reset(self):

        self.__init__()


# ============================================================
# 5. ADAPTIVE MODEL
# ============================================================

class AdaptiveDynamics:

    def __init__(
        self,
        use_residual=True,
    ):

        self.identifier = RLSIdentifier()

        self.residual = ResidualRLS()

        self.use_residual = bool(
            use_residual
        )

    @property
    def motor_gain(self):

        return float(
            self.identifier.theta[0]
        )

    @property
    def drag(self):

        return float(
            self.identifier.theta[1]
        )

    def predict_acceleration(
        self,
        state,
        action,
    ):

        acceleration = (
            self.motor_gain
            * MAX_ACCEL
            * action
            - self.drag * state[2:4]
            + state[4:6]
        )

        if self.use_residual:

            acceleration += (
                self.residual.predict(
                    state,
                    action,
                )
            )

        return acceleration

    def predict_next(
        self,
        state,
        action,
    ):

        next_state = state.copy()

        acceleration = (
            self.predict_acceleration(
                state,
                action,
            )
        )

        next_state[2:4] += (
            acceleration * DT
        )

        speed = np.linalg.norm(
            next_state[2:4]
        )

        if speed > MAX_SPEED:

            next_state[2:4] *= (
                MAX_SPEED / (speed + 1e-8)
            )

        next_state[:2] += (
            next_state[2:4] * DT
        )

        next_state[:2] = np.clip(
            next_state[:2],
            DRONE_RADIUS,
            [
                WORLD_W - DRONE_RADIUS,
                WORLD_H - DRONE_RADIUS,
            ],
        )

        # Future wind prediction uses its expected decay.
        next_state[4:6] *= WIND_DECAY

        next_state[4:6] = np.clip(
            next_state[4:6],
            -55.0,
            55.0,
        )

        return next_state

    def observe(
        self,
        previous_state,
        action,
        next_state,
    ):

        """
        Update order:

        1. Compute actual acceleration from observed velocity change.
        2. Update the linear physical parameter estimates.
        3. Compute the linear model's prediction using the old
           parameters, before the RLS update.
        4. Learn the residual using the observed acceleration.

        This keeps the residual target tied to the actual model
        prediction error rather than the already-updated estimate.
        """

        old_gain = self.motor_gain
        old_drag = self.drag

        actual_acceleration = (
            next_state[2:4]
            - previous_state[2:4]
        ) / DT

        linear_prediction = (
            old_gain * MAX_ACCEL * action
            - old_drag * previous_state[2:4]
            + previous_state[4:6]
        )

        # The RLS identifier uses both velocity axes.
        self.identifier.update(
            previous_state,
            action,
            next_state,
        )

        if self.use_residual:

            self.residual.update(
                previous_state,
                action,
                actual_acceleration,
                linear_prediction,
            )

    def reset(self):

        self.identifier.reset()
        self.residual.reset()


# ============================================================
# 6. FIXED MODEL AND ORACLE MODEL
# ============================================================

class FixedDynamics:

    def __init__(self):

        self.motor_gain = 1.0
        self.drag = 0.065

    def predict_next(
        self,
        state,
        action,
    ):

        next_state = state.copy()

        acceleration = (
            self.motor_gain
            * MAX_ACCEL
            * action
            - self.drag * state[2:4]
            + state[4:6]
        )

        next_state[2:4] += (
            acceleration * DT
        )

        speed = np.linalg.norm(
            next_state[2:4]
        )

        if speed > MAX_SPEED:

            next_state[2:4] *= (
                MAX_SPEED / (speed + 1e-8)
            )

        next_state[:2] += (
            next_state[2:4] * DT
        )

        next_state[:2] = np.clip(
            next_state[:2],
            DRONE_RADIUS,
            [
                WORLD_W - DRONE_RADIUS,
                WORLD_H - DRONE_RADIUS,
            ],
        )

        next_state[4:6] *= WIND_DECAY

        return next_state


class OracleDynamics:

    def __init__(self, domain):

        self.domain = domain

    def predict_next(
        self,
        state,
        action,
    ):

        next_state = state.copy()

        domain = self.domain

        acceleration = (
            domain["motor_gain"]
            * MAX_ACCEL
            * action
            - domain["drag"] * state[2:4]
            + state[4:6]
            - domain["qdrag"]
            * np.abs(state[2:4])
            * state[2:4]
            + domain["bias"]
        )

        next_state[2:4] += (
            acceleration * DT
        )

        speed = np.linalg.norm(
            next_state[2:4]
        )

        if speed > MAX_SPEED:

            next_state[2:4] *= (
                MAX_SPEED / (speed + 1e-8)
            )

        next_state[:2] += (
            next_state[2:4] * DT
        )

        next_state[:2] = np.clip(
            next_state[:2],
            DRONE_RADIUS,
            [
                WORLD_W - DRONE_RADIUS,
                WORLD_H - DRONE_RADIUS,
            ],
        )

        next_state[4:6] *= WIND_DECAY

        return next_state


# ============================================================
# 7. PD GUIDANCE AND OBSTACLE CORRECTION
# ============================================================

def pd_action(
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


def safety_action(
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

        correction = (
            delta / distance
        ) * (
            1.25
            * (SAFE_RADIUS - distance)
            / SAFE_RADIUS
        )

        action = (
            action + correction
        )

    return np.clip(
        action,
        -1.0,
        1.0,
    )


# ============================================================
# 8. MPC
# ============================================================

def trajectory_cost(
    state,
    action,
    target_position,
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

    boundary_margin = min(
        state[0],
        WORLD_W - state[0],
        state[1],
        WORLD_H - state[1],
    )

    boundary_violation = max(
        0.0,
        45.0 - boundary_margin,
    )

    return (
        0.045 * distance
        + 2.5 * obstacle_violation ** 2
        + 1.5 * boundary_violation ** 2
        + 0.015 * np.sum(action ** 2)
    )


def make_nominal_sequence(
    state,
    target,
    dynamics,
):

    sequence = []

    predicted_state = state.copy()

    for step in range(
        1,
        MPC_HORIZON + 1,
    ):

        action = pd_action(
            predicted_state,
            future_target_position(
                target,
                step,
            ),
            future_target_velocity(
                target,
                step,
            ),
        )

        action = safety_action(
            action,
            predicted_state[:2],
        )

        sequence.append(action)

        predicted_state = (
            dynamics.predict_next(
                predicted_state,
                action,
            )
        )

    return np.asarray(
        sequence,
        dtype=np.float64,
    )


def evaluate_sequence(
    state,
    sequence,
    target,
    dynamics,
):

    predicted_state = state.copy()

    total_cost = 0.0

    path = []

    for step, action in enumerate(
        sequence,
        start=1,
    ):

        predicted_state = (
            dynamics.predict_next(
                predicted_state,
                action,
            )
        )

        total_cost += trajectory_cost(
            predicted_state,
            action,
            future_target_position(
                target,
                step,
            ),
        )

        path.append(
            predicted_state[:2].copy()
        )

    return (
        float(total_cost),
        np.asarray(path),
    )


def choose_mpc_action(
    state,
    target,
    dynamics,
):

    nominal = make_nominal_sequence(
        state,
        target,
        dynamics,
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
    )

    candidates = np.clip(
        candidates,
        -1.0,
        1.0,
    )

    # Preserve a safe nominal candidate.
    candidates[0] = nominal

    best_cost = float("inf")
    best_sequence = nominal
    best_path = np.empty(
        (0, 2),
        dtype=np.float64,
    )

    for sequence in candidates:

        cost, path = evaluate_sequence(
            state,
            sequence,
            target,
            dynamics,
        )

        if cost < best_cost:

            best_cost = cost
            best_sequence = sequence
            best_path = path

    return (
        best_sequence[0].copy(),
        best_path,
        best_cost,
    )


# ============================================================
# 9. CONTROLLER FACTORY
# ============================================================

def make_controller(
    mode,
    domain,
):

    if mode == "Fixed-parameter MPC":

        return FixedDynamics()

    if mode == "Adaptive MPC":

        return AdaptiveDynamics(
            use_residual=False
        )

    if mode == "Adaptive + Residual MPC":

        return AdaptiveDynamics(
            use_residual=True
        )

    if mode == "Oracle MPC":

        return OracleDynamics(
            domain
        )

    raise ValueError(
        f"Unknown controller: {mode}"
    )


# ============================================================
# 10. EPISODE EXECUTION
# ============================================================

def run_episode(
    mode,
    domain,
    steps=BENCHMARK_STEPS,
    wind_noise=WIND_NOISE,
    keep_history=False,
):

    env = DroneEnv(
        domain=domain,
        wind_noise=wind_noise,
    )

    env.reset()

    dynamics = make_controller(
        mode,
        domain,
    )

    distances = []
    clearances = []
    efforts = []

    parameter_history = []
    residual_errors = []

    drone_history = []

    collisions = 0

    for _ in range(steps):

        previous_state = env.state()

        if mode == "PD":

            action = pd_action(
                previous_state,
                env.target.position(),
                env.target.velocity(),
            )

            path = np.empty(
                (0, 2),
                dtype=np.float64,
            )

        else:

            action, path, _ = (
                choose_mpc_action(
                    previous_state,
                    env.target,
                    dynamics,
                )
            )

        action = safety_action(
            action,
            previous_state[:2],
        )

        env.step(action)

        next_state = env.state()

        # Online learning uses the real observed transition.
        if isinstance(
            dynamics,
            AdaptiveDynamics,
        ):

            dynamics.observe(
                previous_state,
                action,
                next_state,
            )

            parameter_history.append(
                dynamics.identifier.theta.copy()
            )

            if dynamics.use_residual:

                if dynamics.residual.error_history:

                    residual_errors.append(
                        dynamics.residual.error_history[-1]
                    )

        target_position = (
            env.target.position()
        )

        distance = np.linalg.norm(
            env.position - target_position
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

        drone_history.append(
            env.position.copy()
        )

        env.target.advance()

    result = {
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

    if keep_history:

        result["distances"] = np.asarray(
            distances
        )

        result["clearances"] = np.asarray(
            clearances
        )

        result["parameter_history"] = (
            np.asarray(parameter_history)
        )

        result["residual_errors"] = (
            np.asarray(residual_errors)
        )

        result["drone_history"] = (
            np.asarray(drone_history)
        )

    return result


# ============================================================
# 11. BENCHMARK
# ============================================================

def benchmark():

    modes = [
        "PD",
        "Fixed-parameter MPC",
        "Adaptive MPC",
        "Adaptive + Residual MPC",
        "Oracle MPC",
    ]

    test_domains = DOMAINS

    results = {}

    for domain in test_domains:

        print(
            f"\nTesting domain: {domain['name']}"
        )

        results[domain["name"]] = {}

        for mode in modes:

            episode_results = []

            for episode in range(
                BENCHMARK_EPISODES
            ):

                result = run_episode(
                    mode,
                    domain,
                    steps=BENCHMARK_STEPS,
                    wind_noise=WIND_NOISE,
                )

                episode_results.append(
                    result
                )

            averaged = {
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

            results[domain["name"]][mode] = (
                averaged
            )

            print(
                f"  {mode:<27}"
                f"distance={averaged['mean_distance']:8.2f}  "
                f"clearance={averaged['minimum_clearance']:8.2f}  "
                f"collision steps={averaged['collision_steps']:6.1f}"
            )

    return results


def print_benchmark_table(results):

    print("\n" + "=" * 112)
    print("CONTROLLER COMPARISON")
    print("=" * 112)

    for domain, values in results.items():

        print(
            f"\nDOMAIN: {domain}"
        )

        print(
            f"{'Controller':<28}"
            f"{'Distance':>12}"
            f"{'Clearance':>13}"
            f"{'Collisions':>13}"
            f"{'Effort':>11}"
        )

        print("-" * 80)

        for mode, metrics in values.items():

            print(
                f"{mode:<28}"
                f"{metrics['mean_distance']:>12.2f}"
                f"{metrics['minimum_clearance']:>13.2f}"
                f"{metrics['collision_steps']:>13.2f}"
                f"{metrics['mean_effort']:>11.3f}"
            )


# ============================================================
# 12. VISUALIZATION
# ============================================================

def visualize_parameter_convergence(
    domain,
):

    result = run_episode(
        "Adaptive + Residual MPC",
        domain,
        steps=220,
        keep_history=True,
    )

    history = result[
        "parameter_history"
    ]

    if len(history) == 0:
        return

    true_gain = domain[
        "motor_gain"
    ]

    true_drag = domain[
        "drag"
    ]

    plt.figure(figsize=(10, 5))

    plt.plot(
        history[:, 0],
        label="Estimated motor gain",
    )

    plt.axhline(
        true_gain,
        linestyle="--",
        label="True motor gain",
    )

    plt.xlabel("Observed transition")
    plt.ylabel("Motor gain")
    plt.title("Online Motor-Gain Identification")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))

    plt.plot(
        history[:, 1],
        label="Estimated linear drag",
    )

    plt.axhline(
        true_drag,
        linestyle="--",
        label="True linear drag",
    )

    plt.xlabel("Observed transition")
    plt.ylabel("Drag coefficient")
    plt.title("Online Aerodynamic-Drag Identification")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    residual_errors = result[
        "residual_errors"
    ]

    if len(residual_errors) > 0:

        plt.figure(figsize=(10, 5))

        plt.plot(
            residual_errors,
            alpha=0.7,
            label="Residual prediction error",
        )

        window = min(
            20,
            len(residual_errors),
        )

        if window > 1:

            moving_average = np.convolve(
                residual_errors,
                np.ones(window) / window,
                mode="valid",
            )

            plt.plot(
                np.arange(
                    window - 1,
                    len(residual_errors),
                ),
                moving_average,
                label="Moving average",
            )

        plt.xlabel("Residual update")
        plt.ylabel("Acceleration error")
        plt.title("Online Residual Learning")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()


def visualize_benchmark(results):

    domains = list(
        results.keys()
    )

    modes = list(
        next(iter(results.values())).keys()
    )

    metrics = [
        (
            "mean_distance",
            "Mean target distance",
        ),
        (
            "minimum_clearance",
            "Minimum obstacle clearance",
        ),
        (
            "collision_steps",
            "Collision steps",
        ),
    ]

    for metric, title in metrics:

        plt.figure(figsize=(12, 5))

        x = np.arange(
            len(domains)
        )

        width = 0.15

        for i, mode in enumerate(modes):

            values = [
                results[domain][mode][metric]
                for domain in domains
            ]

            plt.bar(
                x + i * width,
                values,
                width,
                label=mode,
            )

        plt.xticks(
            x + width * (len(modes) - 1) / 2,
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
# 13. INTERACTIVE PYGAME DRONE
# ============================================================

def run_pygame():

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
        "Phase 4.21 — Online Adaptive Drone"
    )

    clock = pygame.time.Clock()

    font = pygame.font.SysFont(
        "consolas",
        16,
    )

    domain = dict(
        DOMAINS[3]
    )

    env = DroneEnv(
        domain=domain,
        wind_noise=WIND_NOISE,
    )

    env.reset()

    modes = [
        "Fixed-parameter MPC",
        "Adaptive MPC",
        "Adaptive + Residual MPC",
        "Oracle MPC",
    ]

    mode_index = 2

    dynamics = make_controller(
        modes[mode_index],
        domain,
    )

    paused = False
    speed = 1

    trail = []

    planned_path = np.empty(
        (0, 2),
        dtype=np.float64,
    )

    parameter_history = []

    last_action = np.zeros(
        2,
        dtype=np.float64,
    )

    def reset_simulation():

        nonlocal env
        nonlocal dynamics
        nonlocal trail
        nonlocal planned_path
        nonlocal parameter_history
        nonlocal last_action

        env = DroneEnv(
            domain=domain,
            wind_noise=WIND_NOISE,
        )

        env.reset()

        dynamics = make_controller(
            modes[mode_index],
            domain,
        )

        trail = []
        planned_path = np.empty(
            (0, 2),
            dtype=np.float64,
        )

        parameter_history = []

        last_action = np.zeros(
            2,
            dtype=np.float64,
        )

    def change_mode(index):

        nonlocal mode_index
        nonlocal dynamics

        mode_index = index

        dynamics = make_controller(
            modes[mode_index],
            domain,
        )

        parameter_history.clear()

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

        norm = np.linalg.norm(
            velocity
        )

        if norm < 1e-6:

            direction = np.array(
                [1.0, 0.0]
            )

        else:

            direction = velocity / norm

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

                endpoint = (
                    center + offset
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
                    (95, 195, 245),
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
            (65, 105, 75),
            True,
            points,
            2,
        )

        position = env.target.position()
        velocity = env.target.velocity()

        pygame.draw.circle(
            screen,
            (75, 235, 120),
            position.astype(int),
            12,
        )

        pygame.draw.circle(
            screen,
            (75, 235, 120),
            position.astype(int),
            22,
            1,
        )

        pygame.draw.line(
            screen,
            (75, 255, 130),
            position.astype(int),
            (
                position + velocity * 0.20
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

                    reset_simulation()

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

                    change_mode(0)

                elif event.key == pygame.K_2:

                    change_mode(1)

                elif event.key == pygame.K_3:

                    change_mode(2)

                elif event.key == pygame.K_4:

                    change_mode(3)

            elif event.type == pygame.MOUSEBUTTONDOWN:

                if event.button == 1:

                    click = np.array(
                        event.pos,
                        dtype=np.float64,
                    )

                    if np.linalg.norm(
                        click - OBSTACLE
                    ) > SAFE_RADIUS + 15:

                        env.target.center = click

        mode = modes[mode_index]

        if not paused:

            for _ in range(speed):

                previous_state = env.state()

                if mode == "PD":

                    action = pd_action(
                        previous_state,
                        env.target.position(),
                        env.target.velocity(),
                    )

                    planned_path = np.empty(
                        (0, 2),
                        dtype=np.float64,
                    )

                else:

                    action, planned_path, _ = (
                        choose_mpc_action(
                            previous_state,
                            env.target,
                            dynamics,
                        )
                    )

                action = safety_action(
                    action,
                    previous_state[:2],
                )

                last_action = action.copy()

                env.step(action)

                next_state = env.state()

                if isinstance(
                    dynamics,
                    AdaptiveDynamics,
                ):

                    dynamics.observe(
                        previous_state,
                        action,
                        next_state,
                    )

                    parameter_history.append(
                        dynamics.identifier.theta.copy()
                    )

                trail.append(
                    env.position.copy()
                )

                if len(trail) > 160:

                    trail.pop(0)

        # Draw background.
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

        # Obstacle and safety boundary.
        pygame.draw.circle(
            screen,
            (95, 95, 105),
            OBSTACLE.astype(int),
            OBSTACLE_RADIUS,
        )

        pygame.draw.circle(
            screen,
            (210, 105, 90),
            OBSTACLE.astype(int),
            SAFE_RADIUS,
            2,
        )

        draw_target()

        if len(trail) > 1:

            pygame.draw.lines(
                screen,
                (80, 140, 235),
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
            )
            - OBSTACLE_RADIUS
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

        if isinstance(
            dynamics,
            AdaptiveDynamics,
        ):

            draw_text(
                f"Estimated motor gain: "
                f"{dynamics.motor_gain:.4f}",
                15,
                142,
                (120, 225, 170),
            )

            draw_text(
                f"Estimated linear drag: "
                f"{dynamics.drag:.4f}",
                15,
                168,
                (120, 225, 170),
            )

            draw_text(
                f"RLS observations: "
                f"{dynamics.identifier.observation_count}",
                15,
                194,
                (120, 225, 170),
            )

            if dynamics.use_residual:

                draw_text(
                    f"Residual updates: "
                    f"{dynamics.residual.update_count}",
                    15,
                    220,
                    (240, 190, 110),
                )

        draw_text(
            f"Simulation speed: {speed}x",
            15,
            246,
        )

        draw_text(
            "1 Fixed MPC | 2 Adaptive MPC | "
            "3 Adaptive + Residual | 4 Oracle",
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
# 14. UNIT TESTS
# ============================================================

def run_tests():

    print("\nRunning tests...")

    passed = 0

    def check(condition, description):

        nonlocal passed

        assert condition, description

        passed += 1

        print(
            f"[PASS] {description}"
        )

    # Test 1 — RLS initial parameters.
    identifier = RLSIdentifier()

    check(
        identifier.theta.shape == (2,),
        "RLS estimates two physical parameters",
    )

    # Test 2 — RLS covariance.
    check(
        identifier.P.shape == (2, 2),
        "RLS covariance matrix has correct shape",
    )

    # Test 3 — Parameter bounds.
    check(
        np.all(identifier.theta >= PARAMETER_MIN)
        and np.all(identifier.theta <= PARAMETER_MAX),
        "Initial estimates are within configured bounds",
    )

    # Test 4 — Residual feature dimension.
    residual = ResidualRLS()

    sample_state = np.array(
        [
            100.0,
            120.0,
            15.0,
            -8.0,
            3.0,
            -2.0,
        ]
    )

    sample_action = np.array(
        [0.2, -0.3]
    )

    features = residual.features(
        sample_state,
        sample_action,
    )

    check(
        features.shape == (9,),
        "Residual model generates nine features",
    )

    # Test 5 — Residual prediction dimension.
    check(
        residual.predict(
            sample_state,
            sample_action,
        ).shape == (2,),
        "Residual model predicts two acceleration components",
    )

    # Test 6 — Residual model starts at zero.
    check(
        np.allclose(
            residual.predict(
                sample_state,
                sample_action,
            ),
            0.0,
        ),
        "Residual model initially predicts zero correction",
    )

    # Test 7 — Residual update count.
    residual.update(
        sample_state,
        sample_action,
        np.array([2.0, -1.0]),
        np.array([1.0, -0.5]),
    )

    check(
        residual.update_count == 1,
        "Residual model performs online updates",
    )

    # Test 8 — Residual prediction remains finite.
    check(
        np.all(
            np.isfinite(
                residual.predict(
                    sample_state,
                    sample_action,
                )
            )
        ),
        "Residual prediction is finite",
    )

    # Test 9 — Environment state shape.
    env = DroneEnv(
        domain=DOMAINS[0]
    )

    check(
        env.state().shape == (6,),
        "Drone observation includes position, velocity and wind",
    )

    # Test 10 — Environment transition.
    old_state = env.state()

    env.step_drone(
        np.array([0.5, -0.2])
    )

    new_state = env.state()

    check(
        not np.allclose(
            old_state,
            new_state,
        ),
        "Drone state changes after a control action",
    )

    # Test 11 — Finite environment state.
    check(
        np.all(np.isfinite(new_state)),
        "Environment state remains finite",
    )

    # Test 12 — Bounded speed.
    check(
        np.linalg.norm(
            env.velocity
        ) <= MAX_SPEED + 1e-6,
        "Drone speed remains bounded",
    )

    # Test 13 — Bounded wind.
    check(
        np.all(np.abs(env.wind) <= 55.0),
        "Wind remains within configured bounds",
    )

    # Test 14 — Adaptive model output.
    adaptive = AdaptiveDynamics(
        use_residual=True
    )

    prediction = adaptive.predict_next(
        env.state(),
        np.array([0.2, 0.1]),
    )

    check(
        prediction.shape == (6,)
        and np.all(np.isfinite(prediction)),
        "Adaptive model produces a valid next state",
    )

    # Test 15 — Online parameter update.
    previous = env.state()

    action = np.array(
        [0.7, -0.4]
    )

    env.step_drone(action)

    next_state = env.state()

    adaptive.observe(
        previous,
        action,
        next_state,
    )

    check(
        adaptive.identifier.observation_count == 2,
        "Two-axis RLS update records two observations",
    )

    # Test 16 — Estimated parameters stay bounded.
    check(
        np.all(
            adaptive.identifier.theta >= PARAMETER_MIN
        )
        and np.all(
            adaptive.identifier.theta <= PARAMETER_MAX
        ),
        "Estimated parameters remain within configured bounds",
    )

    # Test 17 — Target moves.
    target = MovingTarget(
        np.array([500.0, 300.0])
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

    # Test 18 — Nominal sequence.
    sequence = make_nominal_sequence(
        env.state(),
        env.target,
        adaptive,
    )

    check(
        sequence.shape == (
            MPC_HORIZON,
            2,
        ),
        "MPC nominal sequence has correct dimensions",
    )

    # Test 19 — Candidate action bounds.
    check(
        np.all(np.abs(sequence) <= 1.0),
        "Nominal MPC actions are bounded",
    )

    # Test 20 — Sequence evaluation.
    cost, path = evaluate_sequence(
        env.state(),
        sequence,
        env.target,
        adaptive,
    )

    check(
        np.isfinite(cost)
        and path.shape == (
            MPC_HORIZON,
            2,
        )
        and np.all(np.isfinite(path)),
        "MPC evaluates a finite predicted trajectory",
    )

    # Test 21 — Fixed model.
    fixed = FixedDynamics()

    check(
        fixed.predict_next(
            env.state(),
            np.zeros(2),
        ).shape == (6,),
        "Fixed model produces a valid next state",
    )

    # Test 22 — Oracle model.
    oracle = OracleDynamics(
        DOMAINS[0]
    )

    check(
        oracle.predict_next(
            env.state(),
            np.zeros(2),
        ).shape == (6,),
        "Oracle model produces a valid next state",
    )

    # Test 23 — Safety action bounds.
    safe = safety_action(
        np.array([2.0, -2.0]),
        env.position,
    )

    check(
        np.all(np.abs(safe) <= 1.0),
        "Safety correction respects action bounds",
    )

    # Test 24 — Reset.
    adaptive.reset()

    check(
        adaptive.identifier.observation_count == 0
        and adaptive.residual.update_count == 0,
        "Adaptive model reset clears online learning state",
    )

    print(
        f"\n{passed}/24 tests passed."
    )

    return passed == 24


# ============================================================
# 15. MAIN
# ============================================================

def main():

    print("=" * 76)

    print(
        "PHASE 4.21 — ONLINE ADAPTIVE MODEL-BASED RL"
    )

    print(
        "SYSTEM IDENTIFICATION / RESIDUAL LEARNING / MPC"
    )

    print("=" * 76)

    print(
        "\n[1/5] Running unit tests..."
    )

    if not run_tests():

        raise RuntimeError(
            "Unit tests failed."
        )

    print(
        "\n[2/5] Visualizing online parameter identification..."
    )

    visualize_parameter_convergence(
        DOMAINS[3]
    )

    print(
        "\n[3/5] Benchmarking controllers..."
    )

    results = benchmark()

    print_benchmark_table(
        results
    )

    print(
        "\n[4/5] Visualizing benchmark..."
    )

    visualize_benchmark(
        results
    )

    print(
        "\n[5/5] Starting interactive drone..."
    )

    run_pygame()


if __name__ == "__main__":

    main()
