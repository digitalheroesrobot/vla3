
"""
File: phase4/safe_online_adaptation_drone.py

PHASE 4.22
==========

SAFE ONLINE ADAPTATION
UNCERTAINTY-AWARE MPC
ADAPTIVE SAFETY MARGINS
CONTROL BARRIER FUNCTIONS

PURPOSE
-------
Build a moving-target drone controller that adapts its physical model
while accounting for uncertainty and obstacle safety.

WHAT YOU LEARN
--------------

STEP 1 — Online System Identification
    Estimate motor gain and aerodynamic drag using RLS.

STEP 2 — Model Uncertainty
    Estimate parameter uncertainty from the RLS covariance matrix
    and prediction uncertainty from observed residual errors.

STEP 3 — Adaptive Safety Margins
    Increase the required obstacle clearance when model uncertainty
    increases. Reduce it cautiously when confidence improves.

STEP 4 — Control Barrier Functions
    Convert obstacle and boundary safety conditions into linear
    constraints on the drone's acceleration action.

STEP 5 — Safety Filtering
    Project an MPC-requested action onto the feasible safe-action
    region using iterative half-space projection.

STEP 6 — Uncertainty-Aware MPC
    Penalize uncertain predicted trajectories and replan after
    every real transition.

STEP 7 — Evaluation
    Compare:
        1. Fixed MPC
        2. Adaptive MPC
        3. Uncertainty-aware MPC
        4. Uncertainty-aware MPC + CBF
        5. Oracle MPC + CBF

STEP 8 — Interactive Drone
    Click to change the moving target's trajectory center.
    Watch the safety margin and model confidence change.

MATHEMATICAL FOUNDATIONS
-----------------------

1. ONLINE PARAMETER ESTIMATION

    a = g * MAX_ACCEL * u - d*v + wind

    theta = [g, d]

    phi = [MAX_ACCEL*u, -v]

    y = phi.T @ theta

RLS:

    K = P*phi / (lambda + phi.T*P*phi)

    theta = theta + K*(y - phi.T*theta)

    P = (P - K*phi.T*P)/lambda

2. PARAMETER UNCERTAINTY

    A simple uncertainty indicator:

        sigma_parameter = sqrt(trace(P))

    This is a confidence indicator, not a calibrated probability
    distribution over the physical parameters.

3. ADAPTIVE SAFETY MARGIN

    R_safe = R_obstacle + base_margin + uncertainty_margin

    uncertainty_margin is bounded to prevent excessive margins.

4. CONTROL BARRIER FUNCTION

    Obstacle barrier:

        h(p) = ||p - p_obstacle||^2 - R_safe^2

    A discrete-time safety condition:

        h(p_next) >= (1 - alpha)*h(p)

    Predicted position:

        p_next = p + v*dt + 0.5*a*dt^2

    Approximate acceleration:

        a = g*MAX_ACCEL*u - d*v + wind + residual

    Linearizing the barrier condition gives a half-space constraint:

        A*u >= b

    The controller projects its requested action onto these
    half-spaces.

5. UNCERTAINTY-AWARE MPC

    J = tracking_cost
        + obstacle_cost
        + boundary_cost
        + control_effort
        + uncertainty_penalty

IMPORTANT
---------
This is an educational simplified 2D drone model.

The CBF constraints are locally linearized. The projection algorithm
is approximate, and the safety filter is not a formal guarantee under
model errors, infeasible constraints, discrete-time effects, or
unmodeled dynamics.

Real flight systems require validated dynamics, robust constraint
handling, state-estimation guarantees, actuator constraints, and
hardware-in-the-loop testing.

INSTALL
-------
    pip install numpy matplotlib pygame

RUN
---
    python phase4/safe_online_adaptation_drone.py

CONTROLS
--------
    1 : Fixed MPC
    2 : Adaptive MPC
    3 : Uncertainty-aware MPC
    4 : Uncertainty-aware MPC + CBF
    5 : Oracle MPC + CBF

    SPACE : Pause
    R     : Reset
    + / - : Simulation speed
    Click : Change target trajectory center
    ESC   : Exit

ROBOTICS APPLICATION
--------------------
This architecture is useful when a robot must navigate while:
    - its dynamics are still being identified;
    - sensor measurements are noisy;
    - payload or motor performance changes;
    - obstacles must be avoided;
    - the controller must account for model uncertainty.
"""

from __future__ import annotations

import math
import random

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 0. CONFIGURATION
# ============================================================

SEED = 422

np.random.seed(SEED)
random.seed(SEED)

WORLD_W = 900
WORLD_H = 650

DT = 0.08

MAX_SPEED = 150.0
MAX_ACCEL = 125.0
DRONE_RADIUS = 12.0

OBSTACLE = np.array(
    [450.0, 330.0],
    dtype=np.float64,
)

OBSTACLE_RADIUS = 62.0

BASE_SAFETY_MARGIN = 34.0

MAX_ADAPTIVE_MARGIN = 65.0

TARGET_A = 165.0
TARGET_B = 105.0
TARGET_OMEGA = 0.55

MPC_HORIZON = 9
MPC_CANDIDATES = 36

ACTION_NOISE = 0.38

RLS_FORGETTING = 0.995

WIND_DECAY = 0.88
WIND_NOISE = 13.0

CBF_ALPHA = 0.35

PROJECTION_ITERATIONS = 20

# Controller uncertainty penalty.
UNCERTAINTY_WEIGHT = 0.055

# Physical domains.
DOMAINS = [
    {
        "name": "nominal",
        "gain": 1.00,
        "drag": 0.065,
        "qdrag": 0.00015,
        "bias": np.array([0.0, 0.0]),
    },
    {
        "name": "heavy_payload",
        "gain": 0.76,
        "drag": 0.090,
        "qdrag": 0.00024,
        "bias": np.array([1.0, -1.0]),
    },
    {
        "name": "motor_degradation",
        "gain": 0.62,
        "drag": 0.110,
        "qdrag": 0.00030,
        "bias": np.array([2.0, -1.5]),
    },
    {
        "name": "unseen_combined",
        "gain": 1.35,
        "drag": 0.125,
        "qdrag": 0.00035,
        "bias": np.array([-2.0, 2.0]),
    },
]

PARAM_MIN = np.array([0.35, 0.0])
PARAM_MAX = np.array([1.65, 0.35])


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


def future_target(target, step):

    phase = (
        target.phase
        + TARGET_OMEGA * DT * step
    )

    return (
        target.position(phase),
        target.velocity(phase),
    )


# ============================================================
# 2. DRONE ENVIRONMENT
# ============================================================

class DroneEnv:

    """
    State:

        [x, y, vx, vy, wind_x, wind_y]

    Action:

        [ax, ay] in [-1, 1]

    Actual acceleration:

        a = gain*MAX_ACCEL*u
            - drag*v
            + wind
            - qdrag*abs(v)*v
            + bias
    """

    def __init__(self, domain):

        self.domain = domain

        self.target = MovingTarget(
            [710.0, 315.0]
        )

        self.reset()

    def reset(self):

        self.position = np.array(
            [110.0, 500.0],
            dtype=np.float64,
        )

        self.velocity = np.zeros(
            2,
            dtype=np.float64,
        )

        self.wind = np.random.normal(
            0.0,
            7.0,
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
        )

    def step_drone(self, action):

        action = np.clip(
            np.asarray(action, dtype=np.float64),
            -1.0,
            1.0,
        )

        old_state = self.state()

        d = self.domain

        acceleration = (
            d["gain"] * MAX_ACCEL * action
            - d["drag"] * self.velocity
            + self.wind
            - d["qdrag"]
            * np.abs(self.velocity)
            * self.velocity
            + d["bias"]
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

        self.position = np.clip(
            self.position,
            DRONE_RADIUS,
            [
                WORLD_W - DRONE_RADIUS,
                WORLD_H - DRONE_RADIUS,
            ],
        )

        self.wind = (
            WIND_DECAY * self.wind
            + np.random.normal(
                0.0,
                WIND_NOISE,
                size=2,
            )
        )

        self.wind = np.clip(
            self.wind,
            -50.0,
            50.0,
        )

        return self.state() - old_state

    def step(self, action):

        self.step_drone(action)
        self.target.advance()

        return self.state()


# ============================================================
# 3. ONLINE SYSTEM IDENTIFICATION
# ============================================================

class RLSIdentifier:

    def __init__(self):

        self.theta = np.array(
            [0.95, 0.06],
            dtype=np.float64,
        )

        self.P = (
            np.eye(2) * 100.0
        )

        self.count = 0

    def update_axis(
        self,
        phi,
        observation,
    ):

        denominator = (
            RLS_FORGETTING
            + phi @ self.P @ phi
        )

        K = (
            self.P @ phi
            / denominator
        )

        error = (
            observation
            - phi @ self.theta
        )

        self.theta += (
            K * error
        )

        self.P = (
            self.P
            - np.outer(
                K,
                phi @ self.P,
            )
        ) / RLS_FORGETTING

        self.theta = np.clip(
            self.theta,
            PARAM_MIN,
            PARAM_MAX,
        )

        self.P = (
            self.P + self.P.T
        ) / 2.0

        self.count += 1

        return float(error)

    def update(
        self,
        previous,
        action,
        current,
    ):

        velocity = previous[2:4]
        wind = previous[4:6]

        delta_velocity = (
            current[2:4]
            - velocity
        )

        errors = []

        for axis in range(2):

            phi = np.array(
                [
                    MAX_ACCEL * action[axis],
                    -velocity[axis],
                ]
            )

            observation = (
                delta_velocity[axis] / DT
                - wind[axis]
            )

            errors.append(
                self.update_axis(
                    phi,
                    observation,
                )
            )

        return np.asarray(errors)

    def uncertainty(self):

        # This is a heuristic scalar confidence indicator.
        return float(
            np.sqrt(
                max(
                    0.0,
                    np.trace(self.P),
                )
            )
        )


# ============================================================
# 4. RESIDUAL ERROR ESTIMATION
# ============================================================

class ResidualEstimator:

    """
    Exponentially weighted residual statistics.

    The mean estimates systematic prediction error.
    The variance estimates remaining unpredictable error.

    This is intentionally a compact residual estimator rather
    than a second high-capacity learned dynamics model.
    """

    def __init__(self):

        self.mean = np.zeros(2)

        self.variance = np.ones(2) * 16.0

        self.count = 0

        self.beta = 0.96

    def update(self, residual):

        residual = np.asarray(
            residual,
            dtype=np.float64,
        )

        old_mean = self.mean.copy()

        self.mean = (
            self.beta * self.mean
            + (1.0 - self.beta) * residual
        )

        squared_error = (
            residual - old_mean
        ) ** 2

        self.variance = (
            self.beta * self.variance
            + (1.0 - self.beta) * squared_error
        )

        self.variance = np.maximum(
            self.variance,
            0.05,
        )

        self.count += 1

    def uncertainty(self):

        return float(
            np.sqrt(
                np.sum(self.variance)
            )
        )

    def reset(self):

        self.__init__()


# ============================================================
# 5. ADAPTIVE DYNAMICS
# ============================================================

class AdaptiveDynamics:

    def __init__(self):

        self.identifier = RLSIdentifier()

        self.residual = ResidualEstimator()

        self.history = []

    @property
    def gain(self):

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

        return (
            self.gain * MAX_ACCEL * action
            - self.drag * state[2:4]
            + state[4:6]
            + self.residual.mean
        )

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

        next_state[4:6] *= WIND_DECAY

        return next_state

    def observe(
        self,
        previous,
        action,
        current,
    ):

        observed_acceleration = (
            current[2:4]
            - previous[2:4]
        ) / DT

        old_prediction = (
            self.gain * MAX_ACCEL * action
            - self.drag * previous[2:4]
            + previous[4:6]
            + self.residual.mean
        )

        residual_error = (
            observed_acceleration
            - old_prediction
        )

        self.identifier.update(
            previous,
            action,
            current,
        )

        self.residual.update(
            residual_error
        )

        self.history.append(
            [
                self.gain,
                self.drag,
                self.uncertainty(),
                self.residual.uncertainty(),
            ]
        )

    def uncertainty(self):

        return (
            self.identifier.uncertainty()
            + 0.08 * self.residual.uncertainty()
        )

    def reset(self):

        self.__init__()


# ============================================================
# 6. FIXED AND ORACLE MODELS
# ============================================================

class FixedDynamics:

    def __init__(self):

        self.gain = 1.0
        self.drag = 0.065

    def predict_acceleration(
        self,
        state,
        action,
    ):

        return (
            self.gain * MAX_ACCEL * action
            - self.drag * state[2:4]
            + state[4:6]
        )

    def predict_next(
        self,
        state,
        action,
    ):

        return constrained_transition(
            state,
            self.predict_acceleration(
                state,
                action,
            ),
        )

    def uncertainty(self):

        return 0.0


class OracleDynamics:

    def __init__(self, domain):

        self.domain = domain

    def predict_acceleration(
        self,
        state,
        action,
    ):

        d = self.domain

        return (
            d["gain"] * MAX_ACCEL * action
            - d["drag"] * state[2:4]
            + state[4:6]
            - d["qdrag"]
            * np.abs(state[2:4])
            * state[2:4]
            + d["bias"]
        )

    def predict_next(
        self,
        state,
        action,
    ):

        return constrained_transition(
            state,
            self.predict_acceleration(
                state,
                action,
            ),
        )

    def uncertainty(self):

        return 0.0


def constrained_transition(
    state,
    acceleration,
):

    next_state = state.copy()

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
# 7. ADAPTIVE SAFETY MARGIN
# ============================================================

def adaptive_safety_margin(dynamics):

    uncertainty = dynamics.uncertainty()

    # The uncertainty indicator is not a calibrated standard
    # deviation. This mapping is a tunable engineering heuristic.

    extra_margin = (
        MAX_ADAPTIVE_MARGIN
        * uncertainty
        / (uncertainty + 8.0)
    )

    return (
        BASE_SAFETY_MARGIN
        + extra_margin
    )


# ============================================================
# 8. CONTROL BARRIER FUNCTION CONSTRAINTS
# ============================================================

def cbf_constraints(
    state,
    dynamics,
    safety_margin,
):
    """
    Build linear constraints A @ action >= b.

    Obstacle:

        h = ||p - obstacle||^2 - R_safe^2

    Approximate next position:

        p_next = p + v*dt + 0.5*a*dt^2

    CBF condition:

        h_next >= (1-alpha)*h

    The quadratic barrier is linearized around the free predicted
    position. This yields an approximate action half-space.

    World boundaries are represented by four additional
    next-position constraints.
    """

    p = state[:2]
    v = state[2:4]

    gain = getattr(
        dynamics,
        "gain",
        1.0,
    )

    drag = getattr(
        dynamics,
        "drag",
        0.065,
    )

    residual = getattr(
        getattr(dynamics, "residual", None),
        "mean",
        np.zeros(2),
    )

    base_acceleration = (
        -drag * v
        + state[4:6]
        + residual
    )

    action_acceleration = (
        gain * MAX_ACCEL
    )

    # Predicted position:
    #
    # p_next = p_free + B @ action

    p_free = (
        p
        + v * DT
        + 0.5 * base_acceleration * DT**2
    )

    B = (
        0.5
        * action_acceleration
        * DT**2
        * np.eye(2)
    )

    constraints_A = []
    constraints_b = []

    # --------------------------------------------------------
    # Obstacle CBF
    # --------------------------------------------------------

    delta = (
        p_free - OBSTACLE
    )

    safe_radius = (
        OBSTACLE_RADIUS
        + safety_margin
        + DRONE_RADIUS
    )

    h_current = (
        np.sum((p - OBSTACLE) ** 2)
        - safe_radius**2
    )

    # First-order approximation of h(p_next).
    #
    # h(p_free + B*u)
    # ~= h(p_free) + grad(h).T @ B*u

    h_free = (
        np.sum(delta**2)
        - safe_radius**2
    )

    gradient = (
        2.0 * delta
    )

    obstacle_A = (
        gradient @ B
    )

    obstacle_b = (
        (1.0 - CBF_ALPHA) * h_current
        - h_free
    )

    constraints_A.append(
        obstacle_A
    )

    constraints_b.append(
        obstacle_b
    )

    # --------------------------------------------------------
    # Boundary constraints
    # --------------------------------------------------------

    lower = (
        DRONE_RADIUS + 4.0
    )

    upper_x = (
        WORLD_W - DRONE_RADIUS - 4.0
    )

    upper_y = (
        WORLD_H - DRONE_RADIUS - 4.0
    )

    # x_next >= lower
    constraints_A.append(
        B[0].copy()
    )

    constraints_b.append(
        lower - p_free[0]
    )

    # x_next <= upper_x
    constraints_A.append(
        -B[0].copy()
    )

    constraints_b.append(
        p_free[0] - upper_x
    )

    # y_next >= lower
    constraints_A.append(
        B[1].copy()
    )

    constraints_b.append(
        lower - p_free[1]
    )

    # y_next <= upper_y
    constraints_A.append(
        -B[1].copy()
    )

    constraints_b.append(
        p_free[1] - upper_y
    )

    return (
        np.asarray(
            constraints_A,
            dtype=np.float64,
        ),
        np.asarray(
            constraints_b,
            dtype=np.float64,
        ),
    )


# ============================================================
# 9. SAFETY FILTER — HALF-SPACE PROJECTION
# ============================================================

def project_action_to_safe_set(
    requested_action,
    A,
    b,
    iterations=PROJECTION_ITERATIONS,
):
    """
    Solve approximately:

        minimize ||u - u_requested||^2

        subject to:
            A_i @ u >= b_i
            -1 <= u <= 1

    Cyclic projections are used for simplicity.

    This is an approximate projection. Constraints can become
    mutually infeasible, and later projections can disturb
    previously satisfied constraints.

    The function returns:
        filtered_action
        maximum_constraint_violation
        feasible_flag
    """

    action = np.clip(
        np.asarray(
            requested_action,
            dtype=np.float64,
        ),
        -1.0,
        1.0,
    ).copy()

    for _ in range(iterations):

        for row, lower_bound in zip(A, b):

            violation = (
                lower_bound - row @ action
            )

            norm_squared = (
                row @ row
            )

            if (
                violation > 0.0
                and norm_squared > 1e-12
            ):

                action += (
                    violation
                    / norm_squared
                ) * row

                action = np.clip(
                    action,
                    -1.0,
                    1.0,
                )

    violations = (
        b - A @ action
    )

    maximum_violation = max(
        0.0,
        float(
            np.max(violations)
        ),
    )

    feasible = (
        maximum_violation < 1e-3
    )

    return (
        action,
        maximum_violation,
        feasible,
    )


def safety_filter(
    requested_action,
    state,
    dynamics,
    use_cbf,
):
    """
    Fixed MPC and ordinary adaptive MPC use a safety margin
    based on the nominal physical model.

    Uncertainty-aware modes enlarge the margin according to
    estimated model uncertainty.

    The CBF filter is enabled only in the CBF modes.
    """

    if not use_cbf:

        return (
            np.clip(
                requested_action,
                -1.0,
                1.0,
            ),
            0.0,
            True,
        )

    margin = adaptive_safety_margin(
        dynamics
    )

    A, b = cbf_constraints(
        state,
        dynamics,
        margin,
    )

    return project_action_to_safe_set(
        requested_action,
        A,
        b,
    )


# ============================================================
# 10. UNCERTAINTY-AWARE MPC
# ============================================================

def pd_action(
    state,
    target_position,
    target_velocity,
    dynamics,
):

    error = (
        target_position - state[:2]
    )

    desired_velocity = (
        target_velocity + 0.55 * error
    )

    desired_velocity = np.clip(
        desired_velocity,
        -MAX_SPEED,
        MAX_SPEED,
    )

    acceleration = (
        1.8 * (
            desired_velocity - state[2:4]
        )
        - state[4:6]
    )

    gain = max(
        0.35,
        getattr(
            dynamics,
            "gain",
            1.0,
        ),
    )

    return np.clip(
        acceleration
        / (gain * MAX_ACCEL),
        -1.0,
        1.0,
    )


def make_nominal_sequence(
    state,
    target,
    dynamics,
    use_cbf,
):

    sequence = []

    predicted_state = state.copy()

    for step in range(
        1,
        MPC_HORIZON + 1,
    ):

        target_position, target_velocity = (
            future_target(
                target,
                step,
            )
        )

        action = pd_action(
            predicted_state,
            target_position,
            target_velocity,
            dynamics,
        )

        if use_cbf:

            action, _, _ = safety_filter(
                action,
                predicted_state,
                dynamics,
                True,
            )

        sequence.append(
            action
        )

        predicted_state = (
            dynamics.predict_next(
                predicted_state,
                action,
            )
        )

    return np.asarray(
        sequence
    )


def evaluate_sequence(
    state,
    sequence,
    target,
    dynamics,
    uncertainty_aware,
    use_cbf,
):

    predicted_state = state.copy()

    total_cost = 0.0

    path = []

    margin = (
        adaptive_safety_margin(dynamics)
        if uncertainty_aware
        else BASE_SAFETY_MARGIN
    )

    for step, requested_action in enumerate(
        sequence,
        start=1,
    ):

        action = requested_action

        if use_cbf:

            action, _, _ = safety_filter(
                action,
                predicted_state,
                dynamics,
                True,
            )

        predicted_state = (
            dynamics.predict_next(
                predicted_state,
                action,
            )
        )

        target_position, _ = (
            future_target(
                target,
                step,
            )
        )

        distance = np.linalg.norm(
            predicted_state[:2]
            - target_position
        )

        obstacle_distance = np.linalg.norm(
            predicted_state[:2]
            - OBSTACLE
        )

        safe_radius = (
            OBSTACLE_RADIUS
            + DRONE_RADIUS
            + margin
        )

        obstacle_violation = max(
            0.0,
            safe_radius - obstacle_distance,
        )

        boundary_margin = min(
            predicted_state[0],
            WORLD_W - predicted_state[0],
            predicted_state[1],
            WORLD_H - predicted_state[1],
        )

        boundary_violation = max(
            0.0,
            38.0 - boundary_margin,
        )

        total_cost += (
            0.045 * distance
            + 2.0 * obstacle_violation**2
            + 1.5 * boundary_violation**2
            + 0.015 * np.sum(action**2)
        )

        if uncertainty_aware:

            total_cost += (
                UNCERTAINTY_WEIGHT
                * dynamics.uncertainty()
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
    uncertainty_aware=False,
    use_cbf=False,
):

    nominal = make_nominal_sequence(
        state,
        target,
        dynamics,
        use_cbf,
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

    # Preserve the safe nominal sequence.
    candidates[0] = nominal

    best_cost = float("inf")

    best_sequence = nominal

    best_path = np.empty(
        (0, 2)
    )

    for sequence in candidates:

        cost, path = evaluate_sequence(
            state,
            sequence,
            target,
            dynamics,
            uncertainty_aware,
            use_cbf,
        )

        if cost < best_cost:

            best_cost = cost
            best_sequence = sequence
            best_path = path

    requested = best_sequence[0].copy()

    filtered, violation, feasible = (
        safety_filter(
            requested,
            state,
            dynamics,
            use_cbf,
        )
    )

    return (
        filtered,
        requested,
        best_path,
        best_cost,
        violation,
        feasible,
    )


# ============================================================
# 11. CONTROLLER MODES
# ============================================================

MODES = [
    "Fixed MPC",
    "Adaptive MPC",
    "Uncertainty-aware MPC",
    "Uncertainty-aware MPC + CBF",
    "Oracle MPC + CBF",
]


def make_dynamics(mode, domain):

    if mode == "Fixed MPC":

        return FixedDynamics()

    if mode in (
        "Adaptive MPC",
        "Uncertainty-aware MPC",
        "Uncertainty-aware MPC + CBF",
    ):

        return AdaptiveDynamics()

    if mode == "Oracle MPC + CBF":

        return OracleDynamics(domain)

    raise ValueError(mode)


def mode_options(mode):

    return {
        "uncertainty_aware": mode in (
            "Uncertainty-aware MPC",
            "Uncertainty-aware MPC + CBF",
        ),
        "use_cbf": mode in (
            "Uncertainty-aware MPC + CBF",
            "Oracle MPC + CBF",
        ),
    }


# ============================================================
# 12. EPISODE EXECUTION
# ============================================================

def run_episode(
    mode,
    domain,
    steps=180,
    keep_history=False,
):

    env = DroneEnv(domain)

    env.reset()

    dynamics = make_dynamics(
        mode,
        domain,
    )

    options = mode_options(mode)

    distances = []
    clearances = []
    margins = []
    uncertainties = []
    violations = []

    collision_steps = 0
    infeasible_steps = 0

    for _ in range(steps):

        previous = env.state()

        (
            action,
            requested,
            path,
            cost,
            violation,
            feasible,
        ) = choose_mpc_action(
            previous,
            env.target,
            dynamics,
            **options,
        )

        env.step(action)

        current = env.state()

        if isinstance(
            dynamics,
            AdaptiveDynamics,
        ):

            dynamics.observe(
                previous,
                action,
                current,
            )

        env.target.advance()

        distance = np.linalg.norm(
            env.position
            - env.target.position()
        )

        clearance = (
            np.linalg.norm(
                env.position - OBSTACLE
            )
            - OBSTACLE_RADIUS
        )

        if clearance < 0:

            collision_steps += 1

        if not feasible:

            infeasible_steps += 1

        distances.append(distance)
        clearances.append(clearance)
        violations.append(violation)

        margins.append(
            adaptive_safety_margin(dynamics)
            if options["uncertainty_aware"]
            else BASE_SAFETY_MARGIN
        )

        uncertainties.append(
            dynamics.uncertainty()
        )

    result = {
        "mean_distance": float(
            np.mean(distances)
        ),
        "minimum_clearance": float(
            np.min(clearances)
        ),
        "collision_steps": int(
            collision_steps
        ),
        "mean_margin": float(
            np.mean(margins)
        ),
        "mean_uncertainty": float(
            np.mean(uncertainties)
        ),
        "infeasible_steps": int(
            infeasible_steps
        ),
    }

    if keep_history:

        result["distances"] = np.asarray(
            distances
        )

        result["clearances"] = np.asarray(
            clearances
        )

        result["margins"] = np.asarray(
            margins
        )

        result["uncertainties"] = np.asarray(
            uncertainties
        )

        result["violations"] = np.asarray(
            violations
        )

        result["parameter_history"] = np.asarray(
            dynamics.history
            if isinstance(dynamics, AdaptiveDynamics)
            else []
        )

    return result


# ============================================================
# 13. BENCHMARK
# ============================================================

def benchmark():

    results = {}

    for domain in DOMAINS:

        print(
            f"\nDomain: {domain['name']}"
        )

        results[domain["name"]] = {}

        for mode in MODES:

            episode_results = []

            for _ in range(2):

                episode_results.append(
                    run_episode(
                        mode,
                        domain,
                        steps=150,
                    )
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
                f"{mode:<34}"
                f"distance={averaged['mean_distance']:8.2f} "
                f"clearance={averaged['minimum_clearance']:8.2f} "
                f"collisions={averaged['collision_steps']:5.1f} "
                f"infeasible={averaged['infeasible_steps']:5.1f}"
            )

    return results


def visualize_benchmark(results):

    domains = list(
        results.keys()
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
        (
            "infeasible_steps",
            "CBF infeasible steps",
        ),
    ]

    for metric, title in metrics:

        plt.figure(figsize=(12, 5))

        x = np.arange(
            len(domains)
        )

        width = 0.15

        for i, mode in enumerate(MODES):

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
            x + width * 2,
            domains,
            rotation=15,
        )

        plt.title(title)
        plt.ylabel(title)
        plt.grid(axis="y")
        plt.legend()
        plt.tight_layout()
        plt.show()


def visualize_adaptation(domain):

    result = run_episode(
        "Uncertainty-aware MPC + CBF",
        domain,
        steps=220,
        keep_history=True,
    )

    history = result[
        "parameter_history"
    ]

    if len(history) == 0:

        print(
            "No adaptive parameter history available."
        )

        return

    plt.figure(figsize=(10, 5))

    plt.plot(
        history[:, 0],
        label="Estimated gain",
    )

    plt.axhline(
        domain["gain"],
        linestyle="--",
        label="True gain",
    )

    plt.title(
        "Online Motor-Gain Estimation"
    )

    plt.xlabel("Transition")
    plt.ylabel("Motor gain")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))

    plt.plot(
        history[:, 1],
        label="Estimated drag",
    )

    plt.axhline(
        domain["drag"],
        linestyle="--",
        label="True drag",
    )

    plt.title(
        "Online Drag Estimation"
    )

    plt.xlabel("Transition")
    plt.ylabel("Drag")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))

    plt.plot(
        result["margins"],
        label="Adaptive safety margin",
    )

    plt.xlabel("Transition")
    plt.ylabel("Safety margin")
    plt.title(
        "Safety Margin Adaptation"
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))

    plt.plot(
        result["uncertainties"],
        label="Model uncertainty indicator",
    )

    plt.xlabel("Transition")
    plt.ylabel("Uncertainty indicator")
    plt.title(
        "Online Model Uncertainty"
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# 14. INTERACTIVE PYGAME DRONE
# ============================================================

def run_pygame():

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
        "Phase 4.22 — Safe Adaptive Drone"
    )

    clock = pygame.time.Clock()

    font = pygame.font.SysFont(
        "consolas",
        15,
    )

    domain = dict(
        DOMAINS[2]
    )

    env = DroneEnv(domain)

    env.reset()

    mode_index = 3

    dynamics = make_dynamics(
        MODES[mode_index],
        domain,
    )

    paused = False
    speed = 1

    trail = []

    planned_path = np.empty(
        (0, 2)
    )

    requested_action = np.zeros(2)
    executed_action = np.zeros(2)

    last_violation = 0.0
    last_feasible = True

    def reset_simulation():

        nonlocal env
        nonlocal dynamics
        nonlocal trail
        nonlocal planned_path
        nonlocal requested_action
        nonlocal executed_action

        env = DroneEnv(domain)

        env.reset()

        dynamics = make_dynamics(
            MODES[mode_index],
            domain,
        )

        trail = []

        planned_path = np.empty(
            (0, 2)
        )

        requested_action = np.zeros(2)
        executed_action = np.zeros(2)

    def change_mode(index):

        nonlocal mode_index
        nonlocal dynamics

        mode_index = index

        dynamics = make_dynamics(
            MODES[mode_index],
            domain,
        )

    def draw_text(
        text,
        x,
        y,
        color=(235, 235, 235),
    ):

        screen.blit(
            font.render(
                text,
                True,
                color,
            ),
            (x, y),
        )

    def draw_drone(position, velocity):

        center = np.asarray(
            position
        )

        speed_value = np.linalg.norm(
            velocity
        )

        if speed_value > 1e-6:

            direction = (
                velocity / speed_value
            )

        else:

            direction = np.array(
                [1.0, 0.0]
            )

        side = np.array(
            [
                -direction[1],
                direction[0],
            ]
        )

        for forward in (-1, 1):

            for lateral in (-1, 1):

                endpoint = (
                    center
                    + 17 * forward * direction
                    + 17 * lateral * side
                )

                pygame.draw.line(
                    screen,
                    (220, 225, 235),
                    center.astype(int),
                    endpoint.astype(int),
                    5,
                )

                pygame.draw.circle(
                    screen,
                    (85, 190, 240),
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
            (255, 190, 55),
            (
                center + direction * 23
            ).astype(int),
            4,
        )

    def draw_target():

        points = []

        for i in range(100):

            phase = (
                env.target.phase
                + 2.0 * math.pi * i / 100
            )

            points.append(
                env.target.position(
                    phase
                ).astype(int)
            )

        pygame.draw.lines(
            screen,
            (60, 100, 70),
            True,
            points,
            2,
        )

        p = env.target.position()
        v = env.target.velocity()

        pygame.draw.circle(
            screen,
            (70, 235, 110),
            p.astype(int),
            11,
        )

        pygame.draw.line(
            screen,
            (70, 255, 120),
            p.astype(int),
            (
                p + 0.18 * v
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

                elif event.key == pygame.K_5:

                    change_mode(4)

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
                        > OBSTACLE_RADIUS + 110
                    ):

                        env.target.center = click

        mode = MODES[mode_index]

        options = mode_options(mode)

        if not paused:

            for _ in range(speed):

                previous = env.state()

                (
                    action,
                    requested,
                    planned_path,
                    cost,
                    last_violation,
                    last_feasible,
                ) = choose_mpc_action(
                    previous,
                    env.target,
                    dynamics,
                    **options,
                )

                requested_action = requested.copy()

                executed_action = action.copy()

                env.step(action)

                current = env.state()

                if isinstance(
                    dynamics,
                    AdaptiveDynamics,
                ):

                    dynamics.observe(
                        previous,
                        action,
                        current,
                    )

                env.target.advance()

                trail.append(
                    env.position.copy()
                )

                if len(trail) > 150:

                    trail.pop(0)

        # ----------------------------------------------------
        # DRAW
        # ----------------------------------------------------

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

        margin = (
            adaptive_safety_margin(dynamics)
            if options["uncertainty_aware"]
            else BASE_SAFETY_MARGIN
        )

        safe_radius = (
            OBSTACLE_RADIUS
            + DRONE_RADIUS
            + margin
        )

        pygame.draw.circle(
            screen,
            (95, 95, 105),
            OBSTACLE.astype(int),
            OBSTACLE_RADIUS,
        )

        pygame.draw.circle(
            screen,
            (210, 110, 90),
            OBSTACLE.astype(int),
            int(safe_radius),
            2,
        )

        draw_target()

        if len(trail) > 1:

            pygame.draw.lines(
                screen,
                (70, 135, 235),
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
                (255, 190, 60),
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
            f"Target distance: {distance:.1f}",
            15,
            38,
        )

        draw_text(
            f"Obstacle clearance: {clearance:.1f}",
            15,
            64,
        )

        draw_text(
            f"Adaptive safety margin: {margin:.1f}",
            15,
            90,
            (255, 190, 90),
        )

        draw_text(
            f"Requested action: "
            f"[{requested_action[0]:+.2f}, "
            f"{requested_action[1]:+.2f}]",
            15,
            116,
            (245, 160, 160),
        )

        draw_text(
            f"Executed action: "
            f"[{executed_action[0]:+.2f}, "
            f"{executed_action[1]:+.2f}]",
            15,
            142,
            (120, 220, 160),
        )

        draw_text(
            f"Maximum CBF violation: "
            f"{last_violation:.5f}",
            15,
            168,
        )

        draw_text(
            f"CBF feasible: {last_feasible}",
            15,
            194,
            (
                (100, 230, 140)
                if last_feasible
                else (255, 100, 100)
            ),
        )

        if isinstance(
            dynamics,
            AdaptiveDynamics,
        ):

            draw_text(
                f"Estimated gain: {dynamics.gain:.4f}",
                15,
                220,
            )

            draw_text(
                f"Estimated drag: {dynamics.drag:.4f}",
                15,
                246,
            )

            draw_text(
                f"Model uncertainty: "
                f"{dynamics.uncertainty():.3f}",
                15,
                272,
                (255, 190, 90),
            )

            draw_text(
                f"RLS observations: "
                f"{dynamics.identifier.count}",
                15,
                298,
            )

        draw_text(
            f"Simulation speed: {speed}x",
            15,
            324,
        )

        draw_text(
            "1 Fixed | 2 Adaptive | 3 Uncertainty-aware",
            15,
            WORLD_H - 100,
        )

        draw_text(
            "4 Adaptive + CBF | 5 Oracle + CBF",
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
# 15. UNIT TESTS
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

    # Test 1 — RLS dimensions.
    identifier = RLSIdentifier()

    check(
        identifier.theta.shape == (2,),
        "RLS estimates motor gain and drag",
    )

    # Test 2 — Covariance shape.
    check(
        identifier.P.shape == (2, 2),
        "RLS covariance has correct dimensions",
    )

    # Test 3 — Positive initial covariance.
    check(
        np.all(
            np.linalg.eigvalsh(identifier.P) > 0
        ),
        "Initial covariance is positive definite",
    )

    # Test 4 — Residual estimator starts finite.
    residual = ResidualEstimator()

    check(
        np.all(np.isfinite(residual.mean))
        and np.all(np.isfinite(residual.variance)),
        "Residual estimator starts with finite statistics",
    )

    # Test 5 — Residual update.
    residual.update(
        np.array([3.0, -2.0])
    )

    check(
        residual.count == 1,
        "Residual estimator accepts observations",
    )

    # Test 6 — Residual variance positive.
    check(
        np.all(residual.variance > 0),
        "Residual variance remains positive",
    )

    # Test 7 — Environment state.
    env = DroneEnv(DOMAINS[0])

    check(
        env.state().shape == (6,),
        "Drone state includes position, velocity and wind",
    )

    # Test 8 — Environment transition.
    previous = env.state()

    env.step_drone(
        np.array([0.5, -0.4])
    )

    current = env.state()

    check(
        not np.allclose(previous, current),
        "Drone moves after a control action",
    )

    # Test 9 — Speed limit.
    check(
        np.linalg.norm(env.velocity)
        <= MAX_SPEED + 1e-8,
        "Drone velocity respects speed limit",
    )

    # Test 10 — Adaptive dynamics prediction.
    adaptive = AdaptiveDynamics()

    prediction = adaptive.predict_next(
        env.state(),
        np.array([0.2, 0.1]),
    )

    check(
        prediction.shape == (6,)
        and np.all(np.isfinite(prediction)),
        "Adaptive model predicts a finite next state",
    )

    # Test 11 — RLS observation count.
    previous = env.state()

    action = np.array([0.6, -0.3])

    env.step_drone(action)

    current = env.state()

    adaptive.observe(
        previous,
        action,
        current,
    )

    check(
        adaptive.identifier.count == 2,
        "RLS updates both physical axes",
    )

    # Test 12 — Parameter bounds.
    check(
        np.all(
            adaptive.identifier.theta >= PARAM_MIN
        )
        and np.all(
            adaptive.identifier.theta <= PARAM_MAX
        ),
        "Estimated physical parameters remain bounded",
    )

    # Test 13 — Uncertainty finite.
    check(
        np.isfinite(
            adaptive.uncertainty()
        ),
        "Adaptive model uncertainty is finite",
    )

    # Test 14 — Safety margin bounded.
    margin = adaptive_safety_margin(
        adaptive
    )

    check(
        BASE_SAFETY_MARGIN
        <= margin
        <= BASE_SAFETY_MARGIN + MAX_ADAPTIVE_MARGIN,
        "Adaptive safety margin remains bounded",
    )

    # Test 15 — CBF constraint dimensions.
    A, b = cbf_constraints(
        env.state(),
        adaptive,
        margin,
    )

    check(
        A.shape == (5, 2)
        and b.shape == (5,),
        "CBF produces obstacle and four boundary constraints",
    )

    # Test 16 — Projection respects action bounds.
    filtered, violation, feasible = (
        project_action_to_safe_set(
            np.array([1.5, -1.5]),
            A,
            b,
        )
    )

    check(
        np.all(np.abs(filtered) <= 1.0),
        "Safety-filter output respects action bounds",
    )

    # Test 17 — Projection diagnostics.
    check(
        violation >= 0.0
        and isinstance(feasible, bool),
        "Projection reports constraint violation and feasibility",
    )

    # Test 18 — Target movement.
    target = MovingTarget(
        [600.0, 300.0]
    )

    before = target.position().copy()

    target.advance()

    check(
        not np.allclose(
            before,
            target.position(),
        ),
        "Target follows a moving trajectory",
    )

    # Test 19 — MPC nominal sequence.
    sequence = make_nominal_sequence(
        env.state(),
        target,
        adaptive,
        True,
    )

    check(
        sequence.shape == (
            MPC_HORIZON,
            2,
        ),
        "MPC creates a valid nominal action sequence",
    )

    # Test 20 — MPC evaluation.
    cost, path = evaluate_sequence(
        env.state(),
        sequence,
        target,
        adaptive,
        True,
        True,
    )

    check(
        np.isfinite(cost)
        and path.shape == (
            MPC_HORIZON,
            2,
        ),
        "MPC evaluates predicted trajectories",
    )

    # Test 21 — Fixed model.
    fixed = FixedDynamics()

    check(
        fixed.predict_next(
            env.state(),
            np.zeros(2),
        ).shape == (6,),
        "Fixed dynamics model is operational",
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
        "Oracle dynamics model is operational",
    )

    # Test 23 — Zero uncertainty for fixed model.
    check(
        fixed.uncertainty() == 0.0,
        "Fixed model reports zero modeled uncertainty",
    )

    # Test 24 — Reset.
    adaptive.reset()

    check(
        adaptive.identifier.count == 0
        and adaptive.residual.count == 0,
        "Adaptive model reset clears learning statistics",
    )

    print(
        f"\n{passed}/24 tests passed."
    )

    return passed == 24


# ============================================================
# 16. MAIN
# ============================================================

def main():

    print("=" * 76)

    print(
        "PHASE 4.22 — SAFE ONLINE ADAPTATION"
    )

    print(
        "UNCERTAINTY-AWARE MPC + CONTROL BARRIER FUNCTIONS"
    )

    print("=" * 76)

    print("\n[1/4] Unit tests")

    if not run_tests():

        raise RuntimeError(
            "Unit tests failed."
        )

    print(
        "\n[2/4] Online adaptation visualization"
    )

    visualize_adaptation(
        DOMAINS[2]
    )

    print(
        "\n[3/4] Controller benchmark"
    )

    results = benchmark()

    visualize_benchmark(
        results
    )

    print(
        "\n[4/4] Interactive quadcopter"
    )

    run_pygame()


if __name__ == "__main__":

    main()
