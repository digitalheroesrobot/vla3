
"""
File: phase4/risk_sensitive_mpc_drone.py

PHASE 4.23 — RISK-SENSITIVE MODEL PREDICTIVE CONTROL

PURPOSE
-------
Train an ensemble of learned drone dynamics models and use them
to select actions for tracking a continuously moving target.

Compare:
    1. Mean MPC
    2. CVaR MPC
    3. Chance-Constrained MPC

WHAT YOU LEARN
--------------

STEP 1 — Dynamics Dataset
    Collect drone transitions under varied control inputs.

STEP 2 — Ensemble Dynamics Learning
    Train multiple bootstrap models to predict the next drone state.

STEP 3 — Prediction Uncertainty
    Compare the ensemble's predicted trajectories.

STEP 4 — Mean MPC
    Minimize average predicted trajectory cost.

STEP 5 — CVaR MPC
    Minimize average cost among the worst alpha fraction of
    predicted outcomes.

STEP 6 — Chance Constraints
    Reject candidate actions when the estimated probability
    of collision exceeds a specified threshold.

STEP 7 — Benchmark
    Compare average tracking error, collision rate and
    risk-sensitive cost.

STEP 8 — Interactive Drone
    Switch between the three planners and move the target
    trajectory center with the mouse.

MATHEMATICAL FOUNDATIONS
------------------------

1. ENSEMBLE DYNAMICS

    f_1(s,a), f_2(s,a), ..., f_M(s,a)

    Each model is trained on a different bootstrap sample.

    Mean prediction:

        mu = (1/M) * sum(f_i(s,a))

    Model disagreement:

        sigma^2 = (1/M) * sum((f_i - mu)^2)

2. MEAN MPC

    J_mean(a) = mean(J_1(a), ..., J_M(a))

3. CONDITIONAL VALUE AT RISK

    Sort predicted costs from worst to best.

    CVaR_alpha:

        CVaR_alpha(J) =
            average of the worst alpha fraction of costs

    Here alpha = 0.25 means that the planner considers
    the worst 25% of ensemble predictions.

4. CHANCE CONSTRAINT

    P(collision | action sequence) <= epsilon

    Estimate:

        p_collision =
            number of ensemble members predicting collision
            / total ensemble members

    A candidate is feasible when:

        p_collision <= epsilon

5. RISK-SENSITIVE OBJECTIVE

    J = mean_cost + lambda * CVaR_alpha

    A large lambda makes the planner more conservative.

IMPORTANT LIMITATIONS
--------------------
- Bootstrap ensemble disagreement is an uncertainty proxy,
  not a calibrated probability distribution.
- A small ensemble can estimate collision probabilities
  only coarsely.
- Chance constraints are evaluated on sampled model predictions.
- The learned dynamics are simplified linear models with
  nonlinear engineered features.
- This is a simulation exercise, not a flight-certified controller.

ROBOTICS APPLICATION
--------------------
Useful for mobile robots and drones operating with uncertain
dynamics, moving obstacles, uncertain payloads, or noisy models.

INSTALL
-------
    pip install numpy matplotlib pygame

RUN
---
    python phase4/risk_sensitive_mpc_drone.py

CONTROLS
--------
    1 : Mean MPC
    2 : CVaR MPC
    3 : Chance-Constrained MPC

    SPACE : Pause
    R     : Reset
    + / - : Simulation speed
    Click : Change moving target trajectory center
    ESC   : Exit
"""

from __future__ import annotations

import math
import random

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 0. CONFIGURATION
# ============================================================

SEED = 423

np.random.seed(SEED)
random.seed(SEED)

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

TARGET_A = 165.0
TARGET_B = 105.0
TARGET_OMEGA = 0.55

ENSEMBLE_SIZE = 7

DATASET_SIZE = 12000

MPC_HORIZON = 10
MPC_CANDIDATES = 48

CVaR_ALPHA = 0.25
CVAR_WEIGHT = 0.8

MAX_COLLISION_PROBABILITY = 0.10

CHANCE_PENALTY = 100000.0

ACTION_NOISE = 0.42

RIDGE = 1e-4

MODES = [
    "Mean MPC",
    "CVaR MPC",
    "Chance-Constrained MPC",
]


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
# 2. TRUE DRONE ENVIRONMENT
# ============================================================

class DroneEnv:

    """
    State:

        [x, y, vx, vy]

    Action:

        [ax, ay] in [-1, 1]

    True dynamics:

        a = gain*MAX_ACCEL*u
            - drag*v
            - qdrag*abs(v)*v
            + wind
            + bias

    The learned models do not know the true parameters.
    """

    def __init__(self, domain=None):

        if domain is None:

            domain = {
                "gain": 0.85,
                "drag": 0.085,
                "qdrag": 0.00022,
                "bias": np.array([1.5, -1.0]),
            }

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

        self.velocity = np.zeros(2)

        self.wind = np.random.normal(
            0.0,
            5.0,
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
            ]
        )

    def step(self, action):

        action = np.clip(
            np.asarray(action, dtype=np.float64),
            -1.0,
            1.0,
        )

        d = self.domain

        acceleration = (
            d["gain"] * MAX_ACCEL * action
            - d["drag"] * self.velocity
            - d["qdrag"]
            * np.abs(self.velocity)
            * self.velocity
            + self.wind
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
            0.88 * self.wind
            + np.random.normal(
                0.0,
                9.0,
                size=2,
            )
        )

        self.wind = np.clip(
            self.wind,
            -35.0,
            35.0,
        )

        self.target.advance()

        return self.state()


# ============================================================
# 3. DYNAMICS FEATURE ENGINEERING
# ============================================================

def dynamics_features(state, action):

    """
    Feature vector:

        [1,
         x, y,
         vx, vy,
         ax, ay,
         |vx|vx,
         |vy|vy,
         vx*vy,
         ax*vx,
         ay*vy]

    The model predicts:

        delta_state = next_state - state

    The position and velocity are included because the
    model learns a one-step state transition.
    """

    x, y, vx, vy = state
    ax, ay = action

    return np.array(
        [
            1.0,
            x / WORLD_W,
            y / WORLD_H,
            vx / MAX_SPEED,
            vy / MAX_SPEED,
            ax,
            ay,
            abs(vx) * vx / MAX_SPEED**2,
            abs(vy) * vy / MAX_SPEED**2,
            vx * vy / MAX_SPEED**2,
            ax * vx / MAX_SPEED,
            ay * vy / MAX_SPEED,
        ],
        dtype=np.float64,
    )


# ============================================================
# 4. COLLECT TRAINING DATA
# ============================================================

def collect_dataset(
    size=DATASET_SIZE,
):

    env = DroneEnv()

    X = []
    Y = []

    state = env.reset()

    for i in range(size):

        # Broad excitation is important for identifying dynamics.
        # The small PD component prevents the dataset from
        # containing only completely random transitions.

        if np.random.rand() < 0.65:

            action = np.random.uniform(
                -1.0,
                1.0,
                size=2,
            )

        else:

            target_position = (
                env.target.position()
            )

            target_velocity = (
                env.target.velocity()
            )

            error = (
                target_position
                - state[:2]
            )

            desired_velocity = (
                target_velocity
                + 0.45 * error
            )

            desired_velocity = np.clip(
                desired_velocity,
                -MAX_SPEED,
                MAX_SPEED,
            )

            desired_acceleration = (
                1.5 * (
                    desired_velocity
                    - state[2:4]
                )
                - env.wind
            )

            action = np.clip(
                desired_acceleration
                / MAX_ACCEL,
                -1.0,
                1.0,
            )

            action += np.random.normal(
                0.0,
                0.25,
                size=2,
            )

            action = np.clip(
                action,
                -1.0,
                1.0,
            )

        old_state = state.copy()

        X.append(
            dynamics_features(
                old_state,
                action,
            )
        )

        state = env.step(action)

        Y.append(
            state - old_state
        )

        if (
            np.linalg.norm(
                state[:2] - OBSTACLE
            )
            < OBSTACLE_RADIUS + 30
        ):

            env.position = np.array(
                [110.0, 500.0]
            )

            env.velocity[:] = 0.0

            state = env.state()

    return (
        np.asarray(X),
        np.asarray(Y),
    )


# ============================================================
# 5. BOOTSTRAP ENSEMBLE MODEL
# ============================================================

class LinearDynamicsModel:

    """
    Multi-output ridge regression.

    theta = (X.T X + lambda I)^-1 X.T Y

    Prediction:

        delta_state = feature(state, action) @ theta

    This is a compact model for teaching ensemble uncertainty.
    """

    def __init__(self):

        self.theta = None

    def fit(self, X, Y):

        identity = np.eye(
            X.shape[1]
        )

        identity[0, 0] = 0.0

        self.theta = np.linalg.solve(
            X.T @ X + RIDGE * identity,
            X.T @ Y,
        )

    def predict_delta(self, state, action):

        feature = dynamics_features(
            state,
            action,
        )

        return (
            feature @ self.theta
        )


class DynamicsEnsemble:

    def __init__(self, size=ENSEMBLE_SIZE):

        self.models = [
            LinearDynamicsModel()
            for _ in range(size)
        ]

    def fit(self, X, Y):

        n = len(X)

        for model in self.models:

            indices = np.random.randint(
                0,
                n,
                size=n,
            )

            model.fit(
                X[indices],
                Y[indices],
            )

    def predict_members(
        self,
        state,
        action,
    ):

        predictions = []

        for model in self.models:

            delta = model.predict_delta(
                state,
                action,
            )

            next_state = (
                state + delta
            )

            predictions.append(
                constrain_state(next_state)
            )

        return np.asarray(
            predictions
        )

    def predict_mean(
        self,
        state,
        action,
    ):

        return np.mean(
            self.predict_members(
                state,
                action,
            ),
            axis=0,
        )

    def uncertainty(
        self,
        state,
        action,
    ):

        predictions = self.predict_members(
            state,
            action,
        )

        return float(
            np.mean(
                np.std(
                    predictions,
                    axis=0,
                )
            )
        )


def constrain_state(state):

    result = np.asarray(
        state,
        dtype=np.float64,
    ).copy()

    speed = np.linalg.norm(
        result[2:4]
    )

    if speed > MAX_SPEED:

        result[2:4] *= (
            MAX_SPEED / (speed + 1e-8)
        )

    result[:2] = np.clip(
        result[:2],
        DRONE_RADIUS,
        [
            WORLD_W - DRONE_RADIUS,
            WORLD_H - DRONE_RADIUS,
        ],
    )

    return result


# ============================================================
# 6. NOMINAL TRACKING CONTROLLER
# ============================================================

def nominal_action(
    state,
    target_position,
    target_velocity,
):

    error = (
        target_position - state[:2]
    )

    desired_velocity = (
        target_velocity + 0.5 * error
    )

    desired_velocity = np.clip(
        desired_velocity,
        -MAX_SPEED,
        MAX_SPEED,
    )

    desired_acceleration = (
        1.7 * (
            desired_velocity - state[2:4]
        )
    )

    return np.clip(
        desired_acceleration / MAX_ACCEL,
        -1.0,
        1.0,
    )


def make_nominal_sequence(
    state,
    target,
    ensemble,
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

        action = nominal_action(
            predicted_state,
            target_position,
            target_velocity,
        )

        sequence.append(
            action
        )

        predicted_state = (
            ensemble.predict_mean(
                predicted_state,
                action,
            )
        )

    return np.asarray(
        sequence
    )


# ============================================================
# 7. ENSEMBLE TRAJECTORY EVALUATION
# ============================================================

def rollout_one_model(
    initial_state,
    sequence,
    target,
    model,
):

    state = initial_state.copy()

    total_cost = 0.0

    collision = False

    path = []

    for step, action in enumerate(
        sequence,
        start=1,
    ):

        state = constrain_state(
            state + model.predict_delta(
                state,
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
            state[:2] - target_position
        )

        obstacle_distance = np.linalg.norm(
            state[:2] - OBSTACLE
        )

        clearance = (
            obstacle_distance
            - OBSTACLE_RADIUS
            - DRONE_RADIUS
        )

        if clearance < 0:

            collision = True

        obstacle_penalty = (
            max(
                0.0,
                35.0 - clearance,
            ) ** 2
        )

        boundary_distance = min(
            state[0],
            WORLD_W - state[0],
            state[1],
            WORLD_H - state[1],
        )

        boundary_penalty = (
            max(
                0.0,
                35.0 - boundary_distance,
            ) ** 2
        )

        control_cost = np.sum(
            action**2
        )

        total_cost += (
            0.05 * distance
            + 0.045 * obstacle_penalty
            + 0.05 * boundary_penalty
            + 0.02 * control_cost
        )

        path.append(
            state[:2].copy()
        )

    return (
        float(total_cost),
        collision,
        np.asarray(path),
    )


def evaluate_ensemble(
    initial_state,
    sequence,
    target,
    ensemble,
):

    costs = []
    collisions = []
    paths = []

    for model in ensemble.models:

        cost, collision, path = (
            rollout_one_model(
                initial_state,
                sequence,
                target,
                model,
            )
        )

        costs.append(cost)
        collisions.append(collision)
        paths.append(path)

    return (
        np.asarray(costs),
        np.asarray(collisions),
        np.asarray(paths),
    )


# ============================================================
# 8. RISK METRICS
# ============================================================

def calculate_cvar(
    costs,
    alpha=CVaR_ALPHA,
):
    """
    Empirical upper-tail CVaR.

    For alpha = 0.25, average the worst 25% of costs.

    At least one sample is always included.
    """

    costs = np.asarray(
        costs,
        dtype=np.float64,
    )

    count = max(
        1,
        int(
            math.ceil(
                alpha * len(costs)
            )
        ),
    )

    worst = np.sort(
        costs
    )[::-1][:count]

    return float(
        np.mean(worst)
    )


def collision_probability(collisions):

    return float(
        np.mean(
            np.asarray(
                collisions,
                dtype=np.float64,
            )
        )
    )


def risk_metrics(costs, collisions):

    return {
        "mean": float(
            np.mean(costs)
        ),
        "cvar": calculate_cvar(
            costs
        ),
        "collision_probability": (
            collision_probability(
                collisions
            )
        ),
        "worst": float(
            np.max(costs)
        ),
    }


# ============================================================
# 9. RISK-SENSITIVE MPC
# ============================================================

def choose_action(
    state,
    target,
    ensemble,
    mode,
):

    nominal = make_nominal_sequence(
        state,
        target,
        ensemble,
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

    candidates[0] = nominal

    best_score = float("inf")

    best_sequence = nominal

    best_path = None

    best_metrics = None

    feasible_count = 0

    for sequence in candidates:

        costs, collisions, paths = (
            evaluate_ensemble(
                state,
                sequence,
                target,
                ensemble,
            )
        )

        metrics = risk_metrics(
            costs,
            collisions,
        )

        mean_cost = metrics["mean"]

        cvar = metrics["cvar"]

        probability = (
            metrics["collision_probability"]
        )

        if mode == "Mean MPC":

            score = mean_cost

        elif mode == "CVaR MPC":

            score = (
                mean_cost
                + CVAR_WEIGHT * cvar
            )

        elif mode == "Chance-Constrained MPC":

            if (
                probability
                > MAX_COLLISION_PROBABILITY
            ):

                score = (
                    mean_cost
                    + CHANCE_PENALTY
                    * (
                        probability
                        - MAX_COLLISION_PROBABILITY
                    )
                )

            else:

                score = mean_cost

                feasible_count += 1

        else:

            raise ValueError(
                f"Unknown mode: {mode}"
            )

        if score < best_score:

            best_score = score

            best_sequence = sequence

            best_path = np.mean(
                paths,
                axis=0,
            )

            best_metrics = metrics

    return {
        "action": best_sequence[0].copy(),
        "path": best_path,
        "score": float(best_score),
        "metrics": best_metrics,
        "feasible_candidates": feasible_count,
    }


# ============================================================
# 10. BENCHMARK
# ============================================================

def run_episode(
    mode,
    ensemble,
    steps=180,
):

    env = DroneEnv()

    state = env.reset()

    distances = []
    clearances = []
    probabilities = []
    cvars = []

    collisions = 0

    for _ in range(steps):

        result = choose_action(
            state,
            env.target,
            ensemble,
            mode,
        )

        action = result["action"]

        state = env.step(action)

        distance = np.linalg.norm(
            state[:2] - env.target.position()
        )

        clearance = (
            np.linalg.norm(
                state[:2] - OBSTACLE
            )
            - OBSTACLE_RADIUS
            - DRONE_RADIUS
        )

        if clearance < 0:

            collisions += 1

        distances.append(distance)

        clearances.append(clearance)

        probabilities.append(
            result["metrics"][
                "collision_probability"
            ]
        )

        cvars.append(
            result["metrics"]["cvar"]
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
        "mean_predicted_collision_probability": float(
            np.mean(probabilities)
        ),
        "mean_cvar": float(
            np.mean(cvars)
        ),
    }


def benchmark(ensemble):

    results = {}

    print("\nRISK-SENSITIVE MPC BENCHMARK")

    for mode in MODES:

        episodes = []

        for _ in range(3):

            episodes.append(
                run_episode(
                    mode,
                    ensemble,
                    steps=150,
                )
            )

        results[mode] = {
            key: float(
                np.mean(
                    [
                        episode[key]
                        for episode in episodes
                    ]
                )
            )
            for key in episodes[0]
        }

        print(
            f"\n{mode}"
        )

        for key, value in results[mode].items():

            print(
                f"  {key:<43}: {value:.4f}"
            )

    return results


# ============================================================
# 11. VISUALIZATION
# ============================================================

def visualize_benchmark(results):

    metrics = [
        (
            "mean_distance",
            "Mean target tracking distance",
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
            "mean_cvar",
            "Mean predicted CVaR",
        ),
    ]

    for metric, title in metrics:

        plt.figure(figsize=(9, 5))

        values = [
            results[mode][metric]
            for mode in MODES
        ]

        plt.bar(
            MODES,
            values,
        )

        plt.title(title)
        plt.ylabel(metric)
        plt.xticks(rotation=12)
        plt.grid(axis="y")
        plt.tight_layout()
        plt.show()


def visualize_risk_example(
    state,
    target,
    ensemble,
):

    nominal = make_nominal_sequence(
        state,
        target,
        ensemble,
    )

    costs, collisions, paths = (
        evaluate_ensemble(
            state,
            nominal,
            target,
            ensemble,
        )
    )

    plt.figure(figsize=(9, 6))

    for i, path in enumerate(paths):

        label = (
            f"Model {i+1}"
        )

        plt.plot(
            path[:, 0],
            path[:, 1],
            alpha=0.65,
            label=label,
        )

    plt.scatter(
        OBSTACLE[0],
        OBSTACLE[1],
        s=350,
        marker="o",
        label="Obstacle",
    )

    plt.scatter(
        state[0],
        state[1],
        marker="s",
        s=80,
        label="Drone",
    )

    plt.scatter(
        target.position()[0],
        target.position()[1],
        marker="*",
        s=180,
        label="Moving target",
    )

    plt.title(
        "Ensemble Predicted Drone Trajectories"
    )

    plt.xlabel("X")
    plt.ylabel("Y")

    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    print(
        "\nEnsemble risk metrics:"
    )

    for key, value in risk_metrics(
        costs,
        collisions,
    ).items():

        print(
            f"{key}: {value:.4f}"
        )


# ============================================================
# 12. INTERACTIVE PYGAME DRONE
# ============================================================

def run_pygame(ensemble):

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
        "Phase 4.23 — Risk-Sensitive Drone MPC"
    )

    clock = pygame.time.Clock()

    font = pygame.font.SysFont(
        "consolas",
        15,
    )

    env = DroneEnv()

    state = env.reset()

    mode_index = 1

    mode = MODES[mode_index]

    paused = False

    speed = 1

    trail = []

    predicted_path = np.empty(
        (0, 2)
    )

    last_metrics = {
        "mean": 0.0,
        "cvar": 0.0,
        "collision_probability": 0.0,
        "worst": 0.0,
    }

    feasible_candidates = 0

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
            (65, 105, 75),
            True,
            points,
            2,
        )

        p = env.target.position()

        v = env.target.velocity()

        pygame.draw.circle(
            screen,
            (70, 240, 110),
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

                    env = DroneEnv()

                    state = env.reset()

                    trail.clear()

                    predicted_path = np.empty(
                        (0, 2)
                    )

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

        if not paused:

            for _ in range(speed):

                result = choose_action(
                    state,
                    env.target,
                    ensemble,
                    mode,
                )

                action = result["action"]

                predicted_path = result["path"]

                last_metrics = result["metrics"]

                feasible_candidates = (
                    result["feasible_candidates"]
                )

                state = env.step(action)

                trail.append(
                    state[:2].copy()
                )

                if len(trail) > 180:

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

        pygame.draw.circle(
            screen,
            (100, 100, 110),
            OBSTACLE.astype(int),
            OBSTACLE_RADIUS,
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

        if len(predicted_path) > 1:

            pygame.draw.lines(
                screen,
                (255, 190, 60),
                False,
                [
                    p.astype(int)
                    for p in predicted_path
                ],
                3,
            )

        draw_drone(
            state[:2],
            state[2:4],
        )

        distance = np.linalg.norm(
            state[:2] - env.target.position()
        )

        clearance = (
            np.linalg.norm(
                state[:2] - OBSTACLE
            )
            - OBSTACLE_RADIUS
            - DRONE_RADIUS
        )

        draw_text(
            f"Mode: {mode}",
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
            f"Mean predicted cost: "
            f"{last_metrics['mean']:.2f}",
            15,
            90,
        )

        draw_text(
            f"CVaR ({CVaR_ALPHA:.0%}): "
            f"{last_metrics['cvar']:.2f}",
            15,
            116,
            (255, 185, 85),
        )

        draw_text(
            f"Predicted collision probability: "
            f"{last_metrics['collision_probability']:.2f}",
            15,
            142,
            (255, 130, 130),
        )

        draw_text(
            f"Worst predicted cost: "
            f"{last_metrics['worst']:.2f}",
            15,
            168,
        )

        draw_text(
            f"Chance-feasible candidates: "
            f"{feasible_candidates}/{MPC_CANDIDATES}",
            15,
            194,
        )

        draw_text(
            f"Ensemble size: {ENSEMBLE_SIZE}",
            15,
            220,
        )

        draw_text(
            f"CVaR alpha: {CVaR_ALPHA}",
            15,
            246,
        )

        draw_text(
            f"Allowed collision probability: "
            f"{MAX_COLLISION_PROBABILITY:.2f}",
            15,
            272,
        )

        draw_text(
            f"Simulation speed: {speed}x",
            15,
            298,
        )

        draw_text(
            "1 Mean MPC | 2 CVaR MPC | 3 Chance-Constrained MPC",
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
# 13. UNIT TESTS
# ============================================================

def run_tests():

    print("\nRunning unit tests...")

    passed = 0

    def check(condition, description):

        nonlocal passed

        assert condition, description

        passed += 1

        print(
            f"[PASS] {description}"
        )

    # Test 1 — Feature dimensions.
    feature = dynamics_features(
        np.array([100.0, 200.0, 10.0, -5.0]),
        np.array([0.2, -0.3]),
    )

    check(
        feature.shape == (12,),
        "Dynamics feature vector has 12 elements",
    )

    # Test 2 — Feature values finite.
    check(
        np.all(np.isfinite(feature)),
        "Dynamics features are finite",
    )

    # Test 3 — Environment state.
    env = DroneEnv()

    state = env.reset()

    check(
        state.shape == (4,),
        "Drone state has four elements",
    )

    # Test 4 — Environment transition.
    next_state = env.step(
        np.array([0.5, -0.2])
    )

    check(
        next_state.shape == (4,)
        and not np.allclose(state, next_state),
        "Drone environment advances",
    )

    # Test 5 — Speed limit.
    check(
        np.linalg.norm(next_state[2:4])
        <= MAX_SPEED + 1e-8,
        "Drone speed is constrained",
    )

    # Test 6 — State bounds.
    check(
        np.all(next_state[:2] >= DRONE_RADIUS)
        and next_state[0] <= WORLD_W - DRONE_RADIUS
        and next_state[1] <= WORLD_H - DRONE_RADIUS,
        "Drone position remains within world boundaries",
    )

    # Test 7 — Dataset collection.
    X, Y = collect_dataset(500)

    check(
        X.shape == (500, 12)
        and Y.shape == (500, 4),
        "Dataset has correct input and output dimensions",
    )

    # Test 8 — Dataset finite.
    check(
        np.all(np.isfinite(X))
        and np.all(np.isfinite(Y)),
        "Training dataset contains finite values",
    )

    # Test 9 — Linear model training.
    model = LinearDynamicsModel()

    model.fit(X, Y)

    prediction = model.predict_delta(
        state,
        np.array([0.1, 0.2]),
    )

    check(
        prediction.shape == (4,)
        and np.all(np.isfinite(prediction)),
        "Linear dynamics model produces finite predictions",
    )

    # Test 10 — Ensemble size.
    ensemble = DynamicsEnsemble(
        size=3
    )

    ensemble.fit(X, Y)

    check(
        len(ensemble.models) == 3,
        "Ensemble contains three trained models",
    )

    # Test 11 — Ensemble prediction shape.
    members = ensemble.predict_members(
        state,
        np.array([0.1, 0.2]),
    )

    check(
        members.shape == (3, 4),
        "Ensemble prediction shape is correct",
    )

    # Test 12 — Ensemble mean.
    mean = ensemble.predict_mean(
        state,
        np.array([0.1, 0.2]),
    )

    check(
        np.allclose(
            mean,
            np.mean(members, axis=0),
        ),
        "Ensemble mean equals average member prediction",
    )

    # Test 13 — Uncertainty nonnegative.
    check(
        ensemble.uncertainty(
            state,
            np.array([0.1, 0.2]),
        ) >= 0.0,
        "Ensemble disagreement is nonnegative",
    )

    # Test 14 — CVaR example.
    example_costs = np.array(
        [1.0, 2.0, 3.0, 4.0]
    )

    check(
        np.isclose(
            calculate_cvar(
                example_costs,
                alpha=0.25,
            ),
            4.0,
        ),
        "CVaR selects the worst 25 percent",
    )

    # Test 15 — CVaR of multiple worst samples.
    check(
        np.isclose(
            calculate_cvar(
                np.array([1.0, 2.0, 3.0, 4.0]),
                alpha=0.50,
            ),
            3.5,
        ),
        "CVaR averages the worst 50 percent",
    )

    # Test 16 — Collision probability.
    check(
        np.isclose(
            collision_probability(
                [False, True, False, True]
            ),
            0.5,
        ),
        "Empirical collision probability is correct",
    )

    # Test 17 — Risk metrics.
    metrics = risk_metrics(
        np.array([1.0, 2.0, 3.0, 4.0]),
        np.array([False, False, False, True]),
    )

    check(
        np.isclose(metrics["mean"], 2.5)
        and np.isclose(metrics["cvar"], 4.0)
        and np.isclose(
            metrics["collision_probability"],
            0.25,
        ),
        "Combined risk metrics are correct",
    )

    # Test 18 — Target motion.
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
        "Target moves continuously",
    )

    # Test 19 — Nominal sequence.
    sequence = make_nominal_sequence(
        state,
        target,
        ensemble,
    )

    check(
        sequence.shape == (
            MPC_HORIZON,
            2,
        ),
        "Nominal MPC sequence has correct shape",
    )

    # Test 20 — Action bounds.
    check(
        np.all(np.abs(sequence) <= 1.0),
        "Nominal actions remain within bounds",
    )

    # Test 21 — Ensemble rollout.
    costs, collisions, paths = evaluate_ensemble(
        state,
        sequence,
        target,
        ensemble,
    )

    check(
        costs.shape == (3,)
        and collisions.shape == (3,)
        and paths.shape == (
            3,
            MPC_HORIZON,
            2,
        ),
        "Ensemble trajectory evaluation has correct dimensions",
    )

    # Test 22 — Mean MPC.
    result = choose_action(
        state,
        target,
        ensemble,
        "Mean MPC",
    )

    check(
        result["action"].shape == (2,)
        and np.all(np.abs(result["action"]) <= 1.0),
        "Mean MPC returns a valid bounded action",
    )

    # Test 23 — CVaR MPC.
    result = choose_action(
        state,
        target,
        ensemble,
        "CVaR MPC",
    )

    check(
        np.isfinite(result["score"])
        and np.isfinite(result["metrics"]["cvar"]),
        "CVaR MPC produces a finite risk-sensitive score",
    )

    # Test 24 — Chance-constrained MPC.
    result = choose_action(
        state,
        target,
        ensemble,
        "Chance-Constrained MPC",
    )

    check(
        0 <= result["metrics"]["collision_probability"] <= 1,
        "Chance-constrained MPC estimates valid collision probability",
    )

    # Test 25 — Constrained state.
    constrained = constrain_state(
        np.array(
            [
                -100.0,
                1000.0,
                500.0,
                500.0,
            ]
        )
    )

    check(
        constrained[0] >= DRONE_RADIUS
        and constrained[1] <= WORLD_H - DRONE_RADIUS
        and np.linalg.norm(constrained[2:4]) <= MAX_SPEED + 1e-8,
        "Predicted states respect position and speed limits",
    )

    print(
        f"\n{passed}/25 tests passed."
    )

    return passed == 25


# ============================================================
# 14. MAIN
# ============================================================

def main():

    print("=" * 72)

    print(
        "PHASE 4.23 — RISK-SENSITIVE MPC"
    )

    print(
        "ENSEMBLE DYNAMICS + CVaR + CHANCE CONSTRAINTS"
    )

    print("=" * 72)

    print("\n[1/5] Running unit tests")

    if not run_tests():

        raise RuntimeError(
            "Unit tests failed."
        )

    print(
        "\n[2/5] Collecting dynamics dataset"
    )

    X, Y = collect_dataset()

    print(
        f"Dataset size: {len(X)}"
    )

    print(
        f"Feature dimensions: {X.shape[1]}"
    )

    print(
        f"Prediction dimensions: {Y.shape[1]}"
    )

    print(
        "\n[3/5] Training ensemble dynamics models"
    )

    ensemble = DynamicsEnsemble()

    ensemble.fit(
        X,
        Y,
    )

    print(
        f"Trained {len(ensemble.models)} dynamics models."
    )

    print(
        "\n[4/5] Visualizing ensemble prediction uncertainty"
    )

    env = DroneEnv()

    state = env.reset()

    visualize_risk_example(
        state,
        env.target,
        ensemble,
    )

    print(
        "\n[5/5] Benchmarking risk-sensitive planners"
    )

    results = benchmark(
        ensemble
    )

    visualize_benchmark(
        results
    )

    print(
        "\nStarting interactive quadcopter visualization..."
    )

    run_pygame(
        ensemble
    )


if __name__ == "__main__":

    main()
