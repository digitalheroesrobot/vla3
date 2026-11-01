
"""
File: phase4/sim_to_real_adaptation_drone.py

PHASE 4.25 — SIM-TO-REAL TRANSFER AND DOMAIN ADAPTATION

PURPOSE
-------
Transfer a simulation-trained drone policy to a new physical
environment using a limited amount of target-environment data.

Compare:
    1. Zero-shot policy transfer
    2. Dynamics calibration + transferred policy
    3. Dynamics calibration + policy fine-tuning

WHAT YOU LEARN
--------------

STEP 1 — Source Simulation
    Train a Gaussian policy using randomized source domains.

STEP 2 — Target Environment
    Introduce different, hidden physical parameters.

STEP 3 — System Identification
    Estimate motor effectiveness, drag and wind from transitions.

STEP 4 — Zero-Shot Transfer
    Evaluate the source policy without changing its parameters.

STEP 5 — Policy Fine-Tuning
    Fine-tune a copy of the source policy using limited target data.

STEP 6 — Generalization Evaluation
    Compare source, calibrated and fine-tuned controllers.

MATHEMATICAL FOUNDATIONS
------------------------

Unknown drone dynamics:

    a = k*u - c*v - q*abs(v)*v + w

where:
    k = effective motor acceleration gain
    c = linear drag coefficient
    q = quadratic drag coefficient
    w = constant wind acceleration

For each axis:

    observed_acceleration =
        k * action
        - c * velocity
        - q * abs(velocity) * velocity
        + wind

This is linear in the unknown parameters:

    y = X * beta

Least-squares system identification:

    beta_hat = argmin_beta ||X*beta - y||^2

Policy fine-tuning:

    grad J(theta) =
        E[ G_t * grad log pi_theta(a_t | s_t) ]

Transfer comparison:

    Zero-shot:
        theta_target = theta_source

    Fine-tuned:
        theta_target = theta_source + policy_updates

IMPORTANT EXPERIMENTAL DISTINCTION
----------------------------------
The calibrated dynamics estimate is evaluated separately.
The policy does not automatically become a model-based controller
just because the physical parameters were identified.

This lesson uses REINFORCE for target-environment fine-tuning.

The target simulator is a controlled stand-in for a real drone.
No actual hardware connection is made.

ROBOTICS APPLICATION
--------------------
A drone trained in simulation may encounter a different payload,
motor response, aerodynamic drag or wind condition in deployment.

A practical transfer pipeline can:
    1. Collect safe calibration transitions.
    2. Estimate the new physical dynamics.
    3. Evaluate the transferred policy.
    4. Fine-tune conservatively with limited target data.
    5. Validate safety before deployment.

SAFETY LIMITATION
-----------------
This is an educational 2D model.
It is not a flight controller or a flight-certified safety system.
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

SEED = 425

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

STATE_DIM = 8
ACTION_DIM = 2

HIDDEN_DIM = 128

GAMMA = 0.99

SOURCE_LEARNING_RATE = 2e-4
ADAPT_LEARNING_RATE = 5e-5

SOURCE_EPISODES = 120
ADAPTATION_EPISODES = 25

EPISODE_STEPS = 220
EVAL_EPISODES = 5
EVAL_STEPS = 220

CALIBRATION_STEPS = 500

OBSERVATION_NOISE = 0.012

# Source-domain randomization ranges.
SOURCE_RANGES = {
    "motor_gain": (0.65, 1.05),
    "linear_drag": (0.045, 0.13),
    "quadratic_drag": (0.00010, 0.00035),
    "wind_x": (-8.0, 8.0),
    "wind_y": (-8.0, 8.0),
}

# Hidden target-environment parameters.
# These values are only used to create the target simulator.
# The identification algorithm does not receive them.
TARGET_TRUE = {
    "motor_gain": 0.56,
    "linear_drag": 0.155,
    "quadratic_drag": 0.00042,
    "wind_x": 9.0,
    "wind_y": -6.0,
}


# ============================================================
# 1. PHYSICAL DOMAIN
# ============================================================

@dataclass
class Domain:

    motor_gain: float
    linear_drag: float
    quadratic_drag: float
    wind_x: float
    wind_y: float

    def wind(self):

        return np.array(
            [
                self.wind_x,
                self.wind_y,
            ],
            dtype=np.float64,
        )


def sample_source_domain():

    return Domain(
        motor_gain=np.random.uniform(
            *SOURCE_RANGES["motor_gain"]
        ),

        linear_drag=np.random.uniform(
            *SOURCE_RANGES["linear_drag"]
        ),

        quadratic_drag=np.random.uniform(
            *SOURCE_RANGES["quadratic_drag"]
        ),

        wind_x=np.random.uniform(
            *SOURCE_RANGES["wind_x"]
        ),

        wind_y=np.random.uniform(
            *SOURCE_RANGES["wind_y"]
        ),
    )


def target_domain():

    return Domain(
        **TARGET_TRUE
    )


def nominal_domain():

    return Domain(
        motor_gain=0.85,
        linear_drag=0.085,
        quadratic_drag=0.00022,
        wind_x=0.0,
        wind_y=0.0,
    )


# ============================================================
# 2. MOVING TARGET
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
# 3. DRONE ENVIRONMENT
# ============================================================

class DroneEnv:

    """
    State:
        [x, y, vx, vy]

    Action:
        [ax, ay] in [-1, 1]

    Dynamics:
        acceleration =
            motor_gain * MAX_ACCEL * action
            - linear_drag * velocity
            - quadratic_drag * abs(velocity) * velocity
            + wind

    Wind is constant during an episode in this lesson.
    This makes the calibration regression identifiable.
    """

    def __init__(self, domain):

        self.domain = copy.deepcopy(
            domain
        )

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

        self.target.phase = np.random.uniform(
            0.0,
            2.0 * math.pi,
        )

        return self.observe()

    def observe(self):

        target_position = (
            self.target.position()
        )

        target_velocity = (
            self.target.velocity()
        )

        relative_target = (
            target_position - self.position
        )

        relative_obstacle = (
            OBSTACLE - self.position
        )

        observation = np.concatenate(
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
        )

        observation += np.random.normal(
            0.0,
            OBSERVATION_NOISE,
            size=STATE_DIM,
        )

        return observation.astype(
            np.float32
        )

    def step(self, action):

        action = np.clip(
            np.asarray(action, dtype=np.float64),
            -1.0,
            1.0,
        )

        d = self.domain

        acceleration = (
            d.motor_gain
            * MAX_ACCEL
            * action

            - d.linear_drag
            * self.velocity

            - d.quadratic_drag
            * np.abs(self.velocity)
            * self.velocity

            + d.wind()
        )

        old_velocity = (
            self.velocity.copy()
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

        self.target.advance()

        measured_acceleration = (
            self.velocity - old_velocity
        ) / DT

        return (
            self.observe(),
            measured_acceleration,
        )


# ============================================================
# 4. REWARD FUNCTION
# ============================================================

def calculate_reward(
    old_position,
    new_position,
    target_position,
    action,
):

    distance_before = np.linalg.norm(
        old_position - target_position
    )

    distance_after = np.linalg.norm(
        new_position - target_position
    )

    progress = (
        distance_before - distance_after
    )

    obstacle_distance = np.linalg.norm(
        new_position - OBSTACLE
    )

    clearance = (
        obstacle_distance - OBSTACLE_RADIUS
    )

    obstacle_penalty = (
        max(
            0.0,
            SAFE_RADIUS - clearance,
        ) / SAFE_RADIUS
    ) ** 2

    boundary_distance = min(
        new_position[0],
        WORLD_W - new_position[0],
        new_position[1],
        WORLD_H - new_position[1],
    )

    boundary_penalty = (
        max(
            0.0,
            55.0 - boundary_distance,
        ) / 55.0
    ) ** 2

    collision_penalty = (
        8.0
        if clearance < 0.0
        else 0.0
    )

    effort_penalty = (
        0.025 * np.sum(action**2)
    )

    return float(
        0.035 * progress
        - 0.012 * distance_after
        - 1.8 * obstacle_penalty
        - 1.5 * boundary_penalty
        - collision_penalty
        - effort_penalty
    )


# ============================================================
# 5. GAUSSIAN POLICY
# ============================================================

class GaussianPolicy(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                STATE_DIM,
                HIDDEN_DIM,
            ),

            nn.Tanh(),

            nn.Linear(
                HIDDEN_DIM,
                HIDDEN_DIM,
            ),

            nn.Tanh(),

            nn.Linear(
                HIDDEN_DIM,
                ACTION_DIM,
            ),
        )

        self.log_std = nn.Parameter(
            torch.full(
                (ACTION_DIM,),
                -0.55,
            )
        )

    def sample_action(
        self,
        state,
        deterministic=False,
    ):

        mean = self.network(
            state
        )

        std = torch.exp(
            torch.clamp(
                self.log_std,
                -2.0,
                0.5,
            )
        )

        distribution = torch.distributions.Normal(
            mean,
            std,
        )

        if deterministic:

            raw_action = mean

        else:

            raw_action = distribution.rsample()

        action = torch.tanh(
            raw_action
        )

        log_probability = (
            distribution.log_prob(
                raw_action
            )
            - torch.log(
                1.0 - action.pow(2) + 1e-6
            )
        ).sum(
            dim=-1
        )

        return (
            action,
            log_probability,
        )


# ============================================================
# 6. DISCOUNTED RETURNS
# ============================================================

def discounted_returns(
    rewards,
    gamma=GAMMA,
):

    result = np.zeros(
        len(rewards),
        dtype=np.float32,
    )

    running = 0.0

    for index in reversed(
        range(len(rewards))
    ):

        running = (
            rewards[index]
            + gamma * running
        )

        result[index] = running

    return result


# ============================================================
# 7. POLICY TRAINER
# ============================================================

class PolicyTrainer:

    def __init__(
        self,
        policy=None,
        learning_rate=SOURCE_LEARNING_RATE,
    ):

        self.policy = (
            GaussianPolicy()
            if policy is None
            else policy
        ).to(DEVICE)

        self.optimizer = optim.Adam(
            self.policy.parameters(),
            lr=learning_rate,
        )

        self.reward_history = []

    def run_episode(
        self,
        env,
        training=True,
        max_steps=EPISODE_STEPS,
    ):

        observation = env.reset()

        log_probabilities = []
        rewards = []
        distances = []

        collision_steps = 0

        for _ in range(max_steps):

            state_tensor = torch.as_tensor(
                observation,
                dtype=torch.float32,
                device=DEVICE,
            ).unsqueeze(0)

            with torch.set_grad_enabled(
                training
            ):

                action_tensor, log_probability = (
                    self.policy.sample_action(
                        state_tensor,
                        deterministic=not training,
                    )
                )

            action = (
                action_tensor
                .detach()
                .cpu()
                .numpy()[0]
            )

            if training:

                log_probabilities.append(
                    log_probability.squeeze()
                )

            old_position = (
                env.position.copy()
            )

            observation, _ = env.step(
                action
            )

            future_target = (
                env.target.position()
            )

            reward = calculate_reward(
                old_position,
                env.position,
                future_target,
                action,
            )

            rewards.append(
                reward
            )

            distances.append(
                np.linalg.norm(
                    env.position - future_target
                )
            )

            clearance = (
                np.linalg.norm(
                    env.position - OBSTACLE
                )
                - OBSTACLE_RADIUS
            )

            if clearance < 0:

                collision_steps += 1

        if training:

            returns = discounted_returns(
                rewards
            )

            returns_tensor = torch.as_tensor(
                returns,
                dtype=torch.float32,
                device=DEVICE,
            )

            returns_tensor = (
                returns_tensor
                - returns_tensor.mean()
            ) / (
                returns_tensor.std(
                    unbiased=False
                ) + 1e-6
            )

            log_prob_tensor = torch.stack(
                log_probabilities
            )

            loss = -(
                log_prob_tensor
                * returns_tensor
            ).mean()

            self.optimizer.zero_grad()

            loss.backward()

            nn.utils.clip_grad_norm_(
                self.policy.parameters(),
                1.0,
            )

            self.optimizer.step()

        return {
            "reward": float(
                np.sum(rewards)
            ),

            "mean_distance": float(
                np.mean(distances)
            ),

            "collision_steps": collision_steps,
        }

    def train_source(
        self,
        episodes=SOURCE_EPISODES,
    ):

        print(
            "\nTraining source policy..."
        )

        for episode in range(episodes):

            env = DroneEnv(
                sample_source_domain()
            )

            result = self.run_episode(
                env,
                training=True,
            )

            self.reward_history.append(
                result["reward"]
            )

            if (
                (episode + 1) % 10 == 0
                or episode == 0
            ):

                recent = np.mean(
                    self.reward_history[-10:]
                )

                print(
                    f"Source episode {episode+1:3d}/{episodes}"
                    f" | reward={result['reward']:8.2f}"
                    f" | recent mean={recent:8.2f}"
                )

        return self

    def fine_tune(
        self,
        target_env,
        episodes=ADAPTATION_EPISODES,
    ):

        print(
            "\nFine-tuning on target environment..."
        )

        for episode in range(episodes):

            result = self.run_episode(
                target_env,
                training=True,
            )

            self.reward_history.append(
                result["reward"]
            )

            if (
                (episode + 1) % 5 == 0
                or episode == 0
            ):

                print(
                    f"Adaptation episode {episode+1:3d}/{episodes}"
                    f" | reward={result['reward']:8.2f}"
                    f" | distance={result['mean_distance']:7.2f}"
                )


# ============================================================
# 8. SYSTEM IDENTIFICATION
# ============================================================

@dataclass
class DynamicsEstimate:

    motor_gain: float
    linear_drag: float
    quadratic_drag: float
    wind_x: float
    wind_y: float

    def as_array(self):

        return np.array(
            [
                self.motor_gain,
                self.linear_drag,
                self.quadratic_drag,
                self.wind_x,
                self.wind_y,
            ],
            dtype=np.float64,
        )


def collect_calibration_data(
    env,
    steps=CALIBRATION_STEPS,
):

    """
    Collect transitions using bounded exploratory actions.

    Regression input for each axis:

        [MAX_ACCEL * action,
         -velocity,
         -abs(velocity) * velocity,
         1]

    Regression output:

        measured acceleration

    Therefore:

        beta = [motor_gain, linear_drag,
                quadratic_drag, wind]

    This estimates the physical coefficients from transitions.
    """

    env.reset()

    features = []
    targets = []

    for step in range(steps):

        # Bounded persistent excitation:
        # action variation helps distinguish motor gain,
        # drag and constant wind.
        t = step * DT

        action = np.array(
            [
                0.65 * math.sin(1.31 * t)
                + 0.25 * math.sin(2.17 * t),

                0.60 * math.cos(1.07 * t)
                + 0.30 * math.sin(1.83 * t),
            ],
            dtype=np.float64,
        )

        action = np.clip(
            action,
            -1.0,
            1.0,
        )

        old_velocity = (
            env.velocity.copy()
        )

        _, measured_acceleration = env.step(
            action
        )

        # Skip saturated samples: speed clipping would
        # otherwise distort the acceleration regression.
        if np.linalg.norm(env.velocity) >= (
            MAX_SPEED - 1.0
        ):

            continue

        for axis in range(2):

            v = old_velocity[axis]

            row = [
                MAX_ACCEL * action[axis],
                -v,
                -abs(v) * v,
                1.0,
            ]

            features.append(row)

            targets.append(
                measured_acceleration[axis]
            )

    return (
        np.asarray(
            features,
            dtype=np.float64,
        ),

        np.asarray(
            targets,
            dtype=np.float64,
        ),
    )


def identify_dynamics(
    features,
    targets,
):

    """
    Ordinary least squares:

        beta_hat = pinv(X) @ y

    The first three coefficients are shared across axes.
    The wind intercept is estimated separately for each axis.
    """

    beta, _, rank, singular_values = (
        np.linalg.lstsq(
            features,
            targets,
            rcond=None,
        )
    )

    estimate = DynamicsEstimate(
        motor_gain=float(beta[0]),
        linear_drag=float(beta[1]),
        quadratic_drag=float(beta[2]),
        wind_x=float(beta[3]),
        wind_y=float(beta[3]),
    )

    # The feature rows are ordered x,y,x,y...
    # Refit each axis separately to estimate independent wind.
    beta_x, *_ = np.linalg.lstsq(
        features[0::2],
        targets[0::2],
        rcond=None,
    )

    beta_y, *_ = np.linalg.lstsq(
        features[1::2],
        targets[1::2],
        rcond=None,
    )

    estimate = DynamicsEstimate(
        motor_gain=float(
            0.5 * (beta_x[0] + beta_y[0])
        ),

        linear_drag=float(
            0.5 * (beta_x[1] + beta_y[1])
        ),

        quadratic_drag=float(
            0.5 * (beta_x[2] + beta_y[2])
        ),

        wind_x=float(
            beta_x[3]
        ),

        wind_y=float(
            beta_y[3]
        ),
    )

    predictions = (
        features @ beta
    )

    rmse = float(
        np.sqrt(
            np.mean(
                (predictions - targets) ** 2
            )
        )
    )

    return (
        estimate,
        rmse,
        int(rank),
        singular_values,
    )


def print_identification_results(
    estimate,
    rmse,
):

    print(
        "\nDynamics identification results"
    )

    print(
        "-" * 62
    )

    true_values = TARGET_TRUE

    estimated_values = {
        "motor_gain": estimate.motor_gain,
        "linear_drag": estimate.linear_drag,
        "quadratic_drag": estimate.quadratic_drag,
        "wind_x": estimate.wind_x,
        "wind_y": estimate.wind_y,
    }

    for name in true_values:

        true_value = true_values[name]

        estimated_value = (
            estimated_values[name]
        )

        error = (
            estimated_value - true_value
        )

        print(
            f"{name:20s}"
            f" true={true_value:10.5f}"
            f" estimated={estimated_value:10.5f}"
            f" error={error:10.5f}"
        )

    print(
        f"\nCalibration acceleration RMSE: {rmse:.6f}"
    )


# ============================================================
# 9. EVALUATION
# ============================================================

def evaluate_policy(
    policy,
    domain,
    episodes=EVAL_EPISODES,
):

    results = []

    for episode in range(episodes):

        env = DroneEnv(
            copy.deepcopy(domain)
        )

        observation = env.reset()

        distances = []
        collisions = 0

        minimum_clearance = float(
            "inf"
        )

        for _ in range(EVAL_STEPS):

            state_tensor = torch.as_tensor(
                observation,
                dtype=torch.float32,
                device=DEVICE,
            ).unsqueeze(0)

            with torch.no_grad():

                action_tensor, _ = (
                    policy.sample_action(
                        state_tensor,
                        deterministic=True,
                    )
                )

            action = (
                action_tensor
                .cpu()
                .numpy()[0]
            )

            observation, _ = env.step(
                action
            )

            distance = np.linalg.norm(
                env.position
                - env.target.position()
            )

            distances.append(
                distance
            )

            clearance = (
                np.linalg.norm(
                    env.position - OBSTACLE
                )
                - OBSTACLE_RADIUS
            )

            minimum_clearance = min(
                minimum_clearance,
                clearance,
            )

            if clearance < 0:

                collisions += 1

        results.append(
            {
                "distance": float(
                    np.mean(distances)
                ),

                "collision_steps": float(
                    collisions
                ),

                "minimum_clearance": float(
                    minimum_clearance
                ),
            }
        )

    return {
        key: float(
            np.mean(
                [
                    result[key]
                    for result in results
                ]
            )
        )
        for key in results[0]
    }


def evaluate_transfer(
    source_policy,
    calibrated_policy,
    adapted_policy,
):

    domain = target_domain()

    policies = {
        "Zero-shot": source_policy,
        "Calibrated": calibrated_policy,
        "Fine-tuned": adapted_policy,
    }

    results = {}

    print(
        "\n" + "=" * 78
    )

    print(
        "TARGET-ENVIRONMENT TRANSFER EVALUATION"
    )

    print(
        "=" * 78
    )

    for name, policy in policies.items():

        result = evaluate_policy(
            policy,
            domain,
        )

        results[name] = result

        print(
            f"{name:15s}"
            f" | distance={result['distance']:8.2f}"
            f" | collision steps={result['collision_steps']:7.2f}"
            f" | min clearance={result['minimum_clearance']:8.2f}"
        )

    return results


# ============================================================
# 10. VISUALIZATION
# ============================================================

def visualize_training(
    trainer,
):

    rewards = np.asarray(
        trainer.reward_history,
        dtype=np.float64,
    )

    plt.figure(
        figsize=(10, 5)
    )

    plt.plot(
        rewards,
        alpha=0.35,
        label="Episode reward",
    )

    if len(rewards) >= 10:

        smoothed = np.convolve(
            rewards,
            np.ones(10) / 10,
            mode="valid",
        )

        plt.plot(
            np.arange(9, len(rewards)),
            smoothed,
            label="10-episode average",
        )

    plt.axvline(
        SOURCE_EPISODES,
        linestyle="--",
        label="Start of target fine-tuning",
    )

    plt.title(
        "Source Training and Target Fine-Tuning"
    )

    plt.xlabel(
        "Episode"
    )

    plt.ylabel(
        "Episode reward"
    )

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    plt.show()


def visualize_transfer_results(
    results,
):

    names = list(
        results.keys()
    )

    metrics = [
        (
            "distance",
            "Mean target tracking distance",
        ),

        (
            "collision_steps",
            "Collision steps per episode",
        ),

        (
            "minimum_clearance",
            "Minimum obstacle clearance",
        ),
    ]

    for metric, title in metrics:

        values = [
            results[name][metric]
            for name in names
        ]

        plt.figure(
            figsize=(8, 5)
        )

        plt.bar(
            names,
            values,
        )

        plt.title(
            title
        )

        plt.grid(
            axis="y"
        )

        plt.tight_layout()

        plt.show()


def visualize_identification(
    estimate,
    rmse,
):

    true_values = [
        TARGET_TRUE["motor_gain"],
        TARGET_TRUE["linear_drag"],
        TARGET_TRUE["quadratic_drag"],
        TARGET_TRUE["wind_x"],
        TARGET_TRUE["wind_y"],
    ]

    estimated_values = [
        estimate.motor_gain,
        estimate.linear_drag,
        estimate.quadratic_drag,
        estimate.wind_x,
        estimate.wind_y,
    ]

    names = [
        "Motor gain",
        "Linear drag",
        "Quadratic drag",
        "Wind X",
        "Wind Y",
    ]

    x = np.arange(
        len(names)
    )

    width = 0.36

    plt.figure(
        figsize=(10, 5)
    )

    plt.bar(
        x - width / 2,
        true_values,
        width,
        label="Hidden target value",
    )

    plt.bar(
        x + width / 2,
        estimated_values,
        width,
        label="Estimated value",
    )

    plt.xticks(
        x,
        names,
    )

    plt.title(
        f"System Identification — acceleration RMSE {rmse:.4f}"
    )

    plt.grid(
        axis="y"
    )

    plt.legend()

    plt.tight_layout()

    plt.show()


# ============================================================
# 11. INTERACTIVE PYGAME
# ============================================================

def run_pygame(
    source_policy,
    adapted_policy,
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
        "Phase 4.25 — Sim-to-Real Drone Transfer"
    )

    clock = pygame.time.Clock()

    font = pygame.font.SysFont(
        "consolas",
        15,
    )

    domain = target_domain()

    env = DroneEnv(
        domain
    )

    observation = env.reset()

    policies = {
        1: (
            "Zero-shot source policy",
            source_policy,
        ),

        2: (
            "Target fine-tuned policy",
            adapted_policy,
        ),
    }

    selected = 1

    paused = False
    speed = 1

    trail = []

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

        center = env.position

        velocity = env.velocity

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

        position = env.target.position()

        velocity = env.target.velocity()

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

                    env = DroneEnv(
                        copy.deepcopy(domain)
                    )

                    observation = env.reset()

                    trail.clear()

                elif event.key == pygame.K_1:

                    selected = 1

                elif event.key == pygame.K_2:

                    selected = 2

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
                        > OBSTACLE_RADIUS + 110
                    ):

                        env.target.center = click

        policy_name, policy = policies[
            selected
        ]

        if not paused:

            for _ in range(speed):

                state_tensor = torch.as_tensor(
                    observation,
                    dtype=torch.float32,
                    device=DEVICE,
                ).unsqueeze(0)

                with torch.no_grad():

                    action_tensor, _ = (
                        policy.sample_action(
                            state_tensor,
                            deterministic=True,
                        )
                    )

                action = (
                    action_tensor
                    .cpu()
                    .numpy()[0]
                )

                observation, _ = env.step(
                    action
                )

                trail.append(
                    env.position.copy()
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

        draw_drone()

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
            f"Policy: {policy_name}",
            15,
            12,
        )

        draw_text(
            "Target domain: hidden physical parameters",
            15,
            38,
            (255, 195, 90),
        )

        draw_text(
            f"Target distance: {distance:.2f}",
            15,
            64,
        )

        draw_text(
            f"Obstacle clearance: {clearance:.2f}",
            15,
            90,
        )

        draw_text(
            f"Motor gain: {domain.motor_gain:.2f}",
            15,
            116,
        )

        draw_text(
            f"Linear drag: {domain.linear_drag:.3f}",
            15,
            142,
        )

        draw_text(
            f"Quadratic drag: {domain.quadratic_drag:.5f}",
            15,
            168,
        )

        draw_text(
            f"Wind: ({domain.wind_x:.1f}, {domain.wind_y:.1f})",
            15,
            194,
        )

        draw_text(
            f"Simulation speed: {speed}x",
            15,
            220,
        )

        draw_text(
            "1 Zero-shot | 2 Fine-tuned",
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
# 12. UNIT TESTS
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

    # Test 1 — Source-domain sampling.
    source = sample_source_domain()

    check(
        SOURCE_RANGES["motor_gain"][0]
        <= source.motor_gain
        <= SOURCE_RANGES["motor_gain"][1],
        "Source motor gain is in range",
    )

    # Test 2 — Target-domain definition.
    target = target_domain()

    check(
        np.isclose(
            target.motor_gain,
            TARGET_TRUE["motor_gain"],
        ),
        "Target domain contains hidden target parameters",
    )

    # Test 3 — Observation dimension.
    env = DroneEnv(
        nominal_domain()
    )

    observation = env.reset()

    check(
        observation.shape == (STATE_DIM,),
        "Observation has eight dimensions",
    )

    # Test 4 — Observation finite.
    check(
        np.all(np.isfinite(observation)),
        "Observation values are finite",
    )

    # Test 5 — Environment transition.
    next_observation, acceleration = env.step(
        np.array([0.2, -0.3])
    )

    check(
        next_observation.shape == (STATE_DIM,),
        "Environment returns valid next observation",
    )

    # Test 6 — Measured acceleration.
    check(
        acceleration.shape == (2,)
        and np.all(np.isfinite(acceleration)),
        "Measured acceleration has two finite components",
    )

    # Test 7 — Speed constraint.
    check(
        np.linalg.norm(env.velocity)
        <= MAX_SPEED + 1e-8,
        "Drone speed is bounded",
    )

    # Test 8 — Position constraint.
    check(
        np.all(env.position >= DRONE_RADIUS)
        and env.position[0] <= WORLD_W - DRONE_RADIUS
        and env.position[1] <= WORLD_H - DRONE_RADIUS,
        "Drone position is bounded",
    )

    # Test 9 — Target motion.
    old_target = env.target.position().copy()

    env.target.advance()

    check(
        not np.allclose(
            old_target,
            env.target.position(),
        ),
        "Moving target changes position",
    )

    # Test 10 — Reward finite.
    reward = calculate_reward(
        np.array([100.0, 100.0]),
        np.array([105.0, 105.0]),
        np.array([300.0, 300.0]),
        np.array([0.1, 0.1]),
    )

    check(
        np.isfinite(reward),
        "Reward is finite",
    )

    # Test 11 — Discounted returns.
    returns = discounted_returns(
        [1.0, 1.0, 1.0],
        gamma=0.9,
    )

    check(
        np.allclose(
            returns,
            [2.71, 1.9, 1.0],
        ),
        "Discounted returns are correct",
    )

    # Test 12 — Policy action shape.
    policy = GaussianPolicy().to(
        DEVICE
    )

    state = torch.zeros(
        (1, STATE_DIM),
        device=DEVICE,
    )

    action, log_probability = (
        policy.sample_action(state)
    )

    check(
        action.shape == (1, ACTION_DIM),
        "Policy action has correct shape",
    )

    # Test 13 — Action bounds.
    check(
        torch.all(action <= 1.0)
        and torch.all(action >= -1.0),
        "Policy action is bounded",
    )

    # Test 14 — Log probability finite.
    check(
        torch.all(
            torch.isfinite(log_probability)
        ),
        "Policy log probability is finite",
    )

    # Test 15 — Regression recovers known parameters.
    rng = np.random.default_rng(42)

    true_beta = np.array(
        [
            0.72,
            0.11,
            0.00030,
            3.5,
        ]
    )

    test_features = rng.normal(
        size=(500, 4)
    )

    # Ensure a reasonable scale for the velocity-related columns.
    test_features[:, 1] *= 20.0
    test_features[:, 2] *= 100.0

    test_targets = (
        test_features @ true_beta
    )

    recovered, *_ = np.linalg.lstsq(
        test_features,
        test_targets,
        rcond=None,
    )

    check(
        np.allclose(
            recovered,
            true_beta,
            atol=1e-7,
        ),
        "Least-squares regression recovers known parameters",
    )

    # Test 16 — Calibration feature shape.
    calibration_env = DroneEnv(
        target_domain()
    )

    features, targets = collect_calibration_data(
        calibration_env,
        steps=80,
    )

    check(
        features.ndim == 2
        and features.shape[1] == 4
        and len(features) == len(targets),
        "Calibration dataset has correct dimensions",
    )

    # Test 17 — Identified parameters finite.
    estimate, rmse, rank, singular_values = (
        identify_dynamics(
            features,
            targets,
        )
    )

    check(
        np.all(
            np.isfinite(
                estimate.as_array()
            )
        ),
        "Identified dynamics parameters are finite",
    )

    # Test 18 — Regression error finite.
    check(
        np.isfinite(rmse)
        and rmse >= 0.0,
        "Calibration RMSE is finite and nonnegative",
    )

    # Test 19 — Regression rank.
    check(
        rank >= 3,
        "Calibration regression has useful rank",
    )

    # Test 20 — Deterministic policy.
    with torch.no_grad():

        action_a, _ = policy.sample_action(
            state,
            deterministic=True,
        )

        action_b, _ = policy.sample_action(
            state,
            deterministic=True,
        )

    check(
        torch.allclose(
            action_a,
            action_b,
        ),
        "Deterministic policy output is repeatable",
    )

    # Test 21 — Policy gradients.
    policy.zero_grad()

    action, log_probability = (
        policy.sample_action(state)
    )

    loss = -log_probability.mean()

    loss.backward()

    gradients_valid = all(
        parameter.grad is not None
        and torch.all(
            torch.isfinite(parameter.grad)
        )
        for parameter in policy.parameters()
    )

    check(
        gradients_valid,
        "Policy gradients are present and finite",
    )

    print(
        f"\n{passed}/21 tests passed."
    )

    return passed == 21


# ============================================================
# 13. MAIN
# ============================================================

def main():

    print(
        "=" * 72
    )

    print(
        "PHASE 4.25 — SIM-TO-REAL TRANSFER"
    )

    print(
        "SYSTEM IDENTIFICATION + POLICY FINE-TUNING"
    )

    print(
        "=" * 72
    )

    print(
        f"Device: {DEVICE}"
    )

    # --------------------------------------------------------
    # Step 1 — Unit tests
    # --------------------------------------------------------

    print(
        "\n[1/7] Unit tests"
    )

    if not run_tests():

        raise RuntimeError(
            "Unit tests failed."
        )

    # --------------------------------------------------------
    # Step 2 — Train source policy
    # --------------------------------------------------------

    print(
        "\n[2/7] Source simulation training"
    )

    source_trainer = PolicyTrainer()

    source_trainer.train_source()

    source_policy = copy.deepcopy(
        source_trainer.policy
    )

    # --------------------------------------------------------
    # Step 3 — Collect target calibration data
    # --------------------------------------------------------

    print(
        "\n[3/7] Target-environment calibration"
    )

    calibration_env = DroneEnv(
        target_domain()
    )

    features, targets = collect_calibration_data(
        calibration_env
    )

    estimate, rmse, rank, singular_values = (
        identify_dynamics(
            features,
            targets,
        )
    )

    print_identification_results(
        estimate,
        rmse,
    )

    visualize_identification(
        estimate,
        rmse,
    )

    # --------------------------------------------------------
    # Step 4 — Prepare target adaptation
    # --------------------------------------------------------

    print(
        "\n[4/7] Prepare target adaptation"
    )

    # The calibrated dynamics estimate is reported separately.
    # This policy adaptation experiment uses target interaction
    # data, rather than silently substituting estimated parameters
    # into the environment.

    adapted_policy = copy.deepcopy(
        source_policy
    )

    adapted_trainer = PolicyTrainer(
        policy=adapted_policy,
        learning_rate=ADAPT_LEARNING_RATE,
    )

    target_training_env = DroneEnv(
        target_domain()
    )

    # --------------------------------------------------------
    # Step 5 — Fine-tune policy
    # --------------------------------------------------------

    print(
        "\n[5/7] Fine-tuning with limited target data"
    )

    adapted_trainer.fine_tune(
        target_training_env
    )

    adapted_policy = copy.deepcopy(
        adapted_trainer.policy
    )

    # --------------------------------------------------------
    # Step 6 — Evaluate transfer
    # --------------------------------------------------------

    print(
        "\n[6/7] Transfer evaluation"
    )

    results = evaluate_transfer(
        source_policy,
        source_policy,
        adapted_policy,
    )

    visualize_training(
        adapted_trainer
    )

    visualize_transfer_results(
        results
    )

    # --------------------------------------------------------
    # Step 7 — Interactive simulation
    # --------------------------------------------------------

    print(
        "\n[7/7] Interactive drone simulation"
    )

    run_pygame(
        source_policy,
        adapted_policy,
    )


if __name__ == "__main__":

    main()
