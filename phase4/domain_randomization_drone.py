
"""
File: phase4/domain_randomization_drone.py

PHASE 4.24 — DOMAIN RANDOMIZATION FOR SIM-TO-REAL TRANSFER

PURPOSE
-------
Train continuous-action drone controllers under different
physical conditions and evaluate their generalization.

Compare:
    1. Nominal REINFORCE policy
    2. Domain-randomized REINFORCE policy

WHAT YOU LEARN
--------------

STEP 1 — Domain Parameters
    Represent motor gain, drag, payload, wind and sensor noise.

STEP 2 — Domain Randomization
    Randomize physical parameters at the beginning of episodes.

STEP 3 — Continuous Policy
    Use a Gaussian neural-network policy with tanh-bounded actions.

STEP 4 — REINFORCE
    Train the policy using Monte Carlo returns and policy gradients.

STEP 5 — Generalization
    Evaluate the policies in nominal and unseen physical domains.

STEP 6 — Interactive Simulation
    Fly the quadcopter using the trained domain-randomized policy.

MATHEMATICAL FOUNDATIONS
------------------------

Domain randomization:

    theta ~ p(theta)

    s_(t+1) = f(s_t, a_t; theta)

    The controller is trained over a distribution of environments,
    rather than one fixed environment.

REINFORCE:

    grad J(theta) =
        E[ G_t * grad log pi_theta(a_t | s_t) ]

Discounted return:

    G_t = sum(k=0 to T-t) gamma^k * r_(t+k)

Baseline:

    A_t = G_t - b

Policy loss:

    L = -mean(A_t * log pi_theta(a_t | s_t))

The policy uses a Gaussian distribution followed by tanh:

    u = tanh(z)
    z ~ Normal(mu, sigma)

The log-probability includes the tanh Jacobian correction.

SIM-TO-REAL LIMITATIONS
----------------------
- Domain randomization improves robustness only within the
  range and types of variation represented during training.
- It does not guarantee real-world transfer.
- Sensor noise is simulated as observation noise.
- This is a simplified 2D drone model, not flight dynamics.
- The demonstration is not flight-certified.

ROBOTICS APPLICATION
--------------------
A drone may encounter different payloads, battery levels,
motor degradation, aerodynamic drag, and wind conditions.
Training across randomized conditions can reduce dependence
on one exact simulator configuration.

INSTALL
-------
    pip install numpy matplotlib pygame torch

RUN
---
    python phase4/domain_randomization_drone.py

CONTROLS
--------
    1 : Domain-randomized policy
    2 : Nominal policy

    SPACE : Pause
    R     : Reset
    + / - : Simulation speed
    Click : Change moving target trajectory center
    ESC   : Exit
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

SEED = 424

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
SAFE_RADIUS = 105.0

TARGET_A = 165.0
TARGET_B = 105.0
TARGET_OMEGA = 0.55

STATE_DIM = 8
ACTION_DIM = 2

HIDDEN_DIM = 128

GAMMA = 0.99
LEARNING_RATE = 2e-4

TRAIN_EPISODES = 180
MAX_EPISODE_STEPS = 240

EVAL_EPISODES = 5
EVAL_STEPS = 220

UPDATE_EVERY = 5

# Sensor noise used in the observation, not in the true state.
OBSERVATION_NOISE = 0.015

# Randomized domain ranges.
DOMAIN_RANGES = {
    "motor_gain": (0.60, 1.15),
    "linear_drag": (0.045, 0.15),
    "quadratic_drag": (0.00010, 0.00045),
    "wind_strength": (0.0, 22.0),
    "payload_factor": (0.80, 1.35),
}

# Physical domains used for evaluation.
EVALUATION_DOMAINS = {
    "Nominal": {
        "motor_gain": 0.85,
        "linear_drag": 0.085,
        "quadratic_drag": 0.00022,
        "wind_strength": 5.0,
        "payload_factor": 1.0,
    },

    "Heavy payload": {
        "motor_gain": 0.68,
        "linear_drag": 0.105,
        "quadratic_drag": 0.00035,
        "wind_strength": 5.0,
        "payload_factor": 1.30,
    },

    "Motor degradation": {
        "motor_gain": 0.55,
        "linear_drag": 0.095,
        "quadratic_drag": 0.00025,
        "wind_strength": 8.0,
        "payload_factor": 1.15,
    },

    "High drag": {
        "motor_gain": 0.80,
        "linear_drag": 0.17,
        "quadratic_drag": 0.00050,
        "wind_strength": 4.0,
        "payload_factor": 1.0,
    },

    "Strong wind": {
        "motor_gain": 0.85,
        "linear_drag": 0.085,
        "quadratic_drag": 0.00022,
        "wind_strength": 25.0,
        "payload_factor": 1.0,
    },

    # A combination deliberately outside the training ranges.
    "Unseen combined": {
        "motor_gain": 0.50,
        "linear_drag": 0.19,
        "quadratic_drag": 0.00055,
        "wind_strength": 28.0,
        "payload_factor": 1.40,
    },
}


# ============================================================
# 1. PHYSICAL DOMAIN
# ============================================================

@dataclass
class Domain:

    motor_gain: float = 0.85

    linear_drag: float = 0.085

    quadratic_drag: float = 0.00022

    wind_strength: float = 5.0

    payload_factor: float = 1.0

    def as_dict(self):

        return {
            "motor_gain": self.motor_gain,
            "linear_drag": self.linear_drag,
            "quadratic_drag": self.quadratic_drag,
            "wind_strength": self.wind_strength,
            "payload_factor": self.payload_factor,
        }


def nominal_domain():

    return Domain()


def random_domain():

    return Domain(
        motor_gain=np.random.uniform(
            *DOMAIN_RANGES["motor_gain"]
        ),

        linear_drag=np.random.uniform(
            *DOMAIN_RANGES["linear_drag"]
        ),

        quadratic_drag=np.random.uniform(
            *DOMAIN_RANGES["quadratic_drag"]
        ),

        wind_strength=np.random.uniform(
            *DOMAIN_RANGES["wind_strength"]
        ),

        payload_factor=np.random.uniform(
            *DOMAIN_RANGES["payload_factor"]
        ),
    )


def domain_from_dict(values):

    return Domain(
        **{
            key: float(value)
            for key, value in values.items()
        }
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

        self.phase += TARGET_OMEGA * DT


# ============================================================
# 3. DRONE ENVIRONMENT
# ============================================================

class DroneEnv:

    """
    State:
        [x, y, vx, vy]

    Action:
        [ax, ay], normalized to [-1, 1]

    Dynamics:
        acceleration =
            motor_gain * MAX_ACCEL * action / payload_factor
            - linear_drag * velocity
            - quadratic_drag * abs(velocity) * velocity
            + wind

    Wind changes over time, so the environment is not perfectly
    deterministic even when the domain parameters are fixed.
    """

    def __init__(self, domain=None):

        self.domain = (
            nominal_domain()
            if domain is None
            else domain
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

        self.wind = np.random.uniform(
            -self.domain.wind_strength,
            self.domain.wind_strength,
            size=2,
        )

        self.target.phase = np.random.uniform(
            0.0,
            2.0 * math.pi,
        )

        return self.observe()

    def true_state(self):

        return np.concatenate(
            [
                self.position,
                self.velocity,
            ]
        )

    def observe(self):

        """
        Relative coordinates make the policy less dependent
        on absolute world position.

        State:
            target relative position: 2
            target relative velocity: 2
            drone velocity: 2
            obstacle relative position: 2
        """

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
            / d.payload_factor

            - d.linear_drag
            * self.velocity

            - d.quadratic_drag
            * np.abs(self.velocity)
            * self.velocity

            + self.wind
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
            0.90 * self.wind
            + np.random.normal(
                0.0,
                max(
                    0.5,
                    d.wind_strength * 0.12,
                ),
                size=2,
            )
        )

        self.wind = np.clip(
            self.wind,
            -2.0 * d.wind_strength - 1.0,
            2.0 * d.wind_strength + 1.0,
        )

        self.target.advance()

        return self.observe()


# ============================================================
# 4. REWARD FUNCTION
# ============================================================

def calculate_reward(
    old_position,
    new_position,
    target_position,
    action,
):

    """
    Reward combines:
        - tracking distance
        - progress toward target
        - obstacle clearance
        - boundary clearance
        - control effort

    Progress is measured against the same future target
    position before and after the drone moves.
    """

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

    control_penalty = (
        0.025 * np.sum(action**2)
    )

    reward = (
        0.035 * progress
        - 0.012 * distance_after
        - 1.8 * obstacle_penalty
        - 1.5 * boundary_penalty
        - collision_penalty
        - control_penalty
    )

    return float(reward)


# ============================================================
# 5. GAUSSIAN POLICY
# ============================================================

class GaussianPolicy(nn.Module):

    """
    Neural policy:

        state -> MLP -> mean action

    A learned log standard deviation defines exploration.

    The sampled Gaussian action is passed through tanh so that
    the physical control remains inside [-1, 1].
    """

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

    def distribution(self, state):

        mean = self.network(state)

        log_std = torch.clamp(
            self.log_std,
            -2.0,
            0.5,
        )

        std = torch.exp(
            log_std
        )

        return torch.distributions.Normal(
            mean,
            std,
        )

    def sample_action(
        self,
        state,
        deterministic=False,
    ):

        distribution = self.distribution(
            state
        )

        if deterministic:

            raw_action = distribution.mean

        else:

            raw_action = distribution.rsample()

        action = torch.tanh(
            raw_action
        )

        # Change-of-variables correction for tanh.
        log_probability = (
            distribution.log_prob(raw_action)
            - torch.log(
                1.0 - action.pow(2) + 1e-6
            )
        ).sum(
            dim=-1
        )

        return action, log_probability


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

    running_return = 0.0

    for index in reversed(
        range(len(rewards))
    ):

        running_return = (
            rewards[index]
            + gamma * running_return
        )

        result[index] = running_return

    return result


# ============================================================
# 7. REINFORCE TRAINER
# ============================================================

class ReinforceTrainer:

    def __init__(self):

        self.policy = GaussianPolicy().to(
            DEVICE
        )

        self.optimizer = optim.Adam(
            self.policy.parameters(),
            lr=LEARNING_RATE,
        )

        self.reward_history = []

    def collect_episode(
        self,
        env,
        max_steps=MAX_EPISODE_STEPS,
        training=True,
    ):

        observation = env.reset()

        log_probabilities = []
        rewards = []

        distances = []

        collisions = 0

        for step in range(max_steps):

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

            next_observation = env.step(
                action
            )

            future_target_position = (
                env.target.position()
            )

            reward = calculate_reward(
                old_position,
                env.position,
                future_target_position,
                action,
            )

            rewards.append(reward)

            distance = np.linalg.norm(
                env.position
                - env.target.position()
            )

            distances.append(distance)

            clearance = (
                np.linalg.norm(
                    env.position - OBSTACLE
                )
                - OBSTACLE_RADIUS
            )

            if clearance < 0:

                collisions += 1

            observation = next_observation

        if training:

            returns = discounted_returns(
                rewards
            )

            returns_tensor = torch.as_tensor(
                returns,
                dtype=torch.float32,
                device=DEVICE,
            )

            # Normalize returns within the episode.
            # A small standard deviation is protected against.
            if returns_tensor.numel() > 1:

                returns_tensor = (
                    returns_tensor
                    - returns_tensor.mean()
                ) / (
                    returns_tensor.std(
                        unbiased=False
                    ) + 1e-6
                )

            log_probability_tensor = torch.stack(
                log_probabilities
            )

            loss = -(
                log_probability_tensor
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
            "episode_reward": float(
                np.sum(rewards)
            ),

            "mean_distance": float(
                np.mean(distances)
            ),

            "collision_steps": collisions,

            "episode_length": len(rewards),
        }

    def train(
        self,
        randomized,
        episodes=TRAIN_EPISODES,
        label="Policy",
    ):

        print(
            f"\nTraining {label}"
        )

        for episode in range(episodes):

            if randomized:

                domain = random_domain()

            else:

                domain = nominal_domain()

            env = DroneEnv(
                domain
            )

            result = self.collect_episode(
                env,
                training=True,
            )

            self.reward_history.append(
                result["episode_reward"]
            )

            if (
                (episode + 1) % UPDATE_EVERY == 0
                or episode == 0
            ):

                recent = np.mean(
                    self.reward_history[-UPDATE_EVERY:]
                )

                print(
                    f"Episode {episode+1:4d}/{episodes} "
                    f"| reward {result['episode_reward']:8.2f} "
                    f"| recent mean {recent:8.2f} "
                    f"| domain {'randomized' if randomized else 'nominal'}"
                )

        return self


# ============================================================
# 8. TRAINING COMPARISON
# ============================================================

def train_both_policies():

    print(
        "\n" + "=" * 65
    )

    print(
        "TRAINING NOMINAL POLICY"
    )

    print(
        "=" * 65
    )

    nominal_trainer = ReinforceTrainer()

    nominal_trainer.train(
        randomized=False,
        label="Nominal policy",
    )

    print(
        "\n" + "=" * 65
    )

    print(
        "TRAINING DOMAIN-RANDOMIZED POLICY"
    )

    print(
        "=" * 65
    )

    randomized_trainer = ReinforceTrainer()

    randomized_trainer.train(
        randomized=True,
        label="Domain-randomized policy",
    )

    return (
        nominal_trainer,
        randomized_trainer,
    )


# ============================================================
# 9. POLICY EVALUATION
# ============================================================

def evaluate_policy(
    policy,
    domain_values,
    episodes=EVAL_EPISODES,
):

    domain = domain_from_dict(
        domain_values
    )

    results = []

    for episode in range(episodes):

        env = DroneEnv(
            copy.deepcopy(domain)
        )

        observation = env.reset()

        distances = []

        collisions = 0

        minimum_clearance = float("inf")

        for step in range(EVAL_STEPS):

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

            observation = env.step(
                action
            )

            distance = np.linalg.norm(
                env.position
                - env.target.position()
            )

            distances.append(distance)

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
                "mean_distance": float(
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


def evaluate_both(
    nominal_trainer,
    randomized_trainer,
):

    all_results = {}

    policies = {
        "Nominal policy": nominal_trainer.policy,
        "Domain-randomized policy": randomized_trainer.policy,
    }

    print(
        "\n" + "=" * 75
    )

    print(
        "GENERALIZATION EVALUATION"
    )

    print(
        "=" * 75
    )

    for domain_name, domain_values in (
        EVALUATION_DOMAINS.items()
    ):

        print(
            f"\nDomain: {domain_name}"
        )

        all_results[domain_name] = {}

        for policy_name, policy in policies.items():

            result = evaluate_policy(
                policy,
                domain_values,
            )

            all_results[domain_name][
                policy_name
            ] = result

            print(
                f"  {policy_name:<27}"
                f" | distance: {result['mean_distance']:8.2f}"
                f" | collision steps: {result['collision_steps']:7.2f}"
                f" | min clearance: {result['minimum_clearance']:8.2f}"
            )

    return all_results


# ============================================================
# 10. VISUALIZATION
# ============================================================

def moving_average(
    values,
    window=12,
):

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    if len(values) < window:

        return values

    return np.convolve(
        values,
        np.ones(window) / window,
        mode="valid",
    )


def visualize_training(
    nominal_trainer,
    randomized_trainer,
):

    plt.figure(
        figsize=(10, 5)
    )

    plt.plot(
        moving_average(
            nominal_trainer.reward_history
        ),
        label="Nominal policy",
    )

    plt.plot(
        moving_average(
            randomized_trainer.reward_history
        ),
        label="Domain-randomized policy",
    )

    plt.title(
        "REINFORCE Training Reward"
    )

    plt.xlabel(
        "Episode"
    )

    plt.ylabel(
        "Smoothed episode reward"
    )

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    plt.show()


def visualize_evaluation(results):

    domain_names = list(
        results.keys()
    )

    metrics = [
        (
            "mean_distance",
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

    policy_names = [
        "Nominal policy",
        "Domain-randomized policy",
    ]

    x = np.arange(
        len(domain_names)
    )

    width = 0.36

    for metric, title in metrics:

        plt.figure(
            figsize=(12, 5)
        )

        for index, policy_name in enumerate(
            policy_names
        ):

            values = [
                results[domain][policy_name][metric]
                for domain in domain_names
            ]

            plt.bar(
                x + (index - 0.5) * width,
                values,
                width,
                label=policy_name,
            )

        plt.xticks(
            x,
            domain_names,
            rotation=15,
        )

        plt.title(title)

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
    nominal_trainer,
    randomized_trainer,
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
        "Phase 4.24 — Domain Randomization Drone"
    )

    clock = pygame.time.Clock()

    font = pygame.font.SysFont(
        "consolas",
        15,
    )

    policies = {
        1: (
            "Domain-randomized policy",
            randomized_trainer.policy,
        ),

        2: (
            "Nominal policy",
            nominal_trainer.policy,
        ),
    }

    selected = 1

    domain = domain_from_dict(
        EVALUATION_DOMAINS[
            "Unseen combined"
        ]
    )

    env = DroneEnv(
        domain
    )

    observation = env.reset()

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

                observation = env.step(
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
            "Test domain: Unseen combined",
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
            f"Payload factor: {domain.payload_factor:.2f}",
            15,
            168,
        )

        draw_text(
            f"Wind strength: {domain.wind_strength:.1f}",
            15,
            194,
        )

        draw_text(
            f"Simulation speed: {speed}x",
            15,
            220,
        )

        draw_text(
            "1 Domain-randomized | 2 Nominal",
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

    # Test 1 — Nominal domain.
    domain = nominal_domain()

    check(
        0.0 < domain.motor_gain < 2.0,
        "Nominal motor gain is valid",
    )

    # Test 2 — Random domain.
    domain = random_domain()

    check(
        DOMAIN_RANGES["motor_gain"][0]
        <= domain.motor_gain
        <= DOMAIN_RANGES["motor_gain"][1],
        "Randomized motor gain is inside configured range",
    )

    # Test 3 — All random parameters inside ranges.
    values = domain.as_dict()

    check(
        all(
            DOMAIN_RANGES[key][0]
            <= values[key]
            <= DOMAIN_RANGES[key][1]
            for key in DOMAIN_RANGES
        ),
        "All randomized parameters are inside configured ranges",
    )

    # Test 4 — Environment observation.
    env = DroneEnv()

    observation = env.reset()

    check(
        observation.shape == (STATE_DIM,),
        "Observation has eight dimensions",
    )

    # Test 5 — Observation finite.
    check(
        np.all(np.isfinite(observation)),
        "Observation contains finite values",
    )

    # Test 6 — Environment transition.
    next_observation = env.step(
        np.array([0.2, -0.3])
    )

    check(
        next_observation.shape == (STATE_DIM,),
        "Environment returns valid next observation",
    )

    # Test 7 — Physical speed limit.
    check(
        np.linalg.norm(env.velocity)
        <= MAX_SPEED + 1e-8,
        "Physical drone speed is bounded",
    )

    # Test 8 — Physical position bounds.
    check(
        np.all(env.position >= DRONE_RADIUS)
        and env.position[0] <= WORLD_W - DRONE_RADIUS
        and env.position[1] <= WORLD_H - DRONE_RADIUS,
        "Physical drone position is bounded",
    )

    # Test 9 — Target motion.
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
        "Reward calculation is finite",
    )

    # Test 11 — Reward improves with progress.
    reward_near = calculate_reward(
        np.array([100.0, 100.0]),
        np.array([150.0, 150.0]),
        np.array([160.0, 160.0]),
        np.zeros(2),
    )

    reward_far = calculate_reward(
        np.array([100.0, 100.0]),
        np.array([100.0, 100.0]),
        np.array([300.0, 300.0]),
        np.zeros(2),
    )

    check(
        np.isfinite(reward_near)
        and np.isfinite(reward_far),
        "Reward handles different tracking distances",
    )

    # Test 12 — Discounted returns.
    returns = discounted_returns(
        [1.0, 1.0, 1.0],
        gamma=0.9,
    )

    check(
        np.allclose(
            returns,
            [2.71, 1.9, 1.0],
        ),
        "Discounted return calculation is correct",
    )

    # Test 13 — Policy output.
    policy = GaussianPolicy().to(
        DEVICE
    )

    state = torch.zeros(
        (1, STATE_DIM),
        device=DEVICE,
    )

    with torch.no_grad():

        action, log_probability = (
            policy.sample_action(
                state
            )
        )

    check(
        action.shape == (1, ACTION_DIM),
        "Policy returns two-dimensional action",
    )

    # Test 14 — Action bounds.
    check(
        torch.all(action <= 1.0)
        and torch.all(action >= -1.0),
        "Tanh policy action is bounded",
    )

    # Test 15 — Log probability finite.
    check(
        torch.all(
            torch.isfinite(log_probability)
        ),
        "Policy log probability is finite",
    )

    # Test 16 — Deterministic action.
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
        "Deterministic policy gives repeatable actions",
    )

    # Test 17 — Evaluation domain definitions.
    check(
        "Unseen combined" in EVALUATION_DOMAINS,
        "Unseen combined evaluation domain exists",
    )

    # Test 18 — Domain conversion.
    converted = domain_from_dict(
        EVALUATION_DOMAINS["Nominal"]
    )

    check(
        np.isclose(
            converted.motor_gain,
            EVALUATION_DOMAINS["Nominal"]["motor_gain"],
        ),
        "Domain dictionary conversion is correct",
    )

    # Test 19 — Observation relative-position dimension.
    check(
        len(env.observe()) == 8,
        "State representation includes target and obstacle information",
    )

    # Test 20 — Policy parameter gradients.
    policy.zero_grad()

    action, log_probability = policy.sample_action(
        state
    )

    loss = -log_probability.mean()

    loss.backward()

    gradients_exist = all(
        parameter.grad is not None
        and torch.all(
            torch.isfinite(parameter.grad)
        )
        for parameter in policy.parameters()
    )

    check(
        gradients_exist,
        "Policy gradients are present and finite",
    )

    print(
        f"\n{passed}/20 tests passed."
    )

    return passed == 20


# ============================================================
# 13. MAIN
# ============================================================

def main():

    print(
        "=" * 72
    )

    print(
        "PHASE 4.24 — DOMAIN RANDOMIZATION"
    )

    print(
        "REINFORCE + CONTINUOUS DRONE CONTROL"
    )

    print(
        "=" * 72
    )

    print(
        f"Device: {DEVICE}"
    )

    print(
        "\n[1/5] Unit tests"
    )

    if not run_tests():

        raise RuntimeError(
            "Unit tests failed."
        )

    print(
        "\n[2/5] Training two policies"
    )

    nominal_trainer, randomized_trainer = (
        train_both_policies()
    )

    print(
        "\n[3/5] Visualizing training"
    )

    visualize_training(
        nominal_trainer,
        randomized_trainer,
    )

    print(
        "\n[4/5] Evaluating generalization"
    )

    results = evaluate_both(
        nominal_trainer,
        randomized_trainer,
    )

    visualize_evaluation(
        results
    )

    print(
        "\n[5/5] Interactive drone simulation"
    )

    print(
        "The interactive simulation uses the unseen combined domain."
    )

    run_pygame(
        nominal_trainer,
        randomized_trainer,
    )


if __name__ == "__main__":

    main()
