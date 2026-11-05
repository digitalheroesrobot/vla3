
"""
File: phase5/continuous_action_policies.py

PHASE 5.10 — CONTINUOUS-ACTION POLICIES
=======================================

PURPOSE
-------
Understand and implement continuous-action probability distributions
for reinforcement learning and robotic control.

WHAT YOU LEARN
--------------

STEP 1 — Gaussian Policy
    A policy outputs a mean and standard deviation:

        a ~ Normal(mu(s), sigma(s))

STEP 2 — Sampling
    Generate continuous actions using:

        a = mu + sigma * epsilon

        epsilon ~ Normal(0, 1)

STEP 3 — Tanh Squashing
    Restrict actions to [-1, 1]:

        a = tanh(u)

STEP 4 — Log Probability
    Calculate the likelihood of sampled actions.

STEP 5 — Jacobian Correction
    Account for the change in probability density caused by tanh:

        log pi(a|s) =
            log N(u; mu, sigma)
            - sum(log(1 - tanh(u)^2))

STEP 6 — REINFORCE
    Optimize the policy using:

        grad J(theta) =
            E[grad log pi(a|s) * G_t]

STEP 7 — Robotics Application
    Control a quadcopter using continuous acceleration commands
    while tracking a moving target.

IMPORTANT MATHEMATICAL DETAILS
------------------------------

1. Gaussian entropy:

       H = sum(log(sigma) + 0.5*log(2*pi*e))

2. Gaussian log probability:

       log N(u; mu, sigma) =
           -0.5 * [
               ((u-mu)/sigma)^2
               + 2*log(sigma)
               + log(2*pi)
           ]

3. Tanh transformation:

       a = tanh(u)

4. Tanh Jacobian:

       da/du = 1 - tanh(u)^2

5. Policy gradient:

       L = -log pi(a|s) * advantage

The entropy of the original Gaussian is not the exact entropy
of its tanh-transformed distribution. This lesson uses the
Gaussian entropy as a practical exploration regularizer.

This is a simplified educational 2D drone simulator, not a
flight-certified controller.
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

SEED = 510

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

OBS_DIM = 10
ACTION_DIM = 2

WORLD_W = 900
WORLD_H = 650

DT = 0.08

MAX_SPEED = 150.0
MAX_ACCEL = 125.0
DRONE_RADIUS = 12.0

OBSTACLE = np.array(
    [450.0, 330.0],
    dtype=np.float32,
)

OBSTACLE_RADIUS = 62.0
SAFE_RADIUS = 112.0

TARGET_A = 145.0
TARGET_B = 95.0
TARGET_OMEGA = 0.55

GAMMA = 0.99

LEARNING_RATE = 3e-4
ENTROPY_COEFFICIENT = 0.005

TRAIN_EPISODES = 350
MAX_EPISODE_STEPS = 180


# ============================================================
# 1. GAUSSIAN DISTRIBUTION MATHEMATICS
# ============================================================

def gaussian_log_prob(
    actions,
    mean,
    log_std,
):
    """
    Elementwise log probability of a diagonal Gaussian.

    log_std is used instead of std directly because it is
    unconstrained and guarantees positive std after exponentiation.
    """

    std = torch.exp(log_std)

    normalized = (
        actions - mean
    ) / std

    return -0.5 * (
        normalized.pow(2)
        + 2.0 * log_std
        + math.log(2.0 * math.pi)
    )


def gaussian_entropy(log_std):
    """
    Entropy of a diagonal Gaussian:

        H = sum(log_std + 0.5*log(2*pi*e))
    """

    return (
        log_std
        + 0.5 * math.log(2.0 * math.pi * math.e)
    ).sum(dim=-1)


def gaussian_sample(
    mean,
    log_std,
):

    std = torch.exp(log_std)

    epsilon = torch.randn_like(
        mean
    )

    return mean + std * epsilon


# ============================================================
# 2. TANH-SQUASHED GAUSSIAN
# ============================================================

def tanh_squash(raw_actions):

    return torch.tanh(
        raw_actions
    )


def tanh_log_jacobian(raw_actions):
    """
    Numerically stable calculation of:

        log(1 - tanh(u)^2)

    Identity:

        log(1 - tanh(u)^2)
        = 2 * (log(2) - u - softplus(-2u))
    """

    return 2.0 * (
        math.log(2.0)
        - raw_actions
        - torch.nn.functional.softplus(
            -2.0 * raw_actions
        )
    )


def squashed_gaussian_log_prob(
    raw_actions,
    mean,
    log_std,
):
    """
    Probability density after tanh transformation.

    The raw Gaussian sample is retained because atanh(a) becomes
    numerically unstable when a approaches exactly +/-1.
    """

    base_log_prob = gaussian_log_prob(
        raw_actions,
        mean,
        log_std,
    ).sum(dim=-1)

    log_jacobian = tanh_log_jacobian(
        raw_actions
    ).sum(dim=-1)

    return (
        base_log_prob - log_jacobian
    )


def squashed_gaussian_sample(
    mean,
    log_std,
):

    raw_actions = gaussian_sample(
        mean,
        log_std,
    )

    actions = tanh_squash(
        raw_actions
    )

    log_prob = squashed_gaussian_log_prob(
        raw_actions,
        mean,
        log_std,
    )

    return (
        actions,
        raw_actions,
        log_prob,
    )


# ============================================================
# 3. CONTINUOUS POLICY NETWORK
# ============================================================

class GaussianPolicy(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(
            nn.Linear(OBS_DIM, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
        )

        self.mean_head = nn.Linear(
            128,
            ACTION_DIM,
        )

        # State-independent log standard deviation.
        # This is a common practical parameterization.

        self.log_std = nn.Parameter(
            torch.full(
                (ACTION_DIM,),
                -0.5,
            )
        )

    def forward(self, states):

        features = self.features(
            states
        )

        mean = self.mean_head(
            features
        )

        log_std = torch.clamp(
            self.log_std,
            -5.0,
            1.0,
        )

        return mean, log_std.expand_as(mean)

    def sample(self, states):

        mean, log_std = self(states)

        return squashed_gaussian_sample(
            mean,
            log_std,
        )

    def evaluate_raw_actions(
        self,
        states,
        raw_actions,
    ):

        mean, log_std = self(states)

        log_prob = squashed_gaussian_log_prob(
            raw_actions,
            mean,
            log_std,
        )

        entropy = gaussian_entropy(
            log_std
        )

        return (
            log_prob,
            entropy,
        )

    def deterministic_action(self, states):

        mean, _ = self(states)

        return torch.tanh(
            mean
        )


# ============================================================
# 4. MOVING TARGET
# ============================================================

class MovingTarget:

    def __init__(self, center):

        self.center = np.asarray(
            center,
            dtype=np.float32,
        )

        self.phase = 0.0

    def position(self, phase=None):

        if phase is None:
            phase = self.phase

        return self.center + np.array(
            [
                TARGET_A * math.cos(phase),
                TARGET_B * math.sin(phase),
            ],
            dtype=np.float32,
        )

    def velocity(self, phase=None):

        if phase is None:
            phase = self.phase

        return np.array(
            [
                -TARGET_A * TARGET_OMEGA * math.sin(phase),
                TARGET_B * TARGET_OMEGA * math.cos(phase),
            ],
            dtype=np.float32,
        )

    def advance(self):

        self.phase += (
            TARGET_OMEGA * DT
        )


# ============================================================
# 5. DRONE DYNAMICS
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


def drone_dynamics(state, action):

    action = np.clip(
        action,
        -1.0,
        1.0,
    )

    velocity = state[2:4]

    acceleration = (
        MAX_ACCEL * 0.78 * action
        - 0.13 * velocity
        + np.array(
            [3.0, -2.0],
            dtype=np.float32,
        )
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
# 6. DRONE ENVIRONMENT
# ============================================================

class DroneEnv:

    def __init__(self, seed=None):

        self.rng = np.random.default_rng(
            seed
        )

        self.target = MovingTarget(
            [700.0, 310.0]
        )

        self.state = None
        self.steps = 0

        self.reset()

    def observation(self):

        position = self.state[:2]
        velocity = self.state[2:4]

        target_position = (
            self.target.position()
        )

        target_velocity = (
            self.target.velocity()
        )

        return np.concatenate(
            [
                (target_position - position) / 400.0,
                target_velocity / 200.0,
                velocity / MAX_SPEED,
                (OBSTACLE - position) / 400.0,
                [
                    math.sin(self.target.phase),
                    math.cos(self.target.phase),
                ],
            ]
        ).astype(np.float32)

    def reset(self):

        self.state = np.array(
            [
                self.rng.uniform(80, 140),
                self.rng.uniform(440, 540),
                0.0,
                0.0,
            ],
            dtype=np.float32,
        )

        self.target.phase = self.rng.uniform(
            0.0,
            2.0 * math.pi,
        )

        self.steps = 0

        return self.observation()

    def step(self, action):

        action = np.clip(
            np.asarray(
                action,
                dtype=np.float32,
            ),
            -1.0,
            1.0,
        )

        old_position = (
            self.state[:2].copy()
        )

        future_phase = (
            self.target.phase
            + TARGET_OMEGA * DT
        )

        future_target = (
            self.target.position(
                future_phase
            )
        )

        distance_before = np.linalg.norm(
            old_position - future_target
        )

        self.state = drone_dynamics(
            self.state,
            action,
        )

        self.target.advance()

        distance_after = np.linalg.norm(
            self.state[:2]
            - self.target.position()
        )

        obstacle_distance = np.linalg.norm(
            self.state[:2] - OBSTACLE
        )

        boundary_distance = min(
            self.state[0],
            WORLD_W - self.state[0],
            self.state[1],
            WORLD_H - self.state[1],
        )

        collision = (
            obstacle_distance < OBSTACLE_RADIUS
        )

        reward = (
            0.08 * (
                distance_before - distance_after
            )
            - 0.010 * distance_after
            - 0.10 * np.sum(action ** 2)
        )

        reward -= (
            0.006
            * max(
                0.0,
                SAFE_RADIUS - obstacle_distance,
            ) ** 2
        )

        reward -= (
            0.015
            * max(
                0.0,
                55.0 - boundary_distance,
            ) ** 2
        )

        if collision:

            reward -= 100.0

        self.steps += 1

        terminated = bool(
            collision
        )

        truncated = (
            self.steps >= MAX_EPISODE_STEPS
        )

        return (
            self.observation(),
            float(reward),
            terminated,
            truncated,
            {
                "distance": float(distance_after),
                "collision": collision,
            },
        )


# ============================================================
# 7. REINFORCE TRAINING
# ============================================================

def compute_returns(rewards):

    returns = []

    running_return = 0.0

    for reward in reversed(rewards):

        running_return = (
            reward + GAMMA * running_return
        )

        returns.append(
            running_return
        )

    returns.reverse()

    return np.asarray(
        returns,
        dtype=np.float32,
    )


def train():

    env = DroneEnv(
        seed=SEED + 1
    )

    policy = GaussianPolicy().to(
        DEVICE
    )

    optimizer = optim.Adam(
        policy.parameters(),
        lr=LEARNING_RATE,
    )

    history = {
        "return": [],
        "distance": [],
        "entropy": [],
        "std": [],
        "loss": [],
    }

    for episode_index in range(
        TRAIN_EPISODES
    ):

        observation = env.reset()

        observations = []
        raw_actions = []
        rewards = []
        log_probs = []
        entropies = []
        distances = []

        for _ in range(
            MAX_EPISODE_STEPS
        ):

            state_tensor = torch.as_tensor(
                observation,
                dtype=torch.float32,
                device=DEVICE,
            ).unsqueeze(0)

            action, raw_action, log_prob = (
                policy.sample(
                    state_tensor
                )
            )

            mean, log_std = policy(
                state_tensor
            )

            entropy = gaussian_entropy(
                log_std
            )

            action_np = (
                action.squeeze(0)
                .detach()
                .cpu()
                .numpy()
            )

            (
                next_observation,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(action_np)

            observations.append(
                observation.copy()
            )

            raw_actions.append(
                raw_action.squeeze(0)
                .detach()
                .cpu()
                .numpy()
            )

            rewards.append(reward)

            log_probs.append(
                log_prob.squeeze(0)
            )

            entropies.append(
                entropy.squeeze(0)
            )

            distances.append(
                info["distance"]
            )

            observation = next_observation

            if terminated or truncated:

                break

        returns = compute_returns(
            rewards
        )

        returns_tensor = torch.as_tensor(
            returns,
            dtype=torch.float32,
            device=DEVICE,
        )

        # Normalize returns to reduce gradient scale variance.
        # This is an episode-level REINFORCE baseline-like
        # normalization, not a learned state-value critic.

        if len(returns_tensor) > 1:

            normalized_returns = (
                returns_tensor - returns_tensor.mean()
            ) / (
                returns_tensor.std(unbiased=False) + 1e-8
            )

        else:

            normalized_returns = returns_tensor

        log_prob_tensor = torch.stack(
            log_probs
        )

        entropy_tensor = torch.stack(
            entropies
        )

        policy_loss = -(
            log_prob_tensor
            * normalized_returns.detach()
        ).mean()

        entropy_bonus = (
            entropy_tensor.mean()
        )

        loss = (
            policy_loss
            - ENTROPY_COEFFICIENT * entropy_bonus
        )

        optimizer.zero_grad()

        loss.backward()

        nn.utils.clip_grad_norm_(
            policy.parameters(),
            0.5,
        )

        optimizer.step()

        episode_return = float(
            np.sum(rewards)
        )

        history["return"].append(
            episode_return
        )

        history["distance"].append(
            float(np.mean(distances))
        )

        history["entropy"].append(
            float(entropy_bonus.detach().item())
        )

        history["std"].append(
            float(
                torch.exp(
                    policy.log_std.detach()
                ).mean().item()
            )
        )

        history["loss"].append(
            float(loss.detach().item())
        )

        if (
            episode_index + 1
        ) % 10 == 0:

            recent = slice(-10, None)

            print(
                f"Episode {episode_index + 1:4d}/{TRAIN_EPISODES}"
                f" | Return={np.mean(history['return'][recent]):8.2f}"
                f" | Distance={np.mean(history['distance'][recent]):7.2f}"
                f" | Std={history['std'][-1]:.3f}"
                f" | Entropy={history['entropy'][-1]:.3f}"
            )

    return policy, history


# ============================================================
# 8. VISUALIZATION
# ============================================================

def moving_average(values, window=20):

    values = np.asarray(
        values,
        dtype=np.float32,
    )

    if len(values) < window:

        return values

    kernel = np.ones(
        window,
        dtype=np.float32,
    ) / window

    return np.convolve(
        values,
        kernel,
        mode="valid",
    )


def plot_history(history):

    plots = [
        ("return", "Episode Return"),
        ("distance", "Mean Target Distance"),
        ("entropy", "Gaussian Entropy"),
        ("std", "Action Standard Deviation"),
        ("loss", "Policy Loss"),
    ]

    for key, title in plots:

        plt.figure(
            figsize=(10, 4)
        )

        plt.plot(
            history[key],
            alpha=0.3,
            label="Raw",
        )

        plt.plot(
            moving_average(history[key]),
            label="Moving average",
        )

        plt.title(title)

        plt.xlabel("Episode")

        plt.grid(True)

        plt.legend()

        plt.tight_layout()

        plt.show()


def plot_action_distribution():

    mean = torch.tensor(
        [0.0, 0.0]
    )

    log_std = torch.tensor(
        [-0.2, -0.7]
    )

    raw = torch.linspace(
        -3.0,
        3.0,
        500,
    )

    raw_prob = torch.exp(
        gaussian_log_prob(
            torch.stack(
                [raw, torch.zeros_like(raw)],
                dim=1,
            ),
            mean,
            log_std,
        )[:, 0]
    )

    squashed = torch.tanh(raw)

    plt.figure(
        figsize=(10, 4)
    )

    plt.plot(
        raw.numpy(),
        raw_prob.numpy(),
        label="Gaussian density (first action)",
    )

    plt.title(
        "Gaussian Action Distribution"
    )

    plt.xlabel("Raw action")

    plt.ylabel("Probability density")

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    plt.show()

    plt.figure(
        figsize=(10, 4)
    )

    plt.plot(
        raw.numpy(),
        squashed.numpy(),
        label="tanh transformation",
    )

    plt.axhline(
        1.0,
        linestyle="--",
    )

    plt.axhline(
        -1.0,
        linestyle="--",
    )

    plt.title(
        "Tanh Squashing Function"
    )

    plt.xlabel("Raw Gaussian action")

    plt.ylabel("Bounded action")

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    plt.show()


# ============================================================
# 9. PYGAME QUADCOPTER VISUALIZATION
# ============================================================

def run_pygame(policy):

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
        "Phase 5.10 — Continuous-Action Drone Policy"
    )

    clock = pygame.time.Clock()

    font = pygame.font.SysFont(
        "consolas",
        15,
    )

    env = DroneEnv(
        seed=SEED + 100
    )

    observation = env.reset()

    paused = False
    speed = 1

    trail = []

    total_reward = 0.0

    def clamp_center(position):

        return np.array(
            [
                np.clip(
                    position[0],
                    TARGET_A + 25,
                    WORLD_W - TARGET_A - 25,
                ),

                np.clip(
                    position[1],
                    TARGET_B + 25,
                    WORLD_H - TARGET_B - 25,
                ),
            ],
            dtype=np.float32,
        )

    def draw_text(message, x, y):

        screen.blit(
            font.render(
                message,
                True,
                (240, 240, 240),
            ),
            (x, y),
        )

    def draw_drone():

        center = env.state[:2]

        velocity = env.state[2:4]

        magnitude = np.linalg.norm(
            velocity
        )

        if magnitude > 1e-5:

            direction = velocity / magnitude

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

                rotor = (
                    center
                    + direction * forward * 17
                    + side * lateral * 17
                )

                pygame.draw.line(
                    screen,
                    (230, 235, 245),
                    center.astype(int),
                    rotor.astype(int),
                    5,
                )

                pygame.draw.circle(
                    screen,
                    (80, 190, 245),
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
            (255, 185, 55),
            (
                center + direction * 22
            ).astype(int),
            4,
        )

        pygame.draw.line(
            screen,
            (90, 160, 255),
            center.astype(int),
            (
                center + velocity * 0.4
            ).astype(int),
            3,
        )

    def draw_target():

        points = []

        for index in range(100):

            phase = (
                env.target.phase
                + 2 * math.pi * index / 100
            )

            points.append(
                env.target.position(
                    phase
                ).astype(int)
            )

        pygame.draw.lines(
            screen,
            (50, 100, 70),
            True,
            points,
            2,
        )

        position = env.target.position()

        velocity = env.target.velocity()

        pygame.draw.circle(
            screen,
            (70, 245, 105),
            position.astype(int),
            11,
        )

        pygame.draw.line(
            screen,
            (70, 255, 115),
            position.astype(int),
            (
                position + velocity * 0.18
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

                    observation = env.reset()

                    trail.clear()

                    total_reward = 0.0

                elif event.key in (
                    pygame.K_PLUS,
                    pygame.K_EQUALS,
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

                    env.target.center = clamp_center(
                        np.asarray(
                            event.pos,
                            dtype=np.float32,
                        )
                    )

        if not paused:

            for _ in range(speed):

                state_tensor = torch.as_tensor(
                    observation,
                    dtype=torch.float32,
                    device=DEVICE,
                ).unsqueeze(0)

                with torch.no_grad():

                    action = (
                        policy.deterministic_action(
                            state_tensor
                        )
                    )

                action = (
                    action.squeeze(0)
                    .cpu()
                    .numpy()
                )

                (
                    observation,
                    reward,
                    terminated,
                    truncated,
                    info,
                ) = env.step(action)

                total_reward += reward

                trail.append(
                    env.state[:2].copy()
                )

                if len(trail) > 250:

                    trail.pop(0)

                if terminated or truncated:

                    observation = env.reset()

                    trail.clear()

                    total_reward = 0.0

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
            (105, 105, 115),
            OBSTACLE.astype(int),
            OBSTACLE_RADIUS,
        )

        pygame.draw.circle(
            screen,
            (190, 95, 95),
            OBSTACLE.astype(int),
            SAFE_RADIUS,
            1,
        )

        draw_target()

        if len(trail) > 1:

            pygame.draw.lines(
                screen,
                (75, 135, 235),
                False,
                [
                    point.astype(int)
                    for point in trail
                ],
                2,
            )

        draw_drone()

        draw_text(
            "PHASE 5.10 — CONTINUOUS-ACTION POLICY",
            15,
            12,
        )

        draw_text(
            f"Target distance: {info['distance'] if 'info' in locals() else 0:.2f}",
            15,
            40,
        )

        draw_text(
            f"Episode reward: {total_reward:.2f}",
            15,
            65,
        )

        draw_text(
            f"Action std: {torch.exp(policy.log_std).mean().item():.3f}",
            15,
            90,
        )

        draw_text(
            f"Simulation speed: {speed}x",
            15,
            115,
        )

        draw_text(
            "Click: change moving-target trajectory center",
            15,
            WORLD_H - 100,
        )

        draw_text(
            "SPACE pause | R reset | +/- speed | ESC exit",
            15,
            WORLD_H - 70,
        )

        if paused:

            draw_text(
                "PAUSED",
                WORLD_W - 100,
                15,
            )

        pygame.display.flip()

    pygame.quit()


# ============================================================
# 10. NUMERICAL TESTS
# ============================================================

def run_tests():

    print(
        "\nRunning continuous-action policy tests..."
    )

    passed = 0

    def check(condition, description):

        nonlocal passed

        assert condition, description

        passed += 1

        print(
            "[PASS]",
            description,
        )

    # Test 1: Gaussian log probability.

    mean = torch.zeros(
        (4, 2)
    )

    log_std = torch.zeros(
        (4, 2)
    )

    actions = torch.zeros(
        (4, 2)
    )

    log_prob = gaussian_log_prob(
        actions,
        mean,
        log_std,
    )

    expected = -0.5 * math.log(
        2.0 * math.pi
    )

    check(
        torch.allclose(
            log_prob,
            torch.full(
                (4, 2),
                expected,
            ),
            atol=1e-6,
        ),
        "Standard Gaussian log probability",
    )

    # Test 2: Gaussian entropy.

    entropy = gaussian_entropy(
        torch.zeros(
            (4, 2)
        )
    )

    expected_entropy = (
        math.log(2.0 * math.pi * math.e)
    )

    check(
        torch.allclose(
            entropy,
            torch.full(
                (4,),
                expected_entropy,
            ),
            atol=1e-6,
        ),
        "Standard Gaussian entropy",
    )

    # Test 3: Sampling shape.

    sample = gaussian_sample(
        mean,
        log_std,
    )

    check(
        sample.shape == (4, 2),
        "Gaussian sample dimensions",
    )

    # Test 4: Tanh bounds.

    bounded = tanh_squash(
        torch.randn(
            1000,
            2,
        ) * 5
    )

    check(
        bool(
            torch.all(bounded.abs() <= 1.0)
        ),
        "Tanh actions stay within [-1, 1]",
    )

    # Test 5: Tanh Jacobian at zero.

    raw = torch.zeros(
        5,
        2,
    )

    jacobian = tanh_log_jacobian(
        raw
    )

    check(
        torch.allclose(
            jacobian,
            torch.zeros_like(jacobian),
            atol=1e-6,
        ),
        "Tanh log Jacobian at zero",
    )

    # Test 6: Squashed log probability is finite.

    test_mean = torch.zeros(
        8,
        2,
    )

    test_log_std = torch.zeros(
        8,
        2,
    )

    test_raw = torch.randn(
        8,
        2,
    )

    squashed_log_prob = (
        squashed_gaussian_log_prob(
            test_raw,
            test_mean,
            test_log_std,
        )
    )

    check(
        bool(
            torch.isfinite(
                squashed_log_prob
            ).all()
        ),
        "Squashed Gaussian log probability is finite",
    )

    # Test 7: Squashed policy sample.

    action, raw_action, log_prob = (
        squashed_gaussian_sample(
            test_mean,
            test_log_std,
        )
    )

    check(
        action.shape == (8, 2)
        and raw_action.shape == (8, 2)
        and log_prob.shape == (8,),
        "Squashed Gaussian sample dimensions",
    )

    # Test 8: Squashed actions bounded.

    check(
        bool(
            torch.all(action.abs() <= 1.0)
        ),
        "Squashed Gaussian actions are bounded",
    )

    # Test 9: Numerical stability for large raw actions.

    extreme_raw = torch.tensor(
        [
            [20.0, -20.0],
            [15.0, -15.0],
        ]
    )

    stable_jacobian = tanh_log_jacobian(
        extreme_raw
    )

    check(
        bool(
            torch.isfinite(
                stable_jacobian
            ).all()
        ),
        "Stable tanh Jacobian for extreme actions",
    )

    # Test 10: Policy forward pass.

    policy = GaussianPolicy().to(
        DEVICE
    )

    states = torch.randn(
        16,
        OBS_DIM,
        device=DEVICE,
    )

    mean, log_std = policy(
        states
    )

    check(
        mean.shape == (16, ACTION_DIM)
        and log_std.shape == (16, ACTION_DIM),
        "Policy output dimensions",
    )

    # Test 11: Policy gradients.

    actions, raw_actions, log_probs = (
        policy.sample(
            states
        )
    )

    loss = -log_probs.mean()

    policy.zero_grad()

    loss.backward()

    gradients_finite = all(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        for parameter in policy.parameters()
    )

    check(
        gradients_finite,
        "Policy gradients are finite",
    )

    # Test 12: Action deterministic output.

    deterministic = policy.deterministic_action(
        states
    )

    check(
        deterministic.shape == (16, ACTION_DIM)
        and bool(
            torch.all(
                deterministic.abs() <= 1.0
            )
        ),
        "Deterministic policy output is bounded",
    )

    # Test 13: Environment observation.

    env = DroneEnv()

    observation = env.reset()

    check(
        observation.shape == (OBS_DIM,),
        "Drone observation dimensions",
    )

    # Test 14: Environment transition.

    next_obs, reward, terminated, truncated, info = (
        env.step(
            np.zeros(
                ACTION_DIM,
                dtype=np.float32,
            )
        )
    )

    check(
        next_obs.shape == (OBS_DIM,)
        and np.isfinite(reward),
        "Drone environment transition",
    )

    # Test 15: Target velocity.

    target = MovingTarget(
        [400.0, 300.0]
    )

    check(
        target.velocity().shape == (2,),
        "Moving target velocity dimensions",
    )

    print(
        f"\n{passed}/15 tests passed."
    )

    return passed == 15


# ============================================================
# 11. MAIN
# ============================================================

def main():

    print("=" * 72)

    print(
        "PHASE 5.10 — CONTINUOUS-ACTION POLICIES"
    )

    print("=" * 72)

    print(
        "Device:",
        DEVICE,
    )

    if not run_tests():

        raise RuntimeError(
            "Continuous-action policy tests failed."
        )

    print(
        "\nVisualizing Gaussian and tanh distributions..."
    )

    plot_action_distribution()

    print(
        "\nTraining continuous-action REINFORCE..."
    )

    policy, history = train()

    print(
        "\nPlotting training history..."
    )

    plot_history(
        history
    )

    print(
        "\nStarting interactive quadcopter..."
    )

    run_pygame(
        policy
    )

    print(
        "\nPhase 5.10 completed."
    )


if __name__ == "__main__":

    main()
