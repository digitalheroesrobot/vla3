
"""
File: phase5/reinforce_drone.py

PHASE 5.1 — REINFORCE: MONTE CARLO POLICY GRADIENT

============================================================
PURPOSE
============================================================

Train a quadcopter to track a continuously moving target
using a stochastic neural-network policy.

Unlike DQN:
    The network does not output Q-values.

Instead:
    The network outputs a probability distribution
    over continuous drone-control actions.

============================================================
WHAT YOU LEARN
============================================================

STEP 1 — Stochastic Policy
    Represent the policy as a Gaussian distribution.

STEP 2 — Policy Sampling
    Sample continuous acceleration commands.

STEP 3 — Episode Collection
    Collect complete episodes using the current policy.

STEP 4 — Monte Carlo Returns
    Calculate discounted returns from each time step.

STEP 5 — REINFORCE Gradient
    Increase the probability of actions associated
    with higher returns.

STEP 6 — Policy Optimization
    Update policy parameters using gradient ascent,
    implemented as gradient descent on a negative objective.

STEP 7 — Numerical Verification
    Verify return calculations, probability shapes,
    action bounds and policy-gradient calculations.

STEP 8 — Interactive Robotics
    Visualize the trained drone tracking a moving target.

============================================================
MATHEMATICS
============================================================

Stochastic policy:

    pi_theta(a|s) = Normal(mu_theta(s), sigma_theta(s))

Discounted return:

    G_t = r_t + gamma*r_(t+1) + ... + gamma^(T-t-1)*r_(T-1)

Policy-gradient objective:

    J(theta) = E_pi[ G_0 ]

REINFORCE gradient estimator:

    grad J(theta) =
        E_pi[
            sum_t G_t * grad_theta log pi_theta(a_t|s_t)
        ]

Loss minimized by gradient descent:

    L(theta) =
        -sum_t G_t * log pi_theta(a_t|s_t)

IMPORTANT:
    Vanilla REINFORCE uses complete episode returns.
    It does not bootstrap from a value function.
    Its gradient estimator can have high variance.

Continuous actions:

    u ~ Normal(mu, std)

    a = tanh(u)

The tanh transformation bounds actions to [-1, 1].

The log probability must include the transformation
correction:

    log pi(a|s) =
        log Normal(u; mu, std)
        - sum log(1 - tanh(u)^2)

============================================================
ROBOTICS APPLICATION
============================================================

- Continuous drone steering
- Autonomous moving-target tracking
- Robot navigation
- Direct stochastic policy learning
- Foundation for Actor-Critic and PPO

This is a simplified 2D educational simulation,
not a real-flight controller.
"""

from __future__ import annotations

import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt


# ============================================================
# 0. CONFIGURATION
# ============================================================

SEED = 531

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

WORLD_W = 900
WORLD_H = 650

DT = 0.08

DRONE_RADIUS = 12.0
MAX_SPEED = 150.0
MAX_ACCEL = 125.0

OBSTACLE = np.array(
    [450.0, 330.0],
    dtype=np.float32,
)

OBSTACLE_RADIUS = 62.0
SAFE_RADIUS = 112.0

TARGET_A = 145.0
TARGET_B = 95.0
TARGET_OMEGA = 0.55

OBS_DIM = 10
ACTION_DIM = 2

GAMMA = 0.99

LEARNING_RATE = 1e-4

TRAIN_EPISODES = 300
MAX_EPISODE_STEPS = 180

PRINT_INTERVAL = 10

# Reduce these for a quick experiment.
#
# TRAIN_EPISODES = 20
# MAX_EPISODE_STEPS = 100


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

        self.phase += TARGET_OMEGA * DT


# ============================================================
# 2. DRONE PHYSICS
# ============================================================

def clip_physical_state(state):

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


def drone_dynamics(
    state,
    action,
):

    """
    Simplified 2D quadcopter dynamics.

    state = [x, y, vx, vy]
    action = [ax_command, ay_command]

    The action is normalized to [-1, 1].
    """

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
        0.78 * MAX_ACCEL * action
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

    return clip_physical_state(
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
        ).astype(
            np.float32
        )

    def reset(
        self,
        target_center=None,
    ):

        self.state = np.array(
            [
                105.0,
                500.0,
                0.0,
                0.0,
            ],
            dtype=np.float32,
        )

        if target_center is not None:

            self.target.center = np.asarray(
                target_center,
                dtype=np.float32,
            )

        self.target.phase = np.random.uniform(
            0.0,
            2.0 * math.pi,
        )

        self.steps = 0

        return self.observation()

    def step(self, action):

        old_position = (
            self.state[:2].copy()
        )

        # Compare both positions against the same
        # future target position. This avoids
        # confusing target movement with drone progress.

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
            - 0.10 * np.sum(
                np.asarray(action) ** 2
            )
        )

        # Soft obstacle penalty.
        reward -= (
            0.006
            * max(
                0.0,
                SAFE_RADIUS - obstacle_distance,
            ) ** 2
        )

        # Soft boundary penalty.
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

        info = {
            "distance": float(distance_after),
            "collision": collision,
        }

        return (
            self.observation(),
            float(reward),
            terminated,
            truncated,
            info,
        )


# ============================================================
# 4. STOCHASTIC POLICY NETWORK
# ============================================================

class GaussianPolicy(nn.Module):

    def __init__(self):

        super().__init__()

        self.body = nn.Sequential(
            nn.Linear(OBS_DIM, 128),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.ReLU(),
        )

        self.mean_layer = nn.Linear(
            128,
            ACTION_DIM,
        )

        self.log_std_layer = nn.Linear(
            128,
            ACTION_DIM,
        )

    def forward(self, observation):

        hidden = self.body(
            observation
        )

        mean = self.mean_layer(
            hidden
        )

        log_std = self.log_std_layer(
            hidden
        )

        log_std = torch.clamp(
            log_std,
            -5.0,
            1.0,
        )

        return mean, log_std

    def sample(self, observation):

        mean, log_std = self(
            observation
        )

        std = log_std.exp()

        distribution = torch.distributions.Normal(
            mean,
            std,
        )

        # REINFORCE uses an ordinary stochastic sample.
        # Do not use rsample() here: the gradient should
        # flow through log probability, not through
        # a pathwise action derivative.

        raw_action = distribution.sample()

        action = torch.tanh(
            raw_action
        )

        log_probability = distribution.log_prob(
            raw_action
        )

        # Correct probability for the tanh transformation.
        log_probability -= torch.log(
            1.0 - action.pow(2) + 1e-6
        )

        log_probability = log_probability.sum(
            dim=-1
        )

        entropy_estimate = -log_probability

        return (
            action,
            log_probability,
            entropy_estimate,
        )

    def deterministic_action(
        self,
        observation,
    ):

        mean, _ = self(
            observation
        )

        return torch.tanh(
            mean
        )


# ============================================================
# 5. MONTE CARLO RETURN CALCULATION
# ============================================================

def discounted_returns(
    rewards,
    gamma=GAMMA,
):

    """
    Calculate:

        G_t = r_t + gamma*r_(t+1) + ...

    The calculation is performed backwards:

        G_t = r_t + gamma*G_(t+1)

    Vanilla REINFORCE uses complete episode returns.
    There is no value-function bootstrap here.
    """

    returns = []

    running_return = 0.0

    for reward in reversed(
        rewards
    ):

        running_return = (
            reward + gamma * running_return
        )

        returns.append(
            running_return
        )

    returns.reverse()

    return np.asarray(
        returns,
        dtype=np.float32,
    )


# ============================================================
# 6. REINFORCE AGENT
# ============================================================

class REINFORCEAgent:

    def __init__(self):

        self.policy = GaussianPolicy().to(
            DEVICE
        )

        self.optimizer = optim.Adam(
            self.policy.parameters(),
            lr=LEARNING_RATE,
        )

        self.training_updates = 0

    def select_action(
        self,
        observation,
        deterministic=False,
    ):

        state = torch.as_tensor(
            observation,
            dtype=torch.float32,
            device=DEVICE,
        ).unsqueeze(0)

        with torch.no_grad():

            if deterministic:

                action = (
                    self.policy.deterministic_action(
                        state
                    )
                )

            else:

                action, _, _ = (
                    self.policy.sample(
                        state
                    )
                )

        return action.squeeze(0).cpu().numpy()

    def update_episode(
        self,
        observations,
        actions,
        rewards,
    ):

        """
        Vanilla REINFORCE update.

        L(theta) = -sum_t G_t * log pi_theta(a_t|s_t)

        Gradient descent on L is equivalent to
        gradient ascent on the expected return.
        """

        returns = discounted_returns(
            rewards
        )

        states = torch.as_tensor(
            np.asarray(observations),
            dtype=torch.float32,
            device=DEVICE,
        )

        actions = torch.as_tensor(
            np.asarray(actions),
            dtype=torch.float32,
            device=DEVICE,
        )

        returns_tensor = torch.as_tensor(
            returns,
            dtype=torch.float32,
            device=DEVICE,
        )

        mean, log_std = self.policy(
            states
        )

        std = log_std.exp()

        distribution = torch.distributions.Normal(
            mean,
            std,
        )

        # The stored actions are already tanh-transformed.
        # Reconstruct the corresponding pre-tanh action.

        clipped_actions = torch.clamp(
            actions,
            -0.999999,
            0.999999,
        )

        raw_actions = torch.atanh(
            clipped_actions
        )

        log_probability = distribution.log_prob(
            raw_actions
        )

        log_probability -= torch.log(
            1.0 - actions.pow(2) + 1e-6
        )

        log_probability = log_probability.sum(
            dim=-1
        )

        loss = -(
            returns_tensor.detach()
            * log_probability
        ).mean()

        self.optimizer.zero_grad()

        loss.backward()

        nn.utils.clip_grad_norm_(
            self.policy.parameters(),
            5.0,
        )

        self.optimizer.step()

        self.training_updates += 1

        return {
            "loss": float(
                loss.detach().cpu()
            ),

            "mean_return": float(
                returns.mean()
            ),

            "return_std": float(
                returns.std()
            ),

            "episode_length": len(
                rewards
            ),
        }


# ============================================================
# 7. COLLECT ONE COMPLETE EPISODE
# ============================================================

def collect_episode(
    env,
    agent,
):

    observation = env.reset()

    observations = []
    actions = []
    rewards = []

    distances = []

    total_reward = 0.0

    collision = False

    for _ in range(
        MAX_EPISODE_STEPS
    ):

        action = agent.select_action(
            observation,
            deterministic=False,
        )

        next_observation, reward, terminated, truncated, info = (
            env.step(
                action
            )
        )

        observations.append(
            observation.copy()
        )

        actions.append(
            action.copy()
        )

        rewards.append(
            reward
        )

        distances.append(
            info["distance"]
        )

        total_reward += reward

        collision = (
            collision or info["collision"]
        )

        observation = next_observation

        if terminated or truncated:

            break

    return {
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "distances": distances,
        "total_reward": total_reward,
        "mean_distance": float(
            np.mean(distances)
        ),
        "collision": collision,
    }


# ============================================================
# 8. TRAINING
# ============================================================

def train_reinforce():

    agent = REINFORCEAgent()

    env = DroneEnv()

    episode_returns = []

    episode_distances = []

    policy_losses = []

    collisions = []

    for episode_index in range(
        TRAIN_EPISODES
    ):

        trajectory = collect_episode(
            env,
            agent,
        )

        update_info = agent.update_episode(
            trajectory["observations"],
            trajectory["actions"],
            trajectory["rewards"],
        )

        episode_returns.append(
            trajectory["total_reward"]
        )

        episode_distances.append(
            trajectory["mean_distance"]
        )

        policy_losses.append(
            update_info["loss"]
        )

        collisions.append(
            float(
                trajectory["collision"]
            )
        )

        if (
            episode_index + 1
        ) % PRINT_INTERVAL == 0:

            recent_returns = episode_returns[
                -PRINT_INTERVAL:
            ]

            recent_distances = episode_distances[
                -PRINT_INTERVAL:
            ]

            recent_collisions = collisions[
                -PRINT_INTERVAL:
            ]

            print(
                f"Episode {episode_index + 1:4d}/{TRAIN_EPISODES}"
                f" | Return={np.mean(recent_returns):9.2f}"
                f" | Distance={np.mean(recent_distances):8.2f}"
                f" | Collision={np.mean(recent_collisions):.2f}"
                f" | Policy loss={update_info['loss']:9.3f}"
            )

    history = {
        "returns": episode_returns,
        "distances": episode_distances,
        "losses": policy_losses,
        "collisions": collisions,
    }

    return agent, history


# ============================================================
# 9. VISUALIZATION — TRAINING
# ============================================================

def moving_average(
    values,
    window=15,
):

    values = np.asarray(
        values,
        dtype=np.float32,
    )

    if len(values) < window:

        return values

    return np.convolve(
        values,
        np.ones(window) / window,
        mode="valid",
    )


def plot_training(history):

    plt.figure(
        figsize=(10, 5)
    )

    plt.plot(
        history["returns"],
        alpha=0.35,
        label="Episode return",
    )

    smoothed = moving_average(
        history["returns"]
    )

    if len(smoothed):

        plt.plot(
            np.arange(
                len(smoothed)
            ) + 14,
            smoothed,
            label="Moving average",
        )

    plt.xlabel(
        "Episode"
    )

    plt.ylabel(
        "Return"
    )

    plt.title(
        "REINFORCE — Episode Return"
    )

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    plt.show()

    plt.figure(
        figsize=(10, 5)
    )

    plt.plot(
        history["distances"],
        alpha=0.35,
        label="Mean distance",
    )

    smoothed = moving_average(
        history["distances"]
    )

    if len(smoothed):

        plt.plot(
            np.arange(
                len(smoothed)
            ) + 14,
            smoothed,
            label="Moving average",
        )

    plt.xlabel(
        "Episode"
    )

    plt.ylabel(
        "Distance",
    )

    plt.title(
        "Moving Target Tracking Performance"
    )

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    plt.show()

    plt.figure(
        figsize=(10, 5)
    )

    plt.plot(
        history["losses"],
        label="Policy loss",
    )

    plt.xlabel(
        "Episode"
    )

    plt.ylabel(
        "Loss"
    )

    plt.title(
        "REINFORCE Policy Loss"
    )

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    plt.show()

    plt.figure(
        figsize=(10, 5)
    )

    plt.plot(
        history["collisions"],
        label="Collision indicator",
    )

    plt.xlabel(
        "Episode"
    )

    plt.ylabel(
        "Collision"
    )

    plt.title(
        "Collision History"
    )

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    plt.show()


# ============================================================
# 10. EVALUATION
# ============================================================

def evaluate_policy(
    agent,
    episodes=5,
):

    env = DroneEnv()

    returns = []

    distances = []

    collision_rates = []

    for _ in range(
        episodes
    ):

        trajectory = collect_episode(
            env,
            agent,
        )

        # Evaluation must not update the policy.
        # collect_episode samples stochastically, so
        # use a separate deterministic evaluation loop.

        observation = env.reset()

        total_reward = 0.0

        episode_distances = []

        collision = False

        for _ in range(
            MAX_EPISODE_STEPS
        ):

            action = agent.select_action(
                observation,
                deterministic=True,
            )

            (
                observation,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(
                action
            )

            total_reward += reward

            episode_distances.append(
                info["distance"]
            )

            collision = (
                collision or info["collision"]
            )

            if terminated or truncated:

                break

        returns.append(
            total_reward
        )

        distances.append(
            np.mean(episode_distances)
        )

        collision_rates.append(
            float(collision)
        )

    print(
        "\nDeterministic policy evaluation"
    )

    print(
        f"Mean return:       {np.mean(returns):.2f}"
    )

    print(
        f"Mean target distance: {np.mean(distances):.2f}"
    )

    print(
        f"Collision rate:    {np.mean(collision_rates):.2f}"
    )

    return {
        "returns": returns,
        "distances": distances,
        "collisions": collision_rates,
    }


# ============================================================
# 11. PYGAME DRONE VISUALIZATION
# ============================================================

def run_pygame(agent):

    try:

        import pygame

    except ImportError:

        print(
            "Pygame is not installed."
        )

        print(
            "Install with: pip install pygame"
        )

        return

    pygame.init()

    screen = pygame.display.set_mode(
        (WORLD_W, WORLD_H)
    )

    pygame.display.set_caption(
        "Phase 5.1 — REINFORCE Drone"
    )

    clock = pygame.time.Clock()

    font = pygame.font.SysFont(
        "consolas",
        15,
    )

    env = DroneEnv()

    observation = env.reset()

    paused = False

    speed = 1

    trail = []

    total_reward = 0.0

    last_distance = 0.0

    def clamp_target_center(position):

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

    def draw_text(
        text,
        x,
        y,
        color=(235, 235, 235),
    ):

        surface = font.render(
            text,
            True,
            color,
        )

        screen.blit(
            surface,
            (x, y),
        )

    def draw_target():

        path = []

        for index in range(100):

            phase = (
                env.target.phase
                + 2.0 * math.pi * index / 100
            )

            path.append(
                env.target.position(
                    phase
                ).astype(int)
            )

        pygame.draw.lines(
            screen,
            (55, 105, 75),
            True,
            path,
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
            (65, 240, 105),
            target_position.astype(int),
            11,
        )

        pygame.draw.line(
            screen,
            (65, 255, 115),
            target_position.astype(int),
            (
                target_position
                + target_velocity * 0.18
            ).astype(int),
            3,
        )

    def draw_drone():

        center = env.state[:2]

        velocity = env.state[2:4]

        speed_value = np.linalg.norm(
            velocity
        )

        if speed_value > 1e-5:

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

                rotor = (
                    center
                    + direction * forward * 17
                    + side * lateral * 17
                )

                pygame.draw.line(
                    screen,
                    (225, 230, 240),
                    center.astype(int),
                    rotor.astype(int),
                    5,
                )

                pygame.draw.circle(
                    screen,
                    (95, 195, 245),
                    rotor.astype(int),
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

        pygame.draw.line(
            screen,
            (90, 165, 255),
            center.astype(int),
            (
                center + velocity * 0.4
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

                    center = clamp_target_center(
                        np.array(
                            event.pos,
                            dtype=np.float32,
                        )
                    )

                    # Preserve the drone state, but
                    # change the moving target center.
                    env.target.center = center

        if not paused:

            for _ in range(speed):

                action = agent.select_action(
                    observation,
                    deterministic=True,
                )

                (
                    observation,
                    reward,
                    terminated,
                    truncated,
                    info,
                ) = env.step(
                    action
                )

                total_reward += reward

                last_distance = (
                    info["distance"]
                )

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
            (100, 100, 110),
            OBSTACLE.astype(int),
            OBSTACLE_RADIUS,
        )

        pygame.draw.circle(
            screen,
            (185, 95, 95),
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
            "PHASE 5.1 — REINFORCE",
            15,
            12,
        )

        draw_text(
            f"Target distance: {last_distance:.2f}",
            15,
            40,
        )

        draw_text(
            f"Episode reward: {total_reward:.2f}",
            15,
            65,
        )

        draw_text(
            f"Simulation speed: {speed}x",
            15,
            90,
        )

        draw_text(
            "Click: change target trajectory center",
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
                (255, 215, 80),
            )

        pygame.display.flip()

    pygame.quit()


# ============================================================
# 12. NUMERICAL VERIFICATION
# ============================================================

def run_tests():

    print(
        "\nRunning numerical verification..."
    )

    passed = 0

    def check(
        condition,
        description,
    ):

        nonlocal passed

        assert condition, description

        passed += 1

        print(
            f"[PASS] {description}"
        )

    # Test 1 — Discounted returns.
    returns = discounted_returns(
        [1.0, 2.0, 3.0],
        gamma=0.5,
    )

    expected = np.array(
        [
            2.75,
            3.5,
            3.0,
        ]
    )

    check(
        np.allclose(
            returns,
            expected,
        ),
        "Discounted returns are correct",
    )

    # Test 2 — Single-step return.
    check(
        np.allclose(
            discounted_returns(
                [5.0],
                gamma=0.9,
            ),
            [5.0],
        ),
        "Single-step return equals immediate reward",
    )

    # Test 3 — Zero rewards.
    check(
        np.allclose(
            discounted_returns(
                [0.0, 0.0, 0.0]
            ),
            [0.0, 0.0, 0.0],
        ),
        "Zero rewards produce zero returns",
    )

    # Test 4 — Return length.
    check(
        len(
            discounted_returns(
                [1.0, 2.0, 3.0, 4.0]
            )
        ) == 4,
        "Return sequence preserves episode length",
    )

    # Test 5 — Observation dimension.
    env = DroneEnv()

    observation = env.reset()

    check(
        observation.shape == (OBS_DIM,),
        "Observation dimension is correct",
    )

    # Test 6 — Observation finite.
    check(
        np.all(
            np.isfinite(observation)
        ),
        "Observation values are finite",
    )

    # Test 7 — Target movement.
    target = MovingTarget(
        [700.0, 300.0]
    )

    p1 = target.position().copy()

    target.advance()

    p2 = target.position()

    check(
        not np.allclose(
            p1,
            p2,
        ),
        "Moving target changes position",
    )

    # Test 8 — Target velocity.
    check(
        target.velocity().shape == (2,),
        "Target velocity dimension is correct",
    )

    # Test 9 — Drone dynamics.
    next_state = drone_dynamics(
        np.array(
            [100.0, 100.0, 0.0, 0.0]
        ),
        np.array(
            [1.0, 0.0]
        ),
    )

    check(
        next_state.shape == (4,),
        "Drone dynamics return four physical states",
    )

    # Test 10 — Speed clipping.
    fast = clip_physical_state(
        np.array(
            [100.0, 100.0, 500.0, 500.0]
        )
    )

    check(
        np.linalg.norm(
            fast[2:4]
        ) <= MAX_SPEED + 1e-5,
        "Drone speed is bounded",
    )

    # Test 11 — Actor output dimensions.
    policy = GaussianPolicy().to(
        DEVICE
    )

    state = torch.as_tensor(
        observation,
        dtype=torch.float32,
        device=DEVICE,
    ).unsqueeze(0)

    action, log_prob, entropy = (
        policy.sample(
            state
        )
    )

    check(
        action.shape == (1, ACTION_DIM),
        "Policy action dimension is correct",
    )

    # Test 12 — Bounded action.
    check(
        bool(
            torch.all(
                action.abs() <= 1.0
            )
        ),
        "Tanh policy action is bounded",
    )

    # Test 13 — Finite log probability.
    check(
        bool(
            torch.isfinite(
                log_prob
            ).all()
        ),
        "Policy log probability is finite",
    )

    # Test 14 — Finite entropy estimate.
    check(
        bool(
            torch.isfinite(
                entropy
            ).all()
        ),
        "Policy entropy estimate is finite",
    )

    # Test 15 — Deterministic action.
    deterministic = (
        policy.deterministic_action(
            state
        )
    )

    check(
        deterministic.shape == (
            1,
            ACTION_DIM,
        ),
        "Deterministic action has correct shape",
    )

    # Test 16 — Environment output.
    result = env.step(
        np.zeros(
            ACTION_DIM
        )
    )

    check(
        len(result) == 5,
        "Environment returns five step values",
    )

    # Test 17 — Reward finite.
    check(
        np.isfinite(
            result[1]
        ),
        "Environment reward is finite",
    )

    # Test 18 — Policy gradient exists.
    test_policy = GaussianPolicy().to(
        DEVICE
    )

    test_optimizer = optim.Adam(
        test_policy.parameters(),
        lr=1e-4,
    )

    test_state = torch.zeros(
        (4, OBS_DIM),
        device=DEVICE,
    )

    test_actions = torch.zeros(
        (4, ACTION_DIM),
        device=DEVICE,
    )

    test_returns = torch.tensor(
        [1.0, 2.0, 3.0, 4.0],
        device=DEVICE,
    )

    mean, log_std = test_policy(
        test_state
    )

    distribution = torch.distributions.Normal(
        mean,
        log_std.exp(),
    )

    raw_actions = torch.atanh(
        torch.clamp(
            test_actions,
            -0.999,
            0.999,
        )
    )

    log_probability = distribution.log_prob(
        raw_actions
    ).sum(
        dim=-1
    )

    loss = -(
        test_returns * log_probability
    ).mean()

    test_optimizer.zero_grad()

    loss.backward()

    has_gradient = any(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        for parameter in test_policy.parameters()
    )

    check(
        has_gradient,
        "REINFORCE loss produces policy gradients",
    )

    # Test 19 — Policy update.
    old_parameters = [
        parameter.detach().clone()
        for parameter in test_policy.parameters()
    ]

    test_optimizer.step()

    changed = any(
        not torch.allclose(
            old,
            new.detach(),
        )
        for old, new in zip(
            old_parameters,
            test_policy.parameters(),
        )
    )

    check(
        changed,
        "Gradient descent updates policy parameters",
    )

    # Test 20 — Reward calculation.
    check(
        isinstance(
            result[1],
            float,
        ),
        "Environment reward is a scalar float",
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
        "=" * 70
    )

    print(
        "PHASE 5.1 — REINFORCE POLICY GRADIENT"
    )

    print(
        "=" * 70
    )

    print(
        f"Device: {DEVICE}"
    )

    if not run_tests():

        raise RuntimeError(
            "Numerical verification failed."
        )

    print(
        "\n[1/4] Training REINFORCE..."
    )

    agent, history = train_reinforce()

    print(
        "\n[2/4] Evaluating deterministic policy..."
    )

    evaluate_policy(
        agent,
        episodes=5,
    )

    print(
        "\n[3/4] Visualizing training..."
    )

    plot_training(
        history
    )

    print(
        "\n[4/4] Starting interactive drone simulation..."
    )

    run_pygame(
        agent
    )

    print(
        "\nPhase 5.1 completed."
    )


if __name__ == "__main__":

    main()
