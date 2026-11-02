
"""
File: phase5/reinforce_baseline_drone.py

PHASE 5.2 — REINFORCE WITH A VALUE-FUNCTION BASELINE

============================================================
PURPOSE
============================================================

Train a quadcopter to track a moving target using policy
gradients while reducing gradient variance with a learned
state-value baseline.

Two neural networks are trained:

ACTOR:
    Learns a stochastic continuous-action policy.

CRITIC:
    Predicts the expected return from the current state.

============================================================
WHAT YOU LEARN
============================================================

STEP 1 — Gaussian Policy
    Produce continuous stochastic drone-control actions.

STEP 2 — Value Function
    Estimate the expected return V(s).

STEP 3 — Monte Carlo Returns
    Calculate actual episode returns G_t.

STEP 4 — Advantage Estimation
    A_t = G_t - V(s_t)

STEP 5 — Actor Update
    Improve the policy using detached advantages.

STEP 6 — Critic Update
    Train V(s) to predict Monte Carlo returns.

STEP 7 — Variance Analysis
    Compare raw returns with baseline-adjusted advantages.

STEP 8 — Robotics Visualization
    Track a moving target with an interactive quadcopter.

============================================================
MATHEMATICS
============================================================

Policy:

    pi_theta(a|s)

State-value function:

    V_phi(s) = E_pi[G_t | s_t=s]

Monte Carlo return:

    G_t = sum(k=0 to T-t-1) gamma^k * r_(t+k)

Advantage:

    A_t = G_t - V_phi(s_t)

Actor objective:

    J(theta) = E[log pi_theta(a_t|s_t) * A_t]

Actor loss:

    L_actor = -E[log pi_theta(a_t|s_t) * A_t]

Critic loss:

    L_critic = E[(V_phi(s_t)-G_t)^2]

IMPORTANT:
    The advantage must be detached from the critic graph
    during the actor update.

WHY DOES A BASELINE NOT BIAS THE POLICY GRADIENT?

    E[ b(s) * grad log pi(a|s) | s ]

    = b(s) * sum_a pi(a|s) * grad log pi(a|s)

    = b(s) * sum_a grad pi(a|s)

    = b(s) * grad sum_a pi(a|s)

    = b(s) * grad 1

    = 0

Therefore, subtracting a state-dependent baseline does
not change the expected policy gradient, provided the
baseline does not depend on the sampled action.

The learned baseline can reduce variance when it
predicts returns effectively.

============================================================
ROBOTICS APPLICATION
============================================================

- Moving-target tracking
- Continuous drone navigation
- Actor-Critic foundations
- Advantage estimation
- Foundation for A2C, A3C and PPO

This is an educational 2D simulation, not a flight-certified
controller.
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

SEED = 532

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

ACTOR_LR = 1e-4
CRITIC_LR = 5e-4

TRAIN_EPISODES = 300
MAX_EPISODE_STEPS = 180

PRINT_INTERVAL = 10

# Quick experiment:
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


def drone_dynamics(state, action):

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
        ).astype(np.float32)

    def reset(self, target_center=None):

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
# 4. ACTOR — STOCHASTIC POLICY
# ============================================================

class GaussianActor(nn.Module):

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

    def forward(self, state):

        hidden = self.body(
            state
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

    def sample(self, state):

        mean, log_std = self(
            state
        )

        distribution = torch.distributions.Normal(
            mean,
            log_std.exp(),
        )

        # Use an ordinary sample for the REINFORCE
        # score-function estimator.

        raw_action = distribution.sample()

        action = torch.tanh(
            raw_action
        )

        log_probability = distribution.log_prob(
            raw_action
        )

        # Change-of-variables correction for tanh.

        log_probability -= torch.log(
            1.0 - action.pow(2) + 1e-6
        )

        log_probability = log_probability.sum(
            dim=-1
        )

        return (
            action,
            raw_action,
            log_probability,
        )

    def evaluate_actions(
        self,
        states,
        raw_actions,
    ):

        mean, log_std = self(
            states
        )

        distribution = torch.distributions.Normal(
            mean,
            log_std.exp(),
        )

        actions = torch.tanh(
            raw_actions
        )

        log_probability = distribution.log_prob(
            raw_actions
        )

        log_probability -= torch.log(
            1.0 - actions.pow(2) + 1e-6
        )

        return log_probability.sum(
            dim=-1
        )

    def deterministic_action(self, state):

        mean, _ = self(
            state
        )

        return torch.tanh(
            mean
        )


# ============================================================
# 5. CRITIC — STATE-VALUE FUNCTION
# ============================================================

class ValueCritic(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(OBS_DIM, 128),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.ReLU(),

            nn.Linear(128, 1),
        )

    def forward(self, state):

        return self.network(
            state
        ).squeeze(-1)


# ============================================================
# 6. MONTE CARLO RETURNS
# ============================================================

def discounted_returns(
    rewards,
    gamma=GAMMA,
):

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
# 7. ADVANTAGE ESTIMATION
# ============================================================

def calculate_advantages(
    returns,
    values,
):

    """
    Advantage:

        A_t = G_t - V(s_t)

    Positive:
        Actual return exceeded critic prediction.

    Negative:
        Actual return was below critic prediction.

    The critic output is detached during actor training.
    """

    return (
        returns - values.detach()
    )


# ============================================================
# 8. ACTOR-CRITIC TRAINING AGENT
# ============================================================

class REINFORCEBaselineAgent:

    def __init__(self):

        self.actor = GaussianActor().to(
            DEVICE
        )

        self.critic = ValueCritic().to(
            DEVICE
        )

        self.actor_optimizer = optim.Adam(
            self.actor.parameters(),
            lr=ACTOR_LR,
        )

        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=CRITIC_LR,
        )

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
                    self.actor.deterministic_action(
                        state
                    )
                )

            else:

                action, _, _ = (
                    self.actor.sample(
                        state
                    )
                )

        return action.squeeze(0).cpu().numpy()

    def update_episode(
        self,
        observations,
        raw_actions,
        rewards,
    ):

        states = torch.as_tensor(
            np.asarray(observations),
            dtype=torch.float32,
            device=DEVICE,
        )

        raw_actions = torch.as_tensor(
            np.asarray(raw_actions),
            dtype=torch.float32,
            device=DEVICE,
        )

        returns_np = discounted_returns(
            rewards
        )

        returns = torch.as_tensor(
            returns_np,
            dtype=torch.float32,
            device=DEVICE,
        )

        # ----------------------------------------------------
        # CRITIC UPDATE
        # ----------------------------------------------------

        predicted_values = self.critic(
            states
        )

        critic_loss = torch.mean(
            (
                predicted_values - returns
            ) ** 2
        )

        self.critic_optimizer.zero_grad()

        critic_loss.backward()

        nn.utils.clip_grad_norm_(
            self.critic.parameters(),
            5.0,
        )

        self.critic_optimizer.step()

        # ----------------------------------------------------
        # ACTOR UPDATE
        # ----------------------------------------------------

        # Recalculate the baseline after the critic update.
        # Detach it so actor loss cannot update critic weights.

        with torch.no_grad():

            baseline = self.critic(
                states
            )

            advantages = (
                returns - baseline
            )

        log_probabilities = (
            self.actor.evaluate_actions(
                states,
                raw_actions,
            )
        )

        actor_loss = -(
            advantages.detach()
            * log_probabilities
        ).mean()

        self.actor_optimizer.zero_grad()

        actor_loss.backward()

        nn.utils.clip_grad_norm_(
            self.actor.parameters(),
            5.0,
        )

        self.actor_optimizer.step()

        # ----------------------------------------------------
        # DIAGNOSTICS
        # ----------------------------------------------------

        raw_return_std = (
            returns.std(unbiased=False)
        )

        advantage_std = (
            advantages.std(unbiased=False)
        )

        return {
            "actor_loss": float(
                actor_loss.detach().cpu()
            ),

            "critic_loss": float(
                critic_loss.detach().cpu()
            ),

            "mean_return": float(
                returns.mean().cpu()
            ),

            "return_std": float(
                raw_return_std.cpu()
            ),

            "advantage_mean": float(
                advantages.mean().cpu()
            ),

            "advantage_std": float(
                advantage_std.cpu()
            ),

            "mean_value": float(
                baseline.mean().cpu()
            ),
        }


# ============================================================
# 9. EPISODE COLLECTION
# ============================================================

def collect_episode(
    env,
    agent,
):

    observation = env.reset()

    observations = []
    raw_actions = []
    rewards = []

    distances = []

    total_reward = 0.0

    collision = False

    for _ in range(
        MAX_EPISODE_STEPS
    ):

        state_tensor = torch.as_tensor(
            observation,
            dtype=torch.float32,
            device=DEVICE,
        ).unsqueeze(0)

        with torch.no_grad():

            action, raw_action, _ = (
                agent.actor.sample(
                    state_tensor
                )
            )

        action_np = (
            action.squeeze(0)
            .cpu()
            .numpy()
        )

        raw_action_np = (
            raw_action.squeeze(0)
            .cpu()
            .numpy()
        )

        (
            next_observation,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(
            action_np
        )

        observations.append(
            observation.copy()
        )

        raw_actions.append(
            raw_action_np.copy()
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
        "raw_actions": raw_actions,
        "rewards": rewards,
        "distances": distances,
        "total_reward": total_reward,
        "mean_distance": float(
            np.mean(distances)
        ),
        "collision": collision,
    }


# ============================================================
# 10. TRAINING
# ============================================================

def train():

    agent = REINFORCEBaselineAgent()

    env = DroneEnv()

    history = {
        "returns": [],
        "distances": [],
        "actor_losses": [],
        "critic_losses": [],
        "return_std": [],
        "advantage_std": [],
        "collisions": [],
        "mean_values": [],
    }

    for episode in range(
        TRAIN_EPISODES
    ):

        trajectory = collect_episode(
            env,
            agent,
        )

        metrics = agent.update_episode(
            trajectory["observations"],
            trajectory["raw_actions"],
            trajectory["rewards"],
        )

        history["returns"].append(
            trajectory["total_reward"]
        )

        history["distances"].append(
            trajectory["mean_distance"]
        )

        history["actor_losses"].append(
            metrics["actor_loss"]
        )

        history["critic_losses"].append(
            metrics["critic_loss"]
        )

        history["return_std"].append(
            metrics["return_std"]
        )

        history["advantage_std"].append(
            metrics["advantage_std"]
        )

        history["collisions"].append(
            float(
                trajectory["collision"]
            )
        )

        history["mean_values"].append(
            metrics["mean_value"]
        )

        if (
            episode + 1
        ) % PRINT_INTERVAL == 0:

            recent = slice(
                -PRINT_INTERVAL,
                None,
            )

            print(
                f"Episode {episode + 1:4d}/{TRAIN_EPISODES}"
                f" | Return={np.mean(history['returns'][recent]):9.2f}"
                f" | Distance={np.mean(history['distances'][recent]):8.2f}"
                f" | Actor loss={metrics['actor_loss']:9.3f}"
                f" | Critic loss={metrics['critic_loss']:9.3f}"
                f" | Adv std={metrics['advantage_std']:8.3f}"
            )

    return agent, history


# ============================================================
# 11. DETERMINISTIC EVALUATION
# ============================================================

def evaluate_policy(
    agent,
    episodes=5,
):

    env = DroneEnv()

    results = []

    for _ in range(episodes):

        observation = env.reset()

        total_reward = 0.0

        distances = []

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

            distances.append(
                info["distance"]
            )

            collision = (
                collision or info["collision"]
            )

            if terminated or truncated:

                break

        results.append(
            {
                "return": total_reward,
                "distance": float(
                    np.mean(distances)
                ),
                "collision": float(collision),
            }
        )

    print(
        "\nDeterministic evaluation"
    )

    print(
        f"Mean return: {np.mean([x['return'] for x in results]):.2f}"
    )

    print(
        f"Mean distance: {np.mean([x['distance'] for x in results]):.2f}"
    )

    print(
        f"Collision rate: {np.mean([x['collision'] for x in results]):.2f}"
    )

    return results


# ============================================================
# 12. VISUALIZATION
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

    # Return curve.

    plt.figure(
        figsize=(10, 5)
    )

    plt.plot(
        history["returns"],
        alpha=0.3,
        label="Episode return",
    )

    smooth = moving_average(
        history["returns"]
    )

    plt.plot(
        np.arange(len(smooth)) + 14,
        smooth,
        label="Moving average",
    )

    plt.title(
        "REINFORCE with Baseline — Episode Return"
    )

    plt.xlabel(
        "Episode"
    )

    plt.ylabel(
        "Return"
    )

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    plt.show()

    # Actor loss.

    plt.figure(
        figsize=(10, 5)
    )

    plt.plot(
        history["actor_losses"],
        label="Actor loss",
    )

    plt.title(
        "Actor Policy Loss"
    )

    plt.xlabel(
        "Episode"
    )

    plt.ylabel(
        "Loss"
    )

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    plt.show()

    # Critic loss.

    plt.figure(
        figsize=(10, 5)
    )

    plt.plot(
        history["critic_losses"],
        label="Critic MSE loss",
    )

    plt.title(
        "Critic Value Prediction Loss"
    )

    plt.xlabel(
        "Episode"
    )

    plt.ylabel(
        "MSE"
    )

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    plt.show()

    # Return standard deviation vs advantage standard deviation.

    plt.figure(
        figsize=(10, 5)
    )

    plt.plot(
        history["return_std"],
        label="Return standard deviation",
    )

    plt.plot(
        history["advantage_std"],
        label="Advantage standard deviation",
    )

    plt.title(
        "Return and Advantage Variability"
    )

    plt.xlabel(
        "Episode"
    )

    plt.ylabel(
        "Standard deviation"
    )

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    plt.show()

    # Target distance.

    plt.figure(
        figsize=(10, 5)
    )

    plt.plot(
        history["distances"],
        label="Mean target distance",
    )

    plt.title(
        "Moving Target Tracking Performance"
    )

    plt.xlabel(
        "Episode"
    )

    plt.ylabel(
        "Distance"
    )

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    plt.show()

    # Collision history.

    plt.figure(
        figsize=(10, 5)
    )

    plt.plot(
        history["collisions"],
        label="Collision indicator",
    )

    plt.title(
        "Collision History"
    )

    plt.xlabel(
        "Episode"
    )

    plt.ylabel(
        "Collision"
    )

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    plt.show()


# ============================================================
# 13. PYGAME INTERACTIVE DRONE
# ============================================================

def run_pygame(agent):

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
        "Phase 5.2 — REINFORCE with Baseline"
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
    distance = 0.0

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

    def text(message, x, y, color=(235, 235, 235)):

        surface = font.render(
            message,
            True,
            color,
        )

        screen.blit(
            surface,
            (x, y),
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
            (55, 105, 75),
            True,
            points,
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

        magnitude = np.linalg.norm(
            velocity
        )

        if magnitude > 1e-5:

            direction = (
                velocity / magnitude
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

                    env.target.center = clamp_center(
                        np.array(
                            event.pos,
                            dtype=np.float32,
                        )
                    )

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

                distance = info["distance"]

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

        text(
            "PHASE 5.2 — REINFORCE WITH BASELINE",
            15,
            12,
        )

        text(
            f"Target distance: {distance:.2f}",
            15,
            40,
        )

        text(
            f"Episode reward: {total_reward:.2f}",
            15,
            65,
        )

        text(
            f"Simulation speed: {speed}x",
            15,
            90,
        )

        text(
            "Click: change moving-target trajectory center",
            15,
            WORLD_H - 100,
        )

        text(
            "SPACE pause | R reset | +/- speed | ESC exit",
            15,
            WORLD_H - 70,
        )

        if paused:

            text(
                "PAUSED",
                WORLD_W - 100,
                15,
                (255, 215, 80),
            )

        pygame.display.flip()

    pygame.quit()


# ============================================================
# 14. NUMERICAL VERIFICATION
# ============================================================

def run_tests():

    print(
        "\nRunning numerical verification..."
    )

    passed = 0

    def check(condition, description):

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

    check(
        np.allclose(
            returns,
            [2.75, 3.5, 3.0],
        ),
        "Discounted return calculation",
    )

    # Test 2 — Single reward.
    check(
        np.allclose(
            discounted_returns([5.0]),
            [5.0],
        ),
        "Single-step return",
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
        "Return length is preserved",
    )

    # Test 5 — Baseline advantage.
    test_returns = torch.tensor(
        [10.0, 8.0, 5.0]
    )

    test_values = torch.tensor(
        [7.0, 9.0, 5.0]
    )

    advantages = calculate_advantages(
        test_returns,
        test_values,
    )

    check(
        torch.allclose(
            advantages,
            torch.tensor(
                [3.0, -1.0, 0.0]
            ),
        ),
        "Advantage equals return minus baseline",
    )

    # Test 6 — Positive advantage.
    check(
        advantages[0].item() > 0,
        "Positive advantage means better-than-expected return",
    )

    # Test 7 — Negative advantage.
    check(
        advantages[1].item() < 0,
        "Negative advantage means worse-than-expected return",
    )

    # Test 8 — Zero advantage.
    check(
        advantages[2].item() == 0,
        "Zero advantage means expected return was matched",
    )

    # Test 9 — Baseline is detached.
    critic_value = torch.tensor(
        [3.0],
        requires_grad=True,
    )

    return_value = torch.tensor(
        [5.0]
    )

    advantage = calculate_advantages(
        return_value,
        critic_value,
    )

    check(
        not advantage.requires_grad,
        "Advantage is detached from critic computation graph",
    )

    # Test 10 — Observation shape.
    env = DroneEnv()

    observation = env.reset()

    check(
        observation.shape == (OBS_DIM,),
        "Observation dimension",
    )

    # Test 11 — Observation finite.
    check(
        np.all(
            np.isfinite(observation)
        ),
        "Observation is finite",
    )

    # Test 12 — Actor output.
    actor = GaussianActor().to(
        DEVICE
    )

    state = torch.as_tensor(
        observation,
        dtype=torch.float32,
        device=DEVICE,
    ).unsqueeze(0)

    action, raw_action, log_probability = (
        actor.sample(
            state
        )
    )

    check(
        action.shape == (1, ACTION_DIM),
        "Actor action dimension",
    )

    # Test 13 — Bounded actions.
    check(
        bool(
            torch.all(
                action.abs() <= 1.0
            )
        ),
        "Actor action is bounded",
    )

    # Test 14 — Raw action dimension.
    check(
        raw_action.shape == (
            1,
            ACTION_DIM,
        ),
        "Pre-tanh action dimension",
    )

    # Test 15 — Log probability finite.
    check(
        bool(
            torch.isfinite(
                log_probability
            ).all()
        ),
        "Policy log probability is finite",
    )

    # Test 16 — Critic output.
    critic = ValueCritic().to(
        DEVICE
    )

    value = critic(
        state
    )

    check(
        value.shape == (1,),
        "Critic returns one value per state",
    )

    # Test 17 — Critic value finite.
    check(
        bool(
            torch.isfinite(value).all()
        ),
        "Critic prediction is finite",
    )

    # Test 18 — Actor gradient.
    actor_optimizer = optim.Adam(
        actor.parameters(),
        lr=1e-4,
    )

    actor_loss = -log_probability.mean()

    actor_optimizer.zero_grad()

    actor_loss.backward()

    actor_has_gradient = any(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        for parameter in actor.parameters()
    )

    check(
        actor_has_gradient,
        "Actor receives finite gradients",
    )

    # Test 19 — Critic gradient.
    critic_optimizer = optim.Adam(
        critic.parameters(),
        lr=1e-4,
    )

    critic_optimizer.zero_grad()

    critic_loss = (
        value - torch.tensor(
            [5.0],
            device=DEVICE,
        )
    ).pow(2).mean()

    critic_loss.backward()

    critic_has_gradient = any(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        for parameter in critic.parameters()
    )

    check(
        critic_has_gradient,
        "Critic receives finite gradients",
    )

    # Test 20 — Environment step.
    result = env.step(
        np.zeros(
            ACTION_DIM
        )
    )

    check(
        len(result) == 5,
        "Environment returns five values",
    )

    # Test 21 — Finite reward.
    check(
        np.isfinite(result[1]),
        "Environment reward is finite",
    )

    # Test 22 — Target velocity.
    check(
        env.target.velocity().shape == (2,),
        "Moving target has two-dimensional velocity",
    )

    print(
        f"\n{passed}/22 tests passed."
    )

    return passed == 22


# ============================================================
# 15. MAIN
# ============================================================

def main():

    print(
        "=" * 70
    )

    print(
        "PHASE 5.2 — REINFORCE WITH BASELINE"
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
        "\n[1/4] Training actor and critic..."
    )

    agent, history = train()

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
        "\n[4/4] Starting interactive drone..."
    )

    run_pygame(
        agent
    )

    print(
        "\nPhase 5.2 completed."
    )


if __name__ == "__main__":

    main()
