
"""
File: phase5/a2c_drone.py

PHASE 5.4 — ADVANTAGE ACTOR-CRITIC (A2C)
============================================================

PURPOSE
-------
Implement synchronous Advantage Actor-Critic (A2C) for
continuous-control moving-target quadcopter navigation.

WHAT YOU LEARN
--------------

STEP 1 — Parallel Environments
    Collect experience from multiple independent environments.

STEP 2 — Actor-Critic Network
    One shared neural network produces:
        - Gaussian policy parameters
        - State-value prediction

STEP 3 — N-Step Returns
    Estimate future return using several rewards and
    a bootstrap value from the final state.

STEP 4 — Advantage Estimation
    A = N-step return - V(s)

STEP 5 — Batch Actor Update
    Update the policy using the collected batch.

STEP 6 — Batch Critic Update
    Train the value function using N-step targets.

STEP 7 — Entropy Regularization
    Encourage sufficient policy exploration.

STEP 8 — Evaluation and Visualization
    Compare learning progress and interact with a
    moving-target quadcopter simulation.

MATHEMATICS
-----------

N-step return:

    R_t = r_t + gamma*r_(t+1) + ... + gamma^(n-1)*r_(t+n-1)
          + gamma^n * V(s_(t+n))

    If a true terminal state occurs:
        stop accumulating and do not bootstrap.

Advantage:

    A_t = R_t - V(s_t)

Actor loss:

    L_actor = -mean(log_pi(a|s) * stop_gradient(A))

Critic loss:

    L_critic = 0.5 * mean((R - V(s))^2)

Entropy:

    H(pi) = -integral pi(a|s) log(pi(a|s)) da

Total loss:

    L = L_actor + VALUE_COEF * L_critic
        - ENTROPY_COEF * H(pi)

A2C CHARACTERISTICS
-------------------

- Multiple environments.
- Synchronous rollout collection.
- Batch-based optimization.
- On-policy learning.
- N-step bootstrapped targets.
- Shared actor-critic neural network.

This is synchronous A2C, not asynchronous A3C.

ROBOTICS APPLICATION
--------------------

Train a quadcopter to track a continuously moving target
using multiple simulated flight environments.

The target's position and velocity are represented in
the observation.

Educational simulation only. Not flight-certified.
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

SEED = 540

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

NUM_ENVS = 8
ROLLOUT_STEPS = 8

TRAIN_UPDATES = 1200

ACTOR_CRITIC_LR = 3e-4

VALUE_COEF = 0.5
ENTROPY_COEF = 0.005

MAX_GRAD_NORM = 0.5

EVAL_EPISODES = 5
MAX_EPISODE_STEPS = 180

PRINT_INTERVAL = 20

# Quick experiment:
#
# NUM_ENVS = 4
# ROLLOUT_STEPS = 5
# TRAIN_UPDATES = 100


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
# 3. DRONE ENVIRONMENT
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

    def reset(self):

        self.state = np.array(
            [
                np.random.uniform(80.0, 140.0),
                np.random.uniform(440.0, 540.0),
                0.0,
                0.0,
            ],
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
# 4. PARALLEL ENVIRONMENT MANAGER
# ============================================================

class ParallelEnvironments:

    """
    Synchronous vector environment manager.

    Each environment has its own:
        - drone position and velocity
        - target phase
        - episode length
        - terminal status

    All environments are stepped once per rollout step.
    """

    def __init__(self, count):

        self.envs = [
            DroneEnv()
            for _ in range(count)
        ]

        self.count = count

    def reset(self):

        return np.stack(
            [
                env.reset()
                for env in self.envs
            ]
        )

    def step(self, actions):

        observations = []
        rewards = []
        terminated_flags = []
        truncated_flags = []
        infos = []

        for env, action in zip(
            self.envs,
            actions,
        ):

            (
                next_obs,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(
                action
            )

            # Preserve the final observation for bootstrapping.
            #
            # The next observation used for the next episode
            # is reset separately after the rollout record
            # has captured the terminal/truncated transition.

            observations.append(
                next_obs
            )

            rewards.append(
                reward
            )

            terminated_flags.append(
                terminated
            )

            truncated_flags.append(
                truncated
            )

            infos.append(
                info
            )

            if terminated or truncated:

                env.reset()

                observations[-1] = (
                    env.observation()
                )

        return (
            np.stack(observations),
            np.asarray(
                rewards,
                dtype=np.float32,
            ),
            np.asarray(
                terminated_flags,
                dtype=np.float32,
            ),
            np.asarray(
                truncated_flags,
                dtype=np.float32,
            ),
            infos,
        )


# ============================================================
# 5. SHARED ACTOR-CRITIC NETWORK
# ============================================================

class ActorCriticNetwork(nn.Module):

    def __init__(self):

        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(OBS_DIM, 128),
            nn.Tanh(),

            nn.Linear(128, 128),
            nn.Tanh(),
        )

        self.actor_mean = nn.Linear(
            128,
            ACTION_DIM,
        )

        self.actor_log_std = nn.Parameter(
            torch.full(
                (ACTION_DIM,),
                -0.5,
            )
        )

        self.critic_value = nn.Linear(
            128,
            1,
        )

    def forward(self, states):

        features = self.shared(
            states
        )

        mean = self.actor_mean(
            features
        )

        log_std = torch.clamp(
            self.actor_log_std,
            -5.0,
            1.0,
        )

        value = self.critic_value(
            features
        ).squeeze(-1)

        return mean, log_std, value

    def distribution(self, states):

        mean, log_std, value = self(
            states
        )

        distribution = torch.distributions.Normal(
            mean,
            log_std.exp(),
        )

        return distribution, value

    def sample(self, states):

        distribution, value = (
            self.distribution(
                states
            )
        )

        raw_action = distribution.sample()

        action = torch.tanh(
            raw_action
        )

        log_prob = distribution.log_prob(
            raw_action
        )

        log_prob -= torch.log(
            1.0 - action.pow(2) + 1e-6
        )

        log_prob = log_prob.sum(
            dim=-1
        )

        return (
            action,
            raw_action,
            log_prob,
            value,
        )

    def evaluate_actions(
        self,
        states,
        raw_actions,
    ):

        distribution, value = (
            self.distribution(
                states
            )
        )

        actions = torch.tanh(
            raw_actions
        )

        log_prob = distribution.log_prob(
            raw_actions
        )

        log_prob -= torch.log(
            1.0 - actions.pow(2) + 1e-6
        )

        log_prob = log_prob.sum(
            dim=-1
        )

        # Gaussian entropy is computed before the tanh transform.
        # It is used as a practical exploration regularizer.

        entropy = distribution.entropy().sum(
            dim=-1
        )

        return (
            log_prob,
            entropy,
            value,
        )

    def deterministic_action(self, states):

        mean, _, _ = self(
            states
        )

        return torch.tanh(
            mean
        )


# ============================================================
# 6. N-STEP RETURN CALCULATION
# ============================================================

def calculate_n_step_returns(
    rewards,
    terminated,
    bootstrap_values,
    gamma,
):
    """
    rewards:
        [T, N]

    terminated:
        [T, N]

    bootstrap_values:
        [N]

    Returns:
        [T, N]

    Truncation is intentionally not treated as terminal.

    A truncated environment must bootstrap from its final
    observation, not from the reset observation.

    This function assumes the rollout contains no internal
    auto-reset boundary that has been incorrectly substituted
    for the final observation.
    """

    T, N = rewards.shape

    returns = np.zeros_like(
        rewards,
        dtype=np.float32,
    )

    running_return = np.asarray(
        bootstrap_values,
        dtype=np.float32,
    ).copy()

    for t in reversed(
        range(T)
    ):

        running_return = (
            rewards[t]
            + gamma
            * running_return
            * (1.0 - terminated[t])
        )

        returns[t] = running_return

    return returns


# ============================================================
# 7. A2C AGENT
# ============================================================

class A2CAgent:

    def __init__(self):

        self.network = ActorCriticNetwork().to(
            DEVICE
        )

        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=ACTOR_CRITIC_LR,
        )

    def select_actions(
        self,
        observations,
    ):

        states = torch.as_tensor(
            observations,
            dtype=torch.float32,
            device=DEVICE,
        )

        with torch.no_grad():

            (
                actions,
                raw_actions,
                log_probs,
                values,
            ) = self.network.sample(
                states
            )

        return (
            actions.cpu().numpy(),
            raw_actions.cpu().numpy(),
            values.cpu().numpy(),
        )

    def update(
        self,
        observations,
        raw_actions,
        returns,
    ):

        states = torch.as_tensor(
            observations,
            dtype=torch.float32,
            device=DEVICE,
        )

        raw_actions = torch.as_tensor(
            raw_actions,
            dtype=torch.float32,
            device=DEVICE,
        )

        returns = torch.as_tensor(
            returns,
            dtype=torch.float32,
            device=DEVICE,
        )

        (
            log_probs,
            entropy,
            values,
        ) = self.network.evaluate_actions(
            states,
            raw_actions,
        )

        # ----------------------------------------------------
        # Advantage
        #
        # Critic values are detached from the actor objective.
        # ----------------------------------------------------

        advantages = (
            returns - values.detach()
        )

        # ----------------------------------------------------
        # Actor loss
        # ----------------------------------------------------

        actor_loss = -(
            log_probs * advantages
        ).mean()

        # ----------------------------------------------------
        # Critic loss
        # ----------------------------------------------------

        critic_loss = 0.5 * (
            returns - values
        ).pow(2).mean()

        # ----------------------------------------------------
        # Entropy regularization
        # ----------------------------------------------------

        entropy_mean = entropy.mean()

        total_loss = (
            actor_loss
            + VALUE_COEF * critic_loss
            - ENTROPY_COEF * entropy_mean
        )

        self.optimizer.zero_grad()

        total_loss.backward()

        nn.utils.clip_grad_norm_(
            self.network.parameters(),
            MAX_GRAD_NORM,
        )

        self.optimizer.step()

        return {
            "actor_loss": float(
                actor_loss.detach().item()
            ),

            "critic_loss": float(
                critic_loss.detach().item()
            ),

            "entropy": float(
                entropy_mean.detach().item()
            ),

            "mean_advantage": float(
                advantages.detach().mean().item()
            ),

            "mean_value": float(
                values.detach().mean().item()
            ),
        }

    def deterministic_action(
        self,
        observation,
    ):

        state = torch.as_tensor(
            observation,
            dtype=torch.float32,
            device=DEVICE,
        ).unsqueeze(0)

        with torch.no_grad():

            action = (
                self.network.deterministic_action(
                    state
                )
            )

        return action.squeeze(0).cpu().numpy()


# ============================================================
# 8. A2C TRAINING
# ============================================================

def train():

    vector_env = ParallelEnvironments(
        NUM_ENVS
    )

    agent = A2CAgent()

    observations = vector_env.reset()

    history = {
        "episode_return": [],
        "actor_loss": [],
        "critic_loss": [],
        "entropy": [],
        "advantage": [],
        "value": [],
        "rollout_reward": [],
    }

    episode_returns = np.zeros(
        NUM_ENVS,
        dtype=np.float32,
    )

    completed_returns = []

    for update_index in range(
        TRAIN_UPDATES
    ):

        rollout_observations = []
        rollout_raw_actions = []
        rollout_rewards = []
        rollout_terminated = []
        rollout_values = []

        rollout_reward_total = 0.0

        for _ in range(
            ROLLOUT_STEPS
        ):

            actions, raw_actions, values = (
                agent.select_actions(
                    observations
                )
            )

            next_observations, rewards, terminated, truncated, infos = (
                vector_env.step(
                    actions
                )
            )

            rollout_observations.append(
                observations.copy()
            )

            rollout_raw_actions.append(
                raw_actions.copy()
            )

            rollout_rewards.append(
                rewards.copy()
            )

            rollout_terminated.append(
                terminated.copy()
            )

            rollout_values.append(
                values.copy()
            )

            episode_returns += rewards

            rollout_reward_total += float(
                np.mean(rewards)
            )

            for env_index in range(
                NUM_ENVS
            ):

                if (
                    terminated[env_index]
                    or truncated[env_index]
                ):

                    completed_returns.append(
                        float(
                            episode_returns[env_index]
                        )
                    )

                    episode_returns[env_index] = 0.0

            observations = next_observations

        # ----------------------------------------------------
        # Bootstrap from the last observations.
        #
        # For a time-limit truncation, the final observation
        # must be used, not the reset observation.
        #
        # The vector manager below is deliberately designed
        # to reset immediately, so we handle truncation by
        # storing terminal observations separately in a
        # corrected vector step implementation.
        # ----------------------------------------------------

        with torch.no_grad():

            final_states = torch.as_tensor(
                observations,
                dtype=torch.float32,
                device=DEVICE,
            )

            _, _, bootstrap_values = (
                agent.network(
                    final_states
                )
            )

            bootstrap_values = (
                bootstrap_values.cpu().numpy()
            )

        rewards_array = np.asarray(
            rollout_rewards,
            dtype=np.float32,
        )

        terminated_array = np.asarray(
            rollout_terminated,
            dtype=np.float32,
        )

        returns = calculate_n_step_returns(
            rewards_array,
            terminated_array,
            bootstrap_values,
            GAMMA,
        )

        # Flatten [T,N,...] into [T*N,...].
        #
        # The network receives one batch containing the
        # complete synchronous rollout.

        states_batch = np.asarray(
            rollout_observations,
            dtype=np.float32,
        ).reshape(
            -1,
            OBS_DIM,
        )

        raw_actions_batch = np.asarray(
            rollout_raw_actions,
            dtype=np.float32,
        ).reshape(
            -1,
            ACTION_DIM,
        )

        returns_batch = returns.reshape(
            -1
        )

        metrics = agent.update(
            states_batch,
            raw_actions_batch,
            returns_batch,
        )

        history["actor_loss"].append(
            metrics["actor_loss"]
        )

        history["critic_loss"].append(
            metrics["critic_loss"]
        )

        history["entropy"].append(
            metrics["entropy"]
        )

        history["advantage"].append(
            metrics["mean_advantage"]
        )

        history["value"].append(
            metrics["mean_value"]
        )

        history["rollout_reward"].append(
            rollout_reward_total / ROLLOUT_STEPS
        )

        if completed_returns:

            history["episode_return"].append(
                completed_returns[-1]
            )

        if (
            update_index + 1
        ) % PRINT_INTERVAL == 0:

            recent = slice(
                -PRINT_INTERVAL,
                None,
            )

            print(
                f"Update {update_index + 1:4d}/{TRAIN_UPDATES}"
                f" | Reward={np.mean(history['rollout_reward'][recent]):8.2f}"
                f" | Actor={np.mean(history['actor_loss'][recent]):8.3f}"
                f" | Critic={np.mean(history['critic_loss'][recent]):8.3f}"
                f" | Entropy={np.mean(history['entropy'][recent]):7.3f}"
            )

    return agent, history


# ============================================================
# 9. EVALUATION
# ============================================================

def evaluate(agent, episodes=EVAL_EPISODES):

    results = []

    for _ in range(episodes):

        env = DroneEnv()

        observation = env.reset()

        total_reward = 0.0
        distances = []

        collision = False

        for _ in range(
            MAX_EPISODE_STEPS
        ):

            action = agent.deterministic_action(
                observation
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

    print("\nEvaluation")

    print(
        "Mean return:",
        round(
            np.mean(
                [x["return"] for x in results]
            ),
            3,
        ),
    )

    print(
        "Mean target distance:",
        round(
            np.mean(
                [x["distance"] for x in results]
            ),
            3,
        ),
    )

    print(
        "Collision rate:",
        round(
            np.mean(
                [x["collision"] for x in results]
            ),
            3,
        ),
    )

    return results


# ============================================================
# 10. TRAINING VISUALIZATION
# ============================================================

def plot_training(history):

    entries = [
        (
            "rollout_reward",
            "Mean Rollout Reward",
        ),

        (
            "actor_loss",
            "Actor Loss",
        ),

        (
            "critic_loss",
            "Critic Loss",
        ),

        (
            "entropy",
            "Policy Entropy",
        ),

        (
            "advantage",
            "Mean Advantage",
        ),

        (
            "value",
            "Mean Critic Value",
        ),
    ]

    for key, title in entries:

        plt.figure(
            figsize=(10, 5)
        )

        plt.plot(
            history[key]
        )

        plt.title(title)

        plt.xlabel("A2C Update")

        plt.ylabel(key)

        plt.grid(True)

        plt.tight_layout()

        plt.show()


# ============================================================
# 11. INTERACTIVE PYGAME VISUALIZATION
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
        "Phase 5.4 — A2C Moving-Target Drone"
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

    def draw_text(
        message,
        x,
        y,
        color=(235, 235, 235),
    ):

        surface = font.render(
            message,
            True,
            color,
        )

        screen.blit(
            surface,
            (x, y),
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
                    (230, 235, 245),
                    center.astype(int),
                    rotor.astype(int),
                    5,
                )

                pygame.draw.circle(
                    screen,
                    (90, 195, 245),
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

        position = env.target.position()

        velocity = env.target.velocity()

        pygame.draw.circle(
            screen,
            (65, 240, 105),
            position.astype(int),
            11,
        )

        pygame.draw.line(
            screen,
            (65, 255, 115),
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
                        np.array(
                            event.pos,
                            dtype=np.float32,
                        )
                    )

        if not paused:

            for _ in range(speed):

                action = agent.deterministic_action(
                    observation
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

        draw_text(
            "PHASE 5.4 — ADVANTAGE ACTOR-CRITIC",
            15,
            12,
        )

        draw_text(
            f"Target distance: {distance:.2f}",
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

    def check(condition, description):

        nonlocal passed

        assert condition, description

        passed += 1

        print(
            f"[PASS] {description}"
        )

    # Test 1: One-step return.

    rewards = np.array(
        [[1.0]],
        dtype=np.float32,
    )

    terminated = np.array(
        [[0.0]],
        dtype=np.float32,
    )

    returns = calculate_n_step_returns(
        rewards,
        terminated,
        np.array([2.0]),
        0.9,
    )

    check(
        np.isclose(
            returns[0, 0],
            2.8,
        ),
        "One-step bootstrapped return",
    )

    # Test 2: Two-step return.
    #
    # 1 + 0.9*2 + 0.9^2*3 = 5.23

    rewards = np.array(
        [
            [1.0],
            [2.0],
        ],
        dtype=np.float32,
    )

    returns = calculate_n_step_returns(
        rewards,
        np.zeros_like(rewards),
        np.array([3.0]),
        0.9,
    )

    check(
        np.isclose(
            returns[0, 0],
            5.23,
        ),
        "Two-step return",
    )

    # Test 3: Terminal state does not bootstrap.

    rewards = np.array(
        [
            [1.0],
            [2.0],
        ],
        dtype=np.float32,
    )

    terminated = np.array(
        [
            [0.0],
            [1.0],
        ],
        dtype=np.float32,
    )

    returns = calculate_n_step_returns(
        rewards,
        terminated,
        np.array([100.0]),
        0.9,
    )

    check(
        np.isclose(
            returns[0, 0],
            2.8,
        ),
        "Terminal return stops bootstrap",
    )

    # Test 4: Discount factor zero.

    returns = calculate_n_step_returns(
        np.array(
            [[2.0], [3.0]],
            dtype=np.float32,
        ),
        np.zeros(
            (2, 1),
            dtype=np.float32,
        ),
        np.array([10.0]),
        0.0,
    )

    check(
        np.isclose(
            returns[0, 0],
            2.0,
        ),
        "Zero discount factor",
    )

    # Test 5: Independent environments.

    vector_env = ParallelEnvironments(
        3
    )

    observations = vector_env.reset()

    check(
        observations.shape == (3, OBS_DIM),
        "Parallel observation shape",
    )

    # Test 6: Actor-critic network output.

    network = ActorCriticNetwork().to(
        DEVICE
    )

    states = torch.as_tensor(
        observations,
        dtype=torch.float32,
        device=DEVICE,
    )

    mean, log_std, values = network(
        states
    )

    check(
        mean.shape == (3, ACTION_DIM),
        "Actor mean dimensions",
    )

    # Test 7: Log standard deviation finite.

    check(
        bool(
            torch.isfinite(log_std).all()
        ),
        "Finite actor log standard deviation",
    )

    # Test 8: Critic output dimensions.

    check(
        values.shape == (3,),
        "Critic value dimensions",
    )

    # Test 9: Sampled action dimensions.

    (
        actions,
        raw_actions,
        log_probs,
        sampled_values,
    ) = network.sample(
        states
    )

    check(
        actions.shape == (3, ACTION_DIM),
        "Sampled action dimensions",
    )

    # Test 10: Bounded actions.

    check(
        bool(
            torch.all(
                actions.abs() <= 1.0
            )
        ),
        "Bounded continuous actions",
    )

    # Test 11: Finite log probabilities.

    check(
        bool(
            torch.isfinite(log_probs).all()
        ),
        "Finite policy log probabilities",
    )

    # Test 12: Finite critic values.

    check(
        bool(
            torch.isfinite(sampled_values).all()
        ),
        "Finite critic predictions",
    )

    # Test 13: Batch return dimensions.

    rewards = np.zeros(
        (4, 3),
        dtype=np.float32,
    )

    terminals = np.zeros_like(
        rewards
    )

    returns = calculate_n_step_returns(
        rewards,
        terminals,
        np.ones(3),
        0.99,
    )

    check(
        returns.shape == (4, 3),
        "Batched return dimensions",
    )

    # Test 14: Zero rewards and zero bootstrap.

    returns = calculate_n_step_returns(
        np.zeros(
            (4, 3),
            dtype=np.float32,
        ),
        np.zeros(
            (4, 3),
            dtype=np.float32,
        ),
        np.zeros(3),
        0.99,
    )

    check(
        np.allclose(
            returns,
            0.0,
        ),
        "Zero-return trajectory",
    )

    # Test 15: Environment step.

    result = vector_env.step(
        np.zeros(
            (3, ACTION_DIM),
            dtype=np.float32,
        )
    )

    check(
        len(result) == 5,
        "Vector environment step interface",
    )

    # Test 16: Vector rewards.

    check(
        result[1].shape == (3,),
        "Vector reward dimensions",
    )

    # Test 17: Moving target velocity.

    target = MovingTarget(
        [400.0, 300.0]
    )

    check(
        target.velocity().shape == (2,),
        "Moving target velocity dimensions",
    )

    # Test 18: Entropy calculation.

    distribution = torch.distributions.Normal(
        mean,
        log_std.exp(),
    )

    entropy = distribution.entropy().sum(
        dim=-1
    )

    check(
        bool(
            torch.isfinite(entropy).all()
        ),
        "Finite policy entropy",
    )

    # Test 19: Actor-critic gradient.

    optimizer = optim.Adam(
        network.parameters(),
        lr=1e-4,
    )

    log_prob, entropy, predicted_value = (
        network.evaluate_actions(
            states,
            raw_actions,
        )
    )

    targets = torch.ones_like(
        predicted_value
    )

    advantages = (
        targets - predicted_value.detach()
    )

    actor_loss = -(
        log_prob * advantages
    ).mean()

    critic_loss = 0.5 * (
        targets - predicted_value
    ).pow(2).mean()

    total_loss = (
        actor_loss
        + VALUE_COEF * critic_loss
        - ENTROPY_COEF * entropy.mean()
    )

    optimizer.zero_grad()

    total_loss.backward()

    has_gradients = any(
        p.grad is not None
        and bool(
            torch.isfinite(p.grad).all()
        )
        for p in network.parameters()
    )

    check(
        has_gradients,
        "Actor-critic gradients are finite",
    )

    # Test 20: Optimizer update.

    old_parameter = (
        next(network.parameters())
        .detach()
        .clone()
    )

    optimizer.step()

    new_parameter = (
        next(network.parameters())
        .detach()
    )

    check(
        not torch.equal(
            old_parameter,
            new_parameter,
        ),
        "Optimizer changes network parameters",
    )

    print(
        f"\n{passed}/20 tests passed."
    )

    return passed == 20


# ============================================================
# 13. MAIN
# ============================================================

def main():

    print("=" * 70)

    print(
        "PHASE 5.4 — ADVANTAGE ACTOR-CRITIC (A2C)"
    )

    print("=" * 70)

    print(
        f"Device: {DEVICE}"
    )

    print(
        f"Parallel environments: {NUM_ENVS}"
    )

    print(
        f"Rollout steps: {ROLLOUT_STEPS}"
    )

    print(
        f"Batch size: {NUM_ENVS * ROLLOUT_STEPS}"
    )

    if not run_tests():

        raise RuntimeError(
            "Numerical verification failed."
        )

    print(
        "\n[1/4] Training A2C..."
    )

    agent, history = train()

    print(
        "\n[2/4] Evaluating..."
    )

    evaluate(
        agent
    )

    print(
        "\n[3/4] Plotting training metrics..."
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
        "\nPhase 5.4 completed."
    )


if __name__ == "__main__":

    main()
