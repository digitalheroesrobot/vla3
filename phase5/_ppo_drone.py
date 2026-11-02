
"""
File: phase5/ppo_drone.py

PHASE 5.6 — PROXIMAL POLICY OPTIMIZATION (PPO)
============================================================

PURPOSE
-------
Implement PPO from scratch using PyTorch for continuous
quadcopter navigation and moving-target tracking.

WHAT YOU LEARN
--------------

STEP 1 — Gaussian Actor-Critic
    Build a continuous stochastic policy and value network.

STEP 2 — Action Transformation
    Transform Gaussian actions through tanh and correctly
    calculate the transformed action's log probability.

STEP 3 — Rollout Collection
    Collect states, actions, rewards, values and old log
    probabilities using the current policy.

STEP 4 — GAE
    Calculate advantages while handling terminal and
    truncated transitions correctly.

STEP 5 — PPO Clipped Objective
    Calculate the probability ratio and clipped surrogate
    objective.

STEP 6 — Multiple Epochs
    Reuse rollout data for multiple shuffled minibatch
    optimization epochs.

STEP 7 — Policy Diagnostics
    Monitor approximate KL divergence, clip fraction,
    entropy, actor loss and critic loss.

STEP 8 — Robotics Visualization
    Train a simulated quadcopter to track a moving target
    while avoiding an obstacle.

MATHEMATICS
-----------

Probability ratio:

    ratio = exp(new_log_prob - old_log_prob)

Clipped surrogate:

    L_clip = mean(
        min(
            ratio * advantage,
            clip(ratio, 1-eps, 1+eps) * advantage
        )
    )

Value loss:

    L_value = 0.5 * mean((V(s)-return)^2)

Entropy:

    H = -E[log pi(a|s)]

Total minimized loss:

    loss = -L_clip
           + VALUE_COEF * L_value
           - ENTROPY_COEF * H

GAE:

    delta[t] = reward[t]
               + gamma * next_value[t] * (1-terminated[t])
               - value[t]

    A[t] = delta[t]
           + gamma * lambda
             * (1-terminated[t])
             * (1-truncated[t])
             * A[t+1]

IMPORTANT IMPLEMENTATION DETAILS
--------------------------------

1. Old log probabilities are stored during rollout collection.
2. New log probabilities are calculated during optimization.
3. PPO ratio compares the new and old policy probabilities.
4. The rollout data remains fixed throughout PPO epochs.
5. True termination suppresses value bootstrapping.
6. Time-limit truncation bootstraps from the final observation.
7. GAE recursion stops at both terminal and truncation boundaries.
8. Tanh-transformed Gaussian actions use the Jacobian correction.

ROBOTICS APPLICATION
--------------------

A simplified 2D quadcopter tracks a continuously moving
target while avoiding a circular obstacle.

This is an educational simulation, not flight-certified.
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

SEED = 560

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
GAE_LAMBDA = 0.95

CLIP_EPSILON = 0.20

VALUE_COEF = 0.5
ENTROPY_COEF = 0.005

LEARNING_RATE = 3e-4
MAX_GRAD_NORM = 0.5

NUM_ENVS = 8
ROLLOUT_STEPS = 128

PPO_EPOCHS = 6
MINIBATCH_SIZE = 128

TRAIN_UPDATES = 500

MAX_EPISODE_STEPS = 180

TARGET_KL = 0.03

PRINT_INTERVAL = 10

# Quick test:
#
# NUM_ENVS = 4
# ROLLOUT_STEPS = 32
# PPO_EPOCHS = 2
# TRAIN_UPDATES = 20


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
# 2. DRONE DYNAMICS
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

        old_position = self.state[:2].copy()

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
# 4. PARALLEL ENVIRONMENTS
# ============================================================

class ParallelEnvironments:

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

        next_observations = []
        final_observations = []

        rewards = []
        terminated_flags = []
        truncated_flags = []

        infos = []

        for env, action in zip(
            self.envs,
            actions,
        ):

            (
                final_obs,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(action)

            final_observations.append(
                final_obs.copy()
            )

            rewards.append(reward)

            terminated_flags.append(
                terminated
            )

            truncated_flags.append(
                truncated
            )

            infos.append(info)

            if terminated or truncated:

                next_obs = env.reset()

            else:

                next_obs = final_obs.copy()

            next_observations.append(
                next_obs
            )

        return (
            np.stack(next_observations),
            np.stack(final_observations),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(terminated_flags, dtype=np.float32),
            np.asarray(truncated_flags, dtype=np.float32),
            infos,
        )


# ============================================================
# 5. ACTOR-CRITIC NETWORK
# ============================================================

class ActorCritic(nn.Module):

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

        self.critic = nn.Linear(
            128,
            1,
        )

    def forward(self, states):

        features = self.shared(states)

        mean = self.actor_mean(
            features
        )

        log_std = torch.clamp(
            self.actor_log_std,
            -5.0,
            1.0,
        )

        value = self.critic(
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
            self.distribution(states)
        )

        # Sample raw Gaussian action.
        raw_action = distribution.sample()

        # Transform to the bounded action interval.
        action = torch.tanh(
            raw_action
        )

        # Change-of-variables correction:
        #
        # log pi(a) =
        # log N(u; mean,std)
        # - sum(log(1-tanh(u)^2))

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
            self.distribution(states)
        )

        action = torch.tanh(
            raw_actions
        )

        log_prob = distribution.log_prob(
            raw_actions
        )

        log_prob -= torch.log(
            1.0 - action.pow(2) + 1e-6
        )

        log_prob = log_prob.sum(
            dim=-1
        )

        # Entropy of the unsquashed Gaussian.
        # This is a common PPO approximation for
        # tanh-squashed continuous policies.

        entropy = distribution.entropy().sum(
            dim=-1
        )

        return (
            log_prob,
            entropy,
            value,
        )

    def deterministic_action(self, states):

        mean, _, _ = self(states)

        return torch.tanh(mean)


# ============================================================
# 6. GAE
# ============================================================

def compute_gae(
    rewards,
    values,
    next_values,
    terminated,
    truncated,
    gamma=GAMMA,
    gae_lambda=GAE_LAMBDA,
):

    T, N = rewards.shape

    advantages = np.zeros(
        (T, N),
        dtype=np.float32,
    )

    gae = np.zeros(
        N,
        dtype=np.float32,
    )

    for t in reversed(
        range(T)
    ):

        bootstrap_mask = (
            1.0 - terminated[t]
        )

        continuation_mask = (
            1.0
            - np.maximum(
                terminated[t],
                truncated[t],
            )
        )

        delta = (
            rewards[t]
            + gamma
            * next_values[t]
            * bootstrap_mask
            - values[t]
        )

        gae = (
            delta
            + gamma
            * gae_lambda
            * continuation_mask
            * gae
        )

        advantages[t] = gae

    returns = (
        advantages + values
    )

    return advantages, returns


# ============================================================
# 7. PPO AGENT
# ============================================================

class PPOAgent:

    def __init__(self):

        self.network = ActorCritic().to(
            DEVICE
        )

        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=LEARNING_RATE,
            eps=1e-5,
        )

    def select_actions(self, observations):

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
            ) = self.network.sample(states)

        return (
            actions.cpu().numpy(),
            raw_actions.cpu().numpy(),
            log_probs.cpu().numpy(),
            values.cpu().numpy(),
        )

    def predict_values(self, observations):

        states = torch.as_tensor(
            observations,
            dtype=torch.float32,
            device=DEVICE,
        )

        with torch.no_grad():

            _, _, values = self.network(states)

        return values.cpu().numpy()

    def deterministic_action(self, observation):

        states = torch.as_tensor(
            observation,
            dtype=torch.float32,
            device=DEVICE,
        ).unsqueeze(0)

        with torch.no_grad():

            actions = self.network.deterministic_action(
                states
            )

        return actions.squeeze(0).cpu().numpy()

    def update(
        self,
        observations,
        raw_actions,
        old_log_probs,
        advantages,
        returns,
    ):

        states = torch.as_tensor(
            observations,
            dtype=torch.float32,
            device=DEVICE,
        )

        actions = torch.as_tensor(
            raw_actions,
            dtype=torch.float32,
            device=DEVICE,
        )

        old_log_probs = torch.as_tensor(
            old_log_probs,
            dtype=torch.float32,
            device=DEVICE,
        )

        advantages = torch.as_tensor(
            advantages,
            dtype=torch.float32,
            device=DEVICE,
        )

        returns = torch.as_tensor(
            returns,
            dtype=torch.float32,
            device=DEVICE,
        )

        # Normalize advantages across the rollout.
        advantages = (
            advantages - advantages.mean()
        ) / (
            advantages.std(unbiased=False) + 1e-8
        )

        batch_size = states.shape[0]

        metrics = {
            "actor_loss": [],
            "critic_loss": [],
            "entropy": [],
            "approx_kl": [],
            "clip_fraction": [],
            "ratio_mean": [],
        }

        stop_early = False

        for epoch in range(
            PPO_EPOCHS
        ):

            indices = torch.randperm(
                batch_size,
                device=DEVICE,
            )

            for start in range(
                0,
                batch_size,
                MINIBATCH_SIZE,
            ):

                minibatch_indices = indices[
                    start:start + MINIBATCH_SIZE
                ]

                mb_states = states[
                    minibatch_indices
                ]

                mb_actions = actions[
                    minibatch_indices
                ]

                mb_old_log_probs = old_log_probs[
                    minibatch_indices
                ]

                mb_advantages = advantages[
                    minibatch_indices
                ]

                mb_returns = returns[
                    minibatch_indices
                ]

                (
                    new_log_probs,
                    entropy,
                    values,
                ) = self.network.evaluate_actions(
                    mb_states,
                    mb_actions,
                )

                # PPO probability ratio.
                log_ratio = (
                    new_log_probs
                    - mb_old_log_probs
                )

                ratio = torch.exp(
                    log_ratio
                )

                # Unclipped policy objective.
                surrogate_1 = (
                    ratio * mb_advantages
                )

                # Clipped policy objective.
                clipped_ratio = torch.clamp(
                    ratio,
                    1.0 - CLIP_EPSILON,
                    1.0 + CLIP_EPSILON,
                )

                surrogate_2 = (
                    clipped_ratio * mb_advantages
                )

                # We minimize negative PPO objective.
                actor_loss = -torch.minimum(
                    surrogate_1,
                    surrogate_2,
                ).mean()

                critic_loss = 0.5 * (
                    values - mb_returns
                ).pow(2).mean()

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

                with torch.no_grad():

                    # Approximate reverse KL estimator.
                    approx_kl = (
                        (ratio - 1.0) - log_ratio
                    ).mean()

                    clip_fraction = (
                        (
                            torch.abs(ratio - 1.0)
                            > CLIP_EPSILON
                        )
                        .float()
                        .mean()
                    )

                metrics["actor_loss"].append(
                    actor_loss.item()
                )

                metrics["critic_loss"].append(
                    critic_loss.item()
                )

                metrics["entropy"].append(
                    entropy_mean.item()
                )

                metrics["approx_kl"].append(
                    approx_kl.item()
                )

                metrics["clip_fraction"].append(
                    clip_fraction.item()
                )

                metrics["ratio_mean"].append(
                    ratio.mean().item()
                )

                # Stop additional epochs if the policy
                # moves too far from the rollout policy.
                #
                # This is an optional safeguard, not
                # a replacement for PPO clipping.

                if (
                    approx_kl.item()
                    > TARGET_KL
                ):

                    stop_early = True

                    break

            if stop_early:

                break

        return {
            key: float(np.mean(values))
            for key, values in metrics.items()
        }


# ============================================================
# 8. ROLLOUT COLLECTION
# ============================================================

def collect_rollout(
    agent,
    envs,
    observations,
):

    obs_buffer = []
    raw_action_buffer = []

    old_log_prob_buffer = []

    reward_buffer = []
    value_buffer = []
    next_value_buffer = []

    terminated_buffer = []
    truncated_buffer = []

    distance_buffer = []

    episode_returns = np.zeros(
        NUM_ENVS,
        dtype=np.float32,
    )

    completed_returns = []

    for _ in range(
        ROLLOUT_STEPS
    ):

        (
            actions,
            raw_actions,
            log_probs,
            values,
        ) = agent.select_actions(
            observations
        )

        (
            next_observations,
            final_observations,
            rewards,
            terminated,
            truncated,
            infos,
        ) = envs.step(actions)

        # Evaluate the true final transition observation,
        # not the reset observation returned for the next step.

        next_values = agent.predict_values(
            final_observations
        )

        obs_buffer.append(
            observations.copy()
        )

        raw_action_buffer.append(
            raw_actions.copy()
        )

        old_log_prob_buffer.append(
            log_probs.copy()
        )

        reward_buffer.append(
            rewards.copy()
        )

        value_buffer.append(
            values.copy()
        )

        next_value_buffer.append(
            next_values.copy()
        )

        terminated_buffer.append(
            terminated.copy()
        )

        truncated_buffer.append(
            truncated.copy()
        )

        distance_buffer.append(
            np.asarray(
                [
                    info["distance"]
                    for info in infos
                ],
                dtype=np.float32,
            )
        )

        episode_returns += rewards

        for i in range(NUM_ENVS):

            if terminated[i] or truncated[i]:

                completed_returns.append(
                    float(episode_returns[i])
                )

                episode_returns[i] = 0.0

        observations = next_observations

    rollout = {
        "observations": np.asarray(
            obs_buffer,
            dtype=np.float32,
        ),

        "raw_actions": np.asarray(
            raw_action_buffer,
            dtype=np.float32,
        ),

        "old_log_probs": np.asarray(
            old_log_prob_buffer,
            dtype=np.float32,
        ),

        "rewards": np.asarray(
            reward_buffer,
            dtype=np.float32,
        ),

        "values": np.asarray(
            value_buffer,
            dtype=np.float32,
        ),

        "next_values": np.asarray(
            next_value_buffer,
            dtype=np.float32,
        ),

        "terminated": np.asarray(
            terminated_buffer,
            dtype=np.float32,
        ),

        "truncated": np.asarray(
            truncated_buffer,
            dtype=np.float32,
        ),

        "distances": np.asarray(
            distance_buffer,
            dtype=np.float32,
        ),
    }

    return (
        rollout,
        observations,
        completed_returns,
    )


# ============================================================
# 9. PPO TRAINING
# ============================================================

def train():

    envs = ParallelEnvironments(
        NUM_ENVS
    )

    agent = PPOAgent()

    observations = envs.reset()

    history = {
        "reward": [],
        "distance": [],
        "actor_loss": [],
        "critic_loss": [],
        "entropy": [],
        "approx_kl": [],
        "clip_fraction": [],
        "ratio_mean": [],
    }

    all_episode_returns = []

    for update in range(
        TRAIN_UPDATES
    ):

        (
            rollout,
            observations,
            completed_returns,
        ) = collect_rollout(
            agent,
            envs,
            observations,
        )

        all_episode_returns.extend(
            completed_returns
        )

        advantages, returns = compute_gae(
            rollout["rewards"],
            rollout["values"],
            rollout["next_values"],
            rollout["terminated"],
            rollout["truncated"],
            GAMMA,
            GAE_LAMBDA,
        )

        # Flatten [T,N,...] into a single batch.

        states = rollout[
            "observations"
        ].reshape(
            -1,
            OBS_DIM,
        )

        raw_actions = rollout[
            "raw_actions"
        ].reshape(
            -1,
            ACTION_DIM,
        )

        old_log_probs = rollout[
            "old_log_probs"
        ].reshape(
            -1,
        )

        advantages_flat = advantages.reshape(
            -1,
        )

        returns_flat = returns.reshape(
            -1,
        )

        metrics = agent.update(
            states,
            raw_actions,
            old_log_probs,
            advantages_flat,
            returns_flat,
        )

        history["reward"].append(
            float(
                rollout["rewards"].mean()
            )
        )

        history["distance"].append(
            float(
                rollout["distances"].mean()
            )
        )

        for key in (
            "actor_loss",
            "critic_loss",
            "entropy",
            "approx_kl",
            "clip_fraction",
            "ratio_mean",
        ):

            history[key].append(
                metrics[key]
            )

        if (
            update + 1
        ) % PRINT_INTERVAL == 0:

            recent = slice(
                -PRINT_INTERVAL,
                None,
            )

            print(
                f"Update {update + 1:4d}/{TRAIN_UPDATES}"
                f" | Reward={np.mean(history['reward'][recent]):8.2f}"
                f" | Distance={np.mean(history['distance'][recent]):7.2f}"
                f" | Actor={np.mean(history['actor_loss'][recent]):8.3f}"
                f" | Critic={np.mean(history['critic_loss'][recent]):8.3f}"
                f" | KL={np.mean(history['approx_kl'][recent]):7.4f}"
                f" | Clip={np.mean(history['clip_fraction'][recent]):6.3f}"
            )

    return (
        agent,
        history,
        all_episode_returns,
    )


# ============================================================
# 10. PPO VISUALIZATION
# ============================================================

def plot_training(history):

    plots = [
        ("reward", "Mean Rollout Reward"),
        ("distance", "Mean Moving-Target Distance"),
        ("actor_loss", "PPO Actor Loss"),
        ("critic_loss", "Critic Loss"),
        ("entropy", "Policy Entropy"),
        ("approx_kl", "Approximate KL Divergence"),
        ("clip_fraction", "PPO Clip Fraction"),
        ("ratio_mean", "Mean Policy Probability Ratio"),
    ]

    for key, title in plots:

        plt.figure(
            figsize=(10, 5)
        )

        plt.plot(
            history[key]
        )

        plt.title(title)

        plt.xlabel("Training Update")

        plt.ylabel(key)

        plt.grid(True)

        plt.tight_layout()

        plt.show()


# ============================================================
# 11. EVALUATION
# ============================================================

def evaluate(agent, episodes=5):

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
            ) = env.step(action)

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
            [
                total_reward,
                float(np.mean(distances)),
                float(collision),
            ]
        )

    results = np.asarray(
        results,
        dtype=np.float32,
    )

    print("\nEvaluation")

    print(
        "Mean return:",
        round(float(results[:, 0].mean()), 3),
    )

    print(
        "Mean target distance:",
        round(float(results[:, 1].mean()), 3),
    )

    print(
        "Collision rate:",
        round(float(results[:, 2].mean()), 3),
    )

    return results


# ============================================================
# 12. INTERACTIVE PYGAME DRONE
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
        "Phase 5.6 — PPO Moving-Target Drone"
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

    def draw_text(message, x, y):

        screen.blit(
            font.render(
                message,
                True,
                (235, 235, 235),
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

        # Four arms and four propellers.

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
                ) = env.step(action)

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
            "PHASE 5.6 — PROXIMAL POLICY OPTIMIZATION",
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
            )

        pygame.display.flip()

    pygame.quit()


# ============================================================
# 13. NUMERICAL VERIFICATION
# ============================================================

def run_tests():

    print(
        "\nRunning PPO numerical verification..."
    )

    passed = 0

    def check(condition, description):

        nonlocal passed

        assert condition, description

        passed += 1

        print(
            f"[PASS] {description}"
        )

    # Test 1:
    # Probability ratio equals one when old and new
    # log probabilities are identical.

    old_log_prob = torch.tensor(
        [-1.0, -2.0]
    )

    new_log_prob = old_log_prob.clone()

    ratio = torch.exp(
        new_log_prob - old_log_prob
    )

    check(
        torch.allclose(
            ratio,
            torch.ones_like(ratio),
        ),
        "Identical policies produce ratio one",
    )

    # Test 2:
    # Clipping limits probability ratio.

    ratio = torch.tensor(
        [0.5, 1.0, 1.5]
    )

    clipped = torch.clamp(
        ratio,
        0.8,
        1.2,
    )

    check(
        torch.allclose(
            clipped,
            torch.tensor(
                [0.8, 1.0, 1.2]
            ),
        ),
        "PPO probability ratio clipping",
    )

    # Test 3:
    # Positive advantage: ratio above upper bound
    # cannot improve the clipped objective.

    advantage = torch.tensor(
        [2.0]
    )

    ratio = torch.tensor(
        [1.5]
    )

    surrogate_1 = ratio * advantage

    surrogate_2 = torch.clamp(
        ratio,
        0.8,
        1.2,
    ) * advantage

    objective = torch.minimum(
        surrogate_1,
        surrogate_2,
    )

    check(
        torch.allclose(
            objective,
            torch.tensor([2.4]),
        ),
        "Positive advantage uses conservative upper clipping",
    )

    # Test 4:
    # Negative advantage: ratio below lower bound
    # cannot improve the clipped objective.

    advantage = torch.tensor(
        [-2.0]
    )

    ratio = torch.tensor(
        [0.5]
    )

    objective = torch.minimum(
        ratio * advantage,
        torch.clamp(
            ratio,
            0.8,
            1.2,
        ) * advantage,
    )

    check(
        torch.allclose(
            objective,
            torch.tensor([-1.6]),
        ),
        "Negative advantage uses conservative lower clipping",
    )

    # Test 5:
    # Terminal transition suppresses bootstrap.

    advantages, _ = compute_gae(
        np.array([[1.0]], dtype=np.float32),
        np.array([[0.5]], dtype=np.float32),
        np.array([[100.0]], dtype=np.float32),
        np.array([[1.0]], dtype=np.float32),
        np.array([[0.0]], dtype=np.float32),
        gamma=0.9,
        gae_lambda=0.95,
    )

    check(
        np.isclose(
            advantages[0, 0],
            0.5,
        ),
        "Terminal transition suppresses bootstrap",
    )

    # Test 6:
    # Truncation bootstraps from final observation.

    advantages, _ = compute_gae(
        np.array([[1.0]], dtype=np.float32),
        np.array([[0.5]], dtype=np.float32),
        np.array([[2.0]], dtype=np.float32),
        np.array([[0.0]], dtype=np.float32),
        np.array([[1.0]], dtype=np.float32),
        gamma=0.9,
        gae_lambda=0.95,
    )

    check(
        np.isclose(
            advantages[0, 0],
            2.3,
        ),
        "Truncation preserves value bootstrap",
    )

    # Test 7:
    # GAE recursion stops at truncation.

    advantages, _ = compute_gae(
        np.array(
            [[1.0], [100.0]],
            dtype=np.float32,
        ),
        np.zeros((2, 1), dtype=np.float32),
        np.zeros((2, 1), dtype=np.float32),
        np.zeros((2, 1), dtype=np.float32),
        np.array(
            [[1.0], [0.0]],
            dtype=np.float32,
        ),
        gamma=1.0,
        gae_lambda=1.0,
    )

    check(
        np.isclose(
            advantages[0, 0],
            1.0,
        ),
        "GAE recursion stops at truncation",
    )

    # Test 8:
    # Tanh actions stay bounded.

    raw = torch.randn(
        (100, ACTION_DIM)
    )

    bounded = torch.tanh(raw)

    check(
        bool(
            torch.all(
                bounded.abs() <= 1.0
            )
        ),
        "Tanh action bounds",
    )

    # Test 9:
    # Network dimensions.

    network = ActorCritic().to(
        DEVICE
    )

    states = torch.zeros(
        (4, OBS_DIM),
        dtype=torch.float32,
        device=DEVICE,
    )

    mean, log_std, values = network(states)

    check(
        mean.shape == (4, ACTION_DIM)
        and log_std.shape == (ACTION_DIM,)
        and values.shape == (4,),
        "Actor-critic output dimensions",
    )

    # Test 10:
    # Sampled log probability is finite.

    (
        actions,
        raw_actions,
        log_probs,
        values,
    ) = network.sample(states)

    check(
        bool(
            torch.isfinite(log_probs).all()
        ),
        "Sampled log probabilities are finite",
    )

    # Test 11:
    # Stored and reevaluated log probabilities agree
    # before any network parameter update.

    reevaluated, entropy, values = (
        network.evaluate_actions(
            states,
            raw_actions,
        )
    )

    check(
        torch.allclose(
            log_probs,
            reevaluated,
            atol=1e-5,
        ),
        "Old and new log probabilities initially agree",
    )

    # Test 12:
    # Entropy is finite.

    check(
        bool(
            torch.isfinite(entropy).all()
        ),
        "Policy entropy is finite",
    )

    # Test 13:
    # Environment returns correct API.

    env = DroneEnv()

    result = env.step(
        np.zeros(
            ACTION_DIM,
            dtype=np.float32,
        )
    )

    check(
        len(result) == 5,
        "Environment step API",
    )

    # Test 14:
    # Moving target velocity has two dimensions.

    target = MovingTarget(
        [400.0, 300.0]
    )

    check(
        target.velocity().shape == (2,),
        "Moving target velocity representation",
    )

    # Test 15:
    # Parallel environment returns final observations.

    envs = ParallelEnvironments(3)

    obs = envs.reset()

    output = envs.step(
        np.zeros(
            (3, ACTION_DIM),
            dtype=np.float32,
        )
    )

    check(
        obs.shape == (3, OBS_DIM)
        and output[0].shape == (3, OBS_DIM)
        and output[1].shape == (3, OBS_DIM),
        "Parallel environment observation dimensions",
    )

    # Test 16:
    # PPO ratio is one when the policy is unchanged.

    old = torch.tensor(
        [-1.0, -0.5, -2.0]
    )

    new = old.clone()

    ratio = torch.exp(
        new - old
    )

    check(
        torch.allclose(
            ratio,
            torch.ones_like(ratio),
        ),
        "Unchanged policy ratio",
    )

    # Test 17:
    # PPO actor-critic optimizer update.

    optimizer = optim.Adam(
        network.parameters(),
        lr=1e-4,
    )

    advantages = torch.ones(
        4,
        device=DEVICE,
    )

    returns = torch.ones(
        4,
        device=DEVICE,
    )

    log_probs, entropy, values = (
        network.evaluate_actions(
            states,
            raw_actions,
        )
    )

    ratio = torch.exp(
        log_probs - log_probs.detach()
    )

    surrogate = torch.minimum(
        ratio * advantages,
        torch.clamp(
            ratio,
            0.8,
            1.2,
        ) * advantages,
    )

    loss = (
        -surrogate.mean()
        + VALUE_COEF
        * 0.5
        * (values - returns).pow(2).mean()
        - ENTROPY_COEF
        * entropy.mean()
    )

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    check(
        bool(
            torch.isfinite(loss)
        ),
        "PPO optimizer update",
    )

    # Test 18:
    # No NaN in a multi-step GAE rollout.

    rewards = np.random.randn(
        10,
        3,
    ).astype(np.float32)

    values = np.random.randn(
        10,
        3,
    ).astype(np.float32)

    next_values = np.random.randn(
        10,
        3,
    ).astype(np.float32)

    terminated = np.zeros(
        (10, 3),
        dtype=np.float32,
    )

    truncated = np.zeros(
        (10, 3),
        dtype=np.float32,
    )

    advantages, returns = compute_gae(
        rewards,
        values,
        next_values,
        terminated,
        truncated,
    )

    check(
        np.isfinite(advantages).all()
        and np.isfinite(returns).all(),
        "GAE rollout remains numerically finite",
    )

    print(
        f"\n{passed}/18 tests passed."
    )

    return passed == 18


# ============================================================
# 14. MAIN
# ============================================================

def main():

    print("=" * 72)

    print(
        "PHASE 5.6 — PROXIMAL POLICY OPTIMIZATION"
    )

    print("=" * 72)

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
        f"PPO epochs: {PPO_EPOCHS}"
    )

    print(
        f"Minibatch size: {MINIBATCH_SIZE}"
    )

    print(
        f"Clip epsilon: {CLIP_EPSILON}"
    )

    if not run_tests():

        raise RuntimeError(
            "Numerical verification failed."
        )

    print(
        "\n[1/4] Training PPO..."
    )

    agent, history, episode_returns = train()

    print(
        "\n[2/4] Evaluating PPO..."
    )

    evaluate(agent)

    print(
        "\n[3/4] Plotting training diagnostics..."
    )

    plot_training(history)

    print(
        "\n[4/4] Starting interactive quadcopter..."
    )

    run_pygame(agent)

    print(
        "\nPhase 5.6 completed."
    )


if __name__ == "__main__":

    main()
