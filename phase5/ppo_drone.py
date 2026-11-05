"""
File: phase5/ppo_drone.py

PHASE 5.7 — ADVANCED PROXIMAL POLICY OPTIMIZATION
=================================================

PURPOSE
-------
Train a quadcopter to track a moving target while avoiding
obstacles using an improved PPO implementation from scratch.

IMPROVEMENTS
------------
1. Separate Actor and Critic networks.
2. Orthogonal initialization.
3. Improved reward shaping.
4. Improved obstacle avoidance.
5. Better Gaussian exploration.
6. Larger rollout horizon.
7. Linear learning-rate annealing.
8. PPO clipped surrogate objective.
9. KL-based early stopping.
10. Correct GAE terminal/truncation handling.
11. Persistent episode-return tracking.
12. Training diagnostics.
13. Evaluation across randomized trajectories.
14. Interactive Pygame visualization.

ALGORITHM
---------
PPO + GAE + Tanh Gaussian Policy

ACTION
------
[ax, ay] in [-1, 1]

OBSERVATION
-----------
1. Relative target position (2)
2. Relative target velocity (2)
3. Drone velocity (2)
4. Relative obstacle position (2)
5. Target phase (2)

Total: 10 dimensions.

IMPORTANT
---------
Educational 2D simulation.
Not suitable for real flight without additional
control, perception and safety systems.
"""

from __future__ import annotations

import math
import random
from collections import deque

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

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ------------------------------------------------------------
# Environment
# ------------------------------------------------------------

WORLD_W = 900
WORLD_H = 650

DT = 0.08

DRONE_RADIUS = 12.0

MAX_SPEED = 180.0
MAX_ACCEL = 180.0

OBSTACLE = np.array(
    [450.0, 330.0],
    dtype=np.float32,
)

OBSTACLE_RADIUS = 58.0
SAFE_RADIUS = 90.0

TARGET_A = 145.0
TARGET_B = 95.0
TARGET_OMEGA = 0.55

MAX_EPISODE_STEPS = 300

# ------------------------------------------------------------
# Observation and action
# ------------------------------------------------------------

OBS_DIM = 10
ACTION_DIM = 2

# ------------------------------------------------------------
# PPO
# ------------------------------------------------------------

GAMMA = 0.99
GAE_LAMBDA = 0.95

CLIP_EPSILON = 0.20

VALUE_COEF = 0.5
ENTROPY_COEF = 0.003

LEARNING_RATE = 3e-4
MIN_LEARNING_RATE = 3e-5

MAX_GRAD_NORM = 0.5

NUM_ENVS = 8

ROLLOUT_STEPS = 256

PPO_EPOCHS = 8

MINIBATCH_SIZE = 256

TRAIN_UPDATES = 500

TARGET_KL = 0.025

PRINT_INTERVAL = 10

EVAL_EPISODES = 10

# ------------------------------------------------------------
# Reward parameters
# ------------------------------------------------------------

TRACKING_SCALE = 100.0

PROGRESS_COEF = 0.8
TRACKING_COEF = 1.2

ACTION_PENALTY = 0.015

OBSTACLE_PENALTY = 0.025

BOUNDARY_PENALTY = 0.02

COLLISION_PENALTY = 30.0

# ------------------------------------------------------------
# Network
# ------------------------------------------------------------

HIDDEN_DIM = 256

INITIAL_LOG_STD = -0.5


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
        MAX_ACCEL * action
        - 0.10 * velocity
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

        relative_position = (
            target_position - position
        )

        relative_velocity = (
            target_velocity - velocity
        )

        relative_obstacle = (
            OBSTACLE - position
        )

        return np.concatenate(
            [
                relative_position / 300.0,

                relative_velocity / 180.0,

                velocity / MAX_SPEED,

                relative_obstacle / 300.0,

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

        # ----------------------------------------------------
        # 1. Tracking reward
        # ----------------------------------------------------

        tracking_reward = (
            TRACKING_COEF
            * math.exp(
                -distance_after / TRACKING_SCALE
            )
        )

        # ----------------------------------------------------
        # 2. Progress reward
        # ----------------------------------------------------

        progress_reward = (
            PROGRESS_COEF
            * np.clip(
                (distance_before - distance_after) / 10.0,
                -2.0,
                2.0,
            )
        )

        # ----------------------------------------------------
        # 3. Action penalty
        # ----------------------------------------------------

        action_cost = (
            ACTION_PENALTY
            * np.sum(
                np.asarray(action) ** 2
            )
        )

        # ----------------------------------------------------
        # 4. Obstacle avoidance
        # ----------------------------------------------------

        obstacle_cost = (
            OBSTACLE_PENALTY
            * (
                max(
                    0.0,
                    SAFE_RADIUS - obstacle_distance,
                ) / SAFE_RADIUS
            ) ** 2
        )

        # ----------------------------------------------------
        # 5. Boundary avoidance
        # ----------------------------------------------------

        boundary_cost = (
            BOUNDARY_PENALTY
            * (
                max(
                    0.0,
                    55.0 - boundary_distance,
                ) / 55.0
            ) ** 2
        )

        reward = (
            tracking_reward
            + progress_reward
            - action_cost
            - obstacle_cost
            - boundary_cost
        )

        collision = (
            obstacle_distance
            < OBSTACLE_RADIUS + DRONE_RADIUS
        )

        if collision:

            reward -= COLLISION_PENALTY

        self.steps += 1

        terminated = bool(
            collision
        )

        truncated = bool(
            self.steps >= MAX_EPISODE_STEPS
        )

        info = {
            "distance": float(distance_after),

            "obstacle_distance": float(
                obstacle_distance
            ),

            "collision": collision,

            "tracking_reward": float(
                tracking_reward
            ),
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
# 5. NETWORK INITIALIZATION
# ============================================================

def initialize_linear(layer, gain):

    nn.init.orthogonal_(
        layer.weight,
        gain=gain,
    )

    nn.init.constant_(
        layer.bias,
        0.0,
    )


# ============================================================
# 6. SEPARATE ACTOR NETWORK
# ============================================================

class Actor(nn.Module):

    def __init__(self):

        super().__init__()

        self.layers = nn.Sequential(

            nn.Linear(
                OBS_DIM,
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
                HIDDEN_DIM,
            ),

            nn.Tanh(),
        )

        self.mean = nn.Linear(
            HIDDEN_DIM,
            ACTION_DIM,
        )

        self.log_std = nn.Parameter(
            torch.full(
                (ACTION_DIM,),
                INITIAL_LOG_STD,
            )
        )

        # Orthogonal initialization.

        for layer in self.layers:

            if isinstance(
                layer,
                nn.Linear,
            ):

                initialize_linear(
                    layer,
                    math.sqrt(2.0),
                )

        initialize_linear(
            self.mean,
            0.01,
        )

    def forward(self, states):

        features = self.layers(
            states
        )

        mean = self.mean(
            features
        )

        log_std = torch.clamp(
            self.log_std,
            -3.0,
            1.0,
        )

        return mean, log_std

    def distribution(self, states):

        mean, log_std = self(states)

        return torch.distributions.Normal(
            mean,
            log_std.exp(),
        )

    def sample(self, states):

        distribution = self.distribution(
            states
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
        )

    def evaluate_actions(
        self,
        states,
        raw_actions,
    ):

        distribution = self.distribution(
            states
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

        entropy = distribution.entropy().sum(
            dim=-1
        )

        return log_prob, entropy

    def deterministic_action(self, states):

        mean, _ = self(states)

        return torch.tanh(mean)


# ============================================================
# 7. SEPARATE CRITIC NETWORK
# ============================================================

class Critic(nn.Module):

    def __init__(self):

        super().__init__()

        self.layers = nn.Sequential(

            nn.Linear(
                OBS_DIM,
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
                HIDDEN_DIM,
            ),

            nn.Tanh(),

            nn.Linear(
                HIDDEN_DIM,
                1,
            ),
        )

        for layer in self.layers:

            if isinstance(
                layer,
                nn.Linear,
            ):

                initialize_linear(
                    layer,
                    math.sqrt(2.0),
                )

        initialize_linear(
            self.layers[-1],
            1.0,
        )

    def forward(self, states):

        return self.layers(
            states
        ).squeeze(-1)


# ============================================================
# 8. ACTOR-CRITIC
# ============================================================

class ActorCritic(nn.Module):

    def __init__(self):

        super().__init__()

        self.actor = Actor()

        self.critic = Critic()

    def sample(self, states):

        actions, raw_actions, log_probs = (
            self.actor.sample(states)
        )

        values = self.critic(
            states
        )

        return (
            actions,
            raw_actions,
            log_probs,
            values,
        )

    def evaluate_actions(
        self,
        states,
        raw_actions,
    ):

        log_probs, entropy = (
            self.actor.evaluate_actions(
                states,
                raw_actions,
            )
        )

        values = self.critic(
            states
        )

        return (
            log_probs,
            entropy,
            values,
        )

    def deterministic_action(self, states):

        return self.actor.deterministic_action(
            states
        )


# ============================================================
# 9. GENERALIZED ADVANTAGE ESTIMATION
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

        # True termination:
        # Do not bootstrap from next state.

        bootstrap_mask = (
            1.0 - terminated[t]
        )

        # Both termination and truncation:
        # Stop GAE recursion across episode boundaries.

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
# 10. PPO AGENT
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

            values = self.network.critic(
                states
            )

        return values.cpu().numpy()

    def deterministic_action(self, observation):

        states = torch.as_tensor(
            observation,
            dtype=torch.float32,
            device=DEVICE,
        ).unsqueeze(0)

        with torch.no_grad():

            actions = (
                self.network.deterministic_action(
                    states
                )
            )

        return actions.squeeze(0).cpu().numpy()

    def set_learning_rate(self, update):

        progress = (
            update / max(
                1,
                TRAIN_UPDATES - 1,
            )
        )

        lr = (
            LEARNING_RATE
            - progress
            * (
                LEARNING_RATE
                - MIN_LEARNING_RATE
            )
        )

        for group in self.optimizer.param_groups:

            group["lr"] = lr

        return lr

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

                mb_indices = indices[
                    start:start + MINIBATCH_SIZE
                ]

                mb_states = states[
                    mb_indices
                ]

                mb_actions = actions[
                    mb_indices
                ]

                mb_old_log_probs = old_log_probs[
                    mb_indices
                ]

                mb_advantages = advantages[
                    mb_indices
                ]

                mb_returns = returns[
                    mb_indices
                ]

                (
                    new_log_probs,
                    entropy,
                    values,
                ) = self.network.evaluate_actions(
                    mb_states,
                    mb_actions,
                )

                # ----------------------------------------
                # PPO probability ratio
                # ----------------------------------------

                log_ratio = (
                    new_log_probs
                    - mb_old_log_probs
                )

                ratio = torch.exp(
                    log_ratio
                )

                # ----------------------------------------
                # Clipped surrogate objective
                # ----------------------------------------

                surrogate_1 = (
                    ratio * mb_advantages
                )

                clipped_ratio = torch.clamp(
                    ratio,
                    1.0 - CLIP_EPSILON,
                    1.0 + CLIP_EPSILON,
                )

                surrogate_2 = (
                    clipped_ratio * mb_advantages
                )

                actor_loss = -torch.minimum(
                    surrogate_1,
                    surrogate_2,
                ).mean()

                # ----------------------------------------
                # Critic loss
                # ----------------------------------------

                critic_loss = 0.5 * (
                    values - mb_returns
                ).pow(2).mean()

                # ----------------------------------------
                # Entropy regularization
                # ----------------------------------------

                entropy_mean = entropy.mean()

                total_loss = (
                    actor_loss
                    + VALUE_COEF * critic_loss
                    - ENTROPY_COEF * entropy_mean
                )

                # ----------------------------------------
                # Optimization
                # ----------------------------------------

                self.optimizer.zero_grad(
                    set_to_none=True
                )

                total_loss.backward()

                nn.utils.clip_grad_norm_(
                    self.network.parameters(),
                    MAX_GRAD_NORM,
                )

                self.optimizer.step()

                # ----------------------------------------
                # Diagnostics
                # ----------------------------------------

                with torch.no_grad():

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

                # KL early stopping.

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
# 11. ROLLOUT COLLECTION
# ============================================================

def collect_rollout(
    agent,
    envs,
    observations,
    episode_returns,
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

        # Always evaluate the actual final observation.
        # Never use an automatically reset observation
        # for time-limit bootstrapping.

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

        # Maintain episode returns across rollouts.

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
        episode_returns,
        completed_returns,
    )


# ============================================================
# 12. TRAINING
# ============================================================

def train():

    envs = ParallelEnvironments(
        NUM_ENVS
    )

    agent = PPOAgent()

    observations = envs.reset()

    episode_returns = np.zeros(
        NUM_ENVS,
        dtype=np.float32,
    )

    completed_return_history = deque(
        maxlen=100
    )

    history = {

        "reward": [],

        "episode_return": [],

        "distance": [],

        "actor_loss": [],

        "critic_loss": [],

        "entropy": [],

        "approx_kl": [],

        "clip_fraction": [],

        "ratio_mean": [],

        "learning_rate": [],
    }

    all_episode_returns = []

    for update in range(
        TRAIN_UPDATES
    ):

        lr = agent.set_learning_rate(
            update
        )

        (
            rollout,
            observations,
            episode_returns,
            completed_returns,
        ) = collect_rollout(
            agent,
            envs,
            observations,
            episode_returns,
        )

        all_episode_returns.extend(
            completed_returns
        )

        completed_return_history.extend(
            completed_returns
        )

        advantages, returns = compute_gae(
            rollout["rewards"],
            rollout["values"],
            rollout["next_values"],
            rollout["terminated"],
            rollout["truncated"],
        )

        # Flatten rollout dimensions.

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

        history["episode_return"].append(
            float(
                np.mean(
                    completed_return_history
                )
            )
            if completed_return_history
            else np.nan
        )

        history["learning_rate"].append(
            lr
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

                f" | Reward={np.mean(history['reward'][recent]):8.3f}"

                f" | Distance={np.mean(history['distance'][recent]):7.2f}"

                f" | Episode={history['episode_return'][-1]:8.2f}"

                f" | Actor={np.mean(history['actor_loss'][recent]):8.4f}"

                f" | Critic={np.mean(history['critic_loss'][recent]):8.4f}"

                f" | Entropy={np.mean(history['entropy'][recent]):7.3f}"

                f" | KL={np.mean(history['approx_kl'][recent]):7.4f}"

                f" | Clip={np.mean(history['clip_fraction'][recent]):6.3f}"

                f" | LR={lr:.6f}"
            )

    return (
        agent,
        history,
        all_episode_returns,
    )


# ============================================================
# 13. TRAINING VISUALIZATION
# ============================================================

def moving_average(values, window=20):

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

    plots = [

        ("reward", "Mean Rollout Reward"),

        ("episode_return", "Mean Episode Return"),

        ("distance", "Mean Target Distance"),

        ("actor_loss", "Actor Loss"),

        ("critic_loss", "Critic Loss"),

        ("entropy", "Policy Entropy"),

        ("approx_kl", "Approximate KL"),

        ("clip_fraction", "Clip Fraction"),

        ("ratio_mean", "Mean Probability Ratio"),

        ("learning_rate", "Learning Rate"),
    ]

    for key, title in plots:

        plt.figure(
            figsize=(10, 5)
        )

        values = history[key]

        plt.plot(
            values,
            alpha=0.3,
            label="Raw",
        )

        if len(values) >= 20:

            smooth = moving_average(
                values,
                20,
            )

            plt.plot(
                np.arange(19, len(values)),
                smooth,
                label="Moving Average",
            )

        plt.title(title)

        plt.xlabel("Training Update")

        plt.ylabel(key)

        plt.legend()

        plt.grid(True)

        plt.tight_layout()

        plt.show()


# ============================================================
# 14. EVALUATION
# ============================================================

def evaluate(agent, episodes=EVAL_EPISODES):

    results = []

    for episode in range(episodes):

        env = DroneEnv()

        observation = env.reset()

        total_reward = 0.0

        distances = []

        collision = False

        tracking_success_steps = 0

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

            if info["distance"] < 50.0:

                tracking_success_steps += 1

            collision = (
                collision or info["collision"]
            )

            if terminated or truncated:

                break

        tracking_rate = (
            tracking_success_steps
            / max(1, len(distances))
        )

        results.append(
            [
                total_reward,

                float(np.mean(distances)),

                float(np.min(distances)),

                float(collision),

                tracking_rate,
            ]
        )

    results = np.asarray(
        results,
        dtype=np.float32,
    )

    print("\n" + "=" * 60)

    print("PPO EVALUATION")

    print("=" * 60)

    print(
        "Mean return:",
        round(float(results[:, 0].mean()), 3),
    )

    print(
        "Mean target distance:",
        round(float(results[:, 1].mean()), 3),
    )

    print(
        "Best target distance:",
        round(float(results[:, 2].mean()), 3),
    )

    print(
        "Collision rate:",
        round(float(results[:, 3].mean()), 3),
    )

    print(
        "Tracking success rate:",
        round(float(results[:, 4].mean() * 100), 2),
        "%",
    )

    print("=" * 60)

    return results


# ============================================================
# 15. SAVE MODEL
# ============================================================

def save_model(agent):

    torch.save(
        {
            "actor_critic": agent.network.state_dict(),

            "optimizer": agent.optimizer.state_dict(),

            "observation_dim": OBS_DIM,

            "action_dim": ACTION_DIM,

            "seed": SEED,
        },

        "ppo_drone_phase5_7.pth",
    )

    print(
        "\nModel saved: ppo_drone_phase5_7.pth"
    )


# ============================================================
# 16. INTERACTIVE PYGAME
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
        "Phase 5.7 — Advanced PPO Drone"
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

            direction = velocity / magnitude

        else:

            direction = np.array(
                [1.0, 0.0],
                dtype=np.float32,
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

        pygame.draw.circle(
            screen,
            (65, 240, 105),
            position.astype(int),
            11,
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

                if len(trail) > 300:

                    trail.pop(0)

                if terminated or truncated:

                    observation = env.reset()

                    trail.clear()

                    total_reward = 0.0

        screen.fill(
            (17, 23, 34)
        )

        # Background grid.

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

        # Obstacle and safety zone.

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

        # Drone trajectory.

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
            "PHASE 5.7 — ADVANCED PPO",
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
            "SPACE pause | R reset | +/- speed | ESC exit",
            15,
            WORLD_H - 45,
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
# 17. NUMERICAL VERIFICATION
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

    # Test 1: Probability ratio.

    old = torch.tensor(
        [-1.0, -2.0]
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

    # Test 2: PPO clipping.

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
        "PPO clipping",
    )

    # Test 3: Positive advantage.

    objective = torch.minimum(
        torch.tensor([1.5]) * 2.0,

        torch.clamp(
            torch.tensor([1.5]),
            0.8,
            1.2,
        ) * 2.0,
    )

    check(
        torch.allclose(
            objective,
            torch.tensor([2.4]),
        ),
        "Positive advantage clipping",
    )

    # Test 4: Negative advantage.

    objective = torch.minimum(
        torch.tensor([0.5]) * -2.0,

        torch.clamp(
            torch.tensor([0.5]),
            0.8,
            1.2,
        ) * -2.0,
    )

    check(
        torch.allclose(
            objective,
            torch.tensor([-1.6]),
        ),
        "Negative advantage clipping",
    )

    # Test 5: Terminal bootstrap.

    advantages, _ = compute_gae(
        np.array([[1.0]], dtype=np.float32),

        np.array([[0.5]], dtype=np.float32),

        np.array([[100.0]], dtype=np.float32),

        np.array([[1.0]], dtype=np.float32),

        np.array([[0.0]], dtype=np.float32),

        gamma=0.9,
    )

    check(
        np.isclose(
            advantages[0, 0],
            0.5,
        ),
        "Terminal bootstrap suppression",
    )

    # Test 6: Truncation bootstrap.

    advantages, _ = compute_gae(
        np.array([[1.0]], dtype=np.float32),

        np.array([[0.5]], dtype=np.float32),

        np.array([[2.0]], dtype=np.float32),

        np.array([[0.0]], dtype=np.float32),

        np.array([[1.0]], dtype=np.float32),

        gamma=0.9,
    )

    check(
        np.isclose(
            advantages[0, 0],
            2.3,
        ),
        "Truncation bootstrap",
    )

    # Test 7: GAE recursion boundary.

    advantages, _ = compute_gae(
        np.array(
            [[1.0], [100.0]],
            dtype=np.float32,
        ),

        np.zeros(
            (2, 1),
            dtype=np.float32,
        ),

        np.zeros(
            (2, 1),
            dtype=np.float32,
        ),

        np.zeros(
            (2, 1),
            dtype=np.float32,
        ),

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
        "GAE truncation boundary",
    )

    # Test 8: Tanh action bounds.

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
        "Bounded actions",
    )

    # Test 9: Actor dimensions.

    actor = Actor().to(DEVICE)

    states = torch.zeros(
        (4, OBS_DIM),
        device=DEVICE,
    )

    mean, log_std = actor(states)

    check(
        mean.shape == (4, ACTION_DIM)
        and log_std.shape == (ACTION_DIM,),
        "Actor dimensions",
    )

    # Test 10: Critic dimensions.

    critic = Critic().to(DEVICE)

    values = critic(states)

    check(
        values.shape == (4,),
        "Critic dimensions",
    )

    # Test 11: Finite probabilities.

    network = ActorCritic().to(DEVICE)

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
        "Finite log probabilities",
    )

    # Test 12: Old/new log probability agreement.

    new_log_probs, entropy = (
        network.actor.evaluate_actions(
            states,
            raw_actions,
        )
    )

    check(
        torch.allclose(
            log_probs,
            new_log_probs,
            atol=1e-5,
        ),
        "Old/new log probability agreement",
    )

    # Test 13: Finite entropy.

    check(
        bool(
            torch.isfinite(entropy).all()
        ),
        "Finite policy entropy",
    )

    # Test 14: Environment API.

    env = DroneEnv()

    result = env.step(
        np.zeros(
            ACTION_DIM,
            dtype=np.float32,
        )
    )

    check(
        len(result) == 5,
        "Environment API",
    )

    # Test 15: Observation dimensions.

    check(
        env.observation().shape == (OBS_DIM,),
        "Observation dimensions",
    )

    # Test 16: Parallel environments.

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
        "Parallel environment dimensions",
    )

    # Test 17: Target velocity.

    target = MovingTarget(
        [400.0, 300.0]
    )

    check(
        target.velocity().shape == (2,),
        "Target velocity",
    )

    # Test 18: GAE numerical stability.

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
        "GAE numerical stability",
    )

    # Test 19: Optimizer.

    optimizer = optim.Adam(
        network.parameters(),
        lr=1e-4,
    )

    (
        log_probs,
        entropy,
        values,
    ) = network.evaluate_actions(
        states,
        raw_actions,
    )

    loss = (
        -log_probs.mean()
        + 0.5 * values.pow(2).mean()
        - ENTROPY_COEF * entropy.mean()
    )

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    check(
        bool(
            torch.isfinite(loss)
        ),
        "Actor-critic optimizer update",
    )

    # Test 20: Deterministic actions.

    deterministic = (
        network.deterministic_action(states)
    )

    check(
        deterministic.shape == (4, ACTION_DIM)
        and bool(
            torch.isfinite(deterministic).all()
        ),
        "Deterministic policy",
    )

    print(
        f"\n{passed}/20 tests passed."
    )

    return passed == 20


# ============================================================
# 18. MAIN
# ============================================================

def main():

    print("=" * 72)

    print(
        "PHASE 5.7 — ADVANCED PPO"
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
        f"Rollout batch: {NUM_ENVS * ROLLOUT_STEPS}"
    )

    print(
        f"PPO epochs: {PPO_EPOCHS}"
    )

    print(
        f"Minibatch size: {MINIBATCH_SIZE}"
    )

    print(
        f"Training updates: {TRAIN_UPDATES}"
    )

    print(
        f"Learning rate: {LEARNING_RATE}"
    )

    print(
        f"Clip epsilon: {CLIP_EPSILON}"
    )

    print(
        f"Target KL: {TARGET_KL}"
    )

    if not run_tests():

        raise RuntimeError(
            "Numerical verification failed."
        )

    print(
        "\n[1/5] Training PPO..."
    )

    agent, history, episode_returns = train()

    print(
        "\n[2/5] Evaluating PPO..."
    )

    evaluation_results = evaluate(
        agent,
        EVAL_EPISODES,
    )

    print(
        "\n[3/5] Saving model..."
    )

    save_model(agent)

    print(
        "\n[4/5] Plotting training diagnostics..."
    )

    plot_training(history)

    print(
        "\n[5/5] Starting interactive simulation..."
    )

    run_pygame(agent)

    print(
        "\nPhase 5.7 completed."
    )


if __name__ == "__main__":

    main()