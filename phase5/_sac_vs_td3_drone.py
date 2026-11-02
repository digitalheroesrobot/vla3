
"""
File: phase5/sac_vs_td3_drone.py

PHASE 5.15 — SAC VS TD3
=======================

PURPOSE
-------
Compare two continuous-action reinforcement learning algorithms
on the same moving-target quadcopter task.

WHAT YOU LEARN
--------------

STEP 1 — TD3
    Deterministic actor.
    Twin critics.
    Clipped double-Q target.
    Target policy smoothing.
    Delayed actor updates.

STEP 2 — SAC
    Gaussian stochastic actor.
    Tanh-squashed actions.
    Twin critics.
    Entropy-regularized Bellman target.
    Automatic entropy temperature.

STEP 3 — FAIR EXPERIMENT
    Both agents use the same environment dynamics.
    Both use the same evaluation seeds.
    Both are evaluated deterministically.
    Evaluation uses identical initial conditions.

STEP 4 — METRICS
    Episode return.
    Mean target distance.
    Collision count.
    Evaluation performance versus training steps.

STEP 5 — VISUALIZATION
    Training curves.
    Paired evaluation comparison.
    Interactive quadcopter simulation.

ROBOTICS APPLICATION
--------------------
A quadcopter tracks a moving inspection target while avoiding
an obstacle.

The target has position and velocity. Target velocity is
included in the observation.

IMPORTANT
---------
This is a simplified educational 2D simulation.
It is not a flight-certified control system.
"""

from __future__ import annotations

import copy
import math
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import matplotlib.pyplot as plt


# ============================================================
# 0. CONFIGURATION
# ============================================================

SEED = 515

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
TAU = 0.005

ACTOR_LR = 3e-4
CRITIC_LR = 3e-4

BATCH_SIZE = 128
REPLAY_CAPACITY = 100_000

INITIAL_RANDOM_STEPS = 2_000
TRAINING_STEPS = 30_000

MAX_EPISODE_STEPS = 400

EVALUATION_INTERVAL = 5_000
EVALUATION_EPISODES = 8

GRADIENT_CLIP = 1.0
EPSILON = 1e-6

LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0

# TD3-specific parameters.
TD3_POLICY_NOISE = 0.20
TD3_NOISE_CLIP = 0.50
TD3_POLICY_DELAY = 2

# SAC-specific parameter.
SAC_TARGET_ENTROPY = -float(ACTION_DIM)

# Set True for a shorter development run.
QUICK_RUN = False

if QUICK_RUN:

    TRAINING_STEPS = 5_000
    INITIAL_RANDOM_STEPS = 500
    BATCH_SIZE = 64
    EVALUATION_INTERVAL = 1_000
    EVALUATION_EPISODES = 3


# ============================================================
# 1. REPRODUCIBILITY
# ============================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


# ============================================================
# 2. MOVING TARGET
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
# 3. DRONE DYNAMICS
# ============================================================

def clip_drone_state(state):

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

    return clip_drone_state(
        np.concatenate(
            [
                next_position,
                next_velocity,
            ]
        )
    )


# ============================================================
# 4. SHARED ENVIRONMENT
# ============================================================

class DroneEnv:

    def __init__(self, seed=None):

        self.rng = np.random.default_rng(seed)

        self.target = MovingTarget(
            [700.0, 310.0]
        )

        self.state = None
        self.steps = 0

        self.reset()

    def reset(self, seed=None):

        if seed is not None:

            self.rng = np.random.default_rng(
                seed
            )

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

    def step(self, action):

        action = np.clip(
            np.asarray(
                action,
                dtype=np.float32,
            ),
            -1.0,
            1.0,
        )

        old_position = self.state[:2].copy()

        # Compare before and after positions against the same
        # future target position. This isolates drone progress
        # from target movement.

        future_phase = (
            self.target.phase
            + TARGET_OMEGA * DT
        )

        future_target = (
            self.target.position(future_phase)
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
            self.state[:2] - self.target.position()
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

        terminated = bool(collision)

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
# 5. REPLAY BUFFER
# ============================================================

class ReplayBuffer:

    def __init__(self):

        self.buffer = deque(
            maxlen=REPLAY_CAPACITY
        )

    def add(
        self,
        state,
        action,
        reward,
        next_state,
        terminated,
    ):

        self.buffer.append(
            (
                np.asarray(
                    state,
                    dtype=np.float32,
                ).copy(),

                np.asarray(
                    action,
                    dtype=np.float32,
                ).copy(),

                float(reward),

                np.asarray(
                    next_state,
                    dtype=np.float32,
                ).copy(),

                float(terminated),
            )
        )

    def sample(self):

        batch = random.sample(
            self.buffer,
            BATCH_SIZE,
        )

        arrays = (
            np.stack([x[0] for x in batch]),
            np.stack([x[1] for x in batch]),
            np.asarray(
                [x[2] for x in batch],
                dtype=np.float32,
            ).reshape(-1, 1),
            np.stack([x[3] for x in batch]),
            np.asarray(
                [x[4] for x in batch],
                dtype=np.float32,
            ).reshape(-1, 1),
        )

        return tuple(
            torch.as_tensor(
                value,
                dtype=torch.float32,
                device=DEVICE,
            )
            for value in arrays
        )

    def __len__(self):

        return len(self.buffer)


# ============================================================
# 6. COMMON NETWORKS
# ============================================================

class Actor(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(OBS_DIM, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, ACTION_DIM),
            nn.Tanh(),
        )

    def forward(self, states):

        return self.network(states)


class GaussianActor(nn.Module):

    def __init__(self):

        super().__init__()

        self.backbone = nn.Sequential(
            nn.Linear(OBS_DIM, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        self.mean = nn.Linear(
            256,
            ACTION_DIM,
        )

        self.log_std = nn.Linear(
            256,
            ACTION_DIM,
        )

    def forward(self, states):

        features = self.backbone(states)

        mean = self.mean(features)

        log_std = torch.clamp(
            self.log_std(features),
            LOG_STD_MIN,
            LOG_STD_MAX,
        )

        return mean, log_std

    def sample(self, states, deterministic=False):

        mean, log_std = self(states)

        if deterministic:

            return torch.tanh(mean), None

        distribution = Normal(
            mean,
            log_std.exp(),
        )

        # Reparameterization is required for SAC's
        # pathwise actor gradient.

        raw_action = distribution.rsample()

        action = torch.tanh(
            raw_action
        )

        raw_log_prob = distribution.log_prob(
            raw_action
        )

        # Correct for the tanh transformation.

        correction = 2.0 * (
            math.log(2.0)
            - raw_action
            - F.softplus(-2.0 * raw_action)
        )

        log_prob = (
            raw_log_prob - correction
        ).sum(
            dim=-1,
            keepdim=True,
        )

        return action, log_prob


class Critic(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                OBS_DIM + ACTION_DIM,
                256,
            ),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, states, actions):

        return self.network(
            torch.cat(
                [states, actions],
                dim=-1,
            )
        )


# ============================================================
# 7. SHARED BASE AGENT
# ============================================================

class BaseAgent:

    def __init__(self):

        self.replay = ReplayBuffer()

        self.critic1 = Critic().to(DEVICE)
        self.critic2 = Critic().to(DEVICE)

        self.target_critic1 = copy.deepcopy(
            self.critic1
        ).to(DEVICE)

        self.target_critic2 = copy.deepcopy(
            self.critic2
        ).to(DEVICE)

        for parameter in self.target_critic1.parameters():
            parameter.requires_grad_(False)

        for parameter in self.target_critic2.parameters():
            parameter.requires_grad_(False)

        self.critic_optimizer = optim.Adam(
            list(self.critic1.parameters())
            + list(self.critic2.parameters()),
            lr=CRITIC_LR,
        )

        self.update_count = 0

    def store(
        self,
        state,
        action,
        reward,
        next_state,
        terminated,
    ):

        self.replay.add(
            state,
            action,
            reward,
            next_state,
            terminated,
        )

    @staticmethod
    def soft_update(source, target):

        with torch.no_grad():

            for source_parameter, target_parameter in zip(
                source.parameters(),
                target.parameters(),
            ):

                target_parameter.mul_(
                    1.0 - TAU
                )

                target_parameter.add_(
                    TAU * source_parameter
                )

    def update_critics(self, targets, states, actions):

        q1 = self.critic1(
            states,
            actions,
        )

        q2 = self.critic2(
            states,
            actions,
        )

        loss = (
            F.mse_loss(q1, targets)
            + F.mse_loss(q2, targets)
        )

        self.critic_optimizer.zero_grad()

        loss.backward()

        nn.utils.clip_grad_norm_(
            list(self.critic1.parameters())
            + list(self.critic2.parameters()),
            GRADIENT_CLIP,
        )

        self.critic_optimizer.step()

        return loss.item()


# ============================================================
# 8. TD3
# ============================================================

class TD3Agent(BaseAgent):

    def __init__(self):

        super().__init__()

        self.actor = Actor().to(DEVICE)

        self.target_actor = copy.deepcopy(
            self.actor
        ).to(DEVICE)

        for parameter in self.target_actor.parameters():
            parameter.requires_grad_(False)

        self.actor_optimizer = optim.Adam(
            self.actor.parameters(),
            lr=ACTOR_LR,
        )

        self.exploration_std = 0.20

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

            action = self.actor(
                state
            ).squeeze(0).cpu().numpy()

        if not deterministic:

            action += np.random.normal(
                0.0,
                self.exploration_std,
                size=ACTION_DIM,
            )

        return np.clip(
            action,
            -1.0,
            1.0,
        ).astype(np.float32)

    def update(self):

        if len(self.replay) < BATCH_SIZE:

            return None

        (
            states,
            actions,
            rewards,
            next_states,
            terminated,
        ) = self.replay.sample()

        # Target policy smoothing.

        with torch.no_grad():

            next_actions = self.target_actor(
                next_states
            )

            noise = (
                torch.randn_like(next_actions)
                * TD3_POLICY_NOISE
            )

            noise = torch.clamp(
                noise,
                -TD3_NOISE_CLIP,
                TD3_NOISE_CLIP,
            )

            next_actions = torch.clamp(
                next_actions + noise,
                -1.0,
                1.0,
            )

            target_q1 = self.target_critic1(
                next_states,
                next_actions,
            )

            target_q2 = self.target_critic2(
                next_states,
                next_actions,
            )

            target_q = torch.minimum(
                target_q1,
                target_q2,
            )

            targets = (
                rewards
                + GAMMA
                * (1.0 - terminated)
                * target_q
            )

        critic_loss = self.update_critics(
            targets,
            states,
            actions,
        )

        self.update_count += 1

        actor_loss_value = None

        # Delayed actor and target updates.

        if self.update_count % TD3_POLICY_DELAY == 0:

            for parameter in self.critic1.parameters():
                parameter.requires_grad_(False)

            policy_actions = self.actor(states)

            actor_loss = -self.critic1(
                states,
                policy_actions,
            ).mean()

            self.actor_optimizer.zero_grad()

            actor_loss.backward()

            nn.utils.clip_grad_norm_(
                self.actor.parameters(),
                GRADIENT_CLIP,
            )

            self.actor_optimizer.step()

            for parameter in self.critic1.parameters():
                parameter.requires_grad_(True)

            self.soft_update(
                self.actor,
                self.target_actor,
            )

            self.soft_update(
                self.critic1,
                self.target_critic1,
            )

            self.soft_update(
                self.critic2,
                self.target_critic2,
            )

            actor_loss_value = actor_loss.item()

        return {
            "critic_loss": critic_loss,
            "actor_loss": actor_loss_value,
        }


# ============================================================
# 9. SAC
# ============================================================

class SACAgent(BaseAgent):

    def __init__(self):

        super().__init__()

        self.actor = GaussianActor().to(DEVICE)

        self.actor_optimizer = optim.Adam(
            self.actor.parameters(),
            lr=ACTOR_LR,
        )

        self.log_alpha = torch.tensor(
            0.0,
            device=DEVICE,
            requires_grad=True,
        )

        self.alpha_optimizer = optim.Adam(
            [self.log_alpha],
            lr=3e-4,
        )

        self.target_entropy = (
            SAC_TARGET_ENTROPY
        )

    @property
    def alpha(self):

        return self.log_alpha.exp().detach()

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

            action, _ = self.actor.sample(
                state,
                deterministic=deterministic,
            )

        return action.squeeze(0).cpu().numpy().astype(
            np.float32
        )

    def update(self):

        if len(self.replay) < BATCH_SIZE:

            return None

        (
            states,
            actions,
            rewards,
            next_states,
            terminated,
        ) = self.replay.sample()

        # Soft Bellman target.

        with torch.no_grad():

            next_actions, next_log_prob = (
                self.actor.sample(
                    next_states
                )
            )

            q1 = self.target_critic1(
                next_states,
                next_actions,
            )

            q2 = self.target_critic2(
                next_states,
                next_actions,
            )

            soft_q = (
                torch.minimum(q1, q2)
                - self.alpha * next_log_prob
            )

            targets = (
                rewards
                + GAMMA
                * (1.0 - terminated)
                * soft_q
            )

        critic_loss = self.update_critics(
            targets,
            states,
            actions,
        )

        # Actor update.

        for parameter in self.critic1.parameters():
            parameter.requires_grad_(False)

        for parameter in self.critic2.parameters():
            parameter.requires_grad_(False)

        sampled_actions, log_prob = (
            self.actor.sample(states)
        )

        q1 = self.critic1(
            states,
            sampled_actions,
        )

        q2 = self.critic2(
            states,
            sampled_actions,
        )

        actor_loss = (
            self.alpha * log_prob
            - torch.minimum(q1, q2)
        ).mean()

        self.actor_optimizer.zero_grad()

        actor_loss.backward()

        nn.utils.clip_grad_norm_(
            self.actor.parameters(),
            GRADIENT_CLIP,
        )

        self.actor_optimizer.step()

        for parameter in self.critic1.parameters():
            parameter.requires_grad_(True)

        for parameter in self.critic2.parameters():
            parameter.requires_grad_(True)

        # Automatic temperature update.

        alpha_loss = -(
            self.log_alpha
            * (
                log_prob.detach()
                + self.target_entropy
            )
        ).mean()

        self.alpha_optimizer.zero_grad()

        alpha_loss.backward()

        self.alpha_optimizer.step()

        # SAC updates target critics every gradient step.

        self.soft_update(
            self.critic1,
            self.target_critic1,
        )

        self.soft_update(
            self.critic2,
            self.target_critic2,
        )

        self.update_count += 1

        return {
            "critic_loss": critic_loss,
            "actor_loss": actor_loss.item(),
            "alpha": self.alpha.item(),
            "entropy": (-log_prob.detach()).mean().item(),
        }


# ============================================================
# 10. PAIRED EVALUATION
# ============================================================

def evaluate_paired(
    agents,
    seeds,
    episodes_per_seed=1,
):
    """
    Both algorithms receive exactly the same reset seed.

    This makes the initial drone state and target phase identical
    for each paired evaluation episode.
    """

    results = {
        name: {
            "returns": [],
            "distances": [],
            "collisions": [],
        }
        for name in agents
    }

    for seed in seeds:

        for name, agent in agents.items():

            for repetition in range(episodes_per_seed):

                env_seed = (
                    seed + repetition * 100_000
                )

                env = DroneEnv(
                    seed=env_seed
                )

                observation = env.reset(
                    seed=env_seed
                )

                episode_return = 0.0
                distances = []
                collision_count = 0

                for _ in range(MAX_EPISODE_STEPS):

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
                    ) = env.step(action)

                    episode_return += reward

                    distances.append(
                        info["distance"]
                    )

                    collision_count += int(
                        info["collision"]
                    )

                    if terminated or truncated:

                        break

                results[name]["returns"].append(
                    episode_return
                )

                results[name]["distances"].append(
                    float(np.mean(distances))
                )

                results[name]["collisions"].append(
                    collision_count
                )

    summary = {}

    for name, values in results.items():

        summary[name] = {
            "mean_return": float(
                np.mean(values["returns"])
            ),

            "std_return": float(
                np.std(values["returns"])
            ),

            "mean_distance": float(
                np.mean(values["distances"])
            ),

            "std_distance": float(
                np.std(values["distances"])
            ),

            "total_collisions": int(
                np.sum(values["collisions"])
            ),
        }

    return summary, results


# ============================================================
# 11. TRAINING
# ============================================================

def train_one_agent(name, agent, seed):

    set_seed(seed)

    env = DroneEnv(
        seed=seed + 1
    )

    observation = env.reset(
        seed=seed + 1
    )

    history = {
        "episode_return": [],
        "episode_distance": [],
        "critic_loss": [],
        "actor_loss": [],
        "alpha": [],
        "entropy": [],
        "evaluation_step": [],
        "evaluation_return": [],
        "evaluation_distance": [],
        "evaluation_collisions": [],
    }

    episode_return = 0.0
    episode_distances = []

    for step in range(TRAINING_STEPS):

        if step < INITIAL_RANDOM_STEPS:

            action = np.random.uniform(
                -1.0,
                1.0,
                ACTION_DIM,
            ).astype(np.float32)

        else:

            action = agent.select_action(
                observation,
                deterministic=False,
            )

        (
            next_observation,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(action)

        agent.store(
            observation,
            action,
            reward,
            next_observation,
            terminated,
        )

        observation = next_observation

        episode_return += reward

        episode_distances.append(
            info["distance"]
        )

        metrics = agent.update()

        if metrics is not None:

            history["critic_loss"].append(
                metrics["critic_loss"]
            )

            if metrics["actor_loss"] is not None:

                history["actor_loss"].append(
                    metrics["actor_loss"]
                )

            if "alpha" in metrics:

                history["alpha"].append(
                    metrics["alpha"]
                )

                history["entropy"].append(
                    metrics["entropy"]
                )

        if terminated or truncated:

            history["episode_return"].append(
                episode_return
            )

            history["episode_distance"].append(
                float(np.mean(episode_distances))
            )

            observation = env.reset()

            episode_return = 0.0
            episode_distances = []

        if (
            step > INITIAL_RANDOM_STEPS
            and (step + 1) % EVALUATION_INTERVAL == 0
        ):

            summary, _ = evaluate_paired(
                {
                    name: agent,
                },
                seeds=[
                    80_000 + i
                    for i in range(
                        EVALUATION_EPISODES
                    )
                ],
            )

            result = summary[name]

            history["evaluation_step"].append(
                step + 1
            )

            history["evaluation_return"].append(
                result["mean_return"]
            )

            history["evaluation_distance"].append(
                result["mean_distance"]
            )

            history["evaluation_collisions"].append(
                result["total_collisions"]
            )

            print(
                f"[{name}] "
                f"Step={step + 1:6d} "
                f"Return={result['mean_return']:8.2f} "
                f"Distance={result['mean_distance']:7.2f} "
                f"Collisions={result['total_collisions']}"
            )

            if isinstance(agent, SACAgent):

                print(
                    f"    alpha={agent.alpha.item():.4f}"
                )

    return history


# ============================================================
# 12. COMPARISON PLOTS
# ============================================================

def smooth(values, window=20):

    values = np.asarray(
        values,
        dtype=np.float32,
    )

    if len(values) < window:

        return values

    return np.convolve(
        values,
        np.ones(
            window,
            dtype=np.float32,
        ) / window,
        mode="valid",
    )


def plot_training(histories):

    for metric, title in [
        (
            "episode_return",
            "Training Episode Return",
        ),
        (
            "episode_distance",
            "Training Mean Target Distance",
        ),
        (
            "critic_loss",
            "Critic Loss",
        ),
        (
            "actor_loss",
            "Actor Loss",
        ),
    ]:

        plt.figure(figsize=(10, 4))

        for name, history in histories.items():

            values = history[metric]

            if len(values) == 0:
                continue

            plt.plot(
                smooth(values),
                label=name,
            )

        plt.title(title)
        plt.xlabel("Episode or update")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    plt.figure(figsize=(10, 4))

    for name, history in histories.items():

        if history["evaluation_step"]:

            plt.plot(
                history["evaluation_step"],
                history["evaluation_distance"],
                marker="o",
                label=name,
            )

    plt.title(
        "Paired-Seed Evaluation: Mean Target Distance"
    )

    plt.xlabel("Training steps")
    plt.ylabel("Mean target distance")

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    if histories["SAC"]["alpha"]:

        plt.figure(figsize=(10, 4))

        plt.plot(
            histories["SAC"]["alpha"],
        )

        plt.title(
            "SAC Automatic Entropy Temperature"
        )

        plt.xlabel("Gradient update")
        plt.ylabel("Alpha")

        plt.grid(True)
        plt.tight_layout()
        plt.show()


def plot_paired_comparison(summary):

    names = list(summary.keys())

    returns = [
        summary[name]["mean_return"]
        for name in names
    ]

    distances = [
        summary[name]["mean_distance"]
        for name in names
    ]

    collisions = [
        summary[name]["total_collisions"]
        for name in names
    ]

    metrics = [
        (
            returns,
            "Mean Evaluation Return",
            "Return",
        ),
        (
            distances,
            "Mean Evaluation Target Distance",
            "Distance",
        ),
        (
            collisions,
            "Total Evaluation Collisions",
            "Collisions",
        ),
    ]

    for values, title, ylabel in metrics:

        plt.figure(figsize=(7, 4))

        plt.bar(
            names,
            values,
        )

        plt.title(title)
        plt.ylabel(ylabel)
        plt.grid(
            axis="y",
            alpha=0.3,
        )

        plt.tight_layout()
        plt.show()


# ============================================================
# 13. INTERACTIVE PYGAME COMPARISON
# ============================================================

def run_pygame(agents):

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
        "SAC vs TD3 — Moving-Target Quadcopter"
    )

    clock = pygame.time.Clock()

    font = pygame.font.SysFont(
        "consolas",
        15,
    )

    selected = "TD3"

    env = DroneEnv(
        seed=90_000
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
            [-direction[1], direction[0]]
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
                    (225, 235, 245),
                    center.astype(int),
                    rotor.astype(int),
                    5,
                )

                pygame.draw.circle(
                    screen,
                    (80, 185, 245),
                    rotor.astype(int),
                    7,
                    2,
                )

        pygame.draw.circle(
            screen,
            (255, 255, 255),
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
            (45, 100, 65),
            True,
            points,
            2,
        )

        target_position = env.target.position()

        target_velocity = env.target.velocity()

        pygame.draw.circle(
            screen,
            (80, 245, 105),
            target_position.astype(int),
            11,
        )

        pygame.draw.line(
            screen,
            (80, 245, 105),
            target_position.astype(int),
            (
                target_position
                + target_velocity * 0.18
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

                elif event.key == pygame.K_1:

                    selected = "TD3"

                elif event.key == pygame.K_2:

                    selected = "SAC"

                elif event.key in (
                    pygame.K_PLUS,
                    pygame.K_EQUALS,
                    pygame.K_KP_PLUS,
                ):

                    speed = min(
                        5,
                        speed + 1,
                    )

                elif event.key in (
                    pygame.K_MINUS,
                    pygame.K_KP_MINUS,
                ):

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

                action = agents[selected].select_action(
                    observation,
                    deterministic=True,
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

        for x in range(0, WORLD_W, 50):

            pygame.draw.line(
                screen,
                (30, 38, 50),
                (x, 0),
                (x, WORLD_H),
            )

        for y in range(0, WORLD_H, 50):

            pygame.draw.line(
                screen,
                (30, 38, 50),
                (0, y),
                (WORLD_W, y),
            )

        pygame.draw.circle(
            screen,
            (100, 105, 115),
            OBSTACLE.astype(int),
            OBSTACLE_RADIUS,
        )

        pygame.draw.circle(
            screen,
            (190, 90, 90),
            OBSTACLE.astype(int),
            SAFE_RADIUS,
            1,
        )

        draw_target()

        if len(trail) > 1:

            pygame.draw.lines(
                screen,
                (80, 140, 235),
                False,
                [
                    point.astype(int)
                    for point in trail
                ],
                2,
            )

        draw_drone()

        draw_text(
            "PHASE 5.15 — SAC VS TD3",
            15,
            12,
        )

        draw_text(
            f"Controller: {selected}",
            15,
            40,
        )

        draw_text(
            f"Episode reward: {total_reward:.2f}",
            15,
            65,
        )

        draw_text(
            f"Speed: {speed}x",
            15,
            90,
        )

        draw_text(
            "1 TD3 | 2 SAC",
            15,
            WORLD_H - 130,
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
            )

        pygame.display.flip()

    pygame.quit()


# ============================================================
# 14. TESTS
# ============================================================

def run_tests():

    print("\nRunning comparison tests...")

    passed = 0

    def check(condition, description):

        nonlocal passed

        assert condition, description

        passed += 1

        print("[PASS]", description)

    # Test 1: Environment observation shape.

    env = DroneEnv(seed=515)

    observation = env.reset()

    check(
        observation.shape == (OBS_DIM,),
        "Observation dimension",
    )

    # Test 2: Observation finite.

    check(
        np.isfinite(observation).all(),
        "Finite environment observation",
    )

    # Test 3: Drone dynamics.

    next_state = drone_dynamics(
        np.array(
            [100.0, 100.0, 0.0, 0.0],
            dtype=np.float32,
        ),
        np.array(
            [1.0, 0.0],
            dtype=np.float32,
        ),
    )

    check(
        next_state.shape == (4,)
        and np.isfinite(next_state).all(),
        "Drone dynamics",
    )

    # Test 4: Speed constraint.

    check(
        np.linalg.norm(next_state[2:4])
        <= MAX_SPEED + 1e-5,
        "Maximum drone speed",
    )

    # Test 5: Moving target velocity.

    check(
        env.target.velocity().shape == (2,)
        and np.isfinite(env.target.velocity()).all(),
        "Moving target velocity",
    )

    # Test 6: Action clipping.

    clipped = drone_dynamics(
        next_state,
        np.array(
            [5.0, -5.0],
            dtype=np.float32,
        ),
    )

    check(
        np.isfinite(clipped).all(),
        "Action clipping",
    )

    # Test 7: TD3 actor bounds.

    td3_actor = Actor().to(DEVICE)

    states = torch.randn(
        8,
        OBS_DIM,
        device=DEVICE,
    )

    td3_actions = td3_actor(states)

    check(
        bool(torch.all(td3_actions.abs() <= 1.0)),
        "TD3 actor action bounds",
    )

    # Test 8: SAC actor action bounds.

    sac_actor = GaussianActor().to(DEVICE)

    sac_actions, sac_log_prob = sac_actor.sample(
        states
    )

    check(
        bool(torch.all(sac_actions.abs() <= 1.0)),
        "SAC actor action bounds",
    )

    # Test 9: SAC log probability finite.

    check(
        bool(torch.isfinite(sac_log_prob).all()),
        "SAC log probability",
    )

    # Test 10: Twin critic shapes.

    critic1 = Critic().to(DEVICE)
    critic2 = Critic().to(DEVICE)

    q1 = critic1(
        states,
        sac_actions,
    )

    q2 = critic2(
        states,
        sac_actions,
    )

    check(
        q1.shape == (8, 1)
        and q2.shape == (8, 1),
        "Twin critic shapes",
    )

    # Test 11: Clipped double Q.

    minimum = torch.minimum(
        torch.tensor(
            [[5.0], [2.0]],
            device=DEVICE,
        ),
        torch.tensor(
            [[3.0], [4.0]],
            device=DEVICE,
        ),
    )

    check(
        torch.allclose(
            minimum,
            torch.tensor(
                [[3.0], [2.0]],
                device=DEVICE,
            ),
        ),
        "Clipped double Q",
    )

    # Test 12: Terminal mask.

    reward = torch.tensor(
        [[2.0]],
        device=DEVICE,
    )

    target_q = torch.tensor(
        [[5.0]],
        device=DEVICE,
    )

    terminal = torch.tensor(
        [[1.0]],
        device=DEVICE,
    )

    target = (
        reward
        + GAMMA * (1.0 - terminal) * target_q
    )

    check(
        torch.allclose(
            target,
            reward,
        ),
        "Terminal target masking",
    )

    # Test 13: Replay buffer.

    replay = ReplayBuffer()

    replay.add(
        np.zeros(OBS_DIM),
        np.zeros(ACTION_DIM),
        1.0,
        np.ones(OBS_DIM),
        False,
    )

    check(
        len(replay) == 1,
        "Replay buffer storage",
    )

    # Test 14: Same reset seed gives same state.

    env_a = DroneEnv(seed=100)
    env_b = DroneEnv(seed=100)

    obs_a = env_a.reset(seed=200)
    obs_b = env_b.reset(seed=200)

    check(
        np.allclose(obs_a, obs_b),
        "Paired reset reproducibility",
    )

    # Test 15: TD3 agent construction.

    td3 = TD3Agent()

    td3_action = td3.select_action(
        observation,
        deterministic=True,
    )

    check(
        td3_action.shape == (ACTION_DIM,)
        and np.all(np.abs(td3_action) <= 1.0),
        "TD3 deterministic action",
    )

    # Test 16: SAC agent construction.

    sac = SACAgent()

    sac_action = sac.select_action(
        observation,
        deterministic=True,
    )

    check(
        sac_action.shape == (ACTION_DIM,)
        and np.all(np.abs(sac_action) <= 1.0)
        and sac.alpha.item() > 0,
        "SAC deterministic action and alpha",
    )

    # Test 17: Environment transition.

    next_observation, reward, terminated, truncated, info = (
        env.step(
            np.zeros(
                ACTION_DIM,
                dtype=np.float32,
            )
        )
    )

    check(
        next_observation.shape == (OBS_DIM,)
        and np.isfinite(reward)
        and "distance" in info,
        "Environment transition",
    )

    print(
        f"\n{passed}/17 tests passed."
    )

    return passed == 17


# ============================================================
# 15. MAIN
# ============================================================

def main():

    print("=" * 72)

    print(
        "PHASE 5.15 — SAC VS TD3"
    )

    print("=" * 72)

    print("Device:", DEVICE)

    print("Training steps per algorithm:", TRAINING_STEPS)

    if not run_tests():

        raise RuntimeError(
            "Tests failed."
        )

    histories = {}

    agents = {}

    # --------------------------------------------------------
    # Train TD3.
    # --------------------------------------------------------

    print("\nTraining TD3...")

    set_seed(SEED)

    td3 = TD3Agent()

    histories["TD3"] = train_one_agent(
        "TD3",
        td3,
        SEED,
    )

    agents["TD3"] = td3

    # --------------------------------------------------------
    # Train SAC.
    # --------------------------------------------------------

    print("\nTraining SAC...")

    set_seed(SEED)

    sac = SACAgent()

    histories["SAC"] = train_one_agent(
        "SAC",
        sac,
        SEED,
    )

    agents["SAC"] = sac

    # --------------------------------------------------------
    # Paired evaluation.
    # --------------------------------------------------------

    print("\nFinal paired evaluation...")

    seeds = [
        100_000 + index
        for index in range(20)
    ]

    summary, raw_results = evaluate_paired(
        agents,
        seeds,
    )

    for name, result in summary.items():

        print(f"\n{name}")

        for metric, value in result.items():

            print(
                f"  {metric}: {value}"
            )

    print("\nPlotting training curves...")

    plot_training(histories)

    print("\nPlotting paired evaluation...")

    plot_paired_comparison(summary)

    print("\nStarting interactive comparison...")

    run_pygame(agents)

    print("\nPhase 5.15 completed.")


if __name__ == "__main__":

    main()
