
"""
File: phase5/ppo_improvements_drone.py

PHASE 5.7 — ADVANCED PPO IMPROVEMENTS
============================================================

PURPOSE
-------
Compare baseline PPO with PPO enhanced by training-stability
techniques, using a continuous-control quadcopter task.

WHAT YOU LEARN
--------------

STEP 1 — Advantage Normalization
    Normalize rollout advantages before policy optimization.

STEP 2 — Value Function Clipping
    Restrict large critic updates relative to old predictions.

STEP 3 — Learning-Rate Annealing
    Reduce optimizer learning rate as training progresses.

STEP 4 — KL Early Stopping
    Stop PPO optimization when policy divergence is excessive.

STEP 5 — Gradient Clipping
    Limit the magnitude of network gradients.

STEP 6 — Ablation Comparison
    Compare baseline PPO with improved PPO.

STEP 7 — Robotics Application
    Track a moving target while avoiding an obstacle.

MATHEMATICS
-----------

POLICY OBJECTIVE:

    ratio = exp(new_log_prob - old_log_prob)

    L_clip = mean(
        min(
            ratio * advantage,
            clip(ratio, 1-eps, 1+eps) * advantage
        )
    )

VALUE CLIPPING:

    value_clipped =
        old_value + clip(
            new_value - old_value,
            -value_clip_epsilon,
            +value_clip_epsilon
        )

    loss_unclipped = (new_value - return)^2

    loss_clipped = (value_clipped - return)^2

    value_loss = 0.5 * mean(
        max(loss_unclipped, loss_clipped)
    )

ADVANTAGE NORMALIZATION:

    A_normalized = (A - mean(A)) / (std(A) + epsilon)

LEARNING RATE ANNEALING:

    lr = initial_lr * (1 - progress)

KL EARLY STOPPING:

    stop optimization if approximate_kl > target_kl

GRADIENT CLIPPING:

    ||gradient|| <= max_gradient_norm

COMPARISON
----------

Baseline PPO:
    - PPO clipped objective
    - GAE
    - Multiple minibatch epochs

Improved PPO:
    - PPO clipped objective
    - GAE
    - Advantage normalization
    - Value clipping
    - Learning-rate annealing
    - KL early stopping
    - Gradient clipping

ROBOTICS APPLICATION
--------------------

A simulated quadcopter tracks a moving target while
avoiding a circular obstacle.

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

SEED = 570

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
VALUE_CLIP_EPSILON = 0.20

VALUE_COEF = 0.5
ENTROPY_COEF = 0.005

LEARNING_RATE = 3e-4
MAX_GRAD_NORM = 0.5

NUM_ENVS = 8
ROLLOUT_STEPS = 128

PPO_EPOCHS = 6
MINIBATCH_SIZE = 128

TRAIN_UPDATES = 300

MAX_EPISODE_STEPS = 180

TARGET_KL = 0.02

PRINT_INTERVAL = 10

# Quick run:
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
# 4. PARALLEL ENVIRONMENTS
# ============================================================

class ParallelEnvironments:

    def __init__(self, count):

        self.envs = [
            DroneEnv()
            for _ in range(count)
        ]

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
# 5. ACTOR-CRITIC
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

        # Gaussian entropy is an approximation to the
        # entropy of the tanh-transformed distribution.

        entropy = distribution.entropy().sum(
            dim=-1
        )

        return log_prob, entropy, value

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

    return (
        advantages,
        advantages + values,
    )


# ============================================================
# 7. PPO AGENT
# ============================================================

class PPOAgent:

    def __init__(
        self,
        improved=True,
    ):

        self.improved = improved

        self.network = ActorCritic().to(
            DEVICE
        )

        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=LEARNING_RATE,
            eps=1e-5,
        )

        self.initial_lr = LEARNING_RATE

    def set_learning_rate(self, progress):

        if not self.improved:

            return

        # progress: 0 at training start, 1 at end.

        lr = (
            self.initial_lr
            * max(0.0, 1.0 - progress)
        )

        for group in self.optimizer.param_groups:

            group["lr"] = lr

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

            action = self.network.deterministic_action(
                states
            )

        return action.squeeze(0).cpu().numpy()

    def update(
        self,
        observations,
        raw_actions,
        old_log_probs,
        old_values,
        advantages,
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

        old_log_probs = torch.as_tensor(
            old_log_probs,
            dtype=torch.float32,
            device=DEVICE,
        )

        old_values = torch.as_tensor(
            old_values,
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

        # Improvement 1:
        # Normalize advantages over the entire rollout.

        if self.improved:

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
            "epochs": 0,
        }

        for epoch in range(
            PPO_EPOCHS
        ):

            indices = torch.randperm(
                batch_size,
                device=DEVICE,
            )

            epoch_kl = []

            for start in range(
                0,
                batch_size,
                MINIBATCH_SIZE,
            ):

                mb = indices[
                    start:start + MINIBATCH_SIZE
                ]

                mb_states = states[mb]
                mb_actions = raw_actions[mb]

                mb_old_log_probs = old_log_probs[mb]
                mb_old_values = old_values[mb]

                mb_advantages = advantages[mb]
                mb_returns = returns[mb]

                (
                    new_log_probs,
                    entropy,
                    new_values,
                ) = self.network.evaluate_actions(
                    mb_states,
                    mb_actions,
                )

                log_ratio = (
                    new_log_probs
                    - mb_old_log_probs
                )

                ratio = torch.exp(
                    log_ratio
                )

                # PPO clipped policy objective.

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

                # Critic loss:
                # Baseline PPO uses ordinary squared error.
                # Improved PPO clips changes from old values.

                if self.improved:

                    clipped_values = (
                        mb_old_values
                        + torch.clamp(
                            new_values - mb_old_values,
                            -VALUE_CLIP_EPSILON,
                            VALUE_CLIP_EPSILON,
                        )
                    )

                    loss_unclipped = (
                        new_values - mb_returns
                    ).pow(2)

                    loss_clipped = (
                        clipped_values - mb_returns
                    ).pow(2)

                    critic_loss = 0.5 * torch.maximum(
                        loss_unclipped,
                        loss_clipped,
                    ).mean()

                else:

                    critic_loss = 0.5 * (
                        new_values - mb_returns
                    ).pow(2).mean()

                entropy_mean = entropy.mean()

                total_loss = (
                    actor_loss
                    + VALUE_COEF * critic_loss
                    - ENTROPY_COEF * entropy_mean
                )

                self.optimizer.zero_grad()

                total_loss.backward()

                # Improvement 2:
                # Gradient clipping.

                if self.improved:

                    nn.utils.clip_grad_norm_(
                        self.network.parameters(),
                        MAX_GRAD_NORM,
                    )

                self.optimizer.step()

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

                epoch_kl.append(
                    approx_kl.item()
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

            metrics["epochs"] += 1

            # Improvement 3:
            # KL-based early stopping.

            if (
                self.improved
                and len(epoch_kl) > 0
                and np.mean(epoch_kl) > TARGET_KL
            ):

                break

        return {
            key: (
                value
                if key == "epochs"
                else float(np.mean(value))
            )
            for key, value in metrics.items()
        }


# ============================================================
# 8. ROLLOUT COLLECTION
# ============================================================

def collect_rollout(
    agent,
    envs,
    observations,
):

    buffers = {
        "observations": [],
        "raw_actions": [],
        "old_log_probs": [],
        "rewards": [],
        "values": [],
        "next_values": [],
        "terminated": [],
        "truncated": [],
        "distances": [],
    }

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

        # Bootstrap from the final observation before reset.

        next_values = agent.predict_values(
            final_observations
        )

        buffers["observations"].append(
            observations.copy()
        )

        buffers["raw_actions"].append(
            raw_actions.copy()
        )

        buffers["old_log_probs"].append(
            log_probs.copy()
        )

        buffers["rewards"].append(
            rewards.copy()
        )

        buffers["values"].append(
            values.copy()
        )

        buffers["next_values"].append(
            next_values.copy()
        )

        buffers["terminated"].append(
            terminated.copy()
        )

        buffers["truncated"].append(
            truncated.copy()
        )

        buffers["distances"].append(
            [
                info["distance"]
                for info in infos
            ]
        )

        observations = next_observations

    return (
        {
            key: np.asarray(
                value,
                dtype=np.float32,
            )
            for key, value in buffers.items()
        },
        observations,
    )


# ============================================================
# 9. TRAIN ONE PPO CONFIGURATION
# ============================================================

def train_one(
    improved,
    label,
):

    print(
        f"\nTraining: {label}"
    )

    envs = ParallelEnvironments(
        NUM_ENVS
    )

    agent = PPOAgent(
        improved=improved
    )

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
        "epochs": [],
        "learning_rate": [],
    }

    for update in range(
        TRAIN_UPDATES
    ):

        progress = (
            update / max(1, TRAIN_UPDATES - 1)
        )

        agent.set_learning_rate(
            progress
        )

        (
            rollout,
            observations,
        ) = collect_rollout(
            agent,
            envs,
            observations,
        )

        advantages, returns = compute_gae(
            rollout["rewards"],
            rollout["values"],
            rollout["next_values"],
            rollout["terminated"],
            rollout["truncated"],
        )

        states = rollout[
            "observations"
        ].reshape(
            -1,
            OBS_DIM,
        )

        actions = rollout[
            "raw_actions"
        ].reshape(
            -1,
            ACTION_DIM,
        )

        old_log_probs = rollout[
            "old_log_probs"
        ].reshape(-1)

        old_values = rollout[
            "values"
        ].reshape(-1)

        advantages = advantages.reshape(-1)
        returns = returns.reshape(-1)

        metrics = agent.update(
            states,
            actions,
            old_log_probs,
            old_values,
            advantages,
            returns,
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
            "epochs",
        ):

            history[key].append(
                metrics[key]
            )

        history["learning_rate"].append(
            agent.optimizer.param_groups[0]["lr"]
        )

        if (
            update + 1
        ) % PRINT_INTERVAL == 0:

            recent = slice(
                -PRINT_INTERVAL,
                None,
            )

            print(
                f"{label} | "
                f"Update {update + 1:4d}/{TRAIN_UPDATES}"
                f" | Reward={np.mean(history['reward'][recent]):8.2f}"
                f" | Distance={np.mean(history['distance'][recent]):7.2f}"
                f" | KL={np.mean(history['approx_kl'][recent]):7.4f}"
                f" | Clip={np.mean(history['clip_fraction'][recent]):6.3f}"
                f" | Epochs={np.mean(history['epochs'][recent]):4.1f}"
            )

    return agent, history


# ============================================================
# 10. ABLATION COMPARISON
# ============================================================

def compare_histories(
    baseline_history,
    improved_history,
):

    comparisons = [
        ("reward", "Mean Rollout Reward"),
        ("distance", "Mean Target Distance"),
        ("actor_loss", "Actor Loss"),
        ("critic_loss", "Critic Loss"),
        ("approx_kl", "Approximate KL"),
        ("clip_fraction", "Clip Fraction"),
        ("entropy", "Policy Entropy"),
        ("learning_rate", "Learning Rate"),
        ("epochs", "Optimization Epochs"),
    ]

    for key, title in comparisons:

        plt.figure(
            figsize=(10, 5)
        )

        plt.plot(
            baseline_history[key],
            label="Baseline PPO",
        )

        plt.plot(
            improved_history[key],
            label="Improved PPO",
        )

        plt.title(title)

        plt.xlabel("Training Update")

        plt.ylabel(key)

        plt.legend()

        plt.grid(True)

        plt.tight_layout()

        plt.show()


# ============================================================
# 11. EVALUATION
# ============================================================

def evaluate(
    agent,
    label,
    episodes=5,
):

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
                np.mean(distances),
                float(collision),
            ]
        )

    results = np.asarray(
        results,
        dtype=np.float32,
    )

    print(
        f"\n{label} evaluation"
    )

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
# 12. INTERACTIVE PYGAME QUADCOPTER
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
        "Phase 5.7 — Improved PPO Drone"
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
            "PHASE 5.7 — IMPROVED PPO",
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
# 13. NUMERICAL TESTS
# ============================================================

def run_tests():

    print(
        "\nRunning PPO improvement tests..."
    )

    passed = 0

    def check(condition, description):

        nonlocal passed

        assert condition, description

        passed += 1

        print(
            f"[PASS] {description}"
        )

    # Test 1: Advantage normalization.

    advantages = torch.tensor(
        [1.0, 2.0, 3.0, 4.0]
    )

    normalized = (
        advantages - advantages.mean()
    ) / (
        advantages.std(unbiased=False) + 1e-8
    )

    check(
        abs(normalized.mean().item()) < 1e-6,
        "Normalized advantages have zero mean",
    )

    # Test 2: Normalized standard deviation.

    check(
        abs(
            normalized.std(unbiased=False).item()
            - 1.0
        ) < 1e-5,
        "Normalized advantages have unit standard deviation",
    )

    # Test 3: Value clipping.

    old_value = torch.tensor(
        [1.0, 2.0]
    )

    new_value = torch.tensor(
        [2.0, 0.5]
    )

    clipped_value = (
        old_value
        + torch.clamp(
            new_value - old_value,
            -0.2,
            0.2,
        )
    )

    check(
        torch.allclose(
            clipped_value,
            torch.tensor(
                [1.2, 1.8]
            ),
        ),
        "Value function clipping",
    )

    # Test 4: Value clipping does not exceed bounds.

    difference = (
        clipped_value - old_value
    )

    check(
        bool(
            torch.all(
                difference.abs() <= 0.2 + 1e-6
            )
        ),
        "Value changes remain inside clipping interval",
    )

    # Test 5: Learning-rate annealing.

    start_lr = LEARNING_RATE * (1.0 - 0.0)

    end_lr = LEARNING_RATE * (1.0 - 1.0)

    check(
        np.isclose(start_lr, LEARNING_RATE)
        and np.isclose(end_lr, 0.0),
        "Learning-rate annealing endpoints",
    )

    # Test 6: PPO probability ratio.

    old_log_prob = torch.tensor(
        [-1.0, -2.0]
    )

    ratio = torch.exp(
        old_log_prob - old_log_prob
    )

    check(
        torch.allclose(
            ratio,
            torch.ones_like(ratio),
        ),
        "Unchanged policy has ratio one",
    )

    # Test 7: PPO clipping.

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
        "PPO ratio clipping",
    )

    # Test 8: GAE terminal mask.

    advantages, _ = compute_gae(
        np.array([[1.0]], dtype=np.float32),
        np.array([[0.5]], dtype=np.float32),
        np.array([[100.0]], dtype=np.float32),
        np.array([[1.0]], dtype=np.float32),
        np.array([[0.0]], dtype=np.float32),
    )

    check(
        np.isclose(
            advantages[0, 0],
            0.5,
        ),
        "Terminal transition suppresses bootstrap",
    )

    # Test 9: GAE truncation bootstrap.

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
        "Truncation bootstraps from final observation",
    )

    # Test 10: GAE recursion stops at truncation.

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

    # Test 11: Network output.

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
        "Actor-critic dimensions",
    )

    # Test 12: Bounded actions.

    actions, raw, log_probs, values = (
        network.sample(states)
    )

    check(
        bool(
            torch.all(
                actions.abs() <= 1.0
            )
        ),
        "Tanh action bounds",
    )

    # Test 13: Finite log probabilities.

    check(
        bool(
            torch.isfinite(log_probs).all()
        ),
        "Finite action log probabilities",
    )

    # Test 14: Finite entropy.

    log_probs, entropy, values = (
        network.evaluate_actions(
            states,
            raw,
        )
    )

    check(
        bool(
            torch.isfinite(entropy).all()
        ),
        "Finite policy entropy",
    )

    # Test 15: Gradient clipping utility.

    parameters = [
        torch.nn.Parameter(
            torch.tensor([100.0])
        )
    ]

    parameters[0].grad = torch.tensor(
        [100.0]
    )

    nn.utils.clip_grad_norm_(
        parameters,
        0.5,
    )

    check(
        parameters[0].grad.abs().item() <= 0.50001,
        "Gradient norm clipping",
    )

    # Test 16: Parallel environment shapes.

    envs = ParallelEnvironments(3)

    observations = envs.reset()

    output = envs.step(
        np.zeros(
            (3, ACTION_DIM),
            dtype=np.float32,
        )
    )

    check(
        observations.shape == (3, OBS_DIM)
        and output[0].shape == (3, OBS_DIM)
        and output[1].shape == (3, OBS_DIM),
        "Parallel environment dimensions",
    )

    # Test 17: Moving target velocity.

    target = MovingTarget(
        [400.0, 300.0]
    )

    check(
        target.velocity().shape == (2,),
        "Moving target velocity",
    )

    # Test 18: Optimizer update.

    optimizer = optim.Adam(
        network.parameters(),
        lr=1e-4,
    )

    log_probs, entropy, values = (
        network.evaluate_actions(
            states,
            raw,
        )
    )

    loss = (
        -log_probs.mean()
        + 0.5 * values.pow(2).mean()
        - 0.005 * entropy.mean()
    )

    optimizer.zero_grad()

    loss.backward()

    nn.utils.clip_grad_norm_(
        network.parameters(),
        0.5,
    )

    optimizer.step()

    check(
        bool(
            torch.isfinite(loss)
        ),
        "Actor-critic optimizer update",
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
        "PHASE 5.7 — ADVANCED PPO IMPROVEMENTS"
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

    if not run_tests():

        raise RuntimeError(
            "Numerical tests failed."
        )

    print(
        "\n[1/5] Training baseline PPO..."
    )

    baseline_agent, baseline_history = train_one(
        improved=False,
        label="Baseline PPO",
    )

    print(
        "\n[2/5] Training improved PPO..."
    )

    improved_agent, improved_history = train_one(
        improved=True,
        label="Improved PPO",
    )

    print(
        "\n[3/5] Comparing PPO training..."
    )

    compare_histories(
        baseline_history,
        improved_history,
    )

    print(
        "\n[4/5] Evaluating both agents..."
    )

    baseline_results = evaluate(
        baseline_agent,
        "Baseline PPO",
    )

    improved_results = evaluate(
        improved_agent,
        "Improved PPO",
    )

    print(
        "\nComparison summary"
    )

    print(
        "Baseline mean distance:",
        float(
            baseline_results[:, 1].mean()
        ),
    )

    print(
        "Improved mean distance:",
        float(
            improved_results[:, 1].mean()
        ),
    )

    print(
        "\n[5/5] Starting improved PPO drone visualization..."
    )

    run_pygame(
        improved_agent
    )

    print(
        "\nPhase 5.7 completed."
    )


if __name__ == "__main__":

    main()
