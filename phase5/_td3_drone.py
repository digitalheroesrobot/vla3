
"""
File: phase5/td3_drone.py

PHASE 5.13 — TWIN DELAYED DEEP DETERMINISTIC POLICY GRADIENT
============================================================

PURPOSE
-------
Implement TD3 and compare it with DDPG for continuous-action
quadcopter target tracking.

WHAT YOU LEARN
--------------

STEP 1 — Twin critics

    Q1(s,a), Q2(s,a)

    Two independent critics reduce the risk of relying on
    one overestimated Q-function.

STEP 2 — Clipped Double Q-learning

    y = r + gamma*(1-done) *
        min(Q1_target(s',a'), Q2_target(s',a'))

    Use the smaller target estimate.

STEP 3 — Target policy smoothing

    a' = mu_target(s') + clip(noise, -c, c)

    a' = clip(a', -1, 1)

    This smooths the target value over nearby actions.

STEP 4 — Delayed actor updates

    Update critics every training iteration.

    Update actor and target networks only once every
    POLICY_DELAY critic updates.

STEP 5 — Soft target updates

    target = tau*online + (1-tau)*target

STEP 6 — Exploration

    Add OU or Gaussian noise to actions collected from
    the environment.

STEP 7 — DDPG vs TD3

    Compare:
        - episode returns
        - tracking distance
        - critic losses
        - evaluation collisions

STEP 8 — ROBOTICS APPLICATION

    A simplified 2D quadcopter tracks a moving target.
    The observation includes target velocity.

IMPORTANT
---------
This is an educational 2D simulation, not a flight-certified
controller.

TD3 reduces certain sources of critic overestimation but does
not guarantee safe or successful control.
"""

from __future__ import annotations

import copy
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

SEED = 513

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
TAU = 0.005

ACTOR_LR = 1e-4
CRITIC_LR = 1e-3

REPLAY_CAPACITY = 100_000
BATCH_SIZE = 128

INITIAL_RANDOM_STEPS = 2_000
TRAINING_STEPS = 35_000

MAX_EPISODE_STEPS = 400

EXPLORATION_TYPE = "ou"

OU_THETA = 0.15
OU_SIGMA_START = 0.30
OU_SIGMA_END = 0.08

POLICY_DELAY = 2

TARGET_POLICY_NOISE = 0.20
TARGET_NOISE_CLIP = 0.50

GRADIENT_CLIP = 1.0

EVALUATION_INTERVAL = 5_000
EVALUATION_EPISODES = 4


# ============================================================
# 1. RANDOM SEED
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

    speed = np.linalg.norm(state[2:4])

    if speed > MAX_SPEED:

        state[2:4] *= MAX_SPEED / (speed + 1e-9)

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
        np.asarray(action, dtype=np.float32),
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
            [next_position, next_velocity]
        )
    )


# ============================================================
# 4. DRONE ENVIRONMENT
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

    def observation(self):

        position = self.state[:2]
        velocity = self.state[2:4]

        target_position = self.target.position()
        target_velocity = self.target.velocity()

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
            np.asarray(action, dtype=np.float32),
            -1.0,
            1.0,
        )

        old_position = self.state[:2].copy()

        # Measure progress relative to the same future target
        # position before and after the drone moves.

        future_phase = (
            self.target.phase + TARGET_OMEGA * DT
        )

        future_target = self.target.position(
            future_phase
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
            0.08 * (distance_before - distance_after)
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
# 5. EXPLORATION NOISE
# ============================================================

class OUNoise:

    def __init__(
        self,
        size,
        theta=OU_THETA,
        sigma=OU_SIGMA_START,
    ):

        self.size = size
        self.theta = theta
        self.sigma = sigma

        self.state = np.zeros(
            size,
            dtype=np.float32,
        )

    def reset(self):

        self.state.fill(0.0)

    def sample(self):

        dx = (
            self.theta * (-self.state)
            + self.sigma * np.random.normal(
                size=self.size
            )
        )

        self.state += dx.astype(np.float32)

        return self.state.copy()


def gaussian_noise(size, sigma):

    return np.random.normal(
        0.0,
        sigma,
        size=size,
    ).astype(np.float32)


# ============================================================
# 6. ACTOR NETWORK
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


# ============================================================
# 7. CRITIC NETWORK
# ============================================================

class Critic(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(OBS_DIM + ACTION_DIM, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, states, actions):

        inputs = torch.cat(
            [states, actions],
            dim=-1,
        )

        return self.network(inputs)


# ============================================================
# 8. REPLAY BUFFER
# ============================================================

class ReplayBuffer:

    def __init__(self, capacity=REPLAY_CAPACITY):

        self.buffer = deque(maxlen=capacity)

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

    def sample(self, batch_size):

        batch = random.sample(
            self.buffer,
            batch_size,
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
                array,
                dtype=torch.float32,
                device=DEVICE,
            )
            for array in arrays
        )

    def __len__(self):

        return len(self.buffer)


# ============================================================
# 9. DDPG AGENT
# ============================================================

class DDPGAgent:

    def __init__(self):

        self.actor = Actor().to(DEVICE)
        self.critic = Critic().to(DEVICE)

        self.target_actor = copy.deepcopy(
            self.actor
        ).to(DEVICE)

        self.target_critic = copy.deepcopy(
            self.critic
        ).to(DEVICE)

        self.actor_optimizer = optim.Adam(
            self.actor.parameters(),
            lr=ACTOR_LR,
        )

        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=CRITIC_LR,
        )

        self.replay = ReplayBuffer()
        self.ou_noise = OUNoise(ACTION_DIM)

        self.updates = 0

    def select_action(
        self,
        observation,
        exploration=False,
        noise_type="ou",
        noise_sigma=0.2,
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

        if exploration:

            if noise_type == "ou":

                noise = (
                    self.ou_noise.sample()
                    * noise_sigma
                    / max(OU_SIGMA_START, 1e-8)
                )

            elif noise_type == "gaussian":

                noise = gaussian_noise(
                    ACTION_DIM,
                    noise_sigma,
                )

            else:

                raise ValueError(
                    "Unknown exploration noise type"
                )

            action += noise

        return np.clip(
            action,
            -1.0,
            1.0,
        ).astype(np.float32)

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

    def update(self):

        if len(self.replay) < BATCH_SIZE:

            return None

        (
            states,
            actions,
            rewards,
            next_states,
            terminated,
        ) = self.replay.sample(BATCH_SIZE)

        with torch.no_grad():

            next_actions = self.target_actor(
                next_states
            )

            next_q = self.target_critic(
                next_states,
                next_actions,
            )

            targets = (
                rewards
                + GAMMA
                * (1.0 - terminated)
                * next_q
            )

        current_q = self.critic(
            states,
            actions,
        )

        critic_loss = nn.functional.mse_loss(
            current_q,
            targets,
        )

        self.critic_optimizer.zero_grad()
        critic_loss.backward()

        nn.utils.clip_grad_norm_(
            self.critic.parameters(),
            GRADIENT_CLIP,
        )

        self.critic_optimizer.step()

        for parameter in self.critic.parameters():
            parameter.requires_grad_(False)

        actor_loss = -self.critic(
            states,
            self.actor(states),
        ).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()

        nn.utils.clip_grad_norm_(
            self.actor.parameters(),
            GRADIENT_CLIP,
        )

        self.actor_optimizer.step()

        for parameter in self.critic.parameters():
            parameter.requires_grad_(True)

        DDPGAgent.soft_update(
            self.actor,
            self.target_actor,
        )

        DDPGAgent.soft_update(
            self.critic,
            self.target_critic,
        )

        self.updates += 1

        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
        }

    @staticmethod
    def soft_update(source, target):

        with torch.no_grad():

            for source_parameter, target_parameter in zip(
                source.parameters(),
                target.parameters(),
            ):

                target_parameter.mul_(1.0 - TAU)

                target_parameter.add_(
                    TAU * source_parameter
                )


# ============================================================
# 10. TD3 AGENT
# ============================================================

class TD3Agent:

    def __init__(self):

        self.actor = Actor().to(DEVICE)

        self.critic1 = Critic().to(DEVICE)
        self.critic2 = Critic().to(DEVICE)

        self.target_actor = copy.deepcopy(
            self.actor
        ).to(DEVICE)

        self.target_critic1 = copy.deepcopy(
            self.critic1
        ).to(DEVICE)

        self.target_critic2 = copy.deepcopy(
            self.critic2
        ).to(DEVICE)

        self.actor_optimizer = optim.Adam(
            self.actor.parameters(),
            lr=ACTOR_LR,
        )

        self.critic_optimizer = optim.Adam(
            list(self.critic1.parameters())
            + list(self.critic2.parameters()),
            lr=CRITIC_LR,
        )

        self.replay = ReplayBuffer()
        self.ou_noise = OUNoise(ACTION_DIM)

        self.updates = 0

    def select_action(
        self,
        observation,
        exploration=False,
        noise_type="ou",
        noise_sigma=0.2,
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

        if exploration:

            if noise_type == "ou":

                noise = (
                    self.ou_noise.sample()
                    * noise_sigma
                    / max(OU_SIGMA_START, 1e-8)
                )

            elif noise_type == "gaussian":

                noise = gaussian_noise(
                    ACTION_DIM,
                    noise_sigma,
                )

            else:

                raise ValueError(
                    "Unknown exploration noise type"
                )

            action += noise

        return np.clip(
            action,
            -1.0,
            1.0,
        ).astype(np.float32)

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

    def update(self):

        if len(self.replay) < BATCH_SIZE:

            return None

        (
            states,
            actions,
            rewards,
            next_states,
            terminated,
        ) = self.replay.sample(BATCH_SIZE)

        # ----------------------------------------------------
        # STEP 1 — TARGET POLICY SMOOTHING
        # ----------------------------------------------------

        with torch.no_grad():

            next_actions = self.target_actor(
                next_states
            )

            noise = torch.randn_like(
                next_actions
            ) * TARGET_POLICY_NOISE

            noise = noise.clamp(
                -TARGET_NOISE_CLIP,
                TARGET_NOISE_CLIP,
            )

            next_actions = (
                next_actions + noise
            ).clamp(-1.0, 1.0)

            # ------------------------------------------------
            # STEP 2 — TWIN CRITIC TARGET
            # ------------------------------------------------

            target_q1 = self.target_critic1(
                next_states,
                next_actions,
            )

            target_q2 = self.target_critic2(
                next_states,
                next_actions,
            )

            # The minimum is used to reduce overestimation.

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

        # ----------------------------------------------------
        # STEP 3 — UPDATE BOTH CRITICS
        # ----------------------------------------------------

        current_q1 = self.critic1(
            states,
            actions,
        )

        current_q2 = self.critic2(
            states,
            actions,
        )

        critic1_loss = nn.functional.mse_loss(
            current_q1,
            targets,
        )

        critic2_loss = nn.functional.mse_loss(
            current_q2,
            targets,
        )

        critic_loss = (
            critic1_loss + critic2_loss
        )

        self.critic_optimizer.zero_grad()

        critic_loss.backward()

        nn.utils.clip_grad_norm_(
            list(self.critic1.parameters())
            + list(self.critic2.parameters()),
            GRADIENT_CLIP,
        )

        self.critic_optimizer.step()

        self.updates += 1

        actor_loss_value = None

        # ----------------------------------------------------
        # STEP 4 — DELAYED ACTOR UPDATE
        # ----------------------------------------------------

        if self.updates % POLICY_DELAY == 0:

            for parameter in self.critic1.parameters():
                parameter.requires_grad_(False)

            actor_loss = -self.critic1(
                states,
                self.actor(states),
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

            actor_loss_value = actor_loss.item()

            # ------------------------------------------------
            # STEP 5 — DELAYED TARGET UPDATES
            # ------------------------------------------------

            DDPGAgent.soft_update(
                self.actor,
                self.target_actor,
            )

            DDPGAgent.soft_update(
                self.critic1,
                self.target_critic1,
            )

            DDPGAgent.soft_update(
                self.critic2,
                self.target_critic2,
            )

        return {
            "actor_loss": actor_loss_value,
            "critic_loss": critic_loss.item(),
            "q1_mean": current_q1.detach().mean().item(),
            "q2_mean": current_q2.detach().mean().item(),
            "actor_updated": (
                self.updates % POLICY_DELAY == 0
            ),
        }


# ============================================================
# 11. TRAINING UTILITIES
# ============================================================

def exploration_schedule(step):

    fraction = min(
        1.0,
        step / TRAINING_STEPS,
    )

    return (
        OU_SIGMA_START
        + fraction * (
            OU_SIGMA_END - OU_SIGMA_START
        )
    )


def evaluate(agent, episodes=EVALUATION_EPISODES):

    returns = []
    distances = []
    collisions = []

    for episode in range(episodes):

        env = DroneEnv(
            seed=20_000 + episode
        )

        observation = env.reset()

        total_reward = 0.0
        episode_distances = []
        collision_count = 0

        for _ in range(MAX_EPISODE_STEPS):

            action = agent.select_action(
                observation,
                exploration=False,
            )

            (
                observation,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(action)

            total_reward += reward

            episode_distances.append(
                info["distance"]
            )

            collision_count += int(
                info["collision"]
            )

            if terminated or truncated:
                break

        returns.append(total_reward)

        distances.append(
            np.mean(episode_distances)
        )

        collisions.append(
            collision_count
        )

    return {
        "return": float(np.mean(returns)),
        "distance": float(np.mean(distances)),
        "collisions": int(np.sum(collisions)),
    }


# ============================================================
# 12. TRAIN ONE ALGORITHM
# ============================================================

def train_agent(agent, algorithm_name):

    env = DroneEnv(
        seed=SEED + 1
    )

    history = {
        "episode_return": [],
        "episode_distance": [],
        "actor_loss": [],
        "critic_loss": [],
        "evaluation_step": [],
        "evaluation_return": [],
        "evaluation_distance": [],
        "evaluation_collisions": [],
        "q1_mean": [],
        "q2_mean": [],
    }

    observation = env.reset()

    episode_return = 0.0
    episode_distances = []

    for step in range(TRAINING_STEPS):

        sigma = exploration_schedule(step)

        if step < INITIAL_RANDOM_STEPS:

            action = np.random.uniform(
                -1.0,
                1.0,
                size=ACTION_DIM,
            ).astype(np.float32)

        else:

            action = agent.select_action(
                observation,
                exploration=True,
                noise_type=EXPLORATION_TYPE,
                noise_sigma=sigma,
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

            if metrics["actor_loss"] is not None:

                history["actor_loss"].append(
                    metrics["actor_loss"]
                )

            history["critic_loss"].append(
                metrics["critic_loss"]
            )

            if "q1_mean" in metrics:

                history["q1_mean"].append(
                    metrics["q1_mean"]
                )

                history["q2_mean"].append(
                    metrics["q2_mean"]
                )

        if terminated or truncated:

            history["episode_return"].append(
                episode_return
            )

            history["episode_distance"].append(
                float(np.mean(episode_distances))
            )

            if len(history["episode_return"]) % 5 == 0:

                print(
                    f"[{algorithm_name}]"
                    f" Step={step + 1:6d}"
                    f" Episodes={len(history['episode_return']):4d}"
                    f" Return={episode_return:8.2f}"
                    f" Distance={np.mean(episode_distances):7.2f}"
                )

            observation = env.reset()

            agent.ou_noise.reset()

            episode_return = 0.0
            episode_distances = []

        if (
            step > INITIAL_RANDOM_STEPS
            and (step + 1) % EVALUATION_INTERVAL == 0
        ):

            result = evaluate(agent)

            history["evaluation_step"].append(
                step + 1
            )

            history["evaluation_return"].append(
                result["return"]
            )

            history["evaluation_distance"].append(
                result["distance"]
            )

            history["evaluation_collisions"].append(
                result["collisions"]
            )

            print(
                f"\n[{algorithm_name} EVALUATION]"
                f" Step={step + 1}"
                f" Return={result['return']:.2f}"
                f" Distance={result['distance']:.2f}"
                f" Collisions={result['collisions']}\n"
            )

    return history


# ============================================================
# 13. COMPARISON VISUALIZATION
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
        np.ones(window, dtype=np.float32) / window,
        mode="valid",
    )


def plot_comparison(histories):

    for key, title in [
        (
            "episode_return",
            "DDPG vs TD3: Episode Return",
        ),
        (
            "episode_distance",
            "DDPG vs TD3: Target Tracking Distance",
        ),
        (
            "critic_loss",
            "DDPG vs TD3: Critic Loss",
        ),
    ]:

        plt.figure(figsize=(10, 4))

        for name, history in histories.items():

            values = history[key]

            if len(values) == 0:
                continue

            plt.plot(
                moving_average(values),
                label=name,
            )

        plt.title(title)
        plt.xlabel("Episode or update")
        plt.grid(True)
        plt.legend()
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
        "Deterministic Evaluation Distance"
    )

    plt.xlabel("Training steps")
    plt.ylabel("Mean target distance")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 4))

    for name, history in histories.items():

        if history["evaluation_step"]:

            plt.plot(
                history["evaluation_step"],
                history["evaluation_collisions"],
                marker="o",
                label=name,
            )

    plt.title(
        "Evaluation Collision Counts"
    )

    plt.xlabel("Training steps")
    plt.ylabel("Collisions across evaluation episodes")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# 14. PYGAME DRONE VISUALIZATION
# ============================================================

def run_pygame(agent, algorithm_name):

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
        f"Phase 5.13 — {algorithm_name} Drone"
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

    episode_reward = 0.0
    last_distance = 0.0

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

    def draw_drone():

        center = env.state[:2]
        velocity = env.state[2:4]

        magnitude = np.linalg.norm(velocity)

        if magnitude > 1e-5:

            direction = velocity / magnitude

        else:

            direction = np.array([1.0, 0.0])

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

        pygame.draw.line(
            screen,
            (90, 160, 255),
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

                    episode_reward = 0.0

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

                action = agent.select_action(
                    observation,
                    exploration=False,
                )

                (
                    observation,
                    reward,
                    terminated,
                    truncated,
                    info,
                ) = env.step(action)

                episode_reward += reward

                last_distance = info["distance"]

                trail.append(
                    env.state[:2].copy()
                )

                if len(trail) > 250:

                    trail.pop(0)

                if terminated or truncated:

                    observation = env.reset()

                    trail.clear()

                    episode_reward = 0.0

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
            f"PHASE 5.13 — {algorithm_name}",
            15,
            12,
        )

        draw_text(
            f"Target distance: {last_distance:.2f}",
            15,
            40,
        )

        draw_text(
            f"Episode reward: {episode_reward:.2f}",
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
# 15. NUMERICAL TESTS
# ============================================================

def run_tests():

    print("\nRunning TD3 tests...")

    passed = 0

    def check(condition, description):

        nonlocal passed

        assert condition, description

        passed += 1

        print("[PASS]", description)

    # Test 1: Actor dimensions.

    actor = Actor().to(DEVICE)

    states = torch.randn(
        16,
        OBS_DIM,
        device=DEVICE,
    )

    actions = actor(states)

    check(
        actions.shape == (16, ACTION_DIM),
        "Actor output dimensions",
    )

    # Test 2: Action bounds.

    check(
        bool(torch.all(actions.abs() <= 1.0)),
        "Actor action bounds",
    )

    # Test 3: Twin critic dimensions.

    critic1 = Critic().to(DEVICE)
    critic2 = Critic().to(DEVICE)

    q1 = critic1(states, actions)
    q2 = critic2(states, actions)

    check(
        q1.shape == (16, 1)
        and q2.shape == (16, 1),
        "Twin critic output dimensions",
    )

    # Test 4: Clipped Double Q calculation.

    q1_test = torch.tensor(
        [[5.0], [2.0]],
        device=DEVICE,
    )

    q2_test = torch.tensor(
        [[3.0], [4.0]],
        device=DEVICE,
    )

    minimum = torch.minimum(
        q1_test,
        q2_test,
    )

    check(
        torch.allclose(
            minimum,
            torch.tensor(
                [[3.0], [2.0]],
                device=DEVICE,
            ),
        ),
        "Clipped Double Q minimum",
    )

    # Test 5: Target policy noise clipping.

    noise = torch.tensor(
        [[0.8, -0.9]],
        device=DEVICE,
    )

    clipped_noise = noise.clamp(
        -TARGET_NOISE_CLIP,
        TARGET_NOISE_CLIP,
    )

    check(
        torch.allclose(
            clipped_noise,
            torch.tensor(
                [[0.5, -0.5]],
                device=DEVICE,
            ),
        ),
        "Target noise clipping",
    )

    # Test 6: Target action clipping.

    target_action = torch.tensor(
        [[0.9, -0.9]],
        device=DEVICE,
    )

    smoothed = (
        target_action + clipped_noise
    ).clamp(-1.0, 1.0)

    check(
        bool(torch.all(smoothed.abs() <= 1.0)),
        "Smoothed target action bounds",
    )

    # Test 7: Terminal Bellman target.

    reward = torch.tensor(
        [[2.0]],
        device=DEVICE,
    )

    terminal = torch.tensor(
        [[1.0]],
        device=DEVICE,
    )

    q_target = torch.tensor(
        [[5.0]],
        device=DEVICE,
    )

    target = (
        reward
        + GAMMA * (1.0 - terminal) * q_target
    )

    check(
        torch.allclose(target, reward),
        "Terminal state suppresses bootstrap",
    )

    # Test 8: Nonterminal Bellman target.

    nonterminal = torch.tensor(
        [[0.0]],
        device=DEVICE,
    )

    target = (
        reward
        + GAMMA * (1.0 - nonterminal) * q_target
    )

    expected = 2.0 + GAMMA * 5.0

    check(
        torch.allclose(
            target,
            torch.tensor(
                [[expected]],
                device=DEVICE,
            ),
        ),
        "Nonterminal Bellman target",
    )

    # Test 9: Delayed update frequency.

    update_flags = [
        (step % POLICY_DELAY == 0)
        for step in range(1, 7)
    ]

    check(
        update_flags == [
            False,
            True,
            False,
            True,
            False,
            True,
        ],
        "Delayed actor update frequency",
    )

    # Test 10: Soft update arithmetic.

    source = nn.Linear(3, 2)
    target_network = nn.Linear(3, 2)

    with torch.no_grad():

        for parameter in source.parameters():
            parameter.fill_(2.0)

        for parameter in target_network.parameters():
            parameter.fill_(0.0)

        for source_parameter, target_parameter in zip(
            source.parameters(),
            target_network.parameters(),
        ):

            target_parameter.mul_(0.75)
            target_parameter.add_(0.25 * source_parameter)

    check(
        all(
            torch.allclose(
                parameter,
                torch.full_like(parameter, 0.5),
            )
            for parameter in target_network.parameters()
        ),
        "Soft target update",
    )

    # Test 11: Actor gradients.

    actor.zero_grad()

    actor_loss = -critic1(
        states,
        actor(states),
    ).mean()

    actor_loss.backward()

    actor_grads = [
        p.grad
        for p in actor.parameters()
        if p.grad is not None
    ]

    check(
        len(actor_grads) > 0
        and all(
            bool(torch.isfinite(g).all())
            for g in actor_grads
        ),
        "Actor gradients",
    )

    # Test 12: Critic gradients.

    critic1.zero_grad()

    loss = critic1(
        states,
        actions.detach(),
    ).pow(2).mean()

    loss.backward()

    critic_grads = [
        p.grad
        for p in critic1.parameters()
        if p.grad is not None
    ]

    check(
        len(critic_grads) > 0
        and all(
            bool(torch.isfinite(g).all())
            for g in critic_grads
        ),
        "Critic gradients",
    )

    # Test 13: Replay buffer.

    replay = ReplayBuffer(100)

    for index in range(20):

        replay.add(
            np.zeros(OBS_DIM),
            np.zeros(ACTION_DIM),
            float(index),
            np.ones(OBS_DIM),
            False,
        )

    check(
        len(replay) == 20,
        "Replay buffer storage",
    )

    # Test 14: Replay batch dimensions.

    batch = replay.sample(8)

    check(
        batch[0].shape == (8, OBS_DIM)
        and batch[1].shape == (8, ACTION_DIM)
        and batch[2].shape == (8, 1)
        and batch[3].shape == (8, OBS_DIM)
        and batch[4].shape == (8, 1),
        "Replay batch dimensions",
    )

    # Test 15: OU noise.

    ou = OUNoise(ACTION_DIM)

    sample = ou.sample()

    check(
        sample.shape == (ACTION_DIM,)
        and np.isfinite(sample).all(),
        "OU noise sampling",
    )

    # Test 16: Environment observation.

    env = DroneEnv(seed=513)

    observation = env.reset()

    check(
        observation.shape == (OBS_DIM,)
        and np.isfinite(observation).all(),
        "Environment observation",
    )

    # Test 17: Environment transition.

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
        and np.isfinite(reward)
        and "distance" in info,
        "Environment transition",
    )

    # Test 18: Target velocity.

    check(
        env.target.velocity().shape == (2,)
        and np.isfinite(env.target.velocity()).all(),
        "Moving target velocity",
    )

    print(
        f"\n{passed}/18 tests passed."
    )

    return passed == 18


# ============================================================
# 16. MAIN
# ============================================================

def main():

    print("=" * 72)

    print(
        "PHASE 5.13 — TD3: TWIN DELAYED DDPG"
    )

    print("=" * 72)

    print("Device:", DEVICE)

    print("Exploration:", EXPLORATION_TYPE)

    if EXPLORATION_TYPE not in (
        "ou",
        "gaussian",
    ):

        raise ValueError(
            "EXPLORATION_TYPE must be 'ou' or 'gaussian'"
        )

    if not run_tests():

        raise RuntimeError(
            "TD3 tests failed."
        )

    print("\nTraining DDPG baseline...")

    set_seed(SEED)

    ddpg = DDPGAgent()

    ddpg_history = train_agent(
        ddpg,
        "DDPG",
    )

    print("\nTraining TD3...")

    set_seed(SEED)

    td3 = TD3Agent()

    td3_history = train_agent(
        td3,
        "TD3",
    )

    histories = {
        "DDPG": ddpg_history,
        "TD3": td3_history,
    }

    print("\nFinal evaluation:")

    print("DDPG:", evaluate(ddpg, episodes=8))

    print("TD3:", evaluate(td3, episodes=8))

    print("\nPlotting comparison...")

    plot_comparison(histories)

    print("\nChoose visualization algorithm:")
    print("1: DDPG")
    print("2: TD3")

    try:

        choice = input(
            "Enter 1 or 2 [default 2]: "
        ).strip()

    except EOFError:

        choice = "2"

    if choice == "1":

        run_pygame(
            ddpg,
            "DDPG",
        )

    else:

        run_pygame(
            td3,
            "TD3",
        )

    print("\nPhase 5.13 completed.")


if __name__ == "__main__":

    main()
