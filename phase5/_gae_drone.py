
"""
File: phase5/gae_drone.py

PHASE 5.5 — GENERALIZED ADVANTAGE ESTIMATION (GAE)
============================================================

PURPOSE
-------
Implement GAE for continuous-control moving-target drone
navigation using synchronous parallel environments.

WHAT YOU LEARN
--------------

STEP 1 — TD Residual
    Calculate one-step temporal-difference errors.

STEP 2 — GAE
    Combine future TD residuals using gamma and lambda.

STEP 3 — Episode Boundaries
    Distinguish true termination from time-limit truncation.

STEP 4 — Advantage Normalization
    Normalize advantages within each collected rollout.

STEP 5 — Actor-Critic Optimization
    Train a Gaussian policy and state-value function.

STEP 6 — Lambda Experiment
    Compare lambda = 0, 0.5, 0.95, and 1.0.

STEP 7 — Robotics Visualization
    Control a quadcopter tracking a moving target.

MATHEMATICS
-----------

TD residual:

    delta[t] = reward[t]
               + gamma * V(next_state[t]) * (1-terminated[t])
               - V(state[t])

GAE recursion:

    A[t] = delta[t]
           + gamma * lambda
             * (1-terminated[t])
             * (1-truncated[t])
             * A[t+1]

The bootstrap term in delta is removed only for TRUE
termination.

The recursive GAE term stops for both termination
and truncation, because the next transition belongs
to a different episode after an environment reset.

Value target:

    target[t] = A[t] + V(state[t])

Actor loss:

    L_actor = -mean(log_pi * stop_gradient(A))

Critic loss:

    L_critic = 0.5 * mean((V-target)^2)

Entropy regularization:

    L = L_actor + c1*L_critic - c2*entropy

IMPORTANT
---------

This is an educational GAE Actor-Critic implementation.
It is not PPO yet.

GAE is an advantage estimator. PPO will use the same
estimator together with a clipped policy objective.

ROBOTICS APPLICATION
--------------------

Train a simulated quadcopter to track a continuously
moving target while avoiding an obstacle.

Target position and velocity are included in the
observation.

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

SEED = 550

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

NUM_ENVS = 8
ROLLOUT_STEPS = 16

TRAIN_UPDATES = 1000

LEARNING_RATE = 3e-4

VALUE_COEF = 0.5
ENTROPY_COEF = 0.005

MAX_GRAD_NORM = 0.5

MAX_EPISODE_STEPS = 180

PRINT_INTERVAL = 20

# Quick experiment:
#
# NUM_ENVS = 4
# ROLLOUT_STEPS = 8
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
# 4. CORRECT VECTOR ENVIRONMENT
# ============================================================

class ParallelEnvironments:

    """
    The important detail:

    final_observations:
        The actual next observation from each transition,
        before any environment reset.

    next_observations:
        The observations used to start the next vector step.
        These may contain reset observations.

    GAE must use final_observations for bootstrapping.
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

        final_observations = []
        next_observations = []

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
# 6. GENERALIZED ADVANTAGE ESTIMATION
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
    """
    All arrays have shape [T, N].

    rewards[t,n]:
        Reward from transition t in environment n.

    values[t,n]:
        V(observation before transition).

    next_values[t,n]:
        V(actual final observation after transition),
        BEFORE environment reset.

    terminated[t,n]:
        True terminal state, such as a collision.

    truncated[t,n]:
        External time limit.

    Two different masks are essential:

    Bootstrap mask:
        1 - terminated

        A time-limit truncation still bootstraps.

    GAE recursion mask:
        1 - (terminated OR truncated)

        Do not carry advantage recursion into the
        next episode after an environment reset.
    """

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

    value_targets = (
        advantages + values
    )

    return advantages, value_targets


# ============================================================
# 7. A2C + GAE AGENT
# ============================================================

class GAEAgent:

    def __init__(self):

        self.network = ActorCritic().to(
            DEVICE
        )

        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=LEARNING_RATE,
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
                _,
                values,
            ) = self.network.sample(states)

        return (
            actions.cpu().numpy(),
            raw_actions.cpu().numpy(),
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

    def update(
        self,
        observations,
        raw_actions,
        advantages,
        value_targets,
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

        advantages = torch.as_tensor(
            advantages,
            dtype=torch.float32,
            device=DEVICE,
        )

        value_targets = torch.as_tensor(
            value_targets,
            dtype=torch.float32,
            device=DEVICE,
        )

        # Normalize advantages for more stable optimization.
        advantages = (
            advantages - advantages.mean()
        ) / (
            advantages.std(unbiased=False) + 1e-8
        )

        (
            log_probs,
            entropy,
            values,
        ) = self.network.evaluate_actions(
            states,
            raw_actions,
        )

        actor_loss = -(
            log_probs * advantages.detach()
        ).mean()

        critic_loss = 0.5 * (
            values - value_targets
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
            "advantage_std": float(
                advantages.detach().std().item()
            ),
        }

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


# ============================================================
# 8. TRAINING
# ============================================================

def train():

    envs = ParallelEnvironments(
        NUM_ENVS
    )

    agent = GAEAgent()

    observations = envs.reset()

    history = {
        "reward": [],
        "actor_loss": [],
        "critic_loss": [],
        "entropy": [],
        "advantage_std": [],
        "mean_distance": [],
    }

    episode_returns = np.zeros(
        NUM_ENVS,
        dtype=np.float32,
    )

    completed_returns = []

    for update_index in range(
        TRAIN_UPDATES
    ):

        rollout_obs = []
        rollout_raw_actions = []

        rollout_rewards = []
        rollout_values = []
        rollout_next_values = []

        rollout_terminated = []
        rollout_truncated = []

        rollout_distances = []

        for _ in range(
            ROLLOUT_STEPS
        ):

            actions, raw_actions, values = (
                agent.select_actions(
                    observations
                )
            )

            (
                next_observations,
                final_observations,
                rewards,
                terminated,
                truncated,
                infos,
            ) = envs.step(actions)

            # Evaluate the actual final observations.
            #
            # Do not use next_observations here:
            # those may already be reset observations.

            next_values = agent.predict_values(
                final_observations
            )

            rollout_obs.append(
                observations.copy()
            )

            rollout_raw_actions.append(
                raw_actions.copy()
            )

            rollout_rewards.append(
                rewards.copy()
            )

            rollout_values.append(
                values.copy()
            )

            rollout_next_values.append(
                next_values.copy()
            )

            rollout_terminated.append(
                terminated.copy()
            )

            rollout_truncated.append(
                truncated.copy()
            )

            rollout_distances.append(
                np.asarray(
                    [
                        info["distance"]
                        for info in infos
                    ],
                    dtype=np.float32,
                )
            )

            episode_returns += rewards

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
        # Convert rollout to [T,N] arrays.
        # ----------------------------------------------------

        rewards_array = np.asarray(
            rollout_rewards,
            dtype=np.float32,
        )

        values_array = np.asarray(
            rollout_values,
            dtype=np.float32,
        )

        next_values_array = np.asarray(
            rollout_next_values,
            dtype=np.float32,
        )

        terminated_array = np.asarray(
            rollout_terminated,
            dtype=np.float32,
        )

        truncated_array = np.asarray(
            rollout_truncated,
            dtype=np.float32,
        )

        # ----------------------------------------------------
        # GAE calculation.
        # ----------------------------------------------------

        advantages, value_targets = compute_gae(
            rewards_array,
            values_array,
            next_values_array,
            terminated_array,
            truncated_array,
            GAMMA,
            GAE_LAMBDA,
        )

        # ----------------------------------------------------
        # Flatten [T,N,...] into one training batch.
        # ----------------------------------------------------

        states_batch = np.asarray(
            rollout_obs,
            dtype=np.float32,
        ).reshape(
            -1,
            OBS_DIM,
        )

        actions_batch = np.asarray(
            rollout_raw_actions,
            dtype=np.float32,
        ).reshape(
            -1,
            ACTION_DIM,
        )

        advantages_batch = advantages.reshape(
            -1
        )

        targets_batch = value_targets.reshape(
            -1
        )

        metrics = agent.update(
            states_batch,
            actions_batch,
            advantages_batch,
            targets_batch,
        )

        history["reward"].append(
            float(rewards_array.mean())
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

        history["advantage_std"].append(
            metrics["advantage_std"]
        )

        history["mean_distance"].append(
            float(
                np.mean(rollout_distances)
            )
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
                f" | Reward={np.mean(history['reward'][recent]):8.2f}"
                f" | Distance={np.mean(history['mean_distance'][recent]):7.2f}"
                f" | Actor={np.mean(history['actor_loss'][recent]):8.3f}"
                f" | Critic={np.mean(history['critic_loss'][recent]):8.3f}"
            )

    return agent, history, completed_returns


# ============================================================
# 9. LAMBDA COMPARISON EXPERIMENT
# ============================================================

def lambda_experiment():

    """
    Compare GAE estimates for different lambda values
    on the SAME synthetic trajectory.

    This experiment isolates the estimator from policy
    learning and makes the bias/variance trade-off easier
    to inspect.
    """

    rewards = np.array(
        [
            [1.0],
            [0.5],
            [-0.2],
            [1.5],
            [0.7],
            [0.0],
            [1.0],
            [0.4],
        ],
        dtype=np.float32,
    )

    values = np.array(
        [
            [0.8],
            [0.9],
            [1.0],
            [1.1],
            [1.2],
            [1.0],
            [0.8],
            [0.5],
        ],
        dtype=np.float32,
    )

    next_values = np.array(
        [
            [0.9],
            [1.0],
            [1.1],
            [1.2],
            [1.0],
            [0.8],
            [0.5],
            [0.0],
        ],
        dtype=np.float32,
    )

    terminated = np.zeros(
        (8, 1),
        dtype=np.float32,
    )

    truncated = np.zeros(
        (8, 1),
        dtype=np.float32,
    )

    results = {}

    for lam in (
        0.0,
        0.5,
        0.95,
        1.0,
    ):

        advantages, targets = compute_gae(
            rewards,
            values,
            next_values,
            terminated,
            truncated,
            gamma=0.99,
            gae_lambda=lam,
        )

        results[lam] = advantages[:, 0]

        print(
            f"lambda={lam:4.2f}"
            f" | mean advantage={advantages.mean():8.4f}"
            f" | std advantage={advantages.std():8.4f}"
        )

    plt.figure(
        figsize=(10, 5)
    )

    for lam, advantages in results.items():

        plt.plot(
            advantages,
            marker="o",
            label=f"lambda={lam}",
        )

    plt.title(
        "GAE Advantage Estimates for Different Lambda Values"
    )

    plt.xlabel("Time Step")

    plt.ylabel("Advantage")

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.show()

    return results


# ============================================================
# 10. TRAINING VISUALIZATION
# ============================================================

def plot_training(history):

    plots = [
        ("reward", "Mean Rollout Reward"),
        ("mean_distance", "Mean Target Distance"),
        ("actor_loss", "Actor Loss"),
        ("critic_loss", "Critic Loss"),
        ("entropy", "Policy Entropy"),
        ("advantage_std", "Normalized Advantage Standard Deviation"),
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
            (
                total_reward,
                float(np.mean(distances)),
                float(collision),
            )
        )

    print("\nEvaluation")

    print(
        "Mean return:",
        round(
            np.mean(
                [x[0] for x in results]
            ),
            3,
        ),
    )

    print(
        "Mean target distance:",
        round(
            np.mean(
                [x[1] for x in results]
            ),
            3,
        ),
    )

    print(
        "Collision rate:",
        round(
            np.mean(
                [x[2] for x in results]
            ),
            3,
        ),
    )

    return results


# ============================================================
# 12. PYGAME QUADCOPTER VISUALIZATION
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
        "Phase 5.5 — GAE Moving-Target Drone"
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
            "PHASE 5.5 — GENERALIZED ADVANTAGE ESTIMATION",
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

    # Test 1:
    # One-step GAE equals the TD residual.

    rewards = np.array(
        [[1.0]],
        dtype=np.float32,
    )

    values = np.array(
        [[0.5]],
        dtype=np.float32,
    )

    next_values = np.array(
        [[2.0]],
        dtype=np.float32,
    )

    terminated = np.zeros(
        (1, 1),
        dtype=np.float32,
    )

    truncated = np.zeros_like(
        terminated
    )

    advantages, targets = compute_gae(
        rewards,
        values,
        next_values,
        terminated,
        truncated,
        gamma=0.9,
        gae_lambda=0.0,
    )

    # delta = 1 + 0.9*2 - 0.5 = 2.3

    check(
        np.isclose(
            advantages[0, 0],
            2.3,
        ),
        "Lambda zero gives one-step TD advantage",
    )

    # Test 2:
    # Value target = advantage + current value.

    check(
        np.isclose(
            targets[0, 0],
            2.8,
        ),
        "GAE value target",
    )

    # Test 3:
    # Two-step recursive GAE.

    rewards = np.array(
        [
            [1.0],
            [2.0],
        ],
        dtype=np.float32,
    )

    values = np.array(
        [
            [0.5],
            [0.7],
        ],
        dtype=np.float32,
    )

    next_values = np.array(
        [
            [0.7],
            [1.0],
        ],
        dtype=np.float32,
    )

    advantages, _ = compute_gae(
        rewards,
        values,
        next_values,
        np.zeros((2, 1)),
        np.zeros((2, 1)),
        gamma=0.9,
        gae_lambda=0.5,
    )

    # delta[0] = 1 + .9*.7 - .5 = 1.13
    # delta[1] = 2 + .9*1 - .7 = 2.20
    # A[1] = 2.20
    # A[0] = 1.13 + .9*.5*2.20 = 2.12

    check(
        np.isclose(
            advantages[0, 0],
            2.12,
        ),
        "Two-step recursive GAE",
    )

    # Test 4:
    # True terminal must not bootstrap.

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
        "True terminal suppresses bootstrap",
    )

    # Test 5:
    # Truncation must bootstrap but stops recursion.

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
        "Truncation bootstraps from final observation",
    )

    # Test 6:
    # GAE recursion stops at a truncation boundary.

    rewards = np.array(
        [
            [1.0],
            [100.0],
        ],
        dtype=np.float32,
    )

    values = np.zeros(
        (2, 1),
        dtype=np.float32,
    )

    next_values = np.zeros(
        (2, 1),
        dtype=np.float32,
    )

    truncated = np.array(
        [
            [1.0],
            [0.0],
        ],
        dtype=np.float32,
    )

    advantages, _ = compute_gae(
        rewards,
        values,
        next_values,
        np.zeros((2, 1)),
        truncated,
        gamma=0.9,
        gae_lambda=0.95,
    )

    check(
        np.isclose(
            advantages[0, 0],
            1.0,
        ),
        "GAE recursion stops at truncation",
    )

    # Test 7:
    # Lambda zero equals one-step TD.

    advantages, _ = compute_gae(
        np.array(
            [[1.0], [2.0]],
            dtype=np.float32,
        ),
        np.array(
            [[0.5], [0.7]],
            dtype=np.float32,
        ),
        np.array(
            [[0.7], [1.0]],
            dtype=np.float32,
        ),
        np.zeros((2, 1)),
        np.zeros((2, 1)),
        gamma=0.9,
        gae_lambda=0.0,
    )

    check(
        np.allclose(
            advantages[:, 0],
            [1.13, 2.20],
        ),
        "Lambda zero matches TD residuals",
    )

    # Test 8:
    # Lambda one produces the full finite-horizon
    # sum of TD residuals, with the supplied bootstrap.

    advantages, _ = compute_gae(
        np.array(
            [[1.0], [2.0]],
            dtype=np.float32,
        ),
        np.array(
            [[0.5], [0.7]],
            dtype=np.float32,
        ),
        np.array(
            [[0.7], [1.0]],
            dtype=np.float32,
        ),
        np.zeros((2, 1)),
        np.zeros((2, 1)),
        gamma=0.9,
        gae_lambda=1.0,
    )

    check(
        np.isclose(
            advantages[0, 0],
            1.13 + 0.9 * 2.20,
        ),
        "Lambda one accumulates full TD residuals",
    )

    # Test 9:
    # Shape preservation.

    rewards = np.zeros(
        (5, 3),
        dtype=np.float32,
    )

    advantages, targets = compute_gae(
        rewards,
        rewards,
        rewards,
        rewards,
        rewards,
    )

    check(
        advantages.shape == (5, 3)
        and targets.shape == (5, 3),
        "GAE batch dimensions",
    )

    # Test 10:
    # Independent environments do not share recursion.

    rewards = np.array(
        [
            [1.0, 10.0],
            [2.0, 20.0],
        ],
        dtype=np.float32,
    )

    advantages, _ = compute_gae(
        rewards,
        np.zeros_like(rewards),
        np.zeros_like(rewards),
        np.zeros_like(rewards),
        np.zeros_like(rewards),
        gamma=1.0,
        gae_lambda=1.0,
    )

    check(
        np.allclose(
            advantages[:, 0],
            [3.0, 2.0],
        )
        and np.allclose(
            advantages[:, 1],
            [30.0, 20.0],
        ),
        "Parallel environments have independent GAE",
    )

    # Test 11:
    # Vector environment shapes and final observations.

    envs = ParallelEnvironments(3)

    obs = envs.reset()

    result = envs.step(
        np.zeros(
            (3, ACTION_DIM),
            dtype=np.float32,
        )
    )

    check(
        obs.shape == (3, OBS_DIM)
        and result[0].shape == (3, OBS_DIM)
        and result[1].shape == (3, OBS_DIM),
        "Vector environment preserves final observations",
    )

    # Test 12:
    # Actor-critic output shapes.

    network = ActorCritic().to(
        DEVICE
    )

    states = torch.zeros(
        (3, OBS_DIM),
        dtype=torch.float32,
        device=DEVICE,
    )

    mean, log_std, value = network(states)

    check(
        mean.shape == (3, ACTION_DIM)
        and log_std.shape == (ACTION_DIM,)
        and value.shape == (3,),
        "Actor-critic output dimensions",
    )

    # Test 13:
    # Actions are bounded.

    actions, raw_actions, log_probs, _ = (
        network.sample(states)
    )

    check(
        bool(
            torch.all(
                actions.abs() <= 1.0
            )
        ),
        "Continuous actions are bounded",
    )

    # Test 14:
    # Log probability is finite.

    check(
        bool(
            torch.isfinite(log_probs).all()
        ),
        "Policy log probability is finite",
    )

    # Test 15:
    # Entropy is finite.

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
        "Policy entropy is finite",
    )

    # Test 16:
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

    # Test 17:
    # Target velocity is represented.

    target = MovingTarget(
        [400.0, 300.0]
    )

    check(
        target.velocity().shape == (2,),
        "Moving target velocity",
    )

    # Test 18:
    # Optimizer can update the network.

    optimizer = optim.Adam(
        network.parameters(),
        lr=1e-4,
    )

    log_prob, entropy, values = (
        network.evaluate_actions(
            states,
            raw_actions,
        )
    )

    advantages = torch.ones_like(
        values
    )

    targets = torch.ones_like(
        values
    )

    loss = (
        -(log_prob * advantages).mean()
        + 0.5 * (values - targets).pow(2).mean()
        - 0.005 * entropy.mean()
    )

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    check(
        bool(
            torch.isfinite(loss)
        ),
        "Actor-critic optimization",
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
        "PHASE 5.5 — GENERALIZED ADVANTAGE ESTIMATION"
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
        f"GAE lambda: {GAE_LAMBDA}"
    )

    if not run_tests():

        raise RuntimeError(
            "Numerical verification failed."
        )

    print(
        "\n[1/5] Comparing GAE lambda values..."
    )

    lambda_experiment()

    print(
        "\n[2/5] Training GAE Actor-Critic..."
    )

    agent, history, completed_returns = train()

    print(
        "\n[3/5] Evaluating learned policy..."
    )

    evaluate(agent)

    print(
        "\n[4/5] Visualizing training..."
    )

    plot_training(history)

    print(
        "\n[5/5] Starting interactive quadcopter..."
    )

    run_pygame(agent)

    print(
        "\nPhase 5.5 completed."
    )


if __name__ == "__main__":

    main()
