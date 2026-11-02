
"""
RL_Course/phase5/parallel_ppo_drone.py

PHASE 5.17 — PARALLEL PPO FOR CONTINUOUS DRONE CONTROL

PURPOSE
-------
Train PPO using multiple independent drone environments in parallel.
Compare the data collection and optimization structure with single-
environment PPO.

WHAT YOU LEARN
--------------
Step 1: Build independent moving-target drone environments.
Step 2: Run multiple environments synchronously as a vectorized batch.
Step 3: Collect Gaussian-policy actions and preserve the exact raw actions.
Step 4: Distinguish true termination from time-limit truncation.
Step 5: Calculate GAE independently for every environment.
Step 6: Flatten the [time, environment] rollout into PPO minibatches.
Step 7: Optimize the clipped PPO objective.
Step 8: Evaluate the trained policy and visualize training statistics.
Step 9: Control an interactive quadcopter with a moving target.

IMPLEMENTATION
--------------
- PyTorch actor-critic network
- Squashed Gaussian continuous-action policy
- Parallel synchronous environment collection
- Generalized Advantage Estimation (GAE)
- PPO clipped surrogate objective
- Value-function loss
- Entropy regularization
- Minibatch updates
- Correct terminal/truncation handling
- Pygame quadcopter visualization

IMPORTANT MATHEMATICS
---------------------
For each environment n and rollout time t:

delta[t,n] =
    reward[t,n]
    + gamma * (1 - terminated[t,n]) * next_value[t,n]
    - value[t,n]

GAE:
    A[t,n] =
        delta[t,n]
        + gamma * lambda * (1 - boundary[t,n]) * A[t+1,n]

where:
    boundary = terminated OR truncated

A true terminal state suppresses value bootstrapping.
A time-limit truncation bootstraps from its final observation.
Both terminal and truncated states stop GAE recursion across
the episode boundary.

PPO:
    ratio = exp(new_log_prob - old_log_prob)

    L_clip = min(
        ratio * advantage,
        clip(ratio, 1-epsilon, 1+epsilon) * advantage
    )

The raw Gaussian action is stored directly. It is NOT reconstructed
from the squashed action during PPO optimization.

ROBOTICS APPLICATION
--------------------
Parallel environments represent multiple simulated drones or
multiple simulated operating conditions.

This can improve data collection efficiency and expose the policy
to different target phases and initial conditions.

This educational 2D model is not flight-certified.
"""

import math
import random
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# ============================================================
# 1. GLOBAL CONFIGURATION
# ============================================================

SEED = 42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

WORLD_W = 900
WORLD_H = 650

DT = 0.08

MAX_SPEED = 180.0
MAX_ACCEL = 230.0

DRONE_RADIUS = 12.0

OBSTACLE_POS = np.array([450.0, 330.0], dtype=np.float32)
OBSTACLE_RADIUS = 55.0
SAFE_RADIUS = 105.0

TARGET_A = 145.0
TARGET_B = 95.0
TARGET_OMEGA = 0.65

GAMMA = 0.99
GAE_LAMBDA = 0.95

PPO_CLIP = 0.20
VALUE_COEF = 0.50
ENTROPY_COEF = 0.005

LEARNING_RATE = 3e-4
MAX_GRAD_NORM = 0.5

NUM_ENVS = 8
ROLLOUT_STEPS = 128

PPO_EPOCHS = 6
MINIBATCH_SIZE = 256

TOTAL_UPDATES = 250

MAX_EPISODE_STEPS = 350

ACTION_DIM = 2
OBS_DIM = 10

HIDDEN_SIZE = 128


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 3. MOVING TARGET
# ============================================================

class MovingTarget:

    def __init__(self, center=None, phase=0.0):

        if center is None:
            center = np.array(
                [680.0, 320.0],
                dtype=np.float32
            )

        self.center = np.array(
            center,
            dtype=np.float32
        )

        self.phase = float(phase)

    def position(self, phase=None):

        if phase is None:
            phase = self.phase

        return self.center + np.array(
            [
                TARGET_A * math.cos(phase),
                TARGET_B * math.sin(phase)
            ],
            dtype=np.float32
        )

    def velocity(self, phase=None):

        if phase is None:
            phase = self.phase

        return np.array(
            [
                -TARGET_A * TARGET_OMEGA * math.sin(phase),
                TARGET_B * TARGET_OMEGA * math.cos(phase)
            ],
            dtype=np.float32
        )

    def advance(self):

        self.phase += TARGET_OMEGA * DT


# ============================================================
# 4. DRONE ENVIRONMENT
# ============================================================

class DroneEnv:

    def __init__(self, seed=None):

        self.rng = np.random.default_rng(seed)

        self.target_center = np.array(
            [680.0, 320.0],
            dtype=np.float32
        )

        self.reset()

    def reset(self):

        self.position = np.array(
            [
                self.rng.uniform(100, 180),
                self.rng.uniform(100, 550)
            ],
            dtype=np.float32
        )

        self.velocity = np.zeros(
            2,
            dtype=np.float32
        )

        self.target = MovingTarget(
            center=self.target_center,
            phase=self.rng.uniform(0, 2 * math.pi)
        )

        self.steps = 0
        self.previous_distance = None

        return self.get_observation()

    def set_target_center(self, center):

        self.target_center = np.array(
            center,
            dtype=np.float32
        )

        self.target.center = self.target_center.copy()

    def get_observation(self):

        target_pos = self.target.position()
        target_vel = self.target.velocity()

        relative_target = (
            target_pos - self.position
        ) / np.array(
            [WORLD_W, WORLD_H],
            dtype=np.float32
        )

        target_velocity = target_vel / MAX_SPEED

        drone_velocity = self.velocity / MAX_SPEED

        relative_obstacle = (
            OBSTACLE_POS - self.position
        ) / np.array(
            [WORLD_W, WORLD_H],
            dtype=np.float32
        )

        phase_features = np.array(
            [
                math.sin(self.target.phase),
                math.cos(self.target.phase)
            ],
            dtype=np.float32
        )

        observation = np.concatenate(
            [
                relative_target,
                target_velocity,
                drone_velocity,
                relative_obstacle,
                phase_features
            ]
        ).astype(np.float32)

        return observation

    def step(self, action):

        action = np.asarray(
            action,
            dtype=np.float32
        )

        action = np.clip(action, -1.0, 1.0)

        acceleration = action * MAX_ACCEL

        future_target = self.target.position(
            self.target.phase + TARGET_OMEGA * DT
        )

        # Simple acceleration-controlled planar drone model.
        self.velocity += acceleration * DT

        speed = np.linalg.norm(self.velocity)

        if speed > MAX_SPEED:
            self.velocity *= MAX_SPEED / (speed + 1e-8)

        self.position += self.velocity * DT

        # Advance target after calculating its future position.
        self.target.advance()

        distance_before = np.linalg.norm(
            self.position - self.velocity * DT - future_target
        )

        distance_after = np.linalg.norm(
            self.position - future_target
        )

        progress = distance_before - distance_after

        reward = (
            0.08 * progress
            - 0.002 * np.linalg.norm(acceleration)
            - 0.01
        )

        obstacle_distance = np.linalg.norm(
            self.position - OBSTACLE_POS
        )

        if obstacle_distance < SAFE_RADIUS:
            penetration = SAFE_RADIUS - obstacle_distance
            reward -= 0.025 * penetration

        boundary_penalty = 0.0

        if self.position[0] < 25:
            boundary_penalty += 25 - self.position[0]

        if self.position[0] > WORLD_W - 25:
            boundary_penalty += self.position[0] - (WORLD_W - 25)

        if self.position[1] < 25:
            boundary_penalty += 25 - self.position[1]

        if self.position[1] > WORLD_H - 25:
            boundary_penalty += self.position[1] - (WORLD_H - 25)

        reward -= 0.05 * boundary_penalty

        collision = (
            obstacle_distance
            < OBSTACLE_RADIUS + DRONE_RADIUS
        )

        outside = (
            self.position[0] < DRONE_RADIUS
            or self.position[0] > WORLD_W - DRONE_RADIUS
            or self.position[1] < DRONE_RADIUS
            or self.position[1] > WORLD_H - DRONE_RADIUS
        )

        terminated = bool(collision or outside)

        if collision:
            reward -= 100.0

        if outside:
            reward -= 100.0

        self.steps += 1

        truncated = (
            self.steps >= MAX_EPISODE_STEPS
            and not terminated
        )

        observation = self.get_observation()

        info = {
            "distance": float(distance_after),
            "collision": collision,
            "success": distance_after < 28.0
        }

        return (
            observation,
            float(reward),
            terminated,
            truncated,
            info
        )


# ============================================================
# 5. PARALLEL ENVIRONMENTS
# ============================================================

class ParallelEnvironments:

    def __init__(self, num_envs, seed=SEED):

        self.envs = [
            DroneEnv(seed + i)
            for i in range(num_envs)
        ]

        self.num_envs = num_envs

    def reset(self):

        observations = [
            env.reset()
            for env in self.envs
        ]

        return np.stack(observations)

    def step(self, actions):

        observations = []
        rewards = []
        terminateds = []
        truncateds = []
        final_observations = []
        infos = []

        for env, action in zip(self.envs, actions):

            obs, reward, terminated, truncated, info = (
                env.step(action)
            )

            boundary = terminated or truncated

            # Preserve the final observation before auto-reset.
            final_obs = obs.copy()

            if boundary:
                next_obs = env.reset()
            else:
                next_obs = obs

            observations.append(next_obs)
            rewards.append(reward)
            terminateds.append(terminated)
            truncateds.append(truncated)
            final_observations.append(final_obs)
            infos.append(info)

        return (
            np.stack(observations),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(terminateds, dtype=np.float32),
            np.asarray(truncateds, dtype=np.float32),
            np.stack(final_observations),
            infos
        )


# ============================================================
# 6. ACTOR-CRITIC NETWORK
# ============================================================

class ActorCritic(nn.Module):

    def __init__(self):

        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(OBS_DIM, HIDDEN_SIZE),
            nn.Tanh(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.Tanh()
        )

        self.actor_mean = nn.Linear(
            HIDDEN_SIZE,
            ACTION_DIM
        )

        self.critic = nn.Linear(
            HIDDEN_SIZE,
            1
        )

        self.log_std = nn.Parameter(
            torch.full(
                (ACTION_DIM,),
                -0.5
            )
        )

    def forward(self, observation):

        features = self.shared(observation)

        mean = self.actor_mean(features)

        value = self.critic(features).squeeze(-1)

        std = torch.exp(
            torch.clamp(
                self.log_std,
                -5.0,
                1.0
            )
        )

        return mean, std, value

    @staticmethod
    def corrected_log_prob(distribution, raw_action):

        # For a = tanh(u):
        # log pi(a|s) = log N(u|mu,sigma)
        #               - sum log(1 - tanh(u)^2)
        #
        # The raw Gaussian action is stored directly in the rollout.

        gaussian_log_prob = distribution.log_prob(
            raw_action
        ).sum(dim=-1)

        correction = torch.log(
            1.0 - torch.tanh(raw_action).pow(2) + 1e-6
        ).sum(dim=-1)

        return gaussian_log_prob - correction

    def sample_action(self, observation):

        mean, std, value = self(observation)

        distribution = torch.distributions.Normal(
            mean,
            std
        )

        raw_action = distribution.sample()

        action = torch.tanh(raw_action)

        log_prob = self.corrected_log_prob(
            distribution,
            raw_action
        )

        return action, raw_action, log_prob, value

    def evaluate_actions(self, observations, raw_actions):

        mean, std, values = self(observations)

        distribution = torch.distributions.Normal(
            mean,
            std
        )

        log_probs = self.corrected_log_prob(
            distribution,
            raw_actions
        )

        # Base Gaussian entropy is used as a stable exploration
        # regularizer. It is not the exact entropy after tanh.
        entropy = distribution.entropy().sum(dim=-1)

        return log_probs, entropy, values


# ============================================================
# 7. GAE
# ============================================================

def calculate_gae(
    rewards,
    values,
    next_values,
    terminated,
    boundaries,
    gamma=GAMMA,
    gae_lambda=GAE_LAMBDA
):

    """
    All inputs have shape [T, N].

    terminated:
        True terminal state. No value bootstrap.

    boundaries:
        terminated OR truncated.
        Stops GAE recursion across episode resets.

    next_values:
        Value of the actual next observation.
        For a truncated episode, this must be the final observation
        before the environment reset.
    """

    T, N = rewards.shape

    advantages = np.zeros(
        (T, N),
        dtype=np.float32
    )

    gae = np.zeros(
        N,
        dtype=np.float32
    )

    for t in reversed(range(T)):

        bootstrap_mask = 1.0 - terminated[t]

        continuation_mask = 1.0 - boundaries[t]

        delta = (
            rewards[t]
            + gamma * bootstrap_mask * next_values[t]
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

    returns = advantages + values

    return advantages, returns


# ============================================================
# 8. ROLLOUT BUFFER
# ============================================================

@dataclass
class Rollout:

    observations: np.ndarray
    raw_actions: np.ndarray
    old_log_probs: np.ndarray
    rewards: np.ndarray
    values: np.ndarray
    next_values: np.ndarray
    terminated: np.ndarray
    boundaries: np.ndarray


# ============================================================
# 9. PPO AGENT
# ============================================================

class PPOAgent:

    def __init__(self):

        self.network = ActorCritic().to(DEVICE)

        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=LEARNING_RATE
        )

    @torch.no_grad()
    def act(self, observations):

        obs = torch.as_tensor(
            observations,
            dtype=torch.float32,
            device=DEVICE
        )

        action, raw_action, log_prob, value = (
            self.network.sample_action(obs)
        )

        return (
            action.cpu().numpy(),
            raw_action.cpu().numpy(),
            log_prob.cpu().numpy(),
            value.cpu().numpy()
        )

    @torch.no_grad()
    def predict_values(self, observations):

        obs = torch.as_tensor(
            observations,
            dtype=torch.float32,
            device=DEVICE
        )

        _, _, values = self.network(obs)

        return values.cpu().numpy()

    def update(self, rollout):

        observations = torch.as_tensor(
            rollout.observations.reshape(-1, OBS_DIM),
            dtype=torch.float32,
            device=DEVICE
        )

        raw_actions = torch.as_tensor(
            rollout.raw_actions.reshape(-1, ACTION_DIM),
            dtype=torch.float32,
            device=DEVICE
        )

        old_log_probs = torch.as_tensor(
            rollout.old_log_probs.reshape(-1),
            dtype=torch.float32,
            device=DEVICE
        )

        advantages, returns = calculate_gae(
            rollout.rewards,
            rollout.values,
            rollout.next_values,
            rollout.terminated,
            rollout.boundaries
        )

        advantages = torch.as_tensor(
            advantages.reshape(-1),
            dtype=torch.float32,
            device=DEVICE
        )

        returns = torch.as_tensor(
            returns.reshape(-1),
            dtype=torch.float32,
            device=DEVICE
        )

        # Normalize advantages over the complete rollout.
        advantages = (
            advantages - advantages.mean()
        ) / (
            advantages.std(unbiased=False) + 1e-8
        )

        total_samples = observations.shape[0]

        policy_losses = []
        value_losses = []
        entropy_values = []
        approximate_kls = []

        for _ in range(PPO_EPOCHS):

            indices = torch.randperm(
                total_samples,
                device=DEVICE
            )

            for start in range(
                0,
                total_samples,
                MINIBATCH_SIZE
            ):

                batch_indices = indices[
                    start:start + MINIBATCH_SIZE
                ]

                batch_obs = observations[batch_indices]
                batch_actions = raw_actions[batch_indices]

                batch_old_log = old_log_probs[batch_indices]
                batch_adv = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                new_log_probs, entropy, new_values = (
                    self.network.evaluate_actions(
                        batch_obs,
                        batch_actions
                    )
                )

                log_ratio = (
                    new_log_probs - batch_old_log
                )

                ratio = torch.exp(log_ratio)

                unclipped = ratio * batch_adv

                clipped = torch.clamp(
                    ratio,
                    1.0 - PPO_CLIP,
                    1.0 + PPO_CLIP
                ) * batch_adv

                policy_loss = -torch.minimum(
                    unclipped,
                    clipped
                ).mean()

                value_loss = (
                    batch_returns - new_values
                ).pow(2).mean()

                entropy_bonus = entropy.mean()

                loss = (
                    policy_loss
                    + VALUE_COEF * value_loss
                    - ENTROPY_COEF * entropy_bonus
                )

                self.optimizer.zero_grad()

                loss.backward()

                nn.utils.clip_grad_norm_(
                    self.network.parameters(),
                    MAX_GRAD_NORM
                )

                self.optimizer.step()

                with torch.no_grad():

                    approximate_kl = (
                        (ratio - 1.0) - log_ratio
                    ).mean()

                policy_losses.append(
                    float(policy_loss.item())
                )

                value_losses.append(
                    float(value_loss.item())
                )

                entropy_values.append(
                    float(entropy_bonus.item())
                )

                approximate_kls.append(
                    float(approximate_kl.item())
                )

        return {
            "policy_loss": float(np.mean(policy_losses)),
            "value_loss": float(np.mean(value_losses)),
            "entropy": float(np.mean(entropy_values)),
            "approx_kl": float(np.mean(approximate_kls))
        }


# ============================================================
# 10. PARALLEL ROLLOUT COLLECTION
# ============================================================

def collect_rollout(agent, vector_env, observations):

    obs_buffer = []
    raw_action_buffer = []
    log_prob_buffer = []
    reward_buffer = []
    value_buffer = []
    next_value_buffer = []
    terminated_buffer = []
    boundary_buffer = []

    for _ in range(ROLLOUT_STEPS):

        actions, raw_actions, log_probs, values = (
            agent.act(observations)
        )

        (
            next_observations,
            rewards,
            terminated,
            truncated,
            final_observations,
            infos
        ) = vector_env.step(actions)

        boundaries = np.logical_or(
            terminated > 0.5,
            truncated > 0.5
        ).astype(np.float32)

        # Value targets must use the actual next observation:
        # - normal transition: returned next observation
        # - truncation: final observation before reset
        # - termination: value will be masked to zero by GAE
        bootstrap_observations = next_observations.copy()

        for i in range(vector_env.num_envs):

            if truncated[i] > 0.5:
                bootstrap_observations[i] = (
                    final_observations[i]
                )

            elif terminated[i] > 0.5:
                bootstrap_observations[i] = (
                    final_observations[i]
                )

        next_values = agent.predict_values(
            bootstrap_observations
        )

        obs_buffer.append(observations.copy())
        raw_action_buffer.append(raw_actions.copy())
        log_prob_buffer.append(log_probs.copy())
        reward_buffer.append(rewards.copy())
        value_buffer.append(values.copy())
        next_value_buffer.append(next_values.copy())
        terminated_buffer.append(terminated.copy())
        boundary_buffer.append(boundaries.copy())

        observations = next_observations

    rollout = Rollout(
        observations=np.asarray(obs_buffer, dtype=np.float32),
        raw_actions=np.asarray(raw_action_buffer, dtype=np.float32),
        old_log_probs=np.asarray(log_prob_buffer, dtype=np.float32),
        rewards=np.asarray(reward_buffer, dtype=np.float32),
        values=np.asarray(value_buffer, dtype=np.float32),
        next_values=np.asarray(next_value_buffer, dtype=np.float32),
        terminated=np.asarray(terminated_buffer, dtype=np.float32),
        boundaries=np.asarray(boundary_buffer, dtype=np.float32)
    )

    return rollout, observations


# ============================================================
# 11. TRAINING
# ============================================================

def train():

    set_seed(SEED)

    vector_env = ParallelEnvironments(
        NUM_ENVS,
        seed=SEED
    )

    observations = vector_env.reset()

    agent = PPOAgent()

    history = {
        "reward": [],
        "policy_loss": [],
        "value_loss": [],
        "entropy": [],
        "kl": []
    }

    episode_returns = np.zeros(
        NUM_ENVS,
        dtype=np.float32
    )

    recent_episode_returns = []

    start_time = time.time()

    for update in range(TOTAL_UPDATES):

        rollout, observations = collect_rollout(
            agent,
            vector_env,
            observations
        )

        # Approximate episodic returns from the collected transitions.
        episode_returns += rollout.rewards.sum(axis=0)

        for env_index in range(NUM_ENVS):

            if np.any(
                rollout.boundaries[:, env_index] > 0.5
            ):

                recent_episode_returns.append(
                    float(episode_returns[env_index])
                )

                episode_returns[env_index] = 0.0

        metrics = agent.update(rollout)

        mean_reward = (
            float(np.mean(recent_episode_returns[-30:]))
            if recent_episode_returns
            else float(np.mean(rollout.rewards))
        )

        history["reward"].append(mean_reward)
        history["policy_loss"].append(metrics["policy_loss"])
        history["value_loss"].append(metrics["value_loss"])
        history["entropy"].append(metrics["entropy"])
        history["kl"].append(metrics["approx_kl"])

        if (update + 1) % 10 == 0:

            print(
                f"Update {update + 1:4d}/{TOTAL_UPDATES} | "
                f"Reward {mean_reward:8.3f} | "
                f"Policy {metrics['policy_loss']:8.4f} | "
                f"Value {metrics['value_loss']:8.4f} | "
                f"KL {metrics['approx_kl']:.5f}"
            )

    elapsed = time.time() - start_time

    print("\nTraining complete.")
    print(f"Elapsed time: {elapsed:.2f} seconds")
    print(f"Device: {DEVICE}")
    print(f"Parallel environments: {NUM_ENVS}")
    print(
        f"Transitions collected: "
        f"{TOTAL_UPDATES * ROLLOUT_STEPS * NUM_ENVS}"
    )

    return agent, history


# ============================================================
# 12. EVALUATION
# ============================================================

@torch.no_grad()
def evaluate(agent, episodes=10):

    results = []

    for episode in range(episodes):

        env = DroneEnv(seed=10000 + episode)

        obs = env.reset()

        total_reward = 0.0
        collisions = 0
        final_distance = None

        for _ in range(MAX_EPISODE_STEPS):

            obs_tensor = torch.as_tensor(
                obs,
                dtype=torch.float32,
                device=DEVICE
            )

            mean, std, _ = agent.network(obs_tensor)

            action = torch.tanh(mean)

            obs, reward, terminated, truncated, info = (
                env.step(action.cpu().numpy())
            )

            total_reward += reward

            collisions += int(info["collision"])

            final_distance = info["distance"]

            if terminated or truncated:
                break

        results.append(
            {
                "reward": total_reward,
                "distance": final_distance,
                "collision": collisions
            }
        )

    print("\nEvaluation:")

    print(
        "Mean reward:",
        np.mean([r["reward"] for r in results])
    )

    print(
        "Mean final distance:",
        np.mean([r["distance"] for r in results])
    )

    print(
        "Collision count:",
        sum(r["collision"] for r in results)
    )

    return results


# ============================================================
# 13. TRAINING VISUALIZATION
# ============================================================

def visualize_training(history):

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(13, 8)
    )

    axes[0, 0].plot(history["reward"])
    axes[0, 0].set_title("Episode reward")
    axes[0, 0].set_xlabel("PPO update")
    axes[0, 0].grid(True)

    axes[0, 1].plot(history["policy_loss"])
    axes[0, 1].set_title("Policy loss")
    axes[0, 1].grid(True)

    axes[1, 0].plot(history["value_loss"])
    axes[1, 0].set_title("Value loss")
    axes[1, 0].grid(True)

    axes[1, 1].plot(history["entropy"])
    axes[1, 1].set_title("Policy entropy")
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(9, 4))
    plt.plot(history["kl"])
    plt.title("Approximate KL divergence")
    plt.xlabel("PPO update")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ============================================================
# 14. PYGAME QUADCOPTER VISUALIZATION
# ============================================================

def run_pygame(agent):

    try:
        import pygame
    except ImportError:
        print("Pygame is not installed.")
        print("Install with: pip install pygame")
        return

    pygame.init()

    screen = pygame.display.set_mode(
        (WORLD_W, WORLD_H)
    )

    pygame.display.set_caption(
        "Parallel PPO — Moving Target Quadcopter"
    )

    clock = pygame.time.Clock()

    font = pygame.font.SysFont(
        "Arial",
        19
    )

    env = DroneEnv(seed=2026)

    obs = env.reset()

    paused = False
    simulation_speed = 1

    trail = []

    target_path = []

    running = True

    while running:

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:

                if event.key == pygame.K_ESCAPE:
                    running = False

                elif event.key == pygame.K_SPACE:
                    paused = not paused

                elif event.key == pygame.K_r:

                    obs = env.reset()
                    trail.clear()
                    target_path.clear()

                elif event.key in (
                    pygame.K_EQUALS,
                    pygame.K_PLUS,
                    pygame.K_KP_PLUS
                ):
                    simulation_speed = min(
                        8,
                        simulation_speed + 1
                    )

                elif event.key in (
                    pygame.K_MINUS,
                    pygame.K_KP_MINUS
                ):
                    simulation_speed = max(
                        1,
                        simulation_speed - 1
                    )

            elif event.type == pygame.MOUSEBUTTONDOWN:

                if event.button == 1:

                    mouse_x, mouse_y = event.pos

                    # Keep the entire target ellipse within the screen.
                    center_x = np.clip(
                        mouse_x,
                        TARGET_A + 20,
                        WORLD_W - TARGET_A - 20
                    )

                    center_y = np.clip(
                        mouse_y,
                        TARGET_B + 20,
                        WORLD_H - TARGET_B - 20
                    )

                    env.set_target_center(
                        [center_x, center_y]
                    )

                    target_path.clear()

        if not paused:

            for _ in range(simulation_speed):

                obs_tensor = torch.as_tensor(
                    obs,
                    dtype=torch.float32,
                    device=DEVICE
                )

                with torch.no_grad():

                    mean, std, value = agent.network(
                        obs_tensor
                    )

                    action = torch.tanh(mean)

                obs, reward, terminated, truncated, info = (
                    env.step(action.cpu().numpy())
                )

                trail.append(
                    env.position.copy()
                )

                if len(trail) > 400:
                    trail.pop(0)

                target_path.append(
                    env.target.position().copy()
                )

                if len(target_path) > 400:
                    target_path.pop(0)

                if terminated or truncated:

                    obs = env.reset()

                    trail.clear()
                    target_path.clear()

        screen.fill((24, 27, 36))

        # World boundary.
        pygame.draw.rect(
            screen,
            (100, 110, 130),
            (0, 0, WORLD_W, WORLD_H),
            3
        )

        # Obstacle and safety radius.
        pygame.draw.circle(
            screen,
            (90, 90, 100),
            OBSTACLE_POS.astype(int),
            int(SAFE_RADIUS),
            2
        )

        pygame.draw.circle(
            screen,
            (190, 70, 70),
            OBSTACLE_POS.astype(int),
            int(OBSTACLE_RADIUS)
        )

        # Moving target orbit.
        center = env.target.center

        pygame.draw.ellipse(
            screen,
            (80, 90, 110),
            (
                center[0] - TARGET_A,
                center[1] - TARGET_B,
                2 * TARGET_A,
                2 * TARGET_B
            ),
            1
        )

        # Drone trail.
        if len(trail) > 1:

            pygame.draw.lines(
                screen,
                (70, 150, 240),
                False,
                [
                    tuple(p.astype(int))
                    for p in trail
                ],
                2
            )

        # Moving target.
        target_pos = env.target.position()
        target_vel = env.target.velocity()

        pygame.draw.circle(
            screen,
            (255, 190, 50),
            target_pos.astype(int),
            13
        )

        pygame.draw.line(
            screen,
            (255, 220, 100),
            target_pos.astype(int),
            (
                target_pos + 0.25 * target_vel
            ).astype(int),
            3
        )

        # Quadcopter silhouette.
        drone_x, drone_y = env.position.astype(int)

        pygame.draw.line(
            screen,
            (220, 230, 245),
            (drone_x - 24, drone_y),
            (drone_x + 24, drone_y),
            5
        )

        pygame.draw.line(
            screen,
            (220, 230, 245),
            (drone_x, drone_y - 24),
            (drone_x, drone_y + 24),
            5
        )

        for dx, dy in [
            (-24, 0),
            (24, 0),
            (0, -24),
            (0, 24)
        ]:

            pygame.draw.circle(
                screen,
                (80, 190, 255),
                (drone_x + dx, drone_y + dy),
                7
            )

        pygame.draw.circle(
            screen,
            (255, 255, 255),
            (drone_x, drone_y),
            5
        )

        lines = [
            "PARALLEL PPO — QUADCOPTER",
            f"Distance: {info_distance(env):.1f}",
            f"Speed: {simulation_speed}x",
            f"Paused: {paused}",
            "Click: change target orbit center",
            "SPACE: pause | R: reset",
            "+/-: simulation speed",
            "ESC: exit"
        ]

        for i, line in enumerate(lines):

            text_surface = font.render(
                line,
                True,
                (235, 235, 245)
            )

            screen.blit(
                text_surface,
                (15, 12 + i * 25)
            )

        pygame.display.flip()

        clock.tick(60)

    pygame.quit()


def info_distance(env):

    return np.linalg.norm(
        env.target.position() - env.position
    )


# ============================================================
# 15. NUMERICAL AND LOGIC TESTS
# ============================================================

def run_tests():

    print("\nRunning numerical tests...")

    # Test 1: Observation dimension.
    env = DroneEnv(seed=1)

    obs = env.reset()

    assert obs.shape == (OBS_DIM,)

    # Test 2: Target position.
    target = MovingTarget(
        center=[400, 300],
        phase=0
    )

    expected = np.array(
        [400 + TARGET_A, 300],
        dtype=np.float32
    )

    assert np.allclose(
        target.position(),
        expected
    )

    # Test 3: Target velocity is finite.
    assert np.all(
        np.isfinite(target.velocity())
    )

    # Test 4: Action clipping.
    obs, reward, terminated, truncated, info = (
        env.step(np.array([5.0, -5.0]))
    )

    assert np.all(np.isfinite(obs))

    # Test 5: Parallel environment shapes.
    vector_env = ParallelEnvironments(
        4,
        seed=10
    )

    observations = vector_env.reset()

    assert observations.shape == (4, OBS_DIM)

    # Test 6: Parallel step shapes.
    actions = np.zeros(
        (4, ACTION_DIM),
        dtype=np.float32
    )

    (
        next_obs,
        rewards,
        terminated,
        truncated,
        final_obs,
        infos
    ) = vector_env.step(actions)

    assert next_obs.shape == (4, OBS_DIM)
    assert rewards.shape == (4,)
    assert terminated.shape == (4,)
    assert truncated.shape == (4,)
    assert final_obs.shape == (4, OBS_DIM)

    # Test 7: GAE with lambda=0 equals one-step TD residual.
    rewards = np.array(
        [[1.0], [2.0]],
        dtype=np.float32
    )

    values = np.array(
        [[0.5], [0.7]],
        dtype=np.float32
    )

    next_values = np.array(
        [[0.7], [0.9]],
        dtype=np.float32
    )

    terminated = np.zeros(
        (2, 1),
        dtype=np.float32
    )

    boundaries = np.zeros(
        (2, 1),
        dtype=np.float32
    )

    advantages, returns = calculate_gae(
        rewards,
        values,
        next_values,
        terminated,
        boundaries,
        gamma=0.9,
        gae_lambda=0.0
    )

    expected_delta = (
        rewards
        + 0.9 * next_values
        - values
    )

    assert np.allclose(
        advantages,
        expected_delta
    )

    # Test 8: True terminal suppresses bootstrap.
    advantages, returns = calculate_gae(
        np.array([[1.0]], dtype=np.float32),
        np.array([[0.5]], dtype=np.float32),
        np.array([[100.0]], dtype=np.float32),
        np.array([[1.0]], dtype=np.float32),
        np.array([[1.0]], dtype=np.float32),
        gamma=0.9,
        gae_lambda=0.95
    )

    assert np.allclose(
        advantages,
        [[0.5]]
    )

    # Test 9: Truncation bootstraps from final observation.
    advantages, returns = calculate_gae(
        np.array([[1.0]], dtype=np.float32),
        np.array([[0.5]], dtype=np.float32),
        np.array([[2.0]], dtype=np.float32),
        np.array([[0.0]], dtype=np.float32),
        np.array([[1.0]], dtype=np.float32),
        gamma=0.9,
        gae_lambda=0.95
    )

    assert np.allclose(
        advantages,
        [[2.3]]
    )

    # Test 10: GAE recursion stops at truncation.
    rewards = np.array(
        [[1.0], [2.0]],
        dtype=np.float32
    )

    values = np.zeros(
        (2, 1),
        dtype=np.float32
    )

    next_values = np.zeros(
        (2, 1),
        dtype=np.float32
    )

    terminated = np.zeros(
        (2, 1),
        dtype=np.float32
    )

    boundaries = np.array(
        [[1.0], [0.0]],
        dtype=np.float32
    )

    advantages, returns = calculate_gae(
        rewards,
        values,
        next_values,
        terminated,
        boundaries,
        gamma=0.9,
        gae_lambda=0.95
    )

    assert np.allclose(
        advantages[0],
        [1.0]
    )

    # Test 11: GAE recursion continues without a boundary.
    boundaries = np.zeros(
        (2, 1),
        dtype=np.float32
    )

    advantages, returns = calculate_gae(
        rewards,
        values,
        next_values,
        terminated,
        boundaries,
        gamma=0.9,
        gae_lambda=0.95
    )

    expected_first = (
        1.0
        + 0.9 * 0.95 * 2.0
    )

    assert np.allclose(
        advantages[0],
        [expected_first]
    )

    # Test 12: Returns equal advantages plus values.
    assert np.allclose(
        returns,
        advantages + values
    )

    # Test 13: Network output shapes.
    network = ActorCritic().to(DEVICE)

    test_obs = torch.zeros(
        (5, OBS_DIM),
        device=DEVICE
    )

    mean, std, value = network(test_obs)

    assert mean.shape == (5, ACTION_DIM)
    assert std.shape == (ACTION_DIM,)
    assert value.shape == (5,)

    # Test 14: Actions remain bounded.
    with torch.no_grad():

        action, raw_action, log_prob, value = (
            network.sample_action(test_obs)
        )

    assert torch.all(action <= 1.0)
    assert torch.all(action >= -1.0)

    # Test 15: Log probabilities are finite.
    assert torch.all(
        torch.isfinite(log_prob)
    )

    # Test 16: Raw action is preserved.
    assert raw_action.shape == (5, ACTION_DIM)

    # Test 17: PPO update produces finite metrics.
    agent = PPOAgent()

    fake_rollout = Rollout(
        observations=np.random.randn(
            8, 4, OBS_DIM
        ).astype(np.float32),

        raw_actions=np.random.randn(
            8, 4, ACTION_DIM
        ).astype(np.float32),

        old_log_probs=np.zeros(
            (8, 4),
            dtype=np.float32
        ),

        rewards=np.random.randn(
            8, 4
        ).astype(np.float32),

        values=np.zeros(
            (8, 4),
            dtype=np.float32
        ),

        next_values=np.zeros(
            (8, 4),
            dtype=np.float32
        ),

        terminated=np.zeros(
            (8, 4),
            dtype=np.float32
        ),

        boundaries=np.zeros(
            (8, 4),
            dtype=np.float32
        )
    )

    metrics = agent.update(fake_rollout)

    assert all(
        np.isfinite(value)
        for value in metrics.values()
    )

    # Test 18: Target center update.
    env.set_target_center([500, 250])

    assert np.allclose(
        env.target.center,
        [500, 250]
    )

    # Test 19: Target phase advances.
    old_phase = env.target.phase

    env.step(np.zeros(2))

    assert env.target.phase > old_phase

    # Test 20: Environment returns finite reward.
    assert np.isfinite(reward)

    print("20 tests passed.")


# ============================================================
# 16. MAIN
# ============================================================

def main():

    print("=" * 65)
    print("PHASE 5.17 — PARALLEL PPO FOR CONTINUOUS DRONE CONTROL")
    print("=" * 65)

    print("Device:", DEVICE)
    print("Parallel environments:", NUM_ENVS)
    print("Rollout steps:", ROLLOUT_STEPS)
    print(
        "Transitions per PPO update:",
        NUM_ENVS * ROLLOUT_STEPS
    )

    run_tests()

    agent, history = train()

    evaluate(agent)

    visualize_training(history)

    run_pygame(agent)


if __name__ == "__main__":
    main()
