"""
File: phase4/muzero_drone.py

PHASE 4.32 — GUIDED MUZERO: PREDICTIVE INTERCEPTION
===================================================

Architecture:
    Representation: h(observation) -> latent
    Prediction:     f(latent) -> policy, value
    Dynamics:       g(latent, action) -> next_latent, reward
    Planning:       PUCT MCTS in latent space

Improvements:
1. Correct unit-test completion logic.
2. Correct checkpoint resume behavior.
3. Reward-aware MCTS backup.
4. Predictive interception teacher.
5. Relative-velocity-aware observation.
6. Future target displacement.
7. Teacher-guided root priors and policy targets.
8. Improved reward shaping.
9. Normalized value and reward learning targets.
10. Recurrent latent consistency training.
11. Replay-buffer training.
12. Obstacle-aware teacher controller.
13. Evaluation metrics.
14. Pygame visualization.
15. Checkpoint saving/loading.

Educational simulation only. Not a flight-ready controller.
"""

from __future__ import annotations

import os
import math
import random
import argparse
from collections import deque

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# ============================================================
# 0. CONFIGURATION
# ============================================================

SEED = 431

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

WORLD_W = 900
WORLD_H = 650

DT = 0.08

MAX_SPEED = 145.0
MAX_ACCEL = 120.0
DRONE_RADIUS = 12.0

OBSTACLE = np.array(
    [450.0, 330.0],
    dtype=np.float32
)

OBSTACLE_RADIUS = 65.0
SAFE_RADIUS = 110.0

TARGET_A = 165.0
TARGET_B = 105.0
TARGET_OMEGA = 0.55

MOTOR_GAIN = 0.72
LINEAR_DRAG = 0.115
QUADRATIC_DRAG = 0.00032

WIND = np.array(
    [5.0, -3.5],
    dtype=np.float32
)

OBS_DIM = 13
LATENT_DIM = 128

ACTION_VECTORS = np.array(
    [
        [0.0, 0.0],
        [-1.0, 0.0],
        [1.0, 0.0],
        [0.0, -1.0],
        [0.0, 1.0],
        [-0.7071, -0.7071],
        [0.7071, -0.7071],
        [-0.7071, 0.7071],
        [0.7071, 0.7071],
    ],
    dtype=np.float32
)

ACTION_COUNT = len(ACTION_VECTORS)

GAMMA = 0.99

MCTS_SIMULATIONS = 16
PUCT_C = 1.5

DIRICHLET_ALPHA = 0.3
ROOT_NOISE_FRACTION = 0.20

TEACHER_LOOKAHEAD = 0.85
TEACHER_TEMPERATURE = 2.0

TEACHER_PRIOR_WEIGHT = 0.65
TEACHER_TARGET_WEIGHT = 0.55

TRAIN_EPISODES = 120
EPISODE_STEPS = 120

REPLAY_CAPACITY = 500
UNROLL_STEPS = 5

LEARNING_RATE = 0.00025
TRAINING_UPDATES_PER_EPISODE = 16

VALUE_LOSS_WEIGHT = 1.0
REWARD_LOSS_WEIGHT = 1.0
LATENT_LOSS_WEIGHT = 0.25

GRADIENT_CLIP = 5.0

REWARD_SCALE = 10.0
VALUE_SCALE = 50.0

VISUALIZATION_SIMULATIONS = 12

CHECKPOINT_PATH = "muzero_drone_checkpoint.pt"


# ============================================================
# 1. UTILITIES
# ============================================================

def safe_norm(vector):
    return float(np.linalg.norm(vector))


def distance(a, b):
    return safe_norm(
        np.asarray(a) - np.asarray(b)
    )


def normalize(vector):
    vector = np.asarray(
        vector,
        dtype=np.float32
    )

    magnitude = safe_norm(vector)

    if magnitude < 1e-8:
        return np.zeros_like(vector)

    return vector / magnitude


def softmax_numpy(logits):
    logits = np.asarray(
        logits,
        dtype=np.float64
    ).copy()

    logits -= np.max(logits)

    probabilities = np.exp(logits)

    probabilities /= (
        probabilities.sum() + 1e-12
    )

    return probabilities


def moving_average(values, window=10):
    values = np.asarray(
        values,
        dtype=np.float32
    )

    if len(values) < window:
        return values

    return np.convolve(
        values,
        np.ones(window, dtype=np.float32) / window,
        mode="valid"
    )


# ============================================================
# 2. MOVING TARGET
# ============================================================

class MovingTarget:

    def __init__(self, center):
        self.center = np.asarray(
            center,
            dtype=np.float32
        )

        self.phase = 0.0

    def position(self, phase=None):
        p = self.phase if phase is None else phase

        return self.center + np.array(
            [
                TARGET_A * math.cos(p),
                TARGET_B * math.sin(p)
            ],
            dtype=np.float32
        )

    def velocity(self, phase=None):
        p = self.phase if phase is None else phase

        return np.array(
            [
                -TARGET_A * TARGET_OMEGA * math.sin(p),
                TARGET_B * TARGET_OMEGA * math.cos(p)
            ],
            dtype=np.float32
        )

    def acceleration(self, phase=None):
        p = self.phase if phase is None else phase

        return np.array(
            [
                -TARGET_A * TARGET_OMEGA**2 * math.cos(p),
                -TARGET_B * TARGET_OMEGA**2 * math.sin(p)
            ],
            dtype=np.float32
        )

    def predict(self, horizon):
        return self.position(
            self.phase + TARGET_OMEGA * horizon
        )

    def advance(self):
        self.phase += TARGET_OMEGA * DT


# ============================================================
# 3. DRONE PHYSICS
# ============================================================

def clip_drone_state(state):
    state = np.asarray(
        state,
        dtype=np.float32
    ).copy()

    velocity = state[2:4]
    speed = safe_norm(velocity)

    if speed > MAX_SPEED:
        state[2:4] *= MAX_SPEED / (speed + 1e-9)

    state[0] = np.clip(
        state[0],
        DRONE_RADIUS,
        WORLD_W - DRONE_RADIUS
    )

    state[1] = np.clip(
        state[1],
        DRONE_RADIUS,
        WORLD_H - DRONE_RADIUS
    )

    return state


def true_drone_transition(state, action):
    action = np.clip(
        np.asarray(action, dtype=np.float32),
        -1.0,
        1.0
    )

    velocity = state[2:4]

    acceleration = (
        MOTOR_GAIN * MAX_ACCEL * action
        - LINEAR_DRAG * velocity
        - QUADRATIC_DRAG * np.abs(velocity) * velocity
        + WIND
    )

    next_velocity = velocity + acceleration * DT
    next_position = state[:2] + next_velocity * DT

    return clip_drone_state(
        np.concatenate(
            [next_position, next_velocity]
        )
    )


# ============================================================
# 4. PREDICTIVE TEACHER
# ============================================================

def predictive_teacher_policy(observation):
    obs = np.asarray(
        observation,
        dtype=np.float32
    )

    relative_position = obs[0:2] * 400.0
    relative_velocity = obs[2:4] * 200.0
    drone_velocity = obs[4:6] * MAX_SPEED
    relative_obstacle = obs[6:8] * 400.0
    predicted_displacement = obs[10:12] * 400.0

    future_relative = (
        relative_position + predicted_displacement
    )

    desired_acceleration = (
        0.85 * future_relative
        + 2.0 * relative_velocity
        - 0.15 * drone_velocity
    )

    obstacle_distance = safe_norm(relative_obstacle)

    if obstacle_distance < SAFE_RADIUS + 75.0:
        away = normalize(-relative_obstacle)

        strength = (
            SAFE_RADIUS + 75.0 - obstacle_distance
        ) / (SAFE_RADIUS + 75.0)

        desired_acceleration += (
            away * 180.0 * strength
        )

    desired_acceleration = np.clip(
        desired_acceleration,
        -MAX_ACCEL,
        MAX_ACCEL
    )

    desired_direction = normalize(
        desired_acceleration
    )

    scores = []

    for action in ACTION_VECTORS:
        alignment = float(
            np.dot(action, desired_direction)
        )

        acceleration_error = safe_norm(
            MAX_ACCEL * action - desired_acceleration
        )

        score = (
            3.0 * alignment
            - 0.012 * acceleration_error
        )

        if safe_norm(relative_position) < 65.0:
            score -= 0.003 * safe_norm(
                action * MAX_ACCEL
            )

        scores.append(score)

    return softmax_numpy(
        np.asarray(scores) / TEACHER_TEMPERATURE
    ).astype(np.float32)


# ============================================================
# 5. ENVIRONMENT
# ============================================================

class DroneEnv:

    def __init__(self):
        self.target = MovingTarget([710.0, 315.0])
        self.state = None
        self.steps = 0
        self.reset()

    def observation(self):
        position = self.state[:2]
        velocity = self.state[2:4]

        target_position = self.target.position()
        target_velocity = self.target.velocity()

        relative_target = target_position - position
        relative_velocity = target_velocity - velocity
        relative_obstacle = OBSTACLE - position

        future_target = self.target.predict(
            TEACHER_LOOKAHEAD
        )

        predicted_displacement = (
            future_target - target_position
        )

        return np.concatenate(
            [
                relative_target / 400.0,
                relative_velocity / 200.0,
                velocity / MAX_SPEED,
                relative_obstacle / 400.0,
                np.array(
                    [
                        math.sin(self.target.phase),
                        math.cos(self.target.phase)
                    ],
                    dtype=np.float32
                ),
                predicted_displacement / 400.0,
                np.array(
                    [safe_norm(relative_target) / 600.0],
                    dtype=np.float32
                )
            ]
        ).astype(np.float32)

    def reset(self):
        self.state = np.array(
            [110.0, 500.0, 0.0, 0.0],
            dtype=np.float32
        )

        self.target.phase = np.random.uniform(
            0.0,
            2.0 * math.pi
        )

        self.steps = 0

        return self.observation()

    def step(self, action_index):
        action_index = int(
            np.clip(action_index, 0, ACTION_COUNT - 1)
        )

        action = ACTION_VECTORS[action_index]

        old_position = self.state[:2].copy()

        future_phase = (
            self.target.phase + TARGET_OMEGA * DT
        )

        future_target = self.target.position(
            future_phase
        )

        distance_before = distance(
            old_position,
            future_target
        )

        self.state = true_drone_transition(
            self.state,
            action
        )

        self.target.advance()

        target_position = self.target.position()

        distance_after = distance(
            self.state[:2],
            target_position
        )

        progress = distance_before - distance_after

        obstacle_distance = distance(
            self.state[:2],
            OBSTACLE
        )

        boundary_distance = min(
            self.state[0],
            WORLD_W - self.state[0],
            self.state[1],
            WORLD_H - self.state[1]
        )

        collision = (
            obstacle_distance
            < OBSTACLE_RADIUS + DRONE_RADIUS
        )

        boundary_collision = (
            boundary_distance < DRONE_RADIUS + 2.0
        )

        progress_reward = 0.35 * progress

        proximity_reward = (
            2.0 * math.exp(-distance_after / 110.0)
        )

        relative_velocity = (
            self.target.velocity() - self.state[2:4]
        )

        velocity_penalty = (
            0.002
            * safe_norm(relative_velocity)
            * math.exp(-distance_after / 100.0)
        )

        action_penalty = (
            0.018 * float(np.sum(action ** 2))
        )

        obstacle_violation = max(
            0.0,
            SAFE_RADIUS - obstacle_distance
        )

        obstacle_penalty = (
            0.018 * obstacle_violation**2
        )

        boundary_violation = max(
            0.0,
            65.0 - boundary_distance
        )

        boundary_penalty = (
            0.035 * boundary_violation**2
        )

        reward = (
            progress_reward
            + proximity_reward
            - velocity_penalty
            - action_penalty
            - obstacle_penalty
            - boundary_penalty
        )

        if collision:
            reward -= 150.0

        if boundary_collision:
            reward -= 150.0

        if distance_after < 45.0:
            reward += 3.0

        self.steps += 1

        terminated = collision or boundary_collision
        truncated = self.steps >= EPISODE_STEPS

        info = {
            "distance": float(distance_after),
            "progress": float(progress),
            "collision": bool(collision),
            "boundary_collision": bool(boundary_collision),
            "obstacle_distance": float(obstacle_distance),
            "close_tracking": bool(distance_after < 60.0)
        }

        return (
            self.observation(),
            float(reward),
            terminated,
            truncated,
            info
        )


# ============================================================
# 6. MUZERO NETWORK
# ============================================================

class MuZeroNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        self.representation = nn.Sequential(
            nn.Linear(OBS_DIM, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, LATENT_DIM),
            nn.Tanh()
        )

        self.dynamics = nn.Sequential(
            nn.Linear(LATENT_DIM + ACTION_COUNT, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )

        self.next_latent = nn.Sequential(
            nn.Linear(256, LATENT_DIM),
            nn.Tanh()
        )

        self.reward_head = nn.Linear(256, 1)

        self.prediction = nn.Sequential(
            nn.Linear(LATENT_DIM, 256),
            nn.LayerNorm(256),
            nn.ReLU()
        )

        self.policy_head = nn.Linear(
            256,
            ACTION_COUNT
        )

        self.value_head = nn.Linear(256, 1)

    def prediction_inference(self, latent):
        features = self.prediction(latent)

        return (
            self.policy_head(features),
            self.value_head(features)
        )

    def initial_inference(self, observation):
        latent = self.representation(observation)

        policy_logits, normalized_value = (
            self.prediction_inference(latent)
        )

        return (
            latent,
            policy_logits,
            normalized_value * VALUE_SCALE
        )

    def recurrent_inference(self, latent, action_index):
        action_one_hot = F.one_hot(
            action_index.long(),
            num_classes=ACTION_COUNT
        ).float()

        x = torch.cat(
            [latent, action_one_hot],
            dim=-1
        )

        hidden = self.dynamics(x)

        next_latent = self.next_latent(hidden)

        normalized_reward = self.reward_head(hidden)

        policy_logits, normalized_value = (
            self.prediction_inference(next_latent)
        )

        return (
            next_latent,
            normalized_reward * REWARD_SCALE,
            policy_logits,
            normalized_value * VALUE_SCALE
        )


# ============================================================
# 7. MCTS DATA STRUCTURES
# ============================================================

class Edge:

    def __init__(self, prior):
        self.prior = float(prior)
        self.visit_count = 0
        self.value_sum = 0.0
        self.reward = 0.0
        self.child = None

    @property
    def value(self):
        if self.visit_count == 0:
            return 0.0

        return self.value_sum / self.visit_count


class Node:

    def __init__(self):
        self.latent = None
        self.value_prediction = 0.0
        self.children = {}

    def expanded(self):
        return bool(self.children)


# ============================================================
# 8. MCTS
# ============================================================

class MuZeroMCTS:

    def __init__(
        self,
        network,
        simulations=MCTS_SIMULATIONS,
        c_puct=PUCT_C
    ):
        self.network = network
        self.simulations = int(simulations)
        self.c_puct = float(c_puct)

    def expand(
        self,
        node,
        latent,
        policy_logits,
        value
    ):
        node.latent = latent.detach()

        node.value_prediction = float(value)

        probabilities = softmax_numpy(policy_logits)

        node.children = {
            action: Edge(probability)
            for action, probability in enumerate(probabilities)
        }

    def select_child(self, node):
        parent_visits = math.sqrt(
            1.0 + sum(
                edge.visit_count
                for edge in node.children.values()
            )
        )

        best_action = None
        best_edge = None
        best_score = -float("inf")

        for action, edge in node.children.items():
            q_value = edge.value

            exploration = (
                self.c_puct
                * edge.prior
                * parent_visits
                / (1 + edge.visit_count)
            )

            score = q_value + exploration

            if score > best_score:
                best_score = score
                best_action = action
                best_edge = edge

        return best_action, best_edge

    def run(self, observation, add_root_noise=True):
        observation_tensor = torch.as_tensor(
            observation,
            dtype=torch.float32,
            device=DEVICE
        ).unsqueeze(0)

        with torch.inference_mode():
            root_latent, root_logits, root_value_tensor = (
                self.network.initial_inference(
                    observation_tensor
                )
            )

        root = Node()

        root_logits_np = (
            root_logits.squeeze(0).cpu().numpy()
        )

        self.expand(
            root,
            root_latent,
            root_logits_np,
            root_value_tensor.item()
        )

        # Teacher-guided root priors.
        teacher_prior = predictive_teacher_policy(
            observation
        )

        for action, edge in root.children.items():
            edge.prior = (
                (1.0 - TEACHER_PRIOR_WEIGHT) * edge.prior
                + TEACHER_PRIOR_WEIGHT * teacher_prior[action]
            )

        # Normalize after prior blending.
        prior_sum = sum(
            edge.prior for edge in root.children.values()
        )

        for edge in root.children.values():
            edge.prior /= max(prior_sum, 1e-12)

        if add_root_noise:
            noise = np.random.dirichlet(
                [DIRICHLET_ALPHA] * ACTION_COUNT
            )

            for action, edge in root.children.items():
                edge.prior = (
                    (1.0 - ROOT_NOISE_FRACTION) * edge.prior
                    + ROOT_NOISE_FRACTION * noise[action]
                )

        # Latent-space MCTS.
        for _ in range(self.simulations):
            node = root
            search_path = []

            while True:
                action, edge = self.select_child(node)
                search_path.append(edge)

                if edge.child is None:
                    action_tensor = torch.tensor(
                        [action],
                        dtype=torch.long,
                        device=DEVICE
                    )

                    with torch.inference_mode():
                        (
                            next_latent,
                            predicted_reward,
                            next_logits,
                            next_value
                        ) = self.network.recurrent_inference(
                            node.latent,
                            action_tensor
                        )

                    edge.reward = float(
                        predicted_reward.item()
                    )

                    child = Node()

                    self.expand(
                        child,
                        next_latent,
                        next_logits.squeeze(0).cpu().numpy(),
                        next_value.item()
                    )

                    edge.child = child

                    leaf_value = child.value_prediction
                    break

                node = edge.child

            # Correct reward-aware backup:
            # each edge reward is included exactly once.
            value = leaf_value

            for edge in reversed(search_path):
                value = edge.reward + GAMMA * value

                edge.value_sum += value
                edge.visit_count += 1

        visit_counts = np.array(
            [
                root.children[action].visit_count
                for action in range(ACTION_COUNT)
            ],
            dtype=np.float32
        )

        if visit_counts.sum() > 0:
            policy = visit_counts / visit_counts.sum()
        else:
            policy = softmax_numpy(root_logits_np).astype(
                np.float32
            )

        total_visits = visit_counts.sum()

        if total_visits > 0:
            root_value = sum(
                root.children[action].value
                * root.children[action].visit_count
                for action in range(ACTION_COUNT)
            ) / total_visits
        else:
            root_value = root.value_prediction

        return (
            root,
            policy.astype(np.float32),
            float(root_value)
        )


# ============================================================
# 9. EPISODE REPLAY
# ============================================================

class Episode:

    def __init__(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.policies = []
        self.teacher_policies = []
        self.root_values = []
        self.distances = []

        self.terminated = False
        self.collisions = 0

    def __len__(self):
        return len(self.actions)


class EpisodeReplay:

    def __init__(self, capacity=REPLAY_CAPACITY):
        self.episodes = deque(maxlen=capacity)

    def add(self, episode):
        if len(episode) > 0:
            self.episodes.append(episode)

    def sample_position(self):
        if not self.episodes:
            raise ValueError(
                "Cannot sample from empty replay."
            )

        episode = random.choice(self.episodes)

        index = random.randrange(len(episode))

        return episode, index

    def __len__(self):
        return len(self.episodes)


# ============================================================
# 10. VALUE TARGET
# ============================================================

def make_value_target(episode, index):
    value = 0.0
    discount = 1.0

    for reward in episode.rewards[index:]:
        value += discount * reward
        discount *= GAMMA

    return float(value)


# ============================================================
# 11. TRAINER
# ============================================================

class MuZeroTrainer:

    def __init__(self):
        self.network = MuZeroNetwork().to(DEVICE)

        self.optimizer = optim.AdamW(
            self.network.parameters(),
            lr=LEARNING_RATE,
            weight_decay=1e-5
        )

        self.replay = EpisodeReplay()

        self.training_losses = []
        self.policy_losses = []
        self.value_losses = []
        self.reward_losses = []
        self.latent_losses = []

    def collect_episode(self, add_root_noise=True):
        env = DroneEnv()
        observation = env.reset()

        episode = Episode()

        search = MuZeroMCTS(self.network)

        self.network.eval()

        for _ in range(EPISODE_STEPS):
            _, policy, root_value = search.run(
                observation,
                add_root_noise=add_root_noise
            )

            teacher_policy = predictive_teacher_policy(
                observation
            )

            guided_policy = (
                (1.0 - TEACHER_TARGET_WEIGHT) * policy
                + TEACHER_TARGET_WEIGHT * teacher_policy
            )

            guided_policy = guided_policy / max(
                guided_policy.sum(),
                1e-12
            )

            if add_root_noise:
                action_index = int(
                    np.random.choice(
                        ACTION_COUNT,
                        p=guided_policy
                    )
                )
            else:
                action_index = int(
                    np.argmax(guided_policy)
                )

            (
                next_observation,
                reward,
                terminated,
                truncated,
                info
            ) = env.step(action_index)

            episode.observations.append(
                observation.copy()
            )

            episode.actions.append(action_index)
            episode.rewards.append(reward)
            episode.policies.append(guided_policy.copy())
            episode.teacher_policies.append(
                teacher_policy.copy()
            )
            episode.root_values.append(root_value)
            episode.distances.append(info["distance"])

            if info["collision"] or info["boundary_collision"]:
                episode.collisions += 1

            observation = next_observation

            if terminated or truncated:
                episode.terminated = terminated
                break

        episode.observations.append(
            observation.copy()
        )

        self.replay.add(episode)

        return episode

    def train_step(self):
        if len(self.replay) == 0:
            return None

        self.network.train()

        episode, start_index = (
            self.replay.sample_position()
        )

        observation = torch.as_tensor(
            episode.observations[start_index],
            dtype=torch.float32,
            device=DEVICE
        ).unsqueeze(0)

        latent = self.network.representation(observation)

        total_policy_loss = torch.zeros((), device=DEVICE)
        total_value_loss = torch.zeros((), device=DEVICE)
        total_reward_loss = torch.zeros((), device=DEVICE)
        total_latent_loss = torch.zeros((), device=DEVICE)

        policy_count = 0
        value_count = 0
        reward_count = 0
        latent_count = 0

        for unroll in range(UNROLL_STEPS + 1):
            index = start_index + unroll

            if index >= len(episode.observations):
                break

            policy_logits, normalized_value = (
                self.network.prediction_inference(latent)
            )

            # Policy loss.
            if index < len(episode.actions):
                target_policy = torch.as_tensor(
                    episode.policies[index],
                    dtype=torch.float32,
                    device=DEVICE
                ).unsqueeze(0)

                policy_loss = -(
                    target_policy
                    * F.log_softmax(policy_logits, dim=-1)
                ).sum(dim=-1).mean()

                total_policy_loss += policy_loss
                policy_count += 1

            # Value loss.
            target_value = make_value_target(
                episode,
                index
            )

            target_value_tensor = torch.tensor(
                [[target_value / VALUE_SCALE]],
                dtype=torch.float32,
                device=DEVICE
            )

            value_loss = F.smooth_l1_loss(
                normalized_value,
                target_value_tensor
            )

            total_value_loss += value_loss
            value_count += 1

            if unroll == UNROLL_STEPS:
                break

            if index >= len(episode.actions):
                break

            action_tensor = torch.tensor(
                [episode.actions[index]],
                dtype=torch.long,
                device=DEVICE
            )

            (
                next_latent,
                predicted_reward,
                _,
                _
            ) = self.network.recurrent_inference(
                latent,
                action_tensor
            )

            # Reward loss.
            target_reward = torch.tensor(
                [[episode.rewards[index] / REWARD_SCALE]],
                dtype=torch.float32,
                device=DEVICE
            )

            reward_loss = F.smooth_l1_loss(
                predicted_reward / REWARD_SCALE,
                target_reward
            )

            total_reward_loss += reward_loss
            reward_count += 1

            # Latent consistency.
            next_observation = torch.as_tensor(
                episode.observations[index + 1],
                dtype=torch.float32,
                device=DEVICE
            ).unsqueeze(0)

            with torch.no_grad():
                target_latent = self.network.representation(
                    next_observation
                )

            latent_loss = F.mse_loss(
                next_latent,
                target_latent
            )

            total_latent_loss += latent_loss
            latent_count += 1

            latent = next_latent

        policy_loss = (
            total_policy_loss / max(1, policy_count)
        )

        value_loss = (
            total_value_loss / max(1, value_count)
        )

        reward_loss = (
            total_reward_loss / max(1, reward_count)
        )

        latent_loss = (
            total_latent_loss / max(1, latent_count)
        )

        total_loss = (
            policy_loss
            + VALUE_LOSS_WEIGHT * value_loss
            + REWARD_LOSS_WEIGHT * reward_loss
            + LATENT_LOSS_WEIGHT * latent_loss
        )

        self.optimizer.zero_grad(set_to_none=True)

        total_loss.backward()

        nn.utils.clip_grad_norm_(
            self.network.parameters(),
            GRADIENT_CLIP
        )

        self.optimizer.step()

        values = {
            "total": float(total_loss.detach().cpu()),
            "policy": float(policy_loss.detach().cpu()),
            "value": float(value_loss.detach().cpu()),
            "reward": float(reward_loss.detach().cpu()),
            "latent": float(latent_loss.detach().cpu())
        }

        self.training_losses.append(values["total"])
        self.policy_losses.append(values["policy"])
        self.value_losses.append(values["value"])
        self.reward_losses.append(values["reward"])
        self.latent_losses.append(values["latent"])

        return values

    def save(self, path=CHECKPOINT_PATH):
        checkpoint = {
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "training_losses": self.training_losses,
            "policy_losses": self.policy_losses,
            "value_losses": self.value_losses,
            "reward_losses": self.reward_losses,
            "latent_losses": self.latent_losses,
            "seed": SEED,
            "obs_dim": OBS_DIM,
            "latent_dim": LATENT_DIM
        }

        torch.save(checkpoint, path)

        print(f"Saved checkpoint: {path}")

    def load(self, path=CHECKPOINT_PATH):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Checkpoint not found: {path}"
            )

        checkpoint = torch.load(
            path,
            map_location=DEVICE,
            weights_only=False
        )

        if checkpoint.get("obs_dim", OBS_DIM) != OBS_DIM:
            raise ValueError(
                "Checkpoint observation dimension mismatch."
            )

        if checkpoint.get("latent_dim", LATENT_DIM) != LATENT_DIM:
            raise ValueError(
                "Checkpoint latent dimension mismatch."
            )

        self.network.load_state_dict(
            checkpoint["network"]
        )

        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(
                checkpoint["optimizer"]
            )

        self.training_losses = checkpoint.get(
            "training_losses", []
        )

        self.policy_losses = checkpoint.get(
            "policy_losses", []
        )

        self.value_losses = checkpoint.get(
            "value_losses", []
        )

        self.reward_losses = checkpoint.get(
            "reward_losses", []
        )

        self.latent_losses = checkpoint.get(
            "latent_losses", []
        )

        print(f"Loaded checkpoint: {path}")


# ============================================================
# 12. TRAINING LOOP
# ============================================================

def train_muzero(trainer=None):
    if trainer is None:
        trainer = MuZeroTrainer()

    returns = []
    distances = []
    collision_rates = []
    close_rates = []

    print("\nStarting Phase 4.32 training")
    print(f"Device: {DEVICE}")
    print(f"Episodes: {TRAIN_EPISODES}")
    print(f"MCTS simulations: {MCTS_SIMULATIONS}")

    for episode_index in range(TRAIN_EPISODES):
        episode = trainer.collect_episode(
            add_root_noise=True
        )

        episode_return = float(
            np.sum(episode.rewards)
        )

        mean_distance = float(
            np.mean(episode.distances)
        )

        close_rate = float(
            np.mean(
                np.asarray(episode.distances) < 60.0
            )
        )

        collision_rate = float(
            episode.collisions > 0
        )

        returns.append(episode_return)
        distances.append(mean_distance)
        collision_rates.append(collision_rate)
        close_rates.append(close_rate)

        losses = []

        for _ in range(TRAINING_UPDATES_PER_EPISODE):
            loss = trainer.train_step()

            if loss is not None:
                losses.append(loss)

        if losses:
            mean_total = np.mean(
                [x["total"] for x in losses]
            )

            mean_policy = np.mean(
                [x["policy"] for x in losses]
            )

            mean_value = np.mean(
                [x["value"] for x in losses]
            )

            mean_reward = np.mean(
                [x["reward"] for x in losses]
            )
        else:
            mean_total = float("nan")
            mean_policy = float("nan")
            mean_value = float("nan")
            mean_reward = float("nan")

        print(
            f"Episode {episode_index + 1:3d}/{TRAIN_EPISODES}"
            f" | Return={episode_return:9.2f}"
            f" | Distance={mean_distance:7.2f}"
            f" | Close={close_rate:5.1%}"
            f" | Collision={int(collision_rate)}"
            f" | Loss={mean_total:.4f}"
            f" | P={mean_policy:.3f}"
            f" | V={mean_value:.3f}"
            f" | R={mean_reward:.3f}"
        )

        if (episode_index + 1) % 20 == 0:
            trainer.save()

    trainer.save()

    history = {
        "returns": returns,
        "distances": distances,
        "collision_rates": collision_rates,
        "close_rates": close_rates,
        "losses": trainer.training_losses,
        "policy_losses": trainer.policy_losses,
        "value_losses": trainer.value_losses,
        "reward_losses": trainer.reward_losses,
        "latent_losses": trainer.latent_losses
    }

    return trainer, history


# ============================================================
# 13. EVALUATION
# ============================================================

def evaluate_muzero(trainer, episodes=5):
    print("\nEvaluating Phase 4.32")

    trainer.network.eval()

    all_returns = []
    all_distances = []
    all_close_rates = []
    all_collisions = []

    search = MuZeroMCTS(
        trainer.network,
        simulations=MCTS_SIMULATIONS
    )

    for episode_index in range(episodes):
        env = DroneEnv()
        observation = env.reset()

        total_return = 0.0
        episode_distances = []
        close_steps = 0
        collision_steps = 0

        for _ in range(EPISODE_STEPS):
            _, policy, _ = search.run(
                observation,
                add_root_noise=False
            )

            teacher_policy = predictive_teacher_policy(
                observation
            )

            guided_policy = (
                (1.0 - TEACHER_TARGET_WEIGHT) * policy
                + TEACHER_TARGET_WEIGHT * teacher_policy
            )

            action = int(np.argmax(guided_policy))

            (
                observation,
                reward,
                terminated,
                truncated,
                info
            ) = env.step(action)

            total_return += reward
            episode_distances.append(info["distance"])

            if info["close_tracking"]:
                close_steps += 1

            if info["collision"] or info["boundary_collision"]:
                collision_steps += 1

            if terminated or truncated:
                break

        mean_distance = float(
            np.mean(episode_distances)
        )

        close_rate = (
            close_steps / max(1, len(episode_distances))
        )

        all_returns.append(total_return)
        all_distances.append(mean_distance)
        all_close_rates.append(close_rate)
        all_collisions.append(collision_steps)

        print(
            f"Evaluation {episode_index + 1}/{episodes}"
            f" | Return={total_return:.2f}"
            f" | Mean distance={mean_distance:.2f}"
            f" | Close tracking={close_rate:.1%}"
            f" | Collision steps={collision_steps}"
        )

    results = {
        "mean_return": float(np.mean(all_returns)),
        "mean_distance": float(np.mean(all_distances)),
        "mean_close_tracking": float(np.mean(all_close_rates)),
        "mean_collision_steps": float(np.mean(all_collisions))
    }

    print("\nEvaluation summary")

    for key, value in results.items():
        print(f"{key}: {value:.4f}")

    return results


# ============================================================
# 14. TRAINING PLOTS
# ============================================================

def plot_training(history):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib unavailable. Skipping plots.")
        return

    plt.figure(figsize=(10, 5))
    plt.plot(history["returns"], alpha=0.4, label="Episode return")

    smooth = moving_average(history["returns"], 10)

    if len(smooth):
        plt.plot(
            np.arange(len(smooth)) + min(
                9,
                len(history["returns"]) - 1
            ),
            smooth,
            label="10-episode average"
        )

    plt.title("Phase 4.32 — Training Return")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(history["distances"], label="Mean tracking distance")
    plt.title("Tracking Distance")
    plt.xlabel("Episode")
    plt.ylabel("Distance")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(
        np.asarray(history["close_rates"]) * 100.0,
        label="Close tracking"
    )
    plt.title("Close Tracking Percentage")
    plt.xlabel("Episode")
    plt.ylabel("Percentage")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(history["losses"], label="Total loss")
    plt.title("MuZero Training Loss")
    plt.xlabel("Update")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(history["policy_losses"], label="Policy")
    plt.plot(history["value_losses"], label="Value")
    plt.plot(history["reward_losses"], label="Reward")
    plt.plot(history["latent_losses"], label="Latent")
    plt.title("MuZero Loss Components")
    plt.xlabel("Update")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# 15. PYGAME VISUALIZATION
# ============================================================

def run_pygame(trainer):
    try:
        import pygame
    except ImportError:
        print("Install pygame: pip install pygame")
        return

    pygame.init()

    screen = pygame.display.set_mode(
        (WORLD_W, WORLD_H)
    )

    pygame.display.set_caption(
        "Phase 4.32 — Guided MuZero Predictive Tracking"
    )

    clock = pygame.time.Clock()

    font = pygame.font.SysFont(
        "consolas",
        15
    )

    env = DroneEnv()
    observation = env.reset()

    search = MuZeroMCTS(
        trainer.network,
        simulations=VISUALIZATION_SIMULATIONS
    )

    trainer.network.eval()

    running = True
    paused = False
    speed = 1

    trail = []

    last_policy = np.ones(
        ACTION_COUNT,
        dtype=np.float32
    ) / ACTION_COUNT

    last_action = 0
    last_value = 0.0
    last_reward = 0.0

    def draw_text(text, x, y, color=(235, 235, 235)):
        surface = font.render(text, True, color)
        screen.blit(surface, (x, y))

    def reset_simulation():
        nonlocal observation
        observation = env.reset()
        trail.clear()

    def draw_grid():
        for x in range(0, WORLD_W, 50):
            pygame.draw.line(
                screen,
                (30, 38, 50),
                (x, 0),
                (x, WORLD_H)
            )

        for y in range(0, WORLD_H, 50):
            pygame.draw.line(
                screen,
                (30, 38, 50),
                (0, y),
                (WORLD_W, y)
            )

    def draw_target():
        orbit = []

        for index in range(120):
            phase = (
                env.target.phase
                + 2.0 * math.pi * index / 120
            )

            orbit.append(
                env.target.position(phase).astype(int)
            )

        pygame.draw.lines(
            screen,
            (65, 105, 75),
            True,
            orbit,
            2
        )

        target_position = env.target.position()
        predicted_position = env.target.predict(
            TEACHER_LOOKAHEAD
        )

        pygame.draw.circle(
            screen,
            (70, 240, 110),
            target_position.astype(int),
            11
        )

        pygame.draw.circle(
            screen,
            (180, 255, 190),
            target_position.astype(int),
            18,
            1
        )

        pygame.draw.circle(
            screen,
            (255, 210, 60),
            predicted_position.astype(int),
            9,
            2
        )

        pygame.draw.line(
            screen,
            (255, 210, 60),
            target_position.astype(int),
            predicted_position.astype(int),
            2
        )

        pygame.draw.line(
            screen,
            (70, 255, 120),
            target_position.astype(int),
            (
                target_position
                + 0.18 * env.target.velocity()
            ).astype(int),
            3
        )

    def draw_drone():
        center = env.state[:2]
        velocity = env.state[2:4]

        speed_value = safe_norm(velocity)

        if speed_value > 1e-6:
            direction = velocity / speed_value
        else:
            direction = np.array(
                [1.0, 0.0],
                dtype=np.float32
            )

        side = np.array(
            [-direction[1], direction[0]],
            dtype=np.float32
        )

        for forward in (-1, 1):
            for lateral in (-1, 1):
                rotor = (
                    center
                    + 17 * forward * direction
                    + 17 * lateral * side
                )

                pygame.draw.line(
                    screen,
                    (220, 225, 235),
                    center.astype(int),
                    rotor.astype(int),
                    5
                )

                pygame.draw.circle(
                    screen,
                    (90, 190, 245),
                    rotor.astype(int),
                    7,
                    2
                )

        pygame.draw.circle(
            screen,
            (250, 250, 250),
            center.astype(int),
            8
        )

        pygame.draw.circle(
            screen,
            (255, 190, 60),
            (center + direction * 22).astype(int),
            4
        )

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
                    reset_simulation()

                elif event.key in (
                    pygame.K_EQUALS,
                    pygame.K_PLUS,
                    pygame.K_KP_PLUS
                ):
                    speed = min(5, speed + 1)

                elif event.key in (
                    pygame.K_MINUS,
                    pygame.K_KP_MINUS
                ):
                    speed = max(1, speed - 1)

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    click = np.asarray(
                        event.pos,
                        dtype=np.float32
                    )

                    if distance(click, OBSTACLE) > SAFE_RADIUS:
                        env.target.center = np.clip(
                            click,
                            [
                                TARGET_A + 20,
                                TARGET_B + 20
                            ],
                            [
                                WORLD_W - TARGET_A - 20,
                                WORLD_H - TARGET_B - 20
                            ]
                        )

        if not paused:
            for _ in range(speed):
                _, last_policy, last_value = search.run(
                    observation,
                    add_root_noise=False
                )

                teacher_policy = predictive_teacher_policy(
                    observation
                )

                guided_policy = (
                    (1.0 - TEACHER_TARGET_WEIGHT) * last_policy
                    + TEACHER_TARGET_WEIGHT * teacher_policy
                )

                last_action = int(np.argmax(guided_policy))

                (
                    observation,
                    last_reward,
                    terminated,
                    truncated,
                    _
                ) = env.step(last_action)

                trail.append(env.state[:2].copy())

                if len(trail) > 300:
                    trail.pop(0)

                if terminated or truncated:
                    reset_simulation()

        screen.fill((17, 23, 34))

        draw_grid()

        pygame.draw.circle(
            screen,
            (100, 100, 110),
            OBSTACLE.astype(int),
            int(OBSTACLE_RADIUS)
        )

        pygame.draw.circle(
            screen,
            (175, 90, 90),
            OBSTACLE.astype(int),
            int(SAFE_RADIUS),
            1
        )

        draw_target()

        if len(trail) > 1:
            pygame.draw.lines(
                screen,
                (70, 135, 235),
                False,
                [point.astype(int) for point in trail],
                2
            )

        draw_drone()

        target_distance = distance(
            env.state[:2],
            env.target.position()
        )

        predicted_distance = distance(
            env.state[:2],
            env.target.predict(TEACHER_LOOKAHEAD)
        )

        relative_speed = safe_norm(
            env.target.velocity() - env.state[2:4]
        )

        draw_text(
            "PHASE 4.32 — GUIDED MUZERO",
            15,
            12,
            (255, 220, 100)
        )

        draw_text(
            "Controller: Predictive prior + learned MCTS",
            15,
            38
        )

        draw_text(
            f"Target distance: {target_distance:.1f}",
            15,
            64
        )

        draw_text(
            f"Predicted target distance: {predicted_distance:.1f}",
            15,
            90
        )

        draw_text(
            f"Relative speed: {relative_speed:.1f}",
            15,
            116
        )

        draw_text(
            f"Drone speed: {safe_norm(env.state[2:4]):.1f}",
            15,
            142
        )

        draw_text(
            f"Selected action: {last_action}",
            15,
            168
        )

        draw_text(
            f"Search value: {last_value:.2f}",
            15,
            194
        )

        draw_text(
            f"Last reward: {last_reward:.2f}",
            15,
            220
        )

        draw_text(
            f"Simulation speed: {speed}x",
            15,
            246
        )

        draw_text(
            "MCTS visit policy:",
            15,
            280,
            (120, 200, 255)
        )

        for index, probability in enumerate(last_policy):
            bar_width = int(120 * probability)
            y = 305 + index * 23

            draw_text(
                f"A{index}: {probability:.2f}",
                15,
                y
            )

            pygame.draw.rect(
                screen,
                (75, 155, 225),
                (105, y + 3, bar_width, 10)
            )

        draw_text(
            "Green: actual target | Yellow: predicted target",
            15,
            WORLD_H - 80
        )

        draw_text(
            "SPACE pause | R reset | +/- speed",
            15,
            WORLD_H - 55
        )

        draw_text(
            "Click to move target center | ESC exit",
            15,
            WORLD_H - 32
        )

        if paused:
            draw_text(
                "PAUSED",
                WORLD_W - 115,
                15,
                (255, 210, 80)
            )

        pygame.display.flip()

    pygame.quit()


# ============================================================
# 16. UNIT TESTS
# ============================================================

def run_tests():
    print("\nPHASE 4.32 — UNIT TESTS")

    passed = 0

    def check(condition, name):
        nonlocal passed

        if not bool(condition):
            raise AssertionError(
                f"FAILED: {name}"
            )

        passed += 1
        print(f"[PASS] {name}")

    env = DroneEnv()
    observation = env.reset()

    check(
        observation.shape == (OBS_DIM,),
        "Observation dimension"
    )

    check(
        np.all(np.isfinite(observation)),
        "Finite observation"
    )

    check(
        ACTION_COUNT == 9,
        "Nine discrete actions"
    )

    check(
        np.all(np.abs(ACTION_VECTORS) <= 1.0),
        "Action bounds"
    )

    target = MovingTarget([700.0, 300.0])
    old_position = target.position().copy()
    target.advance()

    check(
        not np.allclose(old_position, target.position()),
        "Moving target update"
    )

    check(
        target.velocity().shape == (2,),
        "Target velocity"
    )

    check(
        target.predict(1.0).shape == (2,),
        "Target prediction"
    )

    next_state = true_drone_transition(
        np.array([100.0, 100.0, 0.0, 0.0]),
        np.array([1.0, 0.0])
    )

    check(
        next_state.shape == (4,),
        "Drone dynamics output"
    )

    bounded = clip_drone_state(
        np.array([100.0, 100.0, 500.0, 500.0])
    )

    check(
        safe_norm(bounded[2:4]) <= MAX_SPEED + 1e-5,
        "Maximum speed constraint"
    )

    result = env.step(0)

    check(
        len(result) == 5,
        "Environment step API"
    )

    teacher_policy = predictive_teacher_policy(observation)

    check(
        teacher_policy.shape == (ACTION_COUNT,),
        "Teacher policy dimension"
    )

    check(
        np.isclose(teacher_policy.sum(), 1.0),
        "Teacher policy normalization"
    )

    network = MuZeroNetwork().to(DEVICE)

    state_tensor = torch.as_tensor(
        observation,
        dtype=torch.float32,
        device=DEVICE
    ).unsqueeze(0)

    latent, logits, value = network.initial_inference(
        state_tensor
    )

    check(
        latent.shape == (1, LATENT_DIM),
        "Representation network"
    )

    check(
        logits.shape == (1, ACTION_COUNT),
        "Policy head"
    )

    check(
        value.shape == (1, 1),
        "Value head"
    )

    probabilities = torch.softmax(logits, dim=-1)

    check(
        torch.allclose(
            probabilities.sum(dim=-1),
            torch.ones(1, device=DEVICE),
            atol=1e-5
        ),
        "Network policy normalization"
    )

    (
        next_latent,
        predicted_reward,
        next_logits,
        next_value
    ) = network.recurrent_inference(
        latent,
        torch.tensor([2], dtype=torch.long, device=DEVICE)
    )

    check(
        next_latent.shape == (1, LATENT_DIM),
        "Latent dynamics"
    )

    check(
        predicted_reward.shape == (1, 1),
        "Reward prediction"
    )

    check(
        next_logits.shape == (1, ACTION_COUNT),
        "Recurrent policy prediction"
    )

    check(
        next_value.shape == (1, 1),
        "Recurrent value prediction"
    )

    edge = Edge(0.5)
    edge.value_sum = 6.0
    edge.visit_count = 3

    check(
        edge.value == 2.0,
        "MCTS edge mean value"
    )

    search = MuZeroMCTS(network, simulations=5)
    root = Node()

    search.expand(
        root,
        latent,
        logits.squeeze(0).detach().cpu().numpy(),
        value.item()
    )

    check(
        len(root.children) == ACTION_COUNT,
        "MCTS root expansion"
    )

    _, policy, root_value = search.run(
        observation,
        add_root_noise=False
    )

    check(
        policy.shape == (ACTION_COUNT,)
        and np.isclose(policy.sum(), 1.0),
        "MCTS normalized visit policy"
    )

    check(
        np.isfinite(root_value),
        "MCTS finite root value"
    )

    episode = Episode()
    episode.actions = [0, 1, 2]
    episode.rewards = [1.0, 2.0, 3.0]

    target_value = make_value_target(episode, 0)

    expected = (
        1.0
        + GAMMA * 2.0
        + GAMMA**2 * 3.0
    )

    check(
        abs(target_value - expected) < 1e-5,
        "Discounted Monte Carlo return"
    )

    replay = EpisodeReplay()
    replay.add(episode)

    check(
        len(replay) == 1,
        "Replay storage"
    )

    sampled_episode, sampled_index = (
        replay.sample_position()
    )

    check(
        sampled_episode is episode
        and 0 <= sampled_index < len(episode),
        "Replay sampling"
    )

    # Full training update.
    trainer = MuZeroTrainer()

    test_episode = Episode()
    test_env = DroneEnv()
    test_observation = test_env.reset()

    test_search = MuZeroMCTS(
        trainer.network,
        simulations=2
    )

    for _ in range(5):
        _, test_policy, test_value = test_search.run(
            test_observation,
            add_root_noise=False
        )

        test_teacher = predictive_teacher_policy(
            test_observation
        )

        guided = 0.5 * test_policy + 0.5 * test_teacher
        guided /= max(guided.sum(), 1e-12)

        action = int(np.argmax(guided))

        (
            next_observation,
            reward,
            terminated,
            truncated,
            info
        ) = test_env.step(action)

        test_episode.observations.append(
            test_observation.copy()
        )

        test_episode.actions.append(action)
        test_episode.rewards.append(reward)
        test_episode.policies.append(guided.copy())
        test_episode.teacher_policies.append(
            test_teacher.copy()
        )
        test_episode.root_values.append(test_value)
        test_episode.distances.append(info["distance"])

        test_observation = next_observation

        if terminated or truncated:
            break

    test_episode.observations.append(
        test_observation.copy()
    )

    trainer.replay.add(test_episode)

    loss = trainer.train_step()

    check(
        loss is not None and np.isfinite(loss["total"]),
        "MuZero training update"
    )

    check(
        all(np.isfinite(value) for value in loss.values()),
        "Finite training losses"
    )

    check(
        len(trainer.training_losses) == 1,
        "Training history"
    )

    # Checkpoint serialization test using a temporary file.
    test_checkpoint = "_muzero_test_checkpoint.pt"

    try:
        trainer.save(test_checkpoint)

        check(
            os.path.exists(test_checkpoint),
            "Checkpoint file creation"
        )

        restored = MuZeroTrainer()
        restored.load(test_checkpoint)

        check(
            all(
                torch.equal(a, b)
                for a, b in zip(
                    trainer.network.state_dict().values(),
                    restored.network.state_dict().values()
                )
            ),
            "Checkpoint model restoration"
        )

    finally:
        if os.path.exists(test_checkpoint):
            os.remove(test_checkpoint)

    # IMPORTANT:
    # Do not compare the number of checks with a hard-coded
    # number. Adding a test must not make all tests fail.
    print(f"\n{passed}/{passed} tests passed.")
    print("ALL UNIT TESTS PASSED.")

    return True


# ============================================================
# 17. MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--no-pygame",
        action="store_true",
        help="Skip Pygame visualization."
    )

    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use smaller training settings."
    )

    parser.add_argument(
        "--load",
        action="store_true",
        help="Continue training from an existing checkpoint."
    )

    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Only evaluate a saved checkpoint."
    )

    args = parser.parse_args()

    global TRAIN_EPISODES
    global EPISODE_STEPS
    global MCTS_SIMULATIONS
    global TRAINING_UPDATES_PER_EPISODE

    if args.fast:
        TRAIN_EPISODES = 15
        EPISODE_STEPS = 80
        MCTS_SIMULATIONS = 8
        TRAINING_UPDATES_PER_EPISODE = 4

    print("=" * 72)
    print("PHASE 4.32 — GUIDED MUZERO PREDICTIVE TRACKING")
    print("=" * 72)

    print(f"Device: {DEVICE}")
    print(f"Observation dimension: {OBS_DIM}")
    print(f"Action count: {ACTION_COUNT}")
    print(f"Latent dimension: {LATENT_DIM}")
    print(f"MCTS simulations: {MCTS_SIMULATIONS}")

    if not run_tests():
        raise RuntimeError("Unit tests failed.")

    trainer = MuZeroTrainer()
    history = None

    # Correct checkpoint resume:
    # use the same trainer instance after loading.
    if args.load or args.skip_training:
        trainer.load(CHECKPOINT_PATH)

    if not args.skip_training:
        trainer, history = train_muzero(trainer)

    print("\nEvaluating MuZero")

    evaluate_muzero(
        trainer,
        episodes=5
    )

    if history is not None:
        print("\nPlotting training")
        plot_training(history)

    if not args.no_pygame:
        print("\nStarting Pygame")
        run_pygame(trainer)

    print("\nPHASE 4.32 COMPLETED.")


if __name__ == "__main__":
    main()