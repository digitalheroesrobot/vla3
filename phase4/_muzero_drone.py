
"""
File: phase4/muzero_drone.py

PHASE 4.30 — MUZERO: LEARNED PLANNING WITH MCTS

============================================================
PURPOSE
============================================================

Train a drone controller that combines:
    - A learned latent representation
    - A learned latent dynamics model
    - Reward prediction
    - Policy and value prediction
    - Monte Carlo Tree Search (MCTS)

The agent does not use the true drone transition model
inside its planning algorithm.

============================================================
WHAT YOU LEARN
============================================================

STEP 1 — Environment
    Model a moving-target drone task with discrete actions.

STEP 2 — Representation network
    Convert an observation into a latent state.

STEP 3 — Dynamics network
    Predict the next latent state and immediate reward.

STEP 4 — Prediction network
    Predict action probabilities and state value.

STEP 5 — MCTS
    Use PUCT selection, expansion and value backup to
    improve action selection.

STEP 6 — Self-play data collection
    Collect real episodes using MCTS policies.

STEP 7 — MuZero training
    Train representation, dynamics and prediction networks
    using observed rewards and MCTS policy/value targets.

STEP 8 — Interactive visualization
    Control a moving-target drone with the trained agent.

============================================================
CORE MATHEMATICS
============================================================

Representation:

    z_0 = h_theta(observation)

Prediction:

    policy, value = f_theta(z_0)

    policy = softmax(policy_logits)

Dynamics:

    z_(k+1), reward_(k+1) = g_theta(z_k, action_k)

MCTS PUCT selection:

    score(s,a) =
        Q(s,a)
        + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))

Value backup:

    G_k = reward_(k+1) + gamma * G_(k+1)

Training objective:

    L =
        policy_loss
        + value_loss
        + reward_loss
        + latent_consistency_loss

Policy loss:

    L_policy = -sum_a pi_target(a) * log(pi_predicted(a))

Value loss:

    L_value = (v_target - v_predicted)^2

Reward loss:

    L_reward = (r_target - r_predicted)^2

============================================================
IMPORTANT DISTINCTIONS
============================================================

Ordinary model-based RL:
    Learns a model intended to predict physical state.

MuZero:
    Learns a latent model sufficient for planning.

MCTS:
    Evaluates alternative action sequences using the
    learned latent dynamics and predicted values.

The learned latent state is not required to correspond
directly to physical position, velocity or other variables.

============================================================
ROBOTICS APPLICATION
============================================================

- Autonomous drone navigation
- Mobile robot path planning
- Manipulation planning
- Learning action sequences from interaction
- Planning when an exact simulator is unavailable

SAFETY:
    This is a simplified educational 2D simulation.
    MCTS is not a safety controller, and the learned model
    is not guaranteed to predict real drone dynamics.
"""

from __future__ import annotations

import math
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt


# ============================================================
# 0. CONFIGURATION
# ============================================================

SEED = 430

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

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
    dtype=np.float32,
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
    dtype=np.float32,
)

OBS_DIM = 10

# Nine discrete actions:
# 0: no acceleration
# 1: left
# 2: right
# 3: up
# 4: down
# 5: up-left
# 6: up-right
# 7: down-left
# 8: down-right

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
    dtype=np.float32,
)

ACTION_COUNT = len(
    ACTION_VECTORS
)

LATENT_DIM = 128

GAMMA = 0.99

MCTS_SIMULATIONS = 35
PUCT_C = 1.5

TRAIN_EPISODES = 80
EPISODE_STEPS = 120

REPLAY_CAPACITY = 300
BATCH_SIZE = 32

UNROLL_STEPS = 5

LEARNING_RATE = 0.0005

TRAINING_UPDATES_PER_EPISODE = 8

VALUE_LOSS_WEIGHT = 1.0
REWARD_LOSS_WEIGHT = 1.0
LATENT_LOSS_WEIGHT = 0.5

# For a faster first run, try:
#
# TRAIN_EPISODES = 10
# EPISODE_STEPS = 70
# MCTS_SIMULATIONS = 12
# TRAINING_UPDATES_PER_EPISODE = 2
# BATCH_SIZE = 16


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

        p = (
            self.phase
            if phase is None
            else phase
        )

        return self.center + np.array(
            [
                TARGET_A * math.cos(p),
                TARGET_B * math.sin(p),
            ],
            dtype=np.float32,
        )

    def velocity(self, phase=None):

        p = (
            self.phase
            if phase is None
            else phase
        )

        return np.array(
            [
                -TARGET_A * TARGET_OMEGA * math.sin(p),
                TARGET_B * TARGET_OMEGA * math.cos(p),
            ],
            dtype=np.float32,
        )

    def advance(self):

        self.phase += TARGET_OMEGA * DT


# ============================================================
# 2. DRONE ENVIRONMENT
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


def true_drone_transition(
    state,
    action,
):

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
        MOTOR_GAIN * MAX_ACCEL * action
        - LINEAR_DRAG * velocity
        - QUADRATIC_DRAG
        * np.abs(velocity)
        * velocity
        + WIND
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


class DroneEnv:

    def __init__(self):

        self.target = MovingTarget(
            [710.0, 315.0]
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

        relative_target = (
            target_position - position
        )

        relative_obstacle = (
            OBSTACLE - position
        )

        phase_features = np.array(
            [
                math.sin(self.target.phase),
                math.cos(self.target.phase),
            ],
            dtype=np.float32,
        )

        return np.concatenate(
            [
                relative_target / 400.0,
                target_velocity / 200.0,
                velocity / MAX_SPEED,
                relative_obstacle / 400.0,
                phase_features,
            ]
        ).astype(
            np.float32
        )

    def reset(self):

        self.state = np.array(
            [
                110.0,
                500.0,
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

    def step(self, action_index):

        action_index = int(
            action_index
        )

        action = ACTION_VECTORS[
            action_index
        ]

        old_position = self.state[:2].copy()

        future_phase = (
            self.target.phase
            + TARGET_OMEGA * DT
        )

        future_target = self.target.position(
            future_phase
        )

        distance_before = np.linalg.norm(
            old_position - future_target
        )

        self.state = true_drone_transition(
            self.state,
            action,
        )

        self.target.advance()

        distance_after = np.linalg.norm(
            self.state[:2]
            - self.target.position()
        )

        progress = (
            distance_before - distance_after
        )

        obstacle_distance = np.linalg.norm(
            self.state[:2] - OBSTACLE
        )

        collision = (
            obstacle_distance < OBSTACLE_RADIUS
        )

        boundary_distance = min(
            self.state[0],
            WORLD_W - self.state[0],
            self.state[1],
            WORLD_H - self.state[1],
        )

        reward = (
            0.04 * progress
            - 0.015 * distance_after
            - 0.12 * np.sum(
                action**2
            )
        )

        obstacle_violation = max(
            0.0,
            SAFE_RADIUS - obstacle_distance,
        )

        reward -= (
            0.008 * obstacle_violation**2
        )

        boundary_violation = max(
            0.0,
            55.0 - boundary_distance,
        )

        reward -= (
            0.015 * boundary_violation**2
        )

        if collision:

            reward -= 80.0

        self.steps += 1

        terminated = collision

        truncated = (
            self.steps >= EPISODE_STEPS
        )

        return (
            self.observation(),
            float(reward),
            terminated,
            truncated,
            {
                "distance": float(
                    distance_after
                ),
                "collision": collision,
            },
        )


# ============================================================
# 3. MUZERO NETWORK
# ============================================================

class MuZeroNetwork(nn.Module):

    def __init__(self):

        super().__init__()

        # Representation:
        # observation -> latent state
        self.representation = nn.Sequential(
            nn.Linear(OBS_DIM, 256),
            nn.ReLU(),

            nn.Linear(256, LATENT_DIM),
            nn.Tanh(),
        )

        # Dynamics:
        # latent state + action -> next latent state
        self.dynamics = nn.Sequential(
            nn.Linear(
                LATENT_DIM + ACTION_COUNT,
                256,
            ),
            nn.ReLU(),

            nn.Linear(256, 256),
            nn.ReLU(),
        )

        self.next_latent = nn.Sequential(
            nn.Linear(256, LATENT_DIM),
            nn.Tanh(),
        )

        self.reward_head = nn.Linear(
            256,
            1,
        )

        # Prediction:
        # latent state -> policy and value
        self.prediction = nn.Sequential(
            nn.Linear(LATENT_DIM, 256),
            nn.ReLU(),
        )

        self.policy_head = nn.Linear(
            256,
            ACTION_COUNT,
        )

        self.value_head = nn.Linear(
            256,
            1,
        )

    def initial_inference(self, observation):

        latent = self.representation(
            observation
        )

        policy_logits, value = (
            self.prediction_inference(
                latent
            )
        )

        return (
            latent,
            policy_logits,
            value,
        )

    def prediction_inference(self, latent):

        features = self.prediction(
            latent
        )

        policy_logits = self.policy_head(
            features
        )

        value = self.value_head(
            features
        )

        return (
            policy_logits,
            value,
        )

    def recurrent_inference(
        self,
        latent,
        action_index,
    ):

        action_one_hot = F.one_hot(
            action_index.long(),
            num_classes=ACTION_COUNT,
        ).float()

        x = torch.cat(
            [
                latent,
                action_one_hot,
            ],
            dim=-1,
        )

        hidden = self.dynamics(
            x
        )

        next_latent = self.next_latent(
            hidden
        )

        reward = self.reward_head(
            hidden
        )

        policy_logits, value = (
            self.prediction_inference(
                next_latent
            )
        )

        return (
            next_latent,
            reward,
            policy_logits,
            value,
        )


# ============================================================
# 4. MCTS NODE
# ============================================================

class Node:

    def __init__(
        self,
        prior=0.0,
    ):

        self.prior = float(prior)

        self.visit_count = 0

        self.value_sum = 0.0

        self.reward = 0.0

        self.latent = None

        self.children = {}

    @property
    def value(self):

        if self.visit_count == 0:

            return 0.0

        return (
            self.value_sum
            / self.visit_count
        )

    def expanded(self):

        return len(
            self.children
        ) > 0


# ============================================================
# 5. MCTS
# ============================================================

class MuZeroMCTS:

    def __init__(
        self,
        network,
        simulations=MCTS_SIMULATIONS,
        c_puct=PUCT_C,
    ):

        self.network = network

        self.simulations = simulations

        self.c_puct = c_puct

    @staticmethod
    def softmax_numpy(logits):

        logits = np.asarray(
            logits,
            dtype=np.float64,
        )

        logits = logits - np.max(
            logits
        )

        probabilities = np.exp(
            logits
        )

        return probabilities / (
            probabilities.sum() + 1e-12
        )

    def expand(
        self,
        node,
        latent,
        policy_logits,
        reward=0.0,
    ):

        probabilities = self.softmax_numpy(
            policy_logits
        )

        node.latent = latent.detach()

        node.reward = float(
            reward
        )

        for action_index, probability in enumerate(
            probabilities
        ):

            node.children[action_index] = Node(
                prior=probability
            )

    def select_child(self, node):

        parent_visits = math.sqrt(
            node.visit_count + 1
        )

        best_action = None
        best_child = None
        best_score = -float("inf")

        for action_index, child in node.children.items():

            # PUCT balances exploitation and exploration.
            exploration = (
                self.c_puct
                * child.prior
                * parent_visits
                / (1 + child.visit_count)
            )

            score = (
                child.reward
                + GAMMA * child.value
                + exploration
            )

            if score > best_score:

                best_score = score

                best_action = action_index

                best_child = child

        return (
            best_action,
            best_child,
        )

    def run(
        self,
        observation,
        add_root_noise=True,
    ):

        observation_tensor = torch.as_tensor(
            observation,
            dtype=torch.float32,
            device=DEVICE,
        ).unsqueeze(0)

        with torch.no_grad():

            (
                root_latent,
                root_logits,
                root_value,
            ) = self.network.initial_inference(
                observation_tensor
            )

        root = Node()

        self.expand(
            root,
            root_latent,
            root_logits.squeeze(0).cpu().numpy(),
            reward=0.0,
        )

        # Root Dirichlet noise encourages exploration
        # during training, but is disabled in evaluation.
        if add_root_noise:

            noise = np.random.dirichlet(
                [0.3] * ACTION_COUNT
            )

            for action_index, child in root.children.items():

                child.prior = (
                    0.75 * child.prior
                    + 0.25 * noise[action_index]
                )

        root.visit_count = 1

        for _ in range(
            self.simulations
        ):

            node = root

            search_path = [
                node
            ]

            # ----------------------------------------
            # Selection
            # ----------------------------------------

            while node.expanded():

                action_index, child = (
                    self.select_child(
                        node
                    )
                )

                if child.visit_count == 0:

                    # --------------------------------
                    # Expansion through learned model
                    # --------------------------------

                    action_tensor = torch.tensor(
                        [action_index],
                        dtype=torch.long,
                        device=DEVICE,
                    )

                    with torch.no_grad():

                        (
                            next_latent,
                            predicted_reward,
                            next_logits,
                            next_value,
                        ) = self.network.recurrent_inference(
                            node.latent,
                            action_tensor,
                        )

                    self.expand(
                        child,
                        next_latent,
                        next_logits.squeeze(0).cpu().numpy(),
                        reward=predicted_reward.item(),
                    )

                    leaf_value = float(
                        next_value.item()
                    )

                    search_path.append(
                        child
                    )

                    break

                node = child

                search_path.append(
                    node
                )

            else:

                leaf_value = 0.0

            # ----------------------------------------
            # Backup
            # ----------------------------------------

            value = leaf_value

            for path_node in reversed(
                search_path
            ):

                path_node.value_sum += value

                path_node.visit_count += 1

                value = (
                    path_node.reward
                    + GAMMA * value
                )

        visit_counts = np.zeros(
            ACTION_COUNT,
            dtype=np.float32,
        )

        for action_index, child in root.children.items():

            visit_counts[action_index] = (
                child.visit_count
            )

        if visit_counts.sum() == 0:

            policy = np.ones(
                ACTION_COUNT,
                dtype=np.float32,
            ) / ACTION_COUNT

        else:

            policy = (
                visit_counts
                / visit_counts.sum()
            )

        root_value_estimate = float(
            root.value
        )

        return (
            root,
            policy,
            root_value_estimate,
        )


# ============================================================
# 6. EPISODE REPLAY
# ============================================================

class Episode:

    def __init__(self):

        self.observations = []

        self.actions = []

        self.rewards = []

        self.policies = []

        self.root_values = []

        self.terminated = False

    def __len__(self):

        return len(
            self.actions
        )


class EpisodeReplay:

    def __init__(
        self,
        capacity=REPLAY_CAPACITY,
    ):

        self.episodes = deque(
            maxlen=capacity
        )

    def add(self, episode):

        if len(episode) > 0:

            self.episodes.append(
                episode
            )

    def sample_position(self):

        episode = random.choice(
            list(self.episodes)
        )

        index = random.randrange(
            len(episode)
        )

        return (
            episode,
            index,
        )

    def __len__(self):

        return len(
            self.episodes
        )


# ============================================================
# 7. VALUE TARGETS
# ============================================================

def make_value_target(
    episode,
    index,
    td_steps=UNROLL_STEPS,
):

    """
    n-step value target:

        target =
            r[t+1]
            + gamma*r[t+2]
            + ...
            + gamma^n * V(s[t+n])

    If a true terminal state is reached, no bootstrap
    value is added.

    If the sampled horizon ends before termination,
    bootstrap from the stored MCTS root value.
    """

    value = 0.0

    bootstrap_index = index + td_steps

    for offset in range(
        td_steps
    ):

        reward_index = index + offset

        if reward_index >= len(
            episode.rewards
        ):

            break

        value += (
            GAMMA**offset
            * episode.rewards[reward_index]
        )

        if (
            episode.terminated
            and reward_index == len(episode.rewards) - 1
        ):

            return float(value)

    if bootstrap_index < len(
        episode.root_values
    ):

        value += (
            GAMMA**td_steps
            * episode.root_values[bootstrap_index]
        )

    return float(value)


# ============================================================
# 8. MUZERO TRAINER
# ============================================================

class MuZeroTrainer:

    def __init__(self):

        self.network = MuZeroNetwork().to(
            DEVICE
        )

        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=LEARNING_RATE,
        )

        self.replay = EpisodeReplay()

        self.training_losses = []

    def collect_episode(
        self,
        add_root_noise=True,
    ):

        env = DroneEnv()

        observation = env.reset()

        episode = Episode()

        search = MuZeroMCTS(
            self.network
        )

        for _ in range(
            EPISODE_STEPS
        ):

            (
                root,
                policy,
                root_value,
            ) = search.run(
                observation,
                add_root_noise=add_root_noise,
            )

            if add_root_noise:

                # Sample from the MCTS visit distribution.
                action_index = int(
                    np.random.choice(
                        ACTION_COUNT,
                        p=policy,
                    )
                )

            else:

                action_index = int(
                    np.argmax(policy)
                )

            (
                next_observation,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(
                action_index
            )

            episode.observations.append(
                observation.copy()
            )

            episode.actions.append(
                action_index
            )

            episode.rewards.append(
                reward
            )

            episode.policies.append(
                policy.copy()
            )

            episode.root_values.append(
                root_value
            )

            observation = next_observation

            if terminated or truncated:

                episode.terminated = terminated

                break

        # Store the final observation for training unrolls.
        episode.observations.append(
            observation.copy()
        )

        self.replay.add(
            episode
        )

        return episode

    def train_step(self):

        if len(self.replay) == 0:

            return None

        episode, start_index = (
            self.replay.sample_position()
        )

        # --------------------------------------------
        # Initial representation
        # --------------------------------------------

        observation = torch.as_tensor(
            episode.observations[start_index],
            dtype=torch.float32,
            device=DEVICE,
        ).unsqueeze(0)

        latent = self.network.representation(
            observation
        )

        total_policy_loss = torch.zeros(
            (),
            device=DEVICE,
        )

        total_value_loss = torch.zeros(
            (),
            device=DEVICE,
        )

        total_reward_loss = torch.zeros(
            (),
            device=DEVICE,
        )

        total_latent_loss = torch.zeros(
            (),
            device=DEVICE,
        )

        valid_steps = 0

        for unroll in range(
            UNROLL_STEPS + 1
        ):

            index = (
                start_index + unroll
            )

            if index >= len(
                episode.observations
            ):

                break

            (
                policy_logits,
                predicted_value,
            ) = self.network.prediction_inference(
                latent
            )

            # ----------------------------------------
            # Policy target
            # ----------------------------------------

            if index < len(
                episode.policies
            ):

                target_policy = torch.as_tensor(
                    episode.policies[index],
                    dtype=torch.float32,
                    device=DEVICE,
                ).unsqueeze(0)

                policy_loss = -(
                    target_policy
                    * F.log_softmax(
                        policy_logits,
                        dim=-1,
                    )
                ).sum(
                    dim=-1
                ).mean()

                total_policy_loss += policy_loss

            # ----------------------------------------
            # Value target
            # ----------------------------------------

            target_value = make_value_target(
                episode,
                index,
            )

            target_value_tensor = torch.tensor(
                [[target_value]],
                dtype=torch.float32,
                device=DEVICE,
            )

            value_loss = F.smooth_l1_loss(
                predicted_value,
                target_value_tensor,
            )

            total_value_loss += value_loss

            valid_steps += 1

            if unroll == UNROLL_STEPS:

                break

            if index >= len(
                episode.actions
            ):

                break

            # ----------------------------------------
            # Recurrent dynamics
            # ----------------------------------------

            action_tensor = torch.tensor(
                [episode.actions[index]],
                dtype=torch.long,
                device=DEVICE,
            )

            (
                next_latent,
                predicted_reward,
                _,
                _,
            ) = self.network.recurrent_inference(
                latent,
                action_tensor,
            )

            target_reward = torch.tensor(
                [[episode.rewards[index]]],
                dtype=torch.float32,
                device=DEVICE,
            )

            reward_loss = F.smooth_l1_loss(
                predicted_reward,
                target_reward,
            )

            total_reward_loss += reward_loss

            # ----------------------------------------
            # Latent consistency
            # ----------------------------------------

            next_observation = torch.as_tensor(
                episode.observations[index + 1],
                dtype=torch.float32,
                device=DEVICE,
            ).unsqueeze(0)

            with torch.no_grad():

                target_latent = (
                    self.network.representation(
                        next_observation
                    )
                )

            latent_loss = F.mse_loss(
                next_latent,
                target_latent,
            )

            total_latent_loss += latent_loss

            # Stop gradients through the recurrent latent
            # between unroll steps. This limits memory growth
            # and follows a common MuZero training practice.
            latent = next_latent.detach()

        if valid_steps == 0:

            return None

        total_loss = (
            total_policy_loss
            + VALUE_LOSS_WEIGHT * total_value_loss
            + REWARD_LOSS_WEIGHT * total_reward_loss
            + LATENT_LOSS_WEIGHT * total_latent_loss
        ) / valid_steps

        self.optimizer.zero_grad()

        total_loss.backward()

        nn.utils.clip_grad_norm_(
            self.network.parameters(),
            5.0,
        )

        self.optimizer.step()

        loss_value = float(
            total_loss.detach().cpu()
        )

        self.training_losses.append(
            loss_value
        )

        return loss_value


# ============================================================
# 9. TRAINING LOOP
# ============================================================

def train_muzero():

    trainer = MuZeroTrainer()

    episode_returns = []

    episode_distances = []

    for episode_index in range(
        TRAIN_EPISODES
    ):

        episode = trainer.collect_episode(
            add_root_noise=True
        )

        episode_return = float(
            np.sum(
                episode.rewards
            )
        )

        episode_returns.append(
            episode_return
        )

        # Evaluate the trajectory's target distances.
        distances = []

        env = DroneEnv()

        observation = env.reset()

        # This is a separate diagnostic rollout, not the
        # trajectory used for training.
        for _ in range(
            EPISODE_STEPS
        ):

            search = MuZeroMCTS(
                trainer.network,
                simulations=12,
            )

            _, policy, _ = search.run(
                observation,
                add_root_noise=False,
            )

            action = int(
                np.argmax(policy)
            )

            (
                observation,
                _,
                terminated,
                truncated,
                info,
            ) = env.step(
                action
            )

            distances.append(
                info["distance"]
            )

            if terminated or truncated:

                break

        episode_distances.append(
            float(
                np.mean(distances)
            )
        )

        losses = []

        for _ in range(
            TRAINING_UPDATES_PER_EPISODE
        ):

            loss = trainer.train_step()

            if loss is not None:

                losses.append(
                    loss
                )

        mean_loss = (
            float(np.mean(losses))
            if losses
            else float("nan")
        )

        print(
            f"Episode {episode_index + 1:3d}/{TRAIN_EPISODES}"
            f" | Return={episode_return:9.2f}"
            f" | Diagnostic distance={episode_distances[-1]:8.2f}"
            f" | Episode steps={len(episode):3d}"
            f" | Replay episodes={len(trainer.replay):3d}"
            f" | Loss={mean_loss:.5f}"
        )

    history = {
        "returns": episode_returns,
        "distances": episode_distances,
        "losses": trainer.training_losses,
    }

    return trainer, history


# ============================================================
# 10. EVALUATION
# ============================================================

def evaluate_muzero(
    trainer,
    episodes=5,
):

    env = DroneEnv()

    returns = []

    distances = []

    collisions = []

    search = MuZeroMCTS(
        trainer.network,
        simulations=MCTS_SIMULATIONS,
    )

    for _ in range(episodes):

        observation = env.reset()

        total_return = 0.0

        episode_distances = []

        collision_count = 0

        for _ in range(
            EPISODE_STEPS
        ):

            _, policy, _ = search.run(
                observation,
                add_root_noise=False,
            )

            action = int(
                np.argmax(policy)
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

            total_return += reward

            episode_distances.append(
                info["distance"]
            )

            if info["collision"]:

                collision_count += 1

            if terminated or truncated:

                break

        returns.append(
            total_return
        )

        distances.append(
            np.mean(
                episode_distances
            )
        )

        collisions.append(
            collision_count
        )

    results = {
        "mean_return": float(
            np.mean(returns)
        ),

        "mean_distance": float(
            np.mean(distances)
        ),

        "mean_collision_steps": float(
            np.mean(collisions)
        ),
    }

    print(
        "\nMuZero evaluation:"
    )

    for key, value in results.items():

        print(
            f"{key}: {value:.3f}"
        )

    return results


# ============================================================
# 11. VISUALIZATION
# ============================================================

def plot_training(history):

    plt.figure(
        figsize=(10, 5)
    )

    plt.plot(
        history["returns"],
        label="Episode return",
    )

    plt.xlabel(
        "Episode"
    )

    plt.ylabel(
        "Return"
    )

    plt.title(
        "MuZero Training Return"
    )

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    plt.show()

    plt.figure(
        figsize=(10, 5)
    )

    plt.plot(
        history["distances"],
        label="Diagnostic mean target distance",
    )

    plt.xlabel(
        "Episode"
    )

    plt.ylabel(
        "Distance"
    )

    plt.title(
        "Moving-Target Tracking"
    )

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    plt.show()

    if history["losses"]:

        plt.figure(
            figsize=(10, 5)
        )

        plt.plot(
            history["losses"]
        )

        plt.xlabel(
            "Training update"
        )

        plt.ylabel(
            "Loss"
        )

        plt.title(
            "MuZero Training Loss"
        )

        plt.grid(True)

        plt.tight_layout()

        plt.show()


# ============================================================
# 12. PYGAME DRONE VISUALIZATION
# ============================================================

def run_pygame(trainer):

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
        "Phase 4.30 — MuZero Drone Planning"
    )

    clock = pygame.time.Clock()

    font = pygame.font.SysFont(
        "consolas",
        15,
    )

    env = DroneEnv()

    observation = env.reset()

    search = MuZeroMCTS(
        trainer.network,
        simulations=MCTS_SIMULATIONS,
    )

    paused = False

    speed = 1

    trail = []

    last_policy = np.ones(
        ACTION_COUNT,
        dtype=np.float32,
    ) / ACTION_COUNT

    last_value = 0.0

    last_action = 0

    running = True

    def draw_text(
        message,
        x,
        y,
        color=(235, 235, 235),
    ):

        screen.blit(
            font.render(
                message,
                True,
                color,
            ),
            (x, y),
        )

    def draw_drone():

        center = env.state[:2]

        velocity = env.state[2:4]

        speed_value = np.linalg.norm(
            velocity
        )

        direction = (
            velocity / speed_value
            if speed_value > 1e-6
            else np.array(
                [1.0, 0.0]
            )
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
                    + 17 * forward * direction
                    + 17 * lateral * side
                )

                pygame.draw.line(
                    screen,
                    (220, 225, 235),
                    center.astype(int),
                    rotor.astype(int),
                    5,
                )

                pygame.draw.circle(
                    screen,
                    (90, 190, 245),
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

        pygame.draw.circle(
            screen,
            (255, 190, 60),
            (
                center + direction * 22
            ).astype(int),
            4,
        )

    def draw_target():

        orbit = []

        for index in range(100):

            phase = (
                env.target.phase
                + 2.0 * math.pi * index / 100
            )

            orbit.append(
                env.target.position(
                    phase
                ).astype(int)
            )

        pygame.draw.lines(
            screen,
            (65, 105, 75),
            True,
            orbit,
            2,
        )

        position = (
            env.target.position()
        )

        velocity = (
            env.target.velocity()
        )

        pygame.draw.circle(
            screen,
            (70, 240, 110),
            position.astype(int),
            11,
        )

        pygame.draw.line(
            screen,
            (70, 255, 120),
            position.astype(int),
            (
                position + 0.18 * velocity
            ).astype(int),
            3,
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

                    observation = env.reset()

                    trail.clear()

                elif event.key in (
                    pygame.K_EQUALS,
                    pygame.K_PLUS,
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

                    click = np.asarray(
                        event.pos,
                        dtype=np.float32,
                    )

                    if (
                        np.linalg.norm(
                            click - OBSTACLE
                        )
                        > SAFE_RADIUS
                    ):

                        env.target.center = np.clip(
                            click,
                            [
                                TARGET_A + 20,
                                TARGET_B + 20,
                            ],
                            [
                                WORLD_W - TARGET_A - 20,
                                WORLD_H - TARGET_B - 20,
                            ],
                        )

        if not paused:

            for _ in range(speed):

                (
                    _,
                    last_policy,
                    last_value,
                ) = search.run(
                    observation,
                    add_root_noise=False,
                )

                last_action = int(
                    np.argmax(last_policy)
                )

                (
                    observation,
                    reward,
                    terminated,
                    truncated,
                    info,
                ) = env.step(
                    last_action
                )

                trail.append(
                    env.state[:2].copy()
                )

                if len(trail) > 200:

                    trail.pop(0)

                if terminated or truncated:

                    observation = env.reset()

                    trail.clear()

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
            (175, 90, 90),
            OBSTACLE.astype(int),
            SAFE_RADIUS,
            1,
        )

        draw_target()

        if len(trail) > 1:

            pygame.draw.lines(
                screen,
                (70, 135, 235),
                False,
                [
                    point.astype(int)
                    for point in trail
                ],
                2,
            )

        draw_drone()

        distance = np.linalg.norm(
            env.state[:2]
            - env.target.position()
        )

        draw_text(
            "Controller: MuZero + MCTS",
            15,
            12,
        )

        draw_text(
            f"Target distance: {distance:.2f}",
            15,
            38,
        )

        draw_text(
            f"Selected action: {last_action}",
            15,
            64,
        )

        draw_text(
            f"Search value: {last_value:.2f}",
            15,
            90,
        )

        draw_text(
            f"Simulation speed: {speed}x",
            15,
            116,
        )

        draw_text(
            "MCTS visit policy:",
            15,
            150,
        )

        for index, probability in enumerate(
            last_policy
        ):

            draw_text(
                f"A{index}: {probability:.2f}",
                15,
                174 + index * 20,
            )

        draw_text(
            "SPACE pause | R reset | +/- speed | "
            "click target center | ESC exit",
            15,
            WORLD_H - 35,
        )

        if paused:

            draw_text(
                "PAUSED",
                WORLD_W - 115,
                15,
                (255, 210, 80),
            )

        pygame.display.flip()

    pygame.quit()


# ============================================================
# 13. UNIT TESTS
# ============================================================

def run_tests():

    print(
        "\nRunning unit tests..."
    )

    passed = 0

    def check(
        condition,
        description,
    ):

        nonlocal passed

        assert condition, description

        passed += 1

        print(
            f"[PASS] {description}"
        )

    # Test 1 — Observation shape.
    env = DroneEnv()

    observation = env.reset()

    check(
        observation.shape == (OBS_DIM,),
        "Observation dimension is correct",
    )

    # Test 2 — Finite observation.
    check(
        np.all(
            np.isfinite(observation)
        ),
        "Observation is finite",
    )

    # Test 3 — Moving target.
    target = MovingTarget(
        [700.0, 300.0]
    )

    previous = target.position().copy()

    target.advance()

    check(
        not np.allclose(
            previous,
            target.position(),
        ),
        "Target moves over time",
    )

    # Test 4 — Target velocity.
    check(
        target.velocity().shape == (2,),
        "Target velocity has two dimensions",
    )

    # Test 5 — Action count.
    check(
        ACTION_COUNT == 9,
        "Nine discrete actions are configured",
    )

    # Test 6 — Action bounds.
    check(
        np.all(
            np.abs(ACTION_VECTORS) <= 1.0
        ),
        "Discrete actions are bounded",
    )

    # Test 7 — Drone transition.
    next_state = true_drone_transition(
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
        next_state.shape == (4,),
        "Drone transition returns position and velocity",
    )

    # Test 8 — Speed limit.
    bounded = clip_drone_state(
        np.array(
            [100.0, 100.0, 500.0, 500.0]
        )
    )

    check(
        np.linalg.norm(
            bounded[2:4]
        ) <= MAX_SPEED + 1e-5,
        "Drone speed limit is enforced",
    )

    # Test 9 — Environment step.
    result = env.step(0)

    check(
        len(result) == 5,
        "Environment step returns five values",
    )

    # Test 10 — Network representation.
    network = MuZeroNetwork().to(
        DEVICE
    )

    state_tensor = torch.as_tensor(
        observation,
        dtype=torch.float32,
        device=DEVICE,
    ).unsqueeze(0)

    latent, logits, value = (
        network.initial_inference(
            state_tensor
        )
    )

    check(
        latent.shape == (1, LATENT_DIM),
        "Representation network returns latent state",
    )

    # Test 11 — Policy dimension.
    check(
        logits.shape == (1, ACTION_COUNT),
        "Prediction network returns action logits",
    )

    # Test 12 — Value dimension.
    check(
        value.shape == (1, 1),
        "Prediction network returns scalar value",
    )

    # Test 13 — Policy probabilities.
    probabilities = torch.softmax(
        logits,
        dim=-1,
    )

    check(
        torch.allclose(
            probabilities.sum(dim=-1),
            torch.ones(1, device=DEVICE),
            atol=1e-5,
        ),
        "Policy probabilities sum to one",
    )

    # Test 14 — Recurrent inference.
    action_tensor = torch.tensor(
        [2],
        dtype=torch.long,
        device=DEVICE,
    )

    (
        next_latent,
        predicted_reward,
        next_logits,
        next_value,
    ) = network.recurrent_inference(
        latent,
        action_tensor,
    )

    check(
        next_latent.shape == (1, LATENT_DIM),
        "Dynamics network predicts next latent state",
    )

    # Test 15 — Reward prediction.
    check(
        predicted_reward.shape == (1, 1),
        "Dynamics network predicts scalar reward",
    )

    # Test 16 — MCTS softmax.
    policy = MuZeroMCTS.softmax_numpy(
        np.array(
            [1.0, 2.0, 3.0]
        )
    )

    check(
        np.isclose(
            policy.sum(),
            1.0,
        )
        and np.all(policy >= 0.0),
        "MCTS prior probabilities are normalized",
    )

    # Test 17 — MCTS node value.
    node = Node()

    node.value_sum = 6.0

    node.visit_count = 3

    check(
        node.value == 2.0,
        "MCTS node computes mean value correctly",
    )

    # Test 18 — MCTS expansion.
    search = MuZeroMCTS(
        network,
        simulations=5,
    )

    root = Node()

    search.expand(
        root,
        latent,
        logits.squeeze(0).detach().cpu().numpy(),
    )

    check(
        len(root.children) == ACTION_COUNT,
        "MCTS expands one child per action",
    )

    # Test 19 — MCTS policy.
    _, search_policy, _ = search.run(
        observation,
        add_root_noise=False,
    )

    check(
        search_policy.shape == (ACTION_COUNT,)
        and np.isclose(
            search_policy.sum(),
            1.0,
        ),
        "MCTS returns normalized visit policy",
    )

    # Test 20 — Value target terminal behavior.
    episode = Episode()

    episode.actions = [0, 1]

    episode.rewards = [1.0, 2.0]

    episode.root_values = [5.0, 5.0]

    episode.terminated = True

    target_value = make_value_target(
        episode,
        0,
        td_steps=5,
    )

    expected = (
        1.0 + GAMMA * 2.0
    )

    check(
        abs(
            target_value - expected
        ) < 1e-5,
        "Terminal value target does not bootstrap",
    )

    # Test 21 — Value target bootstrap.
    episode = Episode()

    episode.actions = [0, 1, 2, 3]

    episode.rewards = [1.0, 1.0, 1.0, 1.0]

    episode.root_values = [
        0.0,
        0.0,
        0.0,
        2.0,
        2.0,
    ]

    episode.terminated = False

    target_value = make_value_target(
        episode,
        0,
        td_steps=3,
    )

    expected = (
        1.0
        + GAMMA
        + GAMMA**2
        + GAMMA**3 * 2.0
    )

    check(
        abs(
            target_value - expected
        ) < 1e-5,
        "Nonterminal value target bootstraps correctly",
    )

    # Test 22 — Replay episode.
    replay = EpisodeReplay()

    replay.add(
        episode
    )

    check(
        len(replay) == 1,
        "Episode replay stores episodes",
    )

    # Test 23 — Replay sampling.
    sampled_episode, sampled_index = (
        replay.sample_position()
    )

    check(
        sampled_episode is episode
        and 0 <= sampled_index < len(episode),
        "Episode replay samples valid positions",
    )

    # Test 24 — Actor-independent MCTS output.
    check(
        np.all(
            np.isfinite(search_policy)
        ),
        "MCTS policy contains finite values",
    )

    print(
        f"\n{passed}/24 tests passed."
    )

    return passed == 24


# ============================================================
# 14. MAIN
# ============================================================

def main():

    print(
        "=" * 72
    )

    print(
        "PHASE 4.30 — MUZERO AND LEARNED PLANNING"
    )

    print(
        "=" * 72
    )

    print(
        f"Device: {DEVICE}"
    )

    if not run_tests():

        raise RuntimeError(
            "Unit tests failed."
        )

    print(
        "\n[1/4] Training MuZero..."
    )

    trainer, history = train_muzero()

    print(
        "\n[2/4] Evaluating MuZero..."
    )

    evaluate_muzero(
        trainer,
        episodes=5,
    )

    print(
        "\n[3/4] Plotting training results..."
    )

    plot_training(
        history
    )

    print(
        "\n[4/4] Starting interactive drone visualization..."
    )

    run_pygame(
        trainer
    )

    print(
        "\nPhase 4.30 completed."
    )


if __name__ == "__main__":

    main()
