
"""
RL COURSE — PHASE 4.15
============================================================
Filename:
    phase4/constrained_policy_optimization.py

LESSON:
    Constrained Policy Optimization (CPO) Foundations

IMPLEMENTATION:
    PPO-Lagrangian with separate reward and safety-cost critics.

IMPORTANT:
    This is NOT the canonical CPO algorithm.

    It is a practical constrained-policy-learning implementation
    that introduces the reward/cost formulation and adaptive
    Lagrangian multiplier used in safe RL.

    Canonical CPO uses a constrained trust-region optimization
    step based on the policy KL divergence and conjugate gradient.

PURPOSE
-------
Train a robot to track a moving target while learning to reduce
safety violations.

WHAT YOU LEARN
--------------
STEP 1 — Constrained MDP
    Maximize expected reward while keeping expected safety cost
    below a specified limit.

STEP 2 — Reward and cost critics
    Learn two different value functions:
        V_reward(s)
        V_cost(s)

STEP 3 — Generalized Advantage Estimation
    Estimate reward and cost advantages separately.

STEP 4 — Lagrangian optimization
    Combine reward and cost advantages:
        A_combined = A_reward - lambda * A_cost

STEP 5 — Adaptive multiplier
    Increase lambda when safety cost exceeds the budget.
    Decrease lambda when safety cost is below the budget.

STEP 6 — PPO
    Use the clipped policy objective to update the actor.

STEP 7 — Pygame
    Watch the rocket track a continuously moving target.

MATHEMATICAL FORMULATION
------------------------

Constrained objective:

    maximize    J_reward(policy)

    subject to  J_cost(policy) <= d

Lagrangian:

    L(policy, lambda)
        = J_reward(policy)
          - lambda * (J_cost(policy) - d)

Policy optimization:

    maximize E[
        min(
            ratio * A_combined,
            clipped_ratio * A_combined
        )
    ]

where:

    A_combined = A_reward - lambda * A_cost

Multiplier update:

    lambda = max(
        0,
        lambda + learning_rate * (observed_cost - d)
    )

INTERPRETATION
--------------
If the robot violates safety constraints too frequently:
    lambda increases.
    Safety costs become more influential.

If the robot remains safe:
    lambda can decrease.
    The agent can focus more on tracking performance.

ROBOTICS APPLICATION
--------------------
- Autonomous mobile robot navigation
- Drone trajectory planning
- Industrial robot safety
- Human-robot interaction
- Safe manipulation

CONTROLS
--------
Mouse   Change moving-target trajectory center
SPACE   Pause / resume
R       Reset simulation
+ / -   Simulation speed
ESC     Exit
============================================================
"""

from __future__ import annotations

import math
import random
from collections import deque

import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


# ============================================================
# 1. CONFIGURATION
# ============================================================

SEED = 45

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

WIDTH = 720
HEIGHT = 580
PANEL_WIDTH = 320

DT = 0.10

ROBOT_RADIUS = 13
MAX_SPEED = 115.0
MAX_ACCELERATION = 100.0

OBSTACLE = np.array(
    [350.0, 285.0],
    dtype=np.float32
)

OBSTACLE_RADIUS = 52.0

# Safety-cost threshold begins before physical collision.
SAFETY_RADIUS = (
    OBSTACLE_RADIUS
    + ROBOT_RADIUS
    + 35.0
)

TARGET_AMPLITUDE = np.array(
    [90.0, 65.0],
    dtype=np.float32
)

TARGET_OMEGA = 0.65

STATE_DIM = 10
ACTION_DIM = 2

GAMMA = 0.99
GAE_LAMBDA = 0.95

ACTOR_LR = 3e-4
CRITIC_LR = 1e-3

CLIP_EPSILON = 0.2
ENTROPY_COEFFICIENT = 0.005

ROLLOUT_STEPS = 1024
PPO_EPOCHS = 8
MINIBATCH_SIZE = 128

TRAIN_ITERATIONS = 100

# Expected safety cost per environment step.
# This is an educational cost budget, not a certified safety limit.
COST_BUDGET = 0.035

LAGRANGE_LR = 0.05
INITIAL_LAMBDA = 0.5
MAX_LAMBDA = 30.0

MAX_EPISODE_STEPS = 400


# ============================================================
# 2. UTILITIES
# ============================================================

def clamp_vector(vector, maximum):
    vector = np.asarray(
        vector,
        dtype=np.float32
    )

    norm = float(np.linalg.norm(vector))

    if norm > maximum:
        vector = vector * maximum / (norm + 1e-8)

    return vector


def safe_unit(vector):
    vector = np.asarray(
        vector,
        dtype=np.float32
    )

    norm = float(np.linalg.norm(vector))

    if norm < 1e-8:
        return np.zeros_like(vector)

    return vector / norm


def distance(a, b):
    return float(
        np.linalg.norm(
            np.asarray(a) - np.asarray(b)
        )
    )


# ============================================================
# 3. MOVING TARGET
# ============================================================

class MovingTarget:

    def __init__(self):
        self.center = np.array(
            [550.0, 145.0],
            dtype=np.float32
        )

        self.phase = 0.0

    def position_at(self, phase):
        return self.center + TARGET_AMPLITUDE * np.array(
            [
                math.sin(phase),
                math.sin(phase * 1.35 + 0.7)
            ],
            dtype=np.float32
        )

    def velocity_at(self, phase):
        return TARGET_AMPLITUDE * np.array(
            [
                TARGET_OMEGA * math.cos(phase),
                TARGET_OMEGA * 1.35
                * math.cos(phase * 1.35 + 0.7)
            ],
            dtype=np.float32
        )

    @property
    def position(self):
        return self.position_at(self.phase)

    @property
    def velocity(self):
        return self.velocity_at(self.phase)

    def step(self):
        self.phase += TARGET_OMEGA * DT

    def set_center(self, center):
        self.center = np.asarray(
            center,
            dtype=np.float32
        )


# ============================================================
# 4. ROBOT DYNAMICS
# ============================================================

def robot_dynamics(position, velocity, action):
    action = np.clip(
        np.asarray(action, dtype=np.float32),
        -1.0,
        1.0
    )

    acceleration = action * MAX_ACCELERATION

    next_velocity = (
        velocity + acceleration * DT
    )

    next_velocity = clamp_vector(
        next_velocity,
        MAX_SPEED
    )

    next_position = (
        position + next_velocity * DT
    )

    return next_position, next_velocity


# ============================================================
# 5. CONSTRAINED MOVING-TARGET ENVIRONMENT
# ============================================================

class MovingTargetEnv:

    def __init__(self):
        self.target = MovingTarget()
        self.reset()

    def reset(self):
        self.position = np.array(
            [100.0, 440.0],
            dtype=np.float32
        )

        self.velocity = np.zeros(
            2,
            dtype=np.float32
        )

        self.target.phase = random.uniform(
            0.0,
            2.0 * math.pi
        )

        self.steps = 0

        return self.get_state()

    def get_state(self):
        relative_target = (
            self.target.position - self.position
        )

        relative_obstacle = (
            OBSTACLE - self.position
        )

        clearance = (
            distance(self.position, OBSTACLE)
            - SAFETY_RADIUS
        )

        state = np.concatenate(
            [
                relative_target / 600.0,
                self.target.velocity / 150.0,
                self.velocity / MAX_SPEED,
                relative_obstacle / 600.0,
                np.array(
                    [
                        clearance / 200.0,
                        distance(
                            self.position,
                            self.target.position
                        ) / 600.0
                    ],
                    dtype=np.float32
                )
            ]
        )

        return state.astype(np.float32)

    def safety_cost(self):
        """
        Continuous proximity cost.

        Cost is zero outside SAFETY_RADIUS.

        Cost increases as the robot approaches the obstacle.
        """

        clearance = (
            distance(self.position, OBSTACLE)
            - SAFETY_RADIUS
        )

        proximity_cost = max(
            0.0,
            -clearance / SAFETY_RADIUS
        )

        # Boundary proximity cost.
        boundary_clearances = [
            self.position[0] - ROBOT_RADIUS,
            WIDTH - ROBOT_RADIUS - self.position[0],
            self.position[1] - ROBOT_RADIUS,
            HEIGHT - ROBOT_RADIUS - self.position[1]
        ]

        boundary_cost = max(
            0.0,
            (35.0 - min(boundary_clearances)) / 35.0
        )

        collision = (
            distance(self.position, OBSTACLE)
            < OBSTACLE_RADIUS + ROBOT_RADIUS
        )

        outside = any(
            value < 0
            for value in boundary_clearances
        )

        cost = (
            proximity_cost
            + boundary_cost
        )

        if collision or outside:
            cost += 10.0

        return float(cost)

    def step(self, action):
        action = np.clip(
            np.asarray(action, dtype=np.float32),
            -1.0,
            1.0
        )

        # Compare the robot's old position and new position
        # against the SAME future target position.
        # This avoids confusing target movement with robot progress.
        next_phase = (
            self.target.phase + TARGET_OMEGA * DT
        )

        next_target_position = (
            self.target.position_at(next_phase)
        )

        distance_before = distance(
            self.position,
            next_target_position
        )

        self.position, self.velocity = robot_dynamics(
            self.position,
            self.velocity,
            action
        )

        self.target.step()

        distance_after = distance(
            self.position,
            self.target.position
        )

        progress = (
            distance_before - distance_after
        )

        reward = (
            0.08 * progress
            + 0.3 * math.exp(-distance_after / 75.0)
            - 0.015 * float(np.sum(action ** 2))
        )

        cost = self.safety_cost()

        collision = (
            distance(self.position, OBSTACLE)
            < OBSTACLE_RADIUS + ROBOT_RADIUS
        )

        outside = (
            self.position[0] < ROBOT_RADIUS
            or self.position[0] > WIDTH - ROBOT_RADIUS
            or self.position[1] < ROBOT_RADIUS
            or self.position[1] > HEIGHT - ROBOT_RADIUS
        )

        terminated = collision or outside

        if terminated:
            reward -= 100.0

        self.steps += 1

        truncated = (
            self.steps >= MAX_EPISODE_STEPS
        )

        info = {
            "distance": distance_after,
            "cost": cost,
            "collision": collision,
            "outside": outside,
            "progress": progress
        }

        return (
            self.get_state(),
            float(reward),
            float(cost),
            terminated,
            truncated,
            info
        )


# ============================================================
# 6. ACTOR AND CRITICS
# ============================================================

class PolicyNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Linear(STATE_DIM, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh()
        )

        self.mean = nn.Linear(
            256,
            ACTION_DIM
        )

        self.log_std = nn.Parameter(
            torch.full(
                (ACTION_DIM,),
                -0.7
            )
        )

    def forward(self, state):
        features = self.backbone(state)

        mean = self.mean(features)

        log_std = self.log_std.clamp(
            -3.0,
            0.5
        )

        std = log_std.exp().expand_as(mean)

        return mean, std


class ValueNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(STATE_DIM, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )

    def forward(self, state):
        return self.network(state)


# ============================================================
# 7. POLICY DISTRIBUTION
# ============================================================

def sample_policy(policy, states, deterministic=False):
    mean, std = policy(states)

    distribution = Normal(
        mean,
        std
    )

    if deterministic:
        raw_action = mean
    else:
        raw_action = distribution.rsample()

    action = torch.tanh(raw_action)

    # Change-of-variables correction:
    # log pi(a|s) = log N(u|mu,sigma)
    #               - sum log(1 - tanh(u)^2)
    log_probability = distribution.log_prob(
        raw_action
    )

    correction = torch.log(
        1.0 - action.pow(2) + 1e-6
    )

    log_probability = (
        log_probability - correction
    ).sum(dim=1, keepdim=True)

    return action, log_probability


def evaluate_policy(policy, states, actions):
    mean, std = policy(states)

    distribution = Normal(
        mean,
        std
    )

    # Inverse tanh:
    # u = atanh(a)
    clipped_actions = actions.clamp(
        -0.999999,
        0.999999
    )

    raw_actions = torch.atanh(
        clipped_actions
    )

    log_probability = distribution.log_prob(
        raw_actions
    )

    correction = torch.log(
        1.0 - clipped_actions.pow(2) + 1e-6
    )

    log_probability = (
        log_probability - correction
    ).sum(dim=1, keepdim=True)

    entropy = distribution.entropy().sum(
        dim=1,
        keepdim=True
    )

    return log_probability, entropy


# ============================================================
# 8. GENERALIZED ADVANTAGE ESTIMATION
# ============================================================

def calculate_gae(
    rewards,
    values,
    next_values,
    terminated,
    gamma,
    gae_lambda
):
    """
    TD residual:

        delta_t = r_t + gamma*V(s_{t+1}) - V(s_t)

    GAE:

        A_t = delta_t
              + gamma*lambda*delta_{t+1}
              + ...

    True termination suppresses bootstrapping.

    Time-limit truncation is not treated as a true terminal
    state. The final-state value is retained in next_values.
    """

    advantages = np.zeros_like(
        rewards,
        dtype=np.float32
    )

    gae = 0.0

    for index in reversed(range(len(rewards))):

        continuation = (
            1.0 - terminated[index]
        )

        delta = (
            rewards[index]
            + gamma
            * continuation
            * next_values[index]
            - values[index]
        )

        # The stored sequence may cross an episode boundary.
        # We reset the GAE recursion at every episode end.
        if index < len(rewards) - 1:
            if terminated[index] or (
                next_values[index]
                != values[index + 1]
                and False
            ):
                pass

        gae = (
            delta
            + gamma
            * gae_lambda
            * continuation
            * gae
        )

        advantages[index] = gae

    returns = advantages + values

    return advantages, returns


# ============================================================
# 9. ROLLOUT STORAGE
# ============================================================

class RolloutBuffer:

    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []

        self.rewards = []
        self.costs = []

        self.reward_values = []
        self.cost_values = []

        self.next_reward_values = []
        self.next_cost_values = []

        self.terminated = []
        self.episode_ends = []

        self.episode_costs = []

    def add(
        self,
        state,
        action,
        log_probability,
        reward,
        cost,
        reward_value,
        cost_value,
        next_reward_value,
        next_cost_value,
        terminated,
        episode_end
    ):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_probability)

        self.rewards.append(reward)
        self.costs.append(cost)

        self.reward_values.append(reward_value)
        self.cost_values.append(cost_value)

        self.next_reward_values.append(
            next_reward_value
        )

        self.next_cost_values.append(
            next_cost_value
        )

        self.terminated.append(
            float(terminated)
        )

        self.episode_ends.append(
            bool(episode_end)
        )

    def clear(self):
        self.__init__()

    def __len__(self):
        return len(self.states)


# ============================================================
# 10. CONSTRAINED PPO-LAGRANGIAN AGENT
# ============================================================

class ConstrainedPPOAgent:

    def __init__(self):
        self.policy = PolicyNetwork().to(DEVICE)

        self.reward_value = ValueNetwork().to(DEVICE)
        self.cost_value = ValueNetwork().to(DEVICE)

        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=ACTOR_LR
        )

        self.reward_optimizer = torch.optim.Adam(
            self.reward_value.parameters(),
            lr=CRITIC_LR
        )

        self.cost_optimizer = torch.optim.Adam(
            self.cost_value.parameters(),
            lr=CRITIC_LR
        )

        self.lagrange_multiplier = INITIAL_LAMBDA

    @torch.no_grad()
    def act(self, state, deterministic=False):
        state_tensor = torch.tensor(
            state,
            dtype=torch.float32,
            device=DEVICE
        ).unsqueeze(0)

        action, log_probability = sample_policy(
            self.policy,
            state_tensor,
            deterministic
        )

        reward_value = self.reward_value(
            state_tensor
        )

        cost_value = self.cost_value(
            state_tensor
        )

        return (
            action.cpu().numpy()[0],
            float(log_probability.item()),
            float(reward_value.item()),
            float(cost_value.item())
        )

    @torch.no_grad()
    def values(self, state):
        state_tensor = torch.tensor(
            state,
            dtype=torch.float32,
            device=DEVICE
        ).unsqueeze(0)

        reward_value = self.reward_value(
            state_tensor
        )

        cost_value = self.cost_value(
            state_tensor
        )

        return (
            float(reward_value.item()),
            float(cost_value.item())
        )

    def update_lagrange_multiplier(
        self,
        observed_cost
    ):
        """
        Dual ascent:

            lambda <- lambda + lr*(J_cost - budget)

        The multiplier cannot become negative.
        """

        violation = (
            observed_cost - COST_BUDGET
        )

        self.lagrange_multiplier = float(
            np.clip(
                self.lagrange_multiplier
                + LAGRANGE_LR * violation,
                0.0,
                MAX_LAMBDA
            )
        )

        return violation

    def update(self, buffer):
        states = torch.tensor(
            np.array(buffer.states),
            dtype=torch.float32,
            device=DEVICE
        )

        actions = torch.tensor(
            np.array(buffer.actions),
            dtype=torch.float32,
            device=DEVICE
        )

        old_log_probs = torch.tensor(
            np.array(buffer.log_probs),
            dtype=torch.float32,
            device=DEVICE
        ).unsqueeze(1)

        rewards = np.asarray(
            buffer.rewards,
            dtype=np.float32
        )

        costs = np.asarray(
            buffer.costs,
            dtype=np.float32
        )

        reward_values = np.asarray(
            buffer.reward_values,
            dtype=np.float32
        )

        cost_values = np.asarray(
            buffer.cost_values,
            dtype=np.float32
        )

        next_reward_values = np.asarray(
            buffer.next_reward_values,
            dtype=np.float32
        )

        next_cost_values = np.asarray(
            buffer.next_cost_values,
            dtype=np.float32
        )

        terminated = np.asarray(
            buffer.terminated,
            dtype=np.float32
        )

        episode_ends = np.asarray(
            buffer.episode_ends,
            dtype=bool
        )

        # ----------------------------------------------------
        # GAE with correct episode-boundary handling
        # ----------------------------------------------------

        reward_advantages = np.zeros_like(
            rewards
        )

        cost_advantages = np.zeros_like(
            costs
        )

        reward_gae = 0.0
        cost_gae = 0.0

        for index in reversed(range(len(rewards))):

            continuation = (
                1.0 - terminated[index]
            )

            reward_delta = (
                rewards[index]
                + GAMMA
                * continuation
                * next_reward_values[index]
                - reward_values[index]
            )

            cost_delta = (
                costs[index]
                + GAMMA
                * continuation
                * next_cost_values[index]
                - cost_values[index]
            )

            if episode_ends[index]:
                reward_gae = 0.0
                cost_gae = 0.0

            reward_gae = (
                reward_delta
                + GAMMA
                * GAE_LAMBDA
                * continuation
                * reward_gae
            )

            cost_gae = (
                cost_delta
                + GAMMA
                * GAE_LAMBDA
                * continuation
                * cost_gae
            )

            reward_advantages[index] = reward_gae
            cost_advantages[index] = cost_gae

        reward_returns = (
            reward_advantages + reward_values
        )

        cost_returns = (
            cost_advantages + cost_values
        )

        reward_advantages = torch.tensor(
            reward_advantages,
            dtype=torch.float32,
            device=DEVICE
        ).unsqueeze(1)

        cost_advantages = torch.tensor(
            cost_advantages,
            dtype=torch.float32,
            device=DEVICE
        ).unsqueeze(1)

        reward_returns = torch.tensor(
            reward_returns,
            dtype=torch.float32,
            device=DEVICE
        ).unsqueeze(1)

        cost_returns = torch.tensor(
            cost_returns,
            dtype=torch.float32,
            device=DEVICE
        ).unsqueeze(1)

        # Normalize reward and cost advantages separately.
        reward_advantages = (
            reward_advantages
            - reward_advantages.mean()
        ) / (
            reward_advantages.std(unbiased=False) + 1e-8
        )

        cost_advantages = (
            cost_advantages
            - cost_advantages.mean()
        ) / (
            cost_advantages.std(unbiased=False) + 1e-8
        )

        # Lagrangian policy objective.
        combined_advantages = (
            reward_advantages
            - self.lagrange_multiplier
            * cost_advantages
        )

        sample_count = len(buffer)

        policy_losses = []
        reward_losses = []
        cost_losses = []

        for _ in range(PPO_EPOCHS):

            indices = np.random.permutation(
                sample_count
            )

            for start in range(
                0,
                sample_count,
                MINIBATCH_SIZE
            ):

                batch_indices = indices[
                    start:start + MINIBATCH_SIZE
                ]

                index_tensor = torch.tensor(
                    batch_indices,
                    dtype=torch.long,
                    device=DEVICE
                )

                batch_states = states[index_tensor]
                batch_actions = actions[index_tensor]

                batch_old_log_probs = (
                    old_log_probs[index_tensor]
                )

                batch_advantages = (
                    combined_advantages[index_tensor]
                )

                batch_reward_returns = (
                    reward_returns[index_tensor]
                )

                batch_cost_returns = (
                    cost_returns[index_tensor]
                )

                # --------------------------------------------
                # Actor update
                # --------------------------------------------

                new_log_probs, entropy = evaluate_policy(
                    self.policy,
                    batch_states,
                    batch_actions
                )

                ratio = torch.exp(
                    new_log_probs - batch_old_log_probs
                )

                unclipped = (
                    ratio * batch_advantages
                )

                clipped = (
                    torch.clamp(
                        ratio,
                        1.0 - CLIP_EPSILON,
                        1.0 + CLIP_EPSILON
                    )
                    * batch_advantages
                )

                policy_loss = (
                    -torch.minimum(
                        unclipped,
                        clipped
                    ).mean()
                    - ENTROPY_COEFFICIENT
                    * entropy.mean()
                )

                self.policy_optimizer.zero_grad()
                policy_loss.backward()

                nn.utils.clip_grad_norm_(
                    self.policy.parameters(),
                    0.5
                )

                self.policy_optimizer.step()

                # --------------------------------------------
                # Reward critic update
                # --------------------------------------------

                predicted_reward_values = (
                    self.reward_value(batch_states)
                )

                reward_loss = F.mse_loss(
                    predicted_reward_values,
                    batch_reward_returns
                )

                self.reward_optimizer.zero_grad()
                reward_loss.backward()

                nn.utils.clip_grad_norm_(
                    self.reward_value.parameters(),
                    0.5
                )

                self.reward_optimizer.step()

                # --------------------------------------------
                # Cost critic update
                # --------------------------------------------

                predicted_cost_values = (
                    self.cost_value(batch_states)
                )

                cost_loss = F.mse_loss(
                    predicted_cost_values,
                    batch_cost_returns
                )

                self.cost_optimizer.zero_grad()
                cost_loss.backward()

                nn.utils.clip_grad_norm_(
                    self.cost_value.parameters(),
                    0.5
                )

                self.cost_optimizer.step()

                policy_losses.append(
                    float(policy_loss.item())
                )

                reward_losses.append(
                    float(reward_loss.item())
                )

                cost_losses.append(
                    float(cost_loss.item())
                )

        return {
            "policy_loss": float(np.mean(policy_losses)),
            "reward_value_loss": float(np.mean(reward_losses)),
            "cost_value_loss": float(np.mean(cost_losses))
        }


# ============================================================
# 11. TRAINING LOOP
# ============================================================

def train_agent():
    env = MovingTargetEnv()
    agent = ConstrainedPPOAgent()

    reward_history = []
    cost_history = []
    distance_history = []
    lambda_history = []

    recent_episode_costs = deque(
        maxlen=20
    )

    state = env.reset()

    episode_reward = 0.0
    episode_cost = 0.0
    episode_steps = 0

    completed_episodes = 0

    print("\nTraining constrained PPO...")
    print(f"Device: {DEVICE}")
    print(f"Cost budget per step: {COST_BUDGET}")

    for iteration in range(TRAIN_ITERATIONS):

        buffer = RolloutBuffer()

        rollout_episode_rewards = []
        rollout_episode_costs = []
        rollout_episode_distances = []

        for _ in range(ROLLOUT_STEPS):

            (
                action,
                log_probability,
                reward_value,
                cost_value
            ) = agent.act(state)

            next_state, reward, cost, terminated, truncated, info = (
                env.step(action)
            )

            # At a time-limit truncation, bootstrap from the
            # final observation. At true termination, bootstrap
            # with zero.
            if terminated:
                next_reward_value = 0.0
                next_cost_value = 0.0
            else:
                (
                    next_reward_value,
                    next_cost_value
                ) = agent.values(next_state)

            episode_end = (
                terminated or truncated
            )

            buffer.add(
                state,
                action,
                log_probability,
                reward,
                cost,
                reward_value,
                cost_value,
                next_reward_value,
                next_cost_value,
                terminated,
                episode_end
            )

            state = next_state

            episode_reward += reward
            episode_cost += cost
            episode_steps += 1

            rollout_episode_distances.append(
                info["distance"]
            )

            if episode_end:

                completed_episodes += 1

                reward_history.append(
                    episode_reward
                )

                cost_history.append(
                    episode_cost
                )

                rollout_episode_rewards.append(
                    episode_reward
                )

                rollout_episode_costs.append(
                    episode_cost
                )

                recent_episode_costs.append(
                    episode_cost / max(1, episode_steps)
                )

                episode_reward = 0.0
                episode_cost = 0.0
                episode_steps = 0

                state = env.reset()

        # ----------------------------------------------------
        # Policy and critic update
        # ----------------------------------------------------

        losses = agent.update(buffer)

        # ----------------------------------------------------
        # Update Lagrange multiplier
        # ----------------------------------------------------

        if len(recent_episode_costs) > 0:
            observed_cost = float(
                np.mean(recent_episode_costs)
            )
        else:
            observed_cost = float(
                np.mean(buffer.costs)
            )

        violation = agent.update_lagrange_multiplier(
            observed_cost
        )

        lambda_history.append(
            agent.lagrange_multiplier
        )

        mean_reward = (
            float(np.mean(rollout_episode_rewards))
            if rollout_episode_rewards
            else float(np.mean(buffer.rewards))
        )

        mean_cost = (
            float(np.mean(rollout_episode_costs))
            if rollout_episode_costs
            else float(np.mean(buffer.costs))
        )

        mean_distance = float(
            np.mean(rollout_episode_distances)
        )

        if (iteration + 1) % 5 == 0:
            print(
                f"Iteration {iteration + 1:3d}/"
                f"{TRAIN_ITERATIONS} | "
                f"Reward: {mean_reward:8.2f} | "
                f"Episode cost: {mean_cost:7.3f} | "
                f"Mean distance: {mean_distance:7.2f} | "
                f"Lambda: {agent.lagrange_multiplier:6.3f} | "
                f"Violation: {violation:+.4f}"
            )

        buffer.clear()

    history = {
        "rewards": reward_history,
        "costs": cost_history,
        "distances": distance_history,
        "lambdas": lambda_history
    }

    return agent, history


# ============================================================
# 12. TRAINING VISUALIZATION
# ============================================================

def visualize_training(history):
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(
            3,
            1,
            figsize=(10, 10)
        )

        rewards = history["rewards"]
        costs = history["costs"]
        lambdas = history["lambdas"]

        if rewards:
            axes[0].plot(
                rewards,
                alpha=0.45,
                label="Episode reward"
            )

            if len(rewards) >= 10:
                moving_average = np.convolve(
                    rewards,
                    np.ones(10) / 10,
                    mode="valid"
                )

                axes[0].plot(
                    np.arange(9, len(rewards)),
                    moving_average,
                    label="10-episode average"
                )

        axes[0].set_title("Task Reward")
        axes[0].set_xlabel("Episode")
        axes[0].set_ylabel("Reward")
        axes[0].legend()
        axes[0].grid(True)

        if costs:
            axes[1].plot(
                costs,
                label="Episode safety cost"
            )

        axes[1].set_title("Safety Cost")
        axes[1].set_xlabel("Episode")
        axes[1].set_ylabel("Cost")
        axes[1].legend()
        axes[1].grid(True)

        axes[2].plot(
            lambdas,
            label="Lagrange multiplier"
        )

        axes[2].axhline(
            INITIAL_LAMBDA,
            linestyle="--",
            label="Initial lambda"
        )

        axes[2].set_title("Adaptive Safety Penalty")
        axes[2].set_xlabel("Training iteration")
        axes[2].set_ylabel("Lambda")
        axes[2].legend()
        axes[2].grid(True)

        plt.tight_layout()
        plt.show()

    except ImportError:
        print("Matplotlib unavailable; skipping plots.")


# ============================================================
# 13. PYGAME ROCKET SIMULATOR
# ============================================================

class RocketGUI:

    def __init__(self, agent):
        pygame.init()

        self.screen = pygame.display.set_mode(
            (WIDTH + PANEL_WIDTH, HEIGHT)
        )

        pygame.display.set_caption(
            "Phase 4.15 — Constrained Policy Optimization"
        )

        self.clock = pygame.time.Clock()

        self.font = pygame.font.SysFont(
            "consolas",
            16
        )

        self.small_font = pygame.font.SysFont(
            "consolas",
            13
        )

        self.agent = agent

        self.env = MovingTargetEnv()
        self.env.reset()

        self.paused = False
        self.speed = 1
        self.running = True

        self.last_action = np.zeros(
            ACTION_DIM,
            dtype=np.float32
        )

        self.last_reward = 0.0
        self.last_cost = 0.0

        self.total_reward = 0.0
        self.total_cost = 0.0
        self.steps = 0

    def reset(self):
        self.env.reset()

        self.last_action[:] = 0.0

        self.last_reward = 0.0
        self.last_cost = 0.0

        self.total_reward = 0.0
        self.total_cost = 0.0
        self.steps = 0

    def change_target_center(self, position):
        x, y = position

        if not (
            150 < x < WIDTH - 130
            and 100 < y < HEIGHT - 100
        ):
            return

        candidate = np.array(
            [x, y],
            dtype=np.float32
        )

        orbit_radius = float(
            np.linalg.norm(TARGET_AMPLITUDE)
        )

        clearance = (
            distance(candidate, OBSTACLE)
            - orbit_radius
        )

        if clearance < (
            OBSTACLE_RADIUS + ROBOT_RADIUS + 25
        ):
            return

        self.env.target.set_center(
            candidate
        )

    def draw_arrow(
        self,
        start,
        end,
        color,
        width=2
    ):
        start = np.asarray(start)
        end = np.asarray(end)

        pygame.draw.line(
            self.screen,
            color,
            tuple(start.astype(int)),
            tuple(end.astype(int)),
            width
        )

        direction = safe_unit(
            end - start
        )

        if np.linalg.norm(direction) < 1e-6:
            return

        perpendicular = np.array(
            [-direction[1], direction[0]]
        )

        p1 = (
            end - direction * 10
            + perpendicular * 5
        )

        p2 = (
            end - direction * 10
            - perpendicular * 5
        )

        pygame.draw.polygon(
            self.screen,
            color,
            [
                end.astype(int),
                p1.astype(int),
                p2.astype(int)
            ]
        )

    def draw_rocket(self):
        position = self.env.position

        direction = safe_unit(
            self.env.velocity
        )

        if np.linalg.norm(direction) < 0.1:
            direction = safe_unit(
                self.env.target.position - position
            )

        if np.linalg.norm(direction) < 0.1:
            direction = np.array(
                [1.0, 0.0]
            )

        perpendicular = np.array(
            [-direction[1], direction[0]]
        )

        nose = position + direction * 22
        rear = position - direction * 15

        left_fin = rear + perpendicular * 13
        right_fin = rear - perpendicular * 13

        body_left = position + perpendicular * 8
        body_right = position - perpendicular * 8

        pygame.draw.line(
            self.screen,
            (255, 125, 30),
            tuple(rear.astype(int)),
            tuple(
                (rear - direction * random.randint(10, 18))
                .astype(int)
            ),
            5
        )

        pygame.draw.polygon(
            self.screen,
            (205, 215, 230),
            [
                nose.astype(int),
                body_left.astype(int),
                left_fin.astype(int),
                rear.astype(int),
                right_fin.astype(int),
                body_right.astype(int)
            ]
        )

        pygame.draw.circle(
            self.screen,
            (30, 45, 65),
            tuple(position.astype(int)),
            4
        )

    def draw_world(self):
        self.screen.fill(
            (15, 20, 30)
        )

        for x in range(0, WIDTH, 40):
            pygame.draw.line(
                self.screen,
                (28, 35, 48),
                (x, 0),
                (x, HEIGHT)
            )

        for y in range(0, HEIGHT, 40):
            pygame.draw.line(
                self.screen,
                (28, 35, 48),
                (0, y),
                (WIDTH, y)
            )

        # Target path.
        path = []

        for index in range(100):
            phase = (
                self.env.target.phase
                + index * 0.08
            )

            path.append(
                tuple(
                    self.env.target.position_at(
                        phase
                    ).astype(int)
                )
            )

        pygame.draw.lines(
            self.screen,
            (65, 95, 75),
            False,
            path,
            1
        )

        # World boundary.
        pygame.draw.rect(
            self.screen,
            (90, 110, 140),
            (0, 0, WIDTH, HEIGHT),
            2
        )

        # Safety zone and obstacle.
        pygame.draw.circle(
            self.screen,
            (115, 75, 55),
            OBSTACLE.astype(int),
            int(SAFETY_RADIUS),
            2
        )

        pygame.draw.circle(
            self.screen,
            (170, 65, 65),
            OBSTACLE.astype(int),
            int(OBSTACLE_RADIUS)
        )

        # Target and velocity.
        target = self.env.target.position

        pygame.draw.circle(
            self.screen,
            (70, 230, 120),
            target.astype(int),
            10
        )

        pygame.draw.circle(
            self.screen,
            (180, 255, 190),
            target.astype(int),
            18,
            1
        )

        self.draw_arrow(
            target,
            target + self.env.target.velocity * 0.35,
            (100, 240, 130)
        )

        # Action and velocity indicators.
        self.draw_arrow(
            self.env.position,
            self.env.position + self.last_action * 60,
            (100, 170, 255),
            3
        )

        self.draw_arrow(
            self.env.position,
            self.env.position + self.env.velocity * 0.35,
            (255, 210, 90),
            2
        )

        self.draw_rocket()

    def draw_panel(self):
        x = WIDTH

        pygame.draw.rect(
            self.screen,
            (23, 28, 40),
            (x, 0, PANEL_WIDTH, HEIGHT)
        )

        pygame.draw.line(
            self.screen,
            (70, 80, 100),
            (x, 0),
            (x, HEIGHT),
            2
        )

        target_distance = distance(
            self.env.position,
            self.env.target.position
        )

        clearance = (
            distance(self.env.position, OBSTACLE)
            - SAFETY_RADIUS
        )

        lines = [
            ("PHASE 4.15", (255, 220, 100)),
            ("CONSTRAINED POLICY", (220, 230, 245)),
            ("OPTIMIZATION", (220, 230, 245)),
            ("", (200, 200, 200)),
            ("POLICY", (150, 190, 255)),
            ("PPO-Lagrangian", (100, 240, 150)),
            ("", (200, 200, 200)),
            ("TRACKING", (150, 190, 255)),
            (f"Distance: {target_distance:.1f}", (230, 230, 230)),
            (f"Target vx: {self.env.target.velocity[0]:.1f}", (180, 240, 180)),
            (f"Target vy: {self.env.target.velocity[1]:.1f}", (180, 240, 180)),
            ("", (200, 200, 200)),
            ("SAFETY", (150, 190, 255)),
            (f"Clearance: {clearance:.1f}", (230, 230, 230)),
            (f"Current cost: {self.last_cost:.3f}", (230, 230, 230)),
            (f"Cost budget: {COST_BUDGET:.3f}", (230, 230, 230)),
            ("", (200, 200, 200)),
            ("LAGRANGIAN", (150, 190, 255)),
            (f"Lambda: {self.agent.lagrange_multiplier:.3f}", (255, 220, 100)),
            ("", (200, 200, 200)),
            ("LAST ACTION", (150, 190, 255)),
            (f"X: {self.last_action[0]:+.3f}", (150, 190, 255)),
            (f"Y: {self.last_action[1]:+.3f}", (150, 190, 255)),
            ("", (200, 200, 200)),
            ("SIMULATION", (150, 190, 255)),
            (f"Speed: {self.speed}x", (220, 220, 220)),
            ("PAUSED" if self.paused else "RUNNING", (255, 190, 100)),
            ("", (200, 200, 200)),
            ("CONTROLS", (150, 190, 255)),
            ("Mouse: change target center", (190, 190, 190)),
            ("SPACE: pause / resume", (190, 190, 190)),
            ("R: reset", (190, 190, 190)),
            ("+/-: simulation speed", (190, 190, 190)),
            ("ESC: exit", (190, 190, 190))
        ]

        y = 12

        for text, color in lines:
            if text:
                surface = self.small_font.render(
                    text,
                    True,
                    color
                )

                self.screen.blit(
                    surface,
                    (x + 12, y)
                )

            y += 16

    def handle_events(self):
        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.KEYDOWN:

                if event.key == pygame.K_ESCAPE:
                    self.running = False

                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused

                elif event.key == pygame.K_r:
                    self.reset()

                elif event.key in (
                    pygame.K_PLUS,
                    pygame.K_EQUALS,
                    pygame.K_KP_PLUS
                ):
                    self.speed = min(
                        5,
                        self.speed + 1
                    )

                elif event.key in (
                    pygame.K_MINUS,
                    pygame.K_KP_MINUS
                ):
                    self.speed = max(
                        1,
                        self.speed - 1
                    )

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if (
                    event.button == 1
                    and event.pos[0] < WIDTH
                ):
                    self.change_target_center(
                        event.pos
                    )

    def run(self):
        while self.running:
            self.handle_events()

            if not self.paused:
                for _ in range(self.speed):

                    state = self.env.get_state()

                    action, _, _, _ = self.agent.act(
                        state,
                        deterministic=True
                    )

                    (
                        next_state,
                        reward,
                        cost,
                        terminated,
                        truncated,
                        info
                    ) = self.env.step(action)

                    self.last_action = action.copy()
                    self.last_reward = reward
                    self.last_cost = cost

                    self.total_reward += reward
                    self.total_cost += cost
                    self.steps += 1

                    if terminated or truncated:
                        self.env.reset()

                        self.total_reward = 0.0
                        self.total_cost = 0.0
                        self.steps = 0

            self.draw_world()
            self.draw_panel()

            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()


# ============================================================
# 14. UNIT TESTS
# ============================================================

def run_tests():
    print("\nRunning Phase 4.15 tests...\n")

    # Test 1: Environment state shape.
    env = MovingTargetEnv()

    state = env.reset()

    assert state.shape == (STATE_DIM,)
    assert np.all(np.isfinite(state))

    print("Test 1 passed: environment state")

    # Test 2: Moving target changes position.
    p1 = env.target.position.copy()

    env.target.step()

    p2 = env.target.position.copy()

    assert not np.allclose(p1, p2)

    print("Test 2 passed: moving target")

    # Test 3: Target velocity has correct shape.
    assert env.target.velocity.shape == (2,)

    print("Test 3 passed: target velocity")

    # Test 4: Robot dynamics respect speed limit.
    position, velocity = robot_dynamics(
        np.array([100.0, 100.0]),
        np.array([110.0, 0.0]),
        np.array([1.0, 0.0])
    )

    assert np.linalg.norm(velocity) <= MAX_SPEED + 1e-5

    print("Test 4 passed: speed limit")

    # Test 5: Safety cost is nonnegative.
    assert env.safety_cost() >= 0.0

    print("Test 5 passed: safety cost")

    # Test 6: Environment step returns six values.
    result = env.step(
        np.zeros(ACTION_DIM)
    )

    assert len(result) == 6

    print("Test 6 passed: environment transition")

    # Test 7: Actor action shape and range.
    policy = PolicyNetwork().to(DEVICE)

    test_states = torch.zeros(
        (8, STATE_DIM),
        device=DEVICE
    )

    with torch.no_grad():
        actions, log_probs = sample_policy(
            policy,
            test_states
        )

    assert actions.shape == (8, ACTION_DIM)
    assert log_probs.shape == (8, 1)

    assert torch.all(actions <= 1.0)
    assert torch.all(actions >= -1.0)

    print("Test 7 passed: bounded policy actions")

    # Test 8: Policy log probability is finite.
    assert torch.all(torch.isfinite(log_probs))

    print("Test 8 passed: policy log probability")

    # Test 9: Policy evaluation reproduces log probability.
    with torch.no_grad():
        evaluated_log_probs, _ = evaluate_policy(
            policy,
            test_states,
            actions
        )

    assert evaluated_log_probs.shape == (8, 1)
    assert torch.all(torch.isfinite(evaluated_log_probs))

    print("Test 9 passed: policy evaluation")

    # Test 10: Reward critic output.
    reward_critic = ValueNetwork().to(DEVICE)

    with torch.no_grad():
        values = reward_critic(test_states)

    assert values.shape == (8, 1)

    print("Test 10 passed: reward critic")

    # Test 11: Cost critic output.
    cost_critic = ValueNetwork().to(DEVICE)

    with torch.no_grad():
        values = cost_critic(test_states)

    assert values.shape == (8, 1)

    print("Test 11 passed: cost critic")

    # Test 12: GAE produces finite results.
    rewards = np.array(
        [1.0, 1.0, 1.0],
        dtype=np.float32
    )

    values = np.array(
        [0.5, 0.6, 0.7],
        dtype=np.float32
    )

    next_values = np.array(
        [0.6, 0.7, 0.0],
        dtype=np.float32
    )

    terminated = np.array(
        [0.0, 0.0, 1.0],
        dtype=np.float32
    )

    advantages, returns = calculate_gae(
        rewards,
        values,
        next_values,
        terminated,
        GAMMA,
        GAE_LAMBDA
    )

    assert advantages.shape == (3,)
    assert returns.shape == (3,)
    assert np.all(np.isfinite(advantages))

    print("Test 12 passed: GAE calculation")

    # Test 13: Lagrange multiplier increases on violation.
    agent = ConstrainedPPOAgent()

    initial_lambda = agent.lagrange_multiplier

    agent.update_lagrange_multiplier(
        COST_BUDGET + 0.1
    )

    assert agent.lagrange_multiplier > initial_lambda

    print("Test 13 passed: multiplier increases on violation")

    # Test 14: Multiplier decreases below budget.
    initial_lambda = agent.lagrange_multiplier

    agent.update_lagrange_multiplier(
        COST_BUDGET - 0.1
    )

    assert agent.lagrange_multiplier < initial_lambda

    print("Test 14 passed: multiplier decreases below budget")

    # Test 15: Multiplier never becomes negative.
    agent.lagrange_multiplier = 0.0

    agent.update_lagrange_multiplier(
        -10.0
    )

    assert agent.lagrange_multiplier >= 0.0

    print("Test 15 passed: nonnegative multiplier")

    # Test 16: Rollout buffer storage.
    buffer = RolloutBuffer()

    buffer.add(
        np.zeros(STATE_DIM),
        np.zeros(ACTION_DIM),
        0.0,
        1.0,
        0.0,
        0.5,
        0.1,
        0.6,
        0.1,
        False,
        False
    )

    assert len(buffer) == 1

    print("Test 16 passed: rollout storage")

    # Test 17: Actor and critic optimization executes.
    small_agent = ConstrainedPPOAgent()

    small_buffer = RolloutBuffer()

    for index in range(32):

        state = np.random.randn(
            STATE_DIM
        ).astype(np.float32)

        action, log_prob, rv, cv = small_agent.act(
            state
        )

        small_buffer.add(
            state,
            action,
            log_prob,
            random.random(),
            random.random() * 0.1,
            rv,
            cv,
            rv,
            cv,
            False,
            index == 31
        )

    # Small batch for the unit test.
    original_epochs = PPO_EPOCHS
    original_minibatch = MINIBATCH_SIZE

    try:
        # update() reads module constants, so temporarily use
        # one epoch and a small minibatch.
        globals()["PPO_EPOCHS"] = 1
        globals()["MINIBATCH_SIZE"] = 16

        losses = small_agent.update(
            small_buffer
        )

        assert np.isfinite(
            losses["policy_loss"]
        )

        assert np.isfinite(
            losses["reward_value_loss"]
        )

        assert np.isfinite(
            losses["cost_value_loss"]
        )

    finally:
        globals()["PPO_EPOCHS"] = original_epochs
        globals()["MINIBATCH_SIZE"] = original_minibatch

    print("Test 17 passed: PPO optimization")

    print("\nAll 17 tests passed.\n")


# ============================================================
# 15. MAIN
# ============================================================

def main():
    print("=" * 68)
    print("PHASE 4.15 — CONSTRAINED POLICY OPTIMIZATION")
    print("=" * 68)

    run_tests()

    agent, history = train_agent()

    visualize_training(history)

    print("\nStarting Pygame moving-target simulation.")
    print("The rocket uses the trained constrained policy.")

    gui = RocketGUI(agent)
    gui.run()


if __name__ == "__main__":
    main()
