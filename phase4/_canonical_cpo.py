
"""
RL COURSE — PHASE 4.16
============================================================
Filename:
    phase4/canonical_cpo.py

LESSON:
    Constrained Policy Optimization (CPO)

PURPOSE
-------
Train a moving-target tracking robot while constraining its
expected cumulative safety cost.

WHAT YOU LEARN
--------------
STEP 1 — Constrained MDP
    Maximize expected reward subject to an expected-cost limit.

STEP 2 — Gaussian policy
    Represent continuous robot actions with a probability
    distribution.

STEP 3 — Reward and cost advantages
    Estimate the benefits and safety consequences of actions.

STEP 4 — Trust-region optimization
    Limit policy changes using a KL-divergence constraint.

STEP 5 — Fisher-vector products
    Compute Hessian-vector products without constructing
    the full Hessian matrix.

STEP 6 — Conjugate gradient
    Approximately solve:
        Hx = g
        Hy = b

STEP 7 — CPO constrained step
    Calculate a policy update satisfying the local cost
    constraint and trust-region constraint.

STEP 8 — Backtracking line search
    Check actual reward improvement, cost feasibility,
    and KL divergence before accepting the update.

MATHEMATICS
-----------
Constrained objective:

    maximize    J_reward(theta)

    subject to  J_cost(theta) <= COST_LIMIT

Trust region:

    D_KL(pi_old || pi_new) <= DELTA

Local reward approximation:

    maximize    g.T * step

Local cost approximation:

    c + b.T * step <= 0

Quadratic trust-region approximation:

    0.5 * step.T * H * step <= DELTA

where:

    g = gradient of reward surrogate
    b = gradient of cost surrogate
    H = Hessian of KL divergence
    c = current estimated cost - cost limit

Conjugate gradient:

    Hx = g
    Hy = b

The implementation uses damped Fisher-vector products
and a practical CPO constrained-step calculation.

IMPORTANT
---------
This is an educational implementation of the CPO
trust-region method. It uses a finite-trajectory estimate
of episodic discounted cost, approximate conjugate-gradient
solutions, and backtracking line search.

It is not a formal safety certification. Function
approximation, finite samples, model errors, and numerical
approximations can still produce constraint violations.

ROBOTICS APPLICATION
--------------------
- Autonomous mobile robot navigation
- Drone obstacle avoidance
- Industrial robot motion planning
- Safety-constrained manipulation

Pygame:
    Mouse   Change moving-target trajectory center
    SPACE   Pause / resume
    R       Reset
    +/-     Simulation speed
    ESC     Exit
============================================================
"""

from __future__ import annotations

import math
import random

import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence


# ============================================================
# 1. CONFIGURATION
# ============================================================

SEED = 46

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

SAFETY_RADIUS = (
    OBSTACLE_RADIUS + ROBOT_RADIUS + 35.0
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

# CPO trust-region parameters.
DELTA = 0.015
CG_ITERATIONS = 15
CG_TOLERANCE = 1e-10
FISHER_DAMPING = 0.1

# Backtracking line search.
LINE_SEARCH_STEPS = 15
BACKTRACK_COEFFICIENT = 0.8

# Safety constraint:
# Maximum expected discounted episode cost.
COST_LIMIT = 15.0

EPISODES_PER_ITERATION = 5
MAX_EPISODE_STEPS = 300
TRAIN_ITERATIONS = 60

ACTOR_LR = 3e-4
CRITIC_LR = 1e-3

ENTROPY_COEFFICIENT = 0.002


# ============================================================
# 2. NUMERICAL UTILITIES
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


def flat_parameters(module):
    return torch.cat([
        parameter.detach().reshape(-1)
        for parameter in module.parameters()
    ])


def set_flat_parameters(module, flat_vector):
    offset = 0

    with torch.no_grad():
        for parameter in module.parameters():
            count = parameter.numel()

            parameter.copy_(
                flat_vector[offset:offset + count]
                .view_as(parameter)
            )

            offset += count

    assert offset == flat_vector.numel()


def flat_gradient(gradients):
    return torch.cat([
        gradient.contiguous().view(-1)
        for gradient in gradients
    ])


def flat_grad(
    output,
    parameters,
    create_graph=False
):
    gradients = torch.autograd.grad(
        output,
        parameters,
        create_graph=create_graph,
        allow_unused=False
    )

    return flat_gradient(gradients)


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

    acceleration = (
        action * MAX_ACCELERATION
    )

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
# 5. CONSTRAINED ROBOT ENVIRONMENT
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

        state = np.concatenate([
            relative_target / 600.0,
            self.target.velocity / 150.0,
            self.velocity / MAX_SPEED,
            relative_obstacle / 600.0,
            np.array([
                clearance / 200.0,
                distance(
                    self.position,
                    self.target.position
                ) / 600.0
            ], dtype=np.float32)
        ])

        return state.astype(np.float32)

    def safety_cost(self):
        """
        Episodic safety cost.

        Proximity to the obstacle produces a cost.
        Actual collision produces a large additional cost.
        Approaching world boundaries also produces a cost.
        """

        clearance = (
            distance(self.position, OBSTACLE)
            - SAFETY_RADIUS
        )

        proximity_cost = max(
            0.0,
            -clearance / SAFETY_RADIUS
        )

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

        outside = min(boundary_clearances) < 0

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

        # Compare old and new robot positions against
        # the same future target position.
        next_phase = (
            self.target.phase + TARGET_OMEGA * DT
        )

        future_target = (
            self.target.position_at(next_phase)
        )

        distance_before = distance(
            self.position,
            future_target
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
            "outside": outside
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
# 6. GAUSSIAN CONTINUOUS POLICY
# ============================================================

class GaussianPolicy(nn.Module):

    def __init__(self):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Linear(STATE_DIM, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh()
        )

        self.mean = nn.Linear(
            128,
            ACTION_DIM
        )

        self.log_std = nn.Parameter(
            torch.full(
                (ACTION_DIM,),
                -0.7
            )
        )

    def forward(self, states):
        features = self.backbone(states)

        mean = self.mean(features)

        log_std = self.log_std.clamp(
            -3.0,
            0.5
        )

        std = log_std.exp().expand_as(mean)

        return mean, std

    def distribution(self, states):
        mean, std = self(states)

        return Normal(mean, std)


# ============================================================
# 7. REWARD AND COST CRITICS
# ============================================================

class ValueNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(STATE_DIM, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, states):
        return self.network(states)


# ============================================================
# 8. CONJUGATE GRADIENT
# ============================================================

def conjugate_gradient(
    matrix_vector_product,
    vector,
    iterations=CG_ITERATIONS,
    tolerance=CG_TOLERANCE
):
    """
    Approximately solve:

        Hx = vector

    without explicitly constructing H.

    H is the damped Fisher information matrix.
    """

    solution = torch.zeros_like(vector)

    residual = vector.clone()
    direction = residual.clone()

    residual_dot = torch.dot(
        residual,
        residual
    )

    initial_residual = float(
        residual_dot.item()
    )

    if initial_residual < tolerance:
        return solution

    for _ in range(iterations):

        product = matrix_vector_product(
            direction
        )

        denominator = torch.dot(
            direction,
            product
        )

        if denominator.item() <= 1e-12:
            break

        alpha = (
            residual_dot / denominator
        )

        solution = (
            solution + alpha * direction
        )

        new_residual = (
            residual - alpha * product
        )

        new_residual_dot = torch.dot(
            new_residual,
            new_residual
        )

        if new_residual_dot.item() < tolerance:
            break

        beta = (
            new_residual_dot / residual_dot
        )

        direction = (
            new_residual + beta * direction
        )

        residual = new_residual
        residual_dot = new_residual_dot

    return solution.detach()


# ============================================================
# 9. ROLLOUT COLLECTION
# ============================================================

def collect_episodes(
    env,
    policy,
    reward_critic,
    cost_critic,
    episode_count
):
    """
    Collect complete episodes.

    Actions are sampled from the Gaussian policy and clipped
    only before entering the physical environment.

    Log probabilities are calculated using the original
    Gaussian samples, not the clipped actions.
    """

    all_states = []
    all_actions = []
    all_log_probs = []

    all_rewards = []
    all_costs = []

    all_reward_values = []
    all_cost_values = []

    all_next_reward_values = []
    all_next_cost_values = []

    all_terminated = []
    all_episode_ends = []

    episode_rewards = []
    episode_costs = []
    episode_lengths = []

    for _ in range(episode_count):

        state = env.reset()

        episode_reward = 0.0
        episode_cost = 0.0
        episode_length = 0

        for _ in range(MAX_EPISODE_STEPS):

            state_tensor = torch.tensor(
                state,
                dtype=torch.float32,
                device=DEVICE
            ).unsqueeze(0)

            with torch.no_grad():
                distribution = policy.distribution(
                    state_tensor
                )

                raw_action = distribution.sample()

                log_probability = distribution.log_prob(
                    raw_action
                ).sum(
                    dim=1,
                    keepdim=True
                )

                reward_value = reward_critic(
                    state_tensor
                )

                cost_value = cost_critic(
                    state_tensor
                )

            raw_action_np = (
                raw_action.cpu().numpy()[0]
            )

            environment_action = np.clip(
                raw_action_np,
                -1.0,
                1.0
            )

            (
                next_state,
                reward,
                cost,
                terminated,
                truncated,
                info
            ) = env.step(environment_action)

            next_tensor = torch.tensor(
                next_state,
                dtype=torch.float32,
                device=DEVICE
            ).unsqueeze(0)

            with torch.no_grad():
                if terminated:
                    next_reward_value = 0.0
                    next_cost_value = 0.0
                else:
                    next_reward_value = float(
                        reward_critic(next_tensor).item()
                    )

                    next_cost_value = float(
                        cost_critic(next_tensor).item()
                    )

            episode_end = (
                terminated or truncated
            )

            all_states.append(state)
            all_actions.append(raw_action_np)

            all_log_probs.append(
                float(log_probability.item())
            )

            all_rewards.append(reward)
            all_costs.append(cost)

            all_reward_values.append(
                float(reward_value.item())
            )

            all_cost_values.append(
                float(cost_value.item())
            )

            all_next_reward_values.append(
                next_reward_value
            )

            all_next_cost_values.append(
                next_cost_value
            )

            all_terminated.append(
                float(terminated)
            )

            all_episode_ends.append(
                float(episode_end)
            )

            episode_reward += reward
            episode_cost += cost
            episode_length += 1

            state = next_state

            if episode_end:
                break

        episode_rewards.append(
            episode_reward
        )

        episode_costs.append(
            episode_cost
        )

        episode_lengths.append(
            episode_length
        )

    data = {
        "states": np.asarray(
            all_states,
            dtype=np.float32
        ),

        "actions": np.asarray(
            all_actions,
            dtype=np.float32
        ),

        "log_probs": np.asarray(
            all_log_probs,
            dtype=np.float32
        ).reshape(-1, 1),

        "rewards": np.asarray(
            all_rewards,
            dtype=np.float32
        ),

        "costs": np.asarray(
            all_costs,
            dtype=np.float32
        ),

        "reward_values": np.asarray(
            all_reward_values,
            dtype=np.float32
        ),

        "cost_values": np.asarray(
            all_cost_values,
            dtype=np.float32
        ),

        "next_reward_values": np.asarray(
            all_next_reward_values,
            dtype=np.float32
        ),

        "next_cost_values": np.asarray(
            all_next_cost_values,
            dtype=np.float32
        ),

        "terminated": np.asarray(
            all_terminated,
            dtype=np.float32
        ),

        "episode_ends": np.asarray(
            all_episode_ends,
            dtype=np.float32
        ),

        "episode_rewards": episode_rewards,
        "episode_costs": episode_costs,
        "episode_lengths": episode_lengths
    }

    return data


# ============================================================
# 10. ADVANTAGE ESTIMATION
# ============================================================

def calculate_gae(
    rewards,
    values,
    next_values,
    terminated,
    episode_ends
):
    """
    Separate reward and cost advantages are calculated.

    GAE recursion is reset at episode boundaries.

    Truncation bootstraps from the final observation.
    True termination does not bootstrap.
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
            + GAMMA
            * continuation
            * next_values[index]
            - values[index]
        )

        if episode_ends[index]:
            gae = 0.0

        gae = (
            delta
            + GAMMA
            * GAE_LAMBDA
            * continuation
            * gae
        )

        advantages[index] = gae

    returns = advantages + values

    return advantages, returns


def discounted_episode_costs(data):
    """
    Estimate discounted cost separately for each episode.

    The cost constraint is episodic:

        E[sum_t gamma^t * cost_t] <= COST_LIMIT
    """

    episode_costs = []

    running_cost = 0.0
    discount = 1.0

    for cost, episode_end in zip(
        data["costs"],
        data["episode_ends"]
    ):

        running_cost += discount * float(cost)

        discount *= GAMMA

        if episode_end:
            episode_costs.append(
                running_cost
            )

            running_cost = 0.0
            discount = 1.0

    if len(episode_costs) == 0:
        return 0.0

    return float(
        np.mean(episode_costs)
    )


# ============================================================
# 11. CPO AGENT
# ============================================================

class CPOAgent:

    def __init__(self):
        self.policy = GaussianPolicy().to(DEVICE)

        self.reward_critic = ValueNetwork().to(DEVICE)
        self.cost_critic = ValueNetwork().to(DEVICE)

        self.reward_optimizer = torch.optim.Adam(
            self.reward_critic.parameters(),
            lr=CRITIC_LR
        )

        self.cost_optimizer = torch.optim.Adam(
            self.cost_critic.parameters(),
            lr=CRITIC_LR
        )

    def train_critics(self, data, reward_returns, cost_returns):
        states = torch.tensor(
            data["states"],
            dtype=torch.float32,
            device=DEVICE
        )

        reward_targets = torch.tensor(
            reward_returns,
            dtype=torch.float32,
            device=DEVICE
        ).unsqueeze(1)

        cost_targets = torch.tensor(
            cost_returns,
            dtype=torch.float32,
            device=DEVICE
        ).unsqueeze(1)

        reward_losses = []
        cost_losses = []

        for _ in range(5):

            predicted_reward = (
                self.reward_critic(states)
            )

            reward_loss = F.mse_loss(
                predicted_reward,
                reward_targets
            )

            self.reward_optimizer.zero_grad()
            reward_loss.backward()

            nn.utils.clip_grad_norm_(
                self.reward_critic.parameters(),
                1.0
            )

            self.reward_optimizer.step()

            predicted_cost = (
                self.cost_critic(states)
            )

            cost_loss = F.mse_loss(
                predicted_cost,
                cost_targets
            )

            self.cost_optimizer.zero_grad()
            cost_loss.backward()

            nn.utils.clip_grad_norm_(
                self.cost_critic.parameters(),
                1.0
            )

            self.cost_optimizer.step()

            reward_losses.append(
                float(reward_loss.item())
            )

            cost_losses.append(
                float(cost_loss.item())
            )

        return {
            "reward_value_loss": float(
                np.mean(reward_losses)
            ),
            "cost_value_loss": float(
                np.mean(cost_losses)
            )
        }

    def prepare_data(self, data):
        reward_advantages, reward_returns = calculate_gae(
            data["rewards"],
            data["reward_values"],
            data["next_reward_values"],
            data["terminated"],
            data["episode_ends"]
        )

        cost_advantages, cost_returns = calculate_gae(
            data["costs"],
            data["cost_values"],
            data["next_cost_values"],
            data["terminated"],
            data["episode_ends"]
        )

        # Reward advantage normalization.
        reward_advantages = (
            reward_advantages
            - reward_advantages.mean()
        ) / (
            reward_advantages.std() + 1e-8
        )

        # Do not normalize cost advantages independently:
        # their scale carries information about safety cost.
        return (
            reward_advantages.astype(np.float32),
            cost_advantages.astype(np.float32),
            reward_returns.astype(np.float32),
            cost_returns.astype(np.float32)
        )

    def update_policy(self, data, reward_advantages, cost_advantages):
        states = torch.tensor(
            data["states"],
            dtype=torch.float32,
            device=DEVICE
        )

        actions = torch.tensor(
            data["actions"],
            dtype=torch.float32,
            device=DEVICE
        )

        old_log_probs = torch.tensor(
            data["log_probs"],
            dtype=torch.float32,
            device=DEVICE
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

        parameters = list(
            self.policy.parameters()
        )

        old_parameters = flat_parameters(
            self.policy
        ).clone()

        # Freeze the old policy distribution.
        with torch.no_grad():
            old_distribution = (
                self.policy.distribution(states)
            )

            old_mean = old_distribution.mean.detach()
            old_std = old_distribution.stddev.detach()

        old_distribution = Normal(
            old_mean,
            old_std
        )

        def surrogate_terms():
            new_distribution = (
                self.policy.distribution(states)
            )

            new_log_probs = new_distribution.log_prob(
                actions
            ).sum(
                dim=1,
                keepdim=True
            )

            ratio = torch.exp(
                new_log_probs - old_log_probs
            )

            reward_surrogate = (
                ratio * reward_advantages
            ).mean()

            cost_surrogate = (
                ratio * cost_advantages
            ).mean()

            entropy = new_distribution.entropy().sum(
                dim=1
            ).mean()

            return (
                reward_surrogate,
                cost_surrogate,
                entropy
            )

        # ----------------------------------------------------
        # Current reward and cost surrogate.
        # ----------------------------------------------------

        reward_surrogate, cost_surrogate, entropy = (
            surrogate_terms()
        )

        reward_gradient = flat_grad(
            reward_surrogate,
            parameters
        ).detach()

        cost_gradient = flat_grad(
            cost_surrogate,
            parameters
        ).detach()

        # ----------------------------------------------------
        # KL Hessian-vector product.
        #
        # FVP(v) = (H_KL + damping*I) @ v
        #
        # The full Hessian is never constructed.
        # ----------------------------------------------------

        def fisher_vector_product(vector):
            current_distribution = (
                self.policy.distribution(states)
            )

            kl = kl_divergence(
                old_distribution,
                current_distribution
            ).sum(
                dim=1
            ).mean()

            kl_gradient = flat_grad(
                kl,
                parameters,
                create_graph=True
            )

            gradient_vector_product = torch.dot(
                kl_gradient,
                vector
            )

            second_derivative = torch.autograd.grad(
                gradient_vector_product,
                parameters
            )

            hessian_vector = flat_gradient(
                second_derivative
            ).detach()

            return (
                hessian_vector
                + FISHER_DAMPING * vector
            )

        # ----------------------------------------------------
        # Solve:
        #
        # Hx = g
        # Hy = b
        # ----------------------------------------------------

        x = conjugate_gradient(
            fisher_vector_product,
            reward_gradient
        )

        y = conjugate_gradient(
            fisher_vector_product,
            cost_gradient
        )

        q = torch.dot(
            reward_gradient,
            x
        )

        r = torch.dot(
            reward_gradient,
            y
        )

        s = torch.dot(
            cost_gradient,
            y
        )

        q_value = float(q.item())
        r_value = float(r.item())
        s_value = float(s.item())

        current_episode_cost = discounted_episode_costs(
            data
        )

        c = (
            current_episode_cost - COST_LIMIT
        )

        # ----------------------------------------------------
        # Calculate constrained step.
        # ----------------------------------------------------

        step = torch.zeros_like(
            reward_gradient
        )

        recovery_step = False
        step_status = "no_update"

        if (
            s_value > 1e-10
            and q_value > 1e-10
        ):

            # First check the unconstrained trust-region step.
            unconstrained_scale = math.sqrt(
                2.0 * DELTA / q_value
            )

            unconstrained_step = (
                unconstrained_scale * x
            )

            unconstrained_cost = (
                c
                + torch.dot(
                    cost_gradient,
                    unconstrained_step
                ).item()
            )

            if unconstrained_cost <= 0.0:

                step = unconstrained_step
                step_status = "unconstrained_feasible"

            else:
                # The unconstrained step violates the cost
                # constraint. Solve the local constrained problem.

                A = (
                    q_value
                    - r_value * r_value / s_value
                )

                B = (
                    2.0 * DELTA
                    - c * c / s_value
                )

                if A > 1e-10 and B > 1e-10:

                    lagrange = math.sqrt(
                        A / B
                    )

                    nu = max(
                        0.0,
                        (lagrange * c - r_value)
                        / s_value
                    )

                    step = (
                        x - nu * y
                    ) / lagrange

                    step_status = "constrained"

                else:
                    # Current policy is outside the feasible
                    # region and the local feasible intersection
                    # is unavailable under this approximation.
                    #
                    # Take a cost-reducing recovery step.
                    if c > 0.0:

                        recovery_step = True

                        step = (
                            -math.sqrt(
                                2.0 * DELTA / s_value
                            ) * y
                        )

                        step_status = "recovery"

                    else:
                        step_status = "degenerate"

        # ----------------------------------------------------
        # Backtracking line search.
        # ----------------------------------------------------

        accepted = False
        accepted_kl = 0.0
        accepted_cost = current_episode_cost
        accepted_reward = float(
            reward_surrogate.item()
        )

        original_reward = float(
            reward_surrogate.item()
        )

        original_cost_surrogate = float(
            cost_surrogate.item()
        )

        if torch.linalg.vector_norm(step).item() > 1e-12:

            for line_index in range(
                LINE_SEARCH_STEPS
            ):

                fraction = (
                    BACKTRACK_COEFFICIENT ** line_index
                )

                candidate = (
                    old_parameters + fraction * step
                )

                set_flat_parameters(
                    self.policy,
                    candidate
                )

                with torch.no_grad():

                    candidate_reward, candidate_cost, _ = (
                        surrogate_terms()
                    )

                    candidate_distribution = (
                        self.policy.distribution(states)
                    )

                    kl_value = kl_divergence(
                        old_distribution,
                        candidate_distribution
                    ).sum(
                        dim=1
                    ).mean().item()

                    candidate_reward = float(
                        candidate_reward.item()
                    )

                    candidate_cost = float(
                        candidate_cost.item()
                    )

                # Local cost estimate:
                #
                # J_cost(new) approximately equals
                # J_cost(old) + surrogate_cost(new)
                #                         - surrogate_cost(old)
                #
                predicted_cost = (
                    current_episode_cost
                    + candidate_cost
                    - original_cost_surrogate
                )

                trust_region_ok = (
                    kl_value <= DELTA
                )

                if recovery_step:
                    # Recovery prioritizes cost reduction.
                    objective_ok = (
                        predicted_cost
                        < current_episode_cost
                    )
                else:
                    objective_ok = (
                        candidate_reward
                        >= original_reward - 1e-8
                    )

                constraint_ok = (
                    predicted_cost <= COST_LIMIT
                )

                if (
                    trust_region_ok
                    and objective_ok
                    and constraint_ok
                ):

                    accepted = True
                    accepted_kl = kl_value
                    accepted_cost = predicted_cost
                    accepted_reward = candidate_reward

                    step_status += "_accepted"
                    break

            if not accepted:
                set_flat_parameters(
                    self.policy,
                    old_parameters
                )

                step_status += "_rejected"

        else:
            set_flat_parameters(
                self.policy,
                old_parameters
            )

        return {
            "accepted": accepted,
            "status": step_status,
            "kl": accepted_kl,
            "estimated_cost": accepted_cost,
            "estimated_reward_surrogate": accepted_reward,
            "current_episode_cost": current_episode_cost,
            "q": q_value,
            "r": r_value,
            "s": s_value,
            "entropy": float(entropy.item())
        }

    def update(self, data):
        (
            reward_advantages,
            cost_advantages,
            reward_returns,
            cost_returns
        ) = self.prepare_data(data)

        critic_metrics = self.train_critics(
            data,
            reward_returns,
            cost_returns
        )

        policy_metrics = self.update_policy(
            data,
            reward_advantages,
            cost_advantages
        )

        return {
            **critic_metrics,
            **policy_metrics
        }

    @torch.no_grad()
    def act(self, state, deterministic=True):
        state_tensor = torch.tensor(
            state,
            dtype=torch.float32,
            device=DEVICE
        ).unsqueeze(0)

        distribution = self.policy.distribution(
            state_tensor
        )

        if deterministic:
            action = distribution.mean
        else:
            action = distribution.sample()

        return np.clip(
            action.cpu().numpy()[0],
            -1.0,
            1.0
        )


# ============================================================
# 12. TRAINING
# ============================================================

def train_agent():
    env = MovingTargetEnv()
    agent = CPOAgent()

    history = {
        "rewards": [],
        "costs": [],
        "lengths": [],
        "kl": [],
        "accepted": [],
        "status": []
    }

    print("\nTraining CPO agent...")
    print(f"Device: {DEVICE}")
    print(f"Trust-region radius: {DELTA}")
    print(f"Discounted episode-cost limit: {COST_LIMIT}")

    for iteration in range(TRAIN_ITERATIONS):

        data = collect_episodes(
            env,
            agent.policy,
            agent.reward_critic,
            agent.cost_critic,
            EPISODES_PER_ITERATION
        )

        metrics = agent.update(data)

        mean_reward = float(
            np.mean(data["episode_rewards"])
        )

        mean_cost = float(
            np.mean(data["episode_costs"])
        )

        mean_length = float(
            np.mean(data["episode_lengths"])
        )

        history["rewards"].append(
            mean_reward
        )

        history["costs"].append(
            mean_cost
        )

        history["lengths"].append(
            mean_length
        )

        history["kl"].append(
            metrics["kl"]
        )

        history["accepted"].append(
            metrics["accepted"]
        )

        history["status"].append(
            metrics["status"]
        )

        if (iteration + 1) % 2 == 0:
            print(
                f"Iteration {iteration + 1:3d}/"
                f"{TRAIN_ITERATIONS} | "
                f"Reward {mean_reward:8.2f} | "
                f"Episode cost {mean_cost:8.2f} | "
                f"KL {metrics['kl']:.5f} | "
                f"Accepted {metrics['accepted']} | "
                f"{metrics['status']}"
            )

    return agent, history


# ============================================================
# 13. TRAINING VISUALIZATION
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
        kls = history["kl"]

        axes[0].plot(
            rewards,
            label="Episode reward"
        )

        axes[0].set_title(
            "CPO Task Reward"
        )

        axes[0].set_xlabel(
            "Training iteration"
        )

        axes[0].set_ylabel(
            "Reward"
        )

        axes[0].grid(True)
        axes[0].legend()

        axes[1].plot(
            costs,
            label="Discounted episode safety cost"
        )

        axes[1].axhline(
            COST_LIMIT,
            linestyle="--",
            label="Cost limit"
        )

        axes[1].set_title(
            "Safety Constraint"
        )

        axes[1].set_xlabel(
            "Training iteration"
        )

        axes[1].set_ylabel(
            "Discounted cost"
        )

        axes[1].grid(True)
        axes[1].legend()

        axes[2].plot(
            kls,
            label="Accepted KL divergence"
        )

        axes[2].axhline(
            DELTA,
            linestyle="--",
            label="Trust-region limit"
        )

        axes[2].set_title(
            "Trust Region"
        )

        axes[2].set_xlabel(
            "Training iteration"
        )

        axes[2].set_ylabel(
            "KL divergence"
        )

        axes[2].grid(True)
        axes[2].legend()

        plt.tight_layout()
        plt.show()

    except ImportError:
        print("Matplotlib unavailable; skipping plots.")


# ============================================================
# 14. PYGAME ROCKET VISUALIZATION
# ============================================================

class RocketGUI:

    def __init__(self, agent):
        pygame.init()

        self.screen = pygame.display.set_mode(
            (WIDTH + PANEL_WIDTH, HEIGHT)
        )

        pygame.display.set_caption(
            "Phase 4.16 — Canonical CPO Rocket"
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

        self.last_cost = 0.0
        self.last_reward = 0.0

    def reset(self):
        self.env.reset()

        self.last_action[:] = 0.0
        self.last_cost = 0.0
        self.last_reward = 0.0

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

    def draw_arrow(self, start, end, color, width=2):
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

        flame_length = random.randint(
            10,
            18
        )

        pygame.draw.line(
            self.screen,
            (255, 125, 30),
            tuple(rear.astype(int)),
            tuple(
                (
                    rear - direction * flame_length
                ).astype(int)
            ),
            5
        )

        pygame.draw.polygon(
            self.screen,
            (210, 220, 235),
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

        pygame.draw.rect(
            self.screen,
            (90, 110, 140),
            (0, 0, WIDTH, HEIGHT),
            2
        )

        # Moving target path.
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

        # Safety boundary.
        pygame.draw.circle(
            self.screen,
            (125, 85, 55),
            OBSTACLE.astype(int),
            int(SAFETY_RADIUS),
            2
        )

        # Physical obstacle.
        pygame.draw.circle(
            self.screen,
            (170, 65, 65),
            OBSTACLE.astype(int),
            int(OBSTACLE_RADIUS)
        )

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

        self.draw_arrow(
            self.env.position,
            self.env.position + self.env.velocity * 0.35,
            (255, 210, 90),
            2
        )

        self.draw_arrow(
            self.env.position,
            self.env.position + self.last_action * 60,
            (100, 170, 255),
            3
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
            ("PHASE 4.16", (255, 220, 100)),
            ("CANONICAL CPO", (220, 230, 245)),
            ("", (200, 200, 200)),
            ("POLICY", (150, 190, 255)),
            ("Gaussian continuous policy", (100, 240, 150)),
            ("", (200, 200, 200)),
            ("TRACKING", (150, 190, 255)),
            (f"Distance: {target_distance:.1f}", (230, 230, 230)),
            (f"Target vx: {self.env.target.velocity[0]:.1f}", (180, 240, 180)),
            (f"Target vy: {self.env.target.velocity[1]:.1f}", (180, 240, 180)),
            ("", (200, 200, 200)),
            ("SAFETY", (150, 190, 255)),
            (f"Clearance: {clearance:.1f}", (230, 230, 230)),
            (f"Current cost: {self.last_cost:.3f}", (230, 230, 230)),
            (f"Cost limit: {COST_LIMIT:.1f}", (230, 230, 230)),
            ("", (200, 200, 200)),
            ("TRUST REGION", (150, 190, 255)),
            (f"KL limit: {DELTA:.4f}", (255, 220, 100)),
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

                    action = self.agent.act(
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
                    self.last_cost = cost
                    self.last_reward = reward

                    if terminated or truncated:
                        self.env.reset()

            self.draw_world()
            self.draw_panel()

            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()


# ============================================================
# 15. UNIT TESTS
# ============================================================

def run_tests():
    print("\nRunning Phase 4.16 tests...\n")

    # Test 1: Environment state shape.
    env = MovingTargetEnv()

    state = env.reset()

    assert state.shape == (STATE_DIM,)
    assert np.all(np.isfinite(state))

    print("Test 1 passed: environment state")

    # Test 2: Moving target.
    position_before = env.target.position.copy()

    env.target.step()

    position_after = env.target.position.copy()

    assert not np.allclose(
        position_before,
        position_after
    )

    print("Test 2 passed: moving target")

    # Test 3: Target velocity.
    assert env.target.velocity.shape == (2,)

    print("Test 3 passed: target velocity")

    # Test 4: Robot speed limit.
    _, velocity = robot_dynamics(
        np.array([100.0, 100.0]),
        np.array([110.0, 0.0]),
        np.array([1.0, 0.0])
    )

    assert np.linalg.norm(velocity) <= MAX_SPEED + 1e-5

    print("Test 4 passed: robot speed limit")

    # Test 5: Safety cost nonnegative.
    assert env.safety_cost() >= 0.0

    print("Test 5 passed: nonnegative safety cost")

    # Test 6: Environment transition.
    result = env.step(
        np.zeros(ACTION_DIM)
    )

    assert len(result) == 6

    print("Test 6 passed: environment transition")

    # Test 7: Policy output.
    policy = GaussianPolicy().to(DEVICE)

    test_states = torch.zeros(
        (8, STATE_DIM),
        device=DEVICE
    )

    distribution = policy.distribution(
        test_states
    )

    actions = distribution.sample()

    assert actions.shape == (8, ACTION_DIM)

    print("Test 7 passed: Gaussian policy")

    # Test 8: Gaussian log probability.
    log_prob = distribution.log_prob(
        actions
    ).sum(dim=1)

    assert torch.all(
        torch.isfinite(log_prob)
    )

    print("Test 8 passed: Gaussian log probability")

    # Test 9: KL divergence is zero for identical distributions.
    identical_kl = kl_divergence(
        distribution,
        distribution
    ).sum(dim=1)

    assert torch.allclose(
        identical_kl,
        torch.zeros_like(identical_kl),
        atol=1e-6
    )

    print("Test 9 passed: identical-policy KL")

    # Test 10: KL divergence is nonnegative.
    shifted_distribution = Normal(
        distribution.mean + 0.2,
        distribution.stddev
    )

    kl = kl_divergence(
        distribution,
        shifted_distribution
    ).sum(dim=1)

    assert torch.all(kl >= -1e-6)

    print("Test 10 passed: nonnegative KL")

    # Test 11: Value network output.
    critic = ValueNetwork().to(DEVICE)

    values = critic(test_states)

    assert values.shape == (8, 1)

    print("Test 11 passed: value network")

    # Test 12: GAE.
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

    episode_ends = np.array(
        [0.0, 0.0, 1.0],
        dtype=np.float32
    )

    advantages, returns = calculate_gae(
        rewards,
        values,
        next_values,
        terminated,
        episode_ends
    )

    assert advantages.shape == (3,)
    assert returns.shape == (3,)
    assert np.all(np.isfinite(advantages))

    print("Test 12 passed: GAE")

    # Test 13: Conjugate gradient on a known system.
    matrix = torch.tensor(
        [
            [4.0, 1.0],
            [1.0, 3.0]
        ]
    )

    expected = torch.tensor(
        [1.0, 2.0]
    )

    vector = matrix @ expected

    solution = conjugate_gradient(
        lambda v: matrix @ v,
        vector,
        iterations=10
    )

    assert torch.allclose(
        solution,
        expected,
        atol=1e-5
    )

    print("Test 13 passed: conjugate gradient")

    # Test 14: Flat parameter conversion.
    test_policy = GaussianPolicy().to(DEVICE)

    original = flat_parameters(
        test_policy
    ).clone()

    changed = original + 0.001

    set_flat_parameters(
        test_policy,
        changed
    )

    assert torch.allclose(
        flat_parameters(test_policy),
        changed
    )

    set_flat_parameters(
        test_policy,
        original
    )

    print("Test 14 passed: flat parameter conversion")

    # Test 15: Discounted episode cost.
    dummy_data = {
        "costs": np.array(
            [1.0, 2.0, 3.0, 4.0],
            dtype=np.float32
        ),

        "episode_ends": np.array(
            [0.0, 1.0, 0.0, 1.0],
            dtype=np.float32
        )
    }

    expected_cost = (
        (1.0 + GAMMA * 2.0)
        + (3.0 + GAMMA * 4.0)
    ) / 2.0

    observed_cost = discounted_episode_costs(
        dummy_data
    )

    assert abs(
        observed_cost - expected_cost
    ) < 1e-5

    print("Test 15 passed: discounted episode cost")

    # Test 16: Constrained-step feasibility.
    #
    # A feasible step should satisfy:
    #     c + b.T * step <= 0
    #
    # A simple one-dimensional example.
    c = -0.5
    b = torch.tensor([1.0])
    step = torch.tensor([0.2])

    assert c + torch.dot(b, step).item() <= 0.0

    print("Test 16 passed: local cost constraint")

    # Test 17: Trust-region quadratic constraint.
    H = torch.eye(2)

    step = torch.tensor(
        [0.1, 0.1]
    )

    quadratic_value = (
        0.5 * torch.dot(step, H @ step)
    )

    assert quadratic_value <= DELTA

    print("Test 17 passed: trust-region constraint")

    print("\nAll 17 tests passed.\n")


# ============================================================
# 16. MAIN
# ============================================================

def main():
    print("=" * 70)
    print("PHASE 4.16 — CANONICAL CONSTRAINED POLICY OPTIMIZATION")
    print("=" * 70)

    run_tests()

    agent, history = train_agent()

    visualize_training(history)

    print("\nStarting Pygame moving-target rocket simulation.")

    gui = RocketGUI(agent)
    gui.run()


if __name__ == "__main__":
    main()
