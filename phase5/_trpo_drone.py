
"""
File: phase5/trpo_drone.py

PHASE 5.8 — TRUST REGION POLICY OPTIMIZATION (TRPO)
====================================================

PURPOSE
-------
Implement TRPO from scratch using PyTorch.

WHAT YOU LEARN
--------------

STEP 1 — Policy Gradient
    Optimize the expected return using sampled trajectories.

STEP 2 — KL Divergence
    Measure the difference between old and new Gaussian policies.

STEP 3 — Fisher-Vector Product
    Compute Hessian-vector products of KL divergence without
    explicitly constructing the Hessian matrix.

STEP 4 — Conjugate Gradient
    Approximately solve:

        H x = g

    where H is the Fisher information matrix and g is the
    policy gradient.

STEP 5 — Trust Region Update
    Limit the KL divergence between old and new policies.

STEP 6 — Backtracking Line Search
    Accept an update only if:
        - KL divergence is below the limit.
        - The surrogate objective improves.

STEP 7 — Robotics Application
    Train a quadcopter to track a continuously moving target
    while avoiding an obstacle.

MATHEMATICAL FORMULATION
------------------------

SURROGATE OBJECTIVE:

    L(theta) =
        E[
            pi_theta(a|s) / pi_old(a|s) * A_old(s,a)
        ]

TRPO CONSTRAINT:

    maximize L(theta)

    subject to:

        E[ KL(
            pi_old(.|s),
            pi_theta(.|s)
        ) ] <= delta

LOCAL APPROXIMATION:

    maximize:

        g^T x

    subject to:

        0.5 * x^T H x <= delta

ANALYTICAL SOLUTION DIRECTION:

    x = sqrt(
        2 * delta / (g^T H^-1 g)
    ) * H^-1 g

FISHER-VECTOR PRODUCT:

    H v = Hessian(KL) * v

CONJUGATE GRADIENT:

    Approximately solve:

        H x = g

LINE SEARCH:

    theta_new = theta_old + alpha * x

    Try:

        alpha = 1, 1/2, 1/4, 1/8, ...

    Accept if:
        KL <= delta
        surrogate improvement > 0

ADVANTAGE ESTIMATION:

    GAE(lambda):

        delta_t =
            r_t + gamma * V(s_{t+1}) - V(s_t)

        A_t =
            delta_t + gamma * lambda * A_(t+1)

    True terminal transitions suppress bootstrap.
    Time-limit truncations bootstrap from their final state.

IMPORTANT
---------
This is a simplified educational TRPO implementation.

The policy uses a diagonal Gaussian distribution.
Actions are clipped before entering the environment.
The likelihood ratio is evaluated using the original
unclipped sampled actions.

This is not a flight-certified controller.
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

SEED = 580

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

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

OBS_DIM = 10
ACTION_DIM = 2

GAMMA = 0.99
GAE_LAMBDA = 0.95

MAX_EPISODE_STEPS = 180

# TRPO parameters.

MAX_KL = 0.01

CG_ITERATIONS = 10
CG_TOLERANCE = 1e-10

DAMPING = 0.1

BACKTRACK_STEPS = 10
BACKTRACK_COEFFICIENT = 0.5

VALUE_LEARNING_RATE = 3e-4
VALUE_EPOCHS = 5

TRAIN_EPISODES = 500

PRINT_INTERVAL = 10


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

def clip_state(state):

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

    # The policy action is clipped before entering the simulator.

    action = np.clip(
        action,
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

    return clip_state(
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
                np.random.uniform(80, 140),
                np.random.uniform(440, 540),
                0.0,
                0.0,
            ],
            dtype=np.float32,
        )

        self.target.phase = np.random.uniform(
            0,
            2 * math.pi,
        )

        self.steps = 0

        return self.observation()

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
# 4. ACTOR AND CRITIC NETWORKS
# ============================================================

class GaussianPolicy(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(OBS_DIM, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, ACTION_DIM),
        )

        self.log_std = nn.Parameter(
            torch.full(
                (ACTION_DIM,),
                -0.5,
            )
        )

    def forward(self, states):

        mean = self.network(states)

        log_std = torch.clamp(
            self.log_std,
            -5.0,
            1.0,
        )

        return mean, log_std

    def distribution(self, states):

        mean, log_std = self(states)

        return torch.distributions.Normal(
            mean,
            log_std.exp(),
        )

    def log_prob(self, states, actions):

        distribution = self.distribution(
            states
        )

        return distribution.log_prob(
            actions
        ).sum(dim=-1)

    def sample(self, states):

        distribution = self.distribution(
            states
        )

        actions = distribution.sample()

        log_probs = distribution.log_prob(
            actions
        ).sum(dim=-1)

        return (
            actions,
            log_probs,
        )

    def deterministic_action(self, states):

        mean, _ = self(states)

        return torch.tanh(mean)


class ValueNetwork(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(OBS_DIM, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

    def forward(self, states):

        return self.network(
            states
        ).squeeze(-1)


# ============================================================
# 5. FLATTEN PARAMETERS
# ============================================================

def get_flat_parameters(model):

    return torch.cat(
        [
            parameter.detach().reshape(-1)
            for parameter in model.parameters()
        ]
    )


def set_flat_parameters(model, flat_parameters):

    offset = 0

    with torch.no_grad():

        for parameter in model.parameters():

            count = parameter.numel()

            parameter.copy_(
                flat_parameters[
                    offset:offset + count
                ].view_as(parameter)
            )

            offset += count

    assert offset == flat_parameters.numel()


def flat_gradient(
    output,
    model,
    create_graph=False,
):

    gradients = torch.autograd.grad(
        output,
        model.parameters(),
        create_graph=create_graph,
        allow_unused=False,
    )

    return torch.cat(
        [
            gradient.reshape(-1)
            for gradient in gradients
        ]
    )


# ============================================================
# 6. KL DIVERGENCE
# ============================================================

def gaussian_kl(
    old_mean,
    old_log_std,
    new_mean,
    new_log_std,
):

    old_std = old_log_std.exp()
    new_std = new_log_std.exp()

    kl = (
        new_log_std
        - old_log_std
        + (
            old_std.pow(2)
            + (old_mean - new_mean).pow(2)
        ) / (
            2.0 * new_std.pow(2)
        )
        - 0.5
    )

    return kl.sum(
        dim=-1
    )


# ============================================================
# 7. FISHER-VECTOR PRODUCT
# ============================================================

def fisher_vector_product(
    policy,
    states,
    vector,
):

    # The old policy is represented by detached copies
    # of its current Gaussian parameters.

    with torch.no_grad():

        old_mean, old_log_std = policy(states)

        old_mean = old_mean.detach()
        old_log_std = old_log_std.detach()

    new_mean, new_log_std = policy(states)

    kl = gaussian_kl(
        old_mean,
        old_log_std,
        new_mean,
        new_log_std,
    ).mean()

    # First derivative of KL.

    first_gradient = flat_gradient(
        kl,
        policy,
        create_graph=True,
    )

    # Dot the first derivative with an arbitrary vector.

    product = (
        first_gradient * vector
    ).sum()

    # Second derivative multiplied by vector.

    second_gradient = flat_gradient(
        product,
        policy,
        create_graph=False,
    )

    return (
        second_gradient.detach()
        + DAMPING * vector
    )


# ============================================================
# 8. CONJUGATE GRADIENT
# ============================================================

def conjugate_gradient(
    matrix_vector_product,
    b,
    iterations=CG_ITERATIONS,
    tolerance=CG_TOLERANCE,
):

    x = torch.zeros_like(b)

    residual = b.clone()

    direction = residual.clone()

    residual_dot = torch.dot(
        residual,
        residual,
    )

    for _ in range(iterations):

        product = matrix_vector_product(
            direction
        )

        denominator = torch.dot(
            direction,
            product,
        )

        if denominator.abs().item() < 1e-12:

            break

        alpha = (
            residual_dot / denominator
        )

        x = x + alpha * direction

        residual = (
            residual - alpha * product
        )

        new_residual_dot = torch.dot(
            residual,
            residual,
        )

        if (
            new_residual_dot.item()
            < tolerance
        ):

            break

        beta = (
            new_residual_dot
            / (residual_dot + 1e-12)
        )

        direction = (
            residual + beta * direction
        )

        residual_dot = new_residual_dot

    return x


# ============================================================
# 9. COLLECT A COMPLETE EPISODE
# ============================================================

def collect_episode(
    env,
    policy,
    value_network,
):

    observation = env.reset()

    observations = []
    actions = []
    rewards = []
    values = []
    next_values = []
    terminated_flags = []
    truncated_flags = []
    log_probs = []
    distances = []

    for _ in range(
        MAX_EPISODE_STEPS
    ):

        state_tensor = torch.as_tensor(
            observation,
            dtype=torch.float32,
            device=DEVICE,
        ).unsqueeze(0)

        with torch.no_grad():

            action_tensor, log_prob_tensor = (
                policy.sample(
                    state_tensor
                )
            )

            value = value_network(
                state_tensor
            ).item()

        raw_action = (
            action_tensor.squeeze(0)
            .cpu()
            .numpy()
        )

        # Preserve the sampled action for likelihood evaluation.
        # The environment clips the action to its valid range.

        (
            next_observation,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(raw_action)

        next_state_tensor = torch.as_tensor(
            next_observation,
            dtype=torch.float32,
            device=DEVICE,
        ).unsqueeze(0)

        with torch.no_grad():

            next_value = value_network(
                next_state_tensor
            ).item()

        observations.append(
            observation.copy()
        )

        actions.append(
            raw_action.copy()
        )

        rewards.append(reward)

        values.append(value)

        next_values.append(next_value)

        terminated_flags.append(
            float(terminated)
        )

        truncated_flags.append(
            float(truncated)
        )

        log_probs.append(
            log_prob_tensor.item()
        )

        distances.append(
            info["distance"]
        )

        observation = next_observation

        if terminated or truncated:

            break

    return {
        "observations": np.asarray(
            observations,
            dtype=np.float32,
        ),

        "actions": np.asarray(
            actions,
            dtype=np.float32,
        ),

        "rewards": np.asarray(
            rewards,
            dtype=np.float32,
        ),

        "values": np.asarray(
            values,
            dtype=np.float32,
        ),

        "next_values": np.asarray(
            next_values,
            dtype=np.float32,
        ),

        "terminated": np.asarray(
            terminated_flags,
            dtype=np.float32,
        ),

        "truncated": np.asarray(
            truncated_flags,
            dtype=np.float32,
        ),

        "old_log_probs": np.asarray(
            log_probs,
            dtype=np.float32,
        ),

        "distances": np.asarray(
            distances,
            dtype=np.float32,
        ),
    }


# ============================================================
# 10. GAE
# ============================================================

def compute_gae(episode):

    rewards = episode["rewards"]
    values = episode["values"]
    next_values = episode["next_values"]

    terminated = episode["terminated"]
    truncated = episode["truncated"]

    advantages = np.zeros_like(
        rewards
    )

    gae = 0.0

    for t in reversed(
        range(len(rewards))
    ):

        bootstrap_mask = (
            1.0 - terminated[t]
        )

        continuation_mask = (
            1.0
            - max(
                terminated[t],
                truncated[t],
            )
        )

        delta = (
            rewards[t]
            + GAMMA
            * bootstrap_mask
            * next_values[t]
            - values[t]
        )

        gae = (
            delta
            + GAMMA
            * GAE_LAMBDA
            * continuation_mask
            * gae
        )

        advantages[t] = gae

    returns = (
        advantages + values
    )

    return (
        advantages,
        returns,
    )


# ============================================================
# 11. VALUE FUNCTION UPDATE
# ============================================================

def update_value_network(
    value_network,
    optimizer,
    states,
    returns,
):

    losses = []

    for _ in range(
        VALUE_EPOCHS
    ):

        predictions = value_network(
            states
        )

        loss = (
            predictions - returns
        ).pow(2).mean()

        optimizer.zero_grad()

        loss.backward()

        nn.utils.clip_grad_norm_(
            value_network.parameters(),
            0.5,
        )

        optimizer.step()

        losses.append(
            loss.item()
        )

    return float(
        np.mean(losses)
    )


# ============================================================
# 12. TRPO POLICY UPDATE
# ============================================================

def trpo_update(
    policy,
    states,
    actions,
    old_log_probs,
    advantages,
):

    states = states.detach()
    actions = actions.detach()

    old_log_probs = old_log_probs.detach()
    advantages = advantages.detach()

    # Normalize advantages.

    advantages = (
        advantages - advantages.mean()
    ) / (
        advantages.std(unbiased=False) + 1e-8
    )

    # Snapshot the original policy parameters.

    old_parameters = get_flat_parameters(
        policy
    )

    # Snapshot old Gaussian distribution.

    with torch.no_grad():

        old_mean, old_log_std = policy(
            states
        )

        old_mean = old_mean.detach()
        old_log_std = old_log_std.detach()

    # --------------------------------------------------------
    # 12.1 Policy surrogate objective
    # --------------------------------------------------------

    def surrogate_objective():

        new_log_probs = policy.log_prob(
            states,
            actions,
        )

        ratio = torch.exp(
            new_log_probs - old_log_probs
        )

        return (
            ratio * advantages
        ).mean()

    objective = surrogate_objective()

    # Policy gradient.

    gradient = flat_gradient(
        objective,
        policy,
        create_graph=False,
    ).detach()

    gradient_norm = torch.norm(
        gradient
    ).item()

    if (
        not torch.isfinite(gradient).all()
        or gradient_norm < 1e-10
    ):

        return {
            "accepted": False,
            "kl": 0.0,
            "improvement": 0.0,
            "gradient_norm": gradient_norm,
        }

    # --------------------------------------------------------
    # 12.2 Fisher-vector product
    # --------------------------------------------------------

    def fisher_product(vector):

        return fisher_vector_product(
            policy,
            states,
            vector,
        )

    # --------------------------------------------------------
    # 12.3 Conjugate-gradient solution
    # --------------------------------------------------------

    direction = conjugate_gradient(
        fisher_product,
        gradient,
    )

    curvature = torch.dot(
        direction,
        fisher_product(direction),
    )

    if (
        not torch.isfinite(curvature)
        or curvature.item() <= 1e-12
    ):

        set_flat_parameters(
            policy,
            old_parameters,
        )

        return {
            "accepted": False,
            "kl": 0.0,
            "improvement": 0.0,
            "gradient_norm": gradient_norm,
        }

    # --------------------------------------------------------
    # 12.4 Scale direction to trust region
    # --------------------------------------------------------

    scale = torch.sqrt(
        2.0 * MAX_KL
        / (curvature + 1e-12)
    )

    full_step = (
        scale * direction
    )

    expected_improvement = torch.dot(
        gradient,
        full_step,
    ).item()

    # --------------------------------------------------------
    # 12.5 Backtracking line search
    # --------------------------------------------------------

    accepted = False

    accepted_kl = 0.0
    accepted_improvement = 0.0

    old_objective = objective.detach().item()

    for step_index in range(
        BACKTRACK_STEPS
    ):

        fraction = (
            BACKTRACK_COEFFICIENT
            ** step_index
        )

        candidate_parameters = (
            old_parameters
            + fraction * full_step
        )

        set_flat_parameters(
            policy,
            candidate_parameters,
        )

        with torch.no_grad():

            new_objective = (
                surrogate_objective().item()
            )

            new_mean, new_log_std = policy(
                states
            )

            kl = gaussian_kl(
                old_mean,
                old_log_std,
                new_mean,
                new_log_std,
            ).mean().item()

        actual_improvement = (
            new_objective - old_objective
        )

        # The actual improvement should be positive.
        # The measured KL must remain within the trust region.

        if (
            np.isfinite(new_objective)
            and np.isfinite(kl)
            and actual_improvement > 0.0
            and kl <= MAX_KL
        ):

            accepted = True

            accepted_kl = kl

            accepted_improvement = (
                actual_improvement
            )

            break

    if not accepted:

        set_flat_parameters(
            policy,
            old_parameters,
        )

    return {
        "accepted": accepted,
        "kl": accepted_kl,
        "improvement": accepted_improvement,
        "expected_improvement": expected_improvement,
        "gradient_norm": gradient_norm,
        "curvature": curvature.item(),
    }


# ============================================================
# 13. TRAINING
# ============================================================

def train():

    env = DroneEnv()

    policy = GaussianPolicy().to(
        DEVICE
    )

    value_network = ValueNetwork().to(
        DEVICE
    )

    value_optimizer = optim.Adam(
        value_network.parameters(),
        lr=VALUE_LEARNING_RATE,
    )

    history = {
        "return": [],
        "distance": [],
        "value_loss": [],
        "kl": [],
        "improvement": [],
        "accepted": [],
    }

    for episode_index in range(
        TRAIN_EPISODES
    ):

        episode = collect_episode(
            env,
            policy,
            value_network,
        )

        advantages, returns = compute_gae(
            episode
        )

        states = torch.as_tensor(
            episode["observations"],
            dtype=torch.float32,
            device=DEVICE,
        )

        actions = torch.as_tensor(
            episode["actions"],
            dtype=torch.float32,
            device=DEVICE,
        )

        old_log_probs = torch.as_tensor(
            episode["old_log_probs"],
            dtype=torch.float32,
            device=DEVICE,
        )

        advantages_tensor = torch.as_tensor(
            advantages,
            dtype=torch.float32,
            device=DEVICE,
        )

        returns_tensor = torch.as_tensor(
            returns,
            dtype=torch.float32,
            device=DEVICE,
        )

        # Update policy with TRPO.

        policy_metrics = trpo_update(
            policy,
            states,
            actions,
            old_log_probs,
            advantages_tensor,
        )

        # Update value function separately.

        value_loss = update_value_network(
            value_network,
            value_optimizer,
            states,
            returns_tensor,
        )

        episode_return = float(
            episode["rewards"].sum()
        )

        mean_distance = float(
            episode["distances"].mean()
        )

        history["return"].append(
            episode_return
        )

        history["distance"].append(
            mean_distance
        )

        history["value_loss"].append(
            value_loss
        )

        history["kl"].append(
            policy_metrics["kl"]
        )

        history["improvement"].append(
            policy_metrics["improvement"]
        )

        history["accepted"].append(
            float(policy_metrics["accepted"])
        )

        if (
            episode_index + 1
        ) % PRINT_INTERVAL == 0:

            recent = slice(
                -PRINT_INTERVAL,
                None,
            )

            print(
                f"Episode {episode_index + 1:4d}/{TRAIN_EPISODES}"
                f" | Return={np.mean(history['return'][recent]):8.2f}"
                f" | Distance={np.mean(history['distance'][recent]):7.2f}"
                f" | KL={policy_metrics['kl']:.5f}"
                f" | Accepted={policy_metrics['accepted']}"
                f" | Value loss={value_loss:.4f}"
            )

    return (
        policy,
        value_network,
        history,
    )


# ============================================================
# 14. TRAINING VISUALIZATION
# ============================================================

def plot_history(history):

    plots = [
        ("return", "Episode Return"),
        ("distance", "Mean Target Distance"),
        ("value_loss", "Critic Loss"),
        ("kl", "Accepted KL Divergence"),
        ("improvement", "Surrogate Improvement"),
        ("accepted", "Accepted Policy Update"),
    ]

    for key, title in plots:

        plt.figure(
            figsize=(10, 4)
        )

        plt.plot(
            history[key]
        )

        plt.title(title)

        plt.xlabel("Episode")

        plt.ylabel(key)

        plt.grid(True)

        plt.tight_layout()

        plt.show()


# ============================================================
# 15. PYGAME QUADCOPTER VISUALIZATION
# ============================================================

def run_pygame(policy):

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
        "Phase 5.8 — TRPO Drone"
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

    def text(message, x, y):

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

        pygame.draw.circle(
            screen,
            (255, 185, 55),
            (
                center + direction * 22
            ).astype(int),
            4,
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

    def draw_target():

        points = []

        for i in range(100):

            phase = (
                env.target.phase
                + 2 * math.pi * i / 100
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
                        np.asarray(
                            event.pos,
                            dtype=np.float32,
                        )
                    )

        if not paused:

            for _ in range(speed):

                state_tensor = torch.as_tensor(
                    observation,
                    dtype=torch.float32,
                    device=DEVICE,
                ).unsqueeze(0)

                with torch.no_grad():

                    action = (
                        policy.deterministic_action(
                            state_tensor
                        )
                    )

                action = (
                    action.squeeze(0)
                    .cpu()
                    .numpy()
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

        text(
            "PHASE 5.8 — TRUST REGION POLICY OPTIMIZATION",
            15,
            12,
        )

        text(
            f"Target distance: {info['distance'] if 'info' in locals() else 0:.2f}",
            15,
            40,
        )

        text(
            f"Episode reward: {total_reward:.2f}",
            15,
            65,
        )

        text(
            f"Simulation speed: {speed}x",
            15,
            90,
        )

        text(
            "Click: change moving-target trajectory center",
            15,
            WORLD_H - 100,
        )

        text(
            "SPACE pause | R reset | +/- speed | ESC exit",
            15,
            WORLD_H - 70,
        )

        if paused:

            text(
                "PAUSED",
                WORLD_W - 100,
                15,
            )

        pygame.display.flip()

    pygame.quit()


# ============================================================
# 16. NUMERICAL TESTS
# ============================================================

def run_tests():

    print(
        "\nRunning TRPO tests..."
    )

    passed = 0

    def check(condition, description):

        nonlocal passed

        assert condition, description

        passed += 1

        print(
            "[PASS]",
            description,
        )

    # Test 1: Gaussian KL is zero for identical distributions.

    mean = torch.zeros(
        (4, 2),
        device=DEVICE,
    )

    log_std = torch.zeros(
        2,
        device=DEVICE,
    )

    kl = gaussian_kl(
        mean,
        log_std,
        mean,
        log_std,
    )

    check(
        torch.allclose(
            kl,
            torch.zeros_like(kl),
            atol=1e-7,
        ),
        "Identical Gaussian policies have zero KL",
    )

    # Test 2: Gaussian KL is nonnegative.

    shifted_mean = mean + 0.5

    kl = gaussian_kl(
        mean,
        log_std,
        shifted_mean,
        log_std,
    )

    check(
        bool(
            torch.all(kl >= 0)
        ),
        "Gaussian KL is nonnegative",
    )

    # Test 3: Parameter flattening.

    policy = GaussianPolicy().to(
        DEVICE
    )

    original = get_flat_parameters(
        policy
    )

    set_flat_parameters(
        policy,
        original,
    )

    check(
        original.numel()
        == sum(
            parameter.numel()
            for parameter in policy.parameters()
        ),
        "Flat parameter vector has correct size",
    )

    # Test 4: Flat parameter round trip.

    restored = get_flat_parameters(
        policy
    )

    check(
        torch.allclose(
            original,
            restored,
        ),
        "Parameter flatten and restore",
    )

    # Test 5: Conjugate gradient solves identity system.

    b = torch.tensor(
        [1.0, 2.0, 3.0],
        device=DEVICE,
    )

    solution = conjugate_gradient(
        lambda x: x,
        b,
        iterations=5,
    )

    check(
        torch.allclose(
            solution,
            b,
            atol=1e-5,
        ),
        "Conjugate gradient identity system",
    )

    # Test 6: Conjugate gradient solves diagonal system.

    diagonal = torch.tensor(
        [1.0, 2.0, 4.0],
        device=DEVICE,
    )

    solution = conjugate_gradient(
        lambda x: diagonal * x,
        b,
        iterations=10,
    )

    expected = b / diagonal

    check(
        torch.allclose(
            solution,
            expected,
            atol=1e-4,
        ),
        "Conjugate gradient diagonal system",
    )

    # Test 7: Fisher-vector product is finite.

    states = torch.randn(
        16,
        OBS_DIM,
        device=DEVICE,
    )

    vector = torch.randn(
        get_flat_parameters(policy).shape,
        device=DEVICE,
    )

    product = fisher_vector_product(
        policy,
        states,
        vector,
    )

    check(
        bool(
            torch.isfinite(product).all()
        ),
        "Fisher-vector product is finite",
    )

    # Test 8: Fisher-vector product has correct shape.

    check(
        product.shape == vector.shape,
        "Fisher-vector product dimensions",
    )

    # Test 9: Policy log probability is finite.

    actions = torch.randn(
        16,
        ACTION_DIM,
        device=DEVICE,
    )

    log_probs = policy.log_prob(
        states,
        actions,
    )

    check(
        bool(
            torch.isfinite(log_probs).all()
        ),
        "Policy log probabilities are finite",
    )

    # Test 10: Value network output shape.

    value_network = ValueNetwork().to(
        DEVICE
    )

    values = value_network(
        states
    )

    check(
        values.shape == (16,),
        "Value network output shape",
    )

    # Test 11: GAE terminal bootstrap mask.

    episode = {
        "rewards": np.array([1.0]),
        "values": np.array([0.5]),
        "next_values": np.array([100.0]),
        "terminated": np.array([1.0]),
        "truncated": np.array([0.0]),
    }

    advantage, returns = compute_gae(
        episode
    )

    check(
        np.isclose(
            advantage[0],
            0.5,
        ),
        "Terminal transition suppresses bootstrap",
    )

    # Test 12: GAE truncation bootstrap.

    episode = {
        "rewards": np.array([1.0]),
        "values": np.array([0.5]),
        "next_values": np.array([2.0]),
        "terminated": np.array([0.0]),
        "truncated": np.array([1.0]),
    }

    advantage, returns = compute_gae(
        episode
    )

    check(
        np.isclose(
            advantage[0],
            1.0 + GAMMA * 2.0 - 0.5,
        ),
        "Truncation bootstraps from final state",
    )

    # Test 13: Environment observation.

    env = DroneEnv()

    observation = env.reset()

    check(
        observation.shape == (OBS_DIM,),
        "Drone observation dimensions",
    )

    # Test 14: Environment action and output.

    result = env.step(
        np.zeros(
            ACTION_DIM,
            dtype=np.float32,
        )
    )

    check(
        result[0].shape == (OBS_DIM,)
        and np.isfinite(result[1]),
        "Drone environment transition",
    )

    # Test 15: Moving target velocity.

    target = MovingTarget(
        [400.0, 300.0]
    )

    check(
        target.velocity().shape == (2,),
        "Moving target velocity dimensions",
    )

    print(
        f"\n{passed}/15 tests passed."
    )

    return passed == 15


# ============================================================
# 17. MAIN
# ============================================================

def main():

    print("=" * 72)

    print(
        "PHASE 5.8 — TRUST REGION POLICY OPTIMIZATION"
    )

    print("=" * 72)

    print(
        "Device:",
        DEVICE,
    )

    if not run_tests():

        raise RuntimeError(
            "TRPO numerical tests failed."
        )

    print(
        "\nStarting TRPO training..."
    )

    (
        policy,
        value_network,
        history,
    ) = train()

    print(
        "\nPlotting training results..."
    )

    plot_history(
        history
    )

    print(
        "\nStarting interactive quadcopter visualization..."
    )

    run_pygame(
        policy
    )

    print(
        "\nPhase 5.8 completed."
    )


if __name__ == "__main__":

    main()
