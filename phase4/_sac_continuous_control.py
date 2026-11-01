
# ============================================================
# File: phase4/sac_continuous_control.py
#
# PHASE 4.10 — SOFT ACTOR-CRITIC (SAC)
#
# PURPOSE
#   Implement a modern off-policy RL algorithm for continuous
#   robot control.
#
# WHAT YOU LEARN
#   Step 1: Stochastic policies and Gaussian distributions
#   Step 2: Actor network predicting mean and log standard deviation
#   Step 3: Reparameterization trick
#   Step 4: Tanh-squashed Gaussian action distributions
#   Step 5: Twin critics and target critics
#   Step 6: Entropy-regularized Bellman target
#   Step 7: Automatic entropy-temperature adjustment
#   Step 8: Experience replay and SAC training
#   Step 9: Pygame rocket visualization
#   Step 10: Numerical verification and unit tests
#
# IMPORTANT FORMULAS
#
#   Gaussian policy:
#       u = mu(s) + sigma(s) * epsilon
#       epsilon ~ N(0, I)
#
#   Bounded action:
#       a = tanh(u)
#
#   Squashed Gaussian log probability:
#       log pi(a|s)
#       = log N(u; mu, sigma)
#         - sum(log(1 - tanh(u)^2 + epsilon))
#
#   Soft Bellman target:
#       y = r + gamma*(1-done) *
#           [ min(Q1_target(s',a'), Q2_target(s',a'))
#             - alpha*log_pi(a'|s') ]
#
#   Critic loss:
#       L_Q = MSE(Q(s,a), y)
#
#   Actor loss:
#       L_pi = E[alpha*log_pi(a|s) - min(Q1,Q2)]
#
#   Temperature loss:
#       L_alpha = -E[
#           log_alpha * (log_pi(a|s) + target_entropy)
#       ]
#
# SAC learns to maximize expected reward AND policy entropy.
#
# ROBOTICS APPLICATION
#   A mobile robot learns continuous steering and thrust.
#   The stochastic policy can explore multiple control choices
#   rather than always choosing one deterministic action.
#
# PYGAME CONTROLS
#   Left click : Change target
#   SPACE      : Pause / resume
#   R          : Reset simulation
#   +/-        : Change simulation speed
#   ESC        : Exit
#
# INSTALL
#   pip install numpy pygame torch matplotlib
#
# RUN
#   python phase4/sac_continuous_control.py
# ============================================================

from __future__ import annotations

import math
import random
import time

from collections import deque
from dataclasses import dataclass

import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt


# ============================================================
# STEP 0 — CONFIGURATION
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

WORLD_W = 700
WORLD_H = 560

WINDOW_W = 1020
WINDOW_H = 650

STATE_SIZE = 10
ACTION_SIZE = 2

# Action:
#   action[0] = steering in [-1, 1]
#   action[1] = thrust in [-1, 1]

HIDDEN = 256

GAMMA = 0.99
TAU = 0.005

ACTOR_LR = 3e-4
CRITIC_LR = 3e-4
ALPHA_LR = 3e-4

REPLAY_CAPACITY = 100_000
BATCH_SIZE = 128
WARMUP_STEPS = 1_000

TRAINING_EPISODES = 180
MAX_STEPS = 250

INITIAL_ALPHA = 0.2

# For a two-dimensional action space, a common initial choice
# is target_entropy = -ACTION_SIZE.
TARGET_ENTROPY = -float(ACTION_SIZE)

LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0

MAX_SPEED = 5.0
MAX_ANGULAR_SPEED = 0.09

OBSTACLE = np.array(
    [350.0, 280.0],
    dtype=np.float32,
)

OBSTACLE_RADIUS = 55.0
ROCKET_RADIUS = 12.0

SAFE_DISTANCE = (
    OBSTACLE_RADIUS
    + ROCKET_RADIUS
    + 22.0
)


# ============================================================
# STEP 1 — GENERAL HELPERS
# ============================================================

def distance(a, b):
    return float(np.linalg.norm(a - b))


def clamp_action(action):
    return np.clip(
        action,
        -1.0,
        1.0,
    ).astype(np.float32)


def soft_update(source, target, tau=TAU):
    """
    Polyak averaging:

        target = tau * source + (1-tau) * target
    """

    with torch.no_grad():

        for source_param, target_param in zip(
            source.parameters(),
            target.parameters(),
        ):

            target_param.data.mul_(1.0 - tau)

            target_param.data.add_(
                tau * source_param.data
            )


def hard_update(source, target):
    target.load_state_dict(
        source.state_dict()
    )


# ============================================================
# STEP 2 — CONTINUOUS ROBOT ENVIRONMENT
# ============================================================

def safe_position(position):

    x, y = position

    inside_world = (
        ROCKET_RADIUS <= x <= WORLD_W - ROCKET_RADIUS
        and ROCKET_RADIUS <= y <= WORLD_H - ROCKET_RADIUS
    )

    outside_obstacle = (
        distance(position, OBSTACLE) >= SAFE_DISTANCE
    )

    return inside_world and outside_obstacle


class RocketEnv:

    def __init__(self):
        self.reset()

    def random_safe_position(self):

        for _ in range(5000):

            position = np.array(
                [
                    random.uniform(35, WORLD_W - 35),
                    random.uniform(35, WORLD_H - 35),
                ],
                dtype=np.float32,
            )

            if safe_position(position):
                return position

        raise RuntimeError(
            "Unable to generate safe position."
        )

    def reset(self):

        self.position = self.random_safe_position()

        self.target = self.random_safe_position()

        while distance(
            self.position,
            self.target,
        ) < 230:

            self.target = self.random_safe_position()

        self.velocity = np.zeros(
            2,
            dtype=np.float32,
        )

        self.heading = random.uniform(
            -math.pi,
            math.pi,
        )

        self.steps = 0

        self.total_reward = 0.0

        self.success = False
        self.collision = False

        return self.get_state()

    def set_target(self, target):

        target = np.asarray(
            target,
            dtype=np.float32,
        )

        if not safe_position(target):
            return False

        if distance(
            target,
            self.position,
        ) < 25:

            return False

        self.target = target

        self.steps = 0
        self.total_reward = 0.0

        self.success = False
        self.collision = False

        return True

    def get_state(self):

        target_delta = (
            self.target - self.position
        )

        obstacle_delta = (
            OBSTACLE - self.position
        )

        target_angle = math.atan2(
            target_delta[1],
            target_delta[0],
        )

        angle_error = (
            target_angle - self.heading + math.pi
        ) % (2 * math.pi) - math.pi

        target_distance = np.linalg.norm(
            target_delta
        )

        return np.array(
            [
                target_delta[0] / WORLD_W,
                target_delta[1] / WORLD_H,

                target_distance / 900.0,

                math.sin(angle_error),
                math.cos(angle_error),

                self.velocity[0] / MAX_SPEED,
                self.velocity[1] / MAX_SPEED,

                obstacle_delta[0] / WORLD_W,
                obstacle_delta[1] / WORLD_H,

                (
                    np.linalg.norm(obstacle_delta)
                    - SAFE_DISTANCE
                ) / 900.0,
            ],
            dtype=np.float32,
        )

    def step(self, action):

        action = clamp_action(action)

        steering = float(action[0])
        thrust = float(action[1])

        old_distance = distance(
            self.position,
            self.target,
        )

        # ----------------------------------------------------
        # Simplified continuous vehicle dynamics
        # ----------------------------------------------------

        self.heading += (
            steering * MAX_ANGULAR_SPEED
        )

        forward = np.array(
            [
                math.cos(self.heading),
                math.sin(self.heading),
            ],
            dtype=np.float32,
        )

        acceleration = (
            forward * thrust * 0.22
        )

        self.velocity += acceleration

        self.velocity *= 0.96

        speed = np.linalg.norm(
            self.velocity
        )

        if speed > MAX_SPEED:

            self.velocity *= (
                MAX_SPEED / speed
            )

        candidate = (
            self.position + self.velocity
        )

        reward = -0.03

        terminated = False
        truncated = False

        # ----------------------------------------------------
        # Collision and boundary handling
        # ----------------------------------------------------

        if not safe_position(candidate):

            reward -= 20.0

            self.collision = True
            terminated = True

        else:

            self.position = candidate

            new_distance = distance(
                self.position,
                self.target,
            )

            # Reward progress.
            progress = (
                old_distance - new_distance
            )

            reward += 0.20 * progress

            # Discourage unnecessary movement.
            reward -= (
                0.015
                * np.linalg.norm(self.velocity)
            )

            if new_distance < 35:
                reward += 0.3

            if new_distance < 20:

                reward += 25.0

                self.success = True
                terminated = True

        self.steps += 1

        if (
            self.steps >= MAX_STEPS
            and not terminated
        ):

            truncated = True

        self.total_reward += reward

        return (
            self.get_state(),
            float(reward),
            terminated,
            truncated,
        )


# ============================================================
# STEP 3 — STOCHASTIC GAUSSIAN ACTOR
# ============================================================

class GaussianActor(nn.Module):

    def __init__(self):

        super().__init__()

        self.backbone = nn.Sequential(

            nn.Linear(
                STATE_SIZE,
                HIDDEN,
            ),

            nn.ReLU(),

            nn.Linear(
                HIDDEN,
                HIDDEN,
            ),

            nn.ReLU(),
        )

        self.mean_layer = nn.Linear(
            HIDDEN,
            ACTION_SIZE,
        )

        self.log_std_layer = nn.Linear(
            HIDDEN,
            ACTION_SIZE,
        )

    def forward(self, state):

        features = self.backbone(state)

        mean = self.mean_layer(features)

        log_std = self.log_std_layer(features)

        # Prevent excessively small or large variances.
        log_std = torch.clamp(
            log_std,
            LOG_STD_MIN,
            LOG_STD_MAX,
        )

        return mean, log_std

    def sample(
        self,
        state,
        deterministic=False,
    ):
        """
        Reparameterization:

            u = mean + std * epsilon

        where epsilon is standard Gaussian noise.

        Then:

            action = tanh(u)

        The tanh transformation keeps actions in [-1, 1].

        The log-probability must include the tanh Jacobian
        correction. Omitting it gives an incorrect density
        for the transformed action.
        """

        mean, log_std = self.forward(state)

        std = log_std.exp()

        distribution = torch.distributions.Normal(
            mean,
            std,
        )

        if deterministic:

            pre_tanh = mean

        else:

            pre_tanh = distribution.rsample()

        action = torch.tanh(
            pre_tanh
        )

        log_prob = distribution.log_prob(
            pre_tanh
        )

        # Change-of-variables correction:
        #
        # log |d tanh(u) / du|
        # = log(1 - tanh(u)^2)
        #
        # This stable form avoids directly computing
        # log(1 - tanh(u)^2).

        correction = 2.0 * (
            math.log(2.0)
            - pre_tanh
            - F.softplus(-2.0 * pre_tanh)
        )

        log_prob = (
            log_prob - correction
        ).sum(
            dim=-1,
            keepdim=True,
        )

        return (
            action,
            log_prob,
            torch.tanh(mean),
        )


# ============================================================
# STEP 4 — CRITIC NETWORK
# ============================================================

class Critic(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                STATE_SIZE + ACTION_SIZE,
                HIDDEN,
            ),

            nn.ReLU(),

            nn.Linear(
                HIDDEN,
                HIDDEN,
            ),

            nn.ReLU(),

            nn.Linear(
                HIDDEN,
                1,
            ),
        )

    def forward(
        self,
        state,
        action,
    ):

        x = torch.cat(
            [
                state,
                action,
            ],
            dim=1,
        )

        return self.network(x)


# ============================================================
# STEP 5 — REPLAY BUFFER
# ============================================================

@dataclass
class Transition:

    state: np.ndarray
    action: np.ndarray
    reward: float
    next_state: np.ndarray
    terminated: float


class ReplayBuffer:

    def __init__(self, capacity):

        self.memory = deque(
            maxlen=capacity
        )

    def add(
        self,
        state,
        action,
        reward,
        next_state,
        terminated,
    ):

        self.memory.append(
            Transition(
                state=state.copy(),
                action=action.copy(),
                reward=float(reward),
                next_state=next_state.copy(),
                terminated=float(terminated),
            )
        )

    def sample(self, batch_size):

        batch = random.sample(
            self.memory,
            batch_size,
        )

        states = np.stack(
            [
                item.state
                for item in batch
            ]
        )

        actions = np.stack(
            [
                item.action
                for item in batch
            ]
        )

        rewards = np.asarray(
            [
                item.reward
                for item in batch
            ],
            dtype=np.float32,
        ).reshape(-1, 1)

        next_states = np.stack(
            [
                item.next_state
                for item in batch
            ]
        )

        dones = np.asarray(
            [
                item.terminated
                for item in batch
            ],
            dtype=np.float32,
        ).reshape(-1, 1)

        def tensor(array):

            return torch.as_tensor(
                array,
                dtype=torch.float32,
                device=DEVICE,
            )

        return (
            tensor(states),
            tensor(actions),
            tensor(rewards),
            tensor(next_states),
            tensor(dones),
        )

    def __len__(self):
        return len(self.memory)


# ============================================================
# STEP 6 — SAC AGENT
# ============================================================

class SACAgent:

    def __init__(self):

        # Actor.
        self.actor = GaussianActor().to(
            DEVICE
        )

        # Two independent critics.
        self.critic1 = Critic().to(
            DEVICE
        )

        self.critic2 = Critic().to(
            DEVICE
        )

        # Target critics.
        self.target1 = Critic().to(
            DEVICE
        )

        self.target2 = Critic().to(
            DEVICE
        )

        hard_update(
            self.critic1,
            self.target1,
        )

        hard_update(
            self.critic2,
            self.target2,
        )

        self.actor_optimizer = optim.Adam(
            self.actor.parameters(),
            lr=ACTOR_LR,
        )

        self.critic1_optimizer = optim.Adam(
            self.critic1.parameters(),
            lr=CRITIC_LR,
        )

        self.critic2_optimizer = optim.Adam(
            self.critic2.parameters(),
            lr=CRITIC_LR,
        )

        # Learn log(alpha), not alpha directly.
        #
        # This guarantees:
        #       alpha = exp(log_alpha) > 0

        self.log_alpha = torch.tensor(
            math.log(INITIAL_ALPHA),
            dtype=torch.float32,
            device=DEVICE,
            requires_grad=True,
        )

        self.alpha_optimizer = optim.Adam(
            [
                self.log_alpha
            ],
            lr=ALPHA_LR,
        )

        self.replay = ReplayBuffer(
            REPLAY_CAPACITY
        )

        self.critic_losses = []
        self.actor_losses = []
        self.alpha_values = []

        self.training_steps = 0

    @property
    def alpha(self):

        return self.log_alpha.exp().detach()

    def select_action(
        self,
        state,
        deterministic=False,
    ):

        state_tensor = torch.as_tensor(
            state,
            dtype=torch.float32,
            device=DEVICE,
        ).unsqueeze(0)

        self.actor.eval()

        with torch.no_grad():

            action, _, mean_action = (
                self.actor.sample(
                    state_tensor,
                    deterministic=deterministic,
                )
            )

        self.actor.train()

        if deterministic:
            result = mean_action
        else:
            result = action

        return (
            result.squeeze(0)
            .cpu()
            .numpy()
            .astype(np.float32)
        )

    def train_step(self):

        if len(self.replay) < max(
            BATCH_SIZE,
            WARMUP_STEPS,
        ):

            return None

        (
            states,
            actions,
            rewards,
            next_states,
            dones,
        ) = self.replay.sample(
            BATCH_SIZE
        )

        # ----------------------------------------------------
        # STEP 6.1 — SOFT TARGET
        #
        # y = r + gamma*(1-done) *
        #       [min(Q1_target,Q2_target)
        #        - alpha*log_pi]
        #
        # Entropy contributes to the target.
        # ----------------------------------------------------

        with torch.no_grad():

            next_actions, next_log_prob, _ = (
                self.actor.sample(
                    next_states
                )
            )

            target_q1 = self.target1(
                next_states,
                next_actions,
            )

            target_q2 = self.target2(
                next_states,
                next_actions,
            )

            target_min = torch.minimum(
                target_q1,
                target_q2,
            )

            soft_target_q = (
                target_min
                - self.alpha * next_log_prob
            )

            target = (
                rewards
                + GAMMA
                * (1.0 - dones)
                * soft_target_q
            )

        # ----------------------------------------------------
        # STEP 6.2 — CRITIC UPDATES
        # ----------------------------------------------------

        current_q1 = self.critic1(
            states,
            actions,
        )

        current_q2 = self.critic2(
            states,
            actions,
        )

        critic1_loss = F.mse_loss(
            current_q1,
            target,
        )

        critic2_loss = F.mse_loss(
            current_q2,
            target,
        )

        self.critic1_optimizer.zero_grad()

        critic1_loss.backward()

        nn.utils.clip_grad_norm_(
            self.critic1.parameters(),
            10.0,
        )

        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()

        critic2_loss.backward()

        nn.utils.clip_grad_norm_(
            self.critic2.parameters(),
            10.0,
        )

        self.critic2_optimizer.step()

        # ----------------------------------------------------
        # STEP 6.3 — ACTOR UPDATE
        #
        # L_actor =
        #     E[alpha*log_pi - min(Q1,Q2)]
        #
        # The actor tries to:
        #   1. Find actions with high Q-values.
        #   2. Maintain sufficient policy entropy.
        # ----------------------------------------------------

        new_actions, log_prob, _ = (
            self.actor.sample(states)
        )

        q1_pi = self.critic1(
            states,
            new_actions,
        )

        q2_pi = self.critic2(
            states,
            new_actions,
        )

        min_q_pi = torch.minimum(
            q1_pi,
            q2_pi,
        )

        actor_loss = (
            self.alpha * log_prob
            - min_q_pi
        ).mean()

        self.actor_optimizer.zero_grad()

        actor_loss.backward()

        nn.utils.clip_grad_norm_(
            self.actor.parameters(),
            10.0,
        )

        self.actor_optimizer.step()

        # ----------------------------------------------------
        # STEP 6.4 — AUTOMATIC TEMPERATURE ADJUSTMENT
        #
        # log_alpha is optimized so policy entropy approaches
        # the target entropy.
        #
        # A larger alpha emphasizes exploration.
        # A smaller alpha emphasizes reward maximization.
        # ----------------------------------------------------

        alpha_loss = -(
            self.log_alpha
            * (
                log_prob.detach()
                + TARGET_ENTROPY
            )
        ).mean()

        self.alpha_optimizer.zero_grad()

        alpha_loss.backward()

        self.alpha_optimizer.step()

        # ----------------------------------------------------
        # STEP 6.5 — TARGET CRITIC UPDATES
        # ----------------------------------------------------

        soft_update(
            self.critic1,
            self.target1,
        )

        soft_update(
            self.critic2,
            self.target2,
        )

        self.training_steps += 1

        combined_critic_loss = (
            critic1_loss.item()
            + critic2_loss.item()
        ) / 2.0

        self.critic_losses.append(
            combined_critic_loss
        )

        self.actor_losses.append(
            float(actor_loss.item())
        )

        self.alpha_values.append(
            float(self.alpha.item())
        )

        return {
            "critic_loss": combined_critic_loss,
            "actor_loss": float(actor_loss.item()),
            "alpha": float(self.alpha.item()),
            "entropy": float(
                -log_prob.detach().mean().item()
            ),
        }


# ============================================================
# STEP 7 — TRAINING
# ============================================================

def train_agent(
    agent,
    episodes=TRAINING_EPISODES,
):

    env = RocketEnv()

    episode_rewards = []
    episode_successes = []
    episode_collisions = []

    total_steps = 0

    start_time = time.time()

    for episode in range(episodes):

        state = env.reset()

        episode_reward = 0.0

        for step in range(MAX_STEPS):

            # Random exploration during initial warmup.
            if total_steps < WARMUP_STEPS:

                action = np.random.uniform(
                    -1.0,
                    1.0,
                    size=ACTION_SIZE,
                ).astype(np.float32)

            else:

                action = agent.select_action(
                    state,
                    deterministic=False,
                )

            (
                next_state,
                reward,
                terminated,
                truncated,
            ) = env.step(action)

            # Only true MDP termination disables bootstrapping.
            # Time-limit truncation retains the final state.
            agent.replay.add(
                state,
                action,
                reward,
                next_state,
                terminated,
            )

            if total_steps >= WARMUP_STEPS:

                agent.train_step()

            state = next_state

            episode_reward += reward
            total_steps += 1

            if terminated or truncated:
                break

        episode_rewards.append(
            episode_reward
        )

        episode_successes.append(
            int(env.success)
        )

        episode_collisions.append(
            int(env.collision)
        )

        if (episode + 1) % 20 == 0:

            recent = min(
                20,
                len(episode_rewards),
            )

            print(
                f"Episode {episode + 1:3d}/{episodes} | "
                f"Reward={np.mean(episode_rewards[-recent:]):8.2f} | "
                f"Success={100*np.mean(episode_successes[-recent:]):5.1f}% | "
                f"Collision={100*np.mean(episode_collisions[-recent:]):5.1f}% | "
                f"Alpha={agent.alpha.item():.4f}"
            )

    elapsed = time.time() - start_time

    print(
        f"\nTraining completed in {elapsed:.1f} seconds."
    )

    return {
        "rewards": episode_rewards,
        "successes": episode_successes,
        "collisions": episode_collisions,
        "elapsed": elapsed,
    }


# ============================================================
# STEP 8 — TRAINING VISUALIZATION
# ============================================================

def moving_average(values, window=15):

    values = np.asarray(
        values,
        dtype=np.float32,
    )

    if len(values) < window:
        return values

    return np.convolve(
        values,
        np.ones(window) / window,
        mode="valid",
    )


def plot_training(agent, result):

    plt.figure(figsize=(11, 5))

    plt.plot(
        moving_average(
            result["rewards"]
        )
    )

    plt.title(
        "SAC — Episode Reward"
    )

    plt.xlabel("Episode")
    plt.ylabel("Smoothed reward")

    plt.grid(True)
    plt.tight_layout()

    plt.show()

    plt.figure(figsize=(11, 5))

    plt.plot(
        agent.critic_losses,
        label="Critic loss",
    )

    plt.plot(
        agent.actor_losses,
        label="Actor loss",
    )

    plt.title(
        "SAC — Training Loss"
    )

    plt.xlabel("Training update")
    plt.ylabel("Loss")

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.show()

    plt.figure(figsize=(11, 4))

    plt.plot(
        agent.alpha_values,
        label="Temperature alpha",
    )

    plt.title(
        "SAC — Automatic Entropy Temperature"
    )

    plt.xlabel("Training update")
    plt.ylabel("Alpha")

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.show()

    recent = min(
        25,
        len(result["rewards"]),
    )

    print("\nFINAL SAC PERFORMANCE")

    print(
        "Mean reward:",
        round(
            float(
                np.mean(
                    result["rewards"][-recent:]
                )
            ),
            3,
        ),
    )

    print(
        "Success rate:",
        round(
            100 * float(
                np.mean(
                    result["successes"][-recent:]
                )
            ),
            2,
        ),
        "%",
    )

    print(
        "Collision rate:",
        round(
            100 * float(
                np.mean(
                    result["collisions"][-recent:]
                )
            ),
            2,
        ),
        "%",
    )


# ============================================================
# STEP 9 — PYGAME ROCKET VISUALIZATION
# ============================================================

class RocketGUI:

    def __init__(
        self,
        env,
        agent,
    ):

        pygame.init()

        pygame.display.set_caption(
            "Phase 4.10 — SAC Continuous Rocket Control"
        )

        self.screen = pygame.display.set_mode(
            (
                WINDOW_W,
                WINDOW_H,
            )
        )

        self.clock = pygame.time.Clock()

        self.font = pygame.font.SysFont(
            "consolas",
            15,
        )

        self.title_font = pygame.font.SysFont(
            "consolas",
            20,
            bold=True,
        )

        self.env = env
        self.agent = agent

        self.paused = False
        self.running = True

        self.speed = 1

        self.last_action = np.zeros(
            ACTION_SIZE,
            dtype=np.float32,
        )

        self.reset()

    def reset(self):

        self.env.position = np.array(
            [80.0, 280.0],
            dtype=np.float32,
        )

        self.env.target = np.array(
            [620.0, 280.0],
            dtype=np.float32,
        )

        self.env.velocity = np.zeros(
            2,
            dtype=np.float32,
        )

        self.env.heading = 0.0

        self.env.steps = 0
        self.env.total_reward = 0.0

        self.env.success = False
        self.env.collision = False

        self.last_action[:] = 0.0

    def text(
        self,
        message,
        x,
        y,
        color=(230, 230, 230),
        font=None,
    ):

        if font is None:
            font = self.font

        surface = font.render(
            str(message),
            True,
            color,
        )

        self.screen.blit(
            surface,
            (x, y),
        )

    def rotate_point(
        self,
        local_x,
        local_y,
    ):

        x, y = self.env.position

        angle = self.env.heading

        return (
            int(
                x
                + local_x * math.cos(angle)
                - local_y * math.sin(angle)
            ),
            int(
                y
                + local_x * math.sin(angle)
                + local_y * math.cos(angle)
            ),
        )

    def draw_rocket(self):

        thrust = max(
            0.0,
            float(self.last_action[1]),
        )

        if thrust > 0.05:

            flame_length = int(
                8 + 24 * thrust
            )

            pygame.draw.polygon(
                self.screen,
                (255, 125, 25),
                [
                    self.rotate_point(-15, -4),
                    self.rotate_point(
                        -15 - flame_length,
                        0,
                    ),
                    self.rotate_point(-15, 4),
                ],
            )

        rocket_body = [
            (23, 0),
            (5, -9),
            (-11, -7),
            (-15, -4),
            (-15, 4),
            (-11, 7),
            (5, 9),
        ]

        pygame.draw.polygon(
            self.screen,
            (230, 240, 250),
            [
                self.rotate_point(x, y)
                for x, y in rocket_body
            ],
        )

        pygame.draw.polygon(
            self.screen,
            (70, 160, 230),
            [
                self.rotate_point(x, y)
                for x, y in rocket_body
            ],
            2,
        )

        pygame.draw.circle(
            self.screen,
            (40, 140, 245),
            self.rotate_point(8, 0),
            4,
        )

    def draw_world(self):

        self.screen.fill(
            (18, 22, 30)
        )

        pygame.draw.rect(
            self.screen,
            (28, 34, 44),
            (
                0,
                0,
                WORLD_W,
                WORLD_H,
            ),
        )

        for x in range(
            0,
            WORLD_W,
            40,
        ):

            pygame.draw.line(
                self.screen,
                (40, 47, 59),
                (x, 0),
                (x, WORLD_H),
            )

        for y in range(
            0,
            WORLD_H,
            40,
        ):

            pygame.draw.line(
                self.screen,
                (40, 47, 59),
                (0, y),
                (WORLD_W, y),
            )

        pygame.draw.circle(
            self.screen,
            (135, 75, 55),
            tuple(OBSTACLE.astype(int)),
            int(SAFE_DISTANCE),
            2,
        )

        pygame.draw.circle(
            self.screen,
            (165, 55, 55),
            tuple(OBSTACLE.astype(int)),
            int(OBSTACLE_RADIUS),
        )

        tx, ty = self.env.target.astype(int)

        pygame.draw.circle(
            self.screen,
            (70, 240, 130),
            (tx, ty),
            14,
            2,
        )

        pygame.draw.line(
            self.screen,
            (70, 240, 130),
            (tx - 18, ty),
            (tx + 18, ty),
            2,
        )

        pygame.draw.line(
            self.screen,
            (70, 240, 130),
            (tx, ty - 18),
            (tx, ty + 18),
            2,
        )

        self.draw_rocket()

    def draw_panel(self):

        pygame.draw.rect(
            self.screen,
            (22, 27, 37),
            (
                WORLD_W,
                0,
                WINDOW_W - WORLD_W,
                WINDOW_H,
            ),
        )

        self.text(
            "SOFT ACTOR-CRITIC",
            720,
            20,
            (120, 210, 255),
            self.title_font,
        )

        self.text(
            f"Position: {self.env.position.round(1)}",
            720,
            85,
        )

        self.text(
            f"Velocity: {self.env.velocity.round(2)}",
            720,
            115,
        )

        self.text(
            f"Heading: {math.degrees(self.env.heading):.1f} deg",
            720,
            145,
        )

        self.text(
            f"Distance: {distance(self.env.position, self.env.target):.1f}",
            720,
            175,
        )

        self.text(
            f"Steering: {self.last_action[0]:+.3f}",
            720,
            225,
        )

        self.text(
            f"Thrust: {self.last_action[1]:+.3f}",
            720,
            255,
        )

        self.text(
            f"Alpha: {self.agent.alpha.item():.4f}",
            720,
            300,
        )

        self.text(
            f"Steps: {self.env.steps}",
            720,
            330,
        )

        self.text(
            f"Reward: {self.env.total_reward:.2f}",
            720,
            360,
        )

        self.text(
            f"Simulation speed: {self.speed}",
            720,
            390,
        )

        self.text(
            "CONTINUOUS ACTION",
            720,
            445,
            (120, 210, 255),
        )

        self.text(
            "Steering: [-1, +1]",
            720,
            475,
        )

        self.text(
            "Thrust:   [-1, +1]",
            720,
            500,
        )

        self.text(
            "CLICK: TARGET",
            720,
            550,
        )

        self.text(
            "SPACE: PAUSE | R: RESET",
            720,
            575,
        )

        self.text(
            "ESC: EXIT | +/-: SPEED",
            720,
            600,
        )

    def step(self):

        state = self.env.get_state()

        # Deterministic evaluation:
        # use tanh(mean) rather than a random action sample.
        action = self.agent.select_action(
            state,
            deterministic=True,
        )

        self.last_action = action.copy()

        _, _, terminated, truncated = (
            self.env.step(action)
        )

        if terminated or truncated:
            self.paused = True

    def run(self):

        while self.running:

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
                        self.paused = False

                    elif event.key in (
                        pygame.K_PLUS,
                        pygame.K_EQUALS,
                        pygame.K_KP_PLUS,
                    ):

                        self.speed = min(
                            8,
                            self.speed + 1,
                        )

                    elif event.key in (
                        pygame.K_MINUS,
                        pygame.K_KP_MINUS,
                    ):

                        self.speed = max(
                            1,
                            self.speed - 1,
                        )

                elif event.type == pygame.MOUSEBUTTONDOWN:

                    if event.button == 1:

                        x, y = event.pos

                        if (
                            0 <= x <= WORLD_W
                            and 0 <= y <= WORLD_H
                        ):

                            if self.env.set_target(
                                [x, y]
                            ):

                                self.last_action[:] = 0.0

                                self.paused = False

            if not self.paused:

                for _ in range(self.speed):

                    self.step()

                    if self.paused:
                        break

            self.draw_world()
            self.draw_panel()

            if self.paused:

                self.text(
                    "PAUSED",
                    300,
                    25,
                    (255, 220, 80),
                    self.title_font,
                )

            pygame.display.flip()

            self.clock.tick(60)

        pygame.quit()


# ============================================================
# STEP 10 — UNIT TESTS
# ============================================================

def run_tests():

    print("\nPHASE 4.10 — UNIT TESTS")

    passed = 0

    def check(name, condition):

        nonlocal passed

        assert bool(condition), (
            f"FAILED: {name}"
        )

        passed += 1

        print(
            f"[PASS] {name}"
        )

    # Test 1: State dimension.
    env = RocketEnv()

    check(
        "State dimension",
        env.get_state().shape == (STATE_SIZE,),
    )

    # Test 2: Actor output dimensions.
    actor = GaussianActor().to(DEVICE)

    states = torch.randn(
        8,
        STATE_SIZE,
        device=DEVICE,
    )

    mean, log_std = actor(states)

    check(
        "Actor mean dimensions",
        tuple(mean.shape) == (8, ACTION_SIZE),
    )

    # Test 3: Log standard deviation limits.
    check(
        "Log standard deviation bounds",
        bool(
            torch.all(log_std >= LOG_STD_MIN)
            and torch.all(log_std <= LOG_STD_MAX)
        ),
    )

    # Test 4: Sample action range.
    actions, log_prob, mean_action = actor.sample(
        states
    )

    check(
        "Squashed action range",
        bool(
            torch.all(actions >= -1.0)
            and torch.all(actions <= 1.0)
        ),
    )

    # Test 5: Log probability dimensions.
    check(
        "Log probability dimensions",
        tuple(log_prob.shape) == (8, 1),
    )

    # Test 6: Mean action range.
    check(
        "Mean action range",
        bool(
            torch.all(mean_action >= -1.0)
            and torch.all(mean_action <= 1.0)
        ),
    )

    # Test 7: Reparameterized sample supports gradients.
    test_state = torch.randn(
        4,
        STATE_SIZE,
        device=DEVICE,
    )

    test_action, test_log_prob, _ = actor.sample(
        test_state
    )

    gradient = torch.autograd.grad(
        (
            test_action.sum()
            + test_log_prob.sum()
        ),
        list(actor.parameters()),
        allow_unused=True,
    )

    has_gradient = any(
        g is not None
        and torch.isfinite(g).all()
        and torch.any(g != 0)
        for g in gradient
    )

    check(
        "Reparameterized gradient",
        has_gradient,
    )

    # Test 8: Critic output dimensions.
    critic = Critic().to(DEVICE)

    q = critic(
        states,
        actions,
    )

    check(
        "Critic output dimensions",
        tuple(q.shape) == (8, 1),
    )

    # Test 9: Twin critics can be independent.
    critic2 = Critic().to(DEVICE)

    different = any(
        not torch.equal(a, b)
        for a, b in zip(
            critic.parameters(),
            critic2.parameters(),
        )
    )

    check(
        "Independent critics",
        different,
    )

    # Test 10: Replay insertion.
    replay = ReplayBuffer(100)

    replay.add(
        np.zeros(STATE_SIZE, dtype=np.float32),
        np.zeros(ACTION_SIZE, dtype=np.float32),
        1.0,
        np.ones(STATE_SIZE, dtype=np.float32),
        False,
    )

    check(
        "Replay insertion",
        len(replay) == 1,
    )

    # Test 11: Replay batch dimensions.
    for _ in range(BATCH_SIZE):

        replay.add(
            np.zeros(STATE_SIZE, dtype=np.float32),
            np.zeros(ACTION_SIZE, dtype=np.float32),
            1.0,
            np.ones(STATE_SIZE, dtype=np.float32),
            False,
        )

    batch = replay.sample(
        BATCH_SIZE
    )

    check(
        "Replay state batch",
        tuple(batch[0].shape)
        == (BATCH_SIZE, STATE_SIZE),
    )

    check(
        "Replay action batch",
        tuple(batch[1].shape)
        == (BATCH_SIZE, ACTION_SIZE),
    )

    # Test 12: Soft update.
    source = nn.Linear(2, 2)
    target = nn.Linear(2, 2)

    with torch.no_grad():

        for p in source.parameters():
            p.fill_(1.0)

        for p in target.parameters():
            p.fill_(0.0)

    soft_update(
        source,
        target,
        tau=0.1,
    )

    check(
        "Soft target update",
        torch.allclose(
            target.weight,
            torch.full_like(
                target.weight,
                0.1,
            ),
        ),
    )

    # Test 13: Alpha positivity.
    agent = SACAgent()

    check(
        "Positive temperature",
        agent.alpha.item() > 0,
    )

    # Test 14: Environment accepts continuous action.
    transition = env.step(
        np.array(
            [0.25, 0.5],
            dtype=np.float32,
        )
    )

    check(
        "Continuous environment action",
        len(transition) == 4
        and transition[0].shape == (STATE_SIZE,),
    )

    # Test 15: Environment reward is finite.
    check(
        "Finite environment reward",
        np.isfinite(transition[1]),
    )

    # Test 16: Deterministic action selection.
    deterministic_action = agent.select_action(
        env.get_state(),
        deterministic=True,
    )

    check(
        "Deterministic action dimensions and range",
        deterministic_action.shape == (ACTION_SIZE,)
        and np.all(
            np.abs(deterministic_action) <= 1.0
        ),
    )

    # Test 17: Stochastic action selection.
    stochastic_action = agent.select_action(
        env.get_state(),
        deterministic=False,
    )

    check(
        "Stochastic action dimensions and range",
        stochastic_action.shape == (ACTION_SIZE,)
        and np.all(
            np.abs(stochastic_action) <= 1.0
        ),
    )

    print(
        f"\n{passed}/17 tests passed."
    )


# ============================================================
# STEP 11 — MAIN
# ============================================================

def main():

    print("=" * 75)
    print("PHASE 4.10 — SOFT ACTOR-CRITIC")
    print("=" * 75)

    print(
        "Device:",
        DEVICE,
    )

    print(
        "Action space:",
        "continuous steering + continuous thrust",
    )

    run_tests()

    # --------------------------------------------------------
    # Train SAC.
    # --------------------------------------------------------

    print("\nTraining SAC...")

    agent = SACAgent()

    result = train_agent(
        agent,
        TRAINING_EPISODES,
    )

    # --------------------------------------------------------
    # Visualize training results.
    # --------------------------------------------------------

    plot_training(
        agent,
        result,
    )

    # --------------------------------------------------------
    # Save trained actor.
    # --------------------------------------------------------

    torch.save(
        agent.actor.state_dict(),
        "sac_actor.pth",
    )

    print(
        "\nSaved model: sac_actor.pth"
    )

    # --------------------------------------------------------
    # Launch interactive Pygame simulation.
    # --------------------------------------------------------

    print(
        "\nLaunching SAC rocket simulator..."
    )

    gui_env = RocketEnv()

    gui = RocketGUI(
        gui_env,
        agent,
    )

    gui.run()


if __name__ == "__main__":
    main()
