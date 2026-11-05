
"""
File: phase5/actor_critic_td0_drone.py

PHASE 5.3 — ONE-STEP ACTOR-CRITIC WITH TD(0)
============================================================

PURPOSE
-------
Train a continuous-control quadcopter using online
Actor-Critic learning and one-step temporal-difference
advantage estimation.

WHAT YOU LEARN
--------------

STEP 1 — Actor
    Represent a stochastic continuous-action policy.

STEP 2 — Critic
    Estimate the expected future return V(s).

STEP 3 — TD(0)
    Calculate the one-step temporal-difference error.

STEP 4 — Online Critic Learning
    Update the value function after every transition.

STEP 5 — Online Actor Learning
    Update the policy using the TD error.

STEP 6 — Terminal Handling
    Do not bootstrap from a true terminal state.
    Bootstrap through time-limit truncation.

STEP 7 — Evaluation
    Measure reward, tracking distance and collisions.

STEP 8 — Interactive Robotics
    Track a moving target with a quadcopter visualization.

MATHEMATICS
-----------

TD target:

    y_t = r_t + gamma * V(s_{t+1})

TD error:

    delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)

For a true terminal transition:

    delta_t = r_t - V(s_t)

Critic loss:

    L_critic = 0.5 * delta_t^2

Actor loss:

    L_actor = -log(pi(a_t|s_t)) * delta_t

The actor uses a detached TD error. This prevents actor
backpropagation from modifying critic parameters.

IMPORTANT DISTINCTION
--------------------

terminated:
    The MDP reached a true terminal state.
    Do not bootstrap.

truncated:
    The episode ended because of an external time limit.
    Bootstrap from the final observation.

This implementation uses one-step on-policy Actor-Critic.
It is not yet A2C or PPO.

ROBOTICS APPLICATION
--------------------

- Moving-target tracking
- Continuous drone control
- Online value estimation
- Low-latency policy adaptation
- Foundation for A2C and PPO

Educational 2D simulation only. Not flight-certified.
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

SEED = 535

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

ACTOR_LR = 1e-4
CRITIC_LR = 5e-4

TRAIN_EPISODES = 300
MAX_EPISODE_STEPS = 180

PRINT_INTERVAL = 10

# Quick experiment:
#
# TRAIN_EPISODES = 20
# MAX_EPISODE_STEPS = 100


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
# 2. DRONE PHYSICS
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

    def reset(self, target_center=None):

        self.state = np.array(
            [
                105.0,
                500.0,
                0.0,
                0.0,
            ],
            dtype=np.float32,
        )

        if target_center is not None:

            self.target.center = np.asarray(
                target_center,
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
# 4. STOCHASTIC ACTOR
# ============================================================

class GaussianActor(nn.Module):

    def __init__(self):

        super().__init__()

        self.body = nn.Sequential(
            nn.Linear(OBS_DIM, 128),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.ReLU(),
        )

        self.mean_layer = nn.Linear(
            128,
            ACTION_DIM,
        )

        self.log_std_layer = nn.Linear(
            128,
            ACTION_DIM,
        )

    def forward(self, state):

        hidden = self.body(
            state
        )

        mean = self.mean_layer(
            hidden
        )

        log_std = self.log_std_layer(
            hidden
        )

        log_std = torch.clamp(
            log_std,
            -5.0,
            1.0,
        )

        return mean, log_std

    def sample(self, state):

        mean, log_std = self(
            state
        )

        distribution = torch.distributions.Normal(
            mean,
            log_std.exp(),
        )

        # REINFORCE-style score-function sampling.
        raw_action = distribution.sample()

        action = torch.tanh(
            raw_action
        )

        log_probability = distribution.log_prob(
            raw_action
        )

        # Tanh change-of-variables correction.
        log_probability -= torch.log(
            1.0 - action.pow(2) + 1e-6
        )

        log_probability = log_probability.sum(
            dim=-1
        )

        return (
            action,
            raw_action,
            log_probability,
        )

    def evaluate_raw_action(
        self,
        state,
        raw_action,
    ):

        mean, log_std = self(
            state
        )

        distribution = torch.distributions.Normal(
            mean,
            log_std.exp(),
        )

        action = torch.tanh(
            raw_action
        )

        log_probability = distribution.log_prob(
            raw_action
        )

        log_probability -= torch.log(
            1.0 - action.pow(2) + 1e-6
        )

        return log_probability.sum(
            dim=-1
        )

    def deterministic_action(self, state):

        mean, _ = self(
            state
        )

        return torch.tanh(
            mean
        )


# ============================================================
# 5. STATE-VALUE CRITIC
# ============================================================

class ValueCritic(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(OBS_DIM, 128),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.ReLU(),

            nn.Linear(128, 1),
        )

    def forward(self, state):

        return self.network(
            state
        ).squeeze(-1)


# ============================================================
# 6. TD(0) MATHEMATICAL FUNCTION
# ============================================================

def td_target(
    reward,
    next_value,
    gamma,
    terminated,
):

    """
    One-step TD target.

    Nonterminal:
        target = reward + gamma * V(next_state)

    Terminal:
        target = reward

    Notice that truncation is not automatically terminal.
    If the environment reaches an external time limit,
    the final observation can still be bootstrapped.
    """

    if terminated:

        return reward

    return (
        reward + gamma * next_value
    )


def calculate_td_error(
    reward,
    value,
    next_value,
    gamma,
    terminated,
):

    target = td_target(
        reward,
        next_value,
        gamma,
        terminated,
    )

    return target - value


# ============================================================
# 7. ONLINE ACTOR-CRITIC AGENT
# ============================================================

class ActorCriticAgent:

    def __init__(self):

        self.actor = GaussianActor().to(
            DEVICE
        )

        self.critic = ValueCritic().to(
            DEVICE
        )

        self.actor_optimizer = optim.Adam(
            self.actor.parameters(),
            lr=ACTOR_LR,
        )

        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=CRITIC_LR,
        )

    def select_action(
        self,
        observation,
        deterministic=False,
    ):

        state = torch.as_tensor(
            observation,
            dtype=torch.float32,
            device=DEVICE,
        ).unsqueeze(0)

        with torch.no_grad():

            if deterministic:

                action = (
                    self.actor.deterministic_action(
                        state
                    )
                )

            else:

                action, _, _ = (
                    self.actor.sample(
                        state
                    )
                )

        return action.squeeze(0).cpu().numpy()

    def update(
        self,
        observation,
        raw_action,
        reward,
        next_observation,
        terminated,
    ):

        state = torch.as_tensor(
            observation,
            dtype=torch.float32,
            device=DEVICE,
        ).unsqueeze(0)

        next_state = torch.as_tensor(
            next_observation,
            dtype=torch.float32,
            device=DEVICE,
        ).unsqueeze(0)

        raw_action_tensor = torch.as_tensor(
            raw_action,
            dtype=torch.float32,
            device=DEVICE,
        ).unsqueeze(0)

        reward_tensor = torch.tensor(
            [reward],
            dtype=torch.float32,
            device=DEVICE,
        )

        # ----------------------------------------------------
        # 1. Calculate TD target using the current critic.
        # ----------------------------------------------------

        with torch.no_grad():

            value_next = self.critic(
                next_state
            )

            value_current_for_target = self.critic(
                state
            )

            target = (
                reward_tensor
                + GAMMA
                * value_next
                * (0.0 if terminated else 1.0)
            )

            td_error = (
                target
                - value_current_for_target
            )

        # ----------------------------------------------------
        # 2. Critic update.
        #
        # The TD target is detached.
        # The critic learns to predict this target.
        # ----------------------------------------------------

        current_value = self.critic(
            state
        )

        critic_loss = 0.5 * (
            current_value - target
        ).pow(2).mean()

        self.critic_optimizer.zero_grad()

        critic_loss.backward()

        nn.utils.clip_grad_norm_(
            self.critic.parameters(),
            5.0,
        )

        self.critic_optimizer.step()

        # ----------------------------------------------------
        # 3. Actor update.
        #
        # Use the TD error computed before the critic update.
        # Detach it to prevent critic gradients from entering
        # the actor loss.
        # ----------------------------------------------------

        log_probability = (
            self.actor.evaluate_raw_action(
                state,
                raw_action_tensor,
            )
        )

        actor_loss = -(
            log_probability
            * td_error.detach()
        ).mean()

        self.actor_optimizer.zero_grad()

        actor_loss.backward()

        nn.utils.clip_grad_norm_(
            self.actor.parameters(),
            5.0,
        )

        self.actor_optimizer.step()

        return {
            "td_error": float(
                td_error.item()
            ),

            "value": float(
                current_value.detach().item()
            ),

            "next_value": float(
                value_next.item()
            ),

            "target": float(
                target.item()
            ),

            "actor_loss": float(
                actor_loss.detach().item()
            ),

            "critic_loss": float(
                critic_loss.detach().item()
            ),
        }


# ============================================================
# 8. TRAINING
# ============================================================

def train():

    env = DroneEnv()

    agent = ActorCriticAgent()

    history = {
        "episode_return": [],
        "distance": [],
        "actor_loss": [],
        "critic_loss": [],
        "td_error": [],
        "collisions": [],
    }

    for episode in range(
        TRAIN_EPISODES
    ):

        observation = env.reset()

        total_reward = 0.0
        distances = []

        actor_losses = []
        critic_losses = []
        td_errors = []

        collision = False

        for _ in range(
            MAX_EPISODE_STEPS
        ):

            state_tensor = torch.as_tensor(
                observation,
                dtype=torch.float32,
                device=DEVICE,
            ).unsqueeze(0)

            with torch.no_grad():

                action, raw_action, _ = (
                    agent.actor.sample(
                        state_tensor
                    )
                )

            action_np = (
                action.squeeze(0)
                .cpu()
                .numpy()
            )

            raw_action_np = (
                raw_action.squeeze(0)
                .cpu()
                .numpy()
            )

            (
                next_observation,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(
                action_np
            )

            # TD(0) update occurs after every transition.
            #
            # True terminal:
            #     no bootstrap.
            #
            # Time-limit truncation:
            #     bootstrap from final next_observation.

            metrics = agent.update(
                observation=observation,
                raw_action=raw_action_np,
                reward=reward,
                next_observation=next_observation,
                terminated=terminated,
            )

            total_reward += reward

            distances.append(
                info["distance"]
            )

            actor_losses.append(
                metrics["actor_loss"]
            )

            critic_losses.append(
                metrics["critic_loss"]
            )

            td_errors.append(
                metrics["td_error"]
            )

            collision = (
                collision or info["collision"]
            )

            observation = next_observation

            if terminated or truncated:

                break

        history["episode_return"].append(
            total_reward
        )

        history["distance"].append(
            float(np.mean(distances))
        )

        history["actor_loss"].append(
            float(np.mean(actor_losses))
        )

        history["critic_loss"].append(
            float(np.mean(critic_losses))
        )

        history["td_error"].append(
            float(np.mean(np.abs(td_errors)))
        )

        history["collisions"].append(
            float(collision)
        )

        if (
            episode + 1
        ) % PRINT_INTERVAL == 0:

            recent = slice(
                -PRINT_INTERVAL,
                None,
            )

            print(
                f"Episode {episode + 1:4d}/{TRAIN_EPISODES}"
                f" | Return={np.mean(history['episode_return'][recent]):9.2f}"
                f" | Distance={np.mean(history['distance'][recent]):8.2f}"
                f" | TD error={np.mean(history['td_error'][recent]):8.3f}"
                f" | Critic loss={np.mean(history['critic_loss'][recent]):8.3f}"
            )

    return agent, history


# ============================================================
# 9. DETERMINISTIC EVALUATION
# ============================================================

def evaluate(
    agent,
    episodes=5,
):

    env = DroneEnv()

    results = []

    for _ in range(episodes):

        observation = env.reset()

        total_reward = 0.0
        distances = []

        collision = False

        for _ in range(
            MAX_EPISODE_STEPS
        ):

            action = agent.select_action(
                observation,
                deterministic=True,
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
            {
                "return": total_reward,
                "distance": np.mean(distances),
                "collision": float(collision),
            }
        )

    print("\nEvaluation")

    print(
        "Mean return:",
        round(
            np.mean(
                [r["return"] for r in results]
            ),
            3,
        ),
    )

    print(
        "Mean target distance:",
        round(
            np.mean(
                [r["distance"] for r in results]
            ),
            3,
        ),
    )

    print(
        "Collision rate:",
        round(
            np.mean(
                [r["collision"] for r in results]
            ),
            3,
        ),
    )

    return results


# ============================================================
# 10. TRAINING VISUALIZATION
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


def plot_training(history):

    plots = [
        (
            "episode_return",
            "Episode Return",
            "Return",
        ),

        (
            "distance",
            "Moving Target Tracking Distance",
            "Distance",
        ),

        (
            "actor_loss",
            "Actor Policy Loss",
            "Loss",
        ),

        (
            "critic_loss",
            "Critic TD(0) Loss",
            "Loss",
        ),

        (
            "td_error",
            "Mean Absolute TD Error",
            "Absolute TD Error",
        ),

        (
            "collisions",
            "Collision History",
            "Collision",
        ),
    ]

    for key, title, ylabel in plots:

        plt.figure(
            figsize=(10, 5)
        )

        values = history[key]

        plt.plot(
            values,
            alpha=0.35,
            label=key,
        )

        smooth = moving_average(
            values
        )

        if len(smooth) > 0:

            offset = min(
                14,
                len(values) - len(smooth),
            )

            plt.plot(
                np.arange(len(smooth)) + offset,
                smooth,
                label="Moving average",
            )

        plt.title(title)

        plt.xlabel("Episode")

        plt.ylabel(ylabel)

        plt.grid(True)

        plt.legend()

        plt.tight_layout()

        plt.show()


# ============================================================
# 11. INTERACTIVE PYGAME DRONE VISUALIZATION
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
        "Phase 5.3 — TD(0) Actor-Critic Drone"
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

    def draw_text(
        message,
        x,
        y,
        color=(235, 235, 235),
    ):

        surface = font.render(
            message,
            True,
            color,
        )

        screen.blit(
            surface,
            (x, y),
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

        # Quadcopter arms and propellers.

        for forward in (-1, 1):

            for lateral in (-1, 1):

                rotor = (
                    center
                    + direction * forward * 17
                    + side * lateral * 17
                )

                pygame.draw.line(
                    screen,
                    (225, 230, 240),
                    center.astype(int),
                    rotor.astype(int),
                    5,
                )

                pygame.draw.circle(
                    screen,
                    (95, 195, 245),
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

                action = agent.select_action(
                    observation,
                    deterministic=True,
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

        # Grid.

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

        # Obstacle and safety boundary.

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
            "PHASE 5.3 — ONE-STEP ACTOR-CRITIC",
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
                (255, 215, 80),
            )

        pygame.display.flip()

    pygame.quit()


# ============================================================
# 12. NUMERICAL VERIFICATION
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
    # r=1, gamma=0.9, V(next)=2, nonterminal.
    #
    # TD target = 1 + 0.9*2 = 2.8

    check(
        np.isclose(
            td_target(
                1.0,
                2.0,
                0.9,
                False,
            ),
            2.8,
        ),
        "Nonterminal TD target",
    )

    # Test 2:
    # Terminal target must not bootstrap.

    check(
        np.isclose(
            td_target(
                1.0,
                100.0,
                0.9,
                True,
            ),
            1.0,
        ),
        "Terminal TD target does not bootstrap",
    )

    # Test 3:
    # delta = 2.8 - 2.0 = 0.8

    check(
        np.isclose(
            calculate_td_error(
                1.0,
                2.0,
                2.0,
                0.9,
                False,
            ),
            0.8,
        ),
        "One-step TD error",
    )

    # Test 4:
    # Terminal delta = reward - value = 1 - 2 = -1

    check(
        np.isclose(
            calculate_td_error(
                1.0,
                2.0,
                100.0,
                0.9,
                True,
            ),
            -1.0,
        ),
        "Terminal TD error",
    )

    # Test 5:
    # Truncation is not passed as terminal.
    # Thus bootstrap is retained.

    check(
        np.isclose(
            td_target(
                1.0,
                2.0,
                0.9,
                False,
            ),
            2.8,
        ),
        "Truncated transition retains bootstrap",
    )

    # Test 6:
    # Zero gamma means no bootstrap.

    check(
        np.isclose(
            td_target(
                3.0,
                100.0,
                0.0,
                False,
            ),
            3.0,
        ),
        "Zero discount removes bootstrap",
    )

    # Test 7:
    # Zero next value.

    check(
        np.isclose(
            td_target(
                3.0,
                0.0,
                0.9,
                False,
            ),
            3.0,
        ),
        "Zero next-state value",
    )

    # Test 8:
    # Negative reward.

    check(
        np.isclose(
            td_target(
                -2.0,
                5.0,
                0.5,
                False,
            ),
            0.5,
        ),
        "Negative reward TD target",
    )

    # Test 9:
    # Observation dimensions.

    env = DroneEnv()

    observation = env.reset()

    check(
        observation.shape == (OBS_DIM,),
        "Observation dimension",
    )

    # Test 10:
    # Finite observation.

    check(
        np.all(
            np.isfinite(observation)
        ),
        "Finite observation",
    )

    # Test 11:
    # Actor output shape.

    actor = GaussianActor().to(
        DEVICE
    )

    state = torch.as_tensor(
        observation,
        dtype=torch.float32,
        device=DEVICE,
    ).unsqueeze(0)

    action, raw_action, log_probability = (
        actor.sample(
            state
        )
    )

    check(
        action.shape == (1, ACTION_DIM),
        "Actor action shape",
    )

    # Test 12:
    # Bounded action.

    check(
        bool(
            torch.all(
                action.abs() <= 1.0
            )
        ),
        "Actor action bounded by tanh",
    )

    # Test 13:
    # Finite log probability.

    check(
        bool(
            torch.isfinite(
                log_probability
            ).all()
        ),
        "Finite policy log probability",
    )

    # Test 14:
    # Critic output.

    critic = ValueCritic().to(
        DEVICE
    )

    value = critic(
        state
    )

    check(
        value.shape == (1,),
        "Critic output shape",
    )

    # Test 15:
    # Critic prediction finite.

    check(
        bool(
            torch.isfinite(value).all()
        ),
        "Finite critic prediction",
    )

    # Test 16:
    # Actor gradient.

    actor_optimizer = optim.Adam(
        actor.parameters(),
        lr=1e-4,
    )

    actor_optimizer.zero_grad()

    actor_loss = -log_probability.mean()

    actor_loss.backward()

    actor_has_gradient = any(
        p.grad is not None
        and bool(
            torch.isfinite(p.grad).all()
        )
        for p in actor.parameters()
    )

    check(
        actor_has_gradient,
        "Actor receives finite gradients",
    )

    # Test 17:
    # Critic gradient.

    critic_optimizer = optim.Adam(
        critic.parameters(),
        lr=1e-4,
    )

    critic_optimizer.zero_grad()

    critic_loss = (
        value - torch.tensor(
            [5.0],
            device=DEVICE,
        )
    ).pow(2).mean()

    critic_loss.backward()

    critic_has_gradient = any(
        p.grad is not None
        and bool(
            torch.isfinite(p.grad).all()
        )
        for p in critic.parameters()
    )

    check(
        critic_has_gradient,
        "Critic receives finite gradients",
    )

    # Test 18:
    # Environment step structure.

    result = env.step(
        np.zeros(
            ACTION_DIM
        )
    )

    check(
        len(result) == 5,
        "Environment returns five values",
    )

    # Test 19:
    # Finite reward.

    check(
        np.isfinite(result[1]),
        "Environment reward finite",
    )

    # Test 20:
    # Moving target velocity.

    check(
        env.target.velocity().shape == (2,),
        "Moving target velocity dimension",
    )

    # Test 21:
    # Drone speed limit.

    test_state = np.array(
        [
            200.0,
            200.0,
            1000.0,
            1000.0,
        ],
        dtype=np.float32,
    )

    clipped = clip_physical_state(
        test_state
    )

    check(
        np.linalg.norm(clipped[2:4])
        <= MAX_SPEED + 1e-4,
        "Drone speed limit",
    )

    # Test 22:
    # World boundaries.

    check(
        (
            DRONE_RADIUS <= clipped[0]
            <= WORLD_W - DRONE_RADIUS
        ),
        "Drone horizontal boundary constraint",
    )

    # Test 23:
    # Actor deterministic output.

    deterministic_action = (
        actor.deterministic_action(
            state
        )
    )

    check(
        deterministic_action.shape
        == (1, ACTION_DIM),
        "Deterministic actor action",
    )

    # Test 24:
    # TD error is finite for normal inputs.

    check(
        np.isfinite(
            calculate_td_error(
                1.0,
                2.0,
                3.0,
                0.99,
                False,
            )
        ),
        "Finite TD error",
    )

    print(
        f"\n{passed}/24 tests passed."
    )

    return passed == 24


# ============================================================
# 13. MAIN
# ============================================================

def main():

    print("=" * 70)

    print(
        "PHASE 5.3 — ONE-STEP ACTOR-CRITIC WITH TD(0)"
    )

    print("=" * 70)

    print(
        f"Device: {DEVICE}"
    )

    if not run_tests():

        raise RuntimeError(
            "Numerical verification failed."
        )

    print(
        "\n[1/4] Training Actor-Critic..."
    )

    agent, history = train()

    print(
        "\n[2/4] Evaluating policy..."
    )

    evaluate(
        agent,
        episodes=5,
    )

    print(
        "\n[3/4] Plotting training results..."
    )

    plot_training(
        history
    )

    print(
        "\n[4/4] Starting interactive drone..."
    )

    run_pygame(
        agent
    )

    print(
        "\nPhase 5.3 completed."
    )


if __name__ == "__main__":

    main()
