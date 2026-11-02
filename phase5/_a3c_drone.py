
"""
File: phase5/a3c_drone.py

PHASE 5.9 — ASYNCHRONOUS ADVANTAGE ACTOR-CRITIC (A3C)
=====================================================

PURPOSE
-------
Implement A3C using multiple Python worker threads and PyTorch.

WHAT YOU LEARN
--------------

STEP 1 — Global Actor-Critic
    A shared actor and critic network.

STEP 2 — Local Workers
    Each worker owns:
        - An independent drone environment.
        - A local copy of the global network.
        - Its own trajectory buffer.

STEP 3 — N-STEP RETURNS
    Calculate bootstrapped returns over a short horizon.

STEP 4 — ADVANTAGE ACTOR-CRITIC
    Calculate:
        advantage = return - V(state)

STEP 5 — ASYNCHRONOUS UPDATES
    Workers collect data independently.
    Workers calculate local gradients.
    Gradients are applied to shared global parameters.

STEP 6 — ENTROPY REGULARIZATION
    Encourage exploration through policy entropy.

STEP 7 — ROBOTICS APPLICATION
    Multiple simulated quadcopters independently learn
    the task of tracking moving targets.

MATHEMATICAL FORMULATION
------------------------

N-STEP RETURN:

    R_t = r_t
        + gamma*r_(t+1)
        + ...
        + gamma^(n-1)*r_(t+n-1)
        + gamma^n*V(s_(t+n))

ADVANTAGE:

    A_t = R_t - V(s_t)

ACTOR LOSS:

    L_actor = -log(pi(a|s)) * A_t

CRITIC LOSS:

    L_critic = 0.5 * (R_t - V(s_t))^2

ENTROPY:

    H(pi) = -sum_a pi(a|s) log(pi(a|s))

TOTAL LOSS:

    L = L_actor + c_v*L_critic - beta*H(pi)

ASYNC UPDATE
------------

    1. Copy global parameters to local network.
    2. Collect local experience.
    3. Calculate local gradients.
    4. Apply gradients to global network.
    5. Synchronize local network again.

IMPLEMENTATION NOTE
-------------------

This educational implementation uses Python threads.

The global optimizer update is protected by a lock so that
Adam's internal state is not corrupted by simultaneous writes.

Workers still collect trajectories independently and may
compute gradients using slightly stale local parameters.

This is an asynchronous actor-learner design, but it is not
a lock-free multiprocessing implementation of the original
A3C paper.

The Gaussian policy uses a tanh-squashed action. The log
probability includes the tanh Jacobian correction.

True terminal transitions suppress bootstrapping.
Time-limit truncations bootstrap from their final state.

The simulator is a simplified 2D quadcopter model, not
a flight-certified physical model.
"""

from __future__ import annotations

import math
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# ============================================================
# 0. CONFIGURATION
# ============================================================

SEED = 590

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device(
    "cpu"
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

N_STEP = 8

ENTROPY_COEFFICIENT = 0.01
VALUE_COEFFICIENT = 0.5

LEARNING_RATE = 3e-4

GRADIENT_CLIP = 0.5

NUM_WORKERS = 4

MAX_EPISODES_PER_WORKER = 150

MAX_EPISODE_STEPS = 180

PRINT_INTERVAL = 10

# Training is intentionally configurable.
# Start with fewer episodes for a quick smoke test.


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

        self.phase += (
            TARGET_OMEGA * DT
        )


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

    def __init__(self, seed=None):

        self.rng = np.random.default_rng(
            seed
        )

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
            np.asarray(
                action,
                dtype=np.float32,
            ),
            -1.0,
            1.0,
        )

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
# 4. GLOBAL ACTOR-CRITIC NETWORK
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

        self.critic = nn.Linear(
            128,
            1,
        )

        self.log_std = nn.Parameter(
            torch.full(
                (ACTION_DIM,),
                -0.5,
            )
        )

    def forward(self, states):

        features = self.shared(
            states
        )

        mean = self.actor_mean(
            features
        )

        value = self.critic(
            features
        ).squeeze(-1)

        log_std = torch.clamp(
            self.log_std,
            -5.0,
            1.0,
        )

        return (
            mean,
            log_std,
            value,
        )

    def get_distribution(self, states):

        mean, log_std, value = self(
            states
        )

        distribution = torch.distributions.Normal(
            mean,
            log_std.exp(),
        )

        return (
            distribution,
            value,
        )

    def sample_action(self, states):

        distribution, value = (
            self.get_distribution(
                states
            )
        )

        raw_action = distribution.sample()

        action = torch.tanh(
            raw_action
        )

        # Log probability of the transformed action:
        #
        # log pi(a|s) =
        # log N(u; mu, sigma)
        # - sum(log(1 - tanh(u)^2))

        correction = torch.log(
            1.0 - action.pow(2) + 1e-6
        )

        log_prob = (
            distribution.log_prob(
                raw_action
            )
            - correction
        ).sum(
            dim=-1
        )

        # The base Gaussian entropy is used as an exploration
        # regularizer. It is not the exact entropy of the
        # transformed tanh distribution.

        entropy = distribution.entropy().sum(
            dim=-1
        )

        return (
            action,
            log_prob,
            entropy,
            value,
        )

    def deterministic_action(self, states):

        mean, _, _ = self(states)

        return torch.tanh(
            mean
        )


# ============================================================
# 5. N-STEP RETURN CALCULATION
# ============================================================

def calculate_n_step_returns(
    rewards,
    bootstrap_value,
    terminated,
    truncated,
    gamma=GAMMA,
):

    """
    Rewards are ordered from earliest to latest.

    A true terminal suppresses bootstrapping.

    A truncation bootstraps from the final observation,
    but no rewards beyond the truncation boundary are
    included in this local rollout.
    """

    if terminated:

        return_value = 0.0

    else:

        return_value = float(
            bootstrap_value
        )

    for reward in reversed(
        rewards
    ):

        return_value = (
            float(reward)
            + gamma * return_value
        )

    return return_value


# ============================================================
# 6. LOCAL WORKER
# ============================================================

class A3CWorker:

    def __init__(
        self,
        worker_id,
        global_model,
        global_optimizer,
        update_lock,
        global_episode_counter,
        counter_lock,
        history,
        history_lock,
    ):

        self.worker_id = worker_id

        self.global_model = global_model

        self.global_optimizer = global_optimizer

        self.update_lock = update_lock

        self.global_episode_counter = (
            global_episode_counter
        )

        self.counter_lock = counter_lock

        self.history = history

        self.history_lock = history_lock

        self.local_model = ActorCritic().to(
            DEVICE
        )

        self.env = DroneEnv(
            seed=SEED + 1000 + worker_id
        )

        self.rng = np.random.default_rng(
            SEED + 2000 + worker_id
        )

    def synchronize(self):

        self.local_model.load_state_dict(
            self.global_model.state_dict()
        )

    def apply_local_gradients(self):

        # The local network has already calculated gradients.
        # We transfer those gradients to the global parameters.

        with self.update_lock:

            self.global_optimizer.zero_grad(
                set_to_none=True
            )

            for global_parameter, local_parameter in zip(
                self.global_model.parameters(),
                self.local_model.parameters(),
            ):

                if local_parameter.grad is None:

                    global_parameter.grad = None

                else:

                    global_parameter.grad = (
                        local_parameter.grad.detach().clone()
                    )

            nn.utils.clip_grad_norm_(
                self.global_model.parameters(),
                GRADIENT_CLIP,
            )

            self.global_optimizer.step()

    def update_from_rollout(
        self,
        observations,
        actions,
        rewards,
        bootstrap_value,
        terminated,
    ):

        if len(rewards) == 0:

            return 0.0

        # Calculate n-step targets backwards.

        targets = []

        return_value = (
            0.0
            if terminated
            else float(bootstrap_value)
        )

        for reward in reversed(rewards):

            return_value = (
                float(reward)
                + GAMMA * return_value
            )

            targets.append(
                return_value
            )

        targets.reverse()

        states_tensor = torch.as_tensor(
            np.asarray(
                observations,
                dtype=np.float32,
            ),
            dtype=torch.float32,
            device=DEVICE,
        )

        actions_tensor = torch.as_tensor(
            np.asarray(
                actions,
                dtype=np.float32,
            ),
            dtype=torch.float32,
            device=DEVICE,
        )

        targets_tensor = torch.as_tensor(
            targets,
            dtype=torch.float32,
            device=DEVICE,
        )

        distribution, values = (
            self.local_model.get_distribution(
                states_tensor
            )
        )

        # Inverse tanh maps the stored bounded action back
        # to the Gaussian pre-squash action.
        #
        # At saturation, clipping avoids infinite atanh.

        safe_actions = torch.clamp(
            actions_tensor,
            -0.999999,
            0.999999,
        )

        raw_actions = torch.atanh(
            safe_actions
        )

        correction = torch.log(
            1.0 - safe_actions.pow(2) + 1e-6
        )

        log_probs = (
            distribution.log_prob(
                raw_actions
            )
            - correction
        ).sum(
            dim=-1
        )

        advantages = (
            targets_tensor - values
        )

        actor_loss = -(
            log_probs
            * advantages.detach()
        ).mean()

        critic_loss = 0.5 * (
            advantages.pow(2)
        ).mean()

        entropy = (
            distribution.entropy()
            .sum(dim=-1)
            .mean()
        )

        total_loss = (
            actor_loss
            + VALUE_COEFFICIENT * critic_loss
            - ENTROPY_COEFFICIENT * entropy
        )

        self.local_model.zero_grad(
            set_to_none=True
        )

        total_loss.backward()

        # Asynchronous global update.

        self.apply_local_gradients()

        return float(
            total_loss.detach().item()
        )

    def train_episode(self):

        self.synchronize()

        observation = self.env.reset()

        episode_return = 0.0
        episode_distances = []

        rollout_observations = []
        rollout_actions = []
        rollout_rewards = []

        for step in range(
            MAX_EPISODE_STEPS
        ):

            state_tensor = torch.as_tensor(
                observation,
                dtype=torch.float32,
                device=DEVICE,
            ).unsqueeze(0)

            with torch.no_grad():

                (
                    action_tensor,
                    _,
                    _,
                    value_tensor,
                ) = self.local_model.sample_action(
                    state_tensor
                )

            action = (
                action_tensor.squeeze(0)
                .cpu()
                .numpy()
            )

            (
                next_observation,
                reward,
                terminated,
                truncated,
                info,
            ) = self.env.step(action)

            rollout_observations.append(
                observation.copy()
            )

            rollout_actions.append(
                action.copy()
            )

            rollout_rewards.append(
                reward
            )

            episode_return += reward

            episode_distances.append(
                info["distance"]
            )

            observation = next_observation

            # Update after N_STEP transitions or at episode end.

            rollout_complete = (
                len(rollout_rewards) >= N_STEP
                or terminated
                or truncated
            )

            if rollout_complete:

                if terminated:

                    bootstrap_value = 0.0

                else:

                    final_state_tensor = torch.as_tensor(
                        observation,
                        dtype=torch.float32,
                        device=DEVICE,
                    ).unsqueeze(0)

                    with torch.no_grad():

                        _, _, bootstrap = (
                            self.local_model(
                                final_state_tensor
                            )
                        )

                    bootstrap_value = float(
                        bootstrap.item()
                    )

                self.update_from_rollout(
                    rollout_observations,
                    rollout_actions,
                    rollout_rewards,
                    bootstrap_value,
                    terminated,
                )

                # Clear local rollout buffer.

                rollout_observations.clear()
                rollout_actions.clear()
                rollout_rewards.clear()

                # Refresh local parameters after the update.

                self.synchronize()

            if terminated or truncated:

                break

        return {
            "return": float(episode_return),
            "distance": float(
                np.mean(episode_distances)
            ),
            "steps": step + 1,
        }

    def run(self):

        local_episode = 0

        while local_episode < MAX_EPISODES_PER_WORKER:

            result = self.train_episode()

            local_episode += 1

            with self.counter_lock:

                self.global_episode_counter[0] += 1

                global_episode = (
                    self.global_episode_counter[0]
                )

            with self.history_lock:

                self.history["return"].append(
                    result["return"]
                )

                self.history["distance"].append(
                    result["distance"]
                )

                self.history["steps"].append(
                    result["steps"]
                )

                self.history["worker"].append(
                    self.worker_id
                )

            if (
                global_episode % PRINT_INTERVAL == 0
            ):

                with self.history_lock:

                    recent_returns = (
                        self.history["return"][-PRINT_INTERVAL:]
                    )

                    recent_distances = (
                        self.history["distance"][-PRINT_INTERVAL:]
                    )

                print(
                    f"Episode {global_episode:4d}"
                    f" | Worker {self.worker_id}"
                    f" | Return={np.mean(recent_returns):8.2f}"
                    f" | Distance={np.mean(recent_distances):7.2f}"
                )


# ============================================================
# 7. ASYNCHRONOUS TRAINING
# ============================================================

def train_a3c(
    num_workers=NUM_WORKERS,
    episodes_per_worker=MAX_EPISODES_PER_WORKER,
):

    global MAX_EPISODES_PER_WORKER

    MAX_EPISODES_PER_WORKER = (
        episodes_per_worker
    )

    global_model = ActorCritic().to(
        DEVICE
    )

    # Threads share the same global model in memory.

    global_optimizer = optim.Adam(
        global_model.parameters(),
        lr=LEARNING_RATE,
    )

    update_lock = threading.Lock()

    counter_lock = threading.Lock()

    history_lock = threading.Lock()

    global_episode_counter = [0]

    history = {
        "return": [],
        "distance": [],
        "steps": [],
        "worker": [],
    }

    workers = []

    for worker_id in range(
        num_workers
    ):

        worker = A3CWorker(
            worker_id,
            global_model,
            global_optimizer,
            update_lock,
            global_episode_counter,
            counter_lock,
            history,
            history_lock,
        )

        workers.append(
            worker
        )

    start_time = time.time()

    # Each thread independently interacts with its environment.
    # Global optimizer updates are serialized by update_lock.

    with ThreadPoolExecutor(
        max_workers=num_workers
    ) as executor:

        futures = [
            executor.submit(
                worker.run
            )
            for worker in workers
        ]

        for future in futures:

            future.result()

    elapsed = (
        time.time() - start_time
    )

    print(
        f"\nA3C training completed in {elapsed:.1f} seconds."
    )

    return (
        global_model,
        history,
    )


# ============================================================
# 8. TRAINING VISUALIZATION
# ============================================================

def moving_average(values, window=20):

    values = np.asarray(
        values,
        dtype=np.float32,
    )

    if len(values) < window:

        return values

    kernel = np.ones(
        window,
        dtype=np.float32,
    ) / window

    return np.convolve(
        values,
        kernel,
        mode="valid",
    )


def plot_history(history):

    returns = history["return"]

    distances = history["distance"]

    steps = history["steps"]

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(11, 10),
    )

    axes[0].plot(
        returns,
        alpha=0.3,
        label="Episode return",
    )

    axes[0].plot(
        moving_average(returns),
        label="Moving average",
    )

    axes[0].set_title(
        "A3C Training Return"
    )

    axes[0].set_xlabel(
        "Completed episodes"
    )

    axes[0].legend()

    axes[0].grid(True)

    axes[1].plot(
        distances,
        alpha=0.3,
    )

    axes[1].plot(
        moving_average(distances),
    )

    axes[1].set_title(
        "Mean Moving-Target Distance"
    )

    axes[1].set_xlabel(
        "Completed episodes"
    )

    axes[1].grid(True)

    axes[2].plot(
        steps,
        alpha=0.5,
    )

    axes[2].set_title(
        "Episode Length"
    )

    axes[2].set_xlabel(
        "Completed episodes"
    )

    axes[2].grid(True)

    plt.tight_layout()

    plt.show()


# ============================================================
# 9. PYGAME QUADCOPTER VISUALIZATION
# ============================================================

def run_pygame(global_model):

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
        "Phase 5.9 — A3C Quadcopter"
    )

    clock = pygame.time.Clock()

    font = pygame.font.SysFont(
        "consolas",
        15,
    )

    env = DroneEnv(
        seed=SEED + 9000
    )

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

    def draw_text(message, x, y):

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
                [1.0, 0.0],
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

        for index in range(100):

            phase = (
                env.target.phase
                + 2 * math.pi * index / 100
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
                        global_model.deterministic_action(
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

        draw_text(
            "PHASE 5.9 — ASYNCHRONOUS ACTOR-CRITIC",
            15,
            12,
        )

        draw_text(
            f"Target distance: {info['distance'] if 'info' in locals() else 0:.2f}",
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
# 10. NUMERICAL TESTS
# ============================================================

def run_tests():

    print(
        "\nRunning A3C tests..."
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

    # Test 1: Observation dimension.

    env = DroneEnv()

    observation = env.reset()

    check(
        observation.shape == (OBS_DIM,),
        "Drone observation dimensions",
    )

    # Test 2: Target velocity dimension.

    target = MovingTarget(
        [400.0, 300.0]
    )

    check(
        target.velocity().shape == (2,),
        "Moving target velocity dimensions",
    )

    # Test 3: Action dimension.

    model = ActorCritic().to(
        DEVICE
    )

    state_tensor = torch.as_tensor(
        observation,
        dtype=torch.float32,
    ).unsqueeze(0)

    action, log_prob, entropy, value = (
        model.sample_action(
            state_tensor
        )
    )

    check(
        action.shape == (1, ACTION_DIM),
        "Actor action dimensions",
    )

    # Test 4: Bounded tanh actions.

    check(
        bool(
            torch.all(action.abs() <= 1.0)
        ),
        "Tanh actions are bounded",
    )

    # Test 5: Finite log probability.

    check(
        bool(
            torch.isfinite(log_prob).all()
        ),
        "Transformed action log probability is finite",
    )

    # Test 6: Finite entropy.

    check(
        bool(
            torch.isfinite(entropy).all()
        ),
        "Gaussian entropy is finite",
    )

    # Test 7: Critic output shape.

    check(
        value.shape == (1,),
        "Critic output dimensions",
    )

    # Test 8: N-step return.

    result = calculate_n_step_returns(
        rewards=[1.0, 2.0, 3.0],
        bootstrap_value=4.0,
        terminated=False,
        truncated=False,
        gamma=0.9,
    )

    expected = (
        1.0
        + 0.9 * 2.0
        + 0.9**2 * 3.0
        + 0.9**3 * 4.0
    )

    check(
        np.isclose(
            result,
            expected,
        ),
        "N-step return calculation",
    )

    # Test 9: Terminal suppresses bootstrap.

    result = calculate_n_step_returns(
        rewards=[1.0, 2.0],
        bootstrap_value=100.0,
        terminated=True,
        truncated=False,
        gamma=0.9,
    )

    check(
        np.isclose(
            result,
            1.0 + 0.9 * 2.0,
        ),
        "Terminal suppresses bootstrap",
    )

    # Test 10: Truncation preserves bootstrap.

    result = calculate_n_step_returns(
        rewards=[1.0, 2.0],
        bootstrap_value=3.0,
        terminated=False,
        truncated=True,
        gamma=0.9,
    )

    check(
        np.isclose(
            result,
            1.0 + 0.9 * 2.0 + 0.9**2 * 3.0,
        ),
        "Truncation preserves final-state bootstrap",
    )

    # Test 11: Environment transition.

    next_observation, reward, terminated, truncated, info = (
        env.step(
            np.zeros(
                ACTION_DIM,
                dtype=np.float32,
            )
        )
    )

    check(
        next_observation.shape == (OBS_DIM,)
        and np.isfinite(reward),
        "Drone environment transition",
    )

    # Test 12: Model parameters exist.

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    check(
        parameter_count > 0,
        "Actor-critic has trainable parameters",
    )

    # Test 13: Global/local synchronization.

    local_model = ActorCritic().to(
        DEVICE
    )

    local_model.load_state_dict(
        model.state_dict()
    )

    equal = all(
        torch.allclose(
            global_parameter,
            local_parameter,
        )
        for global_parameter, local_parameter
        in zip(
            model.parameters(),
            local_model.parameters(),
        )
    )

    check(
        equal,
        "Local network synchronizes with global network",
    )

    # Test 14: Local gradients can be calculated.

    local_model.zero_grad(
        set_to_none=True
    )

    test_action, test_log_prob, test_entropy, test_value = (
        local_model.sample_action(
            state_tensor
        )
    )

    test_loss = (
        -test_log_prob.mean()
        + 0.5 * test_value.pow(2).mean()
        - 0.01 * test_entropy.mean()
    )

    test_loss.backward()

    has_gradient = all(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        for parameter in local_model.parameters()
    )

    check(
        has_gradient,
        "Actor-critic gradients are finite",
    )

    # Test 15: Environment state remains bounded.

    for _ in range(20):

        next_observation, _, terminated, truncated, _ = (
            env.step(
                np.array(
                    [0.8, -0.6],
                    dtype=np.float32,
                )
            )
        )

        if terminated or truncated:

            env.reset()

    check(
        DRONE_RADIUS <= env.state[0]
        <= WORLD_W - DRONE_RADIUS
        and DRONE_RADIUS <= env.state[1]
        <= WORLD_H - DRONE_RADIUS,
        "Drone position remains inside world bounds",
    )

    print(
        f"\n{passed}/15 tests passed."
    )

    return passed == 15


# ============================================================
# 11. MAIN
# ============================================================

def main():

    print("=" * 72)

    print(
        "PHASE 5.9 — ASYNCHRONOUS ADVANTAGE ACTOR-CRITIC"
    )

    print("=" * 72)

    print(
        "Device:",
        DEVICE,
    )

    print(
        "Workers:",
        NUM_WORKERS,
    )

    if not run_tests():

        raise RuntimeError(
            "A3C numerical tests failed."
        )

    print(
        "\nStarting asynchronous A3C training..."
    )

    global_model, history = train_a3c(
        num_workers=NUM_WORKERS,
        episodes_per_worker=MAX_EPISODES_PER_WORKER,
    )

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
        global_model
    )

    print(
        "\nPhase 5.9 completed."
    )


if __name__ == "__main__":

    main()
