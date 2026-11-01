
"""
File: phase4/autonomous_drone_system.py

PHASE 4.31 — DEEP RL INTEGRATION PROJECT
AUTONOMOUS DRONE LEARNING SYSTEM

============================================================
PURPOSE
============================================================

Integrate the major Deep RL and model-based RL techniques
studied throughout Phase 4 into one autonomous drone system.

The system combines:

1. SAC continuous-control policy
2. Twin Q-networks
3. Automatic entropy-temperature tuning
4. Ensemble learned dynamics models
5. Uncertainty-aware model predictive control
6. Moving-target prediction
7. Approximate control-barrier-function safety filtering
8. Online model adaptation
9. Interactive Pygame visualization
10. Numerical tests and performance evaluation

============================================================
WHAT YOU LEARN
============================================================

STEP 1 — Environment
    Model a 2D quadcopter with continuous acceleration,
    obstacle avoidance and a moving target.

STEP 2 — SAC
    Learn a stochastic continuous-control policy using
    twin critics, target critics and entropy regularization.

STEP 3 — Learned dynamics
    Learn the drone's physical state transition from data.

STEP 4 — Ensemble uncertainty
    Train multiple dynamics models and measure disagreement.

STEP 5 — MPC
    Search candidate future action sequences and execute
    only the first action before replanning.

STEP 6 — Safety filtering
    Modify actions that violate approximate obstacle or
    boundary safety constraints.

STEP 7 — Online adaptation
    Continue updating the dynamics models using newly
    collected transitions.

STEP 8 — Evaluation
    Compare SAC, SAC+MPC and SAC+MPC+Safety.

============================================================
MATHEMATICS
============================================================

SAC objective:

    J(pi) = E[ sum gamma^t *
                (reward + alpha * entropy) ]

Critic target:

    y = r + gamma * (1-done) *
        (min(Q1_target,Q2_target) - alpha*log_pi)

Actor objective:

    L_actor = E[alpha*log_pi - min(Q1,Q2)]

Temperature objective:

    L_alpha = -E[log_alpha * (log_pi + target_entropy)]

Learned dynamics:

    delta_state = f_theta(state, action)

Ensemble disagreement:

    uncertainty = std(
        model_1_prediction,
        model_2_prediction,
        ...
    )

MPC:

    action_sequence* = argmin_sequence
        sum(stage_cost + uncertainty_penalty)

Receding horizon:

    Execute first action only.
    Observe the real next state.
    Replan.

Approximate obstacle barrier:

    h(p) = ||p - obstacle||^2 - safe_radius^2

    Require:

    (p-obstacle) dot (v + a*dt*u)
        >= -alpha*h(p)

This is a simplified educational discrete-time
safety approximation, not a flight-certified guarantee.

============================================================
ROBOTICS APPLICATION
============================================================

- Autonomous drone navigation
- Mobile robot navigation
- Moving-object tracking
- Model-based planning
- Online adaptation to payload changes
- Sim-to-real research foundations

IMPORTANT:
    This is a simplified 2D simulation.
    It is not suitable for controlling a real drone.
"""

from __future__ import annotations

import copy
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

SEED = 431

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

TARGET_A = 160.0
TARGET_B = 105.0
TARGET_OMEGA = 0.55

OBS_DIM = 10
ACTION_DIM = 2

ACTOR_LR = 3e-4
CRITIC_LR = 3e-4
MODEL_LR = 5e-4

GAMMA = 0.99
TAU = 0.005

INITIAL_ALPHA = 0.2

REPLAY_CAPACITY = 60000
BATCH_SIZE = 128

ENSEMBLE_SIZE = 5
MODEL_BATCH_SIZE = 128

TRAIN_EPISODES = 35
EPISODE_STEPS = 160

INITIAL_RANDOM_STEPS = 900

MODEL_TRAIN_INTERVAL = 150
MODEL_TRAIN_UPDATES = 12

MPC_HORIZON = 12
MPC_CANDIDATES = 96
MPC_ITERATIONS = 2

MPC_ACTION_NOISE = 0.38
UNCERTAINTY_WEIGHT = 0.12

CBF_ALPHA = 0.45

# Quick experiment:
#
# TRAIN_EPISODES = 5
# EPISODE_STEPS = 80
# INITIAL_RANDOM_STEPS = 200
# MPC_CANDIDATES = 30
# MPC_ITERATIONS = 1
# MODEL_TRAIN_UPDATES = 3


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
# 2. DRONE PHYSICS
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


def drone_transition(
    state,
    action,
    motor_gain=0.75,
    drag=0.12,
    wind=None,
):

    if wind is None:

        wind = np.array(
            [4.0, -2.5],
            dtype=np.float32,
        )

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
        motor_gain * MAX_ACCEL * action
        - drag * velocity
        - 0.0003 * np.abs(velocity) * velocity
        + wind
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
# 3. ENVIRONMENT
# ============================================================

class DroneEnv:

    def __init__(self):

        self.target = MovingTarget(
            [710.0, 310.0]
        )

        self.motor_gain = 0.75
        self.drag = 0.12

        self.wind = np.array(
            [4.0, -2.5],
            dtype=np.float32,
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
        ).astype(
            np.float32
        )

    def reset(self):

        self.state = np.array(
            [
                105.0,
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

    def set_domain(
        self,
        motor_gain,
        drag,
        wind,
    ):

        self.motor_gain = float(
            motor_gain
        )

        self.drag = float(
            drag
        )

        self.wind = np.asarray(
            wind,
            dtype=np.float32,
        )

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

        next_phase = (
            self.target.phase
            + TARGET_OMEGA * DT
        )

        future_target = self.target.position(
            next_phase
        )

        distance_before = np.linalg.norm(
            old_position - future_target
        )

        self.state = drone_transition(
            self.state,
            action,
            self.motor_gain,
            self.drag,
            self.wind,
        )

        self.target.advance()

        distance_after = np.linalg.norm(
            self.state[:2]
            - self.target.position()
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
            0.05 * (
                distance_before - distance_after
            )
            - 0.012 * distance_after
            - 0.10 * np.sum(action**2)
        )

        reward -= (
            0.006
            * max(
                0.0,
                SAFE_RADIUS - obstacle_distance,
            )**2
        )

        reward -= (
            0.015
            * max(
                0.0,
                55.0 - boundary_distance,
            )**2
        )

        if collision:

            reward -= 100.0

        self.steps += 1

        terminated = collision

        truncated = (
            self.steps >= EPISODE_STEPS
        )

        info = {
            "distance": float(distance_after),
            "collision": bool(collision),
            "obstacle_distance": float(
                obstacle_distance
            ),
        }

        return (
            self.observation(),
            float(reward),
            terminated,
            truncated,
            info,
        )


# ============================================================
# 4. REPLAY BUFFER
# ============================================================

class ReplayBuffer:

    def __init__(
        self,
        capacity=REPLAY_CAPACITY,
    ):

        self.data = deque(
            maxlen=capacity
        )

    def add(
        self,
        state,
        action,
        reward,
        next_state,
        terminated,
        physical_state,
        next_physical_state,
    ):

        self.data.append(
            (
                np.asarray(state, dtype=np.float32),
                np.asarray(action, dtype=np.float32),
                float(reward),
                np.asarray(next_state, dtype=np.float32),
                float(terminated),
                np.asarray(physical_state, dtype=np.float32),
                np.asarray(next_physical_state, dtype=np.float32),
            )
        )

    def sample(self, batch_size):

        batch = random.sample(
            self.data,
            batch_size,
        )

        columns = list(
            zip(*batch)
        )

        return [
            torch.as_tensor(
                np.asarray(column),
                dtype=torch.float32,
                device=DEVICE,
            )
            for column in columns
        ]

    def __len__(self):

        return len(
            self.data
        )


# ============================================================
# 5. SAC NETWORKS
# ============================================================

LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


class Actor(nn.Module):

    def __init__(self):

        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(OBS_DIM, 256),
            nn.ReLU(),

            nn.Linear(256, 256),
            nn.ReLU(),
        )

        self.mean = nn.Linear(
            256,
            ACTION_DIM,
        )

        self.log_std = nn.Linear(
            256,
            ACTION_DIM,
        )

    def forward(self, state):

        hidden = self.net(
            state
        )

        mean = self.mean(
            hidden
        )

        log_std = torch.clamp(
            self.log_std(hidden),
            LOG_STD_MIN,
            LOG_STD_MAX,
        )

        return mean, log_std

    def sample(self, state):

        mean, log_std = self(
            state
        )

        std = log_std.exp()

        distribution = torch.distributions.Normal(
            mean,
            std,
        )

        raw_action = distribution.rsample()

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
            dim=-1,
            keepdim=True,
        )

        deterministic_action = torch.tanh(
            mean
        )

        return (
            action,
            log_probability,
            deterministic_action,
        )


class Critic(nn.Module):

    def __init__(self):

        super().__init__()

        self.q1 = nn.Sequential(
            nn.Linear(
                OBS_DIM + ACTION_DIM,
                256,
            ),
            nn.ReLU(),

            nn.Linear(256, 256),
            nn.ReLU(),

            nn.Linear(256, 1),
        )

        self.q2 = nn.Sequential(
            nn.Linear(
                OBS_DIM + ACTION_DIM,
                256,
            ),
            nn.ReLU(),

            nn.Linear(256, 256),
            nn.ReLU(),

            nn.Linear(256, 1),
        )

    def forward(self, state, action):

        x = torch.cat(
            [state, action],
            dim=-1,
        )

        return (
            self.q1(x),
            self.q2(x),
        )


# ============================================================
# 6. SAC AGENT
# ============================================================

class SACAgent:

    def __init__(self):

        self.actor = Actor().to(
            DEVICE
        )

        self.critic = Critic().to(
            DEVICE
        )

        self.target_critic = copy.deepcopy(
            self.critic
        ).to(
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

        self.log_alpha = torch.tensor(
            math.log(INITIAL_ALPHA),
            dtype=torch.float32,
            device=DEVICE,
            requires_grad=True,
        )

        self.alpha_optimizer = optim.Adam(
            [self.log_alpha],
            lr=ACTOR_LR,
        )

        self.target_entropy = -float(
            ACTION_DIM
        )

        self.updates = 0

    @property
    def alpha(self):

        return self.log_alpha.exp()

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

            action, _, deterministic_action = (
                self.actor.sample(state)
            )

        selected = (
            deterministic_action
            if deterministic
            else action
        )

        return selected.squeeze(0).cpu().numpy()

    def update(
        self,
        replay,
    ):

        if len(replay) < BATCH_SIZE:

            return None

        (
            states,
            actions,
            rewards,
            next_states,
            terminated,
            _,
            _,
        ) = replay.sample(
            BATCH_SIZE
        )

        rewards = rewards.unsqueeze(-1)

        terminated = terminated.unsqueeze(-1)

        # --------------------------------------------
        # Critic target
        # --------------------------------------------

        with torch.no_grad():

            (
                next_actions,
                next_log_probability,
                _,
            ) = self.actor.sample(
                next_states
            )

            target_q1, target_q2 = (
                self.target_critic(
                    next_states,
                    next_actions,
                )
            )

            target_q = torch.minimum(
                target_q1,
                target_q2,
            )

            target_value = (
                target_q
                - self.alpha.detach()
                * next_log_probability
            )

            y = (
                rewards
                + GAMMA
                * (1.0 - terminated)
                * target_value
            )

        # --------------------------------------------
        # Critic update
        # --------------------------------------------

        q1, q2 = self.critic(
            states,
            actions,
        )

        critic_loss = (
            F.mse_loss(q1, y)
            + F.mse_loss(q2, y)
        )

        self.critic_optimizer.zero_grad()

        critic_loss.backward()

        nn.utils.clip_grad_norm_(
            self.critic.parameters(),
            10.0,
        )

        self.critic_optimizer.step()

        # --------------------------------------------
        # Actor update
        # --------------------------------------------

        new_actions, log_probability, _ = (
            self.actor.sample(
                states
            )
        )

        q1_new, q2_new = self.critic(
            states,
            new_actions,
        )

        min_q = torch.minimum(
            q1_new,
            q2_new,
        )

        actor_loss = (
            self.alpha.detach()
            * log_probability
            - min_q
        ).mean()

        self.actor_optimizer.zero_grad()

        actor_loss.backward()

        nn.utils.clip_grad_norm_(
            self.actor.parameters(),
            10.0,
        )

        self.actor_optimizer.step()

        # --------------------------------------------
        # Automatic entropy temperature
        # --------------------------------------------

        alpha_loss = -(
            self.log_alpha
            * (
                log_probability.detach()
                + self.target_entropy
            )
        ).mean()

        self.alpha_optimizer.zero_grad()

        alpha_loss.backward()

        self.alpha_optimizer.step()

        # --------------------------------------------
        # Target critic update
        # --------------------------------------------

        with torch.no_grad():

            for target_parameter, parameter in zip(
                self.target_critic.parameters(),
                self.critic.parameters(),
            ):

                target_parameter.data.mul_(
                    1.0 - TAU
                )

                target_parameter.data.add_(
                    TAU * parameter.data
                )

        self.updates += 1

        return {
            "actor_loss": float(
                actor_loss.detach().cpu()
            ),

            "critic_loss": float(
                critic_loss.detach().cpu()
            ),

            "alpha": float(
                self.alpha.detach().cpu()
            ),
        }


# ============================================================
# 7. ENSEMBLE LEARNED DYNAMICS
# ============================================================

class DynamicsModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(
                4 + ACTION_DIM,
                128,
            ),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.ReLU(),

            nn.Linear(128, 4),
        )

    def forward(
        self,
        physical_state,
        action,
    ):

        x = torch.cat(
            [
                physical_state,
                action,
            ],
            dim=-1,
        )

        return self.net(
            x
        )


class DynamicsEnsemble:

    def __init__(self):

        self.models = [
            DynamicsModel().to(DEVICE)
            for _ in range(ENSEMBLE_SIZE)
        ]

        self.optimizers = [
            optim.Adam(
                model.parameters(),
                lr=MODEL_LR,
            )
            for model in self.models
        ]

        self.model_losses = []

    def predict_batch(
        self,
        physical_states,
        actions,
    ):

        predictions = []

        with torch.no_grad():

            for model in self.models:

                delta = model(
                    physical_states,
                    actions,
                )

                predictions.append(
                    physical_states + delta
                )

        return torch.stack(
            predictions,
            dim=0,
        )

    def predict(
        self,
        physical_state,
        action,
    ):

        state_tensor = torch.as_tensor(
            physical_state,
            dtype=torch.float32,
            device=DEVICE,
        ).unsqueeze(0)

        action_tensor = torch.as_tensor(
            action,
            dtype=torch.float32,
            device=DEVICE,
        ).unsqueeze(0)

        predictions = self.predict_batch(
            state_tensor,
            action_tensor,
        )

        mean = predictions.mean(
            dim=0
        ).squeeze(0).cpu().numpy()

        std = predictions.std(
            dim=0
        ).squeeze(0).cpu().numpy()

        return (
            clip_state(mean),
            std,
        )

    def train(
        self,
        replay,
        updates=MODEL_TRAIN_UPDATES,
    ):

        if len(replay) < MODEL_BATCH_SIZE:

            return None

        losses = []

        for _ in range(updates):

            (
                states,
                actions,
                rewards,
                next_states,
                terminated,
                physical_states,
                next_physical_states,
            ) = replay.sample(
                MODEL_BATCH_SIZE
            )

            target_delta = (
                next_physical_states
                - physical_states
            )

            for model, optimizer in zip(
                self.models,
                self.optimizers,
            ):

                # Bootstrap each model using its own
                # independently sampled batch.
                indices = torch.randint(
                    0,
                    len(physical_states),
                    (MODEL_BATCH_SIZE,),
                    device=DEVICE,
                )

                prediction = model(
                    physical_states[indices],
                    actions[indices],
                )

                target = target_delta[
                    indices
                ]

                loss = F.mse_loss(
                    prediction,
                    target,
                )

                optimizer.zero_grad()

                loss.backward()

                nn.utils.clip_grad_norm_(
                    model.parameters(),
                    10.0,
                )

                optimizer.step()

                losses.append(
                    float(
                        loss.detach().cpu()
                    )
                )

        mean_loss = float(
            np.mean(losses)
        )

        self.model_losses.append(
            mean_loss
        )

        return mean_loss


# ============================================================
# 8. APPROXIMATE SAFETY FILTER
# ============================================================

def safety_filter(
    state,
    requested_action,
):

    """
    Project a requested action onto approximate
    one-step obstacle and boundary constraints.

    The projection is performed using repeated
    half-space corrections.

    This is not a formal safety guarantee:
    - constraints may be infeasible;
    - the dynamics model may be inaccurate;
    - acceleration and clipping are approximated;
    - safety cannot be guaranteed outside the safe set.
    """

    action = np.clip(
        np.asarray(
            requested_action,
            dtype=np.float32,
        ).copy(),
        -1.0,
        1.0,
    )

    position = state[:2]

    velocity = state[2:4]

    dt = DT

    # Approximate next velocity:
    #
    # v_next = v + MAX_ACCEL * dt * action
    #
    # Position is approximated using v_next.

    def constraints(candidate):

        next_velocity = (
            velocity
            + MAX_ACCEL * dt * candidate
        )

        next_position = (
            position
            + next_velocity * dt
        )

        values = []

        # Obstacle barrier.
        relative = (
            position - OBSTACLE
        )

        h = (
            np.dot(relative, relative)
            - SAFE_RADIUS**2
        )

        obstacle_lhs = np.dot(
            relative,
            next_velocity,
        )

        obstacle_rhs = (
            -CBF_ALPHA * h
        )

        values.append(
            (
                obstacle_lhs - obstacle_rhs,
                relative * MAX_ACCEL * dt,
            )
        )

        # Four world boundaries.
        margin = 55.0

        boundary_specs = [
            (
                position[0] - margin,
                next_position[0] - margin,
                np.array([1.0, 0.0]),
            ),
            (
                WORLD_W - margin - position[0],
                WORLD_W - margin - next_position[0],
                np.array([-1.0, 0.0]),
            ),
            (
                position[1] - margin,
                next_position[1] - margin,
                np.array([0.0, 1.0]),
            ),
            (
                WORLD_H - margin - position[1],
                WORLD_H - margin - next_position[1],
                np.array([0.0, -1.0]),
            ),
        ]

        for current_margin, next_margin, direction in boundary_specs:

            values.append(
                (
                    next_margin,
                    direction * MAX_ACCEL * dt * dt,
                )
            )

        return values

    # Iterative projection.
    for _ in range(12):

        changed = False

        for value, gradient in constraints(
            action
        ):

            norm_squared = np.dot(
                gradient,
                gradient,
            )

            if norm_squared < 1e-10:

                continue

            if value < 0.0:

                action += (
                    -value / norm_squared
                ) * gradient

                action = np.clip(
                    action,
                    -1.0,
                    1.0,
                )

                changed = True

        if not changed:

            break

    return action


# ============================================================
# 9. UNCERTAINTY-AWARE MPC
# ============================================================

class MPCController:

    def __init__(
        self,
        actor,
        ensemble,
    ):

        self.actor = actor

        self.ensemble = ensemble

        self.last_plan = None

    def actor_mean_action(
        self,
        observation,
    ):

        state_tensor = torch.as_tensor(
            observation,
            dtype=torch.float32,
            device=DEVICE,
        ).unsqueeze(0)

        with torch.no_grad():

            _, _, action = self.actor.sample(
                state_tensor
            )

        return action.squeeze(0).cpu().numpy()

    def future_observation(
        self,
        physical_state,
        target_phase,
        target_center,
    ):

        position = physical_state[:2]

        velocity = physical_state[2:4]

        target = MovingTarget(
            target_center
        )

        target.phase = target_phase

        return np.concatenate(
            [
                (target.position() - position) / 400.0,
                target.velocity() / 200.0,
                velocity / MAX_SPEED,
                (OBSTACLE - position) / 400.0,
                [
                    math.sin(target_phase),
                    math.cos(target_phase),
                ],
            ]
        ).astype(
            np.float32
        )

    def plan(
        self,
        physical_state,
        target_phase,
        target_center,
        use_uncertainty=True,
    ):

        base_action = self.actor_mean_action(
            self.future_observation(
                physical_state,
                target_phase,
                target_center,
            )
        )

        horizon = MPC_HORIZON

        candidates = np.zeros(
            (
                MPC_CANDIDATES,
                horizon,
                ACTION_DIM,
            ),
            dtype=np.float32,
        )

        # Shift the previous plan for warm-starting.
        if self.last_plan is not None:

            seed = np.concatenate(
                [
                    self.last_plan[1:],
                    self.last_plan[-1:],
                ],
                axis=0,
            )

        else:

            seed = np.repeat(
                base_action[None, :],
                horizon,
                axis=0,
            )

        for index in range(
            MPC_CANDIDATES
        ):

            noise = np.random.normal(
                0.0,
                MPC_ACTION_NOISE,
                size=(horizon, ACTION_DIM),
            )

            candidates[index] = np.clip(
                seed + noise,
                -1.0,
                1.0,
            )

        # Include the unperturbed seed.
        candidates[0] = seed

        best_sequence = None

        best_cost = float("inf")

        for _ in range(
            MPC_ITERATIONS
        ):

            candidate_costs = []

            for candidate in candidates:

                predicted_state = (
                    physical_state.copy()
                )

                phase = target_phase

                total_cost = 0.0

                cumulative_uncertainty = 0.0

                for step, action in enumerate(
                    candidate
                ):

                    predicted_state, uncertainty = (
                        self.ensemble.predict(
                            predicted_state,
                            action,
                        )
                    )

                    phase += (
                        TARGET_OMEGA * DT
                    )

                    target = (
                        MovingTarget(
                            target_center
                        ).position(
                            phase
                        )
                    )

                    distance = np.linalg.norm(
                        predicted_state[:2] - target
                    )

                    obstacle_distance = np.linalg.norm(
                        predicted_state[:2] - OBSTACLE
                    )

                    boundary_distance = min(
                        predicted_state[0],
                        WORLD_W - predicted_state[0],
                        predicted_state[1],
                        WORLD_H - predicted_state[1],
                    )

                    stage_cost = (
                        0.018 * distance
                        + 0.10 * np.sum(action**2)
                    )

                    stage_cost += (
                        0.012
                        * max(
                            0.0,
                            SAFE_RADIUS - obstacle_distance,
                        )**2
                    )

                    stage_cost += (
                        0.025
                        * max(
                            0.0,
                            60.0 - boundary_distance,
                        )**2
                    )

                    uncertainty_score = float(
                        np.linalg.norm(
                            uncertainty
                        )
                    )

                    cumulative_uncertainty += (
                        uncertainty_score
                    )

                    discount = (
                        GAMMA**step
                    )

                    total_cost += (
                        discount * stage_cost
                    )

                    if use_uncertainty:

                        total_cost += (
                            discount
                            * UNCERTAINTY_WEIGHT
                            * uncertainty_score
                        )

                total_cost += (
                    0.03 * cumulative_uncertainty
                )

                candidate_costs.append(
                    total_cost
                )

            best_index = int(
                np.argmin(
                    candidate_costs
                )
            )

            best_cost = float(
                candidate_costs[best_index]
            )

            best_sequence = candidates[
                best_index
            ].copy()

            # Recenter candidates around the best sequence.
            new_candidates = np.zeros_like(
                candidates
            )

            new_candidates[0] = (
                best_sequence
            )

            for index in range(
                1,
                MPC_CANDIDATES
            ):

                noise = np.random.normal(
                    0.0,
                    MPC_ACTION_NOISE
                    * (0.85 ** (_ + 1)),
                    size=(horizon, ACTION_DIM),
                )

                new_candidates[index] = np.clip(
                    best_sequence + noise,
                    -1.0,
                    1.0,
                )

            candidates = new_candidates

        self.last_plan = (
            best_sequence.copy()
        )

        return (
            best_sequence[0].copy(),
            best_sequence,
            best_cost,
        )


# ============================================================
# 10. TRAINING
# ============================================================

def train_system():

    agent = SACAgent()

    ensemble = DynamicsEnsemble()

    replay = ReplayBuffer()

    returns = []

    distances = []

    model_losses = []

    total_steps = 0

    last_update = 0

    for episode_index in range(
        TRAIN_EPISODES
    ):

        env = DroneEnv()

        # Randomize physical parameters during training.
        motor_gain = np.random.uniform(
            0.60,
            0.90,
        )

        drag = np.random.uniform(
            0.08,
            0.18,
        )

        wind = np.random.uniform(
            -7.0,
            7.0,
            size=2,
        ).astype(
            np.float32
        )

        env.set_domain(
            motor_gain,
            drag,
            wind,
        )

        observation = env.reset()

        episode_return = 0.0

        episode_distances = []

        for step in range(
            EPISODE_STEPS
        ):

            physical_state = (
                env.state.copy()
            )

            if total_steps < INITIAL_RANDOM_STEPS:

                action = np.random.uniform(
                    -1.0,
                    1.0,
                    size=ACTION_DIM,
                ).astype(
                    np.float32
                )

            else:

                action = agent.select_action(
                    observation,
                    deterministic=False,
                )

            (
                next_observation,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(
                action
            )

            next_physical_state = (
                env.state.copy()
            )

            replay.add(
                observation,
                action,
                reward,
                next_observation,
                terminated,
                physical_state,
                next_physical_state,
            )

            observation = next_observation

            episode_return += reward

            episode_distances.append(
                info["distance"]
            )

            total_steps += 1

            # SAC updates.
            if len(replay) >= BATCH_SIZE:

                agent.update(
                    replay
                )

            # Online dynamics-model updates.
            if (
                total_steps - last_update
                >= MODEL_TRAIN_INTERVAL
                and len(replay) >= MODEL_BATCH_SIZE
            ):

                model_loss = ensemble.train(
                    replay
                )

                if model_loss is not None:

                    model_losses.append(
                        model_loss
                    )

                last_update = total_steps

            if terminated or truncated:

                break

        returns.append(
            episode_return
        )

        distances.append(
            float(
                np.mean(
                    episode_distances
                )
            )
        )

        print(
            f"Episode {episode_index + 1:3d}/{TRAIN_EPISODES}"
            f" | Return={episode_return:9.2f}"
            f" | Mean distance={distances[-1]:8.2f}"
            f" | Steps={step + 1:3d}"
            f" | Alpha={float(agent.alpha.detach().cpu()):.3f}"
            f" | Replay={len(replay):5d}"
        )

    return (
        agent,
        ensemble,
        replay,
        {
            "returns": returns,
            "distances": distances,
            "model_losses": model_losses,
        },
    )


# ============================================================
# 11. EVALUATION
# ============================================================

def evaluate_controller(
    agent,
    ensemble,
    mode,
    episodes=5,
):

    results = []

    for episode_index in range(
        episodes
    ):

        env = DroneEnv()

        # Test on changed physical dynamics.
        if episode_index % 2 == 0:

            env.set_domain(
                motor_gain=0.60,
                drag=0.18,
                wind=np.array(
                    [7.0, -5.0]
                ),
            )

        observation = env.reset()

        mpc = MPCController(
            agent.actor,
            ensemble,
        )

        total_return = 0.0

        episode_distances = []

        collision = False

        for _ in range(
            EPISODE_STEPS
        ):

            if mode == "SAC":

                action = agent.select_action(
                    observation,
                    deterministic=True,
                )

            else:

                mpc_action, _, _ = mpc.plan(
                    env.state,
                    env.target.phase,
                    env.target.center,
                    use_uncertainty=True,
                )

                if mode == "SAC+MPC":

                    action = mpc_action

                else:

                    action = safety_filter(
                        env.state,
                        mpc_action,
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

                collision = True

            if terminated or truncated:

                break

        results.append(
            {
                "return": total_return,
                "distance": float(
                    np.mean(episode_distances)
                ),
                "collision": collision,
            }
        )

    return results


def print_evaluation(
    agent,
    ensemble,
):

    modes = [
        "SAC",
        "SAC+MPC",
        "SAC+MPC+Safety",
    ]

    print(
        "\nController comparison"
    )

    print(
        "-" * 76
    )

    all_results = {}

    for mode in modes:

        results = evaluate_controller(
            agent,
            ensemble,
            mode,
            episodes=3,
        )

        all_results[mode] = results

        mean_return = np.mean(
            [
                result["return"]
                for result in results
            ]
        )

        mean_distance = np.mean(
            [
                result["distance"]
                for result in results
            ]
        )

        collision_rate = np.mean(
            [
                result["collision"]
                for result in results
            ]
        )

        print(
            f"{mode:20s}"
            f" | Return={mean_return:9.2f}"
            f" | Distance={mean_distance:8.2f}"
            f" | Collision rate={collision_rate:.2f}"
        )

    return all_results


# ============================================================
# 12. VISUALIZATION
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
        "SAC Training Return"
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
        label="Mean target distance",
    )

    plt.xlabel(
        "Episode"
    )

    plt.ylabel(
        "Distance"
    )

    plt.title(
        "Moving-Target Tracking Performance"
    )

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    plt.show()

    if history["model_losses"]:

        plt.figure(
            figsize=(10, 5)
        )

        plt.plot(
            history["model_losses"],
            label="Dynamics ensemble loss",
        )

        plt.xlabel(
            "Model training event"
        )

        plt.ylabel(
            "MSE"
        )

        plt.title(
            "Learned Dynamics Model Training"
        )

        plt.grid(True)

        plt.legend()

        plt.tight_layout()

        plt.show()


# ============================================================
# 13. PYGAME VISUALIZATION
# ============================================================

def run_pygame(
    agent,
    ensemble,
):

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
        "Phase 4.31 — Autonomous Drone Learning System"
    )

    clock = pygame.time.Clock()

    font = pygame.font.SysFont(
        "consolas",
        15,
    )

    env = DroneEnv()

    observation = env.reset()

    mpc = MPCController(
        agent.actor,
        ensemble,
    )

    modes = [
        "SAC",
        "SAC+MPC",
        "SAC+MPC+Safety",
    ]

    mode_index = 2

    paused = False

    speed = 1

    trail = []

    requested_action = np.zeros(
        2,
        dtype=np.float32,
    )

    executed_action = np.zeros(
        2,
        dtype=np.float32,
    )

    predicted_path = []

    uncertainty_value = 0.0

    total_reward = 0.0

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
                    + direction * forward * 17
                    + side * lateral * 17
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
                    (100, 195, 245),
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
            (255, 190, 65),
            (
                center + direction * 22
            ).astype(int),
            4,
        )

        # Requested action: red.
        # Executed action: blue.
        scale = 38.0

        pygame.draw.line(
            screen,
            (245, 80, 80),
            center.astype(int),
            (
                center + requested_action * scale
            ).astype(int),
            3,
        )

        pygame.draw.line(
            screen,
            (70, 150, 255),
            center.astype(int),
            (
                center + executed_action * scale
            ).astype(int),
            3,
        )

    def draw_target():

        points = []

        for index in range(100):

            phase = (
                env.target.phase
                + 2.0 * math.pi * index / 100
            )

            points.append(
                env.target.position(
                    phase
                ).astype(int)
            )

        pygame.draw.lines(
            screen,
            (60, 105, 75),
            True,
            points,
            2,
        )

        position = env.target.position()

        velocity = env.target.velocity()

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

                    predicted_path.clear()

                    mpc.last_plan = None

                    total_reward = 0.0

                elif event.key == pygame.K_1:

                    mode_index = 0

                elif event.key == pygame.K_2:

                    mode_index = 1

                elif event.key == pygame.K_3:

                    mode_index = 2

                elif event.key == pygame.K_PLUS:

                    speed = min(
                        5,
                        speed + 1,
                    )

                elif event.key == pygame.K_EQUALS:

                    speed = min(
                        5,
                        speed + 1,
                    )

                elif event.key == pygame.K_MINUS:

                    speed = max(
                        1,
                        speed - 1,
                    )

                elif event.key == pygame.K_d:

                    # Simulate a sudden dynamics change.
                    env.motor_gain = (
                        0.55
                        if env.motor_gain > 0.65
                        else 0.80
                    )

        if not paused:

            for _ in range(speed):

                mode = modes[
                    mode_index
                ]

                if mode == "SAC":

                    requested_action = (
                        agent.select_action(
                            observation,
                            deterministic=True,
                        )
                    )

                    predicted_path = []

                else:

                    (
                        requested_action,
                        plan,
                        _,
                    ) = mpc.plan(
                        env.state,
                        env.target.phase,
                        env.target.center,
                        use_uncertainty=True,
                    )

                    predicted_path = []

                    predicted_state = (
                        env.state.copy()
                    )

                    phase = env.target.phase

                    for action in plan:

                        predicted_state, _ = (
                            ensemble.predict(
                                predicted_state,
                                action,
                            )
                        )

                        predicted_path.append(
                            predicted_state[:2].copy()
                        )

                        phase += (
                            TARGET_OMEGA * DT
                        )

                if mode == "SAC+MPC+Safety":

                    executed_action = safety_filter(
                        env.state,
                        requested_action,
                    )

                else:

                    executed_action = (
                        requested_action.copy()
                    )

                # Ensemble disagreement diagnostic.
                _, uncertainty = ensemble.predict(
                    env.state,
                    executed_action,
                )

                uncertainty_value = float(
                    np.linalg.norm(
                        uncertainty
                    )
                )

                (
                    observation,
                    reward,
                    terminated,
                    truncated,
                    info,
                ) = env.step(
                    executed_action
                )

                total_reward += reward

                trail.append(
                    env.state[:2].copy()
                )

                if len(trail) > 220:

                    trail.pop(0)

                if terminated or truncated:

                    observation = env.reset()

                    trail.clear()

                    predicted_path.clear()

                    mpc.last_plan = None

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
            (180, 90, 90),
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

        if len(predicted_path) > 1:

            pygame.draw.lines(
                screen,
                (255, 210, 65),
                False,
                [
                    point.astype(int)
                    for point in predicted_path
                ],
                2,
            )

            for index, point in enumerate(
                predicted_path
            ):

                if index % 3 == 0:

                    pygame.draw.circle(
                        screen,
                        (255, 210, 65),
                        point.astype(int),
                        3,
                    )

        draw_drone()

        distance = np.linalg.norm(
            env.state[:2]
            - env.target.position()
        )

        draw_text(
            "AUTONOMOUS DRONE LEARNING SYSTEM",
            15,
            12,
        )

        draw_text(
            f"Mode: {modes[mode_index]}",
            15,
            38,
        )

        draw_text(
            f"Target distance: {distance:.2f}",
            15,
            64,
        )

        draw_text(
            f"Model uncertainty: {uncertainty_value:.3f}",
            15,
            90,
        )

        draw_text(
            f"Motor gain: {env.motor_gain:.2f}",
            15,
            116,
        )

        draw_text(
            f"Simulation speed: {speed}x",
            15,
            142,
        )

        draw_text(
            "Red: requested action",
            15,
            180,
            (245, 100, 100),
        )

        draw_text(
            "Blue: executed action",
            15,
            202,
            (100, 170, 255),
        )

        draw_text(
            "Yellow: predicted MPC path",
            15,
            224,
            (255, 215, 90),
        )

        draw_text(
            "1 SAC | 2 SAC+MPC | 3 SAC+MPC+Safety",
            15,
            WORLD_H - 70,
        )

        draw_text(
            "SPACE pause | R reset | +/- speed | D change dynamics | ESC exit",
            15,
            WORLD_H - 42,
        )

        if paused:

            draw_text(
                "PAUSED",
                WORLD_W - 110,
                15,
                (255, 210, 80),
            )

        pygame.display.flip()

    pygame.quit()


# ============================================================
# 14. UNIT TESTS
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

    # Test 1 — Observation dimensions.
    env = DroneEnv()

    observation = env.reset()

    check(
        observation.shape == (OBS_DIM,),
        "Observation dimension is correct",
    )

    # Test 2 — Observation finite.
    check(
        np.all(
            np.isfinite(observation)
        ),
        "Observation values are finite",
    )

    # Test 3 — Action bounds.
    action = np.array(
        [2.0, -3.0]
    )

    clipped = np.clip(
        action,
        -1.0,
        1.0,
    )

    check(
        np.all(
            np.abs(clipped) <= 1.0
        ),
        "Continuous action clipping works",
    )

    # Test 4 — Physical transition shape.
    next_state = drone_transition(
        np.array(
            [100.0, 100.0, 0.0, 0.0]
        ),
        np.array(
            [1.0, 0.0]
        ),
    )

    check(
        next_state.shape == (4,),
        "Physical transition returns four state values",
    )

    # Test 5 — Speed limit.
    fast_state = clip_state(
        np.array(
            [100.0, 100.0, 500.0, 500.0]
        )
    )

    check(
        np.linalg.norm(
            fast_state[2:4]
        ) <= MAX_SPEED + 1e-5,
        "Speed limit is enforced",
    )

    # Test 6 — Moving target.
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
        "Target position changes with time",
    )

    # Test 7 — Target velocity.
    check(
        target.velocity().shape == (2,),
        "Target velocity is two-dimensional",
    )

    # Test 8 — Environment step.
    result = env.step(
        np.array(
            [0.0, 0.0]
        )
    )

    check(
        len(result) == 5,
        "Environment returns five step values",
    )

    # Test 9 — Actor output.
    actor = Actor().to(
        DEVICE
    )

    state_tensor = torch.as_tensor(
        observation,
        dtype=torch.float32,
        device=DEVICE,
    ).unsqueeze(0)

    action_tensor, log_prob, deterministic = (
        actor.sample(
            state_tensor
        )
    )

    check(
        action_tensor.shape == (1, ACTION_DIM),
        "SAC actor returns two-dimensional action",
    )

    # Test 10 — Actor action bounds.
    check(
        bool(
            torch.all(
                action_tensor.abs() <= 1.0
            )
        ),
        "Tanh actor action is bounded",
    )

    # Test 11 — Log probability.
    check(
        log_prob.shape == (1, 1)
        and bool(
            torch.isfinite(log_prob).all()
        ),
        "SAC log probability is finite",
    )

    # Test 12 — Critic outputs.
    critic = Critic().to(
        DEVICE
    )

    q1, q2 = critic(
        state_tensor,
        action_tensor,
    )

    check(
        q1.shape == (1, 1)
        and q2.shape == (1, 1),
        "Twin critic returns scalar Q values",
    )

    # Test 13 — Dynamics prediction.
    model = DynamicsModel().to(
        DEVICE
    )

    physical = torch.zeros(
        (1, 4),
        device=DEVICE,
    )

    predicted_delta = model(
        physical,
        action_tensor,
    )

    check(
        predicted_delta.shape == (1, 4),
        "Dynamics model predicts four state deltas",
    )

    # Test 14 — Ensemble output.
    ensemble = DynamicsEnsemble()

    predictions = ensemble.predict_batch(
        physical,
        action_tensor,
    )

    check(
        predictions.shape == (
            ENSEMBLE_SIZE,
            1,
            4,
        ),
        "Ensemble returns predictions from every model",
    )

    # Test 15 — Ensemble mean.
    mean = predictions.mean(
        dim=0
    )

    check(
        mean.shape == (1, 4),
        "Ensemble mean has correct shape",
    )

    # Test 16 — Replay buffer.
    replay = ReplayBuffer()

    replay.add(
        observation,
        np.zeros(2),
        1.0,
        observation,
        False,
        np.zeros(4),
        np.zeros(4),
    )

    check(
        len(replay) == 1,
        "Replay buffer stores transition",
    )

    # Test 17 — Replay sampling.
    batch = replay.sample(
        1
    )

    check(
        len(batch) == 7,
        "Replay sampling returns seven fields",
    )

    # Test 18 — Safety filter.
    filtered = safety_filter(
        np.array(
            [100.0, 100.0, 0.0, 0.0]
        ),
        np.array(
            [0.5, -0.5]
        ),
    )

    check(
        filtered.shape == (2,)
        and np.all(
            np.isfinite(filtered)
        ),
        "Safety filter returns finite action",
    )

    # Test 19 — Safety filter bounds.
    check(
        np.all(
            np.abs(filtered) <= 1.0
        ),
        "Safety filter preserves action bounds",
    )

    # Test 20 — SAC alpha positivity.
    agent = SACAgent()

    check(
        float(
            agent.alpha.detach().cpu()
        ) > 0.0,
        "Automatic entropy temperature is positive",
    )

    # Test 21 — MPC construction.
    mpc = MPCController(
        agent.actor,
        ensemble,
    )

    check(
        mpc.last_plan is None,
        "MPC initializes without previous plan",
    )

    # Test 22 — Target future observation.
    future_obs = mpc.future_observation(
        np.array(
            [100.0, 100.0, 0.0, 0.0]
        ),
        0.5,
        np.array(
            [700.0, 300.0]
        ),
    )

    check(
        future_obs.shape == (OBS_DIM,),
        "MPC constructs moving-target observation",
    )

    # Test 23 — Target phase changes.
    check(
        not np.allclose(
            future_obs[-2:],
            [
                math.sin(0.5 + TARGET_OMEGA * DT),
                math.cos(0.5 + TARGET_OMEGA * DT),
            ],
        ),
        "MPC observation uses the supplied target phase",
    )

    # Test 24 — Reward finite.
    check(
        np.isfinite(result[1]),
        "Environment reward is finite",
    )

    print(
        f"\n{passed}/24 tests passed."
    )

    return passed == 24


# ============================================================
# 15. MAIN
# ============================================================

def main():

    print(
        "=" * 72
    )

    print(
        "PHASE 4.31 — AUTONOMOUS DRONE LEARNING SYSTEM"
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
        "\n[1/4] Training SAC and learned dynamics..."
    )

    (
        agent,
        ensemble,
        replay,
        history,
    ) = train_system()

    print(
        "\n[2/4] Evaluating integrated controllers..."
    )

    print_evaluation(
        agent,
        ensemble,
    )

    print(
        "\n[3/4] Plotting training history..."
    )

    plot_training(
        history
    )

    print(
        "\n[4/4] Starting interactive drone visualization..."
    )

    run_pygame(
        agent,
        ensemble,
    )

    print(
        "\nPhase 4.31 completed."
    )


if __name__ == "__main__":

    main()
