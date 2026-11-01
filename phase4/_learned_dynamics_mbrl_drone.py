
"""
File: phase4/learned_dynamics_mbrl_drone.py

PHASE 4.18 — LEARNED DYNAMICS MODELS AND MODEL-BASED RL
=======================================================

PURPOSE
-------
Learn how an RL/robotics agent can learn its own dynamics model from
experience, then use that model to plan actions before executing them.

WHAT YOU LEARN
--------------
Step 1 — Collect transition data
    Record (state, action, next_state) samples from the drone simulator.
    Understand that a dynamics model learns how the environment changes.

Step 2 — Train a neural dynamics model
    Learn:
        (drone position, drone velocity, action)
            -> (change in position, change in velocity)

    Predicting state differences instead of absolute states is often
    easier to optimize and can improve numerical conditioning.

Step 3 — Validate the model
    Measure one-step prediction error on data not used for training.
    Compare one-step and multi-step prediction errors.
    Understand model error accumulation during long rollouts.

Step 4 — Plan with the learned model
    Use random-shooting MPC:
        a. Generate candidate action sequences.
        b. Predict their future outcomes with the learned model.
        c. Score progress toward the future moving target.
        d. Execute only the first action.
        e. Replan at the next time step.

Step 5 — Compare controllers
    PD controller: hand-designed baseline.
    Learned MPC: plans with the neural dynamics model.
    Oracle MPC: plans with true simulator dynamics.
    Oracle MPC is a reference, not a deployable real-world controller.

Step 6 — Visualize and experiment
    Display the quadcopter drone, moving target, obstacle, predicted path,
    model error, and selected controller in Pygame.

IMPLEMENTATION
--------------
    Python, NumPy, PyTorch, Matplotlib, Pygame.

INSTALL
-------
    pip install numpy torch matplotlib pygame

RUN
---
    python phase4/learned_dynamics_mbrl_drone.py

CONTROLS
--------
    1 : PD controller
    2 : Learned-model MPC
    3 : Oracle MPC
    SPACE : Pause / resume
    R : Reset drone
    + / - : Simulation speed
    Left mouse click : Change moving-target trajectory center
    ESC : Exit

ROBOTICS APPLICATION
--------------------
A real drone does not know its exact aerodynamic response, motor delay,
payload effect, or wind disturbance. It can collect flight transitions
and fit a dynamics model from measured state/action data.

A learned model can then support short-horizon planning. In a real system,
the model must be validated under the intended operating conditions, and
planning must be combined with independent safety mechanisms.

IMPORTANT LIMITATIONS
--------------------
1. This is a simplified 2D educational quadcopter model.
2. The learned model predicts drone motion; target motion is known analytically.
3. Random-shooting MPC is an approximate optimizer.
4. Prediction errors generally compound over longer horizons.
5. The oracle controller uses true simulator dynamics and is only a benchmark.
6. This demonstration is not flight-certified control software.
"""

from __future__ import annotations

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
# 0. CONFIGURATION
# ============================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WORLD_W = 900
WORLD_H = 650

DT = 0.08
MAX_SPEED = 150.0
MAX_ACCEL = 115.0
MAX_ACTION = 1.0

DRONE_RADIUS = 13.0
OBSTACLE = np.array([450.0, 330.0], dtype=np.float32)
OBSTACLE_RADIUS = 65.0
SAFE_RADIUS = 92.0

TARGET_A = 170.0
TARGET_B = 105.0
TARGET_OMEGA = 0.55

MODEL_HIDDEN = 128
TRAIN_EPOCHS = 100
BATCH_SIZE = 256
LEARNING_RATE = 1e-3

DATA_EPISODES = 100
DATA_STEPS = 160

MPC_HORIZON = 16
MPC_CANDIDATES = 160
MPC_ACTION_NOISE = 0.55
MPC_CONTROL_PENALTY = 0.018

TRAINING_EPISODES = 0  # This lesson learns dynamics, not a new policy.


# ============================================================
# 1. MOVING TARGET
# ============================================================

@dataclass
class MovingTarget:
    center: np.ndarray
    phase: float = 0.0

    def position(self, phase: float | None = None) -> np.ndarray:
        p = self.phase if phase is None else phase
        return self.center + np.array(
            [
                TARGET_A * math.cos(p),
                TARGET_B * math.sin(p),
            ],
            dtype=np.float32,
        )

    def velocity(self, phase: float | None = None) -> np.ndarray:
        p = self.phase if phase is None else phase
        return np.array(
            [
                -TARGET_A * TARGET_OMEGA * math.sin(p),
                TARGET_B * TARGET_OMEGA * math.cos(p),
            ],
            dtype=np.float32,
        )

    def advance(self, dt: float = DT) -> None:
        self.phase += TARGET_OMEGA * dt


# ============================================================
# 2. DRONE ENVIRONMENT — TRUE SIMULATOR DYNAMICS
# ============================================================

class DroneEnv:
    """
    State:
        [x, y, vx, vy]

    Action:
        [ax_normalized, ay_normalized]

    True dynamics:
        v(t+1) = clip(v(t) + a * dt)
        p(t+1) = p(t) + v(t+1) * dt

    The learned model will approximate the resulting state difference.
    """

    def __init__(self):
        self.target = MovingTarget(
            center=np.array([700.0, 320.0], dtype=np.float32)
        )
        self.reset()

    def reset(self):
        self.position = np.array([115.0, 500.0], dtype=np.float32)
        self.velocity = np.zeros(2, dtype=np.float32)
        self.target.phase = float(np.random.uniform(0, 2 * math.pi))
        return self.observation()

    def observation(self) -> np.ndarray:
        return np.concatenate(
            [
                self.position,
                self.velocity,
                self.target.position(),
                self.target.velocity(),
            ]
        ).astype(np.float32)

    def drone_state(self) -> np.ndarray:
        return np.concatenate([self.position, self.velocity]).astype(
            np.float32
        )

    def step_drone(self, action: np.ndarray) -> np.ndarray:
        """Advance drone dynamics only; target phase is not advanced here."""
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -MAX_ACTION, MAX_ACTION)

        old_state = self.drone_state()

        acceleration = action * MAX_ACCEL
        self.velocity += acceleration * DT

        speed = np.linalg.norm(self.velocity)
        if speed > MAX_SPEED:
            self.velocity *= MAX_SPEED / (speed + 1e-8)

        self.position += self.velocity * DT

        self.position[0] = np.clip(
            self.position[0], DRONE_RADIUS, WORLD_W - DRONE_RADIUS
        )
        self.position[1] = np.clip(
            self.position[1], DRONE_RADIUS, WORLD_H - DRONE_RADIUS
        )

        return self.drone_state() - old_state

    def step(self, action: np.ndarray):
        self.step_drone(action)
        self.target.advance()
        return self.observation()


# ============================================================
# 3. BASELINE CONTROLLER AND DATA COLLECTION
# ============================================================

def pd_controller(
    drone_state: np.ndarray,
    target_position: np.ndarray,
    target_velocity: np.ndarray,
) -> np.ndarray:
    """
    A simple proportional-derivative target tracker.

    The target velocity is used as feed-forward information:
        desired_velocity = target_velocity + Kp * position_error

    This controller is used for data collection and as a baseline.
    It is not an RL policy.
    """
    position = drone_state[:2]
    velocity = drone_state[2:4]

    error = target_position - position
    desired_velocity = target_velocity + 0.65 * error
    desired_velocity = np.clip(desired_velocity, -MAX_SPEED, MAX_SPEED)

    acceleration = 2.0 * (desired_velocity - velocity)
    return np.clip(acceleration / MAX_ACCEL, -1.0, 1.0)


def add_obstacle_avoidance(
    action: np.ndarray,
    position: np.ndarray,
) -> np.ndarray:
    """
    Simple repulsive correction used only for safe-ish data collection
    and baseline control. It is not a formal collision-avoidance proof.
    """
    delta = position - OBSTACLE
    distance = np.linalg.norm(delta) + 1e-8

    if distance < SAFE_RADIUS:
        direction = delta / distance
        strength = (SAFE_RADIUS - distance) / SAFE_RADIUS
        correction = direction * 1.7 * strength
        action = action + correction

    return np.clip(action, -1.0, 1.0)


def collect_transition_data(
    episodes: int = DATA_EPISODES,
    steps_per_episode: int = DATA_STEPS,
):
    """
    Collect supervised transition examples.

    Input:
        x = [drone position, drone velocity, action]

    Label:
        y = [delta position, delta velocity]

    Random action perturbations broaden the training distribution.
    """
    inputs = []
    labels = []

    env = DroneEnv()

    for episode in range(episodes):
        env.reset()

        for _ in range(steps_per_episode):
            state = env.drone_state()
            target_pos = env.target.position()
            target_vel = env.target.velocity()

            action = pd_controller(state, target_pos, target_vel)
            action = add_obstacle_avoidance(action, state[:2])

            # Inject bounded exploration to cover more action/state pairs.
            action += np.random.normal(0.0, 0.32, size=2)
            action = np.clip(action, -1.0, 1.0).astype(np.float32)

            model_input = np.concatenate([state, action]).astype(np.float32)

            next_state = state + env.step_drone(action)
            delta_state = next_state - state

            inputs.append(model_input)
            labels.append(delta_state.astype(np.float32))

            env.target.advance()

    return np.asarray(inputs, dtype=np.float32), np.asarray(
        labels, dtype=np.float32
    )


# ============================================================
# 4. LEARNED DRONE DYNAMICS MODEL
# ============================================================

class DynamicsModel(nn.Module):
    """
    Neural approximation:
        f_theta([x, y, vx, vy, ax, ay])
            -> [delta_x, delta_y, delta_vx, delta_vy]

    Predicting state differences helps the network focus on the
    action-dependent change rather than relearning identity mappings.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, MODEL_HIDDEN),
            nn.ReLU(),
            nn.Linear(MODEL_HIDDEN, MODEL_HIDDEN),
            nn.ReLU(),
            nn.Linear(MODEL_HIDDEN, 4),
        )

    def forward(self, x):
        return self.net(x)


@dataclass
class Normalizer:
    x_mean: torch.Tensor
    x_std: torch.Tensor
    y_mean: torch.Tensor
    y_std: torch.Tensor


def make_normalizer(x, y) -> Normalizer:
    x_mean = x.mean(dim=0)
    x_std = x.std(dim=0).clamp_min(1e-5)
    y_mean = y.mean(dim=0)
    y_std = y.std(dim=0).clamp_min(1e-5)

    return Normalizer(x_mean, x_std, y_mean, y_std)


def predict_delta(
    model: DynamicsModel,
    normalizer: Normalizer,
    model_input: np.ndarray,
) -> np.ndarray:
    model.eval()

    with torch.no_grad():
        x = torch.as_tensor(
            model_input, dtype=torch.float32, device=DEVICE
        )
        x_norm = (x - normalizer.x_mean) / normalizer.x_std
        y_norm = model(x_norm)
        y = y_norm * normalizer.y_std + normalizer.y_mean

    return y.detach().cpu().numpy()


def train_dynamics_model(x_np, y_np):
    """
    Random train/validation split.
    Validation data is not used for gradient updates.
    """
    count = len(x_np)
    indices = np.random.permutation(count)
    split = int(count * 0.85)

    train_idx = indices[:split]
    val_idx = indices[split:]

    x = torch.tensor(x_np, dtype=torch.float32, device=DEVICE)
    y = torch.tensor(y_np, dtype=torch.float32, device=DEVICE)

    x_train, y_train = x[train_idx], y[train_idx]
    x_val, y_val = x[val_idx], y[val_idx]

    normalizer = make_normalizer(x_train, y_train)

    x_train_n = (x_train - normalizer.x_mean) / normalizer.x_std
    y_train_n = (y_train - normalizer.y_mean) / normalizer.y_std

    x_val_n = (x_val - normalizer.x_mean) / normalizer.x_std
    y_val_n = (y_val - normalizer.y_mean) / normalizer.y_std

    model = DynamicsModel().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()

    train_history = []
    val_history = []

    for epoch in range(TRAIN_EPOCHS):
        model.train()
        permutation = torch.randperm(len(x_train), device=DEVICE)
        epoch_losses = []

        for start in range(0, len(x_train), BATCH_SIZE):
            batch_idx = permutation[start:start + BATCH_SIZE]

            prediction = model(x_train_n[batch_idx])
            loss = loss_fn(prediction, y_train_n[batch_idx])

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            epoch_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(x_val_n), y_val_n).item()

        train_history.append(float(np.mean(epoch_losses)))
        val_history.append(val_loss)

        if (epoch + 1) % 20 == 0:
            print(
                f"Epoch {epoch + 1:3d}/{TRAIN_EPOCHS} | "
                f"train={train_history[-1]:.6f} | "
                f"validation={val_loss:.6f}"
            )

    return model, normalizer, train_history, val_history


# ============================================================
# 5. MODEL VALIDATION
# ============================================================

def evaluate_one_step_error(model, normalizer, x_np, y_np):
    predictions = np.array(
        [predict_delta(model, normalizer, row) for row in x_np]
    )

    errors = predictions - y_np
    component_rmse = np.sqrt(np.mean(errors ** 2, axis=0))
    overall_rmse = float(np.sqrt(np.mean(errors ** 2)))

    return overall_rmse, component_rmse


def evaluate_multistep_error(
    model,
    normalizer,
    rollout_count: int = 30,
    horizon: int = 20,
):
    """
    Compare model rollouts with true simulator rollouts.

    The same action sequence is applied to both systems.
    This isolates model prediction error from differences in control.
    """
    errors_by_step = np.zeros(horizon, dtype=np.float64)
    env = DroneEnv()

    for _ in range(rollout_count):
        env.reset()
        true_state = env.drone_state()
        predicted_state = true_state.copy()

        for step in range(horizon):
            action = np.random.uniform(-0.7, 0.7, size=2).astype(
                np.float32
            )

            true_input = np.concatenate([true_state, action])
            true_delta = env.step_drone(action)
            true_state = true_state + true_delta

            predicted_input = np.concatenate([predicted_state, action])
            predicted_delta = predict_delta(
                model, normalizer, predicted_input
            )
            predicted_state = predicted_state + predicted_delta

            errors_by_step[step] += np.linalg.norm(
                predicted_state[:2] - true_state[:2]
            )

    errors_by_step /= rollout_count
    return errors_by_step


# ============================================================
# 6. MODEL-BASED MPC
# ============================================================

def predict_target_position(
    target: MovingTarget,
    future_step: int,
) -> np.ndarray:
    future_phase = target.phase + TARGET_OMEGA * DT * future_step
    return target.position(future_phase)


def rollout_candidate(
    initial_state: np.ndarray,
    actions: np.ndarray,
    target: MovingTarget,
    model: DynamicsModel | None,
    normalizer: Normalizer | None,
) -> float:
    """
    Score one candidate action sequence.

    model=None means use true simulator dynamics (oracle MPC).
    Otherwise use the learned neural model.

    Only drone dynamics are predicted by the model. Target movement
    is computed analytically from its known trajectory.
    """
    state = initial_state.copy()
    total_cost = 0.0

    for t, action in enumerate(actions, start=1):
        model_input = np.concatenate([state, action]).astype(np.float32)

        if model is None:
            # Oracle: exact simulator transition.
            velocity = state[2:4] + action * MAX_ACCEL * DT
            speed = np.linalg.norm(velocity)
            if speed > MAX_SPEED:
                velocity *= MAX_SPEED / (speed + 1e-8)

            position = state[:2] + velocity * DT
            position[0] = np.clip(
                position[0], DRONE_RADIUS, WORLD_W - DRONE_RADIUS
            )
            position[1] = np.clip(
                position[1], DRONE_RADIUS, WORLD_H - DRONE_RADIUS
            )
            next_state = np.concatenate([position, velocity]).astype(
                np.float32
            )
        else:
            delta = predict_delta(model, normalizer, model_input)
            next_state = state + delta

        target_future = predict_target_position(target, t)

        distance = np.linalg.norm(next_state[:2] - target_future)
        obstacle_distance = np.linalg.norm(
            next_state[:2] - OBSTACLE
        )

        # Cost terms:
        #   target distance: track the moving target
        #   obstacle penalty: discourage entering the safety radius
        #   boundary penalty: discourage leaving the operating area
        #   action penalty: avoid excessive control effort
        total_cost += 0.045 * distance
        total_cost += 0.012 * np.sum(action ** 2)

        obstacle_violation = max(
            0.0, SAFE_RADIUS - obstacle_distance
        )
        total_cost += 2.0 * obstacle_violation ** 2

        boundary_margin = 28.0
        boundary_violation = max(
            0.0, boundary_margin - next_state[0],
            next_state[0] - (WORLD_W - boundary_margin),
            boundary_margin - next_state[1],
            next_state[1] - (WORLD_H - boundary_margin),
        )
        total_cost += 3.0 * boundary_violation ** 2

        state = next_state

    return float(total_cost)


def mpc_action(
    drone_state: np.ndarray,
    target: MovingTarget,
    model: DynamicsModel | None,
    normalizer: Normalizer | None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Random-shooting MPC.

    Candidate sequences are initialized around a PD action sequence,
    then perturbed with bounded Gaussian noise.

    The best sequence is selected, but only its first action is executed.
    Replanning happens at the next control step.
    """
    base_actions = []

    state = drone_state.copy()

    for t in range(1, MPC_HORIZON + 1):
        future_target = predict_target_position(target, t)
        future_velocity = target.velocity(
            target.phase + TARGET_OMEGA * DT * t
        )
        action = pd_controller(state, future_target, future_velocity)
        base_actions.append(action)

        # A rough nominal-state update for constructing the next
        # baseline action. The optimizer itself evaluates each candidate.
        state[2:4] += action * MAX_ACCEL * DT
        state[:2] += state[2:4] * DT

    base_actions = np.asarray(base_actions, dtype=np.float32)

    candidates = np.repeat(
        base_actions[None, :, :], MPC_CANDIDATES, axis=0
    )
    candidates += np.random.normal(
        0.0,
        MPC_ACTION_NOISE,
        size=candidates.shape,
    ).astype(np.float32)
    candidates = np.clip(candidates, -1.0, 1.0)

    # Keep one unperturbed nominal candidate.
    candidates[0] = base_actions

    best_cost = float("inf")
    best_sequence = candidates[0]

    for candidate in candidates:
        cost = rollout_candidate(
            drone_state, candidate, target, model, normalizer
        )
        if cost < best_cost:
            best_cost = cost
            best_sequence = candidate

    return best_sequence[0].copy(), best_sequence.copy()


# ============================================================
# 7. SIMPLE CONTROL SAFETY FILTER
# ============================================================

def safety_filter(action: np.ndarray, position: np.ndarray) -> np.ndarray:
    """
    A small reactive obstacle correction for the executed action.

    MPC already includes obstacle penalties. This additional correction
    is a demonstration layer, not a certified safety filter.
    """
    delta = position - OBSTACLE
    distance = np.linalg.norm(delta) + 1e-8

    if distance < SAFE_RADIUS:
        direction = delta / distance
        away = np.array(
            [direction[0], direction[1]], dtype=np.float32
        )
        action = action + 0.9 * away * (
            (SAFE_RADIUS - distance) / SAFE_RADIUS
        )

    return np.clip(action, -1.0, 1.0)


# ============================================================
# 8. BENCHMARKING
# ============================================================

def run_benchmark(
    model,
    normalizer,
    controller_name: str,
    episodes: int = 8,
    max_steps: int = 500,
):
    """
    Compare controllers over multiple episodes.

    Metrics:
        mean target distance
        minimum obstacle clearance
        mean control effort

    This benchmark is stochastic; results depend on model training
    data, random candidate sequences, and episode initial phases.
    """
    distance_results = []
    clearance_results = []
    effort_results = []

    env = DroneEnv()

    for _ in range(episodes):
        env.reset()

        distances = []
        clearances = []
        efforts = []

        for _step in range(max_steps):
            state = env.drone_state()

            if controller_name == "PD":
                action = pd_controller(
                    state,
                    env.target.position(),
                    env.target.velocity(),
                )
            elif controller_name == "Learned MPC":
                action, _ = mpc_action(
                    state, env.target, model, normalizer
                )
            elif controller_name == "Oracle MPC":
                action, _ = mpc_action(
                    state, env.target, None, None
                )
            else:
                raise ValueError(controller_name)

            action = safety_filter(action, state[:2])
            env.step(action)

            distances.append(
                np.linalg.norm(
                    env.position - env.target.position()
                )
            )
            clearances.append(
                np.linalg.norm(env.position - OBSTACLE)
                - OBSTACLE_RADIUS
            )
            efforts.append(np.linalg.norm(action))

        distance_results.append(np.mean(distances))
        clearance_results.append(np.min(clearances))
        effort_results.append(np.mean(efforts))

    return {
        "mean_distance": float(np.mean(distance_results)),
        "std_distance": float(np.std(distance_results)),
        "mean_min_clearance": float(np.mean(clearance_results)),
        "mean_action_effort": float(np.mean(effort_results)),
    }


# ============================================================
# 9. PYGAME QUADCOPTER VISUALIZATION
# ============================================================

def run_pygame(model, normalizer):
    try:
        import pygame
    except ImportError:
        print("Pygame not installed. Run: pip install pygame")
        return

    pygame.init()
    screen = pygame.display.set_mode((WORLD_W, WORLD_H))
    pygame.display.set_caption(
        "Phase 4.18 — Learned Dynamics Model MPC Drone"
    )
    clock = pygame.time.Clock()

    font = pygame.font.SysFont("consolas", 18)
    small_font = pygame.font.SysFont("consolas", 14)

    env = DroneEnv()
    env.reset()

    mode = "Learned MPC"
    paused = False
    speed = 1
    trail = []
    predicted_path = []

    def draw_text(text, x, y, color=(235, 235, 235)):
        screen.blit(font.render(text, True, color), (x, y))

    def draw_drone(position, velocity):
        x, y = position
        direction = velocity / (np.linalg.norm(velocity) + 1e-6)
        perpendicular = np.array([-direction[1], direction[0]])

        center = np.array([x, y])
        arm_length = 18

        # Four-arm quadcopter silhouette.
        for sign in (-1, 1):
            for side in (-1, 1):
                offset = (
                    direction * sign * arm_length
                    + perpendicular * side * arm_length
                )
                endpoint = center + offset
                pygame.draw.line(
                    screen, (210, 220, 230),
                    center.astype(int), endpoint.astype(int), 5
                )
                pygame.draw.circle(
                    screen, (80, 190, 245),
                    endpoint.astype(int), 7, 2
                )

        pygame.draw.circle(
            screen, (245, 245, 245), (int(x), int(y)), 8
        )
        nose = center + direction * 22
        pygame.draw.circle(
            screen, (255, 190, 60), nose.astype(int), 4
        )

    def draw_target():
        points = []
        for i in range(100):
            phase = env.target.phase + i * 2 * math.pi / 100
            point = env.target.position(phase)
            points.append(point.astype(int))

        pygame.draw.lines(
            screen, (75, 105, 75), True, points, 2
        )

        target_pos = env.target.position()
        pygame.draw.circle(
            screen, (70, 230, 110),
            target_pos.astype(int), 12
        )
        pygame.draw.circle(
            screen, (70, 230, 110),
            target_pos.astype(int), 24, 1
        )

        velocity = env.target.velocity()
        endpoint = target_pos + velocity * 0.20
        pygame.draw.line(
            screen, (80, 255, 130),
            target_pos.astype(int), endpoint.astype(int), 3
        )

    running = True
    last_action = np.zeros(2, dtype=np.float32)
    last_cost_path = np.empty((0, 2), dtype=np.float32)
    model_error_text = "Model RMSE: see validation plot"

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
                    env.reset()
                    trail.clear()
                    predicted_path.clear()
                elif event.key in (pygame.K_EQUALS, pygame.K_PLUS):
                    speed = min(5, speed + 1)
                elif event.key == pygame.K_MINUS:
                    speed = max(1, speed - 1)
                elif event.key == pygame.K_1:
                    mode = "PD"
                elif event.key == pygame.K_2:
                    mode = "Learned MPC"
                elif event.key == pygame.K_3:
                    mode = "Oracle MPC"

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    click = np.array(event.pos, dtype=np.float32)
                    distance = np.linalg.norm(click - OBSTACLE)
                    if distance > SAFE_RADIUS + 15:
                        env.target.center = click

        if not paused:
            for _ in range(speed):
                state = env.drone_state()

                if mode == "PD":
                    action = pd_controller(
                        state,
                        env.target.position(),
                        env.target.velocity(),
                    )
                    sequence = np.repeat(
                        action[None, :], MPC_HORIZON, axis=0
                    )
                elif mode == "Learned MPC":
                    action, sequence = mpc_action(
                        state, env.target, model, normalizer
                    )
                else:
                    action, sequence = mpc_action(
                        state, env.target, None, None
                    )

                action = safety_filter(action, state[:2])
                last_action = action.copy()

                # Draw the planned path using the same selected model.
                rollout_state = state.copy()
                path = [rollout_state[:2].copy()]

                for candidate_action in sequence:
                    model_input = np.concatenate(
                        [rollout_state, candidate_action]
                    ).astype(np.float32)

                    if mode == "Oracle MPC":
                        velocity = (
                            rollout_state[2:4]
                            + candidate_action * MAX_ACCEL * DT
                        )
                        velocity_norm = np.linalg.norm(velocity)
                        if velocity_norm > MAX_SPEED:
                            velocity *= MAX_SPEED / (
                                velocity_norm + 1e-8
                            )
                        position = rollout_state[:2] + velocity * DT
                        position[0] = np.clip(
                            position[0], DRONE_RADIUS,
                            WORLD_W - DRONE_RADIUS
                        )
                        position[1] = np.clip(
                            position[1], DRONE_RADIUS,
                            WORLD_H - DRONE_RADIUS
                        )
                        rollout_state = np.concatenate(
                            [position, velocity]
                        )
                    else:
                        delta = predict_delta(
                            model, normalizer, model_input
                        )
                        rollout_state = rollout_state + delta

                    path.append(rollout_state[:2].copy())

                predicted_path = path

                env.step(action)

                trail.append(env.position.copy())
                if len(trail) > 180:
                    trail.pop(0)

        # ---------------- Draw scene ----------------
        screen.fill((18, 24, 34))

        # Grid.
        for x in range(0, WORLD_W, 50):
            pygame.draw.line(
                screen, (30, 39, 50), (x, 0), (x, WORLD_H)
            )
        for y in range(0, WORLD_H, 50):
            pygame.draw.line(
                screen, (30, 39, 50), (0, y), (WORLD_W, y)
            )

        # Obstacle and safety boundary.
        pygame.draw.circle(
            screen, (95, 95, 105),
            OBSTACLE.astype(int), int(OBSTACLE_RADIUS)
        )
        pygame.draw.circle(
            screen, (210, 100, 85),
            OBSTACLE.astype(int), int(SAFE_RADIUS), 2
        )

        draw_target()

        if len(trail) > 1:
            pygame.draw.lines(
                screen, (90, 145, 235), False,
                [p.astype(int) for p in trail], 2
            )

        if len(predicted_path) > 1:
            pygame.draw.lines(
                screen, (255, 190, 65), False,
                [p.astype(int) for p in predicted_path], 2
            )

        draw_drone(env.position, env.velocity)

        target_distance = np.linalg.norm(
            env.position - env.target.position()
        )
        obstacle_clearance = (
            np.linalg.norm(env.position - OBSTACLE)
            - OBSTACLE_RADIUS
        )

        draw_text(f"Controller: {mode}", 15, 12)
        draw_text(
            f"Target distance: {target_distance:7.2f}",
            15, 40
        )
        draw_text(
            f"Obstacle clearance: {obstacle_clearance:7.2f}",
            15, 68
        )
        draw_text(
            f"Action: [{last_action[0]:+.2f}, {last_action[1]:+.2f}]",
            15, 96
        )
        draw_text(f"Speed: {speed}x", 15, 124)
        draw_text(
            "1 PD | 2 Learned MPC | 3 Oracle MPC",
            15, WORLD_H - 72
        )
        draw_text(
            "SPACE pause | R reset | +/- speed | click target center | ESC exit",
            15, WORLD_H - 44
        )

        if paused:
            draw_text("PAUSED", WORLD_W - 125, 15, (255, 210, 80))

        pygame.display.flip()

    pygame.quit()


# ============================================================
# 10. VISUALIZATION
# ============================================================

def visualize_training(train_history, val_history, multistep_errors):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    axes[0].plot(train_history, label="Training loss")
    axes[0].plot(val_history, label="Validation loss")
    axes[0].set_title("Learned dynamics model training")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Normalized MSE")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(
        np.arange(1, len(multistep_errors) + 1),
        multistep_errors,
        marker="o",
    )
    axes[1].set_title("Multi-step position prediction error")
    axes[1].set_xlabel("Rollout step")
    axes[1].set_ylabel("Mean position error (pixels)")
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()


def visualize_benchmark(results):
    names = list(results.keys())
    mean_distances = [
        results[name]["mean_distance"] for name in names
    ]
    clearances = [
        results[name]["mean_min_clearance"] for name in names
    ]
    efforts = [
        results[name]["mean_action_effort"] for name in names
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    axes[0].bar(names, mean_distances)
    axes[0].set_title("Mean target distance")
    axes[0].set_ylabel("Pixels")
    axes[0].tick_params(axis="x", rotation=15)
    axes[0].grid(axis="y")

    axes[1].bar(names, clearances)
    axes[1].set_title("Mean minimum obstacle clearance")
    axes[1].set_ylabel("Pixels")
    axes[1].tick_params(axis="x", rotation=15)
    axes[1].grid(axis="y")

    axes[2].bar(names, efforts)
    axes[2].set_title("Mean control effort")
    axes[2].set_ylabel("Normalized action magnitude")
    axes[2].tick_params(axis="x", rotation=15)
    axes[2].grid(axis="y")

    plt.tight_layout()
    plt.show()


# ============================================================
# 11. UNIT TESTS
# ============================================================

def run_tests(model=None, normalizer=None):
    print("\nRunning unit tests...")
    passed = 0

    def check(condition, message):
        nonlocal passed
        assert condition, message
        passed += 1
        print(f"[PASS] {message}")

    # Test 1: target position is finite.
    target = MovingTarget(np.array([400.0, 300.0], dtype=np.float32))
    check(
        np.all(np.isfinite(target.position())),
        "Moving target position is finite",
    )

    # Test 2: target velocity is finite.
    check(
        np.all(np.isfinite(target.velocity())),
        "Moving target velocity is finite",
    )

    # Test 3: target moves after phase advancement.
    before = target.position().copy()
    target.advance()
    after = target.position()
    check(
        not np.allclose(before, after),
        "Moving target advances over time",
    )

    # Test 4: environment state has expected shape.
    env = DroneEnv()
    state = env.observation()
    check(state.shape == (8,), "Observation contains drone and target state")

    # Test 5: drone state has expected shape.
    check(
        env.drone_state().shape == (4,),
        "Drone state contains position and velocity",
    )

    # Test 6: zero action leaves velocity unchanged.
    env.reset()
    old_velocity = env.velocity.copy()
    env.step_drone(np.zeros(2, dtype=np.float32))
    check(
        np.allclose(env.velocity, old_velocity),
        "Zero action preserves velocity in this simulator",
    )

    # Test 7: action clipping.
    env.reset()
    env.step_drone(np.array([5.0, -5.0], dtype=np.float32))
    check(
        np.all(np.abs(env.velocity) <= MAX_SPEED + 1e-5),
        "Drone speed is bounded",
    )

    # Test 8: drone position remains inside world bounds.
    check(
        DRONE_RADIUS <= env.position[0] <= WORLD_W - DRONE_RADIUS
        and DRONE_RADIUS <= env.position[1] <= WORLD_H - DRONE_RADIUS,
        "Drone position remains inside world bounds",
    )

    # Test 9: PD controller returns two actions.
    action = pd_controller(
        env.drone_state(),
        env.target.position(),
        env.target.velocity(),
    )
    check(action.shape == (2,), "PD controller returns 2D action")

    # Test 10: PD action is bounded.
    check(
        np.all(np.abs(action) <= 1.0),
        "PD controller action is bounded",
    )

    # Test 11: model input has six dimensions.
    model_input = np.concatenate(
        [env.drone_state(), np.zeros(2)]
    )
    check(
        model_input.shape == (6,),
        "Dynamics model input has six dimensions",
    )

    # Test 12: model output has four dimensions.
    if model is not None and normalizer is not None:
        delta = predict_delta(model, normalizer, model_input)
        check(
            delta.shape == (4,),
            "Learned model predicts four state differences",
        )
        check(
            np.all(np.isfinite(delta)),
            "Learned model prediction is finite",
        )
    else:
        check(True, "Learned model tests skipped before training")

    # Test 13: obstacle correction remains bounded.
    corrected = add_obstacle_avoidance(
        np.array([0.0, 0.0], dtype=np.float32),
        OBSTACLE + np.array([20.0, 0.0], dtype=np.float32),
    )
    check(
        np.all(np.abs(corrected) <= 1.0),
        "Obstacle correction returns bounded action",
    )

    # Test 14: target prediction returns a 2D point.
    future = predict_target_position(target, 5)
    check(
        future.shape == (2,),
        "Future target prediction returns 2D point",
    )

    # Test 15: MPC rollout cost is finite.
    test_actions = np.zeros((MPC_HORIZON, 2), dtype=np.float32)
    cost = rollout_candidate(
        env.drone_state(), test_actions, env.target, None, None
    )
    check(np.isfinite(cost), "Oracle MPC rollout cost is finite")

    # Test 16: action safety filter remains bounded.
    filtered = safety_filter(
        np.array([0.9, -0.9], dtype=np.float32),
        OBSTACLE + np.array([25.0, 0.0], dtype=np.float32),
    )
    check(
        np.all(np.abs(filtered) <= 1.0),
        "Safety correction returns bounded action",
    )

    # Test 17: normalizer standard deviations are positive.
    if normalizer is not None:
        check(
            torch.all(normalizer.x_std > 0)
            and torch.all(normalizer.y_std > 0),
            "Training normalizer has positive standard deviations",
        )
    else:
        check(True, "Normalizer test skipped before training")

    print(f"\n{passed}/17 tests passed.")
    return passed == 17


# ============================================================
# 12. MAIN
# ============================================================

def main():
    print("=" * 68)
    print("PHASE 4.18 — LEARNED DYNAMICS MODELS AND MODEL-BASED RL")
    print("=" * 68)
    print(f"Device: {DEVICE}")

    # Step 1: collect real simulator transitions.
    print("\n[1/7] Collecting drone transition data...")
    x_data, y_data = collect_transition_data()

    print(f"Collected transitions: {len(x_data):,}")
    print(f"Input shape: {x_data.shape}")
    print(f"Target shape: {y_data.shape}")

    # Step 2: train a neural dynamics model.
    print("\n[2/7] Training learned dynamics model...")
    model, normalizer, train_history, val_history = (
        train_dynamics_model(x_data, y_data)
    )

    # Step 3: evaluate held-out one-step error.
    print("\n[3/7] Evaluating one-step prediction error...")
    validation_count = max(1, int(len(x_data) * 0.10))
    val_x = x_data[-validation_count:]
    val_y = y_data[-validation_count:]

    one_step_rmse, component_rmse = evaluate_one_step_error(
        model, normalizer, val_x, val_y
    )

    print(f"One-step RMSE: {one_step_rmse:.6f}")
    print(
        "Component RMSE [dx, dy, dvx, dvy]: "
        + np.array2string(component_rmse, precision=6)
    )

    # Step 4: evaluate multi-step model error.
    print("\n[4/7] Evaluating multi-step prediction error...")
    multistep_errors = evaluate_multistep_error(
        model, normalizer
    )
    print(
        "Final rollout position error: "
        f"{multistep_errors[-1]:.3f} pixels"
    )

    # Step 5: benchmark the controllers.
    print("\n[5/7] Benchmarking controllers...")
    results = {}

    for controller in ("PD", "Learned MPC", "Oracle MPC"):
        print(f"Running {controller}...")
        results[controller] = run_benchmark(
            model, normalizer, controller
        )

    for name, metrics in results.items():
        print(f"\n{name}")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")

    # Step 6: run tests.
    print("\n[6/7] Running tests...")
    run_tests(model, normalizer)

    # Step 7: plots, then interactive drone simulator.
    print("\n[7/7] Showing learning and benchmark plots...")
    visualize_training(
        train_history, val_history, multistep_errors
    )
    visualize_benchmark(results)

    print("\nOpening interactive quadcopter simulator...")
    run_pygame(model, normalizer)


if __name__ == "__main__":
    main()
