# ============================================================
# File: phase4/dqn_rocket_navigation.py
#
# PHASE 4.3 — DEEP Q-NETWORK (DQN)
#
# PURPOSE
#   Train a neural network to control a rocket-shaped robot
#   using reinforcement learning rather than teacher labels.
#
# WHAT YOU LEARN
#   1. How a DQN estimates Q(s, a) for every available action.
#   2. How experience replay breaks correlations between samples.
#   3. Why a separate target network stabilizes learning.
#   4. How the Bellman equation creates the DQN training target.
#   5. How epsilon-greedy exploration balances exploration
#      and exploitation.
#   6. How action masks can prevent unsafe movements.
#   7. How a learned policy can be evaluated in an interactive
#      Pygame robotics environment.
#
# IMPLEMENTATION
#   - A 2D navigation environment with one circular obstacle.
#   - Nine discrete actions: 8 directions + stay.
#   - A neural network: 10 -> 128 -> 128 -> 9.
#   - Replay buffer and target network.
#   - Double-checkable Bellman target calculation.
#   - Pygame rocket visualization and clickable target.
#
# LEARNING CYCLE
#   Understand (comments)
#       -> Implement
#       -> Visualize
#       -> Verify
#       -> Experiment
#       -> Apply
#
# ROBOTICS APPLICATION
#   This lesson demonstrates a simplified mobile-robot navigation
#   problem. A real robot would additionally require localization,
#   sensor fusion, motion planning, dynamic-obstacle handling,
#   hardware-level safety supervision, and sim-to-real validation.
#
# IMPORTANT
#   This is a genuine model-free DQN training example:
#   the network is updated from rewards and Bellman targets.
#   It is NOT supervised learning from teacher-generated labels.
#
# CONTROLS
#   Left mouse button : Set a new target
#   SPACE             : Pause / resume
#   R                 : Reset rocket to starting position
#   +/-               : Change simulation speed
#   ESC               : Exit
#
# Install:
#   pip install numpy pygame torch
#
# Run:
#   python phase4/dqn_rocket_navigation.py
# ============================================================

from __future__ import annotations

import math
import random
import sys
from collections import deque
from dataclasses import dataclass

import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.optim as optim


# ============================================================
# 1. CONFIGURATION
# ============================================================

SEED = 42

WORLD_WIDTH = 700
WORLD_HEIGHT = 560

WINDOW_WIDTH = 1020
WINDOW_HEIGHT = 650

OBSTACLE = np.array([350.0, 280.0], dtype=np.float32)
OBSTACLE_RADIUS = 55.0

ROCKET_RADIUS = 11.0
SAFETY_MARGIN = 20.0
SAFE_DISTANCE = OBSTACLE_RADIUS + ROCKET_RADIUS + SAFETY_MARGIN

MOVE_STEP = 8.0
MAX_EPISODE_STEPS = 170

GAMMA = 0.98
LEARNING_RATE = 0.0005

REPLAY_CAPACITY = 50_000
BATCH_SIZE = 64
WARMUP_STEPS = 1_000

TRAIN_EPISODES = 350
TARGET_UPDATE_INTERVAL = 300

EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY_STEPS = 35_000

HIDDEN_SIZE = 128

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Eight compass directions plus a stay action.
ACTION_VECTORS = np.array(
    [
        [1.0, 0.0],                       # 0 RIGHT
        [1.0, -1.0],                      # 1 UP-RIGHT
        [0.0, -1.0],                      # 2 UP
        [-1.0, -1.0],                     # 3 UP-LEFT
        [-1.0, 0.0],                      # 4 LEFT
        [-1.0, 1.0],                      # 5 DOWN-LEFT
        [0.0, 1.0],                       # 6 DOWN
        [1.0, 1.0],                       # 7 DOWN-RIGHT
        [0.0, 0.0],                       # 8 STAY
    ],
    dtype=np.float32,
)

# Normalize diagonal movement so every directional action
# travels the same distance.
for _i in range(8):
    ACTION_VECTORS[_i] /= np.linalg.norm(ACTION_VECTORS[_i])

ACTION_NAMES = [
    "RIGHT",
    "UP-RIGHT",
    "UP",
    "UP-LEFT",
    "LEFT",
    "DOWN-LEFT",
    "DOWN",
    "DOWN-RIGHT",
    "STAY",
]

NUM_ACTIONS = len(ACTION_VECTORS)
STATE_SIZE = 10


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 3. GEOMETRY
# ============================================================

def distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def is_inside_world(position: np.ndarray) -> bool:
    return (
        ROCKET_RADIUS <= position[0] <= WORLD_WIDTH - ROCKET_RADIUS
        and ROCKET_RADIUS <= position[1] <= WORLD_HEIGHT - ROCKET_RADIUS
    )


def is_safe_position(position: np.ndarray) -> bool:
    return distance(position, OBSTACLE) >= SAFE_DISTANCE


def candidate_position(
    position: np.ndarray,
    action: int,
) -> np.ndarray:
    return position + ACTION_VECTORS[action] * MOVE_STEP


def calculate_action_mask(position: np.ndarray) -> np.ndarray:
    """
    Return True for actions that pass the immediate safety check.

    This is a one-step safety filter, not a complete path planner.
    The stay action is always available.
    """
    mask = np.zeros(NUM_ACTIONS, dtype=bool)

    for action in range(NUM_ACTIONS - 1):
        next_position = candidate_position(position, action)

        mask[action] = (
            is_inside_world(next_position)
            and is_safe_position(next_position)
        )

    mask[8] = True
    return mask


def random_valid_position() -> np.ndarray:
    for _ in range(10_000):
        position = np.array(
            [
                random.uniform(35, WORLD_WIDTH - 35),
                random.uniform(35, WORLD_HEIGHT - 35),
            ],
            dtype=np.float32,
        )

        if is_safe_position(position):
            return position

    raise RuntimeError("Could not generate a valid position.")


# ============================================================
# 4. NAVIGATION ENVIRONMENT
# ============================================================

@dataclass
class StepResult:
    state: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    info: dict


class RocketNavigationEnv:
    """
    Simplified 2D robot navigation environment.

    State features:
        0: target relative X
        1: target relative Y
        2: distance to target
        3: obstacle relative X
        4: obstacle relative Y
        5: obstacle clearance
        6: robot X position
        7: robot Y position
        8: previous movement X
        9: previous movement Y

    The DQN learns Q(s,a), the expected discounted return
    from taking action a in state s and following its policy.
    """

    def __init__(self):
        self.position = np.array([80.0, 280.0], dtype=np.float32)
        self.target = np.array([620.0, 280.0], dtype=np.float32)

        self.previous_action = 8
        self.steps = 0
        self.total_reward = 0.0
        self.last_reward = 0.0
        self.success = False
        self.collision = False

    def set_scenario(
        self,
        start: np.ndarray,
        target: np.ndarray,
    ) -> np.ndarray:
        self.position = np.array(start, dtype=np.float32)
        self.target = np.array(target, dtype=np.float32)

        if not is_inside_world(self.position):
            raise ValueError("Start position is outside the world.")

        if not is_inside_world(self.target):
            raise ValueError("Target position is outside the world.")

        if not is_safe_position(self.position):
            raise ValueError("Start position violates safety distance.")

        if not is_safe_position(self.target):
            raise ValueError("Target position violates safety distance.")

        self.previous_action = 8
        self.steps = 0
        self.total_reward = 0.0
        self.last_reward = 0.0
        self.success = False
        self.collision = False

        return self.get_state()

    def reset(self) -> np.ndarray:
        start = random_valid_position()
        target = random_valid_position()

        while distance(start, target) < 250:
            target = random_valid_position()

        return self.set_scenario(start, target)

    def get_state(self) -> np.ndarray:
        target_delta = self.target - self.position
        obstacle_delta = OBSTACLE - self.position

        target_distance = np.linalg.norm(target_delta)
        obstacle_distance = np.linalg.norm(obstacle_delta)

        clearance = obstacle_distance - SAFE_DISTANCE

        previous_vector = ACTION_VECTORS[self.previous_action]

        state = np.array(
            [
                target_delta[0] / WORLD_WIDTH,
                target_delta[1] / WORLD_HEIGHT,
                target_distance / math.hypot(WORLD_WIDTH, WORLD_HEIGHT),

                obstacle_delta[0] / WORLD_WIDTH,
                obstacle_delta[1] / WORLD_HEIGHT,
                clearance / math.hypot(WORLD_WIDTH, WORLD_HEIGHT),

                self.position[0] / WORLD_WIDTH,
                self.position[1] / WORLD_HEIGHT,

                previous_vector[0],
                previous_vector[1],
            ],
            dtype=np.float32,
        )

        return state

    def action_mask(self) -> np.ndarray:
        return calculate_action_mask(self.position)

    def step(self, action: int) -> StepResult:
        if not 0 <= action < NUM_ACTIONS:
            raise ValueError("Invalid action.")

        old_distance = distance(self.position, self.target)
        next_position = candidate_position(self.position, action)

        reward = -0.04
        terminated = False
        truncated = False

        # The safety mask should normally prevent this.
        # Keep the collision check inside the environment too.
        if not is_inside_world(next_position):
            reward -= 5.0
            terminated = True
            self.collision = True

        elif not is_safe_position(next_position):
            reward -= 15.0
            terminated = True
            self.collision = True

        else:
            self.position = next_position

            new_distance = distance(self.position, self.target)

            # Reward progress toward the target.
            progress = old_distance - new_distance
            reward += 0.12 * progress

            # Small penalty for a direct reversal.
            # This discourages unnecessary left-right vibration.
            previous_vector = ACTION_VECTORS[self.previous_action]
            current_vector = ACTION_VECTORS[action]

            if (
                self.previous_action != 8
                and action != 8
                and np.dot(previous_vector, current_vector) < -0.5
            ):
                reward -= 0.08

            self.previous_action = action

            if new_distance <= 18.0:
                reward += 15.0
                terminated = True
                self.success = True

        self.steps += 1

        if self.steps >= MAX_EPISODE_STEPS and not terminated:
            truncated = True

        self.total_reward += reward
        self.last_reward = reward

        return StepResult(
            state=self.get_state(),
            reward=float(reward),
            terminated=terminated,
            truncated=truncated,
            info={
                "success": self.success,
                "collision": self.collision,
                "distance": distance(self.position, self.target),
            },
        )


# ============================================================
# 5. DQN NETWORK
# ============================================================

class DQN(nn.Module):
    """
    Q-network:

        input state (10)
             |
          Linear
             |
           ReLU
             |
          Linear
             |
           ReLU
             |
          Linear
             |
        9 Q-values

    Output[j] estimates Q(state, action_j).
    """

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(STATE_SIZE, HIDDEN_SIZE),
            nn.ReLU(),

            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),

            nn.Linear(HIDDEN_SIZE, NUM_ACTIONS),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


# ============================================================
# 6. EXPERIENCE REPLAY
# ============================================================

@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    terminated: bool
    next_mask: np.ndarray


class ReplayBuffer:
    """
    Store transitions and randomly sample mini-batches.

    Why replay?
      Consecutive robot observations are strongly correlated.
      Random sampling makes training batches less correlated
      and allows experiences to be reused.
    """

    def __init__(self, capacity: int):
        self.memory = deque(maxlen=capacity)

    def add(self, transition: Transition) -> None:
        self.memory.append(transition)

    def sample(self, batch_size: int):
        batch = random.sample(self.memory, batch_size)

        states = np.stack([t.state for t in batch])
        actions = np.array([t.action for t in batch], dtype=np.int64)
        rewards = np.array([t.reward for t in batch], dtype=np.float32)
        next_states = np.stack([t.next_state for t in batch])
        terminated = np.array(
            [t.terminated for t in batch],
            dtype=np.float32,
        )
        next_masks = np.stack([t.next_mask for t in batch])

        return (
            states,
            actions,
            rewards,
            next_states,
            terminated,
            next_masks,
        )

    def __len__(self):
        return len(self.memory)


# ============================================================
# 7. EPSILON-GREEDY EXPLORATION
# ============================================================

def epsilon_by_step(step: int) -> float:
    fraction = min(step / EPSILON_DECAY_STEPS, 1.0)

    return (
        EPSILON_START
        + fraction * (EPSILON_END - EPSILON_START)
    )


def select_training_action(
    q_values: np.ndarray,
    mask: np.ndarray,
    epsilon: float,
) -> int:
    valid_actions = np.flatnonzero(mask)

    if len(valid_actions) == 0:
        return 8

    if random.random() < epsilon:
        return int(random.choice(valid_actions))

    valid_q = q_values[valid_actions]
    return int(valid_actions[np.argmax(valid_q)])


# ============================================================
# 8. DQN AGENT
# ============================================================

class DQNAgent:
    def __init__(self):
        self.policy_net = DQN().to(DEVICE)
        self.target_net = DQN().to(DEVICE)

        self.target_net.load_state_dict(
            self.policy_net.state_dict()
        )

        self.target_net.eval()

        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=LEARNING_RATE,
        )

        self.loss_function = nn.SmoothL1Loss()

        self.replay_buffer = ReplayBuffer(REPLAY_CAPACITY)

        self.training_steps = 0
        self.loss_history = []

    def predict(self, state: np.ndarray) -> np.ndarray:
        self.policy_net.eval()

        with torch.no_grad():
            state_tensor = torch.tensor(
                state,
                dtype=torch.float32,
                device=DEVICE,
            ).unsqueeze(0)

            q_values = self.policy_net(state_tensor)

        self.policy_net.train()

        return q_values.squeeze(0).cpu().numpy()

    def remember(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        terminated: bool,
        next_mask: np.ndarray,
    ) -> None:
        self.replay_buffer.add(
            Transition(
                state=state.copy(),
                action=action,
                reward=reward,
                next_state=next_state.copy(),
                terminated=terminated,
                next_mask=next_mask.copy(),
            )
        )

    def train_step(self) -> float | None:
        if len(self.replay_buffer) < max(
            BATCH_SIZE,
            WARMUP_STEPS,
        ):
            return None

        (
            states,
            actions,
            rewards,
            next_states,
            terminated,
            next_masks,
        ) = self.replay_buffer.sample(BATCH_SIZE)

        states_t = torch.tensor(
            states,
            dtype=torch.float32,
            device=DEVICE,
        )

        actions_t = torch.tensor(
            actions,
            dtype=torch.int64,
            device=DEVICE,
        )

        rewards_t = torch.tensor(
            rewards,
            dtype=torch.float32,
            device=DEVICE,
        )

        next_states_t = torch.tensor(
            next_states,
            dtype=torch.float32,
            device=DEVICE,
        )

        terminated_t = torch.tensor(
            terminated,
            dtype=torch.float32,
            device=DEVICE,
        )

        next_masks_t = torch.tensor(
            next_masks,
            dtype=torch.bool,
            device=DEVICE,
        )

        # ----------------------------------------------------
        # DQN prediction:
        #
        # Q(s,a; theta)
        #
        # Only the Q-value corresponding to the selected
        # action is used for the current-state prediction.
        # ----------------------------------------------------

        current_q = self.policy_net(states_t)

        selected_q = current_q.gather(
            1,
            actions_t.unsqueeze(1),
        ).squeeze(1)

        # ----------------------------------------------------
        # Bellman target:
        #
        # y = r + gamma * max_a' Q_target(s', a')
        #
        # For terminal transitions, y = r.
        #
        # Invalid next actions are excluded by the safety mask.
        # ----------------------------------------------------

        with torch.no_grad():
            next_q = self.target_net(next_states_t)

            next_q = next_q.masked_fill(
                ~next_masks_t,
                -1e9,
            )

            max_next_q = next_q.max(dim=1).values

            target_q = (
                rewards_t
                + GAMMA * (1.0 - terminated_t) * max_next_q
            )

        # ----------------------------------------------------
        # Minimize:
        #
        # L(theta) = SmoothL1(Q(s,a;theta), y)
        #
        # Backpropagation computes the gradient.
        # Adam updates the policy-network parameters.
        # ----------------------------------------------------

        loss = self.loss_function(selected_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()

        # Limit unusually large gradients.
        nn.utils.clip_grad_norm_(
            self.policy_net.parameters(),
            max_norm=5.0,
        )

        self.optimizer.step()

        self.training_steps += 1

        if self.training_steps % TARGET_UPDATE_INTERVAL == 0:
            self.update_target_network()

        loss_value = float(loss.item())
        self.loss_history.append(loss_value)

        return loss_value

    def update_target_network(self):
        self.target_net.load_state_dict(
            self.policy_net.state_dict()
        )

    def save(self, filename: str = "dqn_rocket.pth"):
        torch.save(
            {
                "policy_net": self.policy_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            filename,
        )

    def load(self, filename: str = "dqn_rocket.pth"):
        checkpoint = torch.load(
            filename,
            map_location=DEVICE,
        )

        self.policy_net.load_state_dict(
            checkpoint["policy_net"]
        )

        self.target_net.load_state_dict(
            checkpoint["target_net"]
        )

        self.optimizer.load_state_dict(
            checkpoint["optimizer"]
        )


# ============================================================
# 9. TRAINING
# ============================================================

def train_dqn(
    agent: DQNAgent,
    env: RocketNavigationEnv,
    episodes: int = TRAIN_EPISODES,
):
    global_step = 0

    episode_rewards = []
    success_history = []
    loss_history = []

    print("\n" + "=" * 65)
    print("PHASE 4.3 — DQN TRAINING")
    print("=" * 65)
    print(f"Device: {DEVICE}")
    print(f"Episodes: {episodes}")
    print(f"Replay capacity: {REPLAY_CAPACITY}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Gamma: {GAMMA}")
    print("=" * 65)

    for episode in range(episodes):
        state = env.reset()

        episode_reward = 0.0
        episode_losses = []

        for _ in range(MAX_EPISODE_STEPS):
            mask = env.action_mask()
            q_values = agent.predict(state)

            epsilon = epsilon_by_step(global_step)

            action = select_training_action(
                q_values,
                mask,
                epsilon,
            )

            result = env.step(action)

            # A truncated episode still has a valid final state.
            # Therefore, it may bootstrap from that state.
            #
            # Only true termination disables bootstrapping.
            next_mask = env.action_mask()

            agent.remember(
                state=state,
                action=action,
                reward=result.reward,
                next_state=result.state,
                terminated=result.terminated,
                next_mask=next_mask,
            )

            loss = agent.train_step()

            if loss is not None:
                episode_losses.append(loss)

            state = result.state
            episode_reward += result.reward
            global_step += 1

            if result.terminated or result.truncated:
                break

        episode_rewards.append(episode_reward)
        success_history.append(1 if env.success else 0)

        if episode_losses:
            mean_loss = float(np.mean(episode_losses))
            loss_history.append(mean_loss)

        if (episode + 1) % 25 == 0 or episode == 0:
            recent_rewards = episode_rewards[-25:]
            recent_success = success_history[-25:]

            avg_reward = float(np.mean(recent_rewards))
            success_rate = float(np.mean(recent_success))

            print(
                f"Episode {episode + 1:4d}/{episodes} | "
                f"Reward {avg_reward:8.2f} | "
                f"Success {success_rate * 100:5.1f}% | "
                f"Epsilon {epsilon_by_step(global_step):.3f} | "
                f"Replay {len(agent.replay_buffer):5d}"
            )

    agent.update_target_network()

    print("\nTraining completed.")
    print(f"Total environment steps: {global_step}")
    print(f"Replay transitions: {len(agent.replay_buffer)}")

    return {
        "episode_rewards": episode_rewards,
        "success_history": success_history,
        "loss_history": loss_history,
    }


# ============================================================
# 10. STABLE ACTION SELECTION FOR THE LIVE SIMULATOR
# ============================================================

def choose_stable_action(
    q_values: np.ndarray,
    mask: np.ndarray,
    previous_action: int,
    hysteresis: float = 0.12,
) -> int:
    """
    Add a small hysteresis margin to reduce left-right vibration.

    If the previous action is still safe and its Q-value is close
    to the best available Q-value, keep the previous action.

    The safety mask always takes priority over hysteresis.
    """
    valid_actions = np.flatnonzero(mask)

    if len(valid_actions) == 0:
        return 8

    valid_q = q_values[valid_actions]
    best_action = int(valid_actions[np.argmax(valid_q)])

    if (
        previous_action in valid_actions
        and previous_action != 8
        and best_action != 8
    ):
        previous_q = q_values[previous_action]
        best_q = q_values[best_action]

        if previous_q >= best_q - hysteresis:
            return previous_action

    return best_action


# ============================================================
# 11. PYGAME VISUALIZATION
# ============================================================

class RocketGUI:
    def __init__(
        self,
        env: RocketNavigationEnv,
        agent: DQNAgent,
        training_results: dict,
    ):
        pygame.init()

        pygame.display.set_caption(
            "Phase 4.3 — DQN Rocket Navigation"
        )

        self.screen = pygame.display.set_mode(
            (WINDOW_WIDTH, WINDOW_HEIGHT)
        )

        self.clock = pygame.time.Clock()

        self.font = pygame.font.SysFont("consolas", 17)
        self.small_font = pygame.font.SysFont("consolas", 14)
        self.title_font = pygame.font.SysFont(
            "consolas",
            21,
            bold=True,
        )

        self.env = env
        self.agent = agent
        self.training_results = training_results

        self.paused = False
        self.running = True
        self.speed = 1

        self.previous_action = 8
        self.current_action = 8

        self.q_values = np.zeros(NUM_ACTIONS, dtype=np.float32)
        self.mask = np.ones(NUM_ACTIONS, dtype=bool)

        self.episode_count = 0
        self.status = "READY"

        self.reset_robot()

    def reset_robot(self):
        start = np.array([80.0, 280.0], dtype=np.float32)

        if not is_safe_position(start):
            start = random_valid_position()

        target = np.array([620.0, 280.0], dtype=np.float32)

        if not is_safe_position(target):
            target = random_valid_position()

        self.env.set_scenario(start, target)

        self.previous_action = 8
        self.current_action = 8

        self.status = "READY"

    def set_target_from_mouse(self, mouse_position):
        x, y = mouse_position

        # Ignore clicks in the information panel.
        if not (0 <= x <= WORLD_WIDTH and 0 <= y <= WORLD_HEIGHT):
            return

        target = np.array([x, y], dtype=np.float32)

        if not is_inside_world(target):
            return

        if not is_safe_position(target):
            self.status = "TARGET TOO CLOSE TO OBSTACLE"
            return

        if distance(target, self.env.position) < 20:
            self.status = "TARGET TOO CLOSE TO ROCKET"
            return

        self.env.target = target
        self.env.steps = 0
        self.env.total_reward = 0.0
        self.env.success = False
        self.env.collision = False

        self.previous_action = 8
        self.current_action = 8

        self.status = "NEW TARGET SET"

    def text(self, value, x, y, color=(230, 230, 230), font=None):
        if font is None:
            font = self.font

        surface = font.render(str(value), True, color)
        self.screen.blit(surface, (x, y))

    def draw_world(self):
        self.screen.fill((18, 22, 30))

        pygame.draw.rect(
            self.screen,
            (28, 34, 44),
            (0, 0, WORLD_WIDTH, WORLD_HEIGHT),
        )

        # Grid.
        for x in range(0, WORLD_WIDTH, 40):
            pygame.draw.line(
                self.screen,
                (39, 46, 58),
                (x, 0),
                (x, WORLD_HEIGHT),
                1,
            )

        for y in range(0, WORLD_HEIGHT, 40):
            pygame.draw.line(
                self.screen,
                (39, 46, 58),
                (0, y),
                (WORLD_WIDTH, y),
                1,
            )

        # Safety boundary and obstacle.
        obstacle_center = tuple(OBSTACLE.astype(int))

        pygame.draw.circle(
            self.screen,
            (130, 75, 55),
            obstacle_center,
            int(SAFE_DISTANCE),
            2,
        )

        pygame.draw.circle(
            self.screen,
            (170, 55, 55),
            obstacle_center,
            int(OBSTACLE_RADIUS),
        )

        pygame.draw.circle(
            self.screen,
            (220, 120, 90),
            obstacle_center,
            8,
        )

        self.text(
            "OBSTACLE",
            OBSTACLE[0] - 42,
            OBSTACLE[1] - 12,
            (255, 220, 220),
            self.small_font,
        )

        # Target.
        tx, ty = self.env.target.astype(int)

        pygame.draw.circle(
            self.screen,
            (60, 230, 130),
            (tx, ty),
            13,
            2,
        )

        pygame.draw.line(
            self.screen,
            (60, 230, 130),
            (tx - 19, ty),
            (tx + 19, ty),
            2,
        )

        pygame.draw.line(
            self.screen,
            (60, 230, 130),
            (tx, ty - 19),
            (tx, ty + 19),
            2,
        )

        self.text(
            "TARGET",
            tx + 15,
            ty - 25,
            (100, 255, 160),
            self.small_font,
        )

        # Rocket trail.
        px, py = self.env.position.astype(int)

        pygame.draw.circle(
            self.screen,
            (90, 130, 210),
            (px, py),
            18,
            1,
        )

        self.draw_rocket(px, py)

    def draw_rocket(self, x: int, y: int):
        """
        Draw a rocket-shaped robot.

        The rocket's nose follows the selected movement direction.
        """
        vector = ACTION_VECTORS[self.current_action]

        if np.linalg.norm(vector) < 0.01:
            vector = ACTION_VECTORS[self.previous_action]

        if np.linalg.norm(vector) < 0.01:
            vector = np.array([1.0, 0.0])

        angle = math.atan2(vector[1], vector[0])

        # Local rocket coordinates: nose points along +X.
        local_points = [
            (19, 0),
            (5, -9),
            (-9, -7),
            (-13, -4),
            (-13, 4),
            (-9, 7),
            (5, 9),
        ]

        def rotate(point):
            lx, ly = point

            rx = lx * math.cos(angle) - ly * math.sin(angle)
            ry = lx * math.sin(angle) + ly * math.cos(angle)

            return (
                int(x + rx),
                int(y + ry),
            )

        rocket_points = [rotate(p) for p in local_points]

        # Animated exhaust.
        flame_length = random.randint(9, 17)

        flame_local = [
            (-13, -4),
            (-13 - flame_length, 0),
            (-13, 4),
        ]

        flame_points = [rotate(p) for p in flame_local]

        pygame.draw.polygon(
            self.screen,
            (255, 145, 35),
            flame_points,
        )

        pygame.draw.polygon(
            self.screen,
            (230, 235, 245),
            rocket_points,
        )

        pygame.draw.polygon(
            self.screen,
            (70, 140, 220),
            rocket_points,
            2,
        )

        pygame.draw.circle(
            self.screen,
            (40, 130, 230),
            rotate((7, 0)),
            4,
        )

        # Fins.
        pygame.draw.polygon(
            self.screen,
            (220, 70, 70),
            [
                rotate((-9, -6)),
                rotate((-17, -12)),
                rotate((-12, -2)),
            ],
        )

        pygame.draw.polygon(
            self.screen,
            (220, 70, 70),
            [
                rotate((-9, 6)),
                rotate((-17, 12)),
                rotate((-12, 2)),
            ],
        )

    def draw_q_values(self):
        panel_x = 720
        panel_y = 285

        self.text(
            "DQN ACTION VALUES",
            panel_x,
            panel_y,
            (120, 210, 255),
            self.title_font,
        )

        valid_values = self.q_values[self.mask]

        if len(valid_values) > 0:
            min_q = float(np.min(valid_values))
            max_q = float(np.max(valid_values))
        else:
            min_q = -1.0
            max_q = 1.0

        span = max(max_q - min_q, 1e-6)

        for i, name in enumerate(ACTION_NAMES):
            y = panel_y + 35 + i * 28

            valid = bool(self.mask[i])
            selected = i == self.current_action

            color = (
                (100, 255, 150)
                if selected
                else (210, 210, 210)
            )

            if not valid:
                color = (100, 100, 100)

            self.text(
                f"{name:10s}",
                panel_x,
                y,
                color,
                self.small_font,
            )

            q_value = float(self.q_values[i])

            self.text(
                f"{q_value:8.3f}",
                panel_x + 105,
                y,
                color,
                self.small_font,
            )

            if valid:
                bar_width = int(
                    90 * (q_value - min_q) / span
                )

                pygame.draw.rect(
                    self.screen,
                    (60, 120, 190),
                    (panel_x + 205, y + 3, bar_width, 10),
                )

            else:
                pygame.draw.line(
                    self.screen,
                    (100, 100, 100),
                    (panel_x + 205, y + 8),
                    (panel_x + 270, y + 8),
                    2,
                )

    def draw_training_curve(self):
        x0 = 720
        y0 = 535
        width = 260
        height = 65

        self.text(
            "TRAINING REWARD",
            x0,
            y0 - 23,
            (120, 210, 255),
            self.small_font,
        )

        pygame.draw.rect(
            self.screen,
            (50, 60, 75),
            (x0, y0, width, height),
            1,
        )

        rewards = self.training_results["episode_rewards"]

        if len(rewards) < 2:
            return

        # Plot a moving average rather than noisy individual rewards.
        window = 15

        smoothed = [
            np.mean(rewards[max(0, i - window + 1):i + 1])
            for i in range(len(rewards))
        ]

        low = float(min(smoothed))
        high = float(max(smoothed))
        span = max(high - low, 1e-6)

        points = []

        for i, value in enumerate(smoothed):
            px = x0 + int(i / max(len(smoothed) - 1, 1) * width)

            py = y0 + height - int(
                (value - low) / span * (height - 8)
            )

            points.append((px, py))

        if len(points) >= 2:
            pygame.draw.lines(
                self.screen,
                (70, 220, 150),
                False,
                points,
                2,
            )

    def draw_panel(self):
        pygame.draw.rect(
            self.screen,
            (22, 27, 37),
            (WORLD_WIDTH, 0, WINDOW_WIDTH - WORLD_WIDTH, WINDOW_HEIGHT),
        )

        self.text(
            "DQN ROBOT CONTROL",
            720,
            20,
            (120, 210, 255),
            self.title_font,
        )

        self.text(
            f"Device: {DEVICE}",
            720,
            55,
            (200, 200, 200),
            self.small_font,
        )

        self.text(
            f"Position: ({self.env.position[0]:.1f}, "
            f"{self.env.position[1]:.1f})",
            720,
            90,
            font=self.small_font,
        )

        self.text(
            f"Target distance: "
            f"{distance(self.env.position, self.env.target):.1f}",
            720,
            115,
            font=self.small_font,
        )

        self.text(
            f"Action: {ACTION_NAMES[self.current_action]}",
            720,
            140,
            (120, 255, 160),
            self.small_font,
        )

        self.text(
            f"Steps: {self.env.steps}",
            720,
            165,
            font=self.small_font,
        )

        self.text(
            f"Reward: {self.env.total_reward:.2f}",
            720,
            190,
            font=self.small_font,
        )

        self.text(
            f"Replay: {len(self.agent.replay_buffer)}",
            720,
            215,
            font=self.small_font,
        )

        self.text(
            f"Training updates: {self.agent.training_steps}",
            720,
            240,
            font=self.small_font,
        )

        self.draw_q_values()
        self.draw_training_curve()

        self.text(
            "CONTROLS",
            720,
            615,
            (120, 210, 255),
            self.small_font,
        )

        self.text(
            "Click: target | SPACE: pause | R: reset",
            720,
            635,
            font=self.small_font,
        )

    def update_policy(self):
        self.mask = self.env.action_mask()
        self.q_values = self.agent.predict(
            self.env.get_state()
        )

        self.current_action = choose_stable_action(
            self.q_values,
            self.mask,
            self.previous_action,
        )

    def simulation_step(self):
        self.update_policy()

        result = self.env.step(self.current_action)

        self.previous_action = self.current_action

        if result.terminated:
            if self.env.success:
                self.status = "TARGET REACHED"
            elif self.env.collision:
                self.status = "COLLISION"
            else:
                self.status = "TERMINATED"

        elif result.truncated:
            self.status = "STEP LIMIT REACHED"

        if result.terminated or result.truncated:
            self.episode_count += 1

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
                        self.reset_robot()

                    elif event.key in (
                        pygame.K_PLUS,
                        pygame.K_EQUALS,
                        pygame.K_KP_PLUS,
                    ):
                        self.speed = min(self.speed + 1, 8)

                    elif event.key in (
                        pygame.K_MINUS,
                        pygame.K_KP_MINUS,
                    ):
                        self.speed = max(self.speed - 1, 1)

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        self.set_target_from_mouse(event.pos)

            if not self.paused:
                for _ in range(self.speed):
                    if (
                        self.env.success
                        or self.env.collision
                        or self.env.steps >= MAX_EPISODE_STEPS
                    ):
                        break

                    self.simulation_step()

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

            if self.status != "READY":
                self.text(
                    self.status,
                    20,
                    WORLD_HEIGHT - 28,
                    (255, 220, 100),
                    self.small_font,
                )

            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()


# ============================================================
# 12. NUMERICAL VERIFICATION AND UNIT TESTS
# ============================================================

def bellman_target(
    reward: float,
    gamma: float,
    next_q_values: np.ndarray,
    terminated: bool,
    next_mask: np.ndarray,
) -> float:
    """
    Independent Bellman-target calculation for verification.

    y = r                              if terminal
    y = r + gamma * max Q(s',a')       otherwise
    """
    if terminated:
        return float(reward)

    valid_q = next_q_values[next_mask]

    if len(valid_q) == 0:
        return float(reward)

    return float(reward + gamma * np.max(valid_q))


def run_tests():
    print("\n" + "=" * 65)
    print("UNIT TESTS")
    print("=" * 65)

    tests_passed = 0

    def check(name, condition):
        nonlocal tests_passed

        assert condition, f"FAILED: {name}"

        tests_passed += 1
        print(f"[PASS] {name}")

    # Test 1: action count.
    check(
        "Nine discrete actions",
        NUM_ACTIONS == 9,
    )

    # Test 2: normalized directional movement.
    norms = np.linalg.norm(ACTION_VECTORS[:8], axis=1)

    check(
        "Eight directional vectors have unit length",
        np.allclose(norms, np.ones(8)),
    )

    # Test 3: state dimension.
    env = RocketNavigationEnv()
    state = env.reset()

    check(
        "State dimension is 10",
        state.shape == (STATE_SIZE,),
    )

    # Test 4: valid position.
    check(
        "Generated start position is safe",
        is_safe_position(env.position),
    )

    # Test 5: action mask shape.
    mask = env.action_mask()

    check(
        "Safety mask has nine entries",
        mask.shape == (NUM_ACTIONS,),
    )

    # Test 6: stay action is always available.
    check(
        "Stay action remains available",
        bool(mask[8]),
    )

    # Test 7: network output shape.
    network = DQN().to(DEVICE)

    output = network(
        torch.zeros(
            (1, STATE_SIZE),
            dtype=torch.float32,
            device=DEVICE,
        )
    )

    check(
        "DQN output shape is (1, 9)",
        tuple(output.shape) == (1, NUM_ACTIONS),
    )

    # Test 8: replay buffer.
    replay = ReplayBuffer(10)

    replay.add(
        Transition(
            state=np.zeros(STATE_SIZE, dtype=np.float32),
            action=0,
            reward=1.0,
            next_state=np.ones(STATE_SIZE, dtype=np.float32),
            terminated=False,
            next_mask=np.ones(NUM_ACTIONS, dtype=bool),
        )
    )

    check(
        "Replay buffer stores a transition",
        len(replay) == 1,
    )

    # Test 9: nonterminal Bellman target.
    target = bellman_target(
        reward=1.0,
        gamma=0.9,
        next_q_values=np.array([1.0, 2.0, 3.0]),
        terminated=False,
        next_mask=np.array([True, True, True]),
    )

    check(
        "Bellman target equals 3.7",
        np.isclose(target, 3.7),
    )

    # Test 10: terminal Bellman target.
    terminal_target = bellman_target(
        reward=2.0,
        gamma=0.9,
        next_q_values=np.array([100.0, 200.0]),
        terminated=True,
        next_mask=np.array([True, True]),
    )

    check(
        "Terminal target does not bootstrap",
        np.isclose(terminal_target, 2.0),
    )

    # Test 11: safety mask excludes an unsafe action.
    test_position = np.array(
        [
            OBSTACLE[0] - SAFE_DISTANCE + 1,
            OBSTACLE[1],
        ],
        dtype=np.float32,
    )

    test_mask = calculate_action_mask(test_position)

    check(
        "Safety mask blocks movement into obstacle",
        not bool(test_mask[0]),
    )

    # Test 12: invalid actions are not selected.
    test_q = np.array(
        [100, 90, 80, 70, 60, 50, 40, 30, 0],
        dtype=np.float32,
    )

    test_mask = np.array(
        [False, False, True, True, True, True, True, True, True],
        dtype=bool,
    )

    selected = select_training_action(
        test_q,
        test_mask,
        epsilon=0.0,
    )

    check(
        "Greedy selection respects safety mask",
        selected == 2,
    )

    # Test 13: hysteresis preserves a near-optimal safe action.
    hysteresis_action = choose_stable_action(
        q_values=np.array(
            [1.00, 0.95, 0.10, 0.00, 0.00, 0.00, 0.00, 0.00, -1.0],
            dtype=np.float32,
        ),
        mask=np.array(
            [True, True, True, True, True, True, True, True, True],
            dtype=bool,
        ),
        previous_action=0,
        hysteresis=0.12,
    )

    check(
        "Hysteresis keeps previous near-optimal action",
        hysteresis_action == 0,
    )

    # Test 14: target update copies parameters.
    agent = DQNAgent()

    with torch.no_grad():
        for parameter in agent.policy_net.parameters():
            parameter.fill_(0.25)

    agent.update_target_network()

    parameters_match = all(
        torch.allclose(p1, p2)
        for p1, p2 in zip(
            agent.policy_net.parameters(),
            agent.target_net.parameters(),
        )
    )

    check(
        "Target network update copies policy parameters",
        parameters_match,
    )

    print("-" * 65)
    print(f"Tests passed: {tests_passed}/14")
    print("=" * 65)


# ============================================================
# 13. MAIN
# ============================================================

def main():
    set_seed()

    run_tests()

    env = RocketNavigationEnv()
    agent = DQNAgent()

    # Train the DQN before opening the interactive simulator.
    training_results = train_dqn(
        agent,
        env,
        episodes=TRAIN_EPISODES,
    )

    # Save the learned policy.
    agent.save("dqn_rocket.pth")

    print("\nSaved model: dqn_rocket.pth")

    # Interactive Pygame visualization.
    gui = RocketGUI(
        env=env,
        agent=agent,
        training_results=training_results,
    )

    gui.run()


if __name__ == "__main__":
    main()