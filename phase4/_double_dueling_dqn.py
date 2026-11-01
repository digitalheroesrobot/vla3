
# ============================================================
# File: phase4/double_dueling_dqn.py
#
# PHASE 4.4 — DOUBLE DQN + DUELING DQN
#
# PURPOSE
#   Improve DQN learning stability and value estimation.
#
# WHAT YOU LEARN
#   STEP 1 — DQN overestimation
#       Understand why max_a Q_target(s',a) can overestimate
#       action values.
#
#   STEP 2 — Double DQN
#       Select the next action with the online network:
#           a* = argmax_a Q_online(s',a)
#
#       Evaluate that action with the target network:
#           y = r + gamma * Q_target(s',a*)
#
#   STEP 3 — Dueling Network
#       Learn state value V(s) and action advantage A(s,a)
#       separately.
#
#       Combine them:
#           Q(s,a) = V(s) + A(s,a) - mean_a A(s,a)
#
#   STEP 4 — Experience Replay
#       Reuse randomly sampled robot experiences.
#
#   STEP 5 — Target Network
#       Stabilize the Bellman targets with delayed parameters.
#
#   STEP 6 — Interactive Robotics Simulation
#       Use the trained network to control a rocket-shaped robot.
#
# IMPLEMENTATION
#   - Double DQN target calculation.
#   - Dueling neural network architecture.
#   - Experience replay.
#   - Epsilon-greedy exploration.
#   - Safety-filtered actions.
#   - Pygame interactive navigation.
#   - Numerical verification and unit tests.
#
# ROBOTICS APPLICATION
#   A mobile robot must navigate toward a target while avoiding
#   an obstacle. The network estimates the value of each movement.
#   The safety layer blocks immediately unsafe actions.
#
#   This is a simplified simulation, not a certified robot
#   navigation or collision-avoidance system.
#
# CONTROLS
#   Left click : Set a new target
#   SPACE      : Pause / resume
#   R          : Reset robot
#   + / -      : Change simulation speed
#   ESC        : Exit
#
# LEARNING CYCLE
#   Understand -> Implement -> Visualize -> Verify
#       -> Experiment -> Apply
#
# INSTALL
#   pip install numpy pygame torch
#
# RUN
#   python phase4/double_dueling_dqn.py
# ============================================================

from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass

import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.optim as optim


# ============================================================
# STEP 0 — CONFIGURATION
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

STATE_SIZE = 10
NUM_ACTIONS = 9
HIDDEN_SIZE = 128

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

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

ACTION_VECTORS = np.array(
    [
        [1.0, 0.0],       # RIGHT
        [1.0, -1.0],      # UP-RIGHT
        [0.0, -1.0],      # UP
        [-1.0, -1.0],     # UP-LEFT
        [-1.0, 0.0],      # LEFT
        [-1.0, 1.0],      # DOWN-LEFT
        [0.0, 1.0],       # DOWN
        [1.0, 1.0],       # DOWN-RIGHT
        [0.0, 0.0],       # STAY
    ],
    dtype=np.float32,
)

for i in range(8):
    ACTION_VECTORS[i] /= np.linalg.norm(ACTION_VECTORS[i])

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


# ============================================================
# STEP 1 — REPRODUCIBILITY
# ============================================================

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# STEP 2 — GEOMETRY AND SAFETY
# ============================================================

def distance(a, b):
    return float(np.linalg.norm(a - b))


def is_inside_world(position):
    return (
        ROCKET_RADIUS <= position[0] <= WORLD_WIDTH - ROCKET_RADIUS
        and ROCKET_RADIUS <= position[1] <= WORLD_HEIGHT - ROCKET_RADIUS
    )


def is_safe_position(position):
    return distance(position, OBSTACLE) >= SAFE_DISTANCE


def candidate_position(position, action):
    return position + ACTION_VECTORS[action] * MOVE_STEP


def calculate_action_mask(position):
    """
    True means that the action passes the one-step safety check.

    This is a local safety filter. It does not guarantee that
    a complete route to the target exists.
    """
    mask = np.zeros(NUM_ACTIONS, dtype=bool)

    for action in range(8):
        next_position = candidate_position(position, action)

        mask[action] = (
            is_inside_world(next_position)
            and is_safe_position(next_position)
        )

    mask[8] = True
    return mask


def random_valid_position():
    for _ in range(10_000):
        p = np.array(
            [
                random.uniform(35, WORLD_WIDTH - 35),
                random.uniform(35, WORLD_HEIGHT - 35),
            ],
            dtype=np.float32,
        )

        if is_safe_position(p):
            return p

    raise RuntimeError("Cannot generate a safe position.")


# ============================================================
# STEP 3 — NAVIGATION ENVIRONMENT
# ============================================================

@dataclass
class StepResult:
    state: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    info: dict


class RocketNavigationEnv:

    def __init__(self):
        self.position = np.array([80.0, 280.0], dtype=np.float32)
        self.target = np.array([620.0, 280.0], dtype=np.float32)

        self.previous_action = 8
        self.steps = 0
        self.total_reward = 0.0
        self.success = False
        self.collision = False

    def set_scenario(self, start, target):
        start = np.array(start, dtype=np.float32)
        target = np.array(target, dtype=np.float32)

        if not is_inside_world(start) or not is_inside_world(target):
            raise ValueError("Start or target is outside the world.")

        if not is_safe_position(start) or not is_safe_position(target):
            raise ValueError("Start or target violates safety distance.")

        self.position = start.copy()
        self.target = target.copy()

        self.previous_action = 8
        self.steps = 0
        self.total_reward = 0.0
        self.success = False
        self.collision = False

        return self.get_state()

    def reset(self):
        start = random_valid_position()
        target = random_valid_position()

        while distance(start, target) < 250:
            target = random_valid_position()

        return self.set_scenario(start, target)

    def get_state(self):
        target_delta = self.target - self.position
        obstacle_delta = OBSTACLE - self.position

        target_distance = np.linalg.norm(target_delta)
        obstacle_distance = np.linalg.norm(obstacle_delta)

        clearance = obstacle_distance - SAFE_DISTANCE
        previous_vector = ACTION_VECTORS[self.previous_action]

        return np.array(
            [
                target_delta[0] / WORLD_WIDTH,
                target_delta[1] / WORLD_HEIGHT,
                target_distance / math.hypot(
                    WORLD_WIDTH,
                    WORLD_HEIGHT,
                ),

                obstacle_delta[0] / WORLD_WIDTH,
                obstacle_delta[1] / WORLD_HEIGHT,
                clearance / math.hypot(
                    WORLD_WIDTH,
                    WORLD_HEIGHT,
                ),

                self.position[0] / WORLD_WIDTH,
                self.position[1] / WORLD_HEIGHT,

                previous_vector[0],
                previous_vector[1],
            ],
            dtype=np.float32,
        )

    def action_mask(self):
        return calculate_action_mask(self.position)

    def step(self, action):
        if not 0 <= action < NUM_ACTIONS:
            raise ValueError("Invalid action.")

        old_distance = distance(self.position, self.target)
        next_position = candidate_position(self.position, action)

        reward = -0.04
        terminated = False
        truncated = False

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

            # Positive reward for moving closer to the target.
            reward += 0.12 * (old_distance - new_distance)

            # Mild reversal penalty to discourage unnecessary
            # left-right oscillation.
            old_vector = ACTION_VECTORS[self.previous_action]
            new_vector = ACTION_VECTORS[action]

            if (
                self.previous_action != 8
                and action != 8
                and np.dot(old_vector, new_vector) < -0.5
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
# STEP 4 — DUELING NETWORK ARCHITECTURE
# ============================================================

class DuelingDQN(nn.Module):
    """
    Dueling architecture:

                 State s
                    |
             Shared feature layers
                    |
             +------+------+
             |             |
          Value V(s)   Advantage A(s,a)
             |             |
             +------+------+
                    |
        Q(s,a) = V(s) + A(s,a) - mean(A)
                    |
             Nine Q-values

    Why subtract mean advantage?

    Without this operation, V and A are not uniquely identifiable.
    Many combinations of V and A could produce the same Q-values.

    Subtracting mean(A) gives a consistent decomposition.
    """

    def __init__(self):
        super().__init__()

        self.feature_layer = nn.Sequential(
            nn.Linear(STATE_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
        )

        self.value_stream = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, 64),
            nn.ReLU(),
            nn.Linear(64, NUM_ACTIONS),
        )

    def forward(self, state):
        features = self.feature_layer(state)

        value = self.value_stream(features)
        advantage = self.advantage_stream(features)

        q_values = (
            value
            + advantage
            - advantage.mean(dim=1, keepdim=True)
        )

        return q_values

    def forward_components(self, state):
        features = self.feature_layer(state)

        value = self.value_stream(features)
        advantage = self.advantage_stream(features)

        q_values = (
            value
            + advantage
            - advantage.mean(dim=1, keepdim=True)
        )

        return q_values, value, advantage


# ============================================================
# STEP 5 — EXPERIENCE REPLAY
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

    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def add(self, transition):
        self.memory.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)

        states = np.stack([t.state for t in batch])
        actions = np.array(
            [t.action for t in batch],
            dtype=np.int64,
        )
        rewards = np.array(
            [t.reward for t in batch],
            dtype=np.float32,
        )
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
# STEP 6 — DOUBLE DQN BELLMAN TARGET
# ============================================================

def calculate_double_dqn_targets(
    rewards,
    online_next_q,
    target_next_q,
    terminated,
    next_masks,
    gamma,
):
    """
    Double DQN separates action selection from action evaluation.

    Ordinary DQN:
        y = r + gamma * max_a Q_target(s', a)

    Double DQN:
        a* = argmax_a Q_online(s', a)

        y = r + gamma * Q_target(s', a*)

    The online network selects the action.
    The target network evaluates the selected action.

    Terminal transitions do not bootstrap.

    Invalid next actions are excluded by the safety mask.
    """

    # Online network chooses the next action.
    masked_online_q = online_next_q.masked_fill(
        ~next_masks,
        -1e9,
    )

    best_next_actions = masked_online_q.argmax(
        dim=1,
        keepdim=True,
    )

    # Target network evaluates that selected action.
    selected_target_q = target_next_q.gather(
        1,
        best_next_actions,
    ).squeeze(1)

    targets = (
        rewards
        + gamma * (1.0 - terminated) * selected_target_q
    )

    return targets


# ============================================================
# STEP 7 — EPSILON-GREEDY EXPLORATION
# ============================================================

def epsilon_by_step(step):
    fraction = min(step / EPSILON_DECAY_STEPS, 1.0)

    return (
        EPSILON_START
        + fraction * (EPSILON_END - EPSILON_START)
    )


def select_action(q_values, mask, epsilon):
    valid_actions = np.flatnonzero(mask)

    if len(valid_actions) == 0:
        return 8

    if random.random() < epsilon:
        return int(random.choice(valid_actions))

    valid_q = q_values[valid_actions]

    return int(valid_actions[np.argmax(valid_q)])


# ============================================================
# STEP 8 — DUELING DOUBLE DQN AGENT
# ============================================================

class DuelingDoubleDQNAgent:

    def __init__(self):
        self.policy_net = DuelingDQN().to(DEVICE)
        self.target_net = DuelingDQN().to(DEVICE)

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

    def predict(self, state):
        self.policy_net.eval()

        with torch.no_grad():
            state_t = torch.tensor(
                state,
                dtype=torch.float32,
                device=DEVICE,
            ).unsqueeze(0)

            q_values = self.policy_net(state_t)

        self.policy_net.train()

        return q_values.squeeze(0).cpu().numpy()

    def remember(
        self,
        state,
        action,
        reward,
        next_state,
        terminated,
        next_mask,
    ):
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

    def train_step(self):
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
        # Current Q(s,a) from the online network.
        # ----------------------------------------------------

        current_q_values = self.policy_net(states_t)

        selected_current_q = current_q_values.gather(
            1,
            actions_t.unsqueeze(1),
        ).squeeze(1)

        # ----------------------------------------------------
        # Double DQN target.
        #
        # Online network:
        #   selects argmax action.
        #
        # Target network:
        #   evaluates selected action.
        #
        # Both next-state calculations are detached from
        # gradient computation.
        # ----------------------------------------------------

        with torch.no_grad():
            online_next_q = self.policy_net(next_states_t)
            target_next_q = self.target_net(next_states_t)

            target_q = calculate_double_dqn_targets(
                rewards=rewards_t,
                online_next_q=online_next_q,
                target_next_q=target_next_q,
                terminated=terminated_t,
                next_masks=next_masks_t,
                gamma=GAMMA,
            )

        # ----------------------------------------------------
        # Smooth L1 loss:
        #
        # L(theta) = SmoothL1(Q_online(s,a), y)
        #
        # Only the online network is optimized.
        # ----------------------------------------------------

        loss = self.loss_function(
            selected_current_q,
            target_q,
        )

        self.optimizer.zero_grad()
        loss.backward()

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

    def save(self, filename="dueling_double_dqn.pth"):
        torch.save(
            {
                "policy_net": self.policy_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "training_steps": self.training_steps,
            },
            filename,
        )

    def load(self, filename="dueling_double_dqn.pth"):
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

        self.training_steps = checkpoint.get(
            "training_steps",
            0,
        )


# ============================================================
# STEP 9 — TRAINING
# ============================================================

def train_agent(agent, env, episodes=TRAIN_EPISODES):
    global_step = 0

    episode_rewards = []
    success_history = []
    collision_history = []
    episode_losses = []

    print("\n" + "=" * 70)
    print("PHASE 4.4 — DUELING DOUBLE DQN TRAINING")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Episodes: {episodes}")
    print(f"Network: Dueling DQN")
    print(f"Target calculation: Double DQN")
    print(f"Gamma: {GAMMA}")
    print(f"Replay capacity: {REPLAY_CAPACITY}")
    print(f"Batch size: {BATCH_SIZE}")
    print("=" * 70)

    for episode in range(episodes):
        state = env.reset()

        total_reward = 0.0
        losses = []

        for _ in range(MAX_EPISODE_STEPS):
            mask = env.action_mask()
            q_values = agent.predict(state)

            epsilon = epsilon_by_step(global_step)

            action = select_action(
                q_values,
                mask,
                epsilon,
            )

            result = env.step(action)

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
                losses.append(loss)

            state = result.state
            total_reward += result.reward
            global_step += 1

            if result.terminated or result.truncated:
                break

        episode_rewards.append(total_reward)
        success_history.append(int(env.success))
        collision_history.append(int(env.collision))

        if losses:
            episode_losses.append(float(np.mean(losses)))
        else:
            episode_losses.append(0.0)

        if (episode + 1) % 25 == 0 or episode == 0:
            recent = slice(max(0, episode - 24), episode + 1)

            avg_reward = float(np.mean(episode_rewards[recent]))
            success_rate = float(np.mean(success_history[recent]))
            collision_rate = float(np.mean(collision_history[recent]))

            print(
                f"Episode {episode + 1:4d}/{episodes} | "
                f"Reward {avg_reward:8.2f} | "
                f"Success {success_rate * 100:5.1f}% | "
                f"Collision {collision_rate * 100:5.1f}% | "
                f"Epsilon {epsilon_by_step(global_step):.3f}"
            )

    agent.update_target_network()

    print("\nTraining finished.")
    print(f"Environment steps: {global_step}")
    print(f"Training updates: {agent.training_steps}")

    return {
        "episode_rewards": episode_rewards,
        "success_history": success_history,
        "collision_history": collision_history,
        "episode_losses": episode_losses,
    }


# ============================================================
# STEP 10 — STABLE ACTION SELECTION
# ============================================================

def choose_stable_action(
    q_values,
    mask,
    previous_action,
    hysteresis=0.12,
):
    """
    Avoid switching direction for tiny Q-value differences.

    Safety always takes priority:
    a previously selected action is retained only if it remains
    in the valid action mask.
    """
    valid_actions = np.flatnonzero(mask)

    if len(valid_actions) == 0:
        return 8

    best_action = int(
        valid_actions[
            np.argmax(q_values[valid_actions])
        ]
    )

    if (
        previous_action in valid_actions
        and previous_action != 8
        and best_action != 8
    ):
        if (
            q_values[previous_action]
            >= q_values[best_action] - hysteresis
        ):
            return int(previous_action)

    return best_action


# ============================================================
# STEP 11 — PYGAME ROCKET VISUALIZATION
# ============================================================

class RocketGUI:

    def __init__(self, env, agent, results):
        pygame.init()

        pygame.display.set_caption(
            "Phase 4.4 — Dueling Double DQN Rocket"
        )

        self.screen = pygame.display.set_mode(
            (WINDOW_WIDTH, WINDOW_HEIGHT)
        )

        self.clock = pygame.time.Clock()

        self.font = pygame.font.SysFont("consolas", 16)
        self.small_font = pygame.font.SysFont("consolas", 13)
        self.title_font = pygame.font.SysFont(
            "consolas",
            20,
            bold=True,
        )

        self.env = env
        self.agent = agent
        self.results = results

        self.paused = False
        self.running = True
        self.speed = 1

        self.current_action = 8
        self.previous_action = 8

        self.q_values = np.zeros(NUM_ACTIONS, dtype=np.float32)
        self.mask = np.ones(NUM_ACTIONS, dtype=bool)

        self.status = "READY"

        self.reset_robot()

    def reset_robot(self):
        self.env.set_scenario(
            np.array([80.0, 280.0], dtype=np.float32),
            np.array([620.0, 280.0], dtype=np.float32),
        )

        self.current_action = 8
        self.previous_action = 8
        self.status = "READY"

    def set_target(self, mouse_position):
        x, y = mouse_position

        if not (0 <= x <= WORLD_WIDTH and 0 <= y <= WORLD_HEIGHT):
            return

        target = np.array([x, y], dtype=np.float32)

        if not is_inside_world(target):
            self.status = "INVALID TARGET"
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

        self.current_action = 8
        self.previous_action = 8
        self.env.previous_action = 8

        self.status = "NEW TARGET SET"

    def draw_text(
        self,
        message,
        x,
        y,
        color=(230, 230, 230),
        font=None,
    ):
        if font is None:
            font = self.font

        surface = font.render(str(message), True, color)
        self.screen.blit(surface, (x, y))

    def draw_rocket(self, x, y):
        """
        Rocket-shaped robot.
        Its nose follows the currently selected action.
        """
        vector = ACTION_VECTORS[self.current_action]

        if np.linalg.norm(vector) < 0.01:
            vector = ACTION_VECTORS[self.previous_action]

        if np.linalg.norm(vector) < 0.01:
            vector = np.array([1.0, 0.0])

        angle = math.atan2(vector[1], vector[0])

        local_points = [
            (20, 0),
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

            return int(x + rx), int(y + ry)

        rocket_points = [rotate(p) for p in local_points]

        flame_length = random.randint(9, 16)

        flame_points = [
            rotate((-13, -4)),
            rotate((-13 - flame_length, 0)),
            rotate((-13, 4)),
        ]

        pygame.draw.polygon(
            self.screen,
            (255, 145, 35),
            flame_points,
        )

        pygame.draw.polygon(
            self.screen,
            (235, 240, 250),
            rocket_points,
        )

        pygame.draw.polygon(
            self.screen,
            (70, 150, 230),
            rocket_points,
            2,
        )

        pygame.draw.circle(
            self.screen,
            (50, 150, 245),
            rotate((7, 0)),
            4,
        )

        pygame.draw.polygon(
            self.screen,
            (220, 65, 65),
            [
                rotate((-9, -6)),
                rotate((-17, -12)),
                rotate((-12, -2)),
            ],
        )

        pygame.draw.polygon(
            self.screen,
            (220, 65, 65),
            [
                rotate((-9, 6)),
                rotate((-17, 12)),
                rotate((-12, 2)),
            ],
        )

    def draw_world(self):
        self.screen.fill((18, 22, 30))

        pygame.draw.rect(
            self.screen,
            (28, 34, 44),
            (0, 0, WORLD_WIDTH, WORLD_HEIGHT),
        )

        # Background grid.
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

        center = tuple(OBSTACLE.astype(int))

        pygame.draw.circle(
            self.screen,
            (130, 75, 55),
            center,
            int(SAFE_DISTANCE),
            2,
        )

        pygame.draw.circle(
            self.screen,
            (170, 55, 55),
            center,
            int(OBSTACLE_RADIUS),
        )

        self.draw_text(
            "OBSTACLE",
            OBSTACLE[0] - 40,
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
            (tx - 18, ty),
            (tx + 18, ty),
            2,
        )

        pygame.draw.line(
            self.screen,
            (60, 230, 130),
            (tx, ty - 18),
            (tx, ty + 18),
            2,
        )

        self.draw_text(
            "TARGET",
            tx + 15,
            ty - 25,
            (100, 255, 160),
            self.small_font,
        )

        # Rocket.
        px, py = self.env.position.astype(int)

        pygame.draw.circle(
            self.screen,
            (90, 130, 210),
            (px, py),
            19,
            1,
        )

        self.draw_rocket(px, py)

    def draw_q_values(self):
        x0 = 720
        y0 = 280

        self.draw_text(
            "DUELING DOUBLE DQN Q-VALUES",
            x0,
            y0,
            (120, 210, 255),
            self.title_font,
        )

        valid_values = self.q_values[self.mask]

        if len(valid_values):
            low = float(np.min(valid_values))
            high = float(np.max(valid_values))
        else:
            low, high = -1.0, 1.0

        span = max(high - low, 1e-6)

        for i, name in enumerate(ACTION_NAMES):
            y = y0 + 34 + i * 27

            valid = bool(self.mask[i])
            selected = i == self.current_action

            color = (
                (100, 255, 150)
                if selected
                else (210, 210, 210)
            )

            if not valid:
                color = (100, 100, 100)

            self.draw_text(
                f"{name:10s}",
                x0,
                y,
                color,
                self.small_font,
            )

            self.draw_text(
                f"{self.q_values[i]:8.3f}",
                x0 + 105,
                y,
                color,
                self.small_font,
            )

            if valid:
                width = int(
                    90 * (self.q_values[i] - low) / span
                )

                pygame.draw.rect(
                    self.screen,
                    (60, 120, 190),
                    (x0 + 205, y + 3, width, 10),
                )
            else:
                pygame.draw.line(
                    self.screen,
                    (100, 100, 100),
                    (x0 + 205, y + 8),
                    (x0 + 270, y + 8),
                    2,
                )

    def draw_reward_curve(self):
        x0 = 720
        y0 = 555
        width = 260
        height = 55

        self.draw_text(
            "TRAINING REWARD (MOVING AVG.)",
            x0,
            y0 - 20,
            (120, 210, 255),
            self.small_font,
        )

        pygame.draw.rect(
            self.screen,
            (50, 60, 75),
            (x0, y0, width, height),
            1,
        )

        rewards = self.results["episode_rewards"]

        if len(rewards) < 2:
            return

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
            px = x0 + int(
                i / max(len(smoothed) - 1, 1) * width
            )

            py = y0 + height - int(
                (value - low) / span * (height - 6)
            )

            points.append((px, py))

        if len(points) > 1:
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

        self.draw_text(
            "ROBOT CONTROL",
            720,
            18,
            (120, 210, 255),
            self.title_font,
        )

        self.draw_text(
            f"Device: {DEVICE}",
            720,
            50,
            font=self.small_font,
        )

        self.draw_text(
            f"Position: ({self.env.position[0]:.1f}, "
            f"{self.env.position[1]:.1f})",
            720,
            80,
            font=self.small_font,
        )

        self.draw_text(
            f"Target distance: "
            f"{distance(self.env.position, self.env.target):.1f}",
            720,
            105,
            font=self.small_font,
        )

        self.draw_text(
            f"Action: {ACTION_NAMES[self.current_action]}",
            720,
            130,
            (120, 255, 160),
            self.small_font,
        )

        self.draw_text(
            f"Steps: {self.env.steps}",
            720,
            155,
            font=self.small_font,
        )

        self.draw_text(
            f"Reward: {self.env.total_reward:.2f}",
            720,
            180,
            font=self.small_font,
        )

        self.draw_text(
            f"Replay: {len(self.agent.replay_buffer)}",
            720,
            205,
            font=self.small_font,
        )

        self.draw_text(
            f"Updates: {self.agent.training_steps}",
            720,
            230,
            font=self.small_font,
        )

        self.draw_q_values()
        self.draw_reward_curve()

        self.draw_text(
            "CLICK: TARGET  SPACE: PAUSE  R: RESET",
            720,
            625,
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
                        self.set_target(event.pos)

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
                self.draw_text(
                    "PAUSED",
                    300,
                    25,
                    (255, 220, 80),
                    self.title_font,
                )

            if self.status != "READY":
                self.draw_text(
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
# STEP 12 — NUMERICAL VERIFICATION
# ============================================================

def run_tests():
    print("\n" + "=" * 68)
    print("PHASE 4.4 UNIT TESTS")
    print("=" * 68)

    passed = 0

    def check(name, condition):
        nonlocal passed

        assert condition, f"FAILED: {name}"

        passed += 1
        print(f"[PASS] {name}")

    # Test 1: Network output dimension.
    network = DuelingDQN().to(DEVICE)

    x = torch.zeros(
        (4, STATE_SIZE),
        dtype=torch.float32,
        device=DEVICE,
    )

    q_values = network(x)

    check(
        "Dueling network output shape is (4, 9)",
        tuple(q_values.shape) == (4, NUM_ACTIONS),
    )

    # Test 2: Value and advantage dimensions.
    q, value, advantage = network.forward_components(x)

    check(
        "Value stream output shape is (4, 1)",
        tuple(value.shape) == (4, 1),
    )

    check(
        "Advantage stream output shape is (4, 9)",
        tuple(advantage.shape) == (4, NUM_ACTIONS),
    )

    # Test 3: Dueling combination.
    expected_q = (
        value
        + advantage
        - advantage.mean(dim=1, keepdim=True)
    )

    check(
        "Dueling combination matches Q-values",
        torch.allclose(q, expected_q),
    )

    # Test 4: Mean advantage is removed.
    centered_advantage = (
        advantage - advantage.mean(dim=1, keepdim=True)
    )

    check(
        "Centered advantage has zero mean",
        torch.allclose(
            centered_advantage.mean(dim=1),
            torch.zeros(4, device=DEVICE),
            atol=1e-6,
        ),
    )

    # Test 5: Double DQN selects with online network,
    # then evaluates with target network.
    rewards = torch.tensor([1.0], device=DEVICE)

    online_q = torch.tensor(
        [[1.0, 5.0, 2.0]],
        device=DEVICE,
    )

    target_q = torch.tensor(
        [[10.0, 20.0, 30.0]],
        device=DEVICE,
    )

    terminated = torch.tensor([0.0], device=DEVICE)

    masks = torch.tensor(
        [[True, True, False]],
        dtype=torch.bool,
        device=DEVICE,
    )

    ddqn_target = calculate_double_dqn_targets(
        rewards,
        online_q,
        target_q,
        terminated,
        masks,
        gamma=0.9,
    )

    # Online chooses action 1 because Q_online = 5 is highest.
    # Target evaluates action 1 as 20.
    # y = 1 + 0.9 * 20 = 19.
    check(
        "Double DQN selects online action and evaluates target Q",
        torch.allclose(
            ddqn_target,
            torch.tensor([19.0], device=DEVICE),
        ),
    )

    # Test 6: Ordinary DQN would choose the maximum target Q.
    ordinary_dqn_target = 1.0 + 0.9 * 20.0

    check(
        "Double DQN target calculation is numerically correct",
        np.isclose(float(ddqn_target.item()), ordinary_dqn_target),
    )

    # Test 7: Terminal transitions do not bootstrap.
    terminal_target = calculate_double_dqn_targets(
        rewards=torch.tensor([2.0], device=DEVICE),
        online_next_q=torch.tensor(
            [[100.0, 200.0, 300.0]],
            device=DEVICE,
        ),
        target_next_q=torch.tensor(
            [[100.0, 200.0, 300.0]],
            device=DEVICE,
        ),
        terminated=torch.tensor([1.0], device=DEVICE),
        next_masks=torch.tensor(
            [[True, True, True]],
            dtype=torch.bool,
            device=DEVICE,
        ),
        gamma=0.9,
    )

    check(
        "Terminal target equals immediate reward",
        torch.allclose(
            terminal_target,
            torch.tensor([2.0], device=DEVICE),
        ),
    )

    # Test 8: Invalid actions cannot be selected.
    masked_q = torch.tensor(
        [[100.0, 5.0, 4.0]],
        device=DEVICE,
    )

    mask = torch.tensor(
        [[False, True, True]],
        dtype=torch.bool,
        device=DEVICE,
    )

    selected = masked_q.masked_fill(
        ~mask,
        -1e9,
    ).argmax(dim=1)

    check(
        "Masked invalid action is excluded",
        int(selected.item()) == 1,
    )

    # Test 9: Replay buffer.
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
        "Replay buffer stores transitions",
        len(replay) == 1,
    )

    # Test 10: Action vectors.
    norms = np.linalg.norm(ACTION_VECTORS[:8], axis=1)

    check(
        "Eight directional actions have unit length",
        np.allclose(norms, np.ones(8)),
    )

    # Test 11: State dimension.
    env = RocketNavigationEnv()
    state = env.reset()

    check(
        "Environment state dimension is 10",
        state.shape == (STATE_SIZE,),
    )

    # Test 12: Safety mask.
    action_mask = env.action_mask()

    check(
        "Safety mask has nine actions",
        action_mask.shape == (NUM_ACTIONS,),
    )

    # Test 13: Stay action is available.
    check(
        "Stay action is always available",
        bool(action_mask[8]),
    )

    # Test 14: Hysteresis.
    stable_action = choose_stable_action(
        q_values=np.array(
            [1.0, 0.95, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
            dtype=np.float32,
        ),
        mask=np.ones(NUM_ACTIONS, dtype=bool),
        previous_action=0,
        hysteresis=0.12,
    )

    check(
        "Hysteresis retains near-optimal previous action",
        stable_action == 0,
    )

    # Test 15: Target network synchronization.
    agent = DuelingDoubleDQNAgent()

    with torch.no_grad():
        for parameter in agent.policy_net.parameters():
            parameter.fill_(0.25)

    agent.update_target_network()

    same_parameters = all(
        torch.allclose(p1, p2)
        for p1, p2 in zip(
            agent.policy_net.parameters(),
            agent.target_net.parameters(),
        )
    )

    check(
        "Target network synchronization works",
        same_parameters,
    )

    print("-" * 68)
    print(f"Tests passed: {passed}/15")
    print("=" * 68)


# ============================================================
# STEP 13 — MAIN
# ============================================================

def main():
    set_seed()

    run_tests()

    env = RocketNavigationEnv()
    agent = DuelingDoubleDQNAgent()

    results = train_agent(
        agent,
        env,
        episodes=TRAIN_EPISODES,
    )

    agent.save("dueling_double_dqn.pth")

    print("\nSaved model: dueling_double_dqn.pth")

    gui = RocketGUI(
        env=env,
        agent=agent,
        results=results,
    )

    gui.run()


if __name__ == "__main__":
    main()
