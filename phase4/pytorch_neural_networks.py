
"""
======================================================================
RL COURSE — PHASE 4.2
PYTORCH NEURAL NETWORK + PYGAME ROCKET NAVIGATION
======================================================================

FILE:
    phase4/pytorch_neural_networks.py

STEP 1 — PYTORCH TENSORS
Purpose:
    Replace manual NumPy neural-network calculations.
What you learn:
    Tensor operations, tensor dimensions and device execution.
Implementation:
    torch.Tensor and automatic CPU/GPU detection.

STEP 2 — NEURAL NETWORK
Purpose:
    Approximate action values for rocket navigation.
What you learn:
    Fully connected layers, ReLU and forward propagation.
Implementation:
    10 -> 64 -> 32 -> 4 network.

STEP 3 — AUTOGRAD
Purpose:
    Calculate gradients automatically.
What you learn:
    Computational graphs, backward propagation and gradients.
Implementation:
    loss.backward().

STEP 4 — OPTIMIZATION
Purpose:
    Train the neural network.
What you learn:
    Mini-batch training and Adam.
Implementation:
    torch.optim.Adam and MSELoss.

STEP 5 — GRADIENT VERIFICATION
Purpose:
    Check PyTorch's gradient calculations.
What you learn:
    Compare autograd with finite differences.
Implementation:
    Numerical perturbation of a model parameter.

STEP 6 — ROCKET NAVIGATION
Purpose:
    Apply the trained network to an interactive environment.
What you learn:
    Neural-network inference in a real-time simulation.
Implementation:
    Pygame, obstacle avoidance and target selection.

STEP 7 — ANTI-VIBRATION CONTROL
Purpose:
    Reduce unnecessary direction switching.
What you learn:
    Hysteresis, momentum and geometric progress.
Implementation:
    Stable action selection and independent safety filtering.

REAL-WORLD APPLICATION:
    Autonomous warehouse robot navigation.

IMPORTANT:
    This is a supervised action-value approximation demonstration.
    It is not yet Deep Q-Learning (DQN).
======================================================================
"""

import time
import unittest
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# =====================================================================
# 1. CONFIGURATION
# =====================================================================

SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("PyTorch device:", DEVICE)

WORLD_WIDTH = 700
WORLD_HEIGHT = 580

START = np.array([80.0, 290.0])
OBSTACLE = np.array([350.0, 290.0])
DEFAULT_TARGET = np.array([620.0, 290.0])

ROCKET_RADIUS = 11
OBSTACLE_RADIUS = 58
SAFETY_MARGIN = 20

SAFE_DISTANCE = (
    ROCKET_RADIUS + OBSTACLE_RADIUS + SAFETY_MARGIN
)

STEP_SIZE = 7.0
GOAL_RADIUS = 15

WAYPOINT_CLEARANCE = SAFE_DISTANCE + 22

ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT"]

ACTION_VECTORS = np.array([
    [0.0, -1.0],
    [1.0, 0.0],
    [0.0, 1.0],
    [-1.0, 0.0]
], dtype=np.float64)

NETWORK_WEIGHT = 0.35
PROGRESS_WEIGHT = 0.65

MOMENTUM_BONUS = 0.12
REVERSE_PENALTY = 0.20
SWITCH_PENALTY = 0.04

HYSTERESIS_MARGIN = 0.12


# =====================================================================
# 2. GEOMETRY AND SAFETY
# =====================================================================

def distance(a, b):
    return float(np.linalg.norm(a - b))


def is_inside_world(position):

    x, y = position

    return (
        ROCKET_RADIUS <= x <= WORLD_WIDTH - ROCKET_RADIUS
        and ROCKET_RADIUS <= y <= WORLD_HEIGHT - ROCKET_RADIUS
    )


def is_position_safe(position):

    return (
        is_inside_world(position)
        and distance(position, OBSTACLE) >= SAFE_DISTANCE
    )


def predict_position(position, action):

    return (
        np.asarray(position, dtype=np.float64)
        + ACTION_VECTORS[action] * STEP_SIZE
    )


def segment_intersects_circle(start, end, center, radius):

    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)

    direction = end - start

    denominator = np.dot(direction, direction)

    if denominator < 1e-12:
        return distance(start, center) < radius

    t = np.dot(center - start, direction) / denominator
    t = np.clip(t, 0.0, 1.0)

    closest = start + t * direction

    return distance(closest, center) < radius


# =====================================================================
# 3. TARGET AND WAYPOINT PLANNING
# =====================================================================

def target_is_valid(target):

    return (
        is_inside_world(target)
        and distance(target, OBSTACLE) >= SAFE_DISTANCE
    )


def calculate_waypoint(position, target):

    intersects = segment_intersects_circle(
        position,
        target,
        OBSTACLE,
        SAFE_DISTANCE + 5
    )

    if not intersects:
        return None

    side = (
        1.0
        if target[1] > OBSTACLE[1] + 12
        else -1.0
    )

    return np.array([
        OBSTACLE[0],
        OBSTACLE[1] + side * WAYPOINT_CLEARANCE
    ])


def navigation_target(position, target):

    waypoint = calculate_waypoint(position, target)

    if waypoint is None:
        return target.copy()

    if distance(position, waypoint) > 13:
        return waypoint.copy()

    return target.copy()


# =====================================================================
# 4. STATE ENCODING
# =====================================================================

def encode_state(position, target):

    position = np.asarray(position, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)

    waypoint = navigation_target(position, target)

    target_delta = target - position
    obstacle_delta = OBSTACLE - position
    waypoint_delta = waypoint - position

    return np.array([
        position[0] / WORLD_WIDTH,
        position[1] / WORLD_HEIGHT,

        target_delta[0] / WORLD_WIDTH,
        target_delta[1] / WORLD_HEIGHT,

        obstacle_delta[0] / WORLD_WIDTH,
        obstacle_delta[1] / WORLD_HEIGHT,

        distance(position, OBSTACLE) / WORLD_WIDTH,
        distance(position, target) / WORLD_WIDTH,

        waypoint_delta[0] / WORLD_WIDTH,
        waypoint_delta[1] / WORLD_HEIGHT

    ], dtype=np.float32)


# =====================================================================
# 5. TEACHER ACTION VALUES
# =====================================================================

def teacher_q_values(position, target):

    waypoint = navigation_target(position, target)

    current_distance = distance(position, waypoint)

    q_values = np.zeros(4, dtype=np.float32)

    for action in range(4):

        candidate = predict_position(position, action)

        if not is_position_safe(candidate):

            q_values[action] = -5.0
            continue

        next_distance = distance(candidate, waypoint)

        progress = (
            current_distance - next_distance
        ) / STEP_SIZE

        clearance = (
            distance(candidate, OBSTACLE) - SAFE_DISTANCE
        )

        clearance_bonus = min(
            max(clearance, 0.0) / 100.0,
            0.15
        )

        q_values[action] = progress + clearance_bonus

    return q_values


# =====================================================================
# 6. DATASET
# =====================================================================

def random_safe_position(rng):

    while True:

        position = np.array([
            rng.uniform(20, WORLD_WIDTH - 20),
            rng.uniform(20, WORLD_HEIGHT - 20)
        ])

        if is_position_safe(position):
            return position


def random_target(rng):

    while True:

        target = np.array([
            rng.uniform(20, WORLD_WIDTH - 20),
            rng.uniform(20, WORLD_HEIGHT - 20)
        ])

        if target_is_valid(target):
            return target


def generate_dataset(
    num_samples=16000,
    seed=SEED
):

    rng = np.random.default_rng(seed)

    X = np.zeros((num_samples, 10), dtype=np.float32)
    Y = np.zeros((num_samples, 4), dtype=np.float32)

    for i in range(num_samples):

        position = random_safe_position(rng)
        target = random_target(rng)

        X[i] = encode_state(position, target)

        Y[i] = teacher_q_values(position, target)

    return X, Y


# =====================================================================
# 7. PYTORCH NEURAL NETWORK
# =====================================================================

class RocketQNetwork(nn.Module):

    """
    Input:
        10 state features.

    Hidden layers:
        64 neurons + ReLU.
        32 neurons + ReLU.

    Output:
        4 action values.

    The network approximates:
        Q(state, action)

    These are supervised target values, not yet values learned
    through temporal-difference reinforcement learning.
    """

    def __init__(self):

        super().__init__()

        self.layers = nn.Sequential(

            nn.Linear(10, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 4)

        )

    def forward(self, x):

        return self.layers(x)


# =====================================================================
# 8. GRADIENT CHECK
# =====================================================================

def numerical_gradient_check(
    model,
    x,
    y,
    epsilon=1e-3
):

    """
    Compare:
        PyTorch autograd gradient
        Numerical central-difference gradient

    Numerical derivative:
        df/dw ~= [f(w+eps)-f(w-eps)] / (2*eps)
    """

    model.zero_grad()

    prediction = model(x)

    loss = torch.mean((prediction - y) ** 2)

    loss.backward()

    parameter = model.layers[0].weight

    analytical = parameter.grad[0, 0].item()

    original = parameter.data[0, 0].item()

    with torch.no_grad():

        parameter[0, 0] = original + epsilon

        loss_plus = torch.mean(
            (model(x) - y) ** 2
        ).item()

        parameter[0, 0] = original - epsilon

        loss_minus = torch.mean(
            (model(x) - y) ** 2
        ).item()

        parameter[0, 0] = original

    numerical = (
        loss_plus - loss_minus
    ) / (2 * epsilon)

    return analytical, numerical


# =====================================================================
# 9. TRAINING
# =====================================================================

def train_network(
    model,
    X,
    Y,
    epochs=120,
    batch_size=128,
    learning_rate=0.001
):

    X_tensor = torch.tensor(
        X,
        dtype=torch.float32
    )

    Y_tensor = torch.tensor(
        Y,
        dtype=torch.float32
    )

    dataset = TensorDataset(
        X_tensor,
        Y_tensor
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

    model.to(DEVICE)

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    initial_loss = None
    history = []

    for epoch in range(epochs):

        model.train()

        epoch_loss = 0.0
        count = 0

        for states, targets in loader:

            states = states.to(DEVICE)
            targets = targets.to(DEVICE)

            optimizer.zero_grad()

            predictions = model(states)

            loss = criterion(
                predictions,
                targets
            )

            loss.backward()

            # Prevent unusually large gradients.
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0
            )

            optimizer.step()

            epoch_loss += loss.item() * len(states)
            count += len(states)

        average_loss = epoch_loss / count

        history.append(average_loss)

        if epoch == 0:
            initial_loss = average_loss

        if (epoch + 1) % 20 == 0:

            print(
                f"Epoch {epoch + 1:3d}/{epochs}"
                f" | Loss: {average_loss:.6f}"
            )

    model.eval()

    with torch.no_grad():

        prediction = model(
            X_tensor.to(DEVICE)
        )

        final_loss = criterion(
            prediction,
            Y_tensor.to(DEVICE)
        ).item()

    return {
        "model": model,
        "losses": history,
        "initial_loss": initial_loss,
        "final_loss": final_loss
    }


# =====================================================================
# 10. INFERENCE
# =====================================================================

def predict_q_values(model, state):

    model.eval()

    state_tensor = torch.tensor(
        state,
        dtype=torch.float32,
        device=DEVICE
    ).unsqueeze(0)

    with torch.no_grad():

        q_values = model(state_tensor)

    return q_values.squeeze(0).cpu().numpy()


# =====================================================================
# 11. ANTI-VIBRATION CONTROLLER
# =====================================================================

def safe_action_mask(position):

    return np.array([
        is_position_safe(
            predict_position(position, action)
        )
        for action in range(4)
    ], dtype=bool)


def geometric_progress(position, target):

    current_distance = distance(position, target)

    scores = np.zeros(4)

    for action in range(4):

        candidate = predict_position(position, action)

        scores[action] = (
            current_distance - distance(candidate, target)
        ) / STEP_SIZE

    return scores


def choose_stable_action(
    q_values,
    position,
    target,
    previous_action=None
):

    mask = safe_action_mask(position)

    if not np.any(mask):
        return None, mask, np.zeros(4)

    waypoint = navigation_target(position, target)

    progress = geometric_progress(
        position,
        waypoint
    )

    q_values = np.asarray(q_values, dtype=np.float64)

    q_min = np.min(q_values[mask])
    q_max = np.max(q_values[mask])

    if q_max - q_min > 1e-8:

        normalized_q = (
            q_values - q_min
        ) / (q_max - q_min)

    else:

        normalized_q = np.zeros(4)

    scores = (
        NETWORK_WEIGHT * normalized_q
        + PROGRESS_WEIGHT * progress
    )

    if previous_action is not None:

        for action in range(4):

            if not mask[action]:
                continue

            if action == previous_action:

                scores[action] += MOMENTUM_BONUS

            elif action == (previous_action + 2) % 4:

                scores[action] -= REVERSE_PENALTY

            else:

                scores[action] -= SWITCH_PENALTY

    scores[~mask] = -np.inf

    best_action = int(np.argmax(scores))

    if (
        previous_action is not None
        and mask[previous_action]
    ):

        if (
            scores[best_action] - scores[previous_action]
            <= HYSTERESIS_MARGIN
        ):

            best_action = previous_action

    return best_action, mask, scores


# =====================================================================
# 12. ENVIRONMENT
# =====================================================================

class RobotEnvironment:

    def __init__(self):
        self.reset()

    def reset(self):

        self.position = START.copy()
        self.target = DEFAULT_TARGET.copy()

        self.steps = 0
        self.collisions = 0
        self.success = False

        self.previous_action = None

    def set_target(self, target):

        target = np.asarray(target, dtype=np.float64)

        if not target_is_valid(target):
            return False

        self.target = target.copy()
        self.success = False
        self.steps = 0

        return True

    def step(self, action):

        if action is None:
            return False

        candidate = predict_position(
            self.position,
            action
        )

        if not is_position_safe(candidate):

            self.collisions += 1
            return False

        self.position = candidate
        self.previous_action = action
        self.steps += 1

        if distance(
            self.position,
            self.target
        ) <= GOAL_RADIUS:

            self.success = True

        return self.success


# =====================================================================
# 13. UNIT TESTS
# =====================================================================

def run_tests():

    print("\n========== UNIT TESTS ==========")

    class TestPhase42(unittest.TestCase):

        def test_network_output(self):

            model = RocketQNetwork().to(DEVICE)

            x = torch.zeros(
                (5, 10),
                device=DEVICE
            )

            y = model(x)

            self.assertEqual(
                tuple(y.shape),
                (5, 4)
            )

        def test_gradients_exist(self):

            model = RocketQNetwork().to(DEVICE)

            x = torch.randn(
                (8, 10),
                device=DEVICE
            )

            y = torch.randn(
                (8, 4),
                device=DEVICE
            )

            loss = torch.mean(
                (model(x) - y) ** 2
            )

            loss.backward()

            for parameter in model.parameters():

                self.assertIsNotNone(parameter.grad)

                self.assertTrue(
                    torch.isfinite(parameter.grad).all()
                )

        def test_gradient_check(self):

            torch.manual_seed(42)

            model = RocketQNetwork().to(DEVICE)

            x = torch.randn(
                (4, 10),
                device=DEVICE
            )

            y = torch.randn(
                (4, 4),
                device=DEVICE
            )

            analytical, numerical = numerical_gradient_check(
                model,
                x,
                y
            )

            self.assertAlmostEqual(
                analytical,
                numerical,
                delta=0.003
            )

        def test_dataset(self):

            X, Y = generate_dataset(100)

            self.assertEqual(X.shape, (100, 10))
            self.assertEqual(Y.shape, (100, 4))

        def test_obstacle_safety(self):

            self.assertFalse(
                is_position_safe(OBSTACLE)
            )

            self.assertTrue(
                is_position_safe(START)
            )

        def test_waypoint(self):

            waypoint = calculate_waypoint(
                START,
                DEFAULT_TARGET
            )

            self.assertIsNotNone(waypoint)

            self.assertGreaterEqual(
                distance(waypoint, OBSTACLE),
                SAFE_DISTANCE
            )

        def test_target_update(self):

            env = RobotEnvironment()

            target = np.array([600.0, 150.0])

            self.assertTrue(
                env.set_target(target)
            )

            np.testing.assert_allclose(
                env.target,
                target
            )

        def test_invalid_target(self):

            env = RobotEnvironment()

            self.assertFalse(
                env.set_target(OBSTACLE)
            )

        def test_safety_filter(self):

            position = np.array([
                OBSTACLE[0] - SAFE_DISTANCE + 2,
                OBSTACLE[1]
            ])

            q = np.array([0.0, 100.0, 0.0, 0.0])

            action, mask, scores = choose_stable_action(
                q,
                position,
                DEFAULT_TARGET
            )

            self.assertNotEqual(action, 1)
            self.assertTrue(mask[action])

        def test_hysteresis(self):

            position = np.array([200.0, 100.0])
            target = np.array([300.0, 100.0])

            q = np.array([0.0, 1.0, 0.0, 0.0])

            action, mask, scores = choose_stable_action(
                q,
                position,
                target,
                previous_action=1
            )

            self.assertEqual(action, 1)

        def test_training_reduces_loss(self):

            X, Y = generate_dataset(
                1000,
                seed=123
            )

            torch.manual_seed(123)

            model = RocketQNetwork().to(DEVICE)

            result = train_network(
                model,
                X,
                Y,
                epochs=30,
                batch_size=64
            )

            self.assertLess(
                result["final_loss"],
                result["initial_loss"]
            )

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        TestPhase42
    )

    result = unittest.TextTestRunner(
        verbosity=2
    ).run(suite)

    if not result.wasSuccessful():

        raise RuntimeError("Phase 4.2 tests failed.")

    print("\nAll tests passed.")


# =====================================================================
# 14. PYGAME VISUALIZATION
# =====================================================================

def run_pygame(training_result):

    import pygame

    pygame.init()

    SCREEN_WIDTH = 1180
    SCREEN_HEIGHT = 780

    screen = pygame.display.set_mode(
        (SCREEN_WIDTH, SCREEN_HEIGHT)
    )

    pygame.display.set_caption(
        "Phase 4.2 — PyTorch Rocket Navigation"
    )

    clock = pygame.time.Clock()

    font = pygame.font.SysFont("consolas", 16)
    small = pygame.font.SysFont("consolas", 13)
    title = pygame.font.SysFont("consolas", 20, bold=True)

    BACKGROUND = (22, 26, 34)
    PANEL = (34, 40, 51)
    GRID = (49, 57, 70)

    WHITE = (240, 243, 248)
    MUTED = (165, 175, 190)

    GREEN = (75, 205, 130)
    RED = (235, 85, 85)
    BLUE = (85, 160, 245)
    YELLOW = (245, 205, 85)
    ORANGE = (240, 145, 65)
    PURPLE = (165, 125, 230)

    WORLD_X = 20
    WORLD_Y = 115

    WORLD_RECT = pygame.Rect(
        WORLD_X,
        WORLD_Y,
        WORLD_WIDTH,
        WORLD_HEIGHT
    )

    model = training_result["model"]

    env = RobotEnvironment()

    paused = False
    running = True

    speed = 1.0
    interval = 0.055

    last_step = 0.0
    exhaust_phase = 0.0

    selected_action = None

    q_values = np.zeros(4)
    mask = safe_action_mask(env.position)
    scores = np.zeros(4)

    message = "LEFT CLICK TO SET A TARGET"
    message_color = YELLOW
    message_until = 0.0

    losses = training_result["losses"]

    def screen_point(position):

        return (
            int(WORLD_X + position[0]),
            int(WORLD_Y + position[1])
        )

    def world_point(mouse):

        return np.array([
            mouse[0] - WORLD_X,
            mouse[1] - WORLD_Y
        ], dtype=np.float64)

    def draw_text(
        text,
        x,
        y,
        color=WHITE,
        selected_font=font
    ):

        surface = selected_font.render(
            str(text),
            True,
            color
        )

        screen.blit(surface, (x, y))

    # ---------------------------------------------------------------
    # ROCKET GRAPHICS
    # ---------------------------------------------------------------

    def rotate_point(point, angle):

        x, y = point

        c = np.cos(angle)
        s = np.sin(angle)

        return np.array([
            x * c - y * s,
            x * s + y * c
        ])

    def draw_rocket(position, action):

        if action is None:
            action = 1

        direction = ACTION_VECTORS[action]

        angle = np.arctan2(
            direction[1],
            direction[0]
        )

        center = np.array(
            screen_point(position),
            dtype=np.float64
        )

        def transform(point):

            return tuple(
                np.round(
                    center + rotate_point(point, angle)
                ).astype(int)
            )

        flame_length = 13 + 4 * np.sin(
            exhaust_phase * 18
        )

        pygame.draw.polygon(
            screen,
            ORANGE,
            [
                transform((-7, -4)),
                transform((-7 - flame_length, 0)),
                transform((-7, 4))
            ]
        )

        pygame.draw.polygon(
            screen,
            YELLOW,
            [
                transform((-7, -2)),
                transform((-7 - flame_length * 0.65, 0)),
                transform((-7, 2))
            ]
        )

        body = [
            (17, 0),
            (7, -6),
            (-9, -6),
            (-13, -3),
            (-13, 3),
            (-9, 6),
            (7, 6)
        ]

        pygame.draw.polygon(
            screen,
            BLUE,
            [transform(p) for p in body]
        )

        pygame.draw.polygon(
            screen,
            WHITE,
            [transform(p) for p in body],
            2
        )

        pygame.draw.polygon(
            screen,
            RED,
            [
                transform((17, 0)),
                transform((7, -6)),
                transform((7, 6))
            ]
        )

        pygame.draw.polygon(
            screen,
            PURPLE,
            [
                transform((-7, -5)),
                transform((-15, -12)),
                transform((-11, -4))
            ]
        )

        pygame.draw.polygon(
            screen,
            PURPLE,
            [
                transform((-7, 5)),
                transform((-15, 12)),
                transform((-11, 4))
            ]
        )

        pygame.draw.circle(
            screen,
            (170, 225, 255),
            transform((5, 0)),
            3
        )

    # ---------------------------------------------------------------
    # WORLD
    # ---------------------------------------------------------------

    def draw_world():

        pygame.draw.rect(
            screen,
            PANEL,
            WORLD_RECT,
            border_radius=8
        )

        for x in range(
            WORLD_X,
            WORLD_X + WORLD_WIDTH,
            35
        ):

            pygame.draw.line(
                screen,
                GRID,
                (x, WORLD_Y),
                (x, WORLD_Y + WORLD_HEIGHT)
            )

        for y in range(
            WORLD_Y,
            WORLD_Y + WORLD_HEIGHT,
            35
        ):

            pygame.draw.line(
                screen,
                GRID,
                (WORLD_X, y),
                (WORLD_X + WORLD_WIDTH, y)
            )

        pygame.draw.circle(
            screen,
            ORANGE,
            screen_point(OBSTACLE),
            int(SAFE_DISTANCE),
            1
        )

        pygame.draw.circle(
            screen,
            RED,
            screen_point(OBSTACLE),
            OBSTACLE_RADIUS
        )

        pygame.draw.circle(
            screen,
            WHITE,
            screen_point(OBSTACLE),
            OBSTACLE_RADIUS,
            2
        )

        pygame.draw.circle(
            screen,
            GREEN,
            screen_point(env.target),
            GOAL_RADIUS
        )

        pygame.draw.circle(
            screen,
            WHITE,
            screen_point(env.target),
            GOAL_RADIUS,
            2
        )

        waypoint = calculate_waypoint(
            env.position,
            env.target
        )

        if waypoint is not None:

            if distance(env.position, waypoint) > 13:

                pygame.draw.circle(
                    screen,
                    YELLOW,
                    screen_point(waypoint),
                    9,
                    2
                )

        pygame.draw.circle(
            screen,
            PURPLE,
            screen_point(START),
            4
        )

        for action in range(4):

            start = screen_point(env.position)

            end = screen_point(
                predict_position(
                    env.position,
                    action
                )
            )

            if not mask[action]:
                color = RED

            elif action == selected_action:
                color = YELLOW

            else:
                color = GREEN

            pygame.draw.line(
                screen,
                color,
                start,
                end,
                3
            )

        draw_rocket(
            env.position,
            env.previous_action
        )

        draw_text(
            "PYTORCH ROCKET NAVIGATION",
            WORLD_X + 10,
            WORLD_Y - 28,
            WHITE,
            title
        )

    # ---------------------------------------------------------------
    # Q-VALUE DISPLAY
    # ---------------------------------------------------------------

    def draw_q_values():

        draw_text(
            "PYTORCH ACTION VALUES",
            770,
            150,
            WHITE,
            title
        )

        for i, name in enumerate(ACTIONS):

            y = 200 + i * 42

            if i == selected_action:
                color = YELLOW
                status = "SELECTED"

            elif mask[i]:
                color = GREEN
                status = "SAFE"

            else:
                color = RED
                status = "BLOCKED"

            draw_text(
                name,
                770,
                y,
                color
            )

            draw_text(
                f"Q = {q_values[i]:+.4f}",
                870,
                y,
                WHITE
            )

            draw_text(
                f"S = {scores[i]:+.3f}",
                990,
                y,
                MUTED
            )

            draw_text(
                status,
                1090,
                y,
                color,
                small
            )

    # ---------------------------------------------------------------
    # NETWORK STRUCTURE
    # ---------------------------------------------------------------

    def draw_network():

        draw_text(
            "NETWORK ARCHITECTURE",
            770,
            405,
            WHITE,
            title
        )

        draw_text(
            "10 -> 64 -> 32 -> 4",
            770,
            438,
            GREEN
        )

        draw_text(
            "Framework: PyTorch",
            770,
            465,
            MUTED
        )

        draw_text(
            f"Device: {DEVICE}",
            770,
            490,
            MUTED
        )

        draw_text(
            "Autograd: ENABLED",
            770,
            515,
            GREEN
        )

        draw_text(
            "Optimizer: Adam",
            770,
            540,
            MUTED
        )

        draw_text(
            "Controller: Hysteresis + Safety Filter",
            770,
            565,
            MUTED,
            small
        )

    # ---------------------------------------------------------------
    # TRAINING LOSS
    # ---------------------------------------------------------------

    def draw_loss():

        draw_text(
            "TRAINING LOSS",
            770,
            620,
            WHITE,
            title
        )

        chart = pygame.Rect(
            770,
            650,
            350,
            70
        )

        pygame.draw.rect(
            screen,
            (25, 30, 40),
            chart,
            border_radius=5
        )

        if len(losses) > 1:

            low = min(losses)
            high = max(losses)

            span = max(high - low, 1e-9)

            points = []

            for i, value in enumerate(losses):

                x = chart.left + int(
                    i / (len(losses) - 1)
                    * chart.width
                )

                y = chart.bottom - int(
                    (value - low) / span
                    * (chart.height - 10)
                ) - 5

                points.append((x, y))

            pygame.draw.lines(
                screen,
                GREEN,
                False,
                points,
                2
            )

        draw_text(
            f"Final loss: {training_result['final_loss']:.6f}",
            770,
            730,
            MUTED,
            small
        )

    # ---------------------------------------------------------------
    # MAIN LOOP
    # ---------------------------------------------------------------

    while running:

        clock.tick(60)

        now = time.perf_counter()

        exhaust_phase += 1 / 60

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

                    selected_action = None

                    message = "ROCKET RESET"
                    message_color = YELLOW
                    message_until = now + 1.5

                elif event.key in (
                    pygame.K_PLUS,
                    pygame.K_EQUALS,
                    pygame.K_KP_PLUS
                ):

                    speed = min(speed * 1.5, 8.0)

                elif event.key in (
                    pygame.K_MINUS,
                    pygame.K_KP_MINUS
                ):

                    speed = max(speed / 1.5, 0.2)

            elif (
                event.type == pygame.MOUSEBUTTONDOWN
                and event.button == 1
            ):

                if WORLD_RECT.collidepoint(event.pos):

                    target = world_point(event.pos)

                    if env.set_target(target):

                        selected_action = None

                        message = "TARGET UPDATED"
                        message_color = GREEN

                    else:

                        message = "INVALID TARGET"
                        message_color = RED

                    message_until = now + 2.0

        # Neural-network inference.
        if (
            not paused
            and not env.success
            and now - last_step >= interval / speed
        ):

            state = encode_state(
                env.position,
                env.target
            )

            q_values = predict_q_values(
                model,
                state
            )

            action, mask, scores = choose_stable_action(
                q_values,
                env.position,
                env.target,
                env.previous_action
            )

            selected_action = action

            env.step(action)

            last_step = now

            if env.success:

                message = "ROCKET REACHED TARGET!"
                message_color = GREEN
                message_until = now + 2.0

        screen.fill(BACKGROUND)

        draw_world()
        draw_q_values()
        draw_network()
        draw_loss()

        draw_text(
            f"Steps: {env.steps}",
            20,
            24,
            WHITE,
            title
        )

        draw_text(
            f"Collisions: {env.collisions}",
            180,
            24,
            RED
        )

        draw_text(
            f"Speed: {speed:.1f}x",
            400,
            24,
            YELLOW
        )

        if paused:

            draw_text(
                "PAUSED",
                500,
                65,
                YELLOW,
                title
            )

        elif env.success:

            draw_text(
                "TARGET REACHED",
                500,
                65,
                GREEN,
                title
            )

        if now < message_until:

            draw_text(
                message,
                20,
                65,
                message_color,
                small
            )

        draw_text(
            "LEFT CLICK: TARGET | SPACE: PAUSE | "
            "R: RESET | +/-: SPEED | ESC: EXIT",
            20,
            755,
            MUTED,
            small
        )

        pygame.display.flip()

    pygame.quit()


# =====================================================================
# 15. MAIN
# =====================================================================

def main():

    print("=" * 65)
    print("RL COURSE — PHASE 4.2")
    print("PYTORCH NEURAL NETWORK + ROCKET NAVIGATION")
    print("=" * 65)

    print("\nRunning unit tests...")

    run_tests()

    print("\nGenerating training dataset...")

    X, Y = generate_dataset(
        num_samples=16000
    )

    print("X shape:", X.shape)
    print("Y shape:", Y.shape)

    print("\nCreating PyTorch model...")

    model = RocketQNetwork().to(DEVICE)

    print(model)

    # Verify autograd against numerical gradients.
    x = torch.randn(
        (4, 10),
        device=DEVICE
    )

    y = torch.randn(
        (4, 4),
        device=DEVICE
    )

    analytical, numerical = numerical_gradient_check(
        model,
        x,
        y
    )

    print("\nGradient verification:")

    print("Autograd:", analytical)
    print("Numerical:", numerical)

    print(
        "Absolute error:",
        abs(analytical - numerical)
    )

    print("\nTraining PyTorch neural network...")

    result = train_network(
        model,
        X,
        Y,
        epochs=120,
        batch_size=128,
        learning_rate=0.001
    )

    print("\nTraining completed.")

    print(
        "Initial loss:",
        result["initial_loss"]
    )

    print(
        "Final loss:",
        result["final_loss"]
    )

    print("\nStarting Pygame simulation...")

    run_pygame(result)


if __name__ == "__main__":
    main()
