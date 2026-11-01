
"""
======================================================================
RL COURSE — PHASE 4.1
NEURAL NETWORK + INTERACTIVE ROCKET NAVIGATION
======================================================================

FILE:
    phase4/neural_networks_from_scratch.py

LESSON ROADMAP
--------------

STEP 1 — Target-conditioned robot state
Purpose:
    Give the network information about the robot, target and obstacle.
What you learn:
    Why a navigation policy must know the destination.
Implementation:
    10-dimensional state vector.

STEP 2 — Obstacle-aware waypoint planning
Purpose:
    Avoid the obstacle when the direct path is blocked.
What you learn:
    Geometric route planning and safety clearance.
Implementation:
    Upper/lower waypoint selection.

STEP 3 — Neural network from scratch
Purpose:
    Predict action values.
What you learn:
    Forward propagation, ReLU and Q-value approximation.
Implementation:
    NumPy network: 10 -> 32 -> 4.

STEP 4 — Backpropagation and Adam
Purpose:
    Train the neural network.
What you learn:
    Chain rule, gradients and adaptive optimization.
Implementation:
    Manual backpropagation and Adam.

STEP 5 — Anti-vibration action selection
Purpose:
    Prevent repeated left-right or up-down switching.
What you learn:
    Action hysteresis, movement momentum and progress scoring.
Implementation:
    Combine network predictions with geometric progress.

STEP 6 — Independent safety filter
Purpose:
    Reject actions that would cause an immediate collision.
What you learn:
    Safety filtering independent of neural-network predictions.
Implementation:
    Predict candidate positions and mask unsafe actions.

STEP 7 — Rocket-shaped Pygame simulation
Purpose:
    Visualize navigation and interactively change destinations.
What you learn:
    Connect neural-network inference to a simulated robot.
Implementation:
    Animated rocket, exhaust, target selection and live Q-values.

STEP 8 — Verification
Purpose:
    Verify network dimensions, safety and anti-vibration behavior.
What you learn:
    Numerical and functional testing.
Implementation:
    Automated unit tests.

REAL-WORLD ROBOTICS APPLICATION
-------------------------------
This is a starting point for autonomous mobile robot navigation.

A real robot additionally requires:
    - Localization
    - Sensor-based obstacle detection
    - Dynamic obstacle prediction
    - Continuous trajectory collision checking
    - Emergency stop and hardware safety controls

IMPORTANT:
    This is supervised neural-network action-value approximation.
    It is not yet DQN. DQN comes later in Phase 4.

GUI CONTROLS
------------
LEFT CLICK : Set a new target
SPACE      : Pause / resume
R          : Reset rocket
+ / -      : Change simulation speed
ESC        : Exit

======================================================================
"""

import time
import unittest
import numpy as np


# =====================================================================
# 1. CONFIGURATION
# =====================================================================

SEED = 42

WORLD_WIDTH = 700
WORLD_HEIGHT = 580

ROCKET_RADIUS = 11

OBSTACLE_RADIUS = 58
SAFETY_MARGIN = 20

SAFE_DISTANCE = (
    OBSTACLE_RADIUS
    + ROCKET_RADIUS
    + SAFETY_MARGIN
)

STEP_SIZE = 7.0
GOAL_RADIUS = 15

START = np.array([80.0, 290.0])
OBSTACLE = np.array([350.0, 290.0])
DEFAULT_TARGET = np.array([620.0, 290.0])

WAYPOINT_CLEARANCE = SAFE_DISTANCE + 22.0

ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT"]

ACTION_VECTORS = np.array([
    [0.0, -1.0],
    [1.0, 0.0],
    [0.0, 1.0],
    [-1.0, 0.0],
], dtype=np.float64)

# Anti-vibration controller parameters.
#
# The network remains important, but it is not allowed to make
# arbitrary rapid direction changes when action scores are similar.
#
# PROGRESS_WEIGHT:
#     How strongly the controller prefers movement toward the waypoint.
#
# MOMENTUM_BONUS:
#     Reward for continuing in the previous direction.
#
# REVERSE_PENALTY:
#     Penalty for immediately reversing direction.
#
# SWITCH_PENALTY:
#     Small penalty for changing direction.
#
# HYSTERESIS_MARGIN:
#     If the previous safe action is almost as good as the new best
#     action, keep the previous action.

PROGRESS_WEIGHT = 0.65
NETWORK_WEIGHT = 0.35

MOMENTUM_BONUS = 0.12
REVERSE_PENALTY = 0.20
SWITCH_PENALTY = 0.04

HYSTERESIS_MARGIN = 0.12

# A direction must remain selected for at least this many frames
# before another non-emergency direction change is allowed.
MIN_ACTION_HOLD_TIME = 0.12


# =====================================================================
# 2. GEOMETRY AND SAFETY
# =====================================================================

def distance(a, b):
    return float(np.linalg.norm(a - b))


def is_inside_world(position):

    x, y = position

    return (
        ROCKET_RADIUS <= x <= WORLD_WIDTH - ROCKET_RADIUS
        and
        ROCKET_RADIUS <= y <= WORLD_HEIGHT - ROCKET_RADIUS
    )


def is_position_safe(position):

    position = np.asarray(position, dtype=np.float64)

    if not is_inside_world(position):
        return False

    return (
        distance(position, OBSTACLE)
        >= SAFE_DISTANCE
    )


def predict_position(position, action):

    return (
        np.asarray(position, dtype=np.float64)
        + ACTION_VECTORS[action] * STEP_SIZE
    )


def segment_intersects_circle(
    start,
    end,
    center,
    radius
):

    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)

    direction = end - start

    denominator = float(
        np.dot(direction, direction)
    )

    if denominator < 1e-12:
        return distance(start, center) < radius

    t = np.dot(
        center - start,
        direction
    ) / denominator

    t = float(np.clip(t, 0.0, 1.0))

    closest = start + t * direction

    return distance(closest, center) < radius


# =====================================================================
# 3. TARGET AND WAYPOINT PLANNING
# =====================================================================

def target_is_valid(target):

    target = np.asarray(target, dtype=np.float64)

    if not is_inside_world(target):
        return False

    return (
        distance(target, OBSTACLE)
        >= SAFE_DISTANCE
    )


def calculate_waypoint(position, target):

    position = np.asarray(position, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)

    intersects = segment_intersects_circle(
        position,
        target,
        OBSTACLE,
        SAFE_DISTANCE + 5.0
    )

    if not intersects:
        return None

    vertical_difference = target[1] - OBSTACLE[1]

    if vertical_difference > 12.0:
        side = 1.0
    else:
        side = -1.0

    return np.array([
        OBSTACLE[0],
        OBSTACLE[1] + side * WAYPOINT_CLEARANCE
    ])


def navigation_target(position, target):

    waypoint = calculate_waypoint(
        position,
        target
    )

    if waypoint is None:
        return target.copy()

    if distance(position, waypoint) > 13.0:
        return waypoint

    return target.copy()


# =====================================================================
# 4. STATE ENCODING
# =====================================================================

def encode_state(position, target):

    position = np.asarray(position, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)

    current_target = navigation_target(
        position,
        target
    )

    target_delta = target - position
    obstacle_delta = OBSTACLE - position
    waypoint_delta = current_target - position

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
        waypoint_delta[1] / WORLD_HEIGHT,

    ], dtype=np.float32)


# =====================================================================
# 5. TEACHER ACTION VALUES
# =====================================================================

def teacher_q_values(position, target):

    current_target = navigation_target(
        position,
        target
    )

    current_distance = distance(
        position,
        current_target
    )

    q_values = np.zeros(4, dtype=np.float32)

    for action in range(4):

        candidate = predict_position(
            position,
            action
        )

        if not is_position_safe(candidate):

            q_values[action] = -5.0
            continue

        next_distance = distance(
            candidate,
            current_target
        )

        progress = (
            current_distance - next_distance
        ) / STEP_SIZE

        clearance = (
            distance(candidate, OBSTACLE)
            - SAFE_DISTANCE
        )

        clearance_bonus = min(
            max(clearance, 0.0) / 100.0,
            0.15
        )

        q_values[action] = (
            progress + clearance_bonus
        )

    return q_values


# =====================================================================
# 6. DATASET GENERATION
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
    num_samples=12000,
    seed=SEED
):

    rng = np.random.default_rng(seed)

    X = np.zeros(
        (num_samples, 10),
        dtype=np.float32
    )

    Y = np.zeros(
        (num_samples, 4),
        dtype=np.float32
    )

    for i in range(num_samples):

        position = random_safe_position(rng)
        target = random_target(rng)

        X[i] = encode_state(
            position,
            target
        )

        Y[i] = teacher_q_values(
            position,
            target
        )

    return X, Y


# =====================================================================
# 7. NEURAL NETWORK FROM SCRATCH
# =====================================================================

class NeuralNetwork:

    def __init__(
        self,
        input_size=10,
        hidden_size=32,
        output_size=4,
        seed=SEED
    ):

        rng = np.random.default_rng(seed)

        self.W1 = (
            rng.standard_normal(
                (input_size, hidden_size)
            )
            * np.sqrt(2.0 / input_size)
        ).astype(np.float32)

        self.b1 = np.zeros(
            hidden_size,
            dtype=np.float32
        )

        self.W2 = (
            rng.standard_normal(
                (hidden_size, output_size)
            )
            * np.sqrt(2.0 / hidden_size)
        ).astype(np.float32)

        self.b2 = np.zeros(
            output_size,
            dtype=np.float32
        )

        self.parameters = {
            "W1": self.W1,
            "b1": self.b1,
            "W2": self.W2,
            "b2": self.b2,
        }

        self.gradients = {}
        self.cache = {}

    def forward(self, X):

        X = np.asarray(
            X,
            dtype=np.float32
        )

        Z1 = X @ self.W1 + self.b1

        A1 = np.maximum(0.0, Z1)

        Q = A1 @ self.W2 + self.b2

        self.cache = {
            "X": X,
            "Z1": Z1,
            "A1": A1,
            "Q": Q,
        }

        return Q

    @staticmethod
    def loss_function(predictions, targets):

        return float(
            np.mean(
                (predictions - targets) ** 2
            )
        )

    def backward(self, targets):

        X = self.cache["X"]
        Z1 = self.cache["Z1"]
        A1 = self.cache["A1"]
        Q = self.cache["Q"]

        targets = np.asarray(
            targets,
            dtype=np.float32
        )

        dQ = (
            2.0 * (Q - targets)
            / Q.size
        )

        dW2 = A1.T @ dQ
        db2 = np.sum(dQ, axis=0)

        dA1 = dQ @ self.W2.T

        dZ1 = dA1 * (Z1 > 0.0)

        dW1 = X.T @ dZ1
        db1 = np.sum(dZ1, axis=0)

        self.gradients = {
            "W1": dW1,
            "b1": db1,
            "W2": dW2,
            "b2": db2,
        }

        return self.gradients


# =====================================================================
# 8. ADAM OPTIMIZER
# =====================================================================

class Adam:

    def __init__(
        self,
        learning_rate=0.001,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8
    ):

        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.m = {}
        self.v = {}
        self.t = 0

    def step(self, network):

        self.t += 1

        for name, parameter in network.parameters.items():

            gradient = network.gradients[name]

            if name not in self.m:

                self.m[name] = np.zeros_like(parameter)
                self.v[name] = np.zeros_like(parameter)

            self.m[name] = (
                self.beta1 * self.m[name]
                + (1 - self.beta1) * gradient
            )

            self.v[name] = (
                self.beta2 * self.v[name]
                + (1 - self.beta2) * gradient ** 2
            )

            m_hat = (
                self.m[name]
                / (1 - self.beta1 ** self.t)
            )

            v_hat = (
                self.v[name]
                / (1 - self.beta2 ** self.t)
            )

            parameter -= (
                self.lr * m_hat
                / (np.sqrt(v_hat) + self.epsilon)
            )


# =====================================================================
# 9. TRAINING
# =====================================================================

def train_network(
    network,
    X,
    Y,
    epochs=150,
    batch_size=128,
    learning_rate=0.001,
    seed=SEED
):

    rng = np.random.default_rng(seed)

    optimizer = Adam(learning_rate)

    initial_loss = network.loss_function(
        network.forward(X),
        Y
    )

    losses = []

    for epoch in range(epochs):

        indices = rng.permutation(len(X))

        for start in range(
            0,
            len(X),
            batch_size
        ):

            batch = indices[
                start:start + batch_size
            ]

            network.forward(X[batch])
            network.backward(Y[batch])

            optimizer.step(network)

        predictions = network.forward(X)

        loss = network.loss_function(
            predictions,
            Y
        )

        losses.append(loss)

        if (epoch + 1) % 25 == 0:

            print(
                f"Epoch {epoch + 1:3d}/{epochs}"
                f" | Loss: {loss:.6f}"
            )

    final_loss = network.loss_function(
        network.forward(X),
        Y
    )

    return {
        "network": network,
        "losses": losses,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
    }


# =====================================================================
# 10. SAFETY FILTER AND ANTI-VIBRATION CONTROLLER
# =====================================================================

def safe_action_mask(position):

    mask = np.zeros(4, dtype=bool)

    for action in range(4):

        candidate = predict_position(
            position,
            action
        )

        mask[action] = is_position_safe(
            candidate
        )

    return mask


def geometric_progress(position, target):

    current_distance = distance(
        position,
        target
    )

    scores = np.zeros(4, dtype=np.float64)

    for action in range(4):

        candidate = predict_position(
            position,
            action
        )

        next_distance = distance(
            candidate,
            target
        )

        scores[action] = (
            current_distance - next_distance
        ) / STEP_SIZE

    return scores


def choose_stable_action(
    q_values,
    position,
    target,
    previous_action=None
):
    """
    Combine neural-network values and geometric progress.

    Anti-vibration mechanisms:

    1. Geometric progress:
       Favor actions that reduce waypoint distance.

    2. Momentum:
       Slightly favor the previous action.

    3. Reverse penalty:
       Penalize immediate 180-degree reversals.

    4. Switching penalty:
       Avoid unnecessary direction changes.

    5. Hysteresis:
       Keep the previous safe action when its score is close
       to the best candidate.

    Safety always takes priority over momentum and hysteresis.
    """

    mask = safe_action_mask(position)

    if not np.any(mask):
        return None, mask, np.zeros(4)

    waypoint = navigation_target(
        position,
        target
    )

    progress = geometric_progress(
        position,
        waypoint
    )

    q_values = np.asarray(
        q_values,
        dtype=np.float64
    )

    # Normalize Q-values to reduce sensitivity to their scale.
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

    # Hysteresis: retain the previous safe direction if its
    # score is close enough to the new best score.
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
# 11. ROBOT ENVIRONMENT
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

        target = np.asarray(
            target,
            dtype=np.float64
        )

        if not target_is_valid(target):
            return False

        self.target = target.copy()

        self.success = False
        self.steps = 0

        # Keep momentum for a smoother response to a new target.
        # Reset it if the new target requires an opposite direction.
        if self.previous_action is not None:

            desired = self.target - self.position

            desired_direction = int(
                np.argmax(
                    ACTION_VECTORS @ desired
                )
            )

            if (
                desired_direction
                == (self.previous_action + 2) % 4
            ):

                self.previous_action = None

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
# 12. UNIT TESTS
# =====================================================================

def run_tests():

    print("\n========== UNIT TESTS ==========")

    class TestPhase41(unittest.TestCase):

        def test_state_dimension(self):

            state = encode_state(
                START,
                DEFAULT_TARGET
            )

            self.assertEqual(state.shape, (10,))

        def test_target_changes_state(self):

            a = encode_state(
                START,
                np.array([620.0, 290.0])
            )

            b = encode_state(
                START,
                np.array([620.0, 100.0])
            )

            self.assertFalse(
                np.allclose(a, b)
            )

        def test_network_dimensions(self):

            network = NeuralNetwork()

            output = network.forward(
                np.zeros((5, 10), dtype=np.float32)
            )

            self.assertEqual(output.shape, (5, 4))

        def test_dataset_dimensions(self):

            X, Y = generate_dataset(100)

            self.assertEqual(X.shape, (100, 10))
            self.assertEqual(Y.shape, (100, 4))

        def test_safety(self):

            self.assertTrue(
                is_position_safe(START)
            )

            self.assertFalse(
                is_position_safe(OBSTACLE)
            )

        def test_target_validation(self):

            self.assertTrue(
                target_is_valid(DEFAULT_TARGET)
            )

            self.assertFalse(
                target_is_valid(OBSTACLE)
            )

        def test_waypoint(self):

            waypoint = calculate_waypoint(
                START,
                DEFAULT_TARGET
            )

            self.assertIsNotNone(waypoint)

            self.assertGreater(
                distance(waypoint, OBSTACLE),
                SAFE_DISTANCE
            )

        def test_clear_path(self):

            position = np.array([80.0, 100.0])
            target = np.array([250.0, 100.0])

            self.assertIsNone(
                calculate_waypoint(position, target)
            )

        def test_safety_filter(self):

            position = np.array([
                OBSTACLE[0] - SAFE_DISTANCE + 2,
                OBSTACLE[1]
            ])

            q_values = np.array([
                0.0,
                100.0,
                0.0,
                0.0
            ])

            action, mask, scores = choose_stable_action(
                q_values,
                position,
                DEFAULT_TARGET
            )

            self.assertNotEqual(action, 1)
            self.assertTrue(mask[action])

        def test_hysteresis_keeps_previous_action(self):

            position = np.array([200.0, 100.0])
            target = np.array([300.0, 100.0])

            q_values = np.array([
                0.0,
                1.0,
                0.0,
                0.0
            ])

            action, mask, scores = choose_stable_action(
                q_values,
                position,
                target,
                previous_action=1
            )

            self.assertEqual(action, 1)

        def test_controller_rejects_reverse_when_unnecessary(self):

            position = np.array([200.0, 100.0])
            target = np.array([300.0, 100.0])

            q_values = np.array([
                0.0,
                1.0,
                0.0,
                0.0
            ])

            action, mask, scores = choose_stable_action(
                q_values,
                position,
                target,
                previous_action=3
            )

            self.assertEqual(action, 1)

        def test_environment_target_update(self):

            env = RobotEnvironment()

            target = np.array([600.0, 150.0])

            self.assertTrue(
                env.set_target(target)
            )

            np.testing.assert_allclose(
                env.target,
                target
            )

        def test_environment_rejects_invalid_target(self):

            env = RobotEnvironment()

            old_target = env.target.copy()

            self.assertFalse(
                env.set_target(OBSTACLE)
            )

            np.testing.assert_allclose(
                env.target,
                old_target
            )

        def test_training_reduces_loss(self):

            X, Y = generate_dataset(
                1000,
                seed=123
            )

            network = NeuralNetwork(seed=123)

            result = train_network(
                network,
                X,
                Y,
                epochs=40,
                batch_size=64,
                seed=123
            )

            self.assertLess(
                result["final_loss"],
                result["initial_loss"]
            )

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        TestPhase41
    )

    result = unittest.TextTestRunner(
        verbosity=2
    ).run(suite)

    if not result.wasSuccessful():

        raise RuntimeError(
            "Phase 4.1 tests failed."
        )

    print("\nAll tests passed.")


# =====================================================================
# 13. PYGAME GUI
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
        "Phase 4.1 — Rocket Navigation"
    )

    clock = pygame.time.Clock()

    font = pygame.font.SysFont("consolas", 16)
    small = pygame.font.SysFont("consolas", 13)
    title_font = pygame.font.SysFont(
        "consolas",
        20,
        bold=True
    )

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

    env = RobotEnvironment()

    network = training_result["network"]
    losses = training_result["losses"]

    paused = False
    running = True

    speed = 1.0
    interval = 0.055

    last_step_time = 0.0

    selected_action = None

    last_q_values = np.zeros(4)
    last_mask = safe_action_mask(env.position)

    last_scores = np.zeros(4)

    message = "LEFT CLICK TO SET A NEW TARGET"
    message_color = YELLOW
    message_until = 0.0

    exhaust_phase = 0.0

    def screen_point(position):

        return (
            int(WORLD_X + position[0]),
            int(WORLD_Y + position[1])
        )

    def world_point(mouse_position):

        return np.array([
            mouse_position[0] - WORLD_X,
            mouse_position[1] - WORLD_Y
        ], dtype=np.float64)

    def draw_text(
        value,
        x,
        y,
        color=WHITE,
        selected_font=font
    ):

        surface = selected_font.render(
            str(value),
            True,
            color
        )

        screen.blit(surface, (x, y))

    # -----------------------------------------------------------------
    # ROCKET SHAPE
    # -----------------------------------------------------------------

    def rotate_local_point(local_point, angle):

        x, y = local_point

        c = np.cos(angle)
        s = np.sin(angle)

        return np.array([
            x * c - y * s,
            x * s + y * c
        ])

    def draw_rocket(position, action, phase):

        """
        Draw a rocket with its nose facing the current direction.

        Local coordinate convention:
            +X = rocket nose
            -X = rocket tail
            +Y = rocket bottom side

        The entire shape rotates according to the movement direction.
        """

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
                    center + rotate_local_point(point, angle)
                ).astype(int)
            )

        # Animated exhaust.
        flame_length = 13 + 4 * np.sin(phase * 18)

        flame = [
            (-7, -4),
            (-7 - flame_length, 0),
            (-7, 4)
        ]

        pygame.draw.polygon(
            screen,
            ORANGE,
            [transform(p) for p in flame]
        )

        inner_flame = [
            (-7, -2),
            (-7 - flame_length * 0.65, 0),
            (-7, 2)
        ]

        pygame.draw.polygon(
            screen,
            YELLOW,
            [transform(p) for p in inner_flame]
        )

        # Main rocket body.
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

        # Nose cone.
        nose = [
            (17, 0),
            (7, -6),
            (7, 6)
        ]

        pygame.draw.polygon(
            screen,
            RED,
            [transform(p) for p in nose]
        )

        # Upper and lower fins.
        upper_fin = [
            (-7, -5),
            (-15, -12),
            (-11, -4)
        ]

        lower_fin = [
            (-7, 5),
            (-15, 12),
            (-11, 4)
        ]

        pygame.draw.polygon(
            screen,
            PURPLE,
            [transform(p) for p in upper_fin]
        )

        pygame.draw.polygon(
            screen,
            PURPLE,
            [transform(p) for p in lower_fin]
        )

        # Cockpit.
        pygame.draw.circle(
            screen,
            (170, 225, 255),
            transform((5, 0)),
            3
        )

    # -----------------------------------------------------------------
    # WORLD DRAWING
    # -----------------------------------------------------------------

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

        # Obstacle safety boundary.
        pygame.draw.circle(
            screen,
            ORANGE,
            screen_point(OBSTACLE),
            int(SAFE_DISTANCE),
            1
        )

        # Physical obstacle.
        pygame.draw.circle(
            screen,
            RED,
            screen_point(OBSTACLE),
            OBSTACLE_RADIUS
        )

        pygame.draw.circle(
            screen,
            (255, 150, 150),
            screen_point(OBSTACLE),
            OBSTACLE_RADIUS,
            2
        )

        # Target.
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

        draw_text(
            "TARGET",
            env.target[0] + WORLD_X - 22,
            env.target[1] + WORLD_Y - 34,
            GREEN,
            small
        )

        # Waypoint.
        waypoint = calculate_waypoint(
            env.position,
            env.target
        )

        if waypoint is not None:

            if distance(
                env.position,
                waypoint
            ) > 13:

                pygame.draw.circle(
                    screen,
                    YELLOW,
                    screen_point(waypoint),
                    9,
                    2
                )

                draw_text(
                    "WAYPOINT",
                    waypoint[0] + WORLD_X - 35,
                    waypoint[1] + WORLD_Y - 25,
                    YELLOW,
                    small
                )

        # Starting position.
        pygame.draw.circle(
            screen,
            PURPLE,
            screen_point(START),
            4
        )

        # Candidate directions.
        for action in range(4):

            candidate = predict_position(
                env.position,
                action
            )

            start_point = screen_point(
                env.position
            )

            end_point = screen_point(
                candidate
            )

            if not last_mask[action]:
                color = RED

            elif action == selected_action:
                color = YELLOW

            else:
                color = GREEN

            pygame.draw.line(
                screen,
                color,
                start_point,
                end_point,
                3
            )

        # Rocket.
        draw_rocket(
            env.position,
            env.previous_action,
            exhaust_phase
        )

        draw_text(
            "ROCKET NAVIGATION WORLD",
            WORLD_X + 10,
            WORLD_Y - 28,
            WHITE,
            title_font
        )

        draw_text(
            "Green = safe | Red = blocked | Yellow = selected",
            WORLD_X + 10,
            WORLD_Y + WORLD_HEIGHT + 8,
            MUTED,
            small
        )

    # -----------------------------------------------------------------
    # NETWORK DISPLAY
    # -----------------------------------------------------------------

    def draw_network():

        draw_text(
            "NEURAL NETWORK",
            770,
            130,
            WHITE,
            title_font
        )

        draw_text(
            "10 inputs -> 32 hidden -> 4 outputs",
            770,
            160,
            MUTED,
            small
        )

        xs = [790, 935, 1090]

        input_ys = np.linspace(205, 380, 10)
        hidden_ys = np.linspace(200, 395, 12)
        output_ys = np.linspace(250, 360, 4)

        input_points = [
            (xs[0], int(y))
            for y in input_ys
        ]

        hidden_points = [
            (xs[1], int(y))
            for y in hidden_ys
        ]

        output_points = [
            (xs[2], int(y))
            for y in output_ys
        ]

        w1 = network.W1
        w2 = network.W2

        max_w1 = max(
            float(np.max(np.abs(w1))),
            1e-8
        )

        max_w2 = max(
            float(np.max(np.abs(w2))),
            1e-8
        )

        for i, p1 in enumerate(input_points):

            for j, p2 in enumerate(hidden_points):

                weight = w1[i, j]

                intensity = int(
                    55 + 150 * abs(weight) / max_w1
                )

                base = BLUE if weight >= 0 else RED

                color = tuple(
                    min(255, int(c * intensity / 255))
                    for c in base
                )

                pygame.draw.line(
                    screen,
                    color,
                    p1,
                    p2,
                    1
                )

        for i, p1 in enumerate(hidden_points):

            for j, p2 in enumerate(output_points):

                weight = w2[i, j]

                intensity = int(
                    55 + 150 * abs(weight) / max_w2
                )

                base = GREEN if weight >= 0 else RED

                color = tuple(
                    min(255, int(c * intensity / 255))
                    for c in base
                )

                pygame.draw.line(
                    screen,
                    color,
                    p1,
                    p2,
                    1
                )

        for point in input_points:

            pygame.draw.circle(
                screen,
                PURPLE,
                point,
                4
            )

        for point in hidden_points:

            pygame.draw.circle(
                screen,
                ORANGE,
                point,
                5
            )

        for point in output_points:

            pygame.draw.circle(
                screen,
                GREEN,
                point,
                8
            )

        draw_text("INPUT", 775, 410, PURPLE, small)
        draw_text("HIDDEN", 920, 410, ORANGE, small)
        draw_text("OUTPUT", 1060, 410, GREEN, small)

    # -----------------------------------------------------------------
    # ACTION VALUES
    # -----------------------------------------------------------------

    def draw_action_values():

        draw_text(
            "ACTION SCORES / SAFETY",
            770,
            445,
            WHITE,
            title_font
        )

        for action, name in enumerate(ACTIONS):

            y = 480 + action * 34

            if action == selected_action:

                status = "SELECTED"
                color = YELLOW

            elif last_mask[action]:

                status = "SAFE"
                color = GREEN

            else:

                status = "BLOCKED"
                color = RED

            draw_text(
                name,
                770,
                y,
                color,
                small
            )

            draw_text(
                f"Q={last_q_values[action]:+.3f}",
                850,
                y,
                WHITE,
                small
            )

            draw_text(
                f"S={last_scores[action]:+.3f}",
                950,
                y,
                MUTED,
                small
            )

            draw_text(
                status,
                1060,
                y,
                color,
                small
            )

    # -----------------------------------------------------------------
    # TRAINING LOSS
    # -----------------------------------------------------------------

    def draw_loss():

        draw_text(
            "TRAINING LOSS",
            770,
            630,
            WHITE,
            title_font
        )

        chart = pygame.Rect(
            770,
            660,
            350,
            55
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
            720,
            MUTED,
            small
        )

    # -----------------------------------------------------------------
    # MAIN EVENT LOOP
    # -----------------------------------------------------------------

    while running:

        clock.tick(60)

        now = time.perf_counter()

        exhaust_phase += 1.0 / 60.0

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

                    last_q_values = np.zeros(4)
                    last_mask = safe_action_mask(env.position)
                    last_scores = np.zeros(4)

                    message = "ROCKET RESET"
                    message_color = YELLOW
                    message_until = now + 1.5

                elif event.key in (
                    pygame.K_PLUS,
                    pygame.K_EQUALS,
                    pygame.K_KP_PLUS
                ):

                    speed = min(
                        speed * 1.5,
                        8.0
                    )

                elif event.key in (
                    pygame.K_MINUS,
                    pygame.K_KP_MINUS
                ):

                    speed = max(
                        speed / 1.5,
                        0.2
                    )

            elif (
                event.type == pygame.MOUSEBUTTONDOWN
                and event.button == 1
            ):

                if WORLD_RECT.collidepoint(event.pos):

                    new_target = world_point(event.pos)

                    if env.set_target(new_target):

                        selected_action = None

                        last_q_values = np.zeros(4)

                        last_mask = safe_action_mask(
                            env.position
                        )

                        message = (
                            f"NEW TARGET: "
                            f"({new_target[0]:.0f}, "
                            f"{new_target[1]:.0f})"
                        )

                        message_color = GREEN
                        message_until = now + 2.0

                    else:

                        message = (
                            "INVALID TARGET: "
                            "OBSTACLE SAFETY ZONE OR OUTSIDE WORLD"
                        )

                        message_color = RED
                        message_until = now + 2.0

        # -------------------------------------------------------------
        # Neural-network inference and stable action selection.
        # -------------------------------------------------------------

        if (
            not paused
            and not env.success
            and now - last_step_time >= interval / speed
        ):

            state = encode_state(
                env.position,
                env.target
            )

            q_values = network.forward(
                state.reshape(1, -1)
            )[0]

            action, mask, scores = choose_stable_action(
                q_values,
                env.position,
                env.target,
                env.previous_action
            )

            selected_action = action

            last_q_values = q_values.copy()
            last_mask = mask.copy()
            last_scores = scores.copy()

            env.step(action)

            last_step_time = now

            if env.success:

                message = "ROCKET REACHED TARGET!"
                message_color = GREEN
                message_until = now + 2.0

        # -------------------------------------------------------------
        # Render.
        # -------------------------------------------------------------

        screen.fill(BACKGROUND)

        draw_world()
        draw_network()
        draw_action_values()
        draw_loss()

        draw_text(
            f"Steps: {env.steps}",
            20,
            24,
            WHITE,
            title_font
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
                title_font
            )

        elif env.success:

            draw_text(
                "TARGET REACHED - CLICK TO SELECT ANOTHER",
                140,
                65,
                GREEN,
                title_font
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
            "LEFT CLICK: NEW TARGET | SPACE: PAUSE | "
            "R: RESET | +/-: SPEED | ESC: EXIT",
            20,
            755,
            MUTED,
            small
        )

        pygame.display.flip()

    pygame.quit()


# =====================================================================
# 14. MAIN
# =====================================================================

def main():

    np.random.seed(SEED)

    print("=" * 65)
    print("RL COURSE — PHASE 4.1")
    print("ROCKET NAVIGATION WITH ANTI-VIBRATION CONTROL")
    print("=" * 65)

    run_tests()

    print("\nGenerating training dataset...")

    X, Y = generate_dataset(
        num_samples=12000,
        seed=SEED
    )

    print("X:", X.shape)
    print("Y:", Y.shape)

    network = NeuralNetwork(
        input_size=10,
        hidden_size=32,
        output_size=4
    )

    print("\nTraining neural network...")

    result = train_network(
        network,
        X,
        Y,
        epochs=150,
        batch_size=128,
        learning_rate=0.001
    )

    print("\nTraining complete.")

    print(
        "Initial loss:",
        result["initial_loss"]
    )

    print(
        "Final loss:",
        result["final_loss"]
    )

    print("\nStarting Pygame rocket simulation...")

    run_pygame(result)


if __name__ == "__main__":
    main()
