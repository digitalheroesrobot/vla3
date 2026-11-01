
"""
====================================================================
RL_Course
Phase 0.6 — Mathematical Foundations Review
File: phase0/foundations_review.py
====================================================================

PURPOSE OF THIS LESSON
----------------------
Consolidate the mathematical foundations required for reinforcement
learning and connect them through a practical stochastic-policy example.

LEARNING ROADMAP
----------------

STEP 1 — Linear Algebra Review
Purpose:
    Understand how vectors and matrices represent states, actions,
    neural-network parameters, and value functions in RL.

What you learn:
    - Vector and matrix operations.
    - Dot products and matrix multiplication.
    - Linear transformations.
    - How a linear Q-function maps states to action values.

Implementation:
    Construct a small linear Q-function and calculate its outputs.

STEP 2 — Calculus and Gradient Review
Purpose:
    Understand how an RL agent improves its parameters through
    gradient-based optimization.

What you learn:
    - Derivatives and gradients.
    - Gradient descent.
    - Numerical gradient approximation.
    - Analytical versus numerical gradient verification.

Implementation:
    Minimize a quadratic loss and compare two gradient calculations.

STEP 3 — Probability and Stochastic Policies
Purpose:
    Understand how an RL agent selects actions probabilistically.

What you learn:
    - Probability distributions.
    - Expected value and variance.
    - Softmax action probabilities.
    - Categorical sampling.
    - Policy entropy.

Implementation:
    Build a two-action stochastic policy and calculate its statistics.

STEP 4 — Numerical Stability Review
Purpose:
    Prevent numerical errors when training RL algorithms.

What you learn:
    - Why ordinary Softmax can overflow.
    - Numerically stable Softmax.
    - Log-Sum-Exp.
    - Why these techniques matter in policy-gradient algorithms.

Implementation:
    Compare ordinary and stable Softmax calculations.

STEP 5 — Neural Network Gradient Verification
Purpose:
    Understand the mathematical foundation of neural-network learning.

What you learn:
    - Forward propagation.
    - Backpropagation.
    - Chain rule.
    - Parameter gradients.
    - Gradient checking.

Implementation:
    Build a small neural network manually and verify its gradients.

STEP 6 — Mini RL Policy Experiment
Purpose:
    Connect linear algebra, calculus, probability, and optimization
    in one reinforcement learning example.

What you learn:
    - Policy parameters (logits).
    - Action probabilities.
    - Expected reward.
    - Policy entropy.
    - The gradient of expected reward.
    - How gradient ascent improves a policy.

Implementation:
    Train a two-action stochastic policy to maximize expected reward.

STEP 7 — Visualization and Verification
Purpose:
    Make mathematical behavior observable and verify implementation.

What you learn:
    - How gradients change optimization trajectories.
    - How policy probabilities change during learning.
    - How entropy changes as a policy becomes more deterministic.

Implementation:
    Run numerical tests and display all lesson visualizations.

LEARNING CYCLE
--------------
Understand (comments)
    -> Implement
    -> Visualize
    -> Verify
    -> Experiment
    -> Apply

REQUIREMENTS
------------
pip install numpy matplotlib

RUN
---
python phase0/foundations_review.py
====================================================================
"""

import numpy as np
import matplotlib.pyplot as plt


# ================================================================
# STEP 1 — LINEAR ALGEBRA REVIEW
# ================================================================

def linear_q_function(state, weights, bias):
    """
    Linear Q-function:

        Q(s) = W s + b

    state:  (number_of_state_features,)
    weights: (number_of_actions, number_of_state_features)
    bias:   (number_of_actions,)

    Output:
        Q-values for all available actions.
    """
    state = np.asarray(state, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    bias = np.asarray(bias, dtype=np.float64)

    return weights @ state + bias


def linear_algebra_demo():
    state = np.array([0.5, 1.0, -0.5])

    weights = np.array([
        [0.2, 0.4, 0.1],
        [0.5, -0.2, 0.3]
    ])

    bias = np.array([0.1, -0.1])

    q_values = linear_q_function(state, weights, bias)

    return state, weights, bias, q_values


# ================================================================
# STEP 2 — CALCULUS AND GRADIENT REVIEW
# ================================================================

def quadratic_loss(x):
    """
    Example optimization objective:

        L(x, y) = (x - 3)^2 + 2(y + 1)^2

    Its minimum is at (3, -1).
    """
    x, y = x
    return (x - 3.0) ** 2 + 2.0 * (y + 1.0) ** 2


def analytical_gradient(x):
    """
    Analytical gradient:

        dL/dx = 2(x - 3)
        dL/dy = 4(y + 1)
    """
    x1, x2 = x

    return np.array([
        2.0 * (x1 - 3.0),
        4.0 * (x2 + 1.0)
    ])


def numerical_gradient(function, x, epsilon=1e-6):
    """
    Central finite difference:

        df/dx_i ~= [f(x + eps) - f(x - eps)] / (2 eps)

    This provides an independent numerical approximation
    of the analytical gradient.
    """
    x = np.asarray(x, dtype=np.float64)
    gradient = np.zeros_like(x)

    for i in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()

        x_plus[i] += epsilon
        x_minus[i] -= epsilon

        gradient[i] = (
            function(x_plus) - function(x_minus)
        ) / (2.0 * epsilon)

    return gradient


def gradient_descent(
    function,
    gradient_function,
    initial_position,
    learning_rate=0.1,
    steps=50
):
    """
    Gradient descent:

        theta_new = theta_old - learning_rate * gradient

    Used to minimize a loss function.
    """
    position = np.asarray(
        initial_position,
        dtype=np.float64
    )

    trajectory = [position.copy()]
    losses = [function(position)]

    for _ in range(steps):
        gradient = gradient_function(position)

        position -= learning_rate * gradient

        trajectory.append(position.copy())
        losses.append(function(position))

    return (
        np.asarray(trajectory),
        np.asarray(losses)
    )


# ================================================================
# STEP 3 — PROBABILITY AND STOCHASTIC POLICIES
# ================================================================

def stable_softmax(logits, temperature=1.0):
    """
    Softmax policy:

        pi(a|s) = exp(logit_a / T)
                  -------------------
                  sum_b exp(logit_b / T)

    T controls action-selection randomness.

    T > 1:
        More distributed action probabilities.

    T < 1:
        More concentrated action probabilities.

    Subtracting the maximum logit prevents numerical overflow.
    """
    if temperature <= 0:
        raise ValueError("Temperature must be positive.")

    logits = np.asarray(logits, dtype=np.float64)

    scaled = logits / temperature
    shifted = scaled - np.max(scaled)

    exp_values = np.exp(shifted)

    return exp_values / np.sum(exp_values)


def categorical_sample(probabilities, rng=None):
    """
    Sample an action according to a categorical distribution.
    """
    if rng is None:
        rng = np.random.default_rng()

    probabilities = np.asarray(
        probabilities,
        dtype=np.float64
    )

    if np.any(probabilities < 0):
        raise ValueError("Probabilities cannot be negative.")

    if not np.isclose(np.sum(probabilities), 1.0):
        raise ValueError("Probabilities must sum to 1.")

    return int(
        rng.choice(
            len(probabilities),
            p=probabilities
        )
    )


def expected_value(values, probabilities):
    """
    Expected value:

        E[X] = sum_i p_i * x_i
    """
    return float(
        np.sum(
            np.asarray(values) *
            np.asarray(probabilities)
        )
    )


def variance(values, probabilities):
    """
    Variance:

        Var(X) = E[(X - E[X])^2]
    """
    mean = expected_value(values, probabilities)

    return float(
        np.sum(
            np.asarray(probabilities) *
            (np.asarray(values) - mean) ** 2
        )
    )


def categorical_entropy(probabilities):
    """
    Entropy:

        H(pi) = -sum_a pi(a) log(pi(a))

    High entropy:
        The policy distributes probability across actions.

    Low entropy:
        The policy strongly prefers particular actions.

    Zero-probability terms contribute zero.
    """
    probabilities = np.asarray(
        probabilities,
        dtype=np.float64
    )

    positive = probabilities > 0

    return float(
        -np.sum(
            probabilities[positive] *
            np.log(probabilities[positive])
        )
    )


# ================================================================
# STEP 4 — NUMERICAL STABILITY REVIEW
# ================================================================

def ordinary_softmax(logits):
    """
    Demonstration only.

    Direct exponentiation can overflow for large logits.
    """
    exp_values = np.exp(
        np.asarray(logits, dtype=np.float64)
    )

    return exp_values / np.sum(exp_values)


def log_sum_exp(values):
    """
    Stable Log-Sum-Exp:

        log(sum(exp(x_i)))
        =
        m + log(sum(exp(x_i - m)))

    where m = max(x).

    This technique is widely used in probability calculations
    and machine-learning loss functions.
    """
    values = np.asarray(values, dtype=np.float64)

    maximum = np.max(values)

    return float(
        maximum +
        np.log(np.sum(np.exp(values - maximum)))
    )


# ================================================================
# STEP 5 — NEURAL NETWORK GRADIENT VERIFICATION
# ================================================================

def neural_network_forward(x, parameters):
    """
    A two-layer neural network:

        z1 = W1 x + b1
        h  = tanh(z1)
        y  = W2 h + b2

    This small network is sufficient to demonstrate
    forward propagation and backpropagation.
    """
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    z1 = W1 @ x + b1
    h = np.tanh(z1)
    output = W2 @ h + b2

    cache = {
        "x": x,
        "z1": z1,
        "h": h
    }

    return output, cache


def neural_network_loss(parameters, x, target):
    output, _ = neural_network_forward(x, parameters)

    error = output - target

    return 0.5 * float(np.sum(error ** 2))


def neural_network_backward(parameters, cache, target, output):
    """
    Backpropagation using the chain rule.

    For:

        L = 0.5 * sum((output - target)^2)

    The derivative is propagated backward through
    the output layer and tanh hidden layer.
    """
    W2 = parameters["W2"]
    x = cache["x"]
    h = cache["h"]

    error = output - target

    dW2 = np.outer(error, h)
    db2 = error.copy()

    dh = W2.T @ error

    # Derivative of tanh(z):
    # d(tanh(z))/dz = 1 - tanh(z)^2
    dz1 = dh * (1.0 - h ** 2)

    dW1 = np.outer(dz1, x)
    db1 = dz1.copy()

    return {
        "W1": dW1,
        "b1": db1,
        "W2": dW2,
        "b2": db2
    }


def numerical_parameter_gradient(
    parameters,
    x,
    target,
    epsilon=1e-6
):
    """
    Numerical gradient for every network parameter.

    This is used to verify manual backpropagation.
    """
    gradients = {}

    for name, parameter in parameters.items():
        gradients[name] = np.zeros_like(parameter)

        for index in np.ndindex(parameter.shape):
            original = parameter[index]

            parameter[index] = original + epsilon
            loss_plus = neural_network_loss(
                parameters, x, target
            )

            parameter[index] = original - epsilon
            loss_minus = neural_network_loss(
                parameters, x, target
            )

            parameter[index] = original

            gradients[name][index] = (
                loss_plus - loss_minus
            ) / (2.0 * epsilon)

    return gradients


def initialize_network(seed=42):
    rng = np.random.default_rng(seed)

    return {
        "W1": rng.normal(0.0, 0.3, size=(4, 3)),
        "b1": np.zeros(4),
        "W2": rng.normal(0.0, 0.3, size=(2, 4)),
        "b2": np.zeros(2)
    }


# ================================================================
# STEP 6 — MINI RL POLICY EXPERIMENT
# ================================================================

def policy_statistics(logits, rewards, temperature=1.0):
    """
    A one-state, two-action RL problem.

    Action 0 gives reward 1.
    Action 1 gives reward 3.

    The policy is parameterized by two logits.

    Expected reward:

        J(theta) = sum_a pi(a|theta) * reward(a)

    Policy-gradient intuition:

        dJ/dtheta_i
        =
        pi_i * (reward_i - J)

    This expression shows how each action's reward relative
    to the current expected reward affects its logit.

    The agent should increase the probability of actions
    whose rewards are above the current expected reward.
    """
    probabilities = stable_softmax(
        logits,
        temperature
    )

    expected_reward = expected_value(
        rewards,
        probabilities
    )

    entropy = categorical_entropy(probabilities)

    gradient = (
        probabilities *
        (np.asarray(rewards) - expected_reward)
    ) / temperature

    return {
        "probabilities": probabilities,
        "expected_reward": expected_reward,
        "entropy": entropy,
        "gradient": gradient
    }


def train_stochastic_policy(
    rewards=(1.0, 3.0),
    initial_logits=(0.0, 0.0),
    learning_rate=0.1,
    episodes=100
):
    """
    Policy-gradient-style optimization.

    We maximize expected reward using gradient ascent:

        theta_new = theta_old + alpha * grad J(theta)

    This example uses the exact expected reward gradient,
    rather than a sampled REINFORCE estimate.

    It isolates the policy-gradient mathematics from
    sampling noise and environment complexity.
    """
    rewards = np.asarray(rewards, dtype=np.float64)
    logits = np.asarray(
        initial_logits,
        dtype=np.float64
    )

    history = {
        "logits": [],
        "probabilities": [],
        "expected_reward": [],
        "entropy": []
    }

    for _ in range(episodes):
        statistics = policy_statistics(
            logits,
            rewards
        )

        history["logits"].append(logits.copy())
        history["probabilities"].append(
            statistics["probabilities"].copy()
        )
        history["expected_reward"].append(
            statistics["expected_reward"]
        )
        history["entropy"].append(
            statistics["entropy"]
        )

        # Gradient ascent maximizes expected reward.
        logits += learning_rate * statistics["gradient"]

    return logits, {
        key: np.asarray(value)
        for key, value in history.items()
    }


# ================================================================
# STEP 7 — VISUALIZATION FUNCTIONS
# ================================================================

def visualize_q_values():
    state, weights, bias, q_values = linear_algebra_demo()

    plt.figure(figsize=(7, 4))

    plt.bar(
        ["Action 0", "Action 1"],
        q_values
    )

    plt.title("Linear Q-Function Output")
    plt.ylabel("Q-value")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()


def visualize_gradient_descent():
    trajectory, losses = gradient_descent(
        quadratic_loss,
        analytical_gradient,
        initial_position=(-2.0, 3.0),
        learning_rate=0.1,
        steps=40
    )

    x_values = np.linspace(-3, 5, 150)
    y_values = np.linspace(-3, 5, 150)

    X, Y = np.meshgrid(x_values, y_values)

    Z = (
        (X - 3.0) ** 2 +
        2.0 * (Y + 1.0) ** 2
    )

    plt.figure(figsize=(7, 5))

    plt.contour(
        X,
        Y,
        Z,
        levels=25
    )

    plt.plot(
        trajectory[:, 0],
        trajectory[:, 1],
        "o-",
        markersize=3,
        label="Optimization path"
    )

    plt.scatter(
        [3.0],
        [-1.0],
        marker="*",
        s=180,
        label="Minimum"
    )

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Gradient Descent Trajectory")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()


def visualize_softmax_temperature():
    logits = np.array([1.0, 2.0, 3.0])

    temperatures = [0.3, 0.7, 1.0, 2.0, 5.0]

    distributions = [
        stable_softmax(logits, temperature=t)
        for t in temperatures
    ]

    x = np.arange(len(logits))
    width = 0.14

    plt.figure(figsize=(9, 5))

    for i, (temperature, probabilities) in enumerate(
        zip(temperatures, distributions)
    ):
        plt.bar(
            x + i * width,
            probabilities,
            width=width,
            label=f"T={temperature}"
        )

    plt.xticks(
        x + width * 2,
        ["Action 0", "Action 1", "Action 2"]
    )

    plt.ylabel("Probability")
    plt.title("Softmax Temperature and Action Distribution")
    plt.legend()
    plt.tight_layout()


def visualize_policy_training(history):
    episodes = np.arange(
        len(history["expected_reward"])
    )

    probabilities = history["probabilities"]

    fig, axes = plt.subplots(3, 1, figsize=(8, 10))

    axes[0].plot(
        episodes,
        probabilities[:, 0],
        label="Action 0"
    )

    axes[0].plot(
        episodes,
        probabilities[:, 1],
        label="Action 1"
    )

    axes[0].set_title("Policy Action Probabilities")
    axes[0].set_ylabel("Probability")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(
        episodes,
        history["expected_reward"]
    )

    axes[1].set_title("Expected Reward")
    axes[1].set_ylabel("Expected reward")
    axes[1].grid(alpha=0.3)

    axes[2].plot(
        episodes,
        history["entropy"]
    )

    axes[2].set_title("Policy Entropy")
    axes[2].set_xlabel("Training episode")
    axes[2].set_ylabel("Entropy")
    axes[2].grid(alpha=0.3)

    plt.tight_layout()


def visualize_neural_network_gradient_check():
    parameters = initialize_network()

    x = np.array([0.5, -0.2, 0.8])
    target = np.array([1.0, -1.0])

    output, cache = neural_network_forward(
        x,
        parameters
    )

    analytical = neural_network_backward(
        parameters,
        cache,
        target,
        output
    )

    numerical = numerical_parameter_gradient(
        parameters,
        x,
        target
    )

    names = list(parameters.keys())

    errors = []

    for name in names:
        difference = (
            analytical[name] - numerical[name]
        )

        errors.append(
            np.max(np.abs(difference))
        )

    plt.figure(figsize=(7, 4))

    plt.bar(names, errors)

    plt.yscale("log")
    plt.ylabel("Maximum absolute gradient error")
    plt.title("Neural Network Gradient Verification")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()


# ================================================================
# UNIT TESTS AND NUMERICAL VERIFICATION
# ================================================================

def run_tests():
    print("\n========== UNIT TESTS ==========\n")

    # Test 1: Linear Q-function
    _, _, _, q_values = linear_algebra_demo()

    assert q_values.shape == (2,)
    assert np.all(np.isfinite(q_values))

    print("[PASS] Linear Q-function")

    # Test 2: Analytical versus numerical gradient
    point = np.array([1.5, 2.0])

    analytical = analytical_gradient(point)

    numerical = numerical_gradient(
        quadratic_loss,
        point
    )

    assert np.allclose(
        analytical,
        numerical,
        atol=1e-6
    )

    print("[PASS] Analytical/numerical gradient")

    # Test 3: Gradient descent convergence
    trajectory, losses = gradient_descent(
        quadratic_loss,
        analytical_gradient,
        initial_position=(-2.0, 3.0),
        learning_rate=0.1,
        steps=100
    )

    assert losses[-1] < losses[0]
    assert np.allclose(
        trajectory[-1],
        np.array([3.0, -1.0]),
        atol=1e-4
    )

    print("[PASS] Gradient descent convergence")

    # Test 4: Softmax normalization
    probabilities = stable_softmax(
        np.array([1.0, 2.0, 3.0])
    )

    assert np.all(probabilities >= 0.0)
    assert np.isclose(np.sum(probabilities), 1.0)

    print("[PASS] Softmax normalization")

    # Test 5: Softmax numerical stability
    large_logits = np.array([10000.0, 10001.0, 10002.0])

    stable_probabilities = stable_softmax(large_logits)

    assert np.all(np.isfinite(stable_probabilities))
    assert np.isclose(
        np.sum(stable_probabilities),
        1.0
    )

    print("[PASS] Stable Softmax")

    # Test 6: Expected value
    mean = expected_value(
        [1.0, 3.0],
        [0.25, 0.75]
    )

    assert np.isclose(mean, 2.5)

    print("[PASS] Expected value")

    # Test 7: Variance
    var = variance(
        [1.0, 3.0],
        [0.5, 0.5]
    )

    assert np.isclose(var, 1.0)

    print("[PASS] Variance")

    # Test 8: Entropy
    entropy = categorical_entropy(
        [0.5, 0.5]
    )

    assert np.isclose(
        entropy,
        np.log(2.0)
    )

    print("[PASS] Categorical entropy")

    # Test 9: Log-Sum-Exp
    values = np.array([1000.0, 1001.0])

    result = log_sum_exp(values)

    expected = 1001.0 + np.log(
        1.0 + np.exp(-1.0)
    )

    assert np.isclose(result, expected)

    print("[PASS] Stable Log-Sum-Exp")

    # Test 10: Neural network gradient checking
    parameters = initialize_network()

    x = np.array([0.5, -0.2, 0.8])
    target = np.array([1.0, -1.0])

    output, cache = neural_network_forward(
        x,
        parameters
    )

    analytical = neural_network_backward(
        parameters,
        cache,
        target,
        output
    )

    numerical = numerical_parameter_gradient(
        parameters,
        x,
        target
    )

    for name in parameters:
        assert np.allclose(
            analytical[name],
            numerical[name],
            atol=1e-6,
            rtol=1e-5
        ), f"Gradient mismatch: {name}"

    print("[PASS] Neural network backpropagation")

    # Test 11: Policy gradient
    logits = np.array([0.0, 0.0])
    rewards = np.array([1.0, 3.0])

    statistics = policy_statistics(
        logits,
        rewards
    )

    numerical_policy_gradient = numerical_gradient(
        lambda theta: policy_statistics(
            theta,
            rewards
        )["expected_reward"],
        logits
    )

    assert np.allclose(
        statistics["gradient"],
        numerical_policy_gradient,
        atol=1e-6
    )

    print("[PASS] Policy gradient verification")

    # Test 12: Policy learning
    final_logits, history = train_stochastic_policy()

    initial_reward = history["expected_reward"][0]
    final_reward = policy_statistics(
        final_logits,
        rewards
    )["expected_reward"]

    assert final_reward > initial_reward
    assert final_reward > 2.9

    print("[PASS] Stochastic policy learning")

    print("\nAll 12 tests passed successfully.\n")


# ================================================================
# MAIN
# ================================================================

def main():
    print("\n========================================")
    print("RL_Course — Phase 0.6")
    print("Mathematical Foundations Review")
    print("========================================")

    # STEP 1
    state, weights, bias, q_values = linear_algebra_demo()

    print("\n[STEP 1] Linear Algebra")
    print("State:", state)
    print("Q-values:", q_values)
    print("Selected action:", int(np.argmax(q_values)))

    # STEP 2
    point = np.array([1.5, 2.0])

    print("\n[STEP 2] Calculus")
    print("Position:", point)
    print("Analytical gradient:", analytical_gradient(point))
    print(
        "Numerical gradient:",
        numerical_gradient(quadratic_loss, point)
    )

    # STEP 3
    logits = np.array([0.0, 1.0, 2.0])

    probabilities = stable_softmax(logits)

    print("\n[STEP 3] Probability")
    print("Action probabilities:", probabilities)
    print("Expected value:", expected_value(
        [1.0, 2.0, 3.0],
        probabilities
    ))
    print("Entropy:", categorical_entropy(probabilities))

    # STEP 4
    print("\n[STEP 4] Numerical Stability")

    large_logits = np.array([1000.0, 1001.0, 1002.0])

    print(
        "Stable Softmax:",
        stable_softmax(large_logits)
    )

    print(
        "Stable Log-Sum-Exp:",
        log_sum_exp(large_logits)
    )

    # STEP 5
    print("\n[STEP 5] Neural Network")

    parameters = initialize_network()

    x = np.array([0.5, -0.2, 0.8])
    target = np.array([1.0, -1.0])

    output, cache = neural_network_forward(
        x,
        parameters
    )

    print("Network output:", output)
    print(
        "Loss:",
        neural_network_loss(parameters, x, target)
    )

    # STEP 6
    print("\n[STEP 6] Mini RL Policy Experiment")

    rewards = np.array([1.0, 3.0])

    initial_statistics = policy_statistics(
        [0.0, 0.0],
        rewards
    )

    final_logits, history = train_stochastic_policy(
        rewards=rewards,
        initial_logits=(0.0, 0.0),
        learning_rate=0.1,
        episodes=100
    )

    final_statistics = policy_statistics(
        final_logits,
        rewards
    )

    print(
        "Initial probabilities:",
        initial_statistics["probabilities"]
    )

    print(
        "Final probabilities:",
        final_statistics["probabilities"]
    )

    print(
        "Initial expected reward:",
        initial_statistics["expected_reward"]
    )

    print(
        "Final expected reward:",
        final_statistics["expected_reward"]
    )

    print(
        "Final entropy:",
        final_statistics["entropy"]
    )

    # STEP 7
    print("\n[STEP 7] Unit Tests")

    run_tests()

    print("[STEP 7] Generating visualizations...")

    visualize_q_values()
    visualize_gradient_descent()
    visualize_softmax_temperature()
    visualize_policy_training(history)
    visualize_neural_network_gradient_check()

    plt.show()

    print("\nPhase 0.6 completed.")


if __name__ == "__main__":
    main()
