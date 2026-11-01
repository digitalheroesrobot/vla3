
"""
RL Course - Phase 0.5
Numerical Stability and Optimization

Installation:
    pip install numpy matplotlib

Run:
    python phase0/numerical_stability.py

Topics:
    1. Floating-point precision
    2. Catastrophic cancellation
    3. Overflow and underflow
    4. Stable Softmax
    5. Stable Log-Sum-Exp
    6. Probability clipping
    7. Gradient clipping
    8. Loss scaling and mixed precision
    9. SGD, Momentum, and Adam
    10. Numerical stability in RL targets

Learning cycle:
    Understand -> Implement -> Visualize -> Verify -> Experiment
"""

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. Floating-Point Precision
# ============================================================

def floating_point_demo():
    """
    Computers represent most real numbers approximately.

    Machine epsilon:
        The smallest positive number eps such that:

            1.0 + eps != 1.0

    For float64:
        eps ≈ 2.22e-16

    For float32:
        eps ≈ 1.19e-7

    Lower precision saves memory and computation time,
    but increases numerical error.

    This matters in RL because neural network training
    repeatedly performs matrix multiplication and gradient updates.
    """

    print("\n[1] Floating-Point Precision")

    print("float64 epsilon:", np.finfo(np.float64).eps)
    print("float32 epsilon:", np.finfo(np.float32).eps)
    print("float16 epsilon:", np.finfo(np.float16).eps)

    print(
        "float32: 1.0 + 1e-8 =",
        np.float32(1.0) + np.float32(1e-8)
    )

    print(
        "float64: 1.0 + 1e-8 =",
        np.float64(1.0) + np.float64(1e-8)
    )


# ============================================================
# 2. Catastrophic Cancellation
# ============================================================

def cancellation_demo():
    """
    Catastrophic cancellation occurs when subtracting
    two nearly equal floating-point numbers.

    Example:

        x = 1e16

        (x + 1) - x

    In float64, x + 1 may be rounded back to x.
    The result becomes zero instead of one.

    The mathematical expression is correct.
    The finite-precision representation loses information.
    """

    x = np.float64(1e16)

    with np.errstate(
        over="ignore",
        under="ignore",
        invalid="ignore"
    ):
        direct = (x + 1.0) - x

    # A smaller-scale calculation preserves the increment.
    reference = (
        np.float64(1e8) + 1.0
    ) - np.float64(1e8)

    return direct, reference


# ============================================================
# 3. Overflow and Underflow
# ============================================================

def overflow_underflow_demo():
    """
    Overflow:
        A value exceeds the largest representable number.

    Underflow:
        A very small value becomes zero or loses precision.

    For example:

        exp(1000) -> overflow
        exp(-1000) -> underflow

    Direct exponential calculations are dangerous
    in Softmax and probability calculations.
    """

    with np.errstate(
        over="ignore",
        under="ignore",
        invalid="ignore"
    ):
        overflow = np.exp(1000.0)
        underflow = np.exp(-1000.0)

    return overflow, underflow


# ============================================================
# 4. Softmax
# ============================================================

def softmax_naive(logits):
    """
    Naive Softmax:

        P(a_i) = exp(z_i) / sum_j exp(z_j)

    Problem:
        exp(z_i) can overflow for large logits.

    Do not use this implementation in production RL code.
    """

    logits = np.asarray(logits, dtype=np.float64)

    exp_values = np.exp(logits)

    return exp_values / np.sum(exp_values)


def softmax_stable(logits):
    """
    Numerically stable Softmax:

        P(a_i) = exp(z_i - max(z))
                 -------------------
                 sum_j exp(z_j - max(z))

    Subtracting the maximum does not change the result
    because Softmax is invariant to a common additive constant.

    Proof:

        exp(z_i - c) / sum_j exp(z_j - c)

        = exp(z_i)exp(-c) /
          [exp(-c) sum_j exp(z_j)]

        = exp(z_i) / sum_j exp(z_j)

    Choosing c = max(z) prevents exponential overflow.

    This is useful for:
        - Categorical policies
        - Action selection
        - Classification networks
        - Policy gradient algorithms
    """

    logits = np.asarray(logits, dtype=np.float64)

    shifted = logits - np.max(
        logits,
        axis=-1,
        keepdims=True
    )

    exp_values = np.exp(shifted)

    return exp_values / np.sum(
        exp_values,
        axis=-1,
        keepdims=True
    )


# ============================================================
# 5. Log-Sum-Exp
# ============================================================

def logsumexp_naive(values):
    """
    Naive calculation:

        log(sum(exp(x_i)))

    This can overflow when x_i is large.
    """

    values = np.asarray(values, dtype=np.float64)

    with np.errstate(
        over="ignore",
        under="ignore",
        invalid="ignore"
    ):
        return np.log(np.sum(np.exp(values)))


def logsumexp_stable(values):
    """
    Stable Log-Sum-Exp:

        log(sum(exp(x_i)))

        = m + log(sum(exp(x_i - m)))

    where:

        m = max(x)

    This avoids overflow during exponentiation.

    Log-Sum-Exp is important in:
        - Log probabilities
        - Policy distributions
        - Entropy calculations
        - Probabilistic models
    """

    values = np.asarray(values, dtype=np.float64)

    maximum = np.max(values)

    return maximum + np.log(
        np.sum(np.exp(values - maximum))
    )


def log_softmax_stable(logits):
    """
    Stable Log-Softmax:

        log_softmax(z_i)
        = z_i - logsumexp(z)

    Computing log probabilities directly is more stable
    than calculating Softmax first and then taking log.
    """

    logits = np.asarray(logits, dtype=np.float64)

    maximum = np.max(
        logits,
        axis=-1,
        keepdims=True
    )

    shifted = logits - maximum

    return shifted - np.log(
        np.sum(
            np.exp(shifted),
            axis=-1,
            keepdims=True
        )
    )


# ============================================================
# 6. Probability Clipping
# ============================================================

def clipped_probability(probability, epsilon=1e-8):
    """
    Protect logarithms from zero probabilities.

        p_clipped = clip(p, epsilon, 1-epsilon)

    Without clipping:

        log(0) = -infinity

    This can create infinite losses and invalid gradients.

    Note:
        Clipping is a numerical safeguard.
        It should not be used to hide incorrect probability
        calculations or invalid model outputs.
    """

    return np.clip(
        probability,
        epsilon,
        1.0 - epsilon
    )


def safe_log_probability(probability, epsilon=1e-8):
    """
    Stable log probability after clipping.
    """

    probability = clipped_probability(
        probability,
        epsilon
    )

    return np.log(probability)


# ============================================================
# 7. Gradient Clipping
# ============================================================

def clip_gradient_global_norm(
    gradient,
    max_norm
):
    """
    Global norm clipping:

        ||g|| = sqrt(sum_i(g_i^2))

    If:

        ||g|| > max_norm

    Then:

        g_clipped = g * max_norm / ||g||

    Otherwise:

        g_clipped = g

    This preserves the direction of the gradient.

    Commonly used in recurrent neural networks,
    policy gradient algorithms, and deep RL.
    """

    gradient = np.asarray(
        gradient,
        dtype=np.float64
    )

    norm = np.linalg.norm(gradient)

    if norm > max_norm:

        gradient = (
            gradient *
            max_norm /
            (norm + 1e-12)
        )

    return gradient


def clip_gradient_elementwise(
    gradient,
    min_value=-1.0,
    max_value=1.0
):
    """
    Elementwise clipping:

        g_i = clip(g_i, min_value, max_value)

    Unlike global norm clipping, this can change
    the direction of the gradient.
    """

    return np.clip(
        gradient,
        min_value,
        max_value
    )


# ============================================================
# 8. Loss Scaling / Mixed Precision
# ============================================================

def loss_scaling_demo():

    """
    Demonstrate float16 gradient underflow and loss scaling.

    Original gradient:
        1e-8

    Direct float16 conversion:
        May underflow to zero.

    Loss scaling:
        scaled_gradient = gradient * scale

    Recovery:
        recovered_gradient = scaled_gradient / scale

    Note:
        float16 introduces quantization error.
        Loss scaling reduces underflow but does not
        eliminate floating-point precision limitations.
    """

    original_gradient = np.float32(1e-8)

    scale = np.float32(1024.0)

    # Direct conversion.
    gradient_float16 = np.float16(
        original_gradient
    )

    # Scale before conversion.
    scaled_gradient = np.float16(
        original_gradient * scale
    )

    # Convert back and unscale.
    recovered_gradient = (
        np.float32(scaled_gradient) / scale
    )

    return (
        original_gradient,
        gradient_float16,
        scaled_gradient,
        recovered_gradient
    )


# ============================================================
# 9. Optimization Objective
# ============================================================

def optimization_objective(parameters):
    """
    Ill-conditioned quadratic objective:

        L(x,y) = 0.5 * (100*x^2 + y^2)

    Gradient:

        dL/dx = 100*x
        dL/dy = y

    Hessian:

        H = [[100, 0],
             [0,   1]]

    The condition number is 100.

    The x direction is much steeper than the y direction.

    This is a simple model of an optimization problem
    where different parameter directions have very
    different curvature.
    """

    x, y = parameters

    return 0.5 * (
        100.0*x**2 +
        y**2
    )


def optimization_gradient(parameters):

    x, y = parameters

    return np.array([
        100.0*x,
        y
    ])


# ============================================================
# 10. SGD Optimizer
# ============================================================

class SGD:

    def __init__(self, learning_rate=0.01):

        self.learning_rate = learning_rate

    def update(self, parameters, gradient):

        return (
            parameters -
            self.learning_rate * gradient
        )


# ============================================================
# 11. Momentum Optimizer
# ============================================================

class Momentum:

    def __init__(
        self,
        learning_rate=0.01,
        beta=0.9
    ):

        self.learning_rate = learning_rate
        self.beta = beta
        self.velocity = None

    def update(self, parameters, gradient):

        if self.velocity is None:

            self.velocity = np.zeros_like(
                parameters
            )

        # v_t = beta*v_(t-1) + (1-beta)*g_t
        self.velocity = (
            self.beta * self.velocity +
            (1.0 - self.beta) * gradient
        )

        # theta_t = theta_(t-1) - alpha*v_t
        return (
            parameters -
            self.learning_rate * self.velocity
        )


# ============================================================
# 12. Adam Optimizer
# ============================================================

class Adam:

    def __init__(
        self,
        learning_rate=0.05,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8
    ):

        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.m = None
        self.v = None
        self.t = 0

    def update(self, parameters, gradient):
        """
        Adam equations:

            m_t = beta1*m_(t-1) + (1-beta1)*g_t

            v_t = beta2*v_(t-1) + (1-beta2)*g_t^2

        Bias correction:

            m_hat = m_t / (1-beta1^t)

            v_hat = v_t / (1-beta2^t)

        Update:

            theta = theta - alpha*m_hat /
                    (sqrt(v_hat) + epsilon)

        Bias correction is important because m and v
        are initialized at zero.
        """

        if self.m is None:

            self.m = np.zeros_like(parameters)
            self.v = np.zeros_like(parameters)

        self.t += 1

        self.m = (
            self.beta1 * self.m +
            (1.0 - self.beta1) * gradient
        )

        self.v = (
            self.beta2 * self.v +
            (1.0 - self.beta2) * gradient**2
        )

        m_hat = self.m / (
            1.0 - self.beta1**self.t
        )

        v_hat = self.v / (
            1.0 - self.beta2**self.t
        )

        return (
            parameters -
            self.learning_rate *
            m_hat /
            (np.sqrt(v_hat) + self.epsilon)
        )


# ============================================================
# 13. Optimizer Experiment
# ============================================================

def run_optimizer(
    optimizer,
    initial_parameters,
    iterations=500
):

    parameters = np.asarray(
        initial_parameters,
        dtype=np.float64
    ).copy()

    parameter_history = []
    loss_history = []

    for _ in range(iterations):

        parameter_history.append(
            parameters.copy()
        )

        loss_history.append(
            optimization_objective(parameters)
        )

        gradient = optimization_gradient(
            parameters
        )

        parameters = optimizer.update(
            parameters,
            gradient
        )

    return (
        np.asarray(parameter_history),
        np.asarray(loss_history),
        parameters
    )


def optimizer_experiment():

    initial_parameters = np.array(
        [2.0, 2.0]
    )

    optimizers = {
        "SGD": SGD(learning_rate=0.01),

        "Momentum": Momentum(
            learning_rate=0.01,
            beta=0.9
        ),

        "Adam": Adam(
            learning_rate=0.05
        )
    }

    results = {}

    for name, optimizer in optimizers.items():

        results[name] = run_optimizer(
            optimizer,
            initial_parameters,
            iterations=500
        )

    return results


# ============================================================
# 14. RL Target Calculation
# ============================================================

def calculate_dqn_target(
    reward,
    next_q_values,
    gamma,
    done
):
    """
    DQN target:

        y = r + gamma*(1-done)*max_a Q_target(s',a)

    done:
        1 if the episode terminates
        0 otherwise

    Important:
        Terminal transitions must not bootstrap
        from the next state.

    Numerical stability considerations:
        - Use finite Q-values.
        - Keep rewards and targets in a reasonable scale.
        - Avoid NaN and infinity.
        - Use float32 or float64 deliberately.
    """

    next_q_values = np.asarray(
        next_q_values,
        dtype=np.float64
    )

    if not np.all(np.isfinite(next_q_values)):
        raise ValueError(
            "Q-values contain NaN or infinity."
        )

    if not 0.0 <= gamma <= 1.0:
        raise ValueError(
            "gamma must be between 0 and 1."
        )

    if done:

        return float(reward)

    return float(
        reward +
        gamma * np.max(next_q_values)
    )


# ============================================================
# 15. Numerical Stability Demonstration
# ============================================================

def numerical_stability_demo():

    print("\n[2] Cancellation")

    direct, reference = cancellation_demo()

    print("(1e16 + 1) - 1e16 =", direct)

    print("(1e8 + 1) - 1e8 =", reference)

    print("\n[3] Overflow and Underflow")

    overflow, underflow = overflow_underflow_demo()

    print("exp(1000):", overflow)

    print("exp(-1000):", underflow)

    print("\n[4] Stable Softmax")

    logits = np.array([
        1000.0,
        1001.0,
        999.0
    ])

    with np.errstate(
        over="ignore",
        invalid="ignore"
    ):
        naive = softmax_naive(logits)

    stable = softmax_stable(logits)

    print("Naive Softmax:", naive)

    print("Stable Softmax:", stable)

    print("Probability sum:", np.sum(stable))

    print("\n[5] Stable Log-Sum-Exp")

    print(
        "Stable:",
        logsumexp_stable(logits)
    )

    print(
        "Naive:",
        logsumexp_naive(logits)
    )

    print("\n[6] Loss Scaling")

    original, direct16, scaled16, recovered = (
        loss_scaling_demo()
    )

    print("Original gradient:", original)

    print("Direct float16:", direct16)

    print("Scaled float16:", scaled16)

    print("Recovered gradient:", recovered)

    print("\n[7] Gradient Clipping")

    gradient = np.array([
        100.0,
        200.0,
        -300.0
    ])

    clipped = clip_gradient_global_norm(
        gradient,
        max_norm=10.0
    )

    print("Original norm:", np.linalg.norm(gradient))

    print("Clipped norm:", np.linalg.norm(clipped))


# ============================================================
# 16. Unit Tests
# ============================================================

def run_tests():

    print("\n========== UNIT TESTS ==========")

    # Test 1: Machine epsilon.
    assert (
        np.finfo(np.float64).eps <
        np.finfo(np.float32).eps
    )

    print("[PASS] Floating-point precision")


    # Test 2: Stable Softmax with extreme logits.
    logits = np.array([
        1000.0,
        1001.0,
        999.0
    ])

    probabilities = softmax_stable(logits)

    assert np.all(np.isfinite(probabilities))

    assert np.all(probabilities >= 0.0)

    assert np.isclose(
        np.sum(probabilities),
        1.0,
        atol=1e-12
    )

    print("[PASS] Stable Softmax")


    # Test 3: Softmax translation invariance.
    logits_a = np.array([2.0, 1.0, 0.5])

    logits_b = logits_a + 1000.0

    assert np.allclose(
        softmax_stable(logits_a),
        softmax_stable(logits_b),
        atol=1e-12
    )

    print("[PASS] Softmax translation invariance")


    # Test 4: Stable Log-Sum-Exp.
    result = logsumexp_stable(
        np.array([1000.0, 1001.0, 999.0])
    )

    expected = (
        1001.0 +
        np.log(
            1.0 + np.exp(-1.0) + np.exp(-2.0)
        )
    )

    assert np.isclose(
        result,
        expected,
        atol=1e-12
    )

    assert np.isfinite(result)

    print("[PASS] Stable Log-Sum-Exp")


    # Test 5: Log-Softmax.
    log_probabilities = log_softmax_stable(
        np.array([1000.0, 1001.0, 999.0])
    )

    assert np.all(np.isfinite(log_probabilities))

    assert np.isclose(
        np.sum(np.exp(log_probabilities)),
        1.0,
        atol=1e-12
    )

    print("[PASS] Stable Log-Softmax")


    # Test 6: Probability clipping.
    probabilities = np.array([
        0.0,
        0.5,
        1.0
    ])

    clipped = clipped_probability(probabilities)

    assert np.all(clipped > 0.0)

    assert np.all(clipped < 1.0)

    assert np.all(np.isfinite(np.log(clipped)))

    print("[PASS] Probability clipping")


    # Test 7: Global norm clipping.
    gradient = np.array([
        3.0,
        4.0
    ])

    clipped = clip_gradient_global_norm(
        gradient,
        max_norm=2.0
    )

    assert np.isclose(
        np.linalg.norm(clipped),
        2.0,
        atol=1e-10
    )

    # Global norm clipping preserves direction.
    assert np.allclose(
        clipped / np.linalg.norm(clipped),
        gradient / np.linalg.norm(gradient)
    )

    print("[PASS] Global norm clipping")


    # Test 8: Elementwise clipping.
    gradient = np.array([
        -5.0,
        0.5,
        8.0
    ])

    clipped = clip_gradient_elementwise(
        gradient,
        -1.0,
        1.0
    )

    assert np.allclose(
        clipped,
        np.array([-1.0, 0.5, 1.0])
    )

    print("[PASS] Elementwise clipping")


    # Test 9: Loss scaling.
    original, direct16, scaled16, recovered = (
        loss_scaling_demo()
    )

    assert direct16 == 0.0

    assert scaled16 != 0.0

    assert np.isclose(
        recovered,
        original,
        rtol=2e-3,
        atol=1e-12
    )

    print("[PASS] Loss scaling")


    # Test 10: SGD convergence.
    sgd_history, sgd_losses, sgd_final = run_optimizer(
        SGD(learning_rate=0.01),
        np.array([2.0, 2.0]),
        iterations=500
    )

    assert sgd_losses[-1] < sgd_losses[0]

    assert np.all(np.isfinite(sgd_final))

    print("[PASS] SGD")


    # Test 11: Momentum convergence.
    momentum_history, momentum_losses, momentum_final = (
        run_optimizer(
            Momentum(
                learning_rate=0.01,
                beta=0.9
            ),
            np.array([2.0, 2.0]),
            iterations=500
        )
    )

    assert momentum_losses[-1] < momentum_losses[0]

    assert np.all(np.isfinite(momentum_final))

    print("[PASS] Momentum")


    # Test 12: Adam convergence.
    adam_history, adam_losses, adam_final = run_optimizer(
        Adam(learning_rate=0.05),
        np.array([2.0, 2.0]),
        iterations=500
    )

    assert adam_losses[-1] < adam_losses[0]

    assert np.all(np.isfinite(adam_final))

    print("[PASS] Adam")


    # Test 13: Adam bias correction and iteration count.
    optimizer = Adam()

    parameters = np.array([1.0, 2.0])

    updated = optimizer.update(
        parameters,
        np.array([0.5, -0.5])
    )

    assert optimizer.t == 1

    assert np.all(np.isfinite(updated))

    print("[PASS] Adam bias correction")


    # Test 14: RL target for non-terminal transition.
    target = calculate_dqn_target(
        reward=1.0,
        next_q_values=np.array([2.0, 3.0, 1.0]),
        gamma=0.99,
        done=False
    )

    assert np.isclose(
        target,
        1.0 + 0.99*3.0
    )

    print("[PASS] DQN non-terminal target")


    # Test 15: RL target for terminal transition.
    target = calculate_dqn_target(
        reward=1.0,
        next_q_values=np.array([1000.0, 2000.0]),
        gamma=0.99,
        done=True
    )

    assert target == 1.0

    print("[PASS] DQN terminal target")


    # Test 16: Reject invalid Q-values.
    try:

        calculate_dqn_target(
            reward=1.0,
            next_q_values=np.array([1.0, np.nan]),
            gamma=0.99,
            done=False
        )

        raise AssertionError(
            "Invalid Q-values were not rejected."
        )

    except ValueError:
        pass

    print("[PASS] Invalid RL target detection")


    print("\n================================")
    print("ALL UNIT TESTS PASSED!")
    print("================================\n")


# ============================================================
# 17. Visualization: Stable Softmax
# ============================================================

def visualize_softmax():

    normal_logits = np.array([
        2.0,
        1.0,
        0.5
    ])

    extreme_logits = np.array([
        1000.0,
        1001.0,
        999.0
    ])

    normal_probabilities = softmax_stable(
        normal_logits
    )

    extreme_probabilities = softmax_stable(
        extreme_logits
    )

    with np.errstate(
        over="ignore",
        invalid="ignore"
    ):
        naive_extreme = softmax_naive(
            extreme_logits
        )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11, 4)
    )

    axes[0].bar(
        ["Action 0", "Action 1", "Action 2"],
        normal_probabilities
    )

    axes[0].set_title(
        "Stable Softmax: Normal Logits"
    )

    axes[0].set_ylabel("Probability")

    axes[0].set_ylim(0, 1)

    axes[1].bar(
        ["Action 0", "Action 1", "Action 2"],
        extreme_probabilities
    )

    axes[1].set_title(
        "Stable Softmax: Extreme Logits"
    )

    axes[1].set_ylim(0, 1)

    axes[1].text(
        0.5,
        0.85,
        f"Naive result: {naive_extreme}",
        transform=axes[1].transAxes,
        ha="center",
        fontsize=9
    )

    for ax in axes:
        ax.grid(axis="y")

    fig.tight_layout()


# ============================================================
# 18. Visualization: Floating-Point Precision
# ============================================================

def visualize_precision():

    dtypes = [
        np.float16,
        np.float32,
        np.float64
    ]

    epsilons = [
        np.finfo(dtype).eps
        for dtype in dtypes
    ]

    names = [
        "float16",
        "float32",
        "float64"
    ]

    plt.figure(figsize=(8, 5))

    plt.bar(
        names,
        epsilons
    )

    plt.yscale("log")

    plt.ylabel("Machine epsilon")

    plt.title("Floating-Point Precision Comparison")

    plt.grid(axis="y")


# ============================================================
# 19. Visualization: Gradient Clipping
# ============================================================

def visualize_gradient_clipping():

    gradient = np.array([
        100.0,
        200.0,
        -300.0
    ])

    clipped = clip_gradient_global_norm(
        gradient,
        max_norm=10.0
    )

    indices = np.arange(len(gradient))

    width = 0.35

    plt.figure(figsize=(8, 5))

    plt.bar(
        indices - width/2,
        gradient,
        width,
        label="Original"
    )

    plt.bar(
        indices + width/2,
        clipped,
        width,
        label="Clipped"
    )

    plt.xticks(
        indices,
        ["g1", "g2", "g3"]
    )

    plt.ylabel("Gradient value")

    plt.title("Global Norm Gradient Clipping")

    plt.legend()

    plt.grid(axis="y")


# ============================================================
# 20. Visualization: Optimizer Comparison
# ============================================================

def visualize_optimizer_comparison():

    results = optimizer_experiment()

    plt.figure(figsize=(9, 5))

    for name, result in results.items():

        _, losses, _ = result

        plt.semilogy(
            losses,
            label=name
        )

    plt.xlabel("Iteration")

    plt.ylabel("Loss")

    plt.title("SGD vs Momentum vs Adam")

    plt.grid(True)

    plt.legend()


# ============================================================
# 21. Visualization: Optimization Trajectories
# ============================================================

def visualize_optimizer_trajectories():

    results = optimizer_experiment()

    x = np.linspace(-2.2, 2.2, 150)

    y = np.linspace(-2.2, 2.2, 150)

    X, Y = np.meshgrid(x, y)

    Z = 0.5 * (
        100.0*X**2 +
        Y**2
    )

    plt.figure(figsize=(9, 6))

    plt.contour(
        X,
        Y,
        Z,
        levels=np.logspace(-2, 3, 20)
    )

    for name, result in results.items():

        history, _, _ = result

        plt.plot(
            history[:, 0],
            history[:, 1],
            label=name
        )

    plt.scatter(
        [0.0],
        [0.0],
        marker="*",
        s=150,
        label="Minimum"
    )

    plt.xlabel("x")

    plt.ylabel("y")

    plt.title("Optimizer Trajectories")

    plt.legend()

    plt.grid(True)


# ============================================================
# 22. Visualization: Loss Scaling
# ============================================================

def visualize_loss_scaling():

    original, direct16, scaled16, recovered = (
        loss_scaling_demo()
    )

    values = [
        original,
        float(direct16),
        float(scaled16),
        float(recovered)
    ]

    labels = [
        "Original",
        "Direct float16",
        "Scaled float16",
        "Recovered"
    ]

    plt.figure(figsize=(9, 5))

    plt.bar(
        labels,
        values
    )

    plt.yscale("symlog", linthresh=1e-10)

    plt.ylabel("Gradient value")

    plt.title("Loss Scaling and Float16 Underflow")

    plt.grid(axis="y")


# ============================================================
# 23. Main
# ============================================================

def main():

    print("\n========================================")
    print(" RL COURSE - PHASE 0.5")
    print(" NUMERICAL STABILITY AND OPTIMIZATION")
    print("========================================")

    # Numerical demonstrations.
    floating_point_demo()

    numerical_stability_demo()

    # Run all unit tests.
    run_tests()

    # Integrated visualizations.
    visualize_softmax()

    visualize_precision()

    visualize_gradient_clipping()

    visualize_optimizer_comparison()

    visualize_optimizer_trajectories()

    visualize_loss_scaling()

    plt.tight_layout()

    plt.show()


if __name__ == "__main__":

    main()
