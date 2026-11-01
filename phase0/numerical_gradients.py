
"""
RL Course - Phase 0.4
Numerical Gradients and Gradient Verification

Run:
    pip install numpy matplotlib
    python numerical_gradients.py

Learning cycle:
    Understand (comments)
    -> Implement
    -> Visualize
    -> Verify
    -> Experiment

Topics:
    1. Forward finite differences
    2. Central finite differences
    3. Numerical gradients
    4. Numerical Jacobians
    5. Directional derivatives
    6. Analytical derivatives
    7. Neural network backpropagation
    8. Numerical vs analytical gradient verification
    9. Epsilon sensitivity
    10. Visualization

All theory and mathematical explanations are included in comments.
"""

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. Numerical Differentiation
# ============================================================

def scalar_function(x):
    """
    Scalar test function.

    f(x) = x^3 - 2x^2 + 3x + 1

    f'(x) = 3x^2 - 4x + 3
    """
    return x**3 - 2.0*x**2 + 3.0*x + 1.0


def scalar_function_derivative(x):
    """
    Analytical derivative:

    f'(x) = 3x^2 - 4x + 3
    """
    return 3.0*x**2 - 4.0*x + 3.0


def forward_difference(function, x, epsilon=1e-5):
    """
    Forward finite difference:

        f'(x) ≈ [f(x + ε) - f(x)] / ε

    First-order approximation.
    """
    return (
        function(x + epsilon) - function(x)
    ) / epsilon


def central_difference(function, x, epsilon=1e-5):
    """
    Central finite difference:

        f'(x) ≈ [f(x + ε) - f(x - ε)] / (2ε)

    Usually more accurate than forward difference.
    """
    return (
        function(x + epsilon) -
        function(x - epsilon)
    ) / (2.0 * epsilon)


# ============================================================
# 2. Numerical Gradient
# ============================================================

def numerical_gradient(
    function,
    parameters,
    epsilon=1e-5
):
    """
    Calculate numerical gradient for a scalar function.

    For:

        f(x1, x2, ..., xn)

    The gradient is:

        ∇f = [
            ∂f/∂x1,
            ∂f/∂x2,
            ...,
            ∂f/∂xn
        ]

    Each partial derivative is estimated using
    central finite differences.
    """

    parameters = np.asarray(
        parameters,
        dtype=np.float64
    )

    gradient = np.zeros_like(parameters)

    for i in range(len(parameters)):

        parameters_plus = parameters.copy()
        parameters_minus = parameters.copy()

        parameters_plus[i] += epsilon
        parameters_minus[i] -= epsilon

        gradient[i] = (
            function(parameters_plus) -
            function(parameters_minus)
        ) / (2.0 * epsilon)

    return gradient


# ============================================================
# 3. Numerical Jacobian
# ============================================================

def nonlinear_vector_function(x):
    """
    Nonlinear vector function:

        f1(x1, x2) = x1^2 + 3*x2

        f2(x1, x2) = sin(x1*x2)

    Output dimension: 2
    Input dimension: 2
    """

    x1, x2 = x

    return np.array([
        x1**2 + 3.0*x2,
        np.sin(x1*x2)
    ])


def numerical_jacobian(
    function,
    x,
    epsilon=1e-5
):
    """
    Numerical Jacobian:

        J[i,j] = ∂f_i / ∂x_j

    If input dimension is n and output dimension is m:

        Jacobian shape = (m, n)
    """

    x = np.asarray(
        x,
        dtype=np.float64
    )

    f0 = np.asarray(function(x))

    output_size = len(f0)
    input_size = len(x)

    J = np.zeros(
        (output_size, input_size)
    )

    for j in range(input_size):

        x_plus = x.copy()
        x_minus = x.copy()

        x_plus[j] += epsilon
        x_minus[j] -= epsilon

        f_plus = np.asarray(function(x_plus))
        f_minus = np.asarray(function(x_minus))

        J[:, j] = (
            f_plus - f_minus
        ) / (2.0 * epsilon)

    return J


def analytical_jacobian(x):
    """
    Analytical Jacobian.

    f1 = x1^2 + 3*x2
    f2 = sin(x1*x2)

    Therefore:

        ∂f1/∂x1 = 2*x1
        ∂f1/∂x2 = 3

        ∂f2/∂x1 = x2*cos(x1*x2)
        ∂f2/∂x2 = x1*cos(x1*x2)

    Jacobian:

        [[2*x1,                 3],
         [x2*cos(x1*x2), x1*cos(x1*x2)]]
    """

    x1, x2 = x

    c = np.cos(x1*x2)

    return np.array([
        [2.0*x1, 3.0],
        [x2*c, x1*c]
    ])


# ============================================================
# 4. Directional Derivative
# ============================================================

def quadratic(parameters):
    """
    Quadratic function:

        f(x,y) = x^2 + 3xy + 2y^2

    Analytical gradient:

        ∂f/∂x = 2x + 3y
        ∂f/∂y = 3x + 4y

    At (2,1):

        ∇f = [7, 10]

    IMPORTANT:
        The previous version incorrectly used [7, 7].
    """

    x, y = parameters

    return (
        x**2 +
        3.0*x*y +
        2.0*y**2
    )


def quadratic_analytical_gradient(parameters):

    x, y = parameters

    return np.array([
        2.0*x + 3.0*y,
        3.0*x + 4.0*y
    ])


def directional_derivative_check(
    function,
    parameters,
    direction,
    epsilon=1e-5
):
    """
    Directional derivative:

        D_v f(x) = ∇f(x) · v

    Numerical approximation:

        [f(x + εv) - f(x - εv)] / (2ε)

    Normalize direction to make the comparison meaningful.
    """

    parameters = np.asarray(parameters, dtype=float)
    direction = np.asarray(direction, dtype=float)

    norm = np.linalg.norm(direction)

    if norm == 0:
        raise ValueError(
            "Direction vector must not be zero."
        )

    direction = direction / norm

    numerical = (
        function(parameters + epsilon*direction) -
        function(parameters - epsilon*direction)
    ) / (2.0*epsilon)

    gradient = numerical_gradient(
        function,
        parameters,
        epsilon=epsilon
    )

    analytical_directional = np.dot(
        gradient,
        direction
    )

    return numerical, analytical_directional


# ============================================================
# 5. Neural Network Parameter Management
# ============================================================

INPUT_SIZE = 2
HIDDEN_SIZE = 4
OUTPUT_SIZE = 2


def initialize_network_parameters(seed=42):
    """
    Two-layer neural network:

        Input -> Tanh Hidden Layer -> Linear Output

    Forward equations:

        z1 = W1*x + b1
        h1 = tanh(z1)

        z2 = W2*h1 + b2
        y  = z2

    Parameter arrays are flattened into one vector
    to make numerical gradient checking possible.
    """

    rng = np.random.default_rng(seed)

    W1 = rng.normal(
        0.0,
        0.3,
        size=(HIDDEN_SIZE, INPUT_SIZE)
    )

    b1 = np.zeros(HIDDEN_SIZE)

    W2 = rng.normal(
        0.0,
        0.3,
        size=(OUTPUT_SIZE, HIDDEN_SIZE)
    )

    b2 = np.zeros(OUTPUT_SIZE)

    return pack_parameters(
        W1,
        b1,
        W2,
        b2
    )


def pack_parameters(W1, b1, W2, b2):

    return np.concatenate([
        W1.ravel(),
        b1.ravel(),
        W2.ravel(),
        b2.ravel()
    ])


def unpack_parameters(parameters):

    parameters = np.asarray(
        parameters,
        dtype=np.float64
    )

    index = 0

    w1_size = HIDDEN_SIZE * INPUT_SIZE

    W1 = parameters[
        index:index+w1_size
    ].reshape(
        HIDDEN_SIZE,
        INPUT_SIZE
    )

    index += w1_size

    b1 = parameters[
        index:index+HIDDEN_SIZE
    ]

    index += HIDDEN_SIZE

    w2_size = OUTPUT_SIZE * HIDDEN_SIZE

    W2 = parameters[
        index:index+w2_size
    ].reshape(
        OUTPUT_SIZE,
        HIDDEN_SIZE
    )

    index += w2_size

    b2 = parameters[
        index:index+OUTPUT_SIZE
    ]

    return W1, b1, W2, b2


# ============================================================
# 6. Neural Network Forward Propagation
# ============================================================

def neural_network_forward(
    parameters,
    state
):
    """
    Forward propagation.

    h = tanh(W1*s + b1)

    output = W2*h + b2

    Cache intermediate values for backpropagation.
    """

    W1, b1, W2, b2 = unpack_parameters(
        parameters
    )

    state = np.asarray(state, dtype=float)

    z1 = W1 @ state + b1

    hidden = np.tanh(z1)

    output = W2 @ hidden + b2

    cache = {
        "state": state,
        "z1": z1,
        "hidden": hidden,
        "output": output,
        "W1": W1,
        "W2": W2
    }

    return output, cache


# ============================================================
# 7. Neural Network Loss
# ============================================================

def neural_network_loss(
    parameters,
    state,
    target
):
    """
    Mean squared error is not used here.

    We define:

        L = 1/2 * Σ(output - target)^2

    Therefore:

        ∂L/∂output = output - target

    This convention makes the analytical gradient simple.
    """

    output, _ = neural_network_forward(
        parameters,
        state
    )

    error = output - target

    return 0.5 * np.sum(error**2)


# ============================================================
# 8. Analytical Backpropagation
# ============================================================

def neural_network_analytical_gradient(
    parameters,
    state,
    target
):
    """
    Manual backpropagation.

    Forward:

        z1 = W1*s + b1
        h1 = tanh(z1)

        y = W2*h1 + b2

    Loss:

        L = 1/2 * sum((y-target)^2)

    Backward:

        delta2 = y - target

        dW2 = outer(delta2, h1)
        db2 = delta2

        delta1 = (W2.T @ delta2) * (1-h1^2)

        dW1 = outer(delta1, state)
        db1 = delta1

    The derivative of tanh(z) is:

        d(tanh(z))/dz = 1 - tanh(z)^2

    Finally, gradients must be flattened in EXACTLY
    the same order as pack_parameters().
    """

    output, cache = neural_network_forward(
        parameters,
        state
    )

    state = cache["state"]
    hidden = cache["hidden"]

    W2 = cache["W2"]

    # Output layer gradient.
    delta2 = output - target

    dW2 = np.outer(
        delta2,
        hidden
    )

    db2 = delta2

    # Hidden layer gradient.
    delta1 = (
        W2.T @ delta2
    ) * (
        1.0 - hidden**2
    )

    dW1 = np.outer(
        delta1,
        state
    )

    db1 = delta1

    return pack_parameters(
        dW1,
        db1,
        dW2,
        db2
    )


# ============================================================
# 9. Gradient Error Metrics
# ============================================================

def gradient_error(
    analytical,
    numerical
):

    analytical = np.asarray(analytical)
    numerical = np.asarray(numerical)

    difference = np.abs(
        analytical - numerical
    )

    max_error = np.max(difference)

    mean_error = np.mean(difference)

    relative_error = (
        np.linalg.norm(difference) /
        max(
            np.linalg.norm(analytical) +
            np.linalg.norm(numerical),
            1e-12
        )
    )

    return {
        "difference": difference,
        "max_error": max_error,
        "mean_error": mean_error,
        "relative_error": relative_error
    }


# ============================================================
# 10. Epsilon Sensitivity Experiment
# ============================================================

def epsilon_sensitivity_experiment():

    x = 1.5

    true_value = scalar_function_derivative(x)

    epsilons = np.logspace(
        -12,
        -1,
        50
    )

    forward_errors = []
    central_errors = []

    for epsilon in epsilons:

        forward = forward_difference(
            scalar_function,
            x,
            epsilon
        )

        central = central_difference(
            scalar_function,
            x,
            epsilon
        )

        forward_errors.append(
            abs(forward - true_value)
        )

        central_errors.append(
            abs(central - true_value)
        )

    return (
        epsilons,
        np.array(forward_errors),
        np.array(central_errors)
    )


# ============================================================
# 11. Gradient Descent Demonstration
# ============================================================

def gradient_descent_demo(
    learning_rate=0.05,
    iterations=100
):
    """
    Minimize:

        f(x,y) = x^2 + 3xy + 5y^2

    The Hessian is positive definite:

        H = [[2, 3],
             [3,10]]

    Therefore the function has a unique global minimum at (0,0).

    Update:

        theta_(t+1) = theta_t - alpha * gradient
    """

    parameters = np.array(
        [4.0, -3.0],
        dtype=float
    )

    history = []

    def objective(p):

        x, y = p

        return (
            x**2 +
            3.0*x*y +
            5.0*y**2
        )

    def gradient(p):

        x, y = p

        return np.array([
            2.0*x + 3.0*y,
            3.0*x + 10.0*y
        ])

    for _ in range(iterations):

        history.append(parameters.copy())

        g = gradient(parameters)

        parameters -= learning_rate*g

    history = np.asarray(history)

    return history, objective


# ============================================================
# 12. Unit Tests
# ============================================================

def run_tests():

    print("\n========== UNIT TESTS ==========")

    # --------------------------------------------------------
    # Test 1: Scalar derivative
    # --------------------------------------------------------

    x = 1.5

    analytical = scalar_function_derivative(x)

    numerical = central_difference(
        scalar_function,
        x,
        epsilon=1e-5
    )

    assert np.isclose(
        analytical,
        numerical,
        atol=1e-7,
        rtol=1e-7
    ), (
        f"Scalar derivative failed: "
        f"{analytical} vs {numerical}"
    )

    print("[PASS] Scalar derivative")


    # --------------------------------------------------------
    # Test 2: Numerical gradient
    # --------------------------------------------------------

    parameters = np.array([2.0, 1.0])

    gradient = numerical_gradient(
        quadratic,
        parameters
    )

    # Correct analytical gradient:
    #
    # df/dx = 2x + 3y = 7
    # df/dy = 3x + 4y = 10
    #
    # Previous version incorrectly expected [7, 7].

    expected = np.array([7.0, 10.0])

    assert np.allclose(
        gradient,
        expected,
        atol=1e-6,
        rtol=1e-6
    ), (
        f"Numerical gradient failed: "
        f"{gradient} vs {expected}"
    )

    print("[PASS] Numerical gradient")


    # --------------------------------------------------------
    # Test 3: Analytical quadratic gradient
    # --------------------------------------------------------

    analytical_gradient = quadratic_analytical_gradient(
        parameters
    )

    assert np.allclose(
        analytical_gradient,
        expected,
        atol=1e-12
    )

    print("[PASS] Analytical quadratic gradient")


    # --------------------------------------------------------
    # Test 4: Jacobian
    # --------------------------------------------------------

    x = np.array([1.5, 0.7])

    numerical_J = numerical_jacobian(
        nonlinear_vector_function,
        x
    )

    analytical_J = analytical_jacobian(x)

    assert np.allclose(
        numerical_J,
        analytical_J,
        atol=1e-6,
        rtol=1e-6
    ), (
        f"Jacobian failed:\n"
        f"Numerical:\n{numerical_J}\n"
        f"Analytical:\n{analytical_J}"
    )

    print("[PASS] Numerical Jacobian")


    # --------------------------------------------------------
    # Test 5: Directional derivative
    # --------------------------------------------------------

    direction = np.array([0.3, 0.8])

    numerical_directional, gradient_directional = (
        directional_derivative_check(
            quadratic,
            parameters,
            direction
        )
    )

    assert np.isclose(
        numerical_directional,
        gradient_directional,
        atol=1e-7,
        rtol=1e-7
    ), (
        f"Directional derivative failed: "
        f"{numerical_directional} vs "
        f"{gradient_directional}"
    )

    print("[PASS] Directional derivative")


    # --------------------------------------------------------
    # Test 6: Neural network forward propagation
    # --------------------------------------------------------

    parameters_nn = initialize_network_parameters()

    state = np.array([0.4, -0.7])

    target = np.array([0.8, -0.2])

    output, _ = neural_network_forward(
        parameters_nn,
        state
    )

    assert output.shape == (OUTPUT_SIZE,)

    assert np.all(np.isfinite(output))

    print("[PASS] Neural network forward propagation")


    # --------------------------------------------------------
    # Test 7: Neural network backpropagation
    # --------------------------------------------------------

    analytical_gradient = neural_network_analytical_gradient(
        parameters_nn,
        state,
        target
    )

    numerical_gradient_value = numerical_gradient(
        lambda p: neural_network_loss(
            p,
            state,
            target
        ),
        parameters_nn,
        epsilon=1e-5
    )

    errors = gradient_error(
        analytical_gradient,
        numerical_gradient_value
    )

    print("\nNeural Network Gradient Verification")

    print(
        "Maximum absolute error:",
        errors["max_error"]
    )

    print(
        "Mean absolute error:",
        errors["mean_error"]
    )

    print(
        "Relative error:",
        errors["relative_error"]
    )

    # Central difference and analytical backpropagation
    # should agree closely for this smooth network.
    #
    # The tolerance is intentionally strict enough to detect
    # real backpropagation errors, while allowing floating-point
    # approximation error.

    assert np.allclose(
        analytical_gradient,
        numerical_gradient_value,
        atol=1e-7,
        rtol=1e-5
    ), (
        "Neural network backpropagation failed.\n"
        f"Maximum error: {errors['max_error']}"
    )

    print("[PASS] Neural network backpropagation")


    # --------------------------------------------------------
    # Test 8: Gradient shape and finite values
    # --------------------------------------------------------

    assert (
        analytical_gradient.shape ==
        numerical_gradient_value.shape
    )

    assert np.all(
        np.isfinite(analytical_gradient)
    )

    assert np.all(
        np.isfinite(numerical_gradient_value)
    )

    print("[PASS] Gradient shape and numerical stability")


    # --------------------------------------------------------
    # Test 9: Gradient descent convergence
    # --------------------------------------------------------

    history, objective = gradient_descent_demo()

    initial_loss = objective(history[0])

    final_loss = objective(history[-1])

    assert final_loss < initial_loss

    print("[PASS] Gradient descent convergence")


    print("\n================================")
    print("ALL UNIT TESTS PASSED!")
    print("================================\n")


# ============================================================
# 13. Visualization 1: Scalar Function
# ============================================================

def visualize_scalar_function():

    x = np.linspace(-2.0, 3.0, 300)

    y = scalar_function(x)

    plt.figure(figsize=(8, 5))

    plt.plot(
        x,
        y,
        label="f(x)"
    )

    plt.title("Scalar Function")

    plt.xlabel("x")

    plt.ylabel("f(x)")

    plt.grid(True)

    plt.legend()


# ============================================================
# 14. Visualization 2: Derivative Comparison
# ============================================================

def visualize_derivative_comparison():

    x_values = np.linspace(
        -2.0,
        3.0,
        100
    )

    analytical = []
    numerical = []

    for x in x_values:

        analytical.append(
            scalar_function_derivative(x)
        )

        numerical.append(
            central_difference(
                scalar_function,
                x
            )
        )

    plt.figure(figsize=(8, 5))

    plt.plot(
        x_values,
        analytical,
        label="Analytical derivative"
    )

    plt.plot(
        x_values,
        numerical,
        "--",
        label="Numerical derivative"
    )

    plt.title("Analytical vs Numerical Derivative")

    plt.xlabel("x")

    plt.ylabel("Derivative")

    plt.grid(True)

    plt.legend()


# ============================================================
# 15. Visualization 3: Epsilon Sensitivity
# ============================================================

def visualize_epsilon_sensitivity():

    epsilons, forward_errors, central_errors = (
        epsilon_sensitivity_experiment()
    )

    plt.figure(figsize=(8, 5))

    plt.loglog(
        epsilons,
        forward_errors,
        label="Forward difference"
    )

    plt.loglog(
        epsilons,
        central_errors,
        label="Central difference"
    )

    plt.xlabel("Epsilon")

    plt.ylabel("Absolute error")

    plt.title("Numerical Differentiation: Epsilon Sensitivity")

    plt.grid(True)

    plt.legend()


# ============================================================
# 16. Visualization 4: Jacobian
# ============================================================

def visualize_jacobian():

    x = np.array([1.5, 0.7])

    J = analytical_jacobian(x)

    plt.figure(figsize=(6, 5))

    plt.imshow(
        J,
        cmap="coolwarm",
        aspect="auto"
    )

    plt.colorbar(label="Derivative value")

    plt.xticks(
        [0, 1],
        ["x1", "x2"]
    )

    plt.yticks(
        [0, 1],
        ["f1", "f2"]
    )

    plt.title("Analytical Jacobian")

    for i in range(J.shape[0]):

        for j in range(J.shape[1]):

            plt.text(
                j,
                i,
                f"{J[i,j]:.4f}",
                ha="center",
                va="center"
            )


# ============================================================
# 17. Visualization 5: Neural Network Gradient
# ============================================================

def visualize_neural_network_gradient():

    parameters = initialize_network_parameters()

    state = np.array([0.4, -0.7])

    target = np.array([0.8, -0.2])

    analytical = neural_network_analytical_gradient(
        parameters,
        state,
        target
    )

    numerical = numerical_gradient(
        lambda p: neural_network_loss(
            p,
            state,
            target
        ),
        parameters,
        epsilon=1e-5
    )

    indices = np.arange(len(analytical))

    plt.figure(figsize=(11, 5))

    plt.plot(
        indices,
        analytical,
        "o-",
        label="Analytical"
    )

    plt.plot(
        indices,
        numerical,
        "x--",
        label="Numerical"
    )

    plt.title("Neural Network Gradient Verification")

    plt.xlabel("Parameter index")

    plt.ylabel("Gradient value")

    plt.grid(True)

    plt.legend()


# ============================================================
# 18. Visualization 6: Gradient Descent
# ============================================================

def visualize_gradient_descent():

    history, objective = gradient_descent_demo(
        learning_rate=0.05,
        iterations=100
    )

    x = np.linspace(-5.0, 5.0, 150)

    y = np.linspace(-5.0, 5.0, 150)

    X, Y = np.meshgrid(x, y)

    Z = (
        X**2 +
        3.0*X*Y +
        5.0*Y**2
    )

    plt.figure(figsize=(8, 6))

    plt.contour(
        X,
        Y,
        Z,
        levels=30
    )

    plt.plot(
        history[:, 0],
        history[:, 1],
        "o-",
        label="Gradient descent path"
    )

    plt.scatter(
        [0],
        [0],
        marker="*",
        s=180,
        label="Global minimum"
    )

    plt.xlabel("x")

    plt.ylabel("y")

    plt.title("Gradient Descent Optimization")

    plt.grid(True)

    plt.legend()


# ============================================================
# 19. Main
# ============================================================

def main():

    print("\n")
    print("========================================")
    print(" RL COURSE - PHASE 0.4")
    print(" NUMERICAL GRADIENTS")
    print("========================================")

    # 1. Run all unit tests.
    run_tests()

    # 2. Display derivative comparison.
    visualize_scalar_function()

    visualize_derivative_comparison()

    # 3. Display numerical stability experiment.
    visualize_epsilon_sensitivity()

    # 4. Display Jacobian.
    visualize_jacobian()

    # 5. Display neural network gradient verification.
    visualize_neural_network_gradient()

    # 6. Display gradient descent.
    visualize_gradient_descent()

    # All visualization functions are integrated
    # directly into this main script.

    plt.tight_layout()

    plt.show()


if __name__ == "__main__":

    main()
