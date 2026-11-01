
"""
Phase 0.2: Calculus for Reinforcement Learning
===============================================

Learning objectives:
1. Derivatives and partial derivatives.
2. Gradient vectors.
3. Gradient descent.
4. Chain rule and backpropagation.
5. Analytical gradients for a linear Q-network.
6. Numerical gradient verification.

Core operations are implemented manually.
NumPy is used for array storage and numerical calculations.
"""

import numpy as np


# ============================================================
# 1. Basic derivatives
# ============================================================

def derivative_power(x, n):
    """
    f(x) = x^n
    f'(x) = n * x^(n-1)
    """
    return n * (x ** (n - 1))


def partial_derivatives_example(x, y):
    """
    f(x, y) = x^2 + 3xy + y^2

    df/dx = 2x + 3y
    df/dy = 3x + 2y
    """
    df_dx = 2.0 * x + 3.0 * y
    df_dy = 3.0 * x + 2.0 * y

    return np.array([df_dx, df_dy], dtype=float)


# ============================================================
# 2. Numerical gradient
# ============================================================

def numerical_gradient(function, parameters, epsilon=1e-6):
    """
    Estimate the gradient using central finite differences.

    dL/dtheta_i approximately equals:

        [L(theta + epsilon) - L(theta - epsilon)]
        ------------------------------------------------
                          2 * epsilon

    Central differences are generally more accurate than
    forward differences for the same epsilon.
    """
    parameters = np.array(parameters, dtype=float)
    gradient = np.zeros_like(parameters)

    for i in range(parameters.size):
        plus = parameters.copy()
        minus = parameters.copy()

        plus[i] += epsilon
        minus[i] -= epsilon

        gradient[i] = (
            function(plus) - function(minus)
        ) / (2.0 * epsilon)

    return gradient


# ============================================================
# 3. Gradient descent
# ============================================================

def gradient_descent(
    initial_w,
    learning_rate=0.1,
    iterations=20
):
    """
    Minimize:

        L(w) = (w - 3)^2

    Analytical derivative:

        dL/dw = 2(w - 3)
    """
    w = float(initial_w)
    history = []

    for step in range(iterations):
        loss = (w - 3.0) ** 2
        gradient = 2.0 * (w - 3.0)

        history.append((step, w, loss, gradient))

        w = w - learning_rate * gradient

    return w, history


# ============================================================
# 4. Chain rule
# ============================================================

def chain_rule_example(w, x, b):
    """
    z = wx + b
    y = z^2

    dy/dw = (dy/dz) * (dz/dw)
          = 2z * x
    """
    z = w * x + b
    y = z ** 2

    dy_dz = 2.0 * z
    dz_dw = x

    dy_dw = dy_dz * dz_dw

    return y, dy_dw


# ============================================================
# 5. Linear Q-network
# ============================================================

def q_network(W, state, bias):
    """
    Q(s) = W @ s + bias

    W:     (num_actions, state_dim)
    state: (state_dim,)
    bias:  (num_actions,)

    Output: (num_actions,)
    """
    W = np.asarray(W, dtype=float)
    state = np.asarray(state, dtype=float)
    bias = np.asarray(bias, dtype=float)

    assert W.ndim == 2
    assert state.ndim == 1
    assert bias.ndim == 1

    assert W.shape[1] == state.shape[0]
    assert W.shape[0] == bias.shape[0]

    return W @ state + bias


# ============================================================
# 6. Q-network loss
# ============================================================

def selected_action_loss(W, bias, state, action, target):
    """
    L = 1/2 * (Q(s, action) - target)^2

    Only the selected action is used in this training example.
    """
    q_values = q_network(W, state, bias)
    error = q_values[action] - target

    return 0.5 * error ** 2


# ============================================================
# 7. Analytical Q-network gradients
# ============================================================

def q_network_gradients(W, bias, state, action, target):
    """
    For:

        Q(s, a) = W[a] @ state + bias[a]

        L = 1/2 * (Q(s, a) - target)^2

    Let:

        error = Q(s, a) - target

    Then:

        dL/dW[a] = error * state
        dL/db[a] = error

    All other action rows have zero gradient.
    """
    W = np.asarray(W, dtype=float)
    bias = np.asarray(bias, dtype=float)
    state = np.asarray(state, dtype=float)

    q_values = q_network(W, state, bias)
    error = q_values[action] - target

    grad_W = np.zeros_like(W)
    grad_bias = np.zeros_like(bias)

    grad_W[action] = error * state
    grad_bias[action] = error

    return grad_W, grad_bias


# ============================================================
# 8. Q-network parameter update
# ============================================================

def update_q_network(
    W,
    bias,
    state,
    action,
    target,
    learning_rate=0.01
):
    """
    One gradient-descent update.
    """
    grad_W, grad_bias = q_network_gradients(
        W, bias, state, action, target
    )

    new_W = W - learning_rate * grad_W
    new_bias = bias - learning_rate * grad_bias

    return new_W, new_bias


# ============================================================
# 9. Numerical verification of Q-network gradients
# ============================================================

def verify_q_network_gradients():
    """
    Compare analytical gradients with numerical gradients.
    """
    W = np.array([
        [0.2, 0.1, 0.4],
        [0.5, 0.2, 0.1],
        [0.1, 0.3, 0.2]
    ], dtype=float)

    bias = np.array([0.1, 0.0, 0.2])
    state = np.array([0.2, 0.5, 0.8])

    action = 1
    target = 0.8

    analytical_W, analytical_bias = q_network_gradients(
        W, bias, state, action, target
    )

    # Flatten W and bias into one parameter vector.
    parameters = np.concatenate([W.flatten(), bias])

    def loss_from_flat_parameters(flat_parameters):
        weight_count = W.size

        test_W = flat_parameters[:weight_count].reshape(W.shape)
        test_bias = flat_parameters[weight_count:]

        return selected_action_loss(
            test_W,
            test_bias,
            state,
            action,
            target
        )

    numerical = numerical_gradient(
        loss_from_flat_parameters,
        parameters
    )

    analytical = np.concatenate([
        analytical_W.flatten(),
        analytical_bias
    ])

    max_error = np.max(np.abs(analytical - numerical))

    print("\nGradient verification")
    print("---------------------")
    print("Analytical gradient:")
    print(analytical)

    print("\nNumerical gradient:")
    print(numerical)

    print("\nMaximum absolute error:", max_error)

    assert np.allclose(
        analytical,
        numerical,
        atol=1e-6,
        rtol=1e-5
    )

    print("PASS: analytical and numerical gradients match.")


# ============================================================
# 10. Unit tests
# ============================================================

def run_tests():
    # Derivative
    assert derivative_power(3.0, 2) == 6.0

    # Partial derivatives
    gradient = partial_derivatives_example(2.0, 1.0)
    assert np.allclose(gradient, [7.0, 8.0])

    # Gradient descent
    final_w, history = gradient_descent(
        initial_w=0.0,
        learning_rate=0.1,
        iterations=50
    )
    assert abs(final_w - 3.0) < 1e-4

    # Chain rule
    w, x, b = 2.0, 3.0, 1.0
    y, analytical = chain_rule_example(w, x, b)

    epsilon = 1e-6
    y_plus = ((w + epsilon) * x + b) ** 2
    y_minus = ((w - epsilon) * x + b) ** 2

    numerical = (y_plus - y_minus) / (2.0 * epsilon)

    assert np.isclose(analytical, numerical, atol=1e-6)

    # Q-network dimensions and output
    W = np.array([
        [0.2, 0.1, 0.4],
        [0.5, 0.2, 0.1],
        [0.1, 0.3, 0.2]
    ])

    state = np.array([0.2, 0.5, 0.8])
    bias = np.array([0.1, 0.0, 0.2])

    q_values = q_network(W, state, bias)
    assert q_values.shape == (3,)

    # Gradient verification
    verify_q_network_gradients()

    print("\nAll tests passed.")


# ============================================================
# 11. Demonstration
# ============================================================

def main():
    print("PHASE 0.2: CALCULUS FOR REINFORCEMENT LEARNING")

    print("\n1. Partial derivatives")
    print(partial_derivatives_example(2.0, 1.0))

    print("\n2. Gradient descent")
    final_w, history = gradient_descent(
        initial_w=0.0,
        learning_rate=0.1,
        iterations=10
    )

    for step, w, loss, gradient in history:
        print(
            f"Step {step:2d}: "
            f"w={w:.6f}, "
            f"loss={loss:.6f}, "
            f"gradient={gradient:.6f}"
        )

    print("Final w after 10 updates:", final_w)

    print("\n3. Chain rule")
    y, dy_dw = chain_rule_example(
        w=2.0,
        x=3.0,
        b=1.0
    )
    print("Output y:", y)
    print("Analytical dy/dw:", dy_dw)

    print("\n4. Q-network update")
    W = np.array([
        [0.2, 0.1, 0.4],
        [0.5, 0.2, 0.1],
        [0.1, 0.3, 0.2]
    ])

    state = np.array([0.2, 0.5, 0.8])
    bias = np.array([0.1, 0.0, 0.2])

    action = 1
    target = 0.8

    before = q_network(W, state, bias)
    loss_before = selected_action_loss(
        W, bias, state, action, target
    )

    new_W, new_bias = update_q_network(
        W,
        bias,
        state,
        action,
        target,
        learning_rate=0.1
    )

    after = q_network(new_W, state, new_bias)
    loss_after = selected_action_loss(
        new_W, new_bias, state, action, target
    )

    print("Q-values before:", before)
    print("Q-values after: ", after)
    print("Loss before:", loss_before)
    print("Loss after: ", loss_after)

    print("\n5. Unit tests")
    run_tests()


if __name__ == "__main__":
    main()
