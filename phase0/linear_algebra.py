
# =============================================================================
# phase0/linear_algebra.py
# =============================================================================
#
# WHAT THIS FILE TEACHES
#
#   1. Vector operations and dot products.
#   2. Matrix multiplication implemented using Python loops.
#   3. Matrix transpose and identity matrices.
#   4. Linear transformations: y = Wx + b.
#   5. Numerical Jacobian calculation.
#   6. Analytical vs numerical Jacobian verification.
#   7. Matrix shape validation.
#
# WHY IT MATTERS FOR REINFORCEMENT LEARNING
#
#   - An RL agent represents its state using vectors.
#   - Neural networks use matrix multiplication.
#   - Q-Networks transform states into action-value predictions.
#   - Policy networks transform states into action distributions.
#   - Backpropagation uses derivatives and Jacobians.
#
# REAL-WORLD APPLICATION
#
#   A robot receives a state vector containing:
#
#       joint_position
#       joint_velocity
#       distance_to_target
#       object_position
#
#   A linear layer transforms this state into action-value predictions.
#
#       Q(s) = Ws + b
#
#   The agent can then select the action with the highest predicted value.
#
# REQUIREMENTS
#
#   pip install numpy
#
# EXECUTION
#
#   python phase0/linear_algebra.py
#
# =============================================================================

import numpy as np


# -----------------------------------------------------------------------------
# 1. VECTOR OPERATIONS
# -----------------------------------------------------------------------------

def vector_add(a, b):
    """Add two vectors of identical shape."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    assert a.shape == b.shape, "Vector shape mismatch"

    result = np.zeros_like(a)

    for i in range(a.size):
        result[i] = a[i] + b[i]

    return result


def vector_subtract(a, b):
    """Subtract vector b from vector a."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    assert a.shape == b.shape, "Vector shape mismatch"

    result = np.zeros_like(a)

    for i in range(a.size):
        result[i] = a[i] - b[i]

    return result


def dot_product(a, b):
    """
    Calculate the dot product.

    a: (n,)
    b: (n,)

    output: scalar

    dot(a,b) = sum(a[i] * b[i])
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    assert a.ndim == 1 and b.ndim == 1
    assert a.shape == b.shape, "Vector shape mismatch"

    result = 0.0

    for i in range(a.size):
        result += a[i] * b[i]

    return result


# -----------------------------------------------------------------------------
# 2. MATRIX MULTIPLICATION FROM SCRATCH
# -----------------------------------------------------------------------------

def matmul_naive(A, B):
    """
    Matrix multiplication using three Python loops.

    A: (n, m)
    B: (m, p)

    output: (n, p)

    Complexity:
        O(n * m * p)
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)

    assert A.ndim == 2
    assert B.ndim == 2

    n, m = A.shape
    m2, p = B.shape

    assert m == m2, "Matrix dimension mismatch"

    C = np.zeros((n, p), dtype=np.float64)

    for i in range(n):
        for j in range(p):
            for k in range(m):
                C[i, j] += A[i, k] * B[k, j]

    return C


# -----------------------------------------------------------------------------
# 3. MATRIX TRANSPOSE
# -----------------------------------------------------------------------------

def matrix_transpose(A):
    """
    Transpose a matrix.

    A: (n, m)
    output: (m, n)

    A_transpose[j, i] = A[i, j]
    """
    A = np.asarray(A, dtype=np.float64)

    assert A.ndim == 2

    n, m = A.shape

    result = np.zeros((m, n), dtype=np.float64)

    for i in range(n):
        for j in range(m):
            result[j, i] = A[i, j]

    return result


# -----------------------------------------------------------------------------
# 4. IDENTITY MATRIX
# -----------------------------------------------------------------------------

def identity_matrix(n):
    """
    Create an n x n identity matrix.

    I[i, j] = 1 if i == j
               0 otherwise

    Property:
        A @ I = A
        I @ A = A
    """
    assert n > 0

    I = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        I[i, i] = 1.0

    return I


# -----------------------------------------------------------------------------
# 5. LINEAR TRANSFORMATION
# -----------------------------------------------------------------------------

def linear_layer(W, x, b):
    """
    Implement a single linear neural-network layer.

        y = Wx + b

    W: (output_features, input_features)
    x: (input_features,)
    b: (output_features,)

    output: (output_features,)
    """
    W = np.asarray(W, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    assert W.ndim == 2
    assert x.ndim == 1
    assert b.ndim == 1

    output_features, input_features = W.shape

    assert x.shape == (input_features,)
    assert b.shape == (output_features,)

    y = np.zeros(output_features, dtype=np.float64)

    for i in range(output_features):
        for j in range(input_features):
            y[i] += W[i, j] * x[j]

        y[i] += b[i]

    return y


# -----------------------------------------------------------------------------
# 6. NUMERICAL JACOBIAN
# -----------------------------------------------------------------------------

def numerical_jacobian(f, x, eps=1e-6):
    """
    Calculate Jacobian using central finite differences.

    f: R^n -> R^m

    Returns:
        J with shape (m, n)

    J[i, j] = df_i / dx_j
    """
    x = np.asarray(x, dtype=np.float64)

    f0 = np.asarray(f(x), dtype=np.float64).ravel()

    m = f0.size
    n = x.size

    J = np.zeros((m, n), dtype=np.float64)

    for j in range(n):

        xp = x.copy()
        xm = x.copy()

        xp[j] += eps
        xm[j] -= eps

        fp = np.asarray(f(xp), dtype=np.float64).ravel()
        fm = np.asarray(f(xm), dtype=np.float64).ravel()

        J[:, j] = (fp - fm) / (2.0 * eps)

    return J


# -----------------------------------------------------------------------------
# 7. ANALYTICAL JACOBIAN OF A LINEAR LAYER
# -----------------------------------------------------------------------------

def linear_layer_jacobian(W):
    """
    For:

        y = Wx + b

    The Jacobian is:

        dy/dx = W

    Therefore, the analytical Jacobian is exactly W.
    """
    return np.asarray(W, dtype=np.float64)


# -----------------------------------------------------------------------------
# 8. REINFORCEMENT LEARNING EXAMPLE
# -----------------------------------------------------------------------------

def rl_q_value_demo():
    """
    A simplified Q-network linear layer.

    State:
        4 input features.

    Actions:
        3 possible actions.

    Output:
        3 action-value predictions.

    The values are illustrative, not learned Q-values.
    """

    # Robot state:
    #
    # [joint_position,
    #  joint_velocity,
    #  distance_to_target,
    #  object_position]

    state = np.array([
        0.2,
        0.5,
        0.8,
        0.1
    ])

    # Three actions:
    #
    # action 0: move left
    # action 1: move right
    # action 2: stay

    W = np.array([
        [0.2, 0.1, 0.4, 0.3],
        [0.5, 0.2, 0.1, 0.2],
        [0.1, 0.3, 0.2, 0.4]
    ])

    b = np.array([
        0.1,
        0.0,
        0.2
    ])

    q_values = linear_layer(W, state, b)

    print("\n--- RL Q-value demonstration ---")

    print("State:")
    print(state)

    print("\nQ-values:")
    print(q_values)

    selected_action = np.argmax(q_values)

    action_names = [
        "move left",
        "move right",
        "stay"
    ]

    print("\nSelected action:")
    print(action_names[selected_action])

    return q_values


# -----------------------------------------------------------------------------
# 9. JACOBIAN VERIFICATION
# -----------------------------------------------------------------------------

def jacobian_demo():

    W = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0]
    ])

    b = np.array([0.1, 0.2, 0.3])

    x = np.array([0.5, 0.8])

    def f(x):
        return linear_layer(W, x, b)

    J_analytic = linear_layer_jacobian(W)

    J_numeric = numerical_jacobian(f, x)

    print("\n--- Jacobian verification ---")

    print("Analytical Jacobian:")
    print(J_analytic)

    print("\nNumerical Jacobian:")
    print(J_numeric)

    error = np.max(np.abs(J_analytic - J_numeric))

    print("\nMaximum Jacobian error:")
    print(error)

    assert np.allclose(
        J_analytic,
        J_numeric,
        atol=1e-6
    ), "Jacobian verification failed"


# -----------------------------------------------------------------------------
# 10. UNIT TESTS
# -----------------------------------------------------------------------------

def run_tests():

    print("\n--- Running unit tests ---")

    # Vector addition
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])

    assert np.allclose(
        vector_add(a, b),
        np.array([5, 7, 9])
    )

    # Dot product
    assert np.isclose(dot_product(a, b), 32.0)

    # Matrix multiplication
    A = np.array([
        [1, 2],
        [3, 4]
    ])

    B = np.array([
        [5, 6],
        [7, 8]
    ])

    expected = np.array([
        [19, 22],
        [43, 50]
    ])

    assert np.allclose(matmul_naive(A, B), expected)

    # Compare with NumPy
    rng = np.random.default_rng(0)

    A = rng.standard_normal((3, 4))
    B = rng.standard_normal((4, 5))

    assert np.allclose(
        matmul_naive(A, B),
        A @ B
    )

    # Matrix transpose
    A = np.array([
        [1, 2, 3],
        [4, 5, 6]
    ])

    assert np.allclose(
        matrix_transpose(A),
        A.T
    )

    # Identity matrix
    I = identity_matrix(3)

    assert np.allclose(I, np.eye(3))

    # Linear layer
    W = np.array([
        [1, 2],
        [3, 4]
    ])

    x = np.array([5, 6])

    bias = np.array([1, 1])

    result = linear_layer(W, x, bias)

    assert np.allclose(result, np.array([18, 40]))

    print("All unit tests passed!")


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

if __name__ == "__main__":

    run_tests()

    rl_q_value_demo()

    jacobian_demo()

    print("\nPhase 0.1 completed.")
