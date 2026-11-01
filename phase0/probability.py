
"""
Phase 0.3: Probability and Statistics for Reinforcement Learning
================================================================

Purpose
-------
Build the probability and statistics foundation required for RL.

Topics:
1. Random variables and probability distributions.
2. Conditional probability.
3. Expected value and variance.
4. Categorical distributions.
5. Gaussian distributions.
6. Sampling and Monte Carlo estimation.
7. Entropy and exploration.
8. Stochastic policies.

Dependencies:
    pip install numpy matplotlib

Run:
    python phase0/probability.py

All mathematical explanations and derivations are included in comments.
Core probability operations are implemented manually where practical.
NumPy is used for arrays and random-number generation.
"""


import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. Probability distribution utilities
# ============================================================

def normalize_probabilities(probabilities):
    """
    Convert non-negative values into a valid probability distribution.

    Mathematical requirements:

        p_i >= 0
        sum(p_i) = 1

    Normalization:

        p_i = x_i / sum(x)

    Example:
        [2, 3, 5] -> [0.2, 0.3, 0.5]
    """
    probabilities = np.asarray(probabilities, dtype=float)

    if probabilities.ndim != 1:
        raise ValueError("Probabilities must be a 1D array.")

    if np.any(probabilities < 0):
        raise ValueError("Probabilities cannot be negative.")

    total = np.sum(probabilities)

    if total <= 0:
        raise ValueError("Probability sum must be positive.")

    return probabilities / total


def validate_distribution(probabilities, tolerance=1e-8):
    """Check whether an array is a valid discrete probability distribution."""
    probabilities = np.asarray(probabilities, dtype=float)

    return (
        probabilities.ndim == 1
        and np.all(probabilities >= 0)
        and np.isclose(np.sum(probabilities), 1.0, atol=tolerance)
    )


# ============================================================
# 2. Expected value and variance
# ============================================================

def expected_value(values, probabilities):
    """
    Calculate the expected value of a discrete random variable.

    E[X] = sum(x_i * p_i)

    In RL:
        Expected reward = sum(reward_i * probability_i)

    Expected value is an average over possible outcomes,
    weighted by their probabilities.
    """
    values = np.asarray(values, dtype=float)
    probabilities = normalize_probabilities(probabilities)

    if values.shape != probabilities.shape:
        raise ValueError("Values and probabilities must have equal shapes.")

    return np.sum(values * probabilities)


def variance(values, probabilities):
    """
    Calculate the variance of a discrete random variable.

    Var(X) = E[(X - E[X])^2]

    Equivalent formula:

        Var(X) = E[X^2] - (E[X])^2

    Variance measures uncertainty or dispersion.
    """
    values = np.asarray(values, dtype=float)
    probabilities = normalize_probabilities(probabilities)

    mean = expected_value(values, probabilities)

    return np.sum(probabilities * (values - mean) ** 2)


def standard_deviation(values, probabilities):
    """Standard deviation is the square root of variance."""
    return np.sqrt(variance(values, probabilities))


# ============================================================
# 3. Conditional probability
# ============================================================

def conditional_probability(p_a_and_b, p_b):
    """
    Calculate:

        P(A | B) = P(A and B) / P(B)

    P(A | B) means the probability of A occurring
    given that B has occurred.

    RL example:
        A = robot successfully grasps an object.
        B = robot approaches the object from the correct direction.

    This function expects valid probabilities.
    """
    if not 0.0 <= p_a_and_b <= 1.0:
        raise ValueError("Joint probability must be between 0 and 1.")

    if not 0.0 < p_b <= 1.0:
        raise ValueError("Condition probability must be in (0, 1].")

    if p_a_and_b > p_b:
        raise ValueError("P(A and B) cannot exceed P(B).")

    return p_a_and_b / p_b


# ============================================================
# 4. Softmax and categorical distribution
# ============================================================

def softmax(logits, temperature=1.0):
    """
    Convert action scores (logits) into probabilities.

    Softmax:

        P(a_i) = exp(z_i / T) / sum_j exp(z_j / T)

    z_i = score for action i
    T   = temperature

    Numerical stability:
        exp(z_i - max(z)) is mathematically equivalent after
        normalization, but avoids unnecessarily large exponentials.

    Temperature behavior:
        T -> 0: distribution becomes concentrated on the largest logit.
        T -> infinity: distribution approaches uniform.

    In RL:
        Softmax can convert policy-network outputs into
        probabilities for stochastic action selection.
    """
    logits = np.asarray(logits, dtype=float)

    if logits.ndim != 1:
        raise ValueError("Logits must be a 1D array.")

    if temperature <= 0:
        raise ValueError("Temperature must be positive.")

    scaled = logits / temperature
    shifted = scaled - np.max(scaled)

    exponentials = np.exp(shifted)

    return exponentials / np.sum(exponentials)


def categorical_sample(probabilities, rng=None):
    """
    Sample an action index from a categorical distribution.

    Example:
        probabilities = [0.1, 0.7, 0.2]

        Action 0 has probability 0.1.
        Action 1 has probability 0.7.
        Action 2 has probability 0.2.

    Unlike argmax, sampling does not always select the
    action with the largest probability.
    """
    probabilities = normalize_probabilities(probabilities)

    if rng is None:
        rng = np.random.default_rng()

    return int(rng.choice(len(probabilities), p=probabilities))


def categorical_entropy(probabilities):
    """
    Calculate Shannon entropy:

        H(P) = -sum_i p_i * log(p_i)

    A probability of zero contributes zero to entropy.

    Interpretation:
        Low entropy  -> policy is concentrated on a few actions.
        High entropy -> policy is more spread across actions.

    For a fixed number of actions, maximum entropy occurs
    for the uniform distribution:

        H_max = log(number_of_actions)

    Entropy is important for exploration in algorithms such as
    PPO and SAC.
    """
    probabilities = normalize_probabilities(probabilities)

    nonzero = probabilities > 0

    return -np.sum(
        probabilities[nonzero] * np.log(probabilities[nonzero])
    )


# ============================================================
# 5. Gaussian distribution
# ============================================================

def gaussian_sample(mean, std, sample_count=1, rng=None):
    """
    Generate samples from a Gaussian distribution.

        X ~ N(mu, sigma^2)

    mu    = mean
    sigma = standard deviation

    In continuous-action RL:
        A policy may output the mean and standard deviation
        of a Gaussian distribution over robot actions.

    Example:
        The policy outputs a distribution over robot joint velocity.
    """
    if std <= 0:
        raise ValueError("Standard deviation must be positive.")

    if sample_count <= 0:
        raise ValueError("Sample count must be positive.")

    if rng is None:
        rng = np.random.default_rng()

    return rng.normal(
        loc=mean,
        scale=std,
        size=sample_count
    )


def gaussian_probability_density(x, mean, std):
    """
    Gaussian probability density function:

        p(x) = 1 / (sigma * sqrt(2*pi))
               * exp(-(x-mu)^2 / (2*sigma^2))

    Note:
        A probability density is not itself the probability
        of an exact continuous value. Probabilities are
        obtained by integrating density over an interval.
    """
    if std <= 0:
        raise ValueError("Standard deviation must be positive.")

    coefficient = 1.0 / (std * np.sqrt(2.0 * np.pi))

    exponent = -((x - mean) ** 2) / (2.0 * std ** 2)

    return coefficient * np.exp(exponent)


def gaussian_log_probability(x, mean, std):
    """
    Log probability density of a Gaussian distribution.

        log p(x) =
            -0.5 * log(2*pi)
            - log(sigma)
            - (x-mu)^2 / (2*sigma^2)

    Log probabilities are commonly used in policy-gradient
    algorithms because sums of log probabilities are easier
    and more numerically stable than products of probabilities.
    """
    if std <= 0:
        raise ValueError("Standard deviation must be positive.")

    return (
        -0.5 * np.log(2.0 * np.pi)
        - np.log(std)
        - ((x - mean) ** 2) / (2.0 * std ** 2)
    )


def gaussian_entropy(std):
    """
    Differential entropy of a one-dimensional Gaussian:

        H = 0.5 * log(2*pi*e*sigma^2)

    Increasing sigma increases entropy.

    In continuous-action RL, policy entropy can encourage
    the agent to retain action diversity during learning.
    """
    if std <= 0:
        raise ValueError("Standard deviation must be positive.")

    return 0.5 * np.log(
        2.0 * np.pi * np.e * std ** 2
    )


# ============================================================
# 6. Monte Carlo estimation
# ============================================================

def monte_carlo_mean(samples):
    """
    Estimate expected value using sampled observations.

        E[X] approximately equals (1/N) * sum(x_i)

    The law of large numbers says that, under suitable
    conditions, the sample average approaches the true
    expected value as N increases.
    """
    samples = np.asarray(samples, dtype=float)

    if samples.size == 0:
        raise ValueError("Samples cannot be empty.")

    return np.mean(samples)


def monte_carlo_convergence(
    true_mean,
    sample_count=10000,
    rng=None
):
    """
    Demonstrate convergence of the sample mean.

    Returns:
        samples
        running_means

    The running mean after N samples is:

        M_N = (X_1 + ... + X_N) / N

    In RL:
        Monte Carlo methods estimate returns and values
        using sampled episodes or trajectories.
    """
    if rng is None:
        rng = np.random.default_rng()

    samples = rng.normal(
        loc=true_mean,
        scale=1.0,
        size=sample_count
    )

    cumulative_sum = np.cumsum(samples)

    sample_numbers = np.arange(1, sample_count + 1)

    running_means = cumulative_sum / sample_numbers

    return samples, running_means


# ============================================================
# 7. Numerical verification
# ============================================================

def verify_expected_value():
    """
    Verify the expected value with a large Monte Carlo sample.
    """
    values = np.array([1.0, 2.0, 5.0])
    probabilities = np.array([0.2, 0.5, 0.3])

    theoretical = expected_value(values, probabilities)

    rng = np.random.default_rng(42)

    sampled_indices = rng.choice(
        len(values),
        size=200000,
        p=probabilities
    )

    sampled_values = values[sampled_indices]

    estimated = monte_carlo_mean(sampled_values)

    print("\nExpected value verification")
    print("---------------------------")
    print("Theoretical:", theoretical)
    print("Monte Carlo:", estimated)
    print("Absolute error:", abs(theoretical - estimated))

    assert abs(theoretical - estimated) < 0.03


def verify_gaussian_statistics():
    """
    Verify Gaussian sample mean and variance.

    For X ~ N(mu, sigma^2):

        E[X] = mu
        Var(X) = sigma^2
    """
    mean = 2.0
    std = 3.0

    rng = np.random.default_rng(123)

    samples = gaussian_sample(
        mean,
        std,
        sample_count=200000,
        rng=rng
    )

    estimated_mean = np.mean(samples)
    estimated_variance = np.var(samples)

    print("\nGaussian verification")
    print("---------------------")
    print("Theoretical mean:", mean)
    print("Estimated mean:", estimated_mean)
    print("Theoretical variance:", std ** 2)
    print("Estimated variance:", estimated_variance)

    assert abs(estimated_mean - mean) < 0.05
    assert abs(estimated_variance - std ** 2) < 0.15


def verify_categorical_sampling():
    """
    Verify that empirical action frequencies approximate
    the specified categorical probabilities.
    """
    probabilities = np.array([0.1, 0.7, 0.2])

    rng = np.random.default_rng(42)

    samples = [
        categorical_sample(probabilities, rng)
        for _ in range(100000)
    ]

    frequencies = np.bincount(
        samples,
        minlength=len(probabilities)
    ) / len(samples)

    print("\nCategorical sampling verification")
    print("---------------------------------")
    print("Expected probabilities:", probabilities)
    print("Observed frequencies: ", frequencies)

    assert np.allclose(
        frequencies,
        probabilities,
        atol=0.01
    )


# ============================================================
# 8. RL policy demonstration
# ============================================================

def rl_policy_demo():
    """
    Demonstrate a stochastic policy for a robot.

    Suppose the robot has three possible discrete actions:

        0 = Move Left
        1 = Move Right
        2 = Stay

    The policy network produces logits.

    Softmax converts logits into action probabilities.
    The agent samples an action from this distribution.

    This differs from greedy action selection:

        Greedy: always select argmax(probabilities).
        Stochastic: sample according to probabilities.
    """
    logits = np.array([1.0, 2.0, 0.5])

    action_names = [
        "Move Left",
        "Move Right",
        "Stay"
    ]

    rng = np.random.default_rng(42)

    print("\nRL stochastic policy demonstration")
    print("----------------------------------")

    for temperature in [0.3, 1.0, 2.0]:
        probabilities = softmax(
            logits,
            temperature=temperature
        )

        entropy = categorical_entropy(probabilities)

        selected_action = categorical_sample(
            probabilities,
            rng
        )

        print(f"\nTemperature: {temperature}")
        print("Probabilities:", probabilities)
        print("Entropy:", entropy)
        print("Sampled action:", action_names[selected_action])


# ============================================================
# 9. Visualization functions
# ============================================================

def visualize_discrete_distribution():
    """
    Visualize a categorical policy.

    The bar heights represent action probabilities.
    """
    actions = [
        "Move Left",
        "Move Right",
        "Stay"
    ]

    logits = np.array([1.0, 2.0, 0.5])

    probabilities = softmax(logits, temperature=1.0)

    plt.figure(figsize=(8, 5))
    plt.bar(actions, probabilities)
    plt.ylim(0, 1)
    plt.ylabel("Probability")
    plt.xlabel("Robot Action")
    plt.title("Categorical Policy Distribution")
    plt.grid(axis="y")
    plt.tight_layout()


def visualize_gaussian_distribution():
    """
    Visualize Gaussian policies with different standard deviations.

    A larger standard deviation means more dispersed actions.
    """
    x = np.linspace(-6, 6, 500)

    mean = 0.0
    standard_deviations = [0.5, 1.0, 2.0]

    plt.figure(figsize=(8, 5))

    for std in standard_deviations:
        density = gaussian_probability_density(
            x,
            mean,
            std
        )

        plt.plot(
            x,
            density,
            label=f"Std = {std}"
        )

    plt.xlabel("Action Value")
    plt.ylabel("Probability Density")
    plt.title("Gaussian Policy Distributions")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()


def visualize_monte_carlo_convergence():
    """
    Plot the running sample mean against the true expected value.

    This demonstrates the law of large numbers.
    """
    true_mean = 3.0

    rng = np.random.default_rng(42)

    samples, running_means = monte_carlo_convergence(
        true_mean=true_mean,
        sample_count=10000,
        rng=rng
    )

    plt.figure(figsize=(9, 5))

    plt.plot(
        np.arange(1, len(samples) + 1),
        running_means,
        label="Estimated mean"
    )

    plt.axhline(
        y=true_mean,
        linestyle="--",
        label="True mean"
    )

    plt.xlabel("Number of Samples")
    plt.ylabel("Estimated Expected Value")
    plt.title("Monte Carlo Estimation Convergence")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()


def visualize_entropy_temperature():
    """
    Demonstrate the effect of softmax temperature on entropy.

    Low temperature:
        Concentrated action probabilities and low entropy.

    High temperature:
        More uniform probabilities and higher entropy.

    For non-identical finite logits, entropy approaches the
    maximum log(number of actions) as temperature increases.
    """
    logits = np.array([1.0, 2.0, 0.5])

    temperatures = np.linspace(0.1, 5.0, 100)

    entropies = []

    for temperature in temperatures:
        probabilities = softmax(
            logits,
            temperature=temperature
        )

        entropies.append(
            categorical_entropy(probabilities)
        )

    plt.figure(figsize=(8, 5))

    plt.plot(temperatures, entropies)

    plt.axhline(
        y=np.log(len(logits)),
        linestyle="--",
        label="Maximum entropy"
    )

    plt.xlabel("Softmax Temperature")
    plt.ylabel("Entropy")
    plt.title("Policy Entropy vs Temperature")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()


# ============================================================
# 10. Unit tests
# ============================================================

def run_tests():
    # Probability normalization
    p = normalize_probabilities([2, 3, 5])
    assert np.allclose(p, [0.2, 0.3, 0.5])
    assert validate_distribution(p)

    # Expected value
    values = np.array([1, 2, 5])
    probabilities = np.array([0.2, 0.5, 0.3])

    assert np.isclose(
        expected_value(values, probabilities),
        2.7
    )

    # Variance
    expected_variance = (
        0.2 * (1 - 2.7) ** 2
        + 0.5 * (2 - 2.7) ** 2
        + 0.3 * (5 - 2.7) ** 2
    )

    assert np.isclose(
        variance(values, probabilities),
        expected_variance
    )

    # Conditional probability
    assert np.isclose(
        conditional_probability(0.2, 0.5),
        0.4
    )

    # Softmax
    p = softmax([1.0, 2.0, 0.5])
    assert validate_distribution(p)

    # Higher temperature should increase entropy
    low_entropy = categorical_entropy(
        softmax([1.0, 2.0, 0.5], temperature=0.2)
    )

    high_entropy = categorical_entropy(
        softmax([1.0, 2.0, 0.5], temperature=5.0)
    )

    assert high_entropy > low_entropy

    # Gaussian density is largest at its mean
    density_at_mean = gaussian_probability_density(
        0.0, 0.0, 1.0
    )

    density_away_from_mean = gaussian_probability_density(
        2.0, 0.0, 1.0
    )

    assert density_at_mean > density_away_from_mean

    # Gaussian entropy increases with standard deviation
    assert gaussian_entropy(2.0) > gaussian_entropy(1.0)

    # Log probability should equal logarithm of density
    x = 1.5
    mean = 0.0
    std = 2.0

    assert np.isclose(
        gaussian_log_probability(x, mean, std),
        np.log(gaussian_probability_density(x, mean, std))
    )

    print("\nAll unit tests passed.")


# ============================================================
# 11. Main
# ============================================================

def main():
    print("=" * 65)
    print("PHASE 0.3: PROBABILITY AND STATISTICS FOR REINFORCEMENT LEARNING")
    print("=" * 65)

    # --------------------------------------------------------
    # A. Discrete random variable
    # --------------------------------------------------------

    values = np.array([1.0, 2.0, 5.0])
    probabilities = np.array([0.2, 0.5, 0.3])

    print("\n1. Discrete random variable")
    print("Values:", values)
    print("Probabilities:", probabilities)
    print("Expected value:", expected_value(values, probabilities))
    print("Variance:", variance(values, probabilities))
    print(
        "Standard deviation:",
        standard_deviation(values, probabilities)
    )

    # --------------------------------------------------------
    # B. Conditional probability
    # --------------------------------------------------------

    print("\n2. Conditional probability")

    p_a_and_b = 0.24
    p_b = 0.60

    print(
        "P(A | B):",
        conditional_probability(p_a_and_b, p_b)
    )

    # --------------------------------------------------------
    # C. Categorical distribution
    # --------------------------------------------------------

    print("\n3. Categorical distribution")

    logits = np.array([1.0, 2.0, 0.5])
    action_probabilities = softmax(logits)

    print("Logits:", logits)
    print("Action probabilities:", action_probabilities)
    print("Entropy:", categorical_entropy(action_probabilities))

    # --------------------------------------------------------
    # D. Gaussian distribution
    # --------------------------------------------------------

    print("\n4. Gaussian distribution")

    mean = 0.0
    std = 1.5

    print("Mean:", mean)
    print("Standard deviation:", std)
    print("Entropy:", gaussian_entropy(std))
    print(
        "Density at mean:",
        gaussian_probability_density(mean, mean, std)
    )

    # --------------------------------------------------------
    # E. Monte Carlo estimation
    # --------------------------------------------------------

    print("\n5. Monte Carlo estimation")

    rng = np.random.default_rng(42)

    samples, running_means = monte_carlo_convergence(
        true_mean=3.0,
        sample_count=10000,
        rng=rng
    )

    print("True mean:", 3.0)
    print("Estimated mean:", monte_carlo_mean(samples))
    print("Estimate after 10 samples:", running_means[9])
    print("Estimate after 100 samples:", running_means[99])
    print("Estimate after 10000 samples:", running_means[-1])

    # --------------------------------------------------------
    # F. RL stochastic policy
    # --------------------------------------------------------

    print("\n6. RL policy")
    rl_policy_demo()

    # --------------------------------------------------------
    # G. Numerical verification
    # --------------------------------------------------------

    print("\n7. Numerical verification")

    verify_expected_value()
    verify_gaussian_statistics()
    verify_categorical_sampling()

    # --------------------------------------------------------
    # H. Unit tests
    # --------------------------------------------------------

    print("\n8. Unit tests")
    run_tests()

    # --------------------------------------------------------
    # I. Visualization
    # --------------------------------------------------------
    # All visualizations are integrated into the main script.
    # Set show_plots=False when running automated tests or
    # a headless server.

    print("\n9. Generating visualizations")

    visualize_discrete_distribution()
    visualize_gaussian_distribution()
    visualize_monte_carlo_convergence()
    visualize_entropy_temperature()

    plt.show()

    print("\nPhase 0.3 completed successfully.")


if __name__ == "__main__":
    main()
