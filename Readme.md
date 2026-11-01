# Reinforcement Learning (RL) — Complete Learning Roadmap

## 1. Course Objective

The objective of this course is to develop a deep understanding of Reinforcement Learning (RL), starting from mathematical foundations and progressing to advanced Deep Reinforcement Learning and Robotics.

The course emphasizes understanding algorithms internally rather than simply using existing libraries.

### Final Learning Goals

By the end of this course, I should be able to:

- Understand the mathematical foundations of Reinforcement Learning.
- Derive important RL algorithms mathematically.
- Implement RL algorithms from scratch using Python and NumPy.
- Understand and implement Deep Reinforcement Learning using PyTorch.
- Visualize agent learning processes and algorithm behavior.
- Evaluate algorithm performance experimentally.
- Understand modern RL architectures used in robotics.
- Develop practical RL systems for autonomous agents and robotic applications.

### Prerequisites

- Python programming.
- Basic NumPy.
- Basic Matplotlib.
- Basic object-oriented programming.

Python fundamentals will not be taught from the beginning.

---

# 2. Teaching Methodology

Every major algorithm will follow the same engineering-oriented learning structure.

| No. | Section | Description |
|---|---|---|
| 1 | Technical Purpose | What problem does the algorithm solve? |
| 2 | Real-World Application | Where is it used in actual systems? |
| 3 | Mathematical Foundation | Equations, definitions and derivations |
| 4 | Algorithm Explanation | Step-by-step algorithm operation |
| 5 | From-Scratch Implementation | Python + NumPy implementation |
| 6 | Visualization | Graphs, diagrams and intermediate results |
| 7 | Numerical Verification | Verify mathematical results numerically |
| 8 | Professional Implementation | PyTorch, Gymnasium and other libraries |
| 9 | Performance Analysis | Complexity, memory, convergence and limitations |
| 10 | Practical Assignment | Engineering exercises and experiments |

## Programming Requirements

Every Python source file should include:

- Detailed technical comments.
- Mathematical equations.
- Input and output specifications.
- Tensor and matrix dimensions where applicable.
- Execution examples.
- Assertions and unit tests.
- Numerical verification.
- Visualization where useful.
- Connections to subsequent algorithms.

## Learning Principle

**Understand → Derive → Implement → Visualize → Verify → Experiment → Apply**

We should understand an algorithm mathematically before relying on a professional library implementation.

---

# 3. Complete Course Roadmap

## Phase 0 — Mathematical Foundations for Reinforcement Learning

### Objective

Understand the mathematical concepts required to implement and analyze RL algorithms.

### Topics

1. Linear Algebra
   - Scalars, vectors and matrices.
   - Vector operations.
   - Matrix multiplication.
   - Dot products.
   - Matrix transpose.
   - Identity matrices.
   - Matrix dimensions and computational complexity.

2. Probability and Statistics
   - Probability distributions.
   - Conditional probability.
   - Joint probability.
   - Expected value.
   - Variance and standard deviation.
   - Gaussian distribution.
   - Categorical distribution.
   - Sampling.

3. Calculus
   - Functions and derivatives.
   - Partial derivatives.
   - Gradients.
   - Chain rule.
   - Jacobians.
   - Numerical differentiation.

4. Numerical Methods
   - Finite differences.
   - Numerical gradient verification.
   - Floating-point precision.
   - Numerical stability.
   - Log-sum-exp.
   - Softmax.

### Python Implementation

```text
phase0/
    linear_algebra.py
    calculus.py
    probability.py
    numerical_gradients.py
    numerical_stability.py
```

### Practical Exercises

- Implement matrix multiplication from scratch.
- Implement probability distributions.
- Calculate expected values and variances.
- Implement numerical gradients.
- Implement Jacobians.
- Verify analytical derivatives against numerical derivatives.
- Visualize probability distributions and gradients.

---

## Phase 1 — Reinforcement Learning Fundamentals

### Objective

Understand the basic RL framework and how an agent learns through interaction with an environment.

### Topics

1. What is Reinforcement Learning?
2. Agent and Environment.
3. State and Action.
4. Reward and Return.
5. Policy.
6. Episode and Time Step.
7. Exploration and Exploitation.
8. Markov Property.
9. Markov Decision Process (MDP).
10. Discount Factor.
11. State Transition Probability.
12. Reward Function.

### Mathematical Foundation

Markov Decision Process:

\[
MDP=(S,A,P,R,\gamma)
\]

Where:

- S: State space.
- A: Action space.
- P: State transition probability.
- R: Reward function.
- γ: Discount factor.

Return:

\[
G_t=\sum_{k=0}^{\infty}\gamma^k r_{t+k+1}
\]

### Python Implementation

```text
phase1/
    mdp.py
    agent.py
    environment.py
    reward.py
    policy.py
```

### Practical Project

**Grid-World Environment**

Create a grid-world environment where an agent must reach a target while avoiding obstacles.

Implement:

- Environment initialization.
- State representation.
- Action execution.
- Reward calculation.
- State transitions.
- Episode termination.
- Random agent.

Visualize the agent's movement.

---

## Phase 2 — Dynamic Programming

### Objective

Understand how optimal policies can be calculated when the environment model is known.

### Topics

1. State-Value Function.
2. Action-Value Function.
3. Bellman Expectation Equation.
4. Bellman Optimality Equation.
5. Policy Evaluation.
6. Policy Improvement.
7. Policy Iteration.
8. Value Iteration.

### Important Equations

State-value function:

\[
V^\pi(s)=\mathbb{E}_\pi[G_t|S_t=s]
\]

Action-value function:

\[
Q^\pi(s,a)=\mathbb{E}_\pi[G_t|S_t=s,A_t=a]
\]

Bellman Expectation Equation:

\[
V^\pi(s)=
\sum_a\pi(a|s)
\sum_{s'}P(s'|s,a)
[r(s,a,s')+\gamma V^\pi(s')]
\]

Bellman Optimality Equation:

\[
V^*(s)=
\max_a\sum_{s'}P(s'|s,a)
[r(s,a,s')+\gamma V^*(s')]
\]

### Python Implementation

```text
phase2/
    bellman_equation.py
    policy_evaluation.py
    policy_iteration.py
    value_iteration.py
```

### Practical Project

Implement Policy Iteration and Value Iteration in Grid-World.

Compare:

- Number of iterations.
- Convergence speed.
- Optimal state values.
- Optimal policies.

Visualize value functions using heatmaps.

---

## Phase 3 — Model-Free Reinforcement Learning

### Objective

Learn how an agent can discover an optimal policy without knowing the environment's transition model.

### Topics

1. Model-Based vs Model-Free RL.
2. Monte Carlo Learning.
3. Temporal Difference Learning.
4. TD(0).
5. SARSA.
6. Q-Learning.
7. On-Policy Learning.
8. Off-Policy Learning.
9. Epsilon-Greedy Exploration.
10. Exploration vs Exploitation.

### Important Equations

TD(0) update:

\[
V(S_t)\leftarrow V(S_t)+
\alpha[r_{t+1}+\gamma V(S_{t+1})-V(S_t)]
\]

SARSA update:

\[
Q(S_t,A_t)\leftarrow Q(S_t,A_t)+
\alpha[r_{t+1}+\gamma Q(S_{t+1},A_{t+1})-Q(S_t,A_t)]
\]

Q-Learning update:

\[
Q(S_t,A_t)\leftarrow Q(S_t,A_t)+
\alpha[r_{t+1}+\gamma\max_a Q(S_{t+1},a)-Q(S_t,A_t)]
\]

### Python Implementation

```text
phase3/
    monte_carlo.py
    td_learning.py
    sarsa.py
    q_learning.py
    exploration.py
```

### Practical Project

Implement Q-Learning from scratch.

The agent must learn to navigate a Grid-World environment.

Measure:

- Total reward per episode.
- Number of steps to reach the goal.
- Learning convergence.
- Exploration rate.
- Final policy quality.

Compare SARSA and Q-Learning.

---

## Phase 4 — Deep Reinforcement Learning

### Objective

Understand how neural networks replace traditional Q-tables when the state space becomes large or continuous.

### Topics

1. Neural Network Fundamentals.
2. Perceptron.
3. Forward Propagation.
4. Backpropagation.
5. Gradient Descent.
6. Loss Functions.
7. PyTorch Fundamentals.
8. Deep Q-Network (DQN).
9. Experience Replay.
10. Target Networks.
11. Double DQN.
12. Dueling DQN.
13. Prioritized Experience Replay.

### Important Equation

DQN target:

\[
y=r+\gamma\max_{a'}Q_{\theta^-}(s',a')
\]

Loss:

\[
L(\theta)=
\mathbb{E}[(y-Q_\theta(s,a))^2]
\]

### Python Implementation

```text
phase4/
    neural_network.py
    dqn.py
    replay_buffer.py
    target_network.py
    double_dqn.py
    dueling_dqn.py
```

### Practical Project

Train a DQN agent using a suitable Gymnasium environment.

Evaluate:

- Training reward.
- Loss convergence.
- Replay buffer behavior.
- Target network updates.
- Training stability.

Visualize training performance.

---

## Phase 5 — Policy Gradient Methods

### Objective

Understand how agents can directly learn policies instead of learning only action-value functions.

### Topics

1. Policy-Based Methods.
2. Stochastic Policies.
3. Policy Parameterization.
4. Policy Gradient Theorem.
5. REINFORCE.
6. Baseline.
7. Actor-Critic.
8. Advantage Function.
9. A2C.
10. A3C.

### Important Equation

Policy Gradient:

\[
\nabla_\theta J(\theta)=
\mathbb{E}[
\nabla_\theta\log\pi_\theta(a|s)G_t
]
\]

Advantage Function:

\[
A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s)
\]

### Python Implementation

```text
phase5/
    policy_gradient.py
    reinforce.py
    actor_critic.py
    advantage.py
    a2c.py
    a3c.py
```

### Practical Project

Implement REINFORCE and Actor-Critic.

Compare their:

- Learning stability.
- Sample efficiency.
- Reward convergence.
- Policy behavior.

---

## Phase 6 — Advanced Reinforcement Learning

### Objective

Understand modern RL algorithms used in continuous-control and advanced decision-making problems.

### Topics

1. Proximal Policy Optimization (PPO).
2. Trust Region Policy Optimization (TRPO).
3. Deep Deterministic Policy Gradient (DDPG).
4. Twin Delayed DDPG (TD3).
5. Soft Actor-Critic (SAC).
6. Entropy Regularization.
7. Continuous Action Spaces.
8. On-Policy vs Off-Policy Training.
9. Sample Efficiency.
10. Hyperparameter Optimization.

### Important Equations

PPO Clipped Objective:

\[
L^{CLIP}(\theta)=
\mathbb{E}_t[
\min(
r_t(\theta)A_t,
\operatorname{clip}(r_t(\theta),1-\epsilon,1+\epsilon)A_t
)]
\]

Where:

\[
r_t(\theta)=
\frac{\pi_\theta(a_t|s_t)}
{\pi_{\theta_{old}}(a_t|s_t)}
\]

### Python Implementation

```text
phase6/
    ppo.py
    trpo.py
    ddpg.py
    td3.py
    sac.py
    continuous_control.py
```

### Practical Project

Train PPO and SAC agents in continuous-control environments.

Compare:

- Reward performance.
- Sample efficiency.
- Training stability.
- Action smoothness.
- Generalization.

---

## Phase 7 — Offline RL and Imitation Learning

### Objective

Understand how agents learn from previously collected datasets instead of relying exclusively on direct environment interaction.

### Topics

1. Demonstration Datasets.
2. Behavior Cloning.
3. Dataset Quality.
4. Distribution Shift.
5. Offline Reinforcement Learning.
6. Conservative Q-Learning (CQL).
7. Implicit Q-Learning (IQL).
8. Decision Transformer.
9. Advantage-Weighted Regression.
10. Evaluation of Learned Policies.

### Python Implementation

```text
phase7/
    behavior_cloning.py
    offline_dataset.py
    cql.py
    iql.py
    decision_transformer.py
```

### Practical Project

Train an agent using a fixed demonstration dataset.

Compare:

- Behavior Cloning.
- Online RL.
- Offline RL.

Analyze the impact of dataset quality and distribution shift.

---

## Phase 8 — Robotics and Embodied AI

### Objective

Apply Reinforcement Learning and related learning methods to robotic systems.

### Topics

1. Robotics Fundamentals.
2. Robot State Representation.
3. Joint Position and Velocity.
4. End-Effector Control.
5. Continuous Action Spaces.
6. Robot Simulation.
7. MuJoCo.
8. Sim-to-Real Transfer.
9. Domain Randomization.
10. Reward Design.
11. Visual Reinforcement Learning.
12. Vision-Language-Action Models.
13. Robot Policy Learning.

### Advanced Topics

- RT-1.
- RT-2.
- OpenVLA.
- XVLA.
- Vision-Language-Action architectures.
- Robot manipulation.
- Imitation Learning.
- Reinforcement Learning for robotic control.

### Python Implementation

```text
phase8/
    robot_environment.py
    robot_state.py
    robot_policy.py
    sim_to_real.py
    domain_randomization.py
    visual_rl.py
    robot_policy_training.py
```

### Practical Project

Develop a simulated robotic manipulation agent.

Possible tasks:

- Reach a target position.
- Move an object.
- Pick up an object.
- Place an object at a specified location.
- Perform simple manipulation sequences.

Evaluate policy performance and robustness.

---

# 4. Complete Project Directory

```text
RL_Course/
│
├── README.md
├── requirements.txt
│
├── phase0/
│   ├── linear_algebra.py
│   ├── calculus.py
│   ├── probability.py
│   ├── numerical_gradients.py
│   └── numerical_stability.py
│
├── phase1/
│   ├── mdp.py
│   ├── agent.py
│   ├── environment.py
│   ├── reward.py
│   └── policy.py
│
├── phase2/
│   ├── bellman_equation.py
│   ├── policy_evaluation.py
│   ├── policy_iteration.py
│   └── value_iteration.py
│
├── phase3/
│   ├── monte_carlo.py
│   ├── td_learning.py
│   ├── sarsa.py
│   ├── q_learning.py
│   └── exploration.py
│
├── phase4/
│   ├── neural_network.py
│   ├── dqn.py
│   ├── replay_buffer.py
│   ├── target_network.py
│   ├── double_dqn.py
│   └── dueling_dqn.py
│
├── phase5/
│   ├── policy_gradient.py
│   ├── reinforce.py
│   ├── actor_critic.py
│   ├── advantage.py
│   ├── a2c.py
│   └── a3c.py
│
├── phase6/
│   ├── ppo.py
│   ├── trpo.py
│   ├── ddpg.py
│   ├── td3.py
│   ├── sac.py
│   └── continuous_control.py
│
├── phase7/
│   ├── behavior_cloning.py
│   ├── offline_dataset.py
│   ├── cql.py
│   ├── iql.py
│   └── decision_transformer.py
│
└── phase8/
    ├── robot_environment.py
    ├── robot_state.py
    ├── robot_policy.py
    ├── sim_to_real.py
    ├── domain_randomization.py
    ├── visual_rl.py
    └── robot_policy_training.py
```

---

# 5. Recommended Development Tools

| Tool | Purpose |
|---|---|
| Python | Main programming language |
| NumPy | Mathematical operations and algorithms from scratch |
| Matplotlib | Algorithm visualization |
| PyTorch | Neural networks and Deep RL |
| Gymnasium | RL environments |
| Stable-Baselines3 | Professional RL implementations and comparisons |
| Jupyter Notebook | Experiments and interactive learning |
| MuJoCo | Physics-based robotic simulation |

The course should begin with NumPy implementations and gradually introduce PyTorch and professional RL libraries.

---

# 6. Learning and Evaluation Strategy

Every phase should produce working source code and a practical demonstration.

For each algorithm:

1. Understand its mathematical foundation.
2. Derive its important equations.
3. Implement it independently.
4. Run test cases.
5. Compare analytical and numerical results where applicable.
6. Visualize its internal behavior.
7. Evaluate performance.
8. Compare with alternative algorithms.
9. Complete an engineering assignment.

### Performance Measurements

Depending on the algorithm, measure:

- Cumulative reward.
- Average episode reward.
- Episode length.
- Convergence speed.
- Sample efficiency.
- Training stability.
- Computational complexity.
- Memory consumption.
- Generalization performance.

---

# 7. First Major Learning Project

## Grid-World Reinforcement Learning

This will be our first complete RL engineering project.

### Development Steps

| Step | Implementation | Expected Result |
|---|---|---|
| 1 | Random Agent | Understand environment interaction |
| 2 | MDP | Define the mathematical environment |
| 3 | Bellman Equation | Calculate expected future rewards |
| 4 | Value Iteration | Calculate optimal state values |
| 5 | Policy Iteration | Obtain an optimal policy |
| 6 | Q-Learning | Learn from experience |
| 7 | SARSA | Implement on-policy learning |
| 8 | Visualization | Compare learned policies |

### Final Deliverables

- Complete Python source code.
- Grid-World environment.
- RL agent.
- Q-table visualization.
- State-value heatmap.
- Policy visualization.
- Learning curves.
- Performance comparison report.

---

# 8. Guiding Principle

The objective is not merely to learn how to use RL libraries.

The objective is to understand how Reinforcement Learning algorithms work internally, why their mathematical formulations are designed as they are, how they are implemented, and how they can be applied to real engineering problems.

**The final destination is the ability to design, implement, analyze, and improve intelligent agents for real-world autonomous systems and robotics.**

## First Lesson

**Phase 0 — Mathematical Foundations for Reinforcement Learning**

First source file:

`phase0/linear_algebra.py`

We will then proceed step by step toward our first Q-Learning implementation.