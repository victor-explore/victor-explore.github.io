---
title: "How reinforcement learning works in Unity"
date: 2025-03-01
draft: false
description:
tags: []
categories: []
author:
toc:
weight: 1
---
## Unity Packages for Reinforcement Learning

Unity's modular nature is facilitated by its **Package Manager**, which allows developers to add specific functionalities to their projects. For implementing Reinforcement Learning (RL) within Unity, the primary and most widely used package is:

*   **ML-Agents Toolkit**:
    *   **Purpose**: This is Unity's official open-source package specifically designed to enable games and simulations to serve as environments for training intelligent agents using RL and imitation learning.
    *   **Functionality**: It provides the necessary tools and SDKs to:
        *   Define **Agents**: The actors within the environment that learn to make decisions.
        *   Define **Observations**: The information the agent receives from the environment (e.g., position, velocity, sensor readings).
        *   Define **Actions**: The decisions the agent can make (e.g., move forward, jump, turn).
        *   Define **Rewards**: The signals given to the agent to indicate whether its actions are good or bad towards achieving a goal.
        *   Connect to **Training Algorithms**: It integrates with established Python-based RL libraries (like TensorFlow via `mlagents-learn`) to perform the actual training process, often using algorithms like PPO (Proximal Policy Optimization) or SAC (Soft Actor-Critic).
    *   **Integration**: ML-Agents allows seamless communication between the Unity environment (where the simulation runs) and the Python training process.

While **ML-Agents** is the cornerstone for RL in Unity, developers could potentially integrate other third-party libraries or custom solutions, but these would typically require significantly more manual setup and integration effort compared to the streamlined workflow offered by the official ML-Agents package. For most users looking to apply RL in Unity, ML-Agents is the recommended and standard choice.

## How Unity communicates with Python

The Unity ML-Agents Toolkit acts as a bridge, enabling communication between the game engine environment (running C# code) and the machine learning algorithms (typically running in Python). This separation allows developers to leverage powerful Python libraries for training while using Unity's sophisticated physics and rendering capabilities for simulation.

### The Core Components

*   **Unity Environment**: This is your game or simulation running in the Unity Editor or as a standalone build. It contains:
    *   **Agents**: The actors within the simulation that learn to perform tasks. Each agent has scripts defining its observations, actions, and reward calculation.
    *   **Behavior Parameters**: A Unity component attached to the Agent GameObject that specifies the observation space, action space, and behavior type (e.g., training, inference).
    *   **Academy (Older versions) / Environment Controller (Implicit)**: Manages the overall simulation stepping and communication timing.
*   **Python API (`mlagents-learn`)**: This is the Python package containing the training algorithms (like PPO, SAC) and the communication interface. It runs as a separate process.
    *   **Trainer**: Manages the training process, collects experiences from the Unity environment(s), and updates the agent's policy.
    *   **Policy**: The neural network model that maps observations to actions.

### The Communication Protocol

ML-Agents primarily uses **gRPC (Google Remote Procedure Call)** for communication. gRPC is a high-performance framework that allows the Unity environment (acting as a client) and the Python trainer (acting as a server) to communicate efficiently, exchanging data like observations and actions across different processes and languages.

### The Communication Loop (Step-by-Step)

The interaction follows a cyclical pattern, synchronized between Unity and Python:

1.  **Initialization**:
    *   The Python training process (`mlagents-learn`) starts and listens for incoming connections from Unity environments.
    *   The Unity environment starts. The ML-Agents components initialize and establish a connection with the Python process via gRPC.

2.  **Observation Step (Unity -> Python)**:
    *   At each decision point, Unity collects the current state information for each agent. This is the **Observation** ($s_t$).
    *   Unity also sends any **Reward** ($r_t$) accumulated since the last action and a **Done** signal ($d_t$) indicating if the agent's episode has ended.
    *   This bundle of information ($s_t, r_t, d_t$) is sent to the Python process for each agent.

3.  **Action Step (Python -> Unity)**:
    *   The Python process receives the observations ($s_t$) for each agent.
    *   The current **Policy** ($\pi$) takes the observation $s_t$ as input and computes the next **Action** ($a_t$) for each agent.
        *   $a_t \sim \pi(\cdot | s_t)$
    *   These computed actions ($a_t$) are sent back to the Unity environment.

4.  **Environment Step (Unity)**:
    *   Unity receives the actions ($a_t$) from Python.
    *   It applies these actions to the corresponding agents within the simulation.
    *   Unity's physics engine and game logic advance the simulation by one step.
    *   This results in a new state ($s_{t+1}$), a new reward ($r_{t+1}$), and a new done signal ($d_{t+1}$).
    *   The loop then returns to Step 2.

5.  **Training (Python)**:
    *   Concurrently, the Python trainer collects the transitions experienced by the agents: $(s_t, a_t, r_{t+1}, s_{t+1}, d_{t+1})$.
    *   These experiences are stored in a buffer.
    *   Periodically, the trainer samples batches of experiences from the buffer and uses them to update the agent's policy ($\pi$) using the chosen RL algorithm (e.g., PPO).

This continuous cycle allows the agent in Unity to interact with its environment, while the Python process learns and refines the agent's decision-making policy based on those interactions. During **inference** (using a pre-trained model), the Python side simply provides actions based on the fixed policy without performing updates.


## Prerequisites for Learning Proximal Policy Optimization (PPO)

Proximal Policy Optimization (PPO) is a powerful and widely used reinforcement learning algorithm, particularly effective in continuous control tasks. However, it builds upon several fundamental concepts in RL. To truly understand PPO – why it works and how it differs from other methods – you should have a solid grasp of the following prerequisites:

### 1. Fundamentals of Reinforcement Learning (RL)

*   **Core RL Loop:** Understand the basic interaction cycle between an **agent** and an **environment**.
    *   The agent observes a **state** ($s$).
    *   The agent takes an **action** ($a$).
    *   The environment transitions to a new state ($s'$) and provides a **reward** ($r$).
*   **Goal of RL:** The objective is typically to find a **policy** ($\pi$) – a mapping from states to actions – that maximizes the **cumulative discounted reward** (Return, $G_t$) over an episode or infinite horizon.
    $$
    G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k+1}
    $$
    where:
    *   $r_{t+k+1}$ is the reward received at timestep $t+k+1$.
    *   $\gamma$ is the **discount factor** ($0 \le \gamma \le 1$), which balances immediate vs. future rewards.
*   **Exploration vs. Exploitation:** Understand the fundamental trade-off between trying new actions to discover potentially better rewards (exploration) and choosing actions known to yield good rewards (exploitation).

### 2. Markov Decision Processes (MDPs)

*   **Formal Framework:** RL problems are often formalized as MDPs. You should understand the components:
    *   $S$: A set of possible **states**.
    *   $A$: A set of possible **actions**.
    *   $P(s' | s, a)$: The **transition probability** function, defining the probability of transitioning to state $s'$ given the current state $s$ and action $a$.
    *   $R(s, a, s')$: The **reward function**, defining the reward received after transitioning from state $s$ to $s'$ via action $a$.
    *   $\gamma$: The **discount factor**.
*   **Markov Property:** The assumption that the future state $s'$ depends only on the current state $s$ and action $a$, not on the history of previous states and actions.

### 3. Value Functions

*   **Concept:** Value functions estimate the "goodness" of being in a particular state or taking a specific action in a state, according to the current policy $\pi$.
*   **State-Value Function ($V^\pi(s)$):** The expected return starting from state $s$ and following policy $\pi$ thereafter.
    $$
    V^\pi(s) = E_\pi [ G_t | S_t = s ] = E_\pi \left[ \sum_{k=0}^{\infty} \gamma^k r_{t+k+1} \biggm| S_t = s \right]
    $$
*   **Action-Value Function ($Q^\pi(s, a)$):** The expected return starting from state $s$, taking action $a$, and then following policy $\pi$ thereafter.
    $$
    Q^\pi(s, a) = E_\pi [ G_t | S_t = s, A_t = a ] = E_\pi \left[ \sum_{k=0}^{\infty} \gamma^k r_{t+k+1} \biggm| S_t = s, A_t = a \right]
    $$
*   **Bellman Equations:** Understand how value functions relate recursively (Bellman expectation equations). This forms the basis for many RL algorithms (though PPO doesn't directly solve them in the same way as Q-learning).

### 4. Policy Gradient Methods

*   **Direct Policy Optimization:** Unlike value-based methods (like Q-Learning) that learn value functions and derive a policy, policy gradient methods directly learn the parameters of the policy $\pi_\theta(a|s)$, where $\theta$ represents the policy parameters (e.g., weights of a neural network).
*   **Objective Function:** The goal is to find parameters $\theta$ that maximize an objective function $J(\theta)$, usually the expected total return.
    $$
    J(\theta) = E_{\tau \sim \pi_\theta} [ R(\tau) ] = E_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} r(s_t, a_t) \right]
    $$
    where $\tau = (s_0, a_0, r_1, s_1, a_1, ...)$ is a trajectory sampled by following policy $\pi_\theta$.
*   **Policy Gradient Theorem:** Understand the core idea that we can update the policy parameters $\theta$ by taking steps in the direction of the gradient $\nabla_\theta J(\theta)$. A common form involves:
    $$
    \nabla_\theta J(\theta) = E_{\pi_\theta} [ \nabla_\theta \log \pi_\theta(a|s) \cdot \Psi ]
    $$
    where $\Psi$ is some measure of the "goodness" of taking action $a$ in state $s$. Different algorithms use different forms for $\Psi$ (e.g., total return $G_t$ in REINFORCE, Advantage function in Actor-Critic).
*   **Intuition:** Increase the probability ($\log \pi_\theta(a|s)$) of actions ($a$) that lead to better outcomes ($\Psi$).
*   **REINFORCE Algorithm:** Familiarity with this basic policy gradient algorithm helps understand the core update mechanism.

### 5. Actor-Critic Methods

*   **Hybrid Approach:** These methods combine aspects of both value-based and policy-based methods.
    *   **Actor:** The policy $\pi_\theta(a|s)$, responsible for selecting actions.
    *   **Critic:** A value function estimator (e.g., $V_\phi(s)$ or $Q_\phi(s, a)$ with parameters $\phi$), responsible for evaluating the actions taken by the actor.
*   **Advantage Function ($A(s, a)$):** A crucial concept often used in Actor-Critic methods and central to PPO. It measures how much better taking action $a$ is compared to the average action in state $s$, according to the current policy.
    $$
    A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)
    $$
    Using the advantage function in the policy gradient update often reduces variance compared to using the raw return $G_t$.
    $$
    \nabla_\theta J(\theta) = E_{\pi_\theta} [ \nabla_\theta \log \pi_\theta(a|s) \cdot A^\pi(s, a) ]
    $$
*   **Advantage Actor-Critic (A2C/A3C):** Understanding A2C (or its asynchronous version A3C) is highly recommended, as PPO is a direct improvement addressing some stability issues found in these earlier Actor-Critic methods.

### 6. Basic Deep Learning Concepts

*   **Neural Networks:** PPO typically uses neural networks to represent the policy (actor) and the value function (critic). Understanding basic neural network architectures, activation functions, and forward/backward propagation is necessary.
*   **Optimization Algorithms:** Familiarity with gradient descent and its variants (like Adam, RMSprop) used to train neural networks.
*   **Loss Functions:** Understanding how loss functions quantify the error between predictions and targets, guiding the learning process.

Having a good foundation in these areas will make learning the specifics of PPO – its objective function, clipping mechanism, and implementation details – much more intuitive and manageable.


## What is Proximal Policy Optimization (PPO)?

Proximal Policy Optimization (PPO) is fundamentally an **Actor-Critic** algorithm. This architectural paradigm involves two distinct components working in tandem to learn an optimal policy. 

The **Actor** is responsible for deciding which action to take in a given state; it represents the policy itself, often parameterized by a neural network with parameters $\theta$. We denote the policy as:
$$
\pi_\theta(a|s)
$$
This function outputs the probability of taking action $a$ in state $s$. 

The **Critic**, on the other hand, evaluates the actions taken by the Actor or the states encountered. It estimates a value function, such as the state-value function $V(s)$ or the action-value function $Q(s, a)$, typically using another neural network with parameters $\phi$. We denote the state-value function estimated by the critic as:
$$
V_\phi(s)
$$
The Critic provides feedback (often in the form of an advantage estimate) to the Actor, guiding it on how to adjust its policy parameters $\theta$ to improve performance. PPO leverages this Actor-Critic structure but introduces specific mechanisms (like clipping the objective function) to ensure more stable and reliable policy updates compared to earlier Actor-Critic methods.










## Advantages of PPO

### Online Training
No need for experience replay buffer.


## My setup

### State Space

### Action Space

### Reward Function


