---
title: "Introduction to Reinforcement Learning"
date:
draft: false
description:
tags: []
categories: []
author:
toc:
weight: 1
---

## Core Idea of Reinforcement Learning

Reinforcement Learning (RL) is a computational approach to learning whereby an **agent** seeks to maximize some notion of cumulative reward by taking actions in an **environment**. This process is akin to training a dog to perform a trick: you cannot dictate every move, but you can provide rewards when it performs correctly. In RL, the agent learns through trial and error, receiving feedback in the form of **rewards** or penalties based on its **actions**. The primary objective is to develop a **policy**, a strategy that guides the agent to take actions that maximize the total accumulated reward over time.

![](/content/posts/RL/1.PNG)

Let's rigorously define the key components of a Reinforcement Learning system:

- **Agent**: The decision-making entity that interacts with and learns from the environment. The agent observes states, takes actions, receives rewards, and updates its policy to improve its decision-making over time. Formally, it is defined by its policy $\pi$ and value functions.

- **Environment**: The external system with which the agent interacts. It is characterized by:

  - A state space $\mathcal{S}$ containing all possible states.
  - An action space $\mathcal{A}$ containing all possible actions.
  - A transition function $P(s'|s,a)$ defining the probability of transitioning to state $s'$ given current state $s$ and action $a$.
  - A reward function $R(s,a,s')$ defining the reward received when transitioning from $s$ to $s'$ via action $a$.

- **State ($S_t \in \mathcal{S}$)**: A complete description of the environment at time $t$ that contains all information relevant for future decision making. The state must satisfy the Markov property:

  $$P(S_{t+1}|S_t,A_t) = P(S_{t+1}|S_t,A_t,S_{t-1},A_{t-1},...,S_0,A_0)$$

  This means the current state captures all relevant information from the history.

- **Action ($A_t \in \mathcal{A}$)**: A choice made by the agent at time $t$ that influences the environment. Actions are selected according to the agent's policy $\pi$ and can be:

  - Discrete: $\mathcal{A} = \{a_1, a_2, ..., a_n\}$
  - Continuous: $\mathcal{A} \subseteq \mathbb{R}^n$

- **Reward ($R_t \in \mathbb{R}$)**: The immediate feedback signal received after taking action $A_{t-1}$ in state $S_{t-1}$. The reward function $R: \mathcal{S} \times \mathcal{A} \times \mathcal{S} \rightarrow \mathbb{R}$ defines the reward dynamics:
  $R_t = R(S_{t-1}, A_{t-1}, S_t)$

- **Policy ($\pi$)**: The agent's decision-making strategy that maps states to actions. Two types exist:
  - **Deterministic Policy**: $\pi: \mathcal{S} \rightarrow \mathcal{A}$
    Directly maps each state to an action: $a = \pi(s)$
  - **Stochastic Policy**: $\pi: \mathcal{S} \times \mathcal{A} \rightarrow [0,1]$
    Maps states to probability distributions over actions: $\pi(a|s) = P(A_t=a|S_t=s)$
    Must satisfy: $\sum_{a \in \mathcal{A}} \pi(a|s) = 1, \forall s \in \mathcal{S}$

![](/content/posts/RL/2..PNG)

- **Goal and Optimal Policy**: The agent aims to find an optimal policy $\pi^*$ that maximizes the expected cumulative reward (return) $G_t$:

  $$
  G_t = \sum_{k=0}^{T-t} \gamma^k R_{t+k+1}
  $$

  where $\gamma \in [0,1]$ is a discount factor that determines the relative importance of immediate versus future rewards.
  The optimal policy satisfies: $\pi^* = \arg\max_{\pi} \mathbb{E}_{\pi}[G_t|S_t=s], \forall s \in \mathcal{S}$

- **Value Functions**: Two key functions quantify the expected return:

  - **State-Value Function**: $V_{\pi}(s) = \mathbb{E}_{\pi}[G_t|S_t=s]$
    Expected return starting from state $s$ and following policy $\pi$
  - **Action-Value Function**: $Q_{\pi}(s,a) = \mathbb{E}_{\pi}[G_t|S_t=s,A_t=a]$
    Expected return starting from state $s$, taking action $a$, then following policy $\pi$
    These functions satisfy the Bellman equations, fundamental to RL algorithms.

- **Horizon ($T$)**: The time span of agent-environment interaction:
  - **Finite Horizon**: Fixed number of steps $T < \infty$
  - **Infinite Horizon**: $T = \infty$, typically with discounting ($\gamma < 1$)
  - **Episodic**: Tasks that naturally terminate after variable length sequences

## Core Problems in Reinforcement Learning

In the field of Reinforcement Learning, we encounter two primary types of problems that are essential for understanding and developing effective learning algorithms:

- **Prediction Problem**: Formally, given a policy $\pi$, the prediction problem, also known as policy evaluation, involves determining the value function $V_{\pi}$ or the action-value function $Q_{\pi}$. Mathematically, the value function $V_{\pi}(s)$ is defined as:

  $$ V*{\pi}(s) = \mathbb{E}*{\pi}[G_t | S_t = s] $$

  where $G_t$ is the return, representing the cumulative future rewards. The prediction problem asks: "What is the expected return if the agent follows this policy $\pi$ from state $s$?" This evaluation helps us understand the effectiveness of a given policy by quantifying the expected rewards.

- **Control Problem**: The control problem focuses on finding the optimal policy $\pi^*$. Mathematically, the optimal policy:

  $$
  \pi^* = \arg\max_{\pi} \mathbb{E}_{\pi}[G_t | S_t = s], \forall s \in \mathcal{S}
  $$

  Solving the control problem typically involves iteratively addressing the prediction problem to evaluate the current policy and then improving the policy based on these evaluations. This iterative process is crucial for refining the agent's strategy to achieve optimal performance.

In summary, reinforcement learning fundamentally aims to solve the control problem, with the prediction problem serving as a critical step in evaluating and improving policies to reach optimal decision-making strategies.
