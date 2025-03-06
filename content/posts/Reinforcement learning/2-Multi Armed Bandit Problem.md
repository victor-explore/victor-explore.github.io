---
title: "Multi-Armed Bandit Problem"
date:
draft: false
description:
tags: []
categories: []
author:
toc:
weight: 1
---

## Multi-Armed Bandit Problem

![](/content/posts/RL/3..PNG)

The Multi-Armed Bandit problem is a fundamental concept in reinforcement learning, where an agent must repeatedly select from multiple options, known as "arms" (akin to slot machines), to maximize its cumulative reward. This scenario is simplified by the fact that the environment is **state-less**, meaning the outcome of an action is solely dependent on the action itself, without any influence from a changing state.

Key characteristics of the Multi-Armed Bandit problem include:

- **State-less Environment**: The environment lacks dynamic states that evolve over time, meaning the agent's decision at each step is unaffected by previous states of the environment.
- **Multiple Actions (Arms)**: The agent has a choice among $k$ distinct actions, referred to as arms, represented by the set $\{1, 2, \ldots, k\}$.
- **Unknown Reward Distributions**: Each arm $a$ is linked to an unknown probability distribution of rewards. When the agent selects arm $a$ at time step $t$ (denoted as $A_t = a$), it receives a reward $R_{t+1}$ drawn from the distribution associated with that arm. The agent is initially unaware of these distributions.
- **Objective**: The agent's goal is to learn a strategy for selecting actions over time that maximizes the total expected reward accumulated across a series of trials. This requires balancing exploration (testing different arms to understand their reward distributions) and exploitation (choosing the arm currently believed to yield the highest reward based on existing knowledge).

To formalize this, we define the **true value of an action** $a$, denoted as $q^*(a)$, which is the expected reward when action $a$ is chosen. Mathematically, this is expressed as the expected value of the reward $R_{t+1}$ given that action $A_t = a$ was taken:

$$ q^\*(a) = \mathbb{E}[R_{t+1} | A_t = a] $$

Here, $\mathbb{E}[\cdot]$ denotes the expected value. Intuitively, $q^*(a)$ represents the average reward anticipated from repeatedly selecting arm $a$.

The **optimal action**, denoted as $a^*$, is the action with the highest expected reward. The objective is to identify this optimal action:

$$ a^\* = \arg\max\_{a} q^\*(a) $$

where $\arg\max_{a}$ signifies "the action $a$ that maximizes the expression". In this context, $q^*(a)$ functions as the **action-value function**, which measures the desirability or "value" of taking action $a$, reflecting the expected long-term reward from that action. Since the true reward distributions are unknown, these action values must be estimated through experience.

### Action-value Methods

Action-value methods are crucial for estimating the expected reward of taking a specific action $a$, denoted as $q^*(a)$. This expected reward is not known a priori, so we estimate it through experience by averaging the rewards obtained from taking action $a$.

Define $q_n(a)$ as the estimated action-value for action $a$ after $n$ observations. The formula for $q_n(a)$ is:

<div class="math-katex">
  $$ q*n(a) = \frac{\sum*{i=1}^{n} R*i \cdot \mathbb{I}\{A*{i-1} = a\}}{\sum*{i=1}^{n} \mathbb{I}\{A*{i-1} = a\}} $$
</div>

Breaking down the formula:

- **$q_n(a)$**: The estimated action-value for action $a$ based on experience.

- **$R_i$**: The reward received at time step $i$, which is the feedback from the environment after taking action $A_{i-1}$.

- **$A_{i-1}$**: The action taken at the previous time step ($i-1$).

- **$\mathbb{I}\{A_{i-1} = a\}$**: An indicator function that equals 1 if action $a$ was taken at time step $i-1$, and 0 otherwise.

- **Numerator: $\sum_{i=1}^{n} R_i \cdot \mathbb{I}\{A_{i-1} = a\}$**: This sums all rewards received when action $a$ was taken. For each time step $i$, if $A_{i-1} = a$, $R_i$ is included in the sum; otherwise, it contributes 0.

- **Denominator: $\sum_{i=1}^{n} \mathbb{I}\{A_{i-1} = a\}$**: This counts the number of times action $a$ has been selected in the first $n$ time steps.

Thus, $q_n(a)$ is the average reward received from action $a$ up to the $n^{th}$ time step, continuously updated as more actions are taken and rewards are received.

With these action-value estimates, we can formulate a policy to decide which action to take at each step, balancing exploration and exploitation to optimize long-term rewards.

### Greedy Policy

- **Description**: The greedy policy is an action selection strategy where the agent consistently chooses the action with the highest estimated value based on current knowledge.

- **Formal Definition**: The action $a$ is selected such that it maximizes the estimated action-value $q_n(b)$ over all possible actions $b \in \{1, 2, \ldots, k\}$:

  $$
  a = \arg\max_{b} q_n(b) \quad \text{for } b \in \{1, 2, \ldots, k\}
  $$

- **Explanation**: Intuitively, the agent evaluates its current estimates of the value of each action, denoted as $q_n(1), q_n(2), \ldots, q_n(k)$, and selects the action with the highest value. This approach is termed "greedy" because it focuses on the immediate best option, potentially overlooking exploration of other actions.

- **Limitations**:

  - **Lack of Exploration**: A purely greedy policy does not explore other actions. If an action is initially perceived as slightly better due to random fluctuations or limited data, the agent will repeatedly choose it, missing out on potentially better actions.
  - **Suboptimal Solutions**: In the context of the multi-armed bandit problem, especially early in the learning process, the action-value estimates $q_n(a)$ may be inaccurate. A greedy policy can lead to premature convergence on a suboptimal action if it is initially overestimated, preventing the discovery of superior actions.

- **When to Use**: The greedy policy is generally unsuitable for scenarios where exploration is crucial, such as in the early stages of reinforcement learning. It may be appropriate when initial estimates are highly reliable or when the focus is on exploitation rather than exploration.

### $\epsilon$-Greedy Policy

- **Description**: The $\epsilon$-greedy policy is a strategy designed to balance the trade-off between exploration (trying new actions) and exploitation (choosing the best-known action). It predominantly exploits the best-known action but occasionally explores other actions to improve knowledge.

- **Mechanism**: The policy operates by selecting the action with the highest estimated value with a probability of $1-\epsilon$, and with a probability of $\epsilon$, it selects an action randomly from the set of all possible actions.

- **Formal Definition**:

  - **With probability $1-\epsilon$ (Exploitation)**: Select an action that has the maximum estimated action-value. If there are multiple such actions, any one of them can be chosen. Mathematically, the selected action $a$ belongs to the set of actions that maximize $q_n(b)$:
    $$ a \in \arg\max\_{b \in \{1, 2, \ldots, k\}} q_n(b) $$

  - **With probability $\epsilon$ (Exploration)**: Select an action randomly from the set of all possible actions, with each action having an equal probability of being selected. Mathematically, for each action $b \in \{1, 2, \ldots, k\}$, the probability of selecting $b$ is $\frac{1}{k}$:
    $$ P(a = b) = \frac{1}{k} \quad \text{for } b \in \{1, 2, \ldots, k\} $$

- **Role of $\epsilon$**:

  - $\epsilon$ is a hyperparameter that determines the level of exploration. It ranges from 0 to 1.
  - **High $\epsilon$ (close to 1)**: Encourages more exploration, allowing the agent to discover potentially better actions, though it may slow convergence to the optimal policy.
  - **Low $\epsilon$ (close to 0)**: Favors exploitation, leading to faster convergence if initial estimates are accurate, but risks missing out on better actions due to limited exploration.

- **Hyperparameter Tuning**: The choice of $\epsilon$ is crucial and often requires tuning. A common practice is to start with a higher $\epsilon$ (e.g., 0.1 or 0.2) to promote exploration initially and then reduce it over time.

- **Decaying $\epsilon$**: As the agent's knowledge improves, the need for exploration decreases. Thus, $\epsilon$ can be decayed over time, either linearly or exponentially, to allow more exploitation as confidence in action-value estimates grows.

- **Advantages**:

  - **Balances Exploration and Exploitation**: Offers a straightforward method to explore the action space while exploiting the best-known actions.
  - **Enhanced Long-term Performance**: By incorporating exploration, the $\epsilon$-greedy policy is more likely to identify the optimal action, improving long-term rewards compared to a purely greedy approach.

- **Considerations**:

  - **Choice of $\epsilon$**: The effectiveness of $\epsilon$ (and its decay schedule) can vary depending on the problem and may require experimentation.
  - **Uniform Random Action Selection**: While the policy selects actions uniformly at random during exploration, more advanced strategies might prioritize actions with higher uncertainty or less frequent selection. Despite this, the simplicity of uniform random selection makes $\epsilon$-greedy a popular and effective baseline strategy.

### Incremental Implementation for Efficient Action-Value Updates

- Action-value methods rely on estimating action values as the average of observed rewards. A naive approach of storing all rewards and recalculating the average each time is computationally expensive and memory-intensive, especially as the number of rewards grows. To address this, we can use an incremental update approach that allows for efficient computation with constant memory and per-time-step computation.

- Let $q_n$ be the current estimate of an action's value after it has been selected $n-1$ times, and let $R_n$ be the $n^{th}$ reward received after selecting that action. Instead of recalculating the sum of all rewards each time, we can update the action-value estimate $q_n$ to $q_{n+1}$ using the new reward $R_n$ and the previous estimate $q_n$.

- The incremental update formula is derived from the definition of the average:

  $$
  q_{n+1} = \frac{1}{n} \sum_{i=1}^{n} R_i
  $$

  This can be rewritten incrementally as (derivation skipped):

  $$
  q_{n+1} = q_n + \frac{1}{n} (R_n - q_n)
  $$

  where:

  - $q_{n+1}$ is the updated action-value estimate after observing the $n^{th}$ reward.
  - $q_n$ is the previous action-value estimate.
  - $R_n$ is the $n^{th}$ reward received.
  - $\frac{1}{n}$ acts as a step size, determining how much the new reward influences the old estimate.

- **Intuition**: This update rule adjusts the old estimate $q_n$ towards the new reward $R_n$. The term $(R_n - q_n)$ represents the error or difference between the new reward and the current estimate. We move a fraction of this error, determined by the step size $\frac{1}{n}$, to refine our estimate. As $n$ increases, the step size $\frac{1}{n}$ decreases, meaning that newer rewards have a smaller impact on the estimate, giving more weight to the accumulated history of rewards.

## General Incremental Implementation for Action-Value Estimation

To efficiently track and update action-value estimates incrementally, we can use a generalized update rule. This rule is applied individually to each action $a$ at each time step $t$. It incorporates a step-size parameter, denoted as $\alpha_t$ (also known as the learning rate), to control the update magnitude. The general incremental update rule is mathematically expressed as:

<div class="math-katex">
  $$
  q_{t+1}(a) = q_t(a) + \alpha_t(R_{t+1} - q_t(a)) 
  $$
</div>

This update equation adjusts the current action-value estimate $q_t(a)$ towards the observed reward $R_{t+1}$. The step-size parameter $\alpha_t$ determines the extent of this adjustment.

Let's break down each component of this update rule to understand its role and significance:

- **$q_t(a)$ : Action-Value Estimate at Time $t$**

  - **Mathematical Definition**: $q_t(a)$ represents our current estimate of the true action-value $q_*(a)$ for action $a$ at time step $t$.

  - **Intuitive Explanation**: Think of $q_t(a)$ as our best guess, based on past experiences, of how good it is to take action $a$. As we gather more experience, we refine this estimate. Initially, it might be inaccurate, but with each update, we aim to get closer to the true value $q_*(a)$.

- **$R_{t+1}$ : Reward at Time $t+1$**

  - **Mathematical Definition**: $R_{t+1}$ is the scalar reward received from the environment at time step $t+1$ as a direct consequence of taking action $a$ at time $t$.
  - **Intuitive Explanation**: $R_{t+1}$ is the immediate feedback we get from the environment after performing action $a$. It's the environment's way of telling us how "good" or "bad" our action was in the short term.

- **$\alpha_t$ : Step-Size Parameter (Learning Rate) at Time $t$**

  - **Mathematical Definition**: $\alpha_t$ is a positive scalar at time step $t$ that determines the extent to which new information overrides old estimates. It is often referred to as the learning rate.
  - **Intuitive Explanation**: $\alpha_t$ controls how much we adjust our action-value estimate $q_t(a)$ based on the new reward $R_{t+1}$.
    - A **large $\alpha_t$ (closer to 1)** means we give significant weight to the new reward $R_{t+1}$. The update is aggressive, and $q_{t+1}(a)$ will change considerably based on $R_{t+1}$. This is useful in non-stationary environments or early in learning when estimates are uncertain.
    - A **small $\alpha_t$ (closer to 0)** means we give less weight to $R_{t+1}$ and rely more on our existing estimate $q_t(a)$. The update is conservative, leading to slower but potentially more stable learning. This is beneficial in stationary environments or later in learning when we want to fine-tune estimates.

- **$(R_{t+1} - q_t(a))$ : Prediction Error (Temporal Difference Error)**
  - **Mathematical Definition**: This term represents the difference between the observed reward $R_{t+1}$ and our current action-value estimate $q_t(a)$. It is also known as the Temporal Difference (TD) error.
  - **Intuitive Explanation**: The prediction error is a measure of "surprise". It tells us how much the new reward $R_{t+1}$ deviates from our expectation $q_t(a)$.
    - **Positive error ($R_{t+1} > q_t(a)$)**: The reward is better than expected. We were "pleasantly surprised," so we increase our estimate $q_t(a)$ to be closer to this better-than-expected outcome.
    - **Negative error ($R_{t+1} < q_t(a)$)**: The reward is worse than expected. We were "unpleasantly surprised," so we decrease our estimate $q_t(a)$ to reflect this worse-than-expected outcome.
    - **Zero error ($R_{t+1} = q_t(a)$)**: The reward is exactly as expected. There is no surprise, and our estimate $q_t(a)$ remains unchanged.

### Conditions on the Step-Size Parameter for Convergence

For the sequence of action-value estimates $q_t(a)$ to converge to the true action-values $q_*(a)$, the step-size parameter $\alpha_t$ cannot be chosen arbitrarily. It must satisfy the following two conditions as time $t$ approaches infinity. These are derived from the theory of stochastic approximation and ensure convergence in stochastic iterative algorithms:

1.  **Sum of Step-Sizes is Infinite: $\sum_{t=1}^{\infty} \alpha_t = \infty$ (Robbins-Monro Condition)**

    - **Mathematical Significance**: This condition, known as the Robbins-Monro condition, is essential for overcoming initial conditions and ensuring that we can eventually reach the true value, regardless of the starting estimate $q_0(a)$.
    - **Intuitive Explanation**: Imagine trying to climb a hill to reach the peak (the true action-value). Each step you take is scaled by $\alpha_t$. If the sum of all step-sizes is infinite, it means you have the potential to take arbitrarily large total steps. This guarantees that you can eventually reach the peak, no matter where you start on the hill. In the context of learning, this condition ensures that we continue to make adjustments to our estimates based on new experiences indefinitely, preventing premature convergence to a suboptimal estimate. We need to keep taking steps to explore and refine our estimates.

2.  **Sum of Squared Step-Sizes is Finite: $\sum_{t=1}^{\infty} \alpha_t^2 < \infty$ (Square-Summable Condition)**
    - **Mathematical Significance**: This condition ensures that while the step-sizes are large enough to reach the true value (as per condition 1), they also become progressively smaller over time to ensure stable convergence and prevent oscillations around the true value.
    - **Intuitive Explanation**: While the first condition ensures we can reach the peak, this second condition is about how we approach it. If the sum of the squares of the step-sizes is finite, it implies that the individual step-sizes $\alpha_t$ must approach zero as $t \to \infty$. This gradual decrease in step-size is like taking smaller and smaller steps as you get closer to the peak. It prevents you from overshooting the peak and oscillating back and forth. It stabilizes the learning process, ensuring that our estimates settle down to a specific value rather than fluctuating indefinitely. We need to reduce the step size over time to fine-tune our position as we get closer to the true value and avoid unnecessary jitter.

Both conditions must be satisfied simultaneously to guarantee convergence of $q_t(a)$ to $q_*(a)$ in stochastic approximation.

### Examples of Step-Size Sequences that Satisfy Convergence Conditions

Several step-size sequences satisfy these conditions, each leading to different convergence behaviors. Here are a few common examples:

1.  **Power Law Decay: $\alpha_t = \frac{1}{(t+1)^p}$, where $t \geq 0$ and $0.5 < p \leq 1$**

    - **Example (p=1):** $\alpha_t = \frac{1}{t+1}$. This is the step-size used in the sample-average method.
    - **Verification of Conditions**:
      - $\sum_{t=1}^{\infty} \frac{1}{(t+1)^p} = \infty$ for $p \leq 1$. The harmonic series ($p=1$) diverges, and for $0.5 < p < 1$, the sum also diverges.
      - $\sum_{t=1}^{\infty} \left(\frac{1}{(t+1)^p}\right)^2 = \sum_{t=1}^{\infty} \frac{1}{(t+1)^{2p}} < \infty$ for $2p > 1$, which means $p > 0.5$.
    - **Characteristics**: Power law decay with $0.5 < p \leq 1$ satisfies both conditions. Values of $p$ closer to 1 provide faster initial learning but can lead to slower convergence in later stages.

2.  **Logarithmic Decay: $\alpha_t = \frac{1}{(t+1)\log(t+1)}$, where $t > 1$ (and we can set $\alpha_0 = 1, \alpha_1 = 1$ for initial steps)**

    - **Verification of Conditions**:
      - $\sum_{t=2}^{\infty} \frac{1}{(t+1)\log(t+1)} = \infty$. This sum diverges (by integral test comparison with $\int \frac{1}{x\log x} dx = \log(\log x)$).
      - $\sum_{t=2}^{\infty} \left(\frac{1}{(t+1)\log(t+1)}\right)^2 < \infty$. This sum converges because the denominator grows faster than $t^2$.
    - **Characteristics**: Logarithmic decay results in step-sizes that decrease more slowly than power law decay, especially in later stages. This can lead to slower initial convergence but potentially better long-term exploration and fine-tuning.

3.  **Another Logarithmic Variant: $\alpha_t = \frac{\log(t+1)}{t+1}$, where $t \geq 0$**
    - **Verification of Conditions**:
      - $\sum_{t=0}^{\infty} \frac{\log(t+1)}{t+1} = \infty$. This sum diverges (by integral test comparison with $\int \frac{\log x}{x} dx = \frac{1}{2}(\log x)^2$).
      - $\sum_{t=0}^{\infty} \left(\frac{\log(t+1)}{t+1}\right)^2 < \infty$. This sum converges because, although $\log(t+1)$ grows slowly, the $t^2$ term in the denominator dominates for large $t$, ensuring convergence.
    - **Characteristics**: This variant also exhibits logarithmic decay, but with a slightly different profile. It can offer a balance between initial learning rate and long-term fine-tuning.

### Comparison of Step-Size Sequences

| Step-Size Sequence  | Formula                               | $\sum \alpha_t = \infty$ | $\sum \alpha_t^2 < \infty$ | Convergence Speed | Long-Term Exploration/Fine-tuning |
| :------------------ | :------------------------------------ | :----------------------- | :------------------------- | :---------------- | :-------------------------------- |
| Power Law Decay     | $\frac{1}{(t+1)^p}$, $0.5 < p \leq 1$ | Yes                      | Yes                        | Relatively Fast   | Moderate                          |
| Logarithmic Decay   | $\frac{1}{(t+1)\log(t+1)}$            | Yes                      | Yes                        | Slower            | Higher                            |
| Logarithmic Variant | $\frac{\log(t+1)}{t+1}$               | Yes                      | Yes                        | Slower            | Higher                            |

The choice of step-size parameter $\alpha_t$ is crucial and often problem-dependent. Power law decay is a good starting point for many problems, offering a balance between convergence speed and stability. Logarithmic decay variants might be preferred when more emphasis is needed on long-term exploration or in non-stationary environments where slower adaptation is desired. Experimentation and tuning are often necessary to find the optimal step-size sequence for a specific reinforcement learning task.

## Stochastic Approximation Algorithm

- In many real-world scenarios, especially in reinforcement learning, we often need to find the root $x^*$ of a function $f(x)$, i.e., solve
<div class="math-katex">
  $$
  f(x^*) = 0
  $$
</div>

However, we might not have direct access to the function $f(x)$. Instead, at each step $t$, we can only observe a noisy version of the function's value, denoted as $f(x_t) + y_t$, where $y_t$ is some random noise. The stochastic approximation algorithm provides an iterative method to find the root $x^*$ in such noisy environments.

- The update rule for the stochastic approximation algorithm is given by:
  $$x_{t+1} = x_t + \alpha_t(f(x_t) + y_t)$$
  where:

  - $x_t \in \mathbb{R}^d$: is the current estimate of the root at time step $t$. We start with an initial guess $x_0$.
  - $f(x_t)$: is the function whose root we are trying to find. In practice, we do not observe this value directly.
  - $y_t$: is the noise or random error at time step $t$, representing the uncertainty in our observation of $f(x_t)$.
  - $\alpha_t$: is the step-size parameter at time step $t$. It is a positive sequence that controls how much we update our estimate $x_t$ based on the noisy observation. The conditions for choosing $\alpha_t$ to ensure convergence were discussed in the previous section.

- Starting from an arbitrary initial estimate $x_0$, we iteratively apply this update rule. Intuitively, at each step, we are moving our current estimate $x_t$ in the direction suggested by the noisy observation $f(x_t) + y_t$, but with a step size regulated by $\alpha_t$.

- Under certain conditions on the step-size sequence, the function $f$, and the noise, it can be mathematically proven that the sequence of estimates converges to the root $x^*$ as $t \to \infty$.
This means:
  <div class="math-katex">
    $$ \lim_{t \to \infty} x_t = x^* $$
  </div>
  where $x^*$ is a solution to the equation $f(x) = 0$.

- The convergence of $x_t$ to $x^*$ is guaranteed if the following conditions are met:

  1. **Step-size conditions**: The step-size sequence satisfies the Robbins-Monro conditions:

     - $\sum_{t=0}^{\infty} \alpha_t = \infty$ (ensures we can reach any point in the parameter space)
     - $\sum_{t=0}^{\infty} \alpha_t^2 < \infty$ (ensures convergence by limiting the cumulative effect of noise)

     These conditions ensure that we take sufficiently large steps initially to reach the vicinity of the root, but gradually reduce the step size to stabilize around the root and avoid oscillations due to noise.

  2. **Properties of the function $f$**: The function $f$ needs to satisfy certain regularity conditions, such as being continuous and having a tendency to move towards the root $x^*$.
  3. **Bounded noise**: The noise sequence is typically assumed to be bounded in some probabilistic sense (e.g., having a finite variance). This ensures that the noise does not overwhelm the signal from the function $f$.

- This theoretical framework of stochastic approximation is crucial because it provides the foundation for understanding why our incremental algorithms, like the action-value estimation update rule, converge to the true values even when we are dealing with noisy reward signals in reinforcement learning environments. The next section will explicitly show how action-value estimation fits into this stochastic approximation framework.

![4](/content/posts/RL/4..PNG)

## Applying Stochastic Approximation to Action-Value Estimation

- Let's understand how the action-value update rule aligns with the stochastic approximation framework. Recall that our goal in action-value estimation is to estimate the true action value, denoted as $q^*(a)$, which is the expected reward when action $a$ is taken. Mathematically, the true action value is defined as:
  <div class="math-katex">
    $$q^*(a) = \mathbb{E}[R_{t+1} | A_t = a]$$
  </div>
    where:

  - $q^*(a)$ is the true expected reward for action $a$.
  - $R_{t+1}$ is the reward received at time $t+1$.
  - $A_t = a$ indicates that action $a$ was chosen at time $t$.
  - $\mathbb{E}[\cdot]$ denotes the expected value.

- We use the following incremental update rule to estimate $q^*(a)$:
  $$q_{t+1}(a) = q_t(a) + \alpha_t(R_{t+1} - q_t(a))$$
  where:

  - $q_{t+1}(a)$ is the updated estimate of the action value for action $a$ at time $t+1$.
  - $q_t(a)$ is the current estimate of the action value for action $a$ at time $t$.
  - $\alpha_t$ is the step-size parameter (learning rate) at time $t$, which satisfies the conditions for convergence discussed earlier.
  - $R_{t+1}$ is the reward received after taking action $a$ at time $t$.

- This action-value update rule can be viewed as an instance of the stochastic approximation algorithm:
  $$x_{t+1} = x_t + \alpha_t(f(x_t) + y_t)$$
  In our action-value estimation context, we can identify the components as follows:

  - **Estimate:** $x_t = q_t(a)$ is the current estimate of the action value for action $a$.
  - **Function:** $f(x) = \mathbb{E}[R_{t+1}|A_t=a] - x$. The function whose root we seek.

The root $x*$ such that $f(x^*) = 0$ is the true action value

<div class="math-katex">
  $$
  q^*(a) = \mathbb{E}[R_{t+1}|A_t=a]
  $$
</div>

- **Noise:** $y_t = R_{t+1} - \mathbb{E}[R_{t+1}|A_t=a]$. This represents the random fluctuation of the observed reward $R_{t+1}$ around its expected value.

- The stochastic approximation algorithm converges to the root $x^*$ of $f(x) = 0$. In our case, this implies that $q_t(a)$ converges to $q^*(a)$ as $t \to \infty$, because the root of $f(x) = \mathbb{E}[R_{t+1}|A_t=a] - x = 0$ is indeed $x^* = \mathbb{E}[R_{t+1}|A_t=a] = q^*(a)$.

- In summary, the action-value estimation update is a stochastic approximation method. By iteratively refining $q_t(a)$ using observed rewards and a suitable step-size $\alpha_t$, we achieve convergence to the true action value $q^*(a)$, even with noisy rewards. This connection to stochastic approximation provides a theoretical basis for the effectiveness of action-value estimation in reinforcement learning.

## Upper Confidence Bound (UCB) Algorithm

The Upper Confidence Bound (UCB) algorithm is a principled approach to address the exploration-exploitation dilemma in reinforcement learning and bandit problems. Unlike $\epsilon$-greedy strategies that explore randomly, UCB makes exploration decisions based on the uncertainty associated with action-value estimates. The core idea of UCB is to select actions that not only have high estimated values but also have high uncertainty, effectively exploring actions that have the potential to be optimal but haven't been sampled enough.

The action selection in UCB is governed by the following formula:

$$A_t = \arg\max_{a \in A} \left[Q_t(a) + U_t(a)\right]$$

where:

- $A_t$ is the action selected at time step $t$.
- $A$ is the set of all possible actions.
- $Q_t(a)$ is the current estimate of the action value for action $a$ at time $t$. This term represents the **exploitation** component, favoring actions that are currently believed to be good based on past experiences.
- $U_t(a)$ is the **Upper Confidence Bound** or uncertainty bonus for action $a$ at time $t$. This term represents the **exploration** component, encouraging the selection of actions about which we are uncertain. It is typically defined as:

  $$U_t(a) = c \sqrt{\frac{\ln t}{N_t(a)}}$$

  where:

  - $c > 0$ is an **exploration parameter** that controls the level of exploration. A higher value of $c$ increases the weight of the uncertainty bonus, leading to more exploration.
  - $t$ is the current time step (total number of steps so far). The term $\ln t$ (natural logarithm of $t$) ensures that exploration decreases over time but does so slowly.
  - $N_t(a)$ is the number of times action $a$ has been selected up to time step $t$. If $N_t(a)$ is small, the uncertainty bonus $U_t(a)$ is large, encouraging exploration of action $a$. As $N_t(a)$ increases, the uncertainty bonus decreases, shifting the focus towards exploitation.

**Key Properties of UCB:**

- **Balances Exploration and Exploitation**: UCB explicitly balances exploration and exploitation by considering both the estimated value ($Q_t(a)$) and the uncertainty ($U_t(a)$) of each action.
- **Optimistic Exploration**: UCB is based on the principle of "optimism in the face of uncertainty." It adds an optimistic bonus ($U_t(a)$) to the estimated value, encouraging the agent to try actions that have not been tried often and thus have high uncertainty.
- **Decreasing Exploration over Time**: As time progresses ($t$ increases) and actions are selected more frequently ($N_t(a)$ increases), the uncertainty bonus $U_t(a)$ decreases (due to $\ln t$ growing slower than $N_t(a)$ in the long run). This naturally reduces exploration over time, allowing the algorithm to converge towards exploiting the optimal action.
- **Parameter Sensitivity**: The exploration parameter $c$ plays a crucial role in controlling the exploration-exploitation trade-off. Tuning $c$ appropriately is important for good performance. A larger $c$ leads to more exploration, which can be beneficial in the initial stages but may slow down convergence later. A smaller $c$ leads to less exploration, which might result in faster initial exploitation but could lead to suboptimal solutions if the initial estimates are inaccurate.
- **Logarithmic Regret**: Under certain conditions, UCB algorithms are known to achieve logarithmic regret, which means that the cumulative loss compared to the optimal policy grows logarithmically with time. This is a desirable property, indicating efficient learning and convergence to near-optimal performance.

The UCB algorithm provides a more sophisticated and theoretically grounded approach to exploration compared to simple methods like $\epsilon$-greedy, making it a valuable tool in various reinforcement learning and decision-making problems.

## Gradient Bandit Algorithms

Gradient Bandit algorithms offer an alternative approach to action selection by learning numerical preferences for each action, rather than estimating action values directly. These preferences determine the probability of selecting each action.

- **Action Preferences $H_t(a)$**:

  - **Mathematical Definition**: For each action $a \in A$ and at each time step $t$, we maintain a numerical preference $H_t(a) \in \mathbb{R}$.
  - **Intuitive Explanation**: $H_t(a)$ represents a learned score indicating the desirability of selecting action $a$. A higher preference for an action means it is considered more favorable to choose relative to other actions. These preferences are not action-value estimates; they are simply numerical values that guide action selection probabilities.

- **Softmax Distribution for Action Probabilities $\pi_t(a)$**:

  - **Mathematical Definition**: The probability of selecting action $a$ at time $t$, denoted as $\pi_t(a) = P(A_t=a)$, is determined by a softmax distribution over the action preferences:
    $$\pi_t(a) = P(A_t=a) = \frac{e^{H_t(a)}}{\sum_{b=1}^k e^{H_t(b)}}$$
    where $k$ is the total number of available actions.
  - **Intuitive Explanation**:
    - The softmax function transforms the preferences $H_t(a)$ into probabilities $\pi_t(a)$.
    - Exponentiating the preferences ($e^{H_t(a)}$) ensures that actions with higher preferences have exponentially higher probabilities of being selected.
    - Normalizing by the sum of exponentiated preferences ($\sum_{b=1}^k e^{H_t(b)}$) ensures that the probabilities for all actions sum to 1, forming a valid probability distribution.
    - This probabilistic selection mechanism naturally incorporates exploration. Even actions with lower preferences have a non-zero probability of being chosen, allowing the algorithm to explore different options.

- **Preference Update Rule via Stochastic Gradient Ascent**:

  - **Mathematical Definition**: Preferences are updated using a stochastic gradient ascent rule. For the action $A_t$ selected at time $t$ and receiving reward $R_t$, the preferences are updated as follows:
    $$H_{t+1}(a) = H_t(a) + \alpha(R_t - \bar{R}_t)(I_{a=A_t} - \pi_t(a)) \quad \forall a \in A$$
    where:
    - $\alpha > 0$ is the step-size parameter (learning rate), controlling the magnitude of preference updates.
    - $R_t$ is the reward received at time $t$ after taking action $A_t$.
    - $\bar{R}_t$ is the baseline reward at time $t$, typically the average of all rewards received up to time $t$.
    - $I_{a=A_t}$ is an indicator function that is 1 if $a = A_t$ (i.e., if action $a$ was selected at time $t$), and 0 otherwise.
    - $\pi_t(a)$ is the probability of selecting action $a$ at time $t$ as defined by the softmax distribution.
  - **Intuitive Explanation**:
    - **Gradient Ascent**: The update rule is derived from stochastic gradient ascent, aiming to maximize the expected reward. Although the derivation is skipped here, the update moves the preferences in a direction that is estimated to increase future rewards.
    - **Reward Prediction Error $(R_t - \bar{R}_t)$**: This term represents how much better or worse the received reward $R_t$ is compared to the average reward $\bar{R}_t$. It serves as a signal for adjusting preferences.
      - If $R_t > \bar{R}_t$ (reward is better than average), we want to increase the preference for the action $A_t$ that led to this reward and decrease preferences for other actions.
      - If $R_t < \bar{R}_t$ (reward is worse than average), we want to decrease the preference for action $A_t$ and potentially increase preferences for other actions.
    - **Action-Specific Update $(I_{a=A_t} - \pi_t(a))$**: This term dictates how preferences for each action are adjusted:
      - **For the selected action $A_t$**: The update term becomes $(1 - \pi_t(A_t))$. Since $\pi_t(A_t) \le 1$, this term is always non-negative. If the reward prediction error $(R_t - \bar{R}_t)$ is positive, the preference $H_{t+1}(A_t)$ for the selected action increases.
      - **For unselected actions $a \ne A_t$**: The update term becomes $(0 - \pi_t(a)) = -\pi_t(a)$. This term is always negative. If the reward prediction error $(R_t - \bar{R}_t)$ is positive, the preference $H_{t+1}(a)$ for unselected actions decreases.
    - In essence, if the reward is better than baseline, we reinforce the action taken and weaken others. If the reward is worse, we weaken the action taken and strengthen others (relatively, through the softmax probability recalculation in the next step).

- **Baseline Reward $\bar{R}_t$**:

  - **Mathematical Definition**: The baseline reward $\bar{R}_t$ is typically calculated as the average of all rewards received up to time $t$:
    $$\bar{R}_t = \frac{1}{t} \sum_{i=1}^{t} R_i$$
  - **Intuitive Explanation**:
    - The baseline $\bar{R}_t$ provides a reference point to evaluate the received reward $R_t$. It represents the average performance of the bandit algorithm so far.
    - Using a baseline reduces the variance of the preference updates. This leads to more stable learning and can speed up convergence.
    - The baseline helps the algorithm to differentiate between actions that are truly good (better than average) and those that are just average or below average.

- **Key Aspects of Gradient Bandit Algorithms**:
  - **Relative Preferences**: Gradient bandits learn relative preferences between actions rather than absolute value estimates. This allows them to naturally handle non-stationary environments where the absolute values of actions might change over time, but their relative rankings might be more stable.
  - **Exploration via Probability Distribution**: The softmax distribution ensures inherent exploration. The probability of selecting each action is continuously adjusted based on learned preferences, allowing for a smooth balance between exploration and exploitation without explicitly setting exploration parameters like in $\epsilon$-greedy or UCB.
  - **Direct Preference Updates**: Updates directly adjust action preferences based on reward prediction errors, moving preferences towards actions that yield better-than-average rewards and away from those that yield worse-than-average rewards.
  - **Variance Reduction with Baseline**: The baseline $\bar{R}_t$ significantly reduces the variance of updates, leading to more stable and reliable learning, especially in noisy reward environments.
  - **Adaptability to Non-Stationarity**: Gradient bandit algorithms can adapt relatively quickly to changes in the reward distribution because the preferences are continuously updated based on recent rewards and the baseline adapts over time to reflect the changing average reward.
