---
title: "Value Iteration"
date: 2025-01-01
draft: false
description:
tags: []
categories: []
author:
toc:
weight: 1
---

Value iteration is a numerical method for solving Stochastic Shortest Path (SSP) problems that relies on the dynamic programming operator $T_\mu$. Let's examine how this operator works and its key properties:

### The Dynamic Programming Operator $T_\mu$

For a given policy $\mu$, the dynamic programming operator $T_\mu$ transforms one value function into another. Mathematically, for any state $x$, we have:

<div class="math-block">
$$
\underbrace{(T_\mu J)(x)}_{\substack{\text{New value function} \\ \text{at state } x}} = \underbrace{E_{x_{k+1}}}_{\substack{\text{Expected value over} \\ \text{next states}}} \left[ \underbrace{g(x,\mu(x),x_{k+1})}_{\substack{\text{Stage cost under} \\ \text{policy } \mu}} + \underbrace{J(x_{k+1})}_{\substack{\text{Cost-to-go from} \\ \text{next state}}} \biggm\vert x_k=x \right]
$$
</div>

where:

- $g(x,\mu(x),x_{k+1})$ represents the immediate cost incurred when taking action $\mu(x)$ in state $x$ and transitioning to $x_{k+1}$
- $J(x_{k+1})$ captures the future cost-to-go starting from the next state $x_{k+1}$
- The expectation $E$ averages over all possible next states according to the system's transition probabilities

### Properties of $T_\mu$

1. **Monotonicity**: If $J_1 \leq J_2$, then $T_\mu J_1 \leq T_\mu J_2$

   - This means the operator preserves the ordering of value functions

2. **Contraction**: For any two value functions $J_1$ and $J_2$:
   <div class="math-block">
   $$
   \underbrace{\|T_\mu J_1 - T_\mu J_2\|}_{\substack{\text{Distance between} \\ \text{transformed functions}}} \leq \underbrace{\alpha}_{\substack{\text{Contraction} \\ \text{factor}}} \underbrace{\|J_1 - J_2\|}_{\substack{\text{Distance between} \\ \text{original functions}}}
   $$
   </div>

   where $\alpha < 1$ is a contraction factor

### Value Iteration with $T_\mu$

The algorithm proceeds as follows:

1. **Initialize**: Start with any value function $J_0$

2. **Iterate**: For $k = 0,1,2,\ldots$

<div class="math-block">
$$
\underbrace{J_{k+1}}_{\substack{\text{Updated value} \\ \text{function}}} = \underbrace{T_\mu J_k}_{\substack{\text{Application of} \\ \text{DP operator}}}
$$
</div>

3. **Convergence**: The sequence $\{J_k\}$ converges to the optimal cost-to-go function $J_\mu^*$ for policy $\mu$:
<div class="math-block">
$$
\underbrace{\lim_{k \to \infty} T_\mu^k J}_{\substack{\text{Repeated application} \\ \text{of operator}}} = \underbrace{J_\mu^*}_{\substack{\text{Optimal} \\ \text{cost-to-go}}}
$$
</div>

This convergence is guaranteed by the contraction property of $T_\mu$, making value iteration a reliable method for computing optimal value functions in SSP problems. The rate of convergence is geometric, with error reducing by at least a factor of $\alpha$ in each iteration.

## Gauss - Seidel value iteration

The Gauss-Seidel value iteration is a modification of the standard value iteration algorithm that uses the most recently computed values as soon as they become available, rather than waiting for a complete iteration to finish.

### Algorithm Description

For a given state $i$, the Gauss-Seidel value iteration operator $F$ is defined as:

For $i=1$:

$$
FJ(1) = \min_{u \in \mathcal{A}(1)} \left( \sum_{j=1}^n p_{1j}(u) \left(g(1,u,j) + J(j)\right) \right)
$$

For $i=2,\ldots,n$:

$$
FJ(i) = \min_{u \in \mathcal{A}(i)} \left(\sum_{j=1}^{n} p_{ij}(u)(g(i,u,j) + \sum_{j=i}^{i-1} p_{ij}(u)FJ(j)) + \sum_{j=i}^n p_{ij}(u)J(j))\right)
$$

where:

- $FJ(i)$ represents the updated cost-to-go for state $i$
- $\mathcal{A}(i)$ represents the set of allowable actions in state $i$
- $p_{ij}(u)$ represents the transition probability from state $i$ to state $j$ under action $u$
- $g(i,u,j)$ represents the stage cost when transitioning from state $i$ to $j$ under action $u$

### Key Properties

1. **Monotonicity**: Like the standard Bellman operator $T$, the Gauss-Seidel operator $F$ is monotone:

   $$
   J \leq \bar{J} \implies FJ \leq F\bar{J}
   $$

2. **Contraction**: $F$ is a contraction mapping with respect to the weighted maximum norm $\|\cdot\|_\beta$ with the same contraction factor $\beta$ as $T$:
<div class="math-block">
$$
\|FJ - F\bar{J}\|_\beta \leq \beta\|J - \bar{J}\|_\beta
$$
</div>

3. **Convergence**: Due to the contraction property, Gauss-Seidel value iteration converges to the unique fixed point $J^*$ at a geometric rate:
<div class="math-block">
$$
\lim_{k \to \infty} F^kJ = J^*
$$
</div>

### Advantages

- Faster empirical convergence compared to standard value iteration due to immediate incorporation of updated values
- Same computational complexity per iteration as standard value iteration ($O(n^2|A|)$ where $|A|$ is the maximum number of actions)
- Memory efficient due to in-place updates ($O(n)$ space complexity)

### Disadvantages

- Inherently sequential computation makes parallelization challenging
- Convergence rate depends on state ordering and problem structure
- May require careful implementation to handle numerical stability issues
