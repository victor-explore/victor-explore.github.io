---
title: "Expectation Maximization (EM) algorithm"
date:
draft: false
description:
tags: []
categories: []
author:
toc:
weight: 1
---
Let the data be $D = \{t_i\}_{i=1}^N$ where $t_i \in \mathbb{R}^d$ are iid $p_t(t)$ data points.

## Latent variable models

Let each data point $x_i$ be associated with a latent variable $z_i$ that is not observed. $z_i$ is a random variable that takes values in some finite set ${1, \ldots, K}$ and represents the membership of $x_i$ to one of $K$ clusters.

Hence the data can be represented as 
<div class="math-katex">
$$D = \{(t_i, z_i)\}_{i=1}^N$$
</div>
 where $t_i$ is the observed data and $z_i$ is the latent variable and $(t_i,z_i)$ are iid $p_{tz}$.

The marginal distribution of the observed data is given by:

$$
p_t(t) = \sum_z p_{tz}(t, z)
$$

## Variational inference
We want to maximize the log-likelihood of the observed data:


$$
\begin{aligned}
\ell(\theta) &= \log \sum_z p_{tz}^\theta(t, z) \\
\end{aligned}
$$
Let $q(z)$ be an arbitrary distribution over $z$.

<div class="math-katex">
$$
\begin{aligned}
\ell(\theta) &= \log \sum_z q(z) \frac{p_{tz}^\theta(t, z)}{q(z)} \\
\end{aligned}
$$
</div>
<div class="math-katex">
$$
\ell(\theta) = \log \mathbb{E}_{q(z)} \left[ \frac{p_{tz}^\theta(t, z)}{q(z)} \right]
$$
</div>
By Jensen's inequality, we have:
<div class="math-katex">
$$
\log \mathbb{E}_{q(z)} \left[ \frac{p_{tz}^\theta(t, z)}{q(z)} \right] \geq \mathbb{E}_{q(z)} \left[ \log \frac{p_{tz}^\theta(t, z)}{q(z)} \right]
$$
</div>
Hence, we have:
<div class="math-katex">
$$
\ell(\theta) \geq \mathbb{E}_{q(z)} \left[ \log \frac{p_{tz}^\theta(t, z)}{q(z)} \right]
$$
</div>
$\ell(\theta)$ is the log-likelihood of the observed data it is also called the evidence.
<div class="math-katex">
$\mathbb{E}_{q(z)} \left[ \log \frac{p_{tz}^\theta(t, z)}{q(z)} \right]$ is the lower bound of the log-likelihood and is also called the evidence lower bound (ELBO).
</div>
Note that the ELBO is a function of $q(z)$ and $\theta$.

## Optimizing ELBO
Instead of maximizing the evidence, we maximize the evidence lower bound (ELBO). We do it by maximizing the ELBO with respect to $q(z)$ and $\theta$.

## Making ELBO tight

To make the ELBO tight, we consider the difference between the evidence and the ELBO:
<div class="math-katex">
$$
\begin{aligned}
\ell(\theta) - \text{ELBO}(q, \theta) &= \log p_t^\theta - \mathbb{E}_{q(z)} \left[ \log \frac{p_{tz}^\theta(t, z)}{q(z)} \right] \\
\end{aligned}
$$
</div>
<div class="math-katex">
$$
\begin{aligned}
\ell(\theta) - \text{ELBO}(q, \theta) &= \log p_t^\theta - \mathbb{E}_{q(z)} \left[ \log \frac{p_{z|t}^\theta(z|t)p_t^\theta(t)}{q(z)} \right] \\
\end{aligned}
$$
</div>
<div class="math-katex">
$$
\begin{aligned}
\ell(\theta) - \text{ELBO}(q, \theta) &= \log p_t^\theta - \mathbb{E}_{q(z)} \left[ \log p_{z|t}^\theta(z|t) + \log p_t^\theta(t) - \log q(z) \right] \\
\end{aligned}
$$
</div>

<div class="math-katex">
$$
\begin{aligned}
\ell(\theta) - \text{ELBO}(q, \theta) &= \log p_t^\theta - \mathbb{E}_{q(z)} \left[ \log p_{z|t}^\theta(z|t) \right] - \mathbb{E}_{q(z)} \left[ \log p_t^\theta(t) \right] + \mathbb{E}_{q(z)} \left[ \log q(z) \right] \\
\ell(\theta) - \text{ELBO}(q, \theta) &= \log p_t^\theta - \mathbb{E}_{q(z)} \left[ \log p_{z|t}^\theta(z|t) \right] - \log p_t^\theta(t) + \mathbb{E}_{q(z)} \left[ \log q(z) \right] \\
\ell(\theta) - \text{ELBO}(q, \theta) &= -\mathbb{E}_{q(z)} \left[ \log p_{z|t}^\theta(z|t) \right] + \mathbb{E}_{q(z)} \left[ \log q(z) \right] \\
\ell(\theta) - \text{ELBO}(q, \theta) &= \mathbb{E}_{q(z)} \left[ \log q(z) - \log p_{z|t}^\theta(z|t) \right] \\
\ell(\theta) - \text{ELBO}(q, \theta) &= \mathbb{E}_{q(z)} \left[ \log \frac{q(z)}{p_{z|t}^\theta(z|t)} \right] \\
\ell(\theta) - \text{ELBO}(q, \theta) &= KL(q(z) || p_{z|t}^\theta(z|t))
\end{aligned}
$$
</div>

The difference between the evidence and the ELBO is the Kullback-Leibler (KL) divergence between $q(z)$ and $p_{z|t}^\theta(z|t)$. To make the ELBO tight, we need to minimize this KL divergence.

The KL divergence is always non-negative and equals zero if and only if the two distributions are identical. Therefore, to make the ELBO tight, we set:

$$
q(z) = p_{z|t}^\theta(z|t)
$$

This choice of $q(z)$ makes the ELBO equal to the evidence, achieving the tightest possible bound.

## EM Algorithm

The Expectation-Maximization (EM) algorithm is an iterative method to find maximum likelihood estimates of parameters in statistical models with latent variables. It consists of two main steps:

1. **E-step (Expectation)**: Compute the expected value of the log-likelihood function with respect to the conditional distribution of $z$ given $t$ under the current estimate of the parameters $\theta$.

2. **M-step (Maximization)**: Find the parameter that maximizes this expected log-likelihood.

Formally, the EM algorithm can be described as follows:

1. Initialize $\theta^{(0)}$
2. Repeat until convergence:
   - E-step: Compute $q^{(t)}(z) = p_{z|t}^{\theta^{(t-1)}}(z|t)$
   - M-step: 
     1. <div class="math-katex">$\theta^{(t)} = \arg\max_\theta \mathbb{E}_{q^{(t)}(z)} \left[ \log \frac{p_{tz}^\theta(t, z)}{q(z)} \right]$</div>
     2. <div class="math-katex">$\theta^{(t)} = \arg\max_\theta \mathbb{E}_{q^{(t)}(z)} \left[ \log p_{tz}^\theta(t,z) \right] - \mathbb{E}_{q^{(t)}(z)} \left[ \log q(z) \right]$</div>
     3. <div class="math-katex">$\theta^{(t)} = \arg\max_\theta \mathbb{E}_{q^{(t)}(z)} \left[ \log p_{tz}^\theta(t,z) \right]$ as second term is constant wrt $\theta$</div>

The EM algorithm guarantees that the likelihood increases at each iteration and converges to a local maximum.


## EM algorithm for GMM

Let's apply the EM algorithm to the Gaussian Mixture Model (GMM) we discussed earlier. Recall that in a GMM, we have:

- Observed data points: $\mathbf{x} = (x_1, ..., x_N)$
- Latent variables: $\mathbf{z}$, where $z_i \in \{1, ..., m\}$ indicates gaussian component
- $p_t(t) = \sum_{j=1}^m \alpha_j \mathcal{N}(t; \mu_j, \xi_j)$
- $\alpha_j$ are mixing coefficients, $\sum_{j=1}^m \alpha_j = 1$
- $\mu_j$ are mean vectors
- $\xi_j$ are covariance matrices
- $p(t_i | z_i = k) = \mathcal{N}(\mathbf{t_i}; \mu_k, \Sigma_k)$
- Parameters: $\theta = (\alpha_1, ..., \alpha_m, \mu_1, ..., \mu_m, \xi_1, ..., \xi_m)$

The EM algorithm for GMM proceeds as follows:

1. **Initialization**: 
   Choose initial values for the parameters $\theta = (\alpha_1, ..., \alpha_m, \mu_1, ..., \mu_m, \xi_1, ..., \xi_m)$.

2. **E-step**: 
   Compute the posterior probabilities (responsibilities) for each data point and each Gaussian component:

   $$
   q^{t+1}(z_i=k) = p_{z|t}^{\theta^t}(z_i=k|t_i) = \frac{p_{tz}^{\theta^t}(t_i, z_i=k)}{p_t^{\theta^t}(t_i)} = \frac{\alpha_k^t \mathcal{N}(t_i; \mu_k^t, \xi_k^t)}{\sum_{j=1}^m \alpha_j^t \mathcal{N}(t_i; \mu_j^t, \xi_j^t)}
   $$

   This equation represents the responsibility that the k-th component takes for explaining the i-th data point. Here:
   - $q^{t+1}(z_i=k)$ is the posterior probability of the i-th data point belonging to the k-th Gaussian component
   - $\alpha_k^t$ is the mixing coefficient for the k-th component at iteration t
   - $\mathcal{N}(t_i; \mu_k^t, \xi_k^t)$ is the probability density of $t_i$ under the k-th Gaussian component
   - The denominator normalizes the probabilities to ensure they sum to 1 across all components

3. **M-step**: 
   Update the parameters:

   $$
   \theta^{k+1} = \arg\max_\theta ELBO(q, \theta)
   $$
   <div class="math-katex">
   $$
   \theta^{k+1} = \arg\max_\theta \mathbb{E}_{q(z)} \left[ \log p_{tz}^\theta(t,z) \right]
   $$
   </div>
   $$
   = \arg\max_\theta \sum_{i=1}^N \sum_{k=1}^m q(z_i=k) \log \left( \alpha_k \mathcal{N}(t_i; \mu_k, \xi_k) \right)
   $$

   This leads to the following update equations:

   $$
   \alpha_k^{new} = \frac{1}{N} \sum_{i=1}^N q(z_i=k) \quad \text{(New mixing coefficients)}
   $$
   $$
   \mu_k^{new} = \frac{\sum_{i=1}^N q(z_i=k) t_i}{\sum_{i=1}^N q(z_i=k)} \quad \text{(New means)}
   $$
   $$
   \xi_k^{new} = \frac{\sum_{i=1}^N q(z_i=k) (t_i - \mu_k^{new})(t_i - \mu_k^{new})^T}{\sum_{i=1}^N q(z_i=k)} \quad \text{(New covariances)}
   $$

The EM algorithm for GMM alternates between these steps until convergence, effectively maximizing the likelihood of the observed data under the Gaussian mixture model.





