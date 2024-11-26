---
title: "Cheat sheet"
date:
draft: false
description:
tags: []
categories: []
author:
toc:
weight: 1
---

## Formulas:

- **K L divergence**: $D_{KL}(p; q) = \int p(x) \log \frac{p(x)}{q(x)} dx$

- **F-divergence**: $D_{F}(p \parallel q) = \int q(x) f\left(\frac{p(x)}{q(x)}\right) dx$
  - $f: \mathbb{R}^+ \rightarrow \mathbb{R}$ is lower semi continous convex function such that $f(1) = 0$
  - If $f(x) = x \log x$, then $D_{F}(p \parallel q) = D_{KL}(p \parallel q)$
  - If $f(x) = x\log(x) - (x+1)\log(x+1)$, then $f^*=-\log(1-\exp^x)$, then $D_{F}(p \parallel q)$ = Jensen-Shannon divergence = $D_{JS}(p \parallel q) = \frac{1}{2}D_{KL}(p \parallel m) + \frac{1}{2}D_{KL}(q \parallel m)$, where $m = \frac{1}{2}(p + q)$

- **Jensen's inequality**: For a convex function $f$ and a random variable $X$: $f(\mathbb{E}[X]) \leq \mathbb{E}[f(X)]$
  - Used to prove F-divergence is non-negative
  - Used in VAEs as latent space models


- **Convex functions**: Let f: $\mathbb{R}^N \rightarrow \mathbb{R}$ be a convex function. Then f is convex if and only if $f(\lambda x + (1-\lambda) y) \leq \lambda f(x) + (1-\lambda)f(y)$

- **Convex conjugate**: Let f: $\mathbb{R}^N \rightarrow \mathbb{R}$ be a convex function. The convex conjugate of f is the function $f^*(y) = \sup_x (y^Tx - f(x))$
  - <div class="math-katex">$f^{**}(x)$ is also convex, i.e: $f^{**}(\lambda x + (1-\lambda) y) \leq \lambda f^{**}(x) + (1-\lambda)f^{**}(y)$</div>
  - $f^{**}(y) = f(y)$

- **Wasserstein metric**: The Wasserstein distance $W_p(P,Q)$ of order $p$ is defined as:$W_p(P,Q) = \left( \inf_{\gamma \in \Gamma(P,Q)} \mathbb{E}_{(x,y) \sim \gamma} [d(x,y)^p] \right)^{1/p}$

- **K - Lipschitz**: K lipschitz is defined as: $|f(x) - f(y)| \leq K |x-y|$

- **FID**: $FID = \|\mu_r - \mu_g\|^2 + \operatorname{Tr}(\Sigma_r + \Sigma_g - 2\sqrt{\Sigma_r \cdot \Sigma_g})$

- **GANs:**
  - <div class="math-katex">$\theta^*_{GAN} = \underset{\theta}{\text{argmin}} \underset{\phi}{\sup} \left[\mathbb{E}_{x \sim p_{data}}[T_\phi(x)] - \mathbb{E}_{x \sim p_{generator}}[f^*(T_\phi(x))]\right]$</div>
  - <div class="math-katex">$\theta^*_{NaiveGAN} = \underset{\theta}{\text{argmin}} \underset{\phi}{\sup} \left[\mathbb{E}_{x \sim p_{data}}[\log(D_\phi(x))] + \mathbb{E}_{x \sim p_{generator}}[\log(1-D_\phi(x))]\right]$</div>
  - <div class="math-katex">$\theta^*_{WGAN} = \underset{\theta}{\argmin} \, \sup_{f \in \text{Lip}_1} [\mathbb{E}_{x \sim P_{data}}[f(x)] - \mathbb{E}_{x \sim P_{generator}}[f(x)]]$</div>
  - <div class="math-katex">$\theta^*_{CGAN} = \underset{\theta}{\text{argmin}} \underset{\phi}{\sup}[\mathbb{E}_{x \sim p_{data}(x|y)}[\log D(x|y)] + \mathbb{E}_{z \sim p_z(z)}[\log(1 - D(G(z|y)|y))]]$</div>
  - <div class="math-katex">$\mathcal{L}_{DA} = \mathbb{E}_{x^s \sim p_s(x)}[\log(D(\phi(x^s)))] + \mathbb{E}_{x^t \sim p_t(x)}[\log(1 - D(\phi(x^t)))]$</div>

- **VAEs:**
  - <div class="math-katex">**$ELBO_{VAE}$** = $F_{\theta}(q) = \mathbb{E}_{q(z|x)} \left[ \log p_{\theta}(x|z) \right] - D_{KL} \left( q(z|x) \mid p_{\theta}(z) \right) $</div>
    - <div class="math-katex">**First term**: $ \nabla_{\phi} \mathbb{E}_{q(z|x)}[\log p_{\theta}(x|z)] \approx \frac{1}{N} \sum_{i=1}^N \nabla_{\phi} [\log p_{\theta}(x|g_{\phi}(\epsilon_i))]$ where  $\epsilon \sim p(\epsilon)$ (typically a standard normal distribution) and $g_{\phi}(\epsilon)$ is our reparameterization function typically $z=g_{\phi}(\epsilon) = \mu_{\phi}(x) + \sigma_{\phi}(x) \odot \epsilon$</div>
    - <div class="math-katex">**Second term**: $ D_{KL}(N(z; \mu_\phi(x), \Sigma_\phi(x)) \| N(0, I)) = \frac{1}{2} \sum_{j=1}^J \left( \mu_{\phi,j}^2(x) + \Sigma_{\phi,j}(x) - \log \Sigma_{\phi,j}(x) - 1 \right) $</div>
    - <div class="math-katex">**Aggregated posterior mismatch**: $ q_{\phi}(z) = \int q_{\phi}(z|x) p(x) dx \neq \int p_{\theta}(z|x) p(x) dx = p_{\theta}(z) $</div>
  - <div class="math-katex">**$ELBO_{Regularized VAEs(Beta VAEs)}$** = $ F_{\theta}(q) = \mathbb{E}_{q_{\phi}(z|x)} \left[ \log p_{\theta}(x|z) \right] - \beta D_{KL} \left( q_{\phi}(z|x) \| p(z) \right) $</div>
  - <div class="math-katex">**$ELBO_{InfoVAE}$**: $ =  \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] -\lambda D_{KL}(q_\phi(z) \| p(z)) + \alpha I_q(x;z) $ where $I_q(x;z) = \mathbb{E}{p_D(x)}[D{KL}(q_\phi(z|x) | q_\phi(z))]$</div>

- **DDPMs:**
  - **Forward diffusion process**: $X_t = \sqrt{\alpha_t} X_{t-1} + \sqrt{1-\alpha_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0,I)$
  - $ X_t = \sqrt{\bar{\alpha}_t}X_0 + \sqrt{1-\bar{\alpha}_t}\epsilon $ where $ \bar{\alpha}_t = \prod_{i=1}^t \alpha_i $
  - **$ELBO_{DDPMs}$** = <div class="math-katex">$\mathbb{E}_{q(X_{1:T}|X_0)}\left[\log p_\theta(X_0|X_1) + \log \frac{p(X_T)}{q(X_T|X_0)} + \sum_{t=2}^T \log \left(\frac{p_\theta(X_{t-1}|X_t)}{q(X_{t-1}|X_t,X_0)}\right)\right]$</div> ie reconstruction term + prior matching term + transition matching term

- **Score matching**:
  - **Langevin equation**: $x_{t+1} = x_t + \alpha \cdot \nabla_\theta \log p_\theta(x_t) + \sqrt{2\alpha} \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$
  - **Score function**: $\nabla_x \log p(x) = s(x)$
  - **Explicit score matching**: $J_{ESM}(\theta) = \mathbb{E}_{p(x)} \left[ \frac{1}{2} \| \hat{s}(x; \theta) - \nabla_x \log p(x) \|^2 \right]$
  - **Implicit score matching**:  $J_{ISM}(\theta) = \mathbb{E}_{p(x)} \left[ \frac{1}{2} \| \hat{s}(x; \theta) \|^2 + \text{tr}(\nabla_x \hat{s}(x; \theta)) \right] + C$
  - **Theorem**: $J_{ISM}(\theta) = J_{ESM}(\theta) + C$
  - **Projected score matching**: $J_{PSM}(\theta) = \frac{1}{2} \mathbb{E}_v \mathbb{E}_{p(x)} \left[ \| v^\top \hat{s}_\theta(x) - v^\top s(x) \|^2 \right]$
  - **Sliced score matching**: $J_{SSM}(\theta) = \mathbb{E}_v \mathbb{E}_{p(x)} \left[ \frac{1}{2}(v^\top \hat{s}_\theta(x))^2 + v^\top (\nabla_x \hat{s}_\theta(x)) v \right]$
  - **Theorem**: $J_{PSM}(\theta) = J_{SSM}(\theta) + C$
  - **Deep score matching**: $J_{DSM}(\theta) = \frac{1}{2} \mathbb{E}_{p(x,x')} \left[ \| \hat{s}_\theta(x) - \nabla_{x} \log p(x|x') \|^2 \right]$
  - **Theorem**: $J_{ESM}(\theta) = J_{DSM}(\theta) + C$

- **Conditional score matching**:
  - **Classifier guided**: $ s(x|y) = s(y|x) + s(x) $
  - **Classifier free**: $s_\theta(x|y)_{CFG} = (1+w)s_\theta(x|y) - w s_\theta(x|\text{null})$

- **Stochastic differential equation**:
  - **Standard Stochastic differential equation**: $dx = f(x,t)dt + g(x,t)dB_t$ such that $dB_t = \sqrt{dt}\epsilon \quad \text{where} \quad \epsilon \sim N(0,1)$
  - **Anderson's result**: The corresponding reverse SDE takes the form: $dx = \left(f(x,t) - g^2(t)\nabla_x(\log p_t(x))\right)dt + g(t)dB_t$
  - **Big picture**: 
    - Stochastic discrete forward process to continuous SDE
    - Continuous SDE to continuous reverse SDE
    - Continuous reverse SDE to discrete reverse process

  - **DDIM**: Left
  - **State space model**: Left

  - **Self supervised learning**:
    - **Noise contrastive estimation**: $J(\theta) = \frac{1}{2} \mathbb{E} [\log \sigma(\log p_\theta(x) - \log p_N(x)) + \log(1-\sigma(\log p_\theta(y) - \log p_N(y)))]$
    - **Info NCE**: $\mathcal{L}_{\text{InfoNCE}} = -\mathbb{E}\left[\log \frac{\exp(f_\theta(x)^\top f_\theta(x^+))}{\exp(f_\theta(x)^\top f_\theta(x^+)) + \sum_{i=1}^{N-1} \exp(f_\theta(x)^\top f_\theta(x^-_i))}\right]$
    - **Masked reconstruction**: $\mathcal{L}(\theta) = \mathbb{E}_{x \sim p_\text{data}}\left[d(x, T_\theta(\text{mask}(x)))\right]$
    - **JEPA**: $\mathcal{L}(\theta,\phi,\psi) = \mathbb{E}_{(x_c,x_t) \sim p_\text{data}}\left[\|z_t - \hat{z}_t\|^2\right]$

- **Gradient**: For a scalar-valued differentiable function $f(\mathbf{x})$ of vector $\mathbf{x} \in \mathbb{R}^n$, the gradient $\nabla_{\mathbf{x}} f(\mathbf{x})$ is:
  $$\nabla_{\mathbf{x}} f(\mathbf{x}) = \begin{bmatrix}
  \frac{\partial f}{\partial x_1} \\
  \frac{\partial f}{\partial x_2} \\
  \vdots \\
  \frac{\partial f}{\partial x_n}
  \end{bmatrix}$$

- **Jacobian**: If $f(\mathbf{x})$ is a vector-valued function of vector $\mathbf{x}$ ie $f: \mathbb{R}^n \rightarrow \mathbb{R}^m$, then the Jacobian matrix contains the partial derivatives of each component of $f$ with respect to each component of $\mathbf{x}$:
  $$J = \begin{bmatrix}
  \frac{\partial f_1}{\partial x_1} & \frac{\partial f_1}{\partial x_2} & \cdots & \frac{\partial f_1}{\partial x_n} \\
  \frac{\partial f_2}{\partial x_1} & \frac{\partial f_2}{\partial x_2} & \cdots & \frac{\partial f_2}{\partial x_n} \\
  \vdots & \vdots & \ddots & \vdots \\
  \frac{\partial f_m}{\partial x_1} & \frac{\partial f_m}{\partial x_2} & \cdots & \frac{\partial f_m}{\partial x_n}
  \end{bmatrix}$$

- **Hessian**: The Hessian matrix is the second derivative of scalar valued function $f$ with respect to each component of $\mathbf{x}$, and is a matrix of size $n \times n$. In simple words, hessian is just jacobian of gradient of $f$. It is defined as:
  $$H = \begin{bmatrix}
  \frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1 \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_1 \partial x_n} \\
  \frac{\partial^2 f}{\partial x_2 \partial x_1} & \frac{\partial^2 f}{\partial x_2^2} & \cdots & \frac{\partial^2 f}{\partial x_2 \partial x_n} \\
  \vdots & \vdots & \ddots & \vdots \\
  \frac{\partial^2 f}{\partial x_n \partial x_1} & \frac{\partial^2 f}{\partial x_n \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_n^2}
  \end{bmatrix}$$

## Distributions
### Bernoulli distribution
- Models a random variable which takes one of two values, 0 (failure) or 1 (success). Means $x \in \{0, 1\}$.
- It is defined by a single parameter $\mu$ which is the probability of success. This automatically means that the probability of failure is $1-\mu$.
- The probability mass function is given by:
  - $p(x; \mu) = \mu^x (1-\mu)^{1-x}$ for $x \in \{0, 1\}$

### Binomial distribution
- Models the number of successes in a fixed number of independent Bernoulli trials.
- It is defined by two parameters: $n$ (the number of trials) and $\mu$ (the probability of success).
- The probability mass function is given by:
  - $p(k; n, \mu) = \binom{n}{k} \mu^k (1-\mu)^{n-k}$ for $k \in \{0, 1, 2, ..., n\}$

### Multinomial distribution
- Now, we have more than two possible outcomes in a single trial say $k$ possible outcomes categories.
- It has the following parameters:
  - $n$ - The number of trials
  - $p_1, p_2, ..., p_k$ - The probabilities for each of the $k$ categories, where $\sum_{i=1}^k p_i = 1$
- The probability mass function for counts $x_1, x_2, ..., x_k$ where $n = \sum_{i=1}^k x_i$ is:
  - $P(X_1=x_1, X_2=x_2, ..., X_k=x_k) = \frac{n!}{x_1!x_2!...x_k!}p_1^{x_1}p_2^{x_2}...p_k^{x_k}$
  - Here $n!$ is the number of ways to arrange $n$ items and the denominator accounts for permutations within each category

### Normal distribution
- The probability density function (PDF) for a Normal distribution is:
  - $f(x; \mu, \sigma) = \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$
- Where:
  - $x$ is the random variable
  - $\mu$ is the mean of the distribution
  - $\sigma$ is the standard deviation of the distribution, defined as $\sigma = \sqrt{\mathbb{E}[(X - \mu)^2]}$

### Multivariate distribution
- The probability density function (PDF) for a Multivariate Normal distribution is:
  - $f(x; \mu, \Sigma) = \frac{1}{(2\pi)^{n/2}|\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu)\right)$
- Where:
  - $x$ is the random vector
  - $\mu$ is the mean vector  
  - $n$ is the dimensionality of the random vector
  - $|\Sigma|$ is the determinant of the covariance matrix
  - $\Sigma$ is the covariance matrix, defined as $\Sigma = \mathbb{E}[(\mathbf{X} - \mu)(\mathbf{X} - \mu)^T]$
    - $\Sigma$ is a symmetric matrix, where element $\Sigma_{ij}$ is the covariance between $X_i$ and $X_j$, defined as: $\Sigma_{ij} = \text{Cov}(X_i, X_j) = \mathbb{E}[(X_i - \mu_i)(X_j - \mu_j)]$


## Discrete optimization using lagrange multipliers

- The Lagrangian function for a discrete optimization problem with constraints is defined as:
  $$L(x, \lambda, \mu) = f(x) + \sum_{i=1}^m \lambda_i g_i(x) + \sum_{j=1}^n \mu_j h_j(x)$$
  where:
  - $f(x)$ is the objective function to be minimized
  - $g_i(x) \leq 0$ are the inequality constraint functions
  - $h_j(x) = 0$ are the equality constraint functions
  - $\lambda_i \geq 0$ are the Lagrange multipliers for inequality constraints
  - $\mu_j$ are the Lagrange multipliers for equality constraints (no sign restriction)

- The Karush-Kuhn-Tucker (KKT) conditions for optimality are:
  1. Stationarity: $\nabla_x L(x^*, \lambda^*, \mu^*) = 0$
  2. Primal feasibility: $g_i(x^*) \leq 0$ for all $i$, $h_j(x^*) = 0$ for all $j$
  3. Dual feasibility: $\lambda_i^* \geq 0$ for all $i$
  4. Complementary slackness: $\lambda_i^* g_i(x^*) = 0$ for all $i$

- To solve the optimization problem:
  1. Form the Lagrangian function
  2. Take partial derivatives with respect to x, λ, and μ
  3. Set the partial derivatives to zero
  4. Solve the resulting system of equations
  5. Check the KKT conditions to ensure optimality

## Taylor Series Expansion

### Scalar Function
For a scalar function $f(x)$ that is infinitely differentiable at a point $a$, the Taylor series expansion is:

$$f(x) = f(a) + f'(a)(x-a) + \frac{f''(a)}{2!}(x-a)^2 + \frac{f'''(a)}{3!}(x-a)^3 + \cdots$$

Or more compactly:

$$f(x) = \sum_{n=0}^{\infty} \frac{f^{(n)}(a)}{n!}(x-a)^n$$

where $f^{(n)}(a)$ is the $n$-th derivative of $f$ evaluated at $a$.

### Vector Function
For a vector-valued function $\mathbf{f}(\mathbf{x})$ where $\mathbf{x} \in \mathbb{R}^n$, the Taylor series expansion around a point $\mathbf{a}$ is:

$$\mathbf{f}(\mathbf{x}) = \mathbf{f}(\mathbf{a}) + J_f(\mathbf{a})(\mathbf{x}-\mathbf{a}) + \frac{1}{2!}H_f(\mathbf{a})(\mathbf{x}-\mathbf{a})^2 + \cdots$$

Where:
- $J_f(\mathbf{a})$ is the Jacobian matrix of $\mathbf{f}$ evaluated at $\mathbf{a}$
- $H_f(\mathbf{a})$ is the Hessian tensor of $\mathbf{f}$ evaluated at $\mathbf{a}$

More generally:

$$\mathbf{f}(\mathbf{x}) = \sum_{k=0}^{\infty} \frac{1}{k!} D^k\mathbf{f}(\mathbf{a})(\mathbf{x}-\mathbf{a})^k$$

where $D^k\mathbf{f}(\mathbf{a})$ is the $k$-th order derivative tensor of $\mathbf{f}$ evaluated at $\mathbf{a}$.