---
title: "KL Divergence - Kullback-Leibler Divergence"
date:
draft: false
description:
tags: []
categories: []
author:
toc: true
weight: 1
---

- KL divergence is a measure of how one probability distribution differs from another probability distribution.

- Mathematically, for two probability distributions $P(x)$ and $Q(x)$, the KL divergence from $Q$ to $P$ is defined as:

$$D_{KL}(P || Q) = \sum_{x} p(x) \log\left(\frac{p(x)}{q(x)}\right) = E_{x \sim p(x)}\left[\log\left(\frac{p(x)}{q(x)}\right)\right]$$ for discrete distributions

$$D_{KL}(P || Q) = \int p(x) \log\left(\frac{p(x)}{q(x)}\right) dx = E_{x \sim p(x)}\left[\log\left(\frac{p(x)}{q(x)}\right)\right]$$ for continuous distributions

where the sum/integral is over all possible events $x$. And $p(x)$ and $q(x)$ are the probability density functions of distributions $P(x)$ and $Q(x)$ respectively.

## Intuition

- Intuitively, KL divergence measures how much information is lost when using $q$ to approximate $p$:
  - $p(x)$: Probability of event $x$ in the true distribution.
  - log$\frac{p(x)}{q(x)} = \log p(x) - \log q(x)$: Log difference in probabilities.
  - $p(x) \log\frac{p(x)}{q(x)}$: Weighs the log difference by $p(x)$.
  - The sum/integral over all $x$ accumulates these weighted differences so that we get the expected log difference.

- KL divergence emphasizes discrepancies in high-probability regions of $p$:
  - If $p$ and $q$ differ significantly for some $x$, but $p(x)$ is small, the impact on the overall divergence is minimal.
  - Conversely, even small differences between $p$ and $q$ in regions where $p(x)$ is large can contribute substantially to the KL divergence.


## Properties

- KL divergence is not symmetric: $D_{KL}(P || Q) \neq D_{KL}(Q || P)$
- KL divergence is always non-negative: $D_{KL}(P || Q) \geq 0$
- KL divergence is 0 if and only if $P$ and $Q$ are the same distribution

## Usefulness in Machine Learning - Minimizing KL Divergence is Equivalent to Maximizing Likelihood
- For a typical ML problem, all we have are samples from the true distribution $P(x)$ ie $data = \{(x_i)\}_{i=1}^N$ where $x_i \in \mathbb{R}^d$ are iid samples from the true distribution $P(x)$. 

- We do not know the true distribution $P(x)$ explicitly.

- We try our best to estimate the true distribution $P(x)$ by $Q_\theta(x)$ where $\theta$ are the parameters of the model.

- We need to know how well our model $Q_\theta(x)$ is performing. We can do this by calculating the KL divergence between the true distribution $P(x)$ and the estimated distribution $Q_\theta(x)$.

$$D_{KL}(P || Q) = ∫ p(x)  log\left(\frac{p(x)}{q_\theta(x)}\right) dx$$

$$D_{KL}(P || Q) = E_{x \sim p(x)}[log\left(\frac{p(x)}{q_\theta(x)}\right)]$$

$$D_{KL}(P || Q) = E_{x \sim p(x)}[log\left(p(x)\right)] - E_{x \sim p(x)}[log\left(q_\theta(x)\right)]$$

- We are trying to find the parameters $\theta^*$ that minimize the KL divergence between $p(x)$ and $q_\theta(x)$.

$$\theta^* = \underset{\theta}{argmin} \ D_{KL}(P || Q_\theta)$$

$$\theta^* = \underset{\theta}{argmin} \ E_{x \sim p(x)}[log\left(p(x)\right)] - E_{x \sim p(x)}[log\left(q_\theta(x)\right)]$$

- Because $E_{x \sim p(x)}[log\left(p(x)\right)]$ does not depend on $\theta$, we can ignore it.

$$\theta^* = \underset{\theta}{argmin} \ -E_{x \sim p(x)}[log\left(q_\theta(x)\right)]$$

$$\theta^* = \underset{\theta}{argmax} \ E_{x \sim p(x)}[log\left(q_\theta(x)\right)]$$

- $E_{x \sim p(x)}[log\left(q_\theta(x)\right)]$ is called the **Expected Log Likelihood**,

- By the law of large numbers, we can approximate the expected log likelihood by the average log likelihood of the data:

$$E_{x \sim p(x)}[log(q_\theta(x))] \approx \frac{1}{N} \sum_{i=1}^N log(q_\theta(x_i))$$

- Therefore, our optimization problem becomes:

$$\theta^* = \underset{\theta}{argmax} \ \frac{1}{N} \sum_{i=1}^N log(q_\theta(x_i))$$

- This is equivalent to maximizing the log likelihood of the data under the model $q_\theta(x)$.

$$\theta^* = \underset{\theta}{argmax} \ \frac{1}{N} \sum_{i=1}^N log(q_\theta(x_i))$$

- Hence $\theta$ is also called the **maximum log likelihood estimate**. 


