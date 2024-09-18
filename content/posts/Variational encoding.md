---
title: "Variational encoding"
math: true
draft: false
---

## Problem setting

Given a set of data $D = \{x_i\}_{i=1}^N$ iid samples from some unknown distribution $p_{\text{data}}$.

We want to model $p_{\text{data}}$ as $p_{\theta}$ using a neural network.

We model $p_{\theta}$ as a latent variable model

## Modelling

Recall the relationship between evidence and evidence lower bound (ELBO).

$$ \mathcal{L}(\theta) \geq F_{\theta}(q) $$

where

$$ F_{\theta}(q) = \mathbb{E}_{q(z)} \left[ \log \frac{p_{\theta}(x, z)}{q(z)} \right] $$ 

Here, $q(z)$ was an arbitrary distribution over the latent variables $z$.

Instead of $q(z)$, we use $q(z|x)$ to denote the posterior. Hence the equation becomes

$$ F_{\theta}(q) = \mathbb{E}_{q(z|x)} \left[ \log \frac{p_{\theta}(x, z)}{q(z|x)} \right] $$

$$ F_{\theta}(q) = \mathbb{E}_{q(z|x)} \left[ \log \frac{p_{\theta}(x|z)p_{\theta}(z)}{q(z|x)} \right] $$

$$ F_{\theta}(q) = \mathbb{E}_{q(z|x)} \left[ \log p_{\theta}(x|z) \right] + \mathbb{E}_{q(z|x)} \left[ \log \frac{p_{\theta}(z)}{q(z|x)} \right] $$

$$ F_{\theta}(q) = \mathbb{E}_{q(z|x)} \left[ \log p_{\theta}(x|z) \right] - \mathbb{E}_{q(z|x)} \left[ \log \frac{q(z|x)}{p_{\theta}(z)} \right] $$

$$ F_{\theta}(q) = \mathbb{E}_{q(z|x)} \left[ \log p_{\theta}(x|z) \right] - D_{KL} \left( q(z|x) \mid p_{\theta}(z) \right) $$

Here the first term is the conditional log likelihood of the data under the model. We want to maximise this term.

The second term is the KL divergence between the posterior and the prior. We want to minimise this term.

## Variational Autoencoder

- **Encoder:** We model $q(z|x)$ as a neural network with parameters $\phi$. The network takes in an observation $x$ and outputs the parameters of a Gaussian distribution ie mean $\mu_{\phi}(x)$ and covariance $\Sigma_{\phi}(x)$.

- **Decoder:** We model $p_{\theta}(x|z)$ as a neural network with parameters $\theta$. The network takes in a sampled latent variable $z$ from the distribution with parameters $\mu_{\phi}(x)$ and $\Sigma_{\phi}(x)$ and outputs a data sample $\hat{x}$. Post training, we use the decoder to generate new data samples ie works as generator.

## Motivation for reparameterisation trick

Evidence is intractable, so we are optimizing a lower bound on it. 

We need to be able to compute the gradient of the ELBO to be able to do gradient descent.


ELBO's first term's gradient is tractable hence we use reparameterisation trick to sample the latent variable $z$ from the posterior $q(z|x)$ and compute the gradient of the ELBO with respect to $\theta$.

## Reparameterisation trick

We got:
$$ F_{\theta}(q) = \mathbb{E}_{q(z|x)} \left[ \log p_{\theta}(x|z) \right] - D_{KL} \left( q(z|x) \mid p_{\theta}(z) \right) $$

The first term is $\mathbb{E}_{q(z|x)} \left[ \log p_{\theta}(x|z) \right]$

We represent it as:

$$\mathbb{E}_{q(z|x)} \left[ \log p_{\theta}(x|z) \right] = \mathbb{E}_{p_{\psi}(v)} \left[ f_{\psi}(v) \right]$$

where:
- $v = z/x$
- $p_{\psi}(v) = q(z|x)$
- $f_{\psi}(v) = \log p_{\theta}(x|v)$


Suppose there exists a deterministic function such that

$$v = g_{\psi}(\epsilon)$$

where $\epsilon$ belongs to a arbitrary distribution $p(\epsilon)$ and is independent of  $\psi$.

Then we can rewrite the first term (after applying LOTUS) as:

$$\mathbb{E}_{p_{\psi}(v)} \left[ f_{\psi}(v) \right] = \mathbb{E}_{p(\epsilon)} \left[ f_{\psi}(g_{\psi}(\epsilon)) \right]$$

Hence 
$$ \nabla_{\psi} \mathbb{E}_{p_{\psi}(v)}[f_{\psi}(v)] = \nabla_{\psi} \mathbb{E}_{p(\epsilon)}[f_{\psi}(g_{\psi}(\epsilon))] \approx \frac{1}{N} \sum_{i=1}^N \nabla_{\psi} [f_{\psi}(g_{\psi}(\epsilon_i))]  \quad \epsilon_i \sim_{iid} p_{\epsilon}$$








