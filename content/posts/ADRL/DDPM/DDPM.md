---
title: "Denoising Diffusion Probabilistic Models"
math: true
draft: false
---
# Denoising Diffusion Probabilistic Models (DDPM)

There are various approaches to see DDPM:
- DDPM as solution to discretized stochastic differential equations
- DDPM as special case of variational autoencoders (VAE)
- DDPM as special case of score-based generative models


## DDPM as special case of variational autoencoders (VAE)

Impose following structure on the VAE:
- Multiple latent variables, where each latent variable is associated with a timestep.
- Dim of all the latent variables is the same as the dim of the data. 
- Assume a non learnable encoder unlike VAE where the encoder was learnable.

## Notations

- $X_0, X_1, ..., X_T$: Denote latent variables at different timesteps 
- $p(X_0)$: Denote the model data distribution
- $X_{0:T} = (X_0, X_1, X_2, ..., X_T)$: Denote the sequence of all latent variables

- $q(X_{1:T}|X_0)$: Denote the latent posterior

## Latent posterior

Because $q(X_{1:T}|X_0)$ is not learnable, we need to define it - we make it first order markovian chain with gaussian transitions:

$$X_0 \rightarrow X_1 \rightarrow X_2 - - \rightarrow X_T$$

This is also called encoding or forward process or adding noise process. Here:



$$X_t = \sqrt{\alpha_t} X_{t-1} + \sqrt{1-\alpha_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0,I)$$

where $\alpha_t \in (0,1)$ called variance schedule are hyperparameters that control the variance of the latent variables at different timesteps. Generally, $\alpha_t$ is chosen to be a constant or a slowly decaying function of $t$ to ensure that the variance of the latent variables at different timesteps is not too high or too low.

Because we are intrested in modeling latents:

$$q(X_{1:T}|X_0) = \prod_{t=1}^T q(X_t|X_{t-1})$$
This is due to the chain rule and Markov property of the forward process.

$$q(X_t|X_{t-1}) = \mathcal{N}(X_t; \sqrt{\alpha_t}X_{t-1}, (1-\alpha_t)I)$$

Because of reparameterization of $X_t = \sqrt{\alpha_t} X_{t-1} + \sqrt{1-\alpha_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0,I)$

## Model

We define model $p_\theta(X_0)$ as:

$$p_\theta(X_0) = \int p_\theta(X_{0:T}) \, dx_{1:T}$$

However, this is not tractable to compute directly. Instead, we can use the Evidence Lower Bound (ELBO) to optimize our model:

$$ ELBO = \mathbb{E}_{q(X_{1:T}|X_0)}\left[\log p_\theta(X_0|X_{1:T}) + \log \frac{p_\theta(X_{1:T})}{q(X_{1:T}|X_0)}\right]$$


$$ ELBO = \mathbb{E}_{q(X_{1:T}|X_0)}\left[\log \frac{p_\theta(X_{0:T})}{q(X_{1:T}|X_0)}\right]$$

The markov chain in reverse is also known as reverse/decoding/denoising/sampling process:

$$X_T \rightarrow X_{T-1} \rightarrow X_{T-2} - - - \rightarrow X_0$$
- It is also a first order markov chain with gaussian transitions
- Hence, we can express the modelled joint distribution of the reverse process by applying the chain rule of the reverse process:

$$p_\theta(X_{0:T}) = p_\theta(X_T) \prod_{t=1}^T p_\theta(X_{t-1}|X_t)$$

Where:
- $p_\theta(X_T)$ is the prior distribution of the latent variable at the final timestep
- $p_\theta(X_{t-1}|X_t)$ represents the transition probability from $X_t$ to $X_{t-1}$

Supstitute $p_\theta(X_{0:T})$ in ELBO:
$$ ELBO = \mathbb{E}_{q(X_{1:T}|X_0)}\left[\log \frac{p_\theta(X_T) \prod_{t=1}^T p_\theta(X_{t-1}|X_t)}{q(X_{1:T}|X_0)}\right]$$

Using the Markov property of the forward process, we can rewrite $q(X_{1:T}|X_0)$ as:

$$ q(X_{1:T}|X_0) = \prod_{t=1}^T q(X_t|X_{t-1}) $$

Substituting this into the ELBO equation:

$$ ELBO = \mathbb{E}_{q(X_{1:T}|X_0)}\left[\log \frac{p_\theta(X_T) \prod_{t=1}^T p_\theta(X_{t-1}|X_t)}{\prod_{t=1}^T q(X_t|X_{t-1})}\right]$$

$$ ELBO = \mathbb{E}_{q(X_{1:T}|X_0)}\left[\log \left( \frac{p_\theta(X_T) \prod_{t=1}^T p_\theta(X_{t-1}|X_t)}{q(X_1|X_0) \prod_{t=2}^T q(X_t|X_{t-1})} \right) \right]$$

$$ ELBO = \mathbb{E}_{q(X_{1:T}|X_0)}\left[\log p_\theta(X_T) + \log \left(\frac{p_\theta(X_0|X_1)}{q(X_1|X_0)}\right) + \sum_{t=2}^T \log \left(\frac{p_\theta(X_{t-1}|X_t)}{q(X_t|X_{t-1}, X_0)}\right)\right]$$

Apply bayes rule to the second term:
$$ ELBO = \mathbb{E}_{q(X_{1:T}|X_0)}\left[\log p_\theta(X_T) + \log \left(\frac{p_\theta(X_0|X_1)}{q(X_1|X_0)}\right) + \sum_{t=2}^T \log \left(\frac{p_\theta(X_{t-1}|X_t)q(X_t|X_0)}{q(X_t|X_{t-1}, X_0)q(X_{t-1}|X_0)}\right)\right]$$

After rearranging the terms, we get:

$$ ELBO = \mathbb{E}_{q(X_{1:T}|X_0)}\left[\log p_\theta(X_T) + \log \left(\frac{p(X_T|X_0)}{q(X_T|X_0)}\right) + \sum_{t=2}^T \log \left(\frac{p_\theta(X_{t-1}|X_t)}{q(X_{t-1}|X_t,X_0)}\right)\right]$$
































































