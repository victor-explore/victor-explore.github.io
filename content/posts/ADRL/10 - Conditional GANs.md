---
title: "Conditional GANs(cGANs)"
date:
draft: false
description:
tags: []
categories: []
author:
toc:
weight: 1
---

A Conditional Generative Adversarial Network (cGAN) is a type of Generative Adversarial Network (GAN) where the generation process is conditioned on some additional information, such as class labels or data from other modalities.

## Architecture modification

### Generator
Concatenate the conditional information Y with the noise input Z and feed it to the generator.

<div style="text-align: center;"><img src="https://raw.githubusercontent.com/victor-explore/ADRL-Notes/refs/heads/main/13.JPG" alt="Image Description" width="400" height="auto"/></div>

Here Y can be:
- One hot encoded class label
- Text embedding etc

### Discriminator

The discriminator in a cGAN is modified similarly to the generator. Concatenate the conditional information Y with the input image X and feed this combined input (X, Y) to the discriminator.

<div style="text-align: center;"><img src="https://raw.githubusercontent.com/victor-explore/ADRL-Notes/refs/heads/main/16.JPG" alt="Image Description" width="400" height="auto"/></div>

This allows the discriminator to assess both the realism of the image and its correspondence to the given condition.

## Objective of cGAN
The objective function of cGAN is an extension of the GAN objective, where both the generator and discriminator are conditioned on y:

<div class="math-katex-block">
$$
\min_G \max_D V(D,G) = \mathbb{E}_{x \sim p_{data}(x|y)}[\log D(x|y)] + \mathbb{E}_{z \sim p_z(z)}[\log(1 - D(G(z|y)|y))]
$$
</div>

Where:
- $G$ is the generator
- $D$ is the discriminator
- $x$ is the real data
- $z$ is the random noise input
- $y$ is the conditional information
- $p_{data}(x|y)$ is the conditional probability density function of the real data given $y$
- $p_z(z)$ is the probability density function of the noise distribution

This objective function encourages the generator to produce samples that not only look realistic but also correspond to the given condition $y$. The discriminator, in turn, learns to distinguish between real and fake samples while taking the condition into account.

## Optimization process
The optimization process for cGANs involves alternating between training the discriminator and the generator:

### Discriminator Optimization

The discriminator aims to maximize the probability of correctly classifying real and generated samples, conditioned on $y$:

<div class="math-katex-block">
$$
L_D = -\mathbb{E}_{x \sim p_{data}(x|y)}[\log D(x|y)] - \mathbb{E}_{z \sim p_z(z)}[\log(1 - D(G(z|y)|y))]
$$
</div>

### Generator Optimization

The generator tries to minimize the discriminator's ability to differentiate between real and fake samples by minimizing:

$$
L_G = -\mathbb{E}_{z \sim p_z(z)}[\log(1 - D(G(z|y)|y))]
$$

This is the standard form of the generator's loss in cGANs, derived from the minimax objective. However, in practice, to avoid vanishing gradients early in training, it's often implemented as:

<div class="math-katex-block">
$$
L_G = -\mathbb{E}_{z \sim p_z(z)}[\log D(G(z|y)|y)]
$$
</div>

During training, these two steps are alternated, with the discriminator and generator parameters updated using gradient descent (or a variant) to minimize their respective loss functions.












