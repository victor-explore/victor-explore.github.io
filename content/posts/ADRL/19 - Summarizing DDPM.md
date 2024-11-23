---
title: "Summarizing DDPM"
date:
draft: false
description:
tags: []
categories: []
author:
toc:
weight: 1
---
- VAE:
  <div style="text-align: center;"><img src="https://raw.githubusercontent.com/victor-explore/ADRL-Notes/refs/heads/main/38.JPG" alt="Image Description" width="200" height="auto"/></div> 
- Hierarchical VAE:
  <div style="text-align: center;"><img src="https://raw.githubusercontent.com/victor-explore/ADRL-Notes/refs/heads/main/36.JPG" alt="Image Description" width="600" height="auto"/></div> 
- DDPM(Hierarchical VAE with deterministic forward process):
  <div style="text-align: center;"><img src="https://raw.githubusercontent.com/victor-explore/ADRL-Notes/refs/heads/main/37.JPG" alt="Image Description" width="600" height="auto"/></div> 
Notice that $\phi$ has been removed from $q_\phi()$.

Also see that:
- The ELBO for a Variational Autoencoder (VAE) is given by:
<div class="math-katex">
$$
\text{ELBO (VAE)} = \mathbb{E}_{q_\phi(z|x)} \left[ \log \frac{p(x|z)}{q_\phi(z|x)} \right]
$$
</div>
- For Denoising Diffusion Models (DDPM), the ELBO is expressed as:
<div class="math-katex">
$$
\text{ELBO (DM)} = \mathbb{E}_{q(x_{1:T}|x_0)} \left[ \log \frac{p(x_{0:T})}{q(x_{1:T}|x_0)} \right]
$$
</div>