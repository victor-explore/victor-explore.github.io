---
title: "Generative vs Discriminative Models"
date: 2025-01-01
draft: false
description:
tags:
categories:
author:
toc: true
weight: 1
---

## Generative Model

- Let the training dataset be 
<div class="math-katex">
$$data = \{(x_i)\}_{i=1}^N$$
</div>

  where $x_i \in \mathbb{R}^d$ are called data points. These data points are iid samples from the true data distribution $P(x)$.

- Aim of generative model is to extimate $P(x)$ using the training data and sample new data points from it.

- GANs, GMMs etc are a example of generative model. In GANs we use training data to train a neural network to learn the distribution of the data. Once we have learnt this distribution we can sample new data points from it.

## Discriminative Model

- In this case the data is in a different format
$$data = \{(x_i, y_i)\}_{i=1}^N$$
 where $x_i \in \mathbb{R}^d$  are the input features and $y_i \in \mathbb{R}^k$ are the corresponding output labels or target variables.

- Aim of discriminative model is to estimate the conditional probability $P(y|x)$ using the training data.
- In simple words, we are given an input $x$ and we want to predict the corresponding output $y$.
- If $y$ can take only discrete values, then the model is called a classification model. If $y$ can take continuous values, then the model is called a regression model.

## Difference between generative and discriminative models

- In generative models, we sample new data points from the learned distribution to get a new data point that might look like one of the data points in the training dataset.
- In discriminative models, for a given input $x$, we sample $y$ from the learned conditional distribution $P(y|x)$.
   
## Then what is conditional generative model ?

- In most practical cases, we want the generative model to generate new data point $x_i$ based on some conditioned input $y_i$.
- In this case the the training dataset is 
$$data =\{(x_i, y_i)\}_{i=1}^N$$ 
and we want to estimate $P(x|y)$ and sample from it.
