---
title: "Distillation"
date:
draft: false
description:
tags: []
categories: []
author:
toc:
weight: 1
---

- Teacher model is a large model that was trained on a large dataset as usual.
- Student model is a smaller model that is trained to mimic the behavior of the teacher model.
- The student model has 2 losses:
<div class="math-katex">
$$
\mathcal{L} = \mathcal{L}_\text{task} + \mathcal{L}_\text{distill}
$$
</div>
  - $\mathcal{L}_\text{task}$ is the task-specific loss.
  - $\mathcal{L}_\text{distill}$ is the distillation loss - This loss is used to make the student model's output close to the teacher model's softened outputs.