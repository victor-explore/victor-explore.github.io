---
title: "Applying RL to derive Optimal Policy for human behavior"
date: 2025-05-25
draft: false
description:
tags: []
categories: []
author:
toc:
weight: 1
---

## RL setting

In the reinforcement learning framework, we can model human behavior as an agent interacting with the world. Let's formalize this mathematically.

### Markov Decision Process (MDP) Formulation

The human experience can be modeled as a Markov Decision Process (MDP) defined by the tuple:

$$\mathcal{M} = (\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)$$

where:
- $\mathcal{S}$ is the state space (all possible states of the world and internal states)
- $\mathcal{A}$ is the action space (all possible actions the human can take)
- $\mathcal{P}: \mathcal{S} \times \mathcal{A} \times \mathcal{S} \rightarrow [0,1]$ is the transition probability function
- $\mathcal{R}: \mathcal{S} \times \mathcal{A} \times \mathcal{S} \rightarrow \mathbb{R}$ is the reward function
- $\gamma \in [0,1]$ is the discount factor

### Environment is the world

The environment encompasses everything external to the agent (brain). Mathematically, the environment dynamics are governed by:

$$P(s_{t+1} | s_t, a_t) = \mathcal{P}(s_t, a_t, s_{t+1})$$

This represents the probability of transitioning to state $s_{t+1}$ given current state $s_t$ and action $a_t$.

### Observation space is 5 senses

The observation space $\mathcal{O}$ is not identical to the state space $\mathcal{S}$. Humans have partial observability through sensory channels:
<div class="math-block">
$$
\mathcal{O} = \mathcal{O}_{\text{visual}} \times \mathcal{O}_{\text{auditory}} \times \mathcal{O}_{\text{tactile}} \times \mathcal{O}_{\text{olfactory}} \times \mathcal{O}_{\text{gustatory}}
$$
</div>

The observation function maps states to observations:

$$\Omega: \mathcal{S} \rightarrow \mathcal{O}$$

At each time step, the agent receives observation:

$$o_t = \Omega(s_t) + \epsilon_t$$

where $\epsilon_t$ represents sensory noise.

### Action space is moving limbs, including eyes, ears, nose, mouth, and skin

The action space consists of continuous motor controls:

<div class="math-block">
$$
\mathcal{A} = \mathcal{A}_{\text{motor}} \times \mathcal{A}_{\text{ocular}} \times \mathcal{A}_{\text{vocal}} \times \mathcal{A}_{\text{facial}}
$$
</div>

where each subspace is typically continuous and high-dimensional. For example:
- $\mathcal{A}_{\text{motor}} \subset \mathbb{R}^n$ where $n$ is the number of controllable muscle groups
- $\mathcal{A}_{\text{ocular}} \subset \mathbb{R}^6$ for eye movements (3D rotation for each eye)

### Agent is the brain that has a policy to decide what action to take

The agent (brain) maintains a policy $\pi$ that maps observations to actions. In the partially observable case:

$$\pi: \mathcal{H} \rightarrow \mathcal{P}(\mathcal{A})$$

where $\mathcal{H}$ is the history space of observations and actions, and $\mathcal{P}(\mathcal{A})$ is the probability distribution over actions.

The optimal policy maximizes expected cumulative reward:

<div class="math-block">
$$
\pi^* = \arg\max_{\pi} \mathbb{E}_{\pi} \left[ \sum_{t=0}^{\infty} \gamma^t r_t \right]
$$
</div>

### Reward is serotonin, dopamine, oxytocin, etc.

The reward function is multi-dimensional, corresponding to different neurotransmitter systems:

<div class="math-block">
$$
r_t = \sum_{i} w_i \cdot r_t^{(i)}
$$
</div>

where:
- $r_t^{(\text{dopamine})}$ - reward prediction and motivation
- $r_t^{(\text{serotonin})}$ - mood and well-being
- $r_t^{(\text{oxytocin})}$ - social bonding and trust
- $r_t^{(\text{endorphins})}$ - pleasure and pain relief

### Individual Differences in Reward Functions

Different individuals have different reward function parameters $\theta$:

$$\mathcal{R}_{\theta}(s,a,s') = \theta^T \phi(s,a,s')$$

where $\phi(s,a,s')$ are reward features and $\theta$ are individual-specific weights.

For example:
- **Money-oriented**: $\theta_{\text{wealth}} >> \theta_{\text{social}}$
- **Family-oriented**: $\theta_{\text{family}} >> \theta_{\text{wealth}}$
- **Self-actualization**: $\theta_{\text{growth}} >> \theta_{\text{material}}$

### Value Functions

The state-value function under policy $\pi$:

<div class="math-block">
$$
V^{\pi}(s) = \mathbb{E}_{\pi} \left[ \sum_{t=0}^{\infty} \gamma^t r_t \biggm| s_0 = s \right]
$$
</div>

The action-value function:

<div class="math-block">
$$
Q^{\pi}(s,a) = \mathbb{E}_{\pi} \left[ \sum_{t=0}^{\infty} \gamma^t r_t \biggm| s_0 = s, a_0 = a \right]
$$
</div>

### Bellman Equations

The optimal value functions satisfy:

<div class="math-block">
$$
V^*(s) = \max_{a \in \mathcal{A}} \sum_{s' \in \mathcal{S}} \mathcal{P}(s,a,s') \left[ \mathcal{R}(s,a,s') + \gamma V^*(s') \right]
$$
</div>

<div class="math-block">
$$
Q^*(s,a) = \sum_{s' \in \mathcal{S}} \mathcal{P}(s,a,s') \left[ \mathcal{R}(s,a,s') + \gamma \max_{a' \in \mathcal{A}} Q^*(s',a') \right]
$$
</div>

### Learning and Adaptation

Humans continuously update their policies through experience:

<div class="math-block">
$$
\pi_{t+1} = \text{UPDATE}(\pi_t, (o_t, a_t, r_t, o_{t+1}))
$$
</div>

This update can be modeled using various RL algorithms:
- **Model-based**: Learning $\mathcal{P}$ and $\mathcal{R}$, then planning
- **Model-free**: Directly learning $Q$ or $\pi$ from experience
- **Hybrid**: Combining both approaches, similar to human cognition

### Exploration vs Exploitation

Humans balance exploration and exploitation through various strategies:

$$a_t \sim \begin{cases}
\pi(a|h_t) & \text{with probability } 1-\epsilon \\
\text{Uniform}(\mathcal{A}) & \text{with probability } \epsilon
\end{cases}$$

where $\epsilon$ represents the exploration rate, which typically decreases with experience and age.

## Individual Differences in Human RL Agents

### Cognitive Capacity and Model Complexity

Different individuals have varying capacities for modeling the world, which affects their ability to learn and plan:

#### High IQ / Better Modeling Capacity

Individuals with higher cognitive capacity can maintain more complex world models:

<div class="math-block">
$$
\hat{\mathcal{P}}_{\text{high-IQ}}(s_{t+1}|s_t, a_t) \approx \mathcal{P}(s_{t+1}|s_t, a_t)
$$
</div>

with lower approximation error. This manifests as:

- **Larger state space representation**: $|\mathcal{S}_{\text{internal}}| \propto \text{cognitive capacity}$
- **Better function approximation**: More accurate $\hat{V}(s)$ and $\hat{Q}(s,a)$
- **Deeper planning horizon**: Can simulate longer trajectories

<div class="math-block">
$$
\pi_{\text{high-IQ}}(a|s) = \arg\max_a \mathbb{E}\left[\sum_{k=0}^{H} \gamma^k r_{t+k} \biggm| s_t = s, a_t = a\right]
$$
</div>

where $H$ (planning horizon) is larger for higher cognitive capacity.

#### Limited Modeling Capacity

Individuals with limited cognitive resources use simpler approximations:

- **Reduced state space**: $\mathcal{S}_{\text{reduced}} \subset \mathcal{S}$
- **Simpler policies**: Often rely on habits and heuristics
- **Model-free learning**: Greater reliance on cached values rather than planning

### Discount Factor Variations


The discount factor $\gamma$ varies significantly across individuals, affecting their temporal preferences:

<div class="math-block">
$$
\gamma \in \begin{cases}
0.99 & \text{Future-oriented (long-term planning, delayed gratification)} \\
0.5 & \text{Present-oriented (immediate gratification, impulsive behavior)}
\end{cases}
$$
</div>

This directly impacts value function computation:

<div class="math-block">
$$
\underbrace{V^{\pi}(s) = \mathbb{E}_{\pi}\left[\sum_{t=0}^{\infty} \gamma^t r_t \biggm| s_0 = s\right]}_{\substack{\text{Value function where } \gamma \text{ determines} \\ \text{relative weight of future vs immediate rewards}}}
$$
</div>

### Time-Inconsistent Preferences

Many humans exhibit hyperbolic discounting:

<div class="math-block">
$$
\gamma(t) = \frac{1}{1 + k \cdot t}
$$
</div>

where $k$ is the individual's impulsivity parameter. This leads to preference reversals and self-control problems.

## Policy-Reward Feedback Loop

### Reward Function Plasticity

Unlike traditional RL formulations, human reward functions are not fixed but adapt based on experience:

$$\mathcal{R}_{t+1}(s,a,s') = f(\mathcal{R}_t(s,a,s'), \pi_t, \text{experience}_t)$$

This creates a feedback loop where the policy influences future rewards.

### Addiction as Reward System Hijacking

Consider the sugar addiction example:

#### Initial State
- Reward for sugar: $r_{\text{sugar}}^{(0)} = r_{\text{base}}$
- Policy: $\pi_0(\text{eat sugar}|s) = p_{\text{low}}$

#### After Repeated Exposure
The reward system adapts:

$$r_{\text{sugar}}^{(t+1)} = r_{\text{sugar}}^{(t)} + \alpha \cdot \text{consumption}_t$$

where $\alpha$ is the sensitization rate.

Simultaneously, tolerance develops:

$$r_{\text{baseline}}^{(t+1)} = r_{\text{baseline}}^{(t)} - \beta \cdot \text{consumption}_t$$

This leads to:
1. **Increased craving**: $\pi_t(\text{eat sugar}|s) \rightarrow 1$
2. **Reduced baseline happiness**: Need sugar to feel normal
3. **Narrowed action space**: Other actions become less rewarding

### Mathematical Model of Habit Formation

Habits form through the strengthening of state-action associations:

$$Q_{\text{habit}}(s,a) = Q_{\text{deliberate}}(s,a) + \theta \cdot N(s,a)$$

where:
- $N(s,a)$ is the number of times action $a$ was taken in state $s$
- $\theta$ is the habit strength parameter

Over time, habitual actions dominate:

$$\pi(a|s) \propto \exp\left(\frac{Q_{\text{habit}}(s,a)}{\tau}\right)$$

where $\tau$ (temperature) decreases with repetition, making behavior more deterministic.

### Neuroplasticity and Reward Adaptation

The brain's reward prediction error:

$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

drives both learning and reward system adaptation:

$$\Delta w_{\text{dopamine}} = \eta \cdot \delta_t \cdot \phi(s_t, a_t)$$

This creates several feedback effects:

1. **Tolerance**: Repeated rewards lead to reduced response
   $$r_{\text{effective}}(s,a) = \frac{r_{\text{nominal}}(s,a)}{1 + \tau \cdot N(s,a)}$$

2. **Sensitization**: Cue-triggered wanting increases
   $$\text{craving}(s) = w^T \phi_{\text{cue}}(s) \cdot \sqrt{N_{\text{exposure}}}$$

3. **Withdrawal**: Absence of expected reward becomes punishing
   $$r_{\text{withdrawal}} = -\lambda \cdot (r_{\text{expected}} - r_{\text{received}})$$

### Social and Environmental Feedback

The policy also affects the environment's response:

$$\mathcal{P}_{t+1}(s'|s,a) = g(\mathcal{P}_t(s'|s,a), \pi_t)$$

Examples:
- **Social reinforcement**: Certain behaviors attract like-minded individuals
- **Environmental shaping**: Our actions modify our surroundings
- **Skill development**: Repeated actions improve capabilities

### Breaking the Feedback Loop

To escape negative policy-reward cycles:

1. **Commitment devices**: Constraining future action space
   $$\mathcal{A}_{\text{constrained}} \subset \mathcal{A}$$

2. **Reward shaping**: Artificially modifying rewards
   $$r_{\text{shaped}}(s,a) = r(s,a) + F(s,a)$$

3. **Meta-learning**: Learning to learn better policies
   $$\pi_{\text{meta}} = \arg\max_{\pi} \mathbb{E}\left[\sum_t \gamma^t r_t + \lambda \cdot \text{flexibility}(\pi)\right]$$

## Implications

Understanding humans as RL agents with:
- **Variable cognitive capacities**
- **Different discount factors**
- **Plastic reward functions**
- **Policy-reward feedback loops**

helps explain diverse human behaviors from addiction to achievement, from impulsivity to long-term planning, and from habit formation to behavioral change.

This framework suggests that effective behavior change requires not just modifying policies but also understanding and potentially reshaping the underlying reward systems and feedback mechanisms.

## Stoic Philosophy Through the RL Lens

The Stoic principle of focusing on what is under our control while accepting what is not can be formalized within our RL framework. This philosophy, exemplified by Marcus Aurelius' "You have power over your mind - not outside events. Realize this, and you will find strength," has profound implications for optimal policy design.

### The Dichotomy of Control

In RL terms, we can partition the elements of our MDP:

**Under our control:**
- Policy: $\pi(a|s)$ - We directly choose our actions
- Internal state representation: $s_{\text{internal}} \subset s$ - How we interpret observations
- Attention mechanism: $\alpha(o) \in [0,1]$ - What aspects of observations we focus on

**Not under our control:**
- Environment dynamics: $\mathcal{P}(s'|s,a)$ - How the world responds to our actions
- External rewards: $r_{\text{external}}(s,a,s')$ - Outcomes and consequences
- Other agents' policies: $\pi_{\text{others}}(a|s)$ - How others behave

### Mathematical Formulation of Stoic Control

We can decompose the value function into controllable and uncontrollable components:

<div class="math-block">
$$
V^{\pi}(s) = \underbrace{\mathbb{E}_{a \sim \pi(\cdot|s)}[Q(s,a)]}_{\text{Controllable: Policy choice}} + \underbrace{\mathbb{E}_{s'|\mathcal{P}}[\text{Var}(r|s,a,s')]}_{\text{Uncontrollable: Environmental stochasticity}}
$$
</div>

The Stoic approach suggests optimizing a modified objective:

<div class="math-block">
$$
\pi_{\text{Stoic}}^* = \arg\max_{\pi} \mathbb{E}_{\pi}\left[\sum_{t=0}^{\infty} \gamma^t \cdot \underbrace{r_{\text{internal}}(s_t, a_t)}_{\text{Process-based reward}}\right]
$$
</div>

where $r_{\text{internal}}$ depends only on controllable factors:

<div class="math-block">$$
r_{\text{internal}}(s,a) = f(\text{effort}(a), \text{virtue}(a), \text{intention}(a))$$
</div>

### The Stoic Reward Function

Traditional RL agents optimize for external outcomes:

<div class="math-block">
$$
r_{\text{traditional}}(s,a,s') = r_{\text{external}}(s')
$$
</div>

Stoic agents redefine their reward function to focus on process:

<div class="math-block">
$$
r_{\text{Stoic}}(s,a,s') = \alpha \cdot \underbrace{r_{\text{virtue}}(a)}_{\text{Acting according to values}} + (1-\alpha) \cdot \underbrace{r_{\text{external}}(s')}_{\text{External outcomes}}
$$
</div>

where $\alpha \rightarrow 1$ for a fully Stoic agent.

### Robustness Through Control Focus

By focusing on controllable elements, Stoic policies exhibit robustness to environmental uncertainty:

<div class="math-block">
$$
\pi_{\text{Stoic}}(a|s) = \arg\max_a \min_{\mathcal{P} \in \mathcal{U}} \mathbb{E}_{\mathcal{P}}[r_{\text{internal}}(s,a)]
$$
</div>

where $\mathcal{U}$ is the uncertainty set of possible environment dynamics. This minimax formulation ensures good performance regardless of environmental response.

### Emotional Regulation as State Abstraction

The Stoic practice of emotional regulation can be modeled as learned state abstraction:

<div class="math-block">
$$\phi_{\text{Stoic}}: \mathcal{S} \rightarrow \mathcal{S}_{\text{abstract}}$$
</div>

where the abstraction function filters out uncontrollable elements:

<div class="math-block">
$$\phi_{\text{Stoic}}(s) = \text{ProjectToControllable}(s)$$
</div>

This reduces the effective state space and prevents uncontrollable factors from influencing the policy:

<div class="math-block">
$$\pi_{\text{Stoic}}(a|s) = \pi(a|\phi_{\text{Stoic}}(s))$$
</div>

### The Paradox of Control and Outcomes

Interestingly, focusing on controllable factors often leads to better long-term outcomes:

<div class="math-block">
$$\mathbb{E}\left[\sum_{t=0}^{T} r_{\text{external},t} \biggm| \pi_{\text{Stoic}}\right] \geq \mathbb{E}\left[\sum_{t=0}^{T} r_{\text{external},t} \biggm| \pi_{\text{outcome-focused}}\right]$$
</div>

This occurs because:

1. **Reduced anxiety**: Lower cortisol affects decision-making quality
   $$\text{noise}_{\text{decision}} \propto \text{anxiety level}$$

2. **Consistent execution**: Process focus leads to more stable policies
   $$\text{Var}[\pi_{\text{Stoic}}(a|s)] < \text{Var}[\pi_{\text{outcome}}(a|s)]$$

3. **Compound effects**: Small controllable improvements accumulate
   $$\prod_{t=1}^{T}(1 + \epsilon_t) \gg 1 \text{ for consistent } \epsilon_t > 0$$

### Practical Implementation

The Stoic RL agent implements several key strategies:

#### 1. Premeditation of Evils (Negative Visualization)
Model worst-case scenarios to reduce their impact:
<div class="math-block">
$$V_{\text{robust}}(s) = \mathbb{E}_{\pi}\left[\sum_{t=0}^{\infty} \gamma^t r_t \biggm| s_0 = s, \mathcal{P} = \mathcal{P}_{\text{worst-case}}\right]$$
</div>

#### 2. View from Above
Use hierarchical abstraction to focus on important controllables:
<div class="math-block">
$$s_{\text{abstract}} = \text{HierarchicalAbstraction}(s, \text{level} = k)$$
</div>

where higher $k$ filters out more uncontrollable details.

#### 3. Discipline of Desire
Modify the reward function to align with controllables:
<div class="math-block">
$$r_{\text{disciplined}}(s,a) = \begin{cases}
r_{\text{high}} & \text{if } a \in \mathcal{A}_{\text{virtuous}} \\
r_{\text{neutral}} & \text{otherwise}
\end{cases}$$
</div>

### Neurological Basis

This framework aligns with neuroscience findings:

- **Prefrontal cortex**: Implements the control-focused policy
- **Anterior cingulate cortex**: Monitors controllable vs uncontrollable
- **Amygdala suppression**: Reduced emotional response to uncontrollables

The neural implementation:

$$\text{PFC activity} = w_{\text{control}}^T \phi_{\text{controllable}}(s) - w_{\text{suppress}}^T \phi_{\text{uncontrollable}}(s)$$

### Implications for Well-being

Agents following Stoic principles show:

1. **Lower variance in happiness**: 
   $$\text{Var}[r_{\text{subjective}}] < \text{Var}[r_{\text{external}}]$$

2. **Higher baseline satisfaction**:
   $$\mathbb{E}[r_{\text{subjective}}] > \mathbb{E}[r_{\text{external}}]$$

3. **Antifragility**: Performance improves under adversity
   $$\frac{\partial \pi_{\text{quality}}}{\partial \text{adversity}} > 0$$

This mathematical framework reveals why the ancient Stoic wisdom remains relevant: by optimizing for what we control, we paradoxically achieve better outcomes in what we cannot control, while maintaining equanimity throughout the process.

## The "Love Your Craft" Philosophy: Why Enjoying the Journey is Optimal

Successful people often advise to "have fun along the way" or "love what you do." This seemingly simple advice has profound mathematical justification within our RL framework. Let's explore why intrinsic motivation and process enjoyment lead to optimal long-term performance.

### Intrinsic vs Extrinsic Reward Functions

Consider two types of reward functions:

**Extrinsic-only agent:**
<div class="math-block">
$$
r_{\text{extrinsic}}(s,a,s') = \begin{cases}
R_{\text{big}} & \text{if } s' \in \mathcal{S}_{\text{success}} \\
0 & \text{otherwise}
\end{cases}
$$
</div>

**Intrinsic-motivated agent:**
<div class="math-block">
$$
r_{\text{intrinsic}}(s,a,s') = \underbrace{r_{\text{process}}(a)}_{\text{Joy from doing}} + \underbrace{r_{\text{mastery}}(s,a)}_{\text{Satisfaction from improvement}} + \underbrace{r_{\text{external}}(s')}_{\text{External rewards}}
$$
</div>

### The Temporal Credit Assignment Problem

For long-term goals, the extrinsic-only agent faces severe challenges:

<div class="math-block">
$$
\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\left[\sum_{t=0}^{T} \gamma^t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot \underbrace{\left(\sum_{k=t}^{T} \gamma^{k-t} r_k\right)}_{\text{Return from time } t}\right]
$$
</div>

When $r_k = 0$ for all $k < T$ (no intermediate rewards), the gradient signal becomes:
- **Exponentially weak**: $\gamma^T$ makes distant rewards nearly invisible
- **High variance**: Success/failure is binary
- **Sparse**: Most trajectories provide no learning signal

### Dense Reward Shaping Through Intrinsic Motivation

Agents who "love their craft" effectively implement reward shaping:

<div class="math-block">
$$
r_{\text{shaped}}(s,a,s') = r_{\text{external}}(s') + F(s,a,s')
$$
</div>

where the shaping function $F$ represents intrinsic enjoyment:

<div class="math-block">
$$
F(s,a,s') = \beta_1 \cdot \underbrace{\text{flow}(s,a)}_{\text{Being in the zone}} + \beta_2 \cdot \underbrace{\Delta\text{skill}(s,s')}_{\text{Learning progress}} + \beta_3 \cdot \underbrace{\text{autonomy}(a)}_{\text{Self-direction}}
$$
</div>

### The Flow State as Optimal Policy Execution

The psychological flow state corresponds to optimal policy execution with intrinsic rewards:

<div class="math-block">
$$
\text{flow}(s,a) = \exp\left(-\frac{(\text{challenge}(s,a) - \text{skill}(s))^2}{2\sigma^2}\right)
$$
</div>

This creates a self-balancing system where:
- Too easy → Low flow reward → Seek harder challenges
- Too hard → Low flow reward → Build more skills
- Just right → Maximum intrinsic reward → Sustained engagement

### Exploration Through Curiosity

Intrinsically motivated agents naturally explore more:

<div class="math-block">
$$
r_{\text{curiosity}}(s,a,s') = \eta \cdot \underbrace{\|s' - \hat{s}'(s,a)\|}_{\text{Prediction error}} + \lambda \cdot \underbrace{H[\pi(\cdot|s)]}_{\text{Policy entropy}}
$$
</div>

This leads to:
1. **Broader skill acquisition**: Exploring diverse state-action pairs
2. **Robustness**: Multiple paths to success
3. **Innovation**: Discovering novel solutions

### The Compound Effect of Daily Joy

Consider the cumulative effect over a career spanning $T$ days:

**Extrinsic-only agent:**
<div class="math-block">
$$
V_{\text{extrinsic}} = \gamma^T \cdot R_{\text{success}} + \sum_{t=0}^{T-1} \gamma^t \cdot \underbrace{0}_{\text{No daily reward}}
$$
</div>

**Intrinsic-motivated agent:**
<div class="math-block">
$$
V_{\text{intrinsic}} = \gamma^T \cdot R_{\text{success}} + \sum_{t=0}^{T-1} \gamma^t \cdot r_{\text{daily joy}}
$$
</div>

Even with small daily rewards, the cumulative difference is massive:
<div class="math-block">
$$
\Delta V = \sum_{t=0}^{T-1} \gamma^t \cdot r_{\text{daily}} \approx \frac{r_{\text{daily}}}{1-\gamma} \text{ for large } T
$$
</div>

### Sustainability and Burnout Prevention

The burnout phenomenon can be modeled as resource depletion:

<div class="math-block">
$$
\text{energy}_{t+1} = \text{energy}_t - \underbrace{\text{cost}(a_t)}_{\text{Effort expended}} + \underbrace{r_{\text{intrinsic}}(s_t,a_t)}_{\text{Energy restored}}
$$
</div>

Without intrinsic rewards:
<div class="math-block">
$$
\text{energy}_T = \text{energy}_0 - \sum_{t=0}^{T} \text{cost}(a_t) < 0 \text{ (burnout)}
$$
</div>

With intrinsic rewards:
<div class="math-block">
$$
\text{energy}_T = \text{energy}_0 + \sum_{t=0}^{T} [r_{\text{intrinsic}}(s_t,a_t) - \text{cost}(a_t)] > 0 \text{ (sustainable)}
$$
</div>

### The Mastery Gradient

Loving your craft creates a positive feedback loop:

<div class="math-block">
$$
\frac{d\text{skill}}{dt} = \alpha \cdot \text{practice\_hours} \cdot \underbrace{(1 + \theta \cdot \text{enjoyment})}_{\text{Engagement multiplier}}
$$
</div>

This leads to exponential skill growth:
<div class="math-block">
$$
\text{skill}(t) = \text{skill}_0 \cdot \exp\left(\alpha \int_0^t \text{practice}(\tau) \cdot (1 + \theta \cdot \text{enjoyment}(\tau)) d\tau\right)
$$
</div>

### Why Top Performers Love Their Craft

The correlation between success and craft-love isn't coincidental but causal:

1. **Selection Effect**: Those who enjoy the process self-select into the field
   <div class="math-block">
   $$P(\text{stays in field} | \text{loves craft}) >> P(\text{stays in field} | \text{hates craft})$$
   </div>

2. **Performance Effect**: Intrinsic motivation improves performance
   <div class="math-block">
   $$\mathbb{E}[\text{performance} | \text{loves craft}] > \mathbb{E}[\text{performance} | \text{external motivation only}]$$
   </div>

3. **Compounding Effect**: More practice + better practice = exponential growth
   <div class="math-block">
   $$\text{total\_skill} = \int_0^T \underbrace{\text{hours}(t)}_{\text{More hours}} \cdot \underbrace{\text{quality}(t)}_{\text{Better quality}} dt$$
   </div>

### The Optimal Policy: Maximize Process Utility

The mathematically optimal life policy is:

<div class="math-block">
$$
\pi^* = \arg\max_{\pi} \mathbb{E}_{\pi}\left[\sum_{t=0}^{\infty} \gamma^t \left(\alpha \cdot r_{\text{process}}(s_t,a_t) + (1-\alpha) \cdot r_{\text{outcome}}(s_t')\right)\right]
$$
</div>

with $\alpha > 0.5$, ensuring that more than half the reward comes from the journey itself.

### Practical Implementation Strategies

1. **Gamification of Practice**:
   <div class="math-block">
   $$r_{\text{gamified}}(s,a) = \sum_i \mathbb{1}[\text{achievement}_i(s,a)] \cdot r_i$$
   </div>

2. **Social Rewards**:
   <div class="math-block">
   $$r_{\text{social}}(s,a) = \beta \cdot \text{community\_engagement}(a) + \gamma \cdot \text{peer\_recognition}(s')$$
   </div>

3. **Progress Tracking**:
   <div class="math-block">
   $$r_{\text{progress}}(s,s') = \log\left(\frac{\text{skill}(s')}{\text{skill}(s)}\right)$$
   </div>

### The Paradox of Detachment

Ironically, those who optimize for process often achieve better outcomes:

<div class="math-block">
$$
P(\text{success} | \text{process-focused}) > P(\text{success} | \text{outcome-focused})
$$
</div>

This occurs because:
- **Reduced anxiety**: $\text{performance} = \text{skill} - \text{anxiety}$
- **Increased practice**: 
<div class="math-block">
$$
\text{total\_practice} = \int_0^T \text{enjoyment}(t) \cdot \text{available\_time}(t) dt
$$
</div>

- **Better learning**: Positive emotions enhance neuroplasticity

### Neurological Basis

The brain's reward systems support this framework:

- **Dopamine**: Released during enjoyable practice, not just outcomes
- **Endorphins**: Generated by flow states
- **Serotonin**: Increased by mastery and progress
- **Oxytocin**: Released through community engagement

The neural value function becomes:
<div class="math-block">
$$
V_{\text{neural}}(s) = w_{\text{DA}}^T \phi_{\text{anticipation}}(s) + w_{\text{5HT}}^T \phi_{\text{satisfaction}}(s) + w_{\text{OT}}^T \phi_{\text{connection}}(s)
$$
</div>

### Conclusion: The Mathematics of Fulfillment

The advice to "love your craft" and "enjoy the journey" isn't just feel-good wisdom—it's mathematically optimal for:

1. **Learning efficiency**: Positive emotions enhance neuroplasticity
2. **Sustainability**: Intrinsic rewards prevent burnout
3. **Performance**: Reduced anxiety and increased practice
4. **Innovation**: Curiosity-driven exploration
5. **Life satisfaction**: Cumulative daily joy dominates end-state rewards

The optimal human policy isn't to suffer now for future rewards, but to structure life so that the journey itself is rewarding. This creates a positive spiral where enjoyment leads to practice, practice leads to mastery, and mastery leads to both intrinsic satisfaction and external success.

## World Models and Learning: The Value of Accurate Mental Representations

Humans operate with internal world models that approximate reality. The accuracy of these models directly impacts our ability to maximize rewards. Let's formalize how learning improves these models and why this matters for optimal decision-making.

### The World Model Framework

Every human maintains an internal model of how the world works:

<div class="math-block">
$$
\hat{\mathcal{M}} = (\hat{\mathcal{P}}, \hat{\mathcal{R}})
$$
</div>

where:
- $\hat{\mathcal{P}}(s'|s,a)$ is our belief about state transitions
- $\hat{\mathcal{R}}(s,a,s')$ is our belief about rewards

The true world model is:
<div class="math-block">
$$
\mathcal{M}^* = (\mathcal{P}^*, \mathcal{R}^*)
$$
</div>

### Model Error and Suboptimal Policies

The error in our world model leads to suboptimal policies:

<div class="math-block">
$$
\epsilon_{\text{model}} = \mathbb{E}_{s,a}\left[\text{KL}\left(\mathcal{P}^*(\cdot|s,a) \| \hat{\mathcal{P}}(\cdot|s,a)\right)\right] + \mathbb{E}_{s,a,s'}\left[|\mathcal{R}^*(s,a,s') - \hat{\mathcal{R}}(s,a,s')|\right]
$$
</div>

This model error bounds the suboptimality of our policy:

<div class="math-block">
$$
V^{\pi^*}(s) - V^{\hat{\pi}}(s) \leq \frac{2\gamma}{(1-\gamma)^2} \cdot \epsilon_{\text{model}} \cdot R_{\max}
$$
</div>

where $\hat{\pi}$ is the optimal policy under our incorrect model $\hat{\mathcal{M}}$.

### Learning as Model Refinement

Education and experience refine our world model through Bayesian updates:

<div class="math-block">
$$
\hat{\mathcal{P}}_{t+1}(s'|s,a) = \frac{\hat{\mathcal{P}}_t(s'|s,a) \cdot P(o_t|s',s,a)}{\sum_{s''} \hat{\mathcal{P}}_t(s''|s,a) \cdot P(o_t|s'',s,a)}
$$
</div>

where $o_t$ is the observation at time $t$.

### The Value of Information

Before learning, our expected value is:

<div class="math-block">
$$
V_{\text{before}} = \mathbb{E}_{\hat{\mathcal{M}}}\left[\sum_{t=0}^{\infty} \gamma^t r_t\right]
$$
</div>

After learning that improves model accuracy:

<div class="math-block">
$$
V_{\text{after}} = \mathbb{E}_{\hat{\mathcal{M}}'}\left[\sum_{t=0}^{\infty} \gamma^t r_t\right]
$$
</div>

The value of learning is:
<div class="math-block">
$$
\text{VOI} = V_{\text{after}} - V_{\text{before}} \propto \text{reduction in } \epsilon_{\text{model}}
$$
</div>

### Types of Model Errors and Their Consequences

#### 1. Transition Model Errors
Misunderstanding cause and effect:

<div class="math-block">
$$
\hat{\mathcal{P}}_{\text{naive}}(\text{wealth}|\text{lottery ticket}) >> \mathcal{P}^*(\text{wealth}|\text{lottery ticket})
$$
</div>

This leads to poor financial decisions with expected value:
<div class="math-block">
$$
\mathbb{E}[\text{return}] = \sum_{s'} \hat{\mathcal{P}}(s'|s,a) \cdot r(s') < 0
$$
</div>

#### 2. Reward Model Errors
Misunderstanding what brings happiness:

<div class="math-block">
$$
\hat{\mathcal{R}}_{\text{naive}}(\text{luxury goods}) >> \mathcal{R}^*(\text{luxury goods})
$$
</div>

Leading to hedonic adaptation surprises:
<div class="math-block">
$$
r_{\text{actual}}(t) = r_{\text{initial}} \cdot \exp(-\lambda t) \neq r_{\text{expected}} = r_{\text{initial}}
$$
</div>

#### 3. State Space Errors
Not considering all relevant variables:

<div class="math-block">
$$
\hat{\mathcal{S}} \subset \mathcal{S}^*
$$
</div>

Missing crucial factors like:
- Long-term health consequences
- Relationship impacts
- Opportunity costs

### The Dunning-Kruger Effect in RL Terms

Beginners often have high confidence with poor models:

<div class="math-block">
$$
\text{Confidence}_{\text{beginner}} = \frac{1}{\text{Var}[\hat{\mathcal{P}}]} \text{ (low variance, wrong mean)}
$$
</div>

Experts have calibrated uncertainty:
<div class="math-block">
$$
\text{Confidence}_{\text{expert}} \propto \frac{1}{\text{Var}[\hat{\mathcal{P}}]} \text{ (appropriate variance, accurate mean)}
$$
</div>

### Model Complexity and Sample Efficiency

Simple models learn faster but have limited capacity:

<div class="math-block">
$$
\epsilon_{\text{bias}}^{\text{simple}} = \min_{\theta \in \Theta_{\text{simple}}} \mathbb{E}[(\mathcal{P}^* - \hat{\mathcal{P}}_\theta)^2]
$$
</div>

Complex models can be more accurate but need more data:

<div class="math-block">
$$
\epsilon_{\text{total}} = \underbrace{\epsilon_{\text{bias}}}_{\text{Model limitations}} + \underbrace{\epsilon_{\text{variance}}}_{\text{Estimation error}} \propto \frac{|\Theta|}{n}
$$
</div>

### The Exploration-Exploitation Tradeoff in Learning

Optimal learning balances:

<div class="math-block">
$$
a_t = \begin{cases}
\arg\max_a Q(s_t,a) & \text{with probability } 1-\epsilon_t \text{ (exploit)} \\
\text{learning action} & \text{with probability } \epsilon_t \text{ (explore)}
\end{cases}
$$
</div>

where $\epsilon_t$ should decay as model accuracy improves:
<div class="math-block">
$$
\epsilon_t = \epsilon_0 \cdot \exp(-\beta \cdot \text{model\_accuracy}_t)
$$
</div>

### Formal Education as Systematic Model Improvement

Education provides:

1. **Theoretical Frameworks**: Compressed experience
   <div class="math-block">
   $$\hat{\mathcal{P}}_{\text{educated}} = \text{aggregate}(\{\mathcal{P}_i\}_{i=1}^{N_{\text{historical}}})$$
   </div>

2. **Causal Understanding**: Beyond correlation
   <div class="math-block">
   $$P(Y|do(X)) \neq P(Y|X)$$
   </div>

3. **Abstract Reasoning**: Generalization across domains
   <div class="math-block">
   $$\hat{\mathcal{P}}_{\text{abstract}}(s'|s,a) = f_{\text{general}}(\phi(s), \phi(a))$$
   </div>

### The Compound Effect of Better Models

Small improvements in model accuracy compound over time:

<div class="math-block">
$$
V_T = \sum_{t=0}^{T} \gamma^t \cdot r_t^{\pi_{\hat{\mathcal{M}}}}
$$
</div>

With better models:
<div class="math-block">
$$
\Delta V_T = \sum_{t=0}^{T} \gamma^t \cdot (r_t^{\pi_{\hat{\mathcal{M}}'}} - r_t^{\pi_{\hat{\mathcal{M}}}}) \approx T \cdot \Delta r_{\text{avg}}
$$
</div>

Over a lifetime, this difference is enormous.

### Meta-Learning: Learning How to Learn

Humans can improve their model-learning process itself:

<div class="math-block">
$$
\frac{d\hat{\mathcal{M}}}{dt} = \alpha(t) \cdot \nabla_{\hat{\mathcal{M}}} \mathcal{L}(\hat{\mathcal{M}}, \text{data})
$$
</div>

where the learning rate $\alpha(t)$ can be optimized:

<div class="math-block">
$$
\alpha^*(t) = \arg\max_{\alpha} \mathbb{E}\left[\text{future model accuracy}|\alpha\right]
$$
</div>

### Practical Implications

1. **Invest in Learning**: The ROI on model improvement is:
   <div class="math-block">
   $$\text{ROI}_{\text{learning}} = \frac{\Delta V_{\text{lifetime}}}{\text{cost}_{\text{learning}}} >> 1$$
   </div>

2. **Seek Diverse Experiences**: Broader data improves generalization
   <div class="math-block">
   $$\text{Var}[\hat{\mathcal{P}}] \propto \frac{1}{\text{diversity of experiences}}$$
   </div>

3. **Question Assumptions**: Regularly test model predictions
   <div class="math-block">
   $$\text{model\_update} = \begin{cases}
   \text{large} & \text{if prediction error high} \\
   \text{small} & \text{if prediction error low}
   \end{cases}$$
   </div>

### The Wisdom Premium

The value difference between accurate and inaccurate models:

<div class="math-block">
$$
\Delta V = V^{\pi^*_{\mathcal{M}^*}} - V^{\pi^*_{\hat{\mathcal{M}}}} = \mathbb{E}_{\text{lifetime}}\left[\sum_t \gamma^t \cdot \text{cost}_{\text{mistakes}}(t)\right]
$$
</div>

This "wisdom premium" justifies significant investment in education, mentorship, and deliberate practice.

### Conclusion: Knowledge as Compound Interest

Better world models lead to better decisions, which compound over time. The earlier and more systematically we improve our models, the greater the lifetime benefit. This mathematical framework explains why:

- Education has such high returns
- Experience is valuable
- Wise mentors are sought after
- Continuous learning is optimal
- Humility (acknowledging model uncertainty) leads to better outcomes

The human journey can be seen as a continuous process of refining our world model to better navigate reality and maximize both immediate and long-term rewards.


## Will add later
- There are some space that have high value function, for example being born righ