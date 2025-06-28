---
title: "Decision making framework using probability theory"
date: 2025-06-28
draft: false
description:
tags: []
categories: []
author:
toc:
weight: 1
---
This is how you can use Probability Theory to make decisions.

## Fundamental Terminologies

Before diving into the mathematical framework, let's establish the key concepts that form the foundation of decision-making under uncertainty.

### Action
An **action** is a choice or decision that a decision-maker can take from a set of available alternatives. It represents what we can control or decide upon in a given situation.

**Mathematical Notation:**
$$
a \in A
$$

Where:
- $a$ = a specific action
- $A$ = the set of all possible actions (action space)

**Example:** If you're deciding how to invest $1000, your action space might be:
<div class="math-katex-block">
$$
A = \{\text{stocks}, \text{bonds}, \text{savings account}, \text{real estate}\}
$$
</div>

### The Fundamental Limitation: Single Action Selection

In most decision-making scenarios, we face a crucial constraint: **we can choose only one action** from our available set at any given time. This limitation is what makes decision-making challenging and requires careful analysis.

**Mathematical Representation:**
$$
\text{Choose } a^* \text{ such that } a^* \in A \text{ and } |selected| = 1
$$

**Breaking Down the Notation:**
- $a^*$ = the optimal or chosen action
- $a^* \in A$ = the chosen action must be from our available action set
- $|selected| = 1$ = the **cardinality** (size) of our selection must be exactly 1

**In Simple Terms:** We must choose exactly one action - no more, no less.

This constraint means:
- We cannot hedge by selecting multiple actions simultaneously (in most cases)
- We must commit to one choice despite uncertainty
- The opportunity cost of not choosing other actions becomes relevant
- Every decision involves trade-offs

**Why This Matters:**
- **Irreversibility**: Once an action is taken, we often cannot undo it
- **Opportunity Cost**: Choosing one action means giving up potential benefits from other actions
- **Resource Constraint**: We typically have limited resources (time, money, attention) to allocate

### Outcome
An **outcome** is the result or consequence that occurs after taking a specific action. Outcomes are often uncertain and depend on factors beyond our control (states of nature).

**Mathematical Notation:**
$$
o = f(a, s)
$$

Where:
- $o$ = outcome
- $a$ = chosen action  
- $s$ = state of nature (uncertain factors)
- $f$ = outcome function that maps actions and states to results

**Key Characteristics of Outcomes:**
- **Uncertainty**: We typically don't know the exact outcome before taking action
- **Action-Dependent**: Different actions lead to different possible outcomes
- **State-Dependent**: The same action can yield different outcomes depending on external circumstances

**Example:** 
If you choose to invest in stocks ($a = \text{stocks}$), your outcome depends on market conditions:
- If market goes up ($s_1$): $o_1 = \text{positive return}$
- If market goes down ($s_2$): $o_2 = \text{negative return}$

This uncertainty in outcomes is precisely why we need probability theory to make optimal decisions.

## Descion making framework - Then how do we make decisions?

Given the fundamental challenge that we must choose exactly **one action** despite uncertain outcomes, how do we make optimal decisions? The answer lies in developing a systematic framework that handles uncertainty mathematically.

### The Core Challenge

We face a fundamental asymmetry in decision-making:
- **What we control**: Our choice of action $a \in A$
- **What we don't control**: The state of nature $s$ that determines outcomes
- **What we want**: To maximize our benefit despite this uncertainty

### The Expected Value Framework

Since outcomes are uncertain, we cannot simply compare deterministic results. Instead, we must work with **expected outcomes** - the average result we anticipate across all possible scenarios.

#### Step 1: Quantify Uncertainty with Probabilities

First, we assign probabilities to different states of nature:

$$
P(s) \text{ for each possible state } s \in S
$$

Where:
- $S$ = set of all possible states of nature
- $P(s)$ = probability that state $s$ occurs
- $\sum_{s \in S} P(s) = 1$ (probabilities must sum to 1)

**Intuition**: These probabilities represent our **beliefs** about how likely different scenarios are to occur.

**Evolutionary Bias Warning**: Humans systematically overestimate probabilities of negative outcomes due to evolutionary survival mechanisms - our ancestors who feared threats lived longer than optimists who ignored dangers.

**Note on Information Gathering**: Most relevant situational information is readily available through basic observation and research. Excessive attempts to gather more precise probabilities (like trying to gauge your boss's emotions or mood) often produces noise rather than signal and can become anxiety-inducing without meaningfully improving decision quality.

#### Step 2: Define Utility Function

We need a way to measure the **value** or **utility** we derive from different outcomes:

$$
U(o) = U(f(a,s))
$$

Where:
- $U(\cdot)$ = utility function that converts outcomes to numerical values
- Higher utility values represent more preferred outcomes

**Why We Need This**: 
- Different outcomes have different values to us
- We need a common scale to compare diverse results
- Utility captures our preferences mathematically

#### How to Assign Utilities: The Personal Nature of Values

**Critical Insight**: Only **you** can assign your utility function. No external authority, expert, or algorithm can determine what outcomes should matter to you or how much.

**Requirements for Proper Utility Assignment**:
- **Know Your Values**: Understand what truly matters to you beyond surface-level desires
- **Clarify Your Ultimate Objectives**: What are you ultimately trying to achieve in life?
- **Ignore External Noise**: Don't let fear, social pressure, or others' opinions distort your true preferences

**The Bezos Framework**: Jeff Bezos uses a powerful heuristic for major decisions - he asks: *"How will I feel about this outcome when I'm on my deathbed?"* This approach:
- **Eliminates short-term noise**: Focuses on what truly matters long-term
- **Clarifies priorities**: Separates genuine values from temporary concerns  
- **Incorporates time preference**: Weighs future satisfaction appropriately

#### Step 3: Calculate Expected Utility

For each possible action, we calculate its **expected utility**:

<div class="math-katex-block">
$$
\underbrace{EU(a)}_{\substack{\text{Expected utility} \\ \text{of action } a}} = \underbrace{\sum_{s \in S}}_{\substack{\text{Sum over all} \\ \text{possible states}}} \underbrace{P(s)}_{\substack{\text{Probability of} \\ \text{state } s}} \times \underbrace{U(f(a,s))}_{\substack{\text{Utility of outcome} \\ \text{when taking action } a \\ \text{in state } s}}
$$
</div>

**Breaking This Down**:
- **$P(s)$**: How likely each scenario is
- **$U(f(a,s))$**: How much we value the outcome in each scenario  
- **Product $P(s) \times U(f(a,s))$**: Weighted value of each scenario
- **Sum**: Average across all possible scenarios

**Intuition**: Expected utility is the **probability-weighted average** of all possible outcomes from taking action $a$.

#### Step 4: Choose Optimal Action

Finally, we select the action that maximizes expected utility:

<div class="math-katex-block">
$$
\underbrace{a^*}_{\substack{\text{Optimal} \\ \text{action}}} = \underbrace{\arg\max_{a \in A}}_{\substack{\text{Action that} \\ \text{maximizes}}} \underbrace{EU(a)}_{\substack{\text{Expected} \\ \text{utility}}}
$$
</div>

**Mathematical Expansion**:
$$
a^* = \arg\max_{a \in A} \left[ \sum_{s \in S} P(s) \times U(f(a,s)) \right]
$$

### The Complete Decision-Making Process

The optimal decision-making framework follows these sequential steps:

1. **Identify Action Space**: Define all possible actions $A$
2. **Identify State Space**: Define all possible states of nature $S$  
3. **Assign Probabilities**: Determine $P(s)$ for each state $s \in S$
4. **Define Outcomes**: Specify outcome function $f(a,s)$ for each action-state pair
5. **Assign Utilities**: Define utility function $U(\cdot)$ for all possible outcomes
6. **Calculate Expected Utilities**: Compute $EU(a)$ for each action $a \in A$
7. **Select Optimal Action**: Choose $a^* = \arg\max_{a \in A} EU(a)$

### Why This Framework Works

This approach is optimal because it:

- **Handles Uncertainty**: Uses probabilities to account for unknown future states
- **Incorporates Preferences**: Utility function captures what we actually care about
- **Provides Ranking**: Gives us a systematic way to compare different actions
- **Maximizes Expected Benefit**: Chooses the action with highest average payoff
- **Is Mathematically Rigorous**: Based on solid foundations from probability theory

### Key Insight: Trading Present Certainty for Future Expected Value

The framework essentially asks: *"Given what I know about the probabilities and my preferences, which action gives me the best expected outcome?"*

This transforms the impossible task of predicting the future into the manageable task of:
- Estimating probabilities based on available information
- Clearly defining what we value (utility function)
- Performing mathematical calculations

**The Beauty of This Approach**: We can make optimal decisions even under uncertainty by working with expectations rather than trying to eliminate uncertainty entirely.

## Post-Decision Confidence: Why You Should Not Fear

Once you've made a decision using this expected utility framework, there's an important psychological principle to remember during the waiting period before outcomes materialize:

### The Optimality Guarantee

**Mathematical Certainty:**
$$
\text{Given } \{P(s), U(\cdot), A, S\} \text{ at time } t_0, \text{ you chose } a^* = \arg\max_{a \in A} EU(a)
$$

**Key Insight**: You **cannot have done better** with the information available at decision time $t_0$.

### Why Fear is Irrational

• **Optimal Choice**: You selected the action with highest expected utility given available information
• **Information Constraint**: You worked with the best probability estimates possible at $t_0$  
• **Hindsight Bias**: Future information doesn't invalidate past optimality
• **Process vs. Outcome**: A good process can yield bad outcomes due to randomness

### The Waiting Period Principle

**During the delay between decision and outcome:**

$$
\text{Confidence Level} = \text{High, because } EU(a^*) \geq EU(a) \text{ } \forall a \in A
$$

**Translation**: Rest easy knowing that no alternative action had higher expected value given what you knew when you decided.

**Remember**: Outcomes are governed by probability, but decisions should be governed by **expected utility maximization**. You played the game optimally.

## Real-World Example: The Remote Work Dilemma

Let's apply our decision-making framework to a practical scenario that many remote workers face.

### The Situation

You have a work-from-home job with a policy requiring availability to come to the office when called. You're considering whether to take a vacation to Bali or stay in your city.

### Step 1: Define the Action Space

$$
A = \{\text{Go to Bali}, \text{Stay in city}\}
$$

### Step 2: Identify States of Nature

$$
S = \{\text{Office calls}, \text{Office doesn't call}\}
$$

### Step 3: Assign Probabilities

Based on your experience and company patterns:

$$
P(\text{Office calls}) = 0.2
$$
$$
P(\text{Office doesn't call}) = 0.8
$$

**Note**: These probabilities sum to 1, satisfying our requirement: $0.2 + 0.8 = 1.0$

### Step 4: Define Outcome Function and Utilities

For each action-state combination, we determine the utility:

**Action: Go to Bali**
- **If office calls** ($s_1$): You're unavailable when needed
  $$U(\text{Bali}, \text{calls}) = -7$$
  
Note: This relatively low penalty reflects confidence in job market alternatives due to strong skills. For others with different career security, this utility might be much more negative.

- **If office doesn't call** ($s_2$): You enjoy vacation without consequences  
  $$U(\text{Bali}, \text{no calls}) = +10$$

**Action: Stay in City**
- **If office calls** ($s_1$): You're available as required (neutral outcome)
  $$U(\text{Stay}, \text{calls}) = 0$$
- **If office doesn't call** ($s_2$): You miss vacation opportunity for nothing
  $$U(\text{Stay}, \text{no calls}) = -10$$

### Step 5: Calculate Expected Utilities

**Expected Utility of Going to Bali:**

<div class="math-katex-block">
$$
\begin{align}
EU(\text{Go to Bali}) &= \underbrace{P(\text{calls})}_{\text{0.2}} \times \underbrace{U(\text{Bali}, \text{calls})}_{\text{-7}} + \underbrace{P(\text{no calls})}_{\text{0.8}} \times \underbrace{U(\text{Bali}, \text{no calls})}_{\text{+10}} \\
&= 0.2 \times (-7) + 0.8 \times (+10) \\
&= -1.4 + 8.0 \\
&= +6.6
\end{align}
$$
</div>

**Expected Utility of Staying in City:**

<div class="math-katex-block">
$$
\begin{align}
EU(\text{Stay in city}) &= \underbrace{P(\text{calls})}_{\text{0.2}} \times \underbrace{U(\text{Stay}, \text{calls})}_{\text{0}} + \underbrace{P(\text{no calls})}_{\text{0.8}} \times \underbrace{U(\text{Stay}, \text{no calls})}_{\text{-10}} \\
&= 0.2 \times (0) + 0.8 \times (-10) \\
&= 0 + (-8.0) \\
&= -8.0
\end{align}
$$
</div>

### Step 6: Select Optimal Action

**Comparison of Expected Utilities:**
$$
EU(\text{Go to Bali}) = +6.6 > EU(\text{Stay in city}) = -8.0
$$

**Optimal Decision:**

<div class="math-katex-block">
$$
a^* = \arg\max_{a \in A} EU(a) = \text{Go to Bali}
$$
</div>

### Key Insights from This Example

**Why "Go to Bali" is Optimal:**

• **High probability of no consequences**: 80% chance office won't call
• **Large opportunity cost of staying**: Missing vacation yields -10 utility in most likely scenario  
• **Expected value dominance**: +6.6 > -8.0 by a significant margin

**Risk vs. Reward Analysis:**
- **20% chance of penalty** (-7 utility) if caught
- **80% chance of reward** (+10 utility) if not called
- **Weighted average strongly favors** taking the risk

**What This Demonstrates:**
1. **Intuition can be misleading**: Fear of getting caught might make staying seem safer
2. **Mathematics provides clarity**: Expected utility calculation shows Bali is clearly better
3. **Probabilities matter**: Low probability of being called (0.2) makes risk acceptable
4. **Opportunity costs count**: The -10 utility of missing vacation when office doesn't call is significant

### Post-Decision Confidence

Once you choose to go to Bali, remember:

$$
\text{You made the mathematically optimal choice given your utilities and probability assessments}
$$

**If the office calls while you're in Bali:**
- You're not "unlucky" - this was a known 20% probability
- Your decision process was correct - you maximized expected utility
- The negative outcome doesn't invalidate the optimal choice

**Key Takeaway**: Good decisions can sometimes yield bad outcomes due to randomness, but that doesn't make them wrong decisions.

## The Execution Imperative: Commit to Your Optimal Decision

Once you've selected your optimal action $a^*$ using the expected utility framework, there's a critical final step that determines whether your mathematical rigor translates into real-world success.

### Execute Ferociously

**The Mathematical Foundation for Commitment:**

$$
\text{If } a^* = \arg\max_{a \in A} EU(a) \text{, then commit fully to executing } a^*
$$

**Why Ferocious Execution Matters:**

• **Optimization Requires Implementation**: The best decision in theory becomes worthless without proper execution

• **Partial Execution Destroys Expected Value**: Half-hearted implementation of $a^*$ yields suboptimal outcomes

• **Confidence Compounds**: Full commitment eliminates decision fatigue and second-guessing

### The Danger of Abandoning the Framework

**The Vulnerability Equation:**

$$
\text{No Framework} \Rightarrow \text{External Manipulation} \Rightarrow \text{Suboptimal Outcomes}
$$

**When you abandon this systematic approach, you become:**

#### A Leaf in the Wind

**Mathematical Representation of Chaos:**

**Decision Process:**
$$
\text{Decision} = f(\text{random external influence}) \text{ instead of } \text{Decision} = \arg\max_{a \in A} EU(a)
$$

**Resulting Utility:**
$$
EU(f(\text{random external influence})) < EU(\arg\max_{a \in A} EU(a)) = \max_{a \in A} EU(a)
$$

**What This Means:**
- **No Internal Compass**: Without your utility function, you can't evaluate what's actually good for you
- **Susceptible to Manipulation**: Others with clearer agendas will influence your choices
- **Inconsistent Choices**: Your decisions become random walks rather than optimized paths

#### The Exploitation by Others

**Key Insight**: People who haven't done the mathematical work to understand **your** utility function will make decisions that optimize **their** utility function, not yours.

**The Manipulation Equation:**
$$
\text{Others' Influence} = \arg\max_{a \in A} EU_{\text{them}}(a) \neq \arg\max_{a \in A} EU_{\text{you}}(a)
$$

**Where:**
- $EU_{\text{them}}(a)$ = Expected utility of action $a$ **for them**
- $EU_{\text{you}}(a)$ = Expected utility of action $a$ **for you**
- These are generally **not equal** and often **inversely related**

**Common Sources of Misaligned Advice:**

• **Family Members**: May prioritize their comfort/pride over your optimal path
• **Friends**: Often project their own fears and limitations onto your situation  
• **Colleagues**: May not want you to advance beyond them
• **Society**: Optimizes for conformity and predictability, not individual success
• **Media**: Optimizes for engagement and drama, not your well-being

### The Framework as Protection

**Your Decision Framework Acts As:**

#### A Firewall Against Manipulation

$$
\text{External Pressure} \xrightarrow{\text{Filtered by}} \text{Your Framework} \xrightarrow{\text{Results in}} \text{Optimal Decision}
$$

**The Filter Questions:**
- Does this external input change my probability estimates $P(s)$?
- Does this change my utility function $U(\cdot)$?
- If neither, then this input is **noise** and should be ignored

#### A Source of Unshakeable Confidence

**The Certainty Principle:**
$$
\text{Confidence} = f(\text{Mathematical Rigor of Decision Process})
$$

**When you know you've:**
- Properly defined your action space $A$
- Accurately estimated probabilities $P(s)$  
- Honestly assessed your utilities $U(\cdot)$
- Calculated expected utilities $EU(a)$ correctly
- Selected $a^* = \arg\max_{a \in A} EU(a)$

**Then external doubt becomes irrelevant** because you have mathematical proof of optimality.

### The Implementation Protocol

**Once $a^*$ is determined:**

1. **Immediate Action**: Begin executing $a^*$ without delay
2. **Ignore External Noise**: Filter out opinions that don't update your probabilities or utilities  
3. **Full Commitment**: Execute with 100% intensity, not hedged half-measures
4. **Monitor Only Relevant Information**: Update $P(s)$ only when new **actual evidence** emerges
5. **Stay the Course**: Remember that random negative outcomes don't invalidate optimal choices

### Managing Execution Anxiety: The Information Embargo

**The Fundamental Execution Principle:**

Once you've committed to action $a^*$, there's a critical psychological trap to avoid that can undermine your optimal decision.

#### The Phone-Checking Trap

**Mathematical Reality:**
$$
\text{If you cannot reverse action } a^* \text{ mid-execution, then monitoring becomes pure anxiety generation}
$$

**Example from our Bali scenario:**
- **You've decided**: Go to Bali ($a^* = \text{Bali}$)
- **You're committed**: Flight booked, already traveling  
- **Reality**: You cannot instantly return if office calls
- **Checking phone constantly**: Generates anxiety without providing actionable options

#### The Anxiety-Uncertainty Equation

**The Source of Execution Anxiety:**
$$
\text{Anxiety} = f(\text{Uncertainty} \times \text{Perceived Control Illusion})
$$

**Breaking This Down:**
- **Uncertainty**: You don't know if office will call
- **Perceived Control Illusion**: Feeling like monitoring gives you control
- **Reality**: Monitoring doesn't change $P(\text{office calls})$ or your available responses

#### The Information Embargo Strategy

**Once action $a^*$ is irreversibly initiated:**

$$
\text{Stop Information Gathering} = \text{Eliminate Anxiety Source}
$$

**Practical Implementation:**
- **Turn off notifications** related to the uncertain outcome
- **Don't check email/phone** compulsively during execution
- **Remember**: The decision was optimal with available information at time $t_0$

#### Why This Works Mathematically

**The Irreversibility Theorem:**
$$
\text{If } P(\text{can reverse } a^*) = 0 \text{, then monitoring serves no utility maximization purpose}
$$

**Logical Chain:**
1. You chose $a^*$ because it maximizes expected utility
2. New information during execution cannot change this past optimality
3. You cannot act on new information mid-execution  
4. Therefore: Monitoring = Pure anxiety with zero benefit

#### The Execution Confidence Formula

**During the execution period:**
$$
\text{Confidence} = f(\text{Trust in Original Framework}) - g(\text{Unnecessary Information Seeking})
$$

**Where:**
- **Trust in Original Framework**: High, because you used optimal decision theory
- **Unnecessary Information Seeking**: Reduces confidence by highlighting uncertainties you cannot act upon

**The Result**: Information embargo during execution **maximizes confidence** and **minimizes anxiety**.

#### The Stoic Reinforcement: Marcus Aurelius's Divine Standard

**The Philosophical Complement to Mathematical Optimality:**

Marcus Aurelius, the philosopher-emperor, captured this principle perfectly:

> *"Even the gods cannot ask more from you than to do your best with what you have."*

**Mathematical Translation:**
$$
\text{If } a^* = \arg\max_{a \in A} EU(a) \text{ given information at } t_0, \text{ then no higher standard exists}
$$

**The Divine Optimality Principle:**
- **You used the best available information**: Your probability estimates $P(s)$ were based on all accessible data
- **You clarified your true values**: Your utility function $U(\cdot)$ reflected your authentic preferences  
- **You performed optimal calculation**: You selected $a^* = \arg\max_{a \in A} EU(a)$
- **Even divine beings** could not improve upon this process with the same information set

**Why This Matters:**
$$
\text{Optimal Decision Process} \Rightarrow \text{Moral and Logical Absolution from Outcome}
$$

**The Liberation:**
- **Bad outcomes** don't invalidate optimal decisions
- **Second-guessing** becomes philosophically unjustifiable  
- **External judgment** becomes irrelevant when you've met the divine standard
- **Anxiety** dissolves when you realize you've done everything possible

**The Complete Standard:**
$$
\text{Human Responsibility} = \text{Optimize Given Available Information} \neq \text{Control Outcomes}
$$

You are responsible for the **process**, not the **result**. Even gods work within the constraints of probability and information - and so do you.

### The Ultimate Protection

**The Framework Provides:**

$$
\text{Internal Sovereignty} = \text{Freedom from External Manipulation}
$$

**Because:**
- **You know what you want**: Clear utility function
- **You understand the probabilities**: Rational risk assessment  
- **You've done the math**: Confidence in your choice
- **You commit fully**: Maximum chance of success

**The Alternative - Being a Leaf in the Wind:**
- Decisions made by whoever speaks loudest
- No consistent direction or progress
- Constant second-guessing and anxiety
- Outcomes optimized for others, not you

### Final Principle: Trust Your Framework, Execute Ferociously

**The Complete Decision-Execution Cycle:**

<div class="math-katex-block">
$$
\underbrace{\text{Mathematical Analysis}}_{\substack{\text{Calculate } a^* = \arg\max EU(a)}} \xrightarrow{\text{Leads to}} \underbrace{\text{Ferocious Execution}}_{\substack{\text{Implement } a^* \text{ with full commitment}}} \xrightarrow{\text{Results in}} \underbrace{\text{Optimal Outcomes}}_{\substack{\text{Maximum expected utility} \\ \text{achieved in practice}}}
$$
</div>

**Remember**: The worst thing that can happen is not a bad outcome from your optimal choice - it's abandoning the framework and becoming vulnerable to those who don't have your best interests at heart.

**Your framework is your shield. Your execution is your sword. Use both.**
