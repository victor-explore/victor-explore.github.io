---
title: "Bitcoin Wallet Decision: Ledger vs Coldcard"
date: 2025-06-27
draft: false
description:
tags: []
categories: []
author:
toc:
weight: 1
---

# Bitcoin Wallet Security Decision: Should I Switch from Ledger to Coldcard?

## The Situation

You have **20 lakh INR** worth of Bitcoin stored in a Ledger hardware wallet. However, you're concerned because:
- Ledger's code is **not open source**
- There's a risk Ledger could "go rogue" and compromise your funds
- You're considering switching to **Coldcard** (open source alternative)
- Switching costs: **19,000 INR** + **0.1% risk** of losing money during transfer
- Your assessment: **2% chance** Ledger goes rogue

## Decision Framework Application

### Step 1: Define Action Space

$$
A = \{\text{Stay with Ledger}, \text{Switch to Coldcard}\}
$$

### Step 2: Identify States of Nature

$$
S = \{\text{Ledger goes rogue}, \text{Ledger stays safe}\}
$$

### Step 3: Assign Probabilities

$$
P(\text{Ledger goes rogue}) = 0.02 = 2\%
$$
$$
P(\text{Ledger stays safe}) = 0.98 = 98\%
$$

### Step 4: Calculate Outcomes and Utilities

**Action: Stay with Ledger**

- **If Ledger goes rogue** ($s_1$): You lose entire 20 lakh INR
  $$U(\text{Stay}, \text{rogue}) = -20,00,000 \text{ INR}$$

- **If Ledger stays safe** ($s_2$): You keep all 20 lakh INR
  $$U(\text{Stay}, \text{safe}) = 0 \text{ INR}$$

**Action: Switch to Coldcard**

- **If Ledger would have gone rogue** ($s_1$): You save your money but pay switching costs
  $$U(\text{Switch}, \text{would be rogue}) = -19,000 \text{ INR (switching cost)}$$

- **If Ledger would have stayed safe** ($s_2$): You pay switching costs unnecessarily
  $$U(\text{Switch}, \text{would be safe}) = -19,000 \text{ INR (switching cost)}$$

**Note**: 0.1% transfer risk is negligible compared to other factors, so excluded for simplicity.

### Step 5: Calculate Expected Utilities

**Expected Utility of Staying with Ledger:**

$$
\begin{align}
EU(\text{Stay}) &= P(\text{rogue}) \times U(\text{Stay}, \text{rogue}) + P(\text{safe}) \times U(\text{Stay}, \text{safe}) \\
&= 0.02 \times (-20,00,000) + 0.98 \times (0) \\
&= -40,000 + 0 \\
&= -40,000 \text{ INR}
\end{align}
$$

**Expected Utility of Switching to Coldcard:**

$$
\begin{align}
EU(\text{Switch}) &= P(\text{rogue}) \times U(\text{Switch}, \text{would be rogue}) + P(\text{safe}) \times U(\text{Switch}, \text{would be safe}) \\
&= 0.02 \times (-19,000) + 0.98 \times (-19,000) \\
&= -380 + (-18,620) \\
&= -19,000 \text{ INR}
\end{align}
$$

### Step 6: Select Optimal Action

**Comparison of Expected Utilities:**
$$
EU(\text{Switch}) = -19,000 > EU(\text{Stay}) = -40,000
$$

**Optimal Decision:**

$$
a^* = \arg\max_{a \in A} EU(a) = \text{Switch to Coldcard}
$$

## Key Insights

**Why Switching is Optimal:**

- **Expected loss from staying**: 40,000 INR
- **Guaranteed cost of switching**: 19,000 INR  
- **Net benefit of switching**: 21,000 INR in expected value

**Risk Analysis:**
- **2% chance** of losing 20 lakh INR = Expected loss of 40,000 INR
- **100% chance** of paying 19,000 INR switching cost
- **Switching saves 21,000 INR** in expected value

**Mathematical Certainty:**
$$
\text{Expected Savings} = 40,000 - 19,000 = 21,000 \text{ INR}
$$

## Decision: Switch to Coldcard

Based on the mathematical analysis, you should **switch to Coldcard** because it minimizes your expected loss by 21,000 INR compared to staying with Ledger.

The 19,000 INR switching cost is a worthwhile insurance premium against the 2% risk of losing your entire 20 lakh INR holding.
