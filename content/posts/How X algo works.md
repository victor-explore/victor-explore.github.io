---
title: "The Complete Guide to the X 'For You' Algorithm"
date: 2025-05-24
draft: false
description: "A technical breakdown of how X's recommendation algorithm actually works, based on the open-source code. Not speculation—just what the code says."
tags: ["x", "twitter", "algorithm", "social media", "growth"]
categories: ["guides"]
author: "victor_explore"
toc: true
weight: 1
---

*Everything in this guide comes from the actual source code—not speculation, not "what worked for me," not guru theories.*

---

## Chapter 1: Introduction

### Forget Everything You've Heard

You've heard the advice.

"Post at 9am for maximum reach."

"Use trending hashtags."

"The algorithm suppresses certain topics."

"Blue checkmarks get boosted."

"Follow-to-follower ratio matters."

"Post more to grow faster."

You've heard the gurus—the ones selling courses, the ones with "X Growth Expert" in their bio, the ones who claim they've "cracked the algorithm."

Here's the problem: most of it is speculation. Or outdated. Or just wrong.

I'm not saying these people are lying. Many genuinely believe what they're teaching. But they're making educated guesses based on what _seems_ to work—not what actually happens inside the algorithm.

And here's the thing: **you don't have to guess anymore.**

---

### The Algorithm Is Open Source

X released the source code for their "For You" recommendation algorithm.

Not a blog post explaining it. Not a PR statement. The actual code.

The Rust services that serve your feed. The Python models that predict engagement. The scoring formulas that decide which posts rise and which ones die.

It's all public. Anyone can read it.

**The official announcement:** [X Engineering on X](https://x.com/XEng/status/2013471689087086804)

**The actual code:** [github.com/xai-org/x-algorithm](https://github.com/xai-org/x-algorithm)

The problem? Most people won't. The codebase is complex—hundreds of files across multiple languages, neural network architectures, distributed systems. Unless you're a software engineer who enjoys reading other people's code for fun, you're not going to dig through it.

That's where I come in.

---

### I Read the Code (So You Don't Have To)

I'm a coder by profession. Reading code is literally what I do.

When X released their algorithm, I went through it. Not skimming—actually reading. Following the data flow from when you open your app to when your feed appears. Tracing how a post gets scored, filtered, and ranked. Understanding what the neural networks actually predict.

This guide is the result.

**Who am I?** I'm [@victor_explore](https://x.com/victor_explore) on X.

Everything in this book comes from the actual source code—not speculation, not "what worked for me," not guru theories. When I say "the algorithm does X," I can point to the file and function where it happens.

Does this mean I know everything? No. Some parts of X's system aren't open source (more on that later). But for the core recommendation algorithm—the thing that decides what billions of people see every day—we now have visibility.

And what I found might surprise you.

---

### What This Guide Will Teach You

This isn't a "10 tips to grow on X" listicle. This is a technical breakdown of how the system actually works, translated into language that doesn't require a computer science degree.

Here's what you'll learn:

**How the algorithm decides what YOU see**

When you open X, what happens behind the scenes? How does it pick from millions of posts to show you exactly 30-50? What's your "behavior fingerprint" and how does it shape your feed?

**How the algorithm ranks YOUR posts for others**

When you post, how does the system decide who sees it? What's the scoring formula? What actions help you, and what actions kill your reach?

**Your algorithmic identity**

What is an author embedding, and why does it matter? How are embeddings trained through contrastive learning? Why does your track record compound over time?

**The Reply Game**

Why replies use the same algorithm but behave completely differently. How new accounts and pivoting accounts can bypass their untrained embeddings by reaching someone else's audience.

**The neural networks behind it all**

I'll explain transformers, embeddings, and two-tower models in plain English. You'll understand what the AI is actually predicting—and why that matters for your strategy.

**What you can actually do with this knowledge**

Practical advice for accounts of all sizes—whether you're just starting out, pivoting to a new niche, or looking to break through a growth plateau. Not generic tips—strategies grounded in how the system actually works.

---

### Let's Begin

The X algorithm isn't trying to find the "best" content.

It's trying to predict what you'll **react** to.

That single insight changes everything about how you should think about the platform—both as a viewer and as a creator.

---

## Chapter 2: TL;DR - How the Algorithm Works

### The One-Sentence Truth

The X algorithm doesn't find the "best" content. It predicts what you'll **react** to.

That's it. Every technical detail in this book flows from that single insight. The algorithm is a behavior prediction engine. It looks at your history, looks at a post, and asks: "How likely is this user to like, reply, repost, share, block, or report this?"

The posts with the highest predicted engagement win. Quality is irrelevant. Truth is irrelevant. What matters is: will YOU interact?

---

### As a Viewer: What Happens When You Open Your Feed

When you open X, four things happen in rapid succession.

#### Step 1: Query Hydration

X fetches your **behavior fingerprint**:

| Data Fetched              | Why It's Needed                                    |
| ------------------------- | -------------------------------------------------- |
| Your last 128 engagements | The core signal for predicting what you'll do next |
| Your following list       | Determines "in-network" vs "out-of-network" posts  |
| Blocked accounts          | Posts from these authors are filtered out          |
| Muted accounts            | Posts from these authors are filtered out          |
| Muted keywords            | Posts containing these words are filtered out      |

Your last 128 interactions are the most important. Not your bio, not your follower count, not how long you've been on the platform. Just what you did recently.

#### Step 2: User Embedding

Your history is converted into a **mathematical vector**—a list of numbers that represents "you" to the algorithm.

This happens through a **Transformer neural network** (the same family of AI that powers ChatGPT). It processes your 128 most recent engagements and outputs a single vector: your **user embedding**.

Think of it as coordinates in a high-dimensional space. Users with similar behavior have similar coordinates. Users with different behavior are far apart.

#### Step 3: Candidate Retrieval

X gathers posts from two sources:

| Source                       | What It Is                                                                                 | Typical Count |
| ---------------------------- | ------------------------------------------------------------------------------------------ | ------------- |
| **Thunder** (In-Network)     | Posts from people you follow. Stored in memory for sub-millisecond lookups.                | ~200 posts    |
| **Phoenix** (Out-of-Network) | Posts from strangers, discovered by AI. Uses your user embedding to find relevant content. | ~1000 posts   |

Combined, about 1,200 posts enter the ranking pipeline. Your feed will show ~30-50 of them.

#### Step 4: Phoenix Retrieval (Finding Candidates)

For out-of-network posts, X compares your **user embedding** against each post's **embedding** using a dot product (simple math).

High similarity = likely relevant to you.
Low similarity = probably not relevant.

This is how X discovers content from accounts you've never seen. Posts that are mathematically "close" to your interests enter the candidate pool.

#### Step 5: Phoenix Ranking (Scoring & Selection)

Now the algorithm scores ALL ~1,200 candidates (both in-network and out-of-network) to decide what YOU see.

**What Grok Predicts (19 Probabilities):**

For each post, the Grok transformer predicts how likely YOU are to:

| Positive Signals                     | Negative Signals    |
| ------------------------------------ | ------------------- |
| Like, Reply, Repost, Quote           | Not Interested      |
| Click, Profile Click, Video View     | Block Author        |
| Share, Dwell (linger), Follow Author | Mute Author, Report |

**How the Score is Calculated:**

```
Final Score = Σ (weight × probability)
```

Each predicted probability is multiplied by a weight (some positive, some negative), then summed. Posts with high predicted engagement and low predicted adverse signals score highest.

**Score Adjustments:**

1. **Author Diversity** - If one author has multiple posts in your feed candidates, their 2nd, 3rd, 4th posts get progressively penalized (~25% decay each)
2. **Out-of-Network Penalty** - Posts from strangers get a small penalty (~20%) compared to posts from people you follow

**Final Ranking:**

Posts are sorted by final score. Top 30-50 are selected, run through safety filters (spam, policy violations), and become your feed.

---

### As a Creator: What Happens to Your Posts

When you post, six things determine who sees it.

#### Step 6: Post Embedding

The algorithm takes your **Post ID** and your **Author ID**, runs them through hash functions, and feeds the result into a simple neural network (2-layer MLP). Output: a **post embedding**.

**Critical insight:** The algorithm doesn't read your words. It doesn't care about hashtags. It hashes your Post ID. Meaning emerges later, from who engages.

#### Step 7: Candidate Hydration

Your post is enriched with live data:

- Current like/repost/reply counts
- Author information (your follower count, verification status)
- Media entities (images, videos, links)

This real-time data feeds into the scoring model.

#### Step 8: Pre-Scoring Filters

Before scoring, posts are filtered out:

- Duplicates
- Already-seen posts
- Posts from blocked/muted authors
- Posts with muted keywords
- Old posts (typically 48+ hours)
- The viewer's own posts

About 30-50% of candidates are removed here. This saves computation—no point scoring posts that will never be shown.

#### Step 9: The Scoring Pipeline

This is where the magic happens. Four scoring stages, run sequentially:

**Stage 1: Phoenix Scorer (Grok Transformer)**

A neural network predicts 19 probabilities for each post:

| Positive Signals                          | Negative Signals          |
| ----------------------------------------- | ------------------------- |
| P(like), P(reply), P(repost), P(quote)    | P(not interested)         |
| P(click), P(profile click), P(video view) | P(block author)           |
| P(share), P(dwell), P(follow author)      | P(mute author), P(report) |

**Stage 2: Weighted Scorer**

```
Final Score = Σ (weight × probability)
```

Positive actions add to the score. Negative actions subtract. The exact weights are hidden (not open source), but the formula is public.

**Stage 3: Author Diversity Scorer**

Prevents one author from dominating your feed:

- 1st post from an author: full score
- 2nd post from same author: score × ~0.75
- 3rd post: score × ~0.56
- And so on...

**Stage 4: OON Scorer (Out-of-Network)**

Posts from people you follow get full score. Posts from strangers get a small penalty (~20% reduction). Your network is prioritized over algorithmic discovery.

#### Step 10: Selection

Posts are sorted by final score. Top 30-50 are selected for your feed.

#### Step 11: Post-Selection Filters

Final safety checks:

- Remove deleted posts
- Remove spam
- Remove policy violations (violence, hate speech)
- Deduplicate conversation threads

What remains is your feed.

**Note on Replies:** Replies use the exact same scoring pipeline, but compete in a different context—within threads, not your followers' feeds. This changes everything for new accounts and pivots.

---

### The Complete Pipeline

**The pipeline in brief:**

1. **Input**: Your history (last 128 actions) + posts from network and discovery
2. **Retrieval**: ~1,200 candidate posts gathered
3. **Ranking**: Grok predicts 19 engagement probabilities, applies weights, filters
4. **Output**: Top 30-50 posts become your feed

---

## Chapter 3: The Algorithm as a Viewer

### The Moment You Open X

You tap the X icon. Your feed loads in under a second.

In that moment, a system serving hundreds of millions of users just made thousands of decisions about you specifically. It fetched your history, computed your preferences, gathered over a thousand candidate posts, and ranked them—all before your thumb finished lifting off the screen.

This chapter explains exactly what happens.

---

### Your Behavior Fingerprint

The algorithm doesn't care about your bio. It doesn't care how long you've been on the platform. It doesn't care about your follower count.

It cares about one thing: **what you did recently**.

When you open your feed, X fetches your **behavior fingerprint**—a record of your **last 128 engagements**. This is your digital DNA to the algorithm.

#### What Gets Fetched

| Data | Purpose |
|------|---------|
| Your last 128 engagements | The core signal for predicting what you'll do next |
| Following list | Determines which posts are "in-network" vs "out-of-network" |
| Blocked accounts | Posts from these authors won't appear |
| Muted accounts | Posts from these authors won't appear |
| Muted keywords | Posts containing these words won't appear |
| Subscriptions | Affects eligibility for subscriber-only content |

#### The History That Defines You

Your engagement history is the most important data. For each recent interaction, the system records:

- **The post** you engaged with (via hash IDs)
- **The author** of that post (via hash IDs)
- **What you did**: like, reply, repost, quote, click, video view, share, dwell, follow, or negative actions (not interested, block, mute, report)
- **Where you saw it**: feed, search, profile, notification
- **When it happened**: timestamp for recency weighting

This history isn't just a list. It's a sequence—ordered by time, with your most recent actions carrying more weight.

**Why 128?** That's the configured `history_seq_len` in the Phoenix model. The Transformer uses attention to figure out which of those 128 interactions matter most for predicting what you'll engage with next.

#### Why This Matters

The algorithm treats you as the sum of your last 128 actions. Changed interests? The algorithm will catch up within 128 engagements. Engaged with something out of character? It noticed.

This is why your feed can shift rapidly. You're not locked into a "profile." You're continuously redefined by what you actually do. Your 129th-oldest engagement? Gone from the window. It no longer influences your feed.

---

### From History to Math: User Embedding

Your behavior fingerprint is human-readable—a list of posts and actions. But the algorithm can't work with lists. It needs math.

So it converts your history into a **user embedding**: a single vector of numbers that represents "you" in mathematical space.

#### How It Works

1. **Hash the entities**: Your user ID, the posts you engaged with, and their authors are all converted to numbers using hash functions. Multiple hash functions are used per entity for robustness.

2. **Look up embeddings**: Each hash indexes into a learned embedding table—a giant lookup table where every possible hash has a corresponding vector.

3. **Combine everything**: Your user embeddings, post embeddings, author embeddings, action vectors, and context are concatenated into one long sequence.

4. **Run through a Transformer**: This combined sequence feeds into a Transformer neural network—the same architecture powering ChatGPT and other LLMs. The Transformer processes the sequence and outputs a refined representation.

5. **Pool and normalize**: The outputs are averaged and normalized to create your final **user embedding**—a single vector representing your predicted preferences.

#### What the Embedding Captures

Think of your user embedding as coordinates in a high-dimensional space. Users with similar behavior end up near each other. Users with different behavior are far apart.

- If you engage heavily with tech content, your embedding moves toward the "tech region"
- If you engage with humor, you drift toward the "humor region"
- Engage with both? You're somewhere in between

This embedding is what the algorithm uses to find posts for you. Posts whose embeddings are "close" to yours are likely to interest you.

---

### Thunder: Your In-Network Posts

Your feed comes from two sources. The first is **Thunder**—X's in-memory post store for content from people you follow.

#### What Thunder Does

Thunder maintains a real-time cache of recent posts from followed accounts. When you request your feed, Thunder provides posts from your network with sub-millisecond latency.

#### How It Achieves Speed

Thunder uses several optimizations to serve posts this fast:

| Technique | Benefit |
|-----------|---------|
| **In-memory storage** | No disk I/O, no database queries |
| **Concurrent hash maps** | Lock-free reads for thousands of simultaneous requests |
| **Per-user timelines** | Posts pre-organized by author for O(1) lookups |
| **Categorized storage** | Separate stores for originals, replies/retweets, and videos |
| **Lightweight references** | Timeline stores only post ID + timestamp; full data fetched on demand |

#### The Data Flow

1. A user you follow posts something
2. The post event hits a Kafka message queue in real-time
3. Thunder's consumer threads pick it up and deserialize it
4. The post is indexed by author ID into the appropriate timeline (original, secondary, or video)
5. When you request your feed, Thunder checks which users you follow and pulls their recent posts
6. Results return in under a millisecond

#### What Gets Stored

Thunder keeps posts for approximately 2 days, then auto-trims them. It categorizes each post:

- **Original posts**: Posts that aren't replies or retweets
- **Secondary posts**: Replies and retweets
- **Video posts**: Posts with video content (indexed separately for video-specific requests)

This categorization lets Thunder serve different feed types efficiently without scanning all posts.

---

### Phoenix Retrieval: Discovering Strangers

The second source for your feed is **Phoenix Retrieval**—the AI system that discovers posts from accounts you don't follow.

This is how X shows you content from strangers. Not random strangers—strangers whose content matches your predicted interests.

#### The Two-Tower Architecture

Phoenix Retrieval uses a **two-tower model**: one tower for users, one tower for candidates.

**User Tower:**
- Input: Your user ID + engagement history
- Processing: Transformer neural network
- Output: A single normalized vector (your user embedding)

**Candidate Tower:**
- Input: Post ID + Author ID
- Processing: Two-layer neural network (MLP)
- Output: A single normalized vector per post (candidate embedding)

Both towers output vectors in the same mathematical space. This is the key insight: users and posts live in the same coordinate system.

#### Finding Matches

Once you have a user embedding and candidate embeddings, finding relevant posts is simple math:

```
similarity = dot_product(user_embedding, candidate_embedding)
```

Because both vectors are normalized (length = 1), this dot product equals cosine similarity—a measure of how "aligned" two vectors are.

- **High similarity** (close to 1): The post is predicted to match your interests
- **Low similarity** (close to 0): The post probably isn't for you
- **Negative similarity**: The post is anti-aligned with your interests

#### The Retrieval Process

1. Your user embedding is computed from your history
2. X has a pre-computed corpus of candidate embeddings (potentially millions of posts)
3. The system computes similarity scores between your embedding and every candidate
4. The top ~1,000 most similar posts are selected
5. These become your out-of-network candidates

This process uses Approximate Nearest Neighbor (ANN) search for efficiency—exact exhaustive search over millions of posts would be too slow.

#### Why It Works

The magic is that the embedding space has learned meaningful structure. Posts about similar topics end up near each other. Users who like similar content end up near each other. And critically, users end up near the posts they'd enjoy.

The two towers are trained together so that users and posts they'd engage with have high similarity, while users and posts they'd ignore have low similarity.

---

### The Combination: ~1,200 Candidates

Your feed draws from both sources:

| Source | What It Provides | Typical Count |
|--------|------------------|---------------|
| **Thunder** | Posts from people you follow | ~200 posts |
| **Phoenix Retrieval** | Posts from strangers, discovered by AI | ~1,000 posts |

These run in parallel. While Thunder looks up your following list and fetches their posts, Phoenix Retrieval is computing your embedding and searching the candidate corpus.

The results merge into a single candidate pool of roughly 1,200 posts. Your final feed will show 30-50 of them.

---

### Pre-Scoring Filters: Cleaning the Pool

Before any scoring happens, the candidate pool goes through filters. These remove posts that should never appear, regardless of predicted engagement.

#### The Filter Sequence

Filters run sequentially. Each one can remove candidates:

| Filter | What It Removes |
|--------|-----------------|
| **Duplicates** | Same post appearing twice |
| **Missing Data** | Posts without required metadata |
| **Age** | Posts older than ~48 hours |
| **Self** | Your own posts (you already know about them) |
| **Retweet Dedup** | Multiple retweets of the same original |
| **Subscription Eligibility** | Subscriber-only content you can't access |
| **Previously Seen** | Posts you've already seen (tracked via Bloom filters) |
| **Previously Served** | Posts already shown in this session (for pagination) |
| **Muted Keywords** | Posts containing words you've muted |
| **Blocked/Muted Authors** | Posts from accounts you've blocked or muted |

#### Why Filter Before Scoring?

Efficiency. Scoring is expensive—it requires running every candidate through a neural network. Filtering out ineligible posts first reduces computation. If a post will never be shown (blocked author, already seen), why waste resources scoring it?

About 30-50% of candidates are removed by these filters before scoring even begins.

---

### The Scoring Pipeline: From 700 to 30

After filtering, you have roughly 700 eligible candidates. Your feed shows 30-50. The scoring pipeline decides which ones make the cut.

#### Step 1: The Neural Network Predicts Your Behavior

The Phoenix ranking model—a Grok-based Transformer—processes each candidate and predicts the probability that YOU will take each of **19 possible actions**:

**Positive Actions (good for the post):**

| Action | What It Means |
|--------|---------------|
| favorite | You'll like this post |
| reply | You'll reply to it |
| repost | You'll retweet it |
| quote | You'll quote-tweet it |
| click | You'll click to expand |
| profile_click | You'll click the author's profile |
| follow_author | You'll follow the author |
| share | You'll share it |
| share_via_dm | You'll share it in a DM |
| share_via_copy_link | You'll copy the link |
| photo_expand | You'll expand the photo |
| vqv | You'll watch the video (quality view) |
| dwell | You'll stop scrolling and read |
| quoted_click | You'll click a quoted post |
| dwell_time | How long you'll spend on it (continuous) |

**Negative Actions (bad for the post):**

| Action | What It Means |
|--------|---------------|
| not_interested | You'll mark it "Not Interested" |
| block_author | You'll block the author |
| mute_author | You'll mute the author |
| report | You'll report the post |

For every candidate, the model outputs 19 probabilities. A post might have: P(like) = 0.15, P(reply) = 0.03, P(block) = 0.001, etc.

#### Step 2: The Weighted Score Formula

Those 19 probabilities become a single score using a weighted sum:

```
Score = Σ (weight × probability)
```

**The key insight:** Not all actions are equal.

- A like might have weight 1.0
- A reply might have weight 10.0 (replies are harder to get)
- A repost might have weight 15.0 (even more valuable)
- A block has a large **negative** weight (maybe -50.0)

So a post with:
- P(like) = 0.20, P(reply) = 0.05, P(block) = 0.01

Scores: `(1.0 × 0.20) + (10.0 × 0.05) + (-50.0 × 0.01) = 0.20 + 0.50 - 0.50 = 0.20`

The negative weights are critical. A post you might block gets penalized heavily, even if you might also like it. The algorithm is trying to avoid showing you content that triggers negative reactions.

#### Step 3: Author Diversity

After weighted scoring, the Author Diversity Scorer prevents any single author from dominating your feed.

If @elonmusk has 5 posts in your candidate pool, and they all score highly, you won't see all 5 at the top. The scorer applies exponential decay:

- First post: full score
- Second post: score × 0.8 (example)
- Third post: score × 0.64
- And so on...

This ensures variety. Your feed won't be a wall of posts from one prolific account.

#### Step 4: In-Network Priority

Finally, the OON (Out-of-Network) Scorer adjusts scores based on whether you follow the author.

Posts from people you follow get their score unchanged. Posts from strangers (discovered by Phoenix Retrieval) get multiplied by a factor less than 1.0.

This creates a bias toward your network. You'll see content from people you chose to follow before content the AI discovered for you.

#### Step 5: Final Selection

All candidates are now sorted by their final adjusted score. The top ~30-50 become your feed.

The entire scoring pipeline—from 700 candidates to 30 posts—runs in milliseconds. Every score is personalized to YOU based on YOUR embedding and YOUR predicted behavior.

---

### Why Your Feed Is Unique

No two users see the same feed. Even if two people followed exactly the same accounts, their feeds would differ. Here's why:

#### 1. Different History = Different Embedding

Your user embedding is computed from YOUR engagement history. Your actions over the past days shape where you sit in embedding space. Even slight differences in behavior create different embeddings, leading to different Phoenix Retrieval results.

#### 2. Different Filters

You've blocked different accounts. You've muted different keywords. You've already seen different posts. The filter stage removes different candidates for each user.

#### 3. Different Timing

Thunder serves recent posts from your network. Since you follow different people who post at different times, the available candidate pool varies by the millisecond.

#### 4. The Scoring Is Personalized

Even when two users see the same post as a candidate, the scoring model predicts different engagement probabilities. Your predicted P(like) for a post depends on YOUR embedding, not some universal score.

#### The Butterfly Effect

Small differences compound. A slightly different embedding pulls slightly different Phoenix candidates. Different candidates mean different competition during ranking. Different ranking means different final selection.

Your feed is a function of: `(your history, your network, your filters, this exact moment)`

Change any input, change the output.

---

### What This Means for You

As a viewer, you have more control over your feed than you might think.

#### Your Actions Are Instructions

Every like, reply, repost, or dwell time is training data. The algorithm watches what you do and adjusts. You're not passively receiving content—you're actively teaching the system what to show you.

**Engage with what you want more of.** The algorithm will comply.

#### Negative Signals Matter

"Not Interested," Block, and Mute aren't just hiding posts—they're shaping your embedding. These negative signals move you away from that content in embedding space.

**Use them.** They work.

#### The "Reset" Myth

People ask: "Can I reset my algorithm?"

Not really. But you don't need to. Your feed is based on your last 128 engagements, not a permanent profile. Just start engaging differently. After 128 new interactions, your old behavior is completely out of the window.

The algorithm has a short memory—exactly 128 actions long. Use that to your advantage.

#### Breaking the Bubble

If your feed feels too narrow, you're probably engaging too narrowly. The algorithm is reflecting your behavior back at you.

To diversify:
- Seek out and engage with different content
- Follow accounts outside your usual topics
- Use "Not Interested" on repetitive content

The system will adjust. It always does.

---

## Chapter 4: The Algorithm as a Creator

### The Moment You Post

You tap "Post." Your carefully crafted thought launches into the void.

What happens next determines whether thousands see it or almost nobody does. In milliseconds, your post enters a competition—a battle against other posts for a spot in each viewer's feed.

This chapter flips the perspective. Chapter 3 explained what happens when you **view** your feed. Now we'll see what happens to **your** posts when they compete for someone else's feed.

---

### Your Post Becomes Math

In Chapter 3, you learned how your engagement history becomes a **user embedding**—a mathematical vector representing your preferences. Your posts go through a similar transformation, but simpler.

#### The Process

When you post, the system creates a **candidate embedding** for your content:

```
post_embedding = project(
    concat(post_hash_embedding, author_hash_embedding, context)
)
```

That's it. Three components:

1. **Post hash embedding**: Your post's unique ID, hashed and looked up in an embedding table
2. **Author hash embedding**: Your user ID, hashed and looked up
3. **Context**: Where the post might appear (feed, search, notifications)

#### What This Means for You

**The algorithm doesn't read your words.** Not directly.

It doesn't parse your sentences. It doesn't understand your jokes. It doesn't evaluate your arguments. It starts with two IDs: your post and you.

So how does content matter? Through **learned associations**. The neural network has processed billions of posts and their outcomes. It has learned patterns: posts from authors with certain characteristics, appearing in certain contexts, tend to get certain reactions.

Your **Author ID embedding** carries your track record. If your previous posts got engagement, your embedding reflects that. If your posts got blocked, that's in there too.

This is why new accounts struggle. The algorithm has no history. Your Author ID embedding is essentially random noise until you build a track record.

---

### The Same 19 Actions, Different Perspective

Chapter 3 introduced the 19 actions the algorithm predicts for every post-viewer pair. As a viewer, those predictions determine what you see. As a creator, those same predictions determine who sees you.

The difference is perspective:

| As a Viewer                 | As a Creator            |
| --------------------------- | ----------------------- |
| "Will I like this?"         | "Will they like mine?"  |
| "Will I block this author?" | "Will they block me?"   |
| "Will I share this?"        | "Will they share mine?" |

Every person who might see your post gets their own set of 19 probability predictions. For User A, P(like) might be 0.25. For User B, it might be 0.02. Same post, different viewers, different predictions.

**Your reach is the sum of all these individual predictions.**

A post that one person loves might be one another person blocks. The algorithm sees both. The weighted score accounts for both.

---

### The Four Stages, From Your Post's View

You already know the scoring pipeline from Chapter 3. Here's what each stage means for your content:

#### Stage 1: Phoenix Scorer

The transformer predicts 19 probabilities. From your post's perspective, this is the moment of truth—how engaging does the model think your content will be for this specific viewer?

What influences these predictions:

- **Your track record** (Author ID embedding)
- **The viewer's history** (what they've engaged with before)
- **Pattern matching** (does your post look like content this viewer engages with?)

#### Stage 2: Weighted Scorer

The 19 probabilities become one score:

```
weighted_score = Σ (weight × probability)
```

For creators, the key insight: **negative actions hurt more than positive actions help**.

If 5% of viewers might block you, that negative weight can overwhelm 20% who might like you. Controversial content pays this tax.

#### Stage 3: Author Diversity

If you have multiple posts in the candidate pool, they compete against each other:

```
multiplier = (1 - floor) × decay^position + floor
```

- Your 1st post: full score
- Your 2nd post: ~80% of score
- Your 3rd post: ~65% of score

**Posting more doesn't mean reaching more.** Each additional post competes against your other posts with a handicap.

#### Stage 4: OON Scorer

For viewers who don't follow you:

```
final_score = score × OON_FACTOR  (where OON_FACTOR < 1.0)
```

Your followers see your content at full strength. Strangers see it discounted.

---

### What Helps Your Reach

Based on how the pipeline works, here's what actually moves the needle:

#### 1. Build Predicted Engagement

The algorithm predicts based on patterns. If your content consistently gets likes, replies, and shares, the model learns that your Author ID produces engaging content. Future predictions improve.

**Consistency compounds.** Each engaging post trains the model to predict higher engagement on your next post.

#### 2. Avoid Predicted Adverse Actions

Content that triggers blocks, mutes, and reports carries that signal forward. The model learns "this author's content gets blocked" and applies that pattern to future predictions.

**Even if you don't get blocked on a specific post**, if your author pattern suggests blocks, you're penalized.

#### 3. Post Strategically, Not Frequently

Author diversity decay punishes volume. One great post outperforms five mediocre ones.

**Quality > quantity** isn't just advice—it's how the scoring math works.

#### 4. Build Your Following

The OON penalty makes followers more valuable than algorithmic discovery. A follower sees your content at full score. A stranger sees it discounted.

**Your network is your distribution.** Algorithmic reach is bonus, not baseline.

#### 5. Video Content (Conditional)

Videos longer than a minimum duration get the VQV (Video Quality View) weight applied. Short clips don't qualify.

If you make video content, make it long enough to count.

---

### What Kills Your Reach

#### 1. Triggering Adverse Actions

Block probability has negative weight. A 3% block rate can tank a post even with 20% like rate.

**Controversial ≠ engaging** in the algorithm's eyes. It sees both the likes and the blocks.

#### 2. Posting Too Much

Every additional post in the scoring window gets exponentially penalized. The 10th post in a day would need to be dramatically better than the 1st just to score equally.

**The algorithm explicitly prevents feed domination.**

#### 3. Old Posts

Posts older than ~48 hours are filtered before scoring even begins. They don't get a chance to compete.

**Timing matters.** Post when your audience is active.

#### 4. Being Blocked or Muted

If a viewer has blocked or muted you, your posts are removed entirely for that viewer—not scored, just gone.

**Every block is permanent reach loss** for that viewer.

#### 5. Muted Keywords

If your post contains keywords a viewer has muted, it's filtered out before scoring.

**You can't reach people who've muted your topics.**

#### 6. Not Being Followed

For every non-follower, your score is multiplied by a factor less than 1. You're competing with a handicap against their in-network content.

**Followers are distribution. Non-followers are uphill battles.**

---

### What This Means for Creators

#### 1. Quality Over Quantity

Author diversity decay makes posting frequency a losing strategy. One great post beats five mediocre ones.

#### 2. Avoid Adverse Actions

A small percentage of blocks can outweigh a larger percentage of likes. Content that triggers blocks may underperform "safer" content with the same like rate.

#### 3. Build Your Network

The OON penalty means your followers are worth more than algorithmic discovery. A follower sees your content at full score. A stranger sees it discounted.

#### 4. Timing Matters

The age filter has a hard cutoff. Post when your audience is active, not when they're asleep. A 24-hour-old post competing against fresh content is already at a disadvantage.

#### 5. Your Track Record Matters

The neural network learns from your history. Accounts with consistent engagement build better Author ID embeddings. New accounts start cold—the algorithm has no signal.

---

## Chapter 5: Author Embeddings

### Your Algorithmic Identity

When the algorithm evaluates your post for someone's feed, it doesn't just look at the post itself. It looks at **you**—specifically, a mathematical representation of your track record as a creator.

This is your author embedding. It's your algorithmic identity.

---

### What Is an Author Embedding?

Every account on X has an author embedding—a vector of numbers that represents that account's history as a content creator.

The lookup is simple:

```
author_embedding = lookup(hash(author_id))
```

Your user ID gets hashed (multiple times, for robustness), and those hashes index into a massive embedding table. The result is a vector—hundreds of numbers that encode patterns learned from your posting history.

---

### Where Author Embeddings Are Used

Your author embedding appears in two critical places in the recommendation pipeline:

#### 1. Retrieval (Finding Candidates)

When Phoenix Retrieval searches for posts to show a viewer, it uses the **Candidate Tower**—a neural network that takes your post embedding AND your author embedding as input.

```
candidate_vector = CandidateTower(post_embedding, author_embedding)
```

This candidate vector gets compared against the viewer's user vector. High similarity means your post is more likely to be retrieved for that viewer.

**Your author embedding affects whether you're even considered.**

#### 2. Ranking (Scoring Candidates)

Once retrieved, your post goes through the Phoenix Scorer. Here, your author embedding is concatenated with your post embedding and projected through learned matrices:

```
combined = project(concat(post_embedding, author_embedding))
```

The transformer then uses this combined representation to predict 19 engagement probabilities for each viewer.

**Your author embedding affects how high you score.**

---

### How Author Embeddings Are Trained

The embedding tables don't start with meaningful values. They're trained through a process called **contrastive learning**.

#### The Two-Tower Architecture

X's retrieval system uses two neural networks—called "towers"—that work together:

**User Tower:**

- Input: Viewer's engagement history
- Output: A vector representing the viewer's preferences

**Candidate Tower:**

- Input: Post + Author embeddings
- Output: A vector representing the candidate content

Both towers output vectors in the **same mathematical space**. This is the key insight: users and posts exist in the same coordinate system.

```
similarity = dot_product(user_vector, candidate_vector)
```

High similarity means the post is predicted to match the user's interests.

#### The Training Objective

During training, the system learns from actual engagement data:

**Positive pairs:** User + Post combinations where engagement happened
**Negative pairs:** User + Post combinations where no engagement happened

The training objective:

- **Pull positive pairs closer** in embedding space
- **Push negative pairs apart** in embedding space

This is contrastive learning in action. The model learns by contrasting what users engaged with against what they ignored.

#### In-Batch Negatives

Here's an efficiency trick: instead of sampling random negatives, the system uses **other candidates in the same batch** as negative examples.

If a batch contains 1,000 user-post pairs, each positive pair has 999 implicit negatives—all the other posts that user didn't engage with.

This creates rich learning signal from real engagement data.

#### Why This Creates Meaningful Space

Through millions of training examples, the embedding space develops structure:

- Authors who create similar content end up near each other
- Users who like similar content end up near each other
- Authors end up near the users who would engage with them

Your author embedding isn't just a random vector—it's a **learned position** in a space where proximity means predicted engagement.

---

### What Your Embedding Depends On

Now that you understand how embeddings are trained, the dependencies become clear.

Your author embedding is shaped by **who engages with your content**.

Not just _whether_ people engage—but _which specific people_.

#### You Become Who Engages With You

When a user engages with your post, your author embedding moves closer to **that user's embedding** in the shared space.

If tech enthusiasts consistently like your posts, your embedding drifts toward the "tech enthusiast" region. If fitness accounts engage with you, you move toward that region. If crypto traders block you, you move away from them.

**Your audience literally shapes your algorithmic identity.**

This is why engagement from the "right" people matters more than raw numbers. 100 likes from users in your target niche move your embedding in a useful direction. 100 likes from random accounts scatter your signal.

#### Positive Signals

When someone **likes**, **replies**, **reposts**, or **shares** your content:

- Your author embedding moves closer to that user's embedding
- The more engagement from similar users, the stronger the pull in that direction
- The model learns: "This author's content resonates with these kinds of users"

#### Negative Signals

When someone **blocks**, **mutes**, or **marks "not interested"** on your content:

- Your author embedding moves away from that user's embedding
- Negative signals are weighted heavily—a few blocks can offset many likes
- The model learns: "This author's content should be avoided for these kinds of users"

#### The Projection Matrices

The projection matrices that combine your author and post embeddings also update during training. They learn how to **interpret** the combination of who you are (author embedding) and what you posted (post embedding).

Over time, the system learns patterns like:

- "When this author posts about tech, engagement is high"
- "When this author posts threads, replies increase"
- "When this author is argumentative, blocks increase"

---

### The Cold Start Problem

Here's the challenge for new accounts: **the algorithm doesn't know you yet**.

When you create an account and start posting, your author embedding is essentially random. The hash of your user ID points to some location in the embedding table, but that location has no meaningful signal about your content.

The algorithm can't predict how people will engage with your posts because it has no data.

#### What This Means

- **Your first posts compete with a handicap.** The algorithm can't confidently recommend you to anyone.
- **Your early engagement matters disproportionately.** The first signals you generate start shaping your embedding.
- **Consistency in your early posts helps.** If you post about wildly different topics, the model receives mixed signals.

Think of it as a "proving ground" period. You're not just building an audience—you're building your algorithmic identity.

---

### Why This Matters for Creators

#### Your Embedding Is Your Reputation

Think of your author embedding as your algorithmic reputation. It encodes:

- What topics you typically post about
- What engagement patterns your content generates
- Which types of users engage positively vs negatively

Every post either reinforces or shifts this reputation.

#### Consistency Compounds

The algorithm makes predictions based on patterns. If you consistently post about one topic and consistently get engagement, the model learns to confidently predict engagement on similar future posts.

Inconsistency dilutes this signal. If you post about tech one day, recipes the next, and politics the day after, the model has weaker patterns to learn from.

**This doesn't mean you can't be varied.** It means the algorithm will have higher confidence predicting engagement for accounts with clearer patterns.

#### Topic Pivots Reset Your Signal

If you pivot to a completely new topic, your author embedding becomes partially misaligned. The embedding was trained on your old content; your new content generates different signals.

This explains why established accounts sometimes see reach drops when they change topics. The algorithm's predictions are temporarily less accurate.

The good news: with consistent posting in the new direction, your embedding will update. It just takes time to retrain the patterns.

---

### The Discovery Loop

Author embeddings directly affect your out-of-network discovery—whether strangers see your content.

#### How Discovery Works

Recall from Chapter 3: Phoenix Retrieval uses the two-tower architecture. Your post embedding and author embedding go through the Candidate Tower (an MLP), producing a candidate vector. That candidate vector is compared to the user's vector via similarity search.

Your author embedding influences the candidate vector that comes out. A stronger author embedding—one trained on engagement from users similar to the viewer—produces a candidate vector more likely to match.

#### The Virtuous Cycle

Good engagement creates a positive feedback loop:

1. You post content that resonates
2. Engagement improves your author embedding
3. Better embedding means better retrieval matches
4. Better matches mean more discovery
5. More discovery means more potential engagement
6. Return to step 2

This is why "going viral" can have lasting effects. The engagement spike improves your embedding, which improves future discovery, which increases baseline engagement.

#### The Vicious Cycle

Poor engagement works in reverse:

1. You post content that triggers blocks or mutes
2. Negative signal degrades your author embedding
3. Worse embedding means worse retrieval matches
4. Worse matches mean less discovery
5. Less discovery means lower engagement
6. Return to step 2

This is why reputation damage can be hard to recover from. Negative signals don't just hurt that post—they hurt your embedding, which affects future posts.

---

## Chapter 6: Replies

### The Same Algorithm, Different Game

You've probably heard the advice: "Be a reply guy." Comment on big accounts. Get in front of their audience. Build your following through replies.

The advice is right, but the explanation is usually wrong. Most people think replies work because of "exposure" or "networking" or some vague notion of "getting seen." They're missing the algorithmic reason—and it connects directly to everything you've learned in the previous chapters.

Replies use the exact same algorithm as posts. Same scoring pipeline. Same 19 engagement predictions. Same weighted formula. Same author embedding contribution. There's no special "reply boost" or separate ranking system.

So why do replies behave differently? Why can a new account get traction through replies when their posts go nowhere?

Let's trace the flow:

1. An established author posts something
2. The algorithm shows that post to viewers relevant to that author—people who follow them, people whose embeddings match
3. A viewer sees the post and clicks to read the comments
4. Now the replies are ranked for that viewer
5. Your reply competes against other replies in that thread—not against millions of posts in the main feed

**The audience is different. And the competition is smaller.**

When you post, you're competing against ~1,200 candidates for a spot in someone's feed. When you reply, you're competing against maybe 20-50 other replies in that thread. The scoring is identical, but the battlefield is completely different.

---

### The Audience Difference

When you post, your content competes in your **followers' feeds**. The algorithm uses your author embedding to predict engagement, then shows your post to people it thinks will engage based on your track record.

When you reply, your content appears in the **thread** you replied to. The audience isn't your followers—it's:

- The original author (who gets notified)
- People who follow the original author
- People who engaged with the thread
- People the algorithm shows the thread to

**You're borrowing someone else's audience.**

| Content Type | Audience | Who Sees It |
|--------------|----------|-------------|
| Your post | Your followers | People similar to your existing audience |
| Your reply | Thread viewers | The post author's audience |

This single difference changes everything for new accounts and pivots.

---

### Why This Matters: The Cold Start and Pivot Problem

Chapter 5 explained the cold start problem: new accounts have untrained author embeddings. The algorithm doesn't know you, so it can't confidently predict engagement.

Here's the problem for new accounts:

1. You post something
2. Your author embedding is essentially random
3. The algorithm has no signal for who should see it
4. Few people see it → low engagement
5. Low engagement → embedding trains slowly
6. Return to step 1

You're stuck. You need engagement to train your embedding, but you need a trained embedding to get discovery.

**Pivoting accounts have the same problem in reverse.** Your embedding is trained—but on your OLD audience. When you post about your new topic:

1. The algorithm matches you to your old audience
2. They ignore it (wrong topic for them)
3. Low engagement from the wrong people
4. Embedding receives scattered or negative signal

---

### How Replies Solve This

Replies bypass the embedding problem entirely.

When you reply to a relevant post in your target niche:

1. Your reply appears in that thread
2. The audience is **the post author's followers**—your target audience
3. They see your reply regardless of your embedding
4. If they engage, your embedding starts training toward them
5. You're building signal in the right direction

**You're accessing the audience you want, not the audience you have.**

```
POSTING (when new or pivoting):
Your post → Algorithm uses your weak/misaligned embedding
         → Reaches few people or wrong people
         → Low/scattered engagement
         → Embedding stays stuck

REPLYING (when new or pivoting):
Your reply → Appears in target niche thread
          → Reaches that post's audience (your target)
          → They engage (if good reply)
          → Embedding trains toward target niche
```

This is why being a "reply guy" works. You're not gaming the algorithm—you're using replies to reach the audience that your posts can't reach yet.

---

### The Scoring Is Identical

To be clear: the algorithm doesn't score replies easier or harder than posts.

Same Phoenix predictions (19 actions). Same weighted formula. Same author embedding influence.

What changes is **who's evaluating your content**. When you reply to a founder's post, your reply is scored for the founder's audience. Your author embedding still matters—but you've placed yourself in front of different eyes.

---

### The Volume Difference

One structural advantage of replies: they don't compete with each other across different threads.

When you post 5 times, all 5 posts compete in your followers' feeds. The Author Diversity Scorer applies decay—each additional post scores lower.

When you reply to 5 different threads, each reply exists in its own ranking context. No cross-thread penalty.

This means reply volume doesn't hurt you the way post volume does. But the real advantage isn't volume—it's audience access.

---

### Quality Still Matters

Each reply is scored individually. If your replies are generic or low-effort:

- People ignore them
- Low engagement
- Model learns your replies don't generate engagement
- Your embedding doesn't train in a useful direction

The strategy only works if your replies are good enough to get engagement from your target audience.

One thoughtful reply that gets engagement is worth more than ten generic comments that get ignored.

---

### The Strategic Playbook

#### For New Accounts

Your embedding is cold. Your follower count is low. Posting is a slow path.

Instead:
- Find accounts in your target niche with engaged audiences
- Reply with genuine value (insights, questions, relevant experience)
- Get engagement from your target audience
- Build your embedding in the right direction from day one

#### For Pivoting Accounts

Your embedding is trained on your old audience. Posting reaches the wrong people.

Instead:
- Find accounts in your NEW niche
- Reply to their content
- Access their audience (your new target)
- Retrain your embedding toward the new direction
- Your posts will start reaching the right people as your embedding shifts

---

### Summary

| Aspect | Posts | Replies |
|--------|-------|---------|
| Algorithm | Same | Same |
| Audience | Your followers | Thread viewers |
| When new/pivoting | Reaches few or wrong people | Reaches target audience |
| Embedding training | Slow (no signal) | Faster (right signal) |
| Best use case | Maintaining audience | Building/shifting audience |

The Reply Game isn't about volume. It's about audience access.

When your embedding can't get your posts in front of the right people, replies let you show up anyway. Use that access to build the signal you need.

---

## Chapter 7: What You Can Do

### From Understanding to Action

You now understand how the X algorithm works. You know about user embeddings, author embeddings, the two-tower architecture, the 19 engagement predictions, the scoring pipeline, and how replies differ from posts.

But understanding the system isn't the same as using it effectively.

This chapter translates the technical knowledge from previous chapters into practical strategies. Every recommendation here connects back to a specific mechanism in the code.

---

### Part 1: If Your Feed Isn't What You Want

Your For You feed feels wrong. You're seeing content you don't care about, topics you never asked for, accounts you've never heard of. What can you do?

#### Why Your Feed Looks This Way

Your **user embedding** is computed from your last 128 engagements. These recent interactions define who the algorithm thinks you are.

```
YOUR USER EMBEDDING = f(last 128 engagements)

Engagements include:
- Likes
- Replies
- Reposts
- Profile clicks
- Dwell time
- "Not Interested" clicks
- Blocks/Mutes
```

Phoenix Retrieval uses this embedding to find posts similar to what you've engaged with. Grok Ranking scores those candidates based on predicted engagement. Both depend entirely on your recent behavior.

#### The Good News: Changes Are Fast

Unlike author embeddings (which shift slowly over months), your feed can change in **days**—because it only depends on 128 recent actions.

You can meaningfully change your feed within 2-4 days of active use.

#### Strategy 1: Intentional Engagement

Your feed is a reflection of your behavior. To change what you see, change what you do.

| Goal                      | Action                                                                   |
| ------------------------- | ------------------------------------------------------------------------ |
| See more of Topic X       | Find and engage with 10-20 accounts posting about Topic X                |
| See less of Topic Y       | Use "Not Interested" on Topic Y posts; mute keywords                     |
| Discover new perspectives | Actively search for and engage with accounts outside your current bubble |

**Practical execution:**

1. Identify 10-15 accounts creating content you want to see
2. Like their posts
3. Reply to their posts (replies count as engagement)
4. Do this consistently for 3-5 days
5. Your embedding shifts toward this cluster

#### Strategy 2: Active Negative Signals

The algorithm treats negative signals seriously. Use them.

| Action           | Effect                                                 |
| ---------------- | ------------------------------------------------------ |
| "Not Interested" | Moves your embedding away from that content type       |
| Mute Keywords    | Filters posts containing those words before ranking    |
| Mute Account     | Removes that account's posts from your feed            |
| Block Account    | Removes account and signals strong negative preference |

> **The Negative Signal Advantage**
>
> Negative signals are explicit and unambiguous.

| Signal Type | Strength | Meaning |
|-------------|----------|---------|
| Not engaging | Weak | Maybe you missed it |
| "Not Interested" | Strong | You don't want this |
| Blocking | Strongest | Never show me this |

The algorithm responds to explicit negatives faster than implicit ones.

**Practical execution:**

- Aggressively use "Not Interested" on content you don't want
- Mute keywords for topics you want to avoid entirely
- Block accounts that consistently appear despite disinterest

#### Strategy 3: Train Deliberately, Not Reactively

Most people engage reactively—they like what shows up. This creates a feedback loop where your feed trains itself.

**Practical execution:**

1. Use Search to find content in topics you want
2. Engage with that content
3. Follow accounts that consistently produce what you want
4. Unfollow accounts that don't serve your goals
5. Treat your feed as something you shape, not something that happens to you

#### How Long Until You See Changes?

| Actions Taken                 | Timeline                              |
| ----------------------------- | ------------------------------------- |
| 20-30 intentional engagements | Feed starts shifting noticeably       |
| 50-80 intentional engagements | Significant change in recommendations |
| 128+ intentional engagements  | Old patterns fully replaced           |

At a moderate usage rate (50 engagements/day), you can transform your feed in 2-3 days.

---

### Part 2: If You Want to Increase Your Following

Growing your account means getting the algorithm to show your content to more people who will follow you. This depends on your **author embedding**—how the algorithm understands you as a creator.

Your strategy depends on your starting position.

---

#### For New Accounts

You're starting from zero. No followers, no engagement history, no author embedding signal.

##### The Cold Start Problem

Your author embedding is essentially random until you build a track record. The algorithm can't confidently predict who will engage with your content because it has no data about you.

**The New Account Dilemma:** You need engagement to train your embedding, but you need a good embedding to get discovery, and you need discovery to get engagement. It's a chicken-and-egg problem.

Here's how to break the loop.

##### Strategy 1: Reply Game First

This is your most powerful lever. Replies work differently from posts—not because of different scoring, but because of different audience access.

**The core insight:** When you reply to an established account's post, your content appears in front of their audience, not yours (because you don't have one yet).

**Volume advantage:** The Author Diversity Scorer penalizes multiple posts from the same author in someone's feed. But replies to different threads don't compete with each other—each thread is a separate ranking context.

| Activity                       | Author Diversity Penalty                    |
| ------------------------------ | ------------------------------------------- |
| 5 posts to timeline            | Each successive post penalized (~25% decay) |
| 5 replies to different threads | No penalty—each scored independently        |

**Practical execution:**

1. Find accounts in your niche with engaged audiences (10K-100K followers)
2. Wait for posts where you can add genuine value
3. Write thoughtful replies—not "Great post!" but actual insights
4. Aim for 10-20 quality replies per day across different threads
5. One reply per post (multiple replies to same post does trigger the penalty)

##### Strategy 2: Niche Domination

The embedding space has structure. Users and posts about similar topics cluster together. If you spread your content across many topics, your embedding becomes diluted—a weak signal in many areas.

If you focus intensely on one topic, your embedding becomes concentrated—a strong signal in that specific region.

| Approach | Embedding Result | Discovery Effect |
|----------|------------------|------------------|
| **Diluted** (multiple topics) | Weak signal in many areas | Matches many users weakly |
| **Concentrated** (one topic) | Strong signal in one area | Matches target users strongly, Phoenix retrieves you for users interested in your niche |

**Practical execution:**

1. Pick one topic you can consistently create value around
2. Post exclusively about that topic for your first 3-6 months
3. Reply to established accounts in that same topic
4. Let your embedding become the reference point for that niche

##### Strategy 3: Quality Over Quantity

The Author Diversity Scorer makes posting frequency a losing strategy for new accounts. You don't have the audience to absorb multiple posts per day.

**New Account Posting 5x/Day:**

| Post | Score | Effect |
|------|-------|--------|
| Post 1 | 100% | Shown to ~100 people (your small network) |
| Post 2 | ~75% | Competes with Post 1, penalized |
| Post 3 | ~56% | Penalized further |
| Post 4 | ~42% | Heavily penalized |
| Post 5 | ~32% | Barely shown |

**Result:** Effective reach is ~3x one post, not 5x. Each post cannibalizes the others.

**Better approach:**

- 1-2 high-quality posts per day maximum
- Put your energy into replies (no volume penalty)
- Every post should be your best effort

##### Timeline Expectations

Embedding training happens through engagement data processed in batches. Expect:

| Timeframe | What Happens                                        |
| --------- | --------------------------------------------------- |
| Week 1-4  | Building Reply Game momentum, minimal organic reach |
| Month 2-3 | Embedding starts reflecting your niche engagement   |
| Month 4-6 | Noticeable improvement in out-of-network discovery  |
| Month 6+  | Stable position, compounding growth begins          |

There are no shortcuts. The algorithm needs real engagement from real users to learn about you.

---

#### For Established Accounts

You have followers. You have an engagement history. Your author embedding is trained and positioned in the embedding space. You're past the cold start—now it's about optimization.

##### Your Compounding Advantage

If you've built consistent engagement in your niche, your embedding is strong:

**The Established Account Advantage:**

A strong author embedding helps you in three ways:

1. **Phoenix Retrieval:** High similarity to target users means your posts are retrieved more often
2. **Grok Ranking:** Historical patterns suggest engagement, leading to higher predicted probabilities
3. **In-Network Distribution:** Large follower base gives you immediate reach without needing discovery

Your job is to not break what's working while optimizing for more.

##### Optimization 1: Respect the Diversity Decay

The Author Diversity Scorer applies exponential decay to multiple posts from the same author:

**Decay Formula:** `multiplier = (1 - floor) × decay^position + floor`

| Post | Score Multiplier | Reduction |
|------|------------------|-----------|
| Post 1 | 1.0 | Full score |
| Post 2 | 0.76 | 24% reduction |
| Post 3 | 0.59 | 41% reduction |
| Post 4 | 0.47 | 53% reduction |
| Post 5 | 0.39 | 61% reduction |

**Practical implication:** 2-3 posts per day is optimal. More posts don't mean more reach—they mean your posts compete against each other with handicaps.

##### Optimization 2: Avoid Adverse Actions

The scoring formula: positive actions add to your score, negative actions subtract. But the subtraction is weighted heavily.

**The Asymmetry:** A 20% like rate with weight +1.0 contributes +0.20 to your score. But a 3% block rate with weight -50.0 contributes -1.50. That 3% block rate costs you more than the 20% like rate gained.

Content that polarizes—even if it gets high engagement from supporters—pays a tax from the people it alienates. The algorithm sees both sides.

**Practical implication:** Controversy isn't free. Consider whether the engagement is worth the adverse signal.

##### Optimization 3: Leverage Your Network

The OON Scorer (Out-of-Network) gives your followers' feed positions at full score. Strangers see your content with a penalty.

| Audience | Score |
|----------|-------|
| **In-Network** (followers) | Your post gets full score |
| **Out-of-Network** (discovery) | Your post gets score × OON_FACTOR (where OON_FACTOR < 1.0) |

**Practical implication:** Your followers are your distribution. Algorithmic discovery is bonus, not baseline. Focus on content that serves your existing audience first.

##### Optimization 4: Timing Matters

Posts older than ~48 hours are filtered out before scoring even begins. They don't compete—they're just gone.

**Practical implication:** Post when your audience is active. A great post at 3am (for your audience's timezone) has 48 hours to catch up with a mediocre post at peak hours.

##### The Virtuous Cycle

Established accounts benefit from compounding:

**The Virtuous Cycle:** Good engagement → Embedding reinforced → Better retrieval matches → More discovery → More engagement → (cycle continues)

Your job is to keep feeding the cycle, not disrupt it.

---

#### For Pivoting Accounts

You built an audience around Topic A. Now you want to create about Topic B. Your reach has dropped. What's happening?

##### The Embedding Mismatch Problem

Your author embedding is trained on your old audience. Contrastive learning pushed your embedding toward the users who engaged with your previous content.

**The Pivot Problem:** Your embedding sits in the "Old Topic Cluster" with your old audience, but your new content is meant for the "New Topic Cluster" which is far apart in embedding space. Your embedding is HERE but your content is for THERE.

When you post about your new topic:

1. Phoenix Retrieval uses your old embedding → retrieves wrong users
2. Grok shows to your old followers (in-network) → they don't care
3. Low engagement → post ranked lower → never reaches right audience
4. Your embedding doesn't move → next post has same problem

##### The Valley of Despair

Every pivot has a dip. Your old audience engagement drops before your new audience engagement builds. You must survive this valley.

**Reach During Pivot:** Your reach starts at your old baseline, then dips significantly as your old audience disengages and your new audience hasn't found you yet. If you complete the pivot, your reach eventually climbs to a new baseline. The key: you MUST survive the dip in the middle.

**Timeline:** Expect 3-6 months of reduced reach during the transition.

##### The Half-Pivot Trap

This is critical: **giving up mid-pivot is worse than not pivoting at all.**

**If You Give Up and Go Back:**

1. Your old audience learned to ignore you (push signal recorded)
2. Your embedding moved away from old cluster
3. You never established in new cluster
4. Going back to old content → algorithm predicts low engagement (because your old audience is now trained to scroll past you)

**Result:** Lower reach than before you started pivoting.

Contrastive learning has no undo button. The push signals from your old audience ignoring your new content are baked into your embedding. You can't just "go back."

| Decision       | Outcome                                            |
| -------------- | -------------------------------------------------- |
| Don't pivot    | Keep current baseline (safe but stuck)             |
| Complete pivot | Survive dip → new audience → potential growth      |
| Half pivot     | Lose old position + never gain new = worst outcome |

**If you're going to pivot, commit.**

##### Pivot Strategy 1: The Reply Game

Same mechanism as new accounts, but even more important for pivots. Your posts go to the wrong audience. Replies bypass that entirely.

**Practical execution:**

- Find accounts in your new niche with engaged audiences
- Reply with genuine expertise from your perspective
- This is how you access the audience your posts can't reach

##### Pivot Strategy 2: Bridge Content

Create content that appeals to both your old and new audiences.

**Bridge Content Examples:**

| Old Niche | New Niche | Bridge Content |
|-----------|-----------|----------------|
| Math/Education | Startups | "The mathematical model behind viral growth" |
| Math/Education | Startups | "What probability theory taught me about market sizing" |
| Math/Education | Startups | "Linear algebra applications in product analytics" |

This maintains some engagement from your old audience (preventing pure push signals) while attracting your new target audience.

##### Pivot Strategy 3: The 70/30 Mix

Don't go 100% new topic immediately. Gradual transition:

- **Month 1:** 70% old / 30% new
- **Month 2:** 50% / 50%
- **Month 3:** 30% old / 70% new
- **Month 4+:** 10% old / 90% new

This maintains engagement floor while shifting your embedding direction.

##### The Two Forces Working For You

Contrastive learning creates two forces during a pivot:

| Signal                            | Effect                     |
| --------------------------------- | -------------------------- |
| Old audience ignoring new content | PUSH away from old cluster |
| New audience engaging via replies | PULL toward new cluster    |

When both forces point the same direction, your embedding moves faster. The Reply Game ensures you get both signals working together.

---

### Universal Strategies

These apply regardless of account size or situation.

#### Post Strategically, Not Frequently

The math is clear: Author Diversity Scorer penalizes volume. Quality beats quantity at every account size.

| Posts Per Day | Effective Reach Multiplier |
| ------------- | -------------------------- |
| 1             | 1.0x                       |
| 2             | ~1.7x                      |
| 3             | ~2.3x                      |
| 5             | ~3.2x                      |
| 10            | ~4.5x                      |

Posting 10x gets you ~4.5x reach, not 10x. The diminishing returns are severe.

**Optimal strategy:** 2-3 high-quality posts per day. Put additional energy into replies (no volume penalty).

#### No Penalty for Inconsistency

The algorithm doesn't track:

- How often you post
- When you last posted
- Your "consistency score"

There is no shadow ban for taking a week off. Your embedding persists. Your followers persist. When you return, your posts are scored the same as before.

**Practical implication:** Take breaks when you need them. The algorithm won't punish you.

#### Why Big Accounts Get More Reach (Even at Same Engagement Rate)

Grok doesn't actually see engagement counts like "5,000 likes." The model sees hash-based embeddings and predicts engagement probabilities. So where does the big account advantage come from?

**The compounding loop:**

For **Big Accounts**:
1. 100K followers → Post shown to 100K people in-network, full score
2. 5% engage = 5,000 engagements from diverse users
3. Author embedding trained on many engagement signals
4. Strong, well-defined embedding position
5. Phoenix retrieves for more OON users
6. Cycle compounds

For **Small Accounts**:
1. 10K followers → Post shown to 10K people in-network
2. 5% engage = 500 engagements from smaller pool
3. Fewer training signals for embedding
4. Weaker embedding position
5. Less OON retrieval

The advantage isn't that Grok "sees" more likes. It's that:

1. **In-network distribution scales with followers** — more followers = more initial reach at full score
2. **Embedding training scales with engagement volume** — more diverse engagements = stronger, better-defined embedding
3. **Phoenix retrieval depends on embedding strength** — stronger embedding = retrieved for more users

**Practical implication for smaller accounts:** You can't compete on in-network distribution. Compete on embedding concentration. A small account with focused engagement in a specific niche builds a strong embedding position in that region—which leads to consistent retrieval for users interested in that niche, even if raw numbers are lower.

---

### Does Premium Help?

X Premium claims to boost reply visibility. Here's what we actually know.

#### What's In the Open-Source Code

The open-source algorithm includes infrastructure for subscriptions, but **the actual boost weights are hidden**.

| Component                          | In Open Source? | Purpose                    |
| ---------------------------------- | --------------- | -------------------------- |
| `subscription_hydrator.rs`         | Yes             | Fetches subscription data  |
| `candidate.rs` subscription fields | Yes             | Stores subscription status |
| `REPLY_WEIGHT` parameter           | Referenced      | Used in scoring formula    |
| Actual weight VALUES               | **No**          | In private `params` module |
| Premium boost multipliers          | **No**          | Not released               |

From the weighted scorer, we see:

```rust
+ Self::apply(s.reply_score, p::REPLY_WEIGHT)
```

The `REPLY_WEIGHT` exists, but all actual values are in a params module **explicitly excluded from open source** for "security reasons."

#### What This Means

> **The Premium Reply Boost is REAL but HIDDEN.**
>
> X claims different tiers get different reply boosts:
> - Basic: "Small reply boost"
> - Premium: "Larger reply boost"
> - Premium+: "Largest reply boost"
>
> We cannot verify the mechanism or magnitude from the code.
> X open-sourced the **FRAMEWORK**, not the **TUNING**.

#### Reasonable Inferences (Not Proven)

Based on the algorithm architecture, the boost most likely applies at the **Grok ranking stage** (where weights are applied), not Phoenix retrieval (which uses pre-computed embeddings).

| Stage             | Premium Boost Likely? | Reasoning                                                               |
| ----------------- | --------------------- | ----------------------------------------------------------------------- |
| Phoenix Retrieval | Unlikely              | Uses pre-computed embeddings; no room for real-time subscription checks |
| Grok Ranking      | Likely                | Real-time features available; weighted_scorer applies multipliers       |

#### What Premium Cannot Change

Even if the boost exists and is significant, it operates within constraints:

| Factor                                 | Can Premium Change It?              |
| -------------------------------------- | ----------------------------------- |
| Your author embedding                  | No—trained from engagement history  |
| Who Phoenix retrieves you for          | No—based on embedding similarity    |
| Whether your content is valuable       | No—boost is a multiplier, not magic |
| Your followers' interest in new topics | No—audience mismatch is structural  |

#### The Honest Answer

We don't know exactly how much Premium helps because X kept the weights private. What we can say:

1. **The boost exists**—X explicitly advertises tiered reply boosts
2. **It's in the ranking stage**—based on architecture, not retrieval
3. **It's a multiplier, not a replacement**—bad content boosted is still bad content
4. **It doesn't solve structural problems**—embedding mismatches, wrong audiences

Whether the ROI is worth it depends on your specific situation and how heavily weighted the boost actually is—information we simply don't have access to.

---

### What Doesn't Work

Strategies that seem logical but fail because of how the algorithm actually works.

#### Hashtag Stuffing

The algorithm doesn't read your text. Embeddings are behavioral, not semantic. A post with #startup #entrepreneur #buildinpublic in the text is hashed the same as any other post.

Your content reaches people based on who has engaged with similar content before—not keyword matching.

#### Buying Followers/Bots

Bot accounts don't have rich engagement histories. They don't create training signal that improves your embedding. The algorithm effectively ignores them.

#### Deleting Old Content

Your embedding is trained on engagement patterns, not post content. Deleting old posts doesn't change the training data that already shaped your embedding.

#### Posting More to Wrong Audience

If your posts aren't reaching the right people, posting more doesn't help. You're just generating more "wrong audience ignores you" training signal, pushing your embedding in unhelpful directions.

---

### Summary

| Goal                            | Primary Strategy                                  | Key Insight                                           |
| ------------------------------- | ------------------------------------------------- | ----------------------------------------------------- |
| **Fix your feed**               | Intentional engagement (50-128 actions)           | Changes in days—only depends on last 128 interactions |
| **Grow as new account**         | Reply Game (10-20/day) + niche focus              | Access audiences before you have one                  |
| **Grow as established account** | Quality posts (2-3/day), serve existing followers | Don't break what's working                            |
| **Pivot to new niche**          | Reply Game in new niche + bridge content          | Commit fully—half-pivots are worst outcome            |

#### The Core Principle

> **The Core Principle**
>
> The algorithm learns from real engagement by real users.
>
> - There are no shortcuts.
> - There are no hacks.
> - There is only: create value for the right people, get engagement from those people, repeat.
>
> Everything in this chapter is about positioning yourself to make that core loop work more efficiently.

---

## Chapter 8: Conclusion

The X algorithm is not a black box. It's a system built on embeddings, engagement predictions, and scoring formulas—all designed to show users content they'll interact with. Your success on the platform comes down to one thing: getting real engagement from the right people.

There are no shortcuts. Hashtags don't hack the system. Buying followers doesn't train your embedding. Posting more to the wrong audience only makes things worse. The algorithm learns from behavior, not tricks. If you want to reach a specific audience, you need that audience to engage with you. The Reply Game, niche focus, and strategic posting aren't hacks—they're simply the most efficient ways to generate the engagement signals the algorithm needs to learn who you are and who should see your content.

Now you understand not just what works, but why it works. Use that knowledge.

---

## Want More Help?

I was able to understand this algorithm because I'm not just a coder—I'm deep into machine learning. I've written three books on the subject. Reading transformer architectures, embedding spaces, and two-tower models isn't a stretch for me; it's what I do.

If you want one-on-one coaching where I explain the algorithm in more detail, give you specific advice based on your situation, or provide access to the tools I've built for myself to grow on X, reach out to me at [@victor_explore](https://x.com/victor_explore). This is a paid service—but it's personalized guidance built on everything in this book, applied directly to your account.
