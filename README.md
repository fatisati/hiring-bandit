# Hiring Pipeline Optimization

A system to find the best candidates through a multi-stage interview pipeline while minimizing total interviewing cost.

## Docs

- [Algorithm](docs/algorithm.md) — implementation details, UCB formula, full loop
- [Evaluation](docs/evaluation.md) — metrics, how to read the report, precision/recall tradeoff
- [Data Generation](docs/data_generation.md) — how to generate synthetic candidate data

## Problem

A company runs candidates through a sequence of interview stages (s0, s1, s2, ...). Each stage costs a fixed amount of interviewer time. New candidates arrive continuously while others are already in the pipeline.

The goal is to **find the best candidates while minimizing total cost** — meaning: advance the right candidates, reject the wrong ones early, and don't waste expensive interview time on weak candidates.

---

## How We Model the Problem

### Stages and Costs

Each stage is a fixed-cost step that reveals more information about a candidate's true quality.

| stage | description | cost (hours) |
|---|---|---|
| s0 | resume review | 1 |
| s1 | online interview | 3 |
| s2 | onsite interview | 8 |
| s3 | final panel | 15 |

Cheaper stages come first and act as filters — only candidates who pass proceed to the next (more expensive) stage.

### Candidate Data

Each candidate accumulates scores as they move through stages. A `null` score means the candidate has not yet reached that stage. The final `outcome` is the single signal the model learns from.

| candidate_id | s0_score | s1_score | s2_score | s3_score | ground_truth_hire | outcome |
|---|---|---|---|---|---|---|
| C001 | 72 | 65 | null | null | 0 | 0 |
| C002 | 88 | 91 | 84 | 90 | 1 | 1 |
| C003 | 45 | null | null | null | 0 | 0 |
| C007 | 79 | 82 | 80 | 88 | 0 | 0 |

### Decision at Each Stage

After a candidate completes stage `si`, we make a binary decision:

- **Advance** → pay cost of next stage `si+1`
- **Reject** → stop, pay nothing more

The optimal decision depends on:
1. The candidate's score at the current stage
2. How well that score predicts success in later stages
3. The cost of the next stage vs. the expected value of finding out more

### What "Best" Means

Each candidate has a hidden `true_quality` score the algorithm never sees. The top 25% by true quality are labelled as truly good — these are the candidates the company actually wants to hire.

The outcome is binary:

```
hired AND truly good  →  outcome = 1
otherwise             →  outcome = 0
```

### Reward Signal

The bandit learns from a reward that accounts for both outcome and cost:

```
reward = outcome - cost_weight × hours_spent
```

| situation | outcome | hours spent | reward (cost_weight=0.01) |
|---|---|---|---|
| rejected at s0 | 0 | 1 | −0.01 |
| rejected at s3 | 0 | 27 | −0.27 |
| hired, not truly good | 0 | 27 | −0.27 |
| hired, truly good | 1 | 27 | **+0.73** |
| hired, truly good | 1 | 4 | **+0.96** |

A successful hire is always positive. The cost term creates mild pressure to identify and advance good candidates early — rather than routing everyone through all four expensive stages.

---

## The Solution

> Full implementation details in [docs/algorithm.md](docs/algorithm.md).

### Core Idea

At each stage, the algorithm maintains a **score threshold**. When a candidate completes a stage:

- Score **above** threshold → advance to the next stage
- Score **below** threshold → reject, stop spending

This is simple, but the power is in how the thresholds are set and how they improve over time.

### Why Fixed Thresholds Fail

A naive approach sets thresholds once and never changes them — for example, always reject anyone below 70. This breaks in two ways:

1. **Too strict** → you reject good candidates who had a bad day on s0. You never find out they would have aced s2.
2. **Too loose** → you advance weak candidates all the way to expensive stages, wasting cost.

Worse, fixed thresholds cannot adapt. If the quality of the candidate pool changes, or if you learn that s1 scores actually predict outcomes better than you thought, fixed thresholds stay wrong forever.

### What the Algorithm Actually Does

The algorithm treats each stage threshold as something to be **learned and continuously improved**, not set once. It works in three steps:

**Step 1 — Warm start from historical data**

Before processing any new candidates, the algorithm looks at past candidates and asks: for each stage, what score reliably predicted a good outcome? That becomes the starting threshold.

```
historical data → estimate: candidates who scored above X at s1 had outcome > 3.0
                → set initial threshold for s1 = X
```

**Step 2 — Make decisions and observe outcomes**

As new candidates arrive, the algorithm applies its current thresholds. Some are advanced, some rejected. Over time — as hired candidates get performance reviews — the algorithm observes whether its decisions were right.

**Step 3 — Update thresholds**

After observing outcomes, the algorithm updates its thresholds. If candidates above the threshold turned out worse than expected, raise it. If there are signs good candidates are being rejected too early, lower it slightly.

This is a continuous loop — every new candidate is both a decision and a data point.

### The Exploration-Exploitation Tradeoff

There is a fundamental tension in step 2:

**Exploitation** — trust the current thresholds. Reject anyone below them. This minimizes cost now but means the algorithm stops learning — if a threshold is slightly wrong, it stays wrong.

**Exploration** — occasionally advance a borderline candidate even if they score just below the threshold. This costs more in the short term but reveals whether the threshold is set correctly.

Without exploration, the algorithm is blind to its own mistakes. Without exploitation, it wastes money testing candidates it already knows are weak.

The solution is to balance both: mostly exploit current knowledge, but maintain controlled exploration around the threshold boundary. The further a candidate's score is from the threshold, the more confident the decision — the closer to the threshold, the more valuable it is to explore.

### Why This Is a Multi-Armed Bandit Problem

The bandit framework formalizes this tradeoff precisely.

Each stage threshold is a "policy" — a rule for who to advance. Think of each possible threshold value as an arm on a slot machine. Pulling an arm means: apply this threshold to the next candidate. The reward is the outcome of that decision.

The bandit algorithm (specifically UCB — Upper Confidence Bound) maintains:
- An **estimated reward** for each threshold — how good have decisions at this threshold been?
- An **uncertainty bonus** — how little data do we have at this threshold?

```
score(threshold) = estimated_reward + uncertainty_bonus
```

The uncertainty bonus is high for thresholds we have not tried much — the algorithm naturally explores those. As data accumulates, the bonus shrinks and the algorithm settles on what actually works.

This gives us a principled, automatic balance between exploration and exploitation — no manual tuning required.

**In one sentence:** the algorithm keeps asking "is this threshold good?" and updates its answer every time a new candidate outcome is observed.

### Concrete Example

Say stage s1 has only 3 possible thresholds: 60, 70, 80.

**Candidates 1–3** — nothing tried yet, so each threshold is explored once in order.

The reward shown uses `cost_weight=0.1`. Reaching s1 costs 3 hrs (s0=1 + s1=3 = 4 hrs total), but here we focus on s1's contribution of 3 hrs to keep it simple.

| candidate | s1_score | threshold | decision | final outcome | reward = outcome − 0.1×3hrs |
|---|---|---|---|---|---|
| C001 | 74 | 60 | advance | 4.0 (hired) | 3.7 |
| C002 | 68 | 70 | reject | 0 | −0.3 |
| C003 | 82 | 80 | advance | 4.5 (hired) | 4.2 |

Note: C001 and C003 are rewarded with the **final** outcome — whether they eventually got hired after all later stages. C002 is rejected at s1 so the pipeline never finds out their true quality; reward = 0 minus cost penalty.

```
after 3 candidates:
threshold:   60      70      80
avg_reward:  3.7    -0.3     4.2
```

**Candidate 4** — all tried once, uncertainty is equal, avg_reward dominates → pick threshold 80:

| candidate | s1_score | threshold | decision | final outcome | reward |
|---|---|---|---|---|---|
| C004 | 71 | 80 | reject | 0 | −0.3 |

```
after 4 candidates:
threshold:   60      70      80
avg_reward:  3.7    -0.3     1.95  ← dropped: 80 rejected a borderline candidate
```

**Candidate 5** — threshold 80 tried twice (small bonus), 60 and 70 tried once (large bonus). Threshold 60 wins: avg 3.7 + big uncertainty bonus beats 80's avg 1.95 + small bonus:

| candidate | s1_score | threshold | decision | final outcome | reward |
|---|---|---|---|---|---|
| C005 | 77 | 60 | advance | 3.8 (hired) | 3.5 |

```
after 5 candidates:
threshold:   60      70      80
avg_reward:  3.6    -0.3     1.95
```

Over time threshold 70 stays negative (cost of rejecting adds up, no hires), threshold 80 stays low (too strict, misses good candidates), and threshold 60 converges as the winner — not because it was set, but because the outcomes said so.

---

## Hyperparameters

| parameter | default | where to set | effect |
|---|---|---|---|
| `--exploration` | 0.1 | CLI | UCB exploration constant. Higher = spends more time testing uncertain thresholds. Lower = commits faster to current best. With binary 0/1 rewards, keep this well below 1.0 — otherwise the uncertainty bonus drowns out the reward signal and the algorithm never settles. |
| `--cost-weight` | 0.01 | CLI | Penalty per hour spent on a candidate. 0 = cost-blind. 0.01 (default) = mild pressure to reject early while keeping a good hire profitable (+0.73 reward). Too high (≥0.04) = even good hires get negative reward, causing the algorithm to reject everyone. |
| `--batch-size` | 10 | CLI | How many online candidates per evaluation report. Does not affect learning. |
| `n_bins` | 10 | `bandit.py` | Number of discrete threshold values per stage (spread evenly from 0–100). Fewer bins = more data per arm = faster convergence. 10 bins gives ~11-point resolution across the 0–100 score range. |
| `N_HISTORICAL` | 50 | `generate_data.py` | How many candidates go into the warm start phase. More = better initialized bandits, fewer candidates left for online learning. |

The two that matter most in practice are `--exploration` and `--cost-weight`. Everything else can stay at its default.

---

## Evaluation Over Time

Candidates are split into two phases based on arrival order:

**Historical phase** — the first batch of candidates (default 50). These represent past data the algorithm uses to set its initial thresholds.

**Online phase** — all candidates after that. The algorithm processes them one by one, updating its thresholds as it goes. Every 10 candidates, we measure performance to see if the algorithm is improving.

```
arrival 0-49    → historical phase  (warm start, set initial thresholds)
arrival 50-59   → online batch 0   (algorithm makes first live decisions)
arrival 60-69   → online batch 1   (updated thresholds)
arrival 70-79   → online batch 2   (updated again)
...
```

We track three metrics per batch:

| metric | question |
|---|---|
| precision | of candidates hired, how many were truly good? |
| recall | of truly good candidates, how many did we hire? |
| cost per hire | how many hours did we spend per successful hire? |

A well-functioning algorithm shows precision and recall trending up, cost per hire trending down, and all three flattening as the algorithm converges.
