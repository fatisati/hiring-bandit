# Algorithm

## Overview

The system has three components:

```
ThresholdBandit   — learns the best threshold for one stage
HiringPipeline    — runs candidates through all stages using one bandit per stage
Evaluator         — tracks precision, recall, cost per hire over time
```

---

## ThresholdBandit

One bandit per stage. Learns which score threshold leads to the best outcomes.

### Setup

The score range (0-100) is divided into 20 bins. Each bin is a candidate threshold value. The bandit tracks two things per bin:

```
counts[i]   — how many times threshold bin i was used
rewards[i]  — total outcome score accumulated at bin i
```

### Selecting a threshold (UCB)

Every time a candidate arrives, the bandit picks a threshold using UCB:

```
score(bin) = avg_reward(bin) + exploration * sqrt(log(total_uses) / uses(bin))
```

- `avg_reward` — how good this threshold has been on average
- `exploration bonus` — high when a bin has been used rarely, pushes the algorithm to try less-explored thresholds
- bins never used get score = infinity → always tried first

The bin with the highest score is selected as the threshold.

### Updating

After a candidate's outcome is known:

```python
pipeline.update(thresholds, visited_stages, reward)
```

Only stages the candidate actually visited are updated — if a candidate was rejected at s1, we don't update s2 or s3 since we have no information about them.

---

## Reward Signal

The bandit does not learn from raw outcomes directly. It learns from a **reward** that accounts for both hire quality and cost:

```
reward = base_outcome - cost_weight × total_hours_spent
```

Where `base_outcome` is:
- `performance_score` — if the pipeline hired the candidate and a review exists
- `running mean of all reviews so far` — if hired but no review yet
- `0` — if the pipeline rejected the candidate

```
pipeline rejected                 →  base = 0
pipeline hired, review exists     →  base = performance_score (1–5)
pipeline hired, no review yet     →  base = mean of all reviews seen so far
```

The running mean is seeded from historical hires before the online phase begins and updated each time a new reviewed hire is observed.

### Why each stage gets the final reward

Each stage bandit updates with the **final** reward — not a stage-specific signal. Stage s1 does not know what happened at s2 or s3. It only knows: "I used threshold X, and the candidate's final outcome was Y."

This works because the reward naturally encodes whether s1's decision was good:

```
s1 too loose  →  weak candidate advances → rejected at s2 → reward = 0 − cost penalty
s1 too strict →  good candidate rejected → reward = 0 − cost penalty
s1 correct    →  good candidate advances → hired → reward = 4.2 − cost penalty (positive)
```

All stage bandits are aligned toward the same goal without needing to communicate with each other.

---

## HiringPipeline

Holds one `ThresholdBandit` per stage. Processes candidates sequentially.

### process(candidate)

1. Ask each bandit for its current best threshold
2. Walk the candidate through stages in order
3. At each stage: if score < threshold → reject, stop
4. If candidate passes all stages → hired

Returns: `hired`, `total_cost`, `thresholds used`, `visited_stages`

### warm_start(historical)

Runs all historical candidates through the pipeline and updates bandits after each one. This gives the bandits a starting point before live candidates arrive.

---

## Evaluator

Tracks metrics per batch of candidates (default: 10 per batch).

Per batch:

| metric | formula |
|---|---|
| precision | correct hires / total hires |
| recall | correct hires / truly good candidates |
| cost per hire | total cost / total hires |

`correct hire` = hired AND `ground_truth_hire = 1`

---

## Running

```bash
# Generate data first
python src/generate_data.py

# Run the pipeline
python src/main.py

# Options
python src/main.py --exploration 2.0 --cost-weight 0.1 --batch-size 20
```

| flag | default | effect |
|---|---|---|
| `--exploration` | 1.0 | How aggressively to try untested thresholds. Higher = more exploration, slower convergence. |
| `--cost-weight` | 0.0 | Penalty per hour spent. 0 = optimise hire quality only. 0.1 = mild pressure to reject early. |
| `--batch-size` | 10 | Candidates per evaluation report. Does not affect learning. |

---

## The Full Loop

```
historical candidates
        ↓
   warm_start()              ← bandits get initial signal
   seed reviewed_scores      ← running mean starts from historical reviews
        ↓
new candidate arrives
        ↓
  select_threshold()         ← UCB picks best threshold per stage
        ↓
   process()                 ← advance or reject at each stage
        ↓
  resolve outcome:
    hired + review   → use performance_score
    hired + no review → use running mean of reviews so far
    rejected         → 0
        ↓
  compute_reward()           ← outcome − cost_weight × hours_spent
        ↓
    update()                 ← only visited stages updated
        ↓
  evaluator.record()         ← track metrics
        ↓
next candidate → repeat
```
