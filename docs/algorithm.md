# Algorithm

## Components

```
ThresholdBandit   — learns the best threshold for one stage
HiringPipeline    — runs candidates through all stages using one bandit per stage
Evaluator         — tracks precision, recall, cost per hire over time
```

---

## ThresholdBandit

Divides 0–100 into 20 bins. Each bin is a candidate threshold. Selects using UCB:

```
score(bin) = avg_reward(bin) + exploration * sqrt(log(total_uses) / uses(bin))
```

Unvisited bins get score = ∞ and are always tried first. The bin with the highest score is selected.

Updates only the bins for stages the candidate actually visited — rejected at s1 means s2/s3 get no update.

---

## Reward Signal

```
reward = base_outcome - cost_weight × total_hours_spent
```

| situation | base_outcome |
|---|---|
| pipeline hired, review exists | performance_score (1–5) |
| pipeline hired, no review yet | running mean of all reviews so far |
| pipeline rejected | 0 |

The running mean is seeded from historical reviews before the online phase begins.

---

## Evaluator

Per batch of 10 candidates:

| metric | formula |
|---|---|
| precision | correct hires / total hires |
| recall | correct hires / truly good candidates |
| total cost | sum of hours spent |
| cost per hire | total cost / total hires |

`correct hire` = hired AND `ground_truth_hire = 1`

---

## Running

```bash
python src/generate_data.py          # generate data first
python src/main.py                   # run with defaults
python src/main.py --exploration 2.0 --cost-weight 0.1
```

| flag | default | effect |
|---|---|---|
| `--exploration` | 1.0 | UCB exploration constant. Higher = more exploration. |
| `--cost-weight` | 0.1 | Penalty per hour spent. 0 = cost-blind. |
| `--batch-size` | 10 | Candidates per evaluation report. |

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
    hired + review    → performance_score
    hired + no review → running mean
    rejected          → 0
        ↓
  compute_reward()           ← outcome − cost_weight × hours_spent
        ↓
    update()                 ← only visited stages updated
        ↓
  evaluator.record()         ← track metrics
        ↓
next candidate → repeat
```
