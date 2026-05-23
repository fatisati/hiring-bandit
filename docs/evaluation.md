# Evaluation

The algorithm is evaluated in the online phase only. Every 10 candidates (one batch), three metrics are computed and compared against hidden ground truth the algorithm never sees. This is standard **offline evaluation** — the same approach used in industry for recommendation systems, ad ranking, and hiring tools.

---

## Offline Evaluation

The algorithm makes decisions using only what it observes: noisy stage scores and outcomes of its own hiring decisions (`outcome = 1` if it hired a truly good candidate, `0` otherwise).

A separate evaluation layer compares those decisions against `ground_truth_hire` — the top 25% of candidates by hidden `true_quality`. The algorithm never sees this. Precision and recall are computed externally, purely to judge how well the policy is working.

This separation is intentional and best practice. If the algorithm could see ground truth during learning, it would be training on the very signal it is being evaluated against — making evaluation meaningless.

---

## Metrics

### Precision
> Of the candidates we hired, how many were truly good?

```
precision = correct_hires / total_hires
```

Low precision = advancing weak candidates through expensive stages.

### Recall
> Of all the truly good candidates, how many did we hire?

```
recall = correct_hires / truly_good_candidates
```

Low recall = thresholds too strict, missing good candidates early.

### Cost per Hire
> Hours of interviewer time spent per successful hire.

```
cost_per_hire = total_hours_spent / total_hires
```

Includes cost of rejected candidates — every stage they passed before rejection costs real hours.

---

## Expected Trends

As the algorithm learns, you should expect:

| metric | expected trend | why |
|---|---|---|
| precision | improving | bandit finds thresholds that filter out weak candidates |
| recall | improving (if `cost_weight` is low) | bandit stops missing good candidates |
| cost per hire | decreasing | weak candidates rejected earlier at cheaper stages |

**Important caveats:**
- Trends are not monotonic — early batches are noisy while the bandit is still exploring
- With binary outcomes (0/1), convergence is slower than with continuous reward signals
- The algorithm may briefly perform worse than the naive baseline before improving
- Higher `cost_weight` trades recall for lower cost — recall may decrease as the algorithm learns to reject more aggressively

---

## The Precision–Recall Tradeoff

- **Raise thresholds** → hire fewer, more selectively → precision up, recall down
- **Lower thresholds** → hire more broadly → recall up, precision down

The `--cost-weight` parameter shifts the balance:

| `cost_weight` | effect |
|---|---|
| 0.0 | cost-blind — optimises hire quality only |
| 0.1 (default) | mild cost pressure — pushes toward earlier rejection |
| high | aggressive cost cutting — high precision, low recall |

---

## Baseline

Before running the online algorithm, measure the **naive policy** — the fixed thresholds (60, 65, 70, 75) used during data generation. This is what the company was doing before learning began.

The notebook prints these metrics before the online phase starts. Use them as the floor: the algorithm should exceed this baseline in precision and recall while reducing cost per hire.

---

## Reading the Batch Report

```
 batch |  seen |  precision |  recall | total cost | cost/hire
------------------------------------------------------------------
     0 |    10 |       0.50 |    0.33 |       87.0 |      29.0
     1 |    10 |       0.67 |    0.44 |       72.0 |      24.0
     2 |    10 |       0.75 |    0.60 |       61.0 |      20.3
```

- `batch` — group of 10 online candidates, in arrival order
- `total cost` — hours spent on all 10 candidates (hired + rejected)
- `cost/hire` — total cost divided by number of hires in this batch

Total cost falling while precision rises = algorithm is working.
