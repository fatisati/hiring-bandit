# Evaluation

The algorithm is evaluated in the online phase only. Every 10 candidates (one batch), three metrics are computed.

---

## Ground Truth

Each candidate has a hidden `true_quality` score the algorithm never sees. The top 25% of candidates by true quality are labelled `ground_truth_hire = 1` — these are the people the company actually wants to hire.

The algorithm only sees noisy stage scores. Its job is to identify the top 25% using those scores alone.

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

## What to Look For

| phase | precision | recall | cost/hire |
|---|---|---|---|
| early batches | low — still exploring | low — thresholds not calibrated | high — rejecting late |
| mid batches | rising | rising | falling |
| late batches | stable | stable | stable |

If metrics don't converge, try increasing `N_HISTORICAL` or tuning `--exploration`.

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

---

## Baseline

Before running the online algorithm, measure the **naive policy** — the fixed thresholds (60, 65, 70, 75) used during data generation. This is what the company was doing before learning began.

The notebook prints these metrics before the online phase starts. Use them as the floor: the algorithm should exceed this baseline in precision and recall while reducing cost per hire.

---

## Ground Truth

`ground_truth_hire = 1` for the top 25% of candidates by `true_quality`. The algorithm never sees `true_quality` — it only sees noisy stage scores.

Precision and recall are computed by comparing the algorithm's decisions against this hidden ground truth. This is standard offline evaluation: ground truth is used only by the evaluator, never by the algorithm. The algorithm learns purely from `outcome` (1 if it hired a truly good candidate, 0 otherwise).
