# Evaluation

The algorithm is evaluated in the online phase only. Every 10 candidates (one batch), three metrics are computed.

---

## Metrics

### Precision
> Of the candidates we hired, how many were truly good?

```
precision = correct_hires / total_hires
```

Low precision means the algorithm is advancing weak candidates all the way through — wasting expensive stage time on people who shouldn't have been hired.

### Recall
> Of all the truly good candidates, how many did we hire?

```
recall = correct_hires / truly_good_candidates
```

Low recall means the algorithm is rejecting good candidates too early — setting thresholds too high and missing people who would have performed well.

### Cost per Hire
> How many hours of interviewer time did we spend per successful hire?

```
cost_per_hire = total_hours_spent / total_hires
```

This includes the cost of rejected candidates — every stage they passed through before being rejected is real time spent. Rejecting weak candidates earlier directly reduces this number.

---

## What to Look For

A well-functioning algorithm shows all three metrics improving over batches and flattening as it converges:

| phase | precision | recall | cost/hire |
|---|---|---|---|
| early batches | low — still exploring | low — thresholds not calibrated | high — rejecting late |
| mid batches | rising | rising | falling |
| late batches | stable | stable | stable |

If they don't converge, consider increasing `N_HISTORICAL` (more warm start data) or tuning `--exploration`.

---

## The Precision–Recall Tradeoff

Precision and recall pull in opposite directions:

- **Raise thresholds** → hire fewer, but more selectively → precision up, recall down
- **Lower thresholds** → hire more broadly → recall up, precision down

The algorithm balances this automatically via UCB. The `--cost-weight` parameter shifts the balance:

| `cost_weight` | effect |
|---|---|
| 0.0 | cost-blind — optimises hire quality only, thresholds may stay loose |
| 0.1 (default) | mild cost pressure — pushes thresholds up, earlier rejection |
| high | aggressive cost cutting — high precision, low recall, low cost |

---

## Reading the Batch Report

```
 batch |  seen |  precision |  recall | total cost | cost/hire
------------------------------------------------------------------
     0 |    10 |       0.50 |    0.33 |        87.0 |      29.0
     1 |    10 |       0.67 |    0.44 |        72.0 |      24.0
     2 |    10 |       0.75 |    0.60 |        61.0 |      20.3
```

- `batch` — batch number (0-indexed)
- `seen` — candidates processed in this batch
- `total cost` — hours spent on all candidates in the batch (hired + rejected)
- `cost/hire` — total cost divided by number of hires this batch

Total cost falling while precision rises is the signal the algorithm is working: it's rejecting the right people early and advancing the right people through.

---

## Ground Truth

`ground_truth_hire = 1` if a candidate's hidden `true_quality` exceeds the threshold (default 70). The algorithm never sees `true_quality` — it only sees noisy stage scores. This is what makes the problem hard and what the algorithm has to learn from outcomes alone.
