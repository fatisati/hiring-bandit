# Data Generation

Synthetic candidate data for developing and evaluating the hiring pipeline algorithm.

---

## Generating Sample Data

```bash
python src/generate_data.py
```

This produces 100 candidates by default, saves to `data/candidates.csv`, and prints a summary:

```
--- Dataset Summary ---
Total candidates : 100
Hired            : 35 (35%)
Performance data : 23 (23%)
Total cost (hrs) : 1391
Cost per hire    : 39.7 hrs
```

---

## Customizing the Dataset

Open `src/generate_data.py` and adjust these settings at the top:

```python
# How strict each stage is — candidates must score above these to advance
THRESHOLDS = {
    "s0": 60,
    "s1": 65,
    "s2": 70,
    "s3": 75,
}

# Cost of each stage in hours
STAGE_COSTS = {
    "s0": 1,
    "s1": 3,
    "s2": 8,
    "s3": 15,
}

# Noise per stage — later stages are more accurate (smaller = more accurate)
STAGE_NOISE = {
    "s0": 20,
    "s1": 12,
    "s2": 6,
    "s3": 3,
}

# How many hired candidates have a performance review yet (0.0 to 1.0)
PERFORMANCE_REVIEW_RATE = 0.6

# Candidate is truly good if their hidden true quality exceeds this
GROUND_TRUTH_THRESHOLD = 70
```

To generate more candidates or change the number of stages:

```python
data = generate_dataset(n=500, n_stages=3)
```

---

## Generation Logic

Each candidate is generated in 5 steps:

**Step 1 — Assign a hidden true quality**

Every candidate gets a single hidden number representing their real ability. The algorithm never sees this.

```
true_quality = random, average 65, most fall between 40-90
```

A candidate is "truly good" if `true_quality >= GROUND_TRUTH_THRESHOLD` (default 70).

**Step 2 — Generate noisy stage scores**

Each stage score is the true quality plus random noise. Noise decreases at later stages — later interviews are more accurate.

```
s0_score = true_quality + noise(±20)   # rough — resume screen
s1_score = true_quality + noise(±12)   # better — online test
s2_score = true_quality + noise(±6)    # good — onsite
s3_score = true_quality + noise(±3)    # very accurate — final panel
```

A candidate with true quality 75 might score 60 on s0 (unlucky, gets rejected) or 90 (lucky, advances). This simulates the randomness of real interviews.

**Step 3 — Apply thresholds to decide who advances**

After each stage score is generated, it is compared to the threshold for that stage:

```
s0 score >= 60 → advance to s1, else reject
s1 score >= 65 → advance to s2, else reject
...
```

If rejected at any stage, all later scores are `null` — those interviews never happened.

**Step 4 — Assign hired and performance score**

If the candidate passed all stage thresholds → `hired = 1`. For hired candidates, 60% of the time a performance score (1-5) is generated, correlated with their true quality. The other 40% have no review yet.

**Step 5 — Compute outcome**

```
hired + performance score available  →  outcome = performance score
hired + no performance score yet     →  outcome = mean performance score of all reviewed employees
not hired                            →  outcome = 0
```

The mean is computed from all candidates who were hired and have a performance review — not a fixed constant. It is recalculated each time the dataset is generated.

---

## Ground Truth Labels

The generator assigns each candidate a hidden `true_quality` score. Stage scores are noisy observations of this true quality — later stages have less noise and therefore reveal it more accurately.

```
true_quality = 75          # hidden, never seen by the algorithm

s0_score = 75 + noise(±20)   # rough — resume screen
s1_score = 75 + noise(±12)   # better — online test
s2_score = 75 + noise(±6)    # good — onsite
s3_score = 75 + noise(±3)    # very accurate — final panel
```

`ground_truth_hire = 1` if `true_quality >= GROUND_TRUTH_THRESHOLD`.

This lets you evaluate the algorithm: did it hire the candidates that were truly good, not just the ones that looked good on a noisy resume screen?

---

## Output Format

| column | description |
|---|---|
| `candidate_id` | unique candidate identifier |
| `arrival_order` | the order the candidate arrived (0-indexed) |
| `phase` | `historical` or `online` — see [evaluation](../README.md#evaluation-over-time) |
| `batch` | batch number within the online phase (`null` for historical) |
| `s0_score` ... `s3_score` | observed score at each stage, `null` if not reached |
| `hired` | 1 if hired by the algorithm, 0 if rejected |
| `performance_score` | post-hire review score (1-5), `null` if not available yet |
| `outcome` | combined target signal used for learning |
| `total_cost` | total hours spent on this candidate |
| `true_quality` | hidden true quality (ground truth, for evaluation only) |
| `ground_truth_hire` | 1 if candidate was truly good, 0 otherwise |
