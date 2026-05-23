# Data Generation

## Generate

```bash
python src/generate_data.py
```

Saves 200 candidates to `data/candidates.csv`.

---

## Historical vs Online Phase

Candidates are split by arrival order:

- **Historical** (first 50) — represent the company's past hiring record. Hiring decisions were made using fixed thresholds (60, 65, 70, 75) — a naive baseline policy. The algorithm uses these records to warm start: it learns which scores at each stage correlated with good outcomes, without having made the decisions itself.

- **Online** (remaining 150) — the algorithm makes its own decisions in real time, updating its thresholds after each candidate.

---

## Customize

Edit the constants at the top of `src/generate_data.py`:

```python
THRESHOLDS    = {"s0": 60, "s1": 65, "s2": 70, "s3": 75}  # data-generation filters
STAGE_COSTS   = {"s0": 1,  "s1": 3,  "s2": 8,  "s3": 15}  # hours per stage
STAGE_NOISE   = {"s0": 20, "s1": 12, "s2": 6,  "s3": 3}   # score noise per stage
TOP_PERCENTILE = 0.25   # top 25% by true quality are "truly good"
N_HISTORICAL   = 50     # warm-start candidates
```

To change dataset size or number of stages:

```python
data = generate_dataset(n=500, n_stages=3)
```

---

## Output Format

| column | description |
|---|---|
| `candidate_id` | unique identifier |
| `arrival_order` | 0-indexed arrival order |
| `phase` | `historical` or `online` |
| `batch` | batch number in online phase (`null` for historical) |
| `true_quality` | hidden true quality — never seen by the algorithm, used for evaluation |
| `s0_score` … `s3_score` | observed score at each stage, `null` if not reached |
| `hired` | 1 if passed all data-generation stage thresholds |
| `ground_truth_hire` | 1 if candidate is in the top `TOP_PERCENTILE` by true quality |
| `outcome` | 1 if hired AND truly good, else 0 — the learning signal |
| `total_cost` | hours spent on this candidate |
