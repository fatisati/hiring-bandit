import random
import csv
import os

# Stage thresholds — candidates must score above these to advance
THRESHOLDS = {
    "s0": 60,
    "s1": 65,
    "s2": 70,
    "s3": 75,
}

# Stage costs in hours
STAGE_COSTS = {
    "s0": 1,
    "s1": 3,
    "s2": 8,
    "s3": 15,
}

# Noise per stage — later stages are more accurate
STAGE_NOISE = {
    "s0": 20,
    "s1": 12,
    "s2": 6,
    "s3": 3,
}

# A candidate is truly good if their true quality exceeds this
GROUND_TRUTH_THRESHOLD = 70

# Probability that a hired candidate has a performance review yet
PERFORMANCE_REVIEW_RATE = 0.6

# Initial mean performance score — used before enough reviews exist
# Updated dynamically once candidates with reviews are generated
MEAN_PERFORMANCE = 3.2

# Number of historical candidates before online phase begins
N_HISTORICAL = 50

# Candidates per evaluation batch in the online phase
BATCH_SIZE = 10


def noisy_score(true_quality, stage):
    noise = STAGE_NOISE[stage]
    return round(min(100, max(0, random.gauss(true_quality, noise))), 1)


def generate_candidate(candidate_id, arrival_order, n_stages=4):
    stages = ["s0", "s1", "s2", "s3"][:n_stages]

    phase = "historical" if arrival_order < N_HISTORICAL else "online"
    batch = None if phase == "historical" else (arrival_order - N_HISTORICAL) // BATCH_SIZE

    row = {
        "candidate_id": candidate_id,
        "arrival_order": arrival_order,
        "phase": phase,
        "batch": batch,
    }

    # Hidden true quality — the algorithm never sees this
    true_quality = random.gauss(65, 15)
    row["true_quality"] = round(true_quality, 1)
    row["ground_truth_hire"] = int(true_quality >= GROUND_TRUTH_THRESHOLD)

    # Observed scores — noisy views of true quality, noise decreases each stage
    prev_passed = True
    for stage in stages:
        if not prev_passed:
            row[f"{stage}_score"] = None
            continue

        observed = noisy_score(true_quality, stage)
        row[f"{stage}_score"] = observed
        prev_passed = observed >= THRESHOLDS[stage]

    # Hired if passed all observed stage thresholds
    hired = prev_passed and row.get(f"{stages[-1]}_score") is not None
    row["hired"] = int(hired)

    # Performance score — only for hired candidates, not always available yet
    if hired and random.random() < PERFORMANCE_REVIEW_RATE:
        perf = 1 + (true_quality / 100) * 4 + random.gauss(0, 0.3)
        row["performance_score"] = round(min(5.0, max(1.0, perf)), 1)
    else:
        row["performance_score"] = None

    # Outcome — single target signal for the algorithm
    if not hired:
        row["outcome"] = 0.0
    elif row["performance_score"] is not None:
        row["outcome"] = row["performance_score"]
    else:
        row["outcome"] = MEAN_PERFORMANCE

    # Total cost spent on this candidate
    row["total_cost"] = sum(
        STAGE_COSTS[s] for s in stages if row[f"{s}_score"] is not None
    )

    return row


def generate_dataset(n=200, n_stages=4, seed=42):
    random.seed(seed)
    candidates = [
        generate_candidate(f"C{str(i+1).zfill(3)}", arrival_order=i, n_stages=n_stages)
        for i in range(n)
    ]

    # Recompute outcome using the actual mean across all reviewed employees
    reviewed = [r["performance_score"] for r in candidates if r["performance_score"] is not None]
    mean_perf = sum(reviewed) / len(reviewed) if reviewed else MEAN_PERFORMANCE

    for r in candidates:
        if r["hired"] and r["performance_score"] is None:
            r["outcome"] = round(mean_perf, 2)

    return candidates


def save_csv(data, filename="data/candidates.csv"):
    if not data:
        return
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)
    print(f"Saved {len(data)} candidates to {filename}")


def print_summary(data):
    historical = [r for r in data if r["phase"] == "historical"]
    online = [r for r in data if r["phase"] == "online"]

    def metrics(subset, label):
        if not subset:
            return
        total = len(subset)
        hired = sum(r["hired"] for r in subset)
        truly_good = sum(r["ground_truth_hire"] for r in subset)
        correct_hires = sum(1 for r in subset if r["hired"] and r["ground_truth_hire"])
        total_cost = sum(r["total_cost"] for r in subset)
        precision = correct_hires / hired if hired else 0
        recall = correct_hires / truly_good if truly_good else 0

        print(f"\n--- {label} ---")
        print(f"Candidates  : {total}")
        print(f"Truly good  : {truly_good} ({100*truly_good//total}%)")
        print(f"Hired       : {hired} ({100*hired//total}%)")
        print(f"Precision   : {precision:.2f}")
        print(f"Recall      : {recall:.2f}")
        print(f"Total cost  : {total_cost} hrs")
        print(f"Cost/hire   : {total_cost/hired:.1f} hrs" if hired else "Cost/hire   : n/a")

    metrics(historical, "Historical Phase")
    metrics(online, "Online Phase")

    # Per-batch breakdown for online phase
    if online:
        batches = sorted(set(r["batch"] for r in online))
        print(f"\n--- Online Phase: Per Batch ---")
        print(f"{'batch':>6} | {'seen':>6} | {'precision':>10} | {'recall':>7} | {'cost/hire':>10}")
        print("-" * 50)
        for b in batches:
            batch_data = [r for r in online if r["batch"] == b]
            hired = sum(r["hired"] for r in batch_data)
            truly_good = sum(r["ground_truth_hire"] for r in batch_data)
            correct = sum(1 for r in batch_data if r["hired"] and r["ground_truth_hire"])
            cost = sum(r["total_cost"] for r in batch_data)
            precision = correct / hired if hired else 0
            recall = correct / truly_good if truly_good else 0
            cost_per_hire = cost / hired if hired else 0
            print(f"{b:>6} | {len(batch_data):>6} | {precision:>10.2f} | {recall:>7.2f} | {cost_per_hire:>10.1f}")


if __name__ == "__main__":
    data = generate_dataset(n=200, n_stages=4)
    print_summary(data)
    save_csv(data)
