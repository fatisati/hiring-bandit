import random
import csv
import os

STAGE_COSTS = {
    "s0": 1,
    "s1": 3,
    "s2": 8,
    "s3": 15,
}

STAGE_NOISE = {
    "s0": 20,
    "s1": 12,
    "s2": 6,
    "s3": 3,
}

# Thresholds used during data generation to decide who advances
THRESHOLDS = {
    "s0": 60,
    "s1": 65,
    "s2": 70,
    "s3": 75,
}

# Fraction of the candidate pool that are truly good
TOP_PERCENTILE = 0.25

N_HISTORICAL = 50
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

    true_quality = random.gauss(65, 15)
    row["true_quality"] = round(true_quality, 1)

    prev_passed = True
    for stage in stages:
        if not prev_passed:
            row[f"{stage}_score"] = None
            continue
        observed = noisy_score(true_quality, stage)
        row[f"{stage}_score"] = observed
        prev_passed = observed >= THRESHOLDS[stage]

    hired = prev_passed and row.get(f"{stages[-1]}_score") is not None
    row["hired"] = int(hired)
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

    # Ground truth: top TOP_PERCENTILE of candidates by true quality
    sorted_qualities = sorted([r["true_quality"] for r in candidates], reverse=True)
    gt_threshold = sorted_qualities[int(len(candidates) * TOP_PERCENTILE)]

    for r in candidates:
        r["ground_truth_hire"] = int(r["true_quality"] >= gt_threshold)
        r["outcome"] = float(r["hired"] and r["ground_truth_hire"])

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


if __name__ == "__main__":
    data = generate_dataset(n=200, n_stages=4)
    print_summary(data)
    save_csv(data)
