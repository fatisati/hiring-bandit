import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from bandit import HiringPipeline
from evaluate import Evaluator

STAGE_COSTS = {"s0": 1, "s1": 3, "s2": 8, "s3": 15}
DEFAULT_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "candidates.csv")


def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Data file not found: {path}\nRun first: python src/generate_data.py"
        )
    with open(path) as f:
        reader = csv.DictReader(f)
        data = []
        for row in reader:
            for key in row:
                if row[key] == "" or row[key] == "None":
                    row[key] = None
                elif key in ("arrival_order", "hired", "ground_truth_hire", "batch"):
                    row[key] = int(row[key]) if row[key] is not None else None
                elif key in ("s0_score", "s1_score", "s2_score", "s3_score",
                             "true_quality", "outcome", "total_cost"):
                    row[key] = float(row[key]) if row[key] is not None else None
            data.append(row)
    return data


def main():
    parser = argparse.ArgumentParser(description="Run hiring pipeline optimization")
    parser.add_argument("--data", default=DEFAULT_DATA_PATH, help="Path to candidates CSV")
    parser.add_argument("--exploration", type=float, default=0.1, help="UCB exploration constant")
    parser.add_argument("--cost-weight", type=float, default=0.01, help="Penalty per hour spent (0 = cost blind)")
    parser.add_argument("--batch-size", type=int, default=10, help="Evaluation batch size")
    args = parser.parse_args()

    data = load_data(args.data)
    historical = [c for c in data if c["phase"] == "historical"]
    online = [c for c in data if c["phase"] == "online"]

    pipeline = HiringPipeline(stage_costs=STAGE_COSTS, exploration=args.exploration, cost_weight=args.cost_weight)
    evaluator = Evaluator(batch_size=args.batch_size)

    print(f"Warm starting on {len(historical)} historical candidates...")
    pipeline.warm_start(historical)

    print(f"Running online phase on {len(online)} candidates...")
    for candidate in online:
        result = pipeline.process(candidate)
        reward = pipeline.compute_reward(result, candidate["outcome"])
        pipeline.update(result["thresholds"], result["visited_stages"], reward)
        evaluator.record(candidate, result)

    evaluator.print_report()


if __name__ == "__main__":
    main()
