from dataclasses import dataclass, field


@dataclass
class BatchMetrics:
    batch: int
    candidates: int = 0
    hired: int = 0
    truly_good: int = 0
    correct_hires: int = 0
    total_cost: float = 0.0

    @property
    def precision(self):
        return self.correct_hires / self.hired if self.hired else 0.0

    @property
    def recall(self):
        return self.correct_hires / self.truly_good if self.truly_good else 0.0

    @property
    def cost_per_hire(self):
        return self.total_cost / self.hired if self.hired else 0.0


class Evaluator:
    def __init__(self, batch_size=10):
        self.batch_size = batch_size
        self.batches: list[BatchMetrics] = []
        self._current: BatchMetrics = BatchMetrics(batch=0)
        self._online_count = 0

    def record(self, candidate: dict, result: dict):
        m = self._current
        m.candidates += 1
        m.truly_good += candidate["ground_truth_hire"]
        m.total_cost += result["total_cost"]

        if result["hired"]:
            m.hired += 1
            if candidate["ground_truth_hire"]:
                m.correct_hires += 1

        self._online_count += 1
        if self._online_count % self.batch_size == 0:
            self.batches.append(self._current)
            self._current = BatchMetrics(batch=len(self.batches))

    def print_report(self):
        print(f"\n{'batch':>6} | {'seen':>5} | {'precision':>10} | {'recall':>7} | {'total cost':>11} | {'cost/hire':>10}")
        print("-" * 66)
        for m in self.batches:
            print(
                f"{m.batch:>6} | {m.candidates:>5} | "
                f"{m.precision:>10.2f} | {m.recall:>7.2f} | "
                f"{m.total_cost:>11.1f} | {m.cost_per_hire:>10.1f}"
            )
