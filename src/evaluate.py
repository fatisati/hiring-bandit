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

    def cumulative(self):
        """Running totals across all batches seen so far."""
        hired, correct, truly_good, cost = 0, 0, 0, 0.0
        rows = []
        for m in self.batches:
            hired      += m.hired
            correct    += m.correct_hires
            truly_good += m.truly_good
            cost       += m.total_cost
            rows.append({
                "batch":          m.batch,
                "precision":      correct / hired      if hired      else None,
                "recall":         correct / truly_good if truly_good else None,
                "cost_per_hire":  cost    / hired      if hired      else None,
            })
        return rows

    def print_report(self):
        print(f"\n{'batch':>6} | {'seen':>5} | {'precision':>10} | {'recall':>7} | {'total cost':>11} | {'cost/hire':>10}")
        print("-" * 66)
        for m in self.batches:
            cph = f"{m.cost_per_hire:>10.1f}" if m.hired else f"{'N/A':>10}"
            print(
                f"{m.batch:>6} | {m.candidates:>5} | "
                f"{m.precision:>10.2f} | {m.recall:>7.2f} | "
                f"{m.total_cost:>11.1f} | {cph}"
            )
