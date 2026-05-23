import numpy as np


class ThresholdBandit:
    """UCB-based threshold learner for a single pipeline stage."""

    def __init__(self, min_score=0, max_score=100, n_bins=10, exploration=0.1):
        self.bins = np.linspace(min_score, max_score, n_bins)
        self.exploration = exploration
        self.counts = np.zeros(n_bins)
        self.rewards = np.zeros(n_bins)

    def _ucb_score(self, total_counts):
        with np.errstate(divide="ignore", invalid="ignore"):
            avg = np.where(self.counts > 0, self.rewards / self.counts, 0)
            bonus = np.where(
                self.counts > 0,
                self.exploration * np.sqrt(np.log(total_counts + 1) / self.counts),
                np.inf,
            )
        return avg + bonus

    def select_threshold(self):
        total = self.counts.sum()
        scores = self._ucb_score(total)
        return self.bins[np.argmax(scores)]

    def update(self, threshold, reward):
        idx = np.argmin(np.abs(self.bins - threshold))
        self.counts[idx] += 1
        self.rewards[idx] += reward


class HiringPipeline:
    """Runs candidates through stages using per-stage bandit thresholds."""

    def __init__(self, stage_costs: dict, exploration=0.1, cost_weight=0.0):
        self.stages = list(stage_costs.keys())
        self.stage_costs = stage_costs
        self.cost_weight = cost_weight
        self.bandits = {s: ThresholdBandit(exploration=exploration) for s in self.stages}

    def process(self, candidate: dict) -> dict:
        thresholds = {s: self.bandits[s].select_threshold() for s in self.stages}
        total_cost = 0
        advanced_through = True
        visited_stages = []

        for stage in self.stages:
            score = candidate.get(f"{stage}_score")
            if score is None:
                advanced_through = False
                break
            visited_stages.append(stage)
            total_cost += self.stage_costs[stage]
            if score < thresholds[stage]:
                advanced_through = False
                break

        hired = advanced_through
        return {"hired": hired, "total_cost": total_cost, "thresholds": thresholds, "visited_stages": visited_stages}

    def compute_reward(self, result: dict, candidate_outcome: float) -> float:
        base = candidate_outcome if result["hired"] else 0.0
        return base - self.cost_weight * result["total_cost"]

    def update(self, thresholds: dict, visited_stages: list, reward: float):
        for stage in visited_stages:
            self.bandits[stage].update(thresholds[stage], reward)

    def warm_start(self, historical: list):
        for candidate in historical:
            if not candidate.get("hired"):
                # Rejected candidates have outcome=0 due to the old policy, not quality.
                # Including them would teach the bandit that advancing candidates is bad.
                continue
            result = self.process(candidate)
            reward = self.compute_reward(result, candidate["outcome"])
            self.update(result["thresholds"], result["visited_stages"], reward)
