import numpy as np
from pydantic import BaseModel


class TrainResult(BaseModel):
    episode: int = None
    episode_return: float = None
    episode_loss: float = None
    episode_elapsed_time: float = None

    @classmethod
    def from_train_results(
        cls, scores: list[float], losses: list[float], elapsed_times: list[float]
    ):
        return cls(
            episode=len(scores),
            episode_return=scores[-1],
            episode_loss=losses[-1],
            episode_elapsed_time=elapsed_times[-1],
        )

    def to_dict(self):
        return self.model_dump()

    def __str__(self):
        return f"Episode {self.episode}: Return={self.episode_return:.2f}, Loss={self.episode_loss:.4f}, Elapsed Time={self.episode_elapsed_time:.2f}"


class EvalResult(BaseModel):
    avg_score: float
    std_score: float
    min_score: float
    max_score: float
    all_scores: list[float]
    steps: list[int]

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)

    @classmethod
    def from_eval_results(cls, scores: list[float], steps: list[int] = None):
        return cls(
            avg_score=np.mean(scores),
            std_score=np.std(scores),
            min_score=np.min(scores),
            max_score=np.max(scores),
            all_scores=[f"{s:.2f}" for s in scores],
            steps=steps,
        )

    def to_dict(self):
        return self.model_dump()

    def __lt__(self, other):
        if isinstance(other, EvalResult):
            return self.avg_score < other.avg_score
        return NotImplemented

    def __le__(self, other):
        if isinstance(other, EvalResult):
            return self.avg_score <= other.avg_score
        return NotImplemented

    def __gt__(self, other):
        if isinstance(other, EvalResult):
            return self.avg_score > other.avg_score
        return NotImplemented

    def __ge__(self, other):
        if isinstance(other, EvalResult):
            return self.avg_score >= other.avg_score
        return NotImplemented

    def __eq__(self, other):
        if isinstance(other, EvalResult):
            return self.avg_score == other.avg_score
        return NotImplemented

    def __ne__(self, other):
        if isinstance(other, EvalResult):
            return self.avg_score != other.avg_score
        return NotImplemented

    def __str__(self):
        return (
            f"EvalResult(\n"
            f"  mean_score={self.avg_score:.3f},\n"
            f"  std_score={self.std_score:.3f},\n"
            f"  min_score={self.min_score:.3f},\n"
            f"  max_score={self.max_score:.3f},\n"
            f"  max_steps={max(self.steps)}\n"
            f")"
        )

    def __repr__(self):
        """Concise representation for debugging."""
        return (
            f"EvalResult(mean={self.avg_score:.3f}, std={self.std_score:.3f}, "
            f"range=[{self.min_score:.3f}, {self.max_score:.3f}], "
            f"n={len(self.all_scores)}, max_steps={self.max(self.steps)})"
        )
