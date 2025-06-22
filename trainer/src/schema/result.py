import numpy as np
from pydantic import BaseModel


class SingleEpisodeResult(BaseModel):
    episode_rewards: list[float] = None
    episode_steps: int = None
    episode_losses: list[float] = None
    episode_elapsed_time: float = None

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)

    def to_dict(self):
        return self.model_dump()

    def __str__(self):
        return f"total_rewards={round(np.sum(self.episode_rewards), 2)}, episode_steps={self.episode_steps}, total_loss={round(np.sum(self.episode_losses), 2)}, episode_elapsed_time={round(self.episode_elapsed_time, 2) if self.episode_elapsed_time is not None else None}"

    def __repr__(self):
        return self.__str__()


class TotalTrainResult(BaseModel):
    total_episodes: int = None
    returns: list[float] = None
    losses: list[float] = None
    elapsed_times: list[float] = None
    total_steps: int = None

    @classmethod
    def initialize(cls):
        return cls(
            total_episodes=0,
            returns=[],
            losses=[],
            elapsed_times=[],
            total_steps=0,
        )

    def update(self, episode_result: SingleEpisodeResult):
        self.total_episodes += 1
        self.returns.append(np.sum(episode_result.episode_rewards))
        self.losses.append(np.sum(episode_result.episode_losses))
        self.elapsed_times.append(episode_result.episode_elapsed_time)
        self.total_steps += episode_result.episode_steps

    def to_dict(self):
        return self.model_dump()

    def __str__(self):
        return f"TotalTrainResult(total_episodes={self.total_episodes}, returns={self.returns}, losses={self.losses}, elapsed_times={self.elapsed_times}, total_steps={self.total_steps})"

    def __repr__(self):
        return self.__str__()


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
