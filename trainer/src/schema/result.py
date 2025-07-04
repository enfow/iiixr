from typing import Optional

import numpy as np
from pydantic import BaseModel


class UpdateLoss(BaseModel):
    pass

    def to_dict(self):
        return self.model_dump()


class PPOUpdateLoss(UpdateLoss):
    actor_loss: float
    critic_loss: float
    entropy_loss: float

    @property
    def total_loss(self):
        return self.actor_loss + self.critic_loss + self.entropy_loss


class SACUpdateLoss(UpdateLoss):
    actor_loss: float
    value_loss: float
    critic1_loss: float
    critic2_loss: float
    alpha_loss: float

    @property
    def total_loss(self):
        return self.actor_loss + self.critic1_loss + self.critic2_loss


class DiscreteSACUpdateLoss(UpdateLoss):
    actor_loss: float
    critic1_loss: float
    critic2_loss: float
    alpha_loss: float

    @property
    def total_loss(self):
        return self.actor_loss + self.critic1_loss + self.critic2_loss + self.alpha_loss


class DQNUpdateLoss(UpdateLoss):
    loss: float

    @property
    def total_loss(self):
        return self.loss


class C51UpdateLoss(DQNUpdateLoss):
    pass


class RainbowDQNUpdateLoss(DQNUpdateLoss):
    pass


class DDQNUpdateLoss(DQNUpdateLoss):
    pass


class TD3UpdateLoss(UpdateLoss):
    actor_loss: float
    critic_loss: float

    @property
    def total_loss(self):
        return self.actor_loss + self.critic_loss


class SingleEpisodeResult(BaseModel):
    episode_number: int = None
    episode_total_reward: float = None
    episode_steps: float = None  # average steps if there are multiple episode(ppo)
    episode_losses: list[UpdateLoss] = None
    episode_elapsed_time: float = None

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)

    def to_dict(self):
        return self.model_dump()

    def to_log_dict(self):
        return {
            "ep": self.episode_number,
            "total_rewards": self.episode_total_reward,
            "episode_steps": self.episode_steps,
            "total_loss": self.episode_total_loss,
            "loss_details": [loss.to_dict() for loss in self.episode_losses],
            "episode_elapsed_time": self.episode_elapsed_time,
        }

    @property
    def episode_total_loss(self):
        return np.sum([loss.total_loss for loss in self.episode_losses])

    def __str__(self):
        return f"ep={self.episode_number}, total_rewards={round(self.episode_total_reward, 2)}, episode_steps={self.episode_steps}, total_loss={round(self.episode_total_loss, 2)}, episode_elapsed_time={round(self.episode_elapsed_time, 2) if self.episode_elapsed_time is not None else None}"

    def __repr__(self):
        return self.__str__()


class TotalTrainResult(BaseModel):
    total_episodes: int = None
    returns: list[float] = None
    total_losses: list[float] = None
    losses: list[UpdateLoss] = None
    elapsed_times: list[float] = None
    total_steps: int = None

    @classmethod
    def initialize(cls):
        return cls(
            total_episodes=0,
            returns=[],
            total_losses=[],
            losses=[],
            elapsed_times=[],
            total_steps=0,
        )

    def update(self, episode_result: SingleEpisodeResult):
        self.total_episodes += 1
        self.returns.append(episode_result.episode_total_reward)
        self.total_losses.append(episode_result.episode_total_loss)
        self.losses.append(episode_result.episode_losses)
        self.elapsed_times.append(episode_result.episode_elapsed_time)
        self.total_steps += episode_result.episode_steps

    def to_dict(self):
        return self.model_dump()

    def __str__(self):
        return f"TotalTrainResult(total_episodes={self.total_episodes}, returns={self.returns}, losses={self.total_losses}, elapsed_times={self.elapsed_times}, total_steps={self.total_steps})"

    def __repr__(self):
        return self.__str__()


class EvalResult(BaseModel):
    train_episode_number: Optional[int] = None
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
    def from_eval_results(
        cls,
        scores: list[float],
        steps: list[int] = None,
        train_episode_number: int = None,
    ):
        return cls(
            avg_score=np.mean(scores),
            std_score=np.std(scores),
            min_score=np.min(scores),
            max_score=np.max(scores),
            all_scores=[f"{s:.2f}" for s in scores],
            steps=steps,
            train_episode_number=train_episode_number,
        )

    def to_dict(self):
        return self.model_dump()

    def to_log_dict(self):
        return self.to_dict()

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
            f"  train_ep={self.train_episode_number},\n"
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
            f"EvalResult(train_ep={self.train_episode_number}, mean={self.avg_score:.3f}, std={self.std_score:.3f}, "
            f"range=[{self.min_score:.3f}, {self.max_score:.3f}], "
            f"n={len(self.all_scores)}, max_steps={self.max(self.steps)})"
        )
