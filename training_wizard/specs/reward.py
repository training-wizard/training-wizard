"""Reward modeling."""

from abc import ABC, abstractmethod
from logging import Logger
from typing import Literal, Self

import numpy as np
import structlog
from pydantic import model_validator
from transformers import Trainer

from ..specs.spec import Spec

logger: Logger = structlog.get_logger()


class RewardSpec(Spec, ABC):
    """A specification for a reward model."""

    name: str = "reward"
    """The name of the reward."""

    @property
    def trainer(self) -> Trainer | None:
        """The trainer for the reward."""
        if hasattr(self, "_trainer"):
            return self._trainer
        return None

    def set_trainer(self, trainer: Trainer):
        """Set the trainer for the reward."""
        self._trainer = trainer

    @abstractmethod
    def compute_reward(
        self,
        prompts: list[str] | list[list[dict[str, str]]],
        completions: list[str] | list[list[dict[str, str]]],
        **kwargs,
    ) -> list[float]:
        """Compute the reward for a set of prompts and completions.

        Args:
            prompts: A list of prompts. Can be a list of strings for completion models,
                or a list of lists of dictionaries for chat models.
            completions: A list of completions. Can be a list of strings for completion models,
                or a list of lists of dictionaries for chat models.
            **kwargs: Additional dataset column contents.

        Returns:
            A list of rewards for each prompt-completion pair.
            or
            A tuple containing a list of rewards and a dictionary of metrics to log.
        """
        ...


class MultiRewardSpec(RewardSpec):
    """A specification for a reward model that sums multiple rewards."""

    rewards: list[RewardSpec]
    """The rewards to compute."""

    weights: list[float] | None = None
    """The weights to use for each reward. If not provided, all rewards are weighted equally."""

    strategy: Literal["sum", "mean", "geometric_mean", "product"] = "sum"
    """The strategy to use to combine the rewards."""

    normalize_rewards: bool = False
    """Normalize each reward dimension in the batch."""

    normalize_weights: bool = True
    """Normalize the weights to sum to 1."""

    @model_validator(mode="after")
    def validate_weights(self) -> Self:
        """Validate the weights."""
        if self.weights is not None:
            if len(self.weights) != len(self.rewards):
                raise ValueError("The number of weights must match the number of rewards.")
            if self.strategy == "mean" and self.normalize_weights:
                logger.warning(
                    "Using mean strategy with normalized weights is not recommended. \
                    Use sum or set normalize_weights to False."
                )
        return self

    def __init__(self, **kwargs):
        """Initialize the MultiRewardSpec."""
        super().__init__(**kwargs)
        self._last_logged_step: int = -1

    def set_trainer(self, trainer: Trainer):
        """Set the trainer for the reward."""
        super().set_trainer(trainer)
        for reward in self.rewards:
            reward.set_trainer(trainer)

    def compute_reward(self, prompts: list[str], completions: list[str], **kwargs) -> list[float]:
        """Compute the reward for a set of prompts and completions."""
        rewards = np.array([reward.compute_reward(prompts, completions, **kwargs) for reward in self.rewards]).T

        rewards_mean_logging = rewards.mean(axis=0)

        if self.trainer is not None and self.trainer.is_world_process_zero():
            current_step = self.trainer.state.global_step
            if current_step > self._last_logged_step:
                self.trainer.log({f"rewards/{reward.name}": r for reward, r in zip(self.rewards, rewards_mean_logging)})
                self._last_logged_step = current_step

        if self.normalize_rewards:
            rewards = rewards / rewards.max(axis=0).clip(min=1e-4)[None, :]

        if self.weights is not None:
            weights = np.array(self.weights, dtype=rewards.dtype)
            if self.normalize_weights:
                # Normalize weights to sum to 1 to ensure consistent reward scaling
                weights = weights / weights.sum()
            rewards = rewards * weights[None, :]

        # Combine the rewards per prompt based on the chosen strategy.
        if self.strategy == "sum":
            combined_rewards = np.sum(rewards, axis=1)
        elif self.strategy == "mean":
            combined_rewards = np.mean(rewards, axis=1)
        elif self.strategy == "geometric_mean":
            # Geometric mean requires positive inputs. Since standardized rewards can be negative,
            # shift the rewards per prompt to be all positive before computing the geometric mean.
            min_rewards = np.min(rewards, axis=1, keepdims=True)
            shifted_rewards = rewards - min_rewards + 1e-4  # 1e-4 ensures positivity.
            combined_rewards = np.exp(np.mean(np.log(shifted_rewards), axis=1))
        elif self.strategy == "product":
            combined_rewards = np.log(rewards.clip(min=1e-3)).sum(axis=1)
        else:
            raise ValueError(f"Invalid strategy: {self.strategy}")

        # Ensure that a reward is returned for every prompt.
        assert len(combined_rewards) == len(prompts)
        return combined_rewards.tolist()
