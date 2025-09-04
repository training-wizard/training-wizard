"""System modules for the Training Wizard.

Think of modules like a wrapper around models, with additional info on how to interact with them.
"""

from abc import ABC, abstractmethod
from functools import cached_property
from typing import Any

import torch
from datasets import Dataset
from matplotlib.figure import Figure
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from transformers import PreTrainedModel, PreTrainedTokenizer, Trainer, TrainerCallback

from ..spec import Spec


class WizardModule(Spec, ABC):
    """A system module for the Wizard."""

    @property
    @abstractmethod
    def callbacks(self) -> list[TrainerCallback]:
        """The callbacks to use during training."""
        ...

    @property
    def optimizer(self) -> Optimizer | None:
        """Custom optimizer for this module. Return None to use the default."""
        return None

    @property
    def lr_scheduler(self) -> LRScheduler | None:
        """Custom learning rate scheduler for this module. Return None to use the default."""
        return None

    @abstractmethod
    def validate_dataset(self, ds: Dataset):
        """Validate the dataset format for this module."""
        ...

    @abstractmethod
    def data_collator(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        """Prepare a batch for training."""
        ...

    @cached_property
    @abstractmethod
    def model(self) -> PreTrainedModel:
        """The model used in this module."""
        ...

    @cached_property
    @abstractmethod
    def tokenizer(self) -> PreTrainedTokenizer:
        """The tokenizer used in this module.

        For encoding, use the method in `transformer_spec` instead.
        """
        ...

    @torch.no_grad()
    @abstractmethod
    def batch_metrics(self, inputs: dict[str, Any], model_outputs: Any) -> dict[str, float | Figure]:
        """Evaluate a set of data and return some metrics or figures to log."""
        ...

    @torch.no_grad()
    @abstractmethod
    def eval_metrics(self, eval_dataset: Dataset, batch_size: int) -> dict[str, float | Figure]:
        """Evaluate a dataset and return some metrics to log."""
        ...

    @abstractmethod
    def compute_loss(
        self, trainer: Trainer, inputs: dict[str, Any], return_outputs: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        """Compute the loss for the model."""
        ...
