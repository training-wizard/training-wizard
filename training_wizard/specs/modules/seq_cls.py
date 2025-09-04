"""Sequence classification modules."""

from functools import cached_property
from logging import Logger
from typing import Any, Literal

import numpy as np
import structlog
import torch
from datasets import Dataset
from matplotlib.figure import Figure
from more_itertools import chunked_even
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    PreTrainedModel,
    PreTrainedTokenizer,
    Trainer,
    TrainerCallback,
)

from ...components.dataset_validation import validate_sequence_classification_dataset
from ...components.helper import eval_mode, standard_prepare_batch
from ...components.metrics import binary_classification_metrics, multiclass_classification_metrics
from ..model import TransformerSpec
from .module import WizardModule

logger: Logger = structlog.get_logger()


class SequenceClassifierModule(WizardModule):
    """Sequence classification module. Can do either binary or multi-class classification."""

    transformer_spec: TransformerSpec
    """The transformer model to use for classification."""

    loss_type: Literal["cross_entropy", "binary_cross_entropy"] = "cross_entropy"
    """The type of loss to use for training. One of:
    - "cross_entropy": Cross Entropy Loss (default)
    - "binary_cross_entropy": Binary Cross Entropy Loss
    """

    label_smoothing: float = 0.0
    """The label smoothing to use for training."""

    compute_eval_metrics: bool = True
    """Whether to compute evaluation metrics."""

    save_plots: bool = False
    """If true, generates metric plots during evaluation."""

    @cached_property
    def callbacks(self) -> list[TrainerCallback]:
        """Additional callbacks to use during training."""
        return []

    def validate_dataset(self, ds: Dataset):
        """Validate the dataset."""
        validate_sequence_classification_dataset(ds)

    @cached_property
    def model(self) -> PreTrainedModel:
        """The model used in this module."""
        return self.transformer_spec.create_model(AutoModelForSequenceClassification)

    @cached_property
    def tokenizer(self) -> PreTrainedTokenizer:
        """The tokenizer used in this module.

        For encoding, use the method in `transformer_spec` instead.
        """
        return self.transformer_spec.tokenizer

    def data_collator(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        """Data collator for training.

        A data collator is a function that takes a list of samples from a dataset and collates them into a batch.
        For example, it can pad sequences to the same length, or convert them to tensors.
        In more advanced cases, it can also apply transformations to the data.
        """
        d = standard_prepare_batch(self.transformer_spec.encode_batch, batch)
        d["labels"] = d["labels"].to(dtype=torch.long)
        return d

    def _metrics(self, y_true: torch.Tensor, y_pred: torch.Tensor, plots: bool = False) -> dict[str, float | Figure]:
        """Compute metrics for a classification task."""
        is_binary = y_pred.squeeze().ndim == 1 or self.loss_type == "binary_cross_entropy"
        if is_binary:
            return binary_classification_metrics(y_true.flatten(), y_pred.flatten(), plots=plots)
        else:
            return multiclass_classification_metrics(
                y_true, y_pred, num_classes=self.model.config.num_labels, plots=plots
            )

    @torch.no_grad()
    def batch_metrics(self, inputs: dict[str, Any], model_outputs: Any) -> dict[str, float | Figure]:
        """Evaluate a set of data and return some metrics or figures to log."""
        y_true: torch.Tensor = inputs["labels"]
        y_pred: torch.Tensor = model_outputs.logits
        return self._metrics(y_true, y_pred, plots=False)

    @torch.no_grad()
    def eval_metrics(self, eval_dataset: Dataset, batch_size: int = 8) -> dict[str, float | Figure]:
        """Evaluate a dataset and return some metrics to log."""
        if not self.compute_eval_metrics:
            return {}
        preds = [
            self.predict(batch)  # type: ignore
            for batch in chunked_even(tqdm(eval_dataset, desc="Evaluating wizard dataset metrics..."), batch_size)
        ]
        preds_np = np.concatenate(preds, axis=0)
        y_pred = torch.Tensor(preds_np)
        y_true = torch.LongTensor(eval_dataset["labels"])
        return self._metrics(y_true, y_pred, plots=self.save_plots)

    @torch.no_grad()
    def predict(self, batch: list[dict[str, Any]]) -> np.ndarray:
        """Predict on a batch of data."""
        batch_encoded = self.data_collator(batch)
        batch_encoded.pop("labels")
        batch_encoded = {k: v.to(self.model.device) for k, v in batch_encoded.items()}
        with eval_mode(self.model):
            model_outputs = self.model(**batch_encoded)
        logits: torch.Tensor = model_outputs.logits.squeeze()
        if logits.ndim == 1 or self.loss_type == "binary_cross_entropy":
            return logits.sigmoid().to(dtype=torch.float32, device="cpu").numpy()
        else:
            return logits.softmax(dim=-1).to(dtype=torch.float32, device="cpu").numpy()

    def compute_loss(
        self, trainer: Trainer, inputs: dict[str, Any], return_outputs: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        """Compute the loss for the model."""
        assert trainer.model is not None, "Trainer model is None"

        labels: torch.Tensor = inputs.pop("labels").to(trainer.model.device)
        outputs = trainer.model(**inputs)

        if len(labels.shape) == 1 and labels.dtype != torch.long:
            labels = labels.unsqueeze(1)

        if self.loss_type == "cross_entropy":
            loss = torch.nn.functional.cross_entropy(outputs.logits, labels, label_smoothing=self.label_smoothing)
        elif self.loss_type == "binary_cross_entropy":
            labels = labels.squeeze().float()
            if self.label_smoothing > 0:
                labels = labels * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
            loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs.logits.squeeze(), labels)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        return (loss, outputs) if return_outputs else loss
