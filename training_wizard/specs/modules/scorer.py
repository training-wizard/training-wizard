"""Sequence classification modules."""

from functools import cached_property
from logging import Logger
from typing import Any, Literal, cast

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

from ...components.dataset_validation import validate_regression_dataset
from ...components.helper import eval_mode, standard_prepare_batch
from ...components.metrics import regression_metrics
from ..model import TransformerSpec
from .module import WizardModule

logger: Logger = structlog.get_logger()


class SequenceRegressionModule(WizardModule):
    """Sequence regression module."""

    transformer_spec: TransformerSpec
    """The transformer model to use for classification."""

    loss_type: Literal["mse", "smooth_l1"] = "smooth_l1"
    """The type of loss to use for training. One of:
    - "smooth_l1": Smooth L1 Loss
    - "mse": Mean Squared Error Loss
    """

    activation: Literal["linear", "sigmoid"] = "sigmoid"
    """The activation function to use for the output layer, depending on if your scores are normalized or not."""

    @cached_property
    def callbacks(self) -> list[TrainerCallback]:
        """Additional callbacks to use during training."""
        return []

    def validate_dataset(self, ds: Dataset):
        """Validate the dataset."""
        validate_regression_dataset(ds)

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
        result = standard_prepare_batch(self.transformer_spec.encode_batch, batch)
        if any("sample_weight" in row for row in batch):
            sample_weights = [row.get("sample_weight", 1.0) for row in batch]
            result["sample_weights"] = torch.Tensor(sample_weights)
        return result

    def _process_logits(self, model_outputs: Any) -> torch.Tensor:
        """Get the results from the model outputs."""
        logits: torch.Tensor = model_outputs.logits
        if self.activation == "sigmoid":
            return logits.sigmoid()
        elif self.activation == "linear":
            return logits
        else:
            raise ValueError(f"Invalid activation function: {self.activation}")

    @torch.no_grad()
    def batch_metrics(self, inputs: dict[str, Any], model_outputs: Any) -> dict[str, float | Figure]:
        """Evaluate a set of data and return some metrics or figures to log."""
        y_true: torch.Tensor = inputs["labels"]
        y_pred: torch.Tensor = self._process_logits(model_outputs)
        result = regression_metrics(y_true.flatten(), y_pred.flatten(), plots=False)
        return result

    @torch.no_grad()
    def eval_metrics(self, eval_dataset: Dataset, batch_size: int = 8) -> dict[str, float | Figure]:
        """Evaluate a dataset and return some metrics to log."""
        preds = [
            self.predict(batch)  # type: ignore
            for batch in chunked_even(tqdm(eval_dataset, desc="Evaluating wizard dataset metrics..."), batch_size)
        ]
        preds_np = np.concatenate(preds, axis=0)
        y_pred = torch.Tensor(preds_np)
        y_true = torch.Tensor(eval_dataset["labels"])
        return regression_metrics(y_true.flatten(), y_pred.flatten(), plots=True)

    @torch.no_grad()
    def predict(self, batch: list[dict[str, Any]]) -> np.ndarray:
        """Predict on a batch of data."""
        batch_encoded = self.data_collator(batch)
        batch_encoded = {
            k: v.to(self.model.device) for k, v in batch_encoded.items() if k in {"input_ids", "attention_mask"}
        }
        with eval_mode(self.model):
            model_outputs = self.model(**batch_encoded)
        y_pred = self._process_logits(model_outputs)
        return y_pred.cpu().numpy()

    def compute_loss(
        self, trainer: Trainer, inputs: dict[str, Any], return_outputs: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        """Compute the loss for the model."""
        assert trainer.model is not None, "Trainer model is None"

        labels: torch.Tensor = inputs.pop("labels")
        sample_weights: torch.Tensor = inputs.pop("sample_weights", torch.ones_like(labels, dtype=torch.float32))
        outputs = trainer.model(**inputs)
        labels = labels.to(cast("str", trainer.model.device), dtype=outputs.logits.dtype)
        scores_pred: torch.Tensor = self._process_logits(outputs)

        if self.loss_type == "smooth_l1":
            loss = torch.nn.functional.smooth_l1_loss(
                scores_pred.flatten(), labels.flatten(), beta=0.1, reduction="none"
            )
        elif self.loss_type == "mse":
            loss = torch.nn.functional.mse_loss(scores_pred.flatten(), labels.flatten(), reduction="none")
        loss = (loss * sample_weights).sum() / sample_weights.sum()

        return (loss, outputs) if return_outputs else loss
