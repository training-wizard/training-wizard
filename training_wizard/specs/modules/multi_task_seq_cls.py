"""Sequence classification modules."""

from functools import cached_property
from logging import Logger
from typing import Any, cast

import numpy as np
import structlog
import torch
from datasets import Dataset
from more_itertools import chunked_even, split_into
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
from ...components.metrics import multiclass_classification_metrics
from ..model import TransformerSpec
from .module import WizardModule

logger: Logger = structlog.get_logger()


class MultiTaskSequenceClassifierModule(WizardModule):
    """Sequence classification module. Can do either binary or multi-class classification."""

    transformer_spec: TransformerSpec
    """The transformer model to use for classification."""

    group_sizes: list[int]
    """The number of classes in each group."""

    label_smoothing: float = 0.0
    """The label smoothing to use for training."""

    compute_eval_metrics: bool = True
    """Whether to compute evaluation metrics."""

    show_per_group_metrics: bool = False
    """Whether to show per-group metrics."""

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
        if "weight" in batch[0]:  # optional per-instance weight(s)
            d["weight"] = torch.tensor([row["weight"] for row in batch], dtype=torch.float32)
        return d

    def _split_into_groups(self, batch_logits: torch.Tensor) -> list[torch.Tensor]:
        """Split a logits or one-hot encoded tensor into groups."""
        total_size = sum(self.group_sizes)
        assert batch_logits.shape[-1] == total_size, "The model num_labels must match the sum of group_sizes"
        group_indices = list(split_into(range(total_size), self.group_sizes))
        return [batch_logits[:, group] for group in group_indices]

    def _metrics(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> dict[str, float]:
        """Compute metrics for a classification task."""
        metrics_all = [
            multiclass_classification_metrics(
                y_true[:, i],  # per-group integer labels
                y_pred_grp,  # per-group predicted probabilities/logits
                num_classes=group_size,
                plots=False,
            )
            for i, (y_pred_grp, group_size) in enumerate(zip(self._split_into_groups(y_pred), self.group_sizes))
        ]
        metrics = {}
        for k in metrics_all[0]:
            if all(k in m for m in metrics_all) and all(isinstance(m[k], float) for m in metrics_all):
                values = [cast("float", m[k]) for m in metrics_all]
                metrics[k] = np.mean(values)
                metrics[f"{k}_std"] = np.std(values)

        if self.show_per_group_metrics:
            for i, group_metrics in enumerate(metrics_all):
                for k in group_metrics:
                    metrics[f"group_{i}_{k}"] = group_metrics[k]
        return metrics

    @torch.no_grad()
    def batch_metrics(self, inputs: dict[str, Any], model_outputs: Any) -> dict[str, float]:
        """Evaluate a set of data and return some metrics or figures to log."""
        y_true: torch.Tensor = inputs["labels"]
        y_pred: torch.Tensor = model_outputs.logits
        return self._metrics(y_true, y_pred)

    @torch.no_grad()
    def eval_metrics(self, eval_dataset: Dataset, batch_size: int = 8) -> dict[str, float]:
        """Evaluate a dataset and return some metrics to log."""
        if not self.compute_eval_metrics:
            return {}
        preds = [
            self.predict(batch)  # type: ignore
            for batch in chunked_even(tqdm(eval_dataset, desc="Evaluating wizard dataset metrics..."), batch_size)
        ]
        preds = [np.concatenate(groups, axis=-1) for groups in preds]
        preds_np = np.concatenate(preds, axis=0)
        y_pred = torch.Tensor(preds_np)
        y_true = torch.LongTensor(eval_dataset["labels"])
        return self._metrics(y_true, y_pred)

    @torch.no_grad()
    def predict(self, batch: list[dict[str, Any]]) -> list[np.ndarray]:
        """Predict on a batch of data."""
        batch_encoded = self.data_collator(batch)
        batch_encoded.pop("labels", None)
        batch_encoded.pop("weight", None)
        batch_encoded = {k: v.to(self.model.device) for k, v in batch_encoded.items()}
        with eval_mode(self.model):
            model_outputs = self.model(**batch_encoded)
        logits: torch.Tensor = model_outputs.logits
        return [
            grp.softmax(dim=-1).to(dtype=torch.float32, device="cpu").numpy() for grp in self._split_into_groups(logits)
        ]

    def compute_loss(
        self, trainer: Trainer, inputs: dict[str, Any], return_outputs: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        """Compute the loss for the model."""
        assert trainer.model is not None, "Trainer model is None"

        labels: torch.Tensor = inputs.pop("labels").to(trainer.model.device)
        sample_weight: torch.Tensor | None = inputs.pop("weight", None)
        if sample_weight is not None:
            sample_weight = sample_weight
        outputs = trainer.model(**inputs)
        logits = outputs.logits

        losses = []
        for i, grp_logits in enumerate(self._split_into_groups(logits)):
            if sample_weight is None:
                loss_i = torch.nn.functional.cross_entropy(
                    grp_logits,
                    labels[:, i],
                    label_smoothing=self.label_smoothing,
                )
            else:
                per_sample_loss = torch.nn.functional.cross_entropy(
                    grp_logits,
                    labels[:, i],
                    reduction="none",
                    label_smoothing=self.label_smoothing,
                )
                grp_weight = sample_weight[:, i] if sample_weight.ndim == 2 else sample_weight
                loss_i = (per_sample_loss * grp_weight).sum() / grp_weight.sum().clamp_min(1e-8)
            losses.append(loss_i)

        loss = torch.mean(torch.stack(losses))
        return (loss, outputs) if return_outputs else loss
