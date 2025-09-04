"""Knowledge distillation module for instruction-tuned models."""

from functools import cached_property
from typing import Any, Literal

import torch
import torch.nn.functional as F
from datasets import Dataset
from matplotlib.figure import Figure
from transformers import (
    AutoModelForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizer,
    Trainer,
    TrainerCallback,
)

from ..model import TransformerSpec
from .module import WizardModule


class KnowledgeDistillationModule(WizardModule):
    """Knowledge distillation module for instruction-tuned models.

    Trains a student model to mimic a teacher model using Kullback-Leibler divergence
    loss between their output probability distributions, combined with the standard
    cross-entropy loss against ground-truth labels.
    """

    teacher_spec: TransformerSpec
    """TransformerSpec configured for the teacher."""

    student_module: WizardModule
    """Instruction tuning module configured for the student."""

    loss_type: Literal["kl_divergence"] = "kl_divergence"
    """Loss function to apply during distillation. Only KL divergence is supported."""

    temperature: float = 1.0
    """Temperature used when computing the distillation loss."""

    distillation_weight: float = 0.5
    """Weight (alpha) applied to the distillation loss (KL divergence).
    The standard cross-entropy loss weight will be (1.0 - distillation_weight).
    """

    @cached_property
    def callbacks(self) -> list[TrainerCallback]:
        """Callbacks for the distillation module."""
        return self.student_module.callbacks

    def validate_dataset(self, ds: Dataset):
        """Validate the dataset."""
        self.student_module.validate_dataset(ds)

    @cached_property
    def teacher(self) -> PreTrainedModel:
        """Get the teacher model instance."""
        model = self.teacher_spec.create_model(AutoModelForCausalLM)
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        return model

    @cached_property
    def teacher_tokenizer(self) -> PreTrainedTokenizer:
        """Get the teacher tokenizer instance."""
        return self.teacher_spec.tokenizer

    @cached_property
    def model(self) -> PreTrainedModel:  # type: ignore[override]
        """Get the student model instance."""
        return self.student_module.model

    @cached_property
    def tokenizer(self) -> PreTrainedTokenizer:  # type: ignore[override]
        """Get the student tokenizer instance."""
        return self.student_module.tokenizer

    def data_collator(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        """Prepare a batch for distillation including student labels."""
        return self.student_module.data_collator(batch)

    @torch.no_grad()
    def batch_metrics(self, inputs: dict[str, Any], model_outputs: Any) -> dict[str, float | Figure]:
        """Compute batch-level metrics using the student module."""
        return self.student_module.batch_metrics(inputs, model_outputs)

    @torch.no_grad()
    def eval_metrics(self, eval_dataset: Dataset, batch_size: int) -> dict[str, float | Figure]:
        """Compute evaluation metrics using the student module."""
        return self.student_module.eval_metrics(eval_dataset, batch_size)

    def _distillation_loss(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        """Compute the KL divergence distillation loss.

        Calculates the KL divergence between the softened probability distributions
        of the student and teacher models, after truncating them to the same sequence length.

        Args:
            student_logits: Logits output by the student model.
            teacher_logits: Logits output by the teacher model.

        Returns:
            The computed KL divergence loss tensor.
        """
        min_seq_len = min(student_logits.shape[1], teacher_logits.shape[1])
        student_logits_trunc = student_logits[:, :min_seq_len, :]
        teacher_logits_trunc = teacher_logits[:, :min_seq_len, :]

        student_log_probs = F.log_softmax(student_logits_trunc / self.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits_trunc / self.temperature, dim=-1)
        kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction="mean", log_target=False)
        scaled_kl_loss = kl_loss * (self.temperature**2)
        return scaled_kl_loss

    def compute_loss(
        self,
        trainer: Trainer,
        inputs: dict[str, Any],
        return_outputs: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        """Compute the combined distillation and standard loss."""
        assert trainer.model is not None, "Trainer model is None"

        student_outputs = trainer.model(**inputs)
        standard_loss = student_outputs.loss

        teacher_model = self.teacher
        teacher_model.to(student_outputs.logits.device)
        with torch.inference_mode():
            teacher_outputs = teacher_model(**inputs)

        distill_loss = self._distillation_loss(student_outputs.logits, teacher_outputs.logits)

        alpha = self.distillation_weight
        if standard_loss is None and alpha > 0.0:
            raise ValueError("Standard loss (student_outputs.loss) is None. Ensure labels are provided correctly.")
        combined_loss = alpha * distill_loss + (1.0 - alpha) * (standard_loss or 0.0)

        return (combined_loss, student_outputs) if return_outputs else combined_loss
