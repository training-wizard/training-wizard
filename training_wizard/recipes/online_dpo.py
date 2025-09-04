"""Train a model using Online Direct Preference Optimization (Online DPO) or Cross-Preference Optimization (XPO).

Online DPO is an efficient method for aligning language models using online AI feedback. Unlike offline DPO,
which uses a fixed preference dataset, Online DPO generates and compares responses during training using
either a reward model or judge to provide feedback. This allows the model to learn from its own outputs
and adapt its behavior in real-time.

XPO extends DPO by introducing an alpha parameter that controls the trade-off between conservative updates
(alpha → 0) and aggressive updates (alpha → ∞). This recipe also supports entropy regularization for XPO
to encourage more diverse outputs.

Key benefits:
- Online feedback provides more relevant training signal as the model evolves
- Can use either a reward model or judge for comparing responses
- Supports both standard and conversational datasets
- Only requires prompts (no reference responses needed)
- Optional XPO mode with configurable alpha and entropy regularization

See:
- https://huggingface.co/docs/trl/main/online_dpo_trainer
- https://huggingface.co/docs/trl/main/xpo_trainer
"""

import logging
from dataclasses import asdict
from functools import cached_property
from logging import Logger
from typing import Any

import structlog
import torch
from pydantic import Field, field_validator
from pydantic.dataclasses import dataclass as pydantic_dataclass
from transformers import AutoModelForCausalLM, PreTrainedModel, TrainerCallback
from trl import OnlineDPOConfig, OnlineDPOTrainer, XPOConfig, XPOTrainer

from ..components.dataset_validation import validate_prompt_only_dataset
from ..components.utilities import print_trainable_parameters
from ..specs.base import TrainingRecipe
from ..specs.dataset import DataSourceSpec
from ..specs.judge import JudgeSpec
from ..specs.model import TransformerSpec
from ..specs.spec import Spec
from ..specs.trainer import CallbacksSpec, MLFlowAndTrainingArgsMixin, WizardTrainingArgsDefaultsMixin

logging.getLogger("transformers.generation.utils").addFilter(
    lambda record: "Setting `pad_token_id` to `eos_token_id`" not in record.getMessage()
)

logger: Logger = structlog.get_logger()


class XPOTrainerWithEntropy(XPOTrainer):
    """XPOTrainer with entropy regularization."""

    def __init__(self, *args, entropy_regularization: float | None = None, **kwargs):
        """Initialize the XPOTrainer with entropy regularization."""
        super().__init__(*args, **kwargs)
        self.entropy_regularization = entropy_regularization

    def _compute_losses(
        self,
        model_logprobs_model_data: Any,
        model_logprobs_ref_data: Any,
        ref_logprobs_ref_data: Any,
        ref_logprobs_model_data: Any,
        chosen_mask: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the losses with entropy regularization."""
        loss, dpo_losses, xpo_losses = super()._compute_losses(
            model_logprobs_model_data,
            model_logprobs_ref_data,
            ref_logprobs_ref_data,
            ref_logprobs_model_data,
            chosen_mask,
        )

        # Add entropy term to loss (negative because we want to maximize entropy)
        if self.entropy_regularization is not None:
            # Calculate entropy: H(X) = -∑π(x)log(π(x))
            probs = torch.exp(model_logprobs_model_data)  # π(x)
            entropy = -(probs * model_logprobs_model_data).sum(dim=-1).mean()  # -∑π(x)log(π(x))
            loss -= self.entropy_regularization * entropy

        return loss, dpo_losses, xpo_losses


class XPOExtras(Spec):
    """Extra arguments for XPOTrainer."""

    xpo_alpha: float | list[float] | None = None
    """Alpha parameter for XPOTrainer. Controls the trade-off between conservative (→0) and aggressive (→∞) updates."""

    entropy_regularization: float | None = None
    """Weight for entropy regularization. Higher values encourage more diverse outputs."""


@pydantic_dataclass
class OnlineDPOConfigSpec(WizardTrainingArgsDefaultsMixin, OnlineDPOConfig):
    """Specification for the OnlineDPOConfig.

    We don't use eval sets for online dpo/xpo, so some defaults are adjusted.
    """

    beta: float | list[float] = 0.1
    """Fixes the broken type hint upstream."""

    load_best_model_at_end: bool = False
    """Whether to load the best model at the end of training."""

    eval_strategy: str = "no"
    """The evaluation strategy to use."""

    save_strategy: str = "no"
    """The saving strategy to use."""


class OnlineDPOSpec(TrainingRecipe, MLFlowAndTrainingArgsMixin):
    """Recipe for training models using Online DPO or XPO.

    This recipe implements Online DPO training which uses online feedback to align language models.
    It can optionally use XPO mode for more controlled preference optimization with entropy regularization.

    Requirements:
    - A base model to be trained
    - A dataset containing prompts (see Dataset Format below)
    - Either a reward model or judge for comparing model outputs
    - Training arguments specifying hyperparameters
    - Optional XPO configuration for advanced preference optimization

    Notes:
    - Online DPO only requires prompts - responses are generated during training
    - For conversational format, the last message must be from the user
    - The dataset should not include model responses as these are generated during training
    """

    transformer_spec: TransformerSpec
    """The specification for the transformer model to use."""

    judge_spec: JudgeSpec
    """The specification for the judge model to use."""

    dataset_spec: DataSourceSpec
    """The dataset to use for training."""

    training_args_spec: OnlineDPOConfigSpec
    """The training arguments."""

    mlflow_experiment_name: str | None = None
    """The name of the MLflow experiment to use."""

    xpo: XPOExtras | None = None
    """If set, will use XPO instead of Online DPO."""

    callbacks_spec: CallbacksSpec = Field(default_factory=CallbacksSpec)
    """Callbacks to use during training."""

    filter_length: bool = False
    """Filter inputs that are longer than 0.9 * max response length."""

    @field_validator("dataset_spec")
    @classmethod
    def validate_dataset(cls, v: DataSourceSpec) -> Any:
        """Validate the dataset."""
        validate_prompt_only_dataset(v.dataset)
        return v

    @cached_property
    def callbacks(self) -> list[TrainerCallback]:
        """The callbacks to use during training."""
        logger.debug("Building callbacks")
        return self.callbacks_spec.create_callbacks(
            mlflow_experiment_name=self.mlflow_experiment_name,
        )

    @cached_property
    def model(self) -> PreTrainedModel:
        """The model to use for training."""
        return self.transformer_spec.create_model(AutoModelForCausalLM)

    def filter_too_long(self, row: dict) -> bool:
        """Filter a row if it's longer than the max response length."""
        text = row["prompt"][-1]["content"] if isinstance(row["prompt"], list) else row["prompt"]
        return len(self.transformer_spec.tokenizer.tokenize(text)) <= 0.9 * self.training_args_spec.max_new_tokens

    @cached_property
    def trainer(self) -> OnlineDPOTrainer:
        """The trainer to use for training."""
        train_ds = self.dataset_spec.dataset.shuffle()

        if self.filter_length:
            train_len_before = len(train_ds)
            train_ds = train_ds.filter(self.filter_too_long, desc="Filtering training dataset")
            logger.info(f"Filtered {train_len_before - len(train_ds)} training examples due to length")

        online_dpo_args = self.training_args_spec
        if self.xpo is not None:
            logger.info("XPO mode enabled")
            xpo_args = XPOConfig(
                **{k: v for k, v in asdict(online_dpo_args).items() if not k.startswith("_")},
                alpha=self.xpo.xpo_alpha,  # type: ignore
            )
            trainer = XPOTrainerWithEntropy(
                model=self.model,
                processing_class=self.transformer_spec.tokenizer,
                judge=self.judge_spec,
                train_dataset=train_ds,
                args=xpo_args,
                callbacks=self.callbacks,
                entropy_regularization=self.xpo.entropy_regularization,
            )
        else:
            trainer = OnlineDPOTrainer(
                model=self.model,
                processing_class=self.transformer_spec.tokenizer,
                judge=self.judge_spec,
                train_dataset=train_ds,
                args=online_dpo_args,
                callbacks=self.callbacks,
            )
        return trainer

    def main(self):
        """Run the training procedure of the training recipe."""
        assert self.trainer.model is not None, "Trainer model is None"
        print_trainable_parameters(self.trainer.model)
        logger.info("Training...")
        self.trainer.train()
