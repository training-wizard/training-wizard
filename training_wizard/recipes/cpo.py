"""See https://huggingface.co/docs/trl/main/cpo_trainer."""

import logging
from functools import cached_property
from logging import Logger
from typing import Any

import structlog
from pydantic import Field, field_validator
from pydantic.dataclasses import dataclass as pydantic_dataclass
from transformers import AutoModelForCausalLM, PreTrainedModel, TrainerCallback
from trl import CPOConfig, CPOTrainer

from ..components.dataset_validation import validate_preference_dataset
from ..components.helper import compute_split
from ..components.utilities import print_trainable_parameters
from ..specs.base import TrainingRecipe
from ..specs.dataset import DataSourceSpec
from ..specs.model import TransformerSpec
from ..specs.trainer import CallbacksSpec, MLFlowAndTrainingArgsMixin, WizardTrainingArgsDefaultsMixin

logging.getLogger("transformers.generation.utils").addFilter(
    lambda record: "Setting `pad_token_id` to `eos_token_id`" not in record.getMessage()
)

logger: Logger = structlog.get_logger()


@pydantic_dataclass
class CPOConfigSpec(WizardTrainingArgsDefaultsMixin, CPOConfig):
    """Specification for the CPOConfig."""


class CPOTrainingSpec(TrainingRecipe, MLFlowAndTrainingArgsMixin):
    """Train a model using Contrastive Preference Optimization (CPO).

    CPO is a method for training language models using preference data, introduced as an alternative
    to Direct Preference Optimization (DPO). It aims to mitigate two key limitations of supervised
    fine-tuning (SFT):
    1. SFT's performance is capped by training data quality
    2. SFT lacks explicit mechanisms to avoid generating suboptimal outputs

    Key features:
    - Supports both standard and conversational preference datasets
    - Offers multiple loss functions including sigmoid (default), hinge, IPO, and SimPO
    - Includes BC regularization controlled by cpo_alpha parameter
    - Compatible with Mixture of Experts (MoE) models through auxiliary loss

    See: https://huggingface.co/docs/trl/main/cpo_trainer
    """

    mlflow_experiment_name: str | None = None
    """The name of the MLflow experiment to use."""

    transformer_spec: TransformerSpec
    """The specification for the transformer model to use."""

    training_args_spec: CPOConfigSpec
    """The training arguments."""

    dataset_spec: DataSourceSpec
    """The dataset to use for training."""

    validation: float | int | DataSourceSpec | None = None
    """Dataset to use for validation:

    - If float, it's the fraction of the training dataset to exclude for validation.
    - If int, it's the number of examples from the training dataset to exclude for validation.
    - If DataSourceSpec, it's the dataset to use for validation.
    - If None, will not use a validation dataset.
    """

    callbacks_spec: CallbacksSpec = Field(default_factory=CallbacksSpec)
    """Callbacks to use during training."""

    @field_validator("dataset_spec", "validation")
    @classmethod
    def validate_dataset(cls, v: Any) -> Any:
        """Validate the dataset."""
        if isinstance(v, DataSourceSpec):
            validate_preference_dataset(v.dataset)
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

    @cached_property
    def trainer(self) -> CPOTrainer:
        """The trainer to use for training."""
        train_ds, test_ds = compute_split(self.dataset_spec.dataset, self.validation, seed=self.training_args_spec.seed)

        trainer = CPOTrainer(
            model=self.model,
            processing_class=self.transformer_spec.tokenizer,
            train_dataset=train_ds,
            eval_dataset=test_ds,
            args=self.training_args_spec,
            callbacks=self.callbacks,
        )
        return trainer

    def main(self):
        """Run the training procedure of the training recipe."""
        assert self.trainer.model is not None, "Trainer model is None"
        print_trainable_parameters(self.trainer.model)
        logger.info("Training...")
        self.trainer.train()
