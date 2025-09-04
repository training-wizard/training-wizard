"""Train a model for many tasks..."""

from functools import cached_property
from logging import Logger
from typing import Self

import structlog
from pydantic import Field, model_validator
from pydantic.dataclasses import dataclass as pydantic_dataclass
from transformers import TrainerCallback, TrainingArguments

from ..components.helper import compute_split
from ..components.utilities import print_trainable_parameters
from ..components.wizard_trainer import WizardTrainer
from ..specs.base import TrainingRecipe
from ..specs.dataset import DataSourceSpec, SimpleDataSourceSpec
from ..specs.modules.module import WizardModule
from ..specs.trainer import CallbacksSpec, MLFlowAndTrainingArgsMixin, WizardTrainingArgsDefaultsMixin

logger: Logger = structlog.get_logger()


@pydantic_dataclass
class SimpleArgumentsSpec(WizardTrainingArgsDefaultsMixin, TrainingArguments):
    """Specification for the SimpleArguments."""


class ModularTrainingSpec(TrainingRecipe, MLFlowAndTrainingArgsMixin):
    """Recipe for training many different modes."""

    wizard_module: WizardModule
    """The module to use for training."""

    training_args_spec: SimpleArgumentsSpec
    """The training arguments."""

    dataset_spec: SimpleDataSourceSpec | DataSourceSpec
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

    @model_validator(mode="after")
    def validate_dataset(self) -> Self:
        """Validate the dataset."""
        self.wizard_module.validate_dataset(self.dataset_spec.dataset)
        if isinstance(self.validation, DataSourceSpec):
            self.wizard_module.validate_dataset(self.validation.dataset)
        return self

    @cached_property
    def callbacks(self) -> list[TrainerCallback]:
        """The callbacks to use during training."""
        logger.debug("Building callbacks")
        return (
            self.callbacks_spec.create_callbacks(
                mlflow_experiment_name=self.mlflow_experiment_name,
            )
            + self.wizard_module.callbacks
        )

    @cached_property
    def trainer(self) -> WizardTrainer:
        """The trainer to use for training."""
        train_ds, val_ds = compute_split(self.dataset_spec.dataset, self.validation, seed=self.training_args_spec.seed)
        return WizardTrainer(
            args=self.training_args_spec,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            training_module=self.wizard_module,
            callbacks=self.callbacks,
        )

    def main(self):
        """Run the training procedure of the training recipe."""
        assert self.trainer.model is not None, "Trainer model is None"
        print_trainable_parameters(self.trainer.model)
        logger.info("Training...")
        self.trainer.train()
