"""Trainer related specs."""

from dataclasses import field
from logging import Logger
from pathlib import Path
from typing import Any, Self

import structlog
from accelerate.utils import set_seed as accelerate_set_seed
from pydantic import BaseModel, field_validator, model_validator
from pydantic.dataclasses import dataclass as pydantic_dataclass
from transformers import TrainingArguments
from transformers.integrations import MLflowCallback
from transformers.trainer_callback import (
    EarlyStoppingCallback,
    TrainerCallback,
)

from ..components.callbacks import (
    EvaluateFirstStepCallback,
    SaveInfoBegin,
    SaveInfoEnd,
)
from ..components.mlflow_auth import set_mlflow_environment_variables

logger: Logger = structlog.get_logger()


class CallbacksSpec(BaseModel):
    """Specification for which callbacks to use. Enables no callbacks by default."""

    enable_early_stopping: bool = False
    """Enable early stopping based on the validation loss."""

    early_stopping_patience: int = -1
    """Patience for early stopping."""

    evaluate_first_step: bool = False
    """Run evalulation after the first step."""

    # Added new flag for LoRA merging
    merge_lora: bool = False
    """If True, merge LoRA weights into the base model before saving in SaveInfoEnd callback."""

    @model_validator(mode="after")
    def validate_stopping_patience(self) -> Any:
        """Validate the early stopping patience."""
        assert not self.enable_early_stopping or self.early_stopping_patience >= 0, (
            "early_stopping_patience must be >= 0 if enable_early_stopping is True."
        )
        return self

    def create_callbacks(
        self,
        mlflow_experiment_name: str | None = None,
    ) -> list[TrainerCallback]:
        """The callbacks to use during training."""
        logger.debug("Building callbacks")

        # Default callbacks
        result: list[TrainerCallback] = []

        # The order must be: SaveInfoEnd, MLflowCallback, SaveInfoBegin
        # Otherwise run_name won't be applied in the run
        result.append(SaveInfoEnd(mlflow_experiment_name=mlflow_experiment_name, merge_lora=self.merge_lora))
        if mlflow_experiment_name:
            logger.info('MLflow callback enabled. Name "%s"', mlflow_experiment_name)
            result.append(MLflowCallback())
        result.append(SaveInfoBegin(mlflow_experiment_name=mlflow_experiment_name))

        if self.enable_early_stopping:
            logger.info("Early stopping callback enabled. Patience: %d", self.early_stopping_patience)
            result.append(EarlyStoppingCallback(self.early_stopping_patience))

        if self.evaluate_first_step:
            logger.info("First-step evaluation callback enabled.")
            result.append(EvaluateFirstStepCallback())

        return result


def validate_training_args(mlflow_experiment_name: str | None, arguments: TrainingArguments):
    """Validate the training arguments."""
    if mlflow_experiment_name not in ["", None] and arguments.run_name is None:
        raise ValueError(
            """MLflow experiment name is set, which enables MLflow tracking, but no run name is given.
            Please provide a value for the `run_name` field."""
        )
    if arguments.output_dir is not None:
        v_path = Path(arguments.output_dir)
        if v_path.exists() and v_path.is_dir():
            contents = list(v_path.iterdir())
            if len(contents) == 0:
                pass
            elif len(contents) == 1 and contents[0].name == "reproducibility" and contents[0].is_dir():
                logger.warning("Found existing reproducibility directory in output path. Deleting it.")
                import shutil

                shutil.rmtree(contents[0])
            else:
                raise ValueError(f"Output directory {arguments.output_dir} already exists.")
        elif v_path.exists():
            raise ValueError(f"Output directory {arguments.output_dir} already exists and is not a directory.")


@pydantic_dataclass
class WizardTrainingArgsDefaultsMixin:
    """A mixin that adds default training arguments.

    NOTE: It's very important that you add this as your **first** inherited parent class!
    """

    report_to: list[str] = field(default_factory=list)
    """List of integrations to report to."""

    overwrite_output_dir: bool = False
    """Whether to overwrite the output directory if it exists."""

    remove_unused_columns: bool = False
    """Whether to remove unused columns from the dataset."""

    dataloader_drop_last: bool = False
    """Whether to drop the last incomplete batch."""

    load_best_model_at_end: bool = True
    """Whether to load the best model at the end of training."""

    eval_strategy: str = "epoch"
    """The evaluation strategy to use."""

    save_strategy: str = "epoch"
    """The saving strategy to use."""


class MLFlowAndTrainingArgsMixin(BaseModel):
    """A mixin that adds functionality for MLFlow and training arguments."""

    mlflow_experiment_name: str | None = None
    """The name of the MLflow experiment to use."""

    training_args_spec: Any
    """The training arguments."""

    @field_validator("mlflow_experiment_name")
    @classmethod
    def validate_mlflow_experiment_setup(cls, v: Any) -> Any:
        """Validate the MLflow experiment name."""
        if v is None:
            return None
        set_mlflow_environment_variables(v)
        return v

    @model_validator(mode="before")
    @classmethod
    def validate_mlflow_experiment_name_in_training_args_spec(cls, v: Any) -> Any:
        """Validate the MLflow experiment name."""
        if isinstance(v, dict) and "mlflow_experiment_name" in v.get("training_args_spec", {}):
            logger.warning(
                "mlflow_experiment_name in training_args_spec is deprecated. "
                "Specify mlflow_experiment_name at the recipe level instead."
            )
            v["mlflow_experiment_name"] = v["training_args_spec"].pop("mlflow_experiment_name")
        return v

    @model_validator(mode="after")
    def validate_valid_training_args(self) -> Self:
        """Validate the training arguments."""
        validate_training_args(self.mlflow_experiment_name, self.training_args_spec)
        return self

    @field_validator("training_args_spec")
    @classmethod
    def set_seed(cls, args: TrainingArguments) -> TrainingArguments:
        """Set a reproducible seed for this (distributed) process."""
        logger.info(f"Setting local process seed to {args.seed}")
        accelerate_set_seed(args.seed)
        return args
