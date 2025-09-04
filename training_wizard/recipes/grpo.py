"""Train a model using Group Relative Policy Optimization (GRPO).

GRPO is a variant of PPO that optimizes language models using group-relative advantages.
It computes rewards relatively within a group of generated completions and incorporates
a KL-penalty with a reference policy to ensure stability while optimizing.
See https://huggingface.co/docs/trl/grpo_trainer for more details.
"""

from functools import cached_property
from logging import Logger
from typing import Any, cast

import structlog
from pydantic import Field, field_validator
from pydantic.dataclasses import dataclass as pydantic_dataclass
from transformers import AutoModelForCausalLM, PreTrainedModel, TrainerCallback
from trl import GRPOConfig, GRPOTrainer

from ..components.callbacks import print_output_sample
from ..components.dataset_validation import validate_prompt_only_dataset
from ..components.helper import compute_split
from ..components.python_utils import make_renamed_func
from ..components.utilities import print_trainable_parameters
from ..specs.base import TrainingRecipe
from ..specs.dataset import DataSourceSpec
from ..specs.model import TransformerSpec
from ..specs.reward import RewardSpec
from ..specs.trainer import CallbacksSpec, MLFlowAndTrainingArgsMixin, WizardTrainingArgsDefaultsMixin

logger: Logger = structlog.get_logger()


# Define a pydantic dataclass for GRPOConfig, with wizard defaults
@pydantic_dataclass
class GRPOConfigSpec(WizardTrainingArgsDefaultsMixin, GRPOConfig):
    """Specification for the GRPOConfig."""


class GRPOTrainingSpec(TrainingRecipe, MLFlowAndTrainingArgsMixin):
    """Recipe for training a model with Group Relative Policy Optimization (GRPO).

    This recipe builds on the standard TRL workflow:
      - Loads a transformer model specification.
      - Splits and validates the dataset.
      - Creates a GRPO trainer with the provided training arguments.
      - Starts the training procedure.
    """

    mlflow_experiment_name: str | None = None
    """The name of the MLflow experiment to use."""

    transformer_spec: TransformerSpec
    """The specification for the transformer model to use."""

    training_args_spec: GRPOConfigSpec
    """GRPO training arguments specification."""

    reward_specs: list[RewardSpec]
    """Custom reward specifications. Users should provide subclasss of RewardSpec that implement compute_reward."""

    dataset_spec: DataSourceSpec
    """The training dataset specification containing preference data."""

    validation: float | int | DataSourceSpec | None = None
    """Dataset to use for validation:

    - If float, it's the fraction of the training dataset to exclude for validation.
    - If int, it's the number of examples from the training dataset to exclude for validation.
    - If DataSourceSpec, it's the dataset to use for validation.
    - If None, will not use a validation dataset.
    """

    callbacks_spec: CallbacksSpec = Field(default_factory=CallbacksSpec)
    """Callbacks to use during training."""

    sample_rate: int = 100
    """Print a sample of outputs every `sample_rate` steps. Set to 0 to disable."""

    @field_validator("dataset_spec")
    @classmethod
    def validate_dataset(cls, v: Any) -> Any:
        """Validate that the dataset conforms to preference dataset standards."""
        if isinstance(v, DataSourceSpec):
            validate_prompt_only_dataset(v.dataset)
        return v

    @cached_property
    def callbacks(self) -> list[TrainerCallback]:
        """Instantiate callbacks for training."""
        logger.debug("Building callbacks")
        return self.callbacks_spec.create_callbacks(
            mlflow_experiment_name=self.mlflow_experiment_name,
        )

    @cached_property
    def model(self) -> PreTrainedModel:
        """Load the transformer model."""
        return self.transformer_spec.create_model(AutoModelForCausalLM)

    def fake_sample_reward(
        self,
        prompts: list[str] | list[list[dict[str, str]]],
        completions: list[str] | list[list[dict[str, str]]],
        **kwargs: Any,
    ) -> list[float]:
        """Just for sampling purposes. Kind of a hack."""
        if (
            self.sample_rate > 0
            and self.trainer.is_world_process_zero()
            and self.trainer.state.global_step % self.sample_rate == 0
        ):
            if isinstance(prompts[0], list):
                prompts = cast("list[list[dict[str, str]]]", prompts)
                completions = cast("list[list[dict[str, str]]]", completions)
                print_output_sample(
                    [messages[-1]["content"] for messages in prompts],
                    [messages[-1]["content"] for messages in completions],
                )
            else:
                prompts = cast("list[str]", prompts)
                completions = cast("list[str]", completions)
                print_output_sample(prompts, completions)

        return [0.0] * len(prompts)

    @cached_property
    def trainer(self) -> GRPOTrainer:
        """Create the GRPOTrainer using the model, dataset, training arguments, and reward function."""
        reward_funcs = {}
        if self.sample_rate > 0:
            reward_funcs["fake_sample_reward"] = self.fake_sample_reward

        for spec in self.reward_specs:
            fn_name = spec.name
            i = 0
            while fn_name in reward_funcs:
                fn_name = f"{fn_name}_{i}"
                i += 1
            reward_funcs[fn_name] = make_renamed_func(spec.compute_reward, fn_name)

        # Split dataset for training and validation
        train_ds, eval_ds = compute_split(self.dataset_spec.dataset, self.validation, seed=self.training_args_spec.seed)

        trainer = GRPOTrainer(
            model=self.model,
            reward_funcs=list(reward_funcs.values()),
            processing_class=self.transformer_spec.tokenizer,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            args=self.training_args_spec,
            callbacks=self.callbacks,
        )

        for spec in self.reward_specs:
            spec.set_trainer(trainer)

        return trainer

    def main(self):
        """Run the GRPO training procedure."""
        assert self.trainer.model is not None, "Trainer model is None"
        print_trainable_parameters(self.trainer.model)
        logger.info("Starting GRPO training...")
        self.trainer.train()
