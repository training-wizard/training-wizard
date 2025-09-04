"""Train a model using REINFORCE Leave-One-Out (RLOO)."""

from functools import cached_property
from logging import Logger
from typing import Any

import structlog
import torch
import torch.nn as nn
from pydantic import Field, field_validator
from pydantic.dataclasses import dataclass as pydantic_dataclass
from transformers import AutoModelForCausalLM, PreTrainedModel, TrainerCallback
from trl import RLOOConfig, RLOOTrainer

from ..components.dataset_validation import validate_prompt_only_dataset
from ..components.helper import compute_split
from ..components.utilities import print_trainable_parameters
from ..specs.base import TrainingRecipe
from ..specs.dataset import DataSourceSpec
from ..specs.model import TransformerSpec
from ..specs.reward import RewardSpec
from ..specs.trainer import CallbacksSpec, MLFlowAndTrainingArgsMixin, WizardTrainingArgsDefaultsMixin

logger: Logger = structlog.get_logger()


class DummyRewardModule(nn.Module):
    """A module for a reward model."""

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Forward pass."""
        raise NotImplementedError("This should never be called. It's just here to make TRL happy.")


@pydantic_dataclass
class RLOOConfigSpec(WizardTrainingArgsDefaultsMixin, RLOOConfig):
    """Specification for the RLOOConfig."""


class RLOOTrainingSpec(TrainingRecipe, MLFlowAndTrainingArgsMixin):
    """Train a model using REINFORCE Leave-One-Out (RLOO).

    RLOO is a method for training language models that uses a simplified form of reinforcement learning.
    Instead of using a value function, RLOO generates K completions for each prompt and uses the mean
    scores from the other K-1 completions as a baseline to calculate the advantage.

    Key features:
    - Models entire completion as a single action (unlike PPO which models each token as an action)
    - Uses leave-one-out scoring to compute advantages
    - Supports both standard and conversational datasets
    - Can use reward models or judges for scoring completions
    - Includes whitening of rewards as an option for training stability

    See: https://huggingface.co/docs/trl/v0.12.2/en/rloo_trainer
    """

    transformer_spec: TransformerSpec
    """The transformer to use for training."""

    reference_transformer_spec: TransformerSpec | None = None
    """The reference transformer to use for comparison. If None, makes a copy of the transformer spec."""

    reward_spec: RewardSpec
    """The reward model to use for training."""

    training_args_spec: RLOOConfigSpec
    """The training arguments."""

    mlflow_experiment_name: str | None = None
    """The name of the MLflow experiment to use."""

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

    filter_length: bool = False
    """Filter inputs that are longer than the 0.9 * max response length."""

    @field_validator("dataset_spec", "validation")
    @classmethod
    def validate_dataset_format(cls, v: Any) -> DataSourceSpec:
        """Validate that the dataset follows either language modeling or conversational format."""
        if isinstance(v, DataSourceSpec):
            validate_prompt_only_dataset(v.dataset)
        return v

    @cached_property
    def policy_model(self) -> PreTrainedModel:
        """The model to use for training."""
        return self.transformer_spec.create_model(AutoModelForCausalLM)

    @cached_property
    def ref_model(self) -> PreTrainedModel:
        """The model to use for training."""
        if self.reference_transformer_spec is None:
            return self.transformer_spec.create_model(AutoModelForCausalLM)
        return self.reference_transformer_spec.create_model(AutoModelForCausalLM)

    @cached_property
    def callbacks(self) -> list[TrainerCallback]:
        """The callbacks to use during training."""
        logger.debug("Building callbacks")
        return self.callbacks_spec.create_callbacks(
            mlflow_experiment_name=self.mlflow_experiment_name,
        )

    def encode_batch(self, batch: dict) -> dict:
        """Encode a batch."""
        prompts = batch["prompt"]
        if isinstance(prompts[0], list):  # If it's a list of chat messages
            prompts = [
                self.transformer_spec.tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
                for p in prompts
            ]
        return self.transformer_spec.encode_batch(prompts)  # type: ignore

    def filter_too_long(self, row: dict) -> bool:
        """Filter a row if it's longer than the max response length."""
        text = row["prompt"][-1]["content"] if isinstance(row["prompt"], list) else row["prompt"]
        return len(self.transformer_spec.tokenizer.tokenize(text)) <= 0.9 * self.training_args_spec.response_length

    @cached_property
    def trainer(self) -> RLOOTrainer:
        """The trainer to use for training."""
        train_ds, test_ds = compute_split(self.dataset_spec.dataset, self.validation, seed=self.training_args_spec.seed)
        if self.filter_length:
            train_len_before = len(train_ds)
            train_ds = train_ds.filter(self.filter_too_long, desc="Filtering training dataset")
            logger.info(f"Filtered {train_len_before - len(train_ds)} training examples due to length")

            if test_ds is not None:
                test_len_before = len(test_ds)
                test_ds = test_ds.filter(self.filter_too_long, desc="Filtering validation dataset")
                logger.info(f"Filtered {test_len_before - len(test_ds)} validation examples due to length")

        train_ds = train_ds.map(
            self.encode_batch, batched=True, desc="Encoding training dataset", remove_columns=train_ds.column_names
        )

        if test_ds is not None:
            test_ds = test_ds.map(
                self.encode_batch, batched=True, desc="Encoding validation dataset", remove_columns=test_ds.column_names
            )

        return RLOOTrainer(
            config=self.training_args_spec,
            processing_class=self.transformer_spec.tokenizer,
            policy=self.policy_model,
            ref_policy=self.ref_model,
            reward_model=DummyRewardModule(),
            train_dataset=train_ds,
            eval_dataset=test_ds,
            callbacks=self.callbacks,
        )

    def main(self):
        """Run the training procedure of the training recipe."""
        print_trainable_parameters(self.trainer.model)

        # Define your new reward function with the same signature
        def custom_get_reward(
            reward_model: Any, sequences: torch.Tensor, pad_token_id: int, context_length: int
        ) -> tuple[None, torch.Tensor, None]:
            # Create attention mask where 1s indicate non-padding tokens
            attention_mask = sequences != pad_token_id

            # Use the mask to get actual sequence lengths
            query_mask = attention_mask[:, :context_length]
            completion_mask = attention_mask[:, context_length:]

            # Decode only the non-padded portions while keeping special tokens
            queries = [
                self.transformer_spec.tokenizer.decode(seq[mask], skip_special_tokens=False)
                for seq, mask in zip(sequences[:, :context_length], query_mask)
            ]
            completions = [
                self.transformer_spec.tokenizer.decode(seq[mask], skip_special_tokens=False)
                for seq, mask in zip(sequences[:, context_length:], completion_mask)
            ]
            return (
                None,
                torch.tensor(
                    self.reward_spec.compute_reward(queries, completions), dtype=torch.float32, device=sequences.device
                ),
                None,
            )

        # Patch it
        import trl.trainer.rloo_trainer

        logger.info("Monkey patching reward function")
        trl.trainer.rloo_trainer.get_reward = custom_get_reward

        logger.info("Training...")
        self.trainer.train()
