"""Wizard module implementing the T5 Pretrain Module."""

from functools import cached_property
from logging import Logger
from typing import TYPE_CHECKING, Any, cast

import structlog
import torch
from datasets import Dataset
from matplotlib.figure import Figure
from pydantic import Field
from transformers import (
    DataCollatorForLanguageModeling,
    PreTrainedModel,
    PreTrainedTokenizer,
    Trainer,
    TrainerCallback,
)

from ...components.callbacks import PretrainCallback
from ...components.helper import eval_mode
from ...components.python_utils import resolve_class_path
from ..generation import GenerationConfigSpec
from ..model import TransformerSpec
from .module import WizardModule

if TYPE_CHECKING:
    from collections.abc import Callable

logger: Logger = structlog.get_logger()


class T5PretrainModule(WizardModule):
    """A system module for the Wizard."""

    transformer_spec: TransformerSpec
    """The transformer model to use for causal seq2seq."""

    generation_args: GenerationConfigSpec = Field(default_factory=GenerationConfigSpec)
    """The generation configuration to use for generation."""

    lm_data_collator_kwargs: dict[str, Any] = Field(default_factory=dict)
    """Additional keyword arguments to pass to the data collator for language modeling.

    See https://huggingface.co/docs/transformers/v4.49.0/en/main_classes/data_collator#transformers.DataCollatorForLanguageModeling

    """

    peek_rate: int = 100
    """Peek at the generations every N steps."""

    peek_sample_size: int = 3
    """The number of samples to peek at."""

    pretrained_model_class: str = "transformers.T5ForConditionalGeneration"
    """The class of the pretrained model to use."""

    @cached_property
    def callbacks(self) -> list[TrainerCallback]:
        """Additional callbacks to use during training."""
        if self.peek_rate > 0:
            return [PretrainCallback(self.predict, self.peek_rate, self.peek_sample_size)]
        return []

    def validate_dataset(self, ds: Dataset):
        """Validate the dataset by making sure the `text` column is present and has strings."""
        # Validate that the dataset has a text column
        if "text" not in ds.column_names:
            raise ValueError("Dataset must have column `text`.")
        # Validate that text is a string
        first_row = ds[0]
        if not isinstance(first_row["text"], str):
            raise ValueError(f"`text` column must contain strings, found {type(first_row['text'])}")

    @cached_property
    def tokenizer(self) -> PreTrainedTokenizer:
        """The tokenizer used in this module.

        For encoding, use the method in `transformer_spec` instead.
        """
        return self.transformer_spec.tokenizer

    @cached_property
    def model(self) -> PreTrainedModel:
        """The model used in this module."""
        model_cls = resolve_class_path(self.pretrained_model_class)
        return self.transformer_spec.create_model(model_cls)

    @cached_property
    def lm_data_collator(self) -> DataCollatorForLanguageModeling:
        """The data collator for language modeling."""
        return DataCollatorForLanguageModeling(tokenizer=self.tokenizer, **self.lm_data_collator_kwargs)

    def data_collator(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        """Prepare a batch for training."""
        texts = [row["text"] for row in batch]
        encoded_batch = self.tokenizer.batch_encode_plus(
            texts, return_tensors="pt", **self.transformer_spec.tokenizer_call_kwargs
        )
        # lm_data_collator expects a list of single-example dicts
        # each with input_ids and attention_mask as individual tensors, not batched
        samples = [{k: v[i] for k, v in encoded_batch.items()} for i in range(len(encoded_batch["input_ids"]))]  # type: ignore
        encoded_sources = self.lm_data_collator(samples)
        # but, the mask tokens function expects the encoded input ids
        target_ids, target_mask = self.lm_data_collator.torch_mask_tokens(encoded_batch["input_ids"])
        if self.tokenizer.pad_token_id is not None:
            target_ids = target_ids.clone()  # Prevent in-place modification issues
            target_ids[target_ids == self.tokenizer.pad_token_id] = -100

        return {**encoded_sources, "labels": target_ids, "decoder_attention_mask": target_mask}

    @torch.no_grad()
    def predict(self, source: list[dict[str, str]], **generation_kwargs) -> dict[str, list[str]]:
        """Predict on a batch of data."""
        inputs = self.data_collator(source)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}  # Move all tensors to model device

        with eval_mode(self.model):
            # Generate outputs
            generated_tokens = cast("Callable", self.model.generate)(
                **inputs,
                generation_config=self.generation_args.to_config(),
                tokenizer=self.tokenizer,
                **generation_kwargs,
            )

        decoded_preds = self.tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

        decoded_inputs = self.tokenizer.batch_decode(
            inputs["input_ids"], skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        # Process labels for decoding
        labels = inputs["labels"]
        if self.tokenizer.pad_token_id is not None:
            labels = labels.clone()  # Avoid modifying inputs directly
            labels[labels == -100] = self.tokenizer.pad_token_id

        decoded_targets = self.tokenizer.batch_decode(
            labels, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

        return {"predictions": decoded_preds, "targets": decoded_targets, "sources": decoded_inputs}

    @torch.no_grad()
    def batch_metrics(self, inputs: dict[str, Any], model_outputs: Any) -> dict[str, float | Figure]:
        """Evaluate a set of data and return some metrics or figures to log."""
        # For pretrain tasks, batch metrics are not commonly computed
        return {}

    @torch.no_grad()
    def eval_metrics(self, eval_dataset: Dataset, batch_size: int) -> dict[str, float | Figure]:
        """Evaluate a dataset and return some metrics to log."""
        # For pretrain tasks, eval metrics are not commonly computed
        return {}

    def compute_loss(
        self, trainer: Trainer, inputs: dict[str, Any], return_outputs: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        """Compute the loss for the model."""
        # Perform the initial forward pass and get the outputs
        assert trainer.model is not None, "Trainer model is None"

        outputs = trainer.model(**inputs)
        loss = outputs.loss

        return (loss, outputs) if return_outputs else loss
