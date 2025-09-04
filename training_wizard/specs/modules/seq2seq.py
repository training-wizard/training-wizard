"""Seq2Seq wizard module using Encoder-Decoder models."""

import random
from functools import cached_property
from logging import Logger
from typing import TYPE_CHECKING, Any, cast

import structlog
import torch
from datasets import Dataset
from matplotlib.figure import Figure
from more_itertools import chunked
from pydantic import Field
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, PreTrainedModel, PreTrainedTokenizer, Trainer, TrainerCallback

from ...components.callbacks import SampleCallback
from ...components.dataset_validation import validate_seq2seq_dataset
from ...components.helper import eval_mode
from ...components.metrics import llm_logit_metrics, seq2seq_metrics
from ..generation import GenerationConfigSpec
from ..model import TransformerSpec
from .module import WizardModule

if TYPE_CHECKING:
    from collections.abc import Callable

logger: Logger = structlog.get_logger()


class Seq2SeqModule(WizardModule):
    """Seq2Seq module. Expects two dataset columns: `source` and `target`."""

    transformer_spec: TransformerSpec
    """The transformer model to use for seq2seq."""

    generation_args: GenerationConfigSpec = Field(default_factory=GenerationConfigSpec)
    """The generation configuration to use for generation."""

    eval_sample_size: int = 100
    """The number of samples to use for evaluation."""

    peek_rate: int = 100
    """Peek at the generations every N steps."""

    peek_sample_size: int = 3
    """The number of samples to peek at."""

    @cached_property
    def callbacks(self) -> list[TrainerCallback]:
        """Additional callbacks to use during training."""
        if self.peek_rate > 0:
            return [SampleCallback(self.predict, self.peek_rate, self.peek_sample_size)]
        return []

    def validate_dataset(self, ds: Dataset):
        """Validate the dataset."""
        validate_seq2seq_dataset(ds)

    @cached_property
    def model(self) -> PreTrainedModel:
        """The model used in this module."""
        return self.transformer_spec.create_model(AutoModelForSeq2SeqLM)

    @cached_property
    def tokenizer(self) -> PreTrainedTokenizer:
        """The tokenizer used in this module.

        For encoding, use the method in `transformer_spec` instead.
        """
        return self.transformer_spec.tokenizer

    def data_collator(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        """Prepare a batch for training, including tokenizing source and target texts."""
        source_texts = [row["source"] for row in batch]
        target_texts = [row["target"] for row in batch]

        encoded = dict(
            self.transformer_spec.encode_batch(
                source_texts,
                padding=True,  # Pad to the longest sequence in the batch
                truncation=True,  # Truncate sequences longer than the model's maximum length
                return_tensors="pt",  # Return PyTorch tensors
            )
        )
        targets = self.transformer_spec.encode_batch(
            target_texts,
            padding=True,  # Pad to the longest sequence in the batch
            truncation=True,  # Truncate sequences longer than the model's maximum length
            return_tensors="pt",  # Return PyTorch tensors
        )
        labels = targets.input_ids
        labels[targets.attention_mask == 0] = -100
        encoded["labels"] = labels
        return encoded

    @torch.no_grad()
    def batch_metrics(self, inputs: dict[str, Any], model_outputs: Any) -> dict[str, float | Figure]:
        """Evaluate a batch and return some metrics or figures to log."""
        metrics = {}
        loss = getattr(model_outputs, "loss", None)
        logits = getattr(model_outputs, "logits", None)
        labels = inputs.get("labels")

        # Compute logit-based metrics if logits and labels are available
        if logits is not None and labels is not None:
            logit_metrics_results = llm_logit_metrics(logits, labels, loss=loss)
            metrics.update(logit_metrics_results)

        return metrics

    @torch.no_grad()
    def eval_metrics(self, eval_dataset: Dataset, batch_size: int = 8) -> dict[str, float | Figure]:
        """Evaluate a dataset and return some metrics to log."""
        if self.eval_sample_size <= 0:
            logger.warning("Skipping evaluation as eval_sample_size is <= 0")
            return {}
        elif self.eval_sample_size > len(eval_dataset):
            logger.warning("eval_sample_size is greater than the eval dataset size, using the entire dataset")
            sample_size = len(eval_dataset)
        else:
            sample_size = self.eval_sample_size

        sample = eval_dataset.select(random.sample(range(len(eval_dataset)), sample_size)).to_list()

        results = []
        for batch in chunked(tqdm(sample, desc="Running seq2seq on evaluation dataset..."), batch_size):
            results.extend(self.predict([row["source"] for row in batch]))

        return seq2seq_metrics(
            target=[row["target"] for row in sample],
            hypothesis=results,
            source=[row["source"] for row in sample],
        )

    @torch.no_grad()
    def predict(self, batch: list[str], **generation_kwargs) -> list[str]:
        """Predict on a batch of data."""
        # Prepare the batch
        enc = self.data_collator([{"source": e, "target": e} for e in batch])
        inputs_ids = enc["input_ids"].to(self.model.device)
        att_mask = enc["attention_mask"].to(self.model.device)

        with eval_mode(self.model):
            # Generate outputs
            generated_tokens = cast("Callable", self.model.generate)(
                inputs_ids,
                generation_config=self.generation_args.to_config(),
                tokenizer=self.tokenizer,
                attention_mask=att_mask,
                **generation_kwargs,
            )

        # Decode generated tokens to strings
        decoded_preds = self.tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

        return decoded_preds

    def compute_loss(
        self, trainer: Trainer, inputs: dict[str, Any], return_outputs: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        """Compute the loss for the model."""
        assert trainer.model is not None, "Trainer model is None"

        outputs = trainer.model(**inputs)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss
