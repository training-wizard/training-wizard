"""Seq2Seq wizard module using Encoder-Decoder models."""

import logging
import random
from functools import cached_property
from logging import Logger
from typing import Any, cast

import structlog
import torch
from datasets import Dataset
from matplotlib.figure import Figure
from more_itertools import chunked
from pydantic import Field, model_validator
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    TextGenerationPipeline,
    Trainer,
    TrainerCallback,
    pipeline,
)

from ...components.callbacks import SampleCallback
from ...components.dataset_validation import validate_seq2seq_dataset
from ...components.helper import compute_prefix_mask
from ...components.metrics import llm_logit_metrics, seq2seq_metrics
from ..generation import GenerationConfigSpec
from ..model import TransformerSpec
from .module import WizardModule

logger: Logger = structlog.get_logger()


class CausalSeq2SeqModule(WizardModule):
    """Causal Seq2Seq module. Expects two dataset columns: `source` and `target`."""

    transformer_spec: TransformerSpec
    """The transformer model to use for causal seq2seq."""

    mask_source_tokens: bool = True
    """Whether to mask the source tokens in the labels during training."""

    eval_sample_size: int | None = None
    """Optionally, only use a subset of the evaluation dataset for the extra metrics, because generation is expensive."""  # noqa: E501

    separator_token: str = "<sep>"
    """The token to use as separator between source and target.

    If not part of the vocabulary, will be added as a new token.
    """

    add_eos_token: bool = True
    """Whether to add the EOS token to the end of the target text."""

    generation_args: GenerationConfigSpec = Field(default_factory=GenerationConfigSpec)
    """The generation configuration to use for generation."""

    peek_rate: int = 100
    """Peek at the generations every N steps."""

    peek_sample_size: int = 3
    """The number of samples to peek at."""

    label_smoothing: float = 0.0
    """The label smoothing to use during training."""

    @cached_property
    def callbacks(self) -> list[TrainerCallback]:
        """Additional callbacks to use during training."""
        if self.peek_rate > 0:
            return [SampleCallback(self.translate, self.peek_rate, self.peek_sample_size)]
        return []

    def validate_dataset(self, ds: Dataset):
        """Validate the dataset."""
        validate_seq2seq_dataset(ds)

    @model_validator(mode="after")
    def disable_tokenizer_padding_side_warning(self) -> Any:
        """Disable warnings related to tokenizer padding side."""
        # Suppress specific warning by message
        logging.getLogger("transformers").setLevel(logging.ERROR)
        return self

    @cached_property
    def model(self) -> PreTrainedModel:
        """The model used in this module."""
        return self.transformer_spec.create_model(AutoModelForCausalLM)

    @cached_property
    def tokenizer(self) -> PreTrainedTokenizer:
        """The tokenizer used in this module.

        For encoding, use the method in `transformer_spec` instead.
        """
        return self.transformer_spec.tokenizer

    def data_collator(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        """Prepare a batch for training."""
        source_texts = [row["source"] for row in batch]
        target_texts = [row["target"] for row in batch]
        prefixes = [f"{s}{self.separator_token}" for s in source_texts]
        suffix = self.tokenizer.eos_token if self.add_eos_token else ""
        pair_texts = [f"{prefix}{tgt}{suffix}" for prefix, tgt in zip(prefixes, target_texts)]

        # Tokenize inputs and targets
        model_inputs = self.transformer_spec.encode_batch(
            pair_texts,
            return_tensors="pt",
            padding=True,
            return_offsets_mapping=True,
        )
        model_inputs_dict: dict[str, torch.Tensor] = dict(model_inputs)  # type: ignore
        labels = model_inputs_dict["input_ids"].clone()
        labels[model_inputs_dict["attention_mask"] == 0] = -100
        offsets_mapping = model_inputs_dict["offset_mapping"]

        if self.mask_source_tokens:
            prefix_mask = compute_prefix_mask(source_texts, offsets_mapping)
            labels[prefix_mask == 0] = -100

        # NOTE: labels are not shifted because this is done by the model
        model_inputs_dict["labels"] = labels
        return model_inputs_dict

    @torch.no_grad()
    def batch_metrics(self, inputs: dict[str, Any], model_outputs: Any) -> dict[str, float | Figure]:
        """Evaluate a batch and return some metrics or figures to log."""
        metrics = {}
        loss = getattr(model_outputs, "loss", None)
        logits = getattr(model_outputs, "logits", None)
        labels = inputs.get("labels")

        # Compute logit-based metrics if logits and labels are available
        if logits is not None and labels is not None:
            labels_shifted = labels[:, 1:]
            logits_shifted = logits[:, :-1]
            logit_metrics_results = llm_logit_metrics(logits_shifted, labels_shifted, loss=loss)
            metrics.update(logit_metrics_results)

        return metrics

    @torch.no_grad()
    def eval_metrics(self, eval_dataset: Dataset, batch_size: int) -> dict[str, float | Figure]:
        """Evaluate a dataset and return some metrics to log."""
        if not self.eval_sample_size or self.eval_sample_size <= 0:
            logger.warning("Skipping evaluation as eval_sample_size is <= 0")
            return {}

        sample = eval_dataset.select(random.sample(range(len(eval_dataset)), self.eval_sample_size)).to_list()

        results = []
        for batch in chunked(tqdm(sample, desc="Running seq2seq on evaluation dataset..."), batch_size):
            results.extend(self.translate([row["source"] for row in batch]))

        return seq2seq_metrics(
            target=[row["target"] for row in sample],
            hypothesis=results,
            source=[row["source"] for row in sample],
        )

    @cached_property
    def pipeline(self) -> TextGenerationPipeline:
        """The pipeline used to generate translations."""
        return cast(
            "TextGenerationPipeline",
            pipeline(
                task="text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
            ),
        )

    @torch.inference_mode()
    def translate(self, source: list[str], **generation_kwargs) -> list[str]:
        """Translate a batch of source texts.

        Args:
            source: List of source texts to translate.
            **generation_kwargs: Additional keyword arguments passed to model.generate().

        Returns:
            List of translated texts.
        """
        # Prepare inputs by adding separator token
        x_str = [f"{s}{self.separator_token}" for s in source]

        # Merge generation config with any additional kwargs
        generation_kwargs_all = self.generation_args.model_dump()
        generation_kwargs_all.update(generation_kwargs)

        # Generate translations
        outputs = self.pipeline(
            x_str,
            generation_config=GenerationConfig(**generation_kwargs_all),
            return_full_text=False,
            tokenizer=self.tokenizer,
        )
        outputs = cast("list[list[dict[str, Any]]]", outputs)

        # Extract generated text, removing the input prefix
        result = [output[0]["generated_text"].strip() for output, x in zip(outputs, x_str)]
        return result

    def compute_loss(
        self, trainer: Trainer, inputs: dict[str, Any], return_outputs: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        """Compute the loss for the model with label smoothing.

        Args:
            trainer: The trainer instance.
            inputs: The model inputs.
            return_outputs: Whether to return model outputs along with the loss.

        Returns:
            The computed loss tensor, optionally with model outputs.
        """
        labels = inputs["labels"]

        assert trainer.model is not None, "Trainer model is None"

        outputs = trainer.model(**inputs, return_dict=True)
        labels_shifted = labels[:, 1:]
        logits_shifted = outputs.logits[:, :-1]

        loss = torch.nn.functional.cross_entropy(
            logits_shifted.reshape(-1, logits_shifted.size(-1)),
            labels_shifted.reshape(-1),
            ignore_index=-100,
            label_smoothing=self.label_smoothing,
        )

        return (loss, outputs) if return_outputs else loss
