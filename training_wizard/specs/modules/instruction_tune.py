"""Seq2Seq wizard module using Encoder-Decoder models."""

import random
from functools import cached_property
from logging import Logger
from typing import Any, cast

import structlog
import torch
from datasets import Dataset
from matplotlib.figure import Figure
from more_itertools import chunked, flatten
from pydantic import Field
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    Trainer,
    TrainerCallback,
)
from transformers.pipelines import pipeline
from transformers.pipelines.text_generation import TextGenerationPipeline

from ...components.callbacks import SampleCallbackInstruct
from ...components.dataset_validation import validate_instruction_dataset
from ...components.metrics import llm_logit_metrics, seq2seq_metrics
from ..generation import GenerationConfigSpec
from ..model import TransformerSpec
from .module import WizardModule

logger: Logger = structlog.get_logger()


class InstructionTuningModule(WizardModule):
    """Instruction tuning module. Expects a dataset in the same format as a ChatGPT fine-tune."""

    transformer_spec: TransformerSpec
    """The transformer model to use for causal seq2seq."""

    generation_args: GenerationConfigSpec = Field(default_factory=GenerationConfigSpec)
    """The generation configuration to use for generation."""

    assistant_only: bool = True
    """If True, only train on the last assistant output."""

    eval_sample_size: int = 100
    """The number of examples to compute extra evaluation metrics on."""

    peek_rate: int = 100
    """Peek at the generations every N steps."""

    peek_sample_size: int = 3
    """The number of samples to peek at."""

    @cached_property
    def callbacks(self) -> list[TrainerCallback]:
        """Additional callbacks to use during training."""
        if self.peek_rate > 0:
            return [SampleCallbackInstruct(self.predict, self.peek_rate, self.peek_sample_size)]
        return []

    def validate_dataset(self, ds: Dataset):
        """Validate the dataset."""
        validate_instruction_dataset(ds)

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

    def data_collator(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """Prepare a batch for training with an added 'weights' column.

        Each weight is set to `self.different_tokens_upweight` if the corresponding label id is not -100,
        appears only once in the input_ids for that sample, and only if `self.different_tokens_upweight` is not 1.0.
        Otherwise, the weight is set to 1.
        """
        # Apply the chat template to generate input tensors.
        result = self.tokenizer.apply_chat_template(
            [row["messages"] for row in batch],
            return_tensors="pt",
            padding=True,
            return_dict=True,
            **self.transformer_spec.tokenizer_call_kwargs,
        )
        result = cast("dict[str, torch.Tensor]", result)

        # Create labels from input_ids, masking out padding and non-assistant tokens.
        labels = result["input_ids"].clone()
        labels[result["attention_mask"] == 0] = -100

        if self.assistant_only:
            prompt_result = self.tokenizer.apply_chat_template(
                [row["messages"][:-1] for row in batch],
                return_tensors="pt",
                padding=True,
                return_dict=True,
                add_generation_prompt=True,
                **self.transformer_spec.tokenizer_call_kwargs,
            )
            prompt_result = cast("dict[str, torch.Tensor]", prompt_result)
            prompt_lengths = prompt_result["attention_mask"].sum(dim=1).tolist()
            label_start_indices = torch.argmax(prompt_result["attention_mask"], dim=1).tolist()

            for i in range(labels.shape[0]):
                start_index = label_start_indices[i]
                prompt_len = prompt_lengths[i]
                labels[i, start_index : start_index + prompt_len] = -100

        result["labels"] = labels
        return result

    @torch.no_grad()
    def batch_metrics(self, inputs: dict[str, Any], model_outputs: Any) -> dict[str, float | Figure]:
        """Evaluate a batch and return some metrics or figures to log."""
        metrics = {}
        loss = getattr(model_outputs, "loss", None)
        logits = getattr(model_outputs, "logits", None)
        labels = inputs.get("labels")

        # Compute logit-based metrics if logits and labels are available
        if logits is not None and labels is not None:
            # Shift logits and labels for next-token prediction
            shifted_logits = logits[:, :-1]
            shifted_labels = labels[:, 1:]
            logit_metrics_results = llm_logit_metrics(shifted_logits, shifted_labels, loss=loss)
            metrics.update(logit_metrics_results)

        return metrics

    @torch.no_grad()
    def eval_metrics(self, eval_dataset: Dataset, batch_size: int = 8) -> dict[str, float | Figure]:
        """Evaluate a dataset and return some metrics to log."""
        if self.eval_sample_size == 0:
            return {}
        try:
            sample = random.sample(eval_dataset.to_list(), min(self.eval_sample_size, len(eval_dataset)))
            sample_conversations = [row["messages"][:-1] for row in sample]
            sample_targets = [row["messages"][-1]["content"] for row in sample]
            sample_predictions = (self.predict(batch) for batch in chunked(sample_conversations, batch_size))
            sample_predictions_flat = flatten(sample_predictions)
            preds_first_beams = (beams[0] for beams in sample_predictions_flat)
            preds_final = list(tqdm(preds_first_beams, desc="Evaluating outputs...", total=len(sample)))
            formatted_messages: list[str] = [
                "\n\n".join(f"{msg['role'].upper()}: {msg['content']}" for msg in msgs) for msgs in sample_conversations
            ]
            return seq2seq_metrics(
                target=sample_targets,
                hypothesis=preds_final,
                source=formatted_messages,
            )
        except Exception as e:
            logger.error(f"Error evaluating distances: {e}")
            return {}

    @cached_property
    def _pipe(self) -> TextGenerationPipeline:
        return cast(
            "TextGenerationPipeline",
            pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
            ),
        )

    @torch.inference_mode()
    def predict(self, conversations: list[list[dict[str, Any]]], **generation_kwargs) -> list[list[str]]:
        """Predict responses for a batch of conversation messages.

        Args:
            conversations: List of lists of message dictionaries,
                     where each message has 'role' and 'content' keys.
            **generation_kwargs: Additional keyword arguments passed to model.generate().

        Returns:
            List of generated response strings.
        """
        generation_kwargs_all = self.generation_args.model_dump()
        generation_kwargs_all.update(generation_kwargs)

        training = self.model.training
        self.model.eval()

        try:
            generated_texts = self._pipe(
                conversations,
                generation_config=GenerationConfig(**generation_kwargs_all),
                return_full_text=False,
                tokenizer=self.tokenizer,
            )
        finally:
            self.model.train(training)

        return [[beam["generated_text"] for beam in conversation] for conversation in generated_texts]  # type: ignore

    def compute_loss(
        self, trainer: Trainer, inputs: dict[str, Any], return_outputs: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        """Compute the loss for the model."""
        assert trainer.model is not None, "Trainer model is None"

        outputs = trainer.model(**inputs)
        return (outputs.loss, outputs) if return_outputs else outputs.loss
