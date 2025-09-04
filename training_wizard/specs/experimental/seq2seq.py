"""Seq2Seq wizard module using Encoder-Decoder models."""

import random
from collections import defaultdict
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
from ..modules.module import WizardModule

if TYPE_CHECKING:
    from collections.abc import Callable

logger: Logger = structlog.get_logger()


@torch.no_grad()
def cross_existence_mask(inputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Creates a mask for labels indicating if each element exists in the corresponding row of inputs.

    For each index [x_i, l_i] in the output mask, the value is True if the value
    labels[x_i, l_i] is present at least once in the row inputs[x_i, :].
    Otherwise, the value is False.

    Args:
        inputs: A LongTensor of shape [X, A].
        labels: A LongTensor of shape [X, B].

    Returns:
        A BoolTensor mask of shape [X, B].
    """
    if inputs.dtype != torch.long or labels.dtype != torch.long:
        inputs = inputs.long()
        labels = labels.long()
    if inputs.dim() != 2 or labels.dim() != 2:
        raise ValueError("Input tensors must be 2-dimensional.")
    if inputs.shape[0] != labels.shape[0]:
        raise ValueError(
            f"First dimension must match for both tensors, but got shapes {inputs.shape} and {labels.shape}"
        )

    labels_a_expanded = inputs.unsqueeze(2)  # Shape: [X, A, 1]
    labels_b_expanded = labels.unsqueeze(1)  # Shape: [X, 1, B]
    comparison = labels_a_expanded == labels_b_expanded
    mask = comparison.any(dim=1)

    return mask


@torch.no_grad()
def compute_loss_weights(
    labels: torch.Tensor,  # The labels corresponding to the loss tokens (e.g., shift_labels)
    token_frequencies: torch.Tensor,  # Pre-calculated or running frequencies (size=vocab_size)
    factor: float,  # The frequency_regularization factor
) -> torch.Tensor:
    """Rescales per-token loss based on token frequencies."""
    if factor <= 0:
        return torch.ones_like(labels)

    token_frequencies = token_frequencies.to(labels.device)
    current_freqs = token_frequencies[labels]
    weights = torch.exp(-factor * current_freqs)

    return weights


def compute_token_frequencies(
    text: list[str],
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 1024,
) -> torch.Tensor:
    """Calculates token frequencies."""
    logger.info("Pre-computing token frequencies from the training dataset...")

    counts: dict[int, int] = defaultdict(int)

    for chunk in chunked(tqdm(text, desc="Pre-computing token frequencies..."), batch_size):
        tokenized = tokenizer(chunk, padding=False)
        for ids in tokenized.input_ids:
            for token_id in ids:
                counts[token_id] += 1

    total_tokens = sum(counts.values())
    counts_list = [counts[i] / total_tokens for i in range(tokenizer.vocab_size)]
    return torch.tensor(counts_list, dtype=torch.float32)


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

    professor_forcing_probability: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Probability of replacing a target token with a model sample during training (Professor Forcing).",
    )
    """Applies Professor Forcing technique. During training, replaces a fraction of ground-truth target tokens
       with tokens sampled from the model's own output distribution (using no gradients for the sampling step).
       The loss for these sampled positions is masked out. Helps mitigate exposure bias."""

    frequency_regularization: float = 0.0
    """Regularize the loss to focus on less frequent tokens."""

    copy_penalty: float = 0.0
    """Penalize the loss for copying tokens from the source."""

    label_smoothing: float = 0.0
    """Label smoothing for the loss function."""

    @cached_property
    def callbacks(self) -> list[TrainerCallback]:
        """Additional callbacks to use during training."""
        if self.peek_rate > 0:
            return [SampleCallback(self.predict, self.peek_rate, self.peek_sample_size)]
        return []

    def validate_dataset(self, ds: Dataset):
        """Validate the dataset."""
        validate_seq2seq_dataset(ds)
        if self.frequency_regularization > 0:
            all_texts = [row["target"] for row in ds]  # type: ignore
            self._token_frequencies = compute_token_frequencies(all_texts, self.tokenizer, batch_size=1024)

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
        target_texts = [row["target"] for row in batch] if all("target" in row for row in batch) else None

        encoded = dict(
            self.transformer_spec.encode_batch(
                source_texts,
                padding=True,  # Pad to the longest sequence in the batch
                truncation=True,  # Truncate sequences longer than the model's maximum length
                return_tensors="pt",  # Return PyTorch tensors
            )
        )
        if target_texts:
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
    def professor_force_labels(self, inputs: dict[str, Any], model: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Applies Professor Forcing.

        Runs a forward pass without gradients to get logits, samples tokens based on
        `professor_forcing_probability`, replaces corresponding labels, and returns
        the new labels along with a mask indicating which positions contribute to the loss.

        Args:
            inputs: Dictionary containing model inputs (e.g., 'input_ids', 'attention_mask', 'labels').
                    'labels' are expected here for determining sampling positions and masking.
            model: The model being trained.

        Returns:
            A tuple containing:
            - augmented_labels (torch.Tensor): Labels tensor potentially modified with sampled tokens.
            - label_mask (torch.Tensor): Mask tensor (1 for original labels, 0 for sampled or padding).
        """
        original_labels: torch.Tensor = inputs["labels"]
        if not model.training or self.professor_forcing_probability <= 0:
            mask = (
                (original_labels != -100).long()
                if original_labels is not None
                else torch.ones_like(inputs["input_ids"])
            )
            return original_labels, mask

        inputs = {k: v.to(model.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
        original_labels = original_labels.to(model.device)
        label_mask: torch.Tensor = torch.ones_like(original_labels)
        label_mask[original_labels == -100] = 0
        label_mask[original_labels == self.tokenizer.pad_token_id] = 0

        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        sampled_indices = torch.multinomial(probs.view(-1, logits.shape[-1]), num_samples=1).view_as(original_labels)

        replace_mask = torch.rand(label_mask.shape, device=label_mask.device) * label_mask
        replace_mask = replace_mask < self.professor_forcing_probability
        keep_mask = ~replace_mask

        new_labels = sampled_indices * replace_mask + original_labels * keep_mask
        label_mask = label_mask * keep_mask
        return new_labels, label_mask

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
        enc = self.data_collator([{"source": e} for e in batch])
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
        """Compute the loss for the model, applying Professor Forcing if enabled.

        For a T5 model, we shift the logits and labels so that the decoder output at
        time step t is compared with the label at t+1.

        Loss is manually calculated and masked based on sampled tokens.
        """
        inputs = {k: v.to(self.model.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
        assert trainer.model is not None, "Trainer model is None"

        # If all the regularization and penalty values are 0, we can just return the loss from the model.
        if all(
            prop == 0
            for prop in [
                self.professor_forcing_probability,
                self.frequency_regularization,
                self.copy_penalty,
            ]
        ):
            outputs = trainer.model(**inputs)
            return (outputs.loss, outputs) if return_outputs else outputs.loss

        labels = inputs["labels"]
        label_mask = torch.ones_like(labels)
        label_mask[labels == -100] = 0
        label_mask[labels == self.tokenizer.pad_token_id] = 0

        if trainer.model.training and self.professor_forcing_probability > 0:
            labels, prof_mask = self.professor_force_labels(inputs, trainer.model)
            label_mask = label_mask * prof_mask
            inputs["labels"] = labels

        outputs = trainer.model(**inputs, return_dict=True)
        logits = outputs.logits  # Shape: (batch_size, seq_len, vocab_size)
        vocab_size = logits.size(-1)

        # For T5, shift the logits and labels so that decoder output at time t is compared with label t+1.
        # shift_logits = logits[:, :-1, :].contiguous()
        # shift_labels = labels[:, 1:].contiguous()
        # shift_mask = label_mask[:, 1:].contiguous()
        shift_logits = logits.contiguous()
        shift_labels = labels.contiguous()
        shift_mask = label_mask.contiguous()

        per_token_loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, vocab_size),
            shift_labels.view(-1),
            reduction="none",
            label_smoothing=self.label_smoothing,
        )

        per_token_loss = per_token_loss.view(shift_mask.shape)

        # --- Apply Frequency Regularization ---
        if self.frequency_regularization > 0:
            loss_weights = compute_loss_weights(
                shift_labels,
                self._token_frequencies,
                self.frequency_regularization,
            )
            loss_weights = loss_weights / loss_weights.sum(dim=-1, keepdim=True).clip(min=1e-10)
            loss_weights = loss_weights * shift_mask.sum(dim=-1, keepdim=True)
            per_token_loss = per_token_loss * loss_weights

        # --- Apply Copy Penalty ---
        # Rescale loss for labels that exist in the source sentence.
        if self.copy_penalty > 0:
            existence_mask = cross_existence_mask(inputs["input_ids"], labels)
            # Don't penalize special tokens.
            for special_token in self.tokenizer.all_special_ids:
                existence_mask[labels == special_token] = 0
            new_weights = 1 - (self.copy_penalty * existence_mask.float())
            per_token_loss = per_token_loss * new_weights

        loss = (per_token_loss * shift_mask).sum() / (shift_mask.sum() + 1e-10)
        outputs.loss = loss

        return (loss, outputs) if return_outputs else loss
