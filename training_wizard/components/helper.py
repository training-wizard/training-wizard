"""Helper functions for training wizard."""

from collections.abc import Callable, Generator
from contextlib import contextmanager
from logging import Logger
from typing import Any

import structlog
import torch
from datasets import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizer

from ..specs.dataset import DataSourceSpec

logger: Logger = structlog.get_logger()


@contextmanager
def eval_mode(model: PreTrainedModel) -> Generator[None, None, None]:
    """Set the model to evaluation mode."""
    mode = model.training
    model.eval()
    yield
    model.train(mode)


def ensure_pad_token(tokenizer: PreTrainedTokenizer) -> PreTrainedTokenizer:
    """Ensure that the tokenizer has a padding token."""
    if not tokenizer.pad_token or tokenizer.pad_token_id is None:
        logger.info("No padding token found, using eos_token as padding token")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def standard_prepare_batch(encode_fn: Callable, batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Data collator for training."""
    text = [row["text"] for row in batch]
    text_pair = None
    if all("text_pair" in row for row in batch):
        text_pair = [row["text_pair"] for row in batch]
    batch_encoding = encode_fn(text, text_pair=text_pair, return_tensors="pt", padding=True)
    if all("label" in row for row in batch):
        labels = torch.tensor([row["label"] for row in batch])
        return {**batch_encoding, "labels": labels}
    elif all("labels" in row for row in batch):
        labels = torch.tensor([row["labels"] for row in batch])
        return {**batch_encoding, "labels": labels}
    return dict(batch_encoding)


def compute_split(
    training_dataset: Dataset, validation: float | int | DataSourceSpec | None, seed: int | None = 42
) -> tuple[Dataset, Dataset | None]:
    """Split the dataset into train and eval sets.

    Args:
        training_dataset: The training dataset to potentially split
        validation: Validation dataset specification:
            - If float, fraction of training data to use for validation
            - If int, number of examples to use for validation
            - If DataSourceSpec, separate validation dataset to use
            - If None, no validation set will be used
        seed: The seed to use for shuffling during the split.

    Returns:
        A tuple of (train_dataset, eval_dataset) where eval_dataset may be None

    Raises:
        ValueError: If validation is an invalid type
    """
    if validation is None:
        return training_dataset, None
    elif isinstance(validation, int | float):
        d = training_dataset.train_test_split(test_size=validation, shuffle=True, seed=seed)
        return d["train"], d["test"]
    elif isinstance(validation, DataSourceSpec):
        return training_dataset, validation.dataset
    raise ValueError(f"Invalid validation input: {validation}")


def compute_prefix_mask(prefix: list[str], offset_mapping: torch.Tensor) -> torch.Tensor:
    """Create a binary mask identifying which tokens *do not* belong to the prefix.

    For each sequence in a batch, creates a mask where 0 indicates tokens that are part of the prefix text,
    and 1 indicates tokens that are part of the continuation text.

    Args:
        prefix: List of prefix strings to mask
        offset_mapping: Tensor of shape [batch_size, seq_len, 2] containing the character offsets for each token

    Returns:
        Tensor of shape [batch_size, seq_len] containing 0s for prefix tokens and 1s for continuation tokens
    """
    prefix_mask = torch.ones(*offset_mapping.shape[:2], dtype=torch.long, device=offset_mapping.device)
    for i in range(offset_mapping.shape[0]):
        prefix_end = len(prefix[i])
        for j in range(offset_mapping.shape[1]):
            token_start, token_end = offset_mapping[i, j]
            if token_end < prefix_end:
                prefix_mask[i, j] = 0
            else:
                break
    return prefix_mask
