"""Dataset specifications."""

import os
from abc import ABC, abstractmethod
from functools import cached_property
from logging import Logger
from typing import Any, Literal, cast

import numpy as np
import structlog
from datasets import Dataset, concatenate_datasets, disable_caching, load_dataset
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from .spec import Spec

logger: Logger = structlog.get_logger()
disable_caching()
logger.info("Caching disabled for HuggingFace datasets")


class DataSourceSpec(Spec, ABC):
    """Abstract base class for data sources."""

    @property
    @abstractmethod
    def dataset(self) -> Dataset:
        """Load the files into a datasets Dataset."""


class TokenLengthFilter(BaseModel):
    """Filter out rows with too many tokens."""

    column: str | list[str]
    """The column(s) to check."""

    max_length: int
    """The maximum length of the tokens."""

    pretrained_name: str
    """The name/path of the pretrained tokenizer to use."""

    aggregation: Literal["max", "sum"] = "sum"
    """Only for multi-column filters. Filtering method:
    - `max`: Check `max(len(tokens)) <= max_length`.
    - `sum`: Check `sum(len(tokens)) <= max_length` (more strict).
    """

    @cached_property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        """The tokenizer to use."""
        return AutoTokenizer.from_pretrained(self.pretrained_name)

    def filter_func(self, batch: dict[str, list[str]]) -> list[bool]:
        """Filter function."""
        cols = [self.column] if isinstance(self.column, str) else self.column
        # [N_cols, N_examples]
        lengths = np.array([[len(ids) for ids in self.tokenizer(batch[col], padding=False).input_ids] for col in cols])
        # [N_examples, N_cols]
        lengths = lengths.T
        if self.aggregation == "max":
            return np.max(lengths, axis=1) <= self.max_length
        elif self.aggregation == "sum":
            return np.sum(lengths, axis=1) <= self.max_length
        else:
            raise ValueError(f"Invalid aggregation: {self.aggregation}")

    def apply(self, dataset: Dataset) -> Dataset:
        """Apply the filter to the dataset."""
        logger.info(
            f"Filtering column(s) {self.column} with {self.aggregation}"
            f" token length > {self.max_length} using tokenizer {self.pretrained_name}"
        )
        return dataset.filter(
            self.filter_func,
            batched=True,
            batch_size=1000,
        )


class PreProcessingSpec(Spec):
    """Common utilities for pre-processing a dataset so you don't have to make a subclass each time.

    Preprocessing is applied in the order of the fields below.
    """

    select_columns: list[str] | None = None
    """Only select some columns in the dataset"""

    rename_columns: dict[str, str] | None = None
    """Rename columns in the dataset."""

    shuffle: bool = True
    """Shuffle the dataset."""

    take: int | None = None
    """Only take the first `n` rows."""

    token_length_filter: TokenLengthFilter | None = None
    """Filter out rows with too many tokens."""

    def apply(self, dataset: Dataset) -> Dataset:
        """Apply the pre-processing to the dataset."""
        if self.select_columns:
            dataset = dataset.select_columns(self.select_columns)
        if self.rename_columns:
            dataset = dataset.rename_columns(self.rename_columns)
        if self.take and len(dataset) > self.take:
            dataset = dataset.select(range(self.take))
        if self.token_length_filter:
            dataset = self.token_length_filter.apply(dataset)
        return dataset


class SimpleDataSourceSpec(DataSourceSpec):
    """A data source that loads files with some loader."""

    data_files: str | list[str]
    """The path to the file(s).

    Supports globbing (e.g. `**/*.jsonl`) and lists of paths (e.g. `["a.jsonl", "b.jsonl"]`)
    """

    loader: Literal["csv", "text", "json"] = "json"
    """The loader to use. Defaults to `json` for JSONL files."""

    load_dataset_kwargs: dict[str, Any] = Field(default_factory=dict)
    """Additional keyword arguments to pass to `load_dataset`.

    These keywords are not allowed:
    - `path`
    - `data_files`
    - `split`
    """

    preprocessing: PreProcessingSpec = Field(default_factory=PreProcessingSpec)
    """Pre-processing to apply to the dataset."""

    @cached_property
    def dataset(self) -> Dataset:
        """Load the files into a datasets Dataset."""
        logger.info("Loading dataset from %s", self.data_files)
        kwargs = self.load_dataset_kwargs.copy()
        data_files = self.data_files if isinstance(self.data_files, list) else [self.data_files]

        dataset_objects = []
        ds = load_dataset(
            path=self.loader,
            data_files=data_files,
            split="train",
            cache_dir=None,
            **kwargs,
        )
        ds = cast("Dataset", ds)  # Just for the type hint
        ds = self.preprocessing.apply(ds)
        dataset_objects.append(ds)

        ds_final = concatenate_datasets(dataset_objects)
        logger.info("Dataset loaded with %d examples", len(ds_final))
        return ds_final


def apply_template(row: dict, template: list[dict]) -> list[dict]:
    """Apply a template to a row."""
    template_copy = [d.copy() for d in template]
    for d in template_copy:
        d["content"] = d["content"].format(**row)
    return template_copy


class TemplateInstructDatasourceSpec(DataSourceSpec):
    """Dataset for paraphrasing."""

    parent: DataSourceSpec
    """The parent dataset to use."""

    output_column: str = "messages"
    """The column to store the output in.

    Other possible names:
    - `prompt` for prompt-only datasets (e.g. Online DPO, XPO)
    """

    messages_template: list[dict] = [  # noqa: RUF012
        {
            "role": "system",
            "content": "You are multilingual paraphraser. User inputs are wrapped in <pphr_input>...</pphr_input> tags. Always answer with a paraphrase, wrapped in <pphr_output>...</pphr_output> tags.",  # noqa: E501
        },
        {"role": "user", "content": "<pphr_input>{source}</pphr_input>"},
        {"role": "assistant", "content": "<pphr_output>{target}</pphr_output>"},
    ]

    def map_row(self, row: dict) -> dict:
        """Preprocess a single row."""
        return {self.output_column: apply_template(row, self.messages_template)}

    @cached_property
    def dataset(self) -> Dataset:
        """Load the dataset."""
        ds = self.parent.dataset
        ds = ds.map(self.map_row, num_proc=os.cpu_count())
        return ds


class MultiDataSourceSpec(DataSourceSpec):
    """Dataset for paraphrasing."""

    parents: list[DataSourceSpec]
    """The datasets to concatenate."""

    @cached_property
    def dataset(self) -> Dataset:
        """Load the dataset."""
        return concatenate_datasets([parent.dataset for parent in self.parents])


class TemplatePreferenceDatasourceSpec(DataSourceSpec):
    """Dataset for preference learning."""

    parent: DataSourceSpec
    """Parent dataset to load from."""

    prompt_template: list[dict] = Field(
        default=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "{prompt}"},
        ]
    )
    """Template for the prompt."""

    chosen_template: list[dict] = Field(
        default=[
            {"role": "assistant", "content": "{chosen}"},
        ]
    )
    """Template for the chosen response."""

    rejected_template: list[dict] = Field(
        default=[
            {"role": "assistant", "content": "{rejected}"},
        ]
    )
    """Template for the rejected response."""

    def map_row(self, row: dict) -> dict:
        """Preprocess a single row."""
        chosen = apply_template(row, self.chosen_template)
        rejected = apply_template(row, self.rejected_template)
        prompt = apply_template(row, self.prompt_template)
        return {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
        }

    @cached_property
    def dataset(self) -> Dataset:
        """Load the dataset."""
        return self.parent.dataset.map(self.map_row, num_proc=os.cpu_count())


class SequenceClassificationDatasourceSpec(DataSourceSpec):
    """Dataset for text classification."""

    parent: DataSourceSpec
    """Parent dataset to load from."""

    text_template: str = "{text}"
    """Template for generating the final text to classify from the dataset columns."""

    label_column: str = "label"
    """Column to use for the label."""

    keep_columns: list[str] = Field(default=[])
    """Additional columns to keep in the output dataset."""

    def map_row(self, row: dict) -> dict:
        """Preprocess a single row."""
        text = self.text_template.format(**row)
        label = row[self.label_column]
        return {"text": text, "labels": label}

    @cached_property
    def dataset(self) -> Dataset:
        """Load the dataset."""
        return self.parent.dataset.map(
            self.map_row,
            remove_columns=[c for c in self.parent.dataset.column_names if c not in self.keep_columns],
            num_proc=os.cpu_count(),
        )


class Seq2SeqDatasourceSpec(DataSourceSpec):
    """Dataset for sequence-to-sequence learning."""

    parent: DataSourceSpec
    """Parent dataset to load from."""

    source_template: str = "{source}"
    """Template for the source column."""

    target_template: str = "{target}"
    """Template for the target column."""

    keep_columns: list[str] = Field(default=[])
    """Additional columns to keep in the output dataset."""

    def map_row(self, row: dict) -> dict:
        """Preprocess a single row."""
        row_strings = {k: str(v) for k, v in row.items()}
        source = self.source_template.format(**row_strings)
        target = self.target_template.format(**row_strings)
        return {"source": source, "target": target}

    @cached_property
    def dataset(self) -> Dataset:
        """Load the dataset."""
        remove_columns = [col for col in self.parent.dataset.column_names if col not in self.keep_columns]
        return self.parent.dataset.map(self.map_row, remove_columns=remove_columns, num_proc=os.cpu_count())
