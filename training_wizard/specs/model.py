"""Specification classes for the model to use."""

import os
from enum import StrEnum
from functools import cached_property, partial
from logging import Logger
from typing import Any, cast

import structlog
import torch
from peft import get_peft_model  # type: ignore
from pydantic import AliasChoices, BaseModel, Field, field_validator
from transformers import (
    AutoTokenizer,
    BatchEncoding,
    BitsAndBytesConfig,
    GPTQConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from ..components.helper import ensure_pad_token
from .lora import LoraConfigSpec
from .spec import Spec

logger: Logger = structlog.get_logger()


class ComputationalDtype(StrEnum):
    """Type of the pytorch dtype."""

    bfloat16 = "bfloat16"
    float32 = "float32"
    float16 = "float16"


class Quantization(StrEnum):
    """Quantization type for Bits and Bytes."""

    nf4 = "nf4"
    fp4 = "fp4"


class BitsAndBytesConfigSpec(BaseModel):
    """The specification for the Bits and Bytes model. Mimics the HuggingFace config."""

    load_in_4bit: bool = True
    bnb_4bit_quant_type: Quantization = Quantization.nf4
    bnb_4bit_use_double_quant: bool = False
    bnb_4bit_compute_dtype: ComputationalDtype = ComputationalDtype.bfloat16

    def create_quantization_config(self) -> BitsAndBytesConfig:
        """Create the Bits and Bytes configuration."""
        return BitsAndBytesConfig(**self.model_dump())

    def method_name(self) -> str:
        """Return the quantization method name."""
        return "Bits and Bytes"


class GPTQConfigSpec(BaseModel):
    """The specification for the GPTQ model. Mimics the HuggingFace config."""

    bits: int
    use_exllama: bool

    def create_quantization_config(self) -> GPTQConfig:
        """Create the GPTQ configuration."""
        return GPTQConfig(**self.model_dump())

    def method_name(self) -> str:
        """Return the quantization method name."""
        return "GPTQ"


class TransformerSpec(Spec):
    """The specification for the pretrained model to use."""

    pretrained_name: str
    """The name of the pretrained model to use. Must be a HuggingFace model."""

    quantization_spec: BitsAndBytesConfigSpec | GPTQConfigSpec | None = Field(
        default=None, validation_alias=AliasChoices("bnb_spec", "quantization_spec")
    )
    """Quantize the model using either Bits and Bytes or GPTQ."""

    device_map: str | dict[str, str] | None = None
    """The device map to use for the model. Can be a string or a dictionary."""

    from_pretrained_kwargs: dict[str, Any] = Field(default_factory=dict)
    """Keyword arguments to pass to the model initialization method."""

    tokenizer_init_kwargs: dict[str, Any] = Field(default_factory=dict)
    """Keyword arguments to pass to the tokenizer initialization method."""

    tokenizer_call_kwargs: dict[str, Any] = Field(default_factory=dict)
    """Keyword arguments to pass to the tokenizer call method."""

    tokenizer_special_tokens: dict[str, Any] | list[str] | None = None
    """Additional special tokens to add to the tokenizer."""

    disable_tokenizer_parallelism: bool = False
    """Disable parallelism in the tokenizer if we're using it before forking the process."""

    lora_spec: LoraConfigSpec | None = None
    """Enable LoRA training and specify parameters"""

    apply_chat_template_kwargs: dict[str, Any] = Field(default_factory=dict)
    """Additional keyword arguments passed to `tokenizer.apply_chat_template`."""

    @field_validator("disable_tokenizer_parallelism")
    @classmethod
    def check_disable_tokenizer_parallelism(cls, v: bool) -> bool:
        """Disable parallelism in the tokenizer if we're using it before forking the process."""
        logger.info("Disabling tokenizer parallelism")
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        return v

    @field_validator("from_pretrained_kwargs")
    @classmethod
    def str_to_torch(cls, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Convert string to torch objects."""
        # check if torch_dtype is set
        if "torch_dtype" in kwargs:
            kwargs["torch_dtype"] = {
                "bfloat16": torch.bfloat16,
                "bf16": torch.bfloat16,
                "float32": torch.float32,
                "fp32": torch.float32,
                "float16": torch.float16,
                "fp16": torch.float16,
            }[kwargs["torch_dtype"]]
        return kwargs

    def encode_batch(self, text: str | list[str], text_pair: str | list[str] | None = None, **kwargs) -> BatchEncoding:
        """Encode a batch of text."""
        base_kw = self.tokenizer_call_kwargs.copy()
        base_kw.update(kwargs)
        if text_pair and self.is_text_pair_broken:
            separator = self.tokenizer.sep_token
            if isinstance(text, str):
                assert isinstance(text_pair, str), "If text is a string, text_pair must be a string"
                text = f"{text}{separator}{text_pair}"
            else:
                assert isinstance(text_pair, list), "If text is a list, text_pair must be a list"
                text = [f"{t}{separator}{tp}" for t, tp in zip(text, text_pair)]
            text_pair = None
        resp = self.tokenizer(text=text, text_pair=text_pair, **base_kw)
        return resp

    @cached_property
    def is_text_pair_broken(self) -> bool:
        """Check if the text_pair argument is broken."""
        resp = False
        if self.tokenizer.sep_token is None:
            logger.warning("Tokenizer does not have a separator token, using workaround")
            resp = True
        if self.tokenizer.encode("cat", text_pair="dog") == self.tokenizer.encode("catdog"):
            logger.warning("Tokenizer text_pair argument is broken, using workaround")
            resp = True
        if resp:
            self.tokenizer.sep_token = next(
                tok for tok in self.tokenizer.all_special_tokens if tok != self.tokenizer.pad_token
            )
            self.tokenizer.sep_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.sep_token)
            logger.info(
                f"Using separator token {self.tokenizer.sep_token}. Encoded text will look like this: 'I am a cat {self.tokenizer.sep_token} I am a dog'"  # noqa: E501
            )
        return False

    @cached_property
    def tokenizer(self) -> PreTrainedTokenizer:
        """Create the tokenizer."""
        tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            self.pretrained_name,
            **self.tokenizer_init_kwargs,
        )  # type: ignore
        tokenizer = ensure_pad_token(tokenizer)

        # Add any special tokens if specified and not already present
        if self.tokenizer_special_tokens:
            if isinstance(self.tokenizer_special_tokens, dict):
                num_added = tokenizer.add_special_tokens(
                    self.tokenizer_special_tokens,
                    replace_additional_special_tokens=False,
                )
            else:
                num_added = tokenizer.add_special_tokens(
                    {"additional_special_tokens": self.tokenizer_special_tokens},  # type: ignore
                    replace_additional_special_tokens=False,
                )
            if num_added > 0:
                logger.info(f"Added {num_added} special tokens to the tokenizer: {self.tokenizer_special_tokens}")

        if self.apply_chat_template_kwargs:
            tokenizer.apply_chat_template = partial(tokenizer.apply_chat_template, **self.apply_chat_template_kwargs)

        return tokenizer

    def create_model(self, model_cls: Any) -> PreTrainedModel:
        """Create the model."""
        logger.info(f"Loading model {self.pretrained_name} with class {model_cls}")

        if self.quantization_spec:
            logger.info(f"Quantizing model using {self.quantization_spec.method_name()}.")

        model: PreTrainedModel = model_cls.from_pretrained(
            self.pretrained_name,
            quantization_config=self.quantization_spec.create_quantization_config() if self.quantization_spec else None,
            device_map=self.device_map,
            **self.from_pretrained_kwargs,
        )

        # Ensure that the model and tokenizer have the same pad token
        model.config.pad_token_id = self.tokenizer.pad_token_id  # type: ignore

        # Resize embeddings if special tokens were added
        if self.tokenizer_special_tokens:
            logger.info(f"Resizing token embeddings for {len(self.tokenizer_special_tokens)} special tokens")
            model.resize_token_embeddings(len(self.tokenizer))

        if self.lora_spec:
            logger.info("Preparing model for LoRA training")
            peft_config = self.lora_spec.create_lora_config()
            model = cast("PreTrainedModel", get_peft_model(model, peft_config))
        return model
