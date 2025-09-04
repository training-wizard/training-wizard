"""Specification classes for LoRA and QLoRA."""

from typing import Literal

from peft import LoraConfig  # type: ignore
from pydantic import BaseModel


class LoraConfigSpec(BaseModel, extra="allow"):
    """The specification for the LoRA model. Mimics the HuggingFace config."""

    r: int = 32
    """r parameter for the LoRA model"""

    lora_alpha: int = 64
    """Alpha parameter for the LoRA model. 2*r is a good default value."""

    lora_dropout: float = 0.05
    """The dropout of the LoRA layers"""

    bias: Literal["none", "all", "lora_only"] = "none"
    """The type of bias to use in the LoRA model. Must be one of:
    none
    all
    lora_only
    """

    target_modules: list[str] | str = "all-linear"
    """The modules to apply LoRA to. Must be a non-empty list."""

    def create_lora_config(self) -> LoraConfig:
        """Create the LoRA configuration."""
        return LoraConfig(**self.model_dump())
