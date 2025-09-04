"""Specification for the generation configuration."""

from transformers import GenerationConfig

from .spec import Spec


class GenerationConfigSpec(Spec, extra="allow"):
    """Simplified configuration for text generation with key parameters."""

    max_new_tokens: int = 512
    """The maximum number of new tokens to generate."""

    min_new_tokens: int | None = None
    """The minimum number of new tokens to generate."""

    do_sample: bool = False
    """Whether to use sampling; if False, uses greedy decoding."""

    temperature: float = 1.0
    """Value to modulate the next token probabilities when sampling."""

    top_k: int = 50
    """Number of highest probability tokens to keep for top-k filtering."""

    top_p: float = 1.0
    """Cumulative probability threshold for nucleus sampling."""

    num_beams: int = 1
    """Number of beams for beam search. 1 means no beam search."""

    early_stopping: bool = False
    """Whether to stop the beam search when at least num_beams sentences are finished."""

    repetition_penalty: float = 1.0
    """Penalty for repeated words. 1.0 means no penalty."""

    length_penalty: float = 1.0
    """Penalty to apply to the length. 1.0 means no penalty."""

    no_repeat_ngram_size: int = 0
    """Size of ngrams that can only occur once."""

    num_return_sequences: int = 1
    """Number of independently generated sequences to return."""

    stop_strings: str | list[str] | None = None
    """String or list of strings that will stop the generation."""

    def to_config(self) -> GenerationConfig:
        """Convert the specification to a GenerationConfig."""
        return GenerationConfig(**self.model_dump())
