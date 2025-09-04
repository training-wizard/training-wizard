"""Train a tokenizer with a custom dataset."""

import argparse
from functools import cached_property
from pathlib import Path
from typing import Literal

from pydantic import BaseModel
from sentencepiece import SentencePieceTrainer

from ..specs.spec import parse_config_dict


class TokenizerParams(BaseModel, extra="allow"):
    """Define arguments for the SentencepieceTrainer using a toml configuration file.

    There are more arguments available. These are in line with the PTT5 codebase,
    and the defaults are set to those instead of  the upstream stentencepiece.

    Extra arguments are allowed, but the key should match exactly those of the train method.
    """

    input: Path
    model_prefix: str
    vocab_size: int = 32000
    input_sentence_size: int = 2000000
    shuffle_input_sentence: bool = True
    pad_id: int = 0
    eos_id: int = 1
    unk_id: int = 2
    bos_id: int = -1
    pad_piece: str = "<pad>"
    unk_piece: str = "<unk>"
    eos_piece: str = "</s>"
    character_coverage: float = 1.0
    model_type: Literal["unigram", "bpe", "char", "word"] = "unigram"

    @cached_property
    def arguments(self) -> str:
        """Conform the argument string for `SentencePieceTrainer.train`."""
        base_args = self.model_dump()
        base_args.update(self.__pydantic_extra__ or {})
        return " ".join([f"--{k}={v}" for k, v in base_args.items()])


def main():
    """Train a tokenizer with a custom dataset."""
    parser = argparse.ArgumentParser()
    parser.add_argument("tokenizer_config", type=Path, help="Path to the configuration file for the tokenizer training")
    args = parser.parse_args()
    tokenizer_params = TokenizerParams(**parse_config_dict(args.tokenizer_config))

    SentencePieceTrainer.Train(tokenizer_params.arguments)


if __name__ == "__main__":
    main()
