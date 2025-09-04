"""Utility functions for testing."""

import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from training_wizard.commands.train_tokenizer import SentencePieceTrainer, TokenizerParams
from training_wizard.specs.spec import parse_config, parse_config_dict

if TYPE_CHECKING:
    from training_wizard.specs.base import TrainingRecipe


def run_variants(spec_path: Path, *variants: Callable):
    """Run a set of variants of the trainer tuning recipe.

    Args:
        spec_path: the path to the spec to use
        variants: a list of functions that take a spec and return an altered spec and a validator function
    """
    spec = parse_config_dict(spec_path)

    with tempfile.TemporaryDirectory() as temp_dir:
        if "training_args_spec" in spec:
            output_dir = temp_dir + "/output_ta"
            spec["training_args_spec"]["output_dir"] = output_dir
        elif "output_dir" in spec:
            output_dir = temp_dir + "/output"
            spec["output_dir"] = output_dir
        else:
            raise ValueError(f"No output_dir found in spec config: {spec_path}")
        output_dir_path = Path(output_dir)

        validators = []
        for variant in variants:
            spec, validator = variant(spec)
            validators.append(validator)

        from structlog.testing import capture_logs

        with capture_logs() as caplog:
            spec_instance: TrainingRecipe = parse_config(spec_path, spec)
            spec_instance.main()

        assert output_dir_path.exists(), f"Training did not create output directory: {output_dir}"

        text = "\n".join(e["event"] for e in caplog)
        for validator in validators:
            validator(text)


def run_train_tokenizer(spec_path: Path):
    """Run the training process for the tokenizer.

    Args:
        spec_path: The path to the spec file.
    """
    tokenizer_params = TokenizerParams(**parse_config_dict(spec_path))
    SentencePieceTrainer.Train(tokenizer_params.arguments)
