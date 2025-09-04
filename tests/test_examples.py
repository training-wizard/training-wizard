"""Test the examples with the cli."""

from pathlib import Path

import pytest

from .utils import run_train_tokenizer, run_variants


def test_not_found():
    """Test that the cli fails with a FileNotFoundError if the config file does not exist."""
    try:
        run_variants(Path("this/should/not/exist/config.toml"))
        raise AssertionError("Expected a FileNotFoundError")
    except FileNotFoundError:
        pass


def get_example_directories() -> list[Path]:
    """Get all directories in the examples folder that contain a config.toml file."""
    examples_dir = Path("examples")
    return [d for d in examples_dir.iterdir() if d.is_dir() and (d / "config.toml").exists()]


@pytest.mark.parametrize("example_dir", get_example_directories(), ids=lambda example_dir: example_dir.name)
def test_examples(pytestconfig: pytest.Config, example_dir: Path):
    """Test an example directory.

    Args:
        pytestconfig: The pytest config.
        example_dir: The path to the example directory to test.
    """
    if example_dir.name == "quantization":
        try:
            import llmcompressor  # type: ignore # noqa: F401
        except ImportError:
            pytest.skip("llmcompressor is not installed, skipping quantization example")
    if example_dir.name == "train_tokenizer":
        run_train_tokenizer(example_dir / "config.toml")
    else:
        run_variants(example_dir / "config.toml")
