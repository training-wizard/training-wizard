"""Quantize a LLM using INT8/INT4 quantization.

This recipe implements model quantization using the llm-compressor library to reduce model size
and improve inference performance. It supports both W8A8 (8-bit weights and activations) and
W4A16 (4-bit weights with 16-bit activations) quantization schemes.

The quantization process uses two main components:
1. SmoothQuant - Makes activations easier to quantize by adjusting the weight scales
2. GPTQ - Performs the actual quantization of weights and activations
"""

import contextlib
import os
import shutil
from logging import Logger
from typing import Any, Literal

from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier

try:
    from llmcompressor import oneshot  # type: ignore
except ImportError:
    # For llmcompressor > 0.5.0, the API changes, so we need to import from the new location
    from llmcompressor.transformers.finetune import oneshot

from pydantic import field_validator
from structlog import get_logger

from ..specs.base import TrainingRecipe
from ..specs.dataset import DataSourceSpec
from ..specs.modules.causal_seq2seq import CausalSeq2SeqModule
from ..specs.modules.instruction_tune import InstructionTuningModule
from ..specs.modules.module import WizardModule

logger: Logger = get_logger()


def uncollate(batch: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Uncollate a batch."""
    return [dict(zip(batch, t)) for t in zip(*batch.values())]


def _is_local_dir(p: str | None) -> bool:
    """Check if a path is a local directory."""
    return isinstance(p, str) and os.path.isdir(p)


@contextlib.contextmanager
def bind_hf_assets_locally(model: Any, tokenizer: Any, base_dir: str):
    """Ensure model/tokenizer 'name_or_path' fields point to a local directory.

    This avoids a bug in llm-compressor that fails to save models that were loaded over the HuggingFace Hub.

    Args:
        model: The model whose config._name_or_path needs to be localized.
        tokenizer: The tokenizer whose name_or_path needs to be localized.
        base_dir: Base directory to create the local snapshot in.
    """
    from huggingface_hub import snapshot_download

    # Remember originals
    old_model_nop = getattr(getattr(model, "config", None), "_name_or_path", None)
    old_tok_nop = getattr(tokenizer, "name_or_path", None)
    old_gen_nop = getattr(getattr(model, "generation_config", None), "_name_or_path", None)

    need_local = not (_is_local_dir(old_model_nop) and _is_local_dir(old_tok_nop))
    local_dir = None

    try:
        if need_local:
            # Pick a repo ID to snapshot (prefer tokenizer repo id if available)
            repo_id = old_tok_nop or old_model_nop
            if not isinstance(repo_id, str):
                raise ValueError("Cannot infer repo id; please pass models/tokenizers loaded from HF.")

            local_dir = os.path.join(base_dir, ".hf_snapshot")
            os.makedirs(local_dir, exist_ok=True)

            # Materialize a local snapshot of the repo (no model reload)
            snapshot_path = snapshot_download(
                repo_id=repo_id,
                local_dir=local_dir,
                local_dir_use_symlinks=False,  # safer for later file ops
            )

            # Point all 'name_or_path' anchors to the local folder
            if hasattr(model, "config"):
                model.config._name_or_path = snapshot_path
            if hasattr(model, "generation_config") and model.generation_config is not None:
                model.generation_config._name_or_path = snapshot_path
            tokenizer.name_or_path = snapshot_path

        yield
    finally:
        # Restore originals
        if hasattr(model, "config"):
            model.config._name_or_path = old_model_nop
        if hasattr(model, "generation_config") and model.generation_config is not None:
            model.generation_config._name_or_path = old_gen_nop
        tokenizer.name_or_path = old_tok_nop

        # Clean up the snapshot
        if local_dir and os.path.isdir(local_dir):
            shutil.rmtree(local_dir, ignore_errors=True)


class LLMQuantizationSpec(TrainingRecipe):
    """Recipe for quantizing LLM models.

    This recipe applies quantization to reduce model size while preserving accuracy.
    It uses a combination of SmoothQuant and GPTQ algorithms from the llm-compressor library.
    The quantized model can be used with vLLM for efficient inference.
    """

    wizard_module: CausalSeq2SeqModule | InstructionTuningModule | WizardModule
    """The module to quantize. Only CausalSeq2SeqModule and InstructionTuningModule are tested to work."""

    dataset_spec: DataSourceSpec
    """The dataset to use for training."""

    output_dir: str
    """The output directory to save the quantized model to."""

    scheme: Literal["W8A8", "W4A16"] = "W8A8"
    """The scheme to use for quantization."""

    num_calibration_samples: int = 512
    """The number of calibration samples to use."""

    batch_size: int = 1
    """The batch size to use for calibration."""

    @field_validator("output_dir")
    @classmethod
    def check_output_dir(cls, v: str) -> str:
        """Check if the output directory exists."""
        if os.path.exists(v):
            raise ValueError(f"Output directory '{v}' already exists. Please specify a new directory.")
        return v

    def main(self):
        """Run the quantization procedure.

        The process involves:
        1. Loading the model and tokenizer
        2. Preparing calibration data from the specified dataset
        3. Applying SmoothQuant to adjust activation scales
        4. Quantizing the model using GPTQ
        5. Saving the quantized model in a vLLM-compatible format
        """
        logger.info("Loading model and tokenizer...")
        model = self.wizard_module.model
        tokenizer = self.wizard_module.tokenizer

        # Bind HF assets locally to ensure name_or_path fields point to local directories
        # This avoids the need to save/reload the entire model while fixing llm-compressor compatibility
        assets_base = os.path.join(self.output_dir, "pretrained_assets")
        with bind_hf_assets_locally(model, tokenizer, assets_base):
            logger.info("Preparing calibration data...")
            train_ds = self.dataset_spec.dataset
            if len(train_ds) < self.num_calibration_samples:
                logger.warning(
                    f"Not enough training samples for calibration. "
                    f"Wanted {self.num_calibration_samples}, got {len(train_ds)}. Using all available samples."
                )

            logger.info(f"Using {self.num_calibration_samples} samples for calibration")
            train_ds = train_ds.select(range(min(self.num_calibration_samples, len(train_ds)))).map(
                lambda batch: self.wizard_module.data_collator(uncollate(batch)),
                remove_columns=train_ds.column_names,
                batched=True,
                batch_size=self.batch_size,
                desc="Preparing calibration data",
            )

            logger.info("Applying quantization...")
            recipe = [
                SmoothQuantModifier(smoothing_strength=0.8),
                GPTQModifier(targets="Linear", scheme=self.scheme, ignore=["lm_head"]),
            ]

            os.makedirs(self.output_dir, exist_ok=True)
            logger.info(f"Temporarily setting current working directory to {self.output_dir}")
            current_dir = os.getcwd()
            os.chdir(self.output_dir)
            try:
                oneshot(
                    model=model,
                    dataset=train_ds,
                    recipe=recipe,
                    max_seq_length=tokenizer.model_max_length,
                    num_calibration_samples=self.num_calibration_samples,
                    output_dir="model",
                )
            finally:
                os.chdir(current_dir)
                logger.info(f"Restored current working directory to {current_dir}")

        logger.info(f"Quantization complete! Output saved to {os.path.join(self.output_dir, 'model')}")
