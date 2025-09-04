"""This module contains the callbacks used by the training wizard."""

import os
import random
import shutil
import sys
from collections.abc import Callable
from logging import Logger
from pathlib import Path
from shutil import copytree
from typing import TYPE_CHECKING, Any

import mlflow
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from structlog import get_logger
from transformers import PreTrainedModel, TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from ..components.reproducibility import save_reproducibility_information
from .utilities import DualWriter

if TYPE_CHECKING:
    from torch.utils.data.dataset import Dataset

logger: Logger = get_logger()

DEFAULT_DUAL_WRITER = DualWriter()
sys.stdout = DEFAULT_DUAL_WRITER


def print_output_sample(source: list[str], prediction: list[str], target: list[str] | None = None):
    """Print out a sample of source-prediction-target texts."""
    console = Console()
    table = Table(show_header=True, header_style="bold white", expand=True, padding=(0, 1, 1, 0))
    table.add_column("Source", style="bright_yellow")
    table.add_column("Prediction", style="turquoise2")
    if target is not None:
        table.add_column("Target", style="bright_green")
        for s, p, t in zip(source, prediction, target, strict=True):
            table.add_row(Text(s), Text(p), Text(t))
    else:
        for s, p in zip(source, prediction, strict=True):
            table.add_row(Text(s), Text(p))
    panel = Panel(table, expand=False, title="Output Sample", border_style="bold white")
    console.print(panel)


class SampleCallback(TrainerCallback):
    """This is a callback that will sample from the training dataset and log the results."""

    def __init__(self, generate_fn: Callable[[list[str]], list[str]], freq: int, sample_size: int = 3):
        """Initialize the callback.

        Args:
            generate_fn: The function to use for generating responses. Takes a history and returns an AI response.
            freq: How often to execute the callback (in steps).
            sample_size: Number of samples to generate.
        """
        super().__init__()
        self.response_fn = generate_fn
        self.freq = freq
        self.sample_size = sample_size
        self._cur = None

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Sample from the training dataset and log the results."""
        if state.is_world_process_zero and state.global_step % self.freq == 0 and state.global_step != self._cur:
            try:
                eval_ds: Dataset = kwargs["train_dataloader"].dataset
            except Exception:
                try:
                    eval_ds: Dataset = kwargs["eval_dataloader"].dataset
                except Exception:
                    logger.warning("No dataset found in kwargs, sample callback will not run")
                    return

            sample_rows: list[dict] = [eval_ds[i] for i in random.sample(range(len(eval_ds)), self.sample_size)]  # type: ignore
            source_str = [row["source"] for row in sample_rows]
            target_str = [row["target"] for row in sample_rows]

            # Generate predictions, one at a time to avoid OOM
            prediction_str = [self.response_fn([e])[0] for e in source_str]
            print_output_sample(source=source_str, prediction=prediction_str, target=target_str)


class SampleCallbackInstruct(TrainerCallback):
    """Callback for instruction tuning that samples from the training dataset and logs conversation outputs.

    This callback extracts the conversation history from the dataset's "messages" field,
    generates an AI response using the provided generation function, and prints a formatted output sample.
    """

    def __init__(
        self,
        generate_fn: Callable[[list[list[dict[str, str]]]], list[list[str]]],
        freq: int,
        sample_size: int = 3,
    ):
        """Initialize the callback.

        Args:
            generate_fn: The function to generate AI responses. It accepts a list of conversation histories,
                         where each history is a list of message dictionaries with keys 'role' and 'content',
                         and returns a list of generated responses.
            freq: Frequency (in training steps) at which to execute the callback.
            sample_size: Number of samples to generate.
        """
        super().__init__()
        self.response_fn = generate_fn
        self.freq = freq
        self.sample_size = sample_size
        self._cur: int | None = None

    def on_step_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ):
        """Sample and log results during training steps for instruction tuning models.

        Args:
            args: Training arguments.
            state: Current state of the trainer.
            control: Control object for trainer callbacks.
            **kwargs: Additional keyword arguments (should contain the dataloader with the dataset).
        """
        if state.is_world_process_zero and state.global_step % self.freq == 0 and state.global_step != self._cur:
            try:
                eval_ds: Dataset = kwargs["train_dataloader"].dataset
            except Exception:
                try:
                    eval_ds: Dataset = kwargs["eval_dataloader"].dataset
                except Exception:
                    logger.warning("No dataset found in kwargs, sample callback will not run")
                    return

            # Sample a few rows from the evaluation dataset.
            sample_rows: list[dict] = [
                eval_ds[i]
                for i in random.sample(range(len(eval_ds)), self.sample_size)  # type: ignore
            ]
            # For instruction tuning, extract the conversation histories from the 'messages' field.
            messages_input: list[list[dict[str, str]]] = [row["messages"][:-1] for row in sample_rows]
            messages_target: list[dict[str, str]] = [row["messages"][-1] for row in sample_rows]

            # Generate predictions for each conversation history, wrapping each history in a list.
            prediction_str: list[str] = [beams[0] for beams in self.response_fn(messages_input)]

            # Format the conversation history for display.
            formatted_messages: list[str] = [
                "\n\n".join(f"{msg['role'].upper()}: {msg['content']}" for msg in msgs) for msgs in messages_input
            ]
            print_output_sample(
                source=formatted_messages, prediction=prediction_str, target=[msg["content"] for msg in messages_target]
            )
            self._cur = state.global_step


class EvaluateFirstStepCallback(TrainerCallback):
    """This is a callback that will evaluate the model on the first step."""

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """After every step, check if we should evaluate."""
        if state.global_step == 1:
            control.should_evaluate = True


class SaveInfoBegin(TrainerCallback):
    """This is a callback that will save reproducibility information before training starts."""

    def __init__(self, mlflow_experiment_name: str | None = None):
        """Initialize the callback.

        Args:
            mlflow_experiment_name: The name of the MLflow experiment to log to.
        """
        super().__init__()
        self.mlflow_experiment_name = mlflow_experiment_name

    def _save_reproducibility_directory(self, args: TrainingArguments):
        """Save the reproducibility information to the output directory."""
        assert args.output_dir is not None, "output_dir is required"
        save_reproducibility_information(args.output_dir)
        reproducibility_dir = os.path.join(args.output_dir, "reproducibility")
        logger.info("Saved reproducibility information to %s", reproducibility_dir)
        if self.mlflow_experiment_name:
            mlflow.log_artifacts(local_dir=reproducibility_dir, artifact_path="reproducibility")
            logger.info("Uploaded reproducibility information to MLflow")

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Save the reproducibility information before training starts."""
        if state.is_world_process_zero:
            logger.info("Saving reproducibility information...")
            self._save_reproducibility_directory(args)


class SaveInfoEnd(TrainerCallback):
    """This is a callback that will save the model, the tokenizer as well as the logs after training."""

    def __init__(
        self,
        dual_writer: DualWriter = DEFAULT_DUAL_WRITER,
        mlflow_experiment_name: str | None = None,
        merge_lora: bool = False,
    ):
        """Initialize the callback.

        Args:
            dual_writer: The dual writer to use for logging.
            mlflow_experiment_name: The name of the MLflow experiment to log to.
            merge_lora: If True and the model is a PEFT model, merge LoRA weights into the base model before saving.
        """
        super().__init__()
        self.dual_writer = dual_writer
        self.mlflow_experiment_name = mlflow_experiment_name
        self.merge_lora = merge_lora

    def on_train_end(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, model: PreTrainedModel, **kwargs
    ):
        """Save the model, tokenizer and logs after training finishes."""
        tokenizer = kwargs.get("processing_class", kwargs.get("tokenizer"))
        if state.is_world_process_zero:
            assert args.output_dir is not None, "output_dir is required"
            os.makedirs(args.output_dir, exist_ok=True)
            save_dir = Path(os.path.join(args.output_dir, "final_checkpoint/"))
            save_dir.mkdir(parents=True, exist_ok=True)

            if self.merge_lora and hasattr(model, "merge_adapter"):
                logger.info("Merging LoRA weights into base model.")
                # Try to merge LoRA weights using the most appropriate method available
                if hasattr(model, "merge_and_unload") and callable(model.merge_and_unload):
                    # Preferred method: merge weights and unload adapter layers
                    model = model.merge_and_unload()
                elif hasattr(model, "merge_adapter") and callable(model.merge_adapter):
                    # Alternative method: just merge the weights
                    model.merge_adapter()
                # Clear adapter configuration so that the saved model doesn't include LoRA adapter info.
                if hasattr(model, "config") and hasattr(model.config, "peft_config"):
                    model.config.peft_config = None
                model.save_pretrained(save_dir)
                logger.info("Saved merged model to %s", save_dir)
                if tokenizer is not None:
                    tokenizer.save_pretrained(save_dir)
                    logger.info("Saved tokenizer to %s", save_dir)
                else:
                    logger.warning("MLflow Callback: Trainer didn't provide tokenizer, so it won't be saved.")
            else:
                last_or_best = "best" if args.load_best_model_at_end else "last"
                logger.info("Saving %s checkpoint of the model to %s", last_or_best, save_dir)
                saved = False
                if state.best_model_checkpoint is not None and args.load_best_model_at_end:
                    best_model_checkpoint = Path(state.best_model_checkpoint)
                    if best_model_checkpoint.exists():
                        logger.info("Copying best model checkpoint to %s", save_dir)
                        copytree(best_model_checkpoint, save_dir, dirs_exist_ok=True)
                        for extra in ["optimizer.pt", "scheduler.pt", "training_args.bin"]:
                            if (save_dir / extra).exists():
                                logger.info("Removing unnecessary file %s", extra)
                                (save_dir / extra).unlink()
                        # DeepSpeed checkpoint files
                        for dir_path in save_dir.glob("global_step_*"):
                            if dir_path.is_dir():
                                shutil.rmtree(dir_path)
                        saved = True
                if not saved:
                    logger.info("MLflow Callback: Saving model and tokenizer manually")
                    model.save_pretrained(save_dir)
                    logger.info("Saved model to %s", save_dir)
                    if tokenizer is not None:
                        tokenizer.save_pretrained(save_dir)
                        logger.info("Saved tokenizer to %s", save_dir)
                    else:
                        logger.warning("MLflow Callback: Trainer didn't provide tokenizer, so it won't be saved.")

            out_file = Path(args.output_dir) / "run.log"
            logger.info("Saving training logs to %s", out_file)
            out_file.write_text(self.dual_writer.getvalue(), encoding="utf-8")

            if self.mlflow_experiment_name:
                mlflow.log_artifact(str(out_file))
                logger.info("Uploaded training logs to MLflow")
                logger.info("Uploading model to MLflow. This may take a while...")
                open(save_dir / "MLmodel", "a").close()
                mlflow.log_artifacts(str(save_dir), artifact_path="model")
                logger.info("Uploaded model artifacts to MLflow")


class PretrainCallback(SampleCallback):
    """This is a callback that will sample from the training dataset and log the results."""

    def __init__(self, *args, random_seed: int | None = None, **kwargs):
        """Initialize the callback.

        Args:
            args: arguments for the SampleCallback base class
            random_seed: Seed for the callback to pick random sentences
            kwargs: Arguments for the SampleCallback base class
        """
        super().__init__(*args, **kwargs)
        self.random_seed = random_seed

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Sample from the training dataset and log the results."""
        if state.is_world_process_zero and state.global_step % self.freq == 0 and state.global_step != self._cur:
            try:
                eval_ds: Dataset = kwargs["train_dataloader"].dataset
            except Exception:
                try:
                    eval_ds: Dataset = kwargs["eval_dataloader"].dataset
                except Exception:
                    logger.warning("No dataset found in kwargs, sample callback will not run")
                    return
            sources, hypothesis, targets = [], [], []
            rng = random.Random(self.random_seed) if self.random_seed is not None else random
            for i in rng.sample(range(len(eval_ds)), self.sample_size):  # type: ignore
                # Generate predictions, one at a time to avoid OOM
                source = eval_ds[i]
                p = self.response_fn([source])
                hypothesis.append(p["predictions"][0])  # type: ignore
                sources.append(p["sources"][0])  # type: ignore
                targets.append(p["targets"][0])  # type: ignore
            print_output_sample(source=sources, prediction=hypothesis, target=targets)
