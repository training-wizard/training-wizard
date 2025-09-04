"""Custom Training Wizard Trainer class with bells and whistles. More to come."""

import signal
from collections import deque
from logging import Logger
from pathlib import Path
from statistics import mean
from typing import TYPE_CHECKING, Any, Literal, cast

import mlflow
import structlog
import torch
from datasets import Dataset
from matplotlib.figure import Figure
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, Trainer
from transformers.trainer import (
    TrainingArguments,
    _is_peft_model,
    can_return_loss,
    find_labels,
)

from ..specs.modules.module import WizardModule

if TYPE_CHECKING:
    from peft.peft_model import PeftModel

logger: Logger = structlog.get_logger()


class WizardTrainer(Trainer):
    """Custom trainer."""

    def __init__(
        self,
        args: TrainingArguments,
        train_dataset: Dataset,
        eval_dataset: Dataset | None,
        training_module: WizardModule,
        **kwargs,
    ):
        """Create a custom trainer."""
        model = training_module.model
        super().__init__(
            model=model,
            args=args,
            data_collator=training_module.data_collator,
            train_dataset=train_dataset,  # type: ignore
            eval_dataset=eval_dataset,  # type: ignore
            processing_class=training_module.tokenizer,
            **kwargs,
        )
        assert isinstance(args.logging_steps, int), "Logging steps must be an integer."
        assert args.logging_strategy == "steps", "Logging strategy must be steps."
        assert args.remove_unused_columns is False, "Unused columns must not be removed."
        self.training_module = training_module

        # Workaround for https://github.com/huggingface/transformers/pull/33422
        if _is_peft_model(self.model):
            if hasattr(self.model, "get_base_model"):
                model_to_inspect = cast("PeftModel", self.model).get_base_model()
                default_label_names = find_labels(model_to_inspect.__class__)
                self.can_return_loss = can_return_loss(model_to_inspect.__class__)
        else:
            default_label_names = find_labels(self.model.__class__)
            self.can_return_loss = can_return_loss(self.model.__class__)
        self.label_names = default_label_names if self.args.label_names is None else self.args.label_names

        self._logged_at = set()
        self._plot_cnt = 0
        self._metric_history: dict[str, deque[float]] = {}

    def create_optimizer(self) -> Optimizer:
        """Create optimizer, using custom one from module if provided."""
        if self.training_module.optimizer is not None:
            logger.info("Using custom optimizer from training module")
            self.optimizer = self.training_module.optimizer
        else:
            super().create_optimizer()
        return self.optimizer

    def create_scheduler(
        self, num_training_steps: int, optimizer: torch.optim.Optimizer | None = None
    ) -> LRScheduler | Any:
        """Create scheduler, using custom one from module if provided."""
        if self.training_module.lr_scheduler is not None:
            logger.info("Using custom lr_scheduler from training module")
            self.lr_scheduler = self.training_module.lr_scheduler
        else:
            # Use the provided optimizer or fall back to self.optimizer
            opt = optimizer if optimizer is not None else self.optimizer
            self.lr_scheduler = super().create_scheduler(num_training_steps, opt)
        return self.lr_scheduler

    def get_train_dataloader(self) -> DataLoader:
        """Get the train dataloader.

        Fixes current issue with callback handler train_dataloader not being set.
        """
        retval = super().get_train_dataloader()
        self.callback_handler.train_dataloader = retval  # type: ignore
        return retval

    def train(self, *args: Any, **kwargs: Any) -> dict[str, float]:
        """Run the training loop with a graceful SIGINT handler."""
        original_sigint_handler = signal.getsignal(signal.SIGINT)

        def _graceful_sigint_handler(signum: int, frame: Any | None):
            if not self.control.should_training_stop:
                logger.warning(
                    "SIGINT received. Finishing current step, then stopping training. "
                    "Press Ctrl+C again to exit immediately."
                )
                self.control.should_training_stop = True
            else:
                logger.warning("Second SIGINT received. Exiting forcefully.")
                raise KeyboardInterrupt

        if self.is_world_process_zero():
            signal.signal(signal.SIGINT, _graceful_sigint_handler)

        try:
            result = super().train(*args, **kwargs)
        except KeyboardInterrupt:
            logger.warning("Training interrupted forcefully.")
            result = {}
        finally:
            if self.is_world_process_zero():
                signal.signal(signal.SIGINT, original_sigint_handler)

        return result

    def log_plots(self, metrics_plots: dict[str, Figure]):
        """Log plots."""
        if self.is_world_process_zero():
            logger.info("Logging plots at step %s", self.state.global_step)
            log_mlflow = False
            if mlflow.active_run() and self.args.report_to and "mlflow" in self.args.report_to:
                logger.info("Logging plots to MLflow as well.")
                log_mlflow = True
            assert self.args.output_dir is not None, "output_dir is required"
            plot_dir = Path(self.args.output_dir) / f"plots/{self._plot_cnt}_step_{self.state.global_step}/"
            for name, plot in metrics_plots.items():
                plot_dir.mkdir(parents=True, exist_ok=True)
                plot.savefig(plot_dir / f"{name}.png")
                if log_mlflow:
                    mlflow.log_figure(plot, f"plots/{self._plot_cnt}_step_{self.state.global_step}/{name}.png")
            self._plot_cnt += 1

    def _update_metric_history(self, metrics: dict[str, float]) -> dict[str, float]:
        """Update the metric history and return smoothed metrics."""
        smoothed_metrics = {}
        for k, v in metrics.items():
            if k not in self._metric_history:
                self._metric_history[k] = deque(maxlen=10)
            self._metric_history[k].append(v)
            smoothed_metrics[k] = mean(self._metric_history[k])
        return smoothed_metrics

    def _update_metrics(self, metrics: dict[str, float | Figure]):
        """Update the metric history and log the metrics."""
        self._logged_at.add(self.state.global_step)
        metrics_float = {k: v for k, v in metrics.items() if isinstance(v, float)}
        metrics_plots = {k: v for k, v in metrics.items() if isinstance(v, Figure)}

        if metrics_float:
            smoothed_metrics = self._update_metric_history(metrics_float)
            self.log(smoothed_metrics)

        if metrics_plots:
            self.log_plots(metrics_plots)

    def compute_loss(
        self, model: PreTrainedModel, inputs: dict[str, Any], return_outputs: bool = False, **kwargs
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        """Compute the loss for the model."""
        inputs_for_metrics = inputs.copy()
        loss, outputs = self.training_module.compute_loss(self, inputs, True)

        if (
            model.training
            and self.state.global_step % self.args.logging_steps == 0
            and self.is_world_process_zero()
            and self.state.global_step not in self._logged_at
        ):
            metrics: dict[str, float | Figure] = self.training_module.batch_metrics(inputs_for_metrics, outputs)
            self._update_metrics(metrics)

        return loss if not return_outputs else (loss, outputs)

    def evaluate(
        self,
        eval_dataset: Dataset | None = None,
        ignore_keys: list[str] | None = None,
        metric_key_prefix: Literal["eval"] = "eval",
    ) -> dict[str, float]:
        """Evaluate the model and return combined metrics for early stopping compatibility.

        This method runs both custom evaluation metrics from the training module
        and standard Trainer evaluation, then combines and logs all metrics together
        to ensure early stopping callbacks can access the required metrics.
        """
        # Validate dataset
        dataset = eval_dataset if eval_dataset is not None else cast("Dataset", self.eval_dataset)
        if dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        # Get custom metrics from training module
        custom_metrics: dict[str, float | Figure] = {}
        if hasattr(self.training_module, "compute_eval_metrics") and getattr(
            self.training_module, "compute_eval_metrics", True
        ):
            custom_metrics = self.training_module.eval_metrics(dataset, batch_size=self.args.per_device_eval_batch_size)

        # Get standard metrics (eval_loss, etc.) from base trainer
        eval_dataloader = self.get_eval_dataloader(cast("Any", dataset))
        eval_output = self.evaluation_loop(
            eval_dataloader,
            description="Evaluation",
            prediction_loss_only=True,
            metric_key_prefix=metric_key_prefix,
        )

        # Combine all metrics with proper prefixes
        combined_metrics = {f"{metric_key_prefix}_{k}": v for k, v in custom_metrics.items()}
        if eval_output.metrics:
            combined_metrics.update(eval_output.metrics)

        # Separate float metrics from plots
        float_metrics = {k: v for k, v in combined_metrics.items() if isinstance(v, float)}
        plot_metrics = {k: v for k, v in combined_metrics.items() if isinstance(v, Figure)}

        # Log metrics and plots (only on main process)
        if self.is_world_process_zero():
            smoothed_metrics = self._update_metric_history(float_metrics)
            self.log(smoothed_metrics)
            if plot_metrics:
                self.log_plots(plot_metrics)

        # Trigger evaluation callbacks
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, float_metrics)

        return float_metrics

    def prediction_step(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        prediction_loss_only: bool,
        ignore_keys: list[str] | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """Perform an evaluation step on `model` using `inputs`."""
        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if self.model is not None and hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", ["past_key_values"])
            else:
                ignore_keys = []

        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss, outputs = self.compute_loss(cast("PreTrainedModel", model), inputs, return_outputs=True)
            loss = loss.mean().detach()

        if prediction_loss_only:
            return (loss, None, None)

        if isinstance(outputs, dict):
            assert ignore_keys is not None
            logits = tuple(v for k, v in outputs.items() if k not in [*ignore_keys, "loss"])
        else:
            logits = outputs[1:]
        if len(logits) == 1:
            logits = logits[0]

        return (loss, cast("torch.Tensor | None", logits), inputs.get("labels", None))
