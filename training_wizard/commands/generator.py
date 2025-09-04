"""Generate training configs and commands for the training wizard."""

import argparse
import copy
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, Literal, Self

import tomli_w
from pydantic import BaseModel, field_validator, model_validator

from ..specs.spec import parse_config_dict


def write_toml(obj: dict[str, Any], path: Path):
    """Writes an object to a toml file."""
    with path.with_suffix(".toml").open(mode="wb") as f:
        tomli_w.dump(obj, f)


@dataclass
class ExperimentConfig:
    """Experiment configuration."""

    epochs: int | list[int]
    batch_size: int | list[int]
    dataset_size: int | list[int]
    warmup_steps_ratio: float | list[float]
    gradient_accumulation_steps: int | list[int]
    neftune_noise_alpha: float | list[float]
    optimizer: str | list[str]
    learning_rate: float | list[float]

    def check_all_lists_match(self, num_stages: int) -> bool:
        """Check that all lists have the same length."""
        for key in [attr for attr in dir(self) if not attr.startswith("__")]:
            attr = getattr(self, key)
            if not isinstance(attr, list):
                attr = [attr]
            curr_num = len(attr)
            if curr_num != num_stages and curr_num == 1:
                attr *= num_stages
            setattr(self, key, attr)
            assert len(getattr(self, key)) == num_stages, f"Length of {key} must match the length of name"
        return True


class GeneratorConfig(BaseModel):
    """Training configuration generator class."""

    stages: list[str]
    """Stages to be carried out.

    Each stage will be executed in the order it appears and it must be linked to a toml file.

    """

    base_path: Path
    """Base path where the experiments are to be located."""

    experiment_name: str
    base_model: str
    experiments: list[ExperimentConfig]
    mlflow_experiment_name: str | None = None
    name_template: str = """{base}_{experiment_name}_{experiment_index}_{stage}"""

    executor_command: str = "poetry run training-wizard"
    """You can change this to accelerate launch -m training_wizard, for example"""

    experiment_joiner: Literal["&&", "and", "||", "or"] = "||"
    stage_joiner: Literal["&&", "and", "||", "or"] = "&&"

    @field_validator("experiment_joiner", "stage_joiner")
    @classmethod
    def format_joiner(cls, v: str) -> str:
        """Format the joiner string for fish or bash execution."""
        return f"; {v} " if v in ("and", "or") else f" {v} "

    @model_validator(mode="after")
    def check_experiment_and_stages_length(self) -> Self:
        """Check that experiment lists and self.stages both have the same length."""
        assert all(e.check_all_lists_match(len(self.stages)) for e in self.experiments), (
            "All experiments must have the same number of stages."
        )
        return self

    @model_validator(mode="after")
    def create_experiment_base_path(self) -> Self:
        """Create the base path for the experiments."""
        self.base_path.mkdir(exist_ok=True, parents=True)
        return self

    @model_validator(mode="after")
    def check_experiment_base_config_exist(self) -> Self:
        """Check that the base config files exist."""
        stages = [Path(self.base_path / s).with_suffix(".toml") for s in self.stages]
        assert all(map(Path.exists, stages)), f"Could not find all base config files {stages}"
        return self

    @cached_property
    def base_configs(self) -> list[dict[str, Any]]:
        """Load the base configs."""
        return [parse_config_dict(Path(self.base_path / stage).with_suffix(".toml")) for stage in self.stages]

    @cached_property
    def model_names(self) -> list[list[str]]:
        """Generate the names for the experiment model outputs and run names."""
        return [
            [
                self.name_template.format(
                    base=self.base_model,
                    experiment_name=self.experiment_name,
                    experiment_index=experiment_index,
                    stage=stage,
                )
                for stage in self.stages
            ]
            for experiment_index in range(len(self.experiments))
        ]

    @cached_property
    def output_dirs(self) -> list[list[Path]]:
        """Generate the output directory."""
        return [
            [
                (self.base_path / stage / "output" / name).resolve()
                for stage, name in zip(self.stages, self.model_names[experiment_index])
            ]
            for experiment_index in range(len(self.experiments))
        ]

    @cached_property
    def pretrained_models(self) -> list[list[str]]:
        """Get the pretrain models with the base model at index 0 for piping output models."""
        return [
            [self.base_model, *(str(x / "final_checkpoint") for x in self.output_dirs[experiment_index])]
            for experiment_index in range(len(self.experiments))
        ]

    @classmethod
    def from_toml(cls, path: Path) -> Self:
        """Load the generator configuration from a toml file."""
        toml_path = path.with_suffix(".toml")
        return cls(**parse_config_dict(toml_path))

    def main(self) -> str:
        """Generate the configs and the commands."""
        config = []
        commands = []
        for experiment_index, e in enumerate(self.experiments):
            stages = []
            for stage_index in range(len(self.stages)):
                dataset_size = _maybe_index(e.dataset_size, stage_index)
                batch_size = _maybe_index(e.batch_size, stage_index)
                gradient_accumulation_steps = _maybe_index(e.gradient_accumulation_steps, stage_index)
                num_train_epochs = _maybe_index(e.epochs, stage_index)
                num_total_steps = (dataset_size / (batch_size * gradient_accumulation_steps)) * num_train_epochs
                config = copy.deepcopy(self.base_configs[stage_index])

                if self.mlflow_experiment_name:
                    config["training_args_spec"]["mlflow_experiment_name"] = self.mlflow_experiment_name

                if "preprocessing" not in config["dataset_spec"]:
                    config["dataset_spec"]["preprocessing"] = {}

                config["dataset_spec"]["preprocessing"]["take"] = dataset_size

                # these values are automatically assigned
                config["wizard_module"]["transformer_spec"]["pretrained_name"] = self.pretrained_models[
                    experiment_index
                ][stage_index]
                config["training_args_spec"]["output_dir"] = str(self.output_dirs[experiment_index][stage_index])
                config["training_args_spec"]["run_name"] = (
                    self.model_names[experiment_index][stage_index].replace("_", " ").title()
                )

                # here are the experiment values
                config["training_args_spec"]["per_device_train_batch_size"] = batch_size
                config["training_args_spec"]["per_device_eval_batch_size"] = batch_size
                config["training_args_spec"]["gradient_accumulation_steps"] = gradient_accumulation_steps
                config["training_args_spec"]["neftune_noise_alpha"] = _maybe_index(e.neftune_noise_alpha, stage_index)
                config["training_args_spec"]["warmup_steps"] = int(
                    num_total_steps * _maybe_index(e.warmup_steps_ratio, stage_index)
                )
                config["training_args_spec"]["learning_rate"] = _maybe_index(e.learning_rate, stage_index)
                config["training_args_spec"]["optim"] = _maybe_index(e.optimizer, stage_index)
                config["training_args_spec"]["num_train_epochs"] = num_train_epochs
                # create the config toml file
                self.output_dirs[experiment_index][stage_index].parent.mkdir(exist_ok=True, parents=True)
                write_toml(obj=config, path=self.output_dirs[experiment_index][stage_index])
                stages.append(f"{self.executor_command} {self.output_dirs[experiment_index][stage_index]}.toml")

            commands.append(self.stage_joiner.join(stages))

        command_string = self.experiment_joiner.join(commands)
        print(command_string)
        return command_string


def _maybe_index(value: Any | list[Any], index: int) -> Any:
    """Get the value at the index if it's a list, otherwise return the value."""
    return value[index] if isinstance(value, list) else value


def main():
    """Main command function."""
    parser = argparse.ArgumentParser(
        prog="generator",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("generator_config", type=Path, help="The path to the generator config")
    args = parser.parse_args()
    generator = GeneratorConfig.from_toml(args.generator_config)
    generator.main()
