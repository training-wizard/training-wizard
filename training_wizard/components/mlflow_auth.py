"""Module for setting up MLflow environment variables and authentication."""
import os
from logging import Logger

import structlog

logger: Logger = structlog.get_logger()

def set_mlflow_environment_variables(mlflow_experiment_name: str | None):
    """Validate the MLflow experiment name and set the environment variables accordingly."""
    if mlflow_experiment_name is not None and mlflow_experiment_name != "":
        raise NotImplementedError("MLflow authentication is not implemented.")
    else:
        logger.info("No MLflow experiment name provided. Disabling MLflow integration.")
        if "MLFLOW_EXPERIMENT_NAME" in os.environ:
            os.environ.pop("MLFLOW_EXPERIMENT_NAME")
        os.environ["DISABLE_MLFLOW_INTEGRATION"] = "TRUE"
