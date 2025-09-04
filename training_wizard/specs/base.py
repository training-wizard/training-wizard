"""Base specifications for training."""

from abc import ABC, abstractmethod
from logging import Logger

from structlog import get_logger

from .spec import Spec

logger: Logger = get_logger()


class TrainingRecipe(Spec, ABC):
    """Specification for a training recipe."""

    @abstractmethod
    def main(self):
        """Run the training recipe."""
        ...
