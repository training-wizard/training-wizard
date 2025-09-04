"""Python utilities."""

import importlib
from collections.abc import Callable
from functools import partial
from logging import Logger
from typing import TypeVar

from structlog import get_logger

logger: Logger = get_logger()

T = TypeVar("T", bound=type)


def make_renamed_func(original_func: Callable, new_name: str) -> Callable:
    """Create a wrapper function with the new name."""
    # Using partial to avoid late binding issues in the closure
    wrapped = partial(original_func)
    wrapped.__name__ = new_name  # type: ignore
    return wrapped


def resolve_class_path(class_path: str) -> type[T]:  # type: ignore - This allows declaring the type, like `x: Type[X] = resolve_class_path(...)`
    """Return the class for the training recipe from the path."""
    # Split the module path to get the module and class name
    module_name, class_name = class_path.rsplit(".", 1)

    # Import the module dynamically
    module = importlib.import_module(module_name)

    # Get the class from the imported module
    cls = getattr(module, class_name)

    return cls


def get_class_path(cls: T | type[T]) -> str:
    """Get the class path for a class.

    Args:
        cls: The class to get the path for.

    Returns:
        The class path.
    """
    return cls.__module__ + "." + cls.__name__


def update_nested_dict(d: dict, u: dict):
    """Recursively update a nested dictionary.

    NOTE: This operation mutates the `d` dictionary.

    Args:
        d: Dictionary to update.
        u: Dictionary to update with.

    Returns:
        The updated dictionary.
    """
    for k in u:
        if k in d and isinstance(d[k], dict) and isinstance(u[k], dict):
            update_nested_dict(d[k], u[k])
        else:
            d[k] = u[k]
