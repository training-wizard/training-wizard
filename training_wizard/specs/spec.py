"""Base model for all specifications (spec)."""

import json
import tomllib
from abc import ABC
from logging import Logger
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, GetCoreSchemaHandler, model_serializer
from pydantic_core import CoreSchema, core_schema
from structlog import get_logger

from ..components.python_utils import get_class_path, resolve_class_path, update_nested_dict

logger: Logger = get_logger()


def parse_config_dict(config_path: str | Path) -> dict[str, Any]:
    """Read some configuration file into a dictionary.

    Supports TOML and JSON files.

    Args:
        config_path: Path to a config file.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config path {config_path} does not exist.")
    if config_path.suffix == ".json":
        return json.loads(config_path.read_text("utf-8"))
    elif config_path.suffix == ".toml":
        return tomllib.loads(config_path.read_text("utf-8"))
    else:
        raise ValueError(f"Config path {config_path} must have a .toml or .json extension. Got `{config_path.suffix}`.")


def parse_config(spec_path: str | Path, update: dict[str, Any] | None = None) -> Any:
    """Parse a configuration file into a Spec object.

    Args:
        spec_path: Path to some specification file. Also supports GCS paths.
        update: Optional dictionary to update the loaded config with, before passing it to `Spec.model_validate`.
                Supports nested updates.
    """
    update = update or {}
    base_config = parse_config_dict(spec_path)
    update_nested_dict(base_config, update)
    return Spec.model_validate(base_config)


class Spec(BaseModel, ABC):
    """Base spec for all specs."""

    model_config = ConfigDict(extra="forbid")
    """Disallow extra fields because they cause silent failures.

    Otherwise, mistyping a field name would not raise an error, it would just be ignored.
    """

    spec_class: str = Field(default=None, validate_default=False)  # type: ignore
    """Allows the user to specify the path to any subclass of the current type.

    Take, for example:
    ```python
    class Pasta(Spec):
        thickness: int

    class Penne(Pasta):
        doneness: str

    class Recipe(BaseModel):
        pasta: Pasta
    ```

    One can then specify the kind of pasta to use in TOML:
    ```toml
    # Recipe configuration
    [pasta]
    spec_class = "my_module.Penne"
    thickness = 2
    doneness = "al dente"
    ```"""

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: type["Spec"], handler: GetCoreSchemaHandler) -> CoreSchema:
        """Apply custom validation logic when loading the spec.

        The way it works is this:
        1. We check if the input is a dict (thus hasn't been instantiated yet)
        2. If yes, we find the class type that should validate it
        3. If we're not that type, we route it to the correct type's validator.

        Args:
            source_type: Source type. Should be the same as cls?
            handler: Handler for the core schema.

        Returns:
            The core schema with a custom pre-validator.
        """

        def custom_validator(value: Any) -> Any:
            if isinstance(value, dict) and value.get("spec_class"):
                spec_class = resolve_class_path(value["spec_class"])
                if spec_class != cls:
                    return spec_class.model_validate(value)
                # Optionally ensure the key is set consistently:
                value["spec_class"] = get_class_path(spec_class)
            return value

        return core_schema.no_info_before_validator_function(custom_validator, handler(source_type))

    @model_serializer
    def serialize(self) -> dict[str, Any]:
        """Serialize the spec to a dict."""
        result = {}
        for k in self.model_fields:
            v = getattr(self, k)
            if v is None:
                continue
            elif isinstance(v, BaseModel):
                result[k] = v.model_dump()
            else:
                result[k] = v
        return result

    @classmethod
    def _find_spec_class(cls, obj: Any) -> type["Spec"]:
        """Extract the spec class to use from the input dict.

        Raises:
            ValueError: If the spec class is invalid.

        Returns:
            The correct spec class.
        """
        spec_class: Any = cls
        assert isinstance(obj, dict), "Internal error: obj is not a dict, this should never happen."
        if "spec_class" in obj and obj["spec_class"] is not None:
            spec_class = resolve_class_path(obj["spec_class"])

        if ABC in spec_class.__bases__:
            raise TypeError(f"{spec_class} is an abstract class and cannot be instantiated.")
        elif not issubclass(spec_class, Spec):
            raise TypeError(f"{spec_class} is not a subclass of Spec")
        return spec_class
