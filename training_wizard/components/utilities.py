"""Common utilities."""

import importlib
import io
import math
import re
import sys
from collections.abc import Callable, Iterable
from contextlib import contextmanager
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Literal, TypeVar, cast

import structlog
import torch
import tqdm
from peft.config import PeftConfig
from peft.peft_model import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

logger = structlog.get_logger()


if TYPE_CHECKING:
    from peft.tuners.lora import LoraConfig


def escape_ansi(line: str) -> str:
    """Remove ANSI escape sequences from a string."""
    ansi_escape = re.compile(r"(?:\x1B[@-_]|[\x80-\x9F])[0-?]*[ -/]*[@-~]")
    return ansi_escape.sub("", line)


class DualWriter:
    """A writer that writes to both stdout and a buffer."""

    def __init__(self):
        """Initialize the dual writer."""
        self.terminal = sys.stdout
        self.mem = io.StringIO()

    def write(self, message: str):
        """Write to both stdout and the buffer."""
        self.terminal.write(message)
        self.mem.write(escape_ansi(message))

    def flush(self):
        """Flush both stdout and the buffer."""
        self.terminal.flush()
        self.mem.flush()

    def getvalue(self) -> str:
        """Get the value of the buffer."""
        self.mem.flush()
        return self.mem.getvalue()

    def isatty(self) -> bool:
        """Return whether the terminal is a TTY."""
        return hasattr(self.terminal, "isatty") and self.terminal.isatty()


@lru_cache(None)
def log_once(log_level: Literal["debug", "info", "warning", "error"], msg: str):
    """Log a warning message once."""
    if log_level == "debug":
        logger.debug(msg)
    elif log_level == "info":
        logger.info(msg)
    elif log_level == "warning":
        logger.warning(msg)
    elif log_level == "error":
        logger.error(msg)
    else:
        logger.error("Invalid log level %s while trying to log message: %s", log_level, msg)


T = TypeVar("T")


def resolve_class_path(class_path: str) -> type[T]:  # type: ignore - This allows declaring the type, like `x: Type[X] = resolve_class_path(...)`
    """Return the class for the training recipe from the path."""
    # Split the module path to get the module and class name
    module_name, class_name = class_path.rsplit(".", 1)

    # Import the module dynamically
    module = importlib.import_module(module_name)

    # Get the class from the imported module
    cls = getattr(module, class_name)

    return cls


@contextmanager
def temporary_dict_update(d: dict[str, Any], **kwargs: Any) -> Any:
    """Temporarily update a dictionary.

    Args:
        d: The dictionary to temporarily update.
        **kwargs: The key-value pairs to temporarily add to the dictionary.

    Yields:
        The dictionary with the temporary updates.
    """
    old = d.copy()
    try:
        d.update(kwargs)
        yield d
    finally:
        d.clear()
        d.update(old)


def make_tensors_contiguous(model: torch.nn.Module):
    """Make all tensors in the model contiguous.

    Args:
        model: The model to make the tensors contiguous for.
    """
    for _, param in model.named_parameters():
        if not param.is_contiguous():
            param.data = param.data.contiguous()


def chars_token_ratio(dataset: Iterable[str], tokenizer: PreTrainedTokenizerBase, nb_examples: int = 400) -> float:
    """Estimate the average number of characters per token in the dataset.

    Args:
        dataset: Any string iterable.
        tokenizer: The tokenizer to use to tokenize the dataset.
        nb_examples: The number of examples to use to estimate the average number of characters per token. Defaults to 400.

    Returns:
        The average number of characters per token in the dataset.
    """  # noqa: E501
    total_characters, total_tokens = 0, 0
    for _, text in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples):  # type: ignore
        total_characters += len(text)
        if tokenizer.is_fast:
            total_tokens += len(tokenizer(text).tokens())
        else:
            total_tokens += len(tokenizer.tokenize(text))

    return total_characters / total_tokens


def dict_list_to_list_dict(x: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Convert a dictionary of lists into a list of dictionaries.

    Args:
        x: A dictionary of lists.

    Returns:
        A list of dictionaries.
    """
    batch_size = len(x[next(iter(x.keys()))])
    assert all(len(v) == batch_size for v in x.values()), "All lists in the dictionary must have the same length."
    x_list = [{k: v[i] for k, v in x.items()} for i in range(batch_size)]
    return x_list


def list_dict_to_dict_list(x: list[dict[str, Any]], keys: Iterable[str]) -> dict[str, list[Any]]:
    """Convert a list of dictionaries into a dictionary of lists.

    Args:
        keys: The keys to extract from the dictionaries.
        x: A list of dictionaries. MUST be non-empty.

    Returns:
        A dictionary of lists.
    """
    x_dict = {k: [d[k] for d in x] for k in keys}
    return x_dict


def merge_adapter_main_models(adapter_model_name: str, base_model_name: str, output_name: str):
    """Merge the adapter with the base model.

    Args:
        adapter_model_name: The name of the adapter model.
        base_model_name: The name of the base model.
        output_name: The name of the merged model.
    """
    peft_config = PeftConfig.from_pretrained(adapter_model_name)
    if peft_config.task_type == "SEQ_CLS":
        # peft is for reward model so load sequence classification
        model = AutoModelForSequenceClassification.from_pretrained(
            base_model_name, num_labels=1, torch_dtype=torch.bfloat16
        )
    elif peft_config.task_type == "SEQ_2_SEQ_LM":
        model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name, torch_dtype=torch.bfloat16)
    else:
        model = AutoModelForCausalLM.from_pretrained(base_model_name, return_dict=True, torch_dtype=torch.bfloat16)

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    # Load the Lora model
    model = PeftModel.from_pretrained(model, adapter_model_name)
    model.eval()

    model = model.merge_and_unload()  # type: ignore

    model.save_pretrained(output_name)
    tokenizer.save_pretrained(output_name)


millnames = ["", " Thousand", " Million", " Billion", " Trillion"]


def millify(n: float) -> str:
    """Convert a number into human-readable format."""
    n = float(n)
    millidx = max(
        0,
        min(len(millnames) - 1, math.floor(0 if n == 0 else math.log10(abs(n)) / 3)),
    )

    return f"{n / 10 ** (3 * millidx):.2f}{millnames[millidx]}"


def print_trainable_parameters(model: torch.nn.Module):
    """Print the number of trainable parameters in the model.

    Args:
        model: The model to print the number of trainable parameters for.
    """
    if hasattr(model, "num_parameters"):
        model = cast("PreTrainedModel", model)
        all_parameters = model.num_parameters()
        trainable_parameters = model.num_parameters(only_trainable=True)
    else:
        all_parameters = sum(p.numel() for p in model.parameters())
        trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"All model parameters: {millify(all_parameters)}")
    print(f"Trainable model parameters: {millify(trainable_parameters)}")
    print(f"Percentage of trainable model parameters: {100 * trainable_parameters / all_parameters:.2f}%")


def set_additional_trainable_modules(model: PeftModel, adapter_name: str = "default"):
    """Identifies and sets additional modules within the model to be trainable.

    Parameters within the model required for gradient updates, but not part
    of the LoRA adapters, are identified by name. This function populates the
    PEFT configuration with these module names, ensuring they are included
    during training and model saving processes.

    Args:
        model: The PEFT model instance being configured.
        adapter_name: Optional; the name of the adapter being configured.
                      Defaults to 'default' if not specified.
    """
    peft_config: LoraConfig = model.peft_config[adapter_name]  # type: ignore

    # Extract module names for trainable parameters not part of the LoRA modules
    peft_config.modules_to_save = [
        name_parts[-2] if (name_parts := named_parameter.split("."))[-1] == "weight" else name_parts[-1]
        for named_parameter, tensor in model.named_parameters()
        if tensor.requires_grad and "lora" not in named_parameter
    ]

    # Update the model's PEFT configuration with the identified additional trainable modules
    cast("Callable", model.set_additional_trainable_modules)(peft_config, adapter_name)
