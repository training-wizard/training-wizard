"""Provide useful information about a model in relation to using LoRA with it."""

import argparse
from collections import Counter
from typing import cast

import torch
from tabulate import tabulate
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    Conv1D,
    PreTrainedModel,
)
from trl import AutoModelForCausalLMWithValueHead

from ..components.utilities import millify

forms_of_all_linear_option = (None, "", "all-linear", ["all-linear"])


def find_all_linear_names(model: torch.nn.Module) -> set[str]:
    """Find all linear modules in the model.

    Args:
        model: The model to find linear modules in.

    Returns:
        A set of names of linear modules in the model.
    """
    linear_classes = (torch.nn.Linear, Conv1D)
    # linear_classes = (torch.nn.Linear, torch.nn.Embedding, torch.nn.Conv2d, Conv1D)

    linear_module_names = set()
    for name, module in model.named_modules():
        # match with all linear classes.
        if isinstance(module, linear_classes):
            names = name.rsplit(".", 1)[-1]  # get the base name
            linear_module_names.add(names)

    # ignore the last classification head for text generation models
    if hasattr(model, "get_output_embeddings"):
        model = cast("PreTrainedModel", model)
        output_emb = model.get_output_embeddings()
        if output_emb is not None:
            last_module_name = next(iter([name for name, module in model.named_modules() if module is output_emb]))
            linear_module_names -= {last_module_name}

    return linear_module_names


def find_full_names(model: torch.nn.Module, target_modules: set[str] | list[str] | str | None = None) -> set[str]:
    """Find the full names of the target modules in the model.

    Args:
        model: The model to find the target modules in.
        target_modules: The target modules to select.

    Returns:
        A set of full names of the target modules in the model.
    """
    if target_modules in forms_of_all_linear_option:
        target_modules = find_all_linear_names(model)
    else:
        target_modules = set(target_modules)
    return {name for name, _ in model.named_modules() if target_modules & set(name.split("."))}


def print_number_of_parameters_sizes_of_matrices(
    model: torch.nn.Module, target_modules: set[str] | list[str] | str | None = None
):
    """Print the number of parameters and sizes of matrices in the "adaptable" target modules.

    Args:
        model: The model to print the number of parameters and sizes of matrices for.
        target_modules: The target modules to select.
    """
    target_modules_full_names = find_full_names(model, target_modules)
    all_target_modules_full_names = find_full_names(model)
    ls = []
    target_modules_params, all_target_modules_params = 0, 0
    for name, module in model.named_modules():
        for p in module.parameters():
            if name in target_modules_full_names:
                n, m = p.shape
                ls.append(tuple(p.shape))
                target_modules_params += p.numel()
            if name in all_target_modules_full_names:
                all_target_modules_params += p.numel()
    shape_counts = Counter(ls)
    table_data = [((f"{n} x {m}"), count) for (n, m), count in shape_counts.items()]
    all_parameters = sum(p.numel() for p in model.parameters())
    target_modules = find_all_linear_names(model) if target_modules in forms_of_all_linear_option else target_modules
    print(f"All possible target modules: {sorted(find_all_linear_names(model))}")
    print(f"Current target modules: {sorted(target_modules)}")
    print(f"Number of parameters in all modules: {millify(all_parameters)}")
    print(f"Number of parameters in target modules: {millify(target_modules_params)}")
    print(f"Percentage to all model parameters: {100 * target_modules_params / all_parameters:.2f}%")
    print(f"Percentage to all adaptable parameters: {100 * target_modules_params / all_target_modules_params:.2f}%")
    print("Shapes of matrices in target modules:")
    print(tabulate(table_data, headers=["Shape (n x m)", "Count"], tablefmt="pretty"))


def main():
    """Main command function."""
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", type=str, help="Name of the pretrained model")
    parser.add_argument("--type", type=str, default=None, help="Type of the model")
    parser.add_argument(
        "--target_modules", type=str, nargs="*", default=None, help="Target modules to count parameters for."
    )
    args = parser.parse_args()

    if args.type == "seq2seq":
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    elif args.type == "causal":
        model = AutoModelForCausalLM.from_pretrained(args.model_name)
    elif args.type == "sequence-classification":
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name)
    elif args.type == "causal-value-head":
        model = AutoModelForCausalLMWithValueHead.from_pretrained(args.model_name)
    else:
        model = AutoModel.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name)
    print_number_of_parameters_sizes_of_matrices(model, args.target_modules)
