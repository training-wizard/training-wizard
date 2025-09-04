"""Find all the names of the linear modules in the model."""

import argparse

import bitsandbytes as bnb
from transformers import AutoModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM, PreTrainedModel


def find_all_linear_names_func(model: PreTrainedModel) -> list[str]:
    """Find all the names of the linear modules in the model.

    Excludes the lm_head module.

    Args:
        model: The model to find the linear modules in.

    Returns:
        A list of the names of the linear modules in the model.
    """
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, bnb.nn.Linear4bit):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


def find_all_linear_names():
    """Find all the names of the linear modules in a model."""
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", type=str, help="Name of the pretrained model")
    parser.add_argument(
        "--auto_type", type=str, default="simple", choices=["simple", "causal", "seq2seq"], help="Auto model type."
    )
    parser.add_argument("--load_4bit", type=bool, default=True, help="Load the model in 4-bit mode.")

    args = parser.parse_args()

    if args.auto_type == "simple":
        model = AutoModel.from_pretrained(args.model_name, load_in_4bit=args.load_4bit)
    elif args.auto_type == "causal":
        model = AutoModelForCausalLM.from_pretrained(args.model_name, load_in_4bit=args.load_4bit)
    elif args.auto_type == "seq2seq":
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name, load_in_4bit=args.load_4bit)
    else:
        raise ValueError(f"Unknown auto type {args.auto_type}")

    print("Linear Names:")
    print(find_all_linear_names_func(model))
