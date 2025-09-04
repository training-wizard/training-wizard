"""Print the number of trainable parameters for a huggingface model.

This does NOT need to download the model weights, but will instead load a set
of default weights into CPU memory.
"""

import argparse

from transformers import AutoConfig, AutoModel


def humanize(number: int) -> str:
    """Convert a number to a human-readable string.

    This uses "k" for thousands, "M" for millions, etc.

    Args:
        number: The number to convert.

    Returns:
        The human-readable string.
    """
    oom = {
        0: "",
        3: "k",
        6: "M",
        9: "B",
        12: "T",
    }
    humanized = str(number)
    for o in oom:
        if number > 10 * 10**o:
            humanized = f"{number / 10**o:.0f}{oom[o]}"
        elif number > 10**o:
            humanized = f"{number / 10**o:.1f}"
            if humanized.endswith(".0"):
                humanized = humanized[:-2]
            humanized += oom[o]
    return humanized


def main():
    """Main command function."""
    parser = argparse.ArgumentParser(
        prog="count-parameters",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("model", type=str, help="The model to load with AutoConfig.from_pretrained()")
    parser.add_argument(
        "-H",
        "--human-readable",
        action="store_true",
        help="Format the output as human-readable string.",
    )
    args = parser.parse_args()

    config = AutoConfig.from_pretrained(args.model)
    model = AutoModel.from_config(config)
    num_params = sum(p.numel() for p in model.parameters())
    if args.human_readable:
        print(humanize(num_params))
    else:
        print(num_params)
