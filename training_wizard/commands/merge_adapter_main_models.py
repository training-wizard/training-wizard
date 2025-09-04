"""Merge models and adapters."""

import argparse

from ..components.utilities import merge_adapter_main_models


def main():
    """Main command function."""
    parser = argparse.ArgumentParser(
        prog="merge-adapter-main-models",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("adapter_model_path", type=str, help="The path to the adapter weights")
    parser.add_argument("base_model_path", type=str, help="The path to the base model weights")
    parser.add_argument(
        "-m",
        "--merged_model_path",
        type=str,
        default=None,
        help="The path to the merged model weights. Defaults to the adapter path with the suffix _merged.",
    )
    args = parser.parse_args()
    output_path = args.merged_model_path
    if output_path is None:
        output_path = args.adapter_model_path + "_merged"
    print(f"Merging {args.adapter_model_path} from base model {args.base_model_path}.")
    print(f"Output path: {output_path}")
    merge_adapter_main_models(args.adapter_model_path, args.base_model_path, output_path)
