"""Command-line interface for training."""

import argparse
from pathlib import Path

import structlog

from ..specs.spec import parse_config

logger = structlog.get_logger()


def training_wizard():
    """Command-line interface for training in a single process."""
    parser = argparse.ArgumentParser()
    parser.add_argument("spec", type=Path, help="Path to a TOML specification file.")
    args = parser.parse_args()
    spec_instance = parse_config(args.spec)
    logger.info("Loaded spec %s", type(spec_instance))
    spec_instance.main()
