# Training Wizard
<p align="center">
  <img alt="logo" src="assets/logo.png" width="300" />
</p>

<p align="center">
<i>An alchemy of automation for the modern mage of machine learning</i>
</p>

The training wizard is a [spec-driven](concepts/spec.md) tool for quick **prototyping** and **finetuning** of **transformers** models.

## Overview

The Training Wizard is a tool that streamlines machine learning workflows by providing reusable components, structured experimentation, and TOML-based configuration for rapid model development and deployment.

## Key Features

- **Reusable Components**: Shared library of ML functions and utilities
- **Experiment Tracking**: Built-in MLflow integration for consistent operations
- **Simple Configuration**: TOML-based setup for streamlined model finetuning
- **Multi-GPU Support**: Accelerate integration for distributed training

## Use Cases

- **New Projects**: Offers a templated approach for setting up new ML experiments quickly and efficiently.
- **Model Training**: Abstracts complex details like parallelization and quantization, allowing users to concentrate on problem-solving.
- **Experiment Management**: Integrates seamless tracking of experiments and models, ensuring results are readily accessible.

## Installation

1. `git clone` the project repository to create a working copy on your machine.
2. Ensure that you have uv installed. The [official documentation for installing uv](https://docs.astral.sh/uv/getting-started/installation/) has detailed instructions.
1. Run `uv sync --all-groups --all-extras`.

## Getting Started

Each example in the `examples/` directory contains a complete recipe with its own README. Browse the available recipes and adapt one to your use case:

- **Instruction Tuning**: Fine-tune models for instruction following
- **Sequence Classification**: Train classification models
- **GRPO/CPO/DPO**: Preference optimization techniques
- **Quantization**: Efficient model compression
- **And more...**

Run any recipe with:

```bash
training-wizard examples/[recipe]/config.toml
```

## Contributing

Contributions are welcome! If you've built something that would improve Training Wizard, please open a pull request. Please read first the [developer documentation](develop/index.md) and the [legal terms](legal_terms.md) that apply to the project.