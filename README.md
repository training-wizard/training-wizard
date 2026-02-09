# Training Wizard

<p align="center">
  <img alt="logo" src="docs/assets/logo.png" width="300" />
</p>

<p align="center">
<i>An alchemy of automation for the modern mage of machine learning</i>
</p>

## Overview

The Training Wizard is a tool that streamlines machine learning workflows by providing reusable components, structured experimentation, and TOML-based configuration for rapid model development and deployment.

## Key Features

- **Reusable Components**: Shared library of ML functions and utilities
- **Experiment Tracking**: Built-in MLflow integration for consistent operations
- **Simple Configuration**: TOML-based setup for streamlined model finetuning
- **Multi-GPU Support**: Accelerate integration for distributed training

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

## Multi-GPU Training

For distributed training, use `accelerate`. First configure your setup:

```bash
accelerate config
```

Then launch any recipe with:

```bash
accelerate launch -m training_wizard examples/[recipe]/config.toml
```

## 🗄️ Project Structure

```text
├── 📁 docs/                               Documentation and assets
├── 📁 examples/                           Recipe examples with READMEs
├── 📁 tests/                              Test suite
├── 📁 training_wizard/                    Core library modules
│   ├── 📁 components/                     Reusable ML components
│   ├── 📁 commands/                       CLI command implementations
│   ├── 📁 recipes/                        Training recipe implementations
│   └── 📁 specs/                          Configuration specifications
├── 📄 pyproject.toml                      Project configuration and dependencies
└── 📄 README.md                           This file
```

## Contributing

Contributions are welcome! If you've built something that would improve Training Wizard, please open a pull request.

## Licenses

This project uses a **dual-license model**:

- **Source Code** — licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).  
  You are free to use, modify, and distribute the code under permissive terms, including an express grant of patent rights from contributors.  

- **Documentation** — licensed under the [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/). 
  You may share and adapt the documentation, even for commercial purposes, as long as proper attribution is given.  

For full details, see the [LICENSE](./LICENSE) file (for source code) and [license.md](./LEGAL/LICENSE-docs-CC-BY-4.0.txt) (for documentation).
