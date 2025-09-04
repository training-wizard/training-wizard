# Binary, Multiclass and Multilabel Classification

A classic. You have some text and you want to predict class labels based on it.

Works with both encoders and decoders.

## Data Format

Your dataset **must** contain the columns `text` and `labels` (with an s).

Single label:
{
    "text": "What is the capital of France?",
    "labels": 0
}

Multi-label:
{
    "text": "What is the capital of France?",
    "labels": [1, 1]
}

*Note: check `training_wizard/specs/dataset.py` for common preprocessing solutions. You can also write your own and link to it via `spec_class = your.module.YourDataSource`*

## This example

The example is a simple multiclass classification, using dummy data.

## Config notes

*Note: it's often easiest to look at the code directly. Start with the `main` method in `training_wizard/recipes/modular.py` and work your way backwards to see how the recipe works.*

- `recipe_class` is the path to the broad training training recipe (responsible for multiple training modules)
- `wizard_module` is the meat of the recipe. Modules are swappable and define training semantics, similar to PyTorch Lightning.
- `loss_type` determines the classification type:
  - `cross_entropy`: Softmax + Cross Entropy Loss
  - `binary_cross_entropy`: Sigmoid + Binary Cross Entropy Loss
- `dataset_spec` shows how to load a simple TSV file for training
- `validation` can be either a float (ratio of dataset), integer (number of examples), dataset specification (like `dataset_spec`) or missing (no validation data)
- `training_args_spec` contains all the arguments that `TrainingArguments` in HuggingFace Transformers has, plus `mlflow_experiment_name` (setting this will log your training to MLFlow!)
  - Some arguments are implicit to avoid cluttering the config
  - See the full descriptions and possible fields in `training_wizard/specs/trainer.py`

### Training Args

Some things to keep in mind when setting hyperparams:

- Small learning rates like `1e-5` are recommended
- Use **large** batch sizes
- `bf16` training is enabled by default
- `adamw_bnb_8bit` has performed well for us and it uses less memory than AdamW

## Running the example

After installing the project, you can run the toml config using:

```bash
training-wizard examples/sequence_classification/config.toml
```

### Multi-GPU

We also support multi-GPU training using `accelerate`. First, pick and configure your multi-GPU training setup:

```bash
accelerate config
```

Then run the wizard using `accelerate`:

```bash
accelerate launch -m training_wizard examples/sequence_classification/config.toml
```
