# Seq2Seq Using Encoder-Decoder Models

Translation of all kinds.

## Data Format

Your dataset **must** contain the columns `source` and `target`.

{
    "source": "What is the capital of France?",
    "target": "Paris is the capital of France."
}

*Note: check `training_wizard/specs/dataset.py` for common preprocessing solutions. You can also write your own and link to it via `spec_class = your.module.YourDataSource`*

## This example

The example is for DE Standard paraphrasing.

## Config notes

*Note: it's often easiest to look at the code directly. Start with the `main` method in `training_wizard/recipes/modular.py` and work your way backwards to see how the recipe works.*

- `recipe_class` is the path to the broad training training recipe (responsible for multiple training modules)
- `wizard_module` is the meat of the recipe. Modules are swappable and define training semantics, similar to PyTorch Lightning.
  - `peek_rate` is the rate (in steps) for printing a small sample of input-prediction-target triplets
- `dataset_spec` shows how to load a dataset from the Data Forest
  - `take = 10000` means we only select the first 10k rows and discard the rest
- `validation` can be either a float (ratio of dataset), integer (number of examples), dataset specification (like `dataset_spec`) or missing (no validation data)
- `training_args_spec` contains all the arguments that `TrainingArguments` in HuggingFace Transformers has, plus `mlflow_experiment_name` (setting this will log your training to MLFlow!)
  - Some arguments are implicit to avoid cluttering the config
  - See the full descriptions and possible fields in `training_wizard/specs/trainer.py`

### Training Args

We set some sensible defaults, but you should understand what each setting does. Ideally, also look into the implicit fields to be aware of them.

## Running the example

After installing the project, you can run the toml config using:

```bash
training-wizard examples/seq2seq/config.toml
```

### Multi-GPU

We also support multi-GPU training using `accelerate`. First, pick and configure your multi-GPU training setup:

```bash
accelerate config
```

Then run the wizard using `accelerate`:

```bash
accelerate launch -m training_wizard examples/seq2seq/config.toml
```
