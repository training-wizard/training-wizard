# Sequence Regression

Predict continuous numerical scores from text sequences. This module is useful for tasks like sentiment scoring, quality rating, similarity scoring, or any task where you need to predict a continuous value from text.

Works with both encoders and decoders.

## Data Format

Your dataset **must** contain the columns `text` and `labels` (with an s).

The labels should be continuous numerical values:

```json
{
    "text": "This movie is amazing!",
    "labels": 0.9
}
```

```json
{
    "text": "This product is okay.",
    "labels": 0.5
}
```

Optional sample weighting:

```json
{
    "text": "High confidence example",
    "labels": 0.8,
    "sample_weight": 2.0
}
```

*Note: check `training_wizard/specs/dataset.py` for common preprocessing solutions. You can also write your own and link to it via `spec_class = your.module.YourDataSource`*

## This example

The example demonstrates regression on dummy data, predicting continuous scores from text input.

## Config notes

*Note: it's often easiest to look at the code directly. Start with the `main` method in `training_wizard/recipes/modular.py` and work your way backwards to see how the recipe works.*

- `spec_class` is the path to the broad training recipe (responsible for multiple training modules)
- `wizard_module` is the meat of the recipe. Uses `SequenceRegressionModule` for regression tasks.
- `loss_type` determines the regression loss function:
  - `"smooth_l1"`: Smooth L1 Loss (Huber loss) - robust to outliers
  - `"mse"`: Mean Squared Error Loss - standard regression loss
- `activation` determines the output activation:
  - `"sigmoid"`: For scores normalized to [0, 1] range
  - `"linear"`: For unbounded continuous scores
- `dataset_spec` shows how to load a simple TSV file for training
- `validation` can be either a float (ratio of dataset), integer (number of examples), dataset specification (like `dataset_spec`) or missing (no validation data)
- `training_args_spec` contains all the arguments that `TrainingArguments` in HuggingFace Transformers has, plus `mlflow_experiment_name` (setting this will log your training to MLFlow!)

### Training Args

Some things to keep in mind when setting hyperparams:

- Small learning rates like `1e-5` are recommended
- Use **large** batch sizes when possible
- `bf16` training is enabled by default
- `adamw_bnb_8bit` has performed well for us and uses less memory than AdamW

## Running the example

After installing the project, you can run the config using:

```bash
training-wizard examples/sequence_regression/config.toml
```

### Multi-GPU

We also support multi-GPU training using `accelerate`. First, pick and configure your multi-GPU training setup:

```bash
accelerate config
```

Then run the wizard using `accelerate`:

```bash
accelerate launch -m training_wizard examples/sequence_regression/config.toml
```
