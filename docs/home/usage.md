# Running a Recipe

This page will walk you through the process of setting up your desired recipe and running the `training-wizard` command. It is a high-level overview of how you would use the Training Wizard in practice.

## Make your data accessible

The one thing binding all training recipes is the presence of a **dataset**. Your data must be accessible through the [loading utility of the `datasets` library](https://huggingface.co/docs/datasets/main/en/package_reference/loading_methods#loading-methods), meaning it should be **either**:

- Available on the [HuggingFace Hub](https://huggingface.co/datasets)

- Saved locally in a compatible format (`txt`, `csv`, `json`, `jsonl`, `arrow`, `sql`, etc...)

That's it for now. Further down, we'll talk about how to inject **preprocessing steps** into your chosen recipe to bring your dataset columns into the required format.


## Set up your TOML config

Configuring your recipe might work a bit differently than you're used to. Because of the way the Training Wizard is structured, building a config happens in a somewhat recursive fashion.

!!! tip "Nested Specs"
    The [spec explanation](../concepts/spec.md) also teaches you how to build a config from scratch.

!!! tip "Experiment Tracking"
    You can use [MLflow](https://mlflow.org/) to track your experiments you do with the Training Wizard. To activate MLflow for a specific experiment, you need to add an `mlflow_experiment_name` to your TOML file, such as `mlflow_experiment_name = "test_mlflow"`, which can be found in the demo TOML file.


### Config examples

The `examples` directory contains demo TOML configs for various recipes that can be used as a reference and is a good starting point.


## Run the recipe

Once you have your TOML config set up, you're good to go! 

Run the Wizard and start training with the command: 

```bash
training-wizard path/to/your/config.toml
```

!!! tip
    The training process may take a while (hours or days) to finish. You may want to run this command in a `tmux` session.

### Multi-GPU Training

For distributed training, use `accelerate`. First configure your setup:

```bash
accelerate config
```

Then launch any recipe with:

```bash
accelerate launch -m training_wizard path/to/your/config.toml
```
