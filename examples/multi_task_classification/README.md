# Multi-Task Sequence Classification

Perform multiple classification tasks simultaneously on the same input text. This is useful when you want to predict several different attributes or categories for each text sample, such as sentiment + topic + urgency, or language + domain + complexity.

The model learns shared representations while maintaining separate classification heads for each task.

## Data Format

Your dataset **must** contain the columns `text` and `labels` (with an s).

The labels should be a list/array with one integer label per task:

```json
{
    "text": "This urgent customer complaint about billing is very negative",
    "labels": [2, 0, 1]
}
```

Where the labels might represent:

- First task (sentiment): 0=negative, 1=neutral, 2=positive
- Second task (urgency): 0=urgent, 1=normal  
- Third task (topic): 0=billing, 1=technical, 2=general, 3=feedback

Optional per-task or per-sample weighting:

```json
{
    "text": "Important training example",
    "labels": [1, 0, 2],
    "weight": [1.0, 2.0, 1.5]  # Weight per task
}
```

Or single weight for all tasks:

```json
{
    "text": "High confidence example",
    "labels": [0, 1, 0],
    "weight": 2.0
}
```

*Note: check `training_wizard/specs/dataset.py` for common preprocessing solutions. You can also write your own and link to it via `spec_class = your.module.YourDataSource`*

## This example

The example demonstrates multi-task classification with three tasks:

- Task 1: 3-class classification (sentiment: negative/neutral/positive)
- Task 2: 2-class classification (urgency: urgent/normal)
- Task 3: 4-class classification (topic: billing/technical/general/feedback)

## Config notes

*Note: it's often easiest to look at the code directly. Start with the `main` method in `training_wizard/recipes/modular.py` and work your way backwards to see how the recipe works.*

- `spec_class` is the path to the broad training training recipe
- `wizard_module` uses `MultiTaskSequenceClassifierModule` for multi-task classification
- `group_sizes`: List defining the number of classes for each task
  - Must match the structure of your labels
  - Total `num_labels` in the model config should equal the sum of group_sizes
- `label_smoothing`: Applied to all classification tasks (helps with overconfidence)
- `compute_eval_metrics`: Whether to compute detailed evaluation metrics
- `show_per_group_metrics`: Whether to show metrics for each individual task

### Key Parameters

- **group_sizes**: Defines the multi-task structure
  - `[3, 2, 4]` means 3 tasks with 3, 2, and 4 classes respectively
  - Labels should be structured as `[task1_label, task2_label, task3_label]`
- **num_labels**: Must equal the sum of all group_sizes
- **label_smoothing**: Helps prevent overconfident predictions across all tasks

### Training Args

Multi-task classification specific considerations:

- **Larger batch sizes** often help as the model learns multiple objectives
- **Moderate learning rates** (like `2e-5`) work well for multi-task learning
- **More epochs** may be needed as the model balances multiple objectives
- Monitor both overall metrics and per-task metrics during training

## Metrics

The module computes several types of metrics:

1. **Aggregated metrics**: Averaged across all tasks (e.g., `accuracy`, `f1_score`)
2. **Standard deviation metrics**: Variance across tasks (e.g., `accuracy_std`)
3. **Per-group metrics**: Individual metrics for each task (when `show_per_group_metrics = true`)

This helps you understand both overall performance and task-specific performance.

## Running the example

After installing the project, you can run the config using:

```bash
training-wizard examples/multi_task_classification/config.toml
```

### Multi-GPU

We also support multi-GPU training using `accelerate`. First, configure your setup:

```bash
accelerate config
```

Then run the wizard using `accelerate`:

```bash
accelerate launch -m training_wizard examples/multi_task_classification/config.toml
```

## Use Cases

Multi-task classification is particularly useful for:

- **Content moderation**: toxicity + spam + off-topic detection
- **Customer support**: sentiment + urgency + category classification  
- **Document analysis**: topic + complexity + language detection
- **Social media**: sentiment + emotion + intent classification
- **E-commerce**: category + brand + condition classification

The shared representation learning often leads to better performance than training separate models for each task.
