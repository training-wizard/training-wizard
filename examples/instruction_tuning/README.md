# Instruction Tuning

This example shows how to fine-tune a model using instruction tuning. The model will learn to follow instructions in a chat format, with system, user, and assistant messages.

## Data Format

The dataset must have messages in the ChatML format (System Message is optional):

{
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "Paris is the capital of France."}
    ]
}

## This example

Here we're training a Qwen2.5 model to be a paraphraser. We use a small model for example's sake.

## Config notes

*Note: it's often easiest to look at the code directly. Start with the `main` method in `training_wizard/recipes/modular.py` and work your way backwards to see how the recipe works.*

- `recipe_class` points to the basic training recipe
- `wizard_module.spec_class` points to the instruction tuning module
- `assistant_only` can be enabled to only compute loss on assistant messages
- `transformer_spec` configures the model and tokenizer
- `dataset_spec` configures the data source and message templates
- `take = 1_000` selects the first 1000 rows from the parent, because this is an example

### Training Args

Some things to keep in mind when setting hyperparams:

- Use gradient checkpointing if you run into memory issues
- A learning rate around 1e-5 works well for most cases
- Enable bf16 mixed precision training if your GPU supports it
- The 8-bit optimizer helps reduce memory usage
- NEFTune noise can help with robustness
- Early stopping helps prevent overfitting

## Running the example

After installing the project, you can run the toml config using:

```bash
training-wizard examples/instruction_tuning/config.toml
```

### Multi-GPU

We also support multi-GPU training using `accelerate`. First, pick and configure your multi-GPU training setup:

```bash
accelerate config
```

Then run the wizard using `accelerate`:

```bash
accelerate launch -m training_wizard examples/instruction_tuning/config.toml
```
