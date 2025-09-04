# Causal Seq2Seq Training

This example shows how to train a causal language model (like Qwen) for seq2seq tasks. Unlike encoder-decoder models, causal LMs can only attend to previous tokens, so we need to use a separator token to mark the boundary between input and output.

## Data Format

The dataset must have two columns:

- `source`: The input text
- `target`: The target text

The model will be trained to generate the target text when given the source text followed by a separator token.

## This example

Here we're training a Qwen2.5 model on a DE Standard dataset. We use a small model for example's sake.

## Config notes

*Note: it's often easiest to look at the code directly. Start with the `main` method in `training_wizard/recipes/modular.py` and work your way backwards to see how the recipe works.*

- `recipe_class` points to the basic training recipe
- `wizard_module.spec_class` points to the causal seq2seq module
- `separator_token` is the token used to separate source from target (defaults to <sep>)
- `mask_source_tokens` can be enabled to only compute loss on target tokens
- `generation_args` control how the model generates text during evaluation
- `transformer_spec` configures the model and tokenizer
- `dataset_spec` configures the data source and any preprocessing
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
training-wizard examples/causal_seq2seq/config.toml
```

### Multi-GPU

We also support multi-GPU training using `accelerate`. First, pick and configure your multi-GPU training setup:

```bash
accelerate config
```

Then run the wizard using `accelerate`:

```bash
accelerate launch -m training_wizard examples/causal_seq2seq/config.toml
```
