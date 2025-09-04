# Contrastive Preference Optimization

*CPO aims to mitigate two fundamental shortcomings of SFT. First, SFT’s methodology of minimizing the discrepancy between predicted outputs and gold-standard references inherently caps model performance at the quality level of the training data. Secondly, SFT lacks a mechanism to prevent the model from rejecting mistakes in translations. The CPO objective is derived from the DPO objective.*

Source: [TRL](https://huggingface.co/docs/trl/cpo_trainer)

## Data Format

CPO requires a preference dataset. The dataset must follow one of these two formats:

1. Standard Format:
{
    "prompt": "What is the capital of France?",
    "chosen": "Paris is the capital of France.",
    "rejected": "I believe London is the capital of France."
}

2. Conversational Format (System Message is optional):
{
    "prompt": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ],
    "chosen": [
        {"role": "assistant", "content": "Paris is the capital of France."}
    ],
    "rejected": [
        {"role": "assistant", "content": "I believe London is the capital of France."}
    ]
}

*Note: check `training_wizard/specs/dataset.py` for common preprocessing solutions. You can also write your own and link to it via `spec_class = your.module.YourDataSource`*

## This example

Since CPO is both a fine-tuning and RLHF method, we start with a basic instruct model. We use preference pairs from DE Standard paraphrasing. The base dataset has three columns: `input`, `chosen` and `rejected`, which we put into a template to conform to the conversational format.
In theory, you should also be able to use CPO on a seq2seq model, such as mT5, with the standard format preferences.

## Config notes

*Note: it's often easiest to look at the code directly. Start with the `main` method in `training_wizard/recipes/cpo.py` and work your way backwards to see how the recipe works.*

- `recipe_class` points to the CPO training recipe
- We use a preference dataset from the Data Forest and apply templates to format it correctly
- The `TemplatePreferenceDatasourceSpec` class helps create the required format by applying templates to the input, chosen and rejected responses
- `take = 1000` selects the first 1000 rows from the parent, because this is an example

### Training Args

Some things to keep in mind when setting hyperparams:

- `gradient_checkpointing` can be turned off if you have enough memory
- Use `gradient_accumulation_steps` to increase the effective batch size if needed
- `beta = 0.1` controls how much to penalize the policy for deviating from the reference model
- `cpo_alpha` controls the strength of behavioral cloning regularization
- `simpo_gamma` is used when `loss_type = "simpo"` to control the SimPO loss
- `max_length` and `max_prompt_length` control the sequence lengths
- `bf16` enables bfloat16 precision training
- `optim = "adamw_bnb_8bit"` uses 8-bit Adam optimizer to save memory

## Running the example

After installing the project, you can run the toml config using:

```bash
training-wizard examples/cpo/config.toml
```

### Multi-GPU

We also support multi-GPU training using `accelerate`. First, pick and configure your multi-GPU training setup:

```bash
accelerate config
```

Then run the wizard using `accelerate`:

```bash
accelerate launch -m training_wizard examples/cpo/config.toml
```
