# Online Direct Preference Optimization (DPO)

Online DPO was proposed in [Direct Language Model Alignment from Online AI Feedback](https://arxiv.org/abs/2402.14827). Unlike traditional DPO which uses a fixed preference dataset, Online DPO generates candidates during training and uses a judge model to provide online feedback.

We start with:

- A policy model (the model to be trained)
- A reference model (frozen copy of the policy model)
- A judge (any function that takes (Input, A, B) and decides if A or B is better given the input)

During training:

- We sample some input x from the dataset
- The policy generates a candidate A
- The reference generates a candidate B
- The judge decides which is better. Now we have a preference pair we can apply DPO to.
- In theory, this online method should surpass a fixed preference dataset, like in traditional DPO

## Data Format

Online DPO requires a prompt-only dataset. The rows must follow one of these two formats:

1. Standard Format:
{
    "prompt": "What is the capital of France?"
}

2. Conversational Format (System Message is optional):
{
    "prompt": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ]
}

*Note: check `training_wizard/specs/dataset.py` for common preprocessing solutions. You can also write your own and link to it via `spec_class = your.module.YourDataSource`*

## This example

Here, we're looking at a DE Standard mode paraphrasing model. Normally both the ranker and the policy model would be finetuned, but for example's sake, we're only using a small instruction-tuned Qwen2.5

## Config notes

*Note: it's often easiest to look at the code directly. Start with the `main` method in `training_wizard/recipes/online_dpo.py` and work your way backwards to see how the recipe works.*

- `recipe_class` is the path to the recipe we want to use
- We use a simple judge that evaluates paraphrasing quality
- To create the required prompt-only dataset for training, we take an existing dataset from the Data Forest and apply a template. This is easy to do with the `TemplateInstructDatasourceSpec` class. It creates a dataset with a custom column (`messages`, or `prompt` in our case) containing instruct style messages. `{xyz}` is replaced with the content of the column `xyz` in the parent dataset.
- `take = 1_000` selects the first 1000 rows from the parent, because this is an example
- `filter_length = false` by default. Set to `true` if you have a seq2seq task and you want to filter out long inputs that are more than `0.9 * max_new_tokens` tokens in size.

### Training Args

Some things to keep in mind when setting hyperparams:

- Try to keep gradient checkpointing **off**. It disables the generator cache which slows down candidate generation considerably.
- Try to achieve a high global batch size, at least 64. Use gradient accumulation to help with memory issues.
- Keep the learning rate row, `5e-7` when you have a global batch size of at least 64.
- If the model is too wild with the candidates, consider reducing `temperature`
- `max_new_tokens` should be the maximum number of tokens you want to generate (aka max response length).
- `beta = 0.1` is fairly loose. If the model is reward hacking, up this to `0.3` or `0.4`.

## Running the example

After installing the project, you can run the toml config using:

```bash
training-wizard examples/online_dpo/config.toml
```

### Multi-GPU

We also support multi-GPU training using `accelerate`. First, pick and configure your multi-GPU training setup:

```bash
accelerate config
```

Then run the wizard using `accelerate`:

```bash
accelerate launch -m training_wizard examples/online_dpo/config.toml
```
