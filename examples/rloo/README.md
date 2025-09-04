# Online RLHF with REINFORCE Leave-One-Out (RLOO)

*The idea is that instead of using a value function, RLOO generates K completions for each prompt. For each completion, RLOO uses the mean scores from the other K-1 completions as a baseline to calculate the advantage. RLOO also models the entire completion as a single action, where as PPO models each token as an action. Note that REINFORCE / A2C is a special case of PPO, when the number of PPO epochs is 1 and the number of mini-batches is 1, which is how we implement RLOO in TRL.*

More details [here](https://huggingface.co/docs/trl/rloo_trainer).

## Data Format

RLOO requires a prompt-only dataset. The rows must follow one of these two formats:

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

For once, a real-life example! The reward spec in `reward_spec.py` is designed for paraphrasing and includes many different metrics going together into one training signal for the model.

## Config notes

*Note: it's often easiest to look at the code directly. Start with the `main` method in `training_wizard/recipes/rloo.py` and work your way backwards to see how the recipe works.*

- `recipe_class` points to the RLOO recipe implementation
- The reward spec in `reward_spec.py` defines a comprehensive reward function combining multiple metrics
- To create the required prompt-only dataset for training, we take an existing dataset from the Data Forest and apply a template using `TemplateInstructDatasourceSpec`. This creates a dataset with a `prompt` column containing instruct style messages. `{xyz}` is replaced with the content of the column `xyz` in the parent dataset.
- `take = 1_000` selects the first 1000 rows from the parent, because this is an example
- `filter_length = false` by default. Set to `true` if you have a seq2seq task and you want to filter out long inputs that are more than `0.9 * response_length` tokens in size.

### Training Args

Some things to keep in mind when setting hyperparams:

- Try to keep gradient checkpointing **off** if possible, as it slows down candidate generation considerably
- Try to achieve a high global batch size using gradient accumulation if needed
- Keep the learning rate low, around `1e-6`
- If the model is generating poor candidates, try reducing `temperature` (default 0.5)
- `response_length` controls the maximum number of tokens to generate
- `kl_coef = 0.05` controls how much to penalize divergence from the reference model
- `cliprange = 0.2` limits how much the policy can change in one update
- `rloo_k = 2` sets how many candidates to generate per prompt
- `num_ppo_epochs = 4` controls how many optimization passes to make on each batch
- `whiten_rewards = false` by default. Enable if reward scaling becomes an issue.

## Running the example

After installing the project, you can run the toml config using:

```bash
training-wizard examples/rloo/config.toml
```

### Multi-GPU

We also support multi-GPU training using `accelerate`. First, pick and configure your multi-GPU training setup:

```bash
accelerate config
```

Then run the wizard using `accelerate`:

```bash
accelerate launch -m training_wizard examples/rloo/config.toml
```
