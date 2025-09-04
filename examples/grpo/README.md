# Group Relative Policy Optimization (GRPO)

Group Relative Policy Optimization (GRPO) is a reinforcement learning from human feedback (RLHF) technique that optimizes language models using group-relative advantages. It's an extension of Proximal Policy Optimization (PPO) that computes rewards relatively within a group of generated completions and incorporates a KL-penalty with a reference policy to ensure stability during optimization.

## Data Format

GRPO requires a prompt-only dataset. The dataset must follow one of these two formats:

1. Standard Format:

```json
{
    "prompt": "What is the capital of France?"
}
```

2. Conversational Format (System Message is optional):

```json
{
    "prompt": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ]
}
```

*Note: Check `training_wizard/specs/dataset.py` for common preprocessing solutions. You can also write your own and link to it via `spec_class = your.module.YourDataSource`*

## This Example

In this example, we're implementing a Grammar Error Correction (GEC) task using GRPO. We use a custom reward function that calculates F1 scores based on edit operations between the source text, the model's correction, and the target correction.

The reward function:

1. Extracts edit operations between strings (insertions, deletions, replacements)
2. Computes precision, recall, and F1 scores for the model's edits compared to ground truth edits
3. Returns the F1 score as the reward

## Custom Reward Specification

GRPO allows you to define custom reward functions by subclassing `RewardSpec`. In this example, we've implemented `GECExampleRewardSpec` which:

1. Parses prompts and completions from different formats
2. Computes edit-based metrics between source, hypothesis, and target texts
3. Logs metrics during training
4. Returns F1 scores as rewards

You can create multiple reward functions and combine them using `MultiRewardSpec` with different weighting strategies.

## Config Notes

*Note: It's often easiest to look at the code directly. Start with the `main` method in `training_wizard/recipes/grpo.py` and work your way backwards to see how the recipe works.*

- `recipe_class` points to the GRPO training recipe
- `reward_specs` defines the custom reward function(s) to use
- We use a template-based dataset to format prompts correctly
- `sample_rate = 100` prints a sample of outputs every 100 steps

### Training Args

Some things to keep in mind when setting hyperparameters:

- `gradient_checkpointing = true` saves memory but may slow down training
- `per_device_train_batch_size` and `gradient_accumulation_steps` control the effective batch size
- `beta = 0.0` controls how much to penalize the policy for deviating from the reference model (higher values = more conservative)
- `epsilon = 0.2` controls the clipping threshold for the policy update
- `num_generations = 2` sets how many completions to generate per prompt
- `temperature = 1.0` controls randomness in generation
- `num_iterations = 1` sets how many times to iterate over each batch
- `max_prompt_length` and `max_completion_length` control sequence lengths

### vLLM Acceleration

GRPO supports vLLM for faster generation:

- `use_vllm = false` enables/disables vLLM
- `vllm_gpu_memory_utilization = 0.8` controls GPU memory allocation
- `vllm_max_model_len = 4096` sets the maximum sequence length
- `vllm_dtype = "bfloat16"` sets the data type for generation
- `vllm_enable_prefix_caching = true` enables caching for faster generation

## Running the Example

After installing the project, you can run the config using:

```bash
training-wizard examples/grpo/config.toml
```

### Multi-GPU Training

We also support multi-GPU training using `accelerate`. First, configure your multi-GPU setup:

```bash
accelerate config
```

Then run the wizard using `accelerate`:

```bash
accelerate launch -m training_wizard examples/grpo/config.toml
```

## References

- TRL Documentation: [GRPO Trainer](https://huggingface.co/docs/trl/grpo_trainer)
