# LLM Quantization

This recipe implements model quantization using the [llm-compressor](https://github.com/vllm-project/llm-compressor/tree/main) library to reduce model size and improve inference performance. It supports both W8A8 (8-bit weights and activations) and W4A16 (4-bit weights with 16-bit activations) quantization schemes. The quantization process uses SmoothQuant to make activations easier to quantize by adjusting weight scales, followed by GPTQ for the actual quantization.

## Data Format

The data format depends on the Wizard Module you use, check the other recipes :-) (Only decoders though!)

*Note: check `training_wizard/specs/dataset.py` for common preprocessing solutions. You can also write your own and link to it via `spec_class = your.module.YourDataSource`*

## This example

This example quantizes a Qwen2.5-0.5B model using W8A8 quantization. We use a seq2seq dataset from the Data Forest for calibration.

## Config notes

*Note: it's often easiest to look at the code directly. Start with the `main` method in `training_wizard/recipes/quantize_llm.py` and work your way backwards to see how the recipe works.*

- `recipe_class` points to the quantization recipe implementation
- `scheme` can be either "W8A8" or "W4A16" - W8A8 is more accurate but larger
- `num_calibration_samples` controls how many samples to use for calibration (512 is usually enough)
- `batch_size` should usually be 1 for quantization
- The wizard module is configured for seq2seq translation with flash attention and bfloat16
- We use a simple dataset spec to load data from the Data Forest

## Running the example

After installing the project, you can run the toml config using:

```bash
training-wizard examples/quantization/config.toml
```

### Multi-GPU

We also support multi-GPU training using `accelerate`. First, pick and configure your multi-GPU training setup:

```bash
accelerate config
```

Then run the wizard using `accelerate`:

```bash
accelerate launch -m training_wizard examples/quantization/config.toml
```
