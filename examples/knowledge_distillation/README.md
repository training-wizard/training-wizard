# Knowledge Distillation

Transfer knowledge from a larger, more capable teacher model to a smaller, more efficient student model. This technique allows you to create compact models that retain much of the performance of larger models while being faster and requiring less memory.

The knowledge distillation module combines the standard cross-entropy loss with a distillation loss based on Kullback-Leibler (KL) divergence between the teacher and student output distributions.

## How it works

1. **Teacher Model**: A larger, pre-trained model that provides "soft targets" (probability distributions)
2. **Student Model**: A smaller model that learns to mimic the teacher's behavior
3. **Combined Loss**: Weighted combination of:
   - Standard loss (cross-entropy with ground truth labels)
   - Distillation loss (KL divergence between teacher and student outputs)

## Data Format

Your dataset should be in instruction-tuning format with `source` and `target` columns (or adjust the template accordingly):

```json
{
    "source": "What is the capital of France?",
    "target": "The capital of France is Paris."
}
```

Or using the chat format:

```json
{
    "messages": [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."}
    ]
}
```

*Note: check `training_wizard/specs/dataset.py` for common preprocessing solutions. You can also write your own and link to it via `spec_class = your.module.YourDataSource`*

## This example

The example demonstrates distilling knowledge from a medium-sized DialoGPT model to a smaller DialoGPT model using conversational data.

## Config notes

*Note: it's often easiest to look at the code directly. Start with the `main` method in `training_wizard/recipes/modular.py` and work your way backwards to see how the recipe works.*

- `spec_class` is the path to the broad training training recipe
- `wizard_module` uses `KnowledgeDistillationModule` for distillation training
- `teacher_spec`: Configuration for the teacher model (frozen during training)
- `student_module`: The student module configuration (any WizardModule, typically InstructionTuningModule)
- `temperature`: Temperature parameter for softening probability distributions (higher = softer)
- `distillation_weight`: Weight (alpha) for the distillation loss vs. standard loss
  - `0.7` means 70% distillation loss, 30% standard loss
  - `1.0` means only distillation loss, `0.0` means only standard loss

### Key Parameters

- **Temperature**: Controls how "soft" the probability distributions are
  - Higher values (3-5) create softer distributions with more information
  - Lower values (1-2) create sharper distributions closer to hard targets
- **Distillation Weight**: Balances between learning from teacher vs. ground truth
  - Higher values emphasize learning from teacher's knowledge
  - Lower values emphasize learning from ground truth labels

### Training Args

Knowledge distillation specific considerations:

- Use **smaller learning rates** (like `5e-6`) since we're fine-tuning
- **Smaller batch sizes** may be necessary due to memory requirements (both teacher and student models)
- **Gradient accumulation** helps achieve effective larger batch sizes
- Enable `gradient_checkpointing` to save memory

## Running the example

After installing the project, you can run the config using:

```bash
training-wizard examples/knowledge_distillation/config.toml
```

### Multi-GPU

We also support multi-GPU training using `accelerate`. First, configure your setup:

```bash
accelerate config
```

Then run the wizard using `accelerate`:

```bash
accelerate launch -m training_wizard examples/knowledge_distillation/config.toml
```

### Memory Considerations

Knowledge distillation requires loading both teacher and student models, which increases memory usage:

- Consider using smaller batch sizes
- Enable gradient checkpointing
- Use mixed precision training (`bf16 = true`)
- Consider using 8-bit optimizers (`adamw_bnb_8bit`)
- The teacher model is automatically frozen to save memory on gradients
