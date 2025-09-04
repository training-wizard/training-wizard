---
title: Specifications
description: A brief introduction to specs in the Training Wizard.
---
# What is a spec?

In the context of the Training Wizard, a **spec** (short for _specification_) is a type of Python class that can be fully initialized by a config file.

Here's a simple example of a spec:

```py
class LoraConfigSpec(...):
    """The specification for the LoRA model. Mimics the HuggingFace config."""

    r: int = 32
    """r parameter for the LoRA model"""

    lora_alpha: int = 64
    """Alpha parameter for the LoRA model. 2*r is a good default value."""

    lora_dropout: float = 0.05
    """The dropout of the LoRA layers"""

    bias: Literal["none", "all", "lora_only"] = "none"
    """The type of bias to use in the LoRA model. Must be one of:
    none
    all
    lora_only
    """

    target_modules: list[str] | str = "all-linear"
    """The modules to apply LoRA to. Must be a non-empty list."""
```

As it is, this class is really just a dataclass with some fields. We can easily initialize it from a config file, though:

```toml title="config.toml"
r = 4
lora_dropout = 0.1
target_modules = ["q", "k", "v"]
```

```py title="load_lora.py"
import tomllib

with open("config.toml") as f:
    config = tomllib.loads(f.read())
    lora = LoraConfigSpec(**config)
```

!!! Tip

    Note that some fields have default values, so we may choose to omit them.

But what if these specs could do more? Hmm. 🤔

## More than just data containers

Specs in the Training Wizard are powered by [Pydantic](https://docs.pydantic.dev/latest/), a powerful **data validation** library that is great at - you guessed it - _validating data_. 

### Field Validators

Let's ensure that the list of `target_modules` will never be empty by declaring our constraint as a _field validator_:

```py
class LoraConfigSpec(...):
    ... # same fields as before

    @field_validator("target_modules")
    @classmethod
    def target_modules_not_empty(cls, v: List[str]) -> List[str]:
        """Validate that the target modules are valid."""
        assert len(v) > 0, "target_modules must be a non-empty list"
        return v
```

Most of this code is boilerplate. All you need to know is that you can define **validators** for certain **fields** that can **assert properties** at **init time**.

### Model Validators

Sometimes, we may want to check that **combinations of fields** work together. For this, we use **model validators**, which also have access to **self**, the class instance. Let's assume that, for some reason, we always wanted `lora_alpha` to be twice as high as `r`. Here's how we would do that:

```py
class LoraConfigSpec(...):
    ... # same fields as before

    @model_validator(mode="after")
    def alpha_must_be_double_r(self) -> "LoraConfigSpec":
        """Validate the early stopping patience."""
        assert self.lora_alpha == self.r * 2, \
            "Oh no, looks like your alpha value is not twice that or r. Please fix it!"
        return self
```

You'll see such validators used in many places, as it allows the Training Wizard to **fail early** and **fail loudly** - saving you from many bugs and quiet failures that can happen otherwise.

## Specs inside specs inside specs inside ...

Most specs don't just have fields that are core python objects. They most often contain **other specs** that manage a configuration of their own. Here's an example from the `TrainingSpec` class, which is at the core of every recipe in the Training Wizard:

```py
class TrainingSpec(...):
    """Specification for a training recipe."""

    recipe_class: str
    """The path to the recipe class (`module.submodule.class_name`). Checks that it points to a valid class."""

    mlflow_experiment_name: Optional[str] = None
    """The name of the MLflow experiment to log to. If None, do not log to MLflow."""

    training_args_spec: TrainingArgumentsSpec
    """The training arguments."""

    dataset_spec: Union[DatasetSpec, StreamingDatasetSpec]
    """The dataset to use for training."""
```

As you can see, Pydantic even supports nesting different specs (or **models**, as they're called by Pydantic - I think you can see why this is problematic for us), optional specs or even spec _unions_!

### Config Files for Nested Specs

**Every spec** can be uniquely represented by a nested dictionary. More conveniently, though, it can be represented by a `toml` config using [tables](https://toml.io/en/v1.0.0#table). In this example, it would look like this:

```toml title="training_spec_config.toml"
spec_class = "training_wizard.recipes.modular.ModularTrainingSpec"

# Omitting mlflow_experiment_name because we don't want to log yet

[wizard_module]
spec_class = "training_wizard.specs.modules.causal_seq2seq.CausalSeq2SeqModule"
... # Some fields

[training_args_spec]
... # Some more fields

[dataset_spec]
... # Some even more fields
```

!!! Unions

    Fields types as `Spec1 | Spec2` can be configured with _either_ `Spec1`s fields or `Spec2`s fields. Pydantic will automatically pick the right one at runtime.

### Even More Nesting

If `training_args_spec` had another spec field `optimizer_spec`, you would define it in the same file using the TOML table syntax:

```toml title="training_spec_config.toml"
recipe_class = "training_wizard.recipes.CausalLanguageModeling"

# Omitting mlflow_experiment_name because we don't want to log yet

[wizard_module]
spec_class = "training_wizard.specs.modules.causal_seq2seq.CausalSeq2SeqModule"
... # Some fields

[wizard_module.transformer_spec]
... # Some nested fields

[wizard_module.transformer_spec.tokenizer_init_kwargs]
... # Some nested nested fields

[training_args_spec]
... # Some fields again

[dataset_spec]
... # Some even more fields again
```

## Spec Abstraction and Inheritance

Just like other class-based architectures, specs in the Training Wizard benefit from abstraction and inheritance for managing complexity and shared functionality. Specs inherit all the **fields** and **validators** of their parent class, as expected. 

All specs also directly or indirectly inherit from **[BaseModel](https://docs.pydantic.dev/latest/api/base_model/)**, which is a Pydantic base class that enables all the Pydantic features in our classes.

## Specs Doing Stuff

Most specs will provide different **calculated fields** in the form of **properties** (oftentimes `cached_property` properties which are only calculated once). Additionally, recipes (which are also specs) will always provide a `main()` method that executes their training routine.

**Methods** are slightly less important, with the exception of the `main()` method at the core of every [recipe](recipe.md).
