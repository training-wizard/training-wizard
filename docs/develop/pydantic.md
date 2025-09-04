# Pydantic troubleshooting

We are using [Pydantic](https://docs.pydantic.dev/latest/) in a lot of parts to initialize complex classes from a single TOML file. While this is convenient and works out of the box 95% of the time, the remaining 5% can include errors that are hard to debug because they happen in code that is injected and not called explicitly. This page collects problems we've encountered and possible solutions to speed up their resolution in the future.

## I specified an alias, but now the original field name is not accepted anymore

This is intentional (albeit misleading). The [Pydantic documentation](https://docs.pydantic.dev/latest/concepts/alias/) specifies the following:

> An alias is an alternative name for a field, used when serializing and deserializing data.

What this *means* is that the alias name and *only* the alias name of the field will be used for serializing and deserializing. If you want alternatives, you have to use [AliasChoices](https://docs.pydantic.dev/latest/concepts/alias/#aliaspath-and-aliaschoices) instead:

```python
class User(BaseModel):
    first_name: str = Field(validation_alias=AliasChoices('first_name', 'fname'))
    last_name: str = Field(validation_alias=AliasChoices('last_name', 'lname'))
```

## I have a field with a union type, but Pydantic initializes the wrong type

Pydantic is pretty crafty at automatically resolving complex union types in fields, but sometimes it needs a bit of help. One prominent example where this is the case is when `before` validators are involved.

Consider the following example:
    
```python
from pydantic import BaseModel, model_validator

class TypeA(BaseModel):
    a: int

    @model_validator(mode="before")
    @classmethod
    def check_fields(cls, values):
        print(values["a"])
        return values

class TypeB(BaseModel):
    b: int

    @model_validator(mode="before")
    @classmethod
    def check_fields(cls, values):
        print(values["b"])
        return values

class MyClass(BaseModel):
    first_field: TypeA | TypeB

data = {"first_field": {"a": 1}}
print(MyClass(**data))
```

This looks innocent, but will break because the validators of *both* classes that are part of the union type will be called. In other cases, this can even lead to Pydantic trying to *initialize* the wrong class. The reason here is the `@model_validator(mode="before")`. Since we specify `mode="before"`, Pydantic has not yet resolved the Union type. We are just operating on raw dicts here, so it will try to validate `first_field` both as `TypeA` and `TypeB`, one of which will always fail, of course.

The solution for this problem depends on the specific situation, but in the above example, this can be resolved by providing an explicit `Discriminator` in the union type that helps to distinguish which type to use both before and after initialization:

```python
from typing import Annotated
from pydantic import BaseModel, Discriminator, Tag, model_validator

class TypeA(BaseModel):
    a: int

    @model_validator(mode="before")
    @classmethod
    def check_fields(cls, values):
        print(values["a"])
        return values

class TypeB(BaseModel):
    b: int

    @model_validator(mode="before")
    @classmethod
    def check_fields(cls, values):
        print(values["b"])
        return values

def get_discriminator_value(value):
    if isinstance(value, dict) and "a" in value:
        return "alpha"
    elif isinstance(value, dict) and "b" in value:
        return "beta"
    elif isinstance(value, TypeA):
        return "alpha"
    elif isinstance(value, TypeB):
        return "beta"

class MyClass(BaseModel):
    first_field: Annotated[
        Annotated[TypeA, Tag("alpha")] | Annotated[TypeB, Tag("beta")],
        Discriminator(get_discriminator_value),
    ]

data = {"first_field": {"a": 1}}
print(MyClass(**data))
```