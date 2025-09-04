# Coding conventions

Here we collect all the coding conventions and best practices that we want to follow. This mainly concerns Python code, but can also include conventions for other parts of the code.

## 📖 PEP8

Most style questions for Python are governed by [PEP8](https://peps.python.org/pep-0008/), which we adopt with some exceptions (see below).

If you find yourself hard to keep up with PEP8, then you can use the `ruff` formatter. It allows you to think less about formatting and more about the code you write, as it will auto-format the files you are working on. If you use VS Code, you can install the official ruff extension for VS Code. For more details on a set of recommended extensions, read [VS Code tips](vs-code.md).

**⤴️ Exceptions to PEP8 rules**

- The goal of coding conventions is improving readability and maintainability, not uniformity at all costs. Therefore, every rule can be broken when it would otherwise hinder readability. This also may be an argument against using formatters because they do not allow for such flexibility.
- Line length: 79 characters is quite restrictive and only makes sense when working from small terminal windows or when having multiple editors side by side. For our projects, 120 characters are perfectly fine.

## ✍️ Formatting of docstrings

We use [google-style notypes docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings). They may be used by `mkdocstrings` to generate API documentation (along with the method's signature).

We always use type hints in the method's signature. The types of parameters and return values should not be mentioned in the docstring. It is preferable to only have one source for each piece of information to avoid conflicting information (otherwise when a method's signature changes, both places would need updating). Also it makes the docstring look less verbose.

**Example**

```python
def get_token_action(self, token: str, index: int, prob: float, sugg_token: str) -> tuple[int, int, str, float] | None:
    """Get a suggested action for a token.

    This method returns a suggested action for a token.
    The action is represented as a tuple of 4 elements:
        starting position, ending position, replacement token, probability of taking the action.

    Args:
        token: token.
        index: index of token in sentence.
        prob: probability of requesting the action.
        sugg_token: suggested token.

    Raises:
        ValueError: if `sugg_token` is unknown.

    Returns:
        Suggested action for token.
    """
```

## 🛟 Linting

A linter is a tool that can help you follow PEP8, as well as proper formatting of docstrings and provide warnings for many other cases.

We use the ruff linter. If you use VS Code, you can install the official ruff extension for VS Code. For more details on a set of recommended extensions, read [VS Code tips](vs-code.md).

A project using ruff should have it in its development dependencies:

**Example**

```toml
[dependency-groups]
dev = ["ruff>=0.9"]
```

The ruff settings are declared within the `pyproject.toml` file. Some recommended settings for the ruff linter are specified below.

**Recommended settings**

```toml
[tool.ruff]
# Allow lines to be as long as 120.
line-length = 120

[tool.ruff.lint]
select = [
    "D",      # pydocstyle
    "W",      # pycodestyle warnings
    "E",      # pycodestyle errors
    "N",      # pep8-naming
    "I",      # isort
    "ANN001", # Missing type annotation for function argument
    "ANN2",   # Missing type annotation for return value
    "F",      # Pyflakes
    "UP",     # pyupgrade
    "RUF",    # Ruff checks
    "A",      # flake8-builtins (shadowing a Python builtin)
    "B",      # flake8-bugbear
    "C4",     # flake8-comprehensions
    "SIM",    # flake8-simplify
    "Q",      # flake8-quotes
    "PIE",    # flake8-pie
    "T10",    # flake8-debugger
    "TCH",    # flake8-type-checking
    "TID",    # flake8-tidy-imports
    "EXE",    # flake8-executable
    "G",      # flake8-logging-format
    "INT",    # flake8-gettext
    "ISC",    # flake8-implicit-str-concat
    "FLY",    # flynt
    "NPY",    # NumPy-specific rules
    "PD",     # pandas-vet
    "PERF",   # Perflint
    "PGH",    # pygrep-hooks
]

ignore = [
    "PGH003", # we want to just use `type: ignore`
    "TID252", # relative imports
]

[tool.ruff.lint.per-file-ignores]
# Ignore `F401` (unused imports) in all `__init__.py` files.
"__init__.py" = ["F401"]
"**/*.ipynb" = ["ALL"]

[tool.ruff.lint.pydocstyle]
# Use Google docstring style.
convention = "google"

[tool.ruff.lint.flake8-annotations]
# Suppress type annotations for none-returning functions.
suppress-none-returning = true

[tool.ruff.format]
# Line endings will be converted to `\n`.
line-ending = "lf"
```

## 🇹 Typing

Python supports [type hints](https://docs.python.org/3/library/typing.html) via the `typing` module. It is encouraged to add them as they can help to document methods and functions and spot errors before running your code.

We use the new typing syntax that doesn't require the `typing` module for most things.

We always use type hints in the method's signature. The local variables *within* a function do not need to be annotated, unless it is a complicated case that would increase understanding of the code.

In VS Code we use Pylance's basic type checking mode. We will evaluate in the future astral's type checker or mypy.

**Annotations and co_annotations**

We do not use [“annotations”](https://peps.python.org/pep-0563/) or [“co_annotations”](https://peps.python.org/pep-0649/) yet for forward references and other cool stuff until they are supported by CPython (without a `from __future__` import). This will happen in [CPython 3.14](https://docs.python.org/3.14/whatsnew/3.14.html#whatsnew314-pep649) and when we make 3.14 our minimum supported version we will use them. For now, we stringify such cases.

## 🖊️ Dependency management

For dependency management, we are using [uv](https://docs.astral.sh/uv/). It combines the features of [Python venv](https://docs.python.org/3/library/venv.html) and pip while still managing the dependencies in the standard ([PEP 621](https://peps.python.org/pep-0621/)) `pyproject.toml` file format in a single location along with a lock file. We are following all the adopted PEPs that are [PyPA specifications](https://packaging.python.org/en/latest/specifications/).

## 🐍 Future Python migrations

We use CPython 3.11 as the minimum supported version.

If we know that some code can be written better in a newer CPython release, we make a comment that starts with `# TODO: py`  (case-insensitive) and describing how we would have done it in the newer CPython release. e.g. `# TODO: Python 3.11 porting: Import "StrEnum" from enum instead of defining it below.`. 

## 📄 README

The README should contain enough information for a new developer to onboard to the project to the point where they can run some examples in the code. It should at least contain a full installation description.

Apart from that, the README should also describe all major features of the project and should be updated accordingly when a new feature is added.

**Minimal requirements for the README**

- The file should start with a few sentences that explain what the repository is there for.
- The README should contain instructions on how to install it and run all the major workflows covered by the repository.
- These instructions can be minimal in the form of shell snippets. If they are, however, a more detailed explanation for onboarding should be found in additional documents in the `docs` folder and there should be links to this information in the main README.

## 🤝 Contributing

If the repository does not have a `CONTRIBUTING.md` file yet, assume that this was an oversight due to time constraints and create one. 😉

A `CONTRIBUTING.md` file documents all major development workflows. This does not pertain to installing and running the code in the repo, but to everything one has to keep in mind when adding code to the repository.

At the minimum, `CONTRIBUTING.md` should detail all steps required to release a new version of the repository. If the repository uses CI scripts, these should be explained here.

## 🇻 Versioning

The versioning scheme of the project depends on how the project is being consumed. There are two cases:

1. If it is a library that has stable API & CLI assurances defined then it should be [SemVer](https://semver.org/) to signal that.
    1. it is good to add a small definition of what are the changes that should trigger a MINOR or a MAJOR version change.
2. If it is a program that isn't a dependency of another project, then it should be [CalVer](https://calver.org/).

## 🗒️ Changelog

If the repository does not have a changelog yet, assume that this was an oversight due to time constraints and create one. 😉

We use the format from [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

It is good to categorize changes to quickly identify those that may have affected a change or introduced a particular bug. Emojis are a good way to accomplish that since they are immediately recognizable but take up only one character of space. If the changelog uses Emojis for categorization, it should explain the meaning of each symbol at the top.

As a general rule, the changelog should be updated every time a pull request is merged into the repository or an issue is closed.

## ✅ Unit tests

We use `pytest` for unit tests.

**Minimal requirements for unit tests**

- The repository should have at least one unit test that covers the main workflow of that repository.
- If a PR introduces code that is not covered by existing unit tests, a new unit test should be added.
- If a PR introduces a function or class with code that is difficult to understand at first glance, that code should be covered by a unit test.
- Each unit test needs at least one sentence in its docstring that explains the hypothesis that is being covered by the test.

## 🎩 Other style guides

- When in doubt, the [Google style guide for Python code](https://google.github.io/styleguide/pyguide.html) is always a good resource.
