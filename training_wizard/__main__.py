"""This module provides a command-line interface for the `training_wizard` module.

If user runs `python -m training_wizard` provide the command-line interface.
"""

from .commands.training_wizard import training_wizard

training_wizard()
