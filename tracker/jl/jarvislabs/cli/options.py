from __future__ import annotations

from typing import Annotated

import click
import typer

from jarvislabs.cli import state

JsonOption = Annotated[bool, typer.Option("--json", help="Output as JSON.")]
YesOption = Annotated[bool, typer.Option("--yes", "-y", help="Skip confirmation prompts.")]


def apply_command_options(*, json_output: bool = False, yes: bool = False) -> None:
    state.json_output = json_output
    state.yes = yes


def option_was_explicit(name: str) -> bool:
    """Tell validation apart from Typer defaults for options with meaningful defaults."""
    ctx = click.get_current_context(silent=True)
    if ctx is None:
        return False
    return ctx.get_parameter_source(name) == click.core.ParameterSource.COMMANDLINE
