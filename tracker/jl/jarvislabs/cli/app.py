"""Root Typer app — entry point for the `jl` CLI."""

from __future__ import annotations

from typing import Annotated

import typer

from jarvislabs.cli.render import die
from jarvislabs.client import Client
from jarvislabs.exceptions import JarvislabsError

app = typer.Typer(
    name="jl",
    help=(
        "[bold cyan]⚡ JarvisLabs[/bold cyan] GPU Cloud CLI\n\n"
        "[dim]Two types of instances:[/dim]\n"
        "  [green]Containers[/green]  [dim]Pre-configured with PyTorch, Jupyter, and IDE. Default.[/dim]\n"
        "  [yellow]VMs[/yellow]          [dim]Bare-metal SSH access. Use --vm.[/dim]\n\n"
        "[dim]Quick examples:[/dim]\n"
        "  [bold]jl create --gpu A100[/bold]                       [rgb(100,100,100)]container instance[/rgb(100,100,100)]\n"
        "  [bold]jl create --gpu H100 --vm[/bold]                  [rgb(100,100,100)]VM instance[/rgb(100,100,100)]\n"
        "  [bold]jl run train.py --gpu L4[/bold]                   [rgb(100,100,100)]run on a fresh GPU[/rgb(100,100,100)]\n"
        "  [bold]jl run . --script train.py --on <id>[/bold]       [rgb(100,100,100)]upload project and run[/rgb(100,100,100)]\n"
        "  [bold]jl ssh <id>[/bold]                                [rgb(100,100,100)]SSH into an instance[/rgb(100,100,100)]\n"
        "  [bold]jl list[/bold]                                    [rgb(100,100,100)]list all instances[/rgb(100,100,100)]\n"
        "  [bold]jl gpus[/bold]                                    [rgb(100,100,100)]GPU availability and pricing[/rgb(100,100,100)]\n"
        "  [bold]jl pause <id>[/bold]                              [rgb(100,100,100)]stop billing, keep data[/rgb(100,100,100)]\n"
        "  [bold]jl destroy <id>[/bold]                            [rgb(100,100,100)]delete permanently[/rgb(100,100,100)]\n\n"
        "[dim]Every command supports[/dim] [bold]--help[/bold] [dim]for details:[/dim]\n"
        "  [dim]jl create --help    jl run --help    jl ssh-key --help[/dim]"
    ),
    rich_markup_mode="rich",
    pretty_exceptions_enable=False,
)


@app.callback(invoke_without_command=True)
def _global_flags(
    ctx: typer.Context,
    version: Annotated[bool, typer.Option("--version", help="Show version and exit.")] = False,
) -> None:
    if version:
        from importlib.metadata import version as pkg_version

        typer.echo(f"jl {pkg_version('jarvislabs')}")
        raise typer.Exit()

    # If no subcommand was given (and --version wasn't handled above), show help
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit()


def get_client() -> Client:
    """Create a Client using the resolved token. Called lazily by commands that need it."""
    try:
        return Client()
    except JarvislabsError as e:
        die(str(e))


def main() -> None:
    """Entry point for `jl` command (wired via pyproject.toml [project.scripts])."""
    # Command modules import `app` from this module, so they can only load after it exists.
    from jarvislabs.cli import commands, deploy, instance, run, setup  # noqa: F401
    from jarvislabs.cli.update_check import finish_update_check, start_update_check

    start_update_check()

    try:
        app()
    except JarvislabsError as e:
        die(str(e))
    finally:
        finish_update_check()
