"""jl setup — authenticate and install agent skills."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import questionary
import typer
from rich import box
from rich.panel import Panel
from rich.text import Text

from jarvislabs.cli import options as cli_options, render, state
from jarvislabs.cli.app import app
from jarvislabs.config import load_config, save_config

if TYPE_CHECKING:
    from jarvislabs.client import Client

# Universal path — always installed, covers most agents
UNIVERSAL_PATH = Path("~/.agents/skills/jarvislabs/SKILL.md")
UNIVERSAL_AGENTS = [
    "Codex",
    "Cursor",
    "OpenCode",
    "Amp",
    "Cline",
    "Gemini CLI",
    "GitHub Copilot",
    "Kimi Code CLI",
    "Warp",
]

# Additional agent-specific paths (agents with their own skill directories)
ADDITIONAL_AGENTS: dict[str, tuple[str, Path]] = {
    "claude-code": ("Claude Code", Path("~/.claude/skills/jarvislabs/SKILL.md")),
}

_MUTED = "#8a8a8a"


# ── Login ────────────────────────────────────────────────────────────────────


def _login_flow(token: str | None) -> Client:
    """Authenticate and save token. Returns authenticated Client. Dies on failure."""
    from jarvislabs.client import Client
    from jarvislabs.exceptions import JarvislabsError

    config = load_config()
    existing = config.get("auth", {}).get("token")

    # Already authenticated — ask to re-authenticate
    if existing and not token:
        try:
            client = Client(api_key=existing)
            info = client.account.user_info()
            render.success(f"Logged in as [bold]{info.name}[/bold]")
            if state.yes or not render.confirm("Re-authenticate with a different account?"):
                return client
        except JarvislabsError:
            render.warning("Saved token is invalid.")

    # Prompt for token
    if not token:
        render.info("Get your API key at: [magenta]https://jarvislabs.ai/settings/api-keys[/magenta]")
        token = render.console.input("[yellow]?[/yellow] API token: ", password=True).strip()

    if not token:
        render.die("No token provided.")

    try:
        with render.spinner("Authenticating..."):
            client = Client(api_key=token)
            info = client.account.user_info()
    except JarvislabsError as e:
        render.die(f"Authentication failed: {e}")

    config.setdefault("auth", {})["token"] = token
    save_config(config)
    render.success(f"Logged in as [bold]{info.name}[/bold]")
    return client


# ── Skill loading ────────────────────────────────────────────────────────────


def _load_bundled_skill() -> str:
    """Load SKILL.md bundled with the package."""
    import importlib.resources

    try:
        ref = importlib.resources.files("jarvislabs.skills") / "SKILL.md"
        return ref.read_text()
    except Exception:
        render.die("Could not find SKILL.md. Try reinstalling jarvislabs.")


# ── Agent selection ──────────────────────────────────────────────────────────


def _select_additional_agents_interactive() -> list[str]:
    """Interactive multi-select for additional agent-specific paths. Returns selected agent keys."""
    if not ADDITIONAL_AGENTS:
        return []

    import questionary.prompts.common as _qc

    _qc.INDICATOR_SELECTED = "✓"
    _qc.INDICATOR_UNSELECTED = "○"

    choices = [
        questionary.Choice(
            title=f"{label:12s}  →  {path.expanduser()}",
            value=key,
        )
        for key, (label, path) in ADDITIONAL_AGENTS.items()
    ]

    render.console.print()
    render.console.print("[bold]Universal (.agents/skills)[/bold] [dim]— always included[/dim]")
    for name in UNIVERSAL_AGENTS:
        render.console.print(f"  [dim]•[/dim] {name}")
    render.console.print()

    selected = questionary.checkbox(
        "Install to additional agent-specific paths?",
        choices=choices,
        style=questionary.Style(
            [
                ("selected", "noreverse fg:#73daca"),
                ("highlighted", "bold"),
                ("pointer", "fg:#73daca bold"),
            ]
        ),
    ).ask()

    if selected is None:  # user pressed Ctrl+C
        raise typer.Exit()

    return selected


def _parse_agents_flag(agents: str) -> list[str]:
    """Parse --agents flag value into validated list of additional agent keys."""
    if agents == "all":
        return list(ADDITIONAL_AGENTS.keys())
    agent_list = [a.strip() for a in agents.split(",")]
    valid_keys = {*ADDITIONAL_AGENTS.keys(), "universal"}
    invalid = [a for a in agent_list if a not in valid_keys]
    if invalid:
        render.die(f"Unknown agents: {', '.join(invalid)}. Valid: {', '.join(ADDITIONAL_AGENTS.keys())}, all")
    # Filter out "universal" since it's always installed
    return [a for a in agent_list if a != "universal"]


# ── Skill installation ───────────────────────────────────────────────────────


def _install_skill_to_path(content: str, path: Path) -> Path:
    """Write skill file to a single path. Returns resolved path."""
    resolved = path.expanduser()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(content)
    return resolved


def _skill_install_flow(agents_flag: str | None) -> list[tuple[str, Path]]:
    """Run the skill install flow. Returns installed (label, path) pairs."""
    render.console.print()
    render.console.print("[bold cyan]Agent Skills[/bold cyan]")
    render.console.print(
        "  Skills teach your coding agents how to use [bold]jl[/bold] to create and manage\n"
        "  GPU instances, run training scripts, and monitor experiments on your behalf."
    )
    render.console.print()

    if not agents_flag and not render.confirm("Install agent skills?", skip=state.yes):
        render.info("Skipped. You can run [bold]jl setup[/bold] again later.")
        return []

    # Determine additional agent-specific installs (validate before any writes)
    if agents_flag:
        additional = _parse_agents_flag(agents_flag)
    elif state.yes:
        # --yes auto-selects all additional agents
        additional = list(ADDITIONAL_AGENTS.keys())
    else:
        additional = _select_additional_agents_interactive()

    content = _load_bundled_skill()
    installed: list[tuple[str, Path]] = []

    # Always install universal (.agents/skills)
    universal_resolved = _install_skill_to_path(content, UNIVERSAL_PATH)
    agents_covered = ", ".join(UNIVERSAL_AGENTS)
    installed.append((f"Universal ({agents_covered})", universal_resolved))

    for key in additional:
        label, path = ADDITIONAL_AGENTS[key]
        resolved = _install_skill_to_path(content, path)
        installed.append((label, resolved))

    render.console.print()
    render.console.print("[bold]Skills installed:[/bold]")
    for label, path in installed:
        render.success(f"{label}  →  {path}")

    return installed


# ── Display ──────────────────────────────────────────────────────────────────


def _show_account_status(client: Client) -> None:
    """Fetch and display account status panel."""
    with render.spinner("Fetching account info..."):
        info = client.account.user_info()
        bal = client.account.balance()
        metrics = client.account.resource_metrics()
        currency = client.account.currency()

    sym = render.currency_symbol(currency)
    render.console.print()
    render.account_status(info, bal, metrics, sym)


def _show_getting_started() -> None:
    """Show getting started guide with Rich panels."""
    con = render.stdout_console

    con.print()

    from rich.console import Group

    # ── Workflow overview
    workflow = Text()
    workflow.append("\n")
    workflow.append("  write code", style="bold cyan")
    workflow.append("  →  ", style=_MUTED)
    workflow.append("run on GPU", style="bold green")
    workflow.append("  →  ", style=_MUTED)
    workflow.append("monitor", style="bold yellow")
    workflow.append("  →  ", style=_MUTED)
    workflow.append("iterate", style="bold magenta")
    workflow.append("\n")

    # ── Using with coding agents
    agent_text = Text()
    agent_text.append("Your coding agents can now use jl to manage GPU instances,\n", style=_MUTED)
    agent_text.append("run experiments, and monitor training. Just ask them:\n", style=_MUTED)
    agent_text.append("\n")
    agent_text.append('"Run my training script train.py on an A100 GPU.\n', style="italic")
    agent_text.append(" Monitor the loss and let me know when it's done.\"\n", style="italic")
    agent_text.append("\n")
    agent_text.append("The agent will create instances, upload code, start runs,\n", style=_MUTED)
    agent_text.append("monitor logs, and clean up — all using jl commands.", style=_MUTED)

    # ── Instance lifecycle
    instance_text = Text()
    instance_text.append("Containers", style="bold green")
    instance_text.append("  Pre-configured with PyTorch, Jupyter, and IDE. ", style=_MUTED)
    instance_text.append("Default.\n", style=_MUTED)
    instance_text.append("VMs       ", style="bold yellow")
    instance_text.append("  Bare-metal SSH access. Use ", style=_MUTED)
    instance_text.append("--vm", style="bold")
    instance_text.append(".\n\n", style=_MUTED)
    instance_text.append("Create    ", style=_MUTED)
    instance_text.append("jl create --gpu A100\n", style="bold")
    instance_text.append("Create VM ", style=_MUTED)
    instance_text.append("jl create --gpu A100-80GB --vm\n", style="bold")
    instance_text.append("List      ", style=_MUTED)
    instance_text.append("jl list\n", style="bold")
    instance_text.append("Pause     ", style=_MUTED)
    instance_text.append("jl pause <id>", style="bold")
    instance_text.append("              stops billing, keeps storage\n", style=_MUTED)
    instance_text.append("Resume    ", style=_MUTED)
    instance_text.append("jl resume <id>", style="bold")
    instance_text.append("             pick up where you left off\n", style=_MUTED)
    instance_text.append("Destroy   ", style=_MUTED)
    instance_text.append("jl destroy <id>", style="bold")
    instance_text.append("            deletes instance + all data\n", style=_MUTED)
    instance_text.append("\n")
    instance_text.append("Tip: always destroy instances you no longer need to stop billing.", style=f"{_MUTED} italic")

    # ── Run management
    run_text = Text()
    run_text.append("Run a file     ", style=_MUTED)
    run_text.append("jl run train.py --gpu A100\n", style="bold")
    run_text.append("Run a project  ", style=_MUTED)
    run_text.append("jl run . --script train.py --on <id>\n", style="bold")
    run_text.append("Check logs     ", style=_MUTED)
    run_text.append("jl run logs <run_id> --tail 50\n", style="bold")
    run_text.append("Stop a run     ", style=_MUTED)
    run_text.append("jl run stop <run_id>\n", style="bold")
    run_text.append("List runs      ", style=_MUTED)
    run_text.append("jl run list --status running", style="bold")

    # ── Quick reference
    quick = Text()
    quick.append("$ jl gpus", style="bold")
    quick.append("                  available GPUs and pricing\n", style=_MUTED)
    quick.append("$ jl status", style="bold")
    quick.append("                account balance and resources\n", style=_MUTED)
    quick.append("$ jl <command> --help", style="bold")
    quick.append("   help for any command\n", style=_MUTED)
    quick.append("  e.g. ", style=_MUTED)
    quick.append("jl create --help", style="bold")
    quick.append("  ", style=_MUTED)
    quick.append("jl run --help", style="bold")
    quick.append("  ", style=_MUTED)
    quick.append("jl ssh-key --help", style="bold")

    # ── Wrap everything in one outer panel
    con.print(
        Panel(
            Group(
                workflow,
                Panel(
                    agent_text,
                    title="[bold]using with coding agents[/bold]",
                    border_style="magenta",
                    box=box.ROUNDED,
                    padding=(0, 1),
                    expand=False,
                ),
                Panel(
                    instance_text,
                    title="[bold]instance lifecycle[/bold]",
                    border_style="green",
                    box=box.ROUNDED,
                    padding=(0, 1),
                    expand=False,
                ),
                Panel(
                    run_text,
                    title="[bold]run management[/bold]",
                    border_style="yellow",
                    box=box.ROUNDED,
                    padding=(0, 1),
                    expand=False,
                ),
                Panel(
                    quick,
                    title="[bold]quick reference[/bold]",
                    border_style=_MUTED,
                    box=box.ROUNDED,
                    padding=(0, 1),
                    expand=False,
                ),
            ),
            title="[bold]get started[/bold]",
            border_style="cyan",
            box=box.HEAVY,
            padding=(0, 1),
        )
    )


# ── Command ──────────────────────────────────────────────────────────────────


@app.command(rich_help_panel="Account")
def setup(
    token: Annotated[str | None, typer.Option("--token", "-t", help="API token (skips interactive prompt).")] = None,
    agents: Annotated[
        str | None,
        typer.Option(
            "--agents", help="Comma-separated additional agents: claude-code,all (.agents/skills always installed)"
        ),
    ] = None,
    yes: cli_options.YesOption = False,
) -> None:
    """Set up the JarvisLabs CLI: authenticate and install agent skills."""
    cli_options.apply_command_options(yes=yes)

    render.console.print()
    render.console.print("[bold cyan]⚡ JarvisLabs CLI Setup[/bold cyan]")

    # Step 1: Authentication
    client = _login_flow(token)

    # Step 2: Skill installation
    _skill_install_flow(agents)

    # Show account status + getting started guide
    _show_account_status(client)
    _show_getting_started()
