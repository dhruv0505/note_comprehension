"""Serverless model deployment commands — `jl deploy`.

Thin wrapper over ``client.deployments``: parse flags, call the SDK, render output.
"""

from __future__ import annotations

import os
import sys
from typing import Annotated

import typer

from jarvislabs.cli import options as cli_options, render, state
from jarvislabs.cli.app import app, get_client

BETA_NOTICE_TEXT = (
    "Serverless is in beta. Limits and behavior may change. "
    "If you hit issues or have feedback, email support@jarvislabs.ai."
)
BETA_NOTICE = f"[bold bright_yellow]{BETA_NOTICE_TEXT}[/bold bright_yellow]"
BETA_HELP_NOTICE = (
    "[bold bright_yellow]Serverless is in beta. We're actively improving it and would love early feedback. "
    "Limits and behavior may change; if you hit issues or have suggestions, email support@jarvislabs.ai.[/bold bright_yellow]"
)

deploy_app = typer.Typer(
    name="deploy",
    help=(
        "Create and manage serverless model deployments [bold bright_yellow](beta)[/bold bright_yellow].\n\n"
        + BETA_HELP_NOTICE
    ),
)
app.add_typer(deploy_app, rich_help_panel="[bold bright_yellow]Serverless (Beta)[/bold bright_yellow]")

RegionHintOption = Annotated[
    str | None, typer.Option("--region", help="Region of the deployment (skips the cross-region search).")
]


def _parse_kv(pairs: list[str] | None, *, flag: str, redact: bool = False) -> dict[str, str]:
    """Parse repeatable ``key=value`` flags. Split on the first `=`; last key wins.

    ``redact`` keeps the offending value out of the error message (for secrets).
    """
    result: dict[str, str] = {}
    for raw in pairs or []:
        if "=" not in raw:
            shown = "" if redact else f" {raw!r}"
            render.die(f"Invalid {flag}{shown}: expected key=value")
        key, value = raw.split("=", 1)
        result[key] = value
    return result


@deploy_app.command("create")
def deploy_create(
    name: Annotated[str, typer.Option("--name", help="Deployment name.")],
    region: Annotated[str, typer.Option("--region", help="Region (IN1 or IN2).")],
    framework: Annotated[str, typer.Option("--framework", help="vllm | sglang | ollama.")],
    gpu: Annotated[str, typer.Option("--gpu", help="GPU type.")],
    gpus_per_worker: Annotated[int, typer.Option("--gpus-per-worker", help="GPUs per worker.")],
    min_workers: Annotated[int, typer.Option("--min-workers", help="Minimum workers (0-100).")],
    max_workers: Annotated[int, typer.Option("--max-workers", help="Maximum workers (0-100).")],
    idle_timeout: Annotated[int, typer.Option("--idle-timeout", help="Idle timeout in seconds (0-86400).")],
    wait_time: Annotated[int, typer.Option("--wait-time", help="Wait time in seconds (0-600).")],
    storage: Annotated[int, typer.Option("--storage", help="Deployment filesystem size in GB (min 50).")],
    model: Annotated[str, typer.Option("--model", help="Model id (e.g. a Hugging Face repo id).")],
    concurrent: Annotated[
        int | None, typer.Option("--concurrent", help="Concurrent requests per worker (default 1).")
    ] = None,
    arg: Annotated[
        list[str] | None,
        typer.Option("--arg", help="Repeatable framework flag as key=value (e.g. --arg max-model-len=8192)."),
    ] = None,
    env: Annotated[
        list[str] | None, typer.Option("--env", help="Repeatable env key=value. HF token via --env HF_TOKEN=...")
    ] = None,
    detach: Annotated[bool, typer.Option("--detach", help="Skip the wait: print the id and exit.")] = False,
    yes: cli_options.YesOption = False,
    json_output: cli_options.JsonOption = False,
) -> None:
    """Create a serverless deployment and poll until running (unless --detach).

    Use --arg to pass extra framework flags as key=value.

    [bold bright_yellow]Serverless is in beta. We're actively improving it and would love early feedback. Limits and behavior may change; if you hit issues or have suggestions, email support@jarvislabs.ai.[/bold bright_yellow]
    """
    cli_options.apply_command_options(json_output=json_output, yes=yes)
    args = _parse_kv(arg, flag="--arg")
    env_dict = _parse_kv(env, flag="--env", redact=True)

    details = (
        f"name={name!r}, region={render.region_input_label(region)}, framework={framework}, "
        f"gpu={gpus_per_worker}x {gpu}, workers={min_workers}-{max_workers}, storage={storage}GB"
    )
    render.warning(BETA_NOTICE)
    if not render.confirm(f"Create deployment ({details})?", skip=state.yes):
        raise typer.Exit()

    client = get_client()
    with render.spinner("Creating deployment..."):
        deployment_id = client.deployments.create(
            name=name,
            region=region,
            framework=framework,
            gpu=gpu,
            model=model,
            gpus_per_worker=gpus_per_worker,
            min_workers=min_workers,
            max_workers=max_workers,
            idle_timeout=idle_timeout,
            wait_time=wait_time,
            storage=storage,
            concurrent=concurrent,
            args=args or None,
            env=env_dict or None,
        )
    # Print the id immediately so Ctrl-C can never orphan it silently.
    render.info(f"Deployment {deployment_id} created.")
    followup = f"Check: jl deploy get {deployment_id} · Delete: jl deploy delete {deployment_id}"

    if detach:
        if state.json_output:
            render.print_json({"deployment_id": deployment_id, "region": render.region_input_label(region)})
            return
        render.info(f"Detached. {followup}")
        return

    try:
        with render.spinner("Deploying..."):
            deployment = client.deployments.wait_until_running(deployment_id, region=region)
    except KeyboardInterrupt:
        # The deployment keeps building — hand back the id in both modes so it's never lost.
        if state.json_output:
            render.print_json({"deployment_id": deployment_id, "region": render.region_input_label(region)})
        else:
            render.info(f"Detached. The deployment keeps running. {followup}")
        raise typer.Exit() from None

    base_url = client.deployments.openai_base_url(deployment_id, region=deployment.region)
    if state.json_output:
        render.print_json({**deployment.model_dump(), "openai_base_url": base_url})
        return
    render.deployment_running_handoff(base_url, deployment.served_model)


@deploy_app.command("list")
def deploy_list(
    wide: Annotated[bool, typer.Option("--wide", help="Show all columns (framework, workers, concurrency).")] = False,
    json_output: cli_options.JsonOption = False,
) -> None:
    """List your deployments across all serverless regions, newest first."""
    cli_options.apply_command_options(json_output=json_output)
    client = get_client()
    with render.spinner("Fetching deployments..."):
        result = client.deployments.list()

    if state.json_output:
        render.print_json(result)
        return
    render.deployments_table(result, wide=wide)


@deploy_app.command("get")
def deploy_get(
    deployment_id: Annotated[str, typer.Argument(help="Deployment id.")],
    region: RegionHintOption = None,
    json_output: cli_options.JsonOption = False,
) -> None:
    """Show full detail of one deployment, including worker info."""
    cli_options.apply_command_options(json_output=json_output)
    client = get_client()
    with render.spinner("Fetching deployment..."):
        deployment = client.deployments.get(deployment_id, region=region)

    base_url = None
    if deployment.status == "running":
        base_url = client.deployments.openai_base_url(deployment_id, region=deployment.region)

    if state.json_output:
        payload = deployment.model_dump()
        if base_url:
            payload["openai_base_url"] = base_url
        render.print_json(payload)
        return
    render.deployment_detail(deployment, base_url=base_url)


@deploy_app.command("status")
def deploy_status(
    deployment_id: Annotated[str, typer.Argument(help="Deployment id.")],
    region: RegionHintOption = None,
    json_output: cli_options.JsonOption = False,
) -> None:
    """Status check for one deployment."""
    cli_options.apply_command_options(json_output=json_output)
    client = get_client()
    with render.spinner("Checking status..."):
        status = client.deployments.status(deployment_id, region=region)

    if state.json_output:
        render.print_json(status)
        return
    render.deployment_status_line(status)


@deploy_app.command("logs")
def deploy_logs(
    deployment_id: Annotated[str, typer.Argument(help="Deployment id.")],
    region: RegionHintOption = None,
    tail: Annotated[int, typer.Option("--tail", help="Last N lines from each worker (0 = live only).")] = 100,
    follow: Annotated[
        bool, typer.Option("--follow/--no-follow", "-f", help="Keep streaming new lines (Ctrl+C to stop).")
    ] = True,
    worker: Annotated[
        int | None, typer.Option("--worker", help="Only this worker id (ids shown in `jl deploy get`).")
    ] = None,
    json_output: cli_options.JsonOption = False,
) -> None:
    """Stream a deployment's logs, live from its running workers (logs are not stored)."""
    cli_options.apply_command_options(json_output=json_output)
    if state.json_output:
        render.die("--json is not supported for log streaming.")

    client = get_client()
    # Read the record before streaming: it resolves the region once (so logs()
    # skips its own search) and carries the status used to explain an empty
    # stream — no parsing it out of notice text.
    with render.spinner("Fetching deployment..."):
        deployment = client.deployments.get(deployment_id, region=region)

    events = client.deployments.logs(deployment_id, region=deployment.region, tail=tail, follow=follow, worker=worker)
    try:
        for kind, text in events:
            if kind == "log":
                print(text, flush=True)  # noqa: T201 - deployment logs intentionally stream raw lines to stdout.
                continue
            if text == "no active workers":
                text = _no_workers_message(deployment.status, deployment_id)
            render.info(text)
    except KeyboardInterrupt:
        raise typer.Exit() from None
    except BrokenPipeError:
        # The reader went away (e.g. `| head`); exit quietly. Point stdout at
        # devnull so the interpreter's final flush doesn't complain.
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        raise typer.Exit() from None


def _no_workers_message(status: str, deployment_id: str) -> str:
    """Why the log stream is empty, in terms of the deployment's status."""
    if status in ("failed", "cleaning"):
        return (
            "No running workers — the deployment failed; logs stream live and are not stored. "
            f"For the failure reason: jl deploy get {deployment_id}"
        )
    if status == "deleting":
        return "No running workers — the deployment is being deleted."
    return "No workers up yet — logs start once a worker is running. Try again shortly."


@deploy_app.command("update")
def deploy_update(
    deployment_id: Annotated[str, typer.Argument(help="Deployment id.")],
    name: Annotated[str | None, typer.Option("--name", help="New name.")] = None,
    idle_timeout: Annotated[int | None, typer.Option("--idle-timeout", help="New idle timeout in seconds.")] = None,
    wait_time: Annotated[int | None, typer.Option("--wait-time", help="New wait time in seconds.")] = None,
    region: RegionHintOption = None,
    json_output: cli_options.JsonOption = False,
) -> None:
    """Patch a running deployment. Only name/idle_timeout/wait_time are mutable."""
    cli_options.apply_command_options(json_output=json_output)

    client = get_client()
    with render.spinner("Updating deployment..."):
        deployment = client.deployments.update(
            deployment_id, name=name, idle_timeout=idle_timeout, wait_time=wait_time, region=region
        )

    if state.json_output:
        render.print_json(deployment)
        return
    render.deployment_detail(deployment)


@deploy_app.command("delete")
def deploy_delete(
    deployment_id: Annotated[str, typer.Argument(help="Deployment id.")],
    yes: cli_options.YesOption = False,
    region: RegionHintOption = None,
    json_output: cli_options.JsonOption = False,
) -> None:
    """Tear down a deployment."""
    cli_options.apply_command_options(json_output=json_output, yes=yes)

    if not render.confirm(f"Delete deployment {deployment_id}?", skip=state.yes):
        raise typer.Exit()

    client = get_client()
    with render.spinner("Deleting deployment..."):
        client.deployments.delete(deployment_id, region=region)

    if state.json_output:
        render.print_json({"success": True, "deployment_id": deployment_id})
        return
    render.success(f"Deleting deployment {deployment_id}.")
