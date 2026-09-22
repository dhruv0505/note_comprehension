from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from jarvislabs.cli import options as cli_options, render, state
from jarvislabs.cli.app import app, get_client
from jarvislabs.config import load_config, save_config
from jarvislabs.constants import EUROPE_GPU_TYPES, EUROPE_REGION, INDIA_NOIDA_REGION

EUROPE_AVAILABILITY_NOTE = "EU1 H100/H200 launches are currently limited to 1 GPU."


def _maybe_eu_note(gpus) -> None:
    if any(gpu.region == EUROPE_REGION and gpu.gpu_type in EUROPE_GPU_TYPES for gpu in gpus):
        render.info(EUROPE_AVAILABILITY_NOTE)


@app.command(rich_help_panel="Account")
def logout(
    json_output: cli_options.JsonOption = False,
) -> None:
    """Remove saved API token from config file."""
    cli_options.apply_command_options(json_output=json_output)
    config = load_config()
    logged_out = "auth" in config and "token" in config["auth"]
    if logged_out:
        del config["auth"]["token"]
        if not config["auth"]:
            del config["auth"]
        save_config(config)

    if state.json_output:
        render.print_json({"success": True, "logged_out": logged_out})
        return
    if logged_out:
        render.success("Logged out — token removed from config.")
    else:
        render.info("No saved token found.")


@app.command(rich_help_panel="Account")
def status(
    json_output: cli_options.JsonOption = False,
) -> None:
    """Show account info, balance, and resource summary."""
    cli_options.apply_command_options(json_output=json_output)
    client = get_client()
    with render.spinner("Fetching account info..."):
        info = client.account.user_info()
        bal = client.account.balance()
        metrics = client.account.resource_metrics()
        currency = client.account.currency()
        sym = render.currency_symbol(currency)

    if state.json_output:
        render.print_json(
            {
                "user": info.model_dump(),
                "balance": bal.model_dump(),
                "resources": metrics.model_dump(),
                "currency": currency,
            }
        )
        return

    render.account_status(info, bal, metrics, sym)


@app.command(rich_help_panel="Resources")
def gpus(
    json_output: cli_options.JsonOption = False,
) -> None:
    """Show GPU availability and pricing."""
    cli_options.apply_command_options(json_output=json_output)
    client = get_client()
    with render.spinner("Fetching GPU availability..."):
        availability = client.account.gpu_availability()
        currency = client.account.currency()

    if state.json_output:
        render.print_json(availability)
        return

    render.gpu_table(availability, currency)
    _maybe_eu_note(availability)


@app.command(rich_help_panel="Resources")
def resources(
    json_output: cli_options.JsonOption = False,
) -> None:
    """Show GPU and CPU VM availability and pricing."""
    cli_options.apply_command_options(json_output=json_output)
    client = get_client()
    with render.spinner("Fetching resources..."):
        meta = client.account.resources()
        gpus = client.account.gpu_availability_from(meta)
        currency = client.account.currency()

    if state.json_output:
        render.print_json({"gpus": gpus, "cpu_meta": render.jsonable_cpu_meta(meta.cpu_meta)})
        return

    render.gpu_table(gpus, currency, show_legend=False)
    render.cpu_vm_table(meta.cpu_meta, currency, show_legend=False)
    render.availability_legend()
    _maybe_eu_note(gpus)


@app.command(rich_help_panel="Resources")
def cpus(
    json_output: cli_options.JsonOption = False,
) -> None:
    """Show CPU VM availability and pricing."""
    cli_options.apply_command_options(json_output=json_output)
    client = get_client()
    with render.spinner("Fetching CPU availability..."):
        meta = client.account.resources()
        currency = client.account.currency()

    if state.json_output:
        render.print_json(render.jsonable_cpu_meta(meta.cpu_meta))
        return

    render.cpu_vm_table(meta.cpu_meta, currency)


@app.command(rich_help_panel="Resources")
def templates(
    json_output: cli_options.JsonOption = False,
) -> None:
    """List available instance templates."""
    cli_options.apply_command_options(json_output=json_output)
    client = get_client()
    with render.spinner("Fetching templates..."):
        tpls = client.account.templates()

    if state.json_output:
        render.print_json(tpls)
        return

    render.templates_table(tpls)


ssh_key_app = typer.Typer(name="ssh-key", help="Manage SSH keys.")
app.add_typer(ssh_key_app, rich_help_panel="Infrastructure")

scripts_app = typer.Typer(name="scripts", help="Manage startup scripts.")
app.add_typer(scripts_app, rich_help_panel="Infrastructure")

filesystem_app = typer.Typer(name="filesystem", help="Manage persistent filesystems.")
app.add_typer(filesystem_app, rich_help_panel="Infrastructure")

vpc_app = typer.Typer(name="vpc", help="Manage VPCs (private networks for VMs).")
app.add_typer(vpc_app, rich_help_panel="Infrastructure")


@ssh_key_app.command("list")
def ssh_key_list(
    json_output: cli_options.JsonOption = False,
) -> None:
    """List SSH keys."""
    cli_options.apply_command_options(json_output=json_output)
    client = get_client()
    with render.spinner("Fetching SSH keys..."):
        keys = client.ssh_keys.list()

    if state.json_output:
        render.print_json(keys)
        return

    render.ssh_keys_table(keys)


@ssh_key_app.command("add")
def ssh_key_add(
    pubkey_file: Annotated[typer.FileText, typer.Argument(help="Path to public key file.")],
    name: Annotated[str, typer.Option("--name", "-n", help="Name for this key.")],
    json_output: cli_options.JsonOption = False,
) -> None:
    """Add an SSH public key."""
    cli_options.apply_command_options(json_output=json_output)
    key_content = pubkey_file.read().strip()
    if not key_content:
        render.die("Public key file is empty.")

    client = get_client()
    with render.spinner("Adding SSH key..."):
        client.ssh_keys.add(ssh_key=key_content, key_name=name)
    if state.json_output:
        render.print_json({"success": True, "name": name})
        return
    render.success(f"SSH key '{name}' added.")


@ssh_key_app.command("remove")
def ssh_key_remove(
    key_id: Annotated[str, typer.Argument(help="Key ID to remove.")],
    yes: cli_options.YesOption = False,
    json_output: cli_options.JsonOption = False,
) -> None:
    """Remove an SSH key."""
    cli_options.apply_command_options(json_output=json_output, yes=yes)
    if not render.confirm(f"Remove SSH key {key_id}?", skip=state.yes):
        raise typer.Exit()

    client = get_client()
    with render.spinner("Removing SSH key..."):
        client.ssh_keys.remove(key_id)
    if state.json_output:
        render.print_json({"success": True, "key_id": key_id})
        return
    render.success(f"SSH key {key_id} removed.")


@scripts_app.command("list")
def scripts_list(
    json_output: cli_options.JsonOption = False,
) -> None:
    """List startup scripts."""
    cli_options.apply_command_options(json_output=json_output)
    client = get_client()
    with render.spinner("Fetching startup scripts..."):
        scripts = client.scripts.list()

    if state.json_output:
        render.print_json(scripts)
        return

    render.scripts_table(scripts)


@scripts_app.command("add")
def scripts_add(
    script_file: Annotated[typer.FileBinaryRead, typer.Argument(help="Path to script file.")],
    name: Annotated[str | None, typer.Option("--name", "-n", help="Script name (defaults to filename stem).")] = None,
    json_output: cli_options.JsonOption = False,
) -> None:
    """Add a startup script."""
    cli_options.apply_command_options(json_output=json_output)
    content = script_file.read()
    if not content.strip():
        render.die("Script file is empty.")

    script_name = (name or "").strip() or Path(script_file.name).stem
    client = get_client()
    with render.spinner("Adding startup script..."):
        client.scripts.add(content, script_name)

    if state.json_output:
        render.print_json({"success": True, "name": script_name})
        return

    render.success(f"Script '{script_name}' added.")


@scripts_app.command("update")
def scripts_update(
    script_id: Annotated[int, typer.Argument(help="Script ID to update.")],
    script_file: Annotated[typer.FileBinaryRead, typer.Argument(help="Path to script file.")],
    json_output: cli_options.JsonOption = False,
) -> None:
    """Update startup script contents."""
    cli_options.apply_command_options(json_output=json_output)
    content = script_file.read()
    if not content.strip():
        render.die("Script file is empty.")

    client = get_client()
    with render.spinner("Updating startup script..."):
        client.scripts.update(script_id, content)

    if state.json_output:
        render.print_json({"success": True, "script_id": script_id})
        return

    render.success(f"Script {script_id} updated.")


@scripts_app.command("remove")
def scripts_remove(
    script_id: Annotated[int, typer.Argument(help="Script ID to remove.")],
    yes: cli_options.YesOption = False,
    json_output: cli_options.JsonOption = False,
) -> None:
    """Remove a startup script."""
    cli_options.apply_command_options(json_output=json_output, yes=yes)
    if not render.confirm(f"Remove startup script {script_id}?", skip=state.yes):
        raise typer.Exit()

    client = get_client()
    with render.spinner("Removing startup script..."):
        client.scripts.remove(script_id)

    if state.json_output:
        render.print_json({"success": True, "script_id": script_id})
        return

    render.success(f"Script {script_id} removed.")


@filesystem_app.command("list")
def filesystem_list(
    json_output: cli_options.JsonOption = False,
) -> None:
    """List filesystems."""
    cli_options.apply_command_options(json_output=json_output)
    client = get_client()
    with render.spinner("Fetching filesystems..."):
        filesystems = client.filesystems.list()

    if state.json_output:
        render.print_json(filesystems)
        return

    render.filesystems_table(filesystems)


@filesystem_app.command("create")
def filesystem_create(
    name: Annotated[str, typer.Option("--name", "-n", help="Filesystem name.")],
    storage: Annotated[int, typer.Option("--storage", "-s", help="Storage in GB (50-2048).")],
    region: Annotated[str | None, typer.Option("--region", help="Region (IN1 or IN2). Defaults to IN2.")] = None,
    yes: cli_options.YesOption = False,
    json_output: cli_options.JsonOption = False,
) -> None:
    """Create a filesystem."""
    cli_options.apply_command_options(json_output=json_output, yes=yes)
    output_region = render.region_input_label(region, default=INDIA_NOIDA_REGION)
    region_display = f", region={output_region}" if region else ""
    if not render.confirm(f"Create filesystem (name={name!r}, storage={storage}GB{region_display})?", skip=state.yes):
        raise typer.Exit()

    client = get_client()
    with render.spinner("Creating filesystem..."):
        fs_id = client.filesystems.create(fs_name=name, storage=storage, region=region)

    if state.json_output:
        render.print_json(
            {"success": True, "fs_id": fs_id, "fs_name": name, "storage": storage, "region": output_region}
        )
        return

    render.success(f"Filesystem {fs_id} created.")


@filesystem_app.command("edit")
def filesystem_edit(
    fs_id: Annotated[int, typer.Argument(help="Filesystem ID to edit.")],
    storage: Annotated[int, typer.Option("--storage", "-s", help="New storage size in GB (increase only).")],
    yes: cli_options.YesOption = False,
    json_output: cli_options.JsonOption = False,
) -> None:
    """Expand filesystem storage."""
    cli_options.apply_command_options(json_output=json_output, yes=yes)
    if not render.confirm(f"Expand filesystem {fs_id} to {storage}GB?", skip=state.yes):
        raise typer.Exit()

    client = get_client()
    with render.spinner("Updating filesystem..."):
        new_fs_id = client.filesystems.edit(fs_id=fs_id, storage=storage)

    if state.json_output:
        render.print_json({"success": True, "fs_id": new_fs_id, "previous_fs_id": fs_id, "storage": storage})
        return

    render.success(f"Filesystem updated. New filesystem ID: {new_fs_id}.")


@filesystem_app.command("remove")
def filesystem_remove(
    fs_id: Annotated[int, typer.Argument(help="Filesystem ID to remove.")],
    yes: cli_options.YesOption = False,
    json_output: cli_options.JsonOption = False,
) -> None:
    """Delete a filesystem."""
    cli_options.apply_command_options(json_output=json_output, yes=yes)
    if not render.confirm(f"Remove filesystem {fs_id}?", skip=state.yes):
        raise typer.Exit()

    client = get_client()
    with render.spinner("Removing filesystem..."):
        client.filesystems.remove(fs_id)

    if state.json_output:
        render.print_json({"success": True, "fs_id": fs_id})
        return

    render.success(f"Filesystem {fs_id} removed.")


@vpc_app.command("list")
def vpc_list(
    json_output: cli_options.JsonOption = False,
) -> None:
    """List VPCs across all regions."""
    cli_options.apply_command_options(json_output=json_output)
    client = get_client()
    with render.spinner("Fetching VPCs..."):
        vpcs = client.vpcs.list()

    if state.json_output:
        render.print_json(vpcs)
        return

    render.vpcs_table(vpcs)


@vpc_app.command("get")
def vpc_get(
    vpc_id: Annotated[str, typer.Argument(help="VPC ID.")],
    json_output: cli_options.JsonOption = False,
) -> None:
    """Show details of a VPC."""
    cli_options.apply_command_options(json_output=json_output)
    client = get_client()
    with render.spinner("Fetching VPC..."):
        vpc = client.vpcs.get(vpc_id)

    if state.json_output:
        render.print_json(vpc)
        return

    render.vpc_detail(vpc)


@vpc_app.command("ips")
def vpc_ips(
    vpc_id: Annotated[str, typer.Argument(help="VPC ID.")],
    json_output: cli_options.JsonOption = False,
) -> None:
    """List private IPs allocated in a VPC and the machines holding them."""
    cli_options.apply_command_options(json_output=json_output)
    client = get_client()
    with render.spinner("Fetching VPC IPs..."):
        ips = client.vpcs.ips(vpc_id)

    if state.json_output:
        render.print_json(ips)
        return

    render.vpc_ips_table(ips)


@vpc_app.command("create")
def vpc_create(
    name: Annotated[str, typer.Argument(help="VPC name.")],
    cidr: Annotated[str, typer.Option("--cidr", help="IPv4 CIDR block (e.g. 10.50.0.0/24).")],
    region: Annotated[str, typer.Option("--region", help="Region (IN1 or IN2).")],
    yes: cli_options.YesOption = False,
    json_output: cli_options.JsonOption = False,
) -> None:
    """Create a VPC (a private network for VMs)."""
    cli_options.apply_command_options(json_output=json_output, yes=yes)
    region_display = render.region_input_label(region)
    if not render.confirm(f"Create VPC (name={name!r}, cidr={cidr}, region={region_display})?", skip=state.yes):
        raise typer.Exit()

    client = get_client()
    with render.spinner("Creating VPC..."):
        vpc = client.vpcs.create(name=name, cidr=cidr, region=region)

    if state.json_output:
        render.print_json(vpc)
        return

    render.success(f"VPC {vpc.vpc_id} created.")
    render.vpc_detail(vpc)


@vpc_app.command("delete")
def vpc_delete(
    vpc_id: Annotated[str, typer.Argument(help="VPC ID to delete.")],
    yes: cli_options.YesOption = False,
    json_output: cli_options.JsonOption = False,
) -> None:
    """Delete an empty VPC. VPCs with machines in them cannot be deleted."""
    cli_options.apply_command_options(json_output=json_output, yes=yes)
    if not render.confirm(f"Delete VPC {vpc_id}?", skip=state.yes):
        raise typer.Exit()

    client = get_client()
    with render.spinner("Deleting VPC..."):
        client.vpcs.delete(vpc_id)

    if state.json_output:
        render.print_json({"success": True, "vpc_id": vpc_id})
        return

    render.success(f"VPC {vpc_id} deleted.")
