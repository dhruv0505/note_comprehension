"""Machine commands — lifecycle, SSH, exec, and file transfer (registered on root app)."""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Annotated

import typer

from jarvislabs.cli import options as cli_options, render, state
from jarvislabs.cli.app import app, get_client
from jarvislabs.cli.options import option_was_explicit
from jarvislabs.constants import DEFAULT_INSTANCE_NAME, DEFAULT_NUM_GPUS, DEFAULT_STORAGE_GB, DEFAULT_TEMPLATE
from jarvislabs.exceptions import SSHError, ValidationError
from jarvislabs.ssh import (
    build_remote_shell_command,
    build_scp_command,
    harden_ssh_parts,
    parse_ssh_command,
    split_ssh_command,
)

if TYPE_CHECKING:
    from jarvislabs.models import Instance

_MACHINE_PANEL = "Machine Management"
_ACCESS_PANEL = "Remote Access"


def _require_running_for_ssh(machine_id: int, status: str) -> None:
    if status != "Running":
        if status == "Paused":
            render.die(f"Instance {machine_id} is paused. Resume it first: jl resume {machine_id}")
        if status in {"Creating", "Resuming"}:
            render.die(f"Instance {machine_id} is not ready yet (status: {status}). Wait for it to reach Running.")
        render.die(f"Instance {machine_id} is not available for SSH (status: {status}).")


def _ssh_parts_or_die(inst: Instance) -> list[str]:
    try:
        return harden_ssh_parts(split_ssh_command(inst.ssh_command))
    except SSHError:
        render.die(f"Cannot parse SSH command: {inst.ssh_command}")


def _resolve_ssh(machine_id: int) -> tuple[Instance, list[str]]:
    client = get_client()
    with render.spinner("Fetching instance..."):
        inst = client.instances.get(machine_id)

    _require_running_for_ssh(machine_id, inst.status)

    if not inst.ssh_command:
        render.die(f"Instance {machine_id} has no SSH command (status: {inst.status}).")

    return inst, _ssh_parts_or_die(inst)


def remote_home(ssh_command: str | None) -> str:
    """Derive the remote home/workspace directory from the SSH command.

    Containers run as root and use /home as a workspace.
    VMs run as a regular user (ubuntu, cloud, …) whose home is /home/<user>.
    """
    if ssh_command:
        try:
            info = parse_ssh_command(ssh_command)
            if info.user and info.user != "root":
                return f"/home/{info.user}"
        except SSHError:
            pass
    return "/home"


def default_upload_dest(source: Path, ssh_command: str | None = None) -> str:
    name = source.name or source.resolve().name
    return f"{remote_home(ssh_command)}/{name}"


def _default_download_dest(source: str) -> str:
    cleaned = source.rstrip("/")
    name = PurePosixPath(cleaned).name
    if not name:
        raise ValueError(f"Cannot infer a local destination from remote path: {source}")
    return name


def _transfer_json(parts: list[str], payload: dict) -> None:
    """Run a transfer command captured and emit the JSON result. Exits nonzero on failure."""
    completed = subprocess.run(parts, capture_output=True, text=True, check=False)
    render.print_json(payload | {"exit_code": completed.returncode})
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def resolve_vm_template(vm: bool, template: str, http_ports: str) -> str:
    """Validate the --vm/--template/--http-ports combination. VMs use the "vm" template."""
    if vm:
        if template != DEFAULT_TEMPLATE:
            render.die("--vm and --template cannot be used together.")
        template = "vm"
        if http_ports:
            render.die("--http-ports is not supported with --vm. VMs are SSH-only.")
    if template.strip().lower() == "vm" and not vm:
        render.die("Use --vm instead of --template vm.")
    return template


@app.command("list", rich_help_panel=_MACHINE_PANEL)
def instance_list(
    json_output: cli_options.JsonOption = False,
) -> None:
    """List all instances."""
    cli_options.apply_command_options(json_output=json_output)
    client = get_client()
    with render.spinner("Fetching instances..."):
        instances = client.instances.list()
        currency = client.account.currency()

    if state.json_output:
        render.print_json(instances)
        return

    render.instances_table(instances, currency)


@app.command("get", rich_help_panel=_MACHINE_PANEL)
def instance_get(
    machine_id: Annotated[int, typer.Argument(help="Instance ID.")],
    json_output: cli_options.JsonOption = False,
) -> None:
    """Show details of a specific instance."""
    cli_options.apply_command_options(json_output=json_output)
    client = get_client()
    with render.spinner("Fetching instance..."):
        inst = client.instances.get(machine_id)
        currency = client.account.currency()

    if state.json_output:
        render.print_json(inst)
        return

    render.instance_detail(inst, currency)


@app.command("create", rich_help_panel=_MACHINE_PANEL)
def instance_create(
    gpu: Annotated[str | None, typer.Option("--gpu", "-g", help="GPU type (e.g. H100, A100, L4).")] = None,
    vm: Annotated[bool, typer.Option("--vm", help="Create a VM instance (SSH-only, no container).")] = False,
    cpu: Annotated[bool, typer.Option("--cpu", help="Create a CPU VM. Requires --vm.")] = False,
    vcpus: Annotated[int | None, typer.Option("--vcpus", help="CPU VM vCPU count.")] = None,
    ram: Annotated[int | None, typer.Option("--ram", help="CPU VM RAM in GB.")] = None,
    template: Annotated[
        str, typer.Option("--template", "-t", help="Framework template for container instances.")
    ] = DEFAULT_TEMPLATE,
    storage: Annotated[int, typer.Option("--storage", "-s", help="Storage in GB.")] = DEFAULT_STORAGE_GB,
    name: Annotated[str, typer.Option("--name", "-n", help="Instance name.")] = DEFAULT_INSTANCE_NAME,
    num_gpus: Annotated[int, typer.Option("--num-gpus", help="Number of GPUs.")] = DEFAULT_NUM_GPUS,
    region: Annotated[str | None, typer.Option("--region", help="Optional region pin (IN1, IN2, EU1).")] = None,
    http_ports: Annotated[
        str, typer.Option("--http-ports", help="Comma-separated HTTP ports to expose (e.g. 7860,8080).")
    ] = "",
    script_id: Annotated[str | None, typer.Option("--script-id", help="Startup script ID to run on launch.")] = None,
    script_args: Annotated[str, typer.Option("--script-args", help="Arguments passed to startup script.")] = "",
    fs_id: Annotated[int | None, typer.Option("--fs-id", help="Filesystem ID to attach.")] = None,
    spot: Annotated[bool, typer.Option("--spot", help="Create a spot GPU container instance.")] = False,
    vpc_id: Annotated[
        str | None,
        typer.Option("--vpc-id", help="VPC to place the VM in. Defaults to the region's default VPC."),
    ] = None,
    yes: cli_options.YesOption = False,
    json_output: cli_options.JsonOption = False,
) -> None:
    """Create a new GPU instance (container or VM)."""
    cli_options.apply_command_options(json_output=json_output, yes=yes)

    if cpu:
        if not vm:
            render.die("--cpu requires --vm.")
        if gpu is not None:
            render.die("--cpu cannot be used with --gpu.")
        if (vcpus is None) != (ram is None):
            render.die("CPU VM sizing requires both --vcpus and --ram.")
        if option_was_explicit("template"):
            render.die("--template is not supported with CPU VMs.")
        if option_was_explicit("num_gpus"):
            render.die("--num-gpus is not supported with CPU VMs.")
        if option_was_explicit("storage") and storage < DEFAULT_STORAGE_GB:
            render.die(f"CPU VMs require at least {DEFAULT_STORAGE_GB}GB storage.")
        if http_ports:
            render.die("--http-ports is not supported with CPU VMs.")
        if script_id is not None or script_args:
            render.die("Startup scripts are not supported with CPU VMs.")
        if fs_id is not None:
            render.die("--fs-id is not supported with CPU VMs.")
        if spot:
            render.die("--spot is not supported with CPU VMs.")

    template = resolve_vm_template(vm, template, http_ports)
    if not cpu and gpu is None:
        render.die("GPU type is required. Use --gpu <type>, or use --vm --cpu for a CPU VM.")
    if spot and vm:
        render.die("--spot is only supported for GPU container instances.")
    if vpc_id and not vm:
        render.die("--vpc-id requires --vm. Containers cannot join a VPC.")

    client = get_client() if cpu else None
    resolved_cpu_region: str | None = None
    if cpu:
        # A VM and its VPC share a region, so the VPC decides before plan resolution.
        if vpc_id and region is None:
            with render.spinner("Resolving VPC..."):
                region = client.vpcs.get(vpc_id).region
        with render.spinner("Resolving CPU VM plan..."):
            vcpus, ram, resolved_cpu_region = client.instances.resolve_cpu_vm_plan(
                vcpus=vcpus,
                ram=ram,
                region=region,
            )
        region = resolved_cpu_region

    if cpu:
        details = [
            f"cpu={vcpus} vCPU / {ram}GB RAM",
            f"storage={storage}GB",
            f"name={name!r}",
            f"region={render.region_label(resolved_cpu_region)}",
        ]
    else:
        details = [f"gpu={num_gpus}x {gpu}", f"template={template}", f"storage={storage}GB", f"name={name!r}"]
        if spot:
            details.append("spot=true")
    if region and not cpu:
        details.append(f"region={render.region_input_label(region)}")
    if http_ports:
        details.append(f"http_ports={http_ports!r}")
    if script_id:
        details.append(f"script_id={script_id}")
    if script_args:
        details.append(f"script_args={script_args!r}")
    if fs_id is not None:
        details.append(f"fs_id={fs_id}")
    if vpc_id:
        details.append(f"vpc={vpc_id}")
    noun = "CPU VM" if cpu else ("VM" if template == "vm" else "instance")
    prompt = f"Create {noun} ({', '.join(details)})?"
    if not render.confirm(prompt, skip=state.yes):
        raise typer.Exit()

    if client is None:
        client = get_client()
    with render.spinner(f"Creating {noun} — this may take a few seconds..."):
        inst = client.instances.create(
            gpu_type=gpu,
            num_gpus=num_gpus,
            template=template,
            cpu=cpu,
            vcpus=vcpus,
            ram=ram,
            storage=storage,
            name=name,
            region=region,
            http_ports=http_ports,
            script_id=script_id,
            script_args=script_args,
            fs_id=fs_id,
            is_spot=spot,
            vpc_id=vpc_id,
        )

    if state.json_output:
        render.print_json(inst)
        return

    render.success(f"{'VM' if noun == 'VM' else 'Instance'} {inst.machine_id} is Running.")
    render.instance_detail(inst, client.account.currency())


@app.command("rename", rich_help_panel=_MACHINE_PANEL)
def instance_rename(
    machine_id: Annotated[int, typer.Argument(help="Instance ID to rename.")],
    name: Annotated[str, typer.Option("--name", "-n", help="New instance name.")],
    yes: cli_options.YesOption = False,
    json_output: cli_options.JsonOption = False,
) -> None:
    """Rename an instance."""
    cli_options.apply_command_options(json_output=json_output, yes=yes)
    if not render.confirm(f"Rename instance {machine_id} to {name!r}?", skip=state.yes):
        raise typer.Exit()

    client = get_client()
    with render.spinner("Renaming instance..."):
        client.instances.get(machine_id)
        client.instances.rename(machine_id, name)

    if state.json_output:
        render.print_json({"success": True, "machine_id": machine_id, "name": name})
        return

    render.success(f"Instance {machine_id} renamed to {name!r}.")


@app.command("pause", rich_help_panel=_MACHINE_PANEL)
def instance_pause(
    machine_id: Annotated[int, typer.Argument(help="Instance ID to pause.")],
    yes: cli_options.YesOption = False,
    json_output: cli_options.JsonOption = False,
) -> None:
    """Pause a running instance."""
    cli_options.apply_command_options(json_output=json_output, yes=yes)
    client = get_client()
    with render.spinner("Checking instance..."):
        client.instances.get(machine_id)

    if not render.confirm(f"Pause instance {machine_id}?", skip=state.yes):
        raise typer.Exit()

    with render.spinner("Pausing instance..."):
        client.instances.pause(machine_id)

    if state.json_output:
        render.print_json({"success": True, "machine_id": machine_id})
        return

    render.success(f"Instance {machine_id} paused.")


@app.command("resume", rich_help_panel=_MACHINE_PANEL)
def instance_resume(
    machine_id: Annotated[int, typer.Argument(help="Instance ID to resume.")],
    gpu: Annotated[str | None, typer.Option("--gpu", "-g", help="Resume with a different GPU type.")] = None,
    num_gpus: Annotated[int | None, typer.Option("--num-gpus", help="Change number of GPUs.")] = None,
    vcpus: Annotated[int | None, typer.Option("--vcpus", help="CPU VM vCPU count.")] = None,
    ram: Annotated[int | None, typer.Option("--ram", help="CPU VM RAM in GB.")] = None,
    storage: Annotated[
        int | None, typer.Option("--storage", "-s", help="Expand storage (GB). Can only increase.")
    ] = None,
    name: Annotated[str | None, typer.Option("--name", "-n", help="Rename instance.")] = None,
    http_ports: Annotated[
        str, typer.Option("--http-ports", help="Comma-separated HTTP ports to expose (e.g. 7860,8080).")
    ] = "",
    script_id: Annotated[str | None, typer.Option("--script-id", help="Startup script ID to use on resume.")] = None,
    script_args: Annotated[
        str | None, typer.Option("--script-args", help="Arguments passed to startup script.")
    ] = None,
    fs_id: Annotated[int | None, typer.Option("--fs-id", help="Filesystem ID to attach.")] = None,
    spot: Annotated[bool, typer.Option("--spot", help="Resume as a spot GPU container instance.")] = False,
    vpc_id: Annotated[
        str | None,
        typer.Option("--vpc-id", help="Move the VM into this VPC on resume. Omit to keep its current VPC."),
    ] = None,
    yes: cli_options.YesOption = False,
    json_output: cli_options.JsonOption = False,
) -> None:
    """Resume a paused instance. Optionally swap GPU, expand storage, or rename."""
    cli_options.apply_command_options(json_output=json_output, yes=yes)
    changes: list[str] = []
    if gpu:
        changes.append(f"gpu={gpu}")
    if num_gpus is not None:
        changes.append(f"num_gpus={num_gpus}")
    if (vcpus is None) != (ram is None):
        render.die("CPU VM sizing requires both --vcpus and --ram.")
    if vcpus is not None and ram is not None:
        changes.append(f"cpu={vcpus} vCPU / {ram}GB RAM")
    if storage is not None:
        changes.append(f"storage={storage}GB")
    if name is not None:
        changes.append(f"name={name!r}")
    if http_ports:
        changes.append(f"http_ports={http_ports!r}")
    if script_id is not None:
        changes.append(f"script_id={script_id}")
    if script_args is not None:
        changes.append(f"script_args={script_args!r}")
    if fs_id is not None:
        changes.append(f"fs_id={fs_id}")
    if spot:
        changes.append("spot=true")
    if vpc_id:
        changes.append(f"vpc={vpc_id}")

    details = ", ".join(changes) if changes else "current configuration"
    if not render.confirm(f"Resume instance {machine_id} with {details}?", skip=state.yes):
        raise typer.Exit()

    client = get_client()
    with render.spinner("Resuming instance..."):
        inst = client.instances.resume(
            machine_id,
            gpu_type=gpu,
            num_gpus=num_gpus,
            vcpus=vcpus,
            ram=ram,
            storage=storage,
            name=name,
            http_ports=http_ports,
            script_id=script_id,
            script_args=script_args,
            fs_id=fs_id,
            is_spot=spot,
            vpc_id=vpc_id,
        )

    if inst.machine_id != machine_id:
        render.warning(f"Instance ID changed: {machine_id} → {inst.machine_id}")

    if state.json_output:
        render.print_json(inst)
        return

    render.success(f"Instance {inst.machine_id} is Running.")
    render.instance_detail(inst, client.account.currency())


@app.command("destroy", rich_help_panel=_MACHINE_PANEL)
def instance_destroy(
    machine_id: Annotated[int, typer.Argument(help="Instance ID to destroy.")],
    yes: cli_options.YesOption = False,
    json_output: cli_options.JsonOption = False,
) -> None:
    """Permanently destroy an instance."""
    cli_options.apply_command_options(json_output=json_output, yes=yes)
    client = get_client()
    with render.spinner("Checking instance..."):
        client.instances.get(machine_id)

    if not render.confirm(
        f"Destroy instance {machine_id}? This cannot be undone.",
        skip=state.yes,
    ):
        raise typer.Exit()

    with render.spinner("Destroying instance..."):
        client.instances.destroy(machine_id)

    if state.json_output:
        render.print_json({"success": True, "machine_id": machine_id})
        return

    render.success(f"Instance {machine_id} destroyed.")


@app.command("ssh", rich_help_panel=_ACCESS_PANEL)
def instance_ssh(
    machine_id: Annotated[int, typer.Argument(help="Instance ID.")],
    print_command: Annotated[
        bool, typer.Option("--print-command", "-p", help="Print SSH command instead of connecting.")
    ] = False,
    json_output: cli_options.JsonOption = False,
) -> None:
    """SSH into a running instance."""
    cli_options.apply_command_options(json_output=json_output)
    client = get_client()
    with render.spinner("Fetching instance..."):
        inst = client.instances.get(machine_id)

    if not inst.ssh_command:
        render.die(f"Instance {machine_id} has no SSH command (status: {inst.status}).")

    if print_command:
        render.stdout_console.print(inst.ssh_command)
        return

    if state.json_output:
        render.print_json({"ssh_command": inst.ssh_command})
        return

    _require_running_for_ssh(machine_id, inst.status)

    parts = _ssh_parts_or_die(inst)

    render.info(f"Connecting to {machine_id}...")
    raise SystemExit(subprocess.call(parts))


@app.command(
    "exec",
    rich_help_panel=_ACCESS_PANEL,
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def instance_exec(
    ctx: typer.Context,
    machine_id: Annotated[int, typer.Argument(help="Instance ID.")],
    json_output: cli_options.JsonOption = False,
) -> None:
    """Execute a command on a running instance."""
    cli_options.apply_command_options(json_output=json_output)
    if not ctx.args:
        render.die(f"No command specified. Use -- to separate: jl exec {machine_id} -- <command>")

    _, parts = _resolve_ssh(machine_id)
    try:
        remote_command = build_remote_shell_command(ctx.args)
    except (SSHError, ValidationError):
        render.die(f"Cannot prepare SSH command for instance {machine_id}.")

    command_label = shlex.join(ctx.args)
    parts.append(remote_command)

    if state.json_output:
        completed = subprocess.run(parts, capture_output=True, text=True, check=False)
        render.print_json(
            {
                "machine_id": machine_id,
                "command": command_label,
                "exit_code": completed.returncode,
                "stdout": getattr(completed, "stdout", ""),
                "stderr": getattr(completed, "stderr", ""),
            }
        )
        if completed.returncode != 0:
            raise SystemExit(completed.returncode)
        return

    render.info(f"Executing on {machine_id}: {command_label}")
    exit_code = subprocess.call(parts)
    if exit_code != 0:
        if exit_code == 255:
            render.warning(f"SSH or remote command failed on {machine_id} (exit code 255).")
        else:
            render.warning(f"Command on {machine_id} exited with code {exit_code}.")
    raise SystemExit(exit_code)


@app.command("upload", rich_help_panel=_ACCESS_PANEL)
def instance_upload(
    machine_id: Annotated[int, typer.Argument(help="Instance ID.")],
    source: Annotated[
        Path, typer.Argument(exists=True, readable=True, resolve_path=True, help="Local file or directory to upload.")
    ],
    dest: Annotated[
        str | None, typer.Argument(help="Remote destination path. Defaults to remote home directory.")
    ] = None,
    json_output: cli_options.JsonOption = False,
) -> None:
    """Upload a local file or directory to a running instance."""
    cli_options.apply_command_options(json_output=json_output)
    inst, ssh_parts = _resolve_ssh(machine_id)
    remote_dest = dest or default_upload_dest(source, inst.ssh_command)
    recursive = source.is_dir()

    if dest is not None:
        if recursive:
            remote_prep = remote_dest.rstrip("/") or remote_dest
        else:
            remote_prep = PurePosixPath(remote_dest).parent.as_posix()
        prep_command = build_remote_shell_command(["mkdir", "-p", remote_prep])
        if subprocess.call([*ssh_parts, prep_command]) != 0:
            render.die(f"Failed to prepare remote destination {remote_prep}.")

    try:
        parts = build_scp_command(
            inst.ssh_command,
            source=str(source),
            dest=remote_dest,
            upload=True,
            recursive=recursive,
        )
    except SSHError:
        render.die(f"Cannot prepare upload command for instance {machine_id}.")

    if state.json_output:
        _transfer_json(
            parts,
            {
                "machine_id": machine_id,
                "direction": "upload",
                "source": str(source),
                "dest": remote_dest,
                "recursive": recursive,
            },
        )
        return

    render.info(f"Uploading to {machine_id}: {source} -> {remote_dest}")
    raise SystemExit(subprocess.call(parts))


@app.command("download", rich_help_panel=_ACCESS_PANEL)
def instance_download(
    machine_id: Annotated[int, typer.Argument(help="Instance ID.")],
    source: Annotated[str, typer.Argument(help="Remote file or directory to download.")],
    dest: Annotated[
        Path | None, typer.Argument(resolve_path=True, help="Local destination path. Defaults to ./<name>.")
    ] = None,
    recursive: Annotated[bool, typer.Option("--recursive", "-r", help="Download directories recursively.")] = False,
    json_output: cli_options.JsonOption = False,
) -> None:
    """Download a remote file or directory from a running instance."""
    cli_options.apply_command_options(json_output=json_output)
    inst, _ = _resolve_ssh(machine_id)

    try:
        local_dest = dest or Path(_default_download_dest(source))
    except ValueError as exc:
        render.die(str(exc))

    try:
        parts = build_scp_command(
            inst.ssh_command,
            source=source,
            dest=str(local_dest),
            upload=False,
            recursive=recursive,
        )
    except SSHError:
        render.die(f"Cannot prepare download command for instance {machine_id}.")

    if state.json_output:
        _transfer_json(
            parts,
            {
                "machine_id": machine_id,
                "direction": "download",
                "source": source,
                "dest": str(local_dest),
                "recursive": recursive,
            },
        )
        return

    render.info(f"Downloading from {machine_id}: {source} -> {local_dest}")
    raise SystemExit(subprocess.call(parts))
