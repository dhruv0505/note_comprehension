"""Managed run commands built on top of instance SSH primitives."""

from __future__ import annotations

import json
import re
import secrets
import shlex
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Annotated

import typer
from typer.core import TyperGroup

from jarvislabs.cli import options as cli_options, render, state
from jarvislabs.cli.app import app, get_client
from jarvislabs.cli.instance import default_upload_dest, remote_home, resolve_vm_template
from jarvislabs.constants import DEFAULT_NUM_GPUS, DEFAULT_STORAGE_GB, DEFAULT_TEMPLATE
from jarvislabs.exceptions import JarvislabsError, SSHError
from jarvislabs.ssh import build_rsync_upload_command, build_scp_command, harden_ssh_parts, split_ssh_command

if TYPE_CHECKING:
    from jarvislabs.models import Instance

_REMOTE_RUNS_SUFFIX = "jl-runs"
_LOCAL_RUNS_ROOT = Path.home() / ".jl" / "runs"
_DEFAULT_FOLLOW_TAIL_LINES = 20


class RunCommandGroup(TyperGroup):
    def resolve_command(self, ctx, args):  # type: ignore[override]
        default_command = "start"
        if not args or args[0] not in self.commands:
            cmd = self.get_command(ctx, default_command)
            return default_command, cmd, args
        return super().resolve_command(ctx, args)

    def get_help(self, ctx):  # type: ignore[override]
        help_text = super().get_help(ctx).rstrip()
        start_help = (
            "\n\n"
            "Start a run directly:\n"
            "  jl run TARGET --on <instance_id> [-- script args]\n"
            "  jl run TARGET --gpu <gpu> [-- script args]\n\n"
            "Common examples:\n"
            "  jl run train.py --gpu L4\n"
            "  jl run train.py --gpu L4 -- --epochs 5\n"
            "  jl run . --script train.py --on <instance_id>\n"
            "  jl run --on <instance_id> -- python -c \"print('hi')\"\n\n"
            "Key start options:\n"
            "  --on INTEGER           Run on an existing instance.\n"
            "  --gpu TEXT             Create a fresh instance with this GPU.\n"
            "  --vm                   Create a VM instead of a container.\n"
            "  --http-ports TEXT      Expose comma-separated HTTP ports on a fresh instance.\n"
            "  --script TEXT          Script path inside a directory target.\n"
            "  --requirements PATH    Override auto-detection with a custom requirements file.\n"
            "  --setup TEXT           Shell command to run before the main command.\n"
            "  --follow / --no-follow Stream logs after starting the run.\n"
            "\n"
            "Environment:\n"
            "  For directory targets, pyproject.toml and requirements.txt are\n"
            "  auto-detected and installed. Use --requirements to override.\n"
            "\n"
            "Lifecycle note:\n"
            "  Human workflow: use the default mode where the CLI stays attached to the\n"
            "  run and streams logs if you want it to pause or destroy the instance after\n"
            "  the run finishes.\n"
            "  Agent workflow: --json is for detached runs, so use --keep and have the\n"
            "  agent clean up the instance after the run.\n"
        )
        return help_text + start_help


run_app = typer.Typer(
    name="run",
    cls=RunCommandGroup,
    help="Start and inspect managed runs.",
    rich_markup_mode="rich",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
app.add_typer(run_app, rich_help_panel="Workloads")


@dataclass(frozen=True, slots=True)
class RunPaths:
    run_id: str
    remote_run_dir: str
    remote_log: str
    remote_pid: str
    remote_exit_code: str
    remote_meta: str
    remote_wrapper: str


@dataclass(frozen=True, slots=True)
class RunSpec:
    target_kind: str
    local_target: Path | None
    remote_target: str | None
    working_dir: str | None
    launch_command: str


@dataclass(frozen=True, slots=True)
class LocalRunRecord:
    run_id: str
    machine_id: int
    target_kind: str
    local_target: str | None
    remote_target: str | None
    working_dir: str | None
    remote_log: str
    remote_pid: str
    remote_exit_code: str
    launch_command: str
    started_at: str
    instance_origin: str = "existing"
    lifecycle_policy: str = "none"


@dataclass(frozen=True, slots=True)
class RunStatusSnapshot:
    run_id: str
    machine_id: int
    target_kind: str
    started_at: str
    state: str
    instance_status: str | None
    exit_code: int | None
    remote_log: str
    lifecycle_policy: str
    instance_cost: float | None = None


def _make_run_id() -> str:
    return f"r_{secrets.token_hex(4)}"


def _build_run_paths(run_id: str, ssh_command: str | None = None) -> RunPaths:
    remote_runs_root = PurePosixPath(remote_home(ssh_command)) / _REMOTE_RUNS_SUFFIX
    remote_run_dir = remote_runs_root / run_id
    return RunPaths(
        run_id=run_id,
        remote_run_dir=remote_run_dir.as_posix(),
        remote_log=(remote_run_dir / "output.log").as_posix(),
        remote_pid=(remote_run_dir / "pid").as_posix(),
        remote_exit_code=(remote_run_dir / "exit_code").as_posix(),
        remote_meta=(remote_run_dir / "meta.json").as_posix(),
        remote_wrapper=(remote_run_dir / "run.sh").as_posix(),
    )


def _local_run_file(run_id: str) -> Path:
    return _LOCAL_RUNS_ROOT / f"{run_id}.json"


def _save_local_run(record: LocalRunRecord) -> None:
    _LOCAL_RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    _local_run_file(record.run_id).write_text(json.dumps(asdict(record), indent=2) + "\n")


def _load_local_run(run_id: str) -> LocalRunRecord:
    try:
        payload = json.loads(_local_run_file(run_id).read_text())
    except FileNotFoundError:
        render.die(f"Run {run_id} was not found locally. Use jl run list to see saved runs.")
    except json.JSONDecodeError:
        render.die(f"Run {run_id} has a corrupted local record.")

    try:
        return LocalRunRecord(**payload)
    except TypeError:
        render.die(f"Run {run_id} has an incompatible local record.")


def _iter_local_runs() -> list[LocalRunRecord]:
    if not _LOCAL_RUNS_ROOT.exists():
        return []

    records: list[LocalRunRecord] = []
    for path in _LOCAL_RUNS_ROOT.glob("*.json"):
        try:
            payload = json.loads(path.read_text())
            records.append(LocalRunRecord(**payload))
        except (json.JSONDecodeError, TypeError):
            render.warning(f"Skipping invalid run record: {path.name}")

    records.sort(key=lambda record: record.started_at, reverse=True)
    return records


def _build_remote_command(script: str) -> str:
    return f"sh -lc {shlex.quote(script)}"


def _run_remote(ssh_parts: list[str], script: str) -> int:
    return subprocess.call([*ssh_parts, _build_remote_command(script)])


def _run_remote_capture(ssh_parts: list[str], script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [*ssh_parts, _build_remote_command(script)],
        capture_output=True,
        text=True,
        check=False,
    )


def _upload_to_remote(ssh_command: str, source: Path, dest: str, *, recursive: bool) -> int:
    parts = build_scp_command(
        ssh_command,
        source=str(source),
        dest=dest,
        upload=True,
        recursive=recursive,
    )
    return subprocess.call(parts)


def _sync_directory_to_remote(ssh_command: str, source: Path, dest: str) -> int:
    parts = build_rsync_upload_command(ssh_command, source=str(source), dest=dest)
    return subprocess.call(parts)


def _write_remote_text(ssh_parts: list[str], destination: str, content: str) -> int:
    remote_parent = PurePosixPath(destination).parent.as_posix()
    script = f"mkdir -p {shlex.quote(remote_parent)} && cat > {shlex.quote(destination)}"
    parts = [*ssh_parts, _build_remote_command(script)]
    completed = subprocess.run(parts, input=content, text=True, check=False)
    return completed.returncode


def _ensure_remote_dir(ssh_parts: list[str], destination: str) -> None:
    if _run_remote(ssh_parts, f"mkdir -p {shlex.quote(destination)}") != 0:
        render.die(f"Failed to prepare remote directory {destination}.")


def _upload_file_to_instance(inst, ssh_parts: list[str], source: Path, destination: str, *, label: str) -> None:
    render.info(f"Uploading {label} to {destination}")
    _ensure_remote_dir(ssh_parts, PurePosixPath(destination).parent.as_posix())
    if _upload_to_remote(inst.ssh_command, source, destination, recursive=False) != 0:
        render.die(f"Failed to upload {label} to instance {inst.machine_id}.")


def _get_instance(machine_id: int) -> Instance | None:
    client = get_client()
    try:
        return client.instances.get(machine_id)
    except JarvislabsError:
        return None


_cached_currency_sym: str | None = None


def _currency_symbol() -> str:
    global _cached_currency_sym
    if _cached_currency_sym is None:
        try:
            _cached_currency_sym = render.currency_symbol(get_client().account.currency())
        except Exception:
            return "$"
    return _cached_currency_sym


def _format_instance_cost(machine_id: int) -> str:
    """Best-effort fetch of instance cost for display. Returns empty string on failure."""
    try:
        inst = _get_instance(machine_id)
        if inst is not None:
            sym = _currency_symbol()
            return f" Instance cost: {sym}{inst.cost:.2f}"
    except Exception:
        pass
    return ""


def _ssh_parts_from_instance(inst: Instance) -> list[str] | None:
    if not inst.ssh_command:
        return None
    try:
        return harden_ssh_parts(split_ssh_command(inst.ssh_command))
    except SSHError:
        return None


def _fetch_exit_code_path(ssh_parts: list[str], remote_exit_code: str) -> int | None:
    completed = _run_remote_capture(ssh_parts, f"cat {shlex.quote(remote_exit_code)}")
    if completed.returncode != 0:
        return None
    try:
        return int(completed.stdout.strip())
    except ValueError:
        return None


def _snapshot(
    record: LocalRunRecord,
    *,
    state: str,
    instance_status: str | None = None,
    exit_code: int | None = None,
    instance_cost: float | None = None,
) -> RunStatusSnapshot:
    return RunStatusSnapshot(
        run_id=record.run_id,
        machine_id=record.machine_id,
        target_kind=record.target_kind,
        started_at=record.started_at,
        state=state,
        instance_status=instance_status,
        exit_code=exit_code,
        remote_log=record.remote_log,
        lifecycle_policy=record.lifecycle_policy,
        instance_cost=instance_cost,
    )


def _get_run_snapshot(record: LocalRunRecord) -> RunStatusSnapshot:
    inst = _get_instance(record.machine_id)
    if inst is None:
        return _snapshot(record, state="instance-missing")

    cost = inst.cost

    if inst.status == "Paused":
        return _snapshot(record, state="instance-paused", instance_status=inst.status, instance_cost=cost)

    if inst.status != "Running":
        return _snapshot(
            record, state=f"instance-{inst.status.lower()}", instance_status=inst.status, instance_cost=cost
        )

    ssh_parts = _ssh_parts_from_instance(inst)
    if ssh_parts is None:
        return _snapshot(record, state="unknown", instance_status=inst.status, instance_cost=cost)

    exit_code = _fetch_exit_code_path(ssh_parts, record.remote_exit_code)
    if exit_code is None:
        state_name = "running"
    elif exit_code == 0:
        state_name = "succeeded"
    else:
        state_name = "failed"

    return _snapshot(record, state=state_name, instance_status=inst.status, exit_code=exit_code, instance_cost=cost)


def _resolve_run_ssh(run_id: str) -> tuple[LocalRunRecord, list[str]]:
    record = _load_local_run(run_id)
    inst = _get_instance(record.machine_id)
    if inst is None:
        render.die(f"Run {run_id} belongs to instance {record.machine_id}, but that instance no longer exists.")

    if inst.status == "Paused":
        render.die(
            f"Run {run_id} belongs to paused instance {record.machine_id}. Resume it first: jl resume {record.machine_id}"
        )

    if inst.status != "Running":
        render.die(f"Run {run_id} is on instance {record.machine_id}, which is currently {inst.status}.")

    ssh_parts = _ssh_parts_from_instance(inst)
    if ssh_parts is None:
        render.die(f"Run {run_id} is on instance {record.machine_id}, but SSH is not available.")
    return record, ssh_parts


def _with_setup(command: str, setup_command: str | None) -> str:
    if not setup_command:
        return command
    return f"{setup_command} && {command}"


def _resolve_local_input(path: str | None, *, label: str) -> Path | None:
    if path is None:
        return None
    resolved = Path(path).expanduser()
    if not resolved.exists():
        render.die(f"{label} does not exist: {path}")
    if not resolved.is_file():
        render.die(f"{label} must be a file: {path}")
    return resolved


def _entrypoint_command(path: str, extra_args: list[str]) -> str:
    suffix = Path(path).suffix
    if suffix == ".py":
        parts = ["python3", path, *extra_args]
    elif suffix == ".sh":
        parts = ["bash", path, *extra_args]
    else:
        render.die(
            "Only Python or bash entrypoints are supported directly. "
            "Use --script <train.py|train.sh> or pass a full command after --."
        )
    return shlex.join(parts)


def _build_run_spec(
    target: str | None,
    extra_args: list[str],
    *,
    script_path: str | None = None,
    ssh_command: str | None = None,
) -> RunSpec:
    if target is None:
        if script_path is not None:
            render.die("--script can only be used with a directory target.")
        if not extra_args:
            render.die(
                "No target or remote command provided. Use jl run <file|folder> --on <id> or jl run --on <id> -- <command>."
            )
        return RunSpec(
            target_kind="command",
            local_target=None,
            remote_target=None,
            working_dir=None,
            launch_command=shlex.join(extra_args),
        )

    local_target = Path(target).expanduser()
    if not local_target.exists():
        looks_like_path = "/" in target or target.startswith(".") or target.endswith((".py", ".sh"))
        if looks_like_path:
            render.die(f"Target does not exist: {target}")
        if script_path is not None:
            render.die("--script can only be used with a directory target.")
        return RunSpec(
            target_kind="command",
            local_target=None,
            remote_target=None,
            working_dir=None,
            launch_command=shlex.join([target, *extra_args]),
        )

    if local_target.is_dir():
        remote_target = default_upload_dest(local_target, ssh_command)
        if script_path:
            launch_command = _entrypoint_command(script_path, extra_args)
        elif extra_args:
            launch_command = shlex.join(extra_args)
        else:
            render.die(
                "Directory targets require --script <path> or a command after --. "
                "Example: jl run . --script train.py --gpu L4"
            )
        return RunSpec(
            target_kind="directory",
            local_target=local_target,
            remote_target=remote_target,
            working_dir=remote_target,
            launch_command=launch_command,
        )

    if script_path is not None:
        render.die("--script can only be used with a directory target.")

    if local_target.suffix not in {".py", ".sh"}:
        render.die(
            "Only Python or bash file targets are supported directly right now. "
            "Use a directory target or the instance upload/exec primitives for other files."
        )

    remote_root = PurePosixPath(remote_home(ssh_command))
    remote_target = (remote_root / local_target.name).as_posix()
    return RunSpec(
        target_kind="file",
        local_target=local_target,
        remote_target=remote_target,
        working_dir=remote_root.as_posix(),
        launch_command=_entrypoint_command(local_target.name, extra_args),
    )


def _parse_run_inputs(raw_args: list[str]) -> tuple[str | None, list[str]]:
    if not raw_args:
        return None, []

    candidate = raw_args[0]
    looks_like_path = "/" in candidate or candidate.startswith(".") or candidate.endswith((".py", ".sh"))
    if looks_like_path or Path(candidate).expanduser().exists():
        return candidate, raw_args[1:]
    return None, raw_args


def _write_remote_metadata(ssh_parts: list[str], inst, paths: RunPaths, spec: RunSpec) -> None:
    payload = {
        "run_id": paths.run_id,
        "machine_id": inst.machine_id,
        "target_kind": spec.target_kind,
        "remote_target": spec.remote_target,
        "working_dir": spec.working_dir,
        "launch_command": spec.launch_command,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    status = _write_remote_text(ssh_parts, paths.remote_meta, json.dumps(payload, indent=2) + "\n")
    if status != 0:
        render.die(f"Failed to upload metadata for run {paths.run_id}.")


def _write_remote_wrapper(ssh_parts: list[str], paths: RunPaths, spec: RunSpec) -> None:
    wrapper = f"""#!/bin/sh
set -u

LOG_FILE={shlex.quote(paths.remote_log)}
PID_FILE={shlex.quote(paths.remote_pid)}
EXIT_FILE={shlex.quote(paths.remote_exit_code)}
COMMAND={shlex.quote(spec.launch_command)}
WORKDIR={shlex.quote(spec.working_dir or "~")}

rm -f "$EXIT_FILE"
touch "$LOG_FILE"
export PYTHONUNBUFFERED=1

child_pid=""
cleanup() {{
  if [ -n "$child_pid" ]; then
    pgid=$(ps -o pgid= -p "$child_pid" 2>/dev/null | tr -d ' ')
    if [ -n "$pgid" ] && [ "$pgid" != "0" ]; then
      kill -- -"$pgid" 2>/dev/null || true
    else
      kill "$child_pid" 2>/dev/null || true
    fi
  fi
}}

trap cleanup INT TERM

sh -lc "cd $WORKDIR && $COMMAND" >>"$LOG_FILE" 2>&1 &
child_pid=$!
printf '%s\\n' "$child_pid" >"$PID_FILE"
wait "$child_pid"
status=$?
printf '%s\\n' "$status" >"$EXIT_FILE"
exit 0
"""
    status = _write_remote_text(ssh_parts, paths.remote_wrapper, wrapper)
    if status != 0:
        render.die(f"Failed to upload wrapper script for run {paths.run_id}.")


def _ensure_remote_rsync(machine_id: int, ssh_parts: list[str]) -> None:
    if shutil.which("rsync") is None:
        render.die("rsync is required locally for directory runs. Install rsync and try again.")

    script = (
        "command -v rsync >/dev/null 2>&1 || "
        "(apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y rsync)"
    )
    with render.spinner(f"Ensuring rsync is installed on instance {machine_id}..."):
        completed = _run_remote_capture(ssh_parts, script)

    if completed.returncode != 0:
        render.die(f"Could not install rsync on instance {machine_id}.")


def _prepare_remote_target(inst, ssh_parts: list[str], spec: RunSpec) -> None:
    if spec.local_target is None or spec.remote_target is None:
        return

    if spec.target_kind == "directory":
        _ensure_remote_rsync(inst.machine_id, ssh_parts)
        _ensure_remote_dir(ssh_parts, spec.remote_target)
        render.info(f"Syncing project to {spec.remote_target}")
        if _sync_directory_to_remote(inst.ssh_command, spec.local_target, spec.remote_target) != 0:
            render.die(f"Failed to sync {spec.local_target} to instance {inst.machine_id}.")
        return

    _upload_file_to_instance(inst, ssh_parts, spec.local_target, spec.remote_target, label="file")


def _detect_requirements(spec: RunSpec) -> str | None:
    """Auto-detect a dependency file in the local directory target.

    Returns the filename to install from on the remote (the file is synced via rsync).
    Only applies to directory targets — file and command modes return None.
    """
    if spec.target_kind != "directory" or spec.local_target is None:
        return None

    pyproject = spec.local_target / "pyproject.toml"
    if pyproject.exists():
        try:
            content = pyproject.read_text()
            if re.search(r"^\s*\[project\]", content, re.MULTILINE):
                return "pyproject.toml"
        except OSError:
            pass

    requirements = spec.local_target / "requirements.txt"
    if requirements.exists():
        return "requirements.txt"

    return None


def _prepare_support_files(
    inst,
    ssh_parts: list[str],
    spec: RunSpec,
    *,
    requirements_path: Path | None,
) -> str | None:
    if requirements_path is None:
        return None

    if spec.working_dir is None:
        render.die("--requirements requires a file or directory target.")

    requirements_name = requirements_path.name
    _upload_file_to_instance(
        inst,
        ssh_parts,
        requirements_path,
        (PurePosixPath(spec.working_dir) / requirements_path.name).as_posix(),
        label="requirements file",
    )
    return requirements_name


def _compose_launch_command(
    spec: RunSpec,
    *,
    setup_command: str | None,
    requirements_name: str | None,
) -> str:
    if spec.target_kind == "command":
        if requirements_name:
            render.die("--requirements requires a file or directory target.")
        # Detect existing venv from previous file/dir runs — prepend to PATH if found
        venv_detect = 'if [ -x "$HOME/.venv/bin/python" ]; then export PATH="$HOME/.venv/bin:$PATH"; fi'
        return _with_setup(f"{venv_detect} && {spec.launch_command}", setup_command)

    commands = [
        "command -v uv >/dev/null 2>&1 || { curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1; }",
        'export PATH="$HOME/.local/bin:$PATH"',
    ]

    if spec.target_kind == "file":
        # File targets: shared instance-level venv at $HOME/.venv
        commands.append('(test -d "$HOME/.venv" || uv venv --system-site-packages --seed "$HOME/.venv")')
        commands.append('. "$HOME/.venv/bin/activate"')
    else:
        # Directory targets: per-project venv inside the project directory
        commands.append("(test -d .venv || uv venv --system-site-packages --seed .venv)")
        commands.append(". .venv/bin/activate")

    if requirements_name:
        quoted = shlex.quote(requirements_name)
        commands.append(f"echo '[jl] Installing from' {quoted} && uv pip install -r {quoted}")
    else:
        commands.append('echo "[jl] No dependency file detected, using template packages"')
    if setup_command:
        commands.append(setup_command)
    commands.append(spec.launch_command)
    return " && ".join(commands)


def _start_remote_run(ssh_parts: list[str], paths: RunPaths) -> int:
    script = (
        f"mkdir -p {shlex.quote(paths.remote_run_dir)} && "
        f"chmod +x {shlex.quote(paths.remote_wrapper)} && "
        f"nohup sh {shlex.quote(paths.remote_wrapper)} </dev/null >/dev/null 2>&1 & "
        "echo $!"
    )
    completed = _run_remote_capture(ssh_parts, script)
    if completed.returncode != 0:
        render.die(f"Failed to start run {paths.run_id}.")

    output = completed.stdout.strip()
    if not output.isdigit():
        render.die(f"Could not determine launcher pid for run {paths.run_id}.")
    return int(output)


def _follow_run_logs(ssh_parts: list[str], paths: RunPaths) -> bool:
    script = (
        f"touch {shlex.quote(paths.remote_log)}; "
        f"tail -n {_DEFAULT_FOLLOW_TAIL_LINES} -F {shlex.quote(paths.remote_log)} & "
        "tail_pid=$!; "
        f"while [ ! -f {shlex.quote(paths.remote_exit_code)} ]; do sleep 1; done; "
        "sleep 1; "
        "kill $tail_pid >/dev/null 2>&1 || true; "
        "wait $tail_pid 2>/dev/null || true"
    )
    parts = [*ssh_parts, _build_remote_command(script)]
    try:
        return subprocess.call(parts) == 0
    except KeyboardInterrupt:
        return False


def _tail_remote_log(
    ssh_parts: list[str],
    remote_log: str,
    *,
    follow: bool,
    tail: int | None,
) -> subprocess.CompletedProcess[str] | int:
    if follow:
        if tail is None:
            script = (
                f"touch {shlex.quote(remote_log)} && tail -n {_DEFAULT_FOLLOW_TAIL_LINES} -F {shlex.quote(remote_log)}"
            )
        else:
            script = f"touch {shlex.quote(remote_log)} && tail -n {tail} -F {shlex.quote(remote_log)}"
        return _run_remote(ssh_parts, script)

    if not state.json_output:
        if tail is not None:
            return _run_remote(ssh_parts, f"tail -n {tail} {shlex.quote(remote_log)}")
        return _run_remote(ssh_parts, f"cat {shlex.quote(remote_log)}")

    if tail is not None:
        return _run_remote_capture(ssh_parts, f"tail -n {tail} {shlex.quote(remote_log)}")
    return _run_remote_capture(ssh_parts, f"cat {shlex.quote(remote_log)}")


def _print_run_followups(run_id: str, *, include_list: bool = False) -> None:
    if state.json_output:
        return
    render.info(f"Run ID: {run_id}")
    render.console.print(f"  [dim]Logs[/dim]    [bold cyan]jl run logs {run_id} --follow[/bold cyan]")
    render.console.print(f"  [dim]Status[/dim]  [bold magenta]jl run status {run_id}[/bold magenta]")
    render.console.print(f"  [dim]Stop[/dim]    [bold red]jl run stop {run_id}[/bold red]")
    if include_list:
        render.console.print("  [dim]All runs[/dim] [bold green]jl run list[/bold green]")


def _stop_remote_run(ssh_parts: list[str], pid_file: str, exit_code_file: str) -> str:
    script = (
        f"if [ -f {shlex.quote(exit_code_file)} ]; then echo finished; exit 0; fi; "
        f"if [ ! -f {shlex.quote(pid_file)} ]; then echo missing-pid; exit 3; fi; "
        f"pid=$(cat {shlex.quote(pid_file)}); "
        'if [ -z "$pid" ]; then echo missing-pid; exit 3; fi; '
        'if kill -0 "$pid" 2>/dev/null; then '
        'pgid=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d " "); '
        'if [ -n "$pgid" ] && [ "$pgid" != "0" ]; then kill -- -"$pgid" 2>/dev/null; '
        'else kill "$pid" 2>/dev/null; fi; '
        "sleep 1; "
        'if kill -0 "$pid" 2>/dev/null; then kill -9 -- -"$pgid" 2>/dev/null || kill -9 "$pid" 2>/dev/null; fi; '
        "echo stopped; exit 0; fi; "
        f"if [ -f {shlex.quote(exit_code_file)} ]; then echo finished; exit 0; fi; "
        "echo not-running; exit 4"
    )
    completed = _run_remote_capture(ssh_parts, script)
    output = completed.stdout.strip()
    if output in {"stopped", "not-running", "missing-pid", "finished"}:
        return output
    return "error"


def _print_detach_instructions(machine_id: int, paths: RunPaths) -> None:
    render.warning(f"Detached from logs. Run {paths.run_id} is still running on instance {machine_id}.")
    _print_run_followups(paths.run_id, include_list=True)


def _wait_for_ssh_ready(machine_id: int, *, timeout_s: int = 90, poll_s: int = 3) -> tuple[Instance, list[str]]:
    deadline = time.monotonic() + timeout_s
    last_status = "unknown"

    while time.monotonic() < deadline:
        inst = _get_instance(machine_id)
        if inst is None:
            render.die(f"Instance {machine_id} no longer exists.")

        last_status = inst.status
        if inst.status == "Paused":
            render.die(f"Instance {machine_id} is paused. Resume it first: jl resume {machine_id}")
        if inst.status != "Running":
            render.die(f"Instance {machine_id} is not available for runs yet (status: {inst.status}).")

        ssh_parts = _ssh_parts_from_instance(inst)
        if ssh_parts is not None and _run_remote_capture(ssh_parts, "true").returncode == 0:
            return inst, ssh_parts

        time.sleep(poll_s)

    render.die(
        f"Instance {machine_id} is {last_status}, but SSH is not ready yet. "
        f"Wait a bit and retry, or check it with: jl exec {machine_id} -- echo ok"
    )


def _start_managed_run(
    target: str | None,
    machine_id: int,
    extra_args: list[str],
    *,
    follow: bool,
    instance_origin: str,
    lifecycle_policy: str,
    script_path: str | None = None,
    setup_command: str | None = None,
    requirements_path: Path | None = None,
) -> tuple[str, int | None]:
    inst, ssh_parts = _wait_for_ssh_ready(machine_id)
    spec = _build_run_spec(target, extra_args, script_path=script_path, ssh_command=inst.ssh_command)
    paths = _build_run_paths(_make_run_id(), ssh_command=inst.ssh_command)

    render.info(f"Preparing run {paths.run_id} on instance {machine_id}")
    if _run_remote(ssh_parts, f"mkdir -p {shlex.quote(paths.remote_run_dir)}") != 0:
        render.die(f"Failed to create remote run directory {paths.remote_run_dir}.")

    _prepare_remote_target(inst, ssh_parts, spec)

    # Determine which requirements to install:
    # 1. Explicit --requirements overrides everything
    # 2. Otherwise auto-detect from local directory (pyproject.toml > requirements.txt)
    if requirements_path:
        requirements_name = _prepare_support_files(
            inst,
            ssh_parts,
            spec,
            requirements_path=requirements_path,
        )
    else:
        requirements_name = _detect_requirements(spec)

    spec = RunSpec(
        target_kind=spec.target_kind,
        local_target=spec.local_target,
        remote_target=spec.remote_target,
        working_dir=spec.working_dir,
        launch_command=_compose_launch_command(
            spec,
            setup_command=setup_command,
            requirements_name=requirements_name,
        ),
    )
    _write_remote_wrapper(ssh_parts, paths, spec)
    _write_remote_metadata(ssh_parts, inst, paths, spec)

    launcher_pid = _start_remote_run(ssh_parts, paths)
    record = LocalRunRecord(
        run_id=paths.run_id,
        machine_id=machine_id,
        target_kind=spec.target_kind,
        local_target=str(spec.local_target) if spec.local_target else None,
        remote_target=spec.remote_target,
        working_dir=spec.working_dir,
        remote_log=paths.remote_log,
        remote_pid=paths.remote_pid,
        remote_exit_code=paths.remote_exit_code,
        launch_command=spec.launch_command,
        started_at=datetime.now(timezone.utc).isoformat(),
        instance_origin=instance_origin,
        lifecycle_policy=lifecycle_policy,
    )
    _save_local_run(record)

    summary = {
        "run_id": paths.run_id,
        "machine_id": machine_id,
        "launcher_pid": launcher_pid,
        "remote_log": paths.remote_log,
        "remote_exit_code": paths.remote_exit_code,
        "target_kind": spec.target_kind,
        "remote_target": spec.remote_target,
        "command": spec.launch_command,
        "instance_origin": instance_origin,
        "lifecycle_policy": lifecycle_policy,
    }

    if state.json_output:
        render.print_json(summary)
        return paths.run_id, None

    render.success(f"Started run {paths.run_id} on instance {machine_id} (launcher pid {launcher_pid}).")
    _print_run_followups(paths.run_id)

    if not follow:
        render.info("Run started in the background. It will keep going if you disconnect.")
        return paths.run_id, None

    render.info("Streaming logs. Press Ctrl+C to detach; the run will keep going.")

    followed_to_completion = _follow_run_logs(ssh_parts, paths)
    if not followed_to_completion:
        _print_detach_instructions(machine_id, paths)
        return paths.run_id, None

    exit_code = _fetch_exit_code_path(ssh_parts, paths.remote_exit_code)
    if exit_code is None:
        render.warning(f"Run {paths.run_id} finished log streaming, but exit status could not be read.")
        _print_run_followups(paths.run_id)
        return paths.run_id, None

    cost_str = _format_instance_cost(machine_id)
    if exit_code == 0:
        render.success(f"Run {paths.run_id} completed successfully.{cost_str}")
    else:
        render.warning(f"Run {paths.run_id} finished with exit code {exit_code}.{cost_str}")

    _print_run_followups(paths.run_id)
    return paths.run_id, exit_code


def _pick_lifecycle_policy(*, pause: bool, destroy: bool, keep: bool) -> str | None:
    selected = [name for name, enabled in (("pause", pause), ("destroy", destroy), ("keep", keep)) if enabled]
    if len(selected) > 1:
        render.die("Choose only one lifecycle option: --pause, --destroy, or --keep.")
    return selected[0] if selected else None


def _apply_lifecycle(machine_id: int, policy: str) -> None:
    if policy == "keep":
        render.info(f"Leaving instance {machine_id} running.")
        return

    client = get_client()
    action = "Pausing" if policy == "pause" else "Destroying"
    with render.spinner(f"{action} instance {machine_id}..."):
        if policy == "pause":
            client.instances.pause(machine_id)
        else:
            client.instances.destroy(machine_id)

    if policy == "pause":
        render.success(f"Instance {machine_id} paused after the run.")
    else:
        render.success(f"Instance {machine_id} destroyed after the run.")


@run_app.command("start", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def run_start(
    ctx: typer.Context,
    on: Annotated[int | None, typer.Option("--on", help="Run on an existing instance.")] = None,
    gpu: Annotated[str | None, typer.Option("--gpu", "-g", help="Create a fresh instance with this GPU.")] = None,
    region: Annotated[
        str | None, typer.Option("--region", help="Optional region pin for fresh instances (IN1, IN2, EU1).")
    ] = None,
    script: Annotated[
        str | None,
        typer.Option(
            "--script",
            help="Python script path inside a directory target. Example: jl run . --script train.py --gpu L4",
        ),
    ] = None,
    vm: Annotated[bool, typer.Option("--vm", help="Create a VM instance instead of a container.")] = False,
    template: Annotated[
        str, typer.Option("--template", "-t", help="Framework template for fresh instances.")
    ] = DEFAULT_TEMPLATE,
    storage: Annotated[
        int, typer.Option("--storage", "-s", help="Storage in GB for fresh instances.")
    ] = DEFAULT_STORAGE_GB,
    name: Annotated[str, typer.Option("--name", "-n", help="Instance name for fresh runs.")] = "jl-run",
    num_gpus: Annotated[int, typer.Option("--num-gpus", help="Number of GPUs for fresh runs.")] = DEFAULT_NUM_GPUS,
    spot: Annotated[bool, typer.Option("--spot", help="Create a spot GPU container for this run.")] = False,
    http_ports: Annotated[
        str, typer.Option("--http-ports", help="Comma-separated HTTP ports to expose on fresh instances.")
    ] = "",
    setup: Annotated[str | None, typer.Option("--setup", help="Shell command to run before the main command.")] = None,
    requirements: Annotated[
        str | None,
        typer.Option(
            "--requirements", help="Override auto-detection: upload and install this requirements file instead."
        ),
    ] = None,
    pause: Annotated[bool, typer.Option("--pause", help="Pause a fresh instance after the run completes.")] = False,
    destroy: Annotated[
        bool, typer.Option("--destroy", help="Destroy a fresh instance after the run completes.")
    ] = False,
    keep: Annotated[
        bool, typer.Option("--keep", help="Leave a fresh instance running after the run completes.")
    ] = False,
    follow: Annotated[bool, typer.Option("--follow/--no-follow", help="Stream logs after starting the run.")] = True,
    yes: cli_options.YesOption = False,
    json_output: cli_options.JsonOption = False,
) -> None:
    """Start a managed run."""
    cli_options.apply_command_options(json_output=json_output, yes=yes)
    target, extra_args = _parse_run_inputs(list(ctx.args))
    requirements_path = _resolve_local_input(requirements, label="Requirements file")

    template = resolve_vm_template(vm, template, http_ports)
    if spot and vm:
        render.die("--spot is only supported for GPU container runs.")

    lifecycle = _pick_lifecycle_policy(pause=pause, destroy=destroy, keep=keep)

    if on is not None and gpu is not None:
        render.die("Use either --on <instance_id> or --gpu <type>, not both.")

    if on is not None:
        if spot:
            render.die("--spot is only supported when creating a fresh instance.")
        if vm:
            render.die("--vm is only supported with --gpu for fresh instances.")
        if region is not None:
            render.die("--region is only supported with --gpu for fresh instances.")
        if lifecycle is not None:
            render.die("Lifecycle flags are only for fresh instances. Do not use --pause/--destroy/--keep with --on.")
        _, exit_code = _start_managed_run(
            target,
            on,
            extra_args,
            follow=follow,
            instance_origin="existing",
            lifecycle_policy="none",
            script_path=script,
            setup_command=setup,
            requirements_path=requirements_path,
        )
        if exit_code not in (None, 0):
            raise SystemExit(exit_code)
        return

    if gpu is None:
        render.die("Use --on <instance_id> to run on an existing instance, or --gpu <type> to create a fresh one.")

    if lifecycle is None:
        if not follow:
            render.die("--no-follow requires an explicit lifecycle flag. Use --keep, --pause, or --destroy.")
        lifecycle = "pause"
        render.info("No lifecycle flag provided for the fresh run. Defaulting to --pause.")

    if state.json_output and lifecycle in {"pause", "destroy"}:
        render.die(
            f"--{lifecycle} is not supported with --json for fresh runs.\n\n"
            "--json is mainly meant for agent workflows. It prints the run details and returns right away, "
            f"so the run becomes detached from this CLI session. Because the CLI is no longer attached, it cannot "
            f"{lifecycle} the instance when the run finishes.\n\n"
            "What to do instead:\n"
            f"  Agent workflow: use --keep --json, then have the agent watch the run and call jl {lifecycle} <machine_id> when it is done.\n"
            f"  Human workflow: drop --json and use the default mode where the CLI stays attached to the run if you want it to apply --{lifecycle} after the run finishes."
        )

    if not follow and lifecycle != "keep":
        render.die("--no-follow is only supported with --keep for fresh runs right now.")

    detail_parts = [f"template={template}", f"storage={storage}GB", f"name={name!r}"]
    if spot:
        detail_parts.append("spot=true")
    if isinstance(region, str) and region:
        detail_parts.append(f"region={render.region_input_label(region)}")
    if http_ports:
        detail_parts.append(f"http_ports={http_ports!r}")
    details = f"Create {num_gpus}x {gpu} instance for jl run ({', '.join(detail_parts)})?"
    if not render.confirm(details, skip=state.yes):
        raise typer.Exit()

    client = get_client()
    with render.spinner("Creating instance for jl run..."):
        inst = client.instances.create(
            gpu_type=gpu,
            num_gpus=num_gpus,
            template=template,
            storage=storage,
            name=name,
            region=region,
            http_ports=http_ports,
            is_spot=spot,
        )

    render.success(f"Fresh instance {inst.machine_id} is ready.")

    try:
        run_id, exit_code = _start_managed_run(
            target,
            inst.machine_id,
            extra_args,
            follow=follow,
            instance_origin="fresh",
            lifecycle_policy=lifecycle,
            script_path=script,
            setup_command=setup,
            requirements_path=requirements_path,
        )
    except SystemExit:
        render.warning(
            f"Run setup failed after creating instance {inst.machine_id}. Manage it manually with jl ssh/pause/destroy."
        )
        raise

    if exit_code is None:
        render.warning(
            f"Run {run_id} is detached. Automatic {lifecycle} will not happen unless the CLI is still connected when the run ends."
        )
        render.info(f"Instance {inst.machine_id} is still running.")
        return

    _apply_lifecycle(inst.machine_id, lifecycle)
    if exit_code != 0:
        raise SystemExit(exit_code)


@run_app.command("list")
def run_list(
    refresh: Annotated[bool, typer.Option("--refresh", help="Refresh live status for each run. Can be slow.")] = False,
    machine: Annotated[int | None, typer.Option("--machine", "-m", help="Filter by instance ID.")] = None,
    limit: Annotated[int | None, typer.Option("--limit", "-l", help="Show only the N most recent runs.")] = None,
    status_filter: Annotated[
        str | None,
        typer.Option("--status", "-s", help="Filter by state (running, succeeded, failed). Implies --refresh."),
    ] = None,
    json_output: cli_options.JsonOption = False,
) -> None:
    """List locally tracked managed runs."""
    cli_options.apply_command_options(json_output=json_output)
    if status_filter is not None:
        refresh = True

    records = _iter_local_runs()
    if machine is not None:
        records = [r for r in records if r.machine_id == machine]
    if status_filter is not None:
        records = [r for r in records if _get_run_snapshot(r).state == status_filter]
    if limit is not None:
        records = records[:limit]

    if state.json_output:
        payload = []
        for record in records:
            item = asdict(record)
            item["state"] = _get_run_snapshot(record).state if refresh else "saved"
            payload.append(item)
        render.print_json(payload)
        return

    if not records:
        render.info("No saved runs found.")
        return

    if not refresh:
        render.info("Showing saved runs from this machine. Use --refresh for live status checks.")

    render.runs_table(
        [
            (
                record.run_id,
                str(record.machine_id),
                record.target_kind,
                _get_run_snapshot(record).state if refresh else "saved",
                record.lifecycle_policy,
                record.started_at[:19].replace("T", " "),
            )
            for record in records
        ]
    )


@run_app.command("status")
def run_status(
    run_id: Annotated[str, typer.Argument(help="Run ID.")],
    json_output: cli_options.JsonOption = False,
) -> None:
    """Show the current status of a managed run."""
    cli_options.apply_command_options(json_output=json_output)
    record = _load_local_run(run_id)
    snapshot = _get_run_snapshot(record)

    if state.json_output:
        render.print_json(asdict(snapshot) | {"launch_command": record.launch_command})
        return

    render.info(f"Run ID: {snapshot.run_id}")
    render.info(f"Machine: {snapshot.machine_id}")
    render.info(f"State: {snapshot.state}")
    if snapshot.instance_cost is not None:
        cost_label = "Storage cost" if snapshot.instance_status == "Paused" else "Instance cost"
        sym = _currency_symbol()
        render.info(f"{cost_label}: {sym}{snapshot.instance_cost:.2f}")
    render.info(f"Lifecycle: {snapshot.lifecycle_policy}")
    render.info(f"Started: {snapshot.started_at}")
    render.info(f"Command: {record.launch_command}")
    render.info(f"Log file: {snapshot.remote_log}")
    if snapshot.exit_code is not None:
        render.info(f"Exit code: {snapshot.exit_code}")
    elif snapshot.state == "running":
        render.info("This run is still active on the remote machine.")
    _print_run_followups(snapshot.run_id, include_list=True)


@run_app.command("logs")
def run_logs(
    run_id: Annotated[str, typer.Argument(help="Run ID.")],
    follow: Annotated[bool, typer.Option("--follow", "-f", help="Follow logs in real time.")] = False,
    tail: Annotated[int | None, typer.Option("--tail", "-n", min=1, help="Show only the last N lines.")] = None,
    json_output: cli_options.JsonOption = False,
) -> None:
    """Show logs for a managed run."""
    cli_options.apply_command_options(json_output=json_output)
    if state.json_output and follow:
        render.die("--json is not supported with --follow for jl run logs.")

    record, ssh_parts = _resolve_run_ssh(run_id)
    if state.json_output:
        completed = _tail_remote_log(ssh_parts, record.remote_log, follow=False, tail=tail)
        run_exit_code = _fetch_exit_code_path(ssh_parts, record.remote_exit_code)
        render.print_json(
            {
                "run_id": record.run_id,
                "machine_id": record.machine_id,
                "remote_log": record.remote_log,
                "content": completed.stdout,
                "run_exit_code": run_exit_code,
            }
        )
        if completed.returncode != 0:
            raise SystemExit(completed.returncode)
        return

    if follow:
        render.info(
            f"Following logs for run {record.run_id} on instance {record.machine_id}. Press Ctrl+C to stop streaming."
        )
        if tail is None:
            render.info(
                f"Showing the latest {_DEFAULT_FOLLOW_TAIL_LINES} log lines first. Use --tail N to change this."
            )
        _print_run_followups(record.run_id, include_list=True)

        try:
            exit_code = _tail_remote_log(ssh_parts, record.remote_log, follow=True, tail=tail)
        except KeyboardInterrupt:
            render.info(f"Stopped log streaming for run {record.run_id}.")
            _print_run_followups(record.run_id, include_list=True)
            return
        raise SystemExit(exit_code)

    # Non-follow mode: show header with run state, then logs, then footer
    run_exit_code = _fetch_exit_code_path(ssh_parts, record.remote_exit_code)
    if run_exit_code is None:
        run_state = "running"
        state_label = "running"
    elif run_exit_code == 0:
        run_state = "succeeded"
        state_label = "succeeded (exit 0)"
    else:
        run_state = "failed"
        state_label = f"failed (exit {run_exit_code})"

    render.console.print(f"[dim]--- run {record.run_id} | machine {record.machine_id} | {state_label} ---[/dim]")
    render.console.print()
    log_rc = _tail_remote_log(ssh_parts, record.remote_log, follow=False, tail=tail)
    render.console.print()
    if run_state == "running":
        render.console.print(f"[dim]--- still running | log: {record.remote_log} ---[/dim]")
    else:
        render.console.print(f"[dim]--- {run_state} | exit code: {run_exit_code} | log: {record.remote_log} ---[/dim]")
    raise SystemExit(log_rc)


@run_app.command("stop")
def run_stop(
    run_id: Annotated[str, typer.Argument(help="Run ID.")],
    json_output: cli_options.JsonOption = False,
) -> None:
    """Stop a managed run by sending TERM to its tracked process."""
    cli_options.apply_command_options(json_output=json_output)
    record = _load_local_run(run_id)
    snapshot = _get_run_snapshot(record)

    if snapshot.state in {"succeeded", "failed"}:
        message = f"Run {run_id} already finished with state {snapshot.state}."
        if state.json_output:
            render.print_json(
                {
                    "run_id": run_id,
                    "machine_id": record.machine_id,
                    "state": snapshot.state,
                    "stopped": False,
                }
            )
            return
        render.info(message)
        _print_run_followups(run_id)
        return

    if snapshot.state in {"instance-paused", "instance-missing"}:
        message = f"Run {run_id} is not on a running instance ({snapshot.state}). Nothing to stop."
        if state.json_output:
            render.print_json(
                {
                    "run_id": run_id,
                    "machine_id": record.machine_id,
                    "state": snapshot.state,
                    "stopped": False,
                }
            )
            return
        render.warning(message)
        _print_run_followups(run_id)
        return

    resolved_record, ssh_parts = _resolve_run_ssh(run_id)
    stop_status = _stop_remote_run(ssh_parts, resolved_record.remote_pid, resolved_record.remote_exit_code)

    if state.json_output:
        if stop_status == "finished":
            refreshed = _get_run_snapshot(resolved_record)
            render.print_json(
                {
                    "run_id": resolved_record.run_id,
                    "machine_id": resolved_record.machine_id,
                    "state": refreshed.state,
                    "exit_code": refreshed.exit_code,
                    "stop_status": stop_status,
                    "stopped": False,
                }
            )
            return
        render.print_json(
            {
                "run_id": resolved_record.run_id,
                "machine_id": resolved_record.machine_id,
                "state": snapshot.state,
                "stop_status": stop_status,
                "stopped": stop_status == "stopped",
            }
        )
        if stop_status == "error":
            raise SystemExit(1)
        return

    if stop_status == "stopped":
        render.success(f"Stop signal sent to run {resolved_record.run_id} on instance {resolved_record.machine_id}.")
        render.info("The process may take a moment to exit cleanly.")
    elif stop_status == "finished":
        refreshed = _get_run_snapshot(resolved_record)
        render.info(f"Run {resolved_record.run_id} already finished with state {refreshed.state}.")
    elif stop_status == "not-running":
        render.warning(f"Run {resolved_record.run_id} no longer has a live process to stop.")
    elif stop_status == "missing-pid":
        render.warning(f"Run {resolved_record.run_id} has no recorded PID on the remote machine.")
    else:
        render.die(f"Could not stop run {resolved_record.run_id}.")

    _print_run_followups(resolved_record.run_id)
