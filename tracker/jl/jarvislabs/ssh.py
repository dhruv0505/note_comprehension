"""Helpers for working with SSH command strings returned by the backend."""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from typing import TYPE_CHECKING

from jarvislabs.exceptions import SSHError, ValidationError

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# ssh flags that consume the following token as their argument.
ARG_FLAGS = frozenset({"-o", "-i", "-F", "-J", "-l", "-p"})

# Safe defaults for non-interactive SSH, added when not already present.
# Order matters — it's the order they get appended to the command.
HARDENING_OPTIONS: tuple[tuple[str, str], ...] = (
    ("BatchMode", "yes"),
    ("ConnectTimeout", "15"),
    ("ServerAliveInterval", "15"),
    ("ServerAliveCountMax", "3"),
    ("UserKnownHostsFile", "/dev/null"),
    ("StrictHostKeyChecking", "no"),
    ("LogLevel", "ERROR"),
)


@dataclass(frozen=True)
class SSHInfo:
    """Structured view of a backend-provided SSH command."""

    user: str
    host: str
    port: int


def split_ssh_command(ssh_command: str) -> list[str]:
    """Split and validate a backend-provided SSH command string."""
    try:
        parts = shlex.split(ssh_command)
    except ValueError as exc:
        raise SSHError(f"Cannot parse SSH command: {ssh_command}") from exc
    if not parts or parts[0] != "ssh":
        raise SSHError(f"Cannot parse SSH command: {ssh_command}")
    return parts


def _iter_args(parts: list[str]) -> Iterator[tuple[str | None, str | None]]:
    """Walk an ssh command's arguments once, yielding ``(flag, arg)``.

    ``arg`` is None for flags that take no value (or a flag missing its value);
    ``flag`` is None for the positional target host.
    """
    i = 1
    while i < len(parts):
        token = parts[i]
        if token in ARG_FLAGS:
            arg = parts[i + 1] if i + 1 < len(parts) else None
            yield token, arg
            i += 2 if arg is not None else 1
        elif token.startswith("-"):
            yield token, None
            i += 1
        else:
            yield None, token
            i += 1


def parse_ssh_command(ssh_command: str) -> SSHInfo:
    """Extract user/host/port from a backend-provided SSH command string."""
    user: str | None = None
    host: str | None = None
    port = 22

    for flag, arg in _iter_args(split_ssh_command(ssh_command)):
        if flag is None:
            host = arg
        elif flag == "-l":
            if arg is None:
                raise SSHError(f"Missing SSH user in command: {ssh_command}")
            user = arg
        elif flag == "-p":
            if arg is None:
                raise SSHError(f"Missing port in SSH command: {ssh_command}")
            try:
                port = int(arg)
            except ValueError as exc:
                raise SSHError(f"Invalid SSH port in command: {ssh_command}") from exc
        elif flag == "-o" and arg is None:
            raise SSHError(f"Missing SSH option value in command: {ssh_command}")

    if host is None:
        raise SSHError(f"Missing target in SSH command: {ssh_command}")

    if "@" in host:
        user, host = host.split("@", 1)
    return SSHInfo(user=user or "root", host=host, port=port)


def _missing_hardening(parts: list[str]) -> list[str]:
    """`-o KEY=VALUE` pairs for hardening defaults not already present in the command."""
    present = {arg.split("=", 1)[0] for flag, arg in _iter_args(parts) if flag == "-o" and arg}
    additions: list[str] = []
    for key, value in HARDENING_OPTIONS:
        if key not in present:
            additions.extend(["-o", f"{key}={value}"])
    return additions


def _passthrough_options(parts: list[str], *, port_flag: str, command: str) -> list[str]:
    """Carry an ssh command's connection options into another tool (scp/rsync),
    translating the port flag and dropping `-l user` and the target host."""
    out: list[str] = []
    for flag, arg in _iter_args(parts):
        if flag == "-p":
            out.extend([port_flag, arg])
        elif flag in {"-o", "-i", "-F", "-J"}:
            if arg is None:
                raise SSHError(f"Missing SSH option value in command: {command}")
            out.extend([flag, arg])
        elif flag is not None and flag != "-l":
            out.append(flag)  # argless flag (e.g. -C); -l and the target are dropped
    return out


def harden_ssh_parts(parts: list[str]) -> list[str]:
    """Add safe non-interactive SSH defaults that aren't already set."""
    if not parts or parts[0] != "ssh":
        raise SSHError("Cannot harden a non-ssh command")

    additions = _missing_hardening(parts)
    if not additions:
        return parts

    insert_at = len(parts) - 1 if len(parts) > 1 else 1
    return [*parts[:insert_at], *additions, *parts[insert_at:]]


def build_scp_command(
    ssh_command: str,
    *,
    source: str,
    dest: str,
    upload: bool,
    recursive: bool = False,
) -> list[str]:
    """Build an scp command that reuses the SSH command's connection options."""
    parts = split_ssh_command(ssh_command)
    info = parse_ssh_command(ssh_command)

    command = ["scp", *_passthrough_options(parts, port_flag="-P", command=ssh_command), *_missing_hardening(parts)]
    if recursive:
        command.append("-r")

    target = f"{info.user}@{info.host}"
    if upload:
        command.extend([source, f"{target}:{dest}"])
    else:
        command.extend([f"{target}:{source}", dest])
    return command


def build_rsync_upload_command(
    ssh_command: str,
    *,
    source: str,
    dest: str,
) -> list[str]:
    """Build an rsync command that tunnels over the SSH command's transport."""
    parts = split_ssh_command(ssh_command)
    info = parse_ssh_command(ssh_command)

    transport = ["ssh", *_passthrough_options(parts, port_flag="-p", command=ssh_command), *_missing_hardening(parts)]

    source_path = source.rstrip("/") + "/"
    dest_path = dest.rstrip("/") + "/"
    command = ["rsync", "-az", "-e", shlex.join(transport), "--delete"]
    command.extend(["--exclude", ".venv/", "--exclude", ".git/", "--exclude", "__pycache__/"])
    command.extend([source_path, f"{info.user}@{info.host}:{dest_path}"])
    return command


def build_remote_shell_command(
    command: Sequence[str],
    *,
    cwd: str | None = None,
    env: Mapping[str, str] | None = None,
) -> str:
    """Build a safely quoted remote shell command for `ssh ... <command>`."""
    if not command:
        raise ValidationError("command cannot be empty")

    segments: list[str] = []

    if cwd:
        segments.append(f"cd {shlex.quote(cwd)}")

    if env:
        for key, value in env.items():
            if not ENV_KEY_RE.match(key):
                raise ValidationError(f"Invalid environment variable name: {key}")
            segments.append(f"export {key}={shlex.quote(value)}")

    rendered_command = " ".join(shlex.quote(part) for part in command)
    segments.append(rendered_command)

    script = " && ".join(segments)
    return f"sh -lc {shlex.quote(script)}"
