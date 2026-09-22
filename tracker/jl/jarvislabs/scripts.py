"""Startup script resource."""

from __future__ import annotations

from typing import TYPE_CHECKING

from jarvislabs.exceptions import APIError, ValidationError
from jarvislabs.models import StartupScript
from jarvislabs.responses import _backend_msg, _declared_failure

if TYPE_CHECKING:
    from jarvislabs.transport import Transport


class Scripts:
    """Manage startup scripts used during instance create/resume."""

    def __init__(self, transport: Transport) -> None:
        self._t = transport

    def list(self) -> list[StartupScript]:
        resp = self._t.request("GET", "scripts/")
        if not isinstance(resp, dict):
            raise APIError(0, "Failed to fetch scripts: unexpected response")
        if _declared_failure(resp):
            raise APIError(0, f"Failed to fetch scripts: {_backend_msg(resp)}")
        return [StartupScript(**item) for item in resp.get("script_meta", [])]

    def add(self, script: bytes | bytearray | str, name: str = "") -> bool:
        content = _coerce_script_bytes(script)
        params = {"name": name} if name else None
        self._t.request(
            "POST",
            "scripts/add",
            params=params,
            files={"script": ("startup.sh", content, "application/x-sh")},
        )
        return True

    def update(self, script_id: int, script: bytes | bytearray | str) -> bool:
        content = _coerce_script_bytes(script)
        self._t.request(
            "POST",
            "scripts/update",
            params={"script_id": script_id},
            files={"script": ("startup.sh", content, "application/x-sh")},
        )
        return True

    def remove(self, script_id: int) -> bool:
        self._t.request("DELETE", "scripts/", params={"script_id": script_id})
        return True


def _coerce_script_bytes(script: bytes | bytearray | str) -> bytes:
    if isinstance(script, bytes):
        content = script
    elif isinstance(script, bytearray):
        content = bytes(script)
    elif isinstance(script, str):
        content = script.encode("utf-8")
    else:
        raise ValidationError("script must be bytes, bytearray, or str")

    if not content.strip():
        raise ValidationError("Script content cannot be empty")
    return content
