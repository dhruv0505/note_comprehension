"""SSH key resource."""

from __future__ import annotations

from typing import TYPE_CHECKING

from jarvislabs.exceptions import APIError
from jarvislabs.models import SSHKey
from jarvislabs.responses import _backend_msg, _normalize_success

if TYPE_CHECKING:
    from jarvislabs.transport import Transport


class SSHKeys:
    """Manage SSH keys."""

    def __init__(self, transport: Transport) -> None:
        self._t = transport

    def list(self) -> list[SSHKey]:
        resp = self._t.request("GET", "ssh/")
        return [SSHKey(**item) for item in resp]

    def add(self, ssh_key: str, key_name: str) -> bool:
        resp = self._t.request("POST", "ssh/", json={"ssh_key": ssh_key, "key_name": key_name})
        if not _normalize_success(resp):
            raise APIError(0, f"Failed to add SSH key: {_backend_msg(resp)}")
        return True

    def remove(self, key_id: str) -> bool:
        resp = self._t.request("DELETE", f"ssh/{key_id}")
        if not _normalize_success(resp):
            raise APIError(0, f"Failed to remove SSH key: {_backend_msg(resp)}")
        return True
