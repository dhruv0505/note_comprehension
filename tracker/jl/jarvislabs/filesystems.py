"""Persistent filesystem resource."""

from __future__ import annotations

from typing import TYPE_CHECKING

from jarvislabs.constants import INDIA_NOIDA_REGION
from jarvislabs.exceptions import APIError, ValidationError
from jarvislabs.models import Filesystem
from jarvislabs.regions import normalize_region, region_base_url
from jarvislabs.responses import _backend_msg, _declared_failure

if TYPE_CHECKING:
    from jarvislabs.transport import Transport


class Filesystems:
    """Manage persistent filesystems."""

    def __init__(self, transport: Transport) -> None:
        self._t = transport

    def list(self) -> list[Filesystem]:
        resp = self._t.request("GET", "filesystem/list")
        if not isinstance(resp, list):
            raise APIError(0, "Failed to fetch filesystems: unexpected response")
        return [Filesystem(**item) for item in resp]

    def create(self, fs_name: str, storage: int, *, region: str | None = None, deployment_id: str | None = None) -> int:
        _validate_filesystem_name(fs_name)
        _validate_filesystem_storage(storage)
        region = normalize_region(region) or INDIA_NOIDA_REGION
        payload: dict[str, str | int] = {"fs_name": fs_name, "storage": storage, "region": region}
        if deployment_id:
            payload["deployment_id"] = deployment_id
        base_url = region_base_url(region)
        resp = self._t.request("POST", "filesystem/create", json=payload, base_url=base_url)
        fs_id = resp.get("fs_id") if isinstance(resp, dict) else None
        if fs_id is None:
            raise APIError(0, f"Failed to create filesystem: {_backend_msg(resp)}")
        return int(fs_id)

    def edit(self, fs_id: int, storage: int) -> int:
        _validate_filesystem_storage(storage)
        base_url = region_base_url(self._fs_region(fs_id))
        resp = self._t.request("POST", "filesystem/edit", json={"fs_id": fs_id, "storage": storage}, base_url=base_url)
        edited_fs_id = resp.get("fs_id") if isinstance(resp, dict) else None
        if edited_fs_id is None:
            raise APIError(0, f"Failed to edit filesystem: {_backend_msg(resp)}")
        return int(edited_fs_id)

    def remove(self, fs_id: int) -> bool:
        base_url = region_base_url(self._fs_region(fs_id))
        resp = self._t.request("POST", "filesystem/delete", params={"fs_id": fs_id}, base_url=base_url)
        if _declared_failure(resp):
            raise APIError(0, f"Failed to remove filesystem: {_backend_msg(resp)}")
        return True

    def _fs_region(self, fs_id: int) -> str:
        """Look up the region of a filesystem by ID."""
        for fs in self.list():
            if fs.fs_id == fs_id:
                if not fs.region:
                    raise ValidationError(
                        f"Filesystem {fs_id} has no region set. "
                        "It may have been created before region support. Please recreate it."
                    )
                return fs.region
        raise ValidationError(f"Filesystem {fs_id} not found. Check the ID with: jl filesystem list")


def _validate_filesystem_name(fs_name: str) -> None:
    if not fs_name or not fs_name.strip():
        raise ValidationError("Filesystem name cannot be empty")
    if len(fs_name) > 30:
        raise ValidationError("Filesystem name must be 30 characters or fewer")


def _validate_filesystem_storage(storage: int) -> None:
    if storage < 50 or storage > 2048:
        raise ValidationError("Filesystem storage must be between 50GB and 2048GB")
