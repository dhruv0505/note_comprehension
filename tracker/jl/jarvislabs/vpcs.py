"""VPC resource — private networks for VM instances.

Validation happens server side and error messages pass through as-is.
list() returns VPCs from every region in one call; create/ips/delete are
routed to the VPC's own region. There is no single-VPC endpoint, so get()
resolves from list().
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from jarvislabs.exceptions import APIError, NotFoundError, ValidationError
from jarvislabs.models import Vpc, VpcIP
from jarvislabs.regions import normalize_region, region_base_url
from jarvislabs.responses import _backend_msg, _declared_failure

if TYPE_CHECKING:
    from jarvislabs.transport import Transport


class Vpcs:
    """Manage VPCs: list, get, create, delete, and inspect allocated IPs."""

    def __init__(self, transport: Transport) -> None:
        self._t = transport

    def list(self) -> list[Vpc]:
        resp = self._t.request("GET", "vpc/list")
        if not isinstance(resp, list):
            raise APIError(0, "Failed to fetch VPCs: unexpected response")
        return [Vpc(**item) for item in resp]

    def get(self, vpc_id: str) -> Vpc:
        for vpc in self.list():
            if vpc.vpc_id == vpc_id:
                return vpc
        raise NotFoundError(f"VPC {vpc_id} not found. Check the ID with: jl vpc list")

    def create(self, name: str, cidr: str, region: str) -> Vpc:
        """Create a VPC."""
        internal_region = normalize_region(region)
        if internal_region is None:
            raise ValidationError("VPC creation requires a region (e.g. 'IN1').")
        resp = self._t.request(
            "POST",
            "vpc/create",
            json={"name": name, "cidr": cidr, "region": internal_region},
            base_url=region_base_url(internal_region),
        )
        if not isinstance(resp, dict) or "vpc_id" not in resp:
            raise APIError(0, f"Failed to create VPC: {_backend_msg(resp)}")
        return Vpc(**resp)

    def ips(self, vpc_id: str) -> list[VpcIP]:
        """List the private IPs allocated in a VPC and the machines holding them."""
        resp = self._t.request("GET", f"vpc/{vpc_id}/ports", base_url=self._vpc_base_url(vpc_id))
        if not isinstance(resp, list):
            raise APIError(0, "Failed to fetch VPC IPs: unexpected response")
        return [VpcIP(**item) for item in resp]

    def delete(self, vpc_id: str) -> bool:
        """Delete an empty VPC.

        The default VPC cannot be deleted. A VPC with machines in it must be
        emptied first: destroy the VMs, or pause and resume them into another
        VPC.
        """
        resp = self._t.request("DELETE", f"vpc/{vpc_id}", base_url=self._vpc_base_url(vpc_id))
        if _declared_failure(resp):
            raise APIError(0, f"Failed to delete VPC: {_backend_msg(resp)}")
        return True

    def _vpc_base_url(self, vpc_id: str) -> str:
        """Look up a VPC's region and return its backend base URL."""
        vpc = self.get(vpc_id)
        return region_base_url(vpc.region)
