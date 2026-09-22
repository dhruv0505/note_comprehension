"""Account resource: balance, user info, resource metrics, catalog views."""

from __future__ import annotations

from typing import TYPE_CHECKING

from jarvislabs.constants import EUROPE_GPU_TYPES, EUROPE_REGION, REGION_URLS
from jarvislabs.models import (
    Balance,
    ResourceMetrics,
    ServerMetaGPU,
    ServerMetaResponse,
    Template,
    UserInfo,
)
from jarvislabs.responses import _normalize_success

if TYPE_CHECKING:
    from jarvislabs.transport import Transport


class Account:
    """Balance, user info, and resource metrics."""

    def __init__(self, transport: Transport) -> None:
        self._t = transport

    def balance(self) -> Balance:
        resp = self._t.request("GET", "users/balance")
        return Balance(**resp)

    def user_info(self) -> UserInfo:
        resp = self._t.request("GET", "users/user_info")
        return UserInfo(**resp)

    def resource_metrics(self) -> ResourceMetrics:
        resp = self._t.request("GET", "misc/resource_metrics")
        return ResourceMetrics(**resp)

    def templates(self) -> list[Template]:
        resp = self._t.request("GET", "misc/frameworks")
        return [Template(**t) for t in resp.get("frameworks", [])]

    def gpu_availability(self) -> list[ServerMetaGPU]:
        """GPU types, pricing, and current availability for new creates."""
        return self.gpu_availability_from(self.resources())

    def gpu_availability_from(self, meta: ServerMetaResponse) -> list[ServerMetaGPU]:
        """Build GPU availability rows from a server_meta response already in memory."""
        return [_listing_gpu_view(gpu) for gpu in meta.server_meta if gpu.region in REGION_URLS]

    def resources(self) -> ServerMetaResponse:
        """Fetch the backend resource catalog used by jl resources."""
        resp = self._t.request("GET", "misc/server_meta")
        return ServerMetaResponse(**resp)

    def currency(self) -> str:
        """Return 'INR' or 'USD' based on user's payment location."""
        resp = self._t.request("GET", "misc/")
        return "INR" if _normalize_success(resp) else "USD"


def _listing_gpu_view(gpu: ServerMetaGPU) -> ServerMetaGPU:
    if gpu.region != EUROPE_REGION or gpu.gpu_type not in EUROPE_GPU_TYPES or gpu.num_free_devices <= 0:
        return gpu
    return gpu.model_copy(update={"num_free_devices": 1})
