"""Capacity policy read from the backend ``misc/server_meta`` catalog.

GPU region resolution and CPU-VM plan selection. Used by the Instances resource.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from jarvislabs.constants import (
    DEFAULT_TEMPLATE,
    INDIA_NOIDA_REGION,
    REGION_PRIORITY,
    REGION_URLS,
)
from jarvislabs.exceptions import ValidationError
from jarvislabs.models import ServerMetaGPU, ServerMetaResponse
from jarvislabs.regions import region_code

if TYPE_CHECKING:
    from jarvislabs.transport import Transport


def _resolve_region(
    transport: Transport,
    *,
    gpu_type: str | None,
    num_gpus: int,
    template: str = DEFAULT_TEMPLATE,
    is_spot: bool = False,
) -> str:
    """Auto-route to the best region via server_meta."""
    fallback = INDIA_NOIDA_REGION

    try:
        resp = transport.request("GET", "misc/server_meta")
        meta = ServerMetaResponse(**resp)
    except Exception:
        return fallback

    all_for_gpu = [s for s in meta.server_meta if s.gpu_type == gpu_type and s.region]
    active_for_gpu = [s for s in all_for_gpu if s.region in REGION_URLS]
    if template == "vm":
        candidates = [s for s in active_for_gpu if s.workload_type in ("vm", None)]
    else:
        candidates = [s for s in active_for_gpu if s.workload_type in ("container", None)]
    if is_spot:
        candidates = [s for s in candidates if s.spot_price is not None]
    if not candidates:
        # GPU exists but not for this workload type — fail early
        if all_for_gpu:
            if template == "vm" and any(s.workload_type == "container" for s in active_for_gpu):
                container_regions = ", ".join(
                    region_code(region)
                    for region in dict.fromkeys(s.region for s in active_for_gpu if s.workload_type == "container")
                )
                raise ValidationError(
                    f"{gpu_type} is available for containers in {container_regions}, but VM support is not available yet."
                )
            if not active_for_gpu:
                raise ValidationError(f"{gpu_type} is not available in any supported region right now.")
            kind = "VM" if template == "vm" else "container"
            if is_spot:
                raise ValidationError(f"Spot is not available for {gpu_type} right now.")
            raise ValidationError(f"{gpu_type} is not available for {kind} instances right now.")
        return fallback

    _priority = {r: i for i, r in enumerate(REGION_PRIORITY)}
    candidates.sort(key=lambda s: _priority.get(s.region, len(REGION_PRIORITY)))

    for server in candidates:
        if _available_devices(server, is_spot=is_spot) >= num_gpus:
            return server.region

    if is_spot:
        raise ValidationError(f"No free spot {gpu_type} GPUs right now. Try again later.")

    return candidates[0].region or fallback


def _available_devices(gpu: ServerMetaGPU, *, is_spot: bool) -> int:
    """Return the capacity count that matches the requested billing type."""
    if is_spot:
        return gpu.num_free_devices
    return gpu.effective_num_free_devices if gpu.effective_num_free_devices is not None else gpu.num_free_devices


def _check_gpu_in_region(
    transport: Transport,
    gpu_type: str,
    num_gpus: int,
    region: str,
    *,
    resume_only: bool = False,
    template: str = DEFAULT_TEMPLATE,
    require_spot: bool = False,
) -> None:
    """Raise early if the requested GPU isn't available in the target region."""
    try:
        resp = transport.request("GET", "misc/server_meta")
        meta = ServerMetaResponse(**resp)
    except Exception:
        return  # Can't check — let the backend decide

    if not meta.server_meta:
        return  # No data — let the backend decide

    in_region = [s for s in meta.server_meta if s.gpu_type == gpu_type and s.region == region]
    if template == "vm":
        in_region = [s for s in in_region if s.workload_type in ("vm", None)]
    else:
        in_region = [s for s in in_region if s.workload_type in ("container", None)]
    label = region_code(region)
    if not in_region:
        if template == "vm" and any(
            s.gpu_type == gpu_type and s.region == region and s.workload_type == "container" for s in meta.server_meta
        ):
            raise ValidationError(
                f"{gpu_type} is available for containers in {label}, but VM support is not available yet."
            )
        if resume_only:
            raise ValidationError(
                f"{gpu_type} is not available in {label}. Paused instances can only resume in their original region."
            )
        raise ValidationError(f"{gpu_type} is not available in {label}.")

    if require_spot and not any(s.spot_price is not None for s in in_region):
        raise ValidationError(f"Spot is not available for {gpu_type} in {label} right now.")

    free = any(_available_devices(s, is_spot=require_spot) >= num_gpus for s in in_region)
    if not free:
        raise ValidationError(f"No free {gpu_type} GPUs in {label} right now. Try again later.")


def _validate_create_region(
    transport: Transport,
    *,
    region: str,
    template: str,
    gpu_type: str,
    num_gpus: int,
    is_spot: bool = False,
) -> None:
    _check_gpu_in_region(transport, gpu_type, num_gpus, region, template=template, require_spot=is_spot)


def choose_cpu_vm_plan(
    transport: Transport,
    *,
    vcpus: int | None,
    ram: int | None,
    region: str | None,
) -> tuple[int, int, str]:
    """Return the requested CPU VM plan, or the smallest available plan.

    The backend owns the plan list in cpu_meta.combinations. The CLI only picks
    from that list and validates region availability.
    """
    try:
        meta = ServerMetaResponse(**transport.request("GET", "misc/server_meta"))
    except Exception as exc:
        raise ValidationError("Could not fetch CPU VM plans. Run jl resources to check availability.") from exc

    combinations = meta.cpu_meta.get("combinations") or []
    if vcpus is None and ram is None:
        available = [combo for combo in combinations if cpu_combo_available(combo)]
        if not available:
            raise ValidationError("No CPU VM plans available right now. Run jl resources to check availability.")
        available.sort(key=lambda combo: (int(combo.get("vcpus") or 0), int(combo.get("ram_gb") or 0)))
        combo = available[0]
    else:
        combo = next(
            (
                item
                for item in combinations
                if int(item.get("vcpus") or 0) == vcpus and int(item.get("ram_gb") or 0) == ram
            ),
            None,
        )
        if combo is None:
            raise ValidationError(f"Invalid CPU VM size: {vcpus} vCPU / {ram}GB RAM.")
        if not cpu_combo_available(combo):
            raise ValidationError(f"CPU VM size {vcpus} vCPU / {ram}GB RAM is not available right now.")

    selected_region = select_cpu_region(combo, region=region)
    return int(combo["vcpus"]), int(combo["ram_gb"]), selected_region


def cpu_combo_available(combo: dict) -> bool:
    """Return True when a CPU size is available in at least one region."""
    return bool(combo.get("available")) and any((combo.get("regions") or {}).values())


def select_cpu_region(combo: dict, *, region: str | None) -> str:
    """Pick the requested CPU VM region, or the first available preferred region."""
    region_availability = combo.get("regions") or {}
    if region is not None:
        if region_availability.get(region) is True:
            return region
        raise ValidationError(
            f"CPU VM size {combo.get('vcpus')} vCPU / {combo.get('ram_gb')}GB RAM is not available in {region_code(region)}."
        )

    for candidate in REGION_PRIORITY:
        if region_availability.get(candidate) is True:
            return candidate
    for candidate, available in region_availability.items():
        if available:
            return candidate
    raise ValidationError("No CPU VM plans available right now. Run jl resources to check availability.")
