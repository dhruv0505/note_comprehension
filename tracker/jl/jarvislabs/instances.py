"""Instance lifecycle resource: list, create, pause, resume, destroy, rename."""

from __future__ import annotations

import re
import time
from typing import TYPE_CHECKING

from jarvislabs.constants import (
    DEFAULT_INSTANCE_NAME,
    DEFAULT_NUM_GPUS,
    DEFAULT_POLL_TIMEOUT_S,
    DEFAULT_REGION,
    DEFAULT_STORAGE_GB,
    DEFAULT_TEMPLATE,
    EUROPE_GPU_COUNTS,
    EUROPE_GPU_TYPES,
    EUROPE_REGION,
    FETCH_RETRY_INTERVAL_S,
    POLL_INTERVAL_S,
    RETRY_STATUS_CODES,
)
from jarvislabs.exceptions import APIError, NotFoundError, ValidationError
from jarvislabs.models import Instance, InstanceListResponse, StatusResponse
from jarvislabs.regions import normalize_region, region_base_url, region_code
from jarvislabs.responses import _backend_msg, _declared_failure, _normalize_success
from jarvislabs.server_meta import (
    _check_gpu_in_region,
    _resolve_region,
    _validate_create_region,
    choose_cpu_vm_plan,
)
from jarvislabs.vpcs import Vpcs

if TYPE_CHECKING:
    from jarvislabs.ssh_keys import SSHKeys
    from jarvislabs.transport import Transport

_INSTANCE_NAME_RE = re.compile(r"[a-zA-Z0-9 _-]+\Z")


def _validate_instance_name(name: str) -> None:
    if not name or not name.strip():
        raise ValidationError("Instance name cannot be empty")
    if len(name) > 40:
        raise ValidationError("Instance name must be 40 characters or fewer")
    if not _INSTANCE_NAME_RE.fullmatch(name):
        raise ValidationError(
            f"Instance name contains unsupported characters: {name!r}. "
            "Only letters, numbers, spaces, hyphens, and underscores are allowed."
        )


def _validate_vpc_template(vpc_id: str | None, template: str) -> None:
    if vpc_id is not None and template != "vm":
        raise ValidationError("VPCs are only supported for VM instances — containers cannot join a VPC.")


def _resolve_vpc_region(transport: Transport, vpc_id: str, region: str | None) -> str:
    """Return the VPC's region, validating any requested region against it."""
    vpc = Vpcs(transport).get(vpc_id)
    if region and region != vpc.region:
        raise ValidationError(
            f"VPC {vpc_id} is in {region_code(vpc.region)}, "
            f"but the instance region is {region_code(region)}. "
            "A VM and its VPC must be in the same region."
        )
    return vpc.region


def _validate_europe(gpu_type: str, num_gpus: int) -> None:
    label = region_code(EUROPE_REGION)
    if gpu_type not in EUROPE_GPU_TYPES:
        raise ValidationError(f"{label} supports only {sorted(EUROPE_GPU_TYPES)} GPUs, got {gpu_type}")
    if num_gpus not in EUROPE_GPU_COUNTS:
        raise ValidationError(f"{label} supports only 1 GPU per instance, got {num_gpus}")


def _ensure_filesystem_exists(transport: Transport, fs_id: int, region: str | None = None) -> None:
    resp = transport.request("GET", "filesystem/list")
    if not isinstance(resp, list):
        raise APIError(0, "Failed to fetch filesystems: unexpected response")

    for item in resp:
        if not isinstance(item, dict):
            continue
        if int(item.get("fs_id", -1)) == fs_id:
            if region and item.get("region"):
                fs_region = item["region"]
                if fs_region != region:
                    raise ValidationError(
                        f"Filesystem {fs_id} is in {region_code(fs_region)}, "
                        f"but the instance is in {region_code(region)}. "
                        f"Filesystems can only be attached to instances in the same region."
                    )
            return

    raise ValidationError(f"Filesystem {fs_id} not found. Check the ID with: jl filesystem list")


def is_cpu_vm(instance: Instance) -> bool:
    """Return True when an instance row represents a CPU VM."""
    return getattr(instance, "template", None) == "vm" and getattr(instance, "gpu_type", None) == "CPU"


def _lifecycle_endpoint(instance: Instance, action: str) -> str:
    """Pick the backend endpoint for a lifecycle action (pause/destroy) by instance kind."""
    if is_cpu_vm(instance):
        return f"templates/vm/cpu/{action}"
    if instance.template == "vm":
        return f"templates/vm/{action}"
    return f"misc/{action}"


class Instances:
    """Instance lifecycle: list, create, pause, resume, destroy."""

    def __init__(self, transport: Transport, ssh_keys: SSHKeys) -> None:
        self._t = transport
        self._ssh_keys = ssh_keys

    def list(self) -> list[Instance]:
        return _fetch_instances(self._t)

    def get(self, machine_id: int) -> Instance:
        return _get_instance(self._t, machine_id)

    def _require_ssh_key(self) -> None:
        if not self._ssh_keys.list():
            raise ValidationError(
                "VM instances require at least one SSH key. Add one with: jl ssh-key add <pubkey-file> --name 'my-key'"
            )

    def resolve_cpu_vm_plan(
        self,
        *,
        vcpus: int | None = None,
        ram: int | None = None,
        region: str | None = None,
    ) -> tuple[int, int, str]:
        """Resolve the CPU VM size and region from backend cpu_meta before prompting."""
        return choose_cpu_vm_plan(
            self._t,
            vcpus=vcpus,
            ram=ram,
            region=normalize_region(region),
        )

    def create(
        self,
        *,
        gpu_type: str | None = None,
        num_gpus: int = DEFAULT_NUM_GPUS,
        template: str = DEFAULT_TEMPLATE,
        cpu: bool = False,
        vcpus: int | None = None,
        ram: int | None = None,
        storage: int = DEFAULT_STORAGE_GB,
        name: str = DEFAULT_INSTANCE_NAME,
        disk_type: str = "ssd",
        http_ports: str = "",
        script_id: str | None = None,
        script_args: str = "",
        fs_id: int | None = None,
        arguments: str = "",
        region: str | None = None,
        is_spot: bool = False,
        vpc_id: str | None = None,
    ) -> Instance:
        """Create a GPU instance by default, or a CPU VM when cpu=True.

        The CLI and SDK both use this as their main create entrypoint. CPU VMs
        use a different backend endpoint, so cpu=True explicitly routes into
        the CPU VM flow after rejecting options that only apply to GPU creates.
        """
        _validate_vpc_template(vpc_id, template)
        if cpu:
            if template != "vm":
                raise ValidationError("CPU instances must be created as VMs. Use: jl create --vm --cpu")
            if gpu_type:
                raise ValidationError("--cpu cannot be used with --gpu.")
            if num_gpus != 1:
                raise ValidationError("--num-gpus is not supported with CPU VMs.")
            if disk_type != "ssd":
                raise ValidationError("--disk-type is not supported with CPU VMs.")
            if http_ports:
                raise ValidationError("--http-ports is not supported with CPU VMs.")
            if script_id is not None or script_args:
                raise ValidationError("Startup scripts are not supported with CPU VMs.")
            if fs_id is not None:
                raise ValidationError("--fs-id is not supported with CPU VMs.")
            if arguments:
                raise ValidationError("--arguments is not supported with CPU VMs.")
            if is_spot:
                raise ValidationError("Spot instances are not available for CPU VMs.")
            return self.create_cpu_vm(
                vcpus=vcpus,
                ram=ram,
                storage=storage,
                name=name,
                region=region,
                vpc_id=vpc_id,
            )

        if not gpu_type:
            raise ValidationError("gpu_type is required (e.g. 'A100', 'H100', 'L4')")
        if is_spot and template == "vm":
            raise ValidationError("Spot instances are only supported for GPU containers.")
        if name:
            _validate_instance_name(name)

        region = normalize_region(region)
        if vpc_id is not None:
            region = _resolve_vpc_region(self._t, vpc_id, region)
        if region is None:
            region = _resolve_region(self._t, gpu_type=gpu_type, num_gpus=num_gpus, template=template, is_spot=is_spot)
        else:
            _validate_create_region(
                self._t,
                region=region,
                template=template,
                gpu_type=gpu_type,
                num_gpus=num_gpus,
                is_spot=is_spot,
            )

        if region == EUROPE_REGION:
            _validate_europe(gpu_type, num_gpus)

        if template == "vm":
            self._require_ssh_key()
        if fs_id is not None:
            _ensure_filesystem_exists(self._t, fs_id, region=region)

        payload: dict = {
            "gpu_type": gpu_type,
            "num_gpus": num_gpus,
            "hdd": storage,
            "region": region,
            "name": name,
            "is_spot": is_spot,
            "duration": "hour",
            "disk_type": disk_type,
            "http_ports": http_ports,
            "script_id": script_id,
            "script_args": script_args,
            "fs_id": fs_id,
            "arguments": arguments,
        }
        if vpc_id is not None:
            payload["vpc_id"] = vpc_id

        base_url = region_base_url(region)
        resp = self._t.request(
            "POST",
            f"templates/{template}/create",
            json=payload,
            base_url=base_url,
        )

        machine_id = resp.get("machine_id")
        if not machine_id:
            raise APIError(0, f"Instance creation failed: {_backend_msg(resp)}")

        # Poll until running, then fetch full instance details
        _poll_until_running(self._t, machine_id, region)
        return _get_instance(self._t, machine_id, retries=3)

    def create_cpu_vm(
        self,
        *,
        vcpus: int | None,
        ram: int | None,
        storage: int,
        name: str,
        region: str | None,
        vpc_id: str | None = None,
    ) -> Instance:
        """Create a CPU VM through the dedicated backend endpoint.

        CPU VM plans are resolved from backend cpu_meta so the CLI does not
        duplicate the backend's allowed vCPU/RAM combinations.
        """
        if (vcpus is None) != (ram is None):
            raise ValidationError("CPU VM sizing requires both --vcpus and --ram.")
        if storage < DEFAULT_STORAGE_GB:
            raise ValidationError(f"CPU VMs require at least {DEFAULT_STORAGE_GB}GB storage.")
        if name:
            _validate_instance_name(name)
        self._require_ssh_key()

        normalized_region = normalize_region(region)
        if vpc_id is not None:
            normalized_region = _resolve_vpc_region(self._t, vpc_id, normalized_region)
        selected_vcpus, selected_ram, selected_region = choose_cpu_vm_plan(
            self._t,
            vcpus=vcpus,
            ram=ram,
            region=normalized_region,
        )

        payload: dict = {
            "num_cpus": 1,
            "vcpus": selected_vcpus,
            "ram_gb": selected_ram,
            "hdd": storage,
            "region": selected_region,
            "name": name,
            "duration": "hour",
            "disk_type": "ssd",
        }
        if vpc_id is not None:
            payload["vpc_id"] = vpc_id

        resp = self._t.request(
            "POST",
            "templates/vm/cpu/create",
            json=payload,
            base_url=region_base_url(selected_region),
        )

        machine_id = resp.get("machine_id")
        if not machine_id:
            raise APIError(0, f"CPU VM creation failed: {_backend_msg(resp)}")

        _poll_until_running(self._t, machine_id, selected_region)
        return _get_instance(self._t, machine_id, retries=3)

    def pause(self, machine_id: int) -> bool:
        instance = _get_instance(self._t, machine_id)
        base_url = region_base_url(instance.region or DEFAULT_REGION)
        endpoint = _lifecycle_endpoint(instance, "pause")

        resp = self._t.request(
            "POST",
            endpoint,
            params={"machine_id": machine_id},
            base_url=base_url,
        )
        if not _normalize_success(resp):
            raise APIError(0, f"Failed to pause instance: {_backend_msg(resp)}")
        return True

    def resume(
        self,
        machine_id: int,
        *,
        gpu_type: str | None = None,
        num_gpus: int | None = None,
        storage: int | None = None,
        name: str | None = None,
        http_ports: str = "",
        script_id: str | None = None,
        script_args: str | None = None,
        fs_id: int | None = None,
        vcpus: int | None = None,
        ram: int | None = None,
        is_spot: bool = False,
        vpc_id: str | None = None,
    ) -> Instance:
        instance = _get_instance(self._t, machine_id)
        if instance.status != "Paused":
            raise ValidationError(f"Can only resume a Paused instance (current status: {instance.status})")
        if name:
            _validate_instance_name(name)
        _validate_vpc_template(vpc_id, instance.template)
        if vpc_id is not None:
            _resolve_vpc_region(self._t, vpc_id, instance.region or DEFAULT_REGION)
        # Resume is region-locked — backend always uses instance's original region
        region = instance.region or DEFAULT_REGION
        base_url = region_base_url(region)
        is_vm = instance.template == "vm"

        if is_cpu_vm(instance):
            if gpu_type is not None:
                raise ValidationError("--gpu is not supported when resuming CPU VMs.")
            if num_gpus is not None:
                raise ValidationError("--num-gpus is not supported when resuming CPU VMs.")
            if http_ports:
                raise ValidationError("--http-ports is not supported when resuming CPU VMs.")
            if script_id is not None or script_args is not None:
                raise ValidationError("Startup scripts are not supported when resuming CPU VMs.")
            if fs_id is not None:
                raise ValidationError("--fs-id is not supported when resuming CPU VMs.")
            if is_spot:
                raise ValidationError("Spot instances are not available for CPU VMs.")
            return self.resume_cpu_vm(
                instance,
                machine_id=machine_id,
                storage=storage,
                name=name,
                vcpus=vcpus,
                ram=ram,
                vpc_id=vpc_id,
            )

        if vcpus is not None or ram is not None:
            raise ValidationError("--vcpus and --ram are only supported for CPU VMs.")
        if is_spot and is_vm:
            raise ValidationError("Spot instances are only supported for GPU containers.")
        if fs_id is not None:
            _ensure_filesystem_exists(self._t, fs_id, region=region)

        # Warn early if the requested GPU isn't available in this region
        if gpu_type and gpu_type != instance.gpu_type:
            _check_gpu_in_region(
                self._t,
                gpu_type,
                num_gpus or instance.num_gpus or 1,
                region,
                resume_only=True,
                template=instance.template,
            )
        if is_spot:
            effective_spot_gpu = gpu_type or instance.gpu_type
            if not effective_spot_gpu:
                raise ValidationError("Cannot resume as spot because the instance GPU type is unknown.")
            _check_gpu_in_region(
                self._t,
                effective_spot_gpu,
                num_gpus or instance.num_gpus or 1,
                region,
                resume_only=True,
                template=instance.template,
                require_spot=True,
            )
        if is_vm:
            self._require_ssh_key()

        effective_gpu = gpu_type or instance.gpu_type
        effective_num = num_gpus or instance.num_gpus or 1
        if region == EUROPE_REGION and effective_gpu:
            _validate_europe(effective_gpu, effective_num)

        payload: dict = {
            "machine_id": machine_id,
            "gpu_type": gpu_type or instance.gpu_type,
            "num_gpus": num_gpus or instance.num_gpus,
            "is_spot": is_spot,
            "hdd": storage or instance.storage_gb,
            "name": name or instance.name,
            "duration": "hour",
            "script_id": script_id,
            "http_ports": http_ports,
            "script_args": script_args or "",
            "fs_id": fs_id if fs_id is not None else instance.fs_id,
            "arguments": "",
        }
        # Omitted vpc_id keeps the VM's current VPC.
        if vpc_id is not None:
            payload["vpc_id"] = vpc_id

        resp = self._t.request(
            "POST",
            f"templates/{instance.template}/resume",
            json=payload,
            base_url=base_url,
        )

        mid = resp.get("machine_id")
        if not mid:
            raise APIError(0, f"Failed to resume instance: {_backend_msg(resp)}")

        _poll_until_running(self._t, mid, region)
        return _get_instance(self._t, mid, retries=3)

    def resume_cpu_vm(
        self,
        instance: Instance,
        *,
        machine_id: int,
        storage: int | None,
        name: str | None,
        vcpus: int | None,
        ram: int | None,
        vpc_id: str | None = None,
    ) -> Instance:
        """Resume a paused CPU VM through the CPU VM backend endpoint."""
        if (vcpus is None) != (ram is None):
            raise ValidationError("CPU VM sizing requires both --vcpus and --ram.")
        if storage is not None and storage < DEFAULT_STORAGE_GB:
            raise ValidationError(f"CPU VMs require at least {DEFAULT_STORAGE_GB}GB storage.")
        if vcpus is not None and ram is not None:
            choose_cpu_vm_plan(self._t, vcpus=vcpus, ram=ram, region=instance.region)
        self._require_ssh_key()

        payload: dict = {
            "machine_id": machine_id,
            "num_cpus": 1,
            "hdd": storage or instance.storage_gb,
            "name": name or instance.name,
        }
        if vcpus is not None and ram is not None:
            payload["vcpus"] = vcpus
            payload["ram_gb"] = ram
        if vpc_id is not None:
            payload["vpc_id"] = vpc_id

        region = instance.region or DEFAULT_REGION
        resp = self._t.request(
            "POST",
            "templates/vm/cpu/resume",
            json=payload,
            base_url=region_base_url(region),
        )

        mid = resp.get("machine_id")
        if not mid:
            raise APIError(0, f"Failed to resume CPU VM: {_backend_msg(resp)}")

        _poll_until_running(self._t, mid, region)
        return _get_instance(self._t, mid, retries=3)

    def destroy(self, machine_id: int) -> bool:
        instance = _get_instance(self._t, machine_id)
        base_url = region_base_url(instance.region or DEFAULT_REGION)
        endpoint = _lifecycle_endpoint(instance, "destroy")

        try:
            resp = self._t.request(
                "POST",
                endpoint,
                params={"machine_id": machine_id},
                base_url=base_url,
            )
        except APIError as exc:
            if _wait_until_instance_destroying_or_missing(self._t, machine_id):
                return True
            raise exc

        if not _normalize_success(resp):
            if _wait_until_instance_destroying_or_missing(self._t, machine_id):
                return True
            raise APIError(0, f"Failed to destroy instance: {_backend_msg(resp)}")
        return True

    def rename(self, machine_id: int, name: str) -> bool:
        _validate_instance_name(name)

        resp = self._t.request(
            "PUT",
            "machines/machine_name",
            params={"machine_id": machine_id, "machine_name": name},
        )
        if _declared_failure(resp):
            raise APIError(0, f"Failed to rename instance: {_backend_msg(resp)}")
        return True


def _poll_until_running(transport: Transport, machine_id: int, region: str) -> None:
    """Poll /misc/status until Running or Failed. Raises on failure/timeout."""
    base_url = region_base_url(region)
    deadline = time.monotonic() + DEFAULT_POLL_TIMEOUT_S
    transient_errors = 0
    max_transient_errors = 5

    while time.monotonic() < deadline:
        try:
            resp = transport.request(
                "GET",
                "misc/status",
                params={"machine_id": machine_id},
                base_url=base_url,
            )
            status = StatusResponse(**resp)
            transient_errors = 0  # reset on success
        except NotFoundError:
            # 404 = no ErrorLog row yet → still creating, keep polling
            transient_errors = 0
            time.sleep(POLL_INTERVAL_S)
            continue
        except APIError as exc:
            # Only retry transient errors (timeouts, server errors, rate limits).
            # Permanent errors (400, 422, etc.) should fail immediately.
            if exc.status_code not in RETRY_STATUS_CODES | {0}:
                raise
            transient_errors += 1
            if transient_errors >= max_transient_errors:
                raise
            time.sleep(POLL_INTERVAL_S)
            continue

        if status.status == "Running":
            return
        if status.status == "Failed":
            code_hint = f" (code={status.code})" if status.code else ""
            raise APIError(0, f"Instance creation failed: {status.error or 'unknown error'}{code_hint}")

        time.sleep(POLL_INTERVAL_S)

    raise APIError(
        0,
        f"Timed out after {DEFAULT_POLL_TIMEOUT_S}s waiting for instance {machine_id} to start. "
        f"It may still come up shortly. Check it with: jl get {machine_id}",
    )


def _fetch_instances(transport: Transport) -> list[Instance]:
    """Fetch all instances."""
    resp = transport.request("GET", "users/fetch")
    if not _normalize_success(resp):
        raise APIError(0, f"Failed to fetch instances: {_backend_msg(resp)}")
    parsed = InstanceListResponse(**resp)
    return parsed.instances


def _get_instance(transport: Transport, machine_id: int, *, retries: int = 0) -> Instance:
    """Fetch a specific instance by machine_id, or raise NotFoundError.

    retries > 0 is used after create/resume to handle brief fetch delays.
    """
    for attempt in range(retries + 1):
        try:
            resp = transport.request("GET", f"users/fetch/{machine_id}")
            if not _normalize_success(resp):
                raise NotFoundError(f"Instance {machine_id} not found")
            instance_data = resp.get("instance")
            if not instance_data:
                raise NotFoundError(f"Instance {machine_id} not found")
            return Instance(**instance_data)
        except NotFoundError as err:
            if attempt < retries:
                time.sleep(FETCH_RETRY_INTERVAL_S)
                continue
            raise NotFoundError(f"Instance {machine_id} not found. Check the ID with: jl list") from err


def _wait_until_instance_destroying_or_missing(
    transport: Transport,
    machine_id: int,
    *,
    retries: int = 20,
    poll_interval_s: float = 1.0,
) -> bool:
    """Return True when the instance is gone or destroy has clearly started."""
    for attempt in range(retries + 1):
        try:
            inst = _get_instance(transport, machine_id)
        except NotFoundError:
            return True
        if str(getattr(inst, "status", "")).lower() == "destroying":
            return True

        if attempt < retries:
            time.sleep(poll_interval_s)

    return False
