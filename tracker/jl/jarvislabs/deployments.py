"""Serverless model deployments resource."""

from __future__ import annotations

import contextlib
import time
from datetime import datetime
from typing import TYPE_CHECKING

from jarvislabs.constants import (
    DEPLOYMENT_POLL_INTERVAL_S,
    DEPLOYMENT_POLL_MAX_TRANSIENT_ERRORS,
    DEPLOYMENT_TRANSIENT_STATUS,
    DEPLOYMENT_WAIT_TIMEOUT_S,
    SERVERLESS_REGIONS,
)
from jarvislabs.exceptions import (
    APIError,
    JarvislabsError,
    NotFoundError,
    RegionResolutionError,
    ValidationError,
)
from jarvislabs.models import (
    Deployment,
    DeploymentListResult,
    DeploymentStatus,
    DeploymentSummary,
    RegionError,
)
from jarvislabs.regions import normalize_serverless_region, region_code, serverless_region_url

if TYPE_CHECKING:
    from collections.abc import Iterator

    from jarvislabs.transport import Transport


class Deployments:
    """Serverless model deployments: create, list, get, status, update, delete.

    A deployment lives in exactly one region, and an id alone doesn't say
    which. Every per-id method therefore takes an optional ``region``: when
    given, we go straight to that region's host; when not, ``_find_deployment``
    asks every region and exactly one is expected to have it.
    """

    def __init__(self, transport: Transport) -> None:
        self._t = transport

    def create(
        self,
        *,
        name: str,
        region: str,
        framework: str,
        gpu: str,
        model: str,
        gpus_per_worker: int,
        min_workers: int,
        max_workers: int,
        idle_timeout: int,
        wait_time: int,
        storage: int,
        concurrent: int | None = None,
        args: dict[str, str] | None = None,
        env: dict[str, str] | None = None,
    ) -> str:
        """Create a deployment and return its id.

        Creation is asynchronous: the id comes back immediately while the
        deployment builds in the background. Poll with ``wait_until_running``.
        """
        internal_region = normalize_serverless_region(region)
        payload: dict = {
            "name": name,
            "region": internal_region,
            "framework": framework,
            "gpus_to_use": {"gpus": [gpu]},
            "gpus_per_worker": gpus_per_worker,
            "min_workers": min_workers,
            "max_workers": max_workers,
            "idle_timeout": idle_timeout,
            "wait_time": wait_time,
            "storage": storage,
            "args": {**(args or {}), "model": model},
        }
        if env is not None:
            payload["env"] = env
        if concurrent is not None:
            payload["concurrent_requests"] = concurrent

        base_url = serverless_region_url(internal_region)
        resp = self._t.request("POST", "management/create", json=payload, base_url=base_url)
        deployment_id = resp.get("deployment_id")
        if not deployment_id:
            raise JarvislabsError("Failed to create deployment: no deployment_id returned.")
        return deployment_id

    def list(self) -> DeploymentListResult:
        """List deployments across all serverless regions, newest first.

        Regions that can't be reached are reported in ``region_errors`` rather
        than hiding the rest; if no region answers at all, raise.
        """
        deployments: list[DeploymentSummary] = []
        region_errors: list[RegionError] = []
        answered = False

        for region in sorted(SERVERLESS_REGIONS):
            try:
                resp = self._t.request("GET", "management/list", base_url=serverless_region_url(region))
            except APIError as exc:
                if exc.status_code not in DEPLOYMENT_TRANSIENT_STATUS:
                    raise
                region_errors.append(RegionError(region=region, error=str(exc)))
                continue
            answered = True
            for item in resp.get("deployments", []):
                deployments.append(DeploymentSummary(**{**item, "region": region}))

        if not answered and region_errors:
            raise JarvislabsError("; ".join(f"{region_code(e.region)}: {e.error}" for e in region_errors))

        deployments.sort(key=_newest_first_key, reverse=True)
        return DeploymentListResult(deployments=deployments, region_errors=region_errors)

    def get(self, deployment_id: str, *, region: str | None = None) -> Deployment:
        """Fetch the full deployment record, including live worker info."""
        if region:
            return self._fetch_deployment(deployment_id, normalize_serverless_region(region))
        _, deployment = self._find_deployment(deployment_id)
        return deployment

    def status(self, deployment_id: str, *, region: str | None = None) -> DeploymentStatus:
        """Fetch just the lifecycle status, with the deployment's region attached."""
        if region:
            resolved_region = normalize_serverless_region(region)
            status = self._fetch_status(deployment_id, resolved_region)
        else:
            resolved_region, deployment = self._find_deployment(deployment_id)
            status = deployment.status
        return DeploymentStatus(deployment_id=deployment_id, region=resolved_region, status=status)

    def update(
        self,
        deployment_id: str,
        *,
        name: str | None = None,
        idle_timeout: int | None = None,
        wait_time: int | None = None,
        region: str | None = None,
    ) -> Deployment:
        """Update a running deployment and return the refreshed record.

        Only ``name``, ``idle_timeout``, and ``wait_time`` can change; scaling
        settings require recreating the deployment.
        """
        patch: dict = {}
        if name is not None:
            patch["name"] = name
        if idle_timeout is not None:
            patch["idle_timeout"] = idle_timeout
        if wait_time is not None:
            patch["wait_time"] = wait_time
        if not patch:
            raise ValidationError("Pass at least one of name / idle_timeout / wait_time.")

        # Read before writing: confirms the deployment exists and gives us its
        # current status for a clear message if the update is rejected.
        if region:
            resolved_region = normalize_serverless_region(region)
            deployment = self._fetch_deployment(deployment_id, resolved_region)
        else:
            resolved_region, deployment = self._find_deployment(deployment_id)

        try:
            self._t.request(
                "PATCH", f"management/{deployment_id}", json=patch, base_url=serverless_region_url(resolved_region)
            )
        except APIError as exc:
            if exc.status_code == 409:
                raise ValidationError(
                    f"Deployment is in '{deployment.status}'; only running deployments can be updated. "
                    "Min/max workers are not patchable — recreate to rescale."
                ) from exc
            raise
        return self.get(deployment_id, region=resolved_region)

    def delete(self, deployment_id: str, *, region: str | None = None) -> bool:
        """Delete a deployment. A 404 on the delete itself (it vanished after
        lookup) is treated as success; an unknown id raises ``NotFoundError``."""
        if region:
            resolved_region = normalize_serverless_region(region)
            self._fetch_deployment(deployment_id, resolved_region)  # confirm it exists before deleting
        else:
            resolved_region, _ = self._find_deployment(deployment_id)

        with contextlib.suppress(NotFoundError):
            self._t.request("DELETE", f"management/{deployment_id}", base_url=serverless_region_url(resolved_region))
        return True

    def wait_until_running(
        self, deployment_id: str, *, region: str | None = None, timeout: float = DEPLOYMENT_WAIT_TIMEOUT_S
    ) -> Deployment:
        """Poll every few seconds until the deployment is running, then return it.

        Raises if the deployment fails or gets deleted while waiting, or once
        ``timeout`` seconds pass without reaching a known final status.
        """
        if region:
            resolved_region = normalize_serverless_region(region)
            seen_status = False
        else:
            resolved_region, _ = self._find_deployment(deployment_id)
            seen_status = True  # the search just saw it exist
        base_url = serverless_region_url(resolved_region)

        transient = 0
        deadline = time.monotonic() + timeout
        while True:
            try:
                resp = self._t.request("GET", f"management/{deployment_id}/status", base_url=base_url)
                status = resp.get("status")
                transient = 0
                seen_status = True
            except NotFoundError:
                transient += 1
                if transient >= DEPLOYMENT_POLL_MAX_TRANSIENT_ERRORS:
                    if seen_status:
                        # The deployment existed and then vanished mid-poll: it
                        # failed and was cleaned up before we could read why.
                        # Report the failure, not a confusing "not found".
                        raise JarvislabsError(f"Deployment {deployment_id} failed: reason unavailable") from None
                    raise
                time.sleep(DEPLOYMENT_POLL_INTERVAL_S)
                continue
            except APIError as exc:
                if exc.status_code not in DEPLOYMENT_TRANSIENT_STATUS:
                    raise
                transient += 1
                if transient >= DEPLOYMENT_POLL_MAX_TRANSIENT_ERRORS:
                    raise
                time.sleep(DEPLOYMENT_POLL_INTERVAL_S)
                continue

            if status == "running":
                return self._fetch_deployment(deployment_id, resolved_region)
            if status in ("failed", "cleaning"):
                raise JarvislabsError(
                    f"Deployment {deployment_id} {status}: {self._failure_reason(deployment_id, resolved_region)}"
                )
            if status == "deleting":
                raise JarvislabsError(f"Deployment {deployment_id} is being deleted (status: {status}).")

            if time.monotonic() >= deadline:
                raise JarvislabsError(
                    f"Stopped waiting on deployment {deployment_id} after {int(timeout)}s "
                    f"(last status: {status!r}). It may still be coming up — large model "
                    f"downloads can take a while. Check: jl deploy get {deployment_id}"
                )
            time.sleep(DEPLOYMENT_POLL_INTERVAL_S)

    def logs(
        self,
        deployment_id: str,
        *,
        region: str | None = None,
        tail: int = 100,
        follow: bool = True,
        worker: int | None = None,
    ) -> Iterator[tuple[str, str]]:
        """Stream a deployment's logs, live from its running workers.

        Starts with the last ``tail`` lines from each worker (0 = live lines
        only), then follows until you stop iterating; ``follow=False`` ends
        after the replay. ``worker`` narrows to one worker id.

        Yields ``(kind, text)`` pairs: kind ``"log"`` is a worker log line,
        kind ``"notice"`` is a status message about the stream itself.
        """
        if region:
            resolved_region = normalize_serverless_region(region)
        else:
            resolved_region, _ = self._find_deployment(deployment_id)

        params: dict = {"tail": tail, "follow": follow}
        if worker is not None:
            params["worker"] = worker

        lines = self._t.stream_lines(
            f"management/{deployment_id}/logs", params=params, base_url=serverless_region_url(resolved_region)
        )
        for line in lines:
            if line.startswith("data: "):
                yield "log", line[len("data: ") :]
            elif line.startswith(":"):
                notice = line[1:].strip()
                if notice and notice != "ping":
                    yield "notice", notice

    def openai_base_url(self, deployment_id: str, *, region: str | None = None) -> str:
        """OpenAI-compatible base URL for the deployment, ready to paste into any
        OpenAI client's ``base_url`` field."""
        if region:
            resolved_region = normalize_serverless_region(region)
        else:
            resolved_region, _ = self._find_deployment(deployment_id)
        host = serverless_region_url(resolved_region).rstrip("/")
        return f"{host}/openai/{deployment_id}/v1"

    # ── internals ──

    def _find_deployment(self, deployment_id: str) -> tuple[str, Deployment]:
        """Find a deployment when its region isn't known: ask each region,
        return (region, deployment) from the one that has it."""
        unreachable: list[str] = []
        for region in sorted(SERVERLESS_REGIONS):
            try:
                return region, self._fetch_deployment(deployment_id, region)
            except NotFoundError:
                continue
            except APIError as exc:
                if exc.status_code not in DEPLOYMENT_TRANSIENT_STATUS:
                    raise
                unreachable.append(region)

        if unreachable:
            codes = ", ".join(region_code(r) for r in unreachable)
            raise RegionResolutionError(
                f"Deployment {deployment_id} not found; could not check {codes}. Retry, or specify the region."
            )
        raise NotFoundError(f"Deployment {deployment_id} not found in any region.")

    def _fetch_deployment(self, deployment_id: str, region: str) -> Deployment:
        base_url = serverless_region_url(region)
        resp = self._t.request("GET", f"management/{deployment_id}", base_url=base_url)
        return Deployment(**{**resp, "region": region})

    def _fetch_status(self, deployment_id: str, region: str) -> str:
        base_url = serverless_region_url(region)
        resp = self._t.request("GET", f"management/{deployment_id}/status", base_url=base_url)
        return resp.get("status", "")

    def _failure_reason(self, deployment_id: str, region: str) -> str:
        """The deployment's error message, or 'reason unavailable' if it can't be read."""
        try:
            return self._fetch_deployment(deployment_id, region).error_message or "reason unavailable"
        except NotFoundError:
            return "reason unavailable"


def _newest_first_key(summary: DeploymentSummary) -> tuple[bool, datetime]:
    """Sort key: newest start_time first, rows without one last.

    Timestamps may arrive timezone-aware or naive; Python can't compare the
    two, so strip the timezone before comparing.
    """
    start = summary.start_time
    if start is not None and start.tzinfo is not None:
        start = start.replace(tzinfo=None)
    return (start is not None, start or datetime.min)
