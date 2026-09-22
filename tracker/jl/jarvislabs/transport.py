"""HTTP transport layer — auth, timeouts, retries, error mapping.

Every API call goes through Transport. client.py never touches httpx directly.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from collections.abc import Iterator

from jarvislabs.constants import (
    DEFAULT_REGION,
    HTTP_TIMEOUT_CONNECT_S,
    HTTP_TIMEOUT_READ_S,
    MAX_RETRIES,
    REGION_URLS,
    RETRY_STATUS_CODES,
)
from jarvislabs.exceptions import APIError, error_from_response

# Only idempotent methods are safe to retry: replaying a POST/PUT/DELETE after a
# timeout can trigger a duplicate operation on the backend.
SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})


def _api_error(resp: httpx.Response) -> APIError:
    """Map a non-success response to a typed exception."""
    try:
        payload = resp.json()
    except ValueError:
        payload = {}
    return error_from_response(resp.status_code, payload)


class Transport:
    """Thin httpx wrapper with auth, retries, and error mapping."""

    def __init__(self, token: str, base_url: str | None = None) -> None:
        self._base_url = base_url or REGION_URLS[DEFAULT_REGION]
        self._client = httpx.Client(
            timeout=httpx.Timeout(
                connect=HTTP_TIMEOUT_CONNECT_S,
                read=HTTP_TIMEOUT_READ_S,
                write=HTTP_TIMEOUT_READ_S,
                pool=HTTP_TIMEOUT_READ_S,
            ),
            headers={"Authorization": f"Bearer {token}"},
        )

    def request(
        self,
        method: str,
        path: str,
        *,
        json: dict | None = None,
        params: dict | None = None,
        files: dict | None = None,
        base_url: str | None = None,
    ) -> dict | list:
        """Send an HTTP request, retrying idempotent calls on transient failures."""
        url = (base_url or self._base_url).rstrip("/") + "/" + path.lstrip("/")
        can_retry = method.upper() in SAFE_METHODS

        for attempt in range(MAX_RETRIES + 1):
            retriable_attempt = can_retry and attempt < MAX_RETRIES
            try:
                resp = self._client.request(method, url, json=json, params=params, files=files)
            except httpx.HTTPError as exc:
                if retriable_attempt:
                    time.sleep(2**attempt)
                    continue
                label = "timed out" if isinstance(exc, httpx.TimeoutException) else "Connection failed"
                raise APIError(0, f"Request {label}: {method} {path}") from exc

            if resp.status_code in RETRY_STATUS_CODES and retriable_attempt:
                time.sleep(2**attempt)  # 1s, 2s, 4s
                continue

            if not resp.is_success:
                raise _api_error(resp)
            return resp.json()

        # The loop always returns or raises on the final attempt; this line only
        # keeps the type checker happy.
        raise AssertionError("unreachable")

    def stream_lines(self, path: str, *, params: dict | None = None, base_url: str | None = None) -> Iterator[str]:
        """Open a streaming GET and yield response lines as they arrive.

        No retries: a stream is long-lived and replaying it would repeat
        lines. The connection closes when the caller stops iterating.
        """
        url = (base_url or self._base_url).rstrip("/") + "/" + path.lstrip("/")
        try:
            with self._client.stream("GET", url, params=params) as resp:
                if not resp.is_success:
                    resp.read()  # streamed bodies aren't loaded until asked
                    raise _api_error(resp)
                yield from resp.iter_lines()
        except httpx.HTTPError as exc:
            label = "timed out" if isinstance(exc, httpx.TimeoutException) else "failed"
            raise APIError(0, f"Stream {label}: GET {path}") from exc

    def close(self) -> None:
        self._client.close()
