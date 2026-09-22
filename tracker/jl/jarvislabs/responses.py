"""Backend response parsing helpers shared across resources."""

from __future__ import annotations


def _backend_msg(resp: object) -> str:
    """Extract a human-readable error from an API response."""
    if isinstance(resp, dict):
        for key in ("message", "error", "detail"):
            if resp.get(key):
                return str(resp[key])
    return "unexpected error"


def _normalize_success(data: dict) -> bool:
    """Normalize success field from various backend response formats."""
    val = data.get("success") or data.get("sucess")
    if isinstance(val, str):
        return val.lower() == "true"
    return bool(val)


def _declared_failure(resp: object) -> bool:
    """True when a response includes a success flag and it is false (an explicit backend failure).

    A missing flag is not a failure — some endpoints simply omit it — so presence is
    checked before the value.
    """
    return isinstance(resp, dict) and ("success" in resp or "sucess" in resp) and not _normalize_success(resp)
