"""Typed exception hierarchy for the SDK.

CLI maps these to exit codes + human-readable messages.
SDK users catch specific types without knowing HTTP internals.
"""


class JarvislabsError(Exception):
    """Base for all SDK errors."""


class AuthError(JarvislabsError):
    """401 — invalid or missing API token."""


class NotFoundError(JarvislabsError):
    """404 — instance, SSH key, or resource not found."""


class InsufficientBalanceError(JarvislabsError):
    """403 — not enough balance to perform the action."""


class ValidationError(JarvislabsError):
    """Client-side validation failure."""


class RegionResolutionError(JarvislabsError):
    """A deployment id could not be uniquely resolved to one region:
    a cross-region collision, or zero matches with a region unreachable."""


class SSHError(JarvislabsError):
    """Base error for local SSH command parsing/execution failures."""


class SSHConnectionError(SSHError):
    """SSH transport/connectivity failure."""


class SSHAuthError(SSHError):
    """SSH authentication failure."""


class APIError(JarvislabsError):
    """HTTP error from the backend that doesn't fit a specific category."""

    def __init__(self, status_code: int, message: str, error_code: str | None = None):
        self.status_code = status_code
        self.message = message
        self.error_code = error_code
        super().__init__(message)


def validation_field(loc: list | None) -> str | None:
    """Last loc element that is a usable field name — a non-empty string that
    isn't "body" (skips list indices). None if there's no usable name."""
    for part in reversed(loc or []):
        if isinstance(part, str) and part and part != "body":
            return part
    return None


def extract_message(payload: dict | list) -> str:
    """Turn an error response body into a human-readable string.

    Handles the ``{"message": ...}`` / ``{"error": ...}`` shapes and the
    structured ``{"detail": [{"loc", "msg", "type"}]}`` validation shape.
    """
    if not isinstance(payload, dict):
        return str(payload) or "Unknown error"
    detail = payload.get("detail")
    if isinstance(detail, list):
        msgs = []
        for item in detail:
            if not isinstance(item, dict):
                msgs.append(str(item))
                continue
            msg = str(item.get("msg", ""))
            field = validation_field(item.get("loc"))
            is_pattern = item.get("type") == "string_pattern_mismatch" or "does not match regex" in msg
            if is_pattern and field:
                msgs.append(f"Invalid {field}")  # don't leak the raw regex to users
            elif field and msg:
                msgs.append(f"{field}: {msg}")
            else:
                msgs.append(msg)
        return "; ".join(msgs)
    return payload.get("message") or payload.get("error") or detail or "Unknown error"


def error_from_response(status_code: int, payload: dict | list) -> JarvislabsError:
    """Build the typed exception for a failed HTTP response."""
    message = extract_message(payload) or f"HTTP {status_code}"
    if status_code == 401:
        return AuthError(message)
    # No machine-readable code exists, so balance is detected from the message.
    if status_code == 403 and "balance" in message.lower():
        return InsufficientBalanceError(message)
    if status_code == 404:
        return NotFoundError(message)
    return APIError(status_code, message)
