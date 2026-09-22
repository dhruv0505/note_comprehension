"""Canonical region helpers.

The single home for translating between internal region ids (e.g. "india-noida-01"),
user-facing display codes (e.g. "IN2"), and backend base URLs. Region *data* lives in
``constants.py``; this module is the logic that operates on it. (Deployment routing —
which serverless host owns a given deployment id — lives in ``deployments.py``.)
"""

from __future__ import annotations

from jarvislabs.constants import (
    DEFAULT_REGION,
    REGION_CODE_TO_REGION,
    REGION_DISPLAY_CODES,
    REGION_URLS,
    SERVERLESS_REGION_URLS,
    SERVERLESS_REGIONS,
)
from jarvislabs.exceptions import ValidationError


def region_code(region: str | None) -> str | None:
    """Internal region id -> user-facing display code (e.g. "india-noida-01" -> "IN2"). None passes through."""
    return REGION_DISPLAY_CODES.get(region, region)


def format_codes() -> str:
    """Valid region codes in product order: IN1, IN2, EU1."""
    return ", ".join(region_code(region) for region in REGION_DISPLAY_CODES)


def region_base_url(region: str | None) -> str:
    """Backend base URL for an internal region id (defaults to the default region)."""
    if region is None:
        return REGION_URLS[DEFAULT_REGION]
    try:
        return REGION_URLS[region]
    except KeyError as exc:
        raise ValidationError(f"Unknown region {region!r}. Use one of: {format_codes()}") from exc


def normalize_region(region: str | None) -> str | None:
    """Validate a user-supplied region (display code or internal id) and return the internal id.

    Returns None for a None/blank input. Raises ValidationError for an unknown region.
    """
    if region is None:
        return None

    normalized = region.strip()
    if not normalized:
        return None

    if normalized in REGION_DISPLAY_CODES:
        return normalized

    upper = normalized.upper()
    if upper in REGION_DISPLAY_CODES.values():
        return REGION_CODE_TO_REGION[upper.lower()]

    raise ValidationError(f"Unknown region {region!r}. Use one of: {format_codes()}")


# ── Serverless ───────────────────────────────────────────────────────────────
# Serverless has its own region set and per-region URLs, and there is no
# cross-region lookup — finding a deployment means asking each region.
# Everything below is serverless-specific.


def serverless_region_url(region: str) -> str:
    """Serverless host URL for an internal region id."""
    try:
        return SERVERLESS_REGION_URLS[region]
    except KeyError:
        raise ValidationError(f"No serverless host for region {region_code(region)}.") from None


def normalize_serverless_region(region: str) -> str:
    """Accept a display code (IN2) or internal id (case-insensitive) and return
    the internal region id, validated against the serverless regions."""
    key = region.strip().lower()
    resolved = REGION_CODE_TO_REGION.get(key, key)
    if resolved not in SERVERLESS_REGIONS:
        codes = ", ".join(sorted(region_code(r) for r in SERVERLESS_REGIONS))
        raise ValidationError(f"Region {region_code(resolved)} is not a valid serverless region. Use one of: {codes}.")
    return resolved
