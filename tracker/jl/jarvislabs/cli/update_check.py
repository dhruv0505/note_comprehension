"""Check for newer versions on PyPI and notify the user.

Non-blocking: reads from a local cache file, refreshes it in a background thread.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

from platformdirs import user_cache_dir

_PKG = "jarvislabs"
_PYPI_URL = f"https://pypi.org/pypi/{_PKG}/json"
_CACHE_DIR = Path(user_cache_dir("jl"))
_CACHE_FILE = _CACHE_DIR / "update_check.json"
_CHECK_TTL = 86_400
_NOTIFY_TTL = 86_400
_TIMEOUT = 3.0


def _read_cache() -> dict:
    """Read cached update data. Returns empty dict on any failure."""
    try:
        return json.loads(_CACHE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_cache(data: dict) -> None:
    """Write cache atomically (temp file + rename). Never raises."""
    tmp = None
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=_CACHE_DIR, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f)
        os.replace(tmp, _CACHE_FILE)
    except Exception:
        if tmp:
            with contextlib.suppress(OSError):
                os.unlink(tmp)


# ── Version parsing ─────────────────────────────────────────────────────────


def _parse_version(v: str) -> tuple[int, ...] | None:
    """Parse 'X.Y.Z' into (X, Y, Z). Returns None for anything unparseable."""
    try:
        parts = tuple(int(x) for x in v.strip().split("."))
        return parts if len(parts) >= 2 else None
    except (ValueError, AttributeError):
        return None


def _get_current_version() -> str:
    from importlib.metadata import version

    return version(_PKG)


# ── PyPI fetch ──────────────────────────────────────────────────────────────


def _refresh_cache() -> None:
    """Fetch latest version from PyPI and update the cache. Runs in a daemon thread."""
    import httpx

    try:
        cache = _read_cache()
        headers = {}
        if cache.get("etag"):
            headers["If-None-Match"] = cache["etag"]

        resp = httpx.get(_PYPI_URL, headers=headers, timeout=_TIMEOUT)

        if resp.status_code == 304:
            cache["checked_at"] = time.time()
            _write_cache(cache)
            return

        if resp.status_code == 200:
            latest = resp.json()["info"]["version"]
            cache["checked_at"] = time.time()
            cache["latest_version"] = latest
            cache["etag"] = resp.headers.get("etag")
            _write_cache(cache)
    except Exception:
        pass


# ── Public API ──────────────────────────────────────────────────────────────

# Module-level state: set by start_update_check(), read by finish_update_check().
_cached_latest: str | None = None
_current: str | None = None


def _should_skip() -> bool:
    """Return True if update checks should be skipped entirely."""
    if os.environ.get("JL_NO_UPDATE_CHECK") or os.environ.get("CI"):
        return True
    return not sys.stderr.isatty()


def _get_upgrade_command() -> str:
    """Detect how jl was installed and return the right upgrade command."""
    prefix = Path(sys.prefix)

    if (prefix / "uv-receipt.toml").is_file():
        return "uv tool upgrade jarvislabs"

    if (prefix / "pipx_metadata.json").is_file():
        return "pipx upgrade jarvislabs"

    from importlib.metadata import distribution

    try:
        installer = (distribution(_PKG).read_text("INSTALLER") or "").strip().lower()
    except Exception:
        installer = ""

    if installer == "uv":
        return "uv pip install --upgrade jarvislabs"
    return "pip install --upgrade jarvislabs"


def start_update_check() -> None:
    """Read cache and spawn background refresh if stale. Call BEFORE app()."""
    global _cached_latest, _current

    if _should_skip():
        return

    try:
        _current = _get_current_version()
        if ".dev" in _current or "+" in _current:
            return

        cache = _read_cache()
        _cached_latest = cache.get("latest_version")

        if time.time() - cache.get("checked_at", 0) >= _CHECK_TTL:
            import threading

            threading.Thread(target=_refresh_cache, daemon=True).start()
    except Exception:
        pass


def finish_update_check() -> None:
    """Show update banner if a newer version is available. Call AFTER app()."""
    try:
        if not _current or not _cached_latest:
            return

        from jarvislabs.cli import state

        if state.json_output:
            return

        current_t = _parse_version(_current)
        latest_t = _parse_version(_cached_latest)
        if not current_t or not latest_t or latest_t <= current_t:
            return

        cache = _read_cache()
        if time.time() - cache.get("notified_at", 0) < _NOTIFY_TTL:
            return

        from jarvislabs.cli.render import console

        console.print(f"\n[yellow]Update available:[/yellow] {_current} → {_cached_latest}")
        console.print(f"Run: [bold]{_get_upgrade_command()}[/bold]")
        console.print("[bold]After upgrading, run:[/bold] jl setup --yes [bold]to update agent skills[/bold]")

        cache["notified_at"] = time.time()
        _write_cache(cache)
    except Exception:
        pass
