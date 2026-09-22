"""Static configuration — region URLs, GPU types, region policy, defaults.

Single source of truth. When a region is added/removed or GPU types change,
only this file needs updating.
"""

# ── Regions ──────────────────────────────────────────────────────────────────

CHENNAI_REGION = "india-chennai-01"
INDIA_NOIDA_REGION = "india-noida-01"
EUROPE_REGION = "europe-01"
DEFAULT_REGION = INDIA_NOIDA_REGION

REGION_URLS: dict[str, str] = {
    CHENNAI_REGION: "https://backendc.jarvislabs.net/",
    INDIA_NOIDA_REGION: "https://backendn.jarvislabs.net/",
    EUROPE_REGION: "https://backendeu.jarvislabs.net/",
}

REGION_DISPLAY_CODES: dict[str, str] = {
    CHENNAI_REGION: "IN1",
    INDIA_NOIDA_REGION: "IN2",
    EUROPE_REGION: "EU1",
}

REGION_CODE_TO_REGION: dict[str, str] = {code.lower(): region for region, code in REGION_DISPLAY_CODES.items()}

# ── Region routing ──────────────────────────────────────────────────────────

# Preferred order for auto-routing when user doesn't specify --region.
# Shared GPUs prefer Noida, then Chennai, then Europe.
REGION_PRIORITY: list[str] = [INDIA_NOIDA_REGION, CHENNAI_REGION, EUROPE_REGION]

# ── Region constraints ────────────────────────────────────────────────────────

EUROPE_GPU_TYPES: frozenset[str] = frozenset({"H100", "H200"})
EUROPE_GPU_COUNTS: frozenset[int] = frozenset({1})

# ── Timeouts & Polling ───────────────────────────────────────────────────────

DEFAULT_POLL_TIMEOUT_S = 600
POLL_INTERVAL_S = 3
FETCH_RETRY_INTERVAL_S = 2
HTTP_TIMEOUT_CONNECT_S = 10
HTTP_TIMEOUT_READ_S = 120
MAX_RETRIES = 3
RETRY_STATUS_CODES: frozenset[int] = frozenset({429, 500, 502, 503, 504})

# ── CLI Defaults ─────────────────────────────────────────────────────────────

DEFAULT_TEMPLATE = "pytorch"
DEFAULT_NUM_GPUS = 1
DEFAULT_STORAGE_GB = 100
DEFAULT_INSTANCE_NAME = "Name me"

# ── Serverless deployments ─────────────────────────────────────────────────────

# Region id -> serverless host URL.
SERVERLESS_REGION_URLS: dict[str, str] = {
    INDIA_NOIDA_REGION: "https://serverlessn.jarvislabs.net/",
    CHENNAI_REGION: "https://serverlessc.jarvislabs.net/",
}

# Reads fan out across these; kept in sync with the URL keys.
SERVERLESS_REGIONS: frozenset[str] = frozenset(SERVERLESS_REGION_URLS)

DEPLOYMENT_POLL_INTERVAL_S = 3
DEPLOYMENT_POLL_MAX_TRANSIENT_ERRORS = 5
# Overall wall-clock cap for wait_until_running (75 min) so an unrecognized
# terminal status can't poll forever. Generous — large model downloads can
# take close to an hour, and giving up early on a healthy deploy is worse
# than a long wait.
DEPLOYMENT_WAIT_TIMEOUT_S = 4500

# Statuses treated as a transient/unreachable region rather than a missing
# deployment (retryable HTTP statuses plus 0 for connection/timeout).
DEPLOYMENT_TRANSIENT_STATUS: frozenset[int] = RETRY_STATUS_CODES | frozenset({0})
