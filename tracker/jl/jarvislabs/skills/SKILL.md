---
name: jarvislabs
description: Agent guide for running GPU experiments and serverless model deployments with the jl CLI on JarvisLabs.ai.
---

# JarvisLabs GPU CLI (`jl`) — Agent Guide

## Getting Started

Verify auth with `jl status --json` before doing anything. If not logged in, use `jl setup --token <token> --yes`. You can also authenticate via `export JL_API_KEY="..."`.

Use `--help` on any command to discover flags (e.g., `jl run --help`, `jl create --help`). If something goes wrong, diagnose with `jl run logs`/`jl run status`/`jl exec` for managed runs, or `jl deploy get`/`jl deploy logs` for deployments — don't guess.

## Mental Model

- Machine commands (`jl create/list/get/pause/resume/destroy/rename/ssh/exec/upload/download`) = GPU instance lifecycle and access.
- `jl vpc` = private networks for VM instances. Run `jl vpc --help` to discover available commands.
- `jl run` = managed job on an instance. Uploads code, sets up a Python environment, runs your script in the background with log tracking.
- `jl exec` = run any command on an instance. Use for system checks (nvidia-smi, ps, df), debugging failed runs, inspecting files, or any raw shell access. No environment setup, no tracking. This is your escape hatch when `jl run` doesn't cover your use case.
- `jl deploy` (beta) = serverless model serving — an OpenAI-compatible endpoint with autoscaling workers, no instance to manage. Separate from instances and runs.

## Instances

### Creating

```bash
jl create --gpu L4 --storage 100 --yes --json
jl create --gpu L4 --spot --yes --json
jl create --vm --cpu --yes --json
```

`--gpu` is required for GPU instances. Use `--spot` only for GPU containers, not GPU VMs or CPU VMs. CPU VMs are created with `--vm --cpu`; omit `--vcpus/--ram` to use the smallest available CPU plan from the backend. Run `jl create --help` for all available flags.

### Lifecycle

Instances have three states that matter: **Running** (billing active), **Paused** (compute billing stopped, storage billing continues, data persists), **Destroyed** (everything deleted).

```bash
jl pause <id> --yes --json      # stop compute billing, keep data
jl resume <id> --yes --json     # restart a paused instance
jl destroy <id> --yes --json    # permanently delete
```

Resume caveats:
- Resume is **region-locked** — an instance always resumes in its original region. GPU swaps are only possible within that region.
- Resume may return a **new machine_id**. Always use the returned ID for subsequent operations.
- Spot resume is explicit: pass `--spot` when you want a paused GPU container to resume as spot. Without `--spot`, resume is on-demand.
- CPU VM resume uses the CPU VM backend path. Pass both `--vcpus` and `--ram` if you want to change CPU size on resume.
- Run `jl resume --help` for all available flags (GPU swap, storage expansion, rename, etc.).

SSH, exec, upload, and download only work on **Running** instances.

### Regions & GPUs

Valid region codes for new instances: `IN1`, `IN2`, `EU1`.

`IN1` is Chennai (`india-chennai-01`). `IN2` is Noida (`india-noida-01`).

If `--region` is omitted, the CLI picks a region based on GPU availability.

Use `jl resources --json` as the source of truth for GPU, region, and workload availability. VM instances require at least one SSH key and a matching VM-capable row (`workload_type="vm"` or `null`).

Template availability is enforced by the backend, not the CLI. If a template isn't available in your chosen region, the create fails server-side — pick another region or template.

Run `jl gpus` to check GPU availability and pricing. Output shows **GPU Containers** and **GPU VMs** tables with separate availability for each. Spot prices are shown only for GPU containers.

Run `jl cpus` to check CPU VM availability and pricing. Output shows vCPU, RAM, region, and hourly price for each CPU VM size.

Run `jl resources` when you also need CPU VM availability and pricing. It shows GPU containers, GPU VMs, and CPU VMs, with one shared available/unavailable legend at the end.

How to read `jl gpus --json` availability:
- `num_free_devices`: GPUs free right now. Spot creates use this count, and require `spot_price` to be present.
- `effective_num_free_devices`: GPUs available for on-demand creates on that server, including GPUs currently used by spot instances that can be preempted. If missing, fall back to `num_free_devices`.
- These counts are per server. They are not the complete regional GPU capacity.
- `workload_type` tells which launch type the row belongs to:
  - `"container"` means use it for normal GPU container creates.
  - `"vm"` means use it for GPU VM creates.
  - `null` means the same row applies to both containers and VMs.

### Ports & Services

Container instances expose default HTTP ports (each gets its own HTTPS URL):

| Port | Service |
|---|---|
| 8889 | JupyterLab (`url` field, shown as "URL" in `jl get`) |
| 7007 | IDE (`vs_url` field) |
| 6006 | Available on generic templates like pytorch (`endpoints[0]`) |

VM instances (`jl create --gpu ... --vm`) get SSH-only access. VMs require at least one SSH key registered (`jl ssh-key add`). Use `ssh_command` from `jl get <id> --json`.

To expose a service (FastAPI, Gradio, etc.), bind to `0.0.0.0:6006` — it's accessible via `endpoints[0]` on generic templates. Use `--http-ports "7860,8080"` at creation or resume to expose custom ports. Custom port URLs appear in `endpoints` after the default 6006 entry.

Run `jl get <id> --json` to find all service URLs (`url`, `vs_url`, `endpoints`).

## Managed Runs

### How `jl run` works

`jl run` uploads your code to an instance, sets up a Python environment, and runs your script in the background with log and exit code tracking. You need either `--on <machine_id>` (existing instance) or `--gpu <type>` (creates a fresh instance).

`run_id` is tracked locally under `~/.jl/runs/`. All run management commands (`logs`, `status`, `stop`, `list`) depend on these local records. Start and monitor runs from the same machine.

### Run targets

| Target | What happens |
|---|---|
| `train.py` | Uploads to `<home>/train.py`, runs in `<home>/` with shared venv at `$HOME/.venv` |
| `.` or `./project` with `--script train.py` | Rsyncs the directory to `<home>/<dirname>/`, runs inside it with project venv at `<home>/<dirname>/.venv` |
| No target, command after `--` | No upload. Runs from `~`. If `$HOME/.venv` exists (from a previous file run), its `bin/` is prepended to PATH so `python` and `pip` resolve to venv versions. Otherwise uses system Python. |

Only `.py` and `.sh` file targets are supported. For other file types, use a directory target or `jl upload` + `jl exec`. Directory targets require `rsync` installed locally.

**Note:** File targets with the same basename overwrite each other on the remote (e.g., `foo/train.py` and `bar/train.py` both land at `/home/train.py`). Use directory targets for projects with nested structure.

Pass script arguments after `--`:
```bash
jl run train.py --on <id> --json --yes -- --epochs 50 --lr 0.001
```

### Environment & setup

`jl run` manages a Python venv on the remote instance. Template packages (torch, etc.) are inherited via `--system-site-packages` — no need to install them. Venvs persist under the remote home directory across pause/resume.

**Venv locations:**
- **File targets:** shared instance-level venv at `$HOME/.venv`. All file runs share it — deps installed for one script are available to all.
- **Directory targets:** per-project venv at `<home>/<dirname>/.venv`. Isolated per project.
- **Command mode:** no venv is created. If `$HOME/.venv` exists from a previous file run, `python` and `pip` automatically resolve to it via PATH prepend.

**How dependencies get installed:**

- **Directory targets** — auto-detected. If your directory has `requirements.txt` or `pyproject.toml` (with `[project]`), deps are installed automatically. No flag needed.
- **File targets** — no auto-detection. Pass `--requirements requirements.txt` if you need extra packages.
- **`--requirements <file>`** — overrides auto-detection. Uploads and installs the specified file instead.
- **`--setup <command>`** — runs a shell command before your script (e.g., `--setup "pip install flash-attn"`). Runs inside the venv for file/dir targets, raw for command mode.

```bash
# Directory — auto-detects requirements.txt
jl run . --script train.py --on <id> --json --yes

# Single file — pass requirements explicitly
jl run train.py --on <id> --requirements requirements.txt --json --yes

# Extra setup command
jl run . --script train.py --on <id> --setup "pip install flash-attn" --json --yes
```

**Command mode** — when you pass a raw command after `--` with no file or directory target. Useful when code already exists on the instance (e.g., uploaded via `jl upload`, written via `jl exec`, or left by a previous run). If `$HOME/.venv` exists from a prior file run, its `bin/` is prepended to PATH so `python` and `pip` resolve to venv versions. You still get `jl run` log tracking (`logs`, `status`, `stop`), which is the main advantage over `jl exec`. `--requirements` is not supported in command mode.

**Important:** Command mode runs from `~` (the remote shell home). Use absolute paths or `cd` explicitly for scripts in specific directories.

```bash
jl run --on <id> --json --yes -- python3 /home/train.py
jl run --on <id> --json --yes -- sh -lc 'cd /home && torchrun --nproc_per_node=2 train.py'
```

### Running on an existing instance

```bash
jl run train.py --on <machine_id> --json --yes
jl run . --script train.py --on <machine_id> --requirements requirements.txt --json --yes
```

Lifecycle flags (`--keep`, `--pause`, `--destroy`) are not allowed with `--on` — the instance is not touched after the run.

### Running on a fresh instance

```bash
jl run . --script train.py --gpu L4 --keep --json --yes
jl run . --script train.py --gpu L4 --spot --keep --json --yes
```

Creates a new instance, uploads code, runs the script. Additional flags: `--spot` (fresh GPU containers only), `--vm` (VM instead of container, disallows `--template` and `--http-ports`), `--template` (default: pytorch; run `jl templates --json` to list available), `--storage` (default: 100GB), `--num-gpus` (default: 1), `--name`/`-n` (default: jl-run), `--region` (IN1/IN2/EU1), `--http-ports`.

**Lifecycle rules for fresh instances:**

- **With `--json` or `--no-follow`:** `--keep` is required. The CLI rejects `--pause` and `--destroy` because it returns immediately and cannot apply lifecycle actions later. Use `--keep` and have the agent pause or destroy the instance after the run completes.
- **Without `--json` or `--no-follow` (human mode):** the CLI stays attached, streams logs, and applies lifecycle when the run finishes. Default lifecycle is `--pause`.

Use separate `jl create` when you need to inspect GPU availability, reuse machines across runs, or attach filesystems/scripts.

## Monitoring & Control

### Reading logs

The primary monitoring command:

```bash
jl run logs <run_id> --tail 50
```

Always use `--tail N` — without it, the entire log file is returned and can be enormous.

The output includes a **header** and **footer** with run state (in non-follow, non-JSON mode):

```
--- run r_abc | machine 123 | running ---

step=100 loss=2.31
step=200 loss=2.11

--- still running | log: <home>/jl-runs/r_abc/output.log ---
```

When done, the footer shows the final state:
```
--- succeeded | exit code: 0 | log: <home>/jl-runs/r_abc/output.log ---
```

Or on failure:
```
--- failed | exit code: 1 | log: <home>/jl-runs/r_abc/output.log ---
```

The log path is `<home>/jl-runs/<run_id>/output.log` (`<home>` = `/home/` on containers, `/home/<user>/` on VMs).

If the instance is paused, missing, or SSH is unavailable, `jl run logs` fails before printing any output. Use `jl run status <run_id> --json` to check those states.

### The agent monitoring loop

1. **Start detached:** `jl run ... --json --yes` — extract `run_id` and `machine_id` from JSON
2. **Early check (catch fast failures):** `sleep 15 && jl run logs <run_id> --tail 30` — if footer says `failed`, fix and retry immediately
3. **Steady-state polling:** `sleep 120 && jl run logs <run_id> --tail 50`
4. Read log body for loss values, errors, or progress
5. Check footer:
   - `still running` → repeat step 3
   - `succeeded | exit code: 0` → download results
   - `failed | exit code: N` → read error, fix, start a new run

Cadence: 60-120s (short experiments), 180-300s (long training), 300-600s (very long runs).

### Checking status

```bash
jl run status <run_id> --json
```

Returns run state, machine_id, exit_code, lifecycle_policy, launch_command, and more — always live, no flags needed. Separately, `jl run list` shows state as `"saved"` until you pass `--refresh` (or `--status`).

### Stopping a run

```bash
jl run stop <run_id> --json
```

Kills the entire process group (training script + all child processes). Escalates to SIGKILL if the process doesn't exit after TERM.

### System checks via exec

```bash
jl exec <id> -- nvidia-smi
jl exec <id> -- ps -ef
jl exec <id> -- df -h
```

Prefer raw output for `jl exec` and `jl run logs` — easier to read and parse. Use `--json` when you need machine-readable state: `create`, `get`, `list`, `run start`, `run status`.

Exit code of the remote command is propagated. For pipes or shell syntax, wrap in `sh -lc`:

```bash
jl exec <id> -- sh -lc 'grep "loss" /path/to/log | tail -5'
```

## File Transfer & Persistence

### Upload and download

```bash
jl upload <id> ./local /remote          # upload file or directory
jl download <id> /remote ./local         # download file
jl download <id> /remote ./local -r      # download directory
```

Default destinations: upload without dest → remote home directory. Download without dest → `./<basename>` in current local directory.

### What persists across pause/resume

The remote home directory (`/home/` on containers, `/home/<user>/` on VMs) persists across pause/resume:

- Files and directories under the home directory
- `$HOME/.venv` (file runs) and `<project>/.venv` (directory runs)
- Attached filesystems (mounted at `/home/jl_fs/`)
- Run metadata under `<home>/jl-runs/<run_id>/`

On **containers**, anything outside the home directory is ephemeral — system installs (`apt-get`, global pip) and files in `/tmp`, `/root`, etc. are lost on pause. VMs are full machines and keep their whole disk. To re-run system installs on each container launch, pass `--setup "<cmd>"` to `jl run`, or attach a startup script (`jl scripts add` + `--script-id`). Venv packages persist either way.

### Remote file paths

`<home>` is `/home/` on containers, `/home/<user>/` on VMs.

- Uploaded files (via `jl run`): `<home>/<filename>` (e.g., `train.py` → `/home/train.py`)
- Uploaded directories (via `jl run`): `<home>/<directory_name>/`
- Uploaded files (via `jl upload`): `<home>/<filename>`
- Shared venv (file runs): `<home>/.venv/`
- Project venv (directory runs): `<home>/<directory_name>/.venv/`
- Run metadata: `<home>/jl-runs/<run_id>/`

### Filesystems & supporting commands

Attach a filesystem at creation with `--fs-id <id>`. Attach a startup script with `--script-id <id>` (and `--script-args`). These flags work on both `jl create` and `jl resume`.

```bash
jl templates --json                    # list available templates
jl ssh-key list --json                 # list registered SSH keys
jl ssh-key add <pubkey-file> --name x  # add SSH key (required for VMs)
jl scripts list --json                 # list startup scripts
jl filesystem list --json              # list filesystems
jl filesystem create --name x --storage 100 --yes --json  # create filesystem
```

**Filesystem caveats:**
- **Region-bound:** A filesystem can only attach to instances in the same region.
- **ID changes on edit:** Expanding a filesystem (`jl filesystem edit`) may return a new `fs_id`. Always use the returned ID.
- The CLI validates that `fs_id` exists and belongs to the same region before creating/resuming.

## Serverless Deployments (Beta)

`jl deploy` = beta serverless model serving: an OpenAI-compatible endpoint with autoscaling workers, **no instance to manage** (no SSH/exec, no pause/resume). You create, call the endpoint, and `delete` the deployment when done.

Subcommands: `create`, `list`, `get`, `status`, `logs`, `update`, `delete`. Run `jl deploy <cmd> --help` for exact flags.

**Regions:** serverless is `IN1`/`IN2` only (not `EU1`). `create` requires `--region`. Per-id commands (`get`/`status`/`logs`/`update`/`delete`) search both regions if `--region` is omitted — pass it to skip the search.

### Creating

```bash
jl deploy create \
  --name qwen --region IN2 --framework vllm --gpu L4 \
  --gpus-per-worker 1 --min-workers 0 --max-workers 2 \
  --idle-timeout 600 --wait-time 60 --storage 50 \
  --model Qwen/Qwen3-0.6B \
  --detach --yes --json
```

- **Async.** Without `--detach`, `create` blocks and polls until `running`. **For agents, pass `--detach`** to get the id immediately, then poll `jl deploy status <id> --json`. The id prints as soon as it's assigned; once printed, Ctrl-C only detaches from the wait — the deployment keeps building.
- Repeatable `--arg key=value` (framework args, e.g. `--arg max-model-len=8192`) and `--env KEY=value` (e.g. `--env HF_TOKEN=...`).
- `--json` requires `--yes`.

### OpenAI-compatible inference

A running deployment exposes an OpenAI-compatible base URL (`openai_base_url` from `jl deploy get <id> --json`, present only when `running`), authed with your JarvisLabs API key. **Use the served model name** — if you set `--arg served-model-name=<x>`, that is the `model` field, not the `--model` id. `create` prints a paste-ready snippet; `get` shows the base URL.

Once `running`, send a real inference request to confirm the model responds as expected before relying on the endpoint. With `--min-workers 0`, the first request may be slow while a worker starts; use a generous first-call timeout.

### Inspecting & managing

```bash
jl deploy status <id> --json            # lightweight lifecycle status (poll this)
jl deploy list                          # compact table with cost
jl deploy list --wide                   # + framework, workers, concurrency
jl deploy get <id> --json               # full detail: workers, cost, error, base URL
jl deploy logs <id> --tail 100 --no-follow
jl deploy update <id> --idle-timeout 900 --json
jl deploy delete <id> --yes --json
```

- Poll until `status` is `running`; `failed`/`cleaning`/`deleting` are terminal. On failure with no live workers, read the reason from `jl deploy get <id>`.
- **`jl deploy logs` defaults to `--follow` (blocks) and rejects `--json`** — pass `--no-follow` for a one-shot read. Logs stream live from workers and are not stored.
- `jl deploy update` only changes `name`/`idle-timeout`/`wait-time`, and only while running. To change GPU/model/workers/storage, recreate.
- Deployments accrue cost while active (compute + storage), so `delete` when done.

## Agent Workflow (End-to-End)

```bash
# 1. Check GPUs and create instance
jl gpus --json
jl create --gpu L4 --storage 100 --yes --json

# 2. Start detached run
jl run . --script train.py --on <machine_id> --requirements requirements.txt --json --yes

# 3. Early check (catch import/syntax/pip failures fast)
sleep 15 && jl run logs <run_id> --tail 30

# 4. Steady-state monitoring (repeat until footer shows succeeded or failed)
sleep 120 && jl run logs <run_id> --tail 50

# 5. Download results (use /home/<user>/ for VMs instead of /home/)
jl download <machine_id> /home/results ./results -r

# 6. Cleanup
jl pause <machine_id> --yes --json
```

For fresh instances without a pre-created instance:

```bash
# Creates instance inline, runs detached — agent must clean up after
jl run . --script train.py --gpu L4 --keep --json --yes
# ... monitor with jl run logs ...
jl pause <machine_id> --yes --json
```

## Error Handling

When `--json` is active, CLI validation and API failures are emitted as `{"error": "..."}` to stdout.

**Confirm-gated commands require `--yes` with `--json`.** Commands that prompt for confirmation (`create`, `pause`, `resume`, `destroy`, `rename`, `filesystem` mutations, `deploy create`/`delete`, fresh-instance `jl run`) cannot prompt in `--json` mode — a missing `--yes` is a hard error (`{"error": "--json requires --yes"}`, exit 1), not a silent skip. Always pair `--yes` with `--json` on these.

Not all non-zero exits use that shape. `jl exec --json` returns its own structured payload with `stdout`, `stderr`, and `exit_code` fields.

Agent rule:
- First inspect the JSON shape
- If it has an `error` key, treat it as a CLI failure
- Otherwise inspect command-specific fields (`exit_code`, `state`, `run_exit_code`)

## Anti-Patterns

- Avoid `jl run logs --follow` for agent polling — it blocks. Use bounded `--tail N` polling instead (humans can use `--follow` interactively). `--json` is incompatible with `--follow`.
- `jl deploy logs` defaults to `--follow` (blocks) and rejects `--json` — pass `--no-follow` for a one-shot read.
- Always use `--json` when starting runs — it returns immediately. Without `--json`, the CLI streams logs and blocks.
- Do not read full logs without `--tail N` — can return megabytes of output.
- Do not poll every few seconds — use 60-600s intervals based on expected run duration.
- Do not use lifecycle flags (`--keep`, `--pause`, `--destroy`) with `--on` — they are rejected. Only for fresh instances.
- Do not use `--pause` or `--destroy` with `--json` for fresh instances — rejected. Use `--keep --json` and clean up yourself.
- Do not use `jl exec` for long-running tasks — it blocks until the command finishes. Use `jl run` which runs in the background with log tracking.
- Do not trust `jl run list` without `--refresh` — state shows as `"saved"` (stale). Use `--refresh` or `--status` for live state.
- Do not assume `machine_id` is stable after `jl resume` — it may return a new ID. Always use the returned ID.
- Do not forget to pause/destroy instances after experiments — they cost money.

## Command Discovery

Every command supports `--help` for full flag details:

```bash
jl create --help       jl run --help       jl ssh-key --help
jl resume --help       jl run logs --help  jl filesystem --help
jl deploy --help       jl deploy create --help
```
