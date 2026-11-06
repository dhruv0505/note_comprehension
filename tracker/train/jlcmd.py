"""jlcmd.py — the Jarvis Labs CLI as the trainer's local side calls it.

The real CLI (../jl/bin/jl.exe) unless JL_FAKE=<dir> is set, in which case
fakejl.py stands in: VMs are folders under <dir> and commands run in Git Bash,
so the whole flow can be tested without renting anything.
"""
import hashlib
import json
import os
import shlex
import subprocess
import tempfile
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CHUNK = 8 * 1024 * 1024         # a bigger upload goes in pieces: a long scp from here drops before it finishes
# ssh's own exit codes for a connection that died, not the command's: -1 comes back as 4294967295 through the CLI
CONNECTION_CODES = {-1, 255, 4294967295}
ENV = dict(os.environ, PYTHONPATH=os.path.join(os.path.dirname(HERE), "jl"), PYTHONIOENCODING="utf-8", PYTHONUTF8="1",
           MSYS_NO_PATHCONV="1")
# A HOME set here (Git Bash sets one, a Windows path) reaches the VM's shell and overrides its own, so "$HOME" in a
# remote command came back as C:\Users\...: never passed on. The CLI finds its config without it (USERPROFILE).
ENV.pop("HOME", None)
# jl.exe and the ssh it runs are console programs: started from a process with no console (the autopilot, a
# background bootstrap) each would open a window of its own
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


class JLError(RuntimeError):
    pass


def _first_json(text):
    starts = [i for i in (text.find("{"), text.find("[")) if i >= 0]
    if not starts:
        raise ValueError("no JSON in output")
    return json.JSONDecoder().raw_decode(text[min(starts):])[0]


class JL:
    def __init__(self, fake=None):
        self.fake = fake if fake is not None else os.environ.get("JL_FAKE")
        if self.fake:
            self.base = [sys.executable, os.path.join(HERE, "fakejl.py"), "--root", self.fake]
        else:
            self.base = [os.path.join(os.path.dirname(HERE), "jl", "bin", "jl.exe")]

    def run(self, *args, timeout=600):
        try:
            # stdin from nowhere: a long-running orchestrator started under a hidden console
            # had its ssh children inherit that console as input, and after a while every
            # call hung until its timeout while the same command from a fresh process ran
            return subprocess.run([*self.base, *map(str, args)], capture_output=True, text=True, encoding="utf-8",
                                  errors="replace", env=ENV, timeout=timeout, stdin=subprocess.DEVNULL,
                                  creationflags=NO_WINDOW)
        except subprocess.TimeoutExpired as error:
            raise JLError(f"jl {args[0]} timed out after {timeout}s") from error

    def json(self, *args, timeout=600):
        result = self.run(*args, "--json", timeout=timeout)
        try:
            doc = _first_json(result.stdout)
        except ValueError:
            raise JLError(f"jl {' '.join(map(str, args))}: exit {result.returncode}: "
                          f"{(result.stdout + result.stderr).strip()[-600:]}") from None
        if result.returncode != 0 and not (isinstance(doc, dict) and "exit_code" in doc):
            raise JLError(f"jl {' '.join(map(str, args))}: exit {result.returncode}: {json.dumps(doc)[-600:]}")
        return doc

    # --- account ------------------------------------------------------------
    def balance(self):
        return float(self.json("status", timeout=120)["balance"]["balance"])

    def gpus(self):
        return self.json("gpus", timeout=120)

    # --- instances ----------------------------------------------------------
    def get(self, machine_id):
        return self.json("get", machine_id, timeout=120)

    def list(self):
        return self.json("list", timeout=120)

    def create_gpu_vm(self, gpu, num_gpus, storage, name, region):
        return self.json("create", "--vm", "--gpu", gpu, "--num-gpus", num_gpus, "--storage", storage,
                         "--name", name, "--region", region, "--yes", timeout=2400)

    def resume(self, machine_id, num_gpus=None, vcpus=None, ram=None, gpu=None):
        """A GPU VM may change its GPU type and count; a CPU VM its size (vCPUs and RAM together). Without `gpu` a
        VM comes back with the GPU type it last had."""
        extra = ["--gpu", gpu] if gpu else []
        extra += ["--num-gpus", num_gpus] if num_gpus else []
        if vcpus and ram:
            extra += ["--vcpus", vcpus, "--ram", ram]
        return self.json("resume", machine_id, *extra, "--yes", timeout=2400)

    def pause(self, machine_id, tries=3):
        last = None
        for attempt in range(tries):
            result = self.run("pause", machine_id, "--yes", timeout=600)
            if result.returncode == 0:
                return True
            last = (result.stdout + result.stderr).strip()[-400:]
            time.sleep(10 * (attempt + 1))
        raise JLError(f"pause {machine_id} failed: {last}")

    # --- commands and files -----------------------------------------------------
    def exec(self, machine_id, command, timeout=300, tries=3):
        """(exit code, stdout, stderr) of `sh -lc command` on the VM. The link here drops mid-call often enough to end
        runs, so a connection-level failure (CONNECTION_CODES, or no result at all) is tried again; commands that are
        sent this way are written to stand being repeated."""
        last = ""
        for attempt in range(tries):
            result = self.run("exec", machine_id, "--json", "--", "sh", "-lc", command, timeout=timeout)
            try:
                doc = _first_json(result.stdout)
                code = int(doc["exit_code"])
            except (ValueError, KeyError, TypeError):
                last = f"exit {result.returncode}: {(result.stdout + result.stderr).strip()[-600:]}"
            else:
                if code not in CONNECTION_CODES or attempt == tries - 1:
                    return code, doc.get("stdout") or "", doc.get("stderr") or ""
                last = f"exit {code}: {((doc.get('stdout') or '') + (doc.get('stderr') or '')).strip()[-300:]}"
            time.sleep(5 * (attempt + 1))
        raise JLError(f"exec on {machine_id} gave no result after {tries} tries: {last}")

    def upload(self, machine_id, local, remote, tries=8, verify=True):
        """scp over a link that drops mid-transfer (a 74 MB bundle takes a while): each attempt is checked against the
        local file's size, a half-written remote file is removed before the next one, and the wait grows. Anything
        larger than CHUNK goes in pieces, since a long transfer from here drops every time."""
        size, last = os.path.getsize(local), ""
        if size > CHUNK:
            return self.upload_in_parts(machine_id, local, remote, size, tries)
        for attempt in range(tries):
            result = self.run("upload", machine_id, local, remote, timeout=1800)
            ok = result.returncode == 0 and "error" not in (result.stdout + result.stderr).lower()
            last = (result.stdout + result.stderr).strip()[-400:] or last
            if ok and verify:
                try:
                    code, out, _ = self.exec(machine_id, f"stat -c %s {shlex.quote(remote)} 2>/dev/null || echo none",
                                             timeout=180)
                    there = out.strip().splitlines()[-1].strip() if out.strip() else "none"
                    ok = code == 0 and there == str(size)
                    if not ok:
                        last = f"{there} bytes on the VM, {size} here"
                except JLError as error:
                    ok, last = False, str(error)[-400:]
            if ok:
                return
            try:        # a part-written file would look finished to the next check
                self.exec(machine_id, f"rm -f {shlex.quote(remote)}", timeout=120)
            except JLError:
                pass
            time.sleep(min(60, 10 * (attempt + 1)))
        raise JLError(f"upload {local} -> {remote} failed after {tries} tries: {last}")

    def upload_in_parts(self, machine_id, local, remote, size, tries):
        """The file in CHUNK-sized pieces, each retried on its own, joined on the VM and checked by sha1: one dropped
        connection then costs a piece rather than the whole upload."""
        digest = hashlib.sha1()
        with open(local, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                digest.update(block)
        parts = -(-size // CHUNK)
        quoted = shlex.quote(remote)
        self.exec(machine_id, f"rm -f {quoted} {quoted}.part*", timeout=300)
        with open(local, "rb") as f:
            for part in range(parts):
                data = f.read(CHUNK)
                handle, path = tempfile.mkstemp(prefix="jlpart")
                try:
                    with os.fdopen(handle, "wb") as piece:
                        piece.write(data)
                    self.upload(machine_id, path, f"{remote}.part{part:03d}", tries=tries)
                finally:
                    os.remove(path)
        code, out, err = self.exec(machine_id, f"cat {quoted}.part* > {quoted} && rm -f {quoted}.part* && "
                                               f"sha1sum {quoted}", timeout=900)
        if code != 0 or digest.hexdigest() not in out:
            raise JLError(f"upload {local} -> {remote}: the {parts} pieces did not join into the same file "
                          f"({digest.hexdigest()}): {(out + err).strip()[-300:]}")

    def download(self, machine_id, remote, local, tries=3):
        for attempt in range(tries):
            result = self.run("download", machine_id, remote, local, timeout=1800)
            if result.returncode == 0 and os.path.exists(local):
                return
            time.sleep(5 * (attempt + 1))
        raise JLError(f"download {remote} failed: {(result.stdout + result.stderr).strip()[-400:]}")
