"""tracker_server.py — the tracker: the dataset run and both training runs on one page, served live from this laptop.

    python tracker_server.py            # -> http://127.0.0.1:8765; start_trackers.vbs runs it at login
    python tracker_server.py --port 8767 --guard-port 8768 --synth-root SCRATCH --run-dir ../run_fake --synth-ready   # local test

This is the only thing to start. The process started keeps the server running in a child process and starts
it again if it exits; the server exits to be started afresh when its own code changes, once no button has
been pressed for a few minutes (so a pause or a learning-rate change sent in the background isn't cut short).
The server keeps the three orchestrators running and on their latest code (orchestrators.py), and a button
that needs one gets it started first; a local test, with run folders of its own, leaves them alone.

tracker.html has a tab per run plus Explore (the val page explorer, served by the seg run's endpoints for now),
and every API call names its run with ?job=synth|seg|marks (seg by default):
  synth   synth/synth_api.py, over what synth_orchestrator.py writes
  seg     train/train_api.py, over orchestrator.py's run dir (--run-dir)
  marks   the same, over orchestrator.py --job marks (--marks-run-dir)
"""
import argparse
import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "synth"), os.path.join(HERE, "train")]
import orchestrators  # noqa: E402
import synth_api  # noqa: E402
import train_api  # noqa: E402

PAGE = os.path.join(HERE, "tracker.html")
RUN_DIR = os.path.join(HERE, "train", "runs", "train")
MARKS_RUN_DIR = os.path.join(HERE, "train", "runs", "marks")
RELOAD = 3                      # the server's exit code for "my code changed: start me again"
QUIET_SECONDS = 360             # a code change waits this long after the last button press
# buttons whose request an orchestrator acts on: it is started, or restarted onto new code, before the request is written
NEEDS_ORCHESTRATOR = {("synth", "/api/start"), ("synth", "/api/resume"),("seg", "/api/start"), ("seg", "/api/test"), ("seg", "/api/reset"), ("seg", "/api/save"),
                      ("seg", "/api/explore/resume"), ("seg", "/api/explore/pause"), ("seg", "/api/explore/run"),
                      ("marks", "/api/start"), ("marks", "/api/test"), ("marks", "/api/reset")}
actions = {"running": 0, "last": 0.0}
actions_lock = threading.Lock()


def stamp():
    return time.strftime("%Y-%m-%d %H:%M:%S")


class Server(ThreadingHTTPServer):
    allow_reuse_address = os.name != "nt"       # on Windows it would let a second server share the port


def make_handler(servers, supervisor):
    class Handler(BaseHTTPRequestHandler):
        def send(self, code, body, kind="application/json"):
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def route(self, method):
            url = urllib.parse.urlparse(self.path)
            q = {k: v[0] for k, v in urllib.parse.parse_qs(url.query).items()}
            job = q.get("job", "seg")
            try:
                if method == "GET" and url.path == "/":
                    return self.send(200, open(PAGE, "rb").read(), "text/html; charset=utf-8")
                if method == "GET":
                    reply = synth_api.get(url.path, q) if job == "synth" else train_api.get(servers, url.path, q)
                    if supervisor and url.path == "/api/status" and reply and reply[0] == 200 and job in orchestrators.JOBS:
                        reply[1]["process"] = supervisor.status(job)
                else:
                    length = int(self.headers.get("Content-Length") or 0)
                    body = json.loads(self.rfile.read(length) or b"{}") if length else {}
                    with actions_lock:
                        actions["running"] += 1
                    try:
                        if supervisor and (job, url.path) in NEEDS_ORCHESTRATOR:
                            supervisor.prepare(job)
                        reply = synth_api.post(url.path, body) if job == "synth" else train_api.post(servers, url.path, q, body)
                    finally:
                        with actions_lock:
                            actions["running"] -= 1
                            actions["last"] = time.time()
                self.send(*(reply or (404, {"error": "not found"})))
            except Exception as error:      # noqa: BLE001 - report rather than drop the connection
                self.send(500, {"error": f"{type(error).__name__}: {error}"})

        def do_GET(self):
            self.route("GET")

        def do_POST(self):
            self.route("POST")

        def log_message(self, *args):
            pass
    return Handler


def watch_code():
    """Exits with RELOAD once one of the server's own files changes and no button's work may still be under way."""
    files = {os.path.abspath(m.__file__) for m in list(sys.modules.values()) if getattr(m, "__file__", None)}
    files = {f for f in files if f.startswith(HERE + os.sep)} | {os.path.abspath(synth_api.so.class_tuning.__file__)}

    def mtimes():
        out = {}
        for path in files:
            try:
                out[path] = os.path.getmtime(path)
            except OSError:
                out[path] = None
        return out
    seen = mtimes()
    while True:
        time.sleep(3)
        if mtimes() == seen:
            continue
        with actions_lock:
            if not actions["running"] and time.time() - actions["last"] > QUIET_SECONDS:
                print(f"[{stamp()}] its code changed: restarting the server", flush=True)
                os._exit(RELOAD)


def exit_with_keeper():
    """The keeping process holds this one's stdin open; when it ends, the server ends too."""
    sys.stdin.buffer.read()
    os._exit(0)


def serve(args):
    real = [os.path.abspath(p) for p in (args.synth_root, args.run_dir, args.marks_run_dir)] == [synth_api.HERE, RUN_DIR, MARKS_RUN_DIR]
    synth_api.configure(args.synth_root)
    servers = train_api.make_servers(args)
    supervisor = orchestrators.Supervisor() if real else None
    if supervisor:
        supervisor.start()
    threading.Thread(target=watch_code, daemon=True).start()
    threading.Thread(target=exit_with_keeper, daemon=True).start()
    httpd = Server(("127.0.0.1", args.port), make_handler(servers, supervisor))
    print(f"[{stamp()}] tracker at http://127.0.0.1:{args.port}" + ("" if supervisor else " (orchestrators not managed)"), flush=True)
    httpd.serve_forever()


def keep_serving(args):
    guard = socket.socket()
    try:
        guard.bind(("127.0.0.1", args.guard_port))      # one tracker at a time: the login script may start a second
    except OSError:
        sys.exit("another tracker_server.py is already running")
    command = [sys.executable, "-u", os.path.abspath(__file__), "--serve", *sys.argv[1:]]
    while True:
        started = time.time()
        code = subprocess.Popen(command, stdin=subprocess.PIPE).wait()
        if code != RELOAD:
            print(f"[{stamp()}] the server exited with code {code}; starting it again", flush=True)
            time.sleep(3 if time.time() - started > 60 else 20)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--guard-port", type=int, default=8797, help="port of the one-tracker-at-a-time lock (local test: another)")
    ap.add_argument("--synth-root", default=synth_api.HERE, help="where the dataset run's files live (local test: a scratch folder)")
    ap.add_argument("--run-dir", default=RUN_DIR, help="the segmentation run")
    ap.add_argument("--marks-run-dir", default=MARKS_RUN_DIR, help="the mark separation run")
    ap.add_argument("--synth-ready", action="store_true", help="training treats the dataset as finished (local test)")
    ap.add_argument("--serve", action="store_true", help=argparse.SUPPRESS)     # the child the keeping process runs
    arguments = ap.parse_args()
    serve(arguments) if arguments.serve else keep_serving(arguments)
