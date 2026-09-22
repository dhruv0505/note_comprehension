"""train_api.py — the tracker's training tabs; ../tracker_server.py serves them.

Two jobs, each with its own run dir and orchestrator.py: the segmentation run
(--run-dir, orchestrator.py) and the mark separation CNN (--marks-run-dir,
orchestrator.py --job marks). Every API call takes ?job=seg|marks, seg by default.

Reads what orchestrator.py writes (state.json, log.jsonl, gpu.jsonl, files/) and
turns the page's buttons into request files for it:
  Start          START.json (once; needs the dataset finished and no run begun). A seg run also needs a mark
                 separator (mark_model): the one picked in the mark separation tab from note_synth/inference/
                 mark_separation/, or else the mark separation run's model copied back. It is snapshotted into
                 mark_model/ and goes to the GPU VM with the code, for its recognition to take marks out with. A seg
                 run can pick its LoRA rank, a model in note_synth/inference/comprehension/ to train further from
                 (snapshotted into init_model/ now, so the model's folder is never touched) or, after a restart that
                 kept one, the previous run's checkpoint (continue_run), and a checkpoint to save to inference/ at the end
  Budget         BUDGET.json: the run's budget from now on, whatever phase it is in
  Save           seg: save/requests/<id>.json, best/ or last/ into note_synth/inference/comprehension/<name> with its
                 val stats. marks: best/ or last/ into note_synth/inference/mark_separation/<name> straight away, from
                 the copy taken when training ended
  Mark separator mark_separator.json beside the run dirs (marks): the saved mark separator Florence Tuning's next
                 Start takes instead of the mark separation run's own, so that run needn't train first
  Stop           KILL, and STOP on the GPU VM straight away; "pause now" pauses without waiting
  Learning rate  control.json on the GPU VM straight away (jl exec), logged in actions.jsonl
  Run on test    TEST.json, created exclusively: a second request is refused
  Restart        RESET.json, once the run has ended, stopped or failed: the orchestrator archives the
                 run and waits for a new Start. {"continue_from": "best" | "last"} (seg) keeps that checkpoint
                 on the GPU VM for the new run to continue from
If the orchestrator stops answering, Stop pauses the VMs itself.
"""
import json
import math
import os
import re
import shlex
import shutil
import threading
import time
import urllib.parse

import inference_models
from jlcmd import JL, JLError

HERE = os.path.dirname(os.path.abspath(__file__))
TAG = re.compile(r"samples_(step_\d+)_(rows|pages|marks)\.jsonl$")
ALIVE_SECONDS = 45
LOCK = threading.Lock()
SAMPLE_KINDS = {"seg": ("pages",), "marks": ("marks",)}       # seg: the cascade's only; rows keep none
ACCOUNT = {"at": 0, "doc": None}        # balance and GPU offers, shared by both jobs
RESETTABLE = ("trained", "done", "stopped", "failed")


def load(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


class Server:
    def __init__(self, args, job="seg", run_dir=None):
        self.args = args
        self.job = job
        self.dir = os.path.abspath(run_dir or args.run_dir)
        self.marks_dir = os.path.abspath(args.marks_run_dir)
        self.files = os.path.realpath(os.path.join(self.dir, "files"))
        os.makedirs(self.files, exist_ok=True)
        self.jl = JL()
        self.kill = {"requested_at": None, "stop_sent": None, "forced": None}

    def path(self, *parts):
        return os.path.join(self.dir, *parts)

    # --- reads ------------------------------------------------------------------------
    def synth(self):
        if self.args.synth_ready:
            return {"ready": True, "detail": "local test", "train_rows": 552, "train_recognition": 0, "train_marks": 40,
                    "val_marks": 12}
        synth_dir = os.path.join(os.path.dirname(HERE), "synth")
        watch = load(os.path.join(synth_dir, "run", "status.json")) or {}
        if (load(os.path.join(synth_dir, "run", "run_meta.json")) or {}).get("mode") in ("sample", "recognition", "additions"):
            # a sample run leaves the dataset alone, and a recognition run only its recognition/ (the last full run's
            # status takes its counts when it finishes): whether it is ready is the last full run's to say
            watch = load(os.path.join(synth_dir, "last_full_status.json")) or {}
        status = watch.get("status") or {}
        splits = status.get("splits") or {}
        done = sum(s.get("pages", 0) for s in splits.values())
        target = sum((s.get("target") or 0) for s in splits.values())
        ready = bool(watch.get("paused")) and watch.get("pause_reason") == "finished"
        recognition = [*((splits.get("train") or {}).get("recognition") or {}).values(),
                       *((splits.get("train") or {}).get("recognition_pool") or {}).values()]    # pages' and the pool's
        return {"ready": ready, "paused": watch.get("paused"), "reason": watch.get("pause_reason"),
                "train_recognition": sum((c or {}).get("samples") or 0 for c in recognition),
                "pages": done, "target": target, "train_rows": (splits.get("train") or {}).get("rows"),
                "val_pages": (splits.get("val") or {}).get("pages"),
                "train_marks": (splits.get("train") or {}).get("marks"), "val_marks": (splits.get("val") or {}).get("marks"),
                "test_marks": (splits.get("test") or {}).get("marks"),
                "detail": "dataset finished" if ready else (f"synthesis {100 * done / target:.0f}% done"
                                                            if target else "synthesis not reporting yet")}

    def mark_choice_path(self):
        return os.path.join(os.path.dirname(self.marks_dir), "mark_separator.json")

    def mark_model(self):
        """The mark separator the segmentation run needs before it can start (its recognition takes marks out with
        it): {ready, detail, source, name, checkpoint, step, mean_f1, run_id, path}. source "inference": the saved one
        picked in the mark separation tab (mark_separator.json); source "run": the mark separation run's model, best/
        once copied back, else last/."""
        choice = load(self.mark_choice_path())
        if choice:
            name = str(choice.get("name") or "")
            model = inference_models.get(name, "mark_separation")
            if model is None:
                return {"ready": False, "source": "inference", "name": name,
                        "detail": f"inference/mark_separation/{name} is gone; pick another mark separator in the mark separation tab"}
            info = model["info"]
            return {"ready": True, "source": "inference", "name": name, "checkpoint": info.get("checkpoint"),
                    "step": info.get("step"), "mean_f1": info.get("val_mean_f1"), "signature": model["signature"],
                    "path": os.path.join(model["folder"], "model.pt"), "detail": f"inference/mark_separation/{name}"}
        state = load(os.path.join(self.marks_dir, "state.json")) or {}
        phase = state.get("phase") or "waiting_for_start"
        local = state.get("checkpoints_local") or {}
        for name in ("best", "last"):
            path = os.path.join(self.marks_dir, "files", "train", "run", name, "model.pt")
            if name in local and os.path.isfile(path):
                entry = local[name]
                return {"ready": True, "source": "run", "checkpoint": name, "step": entry.get("step"),
                        "mean_f1": entry.get("mean_f1"), "run_id": state.get("run_id"), "path": path,
                        "detail": f"mark separation {name}/, step {entry.get('step')}"}
        if self.args.synth_ready:           # local test: the segmentation run starts without one
            return {"ready": True, "source": None, "checkpoint": None, "detail": "local test: no mark separation model"}
        fetch = state.get("mark_model_fetch") or {}
        if phase in ("trained", "done") and (state.get("remote") or {}).get("checkpoints"):
            detail = ("its model could not be copied back from the GPU VM" if fetch.get("done") is False
                      else "its model is being copied back from the GPU VM")
        elif phase in ("provisioning", "training", "stopping", "testing"):
            progress = (state.get("remote") or {}).get("progress") or {}
            detail = (f"mark separation is training, step {progress['step']} of {progress.get('total_steps')}"
                      if progress.get("step") is not None else "mark separation is starting")
        elif phase in ("stopped", "failed", "trained", "done"):
            detail = f"the mark separation run {phase} without a model; restart it"
        else:
            detail = "mark separation hasn't been trained yet"
        return {"ready": False, "source": "run", "detail": detail}

    @staticmethod
    def public(models):
        return [{k: v for k, v in m.items() if k != "folder"} for m in models]

    def state(self):
        state = load(self.path("state.json")) or {}
        alive = bool(state) and time.time() - state.get("heartbeat_at", 0) < ALIVE_SECONDS
        actions = []
        if os.path.exists(self.path("actions.jsonl")):
            actions = [json.loads(l) for l in open(self.path("actions.jsonl")) if l.strip()][-30:]
        validations = {}
        run_files = self.run_files_dir(state)
        if run_files and os.path.isdir(run_files):
            for name in os.listdir(run_files):
                m = TAG.match(name)
                if m:
                    entry = validations.setdefault(m.group(1), {"tag": m.group(1), "step": int(m.group(1)[5:])})
                    entry[m.group(2)] = True
        return {**state, "served_at": time.time(), "orchestrator_alive": alive, "synth": self.synth(),
                "mark_model": self.mark_model(),
                "saved_mark_models": self.public(inference_models.find("mark_separation")) if self.job == "marks" else None,
                "actions": actions, "kill": self.kill,
                "locks": {"start": os.path.exists(self.path("START.json")), "test": os.path.exists(self.path("TEST.json")),
                          "reset": os.path.exists(self.path("RESET.json"))},
                "validations": sorted(validations.values(), key=lambda v: v["step"])}

    def run_files_dir(self, state=None):
        return os.path.join(self.files, "train", "run")

    def local_file(self, remote_path, state):
        home = ((state or {}).get("remote") or {}).get("home")
        if not home:
            return None
        norm = lambda p: p.replace("\\", "/").rstrip("/")
        p, h = norm(remote_path), norm(home)
        if not p.lower().startswith(h.lower() + "/"):
            return None
        return p[len(h) + 1:]

    def records(self, name, since):
        path = self.path(name)
        out, index = [], 0
        if os.path.exists(path):
            with open(path) as f:
                for index, line in enumerate(f, 1):
                    if index > since and line.strip():
                        try:
                            out.append(json.loads(line))
                        except ValueError:
                            pass
        return {"records": out, "next": max(since, index)}

    def file_url(self, rel):
        job = "" if self.job == "seg" else f"job={self.job}&"
        return f"/file?{job}p=" + urllib.parse.quote(rel)

    def samples(self, tag, kind):
        state = load(self.path("state.json")) or {}
        if kind not in SAMPLE_KINDS[self.job] or not re.fullmatch(r"step_\d+", tag):
            return []                   # validation samples only; test content is never served
        path = os.path.join(self.run_files_dir(), f"samples_{tag}_{kind}.jsonl")
        rows = [json.loads(l) for l in open(path) if l.strip()] if os.path.exists(path) else []
        out = []
        for row in rows:
            rel = self.local_file(row.get("image", ""), state)
            item = {**{k: v for k, v in row.items() if k != "raw"}, "image_url": None}
            if rel and os.path.exists(os.path.join(self.files, rel)):
                item["image_url"] = self.file_url(rel)
            if "predicted" in row:
                truth = load(os.path.join(self.files, os.path.splitext(rel)[0] + ".json")) if rel else None
                if truth:
                    item["truth"] = [{k: o[k] for k in ("class", "bbox", "position", "points", "start", "end", "text")
                                      if k in o}
                                     for o in truth.get("objects", [])]
                    item["image_size"] = truth.get("image_size")
            item["name"] = os.path.basename(row.get("image", ""))
            out.append(item)
        return out

    def account_doc(self):
        if time.time() - ACCOUNT["at"] > 120 or ACCOUNT["doc"] is None:
            try:
                vms = [g for g in self.jl.gpus() if g["region"] == "IN2" and g.get("workload_type") == "vm"]
                h100 = [g for g in vms if g["gpu_type"] == "H100"]
                by_type = {}
                for g in vms:           # the dataset VM's private network is in IN2, so only IN2 VMs can reach it
                    best = by_type.get(g["gpu_type"])
                    if best is None or g["num_free_devices"] > best["free"]:
                        by_type[g["gpu_type"]] = {"gpu_type": g["gpu_type"], "price": g["price_per_hour"],
                                                  "free": g["num_free_devices"], "vram": g.get("vram")}
                ACCOUNT.update(at=time.time(), doc={
                    "balance": self.jl.balance(),
                    "h100": {"price": h100[0]["price_per_hour"], "free": h100[0]["num_free_devices"]} if h100 else None,
                    "offers": sorted(by_type.values(), key=lambda o: o["price"])})
            except (JLError, KeyError, IndexError) as error:
                return {"error": str(error)}
        return ACCOUNT["doc"]

    # --- actions ------------------------------------------------------------------------
    def record_action(self, action):
        with LOCK, open(self.path("actions.jsonl"), "a") as f:
            f.write(json.dumps(action) + "\n")

    def start(self, body):
        synth, state = self.synth(), self.state()
        if not synth["ready"]:
            return 409, {"error": f"the dataset is not finished ({synth['detail']})"}
        # request files wait for the orchestrator, which tracker_server.py starts before this if it isn't running
        if (state.get("phase") or "waiting_for_start") != "waiting_for_start":
            return 409, {"error": f"this run has already begun ({state.get('phase')}); restart training to start another"}
        if self.job == "marks":
            return self.start_marks(body)
        try:
            cfg = {"gpu_type": "H100", "num_gpus": int(body["num_gpus"]), "epochs": float(body["epochs"]),
                   "budget": float(body["budget"]), "lr": float(body["lr"]), "batch_size": int(body["batch_size"]),
                   "grad_accum": int(body.get("grad_accum", 1)), "warmup_steps": int(body.get("warmup_steps", 200)),
                   "val_every": int(body["val_every"]), "save_every": int(body["save_every"]),
                   "page_val_max": int(body["page_val_max"]), "eval_batches": int(body.get("eval_batches", 25)),
                   "samples": int(body.get("samples", 8)), "log_every": int(body.get("log_every", 20)),
                   "num_workers": int(body.get("num_workers", 6)),
                   "grad_checkpointing": bool(body.get("grad_checkpointing", False)),
                   "vision_lora": bool(body.get("vision_lora", False)),
                   "lora_rank": int(body.get("lora_rank") or 16), "requested_at": time.time()}
            init_name = str(body["init_model"]) if body.get("init_model") else None
            continue_run = bool(body.get("continue_run"))
            save_to = body.get("save_to") or None
        except (KeyError, TypeError, ValueError) as error:
            return 400, {"error": f"bad settings: {error}"}
        problems = []
        if continue_run:
            cont = state.get("continue_from")
            if not cont:
                problems.append("there is no previous run's checkpoint to continue from")
            elif init_name is not None:
                problems.append("a run continues the previous run or trains from a model in inference/comprehension/, not both")
            else:           # the kept checkpoint's own shape, as for a model from inference/comprehension/
                cfg.update(lora_rank=cont["lora_rank"], lora_alpha=cont["lora_alpha"], continue_from=cont)
        elif init_name is not None:
            model = inference_models.get(init_name)
            if model is None:
                problems.append(f"there is no model {init_name!r} in note_synth/inference/comprehension/")
            elif not isinstance(model["lora_rank"], int):
                problems.append(f"inference/comprehension/{init_name}/adapter_config.json has no LoRA rank")
            else:           # the adapter's own shape: its weights only fit a model built with its rank
                # int(): train.py's --lora-alpha is an int, and a saved adapter's config may hold 128.0
                cfg.update(lora_rank=model["lora_rank"], lora_alpha=int(model["lora_alpha"] or 2 * model["lora_rank"]),
                           init_model={"name": init_name, "signature": model["signature"], "info": model["info"]})
        elif not 1 <= cfg["lora_rank"] <= 256:
            problems.append("LoRA rank must be between 1 and 256")
        cfg.setdefault("lora_alpha", 2 * cfg["lora_rank"])
        if save_to is not None:
            name = str(save_to.get("name") or "").strip() if isinstance(save_to, dict) else ""
            problem = self.name_problem(name)
            if not isinstance(save_to, dict) or save_to.get("checkpoint") not in ("best", "last"):
                problems.append("the checkpoint to save at the end must be best or last")
            elif problem:
                problems.append(problem)
            else:
                cfg["save_to"] = {"checkpoint": save_to["checkpoint"], "name": name}
        if cfg["num_gpus"] not in (1, 2, 4, 8):          # the only H100 VM sizes Jarvis rents
            problems.append("GPUs must be 1, 2, 4 or 8 (Jarvis H100 VM sizes)")
        if not 0 < cfg["epochs"] <= 20:
            problems.append("epochs must be between 0 and 20")
        if not 0 < cfg["lr"] < 1:
            problems.append("learning rate must be between 0 and 1")
        if cfg["batch_size"] < 1 or cfg["grad_accum"] < 1:
            problems.append("batch size and accumulation must be at least 1")
        account = self.account_doc()
        if cfg["budget"] <= 0:
            problems.append("budget must be positive")
        elif account and account.get("balance") is not None and cfg["budget"] > account["balance"]:
            problems.append(f"budget ${cfg['budget']:.2f} is more than the account balance ${account['balance']:.2f}")
        if problems:
            return 400, {"error": "; ".join(problems)}
        marks = self.mark_model()
        if not marks["ready"]:
            return 409, {"error": f"Florence Tuning needs a mark separator, trained or picked in the mark separation tab: {marks['detail']}"}
        if not synth.get("train_recognition") and not self.args.synth_ready:
            return 409, {"error": "the dataset has no train recognition samples to train on with the segmentation samples: "
                                  "regenerate it with Recognition on"}
        if marks.get("path"):
            # this run's own copy of the model its recognition takes marks out with: the marks run can be restarted, or
            # another mark separator picked, meanwhile
            os.makedirs(self.path("mark_model"), exist_ok=True)
            shutil.copy2(marks["path"], self.path("mark_model", "model.pt.part"))
            os.replace(self.path("mark_model", "model.pt.part"), self.path("mark_model", "model.pt"))
            cfg["mark_model"] = {k: marks.get(k) for k in ("source", "name", "checkpoint", "step", "mean_f1", "run_id")}
        if cfg.get("init_model"):
            # this run's own copy: orchestrator.py uploads it, and inference/ can change without affecting the run
            if inference_models.snapshot(init_name, self.path("init_model")) is None:
                return 400, {"error": f"inference/comprehension/{init_name} disappeared while it was being copied"}
        return self.write_start(cfg)

    @staticmethod
    def name_problem(name, kind="comprehension"):
        if not inference_models.NAME.fullmatch(name):
            return "a model name is up to 40 letters, digits, _ - and ., not starting with ."
        if inference_models.taken(name, kind):
            return f"inference/{kind}/{name} already exists; pick another name"
        return None

    def write_start(self, cfg):
        try:
            fd = os.open(self.path("START.json"), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return 409, {"error": "already started"}
        with os.fdopen(fd, "w") as f:
            json.dump(cfg, f)
        return 200, {"ok": True, "config": cfg}

    def start_marks(self, body):
        """The mark separation run's settings: one GPU of any IN2 VM type."""
        try:
            cfg = {"gpu_type": str(body["gpu_type"]), "num_gpus": 1, "epochs": float(body["epochs"]),
                   "budget": float(body["budget"]), "lr": float(body["lr"]), "batch_size": int(body["batch_size"]),
                   "warmup_steps": int(body.get("warmup_steps", 200)), "val_every": int(body["val_every"]),
                   "save_every": int(body["save_every"]), "val_max": int(body.get("val_max", 0)),
                   "samples": int(body.get("samples", 16)), "log_every": int(body.get("log_every", 20)),
                   "requested_at": time.time()}
        except (KeyError, TypeError, ValueError) as error:
            return 400, {"error": f"bad settings: {error}"}
        problems = []
        if not 0 < cfg["epochs"] <= 500:
            problems.append("epochs must be between 0 and 500")
        if not (math.isfinite(cfg["lr"]) and 0 < cfg["lr"] < 1):
            problems.append("learning rate must be between 0 and 1")
        if cfg["batch_size"] < 1 or cfg["log_every"] < 1 or cfg["save_every"] < 1:
            problems.append("batch size, log interval and save interval must be at least 1")
        if cfg["val_every"] < 0 or cfg["val_max"] < 0 or cfg["warmup_steps"] < 0 or not 0 <= cfg["samples"] <= 64:
            problems.append("validation interval, val crops and warmup can't be negative; samples must be 0 to 64")
        account = self.account_doc()
        offers = {o["gpu_type"]: o for o in (account or {}).get("offers") or []}
        if offers and cfg["gpu_type"] not in offers:
            problems.append(f"there is no {cfg['gpu_type']} VM offer in IN2")
        elif offers and offers[cfg["gpu_type"]]["free"] < 1:
            problems.append(f"no {cfg['gpu_type']} is free in IN2 right now")
        if cfg["budget"] <= 0:
            problems.append("budget must be positive")
        elif account and account.get("balance") is not None and cfg["budget"] > account["balance"]:
            problems.append(f"budget ${cfg['budget']:.2f} is more than the account balance ${account['balance']:.2f}")
        if problems:
            return 400, {"error": "; ".join(problems)}
        return self.write_start(cfg)

    def set_lr(self, body):
        state = self.state()
        try:
            lr = float(body["lr"])
        except (KeyError, TypeError, ValueError):
            return 400, {"error": "lr must be a number"}
        if not (math.isfinite(lr) and 0 < lr < 1):
            return 400, {"error": "lr must be between 0 and 1"}
        remote = state.get("remote") or {}
        progress = remote.get("progress") or {}
        if state.get("phase") != "training" or progress.get("phase") not in ("training", "validating", "checkpointing"):
            return 409, {"error": "the learning rate can only be changed while training is running"}
        gpu = (state.get("machines") or {}).get("gpu", {}).get("id")
        seqs = [a.get("seq", 0) for a in state["actions"] if a.get("kind") == "lr"]
        seqs += [(remote.get("control") or {}).get("seq", 0), (remote.get("control_ack") or {}).get("seq", 0),
                 progress.get("control_seq", 0)]
        seq = max(seqs) + 1
        action = {"kind": "lr", "seq": seq, "lr": lr, "previous_peak_lr": progress.get("peak_lr"),
                  "requested_at": time.time(), "status": "sending"}
        self.record_action(action)

        def send():
            doc = json.dumps({"seq": seq, "lr": lr, "requested_at": action["requested_at"]})
            command = (f"cd ~/train/run && printf '%s' {shlex.quote(doc)} > control.json.tmp && "
                       f"mv control.json.tmp control.json && echo WROTE")
            try:
                code, out, err = self.jl.exec(gpu, command, timeout=180)
                ok = code == 0 and "WROTE" in out
                self.record_action({**action, "status": "sent" if ok else "failed", "at": time.time(),
                                    "detail": "" if ok else (out + err)[-300:]})
            except JLError as error:
                self.record_action({**action, "status": "failed", "at": time.time(), "detail": str(error)[-300:]})
        threading.Thread(target=send, daemon=True).start()
        return 200, {"ok": True, "seq": seq}

    def stop(self, force):
        state = self.state()
        now = time.time()
        if self.kill["requested_at"] is None:
            self.kill["requested_at"] = now
        open(self.path("KILL"), "w").write(str(now))
        machines = state.get("machines") or {}
        gpu = machines.get("gpu", {})

        def work():
            if gpu.get("id") and gpu.get("status") != "Paused" and state.get("phase") in ("training", "stopping"):
                try:
                    self.jl.exec(gpu["id"], "touch ~/train/run/STOP", timeout=120)
                    self.kill["stop_sent"] = time.time()
                except JLError:
                    pass
            if force or not state["orchestrator_alive"]:
                for kind in ("gpu", "data"):
                    m = machines.get(kind) or {}
                    if m.get("id") and m.get("status") != "Paused":
                        try:
                            self.jl.pause(m["id"])
                            self.kill["forced"] = {"ok": True, "at": time.time()}
                        except JLError as error:
                            self.kill["forced"] = {"ok": False, "at": time.time(), "detail": str(error)[-300:]}
        threading.Thread(target=work, daemon=True).start()
        return 200, {"ok": True, "kill": self.kill}

    def test(self, body):
        state = self.state()
        if body.get("confirm") is not True:
            return 400, {"error": "the test run needs confirming"}
        if state.get("phase") != "trained":
            return 409, {"error": "the test can run once training has ended and the GPU VM is paused"}
        if not ((state.get("remote") or {}).get("checkpoints")):
            return 409, {"error": "there is no checkpoint to test"}
        try:
            cap = float(body.get("max_spend", 4.0))
        except (TypeError, ValueError):
            return 400, {"error": "max spend must be a number"}
        try:
            fd = os.open(self.path("TEST.json"), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return 409, {"error": "the test set has already been requested; it can only be run once"}
        with os.fdopen(fd, "w") as f:
            json.dump({"max_spend": cap, "confirmed_at": time.time()}, f)
        return 200, {"ok": True}

    def reset(self, body):
        """Restart training (orchestrator.py reset): the run's files are archived and the next Start trains afresh,
        or, with continue_from best / last (seg), may continue from that checkpoint, kept on the GPU VM."""
        state = self.state()
        if body.get("confirm") is not True:
            return 400, {"error": "restarting training needs confirming"}
        if state.get("phase") not in RESETTABLE:
            return 409, {"error": "stop the run first: training can be restarted once it has ended, stopped or failed"}
        keep = body.get("continue_from") or None
        if keep is not None:
            if self.job != "seg" or keep not in ("best", "last"):
                return 400, {"error": "a restart continues from the best or the last checkpoint"}
            if keep not in [c.get("name") for c in ((state.get("remote") or {}).get("checkpoints") or [])]:
                return 409, {"error": f"this run has no {keep}/ checkpoint on the GPU VM to continue from"}
        try:
            fd = os.open(self.path("RESET.json"), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return 409, {"error": "a restart has already been requested"}
        with os.fdopen(fd, "w") as f:
            json.dump({"requested_at": time.time(), "continue_from": keep}, f)
        self.kill = {"requested_at": None, "stop_sent": None, "forced": None}
        return 200, {"ok": True}

    def set_budget(self, body):
        """BUDGET.json: orchestrator.py makes it the run's budget within a few seconds."""
        state = self.state()
        try:
            budget = float(body["budget"])
        except (KeyError, TypeError, ValueError):
            return 400, {"error": "the budget must be a number of dollars"}
        if not (math.isfinite(budget) and budget > 0):
            return 400, {"error": "the budget must be positive"}
        if (state.get("phase") or "waiting_for_start") == "waiting_for_start":
            return 409, {"error": "no run has started: set the budget in Start"}
        spent = (state.get("cost") or {}).get("total") or 0.0
        if budget <= spent:
            return 400, {"error": f"the run has already spent ${spent:.2f}; the budget has to be more than that (Stop ends the run)"}
        account = self.account_doc()
        if account and account.get("balance") is not None and budget - spent > account["balance"]:
            return 400, {"error": f"${budget - spent:.2f} more to spend is more than the account balance ${account['balance']:.2f}"}
        previous = (state.get("config") or {}).get("budget")
        with open(self.path("BUDGET.json.tmp"), "w") as f:
            json.dump({"budget": budget, "requested_at": time.time()}, f)
        os.replace(self.path("BUDGET.json.tmp"), self.path("BUDGET.json"))
        self.record_action({"kind": "budget", "budget": budget, "previous": previous, "requested_at": time.time()})
        return 200, {"ok": True}

    # --- saving to note_synth/inference/ ---------------------------------------------------------
    def save(self, body):
        """A checkpoint of this run into inference/comprehension/ under a new name. orchestrator.py does it: straight
        away from the copy it took when training ended, or after copying it back from the GPU VM (resumed if need be)."""
        if self.job == "marks":
            return self.save_marks(body)
        state = self.state()
        if state.get("phase") not in ("trained", "done"):
            return 409, {"error": "a checkpoint can be saved once training has ended"}
        checkpoint, name = body.get("checkpoint"), str(body.get("name") or "").strip()
        if checkpoint not in ("best", "last"):
            return 400, {"error": "the checkpoint must be best or last"}
        if checkpoint not in {c.get("name") for c in (state.get("remote") or {}).get("checkpoints") or []}:
            return 409, {"error": f"this run has no {checkpoint}/ checkpoint"}
        problem = self.name_problem(name)
        folder = self.path("save", "requests")
        os.makedirs(folder, exist_ok=True)
        pending = [s["name"] for s in state.get("saves") or [] if s.get("state") not in ("done", "failed")]
        pending += [(load(os.path.join(folder, n)) or {}).get("name", "") for n in os.listdir(folder) if n.endswith(".json")]
        if not problem and name.lower() in {p.lower() for p in pending}:
            problem = f"a save as inference/comprehension/{name} is already under way"
        if problem:
            return 400, {"error": problem}
        rid = str(int(time.time() * 1000))
        path = os.path.join(folder, f"{rid}.json")
        with open(path + ".tmp", "w") as f:
            json.dump({"id": rid, "checkpoint": checkpoint, "name": name, "requested_at": time.time()}, f)
        os.replace(path + ".tmp", path)
        return 200, {"ok": True, "id": rid}

    def save_marks(self, body):
        """best/ or last/ of the mark separation run into inference/mark_separation/ under a new name, straight away:
        orchestrator.py copies both back when training ends (fetch_mark_model)."""
        state = self.state()
        checkpoint, name = body.get("checkpoint"), str(body.get("name") or "").strip()
        if checkpoint not in ("best", "last"):
            return 400, {"error": "the checkpoint must be best or last"}
        local = (state.get("checkpoints_local") or {}).get(checkpoint)
        source = os.path.join(self.run_files_dir(), checkpoint)
        if state.get("phase") not in ("trained", "done") or not local or not os.path.isfile(os.path.join(source, "model.pt")):
            return 409, {"error": f"this run's {checkpoint}/ hasn't been copied back from the GPU VM"}
        problem = self.name_problem(name, "mark_separation")
        if problem:
            return 400, {"error": problem}
        cfg, remote = state.get("config") or {}, state.get("remote") or {}
        progress, step = remote.get("progress") or {}, local.get("step")
        record = inference_models.val_record(self.path("log.jsonl"), step, ("val/mean_f1",)) if step is not None else None
        run_info = {
            "checkpoint": "best (val mean F1)" if checkpoint == "best" else "last (end of training)", "step": step,
            "source": f"{remote.get('home')}/train/run/{checkpoint} (GPU VM {((state.get('machines') or {}).get('gpu') or {}).get('id')})",
            "saved_at": time.strftime("%Y-%m-%d %H:%M"), "run_id": state.get("run_id"),
            "base_model": "mark separation CNN (training/marks_cnn)",
            "val_mean_f1": record.get("val/mean_f1") if record else local.get("mean_f1"),
            "best_step": progress.get("best_step"), "total_steps": progress.get("total_steps"), "epochs": cfg.get("epochs"),
            "best_val_mean_f1": progress.get("best_mean_f1"),
            "train_config": {"gpus": f"1 x {cfg.get('gpu_type')}", "batch_size": cfg.get("batch_size"), "lr": cfg.get("lr"),
                             "warmup_steps": cfg.get("warmup_steps"), "train_crops": progress.get("train_rows")}}
        stats = {"checkpoint": checkpoint, "step": step, "val_step": record["step"], "val_crops": record.get("val/crops"),
                 "run_id": state.get("run_id"), "val": record} if record else None
        try:
            inference_models.save(name, source, run_info, stats, "mark_separation")
        except (FileExistsError, ValueError, OSError) as error:
            return 400, {"error": str(error)}
        self.record_action({"kind": "save", "checkpoint": checkpoint, "name": name, "step": step,
                            "val_mean_f1": run_info["val_mean_f1"], "at": time.time()})
        return 200, {"ok": True, "name": name}

    def use_mark_model(self, body):
        """The mark separator Florence Tuning's next Start takes: {"name": a folder in inference/mark_separation/}, or
        {"name": null} for the mark separation run's own model. A run already started keeps the one it has."""
        if self.job != "marks":
            return 404, {"error": "the mark separator is picked in the mark separation tab"}
        name = str(body.get("name") or "").strip() or None
        path = self.mark_choice_path()
        if name is None:
            if os.path.exists(path):
                os.remove(path)
        else:
            model = inference_models.get(name, "mark_separation")
            if model is None:
                return 400, {"error": f"there is no mark separator {name!r} in note_synth/inference/mark_separation/"}
            with open(path + ".tmp", "w") as f:
                json.dump({"name": name, "signature": model["signature"], "chosen_at": time.time()}, f)
            os.replace(path + ".tmp", path)
        self.record_action({"kind": "use_mark_model", "name": name, "at": time.time()})
        return 200, {"ok": True, "mark_model": self.mark_model()}

    def models_doc(self):
        """Every comprehension model in inference/ with the validation stats it was saved with (None for one saved without)."""
        return {"models": [{**m, "stats": inference_models.stats(m["name"])}
                           for m in self.public(inference_models.find())]}

    # --- page explorer (segmentation run) -----------------------------------------------------
    # orchestrator.py resumes / pauses the GPU VM and runs requests; this side only writes
    # request files into <run dir>/explore/ and reads back what was pulled into files/.
    def val_pages(self, state):
        return int((state.get("synth") or {}).get("val_pages") or 2000)

    def explore_action(self, action, body):
        if self.job != "seg":
            return 404, {"error": "the page explorer belongs to the Florence Tuning run"}
        state = self.state()
        ex = state.get("explore") or {"state": "paused", "requests": {}}
        if state.get("phase") not in ("trained", "done"):
            return 409, {"error": "the explorer is available once training has ended"}
        folder = self.path("explore")
        os.makedirs(os.path.join(folder, "requests"), exist_ok=True)
        if action == "resume":
            if ex["state"] not in ("paused", "pausing"):        # one asked while pausing waits for the pause
                return 409, {"error": f"the VM is already {ex['state']}"}
            if state["locks"]["test"] and state.get("phase") == "trained":
                return 409, {"error": "a test run has been requested and needs the VM"}
            open(os.path.join(folder, "RESUME"), "w").close()
            return 200, {"ok": True}
        if action == "pause":
            if ex["state"] == "paused":
                return 409, {"error": "the VM is already paused"}
            open(os.path.join(folder, "PAUSE"), "w").close()
            return 200, {"ok": True}
        try:
            page = int(body["page"])
        except (KeyError, TypeError, ValueError):
            return 400, {"error": "page must be a number"}
        if not 0 <= page < self.val_pages(state):
            return 400, {"error": f"there are {self.val_pages(state)} val pages, numbered from 0"}
        model = body.get("model") or None           # a folder in note_synth/inference/comprehension/; None: this run's own checkpoint
        if model is not None and inference_models.get(model) is None:
            return 400, {"error": f"there is no model {model!r} in note_synth/inference/comprehension/"}
        # a folder in note_synth/inference/mark_separation/; None: the one this run was started with; NO_MARKS: none
        marks = str(body.get("marks") or "") or None
        if marks not in (None, inference_models.NO_MARKS) and inference_models.get(marks, "mark_separation") is None:
            return 400, {"error": f"there is no mark separator {marks!r} in note_synth/inference/mark_separation/"}
        resuming = ex["state"] in ("paused", "pausing")
        if resuming:        # running a page on a paused VM resumes it first; the request waits for the resume
            if state["locks"]["test"] and state.get("phase") == "trained":
                return 409, {"error": "a test run has been requested and needs the VM"}
            open(os.path.join(folder, "RESUME"), "w").close()
        stem = f"p{page:07d}-{model or 'run'}~{marks or '~run'}-"      # "~" is in no model name
        active = [r["id"] for r in (ex.get("requests") or {}).values()
                  if r["page"] == page and r.get("model") == model and r.get("marks") == marks
                  and r["state"] in ("queued", "sent", "running")]
        active += [n[:-5] for n in os.listdir(os.path.join(folder, "requests"))
                   if n.startswith(stem) and n.endswith(".json") and n[len(stem):-5].isdigit()]
        if active:
            return 200, {"ok": True, "id": active[0], "already": True, "resuming": resuming}
        rid = f"{stem}{int(time.time() * 1000)}"
        path = os.path.join(folder, "requests", f"{rid}.json")
        with open(path + ".tmp", "w") as f:
            json.dump({"id": rid, "page": page, "model": model, "marks": marks, "requested_at": time.time()}, f)
        os.replace(path + ".tmp", path)
        return 200, {"ok": True, "id": rid, "resuming": resuming}

    def explore_index(self):
        state = self.state()
        doc = load(os.path.join(self.files, "train", "explore", "index.json"))
        return {"total": self.val_pages(state), "pages": doc.get("pages") if doc else None}

    def explore_models(self):
        """What the page can pick: every model in note_synth/inference/comprehension/ and whether this run has a
        checkpoint of its own on the GPU VM; every mark separator in note_synth/inference/mark_separation/ and the one
        this run was started with (None: it has none)."""
        state = self.state()
        return {"models": self.public(inference_models.find()),
                "run_checkpoint": bool((state.get("remote") or {}).get("checkpoints")),
                "mark_models": self.public(inference_models.find("mark_separation")),
                "run_mark_model": (state.get("config") or {}).get("mark_model")}

    def explore_result(self, page, model=None, marks=None):
        if self.job != "seg":
            return 404, {"error": "the page explorer belongs to the Florence Tuning run"}
        state = self.state()
        done = sorted((r for r in ((state.get("explore") or {}).get("requests") or {}).values()
                       if r.get("page") == page and r.get("model") == model and r.get("marks") == marks
                       and r.get("state") == "done"),
                      key=lambda r: r.get("finished_at") or 0)
        if not done:
            return 404, {"error": "this page hasn't been run with this model and mark separator yet"}
        request = done[-1]
        doc = load(os.path.join(self.files, request["result"]))
        if not doc:
            return 404, {"error": "the result file is missing"}

        def url(remote):
            rel = self.local_file(remote or "", state)
            return self.file_url(rel) if rel and os.path.exists(os.path.join(self.files, rel)) else None

        cascade = doc["cascade"]
        recognition = cascade.get("recognition")        # None: the page's JSON has no ink to read
        if recognition is not None:
            recognition = [{**{k: group.get(k) for k in ("task", "boxes", "targets", "matched", "predictions", "edits",
                                                          "chars", "correct", "totals", "loss", "tokens", "truncated")},
                            "image_url": url(group.get("image"))}
                           for group in recognition]
        return 200, {
            "id": request["id"], "page": page, "model": model, "marks": marks, "mark_model": doc.get("mark_model"),
            "name": doc["name"], "checkpoint": doc.get("checkpoint"),
            "finished_at": doc.get("finished_at"), "seconds": doc.get("seconds"),
            "cascade": {**{k: cascade.get(k) for k in ("predicted", "outputs", "truth_objects", "summary", "seconds")},
                        "name": doc["name"] + ".png", "image_url": url(doc["image"]), "image_size": doc["image_size"],
                        "truth": doc["truth"], "recognition": recognition,
                        "recognition_missed": cascade.get("recognition_missed")},
            "rows": {"summary": doc["rows"]["summary"], "seconds": doc["rows"].get("seconds"),
                     "samples": [{**{k: row.get(k) for k in ("prompt", "target", "prediction", "valid", "errors")},
                                  "name": os.path.basename(row["image"].replace("\\", "/")), "image_url": url(row["image"])}
                                 for row in doc["rows"]["samples"]]}}


def make_servers(args):
    """Both runs by job; args carries run_dir, marks_run_dir and synth_ready."""
    os.makedirs(args.run_dir, exist_ok=True)
    return {"seg": Server(args, "seg", args.run_dir), "marks": Server(args, "marks", args.marks_run_dir)}


def get(servers, path, q):
    """(code, body) or (code, bytes, content type) for a GET; None for a path these tabs don't serve."""
    server = servers.get(q.get("job", "seg"))
    if server is None:
        return 404, {"error": "unknown job"}
    if path == "/api/status":
        return 200, server.state()
    if path == "/api/log":
        return 200, server.records("log.jsonl", int(q.get("since", 0)))
    if path == "/api/gpu":
        return 200, server.records("gpu.jsonl", int(q.get("since", 0)))
    if path == "/api/samples":
        return 200, server.samples(q.get("tag", ""), q.get("kind", "pages"))
    if path == "/api/account":
        return 200, server.account_doc() or {}
    if path == "/api/explore/index":
        return 200, server.explore_index()
    if path == "/api/explore/models":
        return 200, server.explore_models()
    if path == "/api/models":
        return 200, server.models_doc()
    if path == "/api/explore/result":
        try:
            page = int(q.get("page", ""))
        except ValueError:
            return 400, {"error": "page must be a number"}
        return server.explore_result(page, q.get("model") or None, q.get("marks") or None)
    if path == "/file":
        target = os.path.realpath(os.path.join(server.files, q.get("p", "")))
        if not target.startswith(server.files + os.sep) or not os.path.isfile(target):
            return 404, {"error": "not found"}
        return 200, open(target, "rb").read(), "image/png" if target.endswith(".png") else "application/json"
    return None


def post(servers, path, q, body):
    server = servers.get(q.get("job", "seg"))
    if server is None:
        return 404, {"error": "unknown job"}
    if path == "/api/start":
        return server.start(body)
    if path == "/api/lr":
        return server.set_lr(body)
    if path == "/api/kill":
        return server.stop(bool(body.get("force")))
    if path == "/api/test":
        return server.test(body)
    if path == "/api/reset":
        return server.reset(body)
    if path == "/api/budget":
        return server.set_budget(body)
    if path == "/api/save":
        return server.save(body)
    if path == "/api/mark_model":
        return server.use_mark_model(body)
    if path in ("/api/explore/resume", "/api/explore/pause", "/api/explore/run"):
        return server.explore_action(path.rsplit("/", 1)[1], body)
    return None
