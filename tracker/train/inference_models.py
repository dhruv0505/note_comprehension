"""inference_models.py — the models in note_synth/inference/: listed for the tracker, saved from a training run.

Two kinds, each in a folder of its own there, holding one folder per model:
  comprehension/<name>     a LoRA adapter for the Florence-2 model (adapter_config.json and adapter_model.safetensors),
                           saved from the Florence Tuning run
  mark_separation/<name>   the mark separation CNN (model.pt as train_marks.py saves it), saved from the mark separation run
Beside the weights, when there is one:
  run_info.json   where it came from: the run, checkpoint and step, its LoRA rank, the model it was trained from
  stats.json      its validation scores at that checkpoint: the validation record the run logged (comprehension:
                  pages/... and rows/..., per-class precision, recall, F1, TP, FP, FN and the means; mark separation:
                  val/..., per channel and kind of mark), val only
Only the weights go to a GPU VM. train_api.py lists the models for the page (the Models tab compares the comprehension
models' stats), snapshots the model a run trains further from and the mark separator a Florence Tuning run takes marks
out with, and saves the mark separation run's checkpoints; orchestrator.py uploads models for the explorer and saves
the Florence Tuning run's checkpoints. A model folder is never written to once it exists: a save under a name that is
taken is refused.
"""
import json
import os
import re
import shutil

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "inference")
# each kind's files, the weights last: their size and time are the model's signature
KINDS = {"comprehension": ("adapter_config.json", "adapter_model.safetensors"), "mark_separation": ("model.pt",)}
FILES = KINDS["comprehension"]
NAME = re.compile(r"[\w-][\w.-]{0,39}")         # safe as a path part, in a shell command and in a request id
NO_MARKS = "~none"                              # the explorer's mark separator choice for none; never a NAME
INFO_KEYS = ("checkpoint", "step", "best_step", "total_steps", "epochs", "best_val_mean_f1", "val_mean_f1",
             "lora_rank", "trained_from", "copied_at", "saved_at")


def _load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _write(path, doc):
    with open(path, "w") as f:
        json.dump(doc, f, indent=1)


def folder(kind):
    if kind not in KINDS:
        raise ValueError(f"{kind!r} is not a kind of model: {', '.join(KINDS)}")
    return os.path.join(ROOT, kind)


def get(name, kind="comprehension"):
    """The model named `name`, or None: {name, kind, folder, info, size_mb, signature, has_stats}, and for a
    comprehension model its base_model, lora_rank and lora_alpha. The signature changes whenever the weights do, so a
    changed model is uploaded and loaded again."""
    files, path = KINDS[kind], os.path.join(folder(kind), name or "")
    if not NAME.fullmatch(name or "") or not all(os.path.isfile(os.path.join(path, f)) for f in files):
        return None
    weights = os.path.join(path, files[-1])
    info = _load(os.path.join(path, "run_info.json")) or {}
    model = {"name": name, "kind": kind, "folder": path,
             "info": {k: info[k] for k in INFO_KEYS if k in info},
             "size_mb": round(sum(os.path.getsize(os.path.join(path, f)) for f in files) / 1e6, 1),
             "signature": f"{os.path.getsize(weights)}-{int(os.path.getmtime(weights))}",
             "has_stats": os.path.isfile(os.path.join(path, "stats.json"))}
    if kind == "comprehension":
        config = _load(os.path.join(path, "adapter_config.json")) or {}
        model.update(base_model=config.get("base_model_name_or_path"), lora_rank=config.get("r"),
                     lora_alpha=config.get("lora_alpha"))
    return model


def find(kind="comprehension"):
    names = sorted(os.listdir(folder(kind))) if os.path.isdir(folder(kind)) else []
    return [model for model in (get(name, kind) for name in names) if model]


def stats(name, kind="comprehension"):
    return _load(os.path.join(folder(kind), name, "stats.json")) if NAME.fullmatch(name or "") else None


def taken(name, kind="comprehension"):
    """Whether anything in inference/<kind>/ already has this name (compared as Windows does, ignoring case)."""
    names = os.listdir(folder(kind)) if os.path.isdir(folder(kind)) else []
    return (name or "").lower() in {n.lower() for n in names}


def snapshot(name, dest, kind="comprehension"):
    """Copies the model's weights to the folder `dest` (replaced whole), for a run of its own; the model, or None."""
    model = get(name, kind)
    if model is None:
        return None
    part = dest + ".part"
    shutil.rmtree(part, ignore_errors=True)
    os.makedirs(part)
    for file in KINDS[kind]:
        shutil.copy2(os.path.join(model["folder"], file), os.path.join(part, file))
    shutil.rmtree(dest, ignore_errors=True)
    os.replace(part, dest)
    return model


def val_record(log_path, step, keys=("pages/mean_f1", "rows/mean_f1")):
    """The validation record the run logged at `step`, or the latest one before it; None if there is none. A record is
    one with any of `keys`: train.py's by default, ("val/mean_f1",) for train_marks.py's."""
    found = None
    try:
        with open(log_path) as f:
            for line in f:
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                # not an event: a "best" event carries the score it was chosen by, not a validation's scores
                if (any(record.get(k) is not None for k in keys) and not record.get("event")
                        and record.get("step") is not None and record["step"] <= step):
                    if found is None or record["step"] >= found["step"]:
                        found = record
    except OSError:
        return None
    return {k: v for k, v in found.items() if k != "time"} if found else None


def save(name, source, run_info, stats_doc=None, kind="comprehension"):
    """A new model `name` in inference/<kind>/ from the files in `source`, with its run_info.json and stats.json.
    Written beside the others under a hidden name, then renamed into place; raises FileExistsError if the name is taken."""
    if not NAME.fullmatch(name or ""):
        raise ValueError(f"{name!r} is not a model name: up to 40 letters, digits, _ - and ., not starting with .")
    if taken(name, kind):
        raise FileExistsError(f"inference/{kind}/{name} already exists")
    os.makedirs(folder(kind), exist_ok=True)
    part = os.path.join(folder(kind), f".{name}.part")  # a name can't start with ".", so this is never listed
    shutil.rmtree(part, ignore_errors=True)
    os.makedirs(part)
    try:
        for file in KINDS[kind]:
            shutil.copy2(os.path.join(source, file), os.path.join(part, file))
        _write(os.path.join(part, "run_info.json"), run_info)
        if stats_doc:
            _write(os.path.join(part, "stats.json"), stats_doc)
        if taken(name, kind):
            raise FileExistsError(f"inference/{kind}/{name} already exists")
        os.rename(part, os.path.join(folder(kind), name))  # refuses to replace a folder that appeared meanwhile
    except BaseException:
        shutil.rmtree(part, ignore_errors=True)
        raise
    return os.path.join(folder(kind), name)
