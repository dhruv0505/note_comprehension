"""
render_page_val.py — whole pages from the note_synth generator, as the page cascade reads them.

    python render_page_val.py --pages 200                                 # -> data/val_pages/
    python render_page_val.py --pages 200 --seed 2000 --out data/test_pages
    python render_page_val.py --pages 5 --no-check

Each page is two files, pageNNNN.png and pageNNNN.json. The PNG is the page's ink,
black on white, drawn straight from its strokes at --size (1600) with the pen
width scaled up from the board's (seg_prompts.render_board, INK_WIDTH_PX), so
resized to 768 it looks like a training page. The JSON lists every object on the
page, in no order a reader may rely on, coordinates in the PNG's pixels:

    {"format": "note_synth_page/1", "image": "page0000.png",
     "image_size": [1600, 1600], "page_size": [1600, 1600], "coord_order": "x0y0x1y1",
     "source": {"generator": "note_synth", "seed": 1000, "page": 0, "board_size": 1000},
     "objects": [
       {"id": 0, "level": 2, "class": "textblock", "bbox": [x0, y0, x1, y1]},
       {"id": 1, "level": 1, "class": "textline", "parent": 0, "bbox": [x0, y0, x1, y1],
        "text": "its transcription", "ink": [[[x, y, t], ...], ...]},     a formula's text: its LaTeX
       {"id": 2, "level": 1, "class": "vertex", "parent": 5, "position": [x, y]},
       {"id": 4, "level": 1, "class": "arc", "parent": 5, "points": [x1, y1, x2, y2, x3, y3], "bbox": [...]},
       {"id": 6, "level": 1, "class": "segment", "parent": 5, "points": [x1, y1, x2, y2], "bbox": [...]},
       {"id": 3, "level": 0, "class": "arrow", "start": [x, y], "end": [x, y], "bbox": [x0, y0, x1, y1]}]}

An arc's points are its start, middle and end, counterclockwise as the image is
seen; a segment's are its two ends.

`parent` is the id of the level-2 object a level-1 object sits under (absent when
there is none). Classes are the generator's (line_mark); words and unknown ink
are left out. Levels are labels.py's, enclosures and line marks at level 1. Every
arrow has a start (its tail) and an end (its head), found as seg_prompts finds
them for its targets; an arrow without them stops the run.

Checks (unless --no-check). Each stops the run rather than leave a folder the
training side would misread:
  * seg_eval.py's copies of the prompts equal note_synth/seg_prompts.py's
  * the renderer at board size reproduces seg_prompts.render_board pixel for pixel
  * the objects written are exactly labels.export_json's (bar words and unknown ink),
    same levels, classes and coordinates
  * every object in a training target (seg_prompts.page_objects, level1_objects) is
    written, a level-1 one under the parent it is trained under and inside its box
  * every arrow runs tail to head, judged by how it was built rather than by
    seg_prompts._arrow: a flowchart edge starts on its source shape's outline and ends
    on its target's (OHFD's source_id / target_id); a connector starts and ends at its
    shaft's first and last points, with the head drawn at the end. Any other kind of
    arrow is counted as unverified
  * the folder read back through seg_eval.PageSet, with a perfect model answering
    every cascade prompt, scores no FP and exactly one FN per object no prompt
    reaches (reported per class: the cascade's ceiling)
note_synth is imported read-only: no bytecode is written and nothing is saved under it.
"""

from __future__ import annotations
import sys

sys.dont_write_bytecode = True          # before note_synth is imported

import argparse
import json
import math
import os
import random
import re
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NOTE_SYNTH = os.path.abspath(os.path.join(HERE, "..", "note_synth"))
FORMAT = "note_synth_page/1"
MODEL_SIZE = 768                        # the processor's input size


def render_page(blocks, size, board_size, ink_width):
    """seg_prompts.render_board at `size` pixels: the same strokes and pen, scaled.
    note_synth/page_export.py's, which render_seg_data.py and synthesize.py draw with too."""
    from page_export import render_page as draw
    return draw(blocks, size, board_size, ink_width)


def page_objects(blocks, scale):
    """Every labelled object on the page, walked as seg_prompts._collect walks it.
    note_synth/page_export.py's, which synthesize.py writes its pages with too."""
    from page_export import page_objects as collect
    return collect(blocks, scale)


def _where(obj):
    for key in ("position", "points", "bbox"):
        if key in obj:
            return tuple(obj[key])


def check_prompts(sp):
    import seg_eval as E
    ours = (E.LEVEL2_PROMPT, E.ARROW_PROMPT, E.LEVEL1_PROMPTS)
    theirs = (sp.LEVEL2_PROMPT, sp.ARROW_PROMPT, sp.LEVEL1_PROMPTS)
    if ours != theirs:
        raise SystemExit(f"seg_eval.py's prompts differ from seg_prompts.py's:\n  seg_eval:    {ours}\n  seg_prompts: {theirs}")


def check_renderer(blocks, sp):
    board = np.asarray(sp.render_board(blocks), dtype=np.int16)
    ours = np.asarray(render_page(blocks, sp.BOARD_SIZE, sp.BOARD_SIZE, sp.INK_WIDTH_PX), dtype=np.int16)
    if board.shape != ours.shape or np.abs(board - ours).max():
        raise SystemExit("render_page at board size does not reproduce seg_prompts.render_board")


def check_export(blocks, objects, scale, labels, board_size):
    exported = json.loads(labels.export_json(blocks, float(board_size), float(board_size)))["objects"]
    theirs = Counter((o["level"], o["class"], tuple(round(float(v) * scale, 3) for v in _where(o)))
                     for o in exported if o["class"] not in ("word", "unknown"))
    ours = Counter((o["level"], o["class"], _where(o)) for o in objects)
    if theirs != ours:
        raise SystemExit(f"objects differ from labels.export_json:\n  only there: {dict(theirs - ours)}\n"
                         f"  only here:  {dict(ours - theirs)}")


def trained_ids(blocks, objects, scale, sp):
    """Ids of the objects some training target holds, checking each level-1 one is
    written under the parent it is trained under, inside that parent's box."""
    px = lambda values: tuple(round(float(v) * scale, 3) for v in values)
    by_place = {}
    for o in objects:
        by_place.setdefault((o["class"], _where(o)), []).append(o)

    def take(cls, board):
        found = by_place.get((cls, px(board)))
        if not found:
            raise SystemExit(f"a trained {cls} at {px(board)} is not among the objects written")
        return found.pop(0)

    ids = set()
    level2, arrows = sp.page_objects(blocks)
    for obj in level2:
        parent = take(obj["class"], obj["board"])
        ids.add(parent["id"])
        x0, y0, x1, y1 = parent["bbox"]
        for child in sp.level1_objects(obj["node"], obj["class"]):
            entry = take(child["class"], child["board"])
            if entry.get("parent") != parent["id"]:
                raise SystemExit(f"{child['class']} {entry['id']} is trained under {obj['class']} {parent['id']} "
                                 f"but written under {entry.get('parent')}")
            where = _where(entry)
            if not (x0 - 1e-3 <= min(where[0::2]) and max(where[0::2]) <= x1 + 1e-3 and
                    y0 - 1e-3 <= min(where[1::2]) and max(where[1::2]) <= y1 + 1e-3):
                raise SystemExit(f"{child['class']} {entry['id']} lies outside its parent's box")
            ids.add(entry["id"])
    written = [o for o in objects if o["class"] == "arrow"]
    if len(written) != len(arrows):
        raise SystemExit(f"{len(arrows)} arrows in the arrow target, {len(written)} written")
    ids.update(o["id"] for o in written)
    return ids


def check_arrows(blocks, objects, scale):
    """Each written arrow's start is its tail and its end its head, by construction.
    Returns how many arrows of each kind were checked."""
    from make_pdf import _graph_span

    written = iter([o for o in objects if o["class"] == "arrow"])
    kinds = Counter()
    gap = lambda p, b: math.hypot(max(b[0] - p[0], 0, p[0] - b[2]), max(b[1] - p[1], 0, p[1] - b[3]))
    tolerance = 1e-3                    # board units; the JSON rounds to 1e-3 pixels

    def walk(node, shapes):
        meta = node.meta or {}
        if meta.get("flowchart"):
            shapes = {c.meta["id"]: c for c in node.children
                      if c.meta and c.meta.get("id") and c.class_name == "enclosure"}
        if node.class_name == "arrow":
            entry = next(written)
            start = [v / scale for v in entry["start"]]
            end = [v / scale for v in entry["end"]]
            strokes = node.all_strokes()
            if shapes and _graph_span(node, shapes)[0] is not None:
                kind = "flowchart edge"
                ok = (gap(start, shapes[meta["source_id"]].bbox()) < tolerance and
                      gap(end, shapes[meta["target_id"]].bbox()) < tolerance)
            elif node.source in ("geometry", "procedural") and len(strokes) > 1:
                kind = "connector"
                shaft = strokes[0]
                head = [p for stroke in strokes[1:] for p in zip(stroke.x, stroke.y)]
                to_head = lambda p: min(math.hypot(p[0] - x, p[1] - y) for x, y in head)
                ok = (math.hypot(start[0] - shaft.x[0], start[1] - shaft.y[0]) < tolerance and
                      math.hypot(end[0] - shaft.x[-1], end[1] - shaft.y[-1]) < tolerance and
                      to_head(end) < to_head(start))
            else:
                kind, ok = "unverified", True
            if not ok:
                raise SystemExit(f"{kind} arrow {entry['id']} does not run tail to head: "
                                 f"start {entry['start']}, end {entry['end']}")
            kinds[kind] += 1
        for child in node.children:
            walk(child, shapes)

    for block in blocks:
        walk(block, None)
    return kinds


def check_folder(folder, trained, unreachable):
    """The folder read back as training reads it, a perfect model answering every prompt."""
    from torch.utils.data import DataLoader
    import seg_eval as E

    docs = {}

    def doc(path):
        if path not in docs:
            with open(os.path.splitext(path)[0] + ".json") as f:
                docs[path] = json.load(f)
        return docs[path]

    def coords(o):
        return o["start"] + o["end"] if "start" in o else list(_where(o))

    def text(items):
        return E.START + "".join("".join(f"<loc_{min(999, max(0, int(v / MODEL_SIZE * 1000)))}>" for v in c)
                                 + name.replace("_", " ") for name, c in items) + E.EOS

    def answer(requests):
        texts = []
        for r in requests:
            d = doc(r["path"])
            known = [o for o in d["objects"] if o["id"] in trained[os.path.basename(r["path"])]]
            width, height = d["image_size"]
            if r["stage"] in ("level2", "arrow"):
                level = 2 if r["stage"] == "level2" else 0
                items = [(o["class"], [v * MODEL_SIZE / (width if i % 2 == 0 else height)
                                       for i, v in enumerate(coords(o))])
                         for o in known if o["level"] == level]
            else:
                box = r["parent"][1]
                parent = min((o for o in known if o["level"] == 2),
                             key=lambda o: sum(abs(a - b) for a, b in zip(o["bbox"], box)))
                t = r["transform"]
                items = [(o["class"], [v * t["scale"][i % 2] + t["offset"][i % 2] for i, v in enumerate(coords(o))])
                         for o in known if o.get("parent") == parent["id"]]
            texts.append(text(items))
        return texts, [False] * len(texts)

    loader = DataLoader(E.PageSet([folder]), batch_size=4, collate_fn=E.collate_pages)
    summary, _ = E.validate_pages(loader, answer, E.SegMetrics(0.5, 20.0, 30.0, 4.0, unit="pages"), MODEL_SIZE)
    print(f"perfect model through the cascade, {folder} read back through seg_eval:\n{E.format_report(summary)}")
    problems = [] if summary["parse_valid_rate"] == 1.0 else ["not every answer parsed"]
    for name, s in summary["per_class"].items():
        missing = unreachable[name.replace(" ", "_")]
        if s["fp"] or s["fn"] != missing:
            problems.append(f"{name}: FP {s['fp']}, FN {s['fn']} where {missing} are unreachable")
    if problems:
        raise SystemExit("the folder does not score as written:\n  " + "\n  ".join(problems))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1000, help="keep apart from render_seg_data.py's (0)")
    ap.add_argument("--size", type=int, default=1600, help="page image width and height, pixels")
    ap.add_argument("--out", default=os.path.join(HERE, "data", "val_pages"))
    ap.add_argument("--no-check", action="store_true")
    args = ap.parse_args()

    sys.path.insert(0, NOTE_SYNTH)
    import labels
    import seg_prompts as sp
    from make_pdf import build_pools
    from page_generator import generate_page

    check = not args.no_check
    if check:
        check_prompts(sp)
    os.makedirs(args.out, exist_ok=True)
    for name in os.listdir(args.out):                   # a smaller run leaves no stale pages behind
        if re.fullmatch(r"page\d+\.(png|json)", name):
            os.remove(os.path.join(args.out, name))

    random.seed(args.seed)
    pools = build_pools(with_shape_bank=True)
    scale = args.size / sp.BOARD_SIZE
    totals, unreachable, trained, arrow_kinds = Counter(), Counter(), {}, Counter()
    for index in range(args.pages):
        blocks = generate_page(**pools)["board_blocks"]
        objects = page_objects(blocks, scale)
        name = f"page{index:04d}"
        if check:
            if index == 0:
                check_renderer(blocks, sp)
            check_export(blocks, objects, scale, labels, sp.BOARD_SIZE)
            ids = trained_ids(blocks, objects, scale, sp)
            trained[name + ".png"] = ids
            arrow_kinds.update(check_arrows(blocks, objects, scale))
            unreachable.update(o["class"] for o in objects if o["id"] not in ids)
        render_page(blocks, args.size, sp.BOARD_SIZE, sp.INK_WIDTH_PX).save(os.path.join(args.out, name + ".png"))
        with open(os.path.join(args.out, name + ".json"), "w") as f:
            json.dump({"format": FORMAT, "image": name + ".png", "image_size": [args.size, args.size],
                       "page_size": [args.size, args.size], "coord_order": "x0y0x1y1",
                       "source": {"generator": "note_synth", "seed": args.seed, "page": index,
                                  "board_size": sp.BOARD_SIZE},
                       "objects": objects}, f)
        counts = Counter(o["class"] for o in objects)
        totals.update(counts)
        print(f"  {name}: {len(objects)} objects ({', '.join(f'{k} {v}' for k, v in sorted(counts.items()))})")

    print(f"Wrote {args.pages} pages ({args.size} x {args.size}) -> {args.out}")
    print(f"  objects: {', '.join(f'{k} {v}' for k, v in totals.most_common())}")
    if check:
        print(f"  checks passed: prompts, renderer, export_json, training targets, arrows tail to head "
              f"({', '.join(f'{k} {v}' for k, v in arrow_kinds.items()) or 'no arrows'}); "
              f"objects no prompt reaches: {dict(unreachable) or 'none'}")
        check_folder(args.out, trained, unreachable)


if __name__ == "__main__":
    main()
