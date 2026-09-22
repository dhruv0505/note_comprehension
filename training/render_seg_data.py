"""
render_seg_data.py — render segmentation samples from the note_synth generator
into a folder the dataset reads.

    python render_seg_data.py --pages 10            # -> data/images/*.png + data/manifest.jsonl

Each page is drawn at --size (1600) pixels, as render_page_val.py draws its pages,
and its samples come from that drawing, each --image (768) square, greyscale,
black ink on white: the full page downsized, then each level-2 object cut out of
the 1600 page and scaled so its longer side is 768, centred on white — the cut
seg_eval.py makes at inference (note_synth/page_export.py). Each prompt asked of
a sample is one manifest row:

    {"image": "images/p0003_s02.png", "prompt": "...", "target": "..."}

so a full page contributes two rows sharing an image, a textblock/table/geometry/
flowchart crop one, and a drawing crop none (it has no prompt). note_synth is
imported read-only: no bytecode is written and nothing is saved under it. A full
dataset split is note_synth/synthesize.py, which writes these same files.
"""

from __future__ import annotations
import sys

sys.dont_write_bytecode = True          # before note_synth is imported

import argparse
import json
import os
import random
import re

HERE = os.path.dirname(os.path.abspath(__file__))
NOTE_SYNTH = os.path.abspath(os.path.join(HERE, "..", "note_synth"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--size", type=int, default=1600, help="page drawing the samples are cut from, pixels")
    ap.add_argument("--image", type=int, default=768, help="sample image width and height, pixels")
    ap.add_argument("--out", default=os.path.join(HERE, "data"))
    args = ap.parse_args()

    sys.path.insert(0, NOTE_SYNTH)
    import seg_prompts as sp
    sp.configure(image=args.image)
    from make_pdf import build_pools
    from page_export import render_page, write_samples
    from page_generator import generate_page

    image_dir = os.path.join(args.out, "images")
    os.makedirs(image_dir, exist_ok=True)
    for name in os.listdir(image_dir):                  # a smaller run leaves no stale samples behind
        if re.fullmatch(r"p\d+_s\d+\.png", name):
            os.remove(os.path.join(image_dir, name))

    random.seed(args.seed)
    pools = build_pools(with_shape_bank=True)
    scale = args.size / sp.BOARD_SIZE
    rows = []
    for page_index in range(args.pages):
        blocks = generate_page(**pools)["board_blocks"]
        page = render_page(blocks, args.size, sp.BOARD_SIZE, sp.INK_WIDTH_PX)
        page_rows, _ = write_samples(blocks, page, scale, args.out, "images", f"p{page_index:04d}")
        rows.extend(page_rows)
        print(f"  page {page_index}: {len(rows)} rows so far")

    with open(os.path.join(args.out, "manifest.jsonl"), "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"Wrote {len(rows)} prompt rows -> {os.path.join(args.out, 'manifest.jsonl')}")


if __name__ == "__main__":
    main()
