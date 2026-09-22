"""
seg_pages.py — the segmentation samples of the pages make_pdf.py draws, as the
model is shown them.

    python3 seg_pages.py                      # 25 pages -> seg_pages.pdf + seg_pages_labels.json
    python3 seg_pages.py --only 9             # page 9 alone (still page 9 of pages.pdf)

A page is a set of samples (seg_prompts.build_samples): the full page at
224 x 224 with its two prompts — level-2 objects and arrows — and then each
of its level-2 objects cut out of the 1000 x 1000 board, scaled and padded to
224 x 224, with its level-1 prompt. Every sample takes three PDF pages:

    1. its 224 x 224 image, pixel for pixel as the model sees it
    2. its ground truth over that image, each target numbered in target order —
       the level-2 objects beside the arrows for the full page, the level-1
       objects for a crop
    3. its prompts, their target strings, and the targets numbered as on 2

so a page with N level-2 objects takes 3 (N + 1). A drawing has no level-1
prompt; its crop is still shown, with an empty ground truth.

The label JSON holds, per page, every sample's transform (image = board * scale
+ offset) and prompts, each target with its exact board coordinates, its image
coordinates and the numbers it contributes to the target string. The seed and
page count default to make_pdf.py's, so page N here is page N of pages.pdf.
"""

from __future__ import annotations
import argparse
import json
import random
import sys
import textwrap

import numpy as np

from labels import COORD_ORDER
from make_pdf import build_pools
from page_generator import generate_page
from seg_prompts import (GRID, IMAGE_SIZE, BOARD_SIZE, INK_WIDTH_PX,
                         LEVEL2_ORDER, build_samples, class_words, render_board, sample_image)

SEG_COLORS = {
    "textblock": "#1f77b4", "table": "#8c564b", "geometry": "#9467bd", "drawing": "#2ca02c",
    "flowchart": "#bcbd22",
    "textline": "#17becf", "formula": "#e377c2", "vertex": "#d62728",
    "arc": "#2ca02c", "segment": "#1f77b4",
    "enclosure": "#d62728", "line_mark": "#ff7f0e", "arrow": "#555555",
}
TEXT_WIDTH = 118            # characters a prompt page wraps at


def _image_axes(fig, rect):
    ax = fig.add_axes(rect)
    ax.set_xlim(0, IMAGE_SIZE)
    ax.set_ylim(IMAGE_SIZE, 0)
    ax.set_aspect("equal")
    ax.set_xticks([0, IMAGE_SIZE])
    ax.set_yticks([0, IMAGE_SIZE])
    ax.tick_params(labelsize=5, length=2, colors="#888")
    for spine in ax.spines.values():
        spine.set_linewidth(0.4)
        spine.set_color("#aaa")
    return ax


def _show(ax, image, faded=False):
    ax.imshow(np.asarray(image), cmap="gray", vmin=0, vmax=255, interpolation="nearest",
              extent=(0, IMAGE_SIZE, IMAGE_SIZE, 0), alpha=0.3 if faded else 1.0)


def _arc_path(c, steps: int = 96):
    """The circle or arc a target's start, middle and end describe, as (xs, ys):
    from the start to the end, counterclockwise as the image is seen (y down)."""
    start, middle, end = (np.asarray(c[i:i + 2], dtype=float) for i in (0, 2, 4))
    if float(np.hypot(*(start - end))) < 1e-6:
        centre, sweep = (start + middle) / 2.0, 2 * np.pi      # a full circle
    else:
        (ax_, ay), (bx, by), (cx, cy) = start, middle, end
        d = 2.0 * (ax_ * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        if abs(d) < 1e-9:
            return [start[0], middle[0], end[0]], [start[1], middle[1], end[1]]
        ua, ub, uc = ax_ * ax_ + ay * ay, bx * bx + by * by, cx * cx + cy * cy
        centre = np.array([(ua * (by - cy) + ub * (cy - ay) + uc * (ay - by)) / d,
                           (ua * (cx - bx) + ub * (ax_ - cx) + uc * (bx - ax_)) / d])
        sweep = (np.arctan2(*(start - centre)[::-1]) - np.arctan2(*(end - centre)[::-1])) % (2 * np.pi)
    radius = float(np.hypot(*(start - centre)))
    # with y down, counterclockwise on the page is a falling angle
    angles = np.arctan2(*(start - centre)[::-1]) - np.linspace(0.0, sweep, steps)
    return centre[0] + radius * np.cos(angles), centre[1] + radius * np.sin(angles)


def _draw_targets(ax, targets):
    """Each target in its class colour, numbered in target order."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    for number, target in enumerate(targets, 1):
        cls, c = target["class"], target["image"]
        color = SEG_COLORS[cls]
        label = str(number)
        if cls == "vertex":
            ax.plot(c[0], c[1], "o", ms=2.2, color=color)
            at, va = (c[0] + 1.0, c[1] - 1.0), "bottom"
        elif cls == "arc":
            ax.plot(*_arc_path(c), "-", color=color, lw=0.6)
            ax.plot(c[0], c[1], "o", ms=1.6, color=color)         # the start
            ax.plot(c[2], c[3], "o", ms=1.0, color=color, alpha=0.6)
            at, va = (c[0] + 1.0, c[1] - 1.0), "bottom"
        elif cls == "segment":
            ax.plot([c[0], c[2]], [c[1], c[3]], "-", color=color, lw=0.6)
            at, va = ((c[0] + c[2]) / 2 + 1.0, (c[1] + c[3]) / 2 - 1.0), "bottom"
        elif cls == "arrow" and not target.get("no_direction"):
            ax.annotate("", xy=(c[2], c[3]), xytext=(c[0], c[1]),
                        arrowprops=dict(arrowstyle="-|>", color=color, lw=0.7,
                                        shrinkA=0, shrinkB=0, mutation_scale=7))
            ax.plot(c[0], c[1], "o", ms=1.4, color=color)
            at, va = (c[0] + 1.0, c[1] - 1.0), "bottom"
        else:
            ax.add_patch(Rectangle((c[0], c[1]), max(c[2] - c[0], 0.3), max(c[3] - c[1], 0.3),
                                   fill=False, ec=color, lw=0.6,
                                   ls="--" if target.get("no_direction") else "-"))
            at, va = (c[0] + 0.4, c[1] + 0.4), "top"
        ax.text(*at, label, fontsize=4.2, color=color, va=va, ha="left",
                clip_on=True)

    present = [cls for cls in SEG_COLORS if any(t["class"] == cls for t in targets)]
    handles = [Line2D([0], [0], color=SEG_COLORS[cls], lw=1.2, label=cls) for cls in present]
    if handles:
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02),
                  ncol=len(handles), fontsize=6, frameon=False)


def image_page(pdf, plt, image, title):
    fig = plt.figure(figsize=(8.5, 8.5))
    ax = _image_axes(fig, [0.06, 0.05, 0.88, 0.88])
    _show(ax, image)
    fig.suptitle(title, fontsize=8, color="#444", y=0.975)
    pdf.savefig(fig)
    plt.close(fig)


def page_truth_page(pdf, plt, image, sample, title):
    level2, arrows = sample["prompts"]
    fig = plt.figure(figsize=(11, 6.4))
    for rect, prompt, name in (([0.03, 0.1, 0.45, 0.78], level2, "level 2 objects"),
                               ([0.52, 0.1, 0.45, 0.78], arrows, "arrows")):
        ax = _image_axes(fig, rect)
        _show(ax, image, faded=True)
        _draw_targets(ax, prompt["targets"])
        ax.set_title(f"{name} ({len(prompt['targets'])})", fontsize=7, color="#444")
    fig.suptitle(title, fontsize=8, color="#444", y=0.97)
    pdf.savefig(fig)
    plt.close(fig)


def crop_truth_page(pdf, plt, image, sample, title):
    from matplotlib.patches import Rectangle

    fig = plt.figure(figsize=(8.5, 8.5))
    ax = _image_axes(fig, [0.06, 0.07, 0.88, 0.86])
    _show(ax, image, faded=True)
    (pad_x, pad_y), (w, h) = sample["transform"]["pad"], sample["transform"]["size"]
    # where the parent's box landed; the rest is padding
    ax.add_patch(Rectangle((pad_x, pad_y), w, h, fill=False, ec="#999", lw=0.5, ls=":"))
    if sample["prompts"]:
        _draw_targets(ax, sample["prompts"][0]["targets"])
    else:
        ax.text(IMAGE_SIZE / 2, IMAGE_SIZE / 2, "no level-1 prompt for a drawing",
                fontsize=9, color="#888", ha="center", va="center")
    fig.suptitle(title, fontsize=8, color="#444", y=0.975)
    pdf.savefig(fig)
    plt.close(fig)


def _columns(entries, width=TEXT_WIDTH):
    """Short entries laid out down columns, as lines."""
    if not entries:
        return []
    cell = max(len(e) for e in entries) + 3
    n_cols = max(1, width // cell)
    n_rows = -(-len(entries) // n_cols)
    return ["".join(entries[r + c * n_rows].ljust(cell)
                    for c in range(n_cols) if r + c * n_rows < len(entries)).rstrip()
            for r in range(n_rows)]


def prompt_page(pdf, plt, title, header, prompts, empty_note=None):
    lines = [title, ""] + header + [""]
    for n, prompt in enumerate(prompts, 1):
        lines += [f"PROMPT {n}"] + textwrap.wrap(prompt["prompt"], TEXT_WIDTH) + [""]
        lines += [f"TARGET  ({len(prompt['targets'])} objects)"]
        lines += textwrap.wrap(prompt["target"], TEXT_WIDTH) or ["(empty)"]
        lines += ["", "numbered as on the ground-truth page:"]
        lines += _columns([f"#{k:<3d} {class_words(t):<9s} "
                           + " ".join(str(v) for v in t["tokens"])
                           + (" (no direction)" if t.get("no_direction") else "")
                           for k, t in enumerate(prompt["targets"], 1)])
        lines += ["", ""]
    if not prompts and empty_note:
        lines.append(empty_note)

    width = max(len(line) for line in lines)
    size = min(8.0, 0.92 * 8.5 * 72 / (width * 0.6), 0.93 * 11 * 72 / (len(lines) * 1.2))
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.04, 0.97, "\n".join(lines), family="monospace", fontsize=size,
             va="top", ha="left", linespacing=1.2)
    pdf.savefig(fig)
    plt.close(fig)


def _order_lines(page_level: bool):
    coords = (f"coords  0-{GRID - 1} across the {IMAGE_SIZE}px image, "
              f"{'x before y' if COORD_ORDER == 'x0y0x1y1' else 'y before x'}; "
              "box = top-left bottom-right, vertex = one point, arrow = start end, "
              "arc = start middle end counterclockwise, segment = upper end, lower end")
    if page_level:
        order = (f"order   classes in prompt order; level 2 by {LEVEL2_ORDER} order; "
                 "arrows by exact (y, x) of their start")
    else:
        order = ("order   classes in prompt order; textblock/table lines in creation order; "
                 "geometry/flowchart lines, arcs, segments and vertices by exact (y, x); "
                 "enclosures and line marks "
                 "by the text they mark")
    return [coords, order]


def main():
    global IMAGE_SIZE, BOARD_SIZE
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=int, default=25)
    ap.add_argument("--only", type=int, default=None,
                    help="render just this page (1-based); earlier pages are still "
                         "generated so it matches that page of pages.pdf")
    ap.add_argument("--out", default="seg_pages.pdf")
    ap.add_argument("--labels", default="seg_pages_labels.json",
                    help="write every page's samples here ('' to skip)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--board", type=float, default=None,
                    help="board a page is laid out and rasterised on (default layout.PAGE_W)")
    ap.add_argument("--image", type=float, default=None,
                    help="size of every image a prompt is asked of (default layout.OUTPUT_SIZE)")
    ap.add_argument("--cell-size", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                    help="cell width/height range as a share of the writing area "
                         "(default layout.CELL_SIZE); smaller is denser")
    args = ap.parse_args()

    import layout as ly
    import seg_prompts as sp
    sp.configure(board=args.board, image=args.image)
    if args.cell_size:
        ly.CELL_SIZE = tuple(args.cell_size)
    IMAGE_SIZE, BOARD_SIZE = sp.IMAGE_SIZE, sp.BOARD_SIZE

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    random.seed(args.seed)
    pools = build_pools(with_shape_bank=True)
    if not pools["text_pool"]:
        sys.exit("No text pool — IAM-OnDB and DeepWriting are both missing; see README.md")

    wanted = [args.only] if args.only else list(range(1, args.pages + 1))
    all_labels = []
    counts: dict = {}
    pdf_pages = 0
    with PdfPages(args.out) as pdf:
        for number in range(1, max(wanted) + 1):
            page = generate_page(**pools)
            if number not in wanted:
                continue
            board = render_board(page["board_blocks"])
            samples = build_samples(page["board_blocks"])
            n_objects = len(samples) - 1

            for sample in samples:
                image = sample_image(board, sample["transform"])
                t = sample["transform"]
                if sample["image"] == "page":
                    name = f"page {number} — full page"
                    image_page(pdf, plt, image, f"{name} — input {IMAGE_SIZE}x{IMAGE_SIZE}")
                    page_truth_page(pdf, plt, image, sample, f"{name} — ground truth")
                    header = [f"image   full page: board {BOARD_SIZE}x{BOARD_SIZE} -> "
                              f"{IMAGE_SIZE}x{IMAGE_SIZE} (scale {t['scale'][0]:.3f}); "
                              f"{n_objects} level-2 objects follow"] + _order_lines(True)
                    prompt_page(pdf, plt, f"{name} — prompts", header, sample["prompts"])
                else:
                    parent = sample["parent"]
                    name = (f"page {number} — object {parent['index'] + 1}/{n_objects} "
                            f"({parent['class']})")
                    image_page(pdf, plt, image, f"{name} — input {IMAGE_SIZE}x{IMAGE_SIZE}")
                    crop_truth_page(pdf, plt, image, sample, f"{name} — ground truth")
                    x0, y0, x1, y1 = t["board_box"]
                    header = [f"image   {parent['class']} #{parent['index'] + 1} of the level-2 "
                              f"target: board box ({x0:.1f}, {y0:.1f})-({x1:.1f}, {y1:.1f}) -> "
                              f"{t['size'][0]}x{t['size'][1]} (scale {t['scale'][0]:.3f}), "
                              f"padded at ({t['pad'][0]}, {t['pad'][1]})"] + _order_lines(False)
                    prompt_page(pdf, plt, f"{name} — prompts", header, sample["prompts"],
                                empty_note="No level-1 prompt is defined for a drawing.")
                pdf_pages += 3
                for prompt in sample["prompts"]:
                    for target in prompt["targets"]:
                        counts[target["class"]] = counts.get(target["class"], 0) + 1

            all_labels.append({
                "page_index": number - 1,
                "board_size": BOARD_SIZE, "image_size": IMAGE_SIZE, "grid": GRID,
                "coord_order": COORD_ORDER, "level2_order": LEVEL2_ORDER,
                "ink_width_px": INK_WIDTH_PX, "cell_size": list(ly.CELL_SIZE),
                "samples": samples,
            })
            print(f"  page {number}: {n_objects} level-2 objects -> {3 * (n_objects + 1)} PDF pages")

    print(f"Wrote {len(all_labels)} page(s), {pdf_pages} PDF pages -> {args.out}")
    if args.labels:
        with open(args.labels, "w") as f:
            json.dump(all_labels, f)
        print(f"Wrote samples for {len(all_labels)} page(s) -> {args.labels}")
    print("  targets by class:",
          ", ".join(f"{k}={v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1])))
    undirected = sum(1 for p in all_labels for s in p["samples"] for q in s["prompts"]
                     for t in q["targets"] if t.get("no_direction"))
    if undirected:
        print(f"  arrows with no start/end found: {undirected}")


if __name__ == "__main__":
    main()
