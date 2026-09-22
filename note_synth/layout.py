"""
layout.py — build the page's grid of cells.

A page is tiled with cells grown from corners: starting at the top-left, each
cell gets a random width and height (cut off at the page edge, never under 15%
of the writing area) and seeds new cells at its top-right and bottom-left, until
the writing area is full. A line
across the page crosses about 3.5 cells, and because cells in one row need not
line up with the next, a tall cell can sit beside several short ones.

Two things this deliberately does NOT do, both of which earlier versions got
wrong:

  * It does not stack every block in one column down the page. Cells side by
    side give the sideways variety real notes have — a sketch beside a
    paragraph, a wide diagram spanning the full width under them.

  * It does not stretch text to fill its cell. Doing that made handwriting size
    depend on how big a cell happened to be, so one page could hold 8px and 40px
    writing side by side. A single page-wide scale (see page_text_scale) applies
    to all of it, and the *content* adapts to the cell instead: text re-wraps to
    the cell width and takes as many lines as its height allows.
"""

from __future__ import annotations
from typing import List, Optional, Tuple
import heapq
import random

from block_generators import LINE_HEIGHT, LINE_GAP


# The page is square, and it is laid out at this working size — every size in
# the generators (handwriting, margins, gaps, stroke lengths) is in these units
# — then scaled down to OUTPUT_SIZE once it is finished (page_generator
# .generate_page). Building it small instead would have meant re-tuning all of
# them, and a 224-unit page cannot hold the detail they were tuned to draw.
PAGE_W = 1000.0
PAGE_H = 1000.0
OUTPUT_SIZE = 224.0     # the finished page's width and height, labels included
MARGIN = 40.0

# Every cell's width and height are drawn independently from this range, as a
# fraction of the writing area's width / height. The lower end is also a hard
# minimum: a cell cut short by the page edge (or the next cell) is kept only if
# it is still this big, and otherwise its space is shared out equally among the
# cells before it in that row or column. With these bounds a straight line across
# the writing area crosses 3.5 cells on average (3.49 over 400 simulated pages,
# rows and columns alike), with 9-19 cells a page.
CELL_SIZE = (0.15, 0.39)

GUTTER = 16.0           # blank space between neighbouring cells
CELL_PAD = (0.0, 0.10)  # extra slack inside a cell, as a fraction of its size

# One handwriting size per page, shared by every cell: it varies between pages
# (different writer, different pen) but never within one.
LINE_PX_RANGE = (15.0, 23.0)


_EPS = 1e-6


def _plan_run(avail: float, extent: float, rng: random.Random) -> List[float]:
    """Sizes for cells laid end to end (a gutter apart) across exactly `avail`.

    Sizes are drawn from CELL_SIZE (scaled by `extent`) until the run reaches
    `avail`; the last is cut short there. If the cut leaves it under the minimum
    it is dropped and its space divided equally among the cells before it. An
    empty list means not even one minimum-size cell fits.
    """
    lo, hi = CELL_SIZE[0] * extent, CELL_SIZE[1] * extent
    sizes: List[float] = []
    end = -GUTTER
    while True:
        start = end + GUTTER
        size = rng.uniform(lo, hi)
        if start + size < avail - _EPS:
            sizes.append(size)
            end = start + size
            continue
        last = avail - start
        if last >= lo - _EPS:
            sizes.append(last)
        elif sizes:
            spare = (avail - (sum(sizes) + GUTTER * (len(sizes) - 1))) / len(sizes)
            sizes = [s + spare for s in sizes]
        return sizes


def build_grid(page_w: float = PAGE_W, page_h: float = PAGE_H, margin: float = MARGIN,
                rng: random.Random = random) -> List[Tuple[float, float, float, float]]:
    """The page's cells, in reading order: top to bottom, left to right.

    Cells grow from corners. The first sits at the top-left of the writing area;
    each cell seeds the next one at its top-right and at its bottom-left (a
    gutter away). Seeds are taken top to bottom, left to right, so every cell
    already placed starts at or above the current seed: a seed landing on a cell
    slides right past it, and a row only ever stops short on its right, where a
    taller neighbour from an earlier row may be standing.

    Rows and columns are sized as runs (see _plan_run) so that no cell is under
    the minimum. A row is planned in full from the seed it starts at, up to the
    page edge or the next cell. A column is planned when its top cell is placed,
    and the plan travels down with each bottom-left seed; if that seed gets
    pushed sideways by another cell, the column is cut there and a fresh one is
    planned from the new position.
    """
    left, top = margin, margin
    right, bottom = page_w - margin, page_h - margin
    usable_w, usable_h = right - left, bottom - top

    boxes: List[Tuple[float, float, float, float]] = []
    # (y, x, the rest of the column's planned heights)
    seeds: List[Tuple[float, float, Tuple[float, ...]]] = [(top, left, ())]
    while seeds:
        y, x, column = heapq.heappop(seeds)

        # Slide right past any cell (with its gutter) the seed falls on.
        moved = True
        while moved:
            moved = False
            for bx0, by0, bx1, by1 in boxes:
                if (bx0 - _EPS <= x < bx1 + GUTTER - _EPS
                        and by0 - _EPS <= y < by1 + GUTTER - _EPS):
                    x = bx1 + GUTTER
                    column = ()
                    moved = True

        # The nearest cell to the right on this line bounds the row.
        limit = right
        for bx0, by0, bx1, by1 in boxes:
            if by0 - _EPS <= y < by1 + GUTTER - _EPS and bx0 > x + _EPS:
                limit = min(limit, bx0 - GUTTER)

        for i, w in enumerate(_plan_run(limit - x, usable_w, rng)):
            heights = column if i == 0 and column else _plan_run(bottom - y, usable_h, rng)
            if heights:
                h = heights[0]
                boxes.append((x, y, x + w, y + h))
                if len(heights) > 1:
                    heapq.heappush(seeds, (y + h + GUTTER, x, tuple(heights[1:])))
            x += w + GUTTER

    cells: List[Tuple[float, float, float, float]] = []
    for x0, y0, x1, y1 in boxes:
        # a little slack inside the cell so the page doesn't look tiled
        px = (x1 - x0) * rng.uniform(*CELL_PAD)
        py = (y1 - y0) * rng.uniform(*CELL_PAD)
        cells.append((x0 + px, y0 + py, x1 - px, y1 - py))
    return cells


def page_text_scale(rng: random.Random = random) -> float:
    """The single handwriting scale this page uses, in page units per text unit."""
    return rng.uniform(*LINE_PX_RANGE) / LINE_HEIGHT


def cell_capacity(cell, scale: float) -> Tuple[float, float]:
    """A cell's size in text units — what the content generators lay out in."""
    x0, y0, x1, y1 = cell
    return (x1 - x0) / scale, (y1 - y0) / scale


def min_text_height(scale: float) -> float:
    """Shortest cell worth giving to a textblock, in page units: one row of writing.

    There is no matching width floor. A narrow textblock leans to formulas
    instead (page_generator.TEXT_HEAVY_WIDTH_XH): a prose line is cut where it
    meets the cell's edge, and a median IAM line is 22 x-heights wide (p25 is
    17), while a formula is scaled to fit.
    """
    return (LINE_HEIGHT + LINE_GAP) * scale
