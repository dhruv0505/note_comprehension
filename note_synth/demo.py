"""
demo.py — sanity-check the pipeline end-to-end, on the real datasets when they
are present under data/ and on lightweight MOCK stand-ins otherwise.

    python demo.py            # real data where available, mocks for the rest
    python demo.py --mock     # mocks only (fast, no data files needed)

Each pool degrades independently: a missing archive just falls back to its mock
(or to the procedural diagram builder for OHFD), so you can start generating
pages with whichever datasets you've downloaded so far. See README.md for what
goes in data/.

Everything downstream (block_generators, layout, page_generator, labels) is
dataset-agnostic and doesn't change.
"""

import random
import sys

from core import Stroke, Ink

random.seed(0)

# Enough content for varied pages without spending a minute parsing 12k line
# XMLs and a multi-GB QuickDraw dump on every run.
# Enough lines that a single writer has a full page's worth (see
# page_generator.MIN_LINES_PER_WRITER) without a slow startup.
DEMO_LIMITS = {"iam": 1500, "deepwriting": 1500, "quickdraw": 40, "math": 60,
               "ohfd": 30, "emnist": 8}


def _fake_cursive_stroke(n_words=1, seed_x=0.0):
    """Rough zig-zag stand-in for a handwritten word/line — NOT real handwriting,
    purely to exercise geometry/layout code paths."""
    pts = []
    x = seed_x
    for _ in range(n_words * random.randint(3, 6)):
        x += random.uniform(3, 7)
        y = random.uniform(-3, 3)
        pts.append((x, y))
    return Stroke(x=[p[0] for p in pts], y=[p[1] for p in pts])


def mock_text_pool(n_lines=40) -> list:
    """Stand-in for load_iam_ondb()/load_deepwriting() output: line Inks with
    word children."""
    pool = []
    for _ in range(n_lines):
        n_words = random.randint(3, 8)
        words = []
        x = 0.0
        for _ in range(n_words):
            s = _fake_cursive_stroke(1, seed_x=x)
            words.append(Ink(strokes=[s], class_name="word", source="mock_text"))
            x = max(s.x) + random.uniform(4, 8)
        all_strokes = []
        line = Ink(strokes=[], class_name="textline", children=words, source="mock_text")
        pool.append(line)
    return pool


def mock_quickdraw_pool(n=20) -> list:
    pool = []
    for _ in range(n):
        n_strokes = random.randint(1, 3)
        strokes = []
        for _ in range(n_strokes):
            n_pts = random.randint(5, 15)
            xs = [random.uniform(0, 40) for _ in range(n_pts)]
            ys = [random.uniform(0, 40) for _ in range(n_pts)]
            strokes.append(Stroke(x=xs, y=ys))
        pool.append(Ink(strokes=strokes, class_name="drawing", label="mock_sketch", source="mock_quickdraw"))
    return pool


CLASS_COLORS = {"textblock": "#000", "table": "#8b4513",
                "geometry": "#5b2d8e",
                "drawing": "#2ca02c",
                "enclosure": "#1f77b4", "line_mark": "#ff7f0e", "arrow": "#1f77b4",
                "word": "#000", "textline": "#000", "unknown": "#9467bd"}


def render_svg(page: dict, path: str, color_by_class: bool = False):
    """Renders the page as plain black ink strokes on a plain background — this
    is also the correct representation for the segmentation TRAINING input
    itself (the paper's Eq. 2 time/distance color-channel rendering is specific
    to the recognition task, not segmentation). For real training you'd want
    nicer stroke rendering (proper pen width, antialiasing, maybe slight blur)
    rather than raw SVG polylines, but the underlying idea — plain rendering, no
    color encoding — is correct as-is.

    `color_by_class` tints strokes by taxonomy class. That is a debugging aid for
    eyeballing which block a stroke ended up in — never use it to render training
    input, since it leaks the labels into the image. Nothing in the Ink model
    carries color; this is purely a property of this renderer.
    """
    w, h = page["page_w"], page["page_h"]
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">']
    lines.append(f'<rect x="0" y="0" width="{w}" height="{h}" fill="white" stroke="black"/>')

    def walk(node: Ink):
        color = CLASS_COLORS.get(node.class_name, "#000") if color_by_class else "#000"
        for s in node.strokes:
            pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(s.x, s.y))
            lines.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.2"/>')
        for c in node.children:
            walk(c)

    for b in page["blocks"]:
        walk(b)

    lines.append("</svg>")
    with open(path, "w") as f:
        f.write("\n".join(lines))


def _try_load(name, fn, *args, **kwargs):
    """Load one pool, reporting what happened instead of failing the whole demo.

    A missing archive is the expected case while you're still downloading
    datasets, so it's reported like any other empty pool rather than raised.
    """
    try:
        pool = fn(*args, **kwargs)
    except FileNotFoundError as exc:
        print(f"  {name:12s} skipped ({exc})")
        return []
    except ImportError as exc:
        print(f"  {name:12s} skipped (missing dependency: {exc})")
        return []
    print(f"  {name:12s} {len(pool)} items")
    return pool


def load_real_pools() -> dict:
    """Build the four generator pools from whatever is present under data/."""
    import loaders

    print("Loading datasets from", loaders.DATA_DIR)
    text_pool = (_try_load("iam-ondb", loaders.load_iam_ondb,
                           max_items=DEMO_LIMITS["iam"]) +
                 _try_load("deepwriting", loaders.load_deepwriting,
                           max_items=DEMO_LIMITS["deepwriting"]))
    return {
        "text_pool": text_pool,
        # a few hundred per category keeps the sketch pool balanced instead of
        # exhausting the item budget on whatever sorts first alphabetically
        "quickdraw_pool": _try_load("quickdraw", loaders.load_quickdraw,
                                    max_per_category=DEMO_LIMITS["quickdraw"]),
        "ohfd_pool": _try_load("ohfd", loaders.load_ohfd,
                               max_items=DEMO_LIMITS["ohfd"]),
        "math_pool": _try_load("mathwriting", loaders.load_mathwriting,
                               max_items=DEMO_LIMITS["math"]),
        # keyed by character rather than a flat list: it is what a geometry
        # figure looks up to write the letter naming a corner
        "glyph_pool": _try_load("emnist", loaders.load_emnist_pool,
                                max_per_class=DEMO_LIMITS["emnist"]),
    }


if __name__ == "__main__":
    from page_generator import generate_page

    if "--mock" in sys.argv:
        pools = {"text_pool": [], "quickdraw_pool": [], "ohfd_pool": [],
                 "math_pool": [], "glyph_pool": None}
    else:
        pools = load_real_pools()

    # mocks stand in for whichever pools came back empty, so the demo always runs
    if not pools["text_pool"]:
        print("  using mock text pool")
        pools["text_pool"] = mock_text_pool()
    if not pools["quickdraw_pool"]:
        print("  using mock quickdraw pool")
        pools["quickdraw_pool"] = mock_quickdraw_pool()

    page = generate_page(**pools)

    print("=== level-2 target string ===")
    print(page["targets_by_level"][2][:300], "...")
    print("\n=== level-1 target string ===")
    print(page["targets_by_level"][1][:300], "...")
    print("\n=== level-0 target string ===")
    print(page["targets_by_level"][0][:300], "...")

    with open("page_labels.json", "w") as f:
        f.write(page["labels_json"])

    render_svg(page, "page_preview.svg")
    print("\nWrote page_labels.json and page_preview.svg")
