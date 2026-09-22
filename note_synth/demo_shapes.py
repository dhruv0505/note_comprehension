"""
demo_shapes.py — show what shape_mining.py finds, one panel per example.

    python demo_shapes.py                      # 8 examples per category
    python demo_shapes.py --per-kind 12 --per-pool 200
    python demo_shapes.py --worst              # the marginal hits, not the best

Each panel draws the ink itself in black, the fitted primitive in red, and the
measurement that qualified it, and its height-over-width, which is metadata the
detector does not classify on. The fit is drawn because a number alone cannot
tell you whether a threshold is doing what you think: a chord that runs down the
middle of the stroke, a rectangle flush with the ink, an ellipse concentric with
a loop are all visible at a glance, and a bad one is obvious immediately.

By default it samples across the score range rather than showing the eight
best-fitting shapes, since a wall of perfect examples says nothing about where
the thresholds actually sit. `--worst` shows the marginal end alone.
"""

from __future__ import annotations
import argparse
import math

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import shape_mining as S

KINDS = ("line", "rectangle", "oval")


def _spread(hits, count):
    """Take `count` hits spanning the score range, best first.

    Showing the top N would fill the sheet with the same near-perfect shape and
    hide whatever is scraping past the threshold, which is the part worth
    looking at.
    """
    ranked = sorted(hits, key=lambda h: -h.score)
    if len(ranked) <= count:
        return ranked
    picks = np.linspace(0, len(ranked) - 1, count).round().astype(int)
    return [ranked[i] for i in dict.fromkeys(picks.tolist())]


def _draw_fit(ax, hit):
    """Overlay the primitive that the hit was matched against."""
    points = hit.points
    if hit.kind == "line":
        ax.plot([points[0, 0], points[-1, 0]], [points[0, 1], points[-1, 1]],
                "-", color="#d62728", lw=1.0, alpha=0.9, zorder=3)
        ax.plot(points[[0, -1], 0], points[[0, -1], 1], "o", color="#d62728",
                ms=2.5, zorder=4)
    elif hit.kind == "rectangle":
        fitted = S.min_area_rect(points)
        if fitted is not None:
            corners = fitted[3]
            loop = np.vstack([corners, corners[:1]])
            ax.plot(loop[:, 0], loop[:, 1], "-", color="#d62728", lw=1.0,
                    alpha=0.9, zorder=3)
    elif hit.kind == "oval":
        fitted = S.fit_ellipse(points)
        if fitted is not None:
            (cx, cy), (a, b), angle = fitted
            theta = np.linspace(0, 2 * math.pi, 160)
            outline = np.column_stack([a * np.cos(theta), b * np.sin(theta)])
            rotation = np.array([[math.cos(angle), -math.sin(angle)],
                                 [math.sin(angle), math.cos(angle)]])
            outline = outline @ rotation.T
            ax.plot(cx + outline[:, 0], cy + outline[:, 1], "-", color="#d62728",
                    lw=1.0, alpha=0.9, zorder=3)
            ax.plot([cx], [cy], "+", color="#d62728", ms=5, zorder=4)


KEY_METRIC = {"line": ("residual", "resid"), "oval": ("radial_cv", "cv"),
              "rectangle": ("residual", "resid")}


def draw_sheet(hits, per_kind, path, title):
    kinds = [k for k in KINDS if any(h.kind == k for h in hits)]
    if not kinds:
        print("nothing found")
        return

    fig, axes = plt.subplots(len(kinds), per_kind,
                             figsize=(per_kind * 1.7, len(kinds) * 2.0),
                             squeeze=False)
    for row, kind in enumerate(kinds):
        chosen = _spread([h for h in hits if h.kind == kind], per_kind)
        for col in range(per_kind):
            ax = axes[row][col]
            ax.axis("off")
            if col >= len(chosen):
                continue
            hit = chosen[col]
            # each stroke on its own: drawing hit.points as one polyline joins
            # the end of one stroke to the start of the next with ink that was
            # never written, which reads as a diagonal slashed across the shape
            for part in (hit.parts or [hit.points]):
                ax.plot(part[:, 0], part[:, 1], "-", color="#111", lw=1.2,
                        solid_capstyle="round", zorder=2)
            _draw_fit(ax, hit)
            ax.set_aspect("equal")
            ax.invert_yaxis()       # the pools are y-down

            key, short = KEY_METRIC[kind]
            value = hit.metrics.get(key, float("nan"))
            caption = f"{hit.source}\n{short}={value:.3f}  s={hit.score:.2f}"
            # h/w is metadata, not a threshold: it is what tells a square from an
            # oblong and a circle from an ellipse, now that neither is its own
            # category, so the sheet has to show it
            if "hw_ratio" in hit.metrics:
                caption += f"  h/w={hit.metrics['hw_ratio']:.2f}"
            if hit.label:
                caption = f"{str(hit.label)[:14]}\n" + caption
            ax.set_title(caption, fontsize=5.5, pad=2)

        axes[row][0].set_ylabel(kind)
        # row label, placed in the axes' own corner so it survives tight_layout
        axes[row][0].text(-0.08, 0.5, f"{kind}\n({sum(1 for h in hits if h.kind == kind)})",
                          transform=axes[row][0].transAxes, rotation=90,
                          va="center", ha="center", fontsize=7)

    fig.suptitle(title, fontsize=9)
    plt.tight_layout(rect=(0.02, 0, 1, 0.97))
    plt.savefig(path, dpi=170)
    plt.close(fig)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--per-kind", type=int, default=8)
    ap.add_argument("--per-pool", type=int, default=400,
                    help="cap items per pool; None-ish values mine everything")
    ap.add_argument("--out", default="shapes_demo.png")
    ap.add_argument("--worst", action="store_true",
                    help="show the marginal hits instead of a spread")
    args = ap.parse_args()

    import make_pdf
    from collections import Counter

    pools = make_pdf.build_pools()
    hits = S.mine(pools, per_pool=args.per_pool)

    counts = Counter(h.kind for h in hits)
    print("\nfound: " + ", ".join(f"{k}={counts[k]}" for k in KINDS if counts[k]))
    for kind in KINDS:
        sources = Counter(h.source for h in hits if h.kind == kind)
        if sources:
            print(f"  {kind:10s} " + ", ".join(f"{s}={n}" for s, n in sources.most_common()))

    if args.worst:
        hits = [h for h in hits if h.score <= 0.35]
        title = "marginal hits (score <= 0.35): ink in black, fitted primitive in red"
    else:
        title = "mined shapes: ink in black, fitted primitive in red"
    draw_sheet(hits, args.per_kind, args.out, title)


if __name__ == "__main__":
    main()
