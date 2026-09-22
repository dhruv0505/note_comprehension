"""
test_recog.py — recog_synth's tests. Siloed: nothing from note_synth is imported,
no dataset is read, and every file is written under a temporary directory.

    python -m unittest test_recog          # from recog_synth/
"""

from __future__ import annotations
import sys

sys.dont_write_bytecode = True

import json
import os
import random
import tempfile
import unittest

import numpy as np
from PIL import Image

import recog_layout as rl
import recog_marks as rm
import recog_pages as rpg
import recog_pools as rp
import recog_synthesize as rs


class Stroke:
    """A stroke with one time per point: `ts`, or the points' indices."""

    def __init__(self, xs, ys, ts=None):
        self.x, self.y = [float(v) for v in xs], [float(v) for v in ys]
        self.t = [float(v) for v in (range(len(self.x)) if ts is None else ts)]


class Untimed:
    def __init__(self, xs, ys):
        self.x, self.y = [float(v) for v in xs], [float(v) for v in ys]


class Ink:
    def __init__(self, strokes, label, item=None):
        self.strokes, self.label, self.children = strokes, label, []
        self.meta = {"item": item} if item else None

    def all_strokes(self):
        return list(self.strokes)


def ink(width, height, label="text", item=None):
    """A diagonal spanning width x height exactly, written first, then three wavy strokes across it."""
    strokes = [Stroke([0, width], [0, height], [0, 1])]
    for k in range(3):
        xs = np.linspace(k * width / 3, (k + 1) * width / 3, 25)
        ys = height / 2 + 0.4 * height * np.sin(np.linspace(0, 6, 25) + k)
        strokes.append(Stroke(xs, ys, 100 * (k + 1) + np.arange(25)))
    return Ink(strokes, label, item)


def dark(image, threshold=250):
    """Where there is ink: any channel below `threshold`, as the background is white."""
    pixels = np.asarray(image)
    return (pixels < threshold).any(axis=-1) if pixels.ndim == 3 else pixels < threshold


class RowTops(unittest.TestCase):
    def test_defaults_centre_the_stack(self):
        tops = rl.row_tops()
        self.assertEqual(tops, [19 + 75 * i for i in range(10)])
        self.assertEqual(rl.CANVAS - (tops[-1] + rl.ROW_HEIGHT), 19)

    def test_odd_remainder_goes_below(self):
        tops = rl.row_tops(canvas=769)
        self.assertEqual((tops[0], 769 - (tops[-1] + 55)), (19, 20))

    def test_rows_that_do_not_fit_are_refused(self):
        with self.assertRaises(ValueError):
            rl.row_tops(rows=11)


class InkColour(unittest.TestCase):
    def test_channels_follow_the_formula(self):
        strokes = [Stroke([0, 3, 9], [0, 4, 4], [10, 15, 18]), Stroke([9], [10], [30])]
        colours = rl.ink_colours(rl.polylines(strokes))
        # t from 10, over 20; |dx| steps 3 and 6 (the last point keeps 6), over 6;
        # |dy| steps 4 and 0 (the last point keeps 0), over 4; the one-point stroke has no step
        expected = [[(0.0, 0.5, 1.0), (0.25, 1.0, 0.0), (0.4, 1.0, 0.0)], [(1.0, 0.0, 0.0)]]
        self.assertEqual([c.dtype for c in colours], [np.uint8, np.uint8])
        for got, want in zip(colours, expected):
            np.testing.assert_allclose(got, np.array(want) * 255, atol=1)

    def test_time_counts_from_the_earliest_point_whatever_the_stroke_order(self):
        strokes = [Stroke([0, 10], [0, 0], [50, 60]), Stroke([0, 10], [5, 5], [40, 45])]
        colours = rl.ink_colours(rl.polylines(strokes))
        self.assertEqual((colours[1][0, 0], colours[0][0, 0], colours[0][1, 0]), (0, 128, 255))

    def test_flat_and_still_items_have_zero_channels(self):
        colours = rl.ink_colours(rl.polylines([Stroke([5, 5], [5, 5], [3, 3])]))
        self.assertEqual(colours[0].tolist(), [[0, 0, 0], [0, 0, 0]])

    def test_item_without_times_is_refused(self):
        for strokes in ([Untimed([0, 1], [0, 1])], [Stroke([0, 1], [0, 1], [0])]):
            with self.assertRaises(ValueError):
                rl.polylines(strokes)
        with self.assertRaises(ValueError):
            rl.text_sample([Ink([Untimed([0, 50], [0, 20])], "x")], [3.0])

    def test_later_stroke_is_drawn_over_an_earlier_one(self):
        across = Stroke([0, 100], [50, 50], [0, 1])        # written first: green
        down = Stroke([50, 50], [0, 100], [100, 101])      # written last: red and blue
        image, _ = rl.drawing_sample(Ink([down, across], "plus"), 8.0)   # listed last-first
        r, g, b = image.getpixel((383, 383))
        self.assertGreater(r, 200)
        self.assertLess(g, 60)
        self.assertGreater(b, 200)


class RowFit(unittest.TestCase):
    def test_item_is_row_height_pen_included(self):
        lines = rl.polylines(ink(300, 60))
        placement = rl.row_fit(lines, pen=3.0)
        w, h = placement["size"]
        self.assertEqual(h, 55)
        self.assertLess(w, 768)
        self.assertEqual(placement["pen"], 3.0)
        image = rl.draw(lines, placement)
        self.assertEqual(image.mode, "RGB")
        mask = dark(image)
        self.assertEqual(mask.shape, (55, w))
        self.assertTrue(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any())

    def test_wide_item_shrinks_ink_and_pen_together(self):
        placement = rl.row_fit(rl.polylines(ink(5000, 60)), pen=3.0)
        full_scale = (55 - 3.0) / 60
        shrink = 768 / (5000 * full_scale + 3.0)
        self.assertEqual(placement["size"][0], 768)
        self.assertLess(placement["size"][1], 55)
        self.assertAlmostEqual(placement["scale"], full_scale * shrink)
        self.assertAlmostEqual(placement["pen"], 3.0 * shrink)

    def test_degenerate_items_still_draw(self):
        dot = Ink([Stroke([5], [5]), Stroke([5, 5], [5, 5])], ".")
        flat = Ink([Stroke([0, 100], [7, 7])], "-")
        for item in (dot, flat):
            image, rows = rl.text_sample([item], [3.0])
            self.assertTrue(dark(image).any())
            self.assertLessEqual(rows[0]["box"][2], 768)


class TextSample(unittest.TestCase):
    def setUp(self):
        widths = [200, 400, 900, 3000, 150, 600, 700, 250, 5000, 320]
        self.items = [ink(w, 50 + 7 * i, label=f"line {i}") for i, w in enumerate(widths)]
        self.image, self.rows = rl.text_sample(self.items, [2.5] * 10)

    def test_image(self):
        self.assertEqual(self.image.size, (768, 768))
        self.assertEqual(self.image.mode, "RGB")
        self.assertEqual(self.image.getpixel((767, 0)), rl.BACKGROUND)

    def test_ink_only_in_slots_left_aligned(self):
        mask = dark(self.image)
        tops = rl.row_tops()
        slots = np.zeros_like(mask)
        for top in tops:
            slots[top:top + 55] = True
        self.assertFalse((mask & ~slots).any(), "ink in a gap or margin")
        for top, row in zip(tops, self.rows):
            band = mask[top:top + 55]
            columns = np.flatnonzero(band.any(axis=0))
            self.assertLessEqual(columns[0], 2)
            x0, y0, x1, y1 = row["box"]
            self.assertTrue(top <= y0 and y1 <= top + 55)
            if not row["shrunk"]:
                self.assertEqual(y1 - y0, 55)

    def test_only_the_too_wide_rows_shrink(self):
        self.assertEqual([i for i, row in enumerate(self.rows) if row["shrunk"]], [3, 8])
        for i in (3, 8):
            self.assertEqual(self.rows[i]["box"][2], 768)
            box = self.rows[i]["box"]
            top = rl.row_tops()[i]
            self.assertLessEqual(abs((box[1] - top) - (top + 55 - box[3])), 1)   # centred in its slot


class Targets(unittest.TestCase):
    def test_rows_joined_by_newlines(self):
        labels = [f"line {i}" for i in range(10)]
        target = rl.text_target(labels)
        self.assertEqual(rl.SEP, "\n")
        self.assertEqual(target.count("\n"), 9)
        self.assertEqual(target.split("\n"), labels)

    def test_clean_label(self):
        self.assertEqual(rl.clean_label("  a   b \n"), "a b")
        self.assertEqual(rl.clean_label("two\nlines"), "two lines")
        self.assertEqual(rl.clean_label(r"\frac{a}{b}\ c"), r"\frac{a}{b}\ c")
        self.assertEqual(rl.clean_label("a < b > c"), "a < b > c")
        for bad in (None, "", "   ", "x<sep>y", "a</s>", "<pad>", "at <loc_12>", "<od>x"):
            self.assertIsNone(rl.clean_label(bad))
        with self.assertRaises(ValueError):
            rl.text_target(["a", ""])

    def test_prompts(self):
        self.assertEqual(rl.PROMPTS["textline"], "What is written in this image in English?")
        self.assertEqual(rl.PROMPTS["formula"], "What is written in this image in LaTeX?")
        self.assertEqual(rl.PROMPTS["drawing"], "What is drawn in this image?")


class DrawingSample(unittest.TestCase):
    def test_longer_side_fills_canvas_centred(self):
        image, placed = rl.drawing_sample(ink(300, 120, label="cat"), 6.0)
        mask = dark(image)
        columns, rows = np.flatnonzero(mask.any(axis=0)), np.flatnonzero(mask.any(axis=1))
        self.assertLessEqual(columns[0], 2)
        self.assertGreaterEqual(columns[-1], 765)
        self.assertLess(abs((rows[-1] - rows[0] + 1) - (120 * (768 - 6.0) / 300 + 6.0)), 3)
        self.assertLess(abs((rows[0] + rows[-1]) / 2 - 767 / 2), 2)

    def test_tall_drawing(self):
        image, placed = rl.drawing_sample(ink(50, 400, label="tree"), 4.0)
        self.assertEqual(placed["box"][3] - placed["box"][1], 768)


class Pools(unittest.TestCase):
    def test_note_synth_not_imported(self):
        for name in ("core", "loaders", "dataset_splits", "mathwriting_store"):
            self.assertNotIn(name, sys.modules)

    def test_list_pool_leaves_out_unusable_items(self):
        inks = [ink(100, 40, "ok", "iam:a:0"), ink(100, 40, "", "iam:a:1"),
                ink(100, 40, "bad<sep>", "iam:a:2"), Ink([], "no ink", "iam:a:3"), ink(100, 40, "also  ok")]
        pool = rp.ListPool(inks, "textline", "train", "iam")
        self.assertEqual(len(pool), 2)
        self.assertEqual([pool.label(i) for i in range(2)], ["ok", "also ok"])
        self.assertEqual([pool.item_id(i) for i in range(2)], ["iam:a:0", "iam:4"])

    def test_math_items(self):
        class Store:
            labels = ["x^2", "", "a<sep>b", r"\sqrt{2}"]

            def ink(self, i):
                return ink(50, 30, self.labels[i])

            def item_id(self, i):
                return f"mw:val:{i}"

        pool = rp.MathItems(Store(), "val")
        self.assertEqual(len(pool), 2)
        self.assertEqual([pool.item_id(i) for i in range(2)], ["mw:val:0", "mw:val:3"])
        self.assertEqual(pool.label(1), r"\sqrt{2}")
        self.assertEqual(pool.ink(1).label, r"\sqrt{2}")


class Dealing(unittest.TestCase):
    def test_a_pass_uses_every_item_once(self):
        dealer = rs.Dealer(50, 10, 0, "train", "textline")
        self.assertEqual(sorted(i for s in range(5) for i in dealer.items(s)), list(range(50)))

    def test_same_sample_same_items_whatever_the_order(self):
        a = rs.Dealer(37, 10, 0, "train", "textline")
        b = rs.Dealer(37, 10, 0, "train", "textline")
        forward = [a.items(s) for s in range(20)]
        self.assertEqual(forward, [b.items(s) for s in reversed(range(20))][::-1])
        for other in (rs.Dealer(37, 10, 1, "train", "textline"), rs.Dealer(37, 10, 0, "val", "textline")):
            self.assertNotEqual(forward, [other.items(s) for s in range(20)])

    def test_no_repeat_within_a_sample(self):
        dealer = rs.Dealer(13, 10, 0, "train", "formula")
        for s in range(200):
            self.assertEqual(len(set(dealer.items(s))), 10)

    def test_pool_too_small(self):
        with self.assertRaises(ValueError):
            rs.Dealer(9, 10, 0, "train", "textline")

    def test_short_samples_still_use_every_item_once_a_pass(self):
        dealer = rs.Dealer(500, 10, 0, "train", "textline", short_prob=0.3)
        samples = dealer.samples_for(500)
        counts = [dealer.count(s) for s in range(samples)]
        self.assertTrue(all(1 <= c <= 10 for c in counts))
        self.assertGreater(sum(c < 10 for c in counts), 5)
        dealt = [i for s in range(samples) for i in dealer.items(s)]
        self.assertEqual([len(dealer.items(s)) for s in range(samples)], counts)
        self.assertEqual(sorted(dealt[:500]), list(range(500)))
        self.assertLess(sum(counts[:-1]), 500)                      # the last sample is the one that reaches it
        again = rs.Dealer(500, 10, 0, "train", "textline", short_prob=0.3)
        self.assertEqual([again.items(s) for s in reversed(range(samples))][::-1],
                         [dealer.items(s) for s in range(samples)])

    def test_no_short_samples_deals_as_before(self):
        plain, zero = rs.Dealer(37, 10, 0, "train", "textline"), rs.Dealer(37, 10, 0, "train", "textline", short_prob=0.0)
        self.assertEqual([plain.items(s) for s in range(12)], [zero.items(s) for s in range(12)])
        self.assertEqual(zero.samples_for(25), 3)
        self.assertEqual(zero.samples_for(20), 2)


class ShortLayout(unittest.TestCase):
    def test_fewer_items_fill_the_top_slots(self):
        items = [ink(300, 50, label=f"line {i}") for i in range(3)]
        image, rows = rl.text_sample(items, [3.0] * 3, slots=10)
        tops = rl.row_tops()
        mask = dark(image)
        self.assertEqual([r["box"][1] >= t and r["box"][3] <= t + 55 for r, t in zip(rows, tops)], [True] * 3)
        self.assertTrue(mask[tops[0]:tops[2] + 55].any())
        self.assertFalse(mask[tops[3]:].any())
        with self.assertRaises(ValueError):
            rl.text_sample([ink(100, 40)] * 11, [3.0] * 11, slots=10)

    def test_short_samples_in_a_run(self):
        items = [ink(200 + 5 * i, 40, label=f"text {i}", item=f"iam:f:{i}") for i in range(60)]
        pool = rp.ListPool(items, "textline", "train", "iam")
        with tempfile.TemporaryDirectory() as out:
            report = rs.run(rs.base_config("textline", "train", out, short_prob=1.0, mark_prob=0.0), pool=pool,
                            log=lambda *_: None)
            rows = Run.read(None, rs.task_dir(out, "train", "textline"))
            sizes = [len(row["items"]) for row in rows]
            self.assertTrue(all(1 <= n <= 10 for n in sizes))
            self.assertEqual(report["short_samples"], sum(n < 10 for n in sizes))
            self.assertGreaterEqual(sum(sizes), 60)
            self.assertEqual({i for row in rows for i in row["items"]}, {pool.item_id(i) for i in range(60)})
            for row in rows:
                self.assertEqual(len(row["target"].split("\n")), len(row["items"]))
                self.assertEqual(len(row["marks"]), len(row["items"]))


class Marks(unittest.TestCase):
    """A mark as mark separation leaves it: only its points on the writing, all at one time after the row's writing."""

    def setUp(self):
        self.item = ink(600, 60)
        self.lines = rl.polylines(self.item)

    def test_no_mark_leaves_the_item_alone(self):
        self.assertEqual(rm.maybe_mark(self.item, "textline", 3.0, random.Random(0), prob=0.0), (self.item, None))
        self.assertEqual(rm.maybe_mark(self.item, "drawing", 3.0, random.Random(0), prob=1.0), (self.item, None))

    def test_only_overlapping_points_are_added_with_times_in_the_row_span(self):
        times = np.concatenate([line[:, 2] for line in self.lines])
        drawn = 0
        for seed in range(80):
            row, mark = rm.maybe_mark(self.item, "textline", 3.0, random.Random(seed), prob=1.0)
            self.assertIn(mark["kind"], ("underline", "strikethrough", "oval", "rectangle"))
            if not mark["points"]:
                self.assertIs(row, self.item)
                continue
            drawn += 1
            original, added = row[:len(self.lines)], row[len(self.lines):]
            for stroke, line in zip(original, self.lines):
                self.assertEqual((stroke.x, stroke.y, stroke.t), (line[:, 0].tolist(), line[:, 1].tolist(), line[:, 2].tolist()))
            points = np.array([(x, y) for s in added for x, y in zip(s.x, s.y)])
            stamps = np.array([t for s in added for t in s.t])
            self.assertEqual(len(points), mark["points"])
            self.assertTrue(rm.overlapping(points, self.lines, 3.0).all())
            self.assertTrue((stamps == stamps[0]).all() and stamps[0] > times.max())   # one section, after the writing
            self.assertEqual(len(rl.polylines(row)), len(row))          # every added stroke carries its times
        self.assertGreater(drawn, 20)

    def test_same_generator_same_mark(self):
        a = rm.maybe_mark(self.item, "textline", 3.0, random.Random(5), prob=1.0)
        b = rm.maybe_mark(self.item, "textline", 3.0, random.Random(5), prob=1.0)
        self.assertEqual(a[1], b[1])
        self.assertEqual([(s.x, s.y, s.t) for s in a[0][len(self.lines):]], [(s.x, s.y, s.t) for s in b[0][len(self.lines):]])

    def test_formulas_are_never_marked(self):
        for seed in range(20):
            self.assertEqual(rm.maybe_mark(self.item, "formula", 3.0, random.Random(seed), prob=1.0), (self.item, None))


def page_line(cls, x0, y0, width, height, text, parent=0, t0=0.0, strokes=3):
    """A line as note_synth's page_export writes one: ink points [x, y, t] timed from the line's own first point, and
    its bbox its ink's extent."""
    ink = []
    for k in range(strokes):
        xs = np.linspace(x0 + k * width / strokes, x0 + (k + 1) * width / strokes, 12)
        ys = y0 + height / 2 + 0.45 * height * np.sin(np.linspace(0, 6, 12) + k)
        ink.append([[round(float(x), 3), round(float(y), 3), float(t0 + 20 * k + j)] for j, (x, y) in enumerate(zip(xs, ys))])
    points = np.array([p for s in ink for p in s])
    return {"class": cls, "level": 1, "parent": parent, "text": text, "ink": ink,
            "bbox": [float(points[:, 0].min()), float(points[:, 1].min()), float(points[:, 0].max()), float(points[:, 1].max())]}


def refresh_box(obj):
    points = np.array([p for s in obj["ink"] for p in s])
    obj["bbox"] = [float(points[:, 0].min()), float(points[:, 1].min()), float(points[:, 0].max()), float(points[:, 1].max())]
    return obj


class Pages(unittest.TestCase):
    """Samples straight from a page, read as the cascade reads it with the ground truth's boxes."""

    def test_groups_of_ten_per_task_in_page_order(self):
        objects = [{"class": "textblock", "level": 2, "bbox": [0, 0, 400, 3000]}]
        objects += [page_line("textline", 0, 100 * i, 300, 40, f"line {i}") for i in range(23)]
        objects += [page_line("formula", 500, 100 * i, 200, 40, f"x^{i}", parent=None) for i in range(4)]
        groups = rpg.page_groups(objects)
        self.assertEqual([len(g["targets"]) for g in groups["textline"]], [10, 10, 3])
        self.assertEqual([t for g in groups["textline"] for t in g["targets"]], [f"line {i}" for i in range(23)])
        self.assertEqual([g["targets"] for g in groups["formula"]], [[f"x^{i}" for i in range(4)]])
        with tempfile.TemporaryDirectory() as folder:
            rows, stats = rpg.write_samples(objects, folder, "images/00000", "p0000007")
            self.assertEqual([r["image"] for r in rows["textline"]], [f"images/00000/p0000007_{k:02d}.png" for k in range(3)])
            self.assertEqual(rows["textline"][2]["target"], "line 20\nline 21\nline 22")
            self.assertEqual({r["prompt"] for r in rows["formula"]}, {rl.PROMPTS["formula"]})
            self.assertEqual((stats["textline"]["samples"], stats["textline"]["rows"], stats["textline"]["short_samples"]), (3, 23, 1))
            self.assertEqual((stats["formula"]["samples"], stats["formula"]["short_samples"]), (1, 1))
            for task in ("textline", "formula"):
                for row in rows[task]:
                    with Image.open(os.path.join(folder, task, row["image"])) as image:
                        self.assertEqual((image.size, image.mode), ((768, 768), "RGB"))
            with Image.open(os.path.join(folder, "textline", rows["textline"][2]["image"])) as image:
                mask = dark(image)
                self.assertTrue(mask[rl.row_tops()[2]:rl.row_tops()[2] + 55].any())
                self.assertFalse(mask[rl.row_tops()[3]:].any())             # a short sample's other slots stay white

    def test_marks_never_happened(self):
        line = page_line("textline", 0, 0, 300, 40, "underlined")
        mark = {"class": "line_mark", "level": 1, "parent": 0, "bbox": [0, 30, 300, 38],
                "ink": [[[float(x), 34.0] for x in range(0, 300, 5)]]}
        with_mark = rpg.page_groups([line, mark])["textline"][0]
        self.assertEqual(len(with_mark["rows"][0]), 3)
        self.assertEqual(with_mark["others"], [0])
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            rpg.write_samples([line, mark], a, "i", "p")
            rpg.write_samples([line], b, "i", "p")
            with Image.open(os.path.join(a, "textline", "i", "p_00.png")) as x, Image.open(os.path.join(b, "textline", "i", "p_00.png")) as y:
                self.assertTrue(np.array_equal(np.asarray(x), np.asarray(y)))

    def test_a_close_line_below_gives_its_reaching_stroke_whole_timed_after(self):
        above = page_line("textline", 0, 0, 300, 40, "the row")            # ink y 2-38
        below = page_line("textline", 0, 60, 300, 40, "the line below")
        # an ascender of the line below: 4 of its 6 points inside the row's box, so it is read with the row, whole
        ascender = [[100.0, y, 500.0 + j] for j, y in enumerate((22.0, 26.0, 30.0, 34.0, 48.0, 52.0))]
        below["ink"].append(ascender)
        refresh_box(below)
        group = rpg.page_groups([above, below])["textline"][0]
        row, own_t = group["rows"][0], [p[2] for s in above["ink"] for p in s]
        self.assertEqual(group["others"][0], 1)             # (the line below's box, stretched by it, takes some of the row's)
        self.assertEqual(len(row), 4)
        taken = [s for s in row if len(s) == 6]
        self.assertEqual(len(taken), 1)
        np.testing.assert_array_equal(taken[0][:, :2], np.array(ascender)[:, :2])      # its own points, not cut
        np.testing.assert_allclose(np.diff(taken[0][:, 2]), np.diff(np.array(ascender)[:, 2]))
        self.assertGreater(taken[0][:, 2].min(), max(own_t))                # the line below is written after the row
        # the row's own line keeps its times exactly: it is the block's first line
        own = [s for s in row if len(s) == 12]
        np.testing.assert_array_equal(np.concatenate([s[:, 2] for s in own]), np.array(own_t))

    def test_every_line_moves_only_as_a_whole_in_writing_order(self):
        lines = [page_line("textline", 0, 100 * i, 300, 40, f"line {i}") for i in (2, 0, 1)]     # listed out of order
        ordered = rpg.write_order(rpg.page_text(lines))
        spans = []
        for before, after in zip(rpg.page_text(lines), ordered):
            t0, t1 = np.concatenate([s[:, 2] for s in before["strokes"]]), np.concatenate([s[:, 2] for s in after["strokes"]])
            shift = t1 - t0
            self.assertTrue(np.allclose(shift, shift[0]), "a line's times changed by more than one offset")
            spans.append((after["bbox"][1], t1.min(), t1.max()))
        spans.sort()
        for (_, _, end), (_, start, _) in zip(spans, spans[1:]):
            self.assertGreater(start, end)                                  # top to bottom, a pause between

    def test_unusable_label_is_no_row_and_untimed_ink_is_left_out(self):
        lines = [page_line("textline", 0, 0, 300, 40, "fine"), page_line("textline", 0, 100, 300, 40, "bad <pad>")]
        self.assertEqual([g["targets"] for g in rpg.page_groups(lines)["textline"]], [["fine"]])
        # a line without recorded times (a geometry label) is neither a row nor read in the row it overlaps
        timed = page_line("textline", 0, 0, 300, 40, "timed")
        untimed = page_line("textline", 0, 5, 300, 30, "A")
        untimed["ink"] = [[p[:2] for p in s] for s in untimed["ink"]]
        counts = {}
        groups = rpg.page_groups([timed, untimed], untimed=counts)["textline"]
        self.assertEqual([g["targets"] for g in groups], [["timed"]])
        self.assertEqual((groups[0]["others"], len(groups[0]["rows"][0]), counts), ([0], 3, {"textline": 1}))


class Run(unittest.TestCase):
    def read(self, folder):
        with open(os.path.join(folder, "manifest.jsonl"), encoding="utf-8") as f:
            return [json.loads(line) for line in f]

    def test_text_run(self):
        items = [ink(100 + 37 * i, 40 + i % 9, label=f"text {i}", item=f"iam:f:{i}") for i in range(25)]
        pool = rp.ListPool(items, "textline", "train", "iam")
        labels = {pool.item_id(i): pool.label(i) for i in range(len(pool))}
        with tempfile.TemporaryDirectory() as out:
            config = rs.base_config("textline", "train", out, short_prob=0.0)
            report = rs.run(config, pool=pool, shard_samples=2, log=lambda *_: None)
            folder = rs.task_dir(out, "train", "textline")
            rows = self.read(folder)
            self.assertEqual(len(rows), 3)                  # ceil(25 / 10), one pass
            self.assertEqual(report["samples"], 3)
            self.assertEqual(report["rows"], 30)
            self.assertEqual(sorted(set(i for row in rows for i in row["items"])), sorted(labels))
            for row in rows:
                self.assertEqual(row["prompt"], "What is written in this image in English?")
                self.assertEqual(row["target"].split("\n"), [labels[i] for i in row["items"]])
                with Image.open(os.path.join(folder, row["image"])) as image:
                    self.assertEqual((image.size, image.mode), ((768, 768), "RGB"))
            with open(os.path.join(folder, "config.json")) as f:
                self.assertEqual(json.load(f)["ink"], rl.INK)

            # a rerun skips finished shards and gives the same rows; other settings are refused
            rs.run(config, pool=pool, shard_samples=2, log=lambda *_: None)
            self.assertEqual(self.read(folder), rows)
            with self.assertRaises(SystemExit):
                rs.run(rs.base_config("textline", "train", out, seed=1), pool=pool, shard_samples=2,
                       log=lambda *_: None)
            # ... and so is finishing a folder whose images were drawn another way
            with self.assertRaises(SystemExit):
                rs.run(rs.base_config("textline", "train", out, ink="greyscale"), pool=pool, shard_samples=2,
                       log=lambda *_: None)

    def test_marks_in_a_run(self):
        items = [ink(300 + 20 * i, 50, label=f"text {i}", item=f"iam:f:{i}") for i in range(20)]
        pool = rp.ListPool(items, "textline", "train", "iam")
        with tempfile.TemporaryDirectory() as out, tempfile.TemporaryDirectory() as plain:
            report = rs.run(rs.base_config("textline", "train", out, mark_prob=1.0, short_prob=0.0), pool=pool,
                            log=lambda *_: None)
            rows = self.read(rs.task_dir(out, "train", "textline"))
            marks = [kind for row in rows for kind in row["marks"]]
            self.assertEqual(len(marks), 20)
            self.assertTrue(set(marks) - {None} <= {"underline", "strikethrough", "oval", "rectangle"})
            self.assertEqual(report["rows_marked"], sum(kind is not None for kind in marks))
            self.assertGreater(report["rows_marked"], 0)
            self.assertEqual(report["mark_prob"], 1.0)

            # with marks off the rows are drawn as before marks existed, and a marked row's image differs
            rs.run(rs.base_config("textline", "train", plain, mark_prob=0.0, short_prob=0.0), pool=pool,
                   log=lambda *_: None)
            unmarked = self.read(rs.task_dir(plain, "train", "textline"))
            self.assertEqual([row["marks"] for row in unmarked], [[None] * 10, [None] * 10])
            with Image.open(os.path.join(rs.task_dir(out, "train", "textline"), rows[0]["image"])) as a, \
                    Image.open(os.path.join(rs.task_dir(plain, "train", "textline"), unmarked[0]["image"])) as b:
                self.assertEqual(np.array_equal(np.asarray(a), np.asarray(b)), set(rows[0]["marks"]) == {None})

    def test_pool_without_times_is_refused_before_anything_is_made(self):
        items = [Ink([Untimed([0, 40 + i], [0, 30])], f"text {i}", f"iam:f:{i}") for i in range(12)]
        pool = rp.ListPool(items, "textline", "train", "iam")
        with tempfile.TemporaryDirectory() as out:
            with self.assertRaises(SystemExit):
                rs.run(rs.base_config("textline", "train", out), pool=pool, log=lambda *_: None)
            self.assertFalse(os.path.exists(rs.task_dir(out, "train", "textline")))

    def test_formula_and_drawing_runs(self):
        with tempfile.TemporaryDirectory() as out:
            formulas = rp.ListPool([ink(80, 40, label=f"x^{i}") for i in range(12)], "formula", "val", "mw:val")
            report = rs.run(rs.base_config("formula", "val", out), pool=formulas, log=lambda *_: None)
            rows = self.read(rs.task_dir(out, "val", "formula"))
            self.assertEqual(rows[0]["prompt"], "What is written in this image in LaTeX?")
            self.assertEqual(report["target_separator"], "\n")

            names = ["cat", "dog", "cat", "tree", "house"]
            drawings = rp.ListPool([ink(60 + 10 * i, 50, label=n, item=f"qd:{i}") for i, n in enumerate(names)],
                                   "drawing", "val", "qd")
            report = rs.run(rs.base_config("drawing", "val", out), pool=drawings, log=lambda *_: None)
            rows = self.read(rs.task_dir(out, "val", "drawing"))
            self.assertEqual(len(rows), 5)
            self.assertEqual({row["items"][0]: row["target"] for row in rows},
                             {f"qd:{i}": n for i, n in enumerate(names)})
            self.assertEqual(rows[0]["prompt"], "What is drawn in this image?")
            self.assertEqual((report["categories"], report["items_per_category"]), (4, {"min": 1, "max": 2}))
            self.assertEqual(report["pen"], list(rl.DRAWING_PEN))

    def test_config_checks(self):
        with self.assertRaises(ValueError):
            rs.base_config("textline", "train", "x", rows=11)
        with self.assertRaises(TypeError):
            rs.base_config("textline", "train", "x", colour="red")


if __name__ == "__main__":
    unittest.main()
