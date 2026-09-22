"""
class_tuning.py — the generator settings that decide how often each class appears
on a page, and a run's overrides of them.

    python synthesize.py ... --tuning tuning.json       # {"flowchart_weight": 30, "underline_prob": 0.3}

KNOBS names each setting a run may change: the module constant it overrides, the
class it tunes, and whether it is a relative weight (0 or more) or a probability
(0 to 1). defaults() reads their values from page_generator.py and
block_generators.py as text, so the synthesis tracker can show them without
importing the generator (or numpy). apply() sets them on the imported modules; it
is called in each synthesis worker before any page is made.

Every setting here is read when a page is built, not bound at import, except
page_generator.ENCLOSURE_PROB, which generate_page takes as a default argument:
synthesize.py passes it explicitly so an override reaches it. `circle`,
`segment` and `vertex` have no setting of their own: every geometry figure labels
all of its circles, segments and vertices, so they follow geometry_weight.
"""

from __future__ import annotations
import ast
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# key: (module, constant, index into a tuple/dict constant or None, kind, class, what it does)
KNOBS = {
    "textblock_weight": ("page_generator", "TEXTBLOCK_WEIGHT", None, "weight", "textblock",
                         "Cell weight for a textblock (flat, whatever the cell's size)"),
    "flowchart_weight": ("page_generator", "FLOWCHART_WEIGHT", None, "weight", "flowchart",
                         "Cell weight for a flowchart, times the cell's area"),
    "geometry_weight": ("page_generator", "GEOMETRY_WEIGHT", None, "weight", "geometry",
                        "Cell weight for a geometry figure, times area and squareness (its circles, segments and vertices follow it)"),
    "table_weight": ("page_generator", "TABLE_WEIGHT", None, "weight", "table",
                     "Cell weight for a table, highest in mid-sized 1.6:1 cells"),
    "drawing_weight": ("page_generator", "DRAWING_WEIGHT", None, "weight", "drawing",
                       "Cell weight for a QuickDraw sketch, highest in small square cells"),
    "prose_lean_narrow": ("page_generator", "TEXT_HEAVY_PROB", 0, "prob", "textline",
                          "Chance a narrow textblock is mostly prose rather than mostly formulas"),
    "prose_lean_wide": ("page_generator", "TEXT_HEAVY_PROB", 1, "prob", "textline",
                        "Chance a wide textblock is mostly prose rather than mostly formulas"),
    "minority_row_prob": ("page_generator", "MINORITY_ROW_PROB", None, "prob", "formula",
                          "Chance any one textblock row is the block's minority kind (prose vs formula)"),
    "table_formula_prob": ("block_generators", "TABLE_FORMULA_PROB", None, "prob", "formula",
                           "Chance a table cell holds a formula rather than prose"),
    "caption_formula_prob": ("block_generators", "SHAPE_FORMULA_PROB", None, "prob", "formula",
                             "Chance a flowchart caption is a formula rather than prose"),
    "enclosure_prob": ("page_generator", "ENCLOSURE_PROB", None, "prob", "enclosure",
                       "Chance a textblock gets a box or oval round itself, a line or words"),
    "underline_prob": ("block_generators", "UNDERLINE_PROB", None, "prob", "line_mark",
                       "Chance a text line is underlined"),
    "strikethrough_prob": ("block_generators", "STRIKETHROUGH_PROB", None, "prob", "line_mark",
                           "Chance a text line is struck through instead"),
    "arrows_0": ("page_generator", "ARROW_COUNT_WEIGHTS", 0, "weight", "arrow",
                 "Weight of a page having no connector arrows"),
    "arrows_1": ("page_generator", "ARROW_COUNT_WEIGHTS", 1, "weight", "arrow",
                 "Weight of 1 connector arrow on a page"),
    "arrows_2": ("page_generator", "ARROW_COUNT_WEIGHTS", 2, "weight", "arrow",
                 "Weight of 2 connector arrows on a page"),
    "arrows_3": ("page_generator", "ARROW_COUNT_WEIGHTS", 3, "weight", "arrow",
                 "Weight of 3 connector arrows on a page"),
    "arrows_4": ("page_generator", "ARROW_COUNT_WEIGHTS", 4, "weight", "arrow",
                 "Weight of 4 connector arrows on a page"),
}
CLASS_ORDER = ("textblock", "flowchart", "geometry", "table", "drawing",
               "textline", "formula", "enclosure", "line_mark", "arrow")
MAX_WEIGHT = 1000.0


def _constants(module: str) -> dict:
    """A module's top-level literal assignments, read from its source."""
    with open(os.path.join(HERE, module + ".py"), encoding="utf-8") as f:
        tree = ast.parse(f.read())
    found = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                found[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                pass
    return found


def defaults() -> dict:
    """Every knob's value as the generator source sets it."""
    constants = {m: _constants(m) for m in {k[0] for k in KNOBS.values()}}
    out = {}
    for key, (module, name, index, _, _, _) in KNOBS.items():
        value = constants[module][name]
        out[key] = float(value if index is None else value[index])
    return out


def describe() -> list:
    """The knobs for a page to show: key, class, kind, default and what it does, in class order."""
    values = defaults()
    rows = [{"key": key, "class": cls, "kind": kind, "default": values[key], "help": text,
             "constant": name if index is None else f"{name}[{index}]"}
            for key, (_, name, index, kind, cls, text) in KNOBS.items()]
    return sorted(rows, key=lambda r: CLASS_ORDER.index(r["class"]))


def validate(overrides) -> dict:
    """`overrides` checked and cleaned: known keys, finite numbers in range, values equal
    to the default dropped. Raises ValueError naming every problem."""
    if overrides is None:
        return {}
    if not isinstance(overrides, dict):
        raise ValueError("tuning must be an object of setting: value")
    base, clean, problems = defaults(), {}, []
    for key, value in overrides.items():
        if key not in KNOBS:
            problems.append(f"unknown setting {key!r}")
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            problems.append(f"{key} must be a number")
            continue
        kind = KNOBS[key][3]
        top = 1.0 if kind == "prob" else MAX_WEIGHT
        if not 0.0 <= value <= top:
            problems.append(f"{key} must be between 0 and {top:g}")
            continue
        if abs(float(value) - base[key]) > 1e-12:
            clean[key] = float(value)
    merged = {**base, **clean}
    if merged["underline_prob"] + merged["strikethrough_prob"] > 1.0:
        problems.append("underline_prob + strikethrough_prob must be at most 1")
    if not any(merged[f"arrows_{n}"] for n in range(5)):
        problems.append("at least one arrows_N weight must be above 0")
    if not any(merged[k] for k in ("textblock_weight", "flowchart_weight", "geometry_weight",
                                   "table_weight", "drawing_weight")):
        problems.append("at least one cell weight must be above 0")
    if problems:
        raise ValueError("; ".join(problems))
    return clean


def apply(overrides) -> dict:
    """Set validated `overrides` on the imported generator modules; returns what was set."""
    import block_generators
    import page_generator
    modules = {"page_generator": page_generator, "block_generators": block_generators}
    clean = validate(overrides)
    for key, value in clean.items():
        module, name, index = KNOBS[key][:3]
        target = modules[module]
        current = getattr(target, name)
        if index is None:
            setattr(target, name, value)
        elif isinstance(current, tuple):
            setattr(target, name, tuple(value if i == index else v for i, v in enumerate(current)))
        else:
            setattr(target, name, {k: (value if k == index else v) for k, v in current.items()})
    return clean
