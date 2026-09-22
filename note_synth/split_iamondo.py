"""
split_iamondo.py — split the IAMonDo database (941 documents) into a val set
(for model selection / hyperparameter tuning, e.g. picking LoRA rank/LR like
the InkFM paper did) and a test set (for final reported metrics).

Default: 80 documents -> val, the rest (~861) -> test.

Supports optional stratification so the small `val` set doesn't end up with
too few (or zero) examples of rare classes like `table` (~6.3% of pages) by
chance, and optional template-grouping so documents copied from the same
template (IAMonDo explicitly has some of these) don't get split across val
and test.

Usage:
    python3 split_iamondo.py --data_dir /path/to/iamondo --val_size 80 \
        --out_dir /path/to/output --seed 0
"""

from __future__ import annotations
import argparse
import json
import random
from pathlib import Path
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple


def list_documents(data_dir: Path, pattern: str = "*.xml") -> List[Path]:
    """Adjust `pattern` to match however your IAMonDo download is laid out
    (one file per document id, e.g. 001.xml, 002.xml, ...)."""
    return sorted(data_dir.glob(pattern))


# ----------------------------------------------------------------------------
# Optional stratification key extraction — implement against your actual
# annotation format. Returning None disables stratification (falls back to
# plain random split).
# ----------------------------------------------------------------------------

RARE_CLASSES = ("table", "diagram")  # extend as needed


def extract_stratify_key(doc_path: Path) -> Optional[Tuple[bool, ...]]:
    """
    Should return a small tuple summarizing what matters for stratification,
    e.g. (has_table, has_diagram). Returning the same key groups documents
    into the same stratum, which are then split proportionally.

    TODO: parse the actual IAMonDo ground-truth annotation for `doc_path` and
    check which of RARE_CLASSES are present on the page. IAMonDo's ground
    truth XML tags content blocks with types like Text/List/Table/Diagram/
    Drawing/Formula — check for those tag names in your actual files.
    Placeholder below always returns None (i.e. stratification disabled)
    until you wire this up.
    """
    return None


def extract_template_key(doc_path: Path) -> Optional[str]:
    """
    Should return a template/group identifier if `doc_path` is a known copy
    of a template also written by other writers, so all copies of the same
    template can be kept in the same split. Returning None means "not part
    of a known template group" (each such document is free to land in either
    split independently).

    TODO: IAMonDo's own documentation/metadata should indicate which document
    IDs share a template, if this matters for your evaluation. Left as a stub
    since this isn't published in an obviously machine-readable place.
    """
    return None


# ----------------------------------------------------------------------------
# Splitting logic
# ----------------------------------------------------------------------------

def stratified_split(docs: List[Path], val_size: int, seed: int,
                      stratify_fn: Callable[[Path], Optional[object]] = extract_stratify_key,
                      group_fn: Callable[[Path], Optional[str]] = extract_template_key
                      ) -> Tuple[List[Path], List[Path]]:
    rng = random.Random(seed)

    # group documents so template-copies move together as one unit
    groups: Dict[str, List[Path]] = defaultdict(list)
    for d in docs:
        gkey = group_fn(d) or f"__singleton__{d.name}"
        groups[gkey].append(d)
    group_items = list(groups.items())
    rng.shuffle(group_items)

    # bucket groups by stratification key (based on their first doc, as an approximation)
    buckets: Dict[object, List[Tuple[str, List[Path]]]] = defaultdict(list)
    for gkey, members in group_items:
        skey = stratify_fn(members[0])
        buckets[skey].append((gkey, members))

    val_docs: List[Path] = []
    test_docs: List[Path] = []

    if len(buckets) == 1 and None in buckets:
        # no stratification info available -> simple proportional group split
        for gkey, members in group_items:
            if len(val_docs) < val_size:
                val_docs.extend(members)
            else:
                test_docs.extend(members)
    else:
        # allocate val quota proportionally across strata, so rare classes
        # aren't accidentally starved out of val
        total = sum(len(m) for items in buckets.values() for _, m in items)
        for skey, items in buckets.items():
            stratum_n = sum(len(m) for _, m in items)
            stratum_val_quota = round(val_size * stratum_n / total)
            taken = 0
            for gkey, members in items:
                if taken < stratum_val_quota:
                    val_docs.extend(members)
                    taken += len(members)
                else:
                    test_docs.extend(members)

    # if rounding left us short/over val_size, fix up by moving whole groups
    # (kept simple: trim/extend from test<->val without breaking template groups)
    rng.shuffle(val_docs)
    rng.shuffle(test_docs)
    while len(val_docs) > val_size and test_docs is not None:
        test_docs.append(val_docs.pop())
    while len(val_docs) < val_size and test_docs:
        val_docs.append(test_docs.pop())

    return val_docs, test_docs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=Path, required=True)
    ap.add_argument("--pattern", type=str, default="*.xml")
    ap.add_argument("--val_size", type=int, default=80)
    ap.add_argument("--out_dir", type=Path, default=Path("."))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    docs = list_documents(args.data_dir, args.pattern)
    if not docs:
        raise SystemExit(f"No files matched {args.pattern} in {args.data_dir} — check --pattern.")
    if args.val_size >= len(docs):
        raise SystemExit(f"val_size ({args.val_size}) >= total docs ({len(docs)})")

    val_docs, test_docs = stratified_split(docs, args.val_size, args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "val_ids.txt").write_text("\n".join(d.name for d in sorted(val_docs)))
    (args.out_dir / "test_ids.txt").write_text("\n".join(d.name for d in sorted(test_docs)))

    print(f"Total documents: {len(docs)}")
    print(f"Val:  {len(val_docs)} documents -> {args.out_dir / 'val_ids.txt'}")
    print(f"Test: {len(test_docs)} documents -> {args.out_dir / 'test_ids.txt'}")


if __name__ == "__main__":
    main()
