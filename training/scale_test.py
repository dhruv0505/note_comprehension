"""
scale_test.py — the dataset and DataLoader on a manifest far larger than the sample.

    python scale_test.py                      # 1,000,000 rows over 2 manifests, 768 x 768 images
    python scale_test.py --rows 5000000 --keep

Builds stress_data/ from data/ (render_seg_data.py's sample): each sample image
upscaled to 768 x 768, and manifests that cycle through the sample rows, with a
blank line mixed in. Then checks:
  * the offset index: build time, memory, and rows read back at random are the
    rows written there
  * a 768 x 768 image is not resized: its pixels are exactly (ink - mean) / std
  * batches through the DataLoader's 4 persistent workers, and their shapes
stress_data/ is removed afterwards unless --keep.
"""

from __future__ import annotations
import argparse
import gc
import json
import os
import random
import shutil
import time

import numpy as np
import torch
from PIL import Image

from seg_dataset import HERE, Collator, SegDataset, make_dataloader, to_loc_target

SIZE = 768


def build(out, rows, manifests):
    source = os.path.join(HERE, "data")
    with open(os.path.join(source, "manifest.jsonl")) as f:
        sample = [json.loads(line) for line in f if line.strip()]
    os.makedirs(os.path.join(out, "images"), exist_ok=True)
    for name in sorted({row["image"] for row in sample}):
        with Image.open(os.path.join(source, name)) as image:
            image.convert("L").resize((SIZE, SIZE), Image.LANCZOS).save(os.path.join(out, name))

    paths, per_file = [], -(-rows // manifests)
    written = 0
    for m in range(manifests):
        path = os.path.join(out, f"manifest_{m}.jsonl")
        with open(path, "w", newline="\n") as f:
            for _ in range(min(per_file, rows - written)):
                f.write(json.dumps(sample[written % len(sample)]) + "\n")
                written += 1
                if written == 3:
                    f.write("\n")                       # a blank line is not a row
        paths.append(path)
    return sample, paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=1_000_000)
    ap.add_argument("--manifests", type=int, default=2)
    ap.add_argument("--batches", type=int, default=40)
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    out = os.path.join(HERE, "stress_data")
    t = time.perf_counter()
    sample, paths = build(out, args.rows, args.manifests)
    size_mb = sum(os.path.getsize(p) for p in paths) / 2 ** 20
    print(f"built {args.rows:,} rows in {len(paths)} manifests ({size_mb:,.0f} MB) in {time.perf_counter() - t:.1f}s")

    dataset = None
    try:
        t = time.perf_counter()
        dataset = SegDataset(paths)
        index_mb = (dataset.offsets.nbytes + dataset.file_index.nbytes) / 2 ** 20
        print(f"  index                    {len(dataset):,} rows in {time.perf_counter() - t:.2f}s, "
              f"{index_mb:.1f} MB in memory")
        assert len(dataset) == args.rows, (len(dataset), args.rows)

        for index in random.Random(0).sample(range(len(dataset)), 2000) + [0, len(dataset) - 1]:
            row, expected = dataset.row(index), sample[index % len(sample)]
            assert (row["prompt"], row["target"]) == (expected["prompt"], expected["target"]), index
            assert row["image"].endswith(expected["image"].replace("/", os.sep)) or \
                row["image"].endswith(expected["image"]), index
        print("  random access            2,002 rows read back as written, across both manifests")

        collator = Collator()
        ink, prompt, target, _ = dataset[5]
        batch = collator([dataset[5]])
        image_processor = collator.processor.image_processor
        mean = torch.tensor(image_processor.image_mean).view(3, 1, 1)
        std = torch.tensor(image_processor.image_std).view(3, 1, 1)
        expected = (torch.from_numpy(ink).permute(2, 0, 1) - mean) / std
        assert batch["pixel_values"].shape == (1, 3, SIZE, SIZE)
        assert torch.allclose(batch["pixel_values"][0], expected, atol=1e-5)
        assert target == to_loc_target(sample[5]["target"])
        print(f"  768 input, no resize     pixels exactly (ink - mean) / std")

        loader = make_dataloader(dataset)
        batches = iter(loader)
        t = time.perf_counter()
        shapes = set()
        for n, batch in enumerate(batches, 1):
            if n == 1:
                first = time.perf_counter() - t
                t = time.perf_counter()
            shapes.add(tuple(batch["pixel_values"].shape))
            assert batch["labels"].shape[0] == batch["input_ids"].shape[0] == 6
            if n == args.batches:
                break
        per_batch = (time.perf_counter() - t) / max(args.batches - 1, 1)
        print(f"  dataloader               {len(loader):,} batches/epoch; first batch {first:.1f}s "
              f"(workers starting), then {per_batch:.2f}s/batch over {args.batches - 1}; "
              f"pixel_values {sorted(shapes)}")
        del batches, loader         # shuts the persistent workers down; they hold the manifests open
    finally:
        gc.collect()
        if dataset is not None:
            dataset.close()
        if not args.keep:
            shutil.rmtree(out)
            print(f"  removed {out}")


if __name__ == "__main__":
    main()
