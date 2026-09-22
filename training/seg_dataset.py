"""
seg_dataset.py — Dataset/DataLoader of segmentation prompts for Florence-2.

    python seg_dataset.py                                 # data/manifest.jsonl: one batch, checked
    python seg_dataset.py --manifest a.jsonl b.jsonl      # several manifests read as one dataset
    python seg_dataset.py --forward                       # also run the batch through the model once

A manifest is JSON lines, one row per prompt: {"image", "prompt", "target"}.
`image` is a greyscale image, dark ink on light, relative to the manifest's own
folder (or absolute). A recognition row (recog_synth's manifests, which also name
the row's "task") has an RGB image, its ink coloured by time and pen movement, and
a plain-text target: its transcriptions, one per line. It is read the same way,
colours kept, and trained on exactly as written (row_target), so segmentation
and recognition manifests can be read as one dataset. A segmentation row's
`target` is coordinates then class per object, the
coordinates integers on the 0-999 grid (render_seg_data.py writes this):

    46 45 218 284 textblock 499 45 655 334 textblock

and is trained on as Florence-2's location tokens, each coordinate one token:

    <loc_46><loc_45><loc_218><loc_284>textblock<loc_499><loc_45><loc_655><loc_334>textblock

A geometry crop's arc is six numbers (start, middle, end) and its segment
four, its two ends: "120 80 410 95 segment".

Per image:
  1. greyscale -> 3 channels by copying the one channel into R, G and B
  2. inverted to background = 0, ink = 1 (float)
  3. the processor's mean/std normalisation, its own 1/255 rescale switched off
     since step 2 already put the image in [0, 1]. Images already at the
     processor's size (768 x 768) are not resized; others are resized to it.

Batch: pixel_values (B, 3, 768, 768) float32, input_ids / attention_mask (B, L)
from the prompts, labels (B, T) from the targets with padding set to -100.

Scale: the dataset holds only each row's byte offset (8 bytes a row) and reads
the row when asked for it, so a manifest of millions of rows costs megabytes and
the DataLoader workers do not each copy it.
"""

from __future__ import annotations
import argparse
import json
import os
import re
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
# model and processor downloads stay inside this folder unless HF_HOME says otherwise
os.environ.setdefault("HF_HOME", os.path.join(HERE, "hf_cache"))
# the workers tokenize; a tokenizer's thread pool does not survive being forked into them
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

MODEL_ID = "microsoft/Florence-2-large"
MODEL_REVISION = "21a599d414c4d928c9032694c424fb94458e3594"   # the remote code this was checked against
DEFAULT_MANIFEST = os.path.join(HERE, "data", "manifest.jsonl")
MAX_TEXT_TOKENS = 4096      # text_config.max_position_embeddings, encoder and decoder alike
GRID = 1000                 # <loc_0> .. <loc_999>
IGNORE_INDEX = -100


def load_processor():
    from transformers import AutoProcessor
    return AutoProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION, trust_remote_code=True)


def load_model():
    """Florence-2's remote code imports flash_attn unconditionally; it is not
    needed, so it is dropped from the import check."""
    from unittest.mock import patch
    from transformers import AutoModelForCausalLM
    from transformers.dynamic_module_utils import get_imports

    def without_flash_attn(filename):
        return [name for name in get_imports(filename) if name != "flash_attn"]

    with patch("transformers.dynamic_module_utils.get_imports", without_flash_attn):
        return AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=MODEL_REVISION,
                                                    trust_remote_code=True,
                                                    attn_implementation="eager")


def to_loc_target(target: str) -> str:
    """Each integer coordinate as its <loc_N> token; each class name kept as
    written, directly after its coordinates as Florence-2's own targets have it.
    A target already in location tokens comes back unchanged."""
    parts, words = [], []
    for token in target.split():
        if token.isdigit():
            if words:
                parts.append(" ".join(words))
                words = []
            value = int(token)
            if value >= GRID:
                raise ValueError(f"coordinate {value} is off the 0-{GRID - 1} grid in {target[:80]!r}")
            parts.append(f"<loc_{value}>")
        else:
            words.append(token)
    if words:
        parts.append(" ".join(words))
    return "".join(parts)


RECOGNITION_TASKS = ("textline", "formula", "drawing")      # recog_synth's rows name their task; segmentation rows don't


def is_recognition(row: dict) -> bool:
    return row.get("task") in RECOGNITION_TASKS


def row_target(row: dict) -> str:
    """The text a row is trained on: a segmentation row's coordinates as location tokens (to_loc_target), a
    recognition row's transcription exactly as written, its rows' newlines and any numbers kept."""
    return row["target"] if is_recognition(row) else to_loc_target(row["target"])


def _line_offsets(path: str, chunk: int = 1 << 24) -> np.ndarray:
    """Byte offset of every non-blank line, found a chunk at a time."""
    starts = [np.zeros(1, dtype=np.int64)]
    size = 0
    with open(path, "rb") as f:
        while block := f.read(chunk):
            newlines = np.flatnonzero(np.frombuffer(block, dtype=np.uint8) == ord("\n"))
            starts.append(newlines.astype(np.int64) + size + 1)
            size += len(block)
    starts = np.concatenate(starts)
    starts = starts[starts < size]                   # a trailing newline starts no line
    lengths = np.diff(np.append(starts, size))
    return starts[lengths > 2]                       # "\n" or "\r\n" alone is blank


class SegDataset(Dataset):
    def __init__(self, manifests=(DEFAULT_MANIFEST,)):
        self.manifests = [os.path.abspath(path) for path in manifests]
        offsets = [_line_offsets(path) for path in self.manifests]
        self.offsets = np.concatenate(offsets)
        self.file_index = np.concatenate([np.full(len(o), i, dtype=np.int32)
                                          for i, o in enumerate(offsets)])
        self._handles, self._pid = {}, None

    def __len__(self):
        return len(self.offsets)

    def close(self):
        """Closes this process's manifest handles; rows can still be read after."""
        for handle in self._handles.values():
            handle.close()
        self._handles = {}

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handles"], state["_pid"] = {}, None      # file handles stay in their process
        return state

    def row(self, index) -> dict:
        """The manifest row, image path made absolute. Each process opens its
        own handles, so forked workers never share a file position."""
        if self._pid != os.getpid():
            self._handles, self._pid = {}, os.getpid()
        file_index = int(self.file_index[index])
        handle = self._handles.get(file_index)
        if handle is None:
            handle = self._handles[file_index] = open(self.manifests[file_index], "rb")
        handle.seek(int(self.offsets[index]))
        line = handle.readline()
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{self.manifests[file_index]} at byte {self.offsets[index]}: {error}") from None
        row["image"] = os.path.join(os.path.dirname(self.manifests[file_index]), row["image"])
        return row

    def __getitem__(self, index):
        row = self.row(index)
        with Image.open(row["image"]) as image:
            # a greyscale image comes out as R = G = B; a recognition image keeps the colours its ink is drawn in
            rgb = np.asarray(image.convert("RGB"))                  # (H, W, 3)
        ink = 1.0 - rgb.astype(np.float32) / 255.0                  # background 0, ink 1
        return ink, row["prompt"], row_target(row), row["image"]


class Collator:
    """Builds the batch with the processor. The processor is loaded on first use,
    so each DataLoader worker loads its own instead of having it pickled over."""

    def __init__(self):
        self.processor = None

    def __call__(self, batch):
        if self.processor is None:
            self.processor = load_processor()
        images, prompts, targets, paths = zip(*batch)
        image_processor = self.processor.image_processor
        tokenizer = self.processor.tokenizer

        # Florence2Processor.__call__ accepts do_rescale but never passes it on, so
        # its image processor would divide the [0, 1] image by 255 again. Its two
        # halves are called directly instead; its prompt rewriting only expands
        # task tokens such as <OD>, which these prompts do not use.
        size = (image_processor.size["height"], image_processor.size["width"])
        pixel_values = image_processor(
            list(images), do_rescale=False, input_data_format="channels_last",
            do_resize=any(image.shape[:2] != size for image in images),
            return_tensors="pt")["pixel_values"]

        max_prompt = MAX_TEXT_TOKENS - self.processor.image_seq_length   # image tokens share the encoder
        inputs = tokenizer(list(prompts), return_tensors="pt", padding="longest", truncation=True,
                           max_length=max_prompt, return_token_type_ids=False)

        label_ids = tokenizer(list(targets), return_token_type_ids=False,
                              return_attention_mask=False)["input_ids"]
        for path, ids in zip(paths, label_ids):
            if len(ids) > MAX_TEXT_TOKENS:
                warnings.warn(f"{path}: target is {len(ids)} tokens, truncated to {MAX_TEXT_TOKENS}")
        labels = pad_sequence([torch.tensor(ids[:MAX_TEXT_TOKENS]) for ids in label_ids],
                              batch_first=True, padding_value=IGNORE_INDEX)
        return {"pixel_values": pixel_values, "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"], "labels": labels}


def make_dataloader(dataset=None):
    return DataLoader(dataset if dataset is not None else SegDataset(), collate_fn=Collator(),
                      shuffle=True, batch_size=6, num_workers=4, pin_memory=True,
                      persistent_workers=True)


def check_loc_targets(dataset, tokenizer, rows):
    """Every coordinate of the first `rows` targets becomes exactly one location
    token, and the token ids decode back to the converted string."""
    loc_ids = set(tokenizer.convert_tokens_to_ids([f"<loc_{i}>" for i in range(GRID)]))
    n = min(rows, len(dataset))
    longest = 0
    for index in range(n):
        row = dataset.row(index)
        if is_recognition(row):             # plain text: check_recog_targets.py checks these
            continue
        raw = row["target"]
        converted = to_loc_target(raw)
        ids = tokenizer(converted, return_token_type_ids=False)["input_ids"]
        longest = max(longest, len(ids))
        numbers = len(re.findall(r"\b\d+\b", raw))
        if sum(i in loc_ids for i in ids) != numbers:
            raise AssertionError(f"row {index}: {numbers} coordinates but a different number of loc tokens")
        decoded = tokenizer.decode(ids, skip_special_tokens=False)
        if decoded != f"{tokenizer.bos_token}{converted}{tokenizer.eos_token}":
            raise AssertionError(f"row {index}: decodes to {decoded[:120]!r}")
    print(f"  loc-token targets        {n} rows: one token per coordinate, round-trip exact; "
          f"longest {longest} tokens (max {MAX_TEXT_TOKENS})")


def check_batch(batch, processor):
    """Confirms the pixels are the inverted image through mean/std: background
    lands on (0 - mean) / std and ink on (1 - mean) / std, per channel."""
    image_processor = processor.image_processor
    mean = torch.tensor(image_processor.image_mean).view(1, 3, 1, 1)
    std = torch.tensor(image_processor.image_std).view(1, 3, 1, 1)
    pixels = batch["pixel_values"] * std + mean                    # undo mean/std -> [0, 1]
    print(f"  un-normalised range      [{pixels.min():.4f}, {pixels.max():.4f}]  (expect ~[0, 1])")
    print(f"  channels identical       {bool(torch.allclose(pixels[:, 0], pixels[:, 1], atol=1e-4) and torch.allclose(pixels[:, 1], pixels[:, 2], atol=1e-4))}")
    background = (pixels < 0.05).float().mean().item()
    print(f"  pixels ~0 (background)   {100 * background:.1f}%  (expect the large majority)")
    corner = batch["pixel_values"][0, :, 0, 0]
    expected = (0 - mean.view(3)) / std.view(3)
    print(f"  corner pixel             {[round(v, 4) for v in corner.tolist()]}  "
          f"(background expected {[round(v, 4) for v in expected.tolist()]})")
    print(f"  label tokens per row     {(batch['labels'] != IGNORE_INDEX).sum(1).tolist()}")
    tokenizer = processor.tokenizer
    print(f"  prompt 0                 {tokenizer.decode(batch['input_ids'][0], skip_special_tokens=True)[:100]!r}")
    first = batch["labels"][0]
    print(f"  target 0                 {tokenizer.decode(first[first != IGNORE_INDEX], skip_special_tokens=False)[:160]!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", nargs="+", default=[DEFAULT_MANIFEST])
    ap.add_argument("--check-rows", type=int, default=1000,
                    help="rows whose loc-token targets are checked before the batch")
    ap.add_argument("--forward", action="store_true", help="run the batch through the model once")
    args = ap.parse_args()

    dataset = SegDataset(args.manifest)
    loader = make_dataloader(dataset)
    print(f"dataset: {len(dataset)} prompt rows in {len(dataset.manifests)} manifest(s), {len(loader)} batches")
    processor = load_processor()
    check_loc_targets(dataset, processor.tokenizer, args.check_rows)

    batch = next(iter(loader))
    for key, value in batch.items():
        print(f"  {key:15s} {str(tuple(value.shape)):20s} {value.dtype}  pinned={value.is_pinned()}")
    check_batch(batch, processor)

    if args.forward:
        model = load_model().eval()
        with torch.no_grad():
            out = model(**batch)
        print(f"  forward: loss {out.loss.item():.4f}, logits {tuple(out.logits.shape)} {out.logits.dtype}")


if __name__ == "__main__":
    main()
