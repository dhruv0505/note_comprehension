"""
check_recog_targets.py — recognition targets against Florence-2's own tokenizer, as training would read them.

    python check_recog_targets.py ../dataset/train/recognition/textline/manifest.jsonl [more manifests ...]
    python check_recog_targets.py manifest.jsonl --max-rows 5000

For every row (up to --max-rows per manifest) the target is tokenized as Collator tokenizes labels and must:
  - decode back to exactly <s>target</s> (skip_special_tokens=False, no clean-up, as seg_eval.generate decodes)
  - hold no <unk> and no special or location token (a label holding "<sep>" or "<loc_5>" would be one)
  - have one newline token (Ċ, a byte-level BPE token of the base vocabulary) per row break, so the rows are
    exactly the target split on recog_layout.SEP
Prints each problem (the first 20) and a summary; exits 1 when any row fails.
"""

from __future__ import annotations
import argparse
import json
import sys

from seg_dataset import load_processor

try:
    import recog_layout as rl
except ImportError:
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "recog_synth"))
    import recog_layout as rl


def check_target(tokenizer, target, newline_id):
    """The problems with one target, or []."""
    ids = tokenizer(target, return_token_type_ids=False, return_attention_mask=False)["input_ids"]
    problems = []
    back = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    if back != f"{tokenizer.bos_token}{target}{tokenizer.eos_token}":
        problems.append(f"decodes to {back[:120]!r}")
    special = set(tokenizer.all_special_ids) - {tokenizer.bos_token_id, tokenizer.eos_token_id}
    inner = ids[1:-1]
    if any(i in special or i >= tokenizer.vocab_size for i in inner):
        problems.append("holds a special or added token: " + ", ".join(
            sorted({tokenizer.convert_ids_to_tokens(i) for i in inner if i in special or i >= tokenizer.vocab_size})))
    if inner.count(newline_id) != target.count(rl.SEP):
        problems.append(f"{target.count(rl.SEP)} row breaks but {inner.count(newline_id)} newline tokens")
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifests", nargs="+")
    ap.add_argument("--max-rows", type=int, default=0, help="rows checked per manifest (0 = all)")
    args = ap.parse_args()
    tokenizer = load_processor().tokenizer
    newline = tokenizer(rl.SEP, add_special_tokens=False)["input_ids"]
    if len(newline) != 1:
        sys.exit(f"the separator {rl.SEP!r} is {len(newline)} tokens, not one")
    failed = checked = 0
    for path in args.manifests:
        with open(path, encoding="utf-8") as f:
            for number, line in enumerate(f, 1):
                if not line.strip():
                    continue
                if args.max_rows and number > args.max_rows:
                    break
                row = json.loads(line)
                checked += 1
                problems = check_target(tokenizer, row["target"], newline[0])
                if problems:
                    failed += 1
                    if failed <= 20:
                        print(f"{path}:{number} {row.get('image')}: " + "; ".join(problems))
    print(f"{checked:,} targets checked, {failed:,} failed; separator {rl.SEP!r} is token "
          f"{tokenizer.convert_ids_to_tokens(newline[0])!r} ({newline[0]})")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
