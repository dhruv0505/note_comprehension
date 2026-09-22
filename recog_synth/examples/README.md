# Recognition examples (val split)

10 samples each of `textline`, `formula` and `drawing`, made by `recog_synth` (`recog_synthesize.run`) from the val split of the shipped data: IAM-OnDB val forms (splits.json), the MathWriting excerpt's `valid` directory, and QuickDraw sample drawings in the val hash buckets (one per category).

Each folder: `<task>_NN.png` (768 x 768), `manifest.jsonl` (the training rows, `image` relative to the folder), `targets.txt` (each target with its `<sep>` rows listed one per line), `report.json`.
