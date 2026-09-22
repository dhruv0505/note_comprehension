"""
model.py — Florence-2-large with LoRA adapters on its language model, ready to train.

    python model.py          # load, wrap, print trainable vs total parameter counts

Florence-2 is a DaViT vision tower feeding a BART-style encoder-decoder
(config: is_encoder_decoder=True). The hub maps it to AutoModelForCausalLM, but
the generation head is seq2seq, so the peft task type is SEQ_2_SEQ_LM.

Linear layers, from model.named_modules():
  language_model.model.encoder.layers.{0..11}.self_attn.{q,k,v,out}_proj, .fc1, .fc2
  language_model.model.decoder.layers.{0..11}.{self_attn,encoder_attn}.{q,k,v,out}_proj, .fc1, .fc2
  language_model.lm_head                                     (tied to the embeddings)
  vision_tower.blocks...{qkv, proj, fn.net.fc1, fn.net.fc2}

There is no o_proj (the output projection is out_proj), and the vision tower has
its own fc1/fc2, so a plain list of suffixes would put LoRA on the vision tower
too. LORA_TARGETS is one regex anchored at the text encoder/decoder layers:
12 x 6 encoder + 12 x 10 decoder = 192 wrapped layers, none in the vision tower.
"""

from __future__ import annotations
import os

HERE = os.path.dirname(os.path.abspath(__file__))
# model and processor downloads stay inside this folder
os.environ.setdefault("HF_HOME", os.path.join(HERE, "hf_cache"))

import torch
from peft import LoraConfig, TaskType, get_peft_model

MODEL_ID = "microsoft/Florence-2-large"
LORA_TARGETS = (r"language_model\.model\.(encoder|decoder)\.layers\.\d+\."
                r"((self_attn|encoder_attn)\.(q_proj|k_proj|v_proj|out_proj)|fc1|fc2)")
# with vision_lora: the DaViT tower's own linear layers as well, so what the decoder reads can adapt to pen strokes.
# Its weights stay frozen; only these adapters train.
VISION_TARGETS = r"vision_tower\..*\.(qkv|proj|fn\.net\.fc1|fn\.net\.fc2)"


def pick_dtype():
    if not torch.cuda.is_available():
        return torch.float32            # a CPU smoke run; half precision is for the GPU
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def load_base(dtype):
    """Florence-2's remote code imports flash_attn unconditionally; it is not
    needed (or installable here), so it is dropped from the import check."""
    from unittest.mock import patch
    from transformers import AutoModelForCausalLM, AutoProcessor
    from transformers.dynamic_module_utils import get_imports

    def without_flash_attn(filename):
        return [name for name in get_imports(filename) if name != "flash_attn"]

    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    with patch("transformers.dynamic_module_utils.get_imports", without_flash_attn):
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True,
                                                     torch_dtype=dtype, attn_implementation="eager")
    return model, processor


def build_model(rank=16, alpha=None, vision_lora=False):
    """Returns (peft model, processor, tokenizer). Only the LoRA matrices are
    trainable: the vision tower is frozen here, and get_peft_model freezes every
    other base weight. peft keeps the adapter weights in float32 on top of the
    half-precision base. `alpha` defaults to 2 x `rank`, the scale rank 16 /
    alpha 32 has always had."""
    dtype = pick_dtype()
    base, processor = load_base(dtype)

    for param in base.vision_tower.parameters():
        param.requires_grad = False

    targets = f"({LORA_TARGETS})|({VISION_TARGETS})" if vision_lora else LORA_TARGETS
    config = LoraConfig(r=rank, lora_alpha=2 * rank if alpha is None else alpha, lora_dropout=0.05, bias="none",
                        target_modules=targets, task_type=TaskType.SEQ_2_SEQ_LM)
    model = get_peft_model(base, config)
    if torch.cuda.is_available():
        model = model.to("cuda")
    return model, processor, processor.tokenizer


def report_parameters(model):
    """Prints trainable vs total parameters and fails if anything other than a
    LoRA matrix outside the vision tower is trainable."""
    trainable, total = model.get_nb_trainable_parameters()
    trainable_names = [name for name, p in model.named_parameters() if p.requires_grad]
    wrapped = [name for name, module in model.named_modules() if hasattr(module, "lora_A")]
    vision = sum(p.numel() for name, p in model.named_parameters() if ".vision_tower." in name)
    base_dtype = model.get_base_model().language_model.lm_head.weight.dtype
    adapter_dtype = next(p for name, p in model.named_parameters() if "lora_A" in name).dtype

    print(f"  trainable params         {trainable:,}")
    print(f"  total params             {total:,}")
    print(f"  trainable fraction       {100 * trainable / total:.3f}%")
    print(f"  LoRA-wrapped layers      {len(wrapped)}  (expect 192, in vision tower: "
          f"{sum('.vision_tower.' in name for name in wrapped)})")
    print(f"  vision tower params      {vision:,}  (all frozen)")
    print(f"  dtype                    base {base_dtype}, adapters {adapter_dtype}")

    # a vision-tower adapter is trainable on purpose with vision_lora; the tower's own weights never are
    stray = [name for name in trainable_names if "lora_" not in name]
    if stray:
        raise RuntimeError(f"non-LoRA parameters are trainable: {stray[:5]}")


def main():
    model, _, _ = build_model()
    report_parameters(model)


if __name__ == "__main__":
    main()
