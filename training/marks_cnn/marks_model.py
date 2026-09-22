"""
marks_model.py — the mark separation network and its loss.

MarkSeparator takes a 1 x 100 x 600 crop as ink (1 - grey / 255, so paper is 0) and
returns (1 x 100 x 600 logits of the writing's ink, 4 label logits: whether the crop
holds an underline, strikethrough, oval, rectangle — marks_data.MARK_TYPES, several at
once). Ten 3 x 3 convolutions, each followed by batch norm and ReLU, a 1 x 1 head for the
writing, and a linear head on the middle's pooled features for the labels:

    100 x 600   enc1    1 -> 32, 32 -> 32                     skip 1
     50 x 300   enc2   32 -> 64, 64 -> 64                     skip 2
     25 x 150   enc3   64 -> 128                              skip 3
     12 x 75    mid   128 -> 128 dilation 2, 128 -> 128 dilation 4
                       -> mean and max over the map, 256 -> 4  labels
     25 x 150   dec3  128 + skip 3 -> 128
     50 x 300   dec2  128 + skip 2 -> 64
    100 x 600   dec1   64 + skip 1 -> 32
                head   32 -> 1, 1 x 1                          writing

Downsampling is 2 x 2 max pooling, upsampling bilinear to the skip's size. The dilated
middle gives a receptive field of a few hundred pixels, so a stroke is judged by the
whole oval or underline it belongs to rather than its neighbourhood alone, and the
labels are read from the same view of the crop.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from marks_data import MARK_TYPES


def conv(cin, cout, dilation=1):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=dilation, dilation=dilation, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class MarkSeparator(nn.Module):
    def __init__(self, width=32, labels=len(MARK_TYPES)):
        super().__init__()
        c1, c2, c3 = width, 2 * width, 4 * width
        self.enc1 = nn.Sequential(conv(1, c1), conv(c1, c1))
        self.enc2 = nn.Sequential(conv(c1, c2), conv(c2, c2))
        self.enc3 = conv(c2, c3)
        self.mid = nn.Sequential(conv(c3, c3, dilation=2), conv(c3, c3, dilation=4))
        self.dec3 = conv(c3 + c3, c3)
        self.dec2 = conv(c3 + c2, c2)
        self.dec1 = conv(c2 + c1, c1)
        self.head = nn.Conv2d(c1, 1, 1)
        self.classify = nn.Linear(2 * c3, labels)

    @staticmethod
    def up(x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([x, skip], 1)

    def forward(self, x):
        s1 = self.enc1(x)
        s2 = self.enc2(F.max_pool2d(s1, 2))
        s3 = self.enc3(F.max_pool2d(s2, 2))
        x = self.mid(F.max_pool2d(s3, 2))
        labels = self.classify(torch.cat([x.mean((2, 3)), x.amax((2, 3))], 1))
        x = self.dec3(self.up(x, s3))
        x = self.dec2(self.up(x, s2))
        x = self.dec1(self.up(x, s1))
        return self.head(x), labels


def build_model(width=32, labels=len(MARK_TYPES)):
    return MarkSeparator(width, labels)


def from_checkpoint(checkpoint):
    """The network in a loaded model.pt, its weights in place."""
    config = checkpoint.get("model_config") or {}
    if "labels" not in config:
        raise ValueError("this model.pt is the older mark separation network (a writing and a mark image, no labels); "
                         "train a new one")
    model = build_model(**config)
    model.load_state_dict(checkpoint["model"])
    return model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def split_batch(batch):
    """uint8 (B, 2, H, W) crops -> (input ink (B, 1, H, W), writing ink (B, 1, H, W)), floats in [0, 1]."""
    ink = 1.0 - batch.float() / 255.0
    return ink[:, :1], ink[:, 1:2]


def mark_loss(writing_logits, writing, label_logits, labels, label_weight=1.0):
    """(loss, {"bce", "dice", "label_bce"}): the writing image's binary cross-entropy over every pixel plus soft
    Dice per crop, so the sparse ink weighs as much as the paper, plus label_weight x the binary cross-entropy of
    the four labels (each its own yes / no). Computed in float32."""
    writing_logits, writing = writing_logits.float(), writing.float()
    bce = F.binary_cross_entropy_with_logits(writing_logits, writing)
    prob = torch.sigmoid(writing_logits)
    inter = (prob * writing).sum((2, 3))
    total = prob.sum((2, 3)) + writing.sum((2, 3))
    dice = 1.0 - ((2.0 * inter + 1.0) / (total + 1.0)).mean()
    label_bce = F.binary_cross_entropy_with_logits(label_logits.float(), labels.float())
    return bce + dice + label_weight * label_bce, {"bce": bce.detach(), "dice": dice.detach(), "label_bce": label_bce.detach()}
