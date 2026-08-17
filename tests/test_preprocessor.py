"""Checks for the rate/motion-conditioned U-Net preprocessor (needs torch)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.preprocessor import FiLM, SFT, VideoPreprocessor


def main() -> None:
    # (a) FiLM (4D) is exact identity at init for any rate condition.
    x = torch.randn(2, 8, 16, 16)
    film = FiLM(cond_dim=1, ch=8)
    for level in (0.0, 0.5, 1.0):
        assert torch.allclose(film(x, torch.full((2, 1), level)), x), "FiLM init != identity"
    for p in film.parameters():
        torch.nn.init.normal_(p, std=0.1)
    assert not torch.allclose(film(x, torch.zeros(2, 1)), film(x, torch.ones(2, 1))), \
        "FiLM output must depend on the rate condition"

    # (b) SFT is identity at init; depends on the (spatial) motion cue once trained.
    sft = SFT(ch=8)
    cue = torch.rand(2, 1, 16, 16)
    assert torch.allclose(sft(x, cue), x), "SFT init != identity"
    for p in sft.parameters():
        torch.nn.init.normal_(p, std=0.1)
    assert not torch.allclose(sft(x, torch.zeros(2, 1, 16, 16)), sft(x, torch.ones(2, 1, 16, 16))), \
        "SFT output must depend on the motion cue"

    # (c) preprocessor: starts as identity (zero-init tail), keeps shape, cond=None ok.
    pre = VideoPreprocessor(base_ch=8, cond_dim=1)
    vid = torch.rand(2, 3, 4, 64, 64)
    out = pre(vid, torch.full((2, 1), 0.7))
    assert out.shape == vid.shape
    assert torch.allclose(out, vid, atol=1e-6), "zero-init tail -> identity at start"
    assert pre(vid).shape == vid.shape, "cond=None default must run"

    # (d) once the tail is non-zero, the rate condition changes the edit.
    torch.nn.init.normal_(pre.unet.tail.weight, std=0.05)
    y_lo = pre(vid, torch.zeros(2, 1))
    y_hi = pre(vid, torch.ones(2, 1))
    assert not torch.allclose(y_lo, y_hi), "preprocessor output must depend on rate cond"

    # (e) motion cue: zero for a static clip, non-zero when frames move.
    static = torch.ones(1, 3, 4, 8, 8)
    assert VideoPreprocessor._motion_cue(static).abs().max() < 1e-6
    moving = torch.rand(1, 3, 4, 8, 8)
    assert VideoPreprocessor._motion_cue(moving).abs().max() > 0
    print("preprocessor self-check passed")


if __name__ == "__main__":
    main()
