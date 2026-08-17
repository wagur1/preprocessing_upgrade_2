"""Rate- and motion-conditioned U-Net video preprocessor (upgrade2).

This is a clean redesign that drops the two-branch temporal/spatial residual
stack of the baseline (which learned one blurry rate-average and lost to the
codec's own RD curve). Instead we use:

    * a **U-Net backbone** (encoder/decoder + skips) as the pixel editor -- the
      same backbone family Yang et al. (TCSVT 2024) use for the machine-vision
      image preprocessor, which demonstrably reaches negative BD-BR;
    * **FiLM** (Perez et al. 2018) conditioning on the target compression
      operating point (normalised QP), so a *single* model adapts across the
      whole rate range instead of averaging it;
    * **SFT** (Wang et al. CVPR 2018) conditioning on a **motion cue** (temporal
      frame difference), so the preprocessor allocates its edits *spatially*
      toward moving / task-relevant regions -- the video-domain novelty over the
      image preprocessors it builds on.

Video is edited frame-wise (2D convs over [B*T, C, H, W]); the only temporal
signal is the motion cue that drives the SFT, which keeps it cheap on a T4 while
still being motion-aware. Temporal *coherence* of the edit is enforced by the
temporal-consistency loss (see ``src/losses.py``), not by 3D convs.

The residual tail and both modulators are zero-initialised, so the network
starts as an exact identity (stable early training) and only "switches on" its
edits as it learns. Only this module is trained; codec + analyzer are frozen.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _act() -> nn.Module:
    return nn.LeakyReLU(0.1, inplace=True)


class FiLM(nn.Module):
    """Global per-channel affine from the rate condition (zero-init -> identity)."""

    def __init__(self, cond_dim: int, ch: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(cond_dim, ch), _act(), nn.Linear(ch, 2 * ch))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.net(cond).chunk(2, dim=1)  # [N, ch] each
        return x * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]


class SFT(nn.Module):
    """Spatially-varying affine from the motion cue (zero-init -> identity)."""

    def __init__(self, ch: int, cue_ch: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cue_ch, ch, 3, padding=1), _act(), nn.Conv2d(ch, 2 * ch, 3, padding=1)
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor, cue: torch.Tensor) -> torch.Tensor:
        if cue.shape[-2:] != x.shape[-2:]:
            cue = F.interpolate(cue, size=x.shape[-2:], mode="bilinear", align_corners=False)
        gamma, beta = self.net(cue).chunk(2, dim=1)  # [N, ch, H, W] each
        return x * (1.0 + gamma) + beta


class _ConvBlock(nn.Module):
    """conv-act-conv, then FiLM (rate) and SFT (motion) modulation."""

    def __init__(self, in_ch: int, out_ch: int, cond_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.film = FiLM(cond_dim, out_ch)
        self.sft = SFT(out_ch)
        self.act = _act()

    def forward(self, x, cond, cue):
        h = self.act(self.conv1(x))
        h = self.act(self.conv2(h))
        h = self.film(h, cond)
        h = self.sft(h, cue)
        return h


class _UNet(nn.Module):
    """3-level U-Net (full, /2, /4). All blocks FiLM+SFT conditioned."""

    def __init__(self, in_ch: int, base_ch: int, cond_dim: int):
        super().__init__()
        c1, c2, c3 = base_ch, base_ch * 2, base_ch * 4
        self.enc1 = _ConvBlock(in_ch, c1, cond_dim)
        self.enc2 = _ConvBlock(c1, c2, cond_dim)
        self.bott = _ConvBlock(c2, c3, cond_dim)
        self.dec2 = _ConvBlock(c3 + c2, c2, cond_dim)
        self.dec1 = _ConvBlock(c2 + c1, c1, cond_dim)
        self.tail = nn.Conv2d(c1, in_ch, 3, padding=1)
        nn.init.zeros_(self.tail.weight)  # start as identity residual
        nn.init.zeros_(self.tail.bias)

    def forward(self, x, cond, cue):
        e1 = self.enc1(x, cond, cue)                       # full
        e2 = self.enc2(F.avg_pool2d(e1, 2), cond, cue)     # /2
        b = self.bott(F.avg_pool2d(e2, 2), cond, cue)      # /4
        d2 = F.interpolate(b, size=e2.shape[-2:], mode="nearest")
        d2 = self.dec2(torch.cat([d2, e2], dim=1), cond, cue)
        d1 = F.interpolate(d2, size=e1.shape[-2:], mode="nearest")
        d1 = self.dec1(torch.cat([d1, e1], dim=1), cond, cue)
        return self.tail(d1)


class VideoPreprocessor(nn.Module):
    """Rate- and motion-conditioned U-Net preprocessor.

    Args:
        in_ch:    input channels (3 for RGB).
        base_ch:  U-Net width at full resolution.
        res_scale: scales the learned residual before adding to the input.
        cond_dim: rate condition width (1 = normalised QP; sized for later
                  appending an explicit log target-rate for rate control).
    """

    def __init__(self, in_ch: int = 3, base_ch: int = 32, res_scale: float = 1.0,
                 cond_dim: int = 1):
        super().__init__()
        self.cond_dim = cond_dim
        self.res_scale = res_scale
        self.unet = _UNet(in_ch, base_ch, cond_dim)

    @staticmethod
    def _motion_cue(x: torch.Tensor) -> torch.Tensor:
        """[B,C,T,H,W] -> [B,1,T,H,W] per-clip-normalised temporal abs-diff.

        Frame 0 reuses frame 1's diff so the cue is defined for every frame."""
        d = (x[:, :, 1:] - x[:, :, :-1]).abs().mean(dim=1, keepdim=True)  # [B,1,T-1,H,W]
        if d.shape[2] == 0:  # single-frame clip: no motion
            return torch.zeros_like(x[:, :1])
        cue = torch.cat([d[:, :, :1], d], dim=2)  # [B,1,T,H,W]
        peak = cue.amax(dim=(2, 3, 4), keepdim=True)
        return cue / (peak + 1e-6)

    @staticmethod
    def _fold(x):
        b, c, t, h, w = x.shape
        return x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w), (b, t)

    @staticmethod
    def _unfold(x, bt):
        b, t = bt
        n, c, h, w = x.shape
        return x.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        """x: [B,C,T,H,W] in [0,1]. cond: [B, cond_dim] operating point (defaults
        to zeros = top quality). Returns edited video in [0,1]."""
        b, c, t, h, w = x.shape
        if cond is None:
            cond = x.new_zeros(b, self.cond_dim)
        cue = self._motion_cue(x)
        frames, bt = self._fold(x)
        cue_f, _ = self._fold(cue)
        cond_f = cond.repeat_interleave(t, dim=0)  # [B*T, cond_dim]
        delta = self.unet(frames, cond_f, cue_f)
        out = frames + self.res_scale * delta
        return self._unfold(out, bt).clamp(0.0, 1.0)
