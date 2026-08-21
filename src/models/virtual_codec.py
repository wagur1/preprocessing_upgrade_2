"""Differentiable block-transform virtual codec (upgrade2).

Rebuilds the paper's hand-crafted virtual codec (Zhao et al.) so the TRAINING
proxy matches x264/x265 *geometry* -- block DCT + scalar quantisation + P-frame
prediction. CompressAI's learned wavelet-ish transform does not have that
geometry, which is the prime suspect for why edits trained against it failed to
transfer to the real block-DCT codecs.

Pipeline, per clip ``[B,C,T,H,W]`` in [0,1]:

    predict (I-frame: none; P-frame: previous SOURCE frame)  -> residual r
    r  -> block DCT (bs x bs, orthonormal)                   -> coeffs
    coeffs / step(quality)  -> y            (step = quantiser coarseness knob)
    y  -> quantise (add-noise train / round eval)            -> y_hat
    rate = per-frequency Gaussian entropy of y  (factorised, parameter-free)
    y_hat * step  -> inverse block DCT -> r_hat -> x_hat = pred + r_hat

Faithful: block DCT, block-wise scalar quant, P-frame temporal prediction, a
factorised per-frequency rate. Deliberate proxy corners (ponytail):
  * open-loop prediction -- reference is the SOURCE previous frame, not the
    reconstructed one, so no drift term. Closed-loop if drift ever matters.
  * parameter-free Gaussian rate instead of a *trained* Balle factorised prior,
    so the codec stays frozen and the optimiser still touches only the
    preprocessor. Swap in a trained EntropyBottleneck if the rate proves coarse.
The honest eval bitrate always comes from real x264/x265 in engine.py; this
module only supplies the differentiable rate+distortion signal during training.
"""

from __future__ import annotations

import math
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


def _dct_basis(n: int) -> torch.Tensor:
    """Orthonormal DCT-II basis ``D`` [n,n] with ``D @ D.T == I``."""
    k = torch.arange(n).view(n, 1).float()
    m = torch.arange(n).view(1, n).float()
    d = torch.cos(math.pi * (2 * m + 1) * k / (2 * n))
    d[0] *= math.sqrt(1.0 / n)
    d[1:] *= math.sqrt(2.0 / n)
    return d


class VirtualCodec(nn.Module):
    """Block-transform differentiable proxy; drop-in for ``CompressAICodec``.

    Args:
        qualities: level ids (kept identical to the CompressAI setup so
            ``qp_to_quality`` and ``_quality_conds`` need no change). Each maps
            to a quantiser step -- higher id = finer step = more bits.
        block: DCT block size (8 like JPEG / an H.26x transform size).
        q_steps: optional explicit {quality: step}. If absent, steps are
            geometrically interpolated ``step_coarse -> step_fine`` as the
            quality id rises. These are physical calibration knobs (they set
            where the rate curve lands); tune them to overlap the x264/x265 bpp
            range, not blindly.
        inter: enable P-frame prediction (previous source frame as reference).
    """

    def __init__(
        self,
        qualities: List[int] | int = (1, 2, 3, 5, 8),
        block: int = 8,
        q_steps: Dict[int, float] | None = None,
        step_coarse: float = 0.25,
        step_fine: float = 0.03,
        inter: bool = True,
    ):
        super().__init__()
        if isinstance(qualities, int):
            qualities = (qualities,)
        self.qualities = list(qualities)
        self.block = int(block)
        self.inter = bool(inter)
        self.register_buffer("_D", _dct_basis(self.block), persistent=False)
        if q_steps:
            self._steps = {int(q): float(s) for q, s in q_steps.items()}
        else:
            qs = sorted(self.qualities)
            lo, hi = qs[0], qs[-1]
            self._steps = {
                q: (step_fine if hi == lo
                    else step_coarse * (step_fine / step_coarse) ** ((q - lo) / (hi - lo)))
                for q in qs
            }

    # -- geometry helpers --------------------------------------------------
    def _pad(self, x: torch.Tensor):
        h, w = x.shape[-2:]
        bs = self.block
        nh, nw = math.ceil(h / bs) * bs, math.ceil(w / bs) * bs
        return F.pad(x, (0, nw - w, 0, nh - h), mode="replicate"), (h, w)

    @staticmethod
    def _crop(x: torch.Tensor, hw) -> torch.Tensor:
        h, w = hw
        return x[..., :h, :w]

    def _residual(self, x: torch.Tensor):
        """P-frame residual: pred = previous source frame (frame 0 = intra)."""
        if not self.inter or x.shape[2] < 2:
            return x, torch.zeros_like(x)
        pred = torch.zeros_like(x)
        pred[:, :, 1:] = x[:, :, :-1]
        return x - pred, pred

    # -- block DCT / inverse (channel layout: [N, C*bs*bs, H/bs, W/bs]) ----
    def _dct(self, r: torch.Tensor) -> torch.Tensor:
        N, C, H, W = r.shape
        bs, D = self.block, self._D
        b = r.view(N, C, H // bs, bs, W // bs, bs)          # [n,c,i,u,j,v]
        coeff = torch.einsum("ku,lv,nciujv->ncikjl", D, D, b)
        return coeff.permute(0, 1, 3, 5, 2, 4).reshape(N, C * bs * bs, H // bs, W // bs)

    def _idct(self, coeff: torch.Tensor, C: int, H: int, W: int) -> torch.Tensor:
        N, bs, D = coeff.shape[0], self.block, self._D
        c = coeff.view(N, C, bs, bs, H // bs, W // bs).permute(0, 1, 4, 2, 5, 3)
        b = torch.einsum("ku,lv,ncikjl->nciujv", D, D, c)   # [n,c,i,u,j,v]
        return b.reshape(N, C, H, W)

    # -- quantise + factorised per-frequency rate --------------------------
    def _quant_rate(self, coeff: torch.Tensor, step: float, training: bool):
        y = coeff / step
        if training:
            y_hat = y + torch.empty_like(y).uniform_(-0.5, 0.5)
        else:
            y_hat = torch.round(y)
        # bits/coeff = rate of a Gaussian source at SNR = signal power / quantiser
        # noise power (uniform step noise, var 1/12): R = 0.5*log2(1 + 12*E[y^2]).
        # Goes to 0 as a coarse step drives the signal below the quantiser -- no
        # spurious floor. (The earlier differential-entropy form bottomed out at
        # ~0.77 bpp, pinning the proxy ~20x above the x264/x265 operating range
        # and training the preprocessor in a near-lossless regime.)
        power = y.pow(2).mean(dim=(0, 2, 3))
        bits = 0.5 * torch.log2(1.0 + 12.0 * power)
        per_ch = y.shape[0] * y.shape[2] * y.shape[3]
        return y_hat, (bits * per_ch).sum()

    # -- shared code path --------------------------------------------------
    def _code(self, x: torch.Tensor, quality: int, training: bool):
        B, C, T, H, W = x.shape
        step = self._steps[int(quality)]
        r, pred = self._residual(x)
        frames = r.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        frames, hw = self._pad(frames)
        ph, pw = frames.shape[-2:]
        y_hat, total_bits = self._quant_rate(self._dct(frames), step, training)
        r_hat = self._crop(self._idct(y_hat * step, C, ph, pw), hw)
        r_hat = r_hat.reshape(B, T, C, H, W).permute(0, 2, 1, 3, 4)
        x_hat = (pred + r_hat).clamp(0.0, 1.0)
        return x_hat, total_bits / (B * T * H * W)

    # -- training path (differentiable) ------------------------------------
    def forward(self, x: torch.Tensor, quality: int):
        return self._code(x, quality, training=True)

    # -- eval path (estimated bpp; honest bitrate comes from x264/x265) ----
    @torch.no_grad()
    def compress_decompress(self, x: torch.Tensor, quality: int):
        x_hat, bpp = self._code(x, quality, training=False)
        return x_hat, float(bpp)


def _demo() -> None:
    torch.manual_seed(0)
    cod = VirtualCodec(qualities=(1, 2, 3, 5, 8), block=8)
    # DCT basis is orthonormal
    D = cod._D
    assert torch.allclose(D @ D.T, torch.eye(8), atol=1e-5), "DCT not orthonormal"
    # DCT round-trip (no quant) is identity
    r = torch.rand(2, 3, 32, 32)
    assert torch.allclose(cod._idct(cod._dct(r), 3, 32, 32), r, atol=1e-4), "DCT round-trip"
    x = torch.rand(2, 3, 4, 32, 32)
    # fine step -> near-identity reconstruction
    xf, _ = cod.compress_decompress(x, 8)
    assert (xf - x).abs().mean() < 0.05, "fine step should reconstruct well"
    # coarser quality id -> fewer bits (monotone rate knob)
    _, bpp_fine = cod.compress_decompress(x, 8)
    _, bpp_coarse = cod.compress_decompress(x, 1)
    assert bpp_coarse < bpp_fine, (bpp_coarse, bpp_fine)
    # forward is differentiable and feeds gradient to its input
    xin = torch.rand(2, 3, 4, 32, 32, requires_grad=True)
    xh, bpp = cod(xin, 3)
    (xh.mean() + bpp).backward()
    assert xin.grad is not None and torch.isfinite(bpp)
    print(f"virtual_codec self-check passed (bpp {bpp_coarse:.3f} < {bpp_fine:.3f})")


if __name__ == "__main__":
    _demo()

