"""Training objective for the machine-vision video preprocessor (upgrade2).

    L = lam_task * L_task + omega * L_distill + beta * L_rate + tau * L_temp

Deliberately **no MSE-to-source distortion term**. That term (the baseline's
``L_D``) pins the reconstruction to the original pixels and directly fights
compression, which is why the baseline never reached negative BD-Rate. Following
Yang et al. (TCSVT 2024) we replace pixel fidelity with a *task-aligned* feature
distillation term and let a real rate weight bite.

  * ``L_task``   : accuracy loss from the frozen analyzer on the reconstruction
                   (cross-entropy for recognition; SiamFC logistic for tracking).
  * ``L_distill``: MSE between the frozen analyzer's intermediate features on the
                   *source* and on the *reconstruction* -- keeps semantics the
                   codec would otherwise destroy (helps most at low bitrate).
  * ``L_rate``   : estimated bits-per-pixel from the codec entropy model.
  * ``L_temp``   : temporal consistency -- match the *inter-frame change* of the
                   reconstruction to that of the source. Preserves motion / kills
                   flicker without pinning absolute pixels (the video novelty).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import torch
import torch.nn.functional as F


@dataclass
class LossWeights:
    lam_task: float = 1.0
    omega: float = 0.5   # feature distillation (Yang et al. use 0.5)
    beta: float = 0.1    # rate; raise until bpp actually bites
    tau: float = 0.1     # temporal consistency


def feature_distillation(analyzer, x_source: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    """Scale-normalised MSE between frozen-analyzer features of source vs recon.

    Source features are the (detached) target; gradients flow only through the
    reconstruction path. Returns 0 if the analyzer exposes no features."""
    feats_src = analyzer.features(x_source)
    feats_hat = analyzer.features(x_hat)
    if not feats_src:
        return x_hat.new_zeros(())
    loss = x_hat.new_zeros(())
    for fs, fh in zip(feats_src, feats_hat):
        fs = fs.detach()
        loss = loss + F.mse_loss(fh, fs) / (fs.pow(2).mean() + 1e-6)
    return loss / len(feats_src)


def temporal_consistency(x_source: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    """Match reconstruction inter-frame deltas to source inter-frame deltas."""
    if x_source.shape[2] < 2:
        return x_hat.new_zeros(())
    ds = x_source[:, :, 1:] - x_source[:, :, :-1]
    dh = x_hat[:, :, 1:] - x_hat[:, :, :-1]
    return F.mse_loss(dh, ds)


def preprocessing_loss(
    analyzer,
    x_source: torch.Tensor,
    x_hat: torch.Tensor,
    bpp: torch.Tensor,
    target: Any,
    w: LossWeights,
) -> Dict[str, torch.Tensor]:
    l_task, _ = analyzer.accuracy_loss(x_hat, target)
    l_dist = feature_distillation(analyzer, x_source, x_hat)
    l_temp = temporal_consistency(x_source, x_hat)
    total = w.lam_task * l_task + w.omega * l_dist + w.beta * bpp + w.tau * l_temp
    return {
        "loss": total,
        "loss_task": l_task.detach(),
        "loss_dist": l_dist.detach(),
        "loss_rate": (bpp.detach() if torch.is_tensor(bpp) else torch.as_tensor(bpp)),
        "loss_temp": l_temp.detach(),
    }
