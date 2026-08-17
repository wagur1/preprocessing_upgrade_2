"""Task analyzer interface.

The "vision task analyzer" of the paper is a *fixed* (frozen) pretrained
network. During preprocessor training it supplies the accuracy loss L_Acc;
during evaluation it supplies the task metric. Both tasks in the paper (action
recognition, object tracking) plug in through this common interface, so the
preprocessor + codec + training loop stay task-agnostic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn


class TaskAnalyzer(nn.Module, ABC):
    """Frozen downstream analyzer.

    Subclasses receive video in [0, 1] with shape [B, C, T, H, W] and must:
      * ``accuracy_loss`` -> differentiable scalar for training (L_Acc),
      * ``predict``       -> outputs used to compute the eval metric.
    """

    task_name: str = "base"

    def freeze(self) -> "TaskAnalyzer":
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()
        return self

    @abstractmethod
    def accuracy_loss(
        self, x_hat: torch.Tensor, target: Any
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Return (L_Acc, aux) where aux may carry logits/preds for logging."""

    @abstractmethod
    @torch.no_grad()
    def predict(self, x_hat: torch.Tensor) -> Any:
        """Return predictions for metric computation (no grad)."""

    def features(self, x: torch.Tensor) -> list:
        """Intermediate feature maps for the distillation loss (differentiable).

        Default: none (distillation disabled). Override to tap the frozen
        backbone's semantic layers. Input is video [B,C,T,H,W] in [0,1]."""
        return []


def build_task(cfg: dict) -> TaskAnalyzer:
    """Factory: cfg['task']['name'] in {'action_recognition', 'tracking'}."""
    from .action_recognition import ActionRecognitionAnalyzer
    from .tracking import TrackingAnalyzer

    name = cfg["task"]["name"]
    if name == "action_recognition":
        return ActionRecognitionAnalyzer(
            backbone=cfg["task"].get("backbone", "r3d_18"),
            clip_size=cfg["task"].get("clip_size", 112),
        ).freeze()
    if name == "tracking":
        return TrackingAnalyzer(
            backbone=cfg["task"].get("backbone", "resnet18"),
            exemplar_size=cfg["task"].get("exemplar_size", 127),
            search_size=cfg["task"].get("search_size", 255),
            pos_radius=cfg["task"].get("pos_radius", 2.0),
        ).freeze()
    raise ValueError(f"unknown task '{name}'")
