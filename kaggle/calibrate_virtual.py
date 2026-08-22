#!/usr/bin/env python
"""Calibrate the VirtualCodec quant step to a target *training* bpp band.

The honest eval bitrate is always real x264/x265; this only sets where the
differentiable TRAINING proxy operates. Training must sit at a healthy bpp
(~0.1-0.5), NOT ~0 -- a proxy coarse enough to erase the signal gives the
preprocessor nothing to learn and it collapses to a constant image (what the
step_coarse/fine=3.0/1.0 run did: train bpp ~0.004, eval accuracy 0).

Runs the codec on a handful of REAL clips at each candidate step and prints
(step -> bpp). Then set, in the config:
    codec.step_coarse = step giving ~0.1 bpp   (coarsest / lowest quality id)
    codec.step_fine   = step giving ~0.5 bpp   (finest  / highest quality id)

Usage:
    python kaggle/calibrate_virtual.py --config configs/action_recognition.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# make "src" importable when run as `python kaggle/calibrate_virtual.py`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader

from src.config import load_config
from src.data import collate_clips
from src.data.video_dataset import VideoClipDataset
from src.models.virtual_codec import VirtualCodec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/action_recognition.yaml")
    ap.add_argument("--index", default="data/index/kinetics_3gb.json")
    ap.add_argument("--clips", type=int, default=16, help="real clips to average bpp over")
    ap.add_argument("--steps", default="0.03,0.05,0.1,0.2,0.35,0.5,0.75,1.0,2.0,3.0",
                    help="comma-separated quant steps to probe")
    args = ap.parse_args()

    cfg = load_config(args.config)
    d, cc = cfg["data"], cfg["codec"]
    ds = VideoClipDataset(
        index_json=args.index, split="train", train=False,
        num_frames=d.get("num_frames", 16),
        frame_size=d.get("frame_size", 128),
        temporal_stride=d.get("temporal_stride", 2),
    )
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=2,
                        collate_fn=collate_clips)
    got = []
    for clips, _ in loader:
        got.append(clips)
        if sum(c.shape[0] for c in got) >= args.clips:
            break
    x = torch.cat(got)[:args.clips]  # [N,C,T,H,W] in [0,1]
    print(f"[calibrate] {x.shape[0]} clips {tuple(x.shape[1:])}  block={cc.get('block',8)} "
          f"inter={cc.get('inter',True)}")
    print(f"{'step':>8} {'bpp':>9}   target train band ~0.1-0.5")
    for s in (float(v) for v in args.steps.split(",")):
        cod = VirtualCodec(qualities=(1,), block=cc.get("block", 8),
                           q_steps={1: s}, inter=cc.get("inter", True))
        _, bpp = cod.compress_decompress(x, 1)
        flag = "  <-- coarse end" if 0.08 <= bpp <= 0.13 else ("  <-- fine end" if 0.4 <= bpp <= 0.6 else "")
        print(f"{s:8.3f} {bpp:9.4f}{flag}")


if __name__ == "__main__":
    main()
