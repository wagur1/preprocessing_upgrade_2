#!/usr/bin/env python
"""One-shot Kaggle driver: prepare (<=3 GB) -> train -> evaluate.

Task-aware. The task is read from the chosen config:

  * action_recognition (Kinetics-400 5%)::

        !git clone https://github.com/wagur1/preprocessing_upgrade_2.git
        %cd preprocessing_upgrade_2
        !pip install -q compressai
        !python kaggle/run_kaggle.py --epochs 3 --max-steps 300

  * tracking (GOT-10k val)::

        !python kaggle/run_kaggle.py --config configs/tracking.yaml \
            --epochs 3 --max-steps 300

It (1) locates the dataset under /kaggle/input, (2) builds an index capped at
3 GB, (3) trains the preprocessor, and (4) evaluates prep+CompressAI against the
CompressAI-only ablation and bare H.264 / H.265, writing curves + BD-Rate.
For tracking the eval reports real GOT-10k success AUC (+AO, SR).

Everything is parameterised so you can trade run-time for fidelity via
--epochs / --max-steps / --cap-gb / --frame-size / --max-seqs / --max-frames.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# make "src" importable when run from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import apply_overrides, load_config  # noqa: E402
from src.engine import evaluate, train  # noqa: E402

_VIDEO_EXTS = (".mp4", ".avi", ".mkv", ".mov", ".webm")


def _seed_everything(seed: int) -> None:
    """Seed python/torch RNGs so a --seed change gives an independent run (CI)."""
    import random as _r

    _r.seed(seed)
    try:
        import numpy as _np
        _np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch as _t
        _t.manual_seed(seed)
        if _t.cuda.is_available():
            _t.cuda.manual_seed_all(seed)
    except Exception:
        pass


# --------------------------------------------------------------------------
# dataset autodetection
# --------------------------------------------------------------------------
def _count_under(root: Path, predicate) -> int:
    n = 0
    for _dp, _dn, files in os.walk(root):
        for fn in files:
            if predicate(fn):
                n += 1
    return n


def _autodetect_kinetics(explicit: str | None) -> str:
    if explicit and Path(explicit).exists():
        return explicit
    for c in ("/kaggle/input/kinetics-train-5per",
              "/kaggle/input/kinetics-train-5per/train"):
        if Path(c).exists():
            return c
    base = Path("/kaggle/input")
    if base.exists():
        best, best_n = None, -1
        for child in base.iterdir():
            if not child.is_dir():
                continue
            n = _count_under(child, lambda fn: fn.lower().endswith(_VIDEO_EXTS))
            if n > best_n:
                best, best_n = child, n
        if best is not None and best_n > 0:
            print(f"[kaggle] autodetected Kinetics dataset: {best} ({best_n} videos)")
            return str(best)
    raise FileNotFoundError("could not locate Kinetics; pass --dataset-dir explicitly")


def _autodetect_got10k(explicit: str | None) -> str:
    """Find a GOT-10k split dir (has sub-dirs each containing groundtruth.txt)."""
    if explicit and Path(explicit).exists():
        return explicit
    for c in ("/kaggle/input/got10k/val", "/kaggle/input/got-10k/val",
              "/kaggle/input/got10k-val", "/kaggle/input/got10k",
              "/kaggle/input/got-10k"):
        if Path(c).exists():
            return c
    # Fall back: the /kaggle/input subtree with the most groundtruth.txt files,
    # preferring a 'val' dir if present.
    base = Path("/kaggle/input")
    if base.exists():
        best, best_n = None, -1
        for child in base.iterdir():
            if not child.is_dir():
                continue
            n = _count_under(child, lambda fn: fn == "groundtruth.txt")
            if n > best_n:
                best, best_n = child, n
        if best is not None and best_n > 0:
            val = best / "val"
            chosen = val if val.exists() else best
            print(f"[kaggle] autodetected GOT-10k root: {chosen} "
                  f"({best_n} sequences under {best})")
            return str(chosen)
    raise FileNotFoundError("could not locate GOT-10k; pass --dataset-dir explicitly")


# --------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description="Kaggle prepare+train+eval driver.")
    p.add_argument("--config", default="configs/action_recognition.yaml")
    p.add_argument("--dataset-dir", default=None, help="dataset root (auto if omitted)")
    p.add_argument("--index", default=None, help="index JSON path (task default if omitted)")
    p.add_argument("--cap-gb", type=float, default=3.0)
    p.add_argument("--val-frac", type=float, default=None,
                   help="val fraction (default 0.1 classification / 0.3 tracking)")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--beta", type=float, default=None,
                   help="rate weight (loss.beta); raise to make bitrate actually bite")
    p.add_argument("--lam-task", type=float, default=None,
                   help="task-accuracy weight (loss.lam_task)")
    p.add_argument("--omega", type=float, default=None,
                   help="feature-distillation weight (loss.omega)")
    p.add_argument("--tau", type=float, default=None,
                   help="temporal-consistency weight (loss.tau)")
    p.add_argument("--delta", type=float, default=None,
                   help="edit-magnitude weight (loss.delta); L1 |x_pre-x|, direct edit-sparsity lever")
    p.add_argument("--res-scale", type=float, default=None,
                   help="scale on the learned residual (model.res_scale); <1 shrinks edit amplitude")
    p.add_argument("--frame-size", type=int, default=None)
    p.add_argument("--max-seqs", type=int, default=None, help="tracking eval: cap val seqs")
    p.add_argument("--max-frames", type=int, default=None, help="tracking eval: cap frames/seq")
    p.add_argument("--out-dir", default="outputs")
    p.add_argument("--resume", action="store_true",
                   help="continue training from outputs/checkpoints/preprocessor_last.pth")
    p.add_argument("--patience", type=int, default=None,
                   help="early-stop patience in epochs (0 = off; overrides config)")
    p.add_argument("--skip-prepare", action="store_true")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--ckpt", default=None, help="checkpoint for eval if skipping train")
    p.add_argument("--seed", type=int, default=0,
                   help="seed for data subset/split + training (vary it for multi-seed CI)")
    args = p.parse_args()

    _seed_everything(args.seed)
    cfg = load_config(args.config)
    task = cfg["task"]["name"]
    is_tracking = task == "tracking"

    index = args.index or (
        "data/index/got10k_3gb.json" if is_tracking else "data/index/kinetics_3gb.json"
    )
    val_frac = args.val_frac if args.val_frac is not None else (0.3 if is_tracking else 0.1)

    cfg["out_dir"] = args.out_dir
    cfg["data"]["index"] = index

    # 1) build the <=3 GB index (task-specific)
    if not args.skip_prepare:
        if is_tracking:
            from src.data.prepare_got10k import build_index as build_got10k_index
            root = _autodetect_got10k(args.dataset_dir)
            build_got10k_index(root=root, out=index, cap_gb=args.cap_gb,
                               val_frac=val_frac, seed=args.seed)
        else:
            from src.data.prepare_3gb import build_index as build_kinetics_index
            root = _autodetect_kinetics(args.dataset_dir)
            build_kinetics_index(
                root=root, out=index, cap_gb=args.cap_gb, val_frac=val_frac,
                backbone=cfg["task"].get("backbone", "r3d_18"), seed=args.seed,
            )

    # config overrides for a Kaggle-sized run
    ov = [f"train.epochs={args.epochs}"]
    if args.max_steps is not None:
        ov.append(f"train.max_steps={args.max_steps}")
    if args.batch_size is not None:
        ov += [f"train.batch_size={args.batch_size}"]
        if not is_tracking:
            ov.append(f"eval.batch_size={args.batch_size}")
    if args.frame_size is not None:
        ov.append(f"data.frame_size={args.frame_size}")
    if args.max_seqs is not None:
        ov.append(f"eval.max_seqs={args.max_seqs}")
    if args.max_frames is not None:
        ov.append(f"eval.max_frames={args.max_frames}")
    apply_overrides(cfg, ov)
    cfg["train"]["resume"] = bool(args.resume)
    if args.patience is not None:
        cfg["train"]["patience"] = args.patience
    loss_cfg = cfg.setdefault("loss", {})
    if args.beta is not None:
        loss_cfg["beta"] = args.beta
    if args.lam_task is not None:
        loss_cfg["lam_task"] = args.lam_task
    if args.omega is not None:
        loss_cfg["omega"] = args.omega
    if args.tau is not None:
        loss_cfg["tau"] = args.tau
    if args.delta is not None:
        loss_cfg["delta"] = args.delta
    if args.res_scale is not None:
        cfg["model"]["res_scale"] = args.res_scale

    # 2) train
    ckpt = args.ckpt
    if not args.skip_train:
        ckpt = train(cfg)
    if ckpt is None:
        ckpt = str(Path(args.out_dir) / "checkpoints" / "preprocessor.pth")

    # 3) evaluate: prep+CompressAI vs CompressAI / H.264 / H.265, BD-Rate
    evaluate(cfg, ckpt)


if __name__ == "__main__":
    main()
