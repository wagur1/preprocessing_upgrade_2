"""Training and evaluation engine.

Training (only the preprocessor learns; codec + analyzer frozen):

    clip x -> preprocessor(x, cond) -> x_pre -> CompressAI(quality q) -> x_hat, bpp
    L = lam_task*L_task + omega*L_distill + beta*bpp + tau*L_temp   (see losses.py)

A QP is sampled per step and mapped to a proxy quality; its normalised value is
fed to the preprocessor's FiLM so one model spans the whole rate range. There is
no MSE-to-source term (it fought compression) and no real-codec-in-the-loop /
STE (incoherent gradients) -- we train fully through the differentiable proxy and
rely on the edit transferring to x264/x265 at eval (Yang et al. TCSVT 2024).

Two tasks dispatch on ``cfg['task']['name']``:

  * ``action_recognition`` -- Kinetics-400 clips, cross-entropy L_task, top-1
    accuracy at eval.
  * ``tracking``           -- GOT-10k clips + boxes, SiamFC logistic L_task,
    real success-plot AUC at eval (run the tracker over each sequence).

Evaluation traces rate-accuracy curves with the *real* entropy/range coders and
compares six pipelines -- prep+{compressai,h264,h265} vs the bare
{compressai,h264,h265} anchors -- then reports BD-Rate (same-codec prep gain is
the real claim).
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .codecs import StandardCodec, ffmpeg_available
from .data import (
    GOT10kClipDataset,
    VideoClipDataset,
    collate_clips,
    collate_got10k,
    iter_sequences,
)
from .losses import LossWeights, preprocessing_loss
from .metrics import aggregate_metrics, bd_metric, bd_rate, sequence_metrics
from .models import CompressAICodec, VideoPreprocessor
from .tasks import build_task


# --------------------------------------------------------------------------
# shared helpers
# --------------------------------------------------------------------------
def _device(cfg: dict) -> torch.device:
    want = cfg.get("device", "auto")
    if want == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(want)


def _build_models(cfg: dict, device: torch.device):
    m = cfg["model"]
    pre = VideoPreprocessor(
        base_ch=m.get("base_ch", 32),
        res_scale=m.get("res_scale", 1.0),
        cond_dim=m.get("cond_dim", 1),
    ).to(device)
    codec = CompressAICodec(
        model=cfg["codec"].get("model", "bmshj2018-factorized"),
        qualities=tuple(cfg["codec"].get("qualities", [1, 2, 3, 4, 5, 6, 7, 8])),
        pretrained=True,
        trainable=False,
    ).to(device)
    analyzer = build_task(cfg).to(device)
    return pre, codec, analyzer


def _optimizer(pre, tr):
    return torch.optim.Adam(pre.parameters(), lr=tr.get("lr", 1e-4))


def _ckpt_path(cfg: dict) -> Path:
    d = Path(cfg.get("out_dir", "outputs")) / "checkpoints"
    d.mkdir(parents=True, exist_ok=True)
    return d / "preprocessor.pth"


# --------------------------------------------------------------------------
# rate conditioning: the operating point fed to the preprocessor's FiLM
# --------------------------------------------------------------------------
def _qp_norm(qp: float, cfg: dict) -> float:
    """Map an x26x QP to a normalised compression level in [0, 1] (1 = most
    compressed). ``model.qp_ref`` sets the reference range spanning train+eval."""
    lo, hi = cfg["model"].get("qp_ref", [20, 51])
    return min(max((float(qp) - lo) / (hi - lo), 0.0), 1.0)


def _quality_conds(cfg: dict, codec: CompressAICodec) -> Dict[int, float]:
    """FiLM condition per CompressAI quality, matching what TRAINING used.

    Training samples a QP and feeds ``cond = qp_norm(qp)`` while running the proxy
    at ``q = qp_to_quality[qp]``. Eval must feed the *same* condition per quality,
    so we invert that map: quality -> mean training QP -> qp_norm. Qualities never
    seen in training (e.g. a higher-rate quality) are extrapolated linearly on the
    (quality, mean-QP) training points, then qp_norm-clamped to [0,1].

    (The old eval used a separate ``_quality_level`` normalisation, so quality 5
    was trained at cond 0.065 but evaluated at 0.429 -- a train/eval mismatch that
    made ``prep+compressai`` not actually in-domain.)"""
    from collections import defaultdict

    qtq = {int(qp): int(q) for qp, q in cfg["train"]["qp_to_quality"].items()}
    by_q: Dict[int, list] = defaultdict(list)
    for qp, q in qtq.items():
        by_q[q].append(qp)
    q2qp = {q: sum(v) / len(v) for q, v in by_q.items()}   # mean training QP per quality

    qs = sorted(q2qp)
    if len(qs) >= 2:  # least-squares line qp ~ a*quality + b for extrapolation
        n, sx = len(qs), sum(qs)
        sy = sum(q2qp[q] for q in qs)
        sxx = sum(q * q for q in qs)
        sxy = sum(q * q2qp[q] for q in qs)
        denom = (n * sxx - sx * sx) or 1.0
        a = (n * sxy - sx * sy) / denom
        b = (sy - a * sx) / n
    else:
        a, b = 0.0, (next(iter(q2qp.values())) if q2qp else 30.0)

    return {q: _qp_norm(q2qp.get(q, a * q + b), cfg) for q in codec.qualities}


def _rate_cond(level: float, batch: int, device, dtype) -> torch.Tensor:
    """Build the [B, cond_dim] condition vector. Currently a single normalised
    rate level; append a log target-rate here for explicit rate control."""
    return torch.full((batch, 1), float(level), device=device, dtype=dtype)


def _training_codec_setup(tr: dict, codec: CompressAICodec):
    """Validate the QP list and its QP->proxy-quality mapping used at train time.

    (STE / real-codec-in-the-loop is dropped in upgrade2: it made forward=x265
    but gradient=proxy, an incoherent mix. Train fully through the proxy; the
    edit transfers to x264/x265 at eval, per Yang et al. TCSVT 2024 Sec IV-F.)"""
    qp_list = [int(qp) for qp in tr.get("qp_list", codec.qualities)]
    if not qp_list:
        raise ValueError("train.qp_list must contain at least one QP")

    raw_mapping = tr.get("qp_to_quality")
    if raw_mapping is None:
        qp_to_quality = {q: q for q in qp_list}
    else:
        qp_to_quality = {int(qp): int(q) for qp, q in raw_mapping.items()}

    missing = [qp for qp in qp_list if qp not in qp_to_quality]
    if missing:
        raise ValueError(f"train.qp_to_quality is missing QPs: {missing}")
    unavailable = sorted({qp_to_quality[qp] for qp in qp_list} - set(codec.qualities))
    if unavailable:
        raise ValueError(f"proxy qualities {unavailable} are not in codec.qualities")

    if raw_mapping is not None:
        ordered = [qp_to_quality[qp] for qp in sorted(qp_list)]
        if any(a < b for a, b in zip(ordered, ordered[1:])):
            raise ValueError("train.qp_to_quality must be monotonic: higher QP -> lower quality")
    return qp_list, qp_to_quality


# --------------------------------------------------------------------------
# training (dispatch)
# --------------------------------------------------------------------------
def _earlystop_update(val, best, min_delta, no_improve, patience):
    """One val observation -> (improved, best, no_improve, stop).

    A drop of at least ``min_delta`` below ``best`` counts as improvement and
    resets the patience counter; otherwise ``no_improve`` grows and we stop once
    it reaches ``patience`` (patience=0 disables early stopping)."""
    if val < best - min_delta:
        return True, val, 0, False
    no_improve += 1
    return False, best, no_improve, bool(patience and no_improve >= patience)


@torch.no_grad()
def _val_loss(pre, codec, analyzer, loader, weights, qp_list, qp_to_quality,
              cfg, prep_batch, max_batches):
    """Mean proxy-only training loss over the val split (coherent stop signal).

    Uses a fixed mid-range QP so the signal is comparable across epochs, and the
    differentiable proxy only (no ffmpeg) so it stays cheap."""
    was_training = pre.training
    pre.eval()
    qp = qp_list[len(qp_list) // 2]
    q = qp_to_quality[qp]
    total, nb = 0.0, 0
    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        clips, target = prep_batch(batch)
        cond = _rate_cond(_qp_norm(qp, cfg), clips.shape[0], clips.device, clips.dtype)
        x_pre = pre(clips, cond)
        x_hat, bpp = codec(x_pre, q)
        parts = preprocessing_loss(analyzer, clips, x_hat, bpp, target, weights, x_pre=x_pre)
        total += parts["loss"].item()
        nb += 1
    if was_training:
        pre.train()
    return total / max(nb, 1)


def train(cfg: dict) -> str:
    if cfg["task"]["name"] == "tracking":
        return _train_tracking(cfg)
    return _train_classification(cfg)


def _fit(cfg, pre, codec, analyzer, train_loader, val_loader, prep_batch,
         tag: str, n_train: int) -> str:
    """Shared training loop: cosine LR, per-epoch val, best/last checkpoints,
    early stopping, and resume. ``prep_batch(batch) -> (clips, acc_loss_fn)``."""
    device = next(pre.parameters()).device
    tr = cfg["train"]
    lw = cfg.get("loss", {})
    weights = LossWeights(
        lam_task=lw.get("lam_task", 1.0),
        omega=lw.get("omega", 0.5),
        beta=lw.get("beta", 0.1),
        tau=lw.get("tau", 0.1),
        delta=lw.get("delta", 0.0),
        gamma=lw.get("gamma", 0.0),
    )
    opt = _optimizer(pre, tr)
    epochs = int(tr.get("epochs", 5))
    max_steps = tr.get("max_steps", None)
    qp_list, qp_to_quality = _training_codec_setup(tr, codec)

    total_steps = len(train_loader) * epochs
    if max_steps:
        total_steps = min(total_steps, int(max_steps))
    use_cosine = bool(tr.get("cosine", True))
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(total_steps, 1))
             if use_cosine else None)
    patience = int(tr.get("patience", 0))
    min_delta = float(tr.get("min_delta", 1e-4))
    val_max_batches = tr.get("val_max_batches", 20)
    ckpt_path = _ckpt_path(cfg)                    # best (what evaluate loads)
    last_path = ckpt_path.with_name("preprocessor_last.pth")

    start_epoch, step, best_val, no_improve = 0, 0, float("inf"), 0
    if bool(tr.get("resume", False)) and last_path.exists():
        st = torch.load(last_path, map_location=device)
        pre.load_state_dict(st["model"])
        if st.get("opt"):
            opt.load_state_dict(st["opt"])
        if sched is not None and st.get("sched"):
            sched.load_state_dict(st["sched"])
        start_epoch = st.get("epoch", 0)
        step, best_val = st.get("global_step", 0), st.get("best_val", float("inf"))
        no_improve = st.get("no_improve", 0)
        print(f"[train] resumed {last_path} @ epoch {start_epoch}, step {step}")
    if start_epoch >= epochs:
        print("[train] resume: already at target epochs; nothing to do")
        return str(ckpt_path if ckpt_path.exists() else last_path)

    def _save(path, ep):
        torch.save({"model": pre.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict() if sched is not None else None,
                    "cfg": cfg, "epoch": ep, "global_step": step,
                    "best_val": best_val, "no_improve": no_improve}, path)

    pre.train()
    print(f"[train] {tag} | {n_train} clips | {len(train_loader)} steps/epoch | "
          f"val={'yes' if val_loader else 'none'} | cosine={use_cosine} | "
          f"patience={patience or 'off'} | device={device}", flush=True)

    stop, epoch = False, start_epoch
    for epoch in range(start_epoch, epochs):
        pbar = tqdm(train_loader, desc=f"epoch {epoch + 1}/{epochs}")
        for batch in pbar:
            clips, target = prep_batch(batch)
            qp = random.choice(qp_list)
            q = qp_to_quality[qp]
            cond = _rate_cond(_qp_norm(qp, cfg), clips.shape[0], clips.device, clips.dtype)
            x_pre = pre(clips, cond)
            x_hat, bpp = codec(x_pre, q)
            parts = preprocessing_loss(analyzer, clips, x_hat, bpp, target, weights, x_pre=x_pre)
            opt.zero_grad(set_to_none=True)
            parts["loss"].backward()
            opt.step()
            if sched is not None:
                sched.step()
            step += 1
            pbar.set_postfix(loss=f"{parts['loss'].item():.3f}",
                             task=f"{parts['loss_task'].item():.3f}",
                             dist=f"{parts['loss_dist'].item():.3f}",
                             bpp=f"{parts['loss_rate'].item():.3f}",
                             tmp=f"{parts['loss_temp'].item():.4f}",
                             dlt=f"{parts['loss_delta'].item():.4f}",
                             tv=f"{parts['loss_tv'].item():.4f}",
                             lr=f"{opt.param_groups[0]['lr']:.1e}", qp=qp)
            if max_steps and step >= max_steps:
                stop = True
                break

        _save(last_path, epoch + 1)
        if val_loader is None:
            _save(ckpt_path, epoch + 1)            # no val -> last is best
        else:
            vl = _val_loss(pre, codec, analyzer, val_loader, weights, qp_list,
                           qp_to_quality, cfg, prep_batch, val_max_batches)
            improved, best_val, no_improve, stop_es = _earlystop_update(
                vl, best_val, min_delta, no_improve, patience)
            print(f"[train] epoch {epoch + 1} val_loss={vl:.4f} best={best_val:.4f} "
                  f"{'*improved' if improved else f'(no_improve={no_improve})'}", flush=True)
            if improved:
                _save(ckpt_path, epoch + 1)
            elif stop_es:
                print(f"[train] early stop: no val gain in {patience} epochs", flush=True)
                stop = True
        if stop:
            break

    if val_loader is None or not ckpt_path.exists():
        _save(ckpt_path, epoch + 1)
    print(f"[train] best checkpoint -> {ckpt_path}")
    return str(ckpt_path)


def _train_classification(cfg: dict) -> str:
    device = _device(cfg)
    pre, codec, analyzer = _build_models(cfg, device)
    tr, d = cfg["train"], cfg["data"]
    common = dict(
        index_json=d["index"],
        num_frames=d.get("num_frames", 16),
        frame_size=d.get("frame_size", 128),
        temporal_stride=d.get("temporal_stride", 2),
    )
    train_ds = VideoClipDataset(split="train", train=True, **common)
    val_ds = VideoClipDataset(split="val", train=False, **common)
    train_loader = DataLoader(
        train_ds, batch_size=tr.get("batch_size", 4), shuffle=True,
        num_workers=tr.get("num_workers", 2), collate_fn=collate_clips,
        drop_last=True, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=tr.get("batch_size", 4), shuffle=False,
        num_workers=tr.get("num_workers", 2), collate_fn=collate_clips,
    ) if len(val_ds) else None

    def prep(batch):
        clips, labels = batch
        clips = clips.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        return clips, labels

    return _fit(cfg, pre, codec, analyzer, train_loader, val_loader, prep,
                tag="action_recognition", n_train=len(train_ds))


def _train_tracking(cfg: dict) -> str:
    device = _device(cfg)
    pre, codec, analyzer = _build_models(cfg, device)
    tr, d = cfg["train"], cfg["data"]
    common = dict(
        index_json=d["index"],
        num_frames=d.get("num_frames", 8),
        frame_size=d.get("frame_size", 256),
        temporal_stride=d.get("temporal_stride", 3),
    )
    train_ds = GOT10kClipDataset(split="train", train=True, **common)
    val_ds = GOT10kClipDataset(split="val", train=False, **common)
    train_loader = DataLoader(
        train_ds, batch_size=tr.get("batch_size", 2), shuffle=True,
        num_workers=tr.get("num_workers", 2), collate_fn=collate_got10k,
        drop_last=True, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=tr.get("batch_size", 2), shuffle=False,
        num_workers=tr.get("num_workers", 2), collate_fn=collate_got10k,
    ) if len(val_ds) else None

    def prep(batch):
        clips, boxes = batch
        clips = clips.to(device, non_blocking=True)
        boxes = boxes.to(device, non_blocking=True)
        return clips, {"boxes": boxes}

    return _fit(cfg, pre, codec, analyzer, train_loader, val_loader, prep,
                tag="tracking", n_train=len(train_ds))


# --------------------------------------------------------------------------
# evaluation (dispatch)
# --------------------------------------------------------------------------
def evaluate(cfg: dict, ckpt_path: str, out_dir: str | None = None) -> dict:
    device = _device(cfg)
    pre, codec, analyzer = _build_models(cfg, device)
    state = torch.load(ckpt_path, map_location=device)
    pre.load_state_dict(state["model"] if "model" in state else state)
    pre.eval()

    out_dir = Path(out_dir or (Path(cfg.get("out_dir", "outputs")) / "eval"))
    out_dir.mkdir(parents=True, exist_ok=True)

    if cfg["task"]["name"] == "tracking":
        return _evaluate_tracking(cfg, pre, codec, analyzer, out_dir)
    return _evaluate_classification(cfg, pre, codec, analyzer, out_dir)


# -- accumulation utilities ------------------------------------------------
def _accumulate(store: dict, method: str, key, bpp: float, score_sum: float, n: int):
    slot = store.setdefault(method, {}).setdefault(
        key, {"bpp_sum": 0.0, "score_sum": 0.0, "n": 0}
    )
    slot["bpp_sum"] += bpp * n
    slot["score_sum"] += score_sum
    slot["n"] += n


def _curve(store_method: dict) -> Dict[str, List[float]]:
    keys = sorted(store_method, key=lambda k: store_method[k]["bpp_sum"] / max(store_method[k]["n"], 1))
    bpp = [store_method[k]["bpp_sum"] / store_method[k]["n"] for k in keys]
    acc = [store_method[k]["score_sum"] / store_method[k]["n"] for k in keys]
    return {"keys": list(keys), "bpp": bpp, "accuracy": acc}


@torch.no_grad()
def _task_metric(analyzer, x_hat, labels):
    """Classification top-1: returns (num_correct, batch_size)."""
    logits = analyzer.predict(x_hat)
    correct = (logits.argmax(dim=1) == labels).sum().item()
    return float(correct), x_hat.shape[0]


# -- classification eval ---------------------------------------------------
def _evaluate_classification(cfg, pre, codec, analyzer, out_dir) -> dict:
    device = next(pre.parameters()).device
    ev = cfg.get("eval", {})
    ds = VideoClipDataset(
        index_json=cfg["data"]["index"],
        split="val",
        num_frames=cfg["data"].get("num_frames", 16),
        frame_size=cfg["data"].get("frame_size", 128),
        temporal_stride=cfg["data"].get("temporal_stride", 2),
        train=False,
    )
    loader = DataLoader(
        ds, batch_size=ev.get("batch_size", 4), shuffle=False,
        num_workers=ev.get("num_workers", 2), collate_fn=collate_clips,
    )
    qps = ev.get("qp_list", [30, 35, 40, 45, 50])
    have_ffmpeg = ffmpeg_available()
    if not have_ffmpeg:
        print("[eval] WARNING: ffmpeg not found -> skipping H.264/H.265 anchors")

    store: dict = {}
    qmid = codec.qualities[len(codec.qualities) // 2]
    qconds = _quality_conds(cfg, codec)  # per-quality FiLM cond matching training
    saved_vis = False
    for clips, labels in tqdm(loader, desc="eval"):
        clips = clips.to(device)
        labels = labels.to(device)
        # Rate-conditioned: the preprocessor output depends on the operating
        # point, so it is recomputed per rate point (cannot preprocess once).
        for q in codec.qualities:
            cond = _rate_cond(qconds[q], clips.shape[0], clips.device, clips.dtype)
            with torch.no_grad():
                x_pre = pre(clips, cond)
            xh, bpp = codec.compress_decompress(x_pre, q)
            s, n = _task_metric(analyzer, xh, labels)
            _accumulate(store, "prep+compressai", q, bpp, s, n)
            if not saved_vis and q == qmid:
                _save_qualitative(out_dir / "qualitative.png", clips, x_pre, xh)
                saved_vis = True
            xh0, bpp0 = codec.compress_decompress(clips, q)
            s0, n0 = _task_metric(analyzer, xh0, labels)
            _accumulate(store, "compressai", q, bpp0, s0, n0)
        if have_ffmpeg:
            for name in ("h264", "h265"):
                for qp in qps:
                    cond = _rate_cond(_qp_norm(qp, cfg), clips.shape[0], clips.device, clips.dtype)
                    with torch.no_grad():
                        x_pre = pre(clips, cond)
                    sc = StandardCodec(codec=name, qp=qp, preset=ev.get("preset", "medium"))
                    xh, bpp = sc.compress_decompress(clips)
                    s, n = _task_metric(analyzer, xh.to(device), labels)
                    _accumulate(store, name, qp, bpp, s, n)
                    xhp, bppp = sc.compress_decompress(x_pre)   # prep + real codec
                    sp, np_ = _task_metric(analyzer, xhp.to(device), labels)
                    _accumulate(store, f"prep+{name}", qp, bppp, sp, np_)

    curves = {m: _curve(store[m]) for m in store}
    return _finalize(curves, out_dir, task="action_recognition", metric="top1",
                     n_eval=len(ds))


# -- tracking eval ---------------------------------------------------------
def _codec_chunked(pre, codec, clip, q, chunk, use_pre, cond=None):
    """Run preprocessor+CompressAI over a long clip in T-chunks (bounds memory)."""
    b, c, t, h, w = clip.shape
    outs, bpp_sum = [], 0.0
    for s in range(0, t, chunk):
        sub = clip[:, :, s : s + chunk]
        with torch.no_grad():
            xp = pre(sub, cond) if use_pre else sub
            xh, bpp = codec.compress_decompress(xp, q)
        outs.append(xh)
        bpp_sum += bpp * sub.shape[2]
    return torch.cat(outs, dim=2), bpp_sum / max(t, 1)


def _pre_chunked(pre, clip, chunk, cond=None):
    """Preprocess a long clip in T-chunks, return the full [B,C,T,H,W] (bounds memory)."""
    t = clip.shape[2]
    outs = []
    for s in range(0, t, chunk):
        with torch.no_grad():
            outs.append(pre(clip[:, :, s : s + chunk], cond))
    return torch.cat(outs, dim=2)


def _acc_track(store, method, key, bpp, pred, gt, valid):
    m = sequence_metrics(pred, gt, valid)
    slot = store.setdefault(method, {}).setdefault(
        key, {"bpp_sum": 0.0, "frames": 0, "seqs": []}
    )
    T = len(gt)
    slot["bpp_sum"] += bpp * T
    slot["frames"] += T
    slot["seqs"].append(m)


def _curve_track(store_method: dict) -> Dict[str, List[float]]:
    keys = sorted(store_method, key=lambda k: store_method[k]["bpp_sum"] / max(store_method[k]["frames"], 1))
    bpp = [store_method[k]["bpp_sum"] / max(store_method[k]["frames"], 1) for k in keys]
    agg = [aggregate_metrics(store_method[k]["seqs"]) for k in keys]
    return {
        "keys": list(keys),
        "bpp": bpp,
        "accuracy": [a["auc"] for a in agg],
        "ao": [a["ao"] for a in agg],
        "sr50": [a["sr50"] for a in agg],
        "sr75": [a["sr75"] for a in agg],
    }


def _resolve_tracker(cfg, analyzer):
    """Pick the eval tracker. Default = the trained-against SiamFC analyzer.

    ``task.tracker`` may request the paper's exact trackers via pytracking, e.g.
    ``pytracking:dimp:dimp50`` / ``pytracking:atom`` / ``pytracking:kys`` /
    ``pytracking:prdimp:prdimp50``. Those are eval-only (the preprocessor is
    always trained with SiamFC's differentiable loss).
    """
    spec = cfg["task"].get("tracker", "siamfc")
    if spec in (None, "siamfc", "default"):
        return analyzer.track
    parts = str(spec).split(":")
    if parts[0] == "pytracking":
        from .tasks.pytracking_adapter import build_tracker

        name = parts[1] if len(parts) > 1 else "dimp"
        param = parts[2] if len(parts) > 2 else None
        trk = build_tracker(name, param)
        print(f"[eval] using pytracking tracker: {name}/{trk.parameter}")
        return trk.track_sequence
    raise ValueError(f"unknown task.tracker '{spec}'")


def _evaluate_tracking(cfg, pre, codec, analyzer, out_dir) -> dict:
    device = next(pre.parameters()).device
    ev = cfg.get("eval", {})
    fs = cfg["data"].get("frame_size", 256)
    max_frames = ev.get("max_frames", 48)
    max_seqs = ev.get("max_seqs", 30)
    chunk = ev.get("codec_chunk", 16)
    qps = ev.get("qp_list", [30, 35, 40, 45, 50])
    track = _resolve_tracker(cfg, analyzer)
    have_ffmpeg = ffmpeg_available()
    if not have_ffmpeg:
        print("[eval] WARNING: ffmpeg not found -> skipping H.264/H.265 anchors")

    seqs = list(iter_sequences(cfg["data"]["index"], "val", fs, max_frames, max_seqs))
    store: dict = {}
    qconds = _quality_conds(cfg, codec)  # per-quality FiLM cond matching training
    for name, clip, gt, valid in tqdm(seqs, desc="eval-track"):
        clip = clip.to(device)
        init = gt[0]
        # Rate-conditioned: preprocess per operating point (output depends on it).
        for q in codec.qualities:
            cond = _rate_cond(qconds[q], clip.shape[0], clip.device, clip.dtype)
            xh, bpp = _codec_chunked(pre, codec, clip, q, chunk, use_pre=True, cond=cond)
            _acc_track(store, "prep+compressai", q, bpp, track(xh, init), gt, valid)
            xh0, bpp0 = _codec_chunked(pre, codec, clip, q, chunk, use_pre=False)
            _acc_track(store, "compressai", q, bpp0, track(xh0, init), gt, valid)
        if have_ffmpeg:
            for cname in ("h264", "h265"):
                for qp in qps:
                    cond = _rate_cond(_qp_norm(qp, cfg), clip.shape[0], clip.device, clip.dtype)
                    clip_pre = _pre_chunked(pre, clip, chunk, cond=cond)  # prep at this QP
                    sc = StandardCodec(codec=cname, qp=qp, preset=ev.get("preset", "medium"))
                    xh, bpp = sc.compress_decompress(clip)
                    _acc_track(store, cname, qp, bpp, track(xh.to(device), init), gt, valid)
                    xhp, bppp = sc.compress_decompress(clip_pre)   # prep + real codec
                    _acc_track(store, f"prep+{cname}", qp, bppp, track(xhp.to(device), init), gt, valid)

    curves = {m: _curve_track(store[m]) for m in store}
    return _finalize(curves, out_dir, task="tracking", metric="auc", n_eval=len(seqs))


# --------------------------------------------------------------------------
# finalize: BD-Rate, save, plot, print (shared by both tasks)
# --------------------------------------------------------------------------
def _bd_pair(curves: dict, test_name: str, anchor_name: str):
    """BD-Rate / BD-accuracy of test curve vs anchor curve (None if either absent)."""
    if test_name not in curves or anchor_name not in curves:
        return None
    a, t = curves[anchor_name], curves[test_name]
    return {
        "bd_rate_pct": bd_rate(a["bpp"], a["accuracy"], t["bpp"], t["accuracy"]),
        "bd_accuracy": bd_metric(a["bpp"], a["accuracy"], t["bpp"], t["accuracy"]),
    }


def _finalize(curves: dict, out_dir: Path, task: str, metric: str, n_eval: int) -> dict:
    # legacy view: prep+compressai vs every anchor (cross-codec, for reference)
    bd = {}
    for anchor in ("compressai", "h264", "h265"):
        e = _bd_pair(curves, "prep+compressai", anchor)
        if e is not None:
            bd[anchor] = e
    # apples-to-apples: preprocessor gain on the SAME codec (the real claim)
    prep_gain = {}
    for test_name, anchor_name in (
        ("prep+compressai", "compressai"),
        ("prep+h264", "h264"),
        ("prep+h265", "h265"),
    ):
        e = _bd_pair(curves, test_name, anchor_name)
        if e is not None:
            prep_gain[f"{test_name} vs {anchor_name}"] = e
    results = {
        "task": task,
        "metric": metric,
        "curves": curves,
        "bd_vs_anchor": bd,
        "bd_prep_gain": prep_gain,
        "n_eval": n_eval,
    }
    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    _write_csv(out_dir / "curves.csv", curves)
    _plot(out_dir / "rate_accuracy.png", curves, metric)
    _print_summary(results)
    print(f"[eval] wrote {out_dir/'results.json'}, curves.csv, rate_accuracy.png")
    return results


def _write_csv(path: Path, curves: dict) -> None:
    import csv

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "rate_point", "bpp", "accuracy"])
        for method, c in curves.items():
            for k, bpp, acc in zip(c["keys"], c["bpp"], c["accuracy"]):
                w.writerow([method, k, f"{bpp:.6f}", f"{acc:.6f}"])


def _plot(path: Path, curves: dict, metric: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"[eval] plot skipped ({e})")
        return

    ylabel = {"top1": "top-1 accuracy", "auc": "tracking success AUC"}.get(metric, metric)
    plt.figure(figsize=(7, 5))
    styles = {
        "prep+compressai": dict(marker="o", lw=2),
        "compressai": dict(marker="s", ls="--"),
        "h264": dict(marker="^", ls=":"),
        "h265": dict(marker="v", ls="-."),
    }
    for method, c in curves.items():
        plt.plot(c["bpp"], c["accuracy"], label=method, **styles.get(method, {}))
    plt.xlabel("bits per pixel (real coded)")
    plt.ylabel(ylabel)
    plt.title("Rate vs machine-vision accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


def _save_qualitative(path, source, x_pre, x_hat, n_frames: int = 4) -> None:
    """Grid PNG: rows = source / preprocessed / reconstructed, cols = sampled
    frames of batch item 0. Lets you eye what the preprocessor actually edits."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"[eval] qualitative viz skipped ({e})")
        return
    t = source.shape[2]
    idx = torch.linspace(0, t - 1, min(n_frames, t)).round().long().tolist()
    rows = [("source", source), ("preprocessed", x_pre), ("recon", x_hat)]
    fig, axes = plt.subplots(
        len(rows), len(idx), figsize=(3 * len(idx), 3 * len(rows)), squeeze=False
    )
    for r, (name, ten) in enumerate(rows):
        for c, fi in enumerate(idx):
            img = ten[0, :, fi].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
            ax = axes[r][c]
            ax.imshow(img)
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(name, fontsize=11)
            if r == 0:
                ax.set_title(f"frame {fi}", fontsize=9)
    fig.suptitle("Qualitative: source vs preprocessed vs reconstructed")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"[eval] wrote {path}")


def _print_summary(results: dict) -> None:
    print("\n=== rate-accuracy summary ===")
    for method, c in results["curves"].items():
        pts = ", ".join(f"({b:.3f}bpp, {a:.3f})" for b, a in zip(c["bpp"], c["accuracy"]))
        print(f"  {method:16s}: {pts}")
    if results["bd_vs_anchor"]:
        print("\n=== BD-Rate of prep+compressai vs anchors (negative = savings) ===")
        for anchor, v in results["bd_vs_anchor"].items():
            print(
                f"  vs {anchor:12s}: BD-Rate {v['bd_rate_pct']:+.2f}%  |  "
                f"BD-Accuracy {v['bd_accuracy']:+.4f}"
            )
    if results.get("bd_prep_gain"):
        print("\n=== preprocessor gain, SAME codec (the real claim; negative = savings) ===")
        for label, v in results["bd_prep_gain"].items():
            print(
                f"  {label:28s}: BD-Rate {v['bd_rate_pct']:+.2f}%  |  "
                f"BD-Accuracy {v['bd_accuracy']:+.4f}"
            )
