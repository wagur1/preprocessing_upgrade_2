# Running on Kaggle

Two tasks, same output shape: a trained preprocessor plus a rate-accuracy
comparison of **preprocessor + CompressAI** vs the **CompressAI-only** ablation,
**bare H.264** and **bare H.265**, with BD-Rate.

* **Action recognition** (Kinetics-400 5%) — metric: top-1 accuracy.
* **Object tracking** (GOT-10k val) — metric: real success-plot AUC (+AO, SR).

## Prerequisites (Kaggle notebook settings)

1. **Add Input**:
   * action recognition → dataset `rohanmallick/kinetics-train-5per`;
   * tracking → any GOT-10k **val** dataset (sequence folders each containing
     `groundtruth.txt` + numbered `.jpg` frames).
2. **Accelerator** → GPU (T4/P100).
3. **Internet** → On (to clone the repo + download pretrained weights).

ffmpeg (with libx264 and libx265) and PyTorch/torchvision are preinstalled on
Kaggle; only CompressAI needs installing.

## Option A — the notebook

Upload / open `kaggle/preprocessing_kaggle.ipynb` and Run All (it has both an
action-recognition and a tracking cell).

## Option B — paste these cells

```python
!git clone https://github.com/wagur1/preprocessing_upgrade_2.git
%cd preprocessing_upgrade_2
!pip install -q compressai
```

Action recognition:

```python
# build <=3GB balanced index -> train -> evaluate vs CompressAI / H.264 / H.265
!python kaggle/run_kaggle.py \
    --config configs/action_recognition.yaml \
    --cap-gb 3 --epochs 3 --max-steps 300 --batch-size 4
```

Object tracking (GOT-10k):

```python
# build <=3GB GOT-10k index -> train -> evaluate real success AUC + BD-Rate
!python kaggle/run_kaggle.py \
    --config configs/tracking.yaml \
    --cap-gb 3 --epochs 3 --max-steps 300 \
    --max-seqs 30 --max-frames 48
```

```python
import json
from IPython.display import Image, display
display(Image('outputs/eval/rate_accuracy.png'))
res = json.load(open('outputs/eval/results.json'))
print("task:", res['task'], "| metric:", res['metric'])
for a, v in res['bd_vs_anchor'].items():
    print(f"vs {a}: BD-Rate {v['bd_rate_pct']:+.2f}%")
```

## Knobs

| flag | meaning | quick run | fuller run |
|------|---------|-----------|-----------|
| `--config` | `action_recognition.yaml` or `tracking.yaml` | — | — |
| `--cap-gb` | dataset size cap | `3` | `3` |
| `--epochs` | training epochs | `3` | `15` |
| `--max-steps` | cap total optimizer steps (remove for full epochs) | `300` | *(omit)* |
| `--batch-size` | clips per batch | `4` (AR) / `2` (track) | `8` / `4` |
| `--frame-size` | working resolution (multiple of 64) | `128` (AR) / `256` (track) | same |
| `--max-seqs` | tracking: val sequences evaluated | `30` | *(omit = all)* |
| `--max-frames` | tracking: frames per sequence | `48` | *(omit = all)* |
| `--beta` | rate weight (`loss.beta`) — raise so bitrate bites | `0.1` | sweep `0.1→1.0` |
| `--lam-task` | task-accuracy weight (`loss.lam_task`) | `1.0` | `1.0` / `0.5` |
| `--omega` | feature-distillation weight (`loss.omega`) | `0.5` | `0.5` |
| `--tau` | temporal-consistency weight (`loss.tau`) | `0.1` | `0.1` |

## Optional: the paper's exact trackers

The default tracker is the self-contained SiamFC (needs no extra install). To
run the paper's exact GOT-10k trackers (KYS / DiMP / ATOM / PrDiMP) install
[pytracking](https://github.com/visionml/pytracking) via
`bash scripts/install_pytracking.sh` (+ its network weights), then select one at
eval by adding to the tracking config: `task.tracker: pytracking:dimp:dimp50`.
Training always uses SiamFC's differentiable loss regardless.

## Outputs

* `outputs/checkpoints/preprocessor.pth` — trained preprocessor
* `outputs/eval/results.json` — curves + BD-Rate (with `task` and `metric`)
* `outputs/eval/curves.csv` — (method, bpp, accuracy) points
* `outputs/eval/rate_accuracy.png` — the plot

## Notes

* The 3 GB cap is enforced by **indexing**, not copying. For Kinetics,
  `prepare_3gb.py` selects videos round-robin across classes until the
  cumulative size hits the cap (class-balanced). For GOT-10k,
  `prepare_got10k.py` adds whole sequences (kept intact so tracking stays
  meaningful) until the cap, then splits train/val.
* Dataset location is auto-detected under `/kaggle/input`; override with
  `--dataset-dir /kaggle/input/<slug>` if needed. For tracking it looks for a
  `val` dir whose sub-folders contain `groundtruth.txt`.
* The H.264/H.265 anchors need ffmpeg on PATH (present on Kaggle). Locally,
  install ffmpeg with libx264+libx265 or the anchors are skipped with a warning.
