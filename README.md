# Adaptive Video Preprocessing for VCM — Upgrade 2 (U-Net + FiLM/SFT, loss task-distillation)

Tiền xử lý (preprocessing) học được, **sửa pixel trước một codec chuẩn (x264/x265) không đổi**,
để máy (recognition / tracking) "nhìn" tốt hơn ở cùng bitrate — bài toán **VCM (Video Coding for
Machines)** ở nhánh preprocessing. Chỉ preprocessor được train; codec và analyzer **đông cứng**.

Repo này **viết lại từ đầu** so với [`upgrade1`](https://github.com/wagur1/preprocessing_upgrade_1)
(fork của Zhao et al., arXiv:2512.15331). Upgrade1 đo được BD-Rate **dương** (tốn bit hơn) kể cả khi
đã ép `λ` mạnh → NO-GO. Chẩn đoán: **lỗi nằm ở hàm loss, không phải codec.**

> **Vì sao đổi hướng.** Yang et al. (*Task-Switchable Pre-Processor…*, IEEE **TCSVT 2024**) chứng minh
> một preprocessor train **xuyên qua một learned-codec** (CompressAI) **chuyển thẳng** sang codec chuẩn
> (BPG≈HEVC-intra) mà **không cần train lại**, và còn tiết kiệm **nhiều hơn** (−32.6% BD-BR detection
> trên BPG vs −24.1% trên learned-codec). Codec **không** phải vấn đề — nó là "module độc lập, tách rời
> codec". Cái upgrade1 làm sai là **loss**: (1) `L_D = MSE(x̂, source)` ép bám pixel gốc ⇒ *chống lại*
> nén; (2) `λ` quá nhỏ (rate ≈0.2% loss) ⇒ nén không "cắn"; (3) không có feature distillation. Yang et al.
> **bỏ hẳn MSE-to-source**, dùng `λ` lớn, và thêm **distillation đặc trưng** của analyzer.

## Điểm mới so với upgrade1

| Thành phần | upgrade1 (bỏ) | **upgrade2 (mới)** | Lý do |
|---|---|---|---|
| Backbone | 2 nhánh temporal-3D / spatial-2D + attention | **U-Net** (encoder/decoder + skip) | biên tập pixel mạnh hơn, đúng họ backbone của Yang et al. đã đạt BD-BR âm |
| Điều kiện rate | FiLM trong res-block | **FiLM** (rate, per-channel) — giữ | 1 model phủ cả dải bitrate |
| Điều kiện không gian | — | **SFT** theo **motion cue** (temporal diff) | phân bổ chỉnh sửa về vùng chuyển động/task — **novelty video** |
| Distortion | `L_D = MSE(x̂, source)` | **bỏ hẳn** | MSE-to-source chống nén; thủ phạm chính |
| Semantics | — | **L_distill**: MSE đặc trưng analyzer(source) vs analyzer(x̂) | giữ ngữ nghĩa mà codec phá, lợi nhất ở bitrate thấp |
| Rate | `α·λ·bpp` (λ≈1e-3, không cắn) | **`β·bpp`** với β đủ lớn để cắn | mở "cửa nén" |
| Thời gian | — | **L_temp**: khớp *biến thiên liên khung* của x̂ với source | giữ chuyển động / chống flicker, không ghim pixel |
| Train-through-x265 | STE (`_straight_through`) | **bỏ hẳn** | forward=x265 nhưng grad=proxy ⇒ gradient lệch, tệ hơn proxy-only |

Data loaders, metrics (BD-Rate/AUC/top-1), tasks (r3d_18, SiamFC), driver Kaggle, anchor ffmpeg
**giữ nguyên** từ bản dựng lại đã kiểm chứng.

## Kiến trúc mô hình

`VideoPreprocessor` (`src/models/preprocessor.py`) — biên tập **theo từng khung** (2D conv trên
`[B·T, C, H, W]`), tín hiệu thời gian đi vào qua **motion cue** → SFT:

```
x [B,C,T,H,W] ─┬─▶ motion cue = |x_t − x_{t−1}|  (chuẩn hoá/clip) ─┐  (novelty: nơi có chuyển động)
               │                                                   │
               │   c = [qp_norm]  (rate operating point) ──┐       │
               ▼                                            ▼       ▼
        fold → frames [B·T,C,H,W] ─▶  U-Net (enc→bott→dec, skip)  ── mỗi ConvBlock:
                                        conv-act-conv                conv → FiLM(c) → SFT(cue)
                                        ▼                            (γ,β per-channel)  (γ,β per-pixel)
                                     tail (zero-init) ─▶ delta
                                        ▼
              out = clamp( x + res_scale · delta )   ← residual, identity lúc init
```

- **FiLM** (Perez et al. 2018): `(1+γ)·feat + β`, `γ,β` per-channel **toàn cục** dự đoán từ rate `c`.
  Zero-init ⇒ khởi tạo là identity; conditioning "bật dần" khi học. `cond_dim=1` mang `qp_norm`, để
  sẵn chỗ nối `log R_target` cho rate-control sau này.
- **SFT** (Wang et al. 2018): `(1+γ(cue))·feat + β(cue)`, `γ,β` **thay đổi theo không gian** dự đoán từ
  motion cue ⇒ preprocessor chỉnh mạnh vùng chuyển động (thường là đối tượng task), nhả bit ở nền tĩnh.
- **Tail zero-init** ⇒ toàn mạng khởi tạo = identity (train ổn định).

Codec proxy (`src/models/codec.py`) = CompressAI `bmshj2018-factorized` (entropy bottleneck factorized-prior
của Ballé et al. 2017 — đúng mô hình rate paper cite), khả vi khi train, range-coder thật khi eval.

## Pipeline

### Train (chỉ preprocessor học)

```
qp ~ qp_list  ─▶  c=[qp_norm],  q=qp_to_quality[qp]
x ─▶ pre(x, c) ─▶ x_pre ─▶ CompressAI(q) ─▶ x̂, bpp
                                             │
   frozen analyzer.features(x)  ┐            ├─▶ analyzer.accuracy_loss(x̂) = L_task
   frozen analyzer.features(x̂)  ┘─▶ L_distill │
   temporal diff(x) vs diff(x̂) ─▶ L_temp     │
        L = λ_task·L_task + ω·L_distill + β·bpp + τ·L_temp   ─▶ backward → chỉ pre
```

Một step: (1) sample `qp` **trước** khi gọi `pre()` (baseline làm ngược → mù rate); (2) dựng
`c=[qp_norm]`, `qp_norm=(qp−qp_lo)/(qp_hi−qp_lo)∈[0,1]`, `qp_lo/hi = model.qp_ref`; (3) `x_pre=pre(x,c)`;
(4) `x̂,bpp = codec(x_pre, q)` (proxy khả vi); (5) tính `L`, backward. **Không** có real-codec-trong-loop.

### Eval (rate-accuracy, coder thật)

Output preprocessor **phụ thuộc điểm nén** ⇒ chạy lại `pre(x,c)` cho **từng** điểm rate:

```
for mỗi điểm rate (q ∈ codec.qualities  hoặc  qp ∈ eval.qp_list):
    c = [level(điểm rate)]              # quality index → level;  QP → qp_norm
    x_pre = pre(x, c)
    prep+compressai : CompressAI range-coder(x_pre, q)
    prep+h264/h265  : ffmpeg(x_pre, qp)
    anchors         : compressai / h264 / h265 chạy thẳng trên x (không preprocessor)
```

So 6 pipeline → **BD-Rate** (âm = tiết kiệm bit ở cùng accuracy). Hai nhóm:
- `bd_prep_gain` — **cùng codec, chỉ khác có/không preprocessor** (`prep+h265 vs h265`,
  `prep+h264 vs h264`, `prep+compressai vs compressai`): **đây là số đúng luận điểm.**
- `bd_vs_anchor` — `prep+compressai` vs từng anchor (tham khảo, khác codec).

Output: `results.json`, `curves.csv`, `rate_accuracy.png`, `qualitative.png`.

## Hàm loss (`src/losses.py`)

```
L = λ_task · L_task  +  ω · L_distill  +  β · L_R  +  τ · L_temp
```
- `L_task`   — cross-entropy (recognition) / SiamFC balanced-logistic (tracking) của analyzer trên **x̂**.
- `L_distill`— MSE (chuẩn hoá theo scale) giữa đặc trưng trung gian của analyzer trên **source** và **x̂**;
  source detach (chỉ grad qua x̂). Recognition: tap `stem/layer1/layer2` của r3d_18; tracking: đặc trưng
  backbone SiamFC trên full-frame. **Thay cho MSE-to-source** — giữ ngữ nghĩa theo hướng task.
- `L_R`      — bpp ước lượng từ entropy model. β đặt đủ lớn để **nén thực sự cắn** (lỗi chí mạng của baseline).
- `L_temp`   — `MSE( (x̂_t−x̂_{t−1}), (x_t−x_{t−1}) )`: khớp **biến thiên liên khung**, giữ chuyển động &
  chống flicker mà **không** ghim pixel tuyệt đối (nên không chống nén như MSE-to-source).
- `L_delta`  — (tuỳ chọn, mặc định tắt) `mean|x_pre − x|` (L1): **phạt biên độ chỉnh sửa của preprocessor**.
  Đòn bẩy *trực tiếp* khi `β` (bpp, gián tiếp qua entropy model) chưa đủ ngăn model thêm chi tiết → thêm bit.
  **Khác MSE-to-source**: phạt *đầu vào codec* (x_pre) chứ không ghim *x̂*, nên đẩy về chỉnh sửa thưa/ít bit
  thay vì bám pixel gốc. L1 ⇒ chỉnh mạnh vài vùng (đối tượng chuyển động) và nhả nền.

Mặc định `λ_task=1, ω=0.5, β=0.1, τ=0.1, delta=0` (ω theo Yang et al.). **Không có** số hạng MSE-to-source.
Hai đòn bẩy chống "thêm bit": `--res-scale <1` (co biên độ residual toàn cục, cứng) và `--delta >0`
(phạt L1 mềm, để model tự giữ chỗ chỉnh đáng tiền).

## Config mới (`configs/*.yaml`)

```yaml
model:
  base_ch: 32              # bề rộng U-Net ở full-res
  res_scale: 1.0           # out = x + res_scale·delta
  cond_dim: 1              # rộng rate-condition (1 = qp_norm; đầu vào FiLM)
  qp_ref: [20, 51]         # dải QP ↦ mức condition [0,1]
loss:
  lam_task: 1.0
  omega: 0.5               # feature distillation
  beta: 0.1                # rate — chỉnh cái này để nén cắn
  tau: 0.1                 # temporal consistency
  delta: 0.0               # L1 |x_pre−x| — phạt biên độ chỉnh sửa (0=tắt), bật khi β chưa đủ ngăn thêm bit
train:
  qp_list: [22,27,32,37,42]
  qp_to_quality: {22:5, 27:3, 32:2, 37:1, 42:1}   # đơn điệu: QP↑ ↔ quality↓
  cosine: true             # cosine LR decay
  patience: 5              # early-stop: dừng sau N epoch val-loss không giảm (0=tắt)
  min_delta: 1.0e-4
  val_max_batches: 20
  resume: false            # train tiếp từ outputs/checkpoints/preprocessor_last.pth
```
`codec.qualities=[1,2,3,5,8]` phủ mọi giá trị trong `qp_to_quality`.

**Checkpoint & dừng.** `_fit` (dùng chung AR & tracking) mỗi epoch đo **val-loss** (proxy-only, QP giữa
cố định — rẻ, coherent), lưu `preprocessor_last.pth` (cho `--resume`) và ghi `preprocessor.pth` =
**best-val** (ckpt `evaluate.py` nạp). Early-stop lo việc dừng — đặt `epochs` dư rồi để nó tự dừng.
Val rỗng → bỏ qua val, `last = best`.

## Chạy trên Kaggle

Cần **GPU** + **Internet ON** (tải weights CompressAI/torchvision). Image Kaggle có sẵn `ffmpeg`
(libx264/libx265) cho anchor lúc eval.

```python
!git clone https://github.com/wagur1/preprocessing_upgrade_2.git
%cd preprocessing_upgrade_2
!pip install -q compressai          # torch/torchvision đã có trên Kaggle

# Action recognition (Kinetics 5%) — one-shot: prepare index → train → eval
!python kaggle/run_kaggle.py --config configs/action_recognition.yaml \
    --cap-gb 6 --epochs 12          # early-stop tự dừng ở best-val; --resume để train tiếp

# Object tracking (GOT-10k val)
!python kaggle/run_kaggle.py --config configs/tracking.yaml \
    --cap-gb 6 --epochs 12 --max-seqs 120 --max-frames 90
```

Không đủ 1 session ≤12h thì tách: session train (`preprocessor_last.pth` lưu ở Kaggle output) →
session sau `--resume`, hoặc `--skip-train --ckpt outputs/checkpoints/preprocessor.pth` để eval.

### Sweep rate (GO/NO-GO)

Để nén "cắn", tăng **`--beta`** (rate) và có thể giảm **`--lam-task`**; **chỉ nhìn
`prep+compressai vs compressai` (in-domain)**:

```python
for beta, lam in [(0.1, 1.0), (0.5, 1.0), (1.0, 0.5)]:
    !python kaggle/run_kaggle.py --config configs/action_recognition.yaml \
        --cap-gb 3 --epochs 8 --beta {beta} --lam-task {lam} --out-dir outputs/b{beta}_l{lam}
```
Âm ở in-domain → mô hình có cửa nén, đầu tư tiếp (mask học từ analyzer, motion-comp temporal loss).
Vẫn dương dù β mạnh → chỉnh **biên độ edit** trực tiếp: `--res-scale 0.5` (co residual) và/hoặc
`--delta 0.02` (phạt L1 `|x_pre−x|`). Ví dụ ép edit ít-mà-đúng-chỗ:

```python
!python kaggle/run_kaggle.py --config configs/action_recognition.yaml \
    --cap-gb 3 --epochs 12 --beta 1.0 --lam-task 0.5 --delta 0.02 --res-scale 0.5 \
    --out-dir outputs/b1_l0.5_d0.02_r0.5
```
Vẫn dương → phải đổi cấu trúc/điểm bán.

Dataset AR: [`rohanmallick/kinetics-train-5per`](https://www.kaggle.com/datasets/rohanmallick/kinetics-train-5per).
`--cap-gb` nới lên khi cần đường cong/CI chắc hơn.

## Kiểm chứng

```bash
python -m compileall src tests            # cú pháp (đã pass)
python tests/test_preprocessor.py         # FiLM/SFT identity lúc init + phụ thuộc cond/cue; motion cue (cần torch)
python tests/test_losses.py               # distillation=0 khi x̂=source; temporal; không có MSE-to-source (cần torch)
python tests/test_earlystop.py            # điều kiện dừng patience/min_delta (cần torch)
```

## Roadmap

1. ✅ U-Net + FiLM(rate) + SFT(motion) + loss task-distillation + temporal (bản này).
2. Mask **học từ analyzer** (task-importance thật) thay motion cue thủ công.
3. Temporal loss **motion-compensated** (warp bằng optical flow) thay diff thô.
4. Held-out analyzer eval (train SiamFC, eval DiMP/ATOM qua pytracking) — bằng chứng generalization.
5. Same-codec BD-rate + 3 seed/bootstrap CI + runtime/FLOPs/param + VMAF/LPIPS.

## Layout

```
src/
  models/preprocessor.py   U-Net + FiLM(rate) + SFT(motion cue) (trained)
  models/codec.py          CompressAI proxy (khả vi + range coder)
  codecs/standard.py       ffmpeg H.264/H.265 anchor
  tasks/                   r3d_18 (AR) + features(); SiamFC + pytracking adapter (tracking) + features()
  data/                    Kinetics + GOT-10k readers, index builders
  metrics/                 BD-Rate / top-k / tracking AUC
  losses.py                L = λ_task·L_task + ω·L_distill + β·bpp + τ·L_temp (KHÔNG có MSE-to-source)
  engine.py                train/eval + rate-cond helpers + _fit (cosine, val, early-stop, resume)
configs/                   action_recognition.yaml, tracking.yaml
tests/                     test_preprocessor.py, test_losses.py, test_earlystop.py
train.py  evaluate.py      CLI          kaggle/  one-shot driver + notebook
```

## Trích dẫn (citations)

Cơ chế mới dựa trên các công trình sau:

1. **Zhao et al.**, *A Preprocessing Framework for Video Machine Vision under Compression*,
   arXiv:2512.15331 — base paper (khung preprocessing cho VCM; repo này là independent implementation,
   virtual codec → CompressAI).
2. **Yang, Ma, Wang, et al.**, *Task-Switchable Pre-Processor for Image Compression for Multiple Machine
   Vision Tasks*, **IEEE TCSVT 2024** — cơ sở quyết định: U-Net + task-attentive SFT/TSM + **feature
   distillation**, bỏ MSE-to-source, chuyển codec-agnostic sang codec chuẩn. **Hướng thiết kế chính.**
3. **Perez, Strub, de Vries, Dumoulin, Courville**, *FiLM: Visual Reasoning with a General Conditioning
   Layer*, **AAAI 2018** — điều kiện rate per-channel `(1+γ)·f+β`.
4. **Wang, Yu, Dong, et al.**, *Recovering Realistic Texture in Image Super-Resolution by Deep Spatial
   Feature Transform (SFT)*, **CVPR 2018** — điều kiện không gian theo motion cue.
5. **Ronneberger, Fischer, Brox**, *U-Net: Convolutional Networks for Biomedical Image Segmentation*,
   **MICCAI 2015** — backbone encoder/decoder + skip.
6. **Ballé, Laparra, Simoncelli**, *End-to-end Optimized Image Compression*, **ICLR 2017** — factorized
   entropy model = `bmshj2018-factorized` (proxy rate/distortion khả vi).
7. **Bjøntegaard**, *Calculation of Average PSNR Differences Between RD-curves*, VCEG-M33, 2001 — BD-Rate.
8. Dữ liệu & analyzer: **Kay et al.** (Kinetics, 2017); **Huang et al.** (GOT-10k, TPAMI 2019);
   **Tran et al.** (R(2+1)D / r3d_18, CVPR 2018); **Bertinetto et al.** (SiamFC, ECCV 2016).

*Independent implementation cho mục đích nghiên cứu — không phải code của các tác giả gốc.*
