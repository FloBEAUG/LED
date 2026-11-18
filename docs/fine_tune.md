# Fine-Tuning LED on Your Own Camera (Low/High Noise Pairs)

> This guide walks you through fine-tuning the LED denoiser on pairs of low-noise and high-noise RAW images captured with your own camera. The output model produces DNG files via [`scripts/image_process.py`](scripts/image_process.py).

---

## 1. Overview

LED is trained with a **calibration-free** strategy: given a noisy-clean RAW pair at a known digital gain (ratio), the network learns to map the noisy Bayer frame to the clean one. Fine-tuning adapts a pretrained LED checkpoint to your specific camera sensor and noise characteristics.

The fine-tuning pipeline has **two phases**:

| Phase | Purpose | Iterations (default) | Learning rate |
|-------|---------|----------------------|---------------|
| **Align** | Align pretrained weights to your data distribution | `align_iter` (1000) | `1e-4` |
| **OOMN** (Out-of-Manifold Noise) | Fine-tune the noise-specific branch | `oomn_iter` (500) | `1e-5` |

Total default iterations: **1500**. Total training time on a single GPU is typically a few minutes.

---

## 2. Data Requirements

You need **paired RAW images** of the same static scene:

- **Low-noise (clean) image**: long exposure, low ISO (e.g. ISO 100–400, several seconds).
- **High-noise (noisy) image**: short exposure, high ISO (e.g. ISO 3200–12800, fraction of a second).

Both images must have the **same exposure value** (ISO × shutter time) so they represent the same scene brightness at different gains.

### Required ratio

For each target digital gain (ratio), collect **two pairs** at different ISOs so the network can learn the linear relationship between ISO and noise variance:

| Pair | Clean (long) | Noisy (short) | Exposure level |
|------|-------------|---------------|----------------|
| Pair A | ISO 100, 30 s | ISO 6400, 1/10 s | 3000 s vs 640 s |
| Pair B | ISO 320, 2 s | ISO 6400, 1/10 s | 640 s vs 640 s |

The ratio is `clean_exposure / noisy_exposure` (e.g. `3000 / 640 ≈ 4.69`).

### Supported formats

- Canon `.CR3`, `.CR2`
- Sony `.ARW`
- Nikon `.NEF`
- Any format readable by **rawpy / LibRaw**

Files are read as-is (no conversion needed). The script reads them directly with `rawpy`.

---

## 3. Folder Structure

```
your_project/
├── data/
│   └── my_camera_pairs/            # dataroot (contains all RAW files)
│       ├── IMG_0001.CR3            # clean (low noise)
│       ├── IMG_0002.CR3            # noisy (high noise)
│       ├── IMG_0003.CR3            # clean
│       └── IMG_0004.CR3            # noisy
├── pairs.txt                       # data pair list
└── pretrained/
    └── LED_Pretrain_....pth        # starting checkpoint
```

---

## 4. Write the Data Pair List

Create a text file where each line lists one **noisy clean** pair (space-separated, relative to `dataroot` or absolute paths):

```txt
IMG_0002.CR3 IMG_0001.CR3
IMG_0004.CR3 IMG_0003.CR3
```

> **Order matters**: the first file is the **noisy (LQ)** input, the second is the **clean (GT)** target.

The dataset (`PairedRAWDataset`) parses the shutter speed from each filename (pattern like `_10s.` or `_0.1s.`) to compute the ratio automatically. Make sure your filenames contain the exposure time.

If your filenames don't contain exposure times, edit `datasets/txtfiles/` manually or preprocess filenames to include them.

---

## 5. Choose a Pretrained Model

Download or locate a pretrained LED checkpoint. The repo ships one:

```
pretrained/network_g/LED_Pretrain_None_None_CVPR20_Setting_Ratio1-200.pth
```

Common choices:

| Checkpoint | Use case |
|------------|----------|
| `Ratio1-200` | General purpose, wide gain range |
| `Ratio100-300` | Extreme low light (high ratios) |

If you don't have a checkpoint, download one from the links in [`docs/pretrained-models.md`](docs/pretrained-models.md).

---

## 6. Fine-Tuning Command

Use the helper script [`scripts/cutomized_denoiser.py`](scripts/cutomized_denoiser.py):

```bash
conda activate LED-ICCV23

python scripts/cutomized_denoiser.py \
    -t my_camera_finetune \
    -p pretrained/network_g/LED_Pretrain_None_None_CVPR20_Setting_Ratio1-200.pth \
    --dataroot /path/to/your/data/my_camera_pairs \
    --data_pair_list /path/to/your/data/pairs.txt
```

### What this does

It launches `led/train.py` with an internal base config (`options/base/demo/base.yaml`) and overrides:

| Override | Value |
|----------|-------|
| `name` | your `-t` tag |
| `path.pretrain_network_g` | your `-p` checkpoint |
| `datasets.train.dataroot` | your `--dataroot` |
| `datasets.train.data_pair_list` | your `--data_pair_list` |

### Optional: force YAML overrides

Append `--force_yml` key=value pairs to tweak training:

```bash
python scripts/cutomized_denoiser.py -t my_camera \
    -p pretrained/network_g/LED_Pretrain_....pth \
    --dataroot data/my_camera_pairs \
    --data_pair_list data/pairs.txt \
    --force_yml train:total_iter=3000 \
    --force_yml train:align_iter=2000 \
    --force_yml train:oomn_iter=1000 \
    --force_yml datasets:train:crop_size=512 \
    --force_yml train:optim_g:lr=5e-5
```

### Checkpoints

Outputs go to:

```
experiments/my_camera_finetune/
├── models/
│   ├── net_g_iter_1000.pth
│   ├── net_g_iter_1500.pth
│   └── ...
└── training_states/
    └── ...
```

The latest model is saved as `net_g_iter_-1.pth` (or the final iteration).

---

## 7. Configuration Reference

### Base fine-tune option (`options/base/finetune/CVPR20_ELD.yaml`)

```yaml
datasets:
  train:
    num_worker_per_gpu: 8
    batch_size_per_gpu: 1
    dataset_enlarge_ratio: 99999
    prefetch_mode: ~

train:
  generalize_first: true
  pixel_opt:
    type: L1Loss
    loss_weight: 1.0
    reduction: mean
  total_iter: 1500
  align_iter: 1000
  oomn_iter: 500
  warmup_iter: -1
  align_opt:
    optim_g: { type: Adam, lr: 1e-4, weight_decay: 0, betas: [0.9, 0.999] }
    scheduler: { type: HandieLR, milestones: [999999], lrs: [0] }
  oomn_opt:
    optim_g: { type: Adam, lr: 1e-5, weight_decay: 0, betas: [0.9, 0.999] }
    scheduler: { type: HandieLR, milestones: [999999], lrs: [0] }
```

### Dataset option (`options/base/dataset/finetune/SID_SonyA7S2_fewshot_raw_pair6.yaml`)

| Key | Meaning | Typical value |
|-----|---------|---------------|
| `type` | `PairedRAWDataset` | fixed |
| `dataroot` | folder with RAW pairs | set by `--dataroot` |
| `postfix` | file extension filter | `~` (RAW directly) |
| `which_meta` | use GT or LQ metadata | `gt` |
| `data_pair_list` | pairs txt | set by `--data_pair_list` |
| `zero_clip` | clamp negatives | `false` |
| `use_hflip` / `use_rot` | augmentation | `true` |
| `crop_size` | patch size for training | `1024` (or `~` for full image) |
| `load_in_mem` | load all into RAM | `true` (fast) or `false` |

---

## 8. Multi-GPU Distributed Training

For multiple GPUs, use [`scripts/dist_train.sh`](scripts/dist_train.sh) directly instead of the helper script:

```bash
bash scripts/dist_train.sh 2 options/LED/finetune/SID_SonyA7S2_CVPR20_ELD_Setting.yaml
```

Then edit the YAML to point `path.pretrain_network_g`, `datasets.train.dataroot`, and `datasets.train.data_pair_list` to your data.

---

## 9. Test Your Fine-Tuned Model (DNG Output)

After training, run inference with [`scripts/image_process.py`](scripts/image_process.py):

```bash
conda activate LED-ICCV23

python scripts/image_process.py \
    -p experiments/my_camera_finetune/models/net_g_iter_1500.pth \
    --data_path /path/to/your/test_raws/ \
    --save_path inference/my_camera_dng/ \
    --network_options options/base/network_g/repnr_unet.yaml \
    --bps 16 \
    --led \
    --ratio 1.0
```

| Flag | Meaning |
|------|---------|
| `-p` | path to your fine-tuned checkpoint |
| `--data_path` | folder of RAW files to denoise |
| `--save_path` | where DNGs are written |
| `--network_options` | architecture YAML (use `repnr_unet.yaml` for LED finetune checkpoints) |
| `--bps` | DNG bit depth (16) |
| `--led` | tells the loader to use `params_deploy` key (repnr deployed weights) |
| `--ratio` | additional digital gain (1.0 = none) |

Output: `RAW_denoised.dng` per input RAW, with camera metadata (black level, white level, color matrix, CFA pattern) copied from the source file via `exiftool`.

---

## 10. End-to-End Example

```bash
# 1. Activate env
conda activate LED-ICCV23

# 2. Organize data
mkdir -p data/eosr7_pairs
cp ~/DCIM/IMG_*.CR3 data/eosr7_pairs/

# 3. Create pairs.txt
cat > data/eosr7_pairs.txt << 'EOF'
IMG_0002.CR3 IMG_0001.CR3
IMG_0004.CR3 IMG_0003.CR3
EOF

# 4. Fine-tune
python scripts/cutomized_denoiser.py \
    -t eosr7_iso6400 \
    -p pretrained/network_g/LED_Pretrain_None_None_CVPR20_Setting_Ratio1-200.pth \
    --dataroot data/eosr7_pairs \
    --data_pair_list data/eosr7_pairs.txt

# 5. Test (produces DNG)
python scripts/image_process.py \
    -p experiments/eosr7_iso6400/models/net_g_iter_1500.pth \
    --data_path data/eosr7_pairs/ \
    --save_path inference/eosr7_dng/ \
    --network_options options/base/network_g/repnr_unet.yaml \
    --bps 16 \
    --led \
    --ratio 1.0

# 6. Open inference/eosr7_dng/*.dng in your DNG-compatible software
```

---

## 11. Troubleshooting

| Problem | Cause / Fix |
|---------|-------------|
| `Missing key(s) in state_dict` | You loaded a RepNR (deploy) checkpoint into a plain UNet, or vice-versa. Use `--led` with repnr checkpoints; remove `--led` for plain UNet checkpoints. |
| `CUDA out of memory` | Reduce `datasets:train:crop_size` (e.g. 512) or set `batch_size_per_gpu: 1` (already default). For inference, enable the tiled path in `image_process.py` or crop large RAVs. |
| `Unsupported file format` | `rawpy` cannot open the file. Use a supported RAW (CR3, ARW, NEF) or convert to DNG first with Adobe DNG Converter / dnglab. |
| Ratio computed as `None` | Filenames don't contain shutter time (e.g. `_10s.`). Rename files to include exposure or compute ratio manually and patch `PairedRAWDataset`. |
| Purple tint in DNG | Black level / white level mismatch between data and DNG tags. The current `save_as_dng` writes the source file's black/white levels; ensure the denoised data is scaled with the same `bl`/`wl` before writing. |
| Training loss doesn't decrease | Check pair order (noisy first, clean second). Verify `dataroot` and `data_pair_list` paths are correct. |

---

## 12. Repository Pointers

| File | Role |
|------|------|
| `scripts/cutomized_denoiser.py` | Fine-tuning launcher |
| `led/train.py` | Training pipeline |
| `led/models/raw_image_denoising_model.py` | `LEDFinetuneModel` (align + OOMN phases) |
| `led/data/paired_raw_dataset.py` | RAW pair loader |
| `led/utils/img_util.py` | `save_as_dng` (DNG writer) |
| `scripts/image_process.py` | Inference → DNG |
| `options/base/finetune/CVPR20_ELD.yaml` | Training hyperparameters |
| `options/base/dataset/finetune/*.yaml` | Dataset configs |
| `docs/demo.md` | Quick demo (inference) |
| `docs/develop.md` | Architecture / RepNR details |

---

*Generated for the LED-ICCV23 fork (conda env `LED-ICCV23`, Python 3.8, torch 1.13.1).*
