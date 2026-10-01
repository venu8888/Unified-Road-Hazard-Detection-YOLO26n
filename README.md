python -m ultralytics train resume model="checkpoint/last.pt"

> **IMPORTANT**: The command above resumes the existing training checkpoint (`checkpoint/last.pt`) directly and continues from Epoch 33 onward up to 100 epochs. It does **NOT** start training from Epoch 1. Run this command directly from the root of the cloned repository. Do **not** run a fresh `train.py` unless you intentionally want to restart training from scratch.

---

# Unified Road Hazard Detection using YOLO26n

An end-to-end unified road hazard detection model trained on real-world road inspection imagery to detect structural cracks, potholes, and manhole defects in a single high-performance neural network.

---

## 1. Target Hazard Classes (9 Classes)

| Class ID | Class Name | Description | Source |
|---|---|---|---|
| **0** | `longitudinal_crack` | Crack running parallel to roadway direction | N-RDD2024 (D00) |
| **1** | `transverse_crack` | Crack running perpendicular to roadway direction | N-RDD2024 (D10) |
| **2** | `alligator_crack` | Interconnected crocodile / fatigue cracking | N-RDD2024 (D20) |
| **3** | `repaired_crack` | Sealed / filled / repaired road crack | N-RDD2024 (D40) |
| **4** | `pothole` | Road surface cavity / bowl depression | N-RDD2024 (D44) |
| **5** | `manhole_good` | Structurally sound, flush manhole | Supplementary (good) |
| **6** | `manhole_broken` | Cracked, fractured, or damaged frame | Supplementary (broke) |
| **7** | `manhole_loose` | Displaced, recessed, or sunken cover | Supplementary (lose) |
| **8** | `manhole_uncovered` | Open cavity / missing manhole cover | Supplementary (uncovered) |

> *Semantic Rule Note*: Generic manholes (N-RDD2024 D70) were excluded from training to eliminate class ambiguity with condition-specific manhole labels.

---

## 2. Current Training Status

- **Architecture**: YOLO26n (Nano detector)
- **Epochs Completed**: **33 / 100**
- **Best Validation Checkpoint**: **Epoch 32**
- **Original Training Hardware**: NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM)
- **Batch Size**: 16
- **Image Size**: 640 x 640
- **Dataloader Workers**: `workers=0` (mandatory on Windows to prevent `WinError 1455` pagefile exhaustion during CUDA multiprocessing spawn)
- **Device**: `device=0` (CUDA GPU)
- **Resume Checkpoint**: `checkpoint/last.pt` (Epoch 33 checkpoint, valid and loadable)
- **Best Checkpoint**: `checkpoint/best.pt` (Epoch 32 weights, peak validation mAP)

---

## 3. Dataset Distribution (70 / 20 / 10 Split)

> **Dataset Transfer Notice**: The `processed/` dataset directory is provided as a separate ZIP archive (`processed.zip`). Extract it directly into the repository root so the layout matches:
> `processed/train/images`, `processed/val/images`, `processed/test/images`.

The dataset was curated, verified, and split strictly with zero data leakage across splits:

| Split | Images | Bounding Boxes | Relative Path |
|---|---|---|---|
| **Train** | 13,440 | 32,713 | `processed/train/images` |
| **Validation** | 3,840 | 9,286 | `processed/val/images` |
| **Test (Held-out)** | 1,920 | 4,659 | `processed/test/images` |
| **Total** | **19,200** | **46,658** | `processed/` |

- Dataset paths in `configs/data.yaml` use portable relative paths (`path: processed`).

---

## 4. Benchmark Performance (Epoch 32 Checkpoint)

> *Notice*: These metrics reflect the intermediate Epoch 32 checkpoint of the 100-epoch training schedule, not a completed final run.

### Best Validation Metrics (Epoch 32)
- **Precision**: 61.51%
- **Recall**: 59.37%
- **mAP@50**: 61.52%
- **mAP@50-95**: 39.12%

### Held-Out Test Set Metrics (1,920 Unseen Images)
- **Precision**: 61.20%
- **Recall**: 54.09%
- **mAP@50**: 58.61%
- **mAP@50-95**: 35.87%

---

## 5. Setup & Continuation Instructions for Another PC

Follow these exact steps to clone, configure, and continue training on another Windows NVIDIA GPU machine:

### Step 1: Clone Repository with Git LFS
Ensure Git LFS is installed first (`git lfs install`), then clone:
```powershell
git clone https://github.com/venu8888/Unified-Road-Hazard-Detection-YOLO26n.git
cd Unified-Road-Hazard-Detection-YOLO26n
git lfs pull
```

### Step 2: Set Up Python Virtual Environment
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### Step 3: Install CUDA-Enabled PyTorch & Dependencies
```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Verify CUDA Availability
```powershell
python -c "import torch; print('CUDA Available:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

### Step 5: Resume Training (First-Line Command)
From the repository root:
```powershell
python -m ultralytics train resume model="checkpoint/last.pt"
```

---

## 6. Project Directory Tree

```
Unified-Road-Hazard-Detection-YOLO26n/
│
├── README.md                           # Main repository documentation & resume command
├── .gitignore                          # Clean repository ignore rules
├── .gitattributes                     # Git LFS tracking configuration
├── requirements.txt                    # Exact environment dependencies
├── TRANSFER_MANIFEST.md                # Checkpoint hashes, image counts & verification
│
├── checkpoint/
│   ├── last.pt                         # Resumption checkpoint (Epoch 33, SHA-256 verified)
│   └── best.pt                         # Best checkpoint (Epoch 32, SHA-256 verified)
│
├── configs/
│   ├── data.yaml                       # Portable relative dataset configuration (9 classes)
│   └── augmentation_config.yaml        # Augmentation hyperparameters & design rationale
│
├── processed/
│   ├── train/
│   │   ├── images/                     # 13,440 training images
│   │   └── labels/                     # 13,440 YOLO annotation files
│   ├── val/
│   │   ├── images/                     # 3,840 validation images
│   │   └── labels/                     # 3,840 YOLO annotation files
│   └── test/
│       ├── images/                     # 1,920 held-out test images
│       └── labels/                     # 1,920 YOLO annotation files
│
├── scripts/
│   ├── evaluate_test_set.py            # Evaluates checkpoints on the test set
│   └── generate_test_predictions.py    # Generates visual detection overlays
│
└── results/
    └── review2_evidence/               # Training distributions, QA sheets & evaluation logs
```

---

## 7. Troubleshooting

1. **`CUDA not available / CPU detected`**:
   - Ensure you installed the CUDA-enabled PyTorch wheel (`torch==2.14.0+cu130` via `requirements.txt`).
   - Run `nvidia-smi` in PowerShell to verify NVIDIA display drivers.

2. **`OSError: [WinError 1455] The paging file is too small for this operation to complete`**:
   - Keep `workers=0` on Windows. Windows multiprocessing duplicates the full CUDA virtual memory footprint per worker, rapidly exhausting pagefile limits.
   - If needed, increase your Windows Virtual Memory (Pagefile) size to at least 16–24 GB in Advanced System Settings.

3. **`Git LFS pointer files / Checkpoint load error`**:
   - If `checkpoint/last.pt` is only ~130 bytes, Git LFS was not installed prior to clone.
   - Run `git lfs install` and then `git lfs pull` to download the binary weights and images.

4. **`Dataset path not found`**:
   - Always run commands from the repository root folder (`Unified-Road-Hazard-Detection-YOLO26n`).
   - `configs/data.yaml` specifies `path: processed`, which resolves cleanly relative to the repo root.
