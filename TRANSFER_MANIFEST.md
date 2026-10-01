# Transfer Package Manifest

## 1. Repository Information
- **Repository URL**: https://github.com/venu8888/Unified-Road-Hazard-Detection-YOLO26n
- **Branch**: main
- **Commit Hash**: `aee54cc76dd3bde8e8158de29590e4ca29c7c627`
- **Target Architecture**: YOLO26n (9 Road Hazard Classes)
- **Training Progression**: Epoch 33 / 100 Completed

## 2. Checkpoints Integrity

| File | Size (Bytes) | SHA-256 Checksum | Description |
|---|---|---|---|
| `checkpoint/last.pt` | 15,573,859 | `C81015C166995B4BB8CC1A44CBB3AB891004031F338C6DDD0CEBFD3191A72257` | Resume checkpoint from Epoch 33 |
| `checkpoint/best.pt` | 15,573,731 | `9087AFCA2FD8ACC369A004A84D85118D5AA443B9BCB31297B6284F1EBB979353` | Best validation weights (Epoch 32) |

## 3. Dataset Verification

| Split | Images | Labels (.txt) | Total Boxes | Overlap with Test |
|---|---|---|---|---|
| `processed/train/` | 13,440 | 13,440 | 32,713 | 0 (Strictly 0%) |
| `processed/val/` | 3,840 | 3,840 | 9,286 | 0 (Strictly 0%) |
| `processed/test/` | 1,920 | 1,920 | 4,659 | 0 (Strictly 0%) |
| **Total** | **19,200** | **19,200** | **46,658** | **Leakage Verified Clean** |

## 4. Class Distribution

| Class ID | Class Name | Total Instances | Train Instances | Val Instances | Test Instances |
|---|---|---|---|---|---|
| 0 | `longitudinal_crack` | 5,666 | 3,936 | 1,162 | 568 |
| 1 | `transverse_crack` | 4,499 | 3,184 | 884 | 431 |
| 2 | `alligator_crack` | 5,263 | 3,694 | 1,027 | 542 |
| 3 | `repaired_crack` | 1,514 | 1,061 | 303 | 150 |
| 4 | `pothole` | 6,368 | 4,498 | 1,247 | 623 |
| 5 | `manhole_good` | 11,570 | 8,110 | 2,305 | 1,155 |
| 6 | `manhole_broken` | 3,822 | 2,660 | 763 | 399 |
| 7 | `manhole_loose` | 4,374 | 3,061 | 871 | 442 |
| 8 | `manhole_uncovered` | 3,582 | 2,509 | 724 | 349 |
| **Sum** | | **46,658** | **32,713** | **9,286** | **4,659** |

## 5. Evaluation Benchmarks

- **Validation (Epoch 32 Best)**: Precision = 61.51%, Recall = 59.37%, mAP50 = 61.52%, mAP50-95 = 39.12%
- **Held-Out Test Set (Unseen 1,920 Images)**: Precision = 61.20%, Recall = 54.09%, mAP50 = 58.61%, mAP50-95 = 35.87%
