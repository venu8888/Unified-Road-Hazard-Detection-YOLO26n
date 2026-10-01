# YOLO26n Option C Test Set Evaluation Summary Report

**Project:** Unified Road Hazard Detection using YOLO26n  
**Evaluation Target:** Held-out Test Set (`processed/test/`)  
**Checkpoint Evaluated:** `best.pt` (Epoch 32 Checkpoint from 33-epoch run)  
**Evaluation Output Directory:** `runs/evaluation/yolo26n_best_epoch32_test`  
**Execution Mode:** Pure Evaluation Only (Zero Training / Zero Fine-tuning)  

---

## 1. Test Set Verification

- **Total Test Images:** **1,920 images** (100% verified `.jpg` format, resolution $640 \times 640$).
- **Total Test Annotation Files:** **1,920 labels** (100% verified paired `.txt` format).
- **Split Integrity & Leakage Check:**
  - Overlap with Training Set (`processed/train/`): **0 images (0.0%)**
  - Overlap with Validation Set (`processed/val/`): **0 images (0.0%)**
  - Missing or unreadable test label files: **0 files**
- **Augmentation Protocol:**
  - The test set is **strictly unaugmented** in accordance with `configs/augmentation_config.yaml`.
  - Clean spatial and photometric realism is preserved for unbiased benchmarking.
- **Ground Truth Target Instances in Test Set:** **4,544 total bounding boxes**.

---

## 2. Model & Checkpoint Metadata

- **Checkpoint File:** `runs/detect/runs/detect/runs/train/yolo26n_unified_hazard_optc/weights/best.pt`
- **File Size:** **14.85 MB** (15,572,992 bytes)
- **Model Architecture:** YOLO26n Detection Model (`task: detect`)
- **Number of Parameters:** **2,376,591 parameters** (2.38M)
- **Model Computational Complexity:** **5.3 GFLOPs** (fused inference)
- **Target Classes:** **9 Classes** (Contiguous IDs 0 to 8)
- **Training Epoch of Checkpoint:** **Epoch 32 / 100** (0-indexed `epoch: 31` in PyTorch state dict)
- **Inference Speed on RTX 4050 Laptop GPU:** **3.6 ms inference + 0.3 ms preprocess + 0.7 ms NMS postprocess** (~217 FPS at batch 1).

---

## 3. Overall Test Set Performance Metrics

Evaluated across all 1,920 held-out test images with standard IoU thresholds:

| Metric | Raw Value | Percentage |
|:---|:---:|:---:|
| **Precision ($P$)** | `0.61204` | **61.20%** |
| **Recall ($R$)** | `0.54092` | **54.09%** |
| **mAP@50** | `0.58606` | **58.61%** |
| **mAP@50-95** | `0.35870` | **35.87%** |

### Additional COCO-Benchmark Metric Breakdown:
- $\text{AP}_{50:95} \text{ (All Sizes)} = \mathbf{0.360}$
- $\text{AP}_{50} \text{ (All Sizes)} = \mathbf{0.586}$
- $\text{AP}_{75} = \mathbf{0.383}$
- $\text{AP}_{\text{Small Objects}} = \mathbf{0.082}$
- $\text{AP}_{\text{Medium Objects}} = \mathbf{0.292}$
- $\text{AP}_{\text{Large Objects}} = \mathbf{0.401}$
- $\text{AR}_{50:95} \text{ (Max 100 Detections)} = \mathbf{0.598}$
- $\text{AR}_{50} = \mathbf{0.922}$

---

## 4. Per-Class Test Set Performance

Measured values from `per_class_metrics.csv` and `test_metrics_summary.json`:

| Class ID | Class Name | Ground Truth Count | Images with Class | Precision ($P$) | Recall ($R$) | AP@50 | AP@50-95 |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **0** | `longitudinal_crack` | 2,149 | 1,101 | 0.6278 (62.78%) | 0.4714 (47.14%) | **0.5193 (51.93%)** | 0.2769 (27.69%) |
| **1** | `transverse_crack` | 1,025 | 638 | 0.6008 (60.08%) | 0.4556 (45.56%) | **0.4982 (49.82%)** | 0.2318 (23.18%) |
| **2** | `alligator_crack` | 758 | 588 | 0.6654 (66.54%) | 0.5092 (50.92%) | **0.5898 (58.98%)** | 0.3195 (31.95%) |
| **3** | `repaired_crack` | 41 | 25 | 0.4992 (49.92%) | 0.1220 (12.20%) | **0.2127 (21.27%)** | 0.1182 (11.82%) |
| **4** | `pothole` | 408 | 233 | 0.6258 (62.58%) | 0.3198 (31.98%) | **0.3886 (38.86%)** | 0.1548 (15.48%) |
| **5** | `manhole_good` | 63 | 53 | 0.5552 (55.52%) | 0.7302 (73.02%) | **0.6874 (68.74%)** | 0.4755 (47.55%) |
| **6** | `manhole_broken` | 18 | 17 | 0.6197 (61.97%) | 0.5556 (55.56%) | **0.6312 (63.12%)** | 0.4375 (43.75%) |
| **7** | `manhole_loose` | 53 | 53 | 0.7275 (72.75%) | 0.7736 (77.36%) | **0.8420 (84.20%)** | 0.5394 (53.94%) |
| **8** | `manhole_uncovered` | 29 | 29 | 0.5869 (58.69%) | 0.9310 (93.10%) | **0.9053 (90.53%)** | 0.6747 (67.47%) |

---

## 5. Validation vs. Test Set Comparison

Comparison between the best Validation metrics recorded at Epoch 32 and the independent Test set:

| Evaluation Metric | Validation Set (Epoch 32) | Test Set (Epoch 32) | Absolute Numerical Difference | Relative Percentage Point Delta |
|:---|:---:|:---:|:---:|:---:|
| **Precision** | `0.61508` (61.51%) | `0.61204` (61.20%) | `-0.00304` | **-0.30%** |
| **Recall** | `0.59372` (59.37%) | `0.54092` (54.09%) | `-0.05280` | **-5.28%** |
| **mAP@50** | `0.61521` (61.52%) | `0.58606` (58.61%) | `-0.02915` | **-2.92%** |
| **mAP@50-95** | `0.39115` (39.12%) | `0.35870` (35.87%) | `-0.03245` | **-3.25%** |

### Numerical Interpretation:
1. **Precision Stability:** Precision shows an extremely small change across splits (-0.30%), demonstrating that the model maintains a reliable confidence threshold on completely unseen imagery without an explosion of false-positive detections.
2. **Recall Variance:** Test recall is 5.28 percentage points lower than validation recall. This is primarily influenced by the low recall on rarer classes:
   - `repaired_crack` (Class 3): $R = 0.1220$ across only 41 test instances.
   - `pothole` (Class 4): $R = 0.3198$ across 408 test instances.
3. **High-Risk Hazard Fidelity:** In contrast to small crack lines, severe roadway hazards demonstrated high sensitivity on the test set:
   - `manhole_uncovered` achieved **93.10% Recall** and **90.53% AP@50**.
   - `manhole_loose` achieved **77.36% Recall** and **84.20% AP@50**.

---

## 6. Confusion Matrix & Curve Interpretation

- **Preserved Figures:**
  - [`confusion_matrix.png`](file:///C:/Users/lucky/Downloads/DL%20PBL/runs/evaluation/yolo26n_best_epoch32_test/confusion_matrix.png)
  - [`confusion_matrix_normalized.png`](file:///C:/Users/lucky/Downloads/DL%20PBL/runs/evaluation/yolo26n_best_epoch32_test/confusion_matrix_normalized.png)
  - [`BoxPR_curve.png`](file:///C:/Users/lucky/Downloads/DL%20PBL/runs/evaluation/yolo26n_best_epoch32_test/BoxPR_curve.png)
  - [`BoxF1_curve.png`](file:///C:/Users/lucky/Downloads/DL%20PBL/runs/evaluation/yolo26n_best_epoch32_test/BoxF1_curve.png)
- **Observations:**
  1. **Background Confusion:** For thin cracks (`longitudinal_crack`, `transverse_crack`), undetected objects primarily fall into the background column (missed detection rather than cross-class confusion).
  2. **Inter-Class Separation:** Manhole condition classes (`manhole_good`, `manhole_broken`, `manhole_loose`, `manhole_uncovered`) exhibit minimal confusion with crack categories, validating the semantic separation established in Option C.
  3. **Peak F1 Score:** As shown in `BoxF1_curve.png`, the optimal overall confidence threshold is approximately **0.30 - 0.40**, balancing precision and recall.

---

## 7. Prediction Visualizations Generated

All prediction images were rendered using actual model outputs from `best.pt` on real held-out test images (located in `runs/evaluation/yolo26n_best_epoch32_test/predictions/`):

1. **Class 0:** [`pred_class_0_longitudinal_crack.jpg`](file:///C:/Users/lucky/Downloads/DL%20PBL/runs/evaluation/yolo26n_best_epoch32_test/predictions/pred_class_0_longitudinal_crack.jpg)
2. **Class 1:** [`pred_class_1_transverse_crack.jpg`](file:///C:/Users/lucky/Downloads/DL%20PBL/runs/evaluation/yolo26n_best_epoch32_test/predictions/pred_class_1_transverse_crack.jpg)
3. **Class 2:** [`pred_class_2_alligator_crack.jpg`](file:///C:/Users/lucky/Downloads/DL%20PBL/runs/evaluation/yolo26n_best_epoch32_test/predictions/pred_class_2_alligator_crack.jpg)
4. **Class 3:** [`pred_class_3_repaired_crack.jpg`](file:///C:/Users/lucky/Downloads/DL%20PBL/runs/evaluation/yolo26n_best_epoch32_test/predictions/pred_class_3_repaired_crack.jpg)
5. **Class 4:** [`pred_class_4_pothole.jpg`](file:///C:/Users/lucky/Downloads/DL%20PBL/runs/evaluation/yolo26n_best_epoch32_test/predictions/pred_class_4_pothole.jpg)
6. **Class 5:** [`pred_class_5_manhole_good.jpg`](file:///C:/Users/lucky/Downloads/DL%20PBL/runs/evaluation/yolo26n_best_epoch32_test/predictions/pred_class_5_manhole_good.jpg)
7. **Class 6:** [`pred_class_6_manhole_broken.jpg`](file:///C:/Users/lucky/Downloads/DL%20PBL/runs/evaluation/yolo26n_best_epoch32_test/predictions/pred_class_6_manhole_broken.jpg)
8. **Class 7:** [`pred_class_7_manhole_loose.jpg`](file:///C:/Users/lucky/Downloads/DL%20PBL/runs/evaluation/yolo26n_best_epoch32_test/predictions/pred_class_7_manhole_loose.jpg)
9. **Class 8:** [`pred_class_8_manhole_uncovered.jpg`](file:///C:/Users/lucky/Downloads/DL%20PBL/runs/evaluation/yolo26n_best_epoch32_test/predictions/pred_class_8_manhole_uncovered.jpg)
10. **Designated Crack Example:** [`example_crack_detection.jpg`](file:///C:/Users/lucky/Downloads/DL%20PBL/runs/evaluation/yolo26n_best_epoch32_test/predictions/example_crack_detection.jpg)
11. **Designated Pothole Example:** [`example_pothole_detection.jpg`](file:///C:/Users/lucky/Downloads/DL%20PBL/runs/evaluation/yolo26n_best_epoch32_test/predictions/example_pothole_detection.jpg)
12. **Designated Manhole Example:** [`example_manhole_condition_detection.jpg`](file:///C:/Users/lucky/Downloads/DL%20PBL/runs/evaluation/yolo26n_best_epoch32_test/predictions/example_manhole_condition_detection.jpg)
13. **Designated Multi-Hazard Example:** [`example_multi_hazard_detection.jpg`](file:///C:/Users/lucky/Downloads/DL%20PBL/runs/evaluation/yolo26n_best_epoch32_test/predictions/example_multi_hazard_detection.jpg)

---

## 8. Limitations & Observations

1. **Epoch Horizon:** This checkpoint was saved at Epoch 32 of a planned 100-epoch schedule. The learning curves indicate that training had not reached convergence, so potential gains remain available if resumed.
2. **Small Target Difficulty:** As evidenced by the COCO metric $\text{AP}_{\text{Small}} = 0.082$ vs $\text{AP}_{\text{Large}} = 0.401$, distant/narrow crack fractures are harder for a 2.38M parameter nano model than localized manhole structures.
3. **Class Imbalance in Test Set:** `repaired_crack` has only 41 instances in the test set, creating high variance in per-class metric estimates for that category.
