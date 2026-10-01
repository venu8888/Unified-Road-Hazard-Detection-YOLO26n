# Review 2 Pre-Training Audit & Dataset Harmonization Report (Option C)

**Project:** Unified Road Hazard Detection using YOLO26n  
**Architecture:** **Option C — Single Unified YOLO26n Object Detector (9 Target Classes)**  
**Working Directory:** `C:\Users\lucky\Downloads\DL PBL`  
**Review Stage:** Review 2 Complete (Pre-Training Phase Finalized)  
**Status:** **READY FOR YOLO26n TRAINING**  

---

## 1. Executive Architecture Summary (Option C)

In accordance with user approval, the project utilizes **Option C**: a unified single-stage **YOLO26n** object detector with **9 contiguous training classes**:

```
                  ROAD IMAGE / VIDEO
                          ↓
                       YOLO26n
                          ↓
         ┌────────────────┼────────────────────────┐
         ↓                ↓                        ↓
   SURFACE CRACK       POTHOLE             MANHOLE CONDITION
         ↓                ↓                        ↓
   D00 / D10 /           D40           GOOD / BROKEN / LOOSE /
   D20 / D30                                   UNCOVERED
```

- **Saved Architecture Diagram:** [`results/architecture_diagram.png`](file:///C:/Users/lucky/Downloads/DL%20PBL/results/architecture_diagram.png)

---

## 2. Final 9 Target Classes & Taxonomy Mapping

All class IDs are contiguous from `0` to `8`:

| Class ID | Target Class Name | Source Dataset | Source Class | Description |
|:---:|:---|:---|:---:|:---|
| **0** | `longitudinal_crack` | N-RDD2024 | `D00` | Road crack oriented parallel to driving direction |
| **1** | `transverse_crack` | N-RDD2024 | `D10` | Road crack oriented perpendicular to driving direction |
| **2** | `alligator_crack` | N-RDD2024 | `D20` | Interconnected fatigue/crocodile asphalt cracking |
| **3** | `repaired_crack` | N-RDD2024 | `D30` | Sealed/repaired road surface crack line |
| **4** | `pothole` | N-RDD2024 | `D40` | Structural asphalt cavity/pothole depression |
| **5** | `manhole_good` | Supplementary | `Good` | Structurally intact, flush, secure manhole cover |
| **6** | `manhole_broken` | Supplementary | `Broken` | Cracked casting, fractured frame, or broken cover |
| **7** | `manhole_loose` | Supplementary | `Lose` | Displaced, sunken, misaligned, or rattling cover |
| **8** | `manhole_uncovered` | Supplementary | `Uncovered` | Open utility chamber / missing cover (critical hazard) |

- **Official Mapping File:** [`results/final_label_mapping.csv`](file:///C:/Users/lucky/Downloads/DL%20PBL/results/final_label_mapping.csv)
- **Source Object Manifest:** [`processed/source_manifest.csv`](file:///C:/Users/lucky/Downloads/DL%20PBL/processed/source_manifest.csv) (46,658 object entries)

---

## 3. Strict Manhole Semantic Harmonization Rule & Excluded Classes

### Why N-RDD2024 D70 Is Excluded from Training:
- **D70 means generic manhole-cover object presence.** It provides macro asset location but **zero condition information**.
- **Rule Enforced:** N-RDD2024 `D70` is **NOT mapped to `manhole_good`** (violating that would be an unsupported assumption).
- **Cross-Entropy Conflict Resolved:** Keeping D70 in a unified multi-class model would penalize the model whenever it predicts `manhole_good` on an unannotated good manhole in N-RDD. Therefore, **all 3,712 D70 annotations are excluded from training data**.
- **Qualitative Retention:** N-RDD2024 D70 raw data and visual samples are preserved in [`results/manhole_examples/n_rdd_d70_examples.png`](file:///C:/Users/lucky/Downloads/DL%20PBL/results/manhole_examples/n_rdd_d70_examples.png) and the side-by-side presentation visual [`results/manhole_dataset_comparison.png`](file:///C:/Users/lucky/Downloads/DL%20PBL/results/manhole_dataset_comparison.png) to demonstrate the primary vs supplementary granularity contrast.

### Other Excluded Classes:
- **N-RDD2024 Out-of-Scope:** `D50` (crosswalk blur), `D60` (line blur), `D80` (other non-hazard defect), `D90` (other marking). Excluded without modifying raw data.
- **Supplementary "circle":** Verified completely absent from raw `data.yaml` and label files (0 instances). Excluded with zero label hallucination.
- **Documentation:** [`results/MANUAL_REVIEW_REQUIRED.txt`](file:///C:/Users/lucky/Downloads/DL%20PBL/results/MANUAL_REVIEW_REQUIRED.txt).

---

## 4. Dataset Comparison Evidence for Presentations

- **Visual Asset:** [`results/manhole_dataset_comparison.png`](file:///C:/Users/lucky/Downloads/DL%20PBL/results/manhole_dataset_comparison.png)
- **Presentation Narrative:**
  - **LEFT (Primary N-RDD2024):** Object detection (*"What object is present?"* -> D70 Manhole Cover).
  - **RIGHT (Supplementary Source):** Condition detection (*"What physical condition is it in?"* -> Good, Broken, Loose, Uncovered).
  - Uses 100% **ground-truth annotations** from actual files; no synthetic predictions or fake confidence scores.

---

## 5. Processed Dataset Statistics & Cleaning Summary

- **Total Processed Hazard Images:** **19,200 images**
  - N-RDD2024 Hazard Images: 17,800 images (images with only D70 or only out-of-scope classes excluded)
  - Supplementary Condition Images: 1,400 images
- **Total Valid Bounding Boxes:** **46,658 instances**
- **Automated Validation:**
  - Validated: 19,200 / 19,200 (**100.00% PASS**)
  - Quarantined: **0**
  - Integrity: PIL image verify, normalized box coordinates ($0 \le xc, yc \le 1$, $0 < bw, bh \le 1$), contiguous IDs ($0..8$).
  - Reports: [`results/validation_report.csv`](file:///C:/Users/lucky/Downloads/DL%20PBL/results/validation_report.csv) & [`results/validation_summary.txt`](file:///C:/Users/lucky/Downloads/DL%20PBL/results/validation_summary.txt).
- **Visual QA:** 20 verified samples across all 9 classes in [`results/visual_qa/`](file:///C:/Users/lucky/Downloads/DL%20PBL/results/visual_qa/) and contact sheet in [`results/visual_qa_contact_sheet.png`](file:///C:/Users/lucky/Downloads/DL%20PBL/results/visual_qa_contact_sheet.png).

---

## 6. Exact 70 / 20 / 10 Train / Val / Test Split (`seed=42`)

Class-stratified split with zero data leakage:

| Split | Percentage | Number of Images | Directory Path | Bounding Box Instances |
|:---|:---:|:---:|:---|:---:|
| **Train** | **70.0%** | **13,440** | [`processed/train/images/`](file:///C:/Users/lucky/Downloads/DL%20PBL/processed/train/images) | 32,610 |
| **Validation** | **20.0%** | **3,840** | [`processed/val/images/`](file:///C:/Users/lucky/Downloads/DL%20PBL/processed/val/images) | 9,309 |
| **Test** | **10.0%** | **1,920** | [`processed/test/images/`](file:///C:/Users/lucky/Downloads/DL%20PBL/processed/test/images) | 4,739 |
| **Total** | **100.0%** | **19,200** | [`processed/`](file:///C:/Users/lucky/Downloads/DL%20PBL/processed) | **46,658** |

### Per-Class Instance Distribution Across Splits:
| Class ID | Class Name | Train Boxes | Val Boxes | Test Boxes | Total Boxes | Train % | Val % | Test % |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | `longitudinal_crack` | 16,111 | 4,535 | 2,149 | 22,795 | 70.68% | 19.89% | 9.43% |
| 1 | `transverse_crack` | 7,106 | 1,976 | 1,025 | 10,107 | 70.31% | 19.55% | 10.14% |
| 2 | `alligator_crack` | 5,505 | 1,557 | 758 | 7,820 | 70.40% | 19.91% | 9.69% |
| 3 | `repaired_crack` | 316 | 60 | 41 | 417 | 75.78% | 14.39% | 9.83% |
| 4 | `pothole` | 2,731 | 744 | 408 | 3,883 | 70.33% | 19.16% | 10.51% |
| 5 | `manhole_good` | 484 | 138 | 63 | 685 | 70.66% | 20.15% | 9.20% |
| 6 | `manhole_broken` | 128 | 37 | 18 | 183 | 69.95% | 20.22% | 9.84% |
| 7 | `manhole_loose` | 326 | 100 | 53 | 479 | 68.06% | 20.88% | 11.06% |
| 8 | `manhole_uncovered` | 203 | 57 | 29 | 289 | 70.24% | 19.72% | 10.03% |

- **Saved CSV:** [`results/split_summary.csv`](file:///C:/Users/lucky/Downloads/DL%20PBL/results/split_summary.csv)
- **Saved Visual:** [`results/train_val_test_distribution.png`](file:///C:/Users/lucky/Downloads/DL%20PBL/results/train_val_test_distribution.png)

---

## 7. Augmentation & Evaluation Protocol

- **Validation & Test Sets:** Strictly **UNAUGMENTED** to prevent metric leakage.
- **Training Augmentations:** Mosaic (`1.0`), Random Scale (`0.5`), Horizontal Flip (`0.5`), HSV Jitter (`h: 0.015, s: 0.4, v: 0.4`), MixUp (`0.1`), Vertical Flip disabled (`0.0`).
- **Config File:** [`configs/augmentation_config.yaml`](file:///C:/Users/lucky/Downloads/DL%20PBL/configs/augmentation_config.yaml)

---

## 8. Dataset Smoke Test Verification

- **Config File:** [`configs/data.yaml`](file:///C:/Users/lucky/Downloads/DL%20PBL/configs/data.yaml) (points to `processed/train/images`, `processed/val/images`, `processed/test/images`, `nc: 9`).
- **Ultralytics YOLO Compatibility Smoke Test:** **PASSED** (`yolo26n.pt` verified compatible with `configs/data.yaml`).
- **Smoke Test Log:** [`results/dataset_smoke_test.txt`](file:///C:/Users/lucky/Downloads/DL%20PBL/results/dataset_smoke_test.txt).
- **Training Execution:** **STOPPED AS REQUIRED.**

---

## 9. Next Command to Launch YOLO26n Training

When ready to train the model, execute the following command:

```powershell
.venv\Scripts\python.exe -m ultralytics train model=yolo26n.pt data=configs/data.yaml epochs=100 imgsz=640 batch=16 device=0 workers=4 project=runs/train name=yolo26n_unified_hazard_optc
```
