# Architectural Decision: Unified Detector vs. Hierarchical Two-Stage Pipeline

**Project:** Unified Road Hazard Detection using YOLO26n  
**Stage:** Review 2 Pre-Training Audit & Dataset Harmonization  
**Document:** `results/ARCHITECTURE_DECISION.md`  

---

## 1. Executive Summary

During the Review 2 audit of the downloaded datasets (**N-RDD2024** and **Supplementary Roboflow Manhole Cover v4**), a fundamental semantic hierarchy difference was identified between the two sources:

- **N-RDD2024 Category `D70`**: Represents an **object-level detector** answering *"Is there a manhole cover on the road?"* (3,712 instances). No condition or integrity assessment is provided.
- **Supplementary Roboflow Dataset**: Represents a **state/condition-level assessor** answering *"What physical state is this manhole cover in?"* (`Good`: 685, `Broken`: 183, `Lose`: 479, `Uncovered`: 289).
- **The "circle" label**: An audit of all 1,427 raw annotation files verified that `circle` is **completely absent** in the downloaded v4 dataset export. It is safely omitted with zero label hallucination.

The core dilemma is whether to train a **Single Unified 10-Class Detector (Option A)** or a **Hierarchical Two-Stage Detector + Condition Interpreter (Option B)**.

---

## 2. Comparison of Architectural Options

### OPTION A: Single Unified YOLO26n Multi-Class Detector (10 Classes)
In Option A, the dataset is merged into a single multi-class object detection dataset with 10 target classes:
- Class 0: `longitudinal_crack` (D00)
- Class 1: `transverse_crack` (D10)
- Class 2: `alligator_crack` (D20)
- Class 3: `repaired_crack` (D30)
- Class 4: `pothole` (D40)
- Class 5: `manhole_generic` (D70 from N-RDD2024, condition unspecified)
- Class 6: `manhole_broken` (Roboflow Broken)
- Class 7: `manhole_good` (Roboflow Good)
- Class 8: `manhole_loose` (Roboflow Lose)
- Class 9: `manhole_uncovered` (Roboflow Uncovered)

#### Pros:
1. Single forward pass in inference: end-to-end real-time speed on the RTX 4050 Laptop GPU.
2. Directly compatible with standard YOLO26n training pipelines and single `data.yaml`.
3. Preserves all labels without losing condition fidelity.

#### Cons / Semantic Hazard:
1. **Semantic Competition & Negative Penalty**: N-RDD images containing sound manhole covers are labeled as Class 5 (`manhole_generic`). If the detector predicts Class 7 (`manhole_good`), standard cross-entropy loss penalizes it as a False Positive, even though visually it is good.
2. **Dataset Domain Shift**: The supplementary dataset has tight crops or specific street scenes that may bias condition predictions.

---

### OPTION B: Hierarchical Two-Stage Detection Pipeline
In Option B, the system is split into two specialized deep learning models:

#### Stage 1: Primary Road Hazard Detector (YOLO26n - 6 Classes)
Detects all roadway physical objects and hazards:
- Class 0: `longitudinal_crack` (D00)
- Class 1: `transverse_crack` (D10)
- Class 2: `alligator_crack` (D20)
- Class 3: `repaired_crack` (D30)
- Class 4: `pothole` (D40)
- Class 5: `manhole` (Harmonized: All D70 + All Supplementary covers unified at object level)

#### Stage 2: Manhole Condition Interpreter (YOLO26n-cls or Secondary Detector - 4 Classes)
When Stage 1 detects a `manhole` (Class 5), the cropped bounding-box region is passed to Stage 2:
- State 0: `Broken` (Hazard)
- State 1: `Good` (Safe)
- State 2: `Loose` (Hazard)
- State 3: `Uncovered` (Critical Hazard)

#### Pros:
1. **Semantically 100% Sound**: Eliminates negative loss competition between `D70` and `manhole_good`. All manholes in Stage 1 share the exact same object class (`manhole`).
2. **Modularity**: Stage 2 can be trained exclusively on the high-fidelity condition dataset without contaminating the primary crack/pothole detector.
3. Allows deployment of confidence thresholds specifically tailored to manhole condition risks.

#### Cons:
1. Requires running two models (or detector + crop classifier), slightly increasing pipeline complexity.

---

## 3. What Can Be Safely Unified Right Now

1. **Crack and Pothole Categories**:
   - `D00` -> `longitudinal_crack`
   - `D10` -> `transverse_crack`
   - `D20` -> `alligator_crack`
   - `D30` -> `repaired_crack`
   - `D40` -> `pothole`
   These categories have clean, mutually exclusive definitions across all 6 countries.

2. **Out-of-Scope Source Exclusions**:
   - `D50`, `D60`, `D80`, `D90` are safely filtered out without modifying raw datasets.

3. **Condition Classes in Supplementary Dataset**:
   - `Broken`, `Good`, `Lose`, `Uncovered` are verified and validated with clean YOLO bounding boxes.

---

## 4. What Requires Manual Confirmation / Final Decision

Before launching training, the user must choose between:

- **CHOICE 1 (Option A)**: Proceed with the **10-Class Unified YOLO26n Detector** (where `manhole_generic` is distinct from the 4 condition classes).
- **CHOICE 2 (Option B)**: Proceed with the **Hierarchical Architecture**:
  - Primary Detector: 6 Classes (`longitudinal_crack`, `transverse_crack`, `alligator_crack`, `repaired_crack`, `pothole`, `manhole`)
  - Secondary Condition Classifier: 4 Classes (`Broken`, `Good`, `Loose`, `Uncovered`).

We prepare the reproducible scripts to support **both** representations so that whichever choice the user approves can be trained immediately.
