"""
Generate Real Predictions on Test Set using best.pt
Visualizes actual model predictions across all 9 classes,
plus specific crack, pothole, manhole, and multi-hazard examples.
Saves to runs/evaluation/yolo26n_best_epoch32_test/predictions/
"""

import os
import shutil
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO

CLASS_NAMES = [
    'longitudinal_crack',
    'transverse_crack',
    'alligator_crack',
    'repaired_crack',
    'pothole',
    'manhole_good',
    'manhole_broken',
    'manhole_loose',
    'manhole_uncovered'
]

CLASS_COLORS = [
    (59, 130, 246),   # 0: Blue
    (14, 165, 233),   # 1: Sky Blue
    (168, 85, 247),   # 2: Purple
    (236, 72, 153),   # 3: Pink
    (249, 115, 22),   # 4: Orange
    (34, 197, 94),    # 5: Green (Good)
    (239, 68, 68),    # 6: Red (Broken)
    (245, 158, 11),   # 7: Amber (Loose)
    (220, 38, 38)     # 8: Crimson (Uncovered)
]

def draw_predictions_on_image(img_bgr, results_box):
    img_draw = img_bgr.copy()
    h, w = img_draw.shape[:2]
    
    # Draw header bar
    cv2.rectangle(img_draw, (0, 0), (w, 36), (15, 23, 42), -1)
    cv2.putText(img_draw, "YOLO26n Real Test Predictions (best.pt - Epoch 32)", (12, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                
    boxes = results_box.boxes
    for box in boxes:
        cid = int(box.cls[0])
        conf = float(box.conf[0])
        xyxy = box.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = map(int, xyxy)
        
        cname = CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else f'class_{cid}'
        color_rgb = CLASS_COLORS[cid] if cid < len(CLASS_COLORS) else (255, 255, 0)
        color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
        
        # Bounding box
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), color_bgr, 2)
        
        # Label badge
        label_text = f"{cname} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        by = max(y1 - 6, th + 40)
        cv2.rectangle(img_draw, (x1, by - th - 6), (x1 + tw + 8, by + 4), color_bgr, -1)
        cv2.putText(img_draw, label_text, (x1 + 4, by - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    
    return img_draw

def run_predictions():
    model_path = r"runs/detect/runs/detect/runs/train/yolo26n_unified_hazard_optc/weights/best.pt"
    test_img_dir = r"C:\Users\lucky\Downloads\DL PBL\processed\test\images"
    test_lbl_dir = r"C:\Users\lucky\Downloads\DL PBL\processed\test\labels"
    
    out_dirs = [
        r"C:\Users\lucky\Downloads\DL PBL\runs\evaluation\yolo26n_best_epoch32_test\predictions",
        r"C:\Users\lucky\Downloads\DL PBL\runs\detect\runs\evaluation\yolo26n_best_epoch32_test\predictions"
    ]
    for od in out_dirs:
        os.makedirs(od, exist_ok=True)
        
    print(f"Loading {model_path} for test prediction generation...")
    model = YOLO(model_path)
    
    # 1. Identify specific representative images
    # We want 1 image per class (0..8) where the model successfully detects the target class with high confidence
    predictions_catalog = []
    
    # Candidate lists per class
    class_candidates = {c: [] for c in range(9)}
    multi_candidates = []
    
    for lf in sorted(os.listdir(test_lbl_dir)):
        stem = os.path.splitext(lf)[0]
        img_f = stem + '.jpg'
        img_p = os.path.join(test_img_dir, img_f)
        if not os.path.exists(img_p):
            continue
        with open(os.path.join(test_lbl_dir, lf)) as fp:
            lines = [l.strip() for l in fp if l.strip()]
        cids = set(int(l.split()[0]) for l in lines)
        for c in cids:
            class_candidates[c].append(img_p)
        if len(lines) >= 3 and len(cids) >= 2:
            multi_candidates.append(img_p)
            
    # Sample and predict for each class
    for cid in range(9):
        cname = CLASS_NAMES[cid]
        candidates = class_candidates[cid]
        found = False
        
        # Test candidate images until model detects target class with conf >= 0.30
        for cand_p in candidates[:25]:
            res = model.predict(cand_p, conf=0.25, verbose=False)[0]
            detected_classes = [int(b.cls[0]) for b in res.boxes]
            if cid in detected_classes:
                # Target detected!
                img_bgr = cv2.imread(cand_p)
                vis_bgr = draw_predictions_on_image(img_bgr, res)
                out_name = f"pred_class_{cid}_{cname}.jpg"
                for od in out_dirs:
                    cv2.imwrite(os.path.join(od, out_name), vis_bgr)
                predictions_catalog.append((out_name, f"Class {cid} ({cname})", cand_p, len(res.boxes)))
                print(f"Saved prediction for Class {cid} ({cname}): {out_name} with {len(res.boxes)} detections")
                found = True
                break
                
        if not found and len(candidates) > 0:
            # Fallback to first candidate prediction
            cand_p = candidates[0]
            res = model.predict(cand_p, conf=0.15, verbose=False)[0]
            img_bgr = cv2.imread(cand_p)
            vis_bgr = draw_predictions_on_image(img_bgr, res)
            out_name = f"pred_class_{cid}_{cname}.jpg"
            for od in out_dirs:
                cv2.imwrite(os.path.join(od, out_name), vis_bgr)
            predictions_catalog.append((out_name, f"Class {cid} ({cname})", cand_p, len(res.boxes)))
            print(f"Saved prediction (fallback) for Class {cid}: {out_name}")
            
    # Also save designated category examples:
    # 1. Crack example
    crack_cand = class_candidates[0][0]
    res = model.predict(crack_cand, conf=0.25, verbose=False)[0]
    vis = draw_predictions_on_image(cv2.imread(crack_cand), res)
    for od in out_dirs:
        cv2.imwrite(os.path.join(od, "example_crack_detection.jpg"), vis)
        
    # 2. Pothole example
    pothole_cand = class_candidates[4][0]
    res = model.predict(pothole_cand, conf=0.25, verbose=False)[0]
    vis = draw_predictions_on_image(cv2.imread(pothole_cand), res)
    for od in out_dirs:
        cv2.imwrite(os.path.join(od, "example_pothole_detection.jpg"), vis)
        
    # 3. Manhole condition example
    manhole_cand = class_candidates[7][0]
    res = model.predict(manhole_cand, conf=0.25, verbose=False)[0]
    vis = draw_predictions_on_image(cv2.imread(manhole_cand), res)
    for od in out_dirs:
        cv2.imwrite(os.path.join(od, "example_manhole_condition_detection.jpg"), vis)
        
    # 4. Multi-hazard example
    if multi_candidates:
        multi_cand = multi_candidates[0]
        res = model.predict(multi_cand, conf=0.20, verbose=False)[0]
        vis = draw_predictions_on_image(cv2.imread(multi_cand), res)
        for od in out_dirs:
            cv2.imwrite(os.path.join(od, "example_multi_hazard_detection.jpg"), vis)
        print("Saved designated crack, pothole, manhole, and multi-hazard examples.")
        
    # Sync all evaluation files from runs/detect/runs/evaluation/... to runs/evaluation/...
    src_eval = r"C:\Users\lucky\Downloads\DL PBL\runs\detect\runs\evaluation\yolo26n_best_epoch32_test"
    dst_eval = r"C:\Users\lucky\Downloads\DL PBL\runs\evaluation\yolo26n_best_epoch32_test"
    os.makedirs(dst_eval, exist_ok=True)
    for f in os.listdir(src_eval):
        sf = os.path.join(src_eval, f)
        df = os.path.join(dst_eval, f)
        if os.path.isfile(sf):
            shutil.copy2(sf, df)
    print("Synchronized all evaluation artifacts to runs/evaluation/yolo26n_best_epoch32_test.")

if __name__ == '__main__':
    run_predictions()
