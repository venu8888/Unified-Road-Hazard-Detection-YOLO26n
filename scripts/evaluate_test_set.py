"""
Official Test Evaluation Script for YOLO26n Option C best.pt
Evaluates strictly on the held-out TEST set (split='test').
Saves results, metrics, plots, and confusion matrices to runs/evaluation/yolo26n_best_epoch32_test
"""

import os
import json
import pandas as pd
from ultralytics import YOLO

def run_test_evaluation():
    model_path = r"runs/detect/runs/detect/runs/train/yolo26n_unified_hazard_optc/weights/best.pt"
    data_yaml = "configs/data.yaml"
    eval_project = "runs/evaluation"
    eval_name = "yolo26n_best_epoch32_test"
    
    print(f"Loading checkpoint: {model_path}")
    model = YOLO(model_path)
    
    print("Executing validation on TEST set...")
    # workers=0 to ensure rock-solid stability on Windows
    metrics = model.val(
        data=data_yaml,
        split='test',
        imgsz=640,
        batch=16,
        device=0,
        workers=0,
        project=eval_project,
        name=eval_name,
        plots=True,
        save_json=True
    )
    
    save_dir = str(metrics.save_dir)
    print(f"Validation finished. Artifacts saved in: {save_dir}")
    
    # Extract Overall Metrics
    mp = float(metrics.box.mp)
    mr = float(metrics.box.mr)
    map50 = float(metrics.box.map50)
    map50_95 = float(metrics.box.map)
    
    print("\n================ OVERALL TEST METRICS ================")
    print(f"Precision: {mp:.5f} ({mp*100:.2f}%)")
    print(f"Recall:    {mr:.5f} ({mr*100:.2f}%)")
    print(f"mAP@50:    {map50:.5f} ({map50*100:.2f}%)")
    print(f"mAP@50-95: {map50_95:.5f} ({map50_95*100:.2f}%)")
    print("======================================================\n")
    
    # Extract Per-Class Metrics
    class_names = model.names
    per_class_data = []
    
    p_per_class = metrics.box.p
    r_per_class = metrics.box.r
    ap50_per_class = metrics.box.ap50
    ap_per_class = metrics.box.ap # mAP50-95 per class
    
    # Ground truth instance counts per class in test set
    nt_per_class = metrics.box.nt_per_class if hasattr(metrics.box, 'nt_per_class') else []
    
    for i, cid in enumerate(metrics.box.ap_class_index):
        cname = class_names[cid]
        p_val = float(p_per_class[i]) if len(p_per_class) > i else float('nan')
        r_val = float(r_per_class[i]) if len(r_per_class) > i else float('nan')
        ap50_val = float(ap50_per_class[i]) if len(ap50_per_class) > i else float('nan')
        ap_val = float(ap_per_class[i]) if len(ap_per_class) > i else float('nan')
        gt_count = int(nt_per_class[cid]) if len(nt_per_class) > cid else -1
        
        per_class_data.append({
            'class_id': int(cid),
            'class_name': cname,
            'precision': p_val,
            'recall': r_val,
            'ap50': ap50_val,
            'ap50_95': ap_val,
            'ground_truth_instances': gt_count
        })
        print(f"[{cid}] {cname:<20}: P={p_val:.4f}, R={r_val:.4f}, AP50={ap50_val:.4f}, AP50-95={ap_val:.4f}, GT_Count={gt_count}")
        
    df_classes = pd.DataFrame(per_class_data)
    csv_out = os.path.join(save_dir, 'per_class_metrics.csv')
    df_classes.to_csv(csv_out, index=False)
    print(f"Saved per-class metrics to {csv_out}")
    
    summary_metrics = {
        'model_path': model_path,
        'dataset_split': 'test',
        'overall': {
            'precision': mp,
            'recall': mr,
            'mAP50': map50,
            'mAP50_95': map50_95
        },
        'per_class': per_class_data
    }
    
    json_out = os.path.join(save_dir, 'test_metrics_summary.json')
    with open(json_out, 'w', encoding='utf-8') as f:
        json.dump(summary_metrics, f, indent=4)
    print(f"Saved test metrics JSON to {json_out}")
    
    return metrics, save_dir

if __name__ == '__main__':
    run_test_evaluation()
