"""
Dataset Visual QA Script for Option C (9-Class Taxonomy)
Generates representative annotated images across all 9 target classes
and a visual QA contact sheet.
Outputs:
  - results/visual_qa/*.png
  - results/visual_qa_contact_sheet.png
"""

import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

random.seed(42)

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

def draw_qa_image(img_path, lbl_path, target_cid=None):
    img = cv2.imread(img_path)
    if img is None:
        return None
    h, w = img.shape[:2]
    
    with open(lbl_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
        
    for line in lines:
        parts = line.split()
        cid = int(parts[0])
        xc, yc, bw, bh = map(float, parts[1:])
        
        xmin = max(0, min(w, int((xc - bw / 2.0) * w)))
        ymin = max(0, min(h, int((yc - bh / 2.0) * h)))
        xmax = max(0, min(w, int((xc + bw / 2.0) * w)))
        ymax = max(0, min(h, int((yc + bh / 2.0) * h)))
        
        cname = CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else f'class_{cid}'
        color = CLASS_COLORS[cid] if cid < len(CLASS_COLORS) else (0, 255, 255)
        
        color_bgr = (color[2], color[1], color[0])
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color_bgr, 2)
        
        label_text = f"{cid}: {cname}"
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        by = max(ymin - 4, th + 4)
        cv2.rectangle(img, (xmin, by - th - 4), (xmin + tw + 6, by + 4), color_bgr, -1)
        cv2.putText(img, label_text, (xmin + 3, by), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def run_visual_qa(processed_dir='processed', out_dir='results/visual_qa'):
    os.makedirs(out_dir, exist_ok=True)
    img_dir = os.path.join(processed_dir, 'images')
    lbl_dir = os.path.join(processed_dir, 'labels')
    
    class_to_files = {i: [] for i in range(len(CLASS_NAMES))}
    all_lbls = sorted(os.listdir(lbl_dir))
    
    for lf in all_lbls:
        stem = os.path.splitext(lf)[0]
        img_f = os.path.join(img_dir, stem + '.jpg')
        if not os.path.exists(img_f):
            continue
        with open(os.path.join(lbl_dir, lf), 'r', encoding='utf-8') as f:
            lines = [l.strip() for l in f if l.strip()]
        for line in lines:
            cid = int(line.split()[0])
            if cid in class_to_files:
                class_to_files[cid].append((img_f, os.path.join(lbl_dir, lf)))
                
    # Sample 2-3 per class to reach at least 20 images
    selected_qa = []
    qa_index = 1
    for cid in range(len(CLASS_NAMES)):
        cname = CLASS_NAMES[cid]
        candidates = list(set(class_to_files[cid]))
        sample_count = 3 if cid in [0, 4] else 2
        sampled = random.sample(candidates, min(sample_count, len(candidates)))
        for img_p, lbl_p in sampled:
            vis = draw_qa_image(img_p, lbl_p, target_cid=cid)
            if vis is not None:
                save_name = f"qa_{qa_index:02d}_cid{cid}_{cname}.png"
                save_p = os.path.join(out_dir, save_name)
                Image.fromarray(vis).save(save_p)
                selected_qa.append((cid, cname, vis))
                qa_index += 1
                
    print(f"Generated {len(selected_qa)} individual Visual QA images in {out_dir}")
    
    # Build 4x5 Contact Sheet (20 images total)
    fig, axes = plt.subplots(4, 5, figsize=(20, 16), facecolor='#0f172a')
    fig.suptitle('Visual QA Contact Sheet: Option C Harmonized Annotations (9 Target Classes)', 
                 fontsize=18, fontweight='bold', color='white', y=0.98)
    
    for idx, ax in enumerate(axes.flat):
        if idx < len(selected_qa):
            cid, cname, vis = selected_qa[idx]
            ax.imshow(vis)
            ax.set_title(f"Class {cid}: {cname}", fontsize=11, fontweight='bold', color='white', pad=6)
        ax.axis('off')
        
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    contact_path = os.path.join('results', 'visual_qa_contact_sheet.png')
    plt.savefig(contact_path, dpi=300, facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved Contact Sheet to {contact_path}")

if __name__ == '__main__':
    run_visual_qa()
