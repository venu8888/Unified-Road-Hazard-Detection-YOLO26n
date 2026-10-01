"""
Polished Presentation Visuals:
1. results/manhole_dataset_comparison.png (Ground-truth manhole comparison)
2. results/architecture_diagram.png (Option C Unified Architecture Diagram)
"""

import os
import zipfile
import xml.etree.ElementTree as ET
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

random.seed(42)

def draw_voc_boxes(img_rgb, xml_content, highlight_class='D70', dataset_tag="N-RDD2024"):
    root = ET.fromstring(xml_content)
    img_draw = img_rgb.copy()
    h, w = img_draw.shape[:2]
    
    cv2.rectangle(img_draw, (0, 0), (w, 42), (15, 23, 42), -1)
    cv2.putText(img_draw, f"[{dataset_tag}] Target: {highlight_class} (Generic Manhole)", (15, 28), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    
    for obj in root.findall('object'):
        cname = obj.find('name').text.strip()
        bndbox = obj.find('bndbox')
        xmin = max(0, min(w, int(float(bndbox.find('xmin').text))))
        ymin = max(0, min(h, int(float(bndbox.find('ymin').text))))
        xmax = max(0, min(w, int(float(bndbox.find('xmax').text))))
        ymax = max(0, min(h, int(float(bndbox.find('ymax').text))))
        
        if cname == highlight_class:
            color = (249, 115, 22) # Orange
            thickness = 3
        else:
            color = (100, 116, 139) # Gray
            thickness = 1
            
        cv2.rectangle(img_draw, (xmin, ymin), (xmax, ymax), (color[2], color[1], color[0]), thickness)
        
        label_text = f"GT: {cname}"
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        by = max(ymin - 6, th + 48)
        cv2.rectangle(img_draw, (xmin, by - th - 6), (xmin + tw + 10, by + 6), (color[2], color[1], color[0]), -1)
        cv2.putText(img_draw, label_text, (xmin + 5, by), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
        
    return img_draw

def draw_yolo_boxes(img_rgb, label_lines, highlight_class, dataset_tag="Supplementary Dataset"):
    img_draw = img_rgb.copy()
    h, w = img_draw.shape[:2]
    
    cv2.rectangle(img_draw, (0, 0), (w, 42), (15, 23, 42), -1)
    cv2.putText(img_draw, f"[{dataset_tag}] Condition: {highlight_class}", (15, 28), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    
    color_map = {
        'Good': (34, 197, 94),      # Green
        'Broken': (239, 68, 68),    # Red
        'Loose': (245, 158, 11),    # Amber
        'Uncovered': (220, 38, 38)  # Crimson
    }
    class_map = {0: 'Broken', 1: 'Good', 2: 'Loose', 3: 'Uncovered'}
    
    for line in label_lines:
        parts = line.strip().split()
        if not parts:
            continue
        cid = int(parts[0])
        cname = class_map.get(cid, f'class_{cid}')
        xc, yc, bw, bh = map(float, parts[1:5])
        
        xmin = max(0, min(w, int((xc - bw / 2) * w)))
        ymin = max(0, min(h, int((yc - bh / 2) * h)))
        xmax = max(0, min(w, int((xc + bw / 2) * w)))
        ymax = max(0, min(h, int((yc + bh / 2) * h)))
        
        rgb_c = color_map.get(cname, (234, 179, 8))
        color_bgr = (rgb_c[2], rgb_c[1], rgb_c[0])
        thickness = 3 if cname == highlight_class else 2
        cv2.rectangle(img_draw, (xmin, ymin), (xmax, ymax), color_bgr, thickness)
        
        label_text = f"GT: {cname}"
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        by = max(ymin - 6, th + 48)
        cv2.rectangle(img_draw, (xmin, by - th - 6), (xmin + tw + 10, by + 6), color_bgr, -1)
        cv2.putText(img_draw, label_text, (xmin + 5, by), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
                    
    return img_draw

def generate_comparison_visual():
    out_path = os.path.join('results', 'manhole_dataset_comparison.png')
    rdd_dir = r'C:\Users\lucky\Downloads\DL PBL\N-RDD2024Road damage and defects\N-RDD2024Road damage and defects\Training and Validation Dataset'
    manhole_dir = r'C:\Users\lucky\Downloads\DL PBL\Manhole Cover Dataset YOLO.v4i.yolo26'
    
    # 2 D70 images from N-RDD
    zpath = os.path.join(rdd_dir, 'Czech Republic_voc.zip')
    d70_imgs = []
    with zipfile.ZipFile(zpath, 'r') as z:
        for xf in ['train/Czech_000006_jpg.rf.0d307f69f62ee0af8ea2477c481498ab.xml',
                   'train/Czech_000100_jpg.rf.a342b227669697cdb84f298e0aeedcea.xml']:
            img_f = xf.replace('.xml', '.jpg')
            img_arr = np.array(Image.open(io.BytesIO(z.read(img_f))).convert('RGB'))
            vis = draw_voc_boxes(img_arr, z.read(xf), highlight_class='D70', dataset_tag="N-RDD2024")
            d70_imgs.append(vis)
            
    # 4 condition images from Supplementary (Good, Broken, Loose, Uncovered)
    class_map = {0: 'Broken', 1: 'Good', 2: 'Loose', 3: 'Uncovered'}
    condition_imgs = {}
    
    for s in ['train', 'valid', 'test']:
        lbl_dir = os.path.join(manhole_dir, s, 'labels')
        img_dir = os.path.join(manhole_dir, s, 'images')
        for lf in os.listdir(lbl_dir):
            stem = os.path.splitext(lf)[0]
            with open(os.path.join(lbl_dir, lf), 'r') as fp:
                lines = [line.strip() for line in fp if line.strip()]
            if len(lines) == 1:
                cid = int(lines[0].split()[0])
                cname = class_map[cid]
                if cname not in condition_imgs:
                    img_f = os.path.join(img_dir, stem + '.jpg')
                    img_arr = np.array(Image.open(img_f).convert('RGB'))
                    vis = draw_yolo_boxes(img_arr, lines, highlight_class=cname, dataset_tag="Supplementary")
                    condition_imgs[cname] = vis
            if len(condition_imgs) == 4:
                break
        if len(condition_imgs) == 4:
            break
            
    left_stack = np.vstack((d70_imgs[0], d70_imgs[1]))
    r_top = np.hstack((condition_imgs['Good'], condition_imgs['Broken']))
    r_bot = np.hstack((condition_imgs['Loose'], condition_imgs['Uncovered']))
    right_grid = np.vstack((r_top, r_bot))
    
    h_target = left_stack.shape[0]
    w_target = int(right_grid.shape[1] * (h_target / right_grid.shape[0]))
    right_grid_resized = cv2.resize(right_grid, (w_target, h_target), interpolation=cv2.INTER_AREA)
    combined_body = np.hstack((left_stack, right_grid_resized))
    
    fig = plt.figure(figsize=(19, 12), facecolor='#0f172a')
    fig.suptitle('Manhole Label Granularity: Primary vs Supplementary Dataset', 
                 fontsize=22, fontweight='bold', color='white', y=0.97)
    
    fig.text(0.27, 0.92, "PRIMARY DATASET — N-RDD2024\nObject-Level: D70 — Manhole Cover", 
             ha='center', fontsize=14, fontweight='bold', color='#60a5fa')
    fig.text(0.74, 0.92, "SUPPLEMENTARY MANHOLE DATASET\nCondition-Level: Good | Broken | Loose | Uncovered", 
             ha='center', fontsize=14, fontweight='bold', color='#4ade80')
             
    ax = fig.add_axes([0.03, 0.09, 0.94, 0.81])
    ax.imshow(combined_body)
    ax.axis('off')
    
    fig.text(0.5, 0.04, 
             "N-RDD2024 provides generic manhole-object annotation.\n"
             "The supplementary dataset provides finer condition-level annotation.", 
             ha='center', fontsize=13, fontweight='bold', color='#fde047')
    fig.text(0.5, 0.015,
             "Ground-truth annotations shown from raw dataset files. No fake predictions or synthetic scores. D70 excluded from training data to avoid semantic conflict.",
             ha='center', fontsize=10, color='#94a3b8', style='italic')
             
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved {out_path}")

def generate_architecture_diagram():
    out_path = os.path.join('results', 'architecture_diagram.png')
    fig, ax = plt.subplots(figsize=(14, 9), facecolor='#0f172a')
    ax.set_facecolor('#0f172a')
    
    # Draw architecture boxes with clean modern styling
    # Top box: ROAD IMAGE / VIDEO
    bbox_input = dict(boxstyle="round,pad=0.6", fc="#1e293b", ec="#38bdf8", lw=2)
    ax.text(0.5, 0.90, "ROAD IMAGE / VIDEO\n(Dashcam / Mobile / Inspection Stream)", 
            ha='center', va='center', fontsize=13, fontweight='bold', color='white', bbox=bbox_input)
    
    # Arrow 1
    ax.annotate('', xy=(0.5, 0.77), xytext=(0.5, 0.83),
                arrowprops=dict(facecolor='#38bdf8', edgecolor='#38bdf8', width=2.5, headwidth=9))
    
    # Model box: YOLO26n Unified Object Detector
    bbox_model = dict(boxstyle="round,pad=0.8", fc="#0369a1", ec="#7dd3fc", lw=2.5)
    ax.text(0.5, 0.71, "YOLO26n UNIFIED OBJECT DETECTOR\n(Option C: 9-Class Unified Hazard Architecture)", 
            ha='center', va='center', fontsize=14, fontweight='bold', color='white', bbox=bbox_model)
    
    # Branching arrows
    ax.annotate('', xy=(0.20, 0.52), xytext=(0.42, 0.64),
                arrowprops=dict(facecolor='#60a5fa', edgecolor='#60a5fa', width=2, headwidth=8))
    ax.annotate('', xy=(0.50, 0.52), xytext=(0.50, 0.64),
                arrowprops=dict(facecolor='#f97316', edgecolor='#f97316', width=2, headwidth=8))
    ax.annotate('', xy=(0.80, 0.52), xytext=(0.58, 0.64),
                arrowprops=dict(facecolor='#4ade80', edgecolor='#4ade80', width=2, headwidth=8))
                
    # Three Branches
    # 1. Crack Branch
    bbox_crack = dict(boxstyle="round,pad=0.6", fc="#1e3a8a", ec="#60a5fa", lw=2)
    ax.text(0.20, 0.46, "SURFACE CRACKS\n(Classes 0..3)", 
            ha='center', va='center', fontsize=12, fontweight='bold', color='white', bbox=bbox_crack)
    ax.annotate('', xy=(0.20, 0.32), xytext=(0.20, 0.39),
                arrowprops=dict(facecolor='#60a5fa', edgecolor='#60a5fa', width=1.5, headwidth=6))
    crack_desc = ("0: longitudinal_crack (D00)\n"
                  "1: transverse_crack (D10)\n"
                  "2: alligator_crack (D20)\n"
                  "3: repaired_crack (D30)")
    ax.text(0.20, 0.22, crack_desc, ha='center', va='center', fontsize=10, color='#bfdbfe',
            bbox=dict(boxstyle="square,pad=0.5", fc="#0f172a", ec="#3b82f6", lw=1))

    # 2. Pothole Branch
    bbox_pothole = dict(boxstyle="round,pad=0.6", fc="#7c2d12", ec="#f97316", lw=2)
    ax.text(0.50, 0.46, "ROAD CAVITIES\n(Class 4)", 
            ha='center', va='center', fontsize=12, fontweight='bold', color='white', bbox=bbox_pothole)
    ax.annotate('', xy=(0.50, 0.32), xytext=(0.50, 0.39),
                arrowprops=dict(facecolor='#f97316', edgecolor='#f97316', width=1.5, headwidth=6))
    pothole_desc = "4: pothole (D40)\n\n(Severe surface\ndepression & cratering)"
    ax.text(0.50, 0.22, pothole_desc, ha='center', va='center', fontsize=10, color='#fed7aa',
            bbox=dict(boxstyle="square,pad=0.5", fc="#0f172a", ec="#f97316", lw=1))

    # 3. Manhole Condition Branch
    bbox_manhole = dict(boxstyle="round,pad=0.6", fc="#14532d", ec="#4ade80", lw=2)
    ax.text(0.80, 0.46, "MANHOLE CONDITIONS\n(Classes 5..8)", 
            ha='center', va='center', fontsize=12, fontweight='bold', color='white', bbox=bbox_manhole)
    ax.annotate('', xy=(0.80, 0.32), xytext=(0.80, 0.39),
                arrowprops=dict(facecolor='#4ade80', edgecolor='#4ade80', width=1.5, headwidth=6))
    manhole_desc = ("5: manhole_good (Good)\n"
                    "6: manhole_broken (Broken)\n"
                    "7: manhole_loose (Loose)\n"
                    "8: manhole_uncovered (Uncovered)\n"
                    "[D70 Generic Manhole EXCLUDED]")
    ax.text(0.80, 0.22, manhole_desc, ha='center', va='center', fontsize=10, color='#bbf7d0',
            bbox=dict(boxstyle="square,pad=0.5", fc="#0f172a", ec="#22c55e", lw=1))
            
    # Footer note
    ax.text(0.50, 0.05, 
            "Semantic Harmonization Rule: N-RDD2024 D70 (generic manhole) is excluded from training to eliminate cross-entropy conflict.\n"
            "Supplementary dataset provides fine-grained physical state labels (Good / Broken / Loose / Uncovered).",
            ha='center', va='center', fontsize=10, color='#94a3b8', style='italic')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved {out_path}")

if __name__ == '__main__':
    generate_comparison_visual()
    generate_architecture_diagram()
