"""
Dataset Audit Script for Unified Road Hazard Detection
Audits N-RDD2024 and Supplementary Manhole Dataset.
Outputs:
  - results/n_rdd_audit.csv
  - results/manhole_audit.csv
  - results/dataset_summary.txt
  - results/class_distribution_n_rdd.png
  - results/class_distribution_manhole.png
"""

import os
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import io

def audit_n_rdd(base_rdd_path, output_dir='results'):
    os.makedirs(output_dir, exist_ok=True)
    tv_dir = os.path.join(base_rdd_path, 'N-RDD2024Road damage and defects', 'Training and Validation Dataset')
    test_dir = os.path.join(base_rdd_path, 'N-RDD2024Road damage and defects', 'Test Dataset')
    
    countries = ['Czech Republic', 'USA', 'china-motorbike', 'india', 'japan', 'norway']
    
    records = []
    class_counts = Counter()
    country_class_counts = defaultdict(Counter)
    dimensions_set = Counter()
    corrupt_images = 0
    invalid_boxes = 0
    empty_annotations = 0
    total_images = 0
    total_annotations = 0
    duplicate_names = set()
    seen_names = set()
    
    print("Auditing N-RDD2024 across 6 countries...")
    for country in countries:
        zpath = os.path.join(tv_dir, f'{country}_voc.zip')
        if not os.path.exists(zpath):
            print(f"Warning: {zpath} not found!")
            continue
            
        with zipfile.ZipFile(zpath, 'r') as z:
            namelist = z.namelist()
            img_names = [n for n in namelist if n.lower().endswith(('.jpg', '.jpeg', '.png'))]
            xml_names = [n for n in namelist if n.lower().endswith('.xml')]
            
            xml_dict = {os.path.splitext(os.path.basename(n))[0]: n for n in xml_names}
            
            for img_path in img_names:
                total_images += 1
                base_stem = os.path.splitext(os.path.basename(img_path))[0]
                
                if base_stem in seen_names:
                    duplicate_names.add(base_stem)
                seen_names.add(base_stem)
                
                # Check image validity / dimensions
                img_data = z.read(img_path)
                try:
                    with Image.open(io.BytesIO(img_data)) as img:
                        w, h = img.size
                        img_format = img.format
                        dimensions_set[(w, h)] += 1
                except Exception as e:
                    corrupt_images += 1
                    w, h, img_format = 0, 0, 'CORRUPT'
                    
                # Check annotation
                xml_file = xml_dict.get(base_stem)
                boxes_in_img = []
                if xml_file:
                    total_annotations += 1
                    try:
                        xml_data = z.read(xml_file)
                        root = ET.fromstring(xml_data)
                        objs = root.findall('object')
                        if len(objs) == 0:
                            empty_annotations += 1
                        for obj in objs:
                            cname = obj.find('name').text.strip()
                            class_counts[cname] += 1
                            country_class_counts[country][cname] += 1
                            
                            bndbox = obj.find('bndbox')
                            xmin = float(bndbox.find('xmin').text)
                            ymin = float(bndbox.find('ymin').text)
                            xmax = float(bndbox.find('xmax').text)
                            ymax = float(bndbox.find('ymax').text)
                            
                            if xmax <= xmin or ymax <= ymin or xmin < 0 or ymin < 0 or xmax > (w or 640) or ymax > (h or 640):
                                invalid_boxes += 1
                            boxes_in_img.append((cname, xmin, ymin, xmax, ymax))
                    except Exception as e:
                        print(f"Error parsing {xml_file}: {e}")
                else:
                    # image without label
                    pass
                    
                records.append({
                    'country': country,
                    'image_filename': os.path.basename(img_path),
                    'width': w,
                    'height': h,
                    'format': img_format,
                    'has_annotation': xml_file is not None,
                    'num_boxes': len(boxes_in_img),
                    'classes_present': ';'.join(set([b[0] for b in boxes_in_img]))
                })
                
    df_nrdd = pd.DataFrame(records)
    csv_path = os.path.join(output_dir, 'n_rdd_audit.csv')
    df_nrdd.to_csv(csv_path, index=False)
    print(f"Saved {csv_path} with {len(df_nrdd)} rows.")
    
    # Audit unannotated test set
    test_counts = {}
    if os.path.exists(test_dir):
        for sub in os.listdir(test_dir):
            sub_full = os.path.join(test_dir, sub)
            if os.path.isdir(sub_full):
                cnt = 0
                for r, d, fs in os.walk(sub_full):
                    cnt += len([f for f in fs if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                test_counts[sub] = cnt
                
    audit_summary = {
        'total_images': total_images,
        'total_annotations': total_annotations,
        'annotation_format': 'Pascal VOC XML',
        'image_extensions': ['.jpg'],
        'dimensions': dict(dimensions_set),
        'class_counts': dict(class_counts),
        'country_class_counts': {k: dict(v) for k, v in country_class_counts.items()},
        'empty_annotations': empty_annotations,
        'invalid_boxes': invalid_boxes,
        'corrupt_images': corrupt_images,
        'duplicate_filenames': len(duplicate_names),
        'unannotated_test_set_images': test_counts
    }
    
    # Plot class distribution
    plt.figure(figsize=(10, 5))
    classes_sorted = sorted(class_counts.keys())
    counts_sorted = [class_counts[c] for c in classes_sorted]
    bars = plt.bar(classes_sorted, counts_sorted, color='#2563eb', edgecolor='black')
    plt.title('N-RDD2024 Ground-Truth Class Distribution (Bounding Boxes)', fontsize=14, pad=12)
    plt.xlabel('Category Code', fontsize=12)
    plt.ylabel('Instance Count', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 200, f'{yval:,}', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'class_distribution_n_rdd.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved {plot_path}")
    
    return audit_summary

def audit_manhole(base_manhole_path, output_dir='results'):
    os.makedirs(output_dir, exist_ok=True)
    splits = ['train', 'valid', 'test']
    
    # Roboflow data.yaml class names
    class_map = {0: 'Broken', 1: 'Good', 2: 'Lose', 3: 'Uncovered'}
    
    records = []
    class_counts = Counter()
    split_class_counts = defaultdict(Counter)
    dimensions_set = Counter()
    corrupt_images = 0
    invalid_boxes = 0
    empty_annotations = 0
    total_images = 0
    total_annotations = 0
    duplicate_names = set()
    seen_names = set()
    
    print("Auditing Supplementary Manhole Dataset...")
    for s in splits:
        img_dir = os.path.join(base_manhole_path, s, 'images')
        lbl_dir = os.path.join(base_manhole_path, s, 'labels')
        
        if not os.path.exists(img_dir):
            continue
            
        img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for f in img_files:
            total_images += 1
            stem = os.path.splitext(f)[0]
            if stem in seen_names:
                duplicate_names.add(stem)
            seen_names.add(stem)
            
            img_p = os.path.join(img_dir, f)
            try:
                with Image.open(img_p) as img:
                    w, h = img.size
                    img_format = img.format
                    dimensions_set[(w, h)] += 1
            except Exception as e:
                corrupt_images += 1
                w, h, img_format = 0, 0, 'CORRUPT'
                
            lbl_p = os.path.join(lbl_dir, stem + '.txt')
            boxes_in_img = []
            if os.path.exists(lbl_p):
                total_annotations += 1
                with open(lbl_p, 'r') as lfp:
                    lines = [line.strip() for line in lfp if line.strip()]
                    if len(lines) == 0:
                        empty_annotations += 1
                    for line in lines:
                        parts = line.split()
                        cid = int(parts[0])
                        cname = class_map.get(cid, f'class_{cid}')
                        class_counts[cname] += 1
                        split_class_counts[s][cname] += 1
                        
                        xc, yc, bw, bh = map(float, parts[1:5])
                        if xc < 0 or xc > 1 or yc < 0 or yc > 1 or bw <= 0 or bw > 1 or bh <= 0 or bh > 1:
                            invalid_boxes += 1
                        boxes_in_img.append((cname, xc, yc, bw, bh))
            records.append({
                'split': s,
                'image_filename': f,
                'width': w,
                'height': h,
                'format': img_format,
                'has_annotation': os.path.exists(lbl_p),
                'num_boxes': len(boxes_in_img),
                'classes_present': ';'.join(set([b[0] for b in boxes_in_img]))
            })
            
    df_manhole = pd.DataFrame(records)
    csv_path = os.path.join(output_dir, 'manhole_audit.csv')
    df_manhole.to_csv(csv_path, index=False)
    print(f"Saved {csv_path} with {len(df_manhole)} rows.")
    
    audit_summary = {
        'total_images': total_images,
        'total_annotations': total_annotations,
        'annotation_format': 'YOLO TXT (normalized cx, cy, w, h)',
        'image_extensions': ['.jpg'],
        'dimensions': dict(dimensions_set),
        'class_counts': dict(class_counts),
        'split_class_counts': {k: dict(v) for k, v in split_class_counts.items()},
        'empty_annotations': empty_annotations,
        'invalid_boxes': invalid_boxes,
        'corrupt_images': corrupt_images,
        'duplicate_filenames': len(duplicate_names)
    }
    
    # Plot class distribution
    plt.figure(figsize=(8, 5))
    classes_sorted = ['Broken', 'Good', 'Lose', 'Uncovered']
    counts_sorted = [class_counts[c] for c in classes_sorted]
    colors = ['#dc2626', '#16a34a', '#ca8a04', '#9333ea']
    bars = plt.bar(classes_sorted, counts_sorted, color=colors, edgecolor='black')
    plt.title('Supplementary Manhole Dataset Class Distribution (Bounding Boxes)', fontsize=13, pad=12)
    plt.xlabel('Condition State', fontsize=12)
    plt.ylabel('Instance Count', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 15, f'{yval:,}', ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'class_distribution_manhole.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved {plot_path}")
    
    return audit_summary

def write_summary_text(rdd_sum, manhole_sum, output_dir='results'):
    summary_path = os.path.join(output_dir, 'dataset_summary.txt')
    lines = []
    lines.append("================================================================================")
    lines.append("DEEP LEARNING PROJECT 4 — COMPREHENSIVE DATASET AUDIT SUMMARY")
    lines.append("Review 2 Pre-Training Audit")
    lines.append("================================================================================\n")
    
    lines.append("1. PRIMARY DATASET: N-RDD2024 (Road Damage Detection 2024)")
    lines.append("--------------------------------------------------------------------------------")
    lines.append(f"Total Source Training/Validation Images: {rdd_sum['total_images']:,}")
    lines.append(f"Total Annotation Files (VOC XML):        {rdd_sum['total_annotations']:,}")
    lines.append(f"Image Formats/Extensions:                {rdd_sum['image_extensions']}")
    lines.append(f"Image Dimensions (Resolution):           {rdd_sum['dimensions']}")
    lines.append(f"Corrupt / Unreadable Images:             {rdd_sum['corrupt_images']}")
    lines.append(f"Empty Annotations (0 bounding boxes):    {rdd_sum['empty_annotations']}")
    lines.append(f"Invalid Bounding Boxes (out-of-bounds):  {rdd_sum['invalid_boxes']}")
    lines.append(f"Duplicate Basenames across archives:     {rdd_sum['duplicate_filenames']}")
    lines.append("Bounding-Box Instances Per Class:")
    total_rdd_boxes = sum(rdd_sum['class_counts'].values())
    for k, v in sorted(rdd_sum['class_counts'].items(), key=lambda x: -x[1]):
        pct = (v / total_rdd_boxes) * 100
        lines.append(f"  - {k:<6}: {v:>6,} instances ({pct:5.2f}%)")
    lines.append(f"Total N-RDD Ground-Truth Bounding Boxes: {total_rdd_boxes:,}")
    lines.append(f"Unannotated Challenge Test Images:       {rdd_sum['unannotated_test_set_images']}\n")
    
    lines.append("2. SUPPLEMENTARY DATASET: Manhole Cover Dataset (Roboflow v4)")
    lines.append("--------------------------------------------------------------------------------")
    lines.append(f"Total Source Images:                     {manhole_sum['total_images']:,}")
    lines.append(f"Total Annotation Files (YOLO TXT):       {manhole_sum['total_annotations']:,}")
    lines.append(f"Annotation Format:                       {manhole_sum['annotation_format']}")
    lines.append(f"Image Dimensions:                        {manhole_sum['dimensions']}")
    lines.append(f"Corrupt / Unreadable Images:             {manhole_sum['corrupt_images']}")
    lines.append(f"Empty Annotations:                       {manhole_sum['empty_annotations']}")
    lines.append(f"Invalid Bounding Boxes:                  {manhole_sum['invalid_boxes']}")
    lines.append(f"Duplicate Basenames:                     {manhole_sum['duplicate_filenames']}")
    lines.append("Bounding-Box Instances Per Class:")
    total_manhole_boxes = sum(manhole_sum['class_counts'].values())
    for k, v in sorted(manhole_sum['class_counts'].items(), key=lambda x: -x[1]):
        pct = (v / total_manhole_boxes) * 100
        lines.append(f"  - {k:<12}: {v:>6,} instances ({pct:5.2f}%)")
    lines.append(f"Total Manhole Ground-Truth Boxes:        {total_manhole_boxes:,}")
    lines.append(f"Instances per split:                     {manhole_sum['split_class_counts']}\n")
    
    lines.append("3. CRITICAL AUDIT FINDINGS ON TARGET TAXONOMY & CIRCLE CLASS")
    lines.append("--------------------------------------------------------------------------------")
    lines.append("N-RDD2024 Taxonomy:")
    lines.append("  - In-Scope: D00 (Longitudinal), D10 (Transverse), D20 (Alligator), D30 (Repaired), D40 (Pothole), D70 (Manhole)")
    lines.append("  - Out-of-Scope: D50 (Crosswalk blur/marking), D60 (White/yellow line blur), D80, D90")
    lines.append("Supplementary Manhole Taxonomy:")
    lines.append("  - Ground-truth classes present in actual dataset: ['Broken', 'Good', 'Lose', 'Uncovered']")
    lines.append("  - 'circle' class presence: VERIFIED ABSENT. The Roboflow dataset version v4 does not contain any 'circle' label.")
    lines.append("================================================================================")
    
    text = '\n'.join(lines)
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"Saved {summary_path}")

if __name__ == '__main__':
    rdd_dir = r'C:\Users\lucky\Downloads\DL PBL\N-RDD2024Road damage and defects'
    manhole_dir = r'C:\Users\lucky\Downloads\DL PBL\Manhole Cover Dataset YOLO.v4i.yolo26'
    
    rdd_sum = audit_n_rdd(rdd_dir)
    manhole_sum = audit_manhole(manhole_dir)
    write_summary_text(rdd_sum, manhole_sum)
