"""
Harmonization and Dataset Construction Script for Option C (9-Class Taxonomy)
Builds the unified processed dataset:
  - processed/images/
  - processed/labels/
  - processed/source_manifest.csv
Taxonomy:
  0: longitudinal_crack (D00)
  1: transverse_crack (D10)
  2: alligator_crack (D20)
  3: repaired_crack (D30)
  4: pothole (D40)
  5: manhole_good (Good)
  6: manhole_broken (Broken)
  7: manhole_loose (Lose)
  8: manhole_uncovered (Uncovered)
N-RDD D70, D50, D60, D80, D90 are EXCLUDED from training.
"""

import os
import zipfile
import shutil
import xml.etree.ElementTree as ET
import pandas as pd
from PIL import Image

def get_yolo_bbox(xmin, ymin, xmax, ymax, img_w=640.0, img_h=640.0):
    xmin = max(0.0, min(float(xmin), img_w))
    ymin = max(0.0, min(float(ymin), img_h))
    xmax = max(0.0, min(float(xmax), img_w))
    ymax = max(0.0, min(float(ymax), img_h))
    
    if xmax <= xmin or ymax <= ymin:
        return None
        
    xc = (xmin + xmax) / 2.0 / img_w
    yc = (ymin + ymax) / 2.0 / img_h
    w = (xmax - xmin) / img_w
    h = (ymax - ymin) / img_h
    
    # Clip to [0, 1]
    xc = max(0.0, min(1.0, xc))
    yc = max(0.0, min(1.0, yc))
    w = max(0.0, min(1.0, w))
    h = max(0.0, min(1.0, h))
    
    return round(xc, 6), round(yc, 6), round(w, 6), round(h, 6)

# Option C 9-Class Mapping
OPTION_C_MAP = {
    # N-RDD2024 classes
    'D00': ('longitudinal_crack', 0),
    'D10': ('transverse_crack', 1),
    'D20': ('alligator_crack', 2),
    'D30': ('repaired_crack', 3),
    'D40': ('pothole', 4),
    # Supplementary Manhole classes (re-mapped 5..8)
    'Good': ('manhole_good', 5),
    'Broken': ('manhole_broken', 6),
    'Lose': ('manhole_loose', 7),
    'Uncovered': ('manhole_uncovered', 8)
}

SUPP_ID_MAP = {
    0: 'Broken',
    1: 'Good',
    2: 'Lose',
    3: 'Uncovered'
}

def clean_directory(d):
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d, exist_ok=True)

def build_harmonized_dataset(rdd_dir, manhole_dir, output_root='processed'):
    img_out = os.path.join(output_root, 'images')
    lbl_out = os.path.join(output_root, 'labels')
    
    # Clean previous generated directories
    clean_directory(img_out)
    clean_directory(lbl_out)
    clean_directory(os.path.join(output_root, 'train'))
    clean_directory(os.path.join(output_root, 'val'))
    clean_directory(os.path.join(output_root, 'test'))
    
    manifest_rows = []
    
    # 1. Process N-RDD2024 (excluding D70, D50, D60, D80, D90)
    tv_dir = os.path.join(rdd_dir, 'N-RDD2024Road damage and defects', 'Training and Validation Dataset')
    countries = ['Czech Republic', 'USA', 'china-motorbike', 'india', 'japan', 'norway']
    
    print("Processing N-RDD2024 for Option C (D70 excluded from training labels)...")
    nrdd_processed_count = 0
    d70_excluded_count = 0
    
    for country in countries:
        zpath = os.path.join(tv_dir, f'{country}_voc.zip')
        if not os.path.exists(zpath):
            continue
            
        with zipfile.ZipFile(zpath, 'r') as z:
            xml_files = [n for n in z.namelist() if n.endswith('.xml')]
            
            for xf in xml_files:
                base_stem = os.path.splitext(os.path.basename(xf))[0]
                img_f = xf.replace('.xml', '.jpg')
                
                xml_data = z.read(xf)
                root = ET.fromstring(xml_data)
                
                size_node = root.find('size')
                img_w = float(size_node.find('width').text) if size_node is not None else 640.0
                img_h = float(size_node.find('height').text) if size_node is not None else 640.0
                
                objs = root.findall('object')
                in_scope_lines = []
                
                for obj in objs:
                    cname = obj.find('name').text.strip()
                    if cname == 'D70':
                        d70_excluded_count += 1
                        continue
                        
                    if cname in OPTION_C_MAP:
                        bndbox = obj.find('bndbox')
                        xmin = float(bndbox.find('xmin').text)
                        ymin = float(bndbox.find('ymin').text)
                        xmax = float(bndbox.find('xmax').text)
                        ymax = float(bndbox.find('ymax').text)
                        
                        yolo_box = get_yolo_bbox(xmin, ymin, xmax, ymax, img_w, img_h)
                        if yolo_box:
                            final_name, final_id = OPTION_C_MAP[cname]
                            in_scope_lines.append((final_id, yolo_box))
                            manifest_rows.append({
                                'processed_filename': f'nrdd_{base_stem}.jpg',
                                'source_dataset': 'N-RDD2024',
                                'source_filename': os.path.basename(img_f),
                                'original_class': cname,
                                'final_class': final_name,
                                'final_class_id': final_id
                            })
                            
                # If image contains remaining in-scope training hazards, save it
                if len(in_scope_lines) > 0:
                    dest_img = os.path.join(img_out, f'nrdd_{base_stem}.jpg')
                    dest_lbl = os.path.join(lbl_out, f'nrdd_{base_stem}.txt')
                    
                    with open(dest_img, 'wb') as f_out:
                        f_out.write(z.read(img_f))
                        
                    with open(dest_lbl, 'w', encoding='utf-8') as f_out:
                        for cid, (xc, yc, bw, bh) in in_scope_lines:
                            f_out.write(f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
                            
                    nrdd_processed_count += 1
                    
    print(f"N-RDD2024: Saved {nrdd_processed_count} hazard images and labels. (Excluded {d70_excluded_count} D70 boxes)")
    
    # 2. Process Supplementary Manhole Dataset
    print("Processing Supplementary Manhole Dataset with Option C mappings...")
    supp_processed_count = 0
    for split in ['train', 'valid', 'test']:
        img_sdir = os.path.join(manhole_dir, split, 'images')
        lbl_sdir = os.path.join(manhole_dir, split, 'labels')
        if not os.path.exists(lbl_sdir):
            continue
            
        for lf in os.listdir(lbl_sdir):
            if not lf.endswith('.txt'):
                continue
            stem = os.path.splitext(lf)[0]
            img_src = os.path.join(img_sdir, stem + '.jpg')
            if not os.path.exists(img_src):
                continue
                
            with open(os.path.join(lbl_sdir, lf), 'r') as fp:
                lines = [l.strip() for l in fp if l.strip()]
                
            in_scope_lines = []
            for line in lines:
                parts = line.split()
                cid = int(parts[0])
                cname = SUPP_ID_MAP.get(cid)
                if cname in OPTION_C_MAP:
                    final_name, final_id = OPTION_C_MAP[cname]
                    xc, yc, bw, bh = map(float, parts[1:5])
                    xc = max(0.0, min(1.0, xc))
                    yc = max(0.0, min(1.0, yc))
                    bw = max(0.0, min(1.0, bw))
                    bh = max(0.0, min(1.0, bh))
                    in_scope_lines.append((final_id, (xc, yc, bw, bh)))
                    manifest_rows.append({
                        'processed_filename': f'supp_{stem}.jpg',
                        'source_dataset': 'Supplementary_Roboflow',
                        'source_filename': f'{stem}.jpg',
                        'original_class': cname,
                        'final_class': final_name,
                        'final_class_id': final_id
                    })
                    
            if len(in_scope_lines) > 0:
                dest_img = os.path.join(img_out, f'supp_{stem}.jpg')
                dest_lbl = os.path.join(lbl_out, f'supp_{stem}.txt')
                
                shutil.copyfile(img_src, dest_img)
                with open(dest_lbl, 'w', encoding='utf-8') as f_out:
                    for cid, (xc, yc, bw, bh) in in_scope_lines:
                        f_out.write(f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
                supp_processed_count += 1
                
    print(f"Supplementary: Saved {supp_processed_count} condition images and labels.")
    
    # Save Source Manifest
    manifest_df = pd.DataFrame(manifest_rows)
    manifest_path = os.path.join(output_root, 'source_manifest.csv')
    manifest_df.to_csv(manifest_path, index=False)
    print(f"Saved {manifest_path} with {len(manifest_df)} object instances.")
    
    total_imgs = len(os.listdir(img_out))
    total_lbls = len(os.listdir(lbl_out))
    print(f"Total processed images: {total_imgs}, Total processed labels: {total_lbls}")
    return total_imgs, total_lbls, len(manifest_df)

if __name__ == '__main__':
    rdd_dir = r'C:\Users\lucky\Downloads\DL PBL\N-RDD2024Road damage and defects'
    manhole_dir = r'C:\Users\lucky\Downloads\DL PBL\Manhole Cover Dataset YOLO.v4i.yolo26'
    build_harmonized_dataset(rdd_dir, manhole_dir)
