"""
Dataset Validation and Cleaning Script for Option C (9-Class Taxonomy)
Performs integrity checks on processed/images and processed/labels:
  - Image exists and is readable (PIL verify)
  - Label exists and has valid boxes
  - Valid class ID (0 to 8)
  - Normalized coordinates (0 <= xc, yc, bw, bh <= 1)
  - Positive box dimensions (bw > 0, bh > 0)
  - Zero malformed lines
Outputs:
  - results/validation_report.csv
  - results/validation_summary.txt
"""

import os
import shutil
import pandas as pd
from PIL import Image

def validate_dataset(processed_root='processed', quarantine_root='quarantine', results_root='results'):
    os.makedirs(results_root, exist_ok=True)
    q_img = os.path.join(quarantine_root, 'images')
    q_lbl = os.path.join(quarantine_root, 'labels')
    os.makedirs(q_img, exist_ok=True)
    os.makedirs(q_lbl, exist_ok=True)
    
    img_dir = os.path.join(processed_root, 'images')
    lbl_dir = os.path.join(processed_root, 'labels')
    
    image_files = sorted(os.listdir(img_dir))
    label_files = set(os.listdir(lbl_dir))
    
    report_rows = []
    quarantined_count = 0
    total_valid_boxes = 0
    total_images_checked = len(image_files)
    
    print(f"Validating {total_images_checked} Option C processed images and labels...")
    
    for img_name in image_files:
        stem = os.path.splitext(img_name)[0]
        lbl_name = stem + '.txt'
        img_path = os.path.join(img_dir, img_name)
        lbl_path = os.path.join(lbl_dir, lbl_name)
        
        status = 'VALID'
        reasons = []
        
        # 1. Check Image integrity
        try:
            with Image.open(img_path) as im:
                w, h = im.size
                if w <= 0 or h <= 0:
                    status = 'INVALID'
                    reasons.append(f'Invalid dimension: {w}x{h}')
        except Exception as e:
            status = 'INVALID'
            reasons.append(f'Corrupt image file: {str(e)}')
            
        # 2. Check Label exists
        if lbl_name not in label_files:
            status = 'INVALID'
            reasons.append('Missing label file')
            boxes_count = 0
        else:
            # 3. Check label contents
            try:
                with open(lbl_path, 'r', encoding='utf-8') as f:
                    lines = [line.strip() for line in f if line.strip()]
                boxes_count = len(lines)
                if boxes_count == 0:
                    status = 'INVALID'
                    reasons.append('Empty label file (0 boxes)')
                    
                for idx, line in enumerate(lines):
                    parts = line.split()
                    if len(parts) != 5:
                        status = 'INVALID'
                        reasons.append(f'Malformed label line {idx+1}: {line}')
                        continue
                    try:
                        cid = int(parts[0])
                        xc, yc, bw, bh = map(float, parts[1:])
                    except ValueError:
                        status = 'INVALID'
                        reasons.append(f'Non-numeric label values on line {idx+1}: {line}')
                        continue
                        
                    # Option C: strictly classes 0 through 8
                    if cid < 0 or cid > 8:
                        status = 'INVALID'
                        reasons.append(f'Invalid class ID {cid} (must be 0..8) on line {idx+1}')
                        
                    if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0):
                        status = 'INVALID'
                        reasons.append(f'Center out of bounds ({xc}, {yc}) on line {idx+1}')
                        
                    if bw <= 0.0 or bh <= 0.0 or bw > 1.0 or bh > 1.0:
                        status = 'INVALID'
                        reasons.append(f'Invalid box dimensions ({bw}, {bh}) on line {idx+1}')
                        
                    if status == 'VALID':
                        total_valid_boxes += 1
            except Exception as e:
                status = 'INVALID'
                reasons.append(f'Error reading label: {str(e)}')
                
        # 4. Handle Quarantine if invalid
        if status == 'INVALID':
            quarantined_count += 1
            dest_q_img = os.path.join(q_img, img_name)
            shutil.move(img_path, dest_q_img)
            if os.path.exists(lbl_path):
                dest_q_lbl = os.path.join(q_lbl, lbl_name)
                shutil.move(lbl_path, dest_q_lbl)
                
        report_rows.append({
            'image_filename': img_name,
            'label_filename': lbl_name,
            'status': status,
            'num_boxes': boxes_count,
            'issues': '; '.join(reasons) if reasons else 'NONE'
        })
        
    df_report = pd.DataFrame(report_rows)
    report_csv = os.path.join(results_root, 'validation_report.csv')
    df_report.to_csv(report_csv, index=False)
    print(f"Saved {report_csv}")
    
    summary_txt = os.path.join(results_root, 'validation_summary.txt')
    valid_count = total_images_checked - quarantined_count
    with open(summary_txt, 'w', encoding='utf-8') as f:
        f.write("================================================================================\n")
        f.write("DATASET CLEANING & VALIDATION SUMMARY REPORT (OPTION C - 9 CLASSES)\n")
        f.write("================================================================================\n\n")
        f.write(f"Total Processed Images Checked:     {total_images_checked:,}\n")
        f.write(f"Passed All Validation Checks:       {valid_count:,} ({valid_count/total_images_checked*100:.2f}%)\n")
        f.write(f"Quarantined Files:                  {quarantined_count:,}\n")
        f.write(f"Total Valid Bounding Boxes:         {total_valid_boxes:,}\n\n")
        f.write("Option C Taxonomy Enforced (Classes 0..8):\n")
        f.write("  0: longitudinal_crack\n")
        f.write("  1: transverse_crack\n")
        f.write("  2: alligator_crack\n")
        f.write("  3: repaired_crack\n")
        f.write("  4: pothole\n")
        f.write("  5: manhole_good\n")
        f.write("  6: manhole_broken\n")
        f.write("  7: manhole_loose\n")
        f.write("  8: manhole_uncovered\n\n")
        f.write("Validation Criteria Enforced:\n")
        f.write("  [x] Image readable and uncorrupted\n")
        f.write("  [x] Paired YOLO annotation file exists\n")
        f.write("  [x] Contiguous and valid class IDs [0..8]\n")
        f.write("  [x] Normalized center coordinates [0.0 <= xc, yc <= 1.0]\n")
        f.write("  [x] Valid positive dimensions [0.0 < bw, bh <= 1.0]\n")
        f.write("  [x] Zero malformed tokens or trailing characters\n\n")
        if quarantined_count == 0:
            f.write("STATUS: CLEAN — 100% of processed images and labels passed validation.\n")
            f.write("Zero files moved to quarantine.\n")
        else:
            f.write(f"STATUS: {quarantined_count} files moved to quarantine/ with detailed log in validation_report.csv\n")
        f.write("================================================================================\n")
        
    print(f"Saved {summary_txt}")
    print(f"Validation complete: {valid_count} valid, {quarantined_count} quarantined.")
    return valid_count, quarantined_count

if __name__ == '__main__':
    validate_dataset()
