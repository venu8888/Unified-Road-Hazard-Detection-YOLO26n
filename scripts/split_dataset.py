"""
Dataset Split Script for Option C (9-Class Taxonomy)
Splits exactly into 70% Train, 20% Validation, 10% Test with seed=42.
Outputs:
  - processed/train/images/, processed/train/labels/
  - processed/val/images/, processed/val/labels/
  - processed/test/images/, processed/test/labels/
  - results/split_summary.csv
  - results/train_val_test_distribution.png
"""

import os
import shutil
import random
from collections import Counter, defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

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

def perform_split(processed_root='processed', results_root='results', seed=42):
    random.seed(seed)
    np.random.seed(seed)
    
    img_dir = os.path.join(processed_root, 'images')
    lbl_dir = os.path.join(processed_root, 'labels')
    
    split_dirs = {
        'train': {
            'images': os.path.join(processed_root, 'train', 'images'),
            'labels': os.path.join(processed_root, 'train', 'labels')
        },
        'val': {
            'images': os.path.join(processed_root, 'val', 'images'),
            'labels': os.path.join(processed_root, 'val', 'labels')
        },
        'test': {
            'images': os.path.join(processed_root, 'test', 'images'),
            'labels': os.path.join(processed_root, 'test', 'labels')
        }
    }
    
    for s in split_dirs:
        if os.path.exists(os.path.join(processed_root, s)):
            shutil.rmtree(os.path.join(processed_root, s))
        os.makedirs(split_dirs[s]['images'], exist_ok=True)
        os.makedirs(split_dirs[s]['labels'], exist_ok=True)
        
    all_images = sorted(os.listdir(img_dir))
    total_imgs = len(all_images)
    print(f"Total Option C images to split: {total_imgs}")
    
    image_metadata = []
    for img_name in all_images:
        stem = os.path.splitext(img_name)[0]
        lbl_name = stem + '.txt'
        lbl_path = os.path.join(lbl_dir, lbl_name)
        
        cids = []
        if os.path.exists(lbl_path):
            with open(lbl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        cids.append(int(parts[0]))
        strat_key = min(cids, key=lambda x: (x in [3, 6, 7, 8], x)) if cids else -1
        prefix = 'nrdd' if img_name.startswith('nrdd_') else 'supp'
        image_metadata.append({
            'img_name': img_name,
            'stem': stem,
            'strat_key': f"{prefix}_{strat_key}",
            'classes': cids
        })
        
    df_meta = pd.DataFrame(image_metadata)
    
    # 70% Train, 30% Temp
    train_df, temp_df = train_test_split(
        df_meta, 
        test_size=0.30, 
        random_state=seed, 
        stratify=df_meta['strat_key']
    )
    
    # From 30% Temp -> 2/3 for Val (20% total), 1/3 for Test (10% total)
    val_df, test_df = train_test_split(
        temp_df, 
        test_size=1.0/3.0, 
        random_state=seed, 
        stratify=temp_df['strat_key']
    )
    
    print(f"Exact Split counts: Train={len(train_df)} ({len(train_df)/total_imgs*100:.1f}%), "
          f"Val={len(val_df)} ({len(val_df)/total_imgs*100:.1f}%), "
          f"Test={len(test_df)} ({len(test_df)/total_imgs*100:.1f}%)")
          
    split_assignments = {
        'train': train_df,
        'val': val_df,
        'test': test_df
    }
    
    class_tallies = {s: Counter() for s in split_assignments}
    split_summary_records = []
    
    for split_name, df_sub in split_assignments.items():
        print(f"Placing {len(df_sub)} files into {split_name}...")
        for _, row in df_sub.iterrows():
            img_src = os.path.join(img_dir, row['img_name'])
            lbl_src = os.path.join(lbl_dir, row['stem'] + '.txt')
            
            img_dst = os.path.join(split_dirs[split_name]['images'], row['img_name'])
            lbl_dst = os.path.join(split_dirs[split_name]['labels'], row['stem'] + '.txt')
            
            shutil.copyfile(img_src, img_dst)
            if os.path.exists(lbl_src):
                shutil.copyfile(lbl_src, lbl_dst)
                
            for cid in row['classes']:
                class_tallies[split_name][cid] += 1
                
    num_classes = len(CLASS_NAMES)
    for cid in range(num_classes):
        cname = CLASS_NAMES[cid]
        tr_cnt = class_tallies['train'][cid]
        val_cnt = class_tallies['val'][cid]
        tst_cnt = class_tallies['test'][cid]
        tot = tr_cnt + val_cnt + tst_cnt
        split_summary_records.append({
            'class_id': cid,
            'class_name': cname,
            'train_boxes': tr_cnt,
            'val_boxes': val_cnt,
            'test_boxes': tst_cnt,
            'total_boxes': tot,
            'train_pct': round((tr_cnt / tot * 100) if tot else 0, 2),
            'val_pct': round((val_cnt / tot * 100) if tot else 0, 2),
            'test_pct': round((tst_cnt / tot * 100) if tot else 0, 2)
        })
        
    df_summary = pd.DataFrame(split_summary_records)
    summary_csv = os.path.join(results_root, 'split_summary.csv')
    df_summary.to_csv(summary_csv, index=False)
    print(f"Saved {summary_csv}")
    
    # Plot Split Distribution
    plt.figure(figsize=(14, 7))
    x = np.arange(num_classes)
    width = 0.25
    
    tr_vals = [df_summary.loc[df_summary['class_id'] == cid, 'train_boxes'].values[0] for cid in range(num_classes)]
    val_vals = [df_summary.loc[df_summary['class_id'] == cid, 'val_boxes'].values[0] for cid in range(num_classes)]
    tst_vals = [df_summary.loc[df_summary['class_id'] == cid, 'test_boxes'].values[0] for cid in range(num_classes)]
    
    plt.bar(x - width, tr_vals, width, label='Train (70%)', color='#2563eb', edgecolor='black')
    plt.bar(x, val_vals, width, label='Val (20%)', color='#16a34a', edgecolor='black')
    plt.bar(x + width, tst_vals, width, label='Test (10%)', color='#f59e0b', edgecolor='black')
    
    plt.xlabel('Target Class ID & Name', fontsize=12)
    plt.ylabel('Instance Count (Bounding Boxes)', fontsize=12)
    plt.title('Option C 70/20/10 Dataset Split Distribution Across All 9 Classes', fontsize=14, pad=12)
    plt.xticks(x, [f"{i}: {CLASS_NAMES[i]}" for i in range(num_classes)], rotation=35, ha='right', fontsize=9)
    plt.legend(fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    dist_plot = os.path.join(results_root, 'train_val_test_distribution.png')
    plt.savefig(dist_plot, dpi=300)
    plt.close()
    print(f"Saved split distribution plot: {dist_plot}")
    
    return len(train_df), len(val_df), len(test_df)

if __name__ == '__main__':
    perform_split()
