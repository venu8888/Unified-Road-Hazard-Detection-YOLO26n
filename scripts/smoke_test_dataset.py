"""
Dataset Load Smoke Test Script for Option C (9-Class Taxonomy)
Verifies configs/data.yaml, dataset paths, train/val/test images and labels,
loads random batches, and verifies YOLO26n model compatibility without training.
Outputs results/dataset_smoke_test.txt
"""

import os
import yaml
import random
from PIL import Image
from ultralytics import YOLO

def smoke_test(data_yaml='configs/data.yaml', output_file='results/dataset_smoke_test.txt'):
    results = []
    results.append("==================================================")
    results.append("YOLO DATASET LOAD SMOKE TEST REPORT (OPTION C - 9 CLASSES)")
    results.append("==================================================")
    
    with open(data_yaml, 'r') as f:
        data_cfg = yaml.safe_load(f)
        
    results.append(f"Config File: {data_yaml}")
    results.append(f"Root Path: {data_cfg.get('path')}")
    results.append(f"Classes ({data_cfg.get('nc')}): {data_cfg.get('names')}")
    
    root_p = data_cfg.get('path')
    splits = {'train': data_cfg['train'], 'val': data_cfg['val'], 'test': data_cfg['test']}
    
    split_counts = {}
    for s_name, rel_p in splits.items():
        img_p = os.path.join(root_p, rel_p)
        lbl_p = img_p.replace('images', 'labels')
        
        img_files = [f for f in os.listdir(img_p) if f.endswith('.jpg')]
        lbl_files = [f for f in os.listdir(lbl_p) if f.endswith('.txt')]
        split_counts[s_name] = {'images': len(img_files), 'labels': len(lbl_files)}
        
        results.append(f"Split [{s_name}]: {len(img_files)} images, {len(lbl_files)} labels (Path: {img_p})")
        
        # Test load 5 random images and labels
        sampled = random.sample(img_files, 5)
        for sf in sampled:
            sim_p = os.path.join(img_p, sf)
            with Image.open(sim_p) as im:
                w, h = im.size
                assert w == 640 and h == 640
            slbl_p = os.path.join(lbl_p, sf.replace('.jpg', '.txt'))
            with open(slbl_p, 'r') as fp:
                lines = fp.readlines()
                assert len(lines) > 0
                for line in lines:
                    cid = int(line.split()[0])
                    assert 0 <= cid <= 8, f"Invalid class ID {cid}"
                
    results.append("\nRandom Sample Verification: PASSED (images 640x640, valid non-empty labels, class IDs in 0..8)")
    
    # Model compatibility test
    print("Testing YOLO26n instantiation against Option C data.yaml...")
    model = YOLO('yolo26n.pt')
    results.append(f"YOLO26n Model Instantiation: PASSED ({type(model).__name__})")
    results.append("Training Status: STOPPED AS INSTRUCTED (No training performed)")
    results.append("==================================================")
    
    out_text = '\n'.join(results)
    print(out_text)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(out_text)
    print(f"Saved {output_file}")

if __name__ == '__main__':
    smoke_test()
