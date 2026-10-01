"""
YOLO26n Training Script for Unified Road Hazard Detection (Option C - 9 Classes)
Configured with workers=0 to prevent Windows WinError 1455 paging memory allocation failures.
"""

import os
import shutil
from ultralytics import YOLO

def main():
    # Load pretrained YOLO26n base weights
    model = YOLO("yolo26n.pt")

    # Clean incomplete previous run directory if it exists without weights
    partial_run_dir = os.path.join("runs", "detect", "runs", "train", "yolo26n_unified_hazard_optc")
    if os.path.exists(partial_run_dir):
        weights_dir = os.path.join(partial_run_dir, "weights")
        if not os.path.exists(weights_dir) or len(os.listdir(weights_dir)) == 0:
            print(f"Cleaning incomplete run directory: {partial_run_dir}")
            try:
                shutil.rmtree(partial_run_dir)
            except Exception as e:
                print(f"Notice: {e}")

    # Start training with configured parameters
    # Setting workers=0 prevents Windows multiprocessing from spawning new Python processes
    # that redundantly reload CUDA DLLs (cublas64_13.dll) and exhaust the pagefile.
    results = model.train(
        data="configs/data.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,           # NVIDIA GeForce RTX 4050 Laptop GPU
        workers=0,          # Critical for Windows stability
        project="runs/detect/runs/train",
        name="yolo26n_unified_hazard_optc",
        exist_ok=True,
        # Augmentations from configs/augmentation_config.yaml
        mosaic=1.0,
        scale=0.5,
        fliplr=0.5,
        flipud=0.0,
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.4,
        mixup=0.1,
        degrees=0.0,
        shear=0.0,
        save=True,
        save_period=10,
        val=True,
        plots=True
    )
    print("Training process finished.")
    return results

if __name__ == "__main__":
    main()
