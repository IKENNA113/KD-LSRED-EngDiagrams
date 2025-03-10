import sys
from ultralytics import YOLO
import os
import numpy as np

# Ensure the script runs only in the main process (Windows Fix)
if __name__ == "__main__":

    # Load trained model
    model = YOLO("C:/Users/ikenn/KD_YOLO/runs/train/yolov8l_custom/weights/best.pt")

    dataset_yaml = 'C:/Users/ikenn/KD_YOLO/private1_data/dataset.yaml'  # Dataset YAML file path

    # Define validation parameters
    test_params = {
        'data': dataset_yaml,  # Path to dataset .yaml file
        'split': 'val',  # Use validation data
        'show': True,
        'batch': 16,  # Batch size
        'imgsz': 640,  # Image size
        'conf': 0.25,  # Confidence threshold
        'iou': 0.45,  # IoU threshold for NMS
        'device': '0',  # Use GPU (set to 'cpu' if no GPU is available)
        'save_json': False,  # Save results to JSON file
        'save_txt': True,  # Save results to text file
        'save_conf': True,  # Save confidence scores
        'save_crop': False,  # Save cropped images
        'project': 'runs/val',  # Save results to this directory
        'name': 'yolov8l_custom_val2',  # Experiment name
        'exist_ok': False,  # Overwrite existing experiment
        'plots': True,  # Generate plots (e.g., confusion matrix, PR curve)
        'workers': 0  # 🚀 **Fix multiprocessing issue by setting workers to 0**
    }

    # Run validation
    metrics = model.val(**test_params)

    # ✅ Use the correct attributes from the latest YOLOv8 version
    print(f"mAP50: {metrics.box.map50:.4f}")  # Mean Average Precision at IoU 0.5
    print(f"mAP50-95: {metrics.box.map:.4f}")  # Mean Average Precision at IoU 0.5-0.95
    print(f"Precision: {metrics.box.mp:.4f}")  # Mean Precision
    print(f"Recall: {metrics.box.mr:.4f}")  # Mean Recall

    # Start TensorBoard
    os.system('tensorboard --logdir runs/train')




