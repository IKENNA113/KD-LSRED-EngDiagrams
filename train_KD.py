

import sys

# Add your preferred path to the start of sys.path
sys.path.insert(0, "C:/Users/ikenn/KD_YOLO")

from ultralytics import YOLO
import os
import numpy as np


# Load the pre-trained YOLOv8l model
model = YOLO('C:/Users/ikenn/KD_YOLO/ultralytics/cfg/models/v8/yolov8.yaml')  # Load the YOLOv8 Large model

# Define the path to your custom dataset .yaml file
dataset_yaml = 'C:/Users/ikenn/KD_YOLO/data_circuit/dataset.yaml'  # Replace with your dataset .yaml file path

# Define hyperparameters for training
train_params = {
    "model":'C:/Users/ikenn/KD_YOLO/yolov8n.pt',  # Pretrained model weights
    'data': dataset_yaml,  # Path to dataset .yaml file
    'epochs': 150,  # Number of training epochs
    'batch': 16,  # Batch size
    'seed': 42,  #  size
    'patience': 20,  #  size
    "freeze": 0,
    'verbose': True,  #  size
    'imgsz': 640,  # Image size
    'device': '0',  # Use GPU (set to 'cpu' if no GPU is available)
    'workers': 0,  # Number of data loading workers
    'optimizer': 'auto',  # Optimizer (auto, SGD, Adam, etc.)
    'lr0': 0.01,  # Initial learning rate
    'momentum': 0.937,  # SGD momentum
    'weight_decay': 0.0005,  # Weight decay
    'warmup_epochs': 3.0,  # Warmup epochs
    'warmup_momentum': 0.8,  # Warmup momentum
    'warmup_bias_lr': 0.1,  # Warmup bias learning rate
    'label_smoothing': 0.0,  # Label smoothing
    'nbs': 64,  # Nominal batch size
    'save': True,  # Save checkpoints
    'save_period': -1,  # Save checkpoint every x epochs
    'cache': False,  # Cache images for faster training
    'resume': False,  # Resume training from last checkpoint
    'amp': True,  # Automatic Mixed Precision (AMP) training
    'project': 'runs/train',  # Save results to this directory
    'name': 'yolov8n_circult',  # Experiment name
    'exist_ok': False,  # Overwrite existing experiment
    'pretrained': True,  # Use pre-trained weights
    'cos_lr': False,  # Use cosine learning rate scheduler
    'close_mosaic': 10,  # Disable mosaic augmentation for last x epochs
    'plots': True,  # Generate plots during training (e.g., loss curves)
}

# Train the model with real-time monitoring
results = model.train(**train_params)

# Test the model on validation data
test_params = {
    'data': dataset_yaml,  # Path to dataset .yaml file
    'split': 'test',  # Use validation data
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
    'name': 'yolov8n_circult_val',  # Experiment name
    'exist_ok': False,  # Overwrite existing experiment
    'plots': True,  # Generate plots (e.g., confusion matrix, PR curve)
    'workers': 0 ,
}

# Validate the model
metrics = model.val(**test_params)

# ✅ Use the correct attributes from the latest YOLOv8 version
print(f"mAP50: {metrics.box.map50:.4f}")  # Mean Average Precision at IoU 0.5
print(f"mAP50-95: {metrics.box.map:.4f}")  # Mean Average Precision at IoU 0.5-0.95
print(f"Precision: {metrics.box.mp:.4f}")  # Mean Precision
print(f"Recall: {metrics.box.mr:.4f}")  # Mean Recall

# Launch TensorBoard for real-time monitoring
os.system('tensorboard --logdir runs/train')

