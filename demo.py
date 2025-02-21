from ultralytics import YOLO
import os
import torch
import psutil


current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
dataset_root = current_dir + "/dataset"

model = YOLO("yolo11n.pt")
results = model(dataset_root + "/images/test/6_00199_l.png")
results[0].show()
