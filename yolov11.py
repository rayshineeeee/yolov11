from ultralytics import YOLO
import os
import torch
import psutil
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
dataset_root = current_dir + "/dataset"
# Load a model
model = YOLO("yolo11n-seg.yaml")

# Display the model architecture

# Check GPU memory usage
if torch.cuda.is_available():
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # Convert to GB
    print(f"Total GPU memory: {gpu_memory:.2f} GB")
    
# Check system RAM
ram_memory = psutil.virtual_memory().total / 1024**3  # Convert to GB
print(f"Total system RAM: {ram_memory:.2f} GB")

# Train the model
train_results = model.train(
    workers=8,
    data=dataset_root + "/dataset.yaml",  # path to dataset YAML
    epochs=150,  # number of training epochs
    imgsz=1024,  # training image size
    batch=32,
    device= "cuda",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
)

# Evaluate model performance on the validation set
metrics = model.val()

# Perform object detection on an image
# results = model(dataset_root + "/images/test/6_00001_l.jpg")
# results[0].show()

# Export the model to ONNX format
path = model.export(format="onnx")  # return path to exported model