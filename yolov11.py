from ultralytics import YOLO
import os
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
dataset_root = current_dir + "/dataset"
# Load a model
model = YOLO("yolo11n-seg.yaml")

# Train the model
train_results = model.train(
    workers=4,
    data=dataset_root + "/dataset.yaml",  # path to dataset YAML
    epochs=50,  # number of training epochs
    imgsz=1024,  # training image size
    device= "mps",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
)

# Evaluate model performance on the validation set
metrics = model.val()

# Perform object detection on an image
results = model(dataset_root + "/images/testing/6_00001_l.jpg")
results[0].show()

# Export the model to ONNX format
path = model.export(format="onnx")  # return path to exported model