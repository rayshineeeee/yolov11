from ultralytics import YOLO
import os

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
dataset_root = current_dir + "/dataset"

model = YOLO(current_dir+"/yolo11n_custom.pt")

results = model(dataset_root + "/images/test/6_00199_l.png") 

# # Access the results
# for result in results:
#     xy = result.masks.xy  # mask in polygon format
#     xyn = result.masks.xyn  # normalized
#     masks = result.masks.data  # mask in matrix format (num_objects x H x W)
results[0].show()  # display to screen
