from ultralytics import YOLO
import os
import torch
import psutil


current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
dataset_root = current_dir + "/dataset"

model = YOLO(current_dir + "/best.pt")
# results = model.predict(dataset_root + "/images/test/6_00001_l.png")
results = model.predict(source=dataset_root + "/images/test", save=True)


# display the first result

import matplotlib.pyplot as plt

result_image = results[0].plot()
plt.imshow(result_image)
plt.axis("off")
plt.show()
