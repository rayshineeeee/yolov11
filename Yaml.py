import yaml
import os

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
dataset_root = current_dir + "/dataset"

data = {
    "path": dataset_root,
    "train": "images/train",
    "val": "images/val",
    "test": "images/test",  # Optiona

    "nc": 1,  # Number of classes (update accordingly)
    "names": {0: "rock"}  # Class names (update accordingly)
}

# Save as dataset.yaml
yaml_path = os.path.join(dataset_root, "dataset.yaml")
with open(yaml_path, "w") as file:
    yaml.dump(data, file, default_flow_style=False)

print(f"Dataset YAML saved at {yaml_path}")