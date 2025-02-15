import os

# Paths
image_folder = "/Users/raysmacbookair/dataset/images/train"
mask_folder = "/Users/raysmacbookair/dataset/masks/train"

# Get filenames without extensions
image_files = {os.path.splitext(f)[0] for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png'))}
mask_files = {os.path.splitext(f)[0] for f in os.listdir(mask_folder) if f.endswith('.png')}  # Masks should be PNG

# Find mismatches
missing_masks = image_files - mask_files
missing_images = mask_files - image_files

# Print results
if missing_masks:
    print(f"⚠️ Missing masks for images: {missing_masks}")
if missing_images:
    print(f"⚠️ Missing images for masks: {missing_images}")

if not missing_masks and not missing_images:
    print("✅ All images and masks match correctly!")