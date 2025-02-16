import os
import cv2
import numpy as np
import glob

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
dataset_root = current_dir + "/dataset"

# =======================
# Configuration variables
# =======================
APPROX_FACTOR = 0.002                  # Lower value = more sensitive polygon approximation
MIN_CONTOUR_AREA = 100                  # Ignore contours with an area smaller than this (in pixels)
IMG_WIDTH, IMG_HEIGHT = 1024, 1024       # Dimensions of the mask images

INPUT_FOLDER = dataset_root + "/masks/test"          # Folder containing mask images (e.g., PNG files)
OUTPUT_FOLDER = dataset_root + "/labels/test"         # Folder to save the TXT files and generated mask images

def fill_polygons(polygons, width, height):
    """
    Creates a blank mask image with a black background and fills the provided polygons in white.

    Args:
        polygons (list): List of numpy arrays representing polygons (each with shape (n_points, 1, 2)).
        width (int): Width of the mask image.
        height (int): Height of the mask image.

    Returns:
        mask (numpy.ndarray): A mask image with the polygons filled in white (255).
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    if polygons:
        cv2.fillPoly(mask, polygons, color=255)
    return mask

def process_mask(mask_path, output_folder):
    """
    Processes a mask image by:
      - Extracting polygons from a binary mask (where pixels exactly (0,0,0) are considered foreground).
      - Saving the polygon coordinates (normalized to [0,1]) to a TXT file.
      - Generating a new mask image (using the original pixel coordinates) with a black background and white fill.
    """
    # Read the image in color.
    image = cv2.imread(mask_path)
    if image is None:
        print(f"Failed to load {mask_path}")
        return

    # Create a binary mask: pixels exactly equal to (0,0,0) become white (255); others become black (0).
    binary_mask = cv2.inRange(image, np.array([0, 0, 0]), np.array([0, 0, 0]))

    # Find all contours (including small ones) using RETR_LIST.
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    polygon_lines = []      # For saving polygon coordinates (normalized) as text.
    polygons_for_mask = []  # For drawing the polygons in the new mask image (using absolute coordinates).

    # Process each contour.
    for cnt in contours:
        # Ignore very small contours.
        if cv2.contourArea(cnt) < MIN_CONTOUR_AREA:
            continue

        # Approximate the contour to a polygon.
        epsilon = APPROX_FACTOR * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # Ensure the approximated polygon has at least 2 points.
        if approx.shape[0] < 2:
            continue

        # Reshape to a list of (x, y) points.
        pts = approx.reshape(-1, 2)

        # Normalize coordinates to [0,1] for the TXT file.
        normalized_pts = pts.astype(np.float32).copy()
        normalized_pts[:, 0] /= IMG_WIDTH
        normalized_pts[:, 1] /= IMG_HEIGHT

        # Create a string of normalized coordinates (6 decimal places).
        coords_str = " ".join(f"{coord:.6f}" for point in normalized_pts for coord in point)
        # Prepend the class index (0) to the string.
        line = "0 " + coords_str
        polygon_lines.append(line)

        # For the mask, use the unnormalized (absolute) coordinates.
        pts_for_poly = pts.reshape((-1, 1, 2))
        polygons_for_mask.append(pts_for_poly)

    # Prepare the base file name for outputs.
    base_name = os.path.splitext(os.path.basename(mask_path))[0]

    # ---------------------------
    # 1. Write the polygon TXT file.
    # ---------------------------
    txt_path = os.path.join(output_folder, base_name + ".txt")
    with open(txt_path, "w") as f:
        for line in polygon_lines:
            f.write(line + "\n")
    print(f"Polygon TXT file saved: {txt_path}")

    # ---------------------------
    # 2. Generate a new mask image using the fill_polygons() method.
    # ---------------------------
    
    def fill():
        new_mask = fill_polygons(polygons_for_mask, IMG_WIDTH, IMG_HEIGHT)
        mask_output_path = os.path.join(output_folder, base_name + "_mask.png")
        cv2.imwrite(mask_output_path, new_mask)
        print(f"Generated mask image saved: {mask_output_path}")
    #fill()

def main():
    # Create the output folder if it doesn't exist.
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Adjust the glob pattern if your images use a different extension.
    mask_files = glob.glob(os.path.join(INPUT_FOLDER, "*.png"))
    if not mask_files:
        print("No mask images found in the input folder.")
        return

    for mask_path in mask_files:
        process_mask(mask_path, OUTPUT_FOLDER)


if __name__ == "__main__":
    main()
