"""
Generate side-by-side comparison visualizations for EAST, CRAFT, and Ensemble detections.
Creates overlays for 5 sample images to showcase in README.
"""

import os
import cv2
import numpy as np
from glob import glob

# Configuration
IMAGE_DIR = "data/icdar2015/test_images"
EAST_DIR = "outputs/east_final_results"
CRAFT_DIR = "outputs/craft_ensemble_ready"
ENSEMBLE_DIR = "outputs/ensemble_union_balanced"
VIZ_DIR = "visualizations"

# Sample images to visualize (diverse examples)
SAMPLE_IMAGES = ["img_10", "img_100", "img_108", "img_112", "img_114"]

os.makedirs(VIZ_DIR, exist_ok=True)


def read_boxes(txt_path):
    """Read bounding boxes from text file."""
    boxes = []
    if not os.path.exists(txt_path):
        return boxes

    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) not in (8, 9):
                continue
            try:
                pts = list(map(float, parts[:8]))
                poly = np.array(pts, dtype=np.float32).reshape(4, 2)
                score = float(parts[8]) if len(parts) == 9 else 0.5
                boxes.append((poly, score))
            except:
                continue
    return boxes


def draw_detections(image, boxes, color=(0, 255, 0), thickness=2, label=""):
    """Draw bounding boxes on image with label."""
    vis_img = image.copy()

    # Draw boxes
    for poly, score in boxes:
        pts = poly.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(vis_img, [pts], True, color, thickness)

    # Add label at top
    h, w = vis_img.shape[:2]
    cv2.putText(vis_img, f"{label} ({len(boxes)} boxes)",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, color, 2, cv2.LINE_AA)

    return vis_img


def create_comparison(image_name):
    """Create side-by-side comparison of EAST, CRAFT, and Ensemble."""
    # Load image
    img_path = os.path.join(IMAGE_DIR, f"{image_name}.jpg")
    if not os.path.exists(img_path):
        img_path = os.path.join(IMAGE_DIR, f"{image_name}.png")

    if not os.path.exists(img_path):
        print(f"[ERROR] Image not found: {image_name}")
        return

    img = cv2.imread(img_path)
    if img is None:
        print(f"[ERROR] Failed to load: {image_name}")
        return

    # Read detections
    east_boxes = read_boxes(os.path.join(EAST_DIR, f"{image_name}_east_boxes.txt"))
    craft_boxes = read_boxes(os.path.join(CRAFT_DIR, f"{image_name}_craft_boxes.txt"))
    ensemble_boxes = read_boxes(os.path.join(ENSEMBLE_DIR, f"{image_name}_fused.txt"))

    # Create visualizations
    east_vis = draw_detections(img, east_boxes, color=(0, 0, 255), label="EAST")  # Red
    craft_vis = draw_detections(img, craft_boxes, color=(255, 0, 0), label="CRAFT")  # Blue
    ensemble_vis = draw_detections(img, ensemble_boxes, color=(0, 255, 0), label="Ensemble")  # Green

    # Resize to fit side-by-side (max height 400px)
    h, w = img.shape[:2]
    if h > 400:
        scale = 400 / h
        new_w = int(w * scale)
        east_vis = cv2.resize(east_vis, (new_w, 400))
        craft_vis = cv2.resize(craft_vis, (new_w, 400))
        ensemble_vis = cv2.resize(ensemble_vis, (new_w, 400))

    # Add separator lines
    h, w = east_vis.shape[:2]
    separator = np.ones((h, 3, 3), dtype=np.uint8) * 255

    # Concatenate horizontally
    comparison = np.hstack([east_vis, separator, craft_vis, separator, ensemble_vis])

    # Save
    output_path = os.path.join(VIZ_DIR, f"{image_name}_comparison.jpg")
    cv2.imwrite(output_path, comparison, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"[OK] Created: {output_path}")

    return output_path


def main():
    print("=" * 60)
    print("    Creating Visualization Comparisons")
    print("=" * 60)
    print()

    for img_name in SAMPLE_IMAGES:
        create_comparison(img_name)

    print()
    print("=" * 60)
    print(f"[OK] All visualizations saved to: {VIZ_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
