"""
Simplified Refined CRAFT - Uses existing craft_text_detector library
But with optimized parameters and NO RefineNet
"""

import os
import cv2
import numpy as np
import glob
from tqdm import tqdm

# Import CRAFT from craft_text_detector
from craft_text_detector import Craft

# Paths
IMG_DIR = "data/icdar2015/test_images"
SAVE_DIR = "outputs/craft_refined_results"
MODEL_PATH = "models/craft_mlt_25k.pth"

os.makedirs(SAVE_DIR, exist_ok=True)

def process_image(craft_model, image_path, save_dir):
    """Process single image with optimized CRAFT"""

    # Load image
    img = cv2.imread(image_path)
    if img is None:
        return None

    # Get prediction WITHOUT RefineNet
    from craft_text_detector.predict import get_prediction

    prediction_result = get_prediction(
        image=img,
        craft_net=craft_model.craft_net,
        refine_net=None,  # ← KEY: NO RefineNet for ICDAR 2015!
        text_threshold=0.7,  # Optimized for ICDAR 2015
        link_threshold=0.4,
        low_text=0.4,
        cuda=craft_model.cuda,
        long_size=2240,  # ← Paper-specific: 2240px for ICDAR 2015
        poly=False
    )

    boxes = prediction_result["boxes"]

    # Save results
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # Save boxes
    txt_path = os.path.join(save_dir, f"{base_name}_craft_boxes.txt")
    with open(txt_path, 'w') as f:
        for box in boxes:
            if box is not None and len(box) == 4:
                coords = ','.join([f"{int(x)},{int(y)}" for x, y in box])
                f.write(f"{coords},0.9000\n")

    # Save visualization
    result_img = img.copy()
    for box in boxes:
        if box is not None:
            box_int = np.array(box, dtype=np.int32)
            cv2.polylines(result_img, [box_int], True, (0, 255, 0), 2)

    result_path = os.path.join(save_dir, f"{base_name}_craft_result.jpg")
    cv2.imwrite(result_path, result_img)

    return len(boxes)

def main():
    print("=" * 60)
    print("REFINED CRAFT - ICDAR 2015 Optimized")
    print("=" * 60)
    print("\nKey optimizations:")
    print("  ✓ NO RefineNet (prevents word merging)")
    print("  ✓ 2240px long-side (paper-specific)")
    print("  ✓ text_threshold=0.7 (optimized)")
    print("  ✓ link_threshold=0.4 (optimized)")
    print("\n" + "=" * 60)

    # Initialize CRAFT WITHOUT refine_net
    print("\nInitializing CRAFT model...")
    craft = Craft(
        output_dir=SAVE_DIR,
        crop_type="box",
        cuda=False,
        rectify=True,
        weight_path_craft_net=MODEL_PATH,
        # Note: We don't load refine_net!
    )
    print("✓ CRAFT model loaded (RefineNet: DISABLED)")

    # Get images
    image_files = glob.glob(os.path.join(IMG_DIR, "*.jpg"))
    print(f"\n📸 Found {len(image_files)} images to process\n")

    # Process all images
    total_detections = 0
    for img_path in tqdm(image_files, desc="Processing"):
        num_boxes = process_image(craft, img_path, SAVE_DIR)
        if num_boxes:
            total_detections += num_boxes

    print(f"\n✓ Processing complete!")
    print(f"Total detections: {total_detections}")
    print(f"Average per image: {total_detections / len(image_files):.1f}")
    print(f"Results saved to: {SAVE_DIR}")

    # Cleanup
    craft.unload_craftnet_model()

if __name__ == "__main__":
    main()
