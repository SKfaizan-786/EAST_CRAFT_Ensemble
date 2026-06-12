
import os
import cv2
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union
from craft_text_detector import Craft
from craft_text_detector.predict import get_prediction

# ==========================================
# CONFIGURATION
# ==========================================
IMAGE_PATH = r"data/WhatsApp Image 2025-12-19 at 13.07.13.jpeg"
OUTPUT_DIR = "outputs/ensemble_test"
EAST_MODEL_PATH = "models/frozen_east_text_detection.pb"
CRAFT_MODEL_PATH = "models/craft_mlt_25k.pth"

# EAST params
EAST_INPUT_SIZE = 640
EAST_SCORE_THRESH = 0.50
EAST_NMS_THRESH = 0.40
EAST_MIN_HEIGHT = 10
EAST_MIN_AREA = 80

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# GEOMETRY FUNCTIONS
# ==========================================
def poly_iou(p1, p2):
    try:
        P1 = Polygon(p1)
        P2 = Polygon(p2)
        if not P1.is_valid or not P2.is_valid:
            return 0.0
        inter = P1.intersection(P2).area
        union = unary_union([P1, P2]).area
        return 0.0 if union == 0 else inter / union
    except:
        return 0.0

def poly_area(poly):
    try:
        return Polygon(poly).area
    except:
        return 0.0

def get_aspect_ratio(poly):
    try:
        x_coords = poly[:, 0]
        y_coords = poly[:, 1]
        width = x_coords.max() - x_coords.min()
        height = y_coords.max() - y_coords.min()
        if height == 0:
            return 100.0
        return width / height
    except:
        return 1.0

def is_valid_text_box(poly, score, area):
    if area < 200: return False
    aspect = get_aspect_ratio(poly)
    if aspect < 0.3 or aspect > 15.0: return False
    if area < 350 and score < 0.80: return False
    if area < 280 and score < 0.83: return False
    if area < 230 and score < 0.87: return False
    return True

def soft_nms_polygons(boxes, iou_thr=0.4):
    final = []
    for poly, score in sorted(boxes, key=lambda x: -x[1]):
        keep = True
        for fpoly, _ in final:
            if poly_iou(poly, fpoly) > iou_thr:
                keep = False
                break
        if keep:
            final.append((poly, score))
    return final

# ==========================================
# ENSEMBLE LOGIC
# ==========================================
def ensemble_union(east_boxes, craft_boxes):
    final = []
    used_craft = set()
    used_east = set()

    # STRATEGY 1A: Strong agreement (IoU >= 0.35)
    for ci, (cpoly, cscore) in enumerate(craft_boxes):
        area = poly_area(cpoly)
        if cscore < 0.60: continue

        best_iou = 0.0
        best_ei = -1
        for ei, (epoly, escore) in enumerate(east_boxes):
            if ei in used_east: continue
            iou = poly_iou(cpoly, epoly)
            if iou > best_iou:
                best_iou = iou
                best_ei = ei

        if best_iou >= 0.35:
            epoly, escore = east_boxes[best_ei]
            if not is_valid_text_box(cpoly, cscore, area): continue
            
            combined_conf = max(cscore, escore)
            avg_conf = (cscore + escore) / 2.0
            iou_weight = (best_iou - 0.35) / 0.65
            final_score = combined_conf * (1.0 + 0.15 * iou_weight) + 0.05 * avg_conf
            final_score = min(1.0, final_score)

            if final_score < 0.66: continue

            used_craft.add(ci)
            used_east.add(best_ei)
            final.append((cpoly, final_score))

    # STRATEGY 1B: Medium agreement (IoU 0.28-0.35)
    for ci, (cpoly, cscore) in enumerate(craft_boxes):
        if ci in used_craft: continue
        area = poly_area(cpoly)
        if cscore < 0.67: continue

        best_iou = 0.0
        best_ei = -1
        for ei, (epoly, escore) in enumerate(east_boxes):
            if ei in used_east: continue
            iou = poly_iou(cpoly, epoly)
            if iou > best_iou:
                best_iou = iou
                best_ei = ei

        if best_iou >= 0.28 and best_iou < 0.35:
            epoly, escore = east_boxes[best_ei]
            if not is_valid_text_box(cpoly, cscore, area): continue
            combined_conf = max(cscore, escore)
            if combined_conf < 0.75: continue
            if area < 350: continue

            used_craft.add(ci)
            used_east.add(best_ei)
            final.append((cpoly, combined_conf))

    # STRATEGY 2: CRAFT singletons
    for ci, (cpoly, cscore) in enumerate(craft_boxes):
        if ci in used_craft: continue
        area = poly_area(cpoly)
        if not is_valid_text_box(cpoly, cscore, area): continue
        if cscore >= 0.87 or (cscore >= 0.80 and area >= 750):
            final.append((cpoly, cscore))

    # STRATEGY 3: EAST singletons
    for ei, (epoly, escore) in enumerate(east_boxes):
        if ei in used_east: continue
        area = poly_area(epoly)
        if not is_valid_text_box(epoly, escore, area): continue
        if escore >= 0.70 and area >= 850:
            overlap = False
            for fpoly, _ in final:
                if poly_iou(epoly, fpoly) > 0.20:
                    overlap = True
                    break
            if not overlap:
                final.append((epoly, escore))

    # STRATEGY 4: Soft NMS
    final = soft_nms_polygons(final, iou_thr=0.29)
    return final

# ==========================================
# INFERENCE RUNNERS
# ==========================================
def decode_east(scores, geometry, scoreThresh):
    (numRows, numCols) = scores.shape[2:4]
    rects, confidences = [], []
    for y in range(numRows):
        scoresData = scores[0, 0, y]
        x0, x1, x2, x3, angles = geometry[0, 0, y], geometry[0, 1, y], geometry[0, 2, y], geometry[0, 3, y], geometry[0, 4, y]
        for x in range(numCols):
            score = float(scoresData[x])
            if score < scoreThresh: continue
            angle = angles[x]
            cos, sin = np.cos(angle), np.sin(angle)
            h, w = x0[x] + x2[x], x1[x] + x3[x]
            endX = int(x * 4.0 + (cos * x1[x]) + (sin * x2[x]))
            endY = int(y * 4.0 - (sin * x1[x]) + (cos * x2[x]))
            startX, startY = int(endX - w), int(endY - h)
            rects.append((startX, startY, endX, endY))
            confidences.append(score)
    return rects, confidences

def run_east(img):
    print("Running EAST...")
    H, W = img.shape[:2]
    newW = (EAST_INPUT_SIZE // 32) * 32
    newH = (EAST_INPUT_SIZE // 32) * 32
    rW, rH = W / float(newW), H / float(newH)

    image = cv2.resize(img, (newW, newH))
    blob = cv2.dnn.blobFromImage(image, 1.0, (newW, newH), (123.68,116.78,103.94), swapRB=True, crop=False)
    
    net = cv2.dnn.readNet(EAST_MODEL_PATH)
    net.setInput(blob)
    scores, geometry = net.forward(["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"])
    
    rects, confs = decode_east(scores, geometry, EAST_SCORE_THRESH)
    idxs = cv2.dnn.NMSBoxes(rects, confs, EAST_SCORE_THRESH, EAST_NMS_THRESH)
    
    boxes = []
    if len(idxs) > 0:
        for i in np.array(idxs).flatten():
            x1, y1, x2, y2 = rects[i]
            x1, y1 = int(x1 * rW), int(y1 * rH)
            x2, y2 = int(x2 * rW), int(y2 * rH)
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W-1, x2), min(H-1, y2)
            
            w, h = x2 - x1, y2 - y1
            if w*h < EAST_MIN_AREA or h < EAST_MIN_HEIGHT: continue
            
            poly = np.array([[x1,y1], [x2,y1], [x2,y2], [x1,y2]], dtype=np.float32)
            boxes.append((poly, confs[i] * 0.7)) # Normalize EAST score like in notebook
    
    print(f"EAST detected {len(boxes)} boxes")
    return boxes

def run_craft(img):
    print("Running CRAFT...")
    # Initialize CRAFT
    # Note: loading model every time is inefficient but fine for single run
    craft = Craft(
        output_dir=OUTPUT_DIR,
        crop_type="box",
        cuda=False, 
        rectify=True,
        weight_path_craft_net=CRAFT_MODEL_PATH
    )
    
    prediction_result = get_prediction(
        image=img,
        craft_net=craft.craft_net,
        refine_net=craft.refine_net,
        text_threshold=craft.text_threshold,
        link_threshold=craft.link_threshold,
        low_text=craft.low_text,
        cuda=craft.cuda,
        long_size=craft.long_size,
        poly=False
    )
    
    boxes_raw = prediction_result["boxes"]
    scores_raw = prediction_result.get("boxes_scores")
    
    boxes = []
    for i, box in enumerate(boxes_raw):
        if box is not None:
             # Ensure box is float32 polygon
            box = np.array(box, dtype=np.float32)
            score = float(scores_raw[i]) if scores_raw is not None and i < len(scores_raw) else 0.90
            boxes.append((box, score))

    craft.unload_craftnet_model()
    craft.unload_refinenet_model()
    
    print(f"CRAFT detected {len(boxes)} boxes")
    return boxes

# ==========================================
# MAIN
# ==========================================
def main():
    if not os.path.exists(IMAGE_PATH):
        print(f"Error: Image not found at {IMAGE_PATH}")
        return

    print(f"Processing {IMAGE_PATH}")
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print("Error: Failed to load image")
        return

    east_boxes = run_east(img)
    craft_boxes = run_craft(img)
    
    fused_boxes = ensemble_union(east_boxes, craft_boxes)
    print(f"Ensemble fused {len(fused_boxes)} boxes")

    # Draw result
    vis_img = img.copy()
    for poly, score in fused_boxes:
        pts = poly.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(vis_img, [pts], True, (0, 255, 0), 2)
        x, y = int(poly[0][0]), int(poly[0][1]) - 5
        cv2.putText(vis_img, f"{score:.2f}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    out_path = os.path.join(OUTPUT_DIR, "final_ensemble_result.jpg")
    cv2.imwrite(out_path, vis_img)
    print(f"Result saved to {out_path}")

if __name__ == "__main__":
    main()
