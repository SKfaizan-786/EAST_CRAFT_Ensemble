# -*- coding: utf-8 -*-
"""
EAST + CRAFT Multi-Tier WBF Ensemble
Test image inference -> output: test_multitierWBF.png
- Red boxes only, no heading, no caption, no border, 300 DPI
"""

import os, cv2, numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union

# ── Paths ────────────────────────────────────────────────────────────────────
IMAGE_PATH       = "test.jpeg"
EAST_MODEL       = "models/frozen_east_text_detection.pb"
CRAFT_MODEL      = "models/craft_mlt_25k.pth"
OUTPUT_PATH      = "test_multitierWBF.png"

# ── EAST params ───────────────────────────────────────────────────────────────
EAST_SIZE        = 640
EAST_SCORE_THR   = 0.50
EAST_NMS_THR     = 0.40
EAST_MIN_AREA    = 80
EAST_MIN_H       = 10

# ═════════════════════════════════════════════════════════════════════════════
# Geometry helpers  (mirror of ensemble_final.ipynb)
# ═════════════════════════════════════════════════════════════════════════════
def poly_iou(p1, p2):
    try:
        P1, P2 = Polygon(p1), Polygon(p2)
        if not P1.is_valid or not P2.is_valid: return 0.0
        inter = P1.intersection(P2).area
        union = unary_union([P1, P2]).area
        return 0.0 if union == 0 else inter / union
    except: return 0.0

def poly_area(poly):
    try: return Polygon(poly).area
    except: return 0.0

def get_aspect_ratio(poly):
    try:
        xc, yc = poly[:,0], poly[:,1]
        w = xc.max()-xc.min(); h = yc.max()-yc.min()
        return 100.0 if h == 0 else w/h
    except: return 1.0

def is_valid(poly, score, area):
    if area < 200: return False
    asp = get_aspect_ratio(poly)
    if asp < 0.3 or asp > 15.0: return False
    if area < 350 and score < 0.80: return False
    if area < 280 and score < 0.83: return False
    if area < 230 and score < 0.87: return False
    return True

def soft_nms(boxes, thr=0.29):
    final = []
    for poly, sc in sorted(boxes, key=lambda x: -x[1]):
        if all(poly_iou(poly, fp) <= thr for fp, _ in final):
            final.append((poly, sc))
    return final

# ═════════════════════════════════════════════════════════════════════════════
# Multi-Tier WBF Ensemble  (exact logic from ensemble_final.ipynb)
# ═════════════════════════════════════════════════════════════════════════════
def ensemble_union(east_boxes, craft_boxes):
    final, used_c, used_e = [], set(), set()

    # Tier 1A – Strong agreement IoU >= 0.35
    for ci, (cp, cs) in enumerate(craft_boxes):
        if cs < 0.60: continue
        area = poly_area(cp)
        best_iou, best_ei = 0.0, -1
        for ei, (ep, es) in enumerate(east_boxes):
            if ei in used_e: continue
            iou = poly_iou(cp, ep)
            if iou > best_iou: best_iou, best_ei = iou, ei
        if best_iou >= 0.35:
            ep, es = east_boxes[best_ei]
            if not is_valid(cp, cs, area): continue
            w = (best_iou - 0.35) / 0.65
            sc = min(1.0, max(cs,es)*(1.0+0.15*w) + 0.05*(cs+es)/2)
            if sc < 0.66: continue
            used_c.add(ci); used_e.add(best_ei)
            final.append((cp, sc))

    # Tier 1B – Medium agreement 0.28–0.35
    for ci, (cp, cs) in enumerate(craft_boxes):
        if ci in used_c or cs < 0.67: continue
        area = poly_area(cp)
        best_iou, best_ei = 0.0, -1
        for ei, (ep, es) in enumerate(east_boxes):
            if ei in used_e: continue
            iou = poly_iou(cp, ep)
            if iou > best_iou: best_iou, best_ei = iou, ei
        if 0.28 <= best_iou < 0.35:
            ep, es = east_boxes[best_ei]
            if not is_valid(cp, cs, area): continue
            sc = max(cs, es)
            if sc < 0.75 or area < 350: continue
            used_c.add(ci); used_e.add(best_ei)
            final.append((cp, sc))

    # Tier 2 – High-confidence CRAFT singletons
    for ci, (cp, cs) in enumerate(craft_boxes):
        if ci in used_c: continue
        area = poly_area(cp)
        if not is_valid(cp, cs, area): continue
        if cs >= 0.87 or (cs >= 0.80 and area >= 750):
            final.append((cp, cs))

    # Tier 3 – High-confidence EAST singletons
    for ei, (ep, es) in enumerate(east_boxes):
        if ei in used_e: continue
        area = poly_area(ep)
        if not is_valid(ep, es, area): continue
        if es >= 0.70 and area >= 850:
            if all(poly_iou(ep, fp) <= 0.20 for fp, _ in final):
                final.append((ep, es))

    return soft_nms(final, thr=0.29)

# ═════════════════════════════════════════════════════════════════════════════
# EAST inference
# ═════════════════════════════════════════════════════════════════════════════
def decode_east(scores, geom, thr):
    nr, nc = scores.shape[2], scores.shape[3]
    rects, confs = [], []
    for y in range(nr):
        sd = scores[0,0,y]
        x0,x1,x2,x3,ang = geom[0,0,y],geom[0,1,y],geom[0,2,y],geom[0,3,y],geom[0,4,y]
        for x in range(nc):
            sc = float(sd[x])
            if sc < thr: continue
            a = ang[x]; cos, sin = np.cos(a), np.sin(a)
            h = x0[x]+x2[x]; w = x1[x]+x3[x]
            ex = int(x*4+(cos*x1[x])+(sin*x2[x]))
            ey = int(y*4-(sin*x1[x])+(cos*x2[x]))
            rects.append((int(ex-w), int(ey-h), ex, ey)); confs.append(sc)
    return rects, confs

def run_east(img):
    print("Running EAST...", flush=True)
    H, W = img.shape[:2]
    nW = nH = (EAST_SIZE//32)*32
    rW, rH = W/nW, H/nH
    blob = cv2.dnn.blobFromImage(cv2.resize(img,(nW,nH)), 1.0, (nW,nH),
                                  (123.68,116.78,103.94), swapRB=True, crop=False)
    net = cv2.dnn.readNet(EAST_MODEL)
    net.setInput(blob)
    sc, geo = net.forward(["feature_fusion/Conv_7/Sigmoid","feature_fusion/concat_3"])
    rects, confs = decode_east(sc, geo, EAST_SCORE_THR)
    idxs = cv2.dnn.NMSBoxes(rects, confs, EAST_SCORE_THR, EAST_NMS_THR)
    boxes = []
    for i in (np.array(idxs).flatten() if len(idxs) else []):
        x1,y1,x2,y2 = rects[i]
        x1,y1 = int(x1*rW),int(y1*rH)
        x2,y2 = int(x2*rW),int(y2*rH)
        x1,y1,x2,y2 = max(0,x1),max(0,y1),min(W-1,x2),min(H-1,y2)
        if (x2-x1)*(y2-y1)<EAST_MIN_AREA or (y2-y1)<EAST_MIN_H: continue
        poly = np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]],dtype=np.float32)
        boxes.append((poly, confs[i]*0.7))
    print(f"  EAST: {len(boxes)} boxes", flush=True)
    return boxes

# ═════════════════════════════════════════════════════════════════════════════
# CRAFT inference
# ═════════════════════════════════════════════════════════════════════════════
def run_craft(img):
    print("Running CRAFT...", flush=True)
    try:
        from craft_text_detector import Craft
        from craft_text_detector.predict import get_prediction
        craft = Craft(output_dir="outputs/tmp_craft", crop_type="box",
                      cuda=False, rectify=True, weight_path_craft_net=CRAFT_MODEL)
        res = get_prediction(image=img, craft_net=craft.craft_net,
                             refine_net=craft.refine_net,
                             text_threshold=craft.text_threshold,
                             link_threshold=craft.link_threshold,
                             low_text=craft.low_text, cuda=craft.cuda,
                             long_size=craft.long_size, poly=False)
        raw_boxes = res["boxes"]
        raw_scores = res.get("boxes_scores")
        boxes = []
        for i, b in enumerate(raw_boxes):
            if b is None: continue
            b = np.array(b, dtype=np.float32)
            sc = float(raw_scores[i]) if raw_scores is not None and i<len(raw_scores) else 0.90
            boxes.append((b, sc))
        craft.unload_craftnet_model(); craft.unload_refinenet_model()
        print(f"  CRAFT: {len(boxes)} boxes", flush=True)
        return boxes
    except Exception as e:
        print(f"  CRAFT failed ({e}) -- using EAST-only mode", flush=True)
        return []

# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════
def main():
    if not os.path.exists(IMAGE_PATH):
        print(f"Image not found: {IMAGE_PATH}"); return

    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print("Failed to load image"); return

    print(f"Image loaded: {img.shape[1]}x{img.shape[0]}")

    east_boxes  = run_east(img)
    craft_boxes = run_craft(img)

    if craft_boxes:
        fused = ensemble_union(east_boxes, craft_boxes)
        mode = "Multi-Tier WBF (EAST+CRAFT)"
    else:
        # CRAFT unavailable: use validated EAST boxes directly
        fused = []
        for poly, sc in east_boxes:
            area = poly_area(poly)
            if is_valid(poly, sc, area):
                fused.append((poly, sc))
        # apply soft NMS
        fused = soft_nms(fused, thr=0.29)
        mode = "EAST-only (CRAFT unavailable)"

    print(f"Mode: {mode}")
    print(f"Ensemble: {len(fused)} boxes after fusion")

    # Draw: RED boxes only, no heading, no caption
    out = img.copy()
    for poly, _ in fused:
        pts = poly.astype(np.int32).reshape(-1,1,2)
        cv2.polylines(out, [pts], True, (0,0,255), 2)   # RED

    # Save at 300 DPI as PNG using PIL for DPI metadata
    from PIL import Image as PILImage
    out_rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    pil_img = PILImage.fromarray(out_rgb)
    pil_img.save(OUTPUT_PATH, dpi=(300, 300))
    print(f"\nSaved: {OUTPUT_PATH}")
    print(f"Boxes drawn : {len(fused)}")
    print(f"DPI         : 300x300")
    print(f"Mode        : {mode}")

if __name__ == "__main__":
    main()
