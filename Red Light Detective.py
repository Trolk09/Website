import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
import imutils
import os
from datetime import datetime

# ------- Config -------
MODEL_PATH = "yolov8n.pt"   # or yolov8s.pt
USE_WEBCAM = True           # True -> webcam, False -> video file
VIDEO_PATH = "traffic.mp4"
RED_THRESHOLD = 0.05        # fraction of red pixels inside traffic light ROI to call 'red'
SAVE_ON_DETECTION = True
OUT_DIR = "detections"
os.makedirs(OUT_DIR, exist_ok=True)

# Classes in COCO: 'car', 'truck', 'bus', 'motorbike' etc. We'll accept these as vehicles.
VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle"}  # names used by model.names

# Initialize models/readers
model = YOLO(MODEL_PATH)
ocr_reader = easyocr.Reader(['en'], gpu=False)  # set gpu=True if you have GPU and easyocr supports it

# Helper: heuristic function to find a license-plate-like rectangle inside a vehicle crop
def find_plate_candidates(vehicle_img):
    # Resize for consistent processing
    h, w = vehicle_img.shape[:2]
    scale = 600 / max(w, h) if max(w, h) > 800 else 1.0
    if scale != 1.0:
        vehicle_img = imutils.resize(vehicle_img, width=int(w * scale))
    gray = cv2.cvtColor(vehicle_img, cv2.COLOR_BGR2GRAY)
    # Enhance
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    # Edge detection
    edged = cv2.Canny(gray, 100, 200)
    # Morph to close gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,3))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel, iterations=2)
    # Find contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for cnt in contours:
        x,y,wc,hc = cv2.boundingRect(cnt)
        area = wc*hc
        if area < 500:  # too small
            continue
        aspect = wc / float(hc) if hc>0 else 0
        # license plates are usually wide rectangles: try wide aspect ratio (2..8)
        if 2.0 <= aspect <= 8.0:
            # crop and return coordinates scaled back if needed
            candidates.append((x,y,wc,hc))
    # Sort by area descending (largest first)
    candidates = sorted(candidates, key=lambda b: b[2]*b[3], reverse=True)
    return candidates, vehicle_img

# Helper: OCR a cropped plate candidate, return text and confidence-ish metric
def ocr_plate(plate_img):
    # EasyOCR expects RGB
    plate_rgb = cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB)
    results = ocr_reader.readtext(plate_rgb, detail=1, paragraph=False)
    # results: list of tuples (bbox, text, confidence)
    if not results:
        return None, 0.0
    # Choose the longest/highest confidence text (heuristic)
    best = max(results, key=lambda r: (len(r[1]), r[2]))
    text = best[1]
    conf = best[2]
    # Clean text a bit: remove weird chars, spaces around
    text = "".join([c for c in text if c.isalnum() or c in "- "]).strip()
    return text, conf

# Video capture
cap = cv2.VideoCapture(0 if USE_WEBCAM else VIDEO_PATH)

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1

    # Run YOLO detection
    results = model(frame, imgsz=640)  # small size for speed; increase imgsz for accuracy
    r = results[0]

    annotated = frame.copy()
    detected_red_light = False
    red_light_boxes = []

    # First pass: find traffic lights and check for red
    for box in r.boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        if label == "traffic light":
            # crop ROI, convert to HSV and check red
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            # red ranges
            mask1 = cv2.inRange(hsv, (0, 100, 70), (10, 255, 255))
            mask2 = cv2.inRange(hsv, (170, 100, 70), (180, 255, 255))
            red_mask = cv2.bitwise_or(mask1, mask2)
            red_pixels = cv2.countNonZero(red_mask)
            total_pixels = max(1, roi.shape[0]*roi.shape[1])
            ratio = red_pixels/total_pixels
            cv2.rectangle(annotated, (x1,y1), (x2,y2), (0,255,255), 2)
            cv2.putText(annotated, f"TL red_ratio:{ratio:.2f}", (x1, y2+15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
            if ratio >= RED_THRESHOLD:
                detected_red_light = True
                red_light_boxes.append((x1,y1,x2,y2))
                cv2.putText(annotated, "RED LIGHT!", (x1, y1-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    # If red light found: look for vehicles underneath / in frame and try to OCR plate
    found_plate_texts = []
    if detected_red_light:
        # find vehicle boxes
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            if label in VEHICLE_CLASSES:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(annotated, (x1,y1), (x2,y2), (255,0,0), 2)
                cv2.putText(annotated, label, (x1, y1-6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

                # Crop vehicle region and attempt to locate license-plate-like rectangles
                vehicle_crop = frame[y1:y2, x1:x2]
                if vehicle_crop.size == 0:
                    continue
                candidates, processed_vehicle = find_plate_candidates(vehicle_crop)
                plate_text = None
                best_conf = 0.0
                best_plate_box = None
                # Try top N candidates
                for (cx,cy,cw,ch) in candidates[:5]:
                    plate_crop = vehicle_crop[cy:cy+ch, cx:cx+cw]
                    text, conf = ocr_plate(plate_crop)
                    # require some minimal text & confidence
                    if text and len(text) >= 4 and conf > 0.3:
                        # convert coords back to full-frame coordinates
                        bf_x1 = x1 + cx
                        bf_y1 = y1 + cy
                        bf_x2 = bf_x1 + cw
                        bf_y2 = bf_y1 + ch
                        # choose the best by confidence and length
                        if conf > best_conf or (conf==best_conf and len(text)> (len(plate_text) if plate_text else 0)):
                            plate_text = text
                            best_conf = conf
                            best_plate_box = (bf_x1, bf_y1, bf_x2, bf_y2)
                if plate_text:
                    found_plate_texts.append((plate_text, best_conf, best_plate_box))
                    # annotate on frame
                    bx1, by1, bx2, by2 = best_plate_box
                    cv2.rectangle(annotated, (bx1,by1), (bx2,by2), (0,255,0), 2)
                    cv2.putText(annotated, f"{plate_text} ({best_conf:.2f})", (bx1, by1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        # If any plates found, optionally save a snapshot/log
        if found_plate_texts and SAVE_ON_DETECTION:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = os.path.join(OUT_DIR, f"det_{ts}.jpg")
            # overlay text summary before saving
            out = annotated.copy()
            y0 = 30
            for idx, (txt, conf, _) in enumerate(found_plate_texts):
                cv2.putText(out, f"{idx+1}: {txt} ({conf:.2f})", (10, y0 + idx*25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.imwrite(fname, out)
            print(f"[INFO] Saved detection image: {fname}")

    # Show annotated frame
    cv2.imshow("Red Light + Plate Detection", annotated)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
