import cv2
from ultralytics import YOLO

# --- Configuration ---
MODEL_PATH = "yolov8n.pt"  # Pretrained model (COCO)
USE_WEBCAM = True           # True for webcam, False for video file
VIDEO_PATH = "food.mp4"     # Path to your video file
CONF_THRESHOLD = 0.4        # Minimum confidence to show detection

# List of food-related classes in YOLOv8 (COCO)
FOOD_CLASSES = {"banana", "apple", "sandwich", "orange", "broccoli", "carrot",
                "hot dog", "pizza", "donut", "cake"}

# Load YOLO model
model = YOLO(MODEL_PATH)

# Choose source
cap = cv2.VideoCapture(0 if USE_WEBCAM else VIDEO_PATH)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    annotated_frame = frame.copy()

    # Draw boxes for detected food
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = model.names[cls_id]
        if label in FOOD_CLASSES and conf >= CONF_THRESHOLD:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"{label} ({conf:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Food Detection", annotated_frame)

    # Press ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
