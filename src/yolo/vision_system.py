import cv2
import time
import numpy as np
import pandas as pd
from ultralytics import YOLO
from datetime import datetime

# -------------------------
# Load Models
# -------------------------
part_model = YOLO("best.pt")
print(part_model.names)

# -------------------------
# Class Names
# -------------------------
PART_CLASSES = {
    0: "gear",
    
    1: "knob",
    2: "washer",
    3: "frame",
    4: "guide",
    5: "mount"
}


# -------------------------
# Report Storage
# -------------------------
inspection_log = []

# -------------------------
# Helper Functions
# -------------------------


# -------------------------
# Video Capture
# -------------------------
cap = cv2.VideoCapture(0)
prev_time = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()
    fps = 1 / (current_time - prev_time) if prev_time else 0
    prev_time = current_time

    # -------------------------
    # Part Detection
    # -------------------------
    part_results = part_model(frame, conf=0.5, iou=0.5, verbose=False)

    for result in part_results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            part_name = PART_CLASSES[cls_id]
            roi = frame[y1:y2, x1:x2]

            if roi.size == 0:
                continue

          

            

            # -------------------------
            # Quality Decision
            # -------------------------

            # -------------------------
            # Draw Part Box
            # -------------------------
            color = (0, 255, 0) 

            label = f"{part_name} | {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # -------------------------
            # Draw Defect Boxes
            # -------------------------
           
            # -------------------------
            # Log Inspection
            # -------------------------
           
    # -------------------------
    # FPS Display
    # -------------------------
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.imshow("Quality Inspection System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -------------------------
# Cleanup
# -------------------------
cap.release()
cv2.destroyAllWindows()

# -------------------------
# Save Report
# -------------------------
df = pd.DataFrame(inspection_log)
df.to_csv("inspection_report.csv", index=False)
print("Inspection report saved.")
