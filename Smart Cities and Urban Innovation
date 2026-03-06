import cv2
from ultralytics import YOLO

# Load YOLO model (COCO pretrained)
model = YOLO("yolov8n.pt")

# Open both videos
cap1 = cv2.VideoCapture("road1.mp4")
cap2 = cv2.VideoCapture("road2.mp4")

# Only count these classes
vehicle_classes = ["car", "motorcycle", "bus", "truck", "bicycle"]

while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not ret1 and not ret2:
        break

    counts1 = [0] * 1
    counts2 = [0] * 1

    # Process first camera
    if ret1:
        results1 = model(frame1)
        for r in results1:
            for box in r.boxes:
                cls = int(box.cls[0])
                label = model.names[cls]
                if label in vehicle_classes:
                    counts1[0] += 1
                    x1,y1,x2,y2 = map(int,box.xyxy[0])
                    cv2.rectangle(frame1, (x1,y1), (x2,y2), (0,255,0), 2)

    # Process second camera
    if ret2:
        results2 = model(frame2)
        for r in results2:
            for box in r.boxes:
                cls = int(box.cls[0])
                label = model.names[cls]
                if label in vehicle_classes:
                    counts2[0] += 1
                    x1,y1,x2,y2 = map(int,box.xyxy[0])
                    cv2.rectangle(frame2, (x1,y1), (x2,y2), (255,0,0), 2)

    # Decide signals
    sig1 = "RED"
    sig2 = "RED"

    if counts1[0] > counts2[0]:
        sig1 = "GREEN"
    elif counts2[0] > counts1[0]:
        sig2 = "GREEN"
    else:
        sig1 = "GREEN"  # tie → make road1 green by default

    # Put text on frames
    if ret1:
        cv2.putText(frame1, f"Road1: {sig1} ({counts1[0]})", (30,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0) if sig1=="GREEN" else (0,0,255), 3)
    if ret2:
        cv2.putText(frame2, f"Road2: {sig2} ({counts2[0]})", (30,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0) if sig2=="GREEN" else (0,0,255), 3)

    # Combine and show
    if ret1 and ret2:
        output = cv2.hconcat([frame1, frame2])
    elif ret1:
        output = frame1
    else:
        output = frame2

    cv2.imshow("AI Traffic Dual Camera Signals", output)
    if cv2.waitKey(1) == 27:
        break

cap1.release()
cap2.release()
cv2.destroyAllWindows()
