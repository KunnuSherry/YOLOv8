import cv2
from ultralytics import YOLO

# Load the trained YOLOv8 model
model = YOLO("model/best.pt")

# Open the webcam
cap = cv2.VideoCapture(0)

# Set resolution (optional but improves clarity)
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height

if not cap.isOpened():
    print("Error: Cannot access the webcam")
    exit()

print("Starting webcam... Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run prediction
    results = model(frame)[0]

    # Draw bounding boxes
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf)
        cls = int(box.cls)
        label = f"{model.names[cls]} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("YOLOv8 Webcam Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
