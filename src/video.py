# webcam_detection.py - Live object detection from webcam!
import cv2
from ultralytics import YOLO

# Load model
model = YOLO("latest_model.pt")

# Use webcam (source=0 means default camera)
results = model.predict(
    source=0,  # Webcam
    show=True,  # Show live feed
    stream=True,  # Real-time streaming
    verbose=False,  # Less terminal output
)

print("📹 Starting webcam detection...")
print("Press 'q' to quit")

# Process live stream
for r in results:
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

print("👋 Webcam detection stopped")

