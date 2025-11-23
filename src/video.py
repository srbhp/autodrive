import cv2
from ultralytics import YOLO

# Load model
model = YOLO("saved_model/latest_model.pt")

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
    # Check if any detections exist
    if r is not None:
        print( r.probs)  # Access predictions directly
    else:
        print("No results returned from the model.")

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

print("👋 Webcam detection stopped")
