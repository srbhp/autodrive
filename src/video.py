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
    if r is not  None:
        # If this is a detection model, results usually have `boxes` with `cls` and `conf`.
        if hasattr(r, "boxes") and r.boxes is not None and len(r.boxes) > 0:
            # r.boxes.cls and r.boxes.conf may be torch tensors; convert if necessary
            try:
                cls_ids = r.boxes.cls.cpu().numpy()
            except Exception:
                cls_ids = r.boxes.cls
            try:
                confs = r.boxes.conf.cpu().numpy()
            except Exception:
                confs = r.boxes.conf

            # `r.names` maps class ids to readable labels (dict or list)
            for cls_id, conf in zip(cls_ids, confs):
                cls_id = int(cls_id)
                # Support both dict and list-style `r.names`
                try:
                    name = r.names[cls_id]
                except Exception:
                    name = r.names.get(cls_id, str(cls_id)) if isinstance(r.names, dict) else str(cls_id)
                if conf >= 0.75:  # Only print predictions with confidence >= 0.5
                    print(f"Predicted: {name} (class {cls_id}) — confidence: {float(conf):.3f}")
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

print("👋 Webcam detection stopped")
