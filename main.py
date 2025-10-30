from fastapi import FastAPI
import cv2
from fastapi import UploadFile
import numpy as np
from ultralytics import YOLO

app = FastAPI()


model = YOLO("saved_model/trafic_sign_model.pt")


@app.post("/detect/")
async def detect_objects(file: UploadFile):
    # Process the uploaded image for object detection
    image_bytes = await file.read()
    image = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # Perform object detection with YOLOv8
    detections = model.predict(image)

    return {"detections": detections}
