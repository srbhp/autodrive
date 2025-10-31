from fastapi import FastAPI, UploadFile, File
import numpy as np
import cv2
from ultralytics import YOLO
import io
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the YOLO model
model = YOLO("saved_model/trafic_sign_model.pt")


@app.post("/detect/")
async def detect_objects(file: UploadFile = File(...)):
    # Read the uploaded file
    image_bytes = await file.read()

    # Convert bytes to NumPy array
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)

    # Decode the image
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    # Perform object detection
    results = model.predict(image, save=True)
    for result in results:
        result_image = result.plot()

    # Encode image to JPEG
    _, buffer = cv2.imencode(".jpg", result_image)
    image_bytes = io.BytesIO(buffer)

    return StreamingResponse(image_bytes, media_type="image/jpeg")
