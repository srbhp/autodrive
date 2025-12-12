# autodrive

## detect traffic sign

 `python src/video.py`

## resources

- https://github.com/lyft/nuscenes-devkit
- Nvidia Data: https://github.com/NVlabs/physical_ai_av 

## Lane Demo (OpenCV, training-free)

There is a minimal demo for lane detection using classic OpenCV methods and
an optional pre-trained segmentation model. It can be run without deep
learning dependencies, or with `torch`/`torchvision` for an optional mask.

Run the demo (webcam):

```bash
python src/lane_demo.py --source 0 --display
```

Run the demo (video file) and save the output:

```bash
python src/lane_demo.py --source ./assets/input.mp4 --display --save out.mp4
```

Use the optional pre-trained segmentation model via torch/torchvision:

```bash
python -m pip install torch torchvision
python src/lane_demo.py --source 0 --use-torch --display
```

Use YOLOv8n for detection and bird's-eye transform (recommended if you have a
YOLO model trained for lanes or drivable edges):

```bash
python -m pip install ultralytics
python src/lane_demo.py --source 0 --use-yolo --display
```
Tips to reduce inference latency:

- Lower the YOLO image size: use `--yolo-imgsz 384` or `320` to reduce compute.
- Use a GPU and `ultralytics` with `torch` installed; the demo will move model to CUDA automatically.
- For embedded deployment (Jetson), convert the model to ONNX and TensorRT or export to OpenVINO for Intel devices.

Example low-latency run (GPU):

```bash
python -m pip install ultralytics torch
python src/lane_demo.py --source 0 --use-yolo --yolo-imgsz 384 --display
```

Lane departure warnings:

- `--offset-threshold`: lateral offset in meters to trigger a warning (default 0.4m)
- `--frames-to-trigger`: consecutive frames above threshold to trigger (default 3)
- `--beep`: optionally play a terminal beep when a warning is activated

Example:

```bash
python src/lane_demo.py --source 0 --use-yolo --yolo-imgsz 384 --offset-threshold 0.35 --frames-to-trigger 5 --beep --display
```

If you have a custom YOLO weights file trained to detect lanes or drivable edges, pass it via `--yolo-weights`.

For lower latency, reduce the YOLO inference size to 384 or 320 and use a GPU:

```bash
python src/lane_demo.py --source 0 --use-yolo --yolo-imgsz 384 --display
```

```bash
python src/lane_demo.py --source 0 --use-yolo --yolo-weights path/to/weights.pt --display
```
```

The script uses the OpenCV pipeline implemented in `src/ui/road_lanes.py`,
and will fall back to the classical pipeline if the segmentation model or
`torch` isn't available.

