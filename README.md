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

## TODO 

 - show traffic sign
 - predict lane 
 - show warning based on the distance and velocity
 - blind spot monitoring 
