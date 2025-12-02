# app.py -- viewer + detection + simple depth estimator
import asyncio, json, time, math, threading
from pathlib import Path
import pyglet
import moderngl_window as mglw
from moderngl_window import geometry
from pyglm import vec3, mat4
import websockets
import queue

# New imports for detection
import cv2
from ultralytics import YOLO
try:
    from picamera2 import Picamera2  # optional for Pi camera
    PICAMERA2_AVAILABLE = True
except Exception:
    PICAMERA2_AVAILABLE = False

# Constants
WS_PORT = 8765
ASSETS = Path(__file__).parent / "assets"

# Camera / detection params (tune for your camera)
USE_PICAMERA2 = False  # set True if you want Picamera2 input
CAM_WIDTH, CAM_HEIGHT = 1280, 720
DETECT_IMGSZ = 640   # yolov11 imgsz (lower -> faster)
YOLO_MODEL = "yolo11n.pt"  # choose a small model for Pi
FOCAL_PIXELS = 800.0  # approximate focal length in pixels (calibrate)
REAL_CAR_WIDTH = 1.8  # meters; used for distance est. for class 'car'. Adjust if needed.

# Thread-safe queues
frame_q = queue.Queue(maxsize=2)
detections_q = queue.Queue(maxsize=4)

# Simple camera grabber (Picamera2 if available else OpenCV)
def camera_grabber():
    if USE_PICAMERA2 and PICAMERA2_AVAILABLE:
        picam2 = Picamera2()
        picam2.preview_configuration.main.size = (CAM_WIDTH, CAM_HEIGHT)
        picam2.preview_configuration.main.format = "RGB888"
        picam2.preview_configuration.align()
        picam2.configure("preview")
        picam2.start()
        while True:
            frame = picam2.capture_array()
            if not frame_q.full():
                frame_q.put(frame)
    else:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue
            # convert BGR->RGB for ultralytics
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if not frame_q.full():
                frame_q.put(frame_rgb)

# Simple monocular distance estimate using object pixel width
def bbox_to_distance(pixel_w, class_name="car"):
    if pixel_w <= 0:
        return None
    # choose real width based on class
    real_w = REAL_CAR_WIDTH if class_name == "car" else 0.5
    return (real_w * FOCAL_PIXELS) / pixel_w

# Detection worker using Ultralytics YOLO
def detection_worker():
    model = YOLO(YOLO_MODEL)
    while True:
        try:
            frame = frame_q.get()
        except Exception:
            continue
        # run inference (synchronous)
        results = model(frame, imgsz=DETECT_IMGSZ, conf=0.35, classes=None, device='cpu')
        det_list = []
        # results can be a list-like; process first item
        r = results[0]
        boxes = getattr(r, 'boxes', None)
        if boxes is None:
            # No detections
            detections_q.put([])
            continue
        for b in boxes:
            # b.xyxy, b.conf, b.cls
            xyxy = b.xyxy[0].cpu().numpy() if hasattr(b.xyxy[0], 'cpu') else b.xyxy[0]
            x1, y1, x2, y2 = map(float, xyxy)
            pixel_w = x2 - x1
            cls_idx = int(b.cls[0]) if hasattr(b, 'cls') else int(b.cls)
            conf = float(b.conf[0]) if hasattr(b, 'conf') else float(b.conf)
            class_name = model.names.get(cls_idx, str(cls_idx))
            dist = bbox_to_distance(pixel_w, class_name=class_name)  # meters
            # compute bearing relative to camera center
            cx = CAM_WIDTH / 2.0
            u = (x1 + x2) / 2.0
            angle = math.atan2((u - cx), FOCAL_PIXELS)  # radians horizontal
            # camera coords: x (right), y (up), z (forward)
            x_cam = dist * math.sin(angle) if dist else 0.0
            z_cam = dist * math.cos(angle) if dist else 0.0
            det = {
                'class': class_name,
                'conf': conf,
                'bbox': [x1, y1, x2, y2],
                'dist': dist,
                'cam_x': x_cam,
                'cam_z': z_cam,
                'ts': time.time()
            }
            det_list.append(det)
        # push detections for viewer
        try:
            if detections_q.full():
                _ = detections_q.get_nowait()
            detections_q.put(det_list)
        except Exception:
            pass

# Minimal renderable object class
class TrackedObject:
    def __init__(self, obj_id, cls, pos, dist, ts):
        self.id = obj_id
        self.cls = cls
        self.pos = pos  # vec3
        self.dist = dist
        self.last_ts = ts
        self.model = None  # placeholder geometry will be assigned externally

# Viewer (extended)
class Viewer(mglw.WindowConfig):
    gl_version = (3, 3)
    title = "Pi 3D Car Viewer + Detection"
    window_size = (1920, 1080)
    resource_dir = (Path(__file__).parent / "assets").as_posix()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ctx.enable_only(self.ctx.DEPTH_TEST | self.ctx.CULL_FACE)
        # load shaders explicitly (avoid resource_dir issues)
        vs = (Path(__file__).parent / "shaders" / "simple_vert.glsl").read_text()
        fs = (Path(__file__).parent / "shaders" / "simple_frag.glsl").read_text()
        self.prog = self.ctx.program(vertex_shader=vs, fragment_shader=fs)
        # simple box geometry as car placeholder
        self.box = geometry.cube(size=(1.0, 0.5, 2.0))
        # camera
        self.camera_pos = vec3(0.0, 5.0, 10.0)
        self.target = vec3(0.0, 0.0, 0.0)
        # pose interpolation (for ego vehicle)
        now = time.time()
        self.prev_pose = {'t': now, 'pos': vec3(0.0,0.0,0.0), 'yaw': 0.0, 'speed':0.0}
        self.next_pose = dict(self.prev_pose)
        # HUD label (pyglet)
        self.hud = pyglet.text.Label('', font_size=18, x=10, y=self.window_size[1]-30, color=(255,255,255,255))
        # tracked objects dict
        self.tracked = {}  # id -> TrackedObject
        self.next_obj_id = 1
        # start camera grabber and detection threads
        threading.Thread(target=camera_grabber, daemon=True).start()
        threading.Thread(target=detection_worker, daemon=True).start()
        # optional websocket server (unchanged from earlier) to receive external telemetry
        threading.Thread(target=lambda: asyncio.run(self.ws_server()), daemon=True).start()

    # websocket handler unchanged
    async def ws_handler(self, websocket, path):
        async for msg in websocket:
            try:
                d = json.loads(msg)
                pose = {'t': d.get('ts', time.time()), 'pos': vec3(d.get('x',0.0), 0.0, d.get('y',0.0)),
                        'yaw': math.radians(d.get('yaw',0.0)), 'speed': d.get('speed',0.0)}
                self.prev_pose = self.next_pose
                self.next_pose = pose
            except Exception as e:
                print("ws parse err", e)

    async def ws_server(self):
        async with websockets.serve(self.ws_handler, "0.0.0.0", WS_PORT):
            await asyncio.Future()

    def render(self, time_delta, frame_time):
        self.ctx.clear(0.08, 0.08, 0.1)
        # Interpolate ego pose
        now = time.time()
        dt = self.next_pose['t'] - self.prev_pose['t'] if (self.next_pose['t'] - self.prev_pose['t'])!=0 else 0.05
        alpha = max(0.0, min(1.0, (now - self.prev_pose['t']) / dt))
        ego_pos = self.prev_pose['pos'] * (1-alpha) + self.next_pose['pos'] * alpha
        ego_yaw = self.prev_pose['yaw'] * (1-alpha) + self.next_pose['yaw'] * alpha
        # Build MVP
        proj = mat4.perspective(45.0, self.wnd.aspect_ratio, 0.1, 100.0)
        cam = mat4.look_at(self.camera_pos, self.target, vec3(0.0,1.0,0.0))
        # Render tracked detections (update from detections_q)
        try:
            dets = detections_q.get_nowait()
            self._update_tracked_from_detections(dets)
        except queue.Empty:
            pass
        # draw each tracked object as a box at its pos (world coords: x,z)
        for obj in list(self.tracked.values()):
            # simple expiration: remove if older than 2s
            if time.time() - obj.last_ts > 2.0:
                del self.tracked[obj.id]
                continue
            model_mat = mat4.translate(obj.pos) * mat4.scale(1.0, 1.0, 1.0) * mat4.rotate(0.0, vec3(0,1,0))
            self.prog['m_proj'].write(proj.to_bytes())
            self.prog['m_cam'].write(cam.to_bytes())
            self.prog['m_model'].write(model_mat.to_bytes())
            self.box.render(self.prog)
        # HUD: show count and optionally the nearest object distance
        nearest = min((o.dist for o in self.tracked.values()), default=0.0)
        self.hud.text = f"Tracked: {len(self.tracked)}    Nearest: {nearest:.2f} m"
        self.hud.draw()

    def _update_tracked_from_detections(self, detections):
        # Simple tracker: create/update object per detection; no re-identification
        for d in detections:
            # Convert camera coords to world coords relative to ego (simple transform: place on ground plane)
            x_cam = d['cam_x']
            z_cam = d['cam_z']
            # Place objects in world ahead of ego. Here ego at origin; if ego moves, apply transform.
            world_x = float(x_cam)
            world_z = float(z_cam)
            pos = vec3(world_x, 0.0, world_z)
            obj = TrackedObject(self.next_obj_id, d['class'], pos, d['dist'], d['ts'])
            obj.model = self.box
            self.tracked[self.next_obj_id] = obj
            self.next_obj_id += 1

if __name__ == "__main__":
    mglw.run_window_config(Viewer)

