# app.py -- minimal 3D car viewer with HUD and WebSocket telemetry
import asyncio
import json
import math
import threading
import time
from pathlib import Path

import moderngl_window as mglw
import pyglet
import pyglm.glm as glm
import websockets
from moderngl_window import geometry
from pyglm.glm import vec3

WS_PORT = 8765
ASSETS = Path(__file__).parent / "assets"


def load_simple_gltf(ctx, path):
    # Placeholder: return a simple cube geometry as car
    return geometry.cube(size=(1.0, 0.5, 2.0))


class Viewer(mglw.WindowConfig):
    gl_version = (3, 3)
    title = "Pi 3D Car Viewer"
    window_size = (1920, 1080)
    # Use package dir as resource_dir so both `assets/` and `shaders/` are available
    resource_dir = Path(__file__).parent.as_posix()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ctx.enable_only(self.ctx.DEPTH_TEST | self.ctx.CULL_FACE)
        self.prog = self.load_program(
            vertex_shader="shaders/simple_vert.glsl",
            fragment_shader="shaders/simple_flag.glsl",
        )
        self.car = load_simple_gltf(self.wnd.ctx, ASSETS / "car.gltf")
        self.car_prog = self.prog
        # Camera
        self.camera_pos = vec3(0.0, 5.0, 10.0)
        self.target = vec3(0.0, 0.0, 0.0)
        # Pose interpolation
        now = time.time()
        self.prev_pose = {
            "t": now,
            "pos": vec3(0.0, 0.0, 0.0),
            "yaw": 0.0,
            "speed": 0.0,
        }
        self.next_pose = dict(self.prev_pose)
        # HUD label (pyglet)
        self.hud = pyglet.text.Label(
            "",
            font_size=18,
            x=10,
            y=self.window_size[1] - 30,
            color=(255, 255, 255, 255),
        )
        # Start websocket receiver in background thread
        threading.Thread(
            target=lambda: asyncio.run(self.ws_server()), daemon=True
        ).start()

    async def ws_handler(self, websocket, path):
        async for msg in websocket:
            try:
                d = json.loads(msg)
                pose = {
                    "t": d.get("ts", time.time()),
                    "pos": vec3(d.get("x", 0.0), 0.0, d.get("y", 0.0)),
                    "yaw": math.radians(d.get("yaw", 0.0)),
                    "speed": d.get("speed", 0.0),
                }
                self.prev_pose = self.next_pose
                self.next_pose = pose
            except Exception as e:
                print("ws parse err", e)

    async def ws_server(self):
        async with websockets.serve(self.ws_handler, "0.0.0.0", WS_PORT):
            await asyncio.Future()

    def render(self, time_delta, frame_time):
        self.ctx.clear(0.08, 0.08, 0.1)
        now = time.time()
        dt = (
            self.next_pose["t"] - self.prev_pose["t"]
            if (self.next_pose["t"] - self.prev_pose["t"]) != 0
            else 0.05
        )
        alpha = max(0.0, min(1.0, (now - self.prev_pose["t"]) / dt))
        p = self.prev_pose["pos"] * (1 - alpha) + self.next_pose["pos"] * alpha
        yaw = self.prev_pose["yaw"] * (1 - alpha) + self.next_pose["yaw"] * alpha
        proj = glm.perspective(45.0, self.wnd.aspect_ratio, 0.1, 100.0)
        cam = glm.lookAt(self.camera_pos, self.target, vec3(0.0, 1.0, 0.0))
        model = glm.translate(p) * glm.rotate(yaw, vec3(0.0, 1.0, 0.0))
        self.car_prog["m_proj"].write(proj.to_bytes())
        self.car_prog["m_cam"].write(cam.to_bytes())
        self.car_prog["m_model"].write(model.to_bytes())
        # set car color (shader expects `u_color`)
        try:
            self.car_prog["u_color"].value = (0.9, 0.1, 0.1, 1.0)
        except Exception:
            pass
        self.car.render(self.car_prog)
        speed = self.next_pose.get("speed", 0.0)
        yaw_deg = math.degrees(yaw)
        self.hud.text = f"Speed: {speed:.1f} m/s    Heading: {yaw_deg:.1f}°"
        self.hud.draw()

    # moderngl-window (newer versions) calls `on_render` on WindowConfig.
    # Provide a compatibility wrapper so both old and new APIs work:
    def on_render(self, time_delta, frame_time):
        """Compatibility wrapper for moderngl_window.WindowConfig.on_render.

        Delegates to the existing `render` implementation so older code still
        works while satisfying WindowConfig's required method.
        """
        return self.render(time_delta, frame_time)


if __name__ == "__main__":
    mglw.run_window_config(Viewer)
