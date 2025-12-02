# ws_sender.py -- sends test telemetry to the viewer
import asyncio, websockets, json, time, math
async def send():
    uri = "ws://localhost:8765"
    async with websockets.connect(uri) as ws:
        for i in range(5000):
            ts = time.time()
            x = math.sin(i*0.02)*5.0
            y = math.cos(i*0.02)*5.0
            yaw = (i*3.0) % 360
            speed = math.hypot(x, y)/1.0
            msg = {'ts': ts, 'x': x, 'y': y, 'yaw': yaw, 'speed': speed}
            await ws.send(json.dumps(msg))
            await asyncio.sleep(0.05)
asyncio.run(send())
