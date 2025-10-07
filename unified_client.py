import asyncio
import struct
import cv2
import json
import time
from typing import Optional, Dict, Any, Tuple
import argparse


class UnifiedClient:
    def __init__(self, server_ip: str = '127.0.0.1', server_port: int = 12350, camera_source=0, send_ad_id: bool = False):
        self.server_ip = server_ip
        self.server_port = server_port
        self.camera_source = camera_source
        self.send_ad_id = send_ad_id

        self.cap = None
        self.connected = False
        self.connection_retry_delay = 2
        self.display_analytics = True
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.last_fps_value: Optional[float] = None

    def initialize_camera(self) -> bool:
        self.cap = cv2.VideoCapture(self.camera_source)
        if not self.cap.isOpened():
            return False
        # Favor FPS: moderate resolution and small buffer
        try:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        # Warmup read
        ret, frame = self.cap.read()
        return bool(ret and frame is not None)

    def set_camera_source(self, new_source: int) -> bool:
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass
        self.camera_source = new_source
        ok = self.initialize_camera()
        if not ok:
            print(f"Failed to switch to camera index {new_source}")
        else:
            print(f"Switched to camera index {new_source}")
        return ok

    async def connect(self) -> Tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        while True:
            try:
                reader, writer = await asyncio.open_connection(self.server_ip, self.server_port)
                self.connected = True
                return reader, writer
            except Exception:
                self.connected = False
                await asyncio.sleep(self.connection_retry_delay)

    @staticmethod
    def _encode_request(frame, ad_id: Optional[str]) -> bytes:
        # flags
        flags = 0
        payload = bytearray()
        if ad_id:
            flags |= 0b00000001
        payload.extend(bytes([flags]))

        if ad_id:
            ad_bytes = ad_id.encode('utf-8')
            payload.extend(struct.pack('>I', len(ad_bytes)))
            payload.extend(ad_bytes)

        ok, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        if not ok:
            raise RuntimeError('encode_failed')
        data = buf.tobytes()
        payload.extend(struct.pack('>I', len(data)))
        payload.extend(data)
        return bytes(payload)

    @staticmethod
    async def _read_response(reader: asyncio.StreamReader) -> Dict[str, Any]:
        sz = struct.unpack('>I', await reader.readexactly(4))[0]
        blob = await reader.readexactly(sz)
        return json.loads(blob.decode('utf-8'))

    def _draw_overlay(self, frame, analytics: Dict[str, Any]):
        if not analytics:
            return frame
        h, w = frame.shape[:2]
        out = frame.copy()
        panel_w, panel_h = 460, 200
        overlay = out.copy()
        import numpy as np
        cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, out, 0.6, 0, out)

        font = cv2.FONT_HERSHEY_SIMPLEX
        fs = 0.55
        y = 35
        def put(text):
            nonlocal y
            cv2.putText(out, text, (20, y), font, fs, (0, 255, 0), 1)
            y += 22

        # Show client-side averaged FPS in panel
        fps_panel = self._calc_fps()
        if fps_panel is not None:
            put(f"Client FPS(avg): {fps_panel:.1f}")
        else:
            put("Client FPS(avg): --")
        if 'server_avg_fps' in analytics:
            put(f"Server FPS(avg): {analytics.get('server_avg_fps', 0):.1f}")
        put(f"Persons: {analytics.get('total_persons', 0)} | Faces: {analytics.get('total_faces', 0)}")
        if 'unique_tracked_persons' in analytics:
            put(f"Unique tracked persons: {analytics['unique_tracked_persons']}")
        if 'current_tracked_persons' in analytics:
            put(f"Current tracked persons: {analytics['current_tracked_persons']}")
        tg = analytics.get('tracked_gender_counts') or {}
        if tg:
            put(f"Tracked genders M:{tg.get('male',0)} F:{tg.get('female',0)} U:{tg.get('unknown',0)}")

        # Draw FPS (top-right)
        fps = self._calc_fps()
        if fps:
            cv2.putText(out, f"FPS: {fps:.1f}", (w - 120, 28), font, 0.6, (0, 255, 255), 2)
        # Draw tracked person boxes with ID labels
        for tp in analytics.get('tracked_persons', []) or []:
            bx = tp.get('bbox')
            tid = tp.get('id')
            if not bx:
                continue
            x, y, bw, bh = bx
            x1 = int(x * w)
            y1 = int(y * h)
            x2 = int((x + bw) * w)
            y2 = int((y + bh) * h)
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 255), 2)
            if tid is not None:
                label = f"ID {tid}"
                cv2.putText(out, label, (x1, max(0, y1 - 6)), font, 0.5, (0, 200, 255), 1)
        return out

    def _calc_fps(self) -> Optional[float]:
        self.fps_counter += 1
        if self.fps_counter >= 30:
            elapsed = time.time() - self.fps_start_time
            if elapsed > 0:
                fps = self.fps_counter / elapsed
                self.fps_counter = 0
                self.fps_start_time = time.time()
                self.last_fps_value = fps
                return fps
        # return last known value to have a stable readout between updates
        return self.last_fps_value

    async def run(self):
        if not self.initialize_camera():
            print('Failed to initialize camera')
            return

        cv2.namedWindow('Unified Client', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Unified Client', 960, 720)

        reader, writer = None, None

        try:
            while True:
                if not self.connected:
                    reader, writer = await self.connect()

                ret, frame = self.cap.read()
                if not ret:
                    await asyncio.sleep(0.01)
                    continue

                # Optional client-side downscale to sustain FPS
                h, w = frame.shape[:2]
                if max(h, w) > 960:
                    scale = 960.0 / max(h, w)
                    frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

                try:
                    ad_id = None  # extend here if you need ad IDs
                    writer.write(self._encode_request(frame, ad_id))
                    await writer.drain()

                    resp = await asyncio.wait_for(self._read_response(reader), timeout=5.0)
                    analytics = resp.get('analytics') if resp and resp.get('status') == 'success' else None
                except Exception:
                    self.connected = False
                    try:
                        if writer:
                            writer.close()
                            await writer.wait_closed()
                    except Exception:
                        pass
                    continue

                disp = self._draw_overlay(frame, analytics) if (self.display_analytics and analytics) else frame
                cv2.imshow('Unified Client', disp)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('a'):
                    self.display_analytics = not self.display_analytics
                elif key == ord('c'):
                    # Toggle between laptop cam (0) and USB cam (1) by default
                    alt = 1 if self.camera_source == 0 else 0
                    self.set_camera_source(alt)

                await asyncio.sleep(0)  # yield
        finally:
            if writer:
                try:
                    writer.close()
                    await writer.wait_closed()
                except Exception:
                    pass
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()


async def main():
    parser = argparse.ArgumentParser(description='Unified client')
    parser.add_argument('--server-ip', default='127.0.0.1')
    parser.add_argument('--server-port', type=int, default=12350)
    parser.add_argument('--camera', type=int, default=0, help='OpenCV camera index (0=laptop, 1=USB)')
    args = parser.parse_args()

    client = UnifiedClient(server_ip=args.server_ip, server_port=args.server_port, camera_source=args.camera)
    await client.run()


if __name__ == '__main__':
    asyncio.run(main())


