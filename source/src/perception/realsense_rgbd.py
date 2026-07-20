from __future__ import annotations

import pyrealsense2 as rs
import numpy as np


class RealSenseRGBDCamera:
    """RGBD RealSense wrapper with depth filtering and auto resolution fallback."""

    def __init__(self, width: int = 1280, height: int = 720, fps: int = 30, clip_max: float = 3.0):
        self.width = width
        self.height = height
        self.fps = fps
        self.clip_max = clip_max

        self.pipeline = None
        self.align = None
        self.intrinsics = None
        self.depth_scale = None
        self.last_color_frame = None
        self.last_depth_frame = None

        self.decimation = rs.decimation_filter()
        self.decimation.set_option(rs.option.filter_magnitude, 1)

        self.spatial = rs.spatial_filter()
        self.spatial.set_option(rs.option.filter_magnitude, 2)
        self.spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
        self.spatial.set_option(rs.option.filter_smooth_delta, 20)

        self.temporal = rs.temporal_filter()
        self.hole_filling = rs.hole_filling_filter()

    def start(self) -> None:
        ctx = rs.context()
        devices = ctx.query_devices()
        if len(devices) == 0:
            raise RuntimeError("No RealSense device detected")

        candidates = [
            (self.width, self.height, self.fps),
            (640, 480, 30),
            (848, 480, 30),
            (640, 360, 30),
        ]
        last_error = None

        for width, height, fps in candidates:
            try:
                self.pipeline = rs.pipeline()
                config = rs.config()
                config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
                config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
                profile = self.pipeline.start(config)

                self.width = width
                self.height = height
                self.fps = fps

                depth_sensor = profile.get_device().first_depth_sensor()
                self.depth_scale = depth_sensor.get_depth_scale()
                self.align = rs.align(rs.stream.color)

                for _ in range(30):
                    self.pipeline.wait_for_frames()

                color_profile = profile.get_stream(rs.stream.color)
                self.intrinsics = color_profile.as_video_stream_profile().get_intrinsics()
                return
            except Exception as exc:
                last_error = exc
                try:
                    if self.pipeline is not None:
                        self.pipeline.stop()
                except Exception:
                    pass
                self.pipeline = None
                self.align = None

        raise RuntimeError(f"RealSense camera start failed. Last error: {last_error}")

    def get_frames(self) -> tuple[bool, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        if self.pipeline is None:
            raise RuntimeError("Camera has not been started. Call start() first.")

        frames = self.pipeline.wait_for_frames()
        aligned = self.align.process(frames)
        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()
        if not depth_frame or not color_frame:
            return False, None, None, None

        depth_frame = self.decimation.process(depth_frame)
        depth_frame = self.spatial.process(depth_frame)
        depth_frame = self.temporal.process(depth_frame)
        depth_frame = self.hole_filling.process(depth_frame)

        self.last_color_frame = color_frame
        self.last_depth_frame = depth_frame

        if self.intrinsics is None:
            self.intrinsics = color_frame.profile.as_video_stream_profile().intrinsics

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        depth_meters = depth_image.astype(np.float32) * self.depth_scale
        return True, color_image, depth_image, depth_meters

    def get_last_aligned_frames(self):
        return self.last_depth_frame, self.last_color_frame

    def release(self) -> None:
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except Exception:
                pass
            finally:
                self.pipeline = None
                self.last_color_frame = None
                self.last_depth_frame = None
