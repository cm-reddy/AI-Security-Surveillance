"""Unit tests for CameraManager and Stream Resilience Layer."""
import unittest
from pathlib import Path
import numpy as np

from camera_manager import CameraManager, CameraInfo, probe_source, render_disconnected_screen


class TestCameraManager(unittest.TestCase):
    def setUp(self):
        self.mgr = CameraManager()

    def test_camera_manager_initialization(self):
        self.assertGreater(len(self.mgr.cameras), 0)
        self.assertIn(self.mgr.active_camera_id, self.mgr.cameras)

    def test_probe_usb_camera(self):
        alive, msg, res = probe_source("0")
        self.assertIsInstance(alive, bool)

    def test_probe_unreachable_ip(self):
        # Guaranteed unreachable IP on reserved documentation network
        alive, msg, res = probe_source("rtsp://admin:pass@192.0.2.1:554/test", timeout=0.5)
        self.assertFalse(alive)
        self.assertIn("unreachable", msg.lower())

    def test_render_disconnected_screen(self):
        frame = render_disconnected_screen(
            camera_name="Test CCTV",
            source="rtsp://192.168.1.100:554/live",
            status_text="Network down",
            countdown_sec=2.5,
            attempt=3,
            width=640,
            height=360,
        )
        self.assertIsInstance(frame, np.ndarray)
        self.assertEqual(frame.shape, (360, 640, 3))
        self.assertEqual(frame.dtype, np.uint8)

    def test_camera_switching_and_cycling(self):
        initial_id = self.mgr.active_camera_id
        next_id = self.mgr.cycle_next_camera()
        self.assertIn(next_id, self.mgr.cameras)
        summary = self.mgr.get_status_summary()
        self.assertIn("cameras", summary)
        self.assertIn("connected_cameras", summary)
        self.assertIn("total_cameras", summary)


if __name__ == "__main__":
    unittest.main()
