from dataclasses import dataclass

import cv2
import numpy as np

from app.core.logging import get_logger

logger = get_logger(__name__)

# 6-point 3D face model for solvePnP (OpenCV camera convention: Y+ = down, Z+ = forward)
FACE_MODEL_3D = np.array([
    [0.0, 0.0, 0.0],         # Nose tip
    [0.0, 330.0, -65.0],     # Chin
    [-225.0, -170.0, -135.0], # Left eye left corner
    [225.0, -170.0, -135.0],  # Right eye right corner
    [-150.0, 150.0, -125.0],  # Left mouth corner
    [150.0, 150.0, -125.0],   # Right mouth corner
], dtype=np.float64)

# Corresponding MediaPipe landmark indices
FACE_MODEL_2D_IDX = [1, 152, 33, 263, 61, 291]


@dataclass
class HeadPoseResult:
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    gaze_direction: str = "unknown"


class HeadPoseEstimator:
    """Reusable head pose estimator using OpenCV solvePnP + MediaPipe landmarks.

    Estimates yaw/pitch/roll from face landmarks and classifies
    gaze direction (center/left/right/up/down).
    """

    def __init__(
        self,
        yaw_threshold: float = 20.0,
        pitch_up_threshold: float = 5.0,
        pitch_down_threshold: float = 30.0,
    ) -> None:
        self.yaw_threshold = yaw_threshold
        self.pitch_up_threshold = pitch_up_threshold
        self.pitch_down_threshold = pitch_down_threshold

    def estimate(
        self, landmarks, img_w: int, img_h: int
    ) -> HeadPoseResult:
        """Estimate head pose from MediaPipe landmarks.

        Args:
            landmarks: MediaPipe face landmarks (list of NormalizedLandmark).
            img_w: Image width in pixels.
            img_h: Image height in pixels.

        Returns:
            HeadPoseResult with yaw/pitch/roll and gaze direction.
        """
        image_pts = np.array([
            (landmarks[i].x * img_w, landmarks[i].y * img_h)
            for i in FACE_MODEL_2D_IDX
        ], dtype=np.float64)

        focal = img_w
        center = (img_w / 2, img_h / 2)
        camera = np.array([
            [focal, 0, center[0]],
            [0, focal, center[1]],
            [0, 0, 1],
        ], dtype=np.float64)

        dist = np.zeros((4, 1), dtype=np.float64)
        _, rvec, _ = cv2.solvePnP(FACE_MODEL_3D, image_pts, camera, dist)

        rmat, _ = cv2.Rodrigues(rvec)
        yaw, pitch, roll = self._rotation_matrix_to_euler(rmat)
        gaze = self._classify_gaze(yaw, pitch)

        return HeadPoseResult(
            yaw=yaw, pitch=pitch, roll=roll, gaze_direction=gaze
        )

    def _classify_gaze(self, yaw: float, pitch: float) -> str:
        """Classify gaze direction based on yaw and pitch thresholds."""
        if abs(yaw) > self.yaw_threshold:
            return "left" if yaw < 0 else "right"
        if pitch < self.pitch_up_threshold:
            return "up"
        if pitch > self.pitch_down_threshold:
            return "down"
        return "center"

    def _rotation_matrix_to_euler(
        self, rmat: np.ndarray
    ) -> tuple[float, float, float]:
        """Convert rotation matrix to yaw, pitch, roll in degrees (ZYX convention)."""
        sy = np.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
        singular = sy < 1e-6

        if not singular:
            x = np.arctan2(rmat[2, 1], rmat[2, 2])
            y = np.arctan2(-rmat[2, 0], sy)
            z = np.arctan2(rmat[1, 0], rmat[0, 0])
        else:
            x = np.arctan2(-rmat[1, 2], rmat[1, 1])
            y = np.arctan2(-rmat[2, 0], sy)
            z = 0

        yaw, pitch, roll = (
            round(np.degrees(y), 2),
            round(np.degrees(x), 2),
            round(np.degrees(z), 2),
        )

        if roll > 90:
            roll -= 180
        elif roll < -90:
            roll += 180

        return yaw, pitch, roll
