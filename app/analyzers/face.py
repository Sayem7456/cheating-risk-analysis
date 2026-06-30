from dataclasses import dataclass, field

import cv2
import numpy as np

from mediapipe import Image, ImageFormat
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.face_landmarker import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    FaceLandmarkerResult,
)

from app.analyzers.head_pose import HeadPoseEstimator
from app.core.config import settings
from app.schemas.features import FaceFeatures
from app.core.logging import get_logger

logger = get_logger(__name__)

# MediaPipe 468 face landmark indices
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0]


@dataclass
class FrameFaceResult:
    face_detected: bool = False
    face_count: int = 0
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    gaze_direction: str = "unknown"
    left_ear: float = 0.0
    right_ear: float = 0.0
    mar: float = 0.0


class FaceAnalyzer:
    """Per-frame face analysis using MediaPipe FaceLandmarker.

    Processes individual frames and aggregates results into FaceFeatures.
    """

    def __init__(
        self,
        model_path: str | None = None,
        head_pose_estimator: HeadPoseEstimator | None = None,
    ) -> None:
        model = model_path or settings.model_asset_path
        logger.info("loading_face_landmarker_model", path=model)

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model),
            num_faces=5,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self.landmarker = FaceLandmarker.create_from_options(options)
        self.head_pose = head_pose_estimator or HeadPoseEstimator()
        logger.info("face_landmarker_loaded")

    def analyze_frame(self, frame: np.ndarray) -> FrameFaceResult:
        """Run MediaPipe face landmarking on a single BGR frame."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)
        result: FaceLandmarkerResult = self.landmarker.detect(mp_image)

        face_count = len(result.face_landmarks)
        if face_count == 0:
            return FrameFaceResult(face_detected=False, face_count=0)

        landmarks = result.face_landmarks[0]
        h, w = frame.shape[:2]
        pts = np.array([(lm.x * w, lm.y * h) for lm in landmarks])

        left_ear = self._compute_ear(pts, LEFT_EYE)
        right_ear = self._compute_ear(pts, RIGHT_EYE)
        mar = self._compute_mar(pts)
        pose = self.head_pose.estimate(landmarks, w, h)

        return FrameFaceResult(
            face_detected=True,
            face_count=face_count,
            yaw=pose.yaw,
            pitch=pose.pitch,
            roll=pose.roll,
            gaze_direction=pose.gaze_direction,
            left_ear=left_ear,
            right_ear=right_ear,
            mar=mar,
        )

    def aggregate(
        self,
        results: list[FrameFaceResult],
        total_frames: int,
        frame_fps: float = 30.0,
    ) -> FaceFeatures:
        """Aggregate per-frame results into FaceFeatures."""
        if not results:
            return FaceFeatures()

        sec_per_frame = 1.0 / max(frame_fps, 1.0)
        face_present_frames = sum(1 for r in results if r.face_detected)
        face_missing_frames = total_frames - face_present_frames

        face_missing_events = self._count_events(
            [r.face_detected for r in results], target=False
        )

        # Screen attention: gaze is "center" or "down" (looking at keyboard is normal)
        attention_frames = sum(
            1 for r in results
            if r.face_detected and r.gaze_direction in ("center", "down")
        )
        screen_attention_ratio = round(
            attention_frames / max(total_frames, 1), 4
        )

        # Look away: face detected but not looking center
        # Exclude "down" gaze — looking at keyboard is normal typing behavior
        look_away_results = [
            r for r in results
            if r.face_detected and r.gaze_direction not in ("center", "down")
        ]
        look_away_duration = round(
            len(look_away_results) * sec_per_frame, 2
        )

        # Multi-face: face_count > 1
        multi_frames = [r.face_count > 1 for r in results]
        multiple_face_events = self._count_events(multi_frames, target=True)
        multiple_face_duration = round(
            sum(multi_frames) * sec_per_frame, 2
        )
        first_occurrence_timestamp = 0.0
        for i, r in enumerate(results):
            if r.face_count > 1:
                first_occurrence_timestamp = round(
                    i * sec_per_frame, 2
                )
                break

        # Side glances: contiguous sequences of looking left or right
        side_glance_count = self._count_side_glances(results)

        blink_frames = sum(
            1 for r in results
            if r.face_detected and r.left_ear < 0.2 and r.right_ear < 0.2
        )
        blink_rate = round(
            (blink_frames / max(face_present_frames, 1)) * frame_fps, 2
        )

        speaking_frames = sum(
            1 for r in results if r.face_detected and r.mar > 0.6
        )
        speaking_events = self._count_events(
            [r.mar > 0.6 if r.face_detected else False for r in results],
            target=True,
        )

        total_eyes_closed = sum(
            1 for r in results
            if r.face_detected and r.left_ear < 0.15 and r.right_ear < 0.15
        )

        return FaceFeatures(
            face_missing_duration=round(
                face_missing_frames * sec_per_frame, 2
            ),
            multiple_face_events=multiple_face_events,
            multiple_face_duration=multiple_face_duration,
            first_occurrence_timestamp=first_occurrence_timestamp,
            look_away_duration=look_away_duration,
            side_glance_count=side_glance_count,
            screen_attention_ratio=screen_attention_ratio,
            blink_rate=blink_rate,
            eyes_closed_duration=round(
                total_eyes_closed * sec_per_frame, 2
            ),
            speaking_events=speaking_events,
            phone_detected_frames=0,
        )

    def _compute_ear(
        self, pts: np.ndarray, indices: list[int]
    ) -> float:
        """Eye Aspect Ratio: (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)."""
        p1, p2, p3, p4, p5, p6 = pts[indices]
        vertical = (
            np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)
        )
        horizontal = np.linalg.norm(p1 - p4) * 2.0
        return float(round(vertical / max(horizontal, 1e-6), 4))

    def _compute_mar(self, pts: np.ndarray) -> float:
        """Mouth Aspect Ratio: (|p2-p8| + |p3-p7| + |p4-p6|) / (2 * |p1-p5|)."""
        p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p0 = (
            pts[idx] for idx in MOUTH
        )
        vertical = (
            np.linalg.norm(p2 - p8)
            + np.linalg.norm(p3 - p7)
            + np.linalg.norm(p4 - p6)
        )
        horizontal = np.linalg.norm(p1 - p5) * 2.0
        return float(round(vertical / max(horizontal, 1e-6), 4))

    def _count_events(
        self, states: list[bool], target: bool
    ) -> int:
        """Count contiguous sequences of target=True or target=False."""
        count = 0
        in_event = False
        for s in states:
            if s == target and not in_event:
                count += 1
                in_event = True
            elif s != target:
                in_event = False
        return count

    def _count_side_glances(
        self, results: list[FrameFaceResult]
    ) -> int:
        """Count contiguous sequences of looking left or right (excluding up/down)."""
        count = 0
        in_glance = False
        for r in results:
            is_side = r.face_detected and r.gaze_direction in ("left", "right")
            if is_side and not in_glance:
                count += 1
                in_glance = True
            elif not is_side:
                in_glance = False
        return count

    def close(self) -> None:
        self.landmarker.close()
