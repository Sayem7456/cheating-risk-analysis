import cv2
import numpy as np

from mediapipe import Image, ImageFormat
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.face_landmarker import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    FaceLandmarkerResult,
)

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class MultiFaceDetector:
    """Detects number of faces in a frame using MediaPipe FaceLandmarker.

    Counts how many faces are visible (0, 1, or multiple) by running
    FaceLandmarker with num_faces > 1.  Designed to be injected as a
    reusable dependency alongside FaceAnalyzer.
    """

    def __init__(
        self,
        model_path: str | None = None,
        max_faces: int = 5,
    ) -> None:
        model = model_path or settings.model_asset_path
        logger.info(
            "loading_multi_face_model",
            path=model,
            max_faces=max_faces,
        )

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model),
            num_faces=max_faces,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self.landmarker = FaceLandmarker.create_from_options(options)
        self.max_faces = max_faces
        logger.info("multi_face_model_loaded")

    def detect(self, frame: np.ndarray) -> int:
        """Return the number of faces detected in a BGR frame."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)
        result: FaceLandmarkerResult = self.landmarker.detect(mp_image)
        return len(result.face_landmarks)

    def close(self) -> None:
        self.landmarker.close()
