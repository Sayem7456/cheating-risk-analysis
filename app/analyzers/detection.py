from dataclasses import dataclass, field

import cv2
import numpy as np

from ultralytics import YOLO

from app.core.logging import get_logger

logger = get_logger(__name__)

# COCO class IDs relevant to exam proctoring
COCO_PHONE = 67
COCO_LAPTOP = 63
COCO_BOOK = 73

# Label mapping from COCO ID to our domain labels
# Laptop (COCO 63) is the expected exam device — not tracked as suspicious.
# COCO has no dedicated tablet class; laptops may be misclassified as such.
LABEL_MAP: dict[int, str] = {
    COCO_PHONE: "phone",
    COCO_BOOK: "book",
}

# Objects we care about for feature output
RELEVANT_LABELS = {"phone", "book"}


@dataclass
class DetectedObject:
    label: str
    confidence: float
    bbox: tuple[float, float, float, float]  # xyxy normalized [0,1]


@dataclass
class FrameDetectionResult:
    objects: list[DetectedObject] = field(default_factory=list)

    @property
    def labels(self) -> set[str]:
        return {o.label for o in self.objects}

    def has_label(self, label: str) -> bool:
        return label in self.labels


@dataclass
class DetectionFeatures:
    phone_detected_frames: int = 0
    tablet_detected_frames: int = 0
    book_detected_frames: int = 0


class ObjectDetectionAnalyzer:
    """Per-frame object detection using Ultralytics YOLOv8.

    Detects proctoring-relevant objects (phone, laptop, book) in each frame
    and produces aggregate DetectionFeatures.
    """

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        confidence_threshold: float = 0.55,
        device: str = "cpu",
    ) -> None:
        logger.info(
            "loading_yolo_model",
            model=model_name,
            conf_threshold=confidence_threshold,
            device=device,
        )
        self.model = YOLO(model_name)
        self.confidence_threshold = confidence_threshold
        self.device = device
        logger.info("yolo_model_loaded")

    def detect(self, frame: np.ndarray) -> FrameDetectionResult:
        """Run YOLOv8 inference on a single BGR frame."""
        results = self.model(
            frame,
            conf=self.confidence_threshold,
            device=self.device,
            verbose=False,
        )

        objects: list[DetectedObject] = []
        if not results or not results[0].boxes:
            return FrameDetectionResult(objects=objects)

        boxes = results[0].boxes
        for box in boxes:
            cls_id = int(box.cls[0].item())
            label = LABEL_MAP.get(cls_id)
            if label is None:
                continue
            conf = round(box.conf[0].item(), 4)
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            h, w = frame.shape[:2]
            objects.append(DetectedObject(
                label=label,
                confidence=conf,
                bbox=(x1 / w, y1 / h, x2 / w, y2 / h),
            ))

        return FrameDetectionResult(objects=objects)

    def aggregate(
        self, results: list[FrameDetectionResult]
    ) -> DetectionFeatures:
        """Aggregate per-frame detection results into DetectionFeatures."""
        phone_frames = sum(1 for r in results if r.has_label("phone"))
        book_frames = sum(1 for r in results if r.has_label("book"))
        # Tablet detection is not reliable with COCO classes (laptop/tablet
        # are both class 63). Laptop is the expected exam device, so we
        # exclude it from suspicious detections entirely.
        tablet_frames = 0

        return DetectionFeatures(
            phone_detected_frames=phone_frames,
            tablet_detected_frames=tablet_frames,
            book_detected_frames=book_frames,
        )
