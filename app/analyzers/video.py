from pathlib import Path
from typing import Any

import cv2

from app.analyzers.base import BaseAnalyzer
from app.analyzers.detection import (
    DetectionFeatures,
    FrameDetectionResult,
    ObjectDetectionAnalyzer,
)
from app.analyzers.face import FaceAnalyzer, FrameFaceResult
from app.core.config import settings
from app.schemas.features import FaceFeatures
from app.core.logging import get_logger
from app.utils.s3_downloader import S3VideoDownloader
from app.utils.video import VideoProcessor

logger = get_logger(__name__)


class VideoAnalyzer(BaseAnalyzer):
    """Orchestrates video download, frame extraction, face analysis, and object detection."""

    def __init__(
        self,
        downloader: S3VideoDownloader,
        processor: VideoProcessor | None = None,
        face_analyzer: FaceAnalyzer | None = None,
        object_detector: ObjectDetectionAnalyzer | None = None,
        fps: int = 1,
    ) -> None:
        self.downloader = downloader
        self.processor = processor or VideoProcessor()
        self.face_analyzer = face_analyzer or FaceAnalyzer()
        self.object_detector = object_detector
        self.fps = fps

    async def analyze(
        self,
        face_records: list[dict[str, Any]],
        session_id: str | None = None,
    ) -> tuple[FaceFeatures, list[FrameFaceResult], list[FrameDetectionResult]]:
        """Run full video analysis pipeline.

        Returns:
            Tuple of (aggregated FaceFeatures, per-frame face results, per-frame detection results).
        """
        if not face_records:
            return FaceFeatures(), [], []

        sid = session_id or "unknown"
        logger.info(
            "video_analysis_started",
            url_count=len(face_records),
            session_id=sid,
        )

        try:
            local_paths = await self.downloader.download_chunks(
                face_records, sid
            )
        except Exception:
            logger.exception("video_download_failed", session_id=sid)
            return FaceFeatures(), [], []

        all_frame_results: list[FrameFaceResult] = []
        detection_results: list[FrameDetectionResult] = []
        total_frames = 0

        for local_path in local_paths:
            if not self.processor.validate_video(local_path):
                logger.warning("skipping_invalid_chunk", path=local_path)
                continue

            frame_dir = Path(local_path).parent / "frames"
            frame_paths = self.processor.extract_frames(
                local_path, str(frame_dir), fps=self.fps
            )

            for fpath in frame_paths:
                frame = cv2.imread(fpath)
                if frame is None:
                    continue
                result = self.face_analyzer.analyze_frame(frame)
                all_frame_results.append(result)

                if self.object_detector is not None:
                    det_result = self.object_detector.detect(frame)
                    detection_results.append(det_result)

                total_frames += 1

        logger.info(
            "video_analysis_complete",
            session_id=sid,
            frames_analyzed=total_frames,
            chunks_processed=len(local_paths),
        )

        # Use analysis FPS (self.fps) so each analysis frame counts correctly
        features = self.face_analyzer.aggregate(
            all_frame_results,
            total_frames=total_frames or len(all_frame_results),
            frame_fps=float(self.fps),
        )

        if detection_results:
            det_features = self.object_detector.aggregate(detection_results)
            features.phone_detected_frames = det_features.phone_detected_frames
            features.tablet_detected_frames = det_features.tablet_detected_frames
            features.book_detected_frames = det_features.book_detected_frames

        return features, all_frame_results, detection_results

    def close(self) -> None:
        self.face_analyzer.close()
        if self.object_detector is not None:
            import gc
            gc.collect()
