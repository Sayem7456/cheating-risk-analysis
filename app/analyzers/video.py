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
from app.utils.video_merger import VideoMerger

logger = get_logger(__name__)


class VideoAnalyzer(BaseAnalyzer):
    """Orchestrates video download, frame extraction, face analysis, and object detection."""

    def __init__(
        self,
        downloader: S3VideoDownloader,
        processor: VideoProcessor | None = None,
        face_analyzer: FaceAnalyzer | None = None,
        object_detector: ObjectDetectionAnalyzer | None = None,
        merger: VideoMerger | None = None,
        fps: int = 1,
    ) -> None:
        self.downloader = downloader
        self.processor = processor or VideoProcessor()
        self.face_analyzer = face_analyzer or FaceAnalyzer()
        self.object_detector = object_detector
        self.merger = merger or VideoMerger(temp_dir=settings.video_temp_dir)
        self.fps = fps

    def _analyze_single_video(
        self, video_path: str, label: str
    ) -> tuple[list[FrameFaceResult], list[FrameDetectionResult], int]:
        """Extract frames from a video and run face + object detection."""
        frame_results: list[FrameFaceResult] = []
        det_results: list[FrameDetectionResult] = []
        count = 0

        if not self.processor.validate_video(video_path):
            logger.warning("invalid_video", path=video_path, label=label)
            return frame_results, det_results, count

        frame_dir = Path(video_path).parent / f"frames_{label}"
        frame_paths = self.processor.extract_frames(
            video_path, str(frame_dir), fps=self.fps
        )
        logger.info(
            "frames_extracted",
            path=video_path,
            label=label,
            frame_count=len(frame_paths),
        )

        for fpath in frame_paths:
            frame = cv2.imread(fpath)
            if frame is None:
                continue
            frame_results.append(self.face_analyzer.analyze_frame(frame))
            if self.object_detector is not None:
                det_results.append(self.object_detector.detect(frame))
            count += 1

        return frame_results, det_results, count

    async def analyze(
        self,
        face_records: list[dict[str, Any]],
        session_id: str | None = None,
    ) -> tuple[FaceFeatures, list[FrameFaceResult], list[FrameDetectionResult], str | None]:
        """Run full video analysis pipeline."""
        if not face_records:
            logger.warning(
                "no_face_records",
                session_id=session_id,
                message="face_records is empty — skipping video analysis",
            )
            return FaceFeatures(), [], [], None

        sid = session_id or "unknown"
        merged_s3_key: str | None = None
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
            return FaceFeatures(), [], [], None

        if not local_paths:
            logger.warning("no_chunks_downloaded", session_id=sid)
            return FaceFeatures(), [], [], None

        logger.info(
            "chunks_downloaded",
            session_id=sid,
            chunk_count=len(local_paths),
            paths=local_paths,
        )

        all_frame_results: list[FrameFaceResult] = []
        detection_results: list[FrameDetectionResult] = []
        total_frames = 0

        # Step 1: Try merge + upload + analyze merged video
        try:
            session_dir = Path(settings.video_temp_dir) / sid
            merged_path = str(session_dir / "merged_video.mp4")
            merged_path = self.merger.merge_chunks(local_paths, merged_path)

            merged_s3_key = self.downloader.upload_merged_video(
                merged_path, sid
            )
            logger.info("merged_video_uploaded", session_id=sid, s3_key=merged_s3_key)

            all_frame_results, detection_results, total_frames = (
                self._analyze_single_video(merged_path, "merged")
            )
            logger.info(
                "merged_analysis_result",
                session_id=sid,
                frames=total_frames,
            )
        except Exception:
            logger.exception("video_merge_or_analyze_failed", session_id=sid)

        # Step 2: If merged analysis produced no frames, analyze chunks directly
        if total_frames == 0:
            logger.info("falling_back_to_chunks", session_id=sid, chunk_count=len(local_paths))
            all_frame_results = []
            detection_results = []
            total_frames = 0

            for i, local_path in enumerate(local_paths):
                fr, dr, count = self._analyze_single_video(local_path, f"chunk_{i}")
                all_frame_results.extend(fr)
                detection_results.extend(dr)
                total_frames += count

            logger.info(
                "chunk_analysis_result",
                session_id=sid,
                frames=total_frames,
            )

        # Clean up
        self.merger.delete_chunks(local_paths)
        self.merger.delete_session_dir(sid)

        logger.info(
            "video_analysis_complete",
            session_id=sid,
            frames_analyzed=total_frames,
            merged_s3_key=merged_s3_key,
        )

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

        return features, all_frame_results, detection_results, merged_s3_key

    def close(self) -> None:
        self.face_analyzer.close()
        if self.object_detector is not None:
            import gc
            gc.collect()
