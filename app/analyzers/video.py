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

    async def analyze(
        self,
        face_records: list[dict[str, Any]],
        session_id: str | None = None,
    ) -> tuple[FaceFeatures, list[FrameFaceResult], list[FrameDetectionResult], str | None]:
        """Run full video analysis pipeline.

        Returns:
            Tuple of (aggregated FaceFeatures, per-frame face results, per-frame detection results, merged_video_s3_key).
        """
        if not face_records:
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

        merged_path = None
        try:
            # Merge chunks into single video
            session_dir = Path(settings.video_temp_dir) / sid
            merged_path = str(session_dir / "merged_video.webm")
            merged_path = self.merger.merge_chunks(local_paths, merged_path)

            # Upload merged video to S3
            merged_s3_key = self.downloader.upload_merged_video(
                merged_path, sid
            )
            logger.info(
                "merged_video_uploaded",
                session_id=sid,
                s3_key=merged_s3_key,
            )

            # Delete chunk files
            self.merger.delete_chunks(local_paths)

        except Exception:
            logger.exception("video_merge_upload_failed", session_id=sid)
            # Continue with analysis even if merge/upload fails

        all_frame_results: list[FrameFaceResult] = []
        detection_results: list[FrameDetectionResult] = []
        total_frames = 0

        # Analyze the merged video instead of individual chunks
        video_to_analyze = merged_path if merged_path and Path(merged_path).exists() else None

        if video_to_analyze:
            if self.processor.validate_video(video_to_analyze):
                frame_dir = Path(video_to_analyze).parent / "frames"
                frame_paths = self.processor.extract_frames(
                    video_to_analyze, str(frame_dir), fps=self.fps
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
        else:
            # Fallback to individual chunks if merge failed
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

        # Clean up session directory
        self.merger.delete_session_dir(sid)

        logger.info(
            "video_analysis_complete",
            session_id=sid,
            frames_analyzed=total_frames,
            used_merged_video=video_to_analyze is not None,
            merged_s3_key=merged_s3_key,
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

        return features, all_frame_results, detection_results, merged_s3_key

    def close(self) -> None:
        self.face_analyzer.close()
        if self.object_detector is not None:
            import gc
            gc.collect()
