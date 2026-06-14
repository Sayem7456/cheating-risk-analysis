import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2

from app.core.logging import get_logger

logger = get_logger(__name__)

JPEG_QUALITY = 85


@dataclass
class VideoMetadata:
    path: str
    width: int = 0
    height: int = 0
    fps: float = 0.0
    total_frames: int | None = None
    duration_seconds: float | None = None
    codec: str | None = None
    valid: bool = False
    errors: list[str] = field(default_factory=list)


class VideoProcessor:
    """Validates, inspects, and extracts frames from video files.

    Uses OpenCV for stream-based processing — memory efficient,
    suitable for large webm files.
    """

    def validate_video(self, path: str) -> bool:
        """Check that a video file exists, is readable, and has content."""
        if not os.path.isfile(path):
            logger.error("video_file_missing", path=path)
            return False

        if os.path.getsize(path) == 0:
            logger.error("video_file_empty", path=path)
            return False

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            logger.error("video_cannot_open", path=path)
            return False

        ret, _ = cap.read()
        cap.release()

        if not ret:
            logger.error("video_no_frames", path=path)
            return False

        return True

    def extract_metadata(self, path: str) -> VideoMetadata:
        """Extract all metadata from a video file."""
        meta = VideoMetadata(path=path)

        if not os.path.isfile(path):
            meta.errors.append("file_not_found")
            return meta

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            meta.errors.append("cannot_open")
            return meta

        meta.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        meta.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        meta.fps = cap.get(cv2.CAP_PROP_FPS)

        raw_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        if raw_fourcc:
            meta.codec = "".join(
                chr((raw_fourcc >> (8 * i)) & 0xFF) for i in range(4)
            )

        raw_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if raw_total > 0:
            meta.total_frames = raw_total
            if meta.fps > 0:
                meta.duration_seconds = raw_total / meta.fps
        else:
            logger.warning("frame_count_unavailable_counting_manually", path=path)
            manual_count = self._count_frames(cap)
            meta.total_frames = manual_count
            if meta.fps > 0 and manual_count is not None:
                meta.duration_seconds = manual_count / meta.fps

        cap.release()
        meta.valid = True
        return meta

    def extract_frames(
        self,
        path: str,
        output_dir: str,
        fps: int = 1,
    ) -> list[str]:
        """Extract frames at a configurable sampling rate.

        Streams through the video one frame at a time — memory efficient.
        Returns sorted list of frame file paths.
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            logger.error("extract_frames_cannot_open", path=path)
            return []

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps <= 0:
            video_fps = 30.0

        frame_interval = max(1, round(video_fps / fps))

        frames: list[str] = []
        frame_idx = 0
        saved_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_interval == 0:
                frame_path = os.path.join(
                    output_dir, f"frame_{saved_idx:06d}.jpg"
                )
                cv2.imwrite(
                    frame_path, frame,
                    [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
                )
                frames.append(frame_path)
                saved_idx += 1
            frame_idx += 1

        cap.release()
        logger.info(
            "frames_extracted",
            path=path,
            total_frames_read=frame_idx,
            frames_saved=saved_idx,
            target_fps=fps,
        )
        return frames

    def process_video(
        self,
        path: str,
        output_dir: str,
        fps: int = 1,
    ) -> tuple[VideoMetadata, list[str]]:
        """Convenience: validate, extract metadata, and extract frames in one call."""
        meta = self.extract_metadata(path)
        if not meta.valid:
            return meta, []

        frames = self.extract_frames(path, output_dir, fps)
        return meta, frames

    def _count_frames(self, cap: cv2.VideoCapture) -> int | None:
        """Fallback frame counter for files where CAP_PROP_FRAME_COUNT is unreliable."""
        count = 0
        try:
            while True:
                ret, _ = cap.read()
                if not ret:
                    break
                count += 1
                if count > 1_000_000:
                    logger.warning("frame_count_limit_reached")
                    break
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return count
        except Exception:
            return None
