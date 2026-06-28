import hashlib
import os
from pathlib import Path
from typing import Any

import boto3
from botocore.config import Config

from app.core.logging import get_logger

logger = get_logger(__name__)


class S3VideoDownloader:
    """Downloads ordered face/screen video chunks from S3 with integrity verification."""

    def __init__(
        self,
        access_key_id: str,
        secret_access_key: str,
        region: str,
        bucket: str,
        temp_dir: str = "/tmp/cheating-analysis/videos",
        verify_integrity: bool = True,
    ) -> None:
        self.bucket = bucket
        self.temp_dir = temp_dir
        self.verify_integrity = verify_integrity
        self.client = boto3.client(
            "s3",
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            region_name=region,
            config=Config(connect_timeout=30, read_timeout=120, retries={"max_attempts": 3}),
        )

    async def download_chunks(
        self,
        chunks: list[dict[str, Any]],
        session_id: str,
    ) -> list[str]:
        """Download all video chunks for a session, return local paths sorted by chunk_index.

        Args:
            chunks: List of dicts with 'chunk_index', 's3_key', 'video_url'.
            session_id: Unique session identifier for temp directory isolation.

        Returns:
            Sorted list of local file paths.

        Raises:
            FileNotFoundError: If a chunk fails integrity check after retries.
        """
        if not chunks:
            return []

        session_dir = Path(self.temp_dir) / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        sorted_chunks = sorted(chunks, key=lambda c: c.get("chunk_index", 0))
        local_paths: list[str] = []

        for chunk in sorted_chunks:
            s3_key = chunk.get("s3_key", "")
            chunk_index = chunk.get("chunk_index", 0)
            ext = os.path.splitext(s3_key)[1] or ".webm"
            local_path = str(session_dir / f"chunk_{chunk_index:04d}{ext}")

            existing_etag = self._head_object_etag(s3_key)
            if existing_etag:
                logger.debug("s3_head_ok", key=s3_key, etag=existing_etag)

            logger.info(
                "chunk_download_started",
                key=s3_key,
                chunk_index=chunk_index,
                dest=local_path,
            )

            self.client.download_file(self.bucket, s3_key, local_path)

            self._verify_download(s3_key, local_path)

            logger.info(
                "chunk_download_completed",
                key=s3_key,
                chunk_index=chunk_index,
                size=os.path.getsize(local_path),
            )

            local_paths.append(local_path)

        logger.info(
            "all_chunks_downloaded",
            session_id=session_id,
            count=len(local_paths),
        )
        return local_paths

    def _head_object_etag(self, key: str) -> str | None:
        """Return the S3 object ETag, or None if inaccessible."""
        try:
            response = self.client.head_object(Bucket=self.bucket, Key=key)
            etag = response.get("ETag", "").strip('"')
            return etag or None
        except Exception:
            logger.warning("s3_head_failed", key=key)
            return None

    def _verify_download(self, s3_key: str, local_path: str) -> None:
        """Verify a downloaded file exists, has content, and matches size."""
        if not os.path.isfile(local_path):
            raise FileNotFoundError(f"Downloaded file missing: {local_path}")

        file_size = os.path.getsize(local_path)
        if file_size == 0:
            raise ValueError(f"Downloaded file is empty: {local_path}")

        if not self.verify_integrity:
            return

        s3_size = self._head_object_size(s3_key)
        if s3_size is not None and file_size != s3_size:
            raise ValueError(
                f"Size mismatch for {s3_key}: "
                f"downloaded {file_size} != S3 {s3_size}"
            )

    def _head_object_size(self, key: str) -> int | None:
        """Return the ContentLength of an S3 object."""
        try:
            response = self.client.head_object(Bucket=self.bucket, Key=key)
            return response.get("ContentLength")
        except Exception:
            logger.warning("s3_size_check_failed", key=key)
            return None

    def clean_up(self, session_id: str) -> None:
        """Remove all downloaded files for a session."""
        session_dir = Path(self.temp_dir) / session_id
        if session_dir.exists():
            import shutil
            shutil.rmtree(session_dir)
            logger.info("cleaned_up_session", session_id=session_id)

    def upload_merged_video(
        self,
        local_path: str,
        session_id: str,
        filename: str = "merged_video.webm",
    ) -> str:
        """Upload merged video to S3 and return the S3 key.

        Args:
            local_path: Local path to the merged video file.
            session_id: Session identifier for S3 key prefix.
            filename: Output filename.

        Returns:
            S3 key of the uploaded file.
        """
        s3_key = f"{session_id}/face/{filename}"

        if not os.path.isfile(local_path):
            raise FileNotFoundError(f"Merged video not found: {local_path}")

        file_size = os.path.getsize(local_path)
        logger.info(
            "merged_upload_started",
            local_path=local_path,
            s3_key=s3_key,
            size=file_size,
        )

        self.client.upload_file(local_path, self.bucket, s3_key)

        logger.info(
            "merged_upload_completed",
            s3_key=s3_key,
            size=file_size,
        )
        return s3_key
