import os
import subprocess
from pathlib import Path

from app.core.logging import get_logger

logger = get_logger(__name__)


class VideoMerger:
    """Merges multiple video chunks into a single video using ffmpeg."""

    def __init__(self, temp_dir: str = "/tmp/cheating-analysis/videos") -> None:
        self.temp_dir = temp_dir

    def merge_chunks(
        self,
        chunk_paths: list[str],
        output_path: str,
    ) -> str:
        """Merge sorted video chunks into a single video file.

        Args:
            chunk_paths: Sorted list of chunk file paths.
            output_path: Path for the merged output file.

        Returns:
            Path to the merged video file.

        Raises:
            FileNotFoundError: If any chunk file is missing.
            RuntimeError: If ffmpeg merge fails.
        """
        if not chunk_paths:
            raise ValueError("No chunks to merge")

        for path in chunk_paths:
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Chunk file not found: {path}")

        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        if len(chunk_paths) == 1:
            import shutil
            shutil.copy2(chunk_paths[0], output_path)
            logger.info(
                "single_chunk_copied",
                source=chunk_paths[0],
                dest=output_path,
            )
            return output_path

        concat_file = str(output_dir / "concat_list.txt")
        with open(concat_file, "w") as f:
            for path in chunk_paths:
                safe_path = path.replace("'", "'\\''")
                f.write(f"file '{safe_path}'\n")

        cmd = [
            "ffmpeg",
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", concat_file,
            "-c", "copy",
            output_path,
        ]

        logger.info(
            "merge_started",
            chunk_count=len(chunk_paths),
            output=output_path,
        )

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"ffmpeg failed (code {result.returncode}): {result.stderr}"
                )
        finally:
            if os.path.exists(concat_file):
                os.remove(concat_file)

        merged_size = os.path.getsize(output_path)
        logger.info(
            "merge_completed",
            output=output_path,
            merged_size=merged_size,
        )
        return output_path

    def delete_chunks(self, chunk_paths: list[str]) -> int:
        """Delete chunk files after merging.

        Returns:
            Number of files deleted.
        """
        deleted = 0
        for path in chunk_paths:
            try:
                if os.path.isfile(path):
                    os.remove(path)
                    deleted += 1
            except OSError as e:
                logger.warning("chunk_delete_failed", path=path, error=str(e))

        if deleted:
            logger.info("chunks_deleted", count=deleted)
        return deleted

    def delete_session_dir(self, session_id: str) -> None:
        """Delete entire session directory including frames."""
        import shutil
        session_dir = Path(self.temp_dir) / session_id
        if session_dir.exists():
            shutil.rmtree(session_dir)
            logger.info("session_dir_deleted", session_id=session_id)
