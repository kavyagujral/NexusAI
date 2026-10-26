import os
import shutil
from pathlib import Path
from typing import Tuple
import aiofiles
from fastapi import UploadFile, HTTPException, status
from backend.config import settings
from backend.utils.logger import logger


def ensure_directories():
    """Ensure all required local storage directories exist."""
    dirs = [
        settings.resolve_path(settings.STORAGE_DIR),
        settings.resolve_path(settings.VIDEOS_DIR),
        settings.resolve_path(settings.FRAMES_DIR),
        settings.resolve_path(settings.AUDIO_DIR),
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Storage directories initialized at {settings.STORAGE_DIR}")


async def save_uploaded_file(file: UploadFile, video_id: str) -> Tuple[Path, int]:
    """
    Asynchronously streams and saves an uploaded video file to the disk.
    Performs extension validation and enforces maximum upload size.
    
    Returns:
        (saved_path, file_size_bytes)
    """
    ensure_directories()
    
    # Validate extension
    ext = Path(file.filename or "").suffix.lower()
    if ext not in settings.allowed_extensions_list:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{ext}'. Allowed formats: {settings.ALLOWED_VIDEO_EXTENSIONS}"
        )
    
    dest_path = settings.resolve_path(settings.VIDEOS_DIR) / f"{video_id}{ext}"
    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    total_bytes = 0

    try:
        async with aiofiles.open(dest_path, "wb") as buffer:
            while chunk := await file.read(1024 * 1024):  # 1MB chunks
                total_bytes += len(chunk)
                if total_bytes > max_bytes:
                    if dest_path.exists():
                        dest_path.unlink()
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"File exceeds maximum allowed size of {settings.MAX_UPLOAD_SIZE_MB}MB."
                    )
                await buffer.write(chunk)
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        logger.error(f"Failed to save uploaded file {file.filename}: {str(e)}")
        if dest_path.exists():
            dest_path.unlink()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save uploaded video: {str(e)}"
        )

    logger.info(f"Successfully saved video [{video_id}] to {dest_path} ({total_bytes / (1024*1024):.2f} MB)")
    return dest_path, total_bytes
