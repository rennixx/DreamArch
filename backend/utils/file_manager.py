"""
File management utilities for Dream Architect
"""

import os
import uuid
from pathlib import Path
import shutil
from datetime import datetime, timedelta


def generate_job_id() -> str:
    """Generate unique job ID"""
    return str(uuid.uuid4())


def get_upload_path(job_id: str, extension: str = "webm") -> str:
    """Get path for uploaded audio file"""
    return f"uploads/{job_id}.{extension}"


def get_output_path(output_type: str, job_id: str, extension: str = "wav") -> str:
    """
    Get path for output file

    Args:
        output_type: Type of output (midi, instrumental, vocals, mixed, final)
        job_id: Unique job identifier
        extension: File extension

    Returns:
        Path to output file
    """
    return f"outputs/{output_type}/{job_id}.{extension}"


def ensure_directories():
    """Ensure all required directories exist"""
    dirs = [
        "uploads",
        "outputs/midi",
        "outputs/instrumental",
        "outputs/vocals",
        "outputs/mixed",
        "outputs/final"
    ]

    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)


def cleanup_old_files(max_age_hours: int = 24):
    """
    Clean up files older than specified age

    Args:
        max_age_hours: Maximum age in hours before deletion
    """
    cutoff = datetime.now() - timedelta(hours=max_age_hours)

    directories = [
        "uploads",
        "outputs/midi",
        "outputs/instrumental",
        "outputs/vocals",
        "outputs/mixed",
        "outputs/final"
    ]

    for directory in directories:
        if not Path(directory).exists():
            continue

        for file_path in Path(directory).iterdir():
            if file_path.is_file():
                file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                if file_time < cutoff:
                    try:
                        file_path.unlink()
                    except Exception as e:
                        print(f"Error deleting {file_path}: {e}")


def get_file_size(file_path: str) -> int:
    """Get file size in bytes"""
    return Path(file_path).stat().st_size


def file_exists(file_path: str) -> bool:
    """Check if file exists"""
    return Path(file_path).exists()


def delete_file(file_path: str):
    """Delete a file if it exists"""
    if file_exists(file_path):
        Path(file_path).unlink()
