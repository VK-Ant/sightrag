# sightrag/utils/video.py
# Video frame extraction utilities

from pathlib import Path
from PIL import Image

SUPPORTED_FORMATS = {".mp4", ".avi", ".mov", ".mkv"}


def extract_frames(video_path: str, fps: int = 1):
    """
    Extract frames from video at given FPS.
    Default: 1 frame per second.
    Returns list of (PIL.Image, timestamp_str) tuples.
    """
    try:
        import cv2
    except ImportError:
        raise ImportError(
            "OpenCV needed for video.\n"
            "Run: pip install opencv-python"
        )

    path = Path(video_path)

    if not path.exists():
        raise FileNotFoundError(f"Video not found: {path}")

    if path.suffix.lower() not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported video format: {path.suffix}\n"
            f"Supported: {SUPPORTED_FORMATS}"
        )

    cap = cv2.VideoCapture(str(path))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, int(video_fps / fps))

    frames = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            # Convert BGR to RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)

            # Generate timestamp
            seconds = frame_count / video_fps
            timestamp = _format_timestamp(seconds)
            frames.append((pil_img, timestamp))

        frame_count += 1

    cap.release()

    if not frames:
        raise ValueError(f"No frames extracted from: {path}")

    return frames


def _format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"
