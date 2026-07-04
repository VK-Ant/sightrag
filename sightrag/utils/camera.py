# sightrag/utils/camera.py
# Live camera frame capture utilities

from PIL import Image


def capture_frames(camera_id: int = 0,
                   fps: int = 1,
                   buffer_seconds: int = 60):
    """
    Capture frames from live camera.
    Default: 1 FPS, 60 second buffer.
    Generator — yields (PIL.Image, timestamp) tuples.
    """
    try:
        import cv2
    except ImportError:
        raise ImportError(
            "OpenCV needed for camera.\n"
            "Run: pip install opencv-python"
        )

    import time

    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open camera {camera_id}.\n"
            "Check camera is connected and not in use."
        )

    print(f"[SightRAG] Camera {camera_id} opened. "
          f"Capturing at {fps} FPS...")

    interval = 1.0 / fps
    last_capture = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[SightRAG] Camera frame failed. Retrying...")
                continue

            now = time.time()
            if now - last_capture >= interval:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)
                timestamp = time.strftime("%H:%M:%S")
                last_capture = now
                yield pil_img, timestamp

    finally:
        cap.release()
        print("[SightRAG] Camera released.")


def capture_single(camera_id: int = 0):
    """Capture a single frame for reference image query."""
    try:
        import cv2
    except ImportError:
        raise ImportError("Run: pip install opencv-python")

    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {camera_id}.")

    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError("Failed to capture frame.")

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)
