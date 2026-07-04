"""
SightRAG Demo — Live Camera
=============================
Run: python demo_sightrag/sightrag_livecam.py

Captures frames from webcam for 10 seconds.
Then searches captured frames with text queries.

Requires: webcam connected to your computer.
"""

import os
import sys
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

print("=" * 50)
print("  SightRAG Demo — Live Camera")
print("  See. Search. Retrieve.")
print("=" * 50)
print()

try:
    import cv2
except ImportError:
    print("❌ OpenCV not installed. Run: pip install opencv-python")
    sys.exit(1)

# Check camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ No camera detected.")
    print("   Connect a webcam and try again.")
    sys.exit(1)

cam_dir = os.path.join(os.path.dirname(__file__), "camera_captures")
os.makedirs(cam_dir, exist_ok=True)

# Capture for 10 seconds at 1 FPS
print("📷 Capturing from webcam for 10 seconds...")
print("   (stay in front of camera)")
print()

from PIL import Image

frames_saved = 0
start = time.time()

while time.time() - start < 10:
    ret, frame = cap.read()
    if ret:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        fname = f"cam_frame_{frames_saved:03d}.jpg"
        pil_img.save(os.path.join(cam_dir, fname))
        frames_saved += 1
        elapsed = int(time.time() - start)
        print(f"\r   Captured {frames_saved} frames ({elapsed}/10 sec)", end="")
    time.sleep(1)

cap.release()
print(f"\n\n✅ Captured {frames_saved} frames in camera_captures/")

if frames_saved == 0:
    print("❌ No frames captured. Camera may not be working.")
    sys.exit(1)

# Now index captured frames
from sightrag import SightRAG

rag = SightRAG()
print(f"\n📁 Indexing {frames_saved} camera frames...")
rag.index(cam_dir)

# Search
queries = ["find person", "find face", "find object"]
for q in queries:
    results = rag.query(q, top_k=2)
    print(f'\n🔍 "{q}"')
    for i, r in enumerate(results, 1):
        print(f"   {i}. {os.path.basename(r['image_path'])} — "
              f"score: {r['score']:.4f} | {r['label']}")

# Reference query — use first captured frame
first_frame = os.path.join(cam_dir, "cam_frame_000.jpg")
results = rag.query(reference=first_frame, top_k=2)
print(f'\n🖼️ Reference: cam_frame_000.jpg')
for i, r in enumerate(results, 1):
    print(f"   {i}. {os.path.basename(r['image_path'])} — score: {r['score']:.4f}")

rag.clear()
print("\n✅ Camera demo complete!")
