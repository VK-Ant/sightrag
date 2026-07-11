"""
SightRAG v0.2 Demo — Live Camera
Run: python demo_sightrag/sightrag_livecam.py
Requires webcam.
"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

print("=" * 55)
print("  SightRAG v0.2 Demo — Live Camera")
print("  See. Search. Retrieve.")
print("=" * 55)

try:
    import cv2
except ImportError:
    print("  Install: pip install opencv-python")
    sys.exit(1)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("  No camera detected.")
    sys.exit(1)

cam_dir = os.path.join(os.path.dirname(__file__), "camera_captures")
os.makedirs(cam_dir, exist_ok=True)

from PIL import Image
print("  Capturing 10 seconds...")
frames_saved = 0
start = time.time()
while time.time() - start < 10:
    ret, frame = cap.read()
    if ret:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        Image.fromarray(rgb).save(os.path.join(cam_dir, f"cam_{frames_saved:03d}.jpg"))
        frames_saved += 1
        print(f"\r  {frames_saved} frames ({int(time.time()-start)}/10s)", end="", flush=True)
    time.sleep(1)
cap.release()
print(f"\n  Captured {frames_saved} frames")

from sightrag import SightRAG
rag = SightRAG()
rag.index(cam_dir)

for q in ["find person", "find object"]:
    results = rag.query(q, top_k=2)
    print(f'\n  "{q}"')
    for i, r in enumerate(results, 1):
        print(f"   {i}. {os.path.basename(r['image_path'])} — score: {r['score']:.4f}")

output_dir = os.path.join(os.path.dirname(__file__), "..", "output")
rag.show(rag.query("find person", top_k=3), save=output_dir)

rag.clear()
print("\n  Camera demo complete!")
