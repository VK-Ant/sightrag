"""
SightRAG v0.4 Demo — Live Camera + OCR
Run: python demo_sightrag/sightrag_livecam.py
"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

print("=" * 55)
print("  SightRAG v0.4 Demo — Live Camera")
print("  See. Search. Retrieve. Understand.")
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
output_dir = os.path.join(os.path.dirname(__file__), "..", "output")

# ─── Without OCR ───
print("\n── Visual Search ──")
rag = SightRAG()
rag.index(cam_dir)
for q in ["find person", "find object"]:
    results = rag.query(q, top_k=2)
    print(f'\n  "{q}"')
    for i, r in enumerate(results, 1):
        print(f"   {i}. {os.path.basename(r['image_path'])} — score: {r['score']:.4f}")
rag.show(rag.query("find person", top_k=2), save=output_dir)
rag.clear()

# ─── With OCR ───
print("\n── OCR Search ──")
try:
    rag_ocr = SightRAG(ocr=True)
    rag_ocr.index(cam_dir)
    results = rag_ocr.query("find text", top_k=2)
    print(f'  "find text"')
    for i, r in enumerate(results, 1):
        print(f"   {i}. {os.path.basename(r['image_path'])} — score: {r['score']:.4f} | OCR: '{r.get('ocr_text','')}'")
    rag_ocr.clear()
except Exception as e:
    print(f"  OCR skipped: {str(e)[:50]}")

print("\n  Camera demo complete!")
