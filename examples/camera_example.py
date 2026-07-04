"""
SightRAG — Camera Example

Run: python examples/camera_example.py
"""

from sightrag import SightRAG

# Initialize
rag = SightRAG()

# Index existing images first
rag.index("./demo_sightrag/input_images/")

# Capture from camera and use as reference query
print("\n📷 Capturing from webcam...")
try:
    from sightrag.utils.camera import capture_single
    frame = capture_single(camera_id=0)
    frame.save("./demo_sightrag/camera_captures/live_capture.jpg")
    print("✅ Frame captured!")

    # Use captured frame as reference query
    results = rag.query(reference="./demo_sightrag/camera_captures/live_capture.jpg", top_k=3)
    print("\n🔍 Results using live camera frame:")
    for r in results:
        print(f"  → {r['image_path']} | score: {r['score']:.4f}")

except Exception as e:
    print(f"⚠️ Camera not available: {e}")

rag.clear()
print("\n✅ Done!")
