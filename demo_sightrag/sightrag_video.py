"""
SightRAG v0.3 Demo — Video Indexing
Run: python demo_sightrag/sightrag_video.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from sightrag import SightRAG

print("=" * 55)
print("  SightRAG v0.3 Demo — Video")
print("  See. Search. Retrieve.")
print("=" * 55)

video_dir = os.path.join(os.path.dirname(__file__), "video_samples")
videos = [f for f in os.listdir(video_dir) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]

if not videos:
    print("\n  Creating sample video from images...")
    try:
        import cv2
        input_dir = os.path.join(os.path.dirname(__file__), "input_images")
        images = sorted(f for f in os.listdir(input_dir) if f.endswith('.jpg'))
        sample_path = os.path.join(video_dir, "sample_video.mp4")
        first = cv2.imread(os.path.join(input_dir, images[0]))
        h, w = first.shape[:2]
        writer = cv2.VideoWriter(sample_path, cv2.VideoWriter_fourcc(*'mp4v'), 1, (w, h))
        for img in images:
            writer.write(cv2.imread(os.path.join(input_dir, img)))
        writer.release()
        videos = ["sample_video.mp4"]
    except Exception as e:
        print(f"  Put your .mp4 file in video_samples/ folder")
        sys.exit(1)

rag = SightRAG()
for v in videos:
    rag.index(os.path.join(video_dir, v), fps=1)

for q in ["find person", "find shelf", "find car"]:
    results = rag.query(q, top_k=2)
    print(f'\n  "{q}"')
    for i, r in enumerate(results, 1):
        ts = r.get('timestamp', '')
        print(f"   {i}. score: {r['score']:.4f} | {r['label']} | t={ts}")

# v0.3: Visualize
output_dir = os.path.join(os.path.dirname(__file__), "..", "output")
rag.show(rag.query("find person", top_k=3), save=output_dir)

rag.clear()
print("\n  Video demo complete!")
