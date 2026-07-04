"""
SightRAG Demo — Video Indexing
===============================
Run: python demo_sightrag/sightrag_video.py

Put your video files in video_samples/ folder first.
If no videos found, creates a sample video from input_images.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sightrag import SightRAG

print("=" * 50)
print("  SightRAG Demo — Video")
print("  See. Search. Retrieve.")
print("=" * 50)
print()

video_dir = os.path.join(os.path.dirname(__file__), "video_samples")
videos = [f for f in os.listdir(video_dir)
          if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]

if not videos:
    # Create a sample video from demo images
    print("📹 No videos found. Creating sample video from images...")
    try:
        import cv2
        import numpy as np
        from PIL import Image

        input_dir = os.path.join(os.path.dirname(__file__), "input_images")
        images = sorted([f for f in os.listdir(input_dir) if f.endswith('.jpg')])

        sample_path = os.path.join(video_dir, "sample_video.mp4")
        first = cv2.imread(os.path.join(input_dir, images[0]))
        h, w = first.shape[:2]

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(sample_path, fourcc, 1, (w, h))

        for img_name in images:
            frame = cv2.imread(os.path.join(input_dir, img_name))
            for _ in range(3):  # 3 seconds per image
                writer.write(frame)

        writer.release()
        print(f"   ✅ Created: {sample_path}")
        videos = ["sample_video.mp4"]

    except Exception as e:
        print(f"   ❌ Could not create sample video: {e}")
        print("   Put your own .mp4 file in video_samples/ folder")
        sys.exit(1)

# Index videos
rag = SightRAG()

for video in videos:
    video_path = os.path.join(video_dir, video)
    print(f"\n📹 Indexing: {video}")
    rag.index(video_path, fps=1)

# Query
queries = ["find person", "find shelf", "find car"]
for q in queries:
    results = rag.query(q, top_k=2)
    print(f'\n🔍 "{q}"')
    for i, r in enumerate(results, 1):
        print(f"   {i}. timestamp: {r['timestamp']} — "
              f"score: {r['score']:.4f} | {r['label']}")

rag.clear()
print("\n✅ Video demo complete!")
