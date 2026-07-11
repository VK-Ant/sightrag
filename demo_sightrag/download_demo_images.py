"""
Download real demo images for SightRAG testing.
Run: python demo_sightrag/download_demo_images.py
"""

import os
import urllib.request

DEMO_DIR = os.path.join(os.path.dirname(__file__), "input_images")
REF_DIR = os.path.join(os.path.dirname(__file__), "reference_images")

# Free sample images from COCO dataset (public URLs)
IMAGES = {
    "input_images": [
        ("https://ultralytics.com/images/bus.jpg", "street_bus.jpg"),
        ("https://ultralytics.com/images/zidane.jpg", "person_sports.jpg"),
    ],
    "reference_images": []
}

print("Downloading real demo images...")
print()

for folder, urls in IMAGES.items():
    target = os.path.join(os.path.dirname(__file__), folder)
    os.makedirs(target, exist_ok=True)
    for url, fname in urls:
        path = os.path.join(target, fname)
        if os.path.exists(path):
            print(f"  Already exists: {fname}")
            continue
        try:
            print(f"  Downloading: {fname}...")
            urllib.request.urlretrieve(url, path)
            print(f"  Saved: {path}")
        except Exception as e:
            print(f"  Failed: {fname} — {e}")

print()
print("OR manually add your own real images:")
print(f"  Input:     {DEMO_DIR}")
print(f"  Reference: {REF_DIR}")
print()
print("Good images for testing:")
print("  - Street photos with people and cars")
print("  - Shelf photos with products")
print("  - Room photos with furniture")
print("  - Any photo with clear objects")
print()
print("Done!")
