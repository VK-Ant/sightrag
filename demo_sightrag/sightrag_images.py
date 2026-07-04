"""
SightRAG Demo — Image Folder Indexing
======================================
Run: python demo_sightrag/sightrag_images.py

Indexes images from input_images/ folder.
Searches with text and reference image.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sightrag import SightRAG

print("=" * 50)
print("  SightRAG Demo — Image Folder")
print("  See. Search. Retrieve.")
print("=" * 50)
print()

# Step 1: Initialize
rag = SightRAG()

# Step 2: Index images
input_dir = os.path.join(os.path.dirname(__file__), "input_images")
print(f"\n📁 Indexing: {input_dir}")
rag.index(input_dir)

# Step 3: Text queries
queries = [
    "find person",
    "find empty shelf",
    "find car in parking lot",
    "find room with furniture"
]

for q in queries:
    results = rag.query(q, top_k=2)
    print(f'\n🔍 "{q}"')
    for i, r in enumerate(results, 1):
        print(f"   {i}. {os.path.basename(r['image_path'])} — "
              f"score: {r['score']:.4f} | {r['label']} | bbox: {r['bbox']}")

# Step 4: Reference image query
ref_dir = os.path.join(os.path.dirname(__file__), "reference_images")
ref_images = [f for f in os.listdir(ref_dir) if f.endswith('.jpg')]

for ref in ref_images:
    ref_path = os.path.join(ref_dir, ref)
    results = rag.query(reference=ref_path, top_k=2)
    print(f'\n🖼️ Reference: {ref}')
    for i, r in enumerate(results, 1):
        print(f"   {i}. {os.path.basename(r['image_path'])} — score: {r['score']:.4f}")

# Cleanup
rag.clear()
print("\n✅ Image demo complete!")
