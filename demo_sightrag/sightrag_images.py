"""
SightRAG v0.2 Demo — Image Folder
Run: python demo_sightrag/sightrag_images.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from sightrag import SightRAG

print("=" * 55)
print("  SightRAG v0.2 Demo — Image Folder")
print("  See. Search. Retrieve.")
print("=" * 55)

rag = SightRAG()

input_dir = os.path.join(os.path.dirname(__file__), "input_images")
rag.index(input_dir)

# Text queries
for q in ["find person", "find empty shelf", "find car in parking lot", "find room with furniture"]:
    results = rag.query(q, top_k=2)
    print(f'\n  "{q}"')
    for i, r in enumerate(results, 1):
        print(f"   {i}. {os.path.basename(r['image_path'])} — score: {r['score']:.4f} | {r['label']}")

# Reference queries
ref_dir = os.path.join(os.path.dirname(__file__), "reference_images")
for ref in sorted(f for f in os.listdir(ref_dir) if f.endswith('.jpg')):
    results = rag.query(reference=os.path.join(ref_dir, ref), top_k=2)
    print(f'\n  Reference: {ref}')
    for i, r in enumerate(results, 1):
        print(f"   {i}. {os.path.basename(r['image_path'])} — score: {r['score']:.4f}")

# Visualize
output_dir = os.path.join(os.path.dirname(__file__), "..", "output")
results = rag.query("find person", top_k=3)
rag.show(results, save=output_dir)

rag.clear()
print("\n  Demo complete!")
