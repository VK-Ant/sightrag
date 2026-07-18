"""
SightRAG v0.3 Demo — Image Folder
Run: python demo_sightrag/sightrag_images.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from sightrag import SightRAG

print("=" * 55)
print("  SightRAG v0.3 Demo — Image Folder")
print("  See. Search. Retrieve.")
print("=" * 55)

input_dir = os.path.join(os.path.dirname(__file__), "input_images")
ref_dir = os.path.join(os.path.dirname(__file__), "reference_images")
output_dir = os.path.join(os.path.dirname(__file__), "..", "output")

# ─── Default (YOLO + CLIP) ───
print("\n── Default: YOLO + CLIP ──")
rag = SightRAG()
rag.index(input_dir)

for q in ["find person", "find empty shelf", "find car", "find room"]:
    results = rag.query(q, top_k=2)
    print(f'\n  "{q}"')
    for i, r in enumerate(results, 1):
        print(f"   {i}. {os.path.basename(r['image_path'])} — score: {r['score']:.4f} | {r['label']}")

# Reference queries
for ref in sorted(f for f in os.listdir(ref_dir) if f.endswith('.jpg')):
    results = rag.query(reference=os.path.join(ref_dir, ref), top_k=2)
    print(f'\n  Reference: {ref}')
    for i, r in enumerate(results, 1):
        print(f"   {i}. {os.path.basename(r['image_path'])} — score: {r['score']:.4f}")

# Visualize
rag.show(rag.query("find person", top_k=3), save=output_dir)

# ─── Grounding DINO (v0.3 new) ───
print("\n── Grounding DINO (any domain) ──")
try:
    rag_dino = SightRAG(detector="grounding-dino")
    rag_dino.index(input_dir)
    results = rag_dino.query("find person standing", top_k=2)
    print(f'  "find person standing"')
    for i, r in enumerate(results, 1):
        print(f"   {i}. {os.path.basename(r['image_path'])} — score: {r['score']:.4f}")
    rag_dino.clear()
except Exception as e:
    print(f"  Skipped: {str(e)[:60]}")

# ─── Re-ranking (v0.3 new) ───
print("\n── Re-ranking (better accuracy) ──")
rag_rerank = SightRAG(rerank=True)
rag_rerank.index(input_dir)
results = rag_rerank.query("find person", top_k=3)
print(f'  "find person" (re-ranked)')
for i, r in enumerate(results, 1):
    print(f"   {i}. {os.path.basename(r['image_path'])} — score: {r['score']:.4f}")
rag_rerank.clear()

rag.clear()
print("\n  Demo complete!")
