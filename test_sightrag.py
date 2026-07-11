"""
SightRAG v0.2 Quick Test
Run: python test_sightrag.py
"""
import os, sys, shutil

print("=" * 55)
print("  SightRAG v0.2 — Quick Test")
print("=" * 55)

INPUT = "./demo_sightrag/input_images"
REF = "./demo_sightrag/reference_images"

images = [f for f in os.listdir(INPUT) if f.endswith(('.jpg','.png'))]
refs = [f for f in os.listdir(REF) if f.endswith(('.jpg','.png'))]
print(f"\n  Input: {len(images)} | Reference: {len(refs)}")

# Test 1: Import
from sightrag import SightRAG, __version__
print(f"\n  1. Import: SightRAG v{__version__}")

# Test 2: Initialize
rag = SightRAG(index_path="./test_index")
print(f"  2. Backend: {rag._backend.name}")

# Test 3: Index
rag.index(INPUT)
print(f"  3. Indexed: {rag.count()} regions")

# Test 4-6: Text queries
for i, q in enumerate(["find person", "find empty shelf", "find car"], 4):
    results = rag.query(q, top_k=2)
    top = os.path.basename(results[0]['image_path']) if results else "none"
    print(f"  {i}. \"{q}\" → {top} ({results[0]['score']:.4f})")

# Test 7-8: Reference queries
for i, ref in enumerate(sorted(refs)[:2], 7):
    results = rag.query(reference=os.path.join(REF, ref), top_k=2)
    top = os.path.basename(results[0]['image_path']) if results else "none"
    print(f"  {i}. Ref: {ref} → {top} ({results[0]['score']:.4f})")

# Test 9: rag.show()
os.makedirs("./test_output", exist_ok=True)
results = rag.query("find person", top_k=2)
rag.show(results, save="./test_output/")
saved = os.listdir("./test_output/")
print(f"  9. rag.show(): {len(saved)} annotated images saved")

# Test 10: Clean
rag.clear()
shutil.rmtree("./test_index", ignore_errors=True)
shutil.rmtree("./test_output", ignore_errors=True)
print(f"  10. Cleanup: done")

print(f"\n  ALL TESTS PASSED")
print("=" * 55)
