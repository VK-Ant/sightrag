"""
SightRAG v0.4 — Test
Run: python test_sightrag.py
"""
import os, sys, shutil

print("=" * 55)
print("  SightRAG v0.4 — Test")
print("=" * 55)

INPUT = "./demo_sightrag/input_images"
REF = "./demo_sightrag/reference_images"

images = [f for f in os.listdir(INPUT) if f.endswith(('.jpg','.png'))]
refs = [f for f in os.listdir(REF) if f.endswith(('.jpg','.png'))]
print(f"\n  Input: {len(images)} images | Reference: {len(refs)} images")

from sightrag import SightRAG, __version__
print(f"\n  1. Import: SightRAG v{__version__}")

# Load once, reuse
rag = SightRAG(index_path="./test_index")
print(f"  2. Backend: {rag._backend.name}")

# Index
rag.index(INPUT)
print(f"  3. Indexed: {rag.count()} regions")

# Text queries
for i, q in enumerate(["find person", "find shelf", "find car"], 4):
    results = rag.query(q, top_k=2)
    if results:
        print(f"  {i}. \"{q}\" → {os.path.basename(results[0]['image_path'])} ({results[0]['score']:.4f})")
    else:
        print(f"  {i}. \"{q}\" → no results")

# Reference query
ref = sorted(refs)[0]
results = rag.query(reference=os.path.join(REF, ref), top_k=2)
if results:
    print(f"  7. Ref: {ref} → {os.path.basename(results[0]['image_path'])} ({results[0]['score']:.4f})")

# Visualize
os.makedirs("./test_output", exist_ok=True)
rag.show(rag.query("find person", top_k=2), save="./test_output/")
print(f"  8. rag.show(): {len(os.listdir('./test_output/'))} images saved")

# OCR (optional)
try:
    import easyocr
    rag_ocr = SightRAG(ocr=True, index_path="./test_ocr")
    rag_ocr.index(INPUT)
    results = rag_ocr.query("find text", top_k=2)
    ocr_text = results[0].get('ocr_text', '') if results else ''
    print(f"  9. OCR: '{ocr_text[:30]}'")
    rag_ocr.clear()
except ImportError:
    print(f"  9. OCR: not installed (pip install easyocr)")
except Exception as e:
    print(f"  9. OCR: {str(e)[:40]}")

# CLI (optional)
try:
    import click
    print(f"  10. CLI: ready")
except ImportError:
    print(f"  10. CLI: not installed (pip install click)")

# Cleanup
rag.clear()
for d in ["test_index", "test_output", "test_ocr"]:
    shutil.rmtree(f"./{d}", ignore_errors=True)

print(f"\n  DONE")
print("=" * 55)
