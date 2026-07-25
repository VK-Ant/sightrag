"""
SightRAG v0.4 Demo — Image Folder + OCR + Multimodal
Run: python demo_sightrag/sightrag_images.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from sightrag import SightRAG

print("=" * 55)
print("  SightRAG v0.4 Demo — Image Folder")
print("  See. Search. Retrieve. Understand.")
print("=" * 55)

input_dir = os.path.join(os.path.dirname(__file__), "input_images")
ref_dir = os.path.join(os.path.dirname(__file__), "reference_images")
output_dir = os.path.join(os.path.dirname(__file__), "..", "output")

# ─── Mode 1: Default (YOLO + CLIP) ───
print("\n── Mode 1: Visual Search (YOLO + CLIP) ──")
rag = SightRAG()
rag.index(input_dir)

for q in ["find person", "find car", "find shelf"]:
    results = rag.query(q, top_k=2)
    print(f'\n  "{q}"')
    for i, r in enumerate(results, 1):
        print(f"   {i}. {os.path.basename(r['image_path'])} — score: {r['score']:.4f} | {r['label']}")

rag.show(rag.query("find person", top_k=3), save=output_dir)
rag.clear()

# ─── Mode 2: OCR (reads text on images) ───
print("\n── Mode 2: OCR Search (reads text) ──")
try:
    rag_ocr = SightRAG(ocr=True)
    rag_ocr.index(input_dir)
    
    # Text on images will be searchable
    for q in ["find person", "find text"]:
        results = rag_ocr.query(q, top_k=2)
        print(f'\n  "{q}"')
        for i, r in enumerate(results, 1):
            ocr = r.get('ocr_text', '')
            print(f"   {i}. {os.path.basename(r['image_path'])} — score: {r['score']:.4f} | OCR: '{ocr}'")
    
    rag_ocr.clear()
except Exception as e:
    print(f"  OCR skipped: {str(e)[:60]}")
    print("  Install: pip install easyocr")

# ─── Mode 3: Grounding DINO ───
print("\n── Mode 3: Grounding DINO (any domain) ──")
try:
    rag_dino = SightRAG(detector="grounding-dino")
    rag_dino.index(input_dir)
    results = rag_dino.query("find person standing", top_k=2)
    print(f'  "find person standing"')
    for i, r in enumerate(results, 1):
        print(f"   {i}. {os.path.basename(r['image_path'])} — score: {r['score']:.4f}")
    rag_dino.clear()
except Exception as e:
    print(f"  Grounding DINO skipped: {str(e)[:60]}")

# ─── Mode 4: Multimodal Understanding ───
print("\n── Mode 4: Multimodal Understanding (optional) ──")
try:
    rag_mm = SightRAG(ocr=True, multimodal="qwen2-vl")
    rag_mm.index(input_dir)
    results = rag_mm.query("find damaged product", top_k=2, understand=True)
    print(f'  "find damaged product" (with LLM understanding)')
    for i, r in enumerate(results, 1):
        print(f"   {i}. {os.path.basename(r['image_path'])} — score: {r['score']:.4f}")
    rag_mm.clear()
except Exception as e:
    print(f"  Multimodal skipped: {str(e)[:60]}")
    print("  Install: pip install sightrag[multimodal]")

print("\n  v0.4 Demo complete!")
