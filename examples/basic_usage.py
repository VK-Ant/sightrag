"""
SightRAG — Basic Usage Example

Run: python examples/basic_usage.py
"""

from sightrag import SightRAG

# Initialize
rag = SightRAG()

# Index images
rag.index("./demo_sightrag/input_images/")

# Text query
print("\n🔍 Text Query: 'find person'")
results = rag.query("find person", top_k=3)
for r in results:
    print(f"  → {r['image_path']} | score: {r['score']:.4f} | {r['label']}")

# Reference image query
print("\n🖼️ Reference Query: ref_shelf_query.jpg")
results = rag.query(reference="./demo_sightrag/reference_images/ref_shelf_query.jpg", top_k=3)
for r in results:
    print(f"  → {r['image_path']} | score: {r['score']:.4f}")

# Cleanup
rag.clear()
print("\n✅ Done!")
