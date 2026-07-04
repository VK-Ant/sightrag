"""
SightRAG — Custom Domain Example

Run: python examples/custom_domain_example.py
"""

from sightrag import SightRAG

# Without domain hint
print("── Without domain_hint ──")
rag = SightRAG(index_path="./index_default")
rag.index("./demo_sightrag/input_images/")
results = rag.query("out of stock products", top_k=3)
for r in results:
    print(f"  → {r['image_path'].split('/')[-1]} | score: {r['score']:.4f}")
rag.clear()

# With retail domain hint
print("\n── With retail domain_hint ──")
rag_retail = SightRAG(
    domain_hint="retail shelf product stock inventory empty facing",
    index_path="./index_retail"
)
rag_retail.index("./demo_sightrag/input_images/")
results = rag_retail.query("out of stock products", top_k=3)
for r in results:
    print(f"  → {r['image_path'].split('/')[-1]} | score: {r['score']:.4f}")
rag_retail.clear()

import shutil
shutil.rmtree("./index_default", ignore_errors=True)
shutil.rmtree("./index_retail", ignore_errors=True)
print("\n✅ Done!")
