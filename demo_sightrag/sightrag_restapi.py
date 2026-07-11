"""
SightRAG v0.2 Demo — REST API
Step 1: python -m sightrag.api
Step 2: python demo_sightrag/sightrag_restapi.py
"""
import os, sys, json
try:
    import requests
except ImportError:
    print("  Install: pip install requests")
    sys.exit(1)

print("=" * 55)
print("  SightRAG v0.2 Demo — REST API")
print("  See. Search. Retrieve.")
print("=" * 55)

BASE = "http://localhost:8000"
try:
    r = requests.get(f"{BASE}/", timeout=3)
    print(f"  Server running: {r.json()}")
except requests.ConnectionError:
    print("  Start server first: python -m sightrag.api")
    sys.exit(1)

input_dir = os.path.join(os.path.dirname(__file__), "input_images")
ref_dir = os.path.join(os.path.dirname(__file__), "reference_images")

requests.post(f"{BASE}/index/folder", data={"path": input_dir})

for q in ["find person", "find empty shelf"]:
    r = requests.post(f"{BASE}/query/text", data={"text": q, "top_k": 2})
    print(f'\n  "{q}"')
    for res in r.json()["results"]:
        print(f"   → {os.path.basename(res['image_path'])} — score: {res['score']:.4f}")

ref_path = os.path.join(ref_dir, "ref_shelf_query.jpg")
with open(ref_path, "rb") as f:
    r = requests.post(f"{BASE}/query/reference", files={"file": f}, data={"top_k": 2})
print(f'\n  Reference: ref_shelf_query.jpg')
for res in r.json()["results"]:
    print(f"   → {os.path.basename(res['image_path'])} — score: {res['score']:.4f}")

requests.delete(f"{BASE}/index")
print("\n  REST API demo complete!")
