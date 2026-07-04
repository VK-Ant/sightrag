"""
SightRAG Demo — REST API
==========================
Step 1: Start server in one terminal:
    python -m sightrag.api

Step 2: Run this in another terminal:
    python demo_sightrag/sightrag_restapi.py
"""

import os
import sys
import json

try:
    import requests
except ImportError:
    print("❌ requests not installed. Run: pip install requests")
    sys.exit(1)

print("=" * 50)
print("  SightRAG Demo — REST API")
print("  See. Search. Retrieve.")
print("=" * 50)
print()

BASE = "http://localhost:8000"

# Check if server is running
try:
    r = requests.get(f"{BASE}/", timeout=3)
    print("✅ Server is running")
    print(json.dumps(r.json(), indent=2))
except requests.ConnectionError:
    print("❌ Server not running!")
    print("   Start it first:")
    print("   python -m sightrag.api")
    sys.exit(1)

input_dir = os.path.join(os.path.dirname(__file__), "input_images")
ref_dir = os.path.join(os.path.dirname(__file__), "reference_images")

# Index images
print("\n── Index Images ──")
r = requests.post(f"{BASE}/index/folder", data={"path": input_dir})
print(f"   {r.json()}")

# Text queries
queries = ["find person", "find empty shelf", "find car"]
for q in queries:
    print(f'\n── Query: "{q}" ──')
    r = requests.post(f"{BASE}/query/text", data={"text": q, "top_k": 2})
    data = r.json()
    for result in data["results"]:
        print(f"   → {os.path.basename(result['image_path'])} — score: {result['score']:.4f}")

# Reference image query
ref_path = os.path.join(ref_dir, "ref_shelf_query.jpg")
print(f"\n── Reference: ref_shelf_query.jpg ──")
with open(ref_path, "rb") as f:
    r = requests.post(f"{BASE}/query/reference", files={"file": f}, data={"top_k": 2})
for result in r.json()["results"]:
    print(f"   → {os.path.basename(result['image_path'])} — score: {result['score']:.4f}")

# Status
print(f"\n── Status ──")
print(f"   {requests.get(f'{BASE}/status').json()}")

# Clear
requests.delete(f"{BASE}/index")
print("\n✅ REST API demo complete!")
