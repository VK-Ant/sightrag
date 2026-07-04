"""
SightRAG — REST API Example

Step 1: Start server
    python -m sightrag.api

Step 2: Run this script in another terminal
    python examples/rest_api_example.py
"""

import requests
import json

BASE = "http://localhost:8000"

# Check API
print("── API Status ──")
r = requests.get(f"{BASE}/")
print(json.dumps(r.json(), indent=2))

# Index images
print("\n── Index Images ──")
r = requests.post(f"{BASE}/index/folder", data={"path": "./demo_sightrag/input_images/"})
print(r.json())

# Text query
print("\n── Text Query ──")
r = requests.post(f"{BASE}/query/text", data={"text": "find person", "top_k": 3})
data = r.json()
print(f"Query: {data['query']}")
for result in data["results"]:
    print(f"  → {result['image_path'].split('/')[-1]} | score: {result['score']:.4f}")

# Reference image query
print("\n── Reference Image Query ──")
with open("./demo_sightrag/reference_images/ref_shelf_query.jpg", "rb") as f:
    r = requests.post(f"{BASE}/query/reference", files={"file": f}, data={"top_k": 3})
data = r.json()
for result in data["results"]:
    print(f"  → {result['image_path'].split('/')[-1]} | score: {result['score']:.4f}")

# Status
print("\n── Status ──")
r = requests.get(f"{BASE}/status")
print(r.json())

# Clear
print("\n── Clear ──")
r = requests.delete(f"{BASE}/index")
print(r.json())

print("\n✅ Done!")
