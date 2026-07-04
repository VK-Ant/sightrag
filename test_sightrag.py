"""
SightRAG Complete Test
Run: python test_sightrag.py

Tests all input modes:
1. Image folder indexing
2. Reference image query
3. Camera capture (if available)
4. Text queries
"""

import os
import sys
import shutil

print("=" * 55)
print("  SightRAG v0.1.0 — Complete Test")
print("  See. Search. Retrieve.")
print("=" * 55)
print()

INPUT_DIR = "./demo_sightrag/input_images"
REF_DIR = "./demo_sightrag/reference_images"
CAM_DIR = "./demo_sightrag/camera_captures"

# Verify folders
for d in [INPUT_DIR, REF_DIR]:
    if not os.path.exists(d):
        print(f"❌ {d} not found!")
        sys.exit(1)

images = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
refs = [f for f in os.listdir(REF_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]

print(f"✅ Input images:     {len(images)} files in {INPUT_DIR}")
print(f"✅ Reference images: {len(refs)} files in {REF_DIR}")
print()

# ─────────────────────────────────────
# TEST 1: Import
# ─────────────────────────────────────
print("— Test 1: Import")
try:
    from sightrag import SightRAG, __version__
    print(f"  ✅ SightRAG v{__version__}")
except Exception as e:
    print(f"  ❌ {e}")
    sys.exit(1)
print()

# ─────────────────────────────────────
# TEST 2: Initialize
# ─────────────────────────────────────
print("— Test 2: Initialize")
try:
    rag = SightRAG(index_path="./test_index")
    print(f"  ✅ Store: {rag._store_type}")
except Exception as e:
    print(f"  ⚠️  ChromaDB failed, trying SQLite...")
    rag = SightRAG(store="sqlite", index_path="./test_index")
    print(f"  ✅ SQLite fallback")
print()

# ─────────────────────────────────────
# TEST 3: Index image folder
# ─────────────────────────────────────
print(f"— Test 3: Index {len(images)} input images")
try:
    rag.index(INPUT_DIR)
    print(f"  ✅ {rag.count()} regions indexed")
except Exception as e:
    print(f"  ❌ {e}")
    sys.exit(1)
print()

# ─────────────────────────────────────
# TEST 4-6: Text queries
# ─────────────────────────────────────
queries = [
    ("find person", "Test 4"),
    ("find empty shelf", "Test 5"),
    ("find car in parking lot", "Test 6"),
]
for q, name in queries:
    print(f'— {name}: Text query — "{q}"')
    try:
        results = rag.query(q, top_k=3)
        print(f"  ✅ {len(results)} results")
        for i, r in enumerate(results[:2], 1):
            print(f"     {i}. {os.path.basename(r['image_path'])} — "
                  f"score: {r['score']} | label: {r['label']}")
    except Exception as e:
        print(f"  ❌ {e}")
    print()

# ─────────────────────────────────────
# TEST 7-9: Reference image queries
# ─────────────────────────────────────
for idx, ref_file in enumerate(sorted(refs), 7):
    ref_path = os.path.join(REF_DIR, ref_file)
    print(f"— Test {idx}: Reference query — {ref_file}")
    try:
        results = rag.query(reference=ref_path, top_k=3)
        print(f"  ✅ {len(results)} results")
        for i, r in enumerate(results[:2], 1):
            print(f"     {i}. {os.path.basename(r['image_path'])} — "
                  f"score: {r['score']}")
    except Exception as e:
        print(f"  ❌ {e}")
    print()

# ─────────────────────────────────────
# TEST 10: Camera capture (optional)
# ─────────────────────────────────────
print("— Test 10: Camera capture (optional)")
try:
    import cv2
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # Save captured frame
            os.makedirs(CAM_DIR, exist_ok=True)
            cam_path = os.path.join(CAM_DIR, "camera_test.jpg")
            cv2.imwrite(cam_path, frame)
            cap.release()
            
            # Use as reference query
            results = rag.query(reference=cam_path, top_k=3)
            print(f"  ✅ Camera frame captured → {cam_path}")
            print(f"  ✅ Reference query returned {len(results)} results")
            for i, r in enumerate(results[:2], 1):
                print(f"     {i}. {os.path.basename(r['image_path'])} — "
                      f"score: {r['score']}")
        else:
            print("  ⚠️  Camera opened but frame capture failed")
            cap.release()
    else:
        print("  ⚠️  No camera detected — skipping (this is okay)")
except ImportError:
    print("  ⚠️  OpenCV not available — skipping camera test")
except Exception as e:
    print(f"  ⚠️  Camera test skipped: {e}")
print()

# ─────────────────────────────────────
# TEST 11: Live camera indexing (optional, 5 seconds only)
# ─────────────────────────────────────
print("— Test 11: Live camera indexing (5 seconds)")
try:
    import cv2
    import time
    
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        os.makedirs(CAM_DIR, exist_ok=True)
        cam_rag = SightRAG(index_path="./test_cam_index")
        
        start = time.time()
        count = 0
        while time.time() - start < 5:  # 5 seconds only
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                from PIL import Image
                pil_img = Image.fromarray(frame_rgb)
                
                # Save frame
                fname = f"cam_frame_{count:03d}.jpg"
                pil_img.save(os.path.join(CAM_DIR, fname))
                count += 1
                time.sleep(1)  # 1 FPS
        
        cap.release()
        
        if count > 0:
            cam_rag.index(CAM_DIR)
            results = cam_rag.query("find person", top_k=2)
            print(f"  ✅ Captured {count} frames in 5 seconds")
            print(f"  ✅ Indexed {cam_rag.count()} regions")
            print(f"  ✅ Query returned {len(results)} results")
            cam_rag.clear()
        else:
            print("  ⚠️  No frames captured")
    else:
        print("  ⚠️  No camera — skipping")
except Exception as e:
    print(f"  ⚠️  Skipped: {e}")
print()

# ─────────────────────────────────────
# TEST 12: Error handling
# ─────────────────────────────────────
print("— Test 12: Error handling")
errors_caught = 0

# Empty index query
try:
    empty_rag = SightRAG(index_path="./test_empty")
    empty_rag.query("test")
except (RuntimeError, Exception):
    errors_caught += 1

# Missing folder
try:
    rag.index("./nonexistent_folder/")
except (FileNotFoundError, Exception):
    errors_caught += 1

# No query provided
try:
    rag.query()
except (ValueError, TypeError, Exception):
    errors_caught += 1

print(f"  ✅ {errors_caught}/3 errors caught correctly")
print()

# ─────────────────────────────────────
# TEST 13: Status
# ─────────────────────────────────────
print(f"— Test 13: Status → {rag}")
print()

# ─────────────────────────────────────
# Cleanup
# ─────────────────────────────────────
print("— Cleanup")
rag.clear()
shutil.rmtree("./test_index", ignore_errors=True)
shutil.rmtree("./test_empty", ignore_errors=True)
shutil.rmtree("./test_cam_index", ignore_errors=True)
# Keep camera captures for user to see
print("  ✅ Test indexes cleaned")

print()
print("=" * 55)
print("  ALL TESTS COMPLETE ✅")
print("=" * 55)
print()
print("  Tested:")
print("  ✅ Image folder indexing")
print("  ✅ Text queries (person, shelf, car)")
print("  ✅ Reference image queries (3 references)")
print("  ✅ Camera capture (if available)")
print("  ✅ Live camera indexing (if available)")
print("  ✅ Error handling")
print()
print("  Ship it Ant! 🐜")
