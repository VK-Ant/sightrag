"""
SightRAG v0.5 — Test (Track Release)
Run: python test_sightrag.py
"""
import os, sys, shutil

print("=" * 60)
print("  SightRAG v0.5 — Track Release Test")
print("=" * 60)

INPUT = "./demo_sightrag/input_images"
REF = "./demo_sightrag/reference_images"

images = [f for f in os.listdir(INPUT) if f.endswith(('.jpg','.png'))] if os.path.isdir(INPUT) else []
refs = [f for f in os.listdir(REF) if f.endswith(('.jpg','.png'))] if os.path.isdir(REF) else []
print(f"\n  Input: {len(images)} images | Reference: {len(refs)} images")

from sightrag import SightRAG, __version__
print(f"\n  1. Import: SightRAG v{__version__}")

# ── Core (same as v0.4) ───────────────────────────────────
rag = SightRAG(index_path="./test_index")
print(f"  2. Init: {rag}")

if images:
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
    if refs:
        ref = sorted(refs)[0]
        results = rag.query(reference=os.path.join(REF, ref), top_k=2)
        if results:
            print(f"  7. Ref: {ref} → {os.path.basename(results[0]['image_path'])} ({results[0]['score']:.4f})")

    # Visualize
    os.makedirs("./test_output", exist_ok=True)
    rag.show(rag.query("find person", top_k=2), save="./test_output/")
    print(f"  8. rag.show(): {len(os.listdir('./test_output/'))} images saved")
else:
    print("  3-8. Skipped (no demo images)")

# ── v0.5: Segmentation module test ────────────────────────
print(f"\n  --- v0.5 Segmentation ---")

try:
    from sightrag.segmentors.base import SegmentorBase
    from sightrag.segmentors.yolo_segmentor import YOLOSegmentor
    print(f"  S1. Segmentor imports: OK")
except ImportError as e:
    print(f"  S1. Segmentor imports: {e}")

try:
    seg = YOLOSegmentor()
    print(f"  S2. Default segmentor init: OK")

    if images:
        from PIL import Image
        test_img = Image.open(os.path.join(INPUT, images[0])).convert("RGB")
        seg_results = seg.segment(test_img)
        masks_count = sum(1 for d in seg_results if d.get("mask") is not None)
        print(f"  S3. Segment: {len(seg_results)} objects, {masks_count} masks")
    else:
        print(f"  S3. Segment: skipped (no images)")
except Exception as e:
    print(f"  S2-S3. Segmentation: {str(e)[:50]}")

try:
    from sightrag.segmentors.hf_segmentor import HFSegmentor
    print(f"  S4. HFSegmentor import: OK")
except ImportError as e:
    print(f"  S4. HFSegmentor import: {e}")

try:
    from sightrag.segmentors.sam2_segmentor import SAM2Segmentor
    print(f"  S5. SAM2Segmentor import: OK")
except ImportError as e:
    print(f"  S5. SAM2Segmentor import: {e}")


# ── v0.5: Tracker module test ─────────────────────────────
print(f"\n  --- v0.5 Tracking ---")

try:
    from sightrag.trackers.base import TrackerBase, Track, TrackState
    from sightrag.trackers.bytetrack import ByteTracker, STrack, KalmanFilter
    print(f"  T1. Tracker imports: OK")
except ImportError as e:
    print(f"  T1. Tracker imports: {e}")

try:
    tracker = ByteTracker()
    print(f"  T2. Default tracker init: OK")

    # Simulate 3 frames of tracking
    dets_f1 = [
        {"bbox": [100, 100, 200, 250], "label": "person", "confidence": 0.9},
        {"bbox": [400, 150, 500, 350], "label": "car", "confidence": 0.85},
    ]
    dets_f2 = [
        {"bbox": [105, 98, 205, 248], "label": "person", "confidence": 0.88},
        {"bbox": [410, 155, 510, 355], "label": "car", "confidence": 0.82},
    ]
    dets_f3 = [
        {"bbox": [112, 95, 212, 245], "label": "person", "confidence": 0.87},
    ]

    states1 = tracker.update(dets_f1, frame_idx=1, timestamp="0.00")
    states2 = tracker.update(dets_f2, frame_idx=2, timestamp="0.20")
    states3 = tracker.update(dets_f3, frame_idx=3, timestamp="0.40")

    tracks = tracker.get_tracks()
    print(f"  T3. ByteTrack 3-frame test: {len(tracks)} tracks, "
          f"IDs={list(tracks.keys())}")

    # Check track consistency
    for tid, track in tracks.items():
        print(f"      Track#{tid}: {track.label}, "
              f"frames={track.frame_count}, "
              f"t={track.first_timestamp}→{track.last_timestamp}")
except Exception as e:
    print(f"  T2-T3. ByteTrack: {str(e)[:50]}")

try:
    from sightrag.trackers.botsort import BoTSORTTracker, STrackReID
    print(f"  T4. BoTSORTTracker import: OK")

    bot = BoTSORTTracker()
    dets = [
        {"bbox": [100, 100, 200, 250], "label": "person",
         "confidence": 0.9, "embedding": None},
    ]
    states = bot.update(dets, frame_idx=1, timestamp="0.00")
    print(f"  T5. BoT-SORT 1-frame: {len(states)} tracked")
except Exception as e:
    print(f"  T4-T5. BoT-SORT: {str(e)[:50]}")


# ── v0.5: Core integration test ───────────────────────────
print(f"\n  --- v0.5 Core Integration ---")

try:
    rag_track = SightRAG(track=True, index_path="./test_track_index")
    print(f"  I1. SightRAG(track=True): OK — {rag_track}")
    rag_track.clear()
except Exception as e:
    print(f"  I1. SightRAG(track=True): {str(e)[:50]}")

try:
    rag_seg = SightRAG(segment=True, index_path="./test_seg_index")
    print(f"  I2. SightRAG(segment=True): OK — {rag_seg}")
    rag_seg.clear()
except Exception as e:
    print(f"  I2. SightRAG(segment=True): {str(e)[:50]}")

try:
    rag_both = SightRAG(track=True, segment=True, index_path="./test_both_index")
    print(f"  I3. SightRAG(track+segment): OK")
    print(f"      {rag_both}")
    rag_both.clear()
except Exception as e:
    print(f"  I3. SightRAG(track+segment): {str(e)[:50]}")


# ── v0.5: Store tracks test ───────────────────────────────
print(f"\n  --- v0.5 Store Tracks ---")

try:
    from sightrag.store.sqlite_store import SQLiteStore
    store = SQLiteStore("./test_store_track")

    store.save_track(
        track_id=1, video_path="test.mp4", label="person",
        first_frame=1, last_frame=30,
        first_timestamp="0.00", last_timestamp="1.00",
        frame_count=30,
        trajectory=[(1, [100, 100, 200, 250]), (30, [300, 100, 400, 250])]
    )

    store.add_track_detection(
        track_id=1, video_path="test.mp4", frame_idx=1,
        timestamp="0.00", bbox=[100, 100, 200, 250],
        label="person", confidence=0.9
    )
    store.add_track_detection(
        track_id=1, video_path="test.mp4", frame_idx=30,
        timestamp="1.00", bbox=[300, 100, 400, 250],
        label="person", confidence=0.85
    )
    store.conn.commit()

    track = store.get_track(1)
    print(f"  D1. save_track + get_track: OK")
    print(f"      Track#{track['track_id']}: {track['label']}, "
          f"{track['frame_count']} frames, "
          f"{len(track['states'])} detections")

    all_t = store.get_all_tracks()
    print(f"  D2. get_all_tracks: {len(all_t)} tracks")

    found = store.find_track_for_detection(
        "test.mp4", [100, 100, 200, 250], "0.00"
    )
    print(f"  D3. find_track_for_detection: "
          f"{'found Track#' + str(found['track_id']) if found else 'not found'}")

    print(f"  D4. track_count: {store.track_count()}")

    store.clear()
    print(f"  D5. clear (with tracks): OK")

    store.conn.close()
except Exception as e:
    print(f"  D1-D5. Store tracks: {str(e)[:60]}")


# ── OCR test (optional) ───────────────────────────────────
print(f"\n  --- Optional Features ---")
try:
    import easyocr
    rag_ocr = SightRAG(ocr=True, index_path="./test_ocr")
    if images:
        rag_ocr.index(INPUT)
        results = rag_ocr.query("find text", top_k=2)
        ocr_text = results[0].get('ocr_text', '') if results else ''
        print(f"  O1. OCR: '{ocr_text[:30]}'")
    rag_ocr.clear()
except ImportError:
    print(f"  O1. OCR: not installed (pip install easyocr)")
except Exception as e:
    print(f"  O1. OCR: {str(e)[:40]}")

try:
    import click
    print(f"  O2. CLI: ready")
except ImportError:
    print(f"  O2. CLI: not installed (pip install click)")


# ── Cleanup ───────────────────────────────────────────────
rag.clear()
for d in ["test_index", "test_output", "test_ocr",
          "test_track_index", "test_seg_index", "test_both_index",
          "test_store_track"]:
    shutil.rmtree(f"./{d}", ignore_errors=True)

print(f"\n  ALL TESTS DONE")
print("=" * 60)
