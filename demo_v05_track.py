"""
SightRAG v0.5 — Track Release Demo
===================================

Demo 1: Basic (same as v0.4 — backward compatible)
Demo 2: Segmentation — pixel-level masks
Demo 3: Tracking — follow objects through video
Demo 4: Find + Track — query and follow in one call
Demo 5: Timeline visualization

Usage:
    python demo_v05_track.py
"""

import os

print("=" * 60)
print("  SightRAG v0.5 — Track Release Demo")
print("=" * 60)

from sightrag import SightRAG, __version__
print(f"  Version: {__version__}\n")


# ── Demo 1: Basic (backward compatible) ──────────────────
print("─── Demo 1: Basic Query (v0.4 compatible) ───")

rag = SightRAG()
print(f"  {rag}")

# Index images
if os.path.isdir("./demo_sightrag/input_images"):
    rag.index("./demo_sightrag/input_images")
    results = rag.query("find person", top_k=3)
    print(f"  Query 'find person': {len(results)} results")
    for r in results:
        print(f"    → {os.path.basename(r['image_path'])} "
              f"| {r['label']} | score={r['score']:.4f}")
    rag.show(results, save="./demo_output/basic/")
else:
    print("  (No demo images — skipping)")

rag.clear()


# ── Demo 2: Segmentation ─────────────────────────────────
print("\n─── Demo 2: Segmentation ───")

rag_seg = SightRAG(segment=True)
print(f"  {rag_seg}")

if os.path.isdir("./demo_sightrag/input_images"):
    rag_seg.index("./demo_sightrag/input_images")
    results = rag_seg.query("find person", top_k=2, segment=True)
    masks = sum(1 for r in results if r.get("mask") is not None)
    print(f"  Query with masks: {len(results)} results, {masks} with masks")
    rag_seg.show(results, save="./demo_output/segment/")
else:
    print("  (No demo images — skipping)")

rag_seg.clear()


# ── Demo 3: Video Tracking ───────────────────────────────
print("\n─── Demo 3: Video Tracking ───")

# Check for a demo video
demo_videos = []
for d in ["./demo_sightrag", "."]:
    if os.path.isdir(d):
        for f in os.listdir(d):
            if f.endswith((".mp4", ".avi", ".mov")):
                demo_videos.append(os.path.join(d, f))

if demo_videos:
    video = demo_videos[0]
    print(f"  Video: {video}")

    rag_track = SightRAG(track=True)
    print(f"  {rag_track}")

    rag_track.index(video, fps=5)

    # List all tracks
    all_tracks = rag_track.get_all_tracks()
    print(f"  Tracked objects: {len(all_tracks)}")
    for t in all_tracks[:10]:
        print(f"    Track#{t['track_id']}: {t['label']} "
              f"({t['first_timestamp']}s → {t['last_timestamp']}s, "
              f"{t['frame_count']} frames)")

    # Query with track info
    results = rag_track.query("find person", top_k=3)
    for r in results:
        tid = r.get("track_id", "?")
        print(f"    → {r['label']} | score={r['score']:.4f} | Track#{tid}")

    rag_track.show(results, save="./demo_output/track/")
    rag_track.clear()
else:
    print("  (No demo video found — skipping)")
    print("  Put a .mp4 file in ./demo_sightrag/ to test tracking")


# ── Demo 4: Find + Track ─────────────────────────────────
print("\n─── Demo 4: Find + Track ───")

if demo_videos:
    video = demo_videos[0]
    rag_ft = SightRAG(track=True)
    rag_ft.index(video, fps=5)

    timelines = rag_ft.find_and_track("person", top_k=2)
    print(f"  Found {len(timelines)} tracked person(s)")

    for tl in timelines:
        print(f"    Track#{tl['track_id']}: {tl['label']}")
        print(f"      First: {tl['first_timestamp']}s | "
              f"Last: {tl['last_timestamp']}s | "
              f"Frames: {tl['frame_count']}")

    rag_ft.clear()
else:
    print("  (No demo video — skipping)")


# ── Demo 5: Different tracker/segmentor combos ───────────
print("\n─── Demo 5: Pluggable Configs ───")

configs = [
    {"desc": "Segment only", "segment": True},
    {"desc": "Track only", "track": True},
    {"desc": "Custom tracker", "track": True, "tracker": "botsort"},
    {"desc": "Segment + Track", "segment": True, "track": True},
]

for cfg in configs:
    desc = cfg.pop("desc")
    try:
        r = SightRAG(index_path=f"./test_cfg_{desc[:8]}", **cfg)
        print(f"  {desc}: OK — {r}")
        r.clear()
    except Exception as e:
        print(f"  {desc}: {str(e)[:50]}")

# Cleanup
import shutil
for d in os.listdir("."):
    if d.startswith("test_cfg_"):
        shutil.rmtree(d, ignore_errors=True)

print(f"\n  DEMO COMPLETE")
print("=" * 60)
