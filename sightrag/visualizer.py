"""SightRAG result visualization — rag.show()"""

import os
import numpy as np
from PIL import Image, ImageDraw


# High contrast colors — visible on any background
COLORS = {
    "person":      (0, 255, 0),       # bright green
    "car":         (255, 50, 50),      # bright red
    "truck":       (255, 80, 0),       # orange
    "bottle":      (0, 150, 255),      # blue
    "cup":         (0, 200, 255),      # cyan
    "chair":       (255, 200, 0),      # yellow
    "dog":         (255, 165, 0),      # orange
    "cat":         (200, 100, 255),    # purple
    "horse":       (255, 255, 0),      # yellow
    "bus":         (255, 0, 100),      # magenta
    "motorcycle":  (0, 255, 200),      # teal
    "whole_image": (100, 100, 100),    # dim gray (de-emphasized)
}
DEFAULT_COLOR = (0, 255, 200)

# Track ID colors — cycle through for different tracks
TRACK_COLORS = [
    (0, 255, 0), (255, 50, 50), (0, 150, 255), (255, 200, 0),
    (200, 100, 255), (0, 255, 200), (255, 80, 0), (255, 0, 100),
    (0, 200, 255), (255, 255, 0), (100, 255, 100), (255, 150, 50),
]


def show_results(results, save=None, max_show=5):
    """
    Display or save results with bounding boxes, masks, and track info.

    Usage:
        rag.show(results)                    # display
        rag.show(results, save="./output/")  # save
    """
    if save:
        os.makedirs(save, exist_ok=True)

    shown = 0
    for r in results[:max_show]:
        path = r.get("image_path", "")
        if not os.path.exists(path):
            continue

        img = Image.open(path).copy()
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Draw mask overlay if present
        mask = r.get("mask")
        if mask is not None:
            img = _draw_mask(img, mask, r.get("label", ""))

        draw = ImageDraw.Draw(img)
        bbox = r.get("bbox", [])
        label = r.get("label", "")
        score = r.get("score", 0)
        track_id = r.get("track_id")

        w, h = img.size

        # Skip drawing if whole_image (full image bbox)
        is_whole = (label == "whole_image" or
                    (bbox and bbox == [0, 0, w, h]))

        if bbox and len(bbox) == 4 and not is_whole:
            x1, y1, x2, y2 = bbox

            # Use track color if tracked, else label color
            if track_id is not None:
                color = TRACK_COLORS[track_id % len(TRACK_COLORS)]
            else:
                color = COLORS.get(label, DEFAULT_COLOR)

            # Thick bounding box with double border for contrast
            draw.rectangle([x1-1, y1-1, x2+1, y2+1],
                           outline=(0, 0, 0), width=4)
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

            # Label text — include track ID if tracked
            if track_id is not None:
                text = f" T{track_id} {label} {score:.2f} "
            else:
                text = f" {label} {score:.2f} "

            try:
                text_bbox = draw.textbbox((0, 0), text)
                tw = text_bbox[2] - text_bbox[0]
                th = text_bbox[3] - text_bbox[1]
            except:
                tw = len(text) * 8
                th = 14

            # Label position — above bbox, or inside if at top edge
            label_y = y1 - th - 6
            if label_y < 0:
                label_y = y1 + 4

            # Dark background for label
            draw.rectangle(
                [x1, label_y, x1 + tw + 8, label_y + th + 6],
                fill=(0, 0, 0)
            )
            draw.text((x1 + 4, label_y + 2), text, fill=color)

            # Corner markers
            corner_len = min(20, (x2-x1)//4, (y2-y1)//4)
            for cx, cy, dx, dy in [
                (x1, y1, 1, 1), (x2, y1, -1, 1),
                (x1, y2, 1, -1), (x2, y2, -1, -1)
            ]:
                draw.line([(cx, cy), (cx + corner_len*dx, cy)],
                          fill=color, width=3)
                draw.line([(cx, cy), (cx, cy + corner_len*dy)],
                          fill=color, width=3)

        # Bottom info bar
        bar_h = 35
        draw.rectangle([0, h - bar_h, w, h], fill=(0, 0, 0))

        # Score bar
        score_w = int(score * (w - 20))
        score_color = ((0, 255, 0) if score > 0.7
                       else (255, 255, 0) if score > 0.4
                       else (255, 100, 0))
        draw.rectangle(
            [10, h - bar_h + 5, 10 + score_w, h - bar_h + 12],
            fill=score_color
        )

        # Info text
        info = f"score: {score:.4f} | {label} | {os.path.basename(path)}"
        if track_id is not None:
            info = f"Track#{track_id} | " + info
        draw.text((10, h - bar_h + 15), info, fill=(255, 255, 255))

        # Timestamp for video
        ts = r.get("timestamp", "")
        if ts:
            draw.text((w - 120, h - bar_h + 15), f"t={ts}",
                       fill=(255, 255, 0))

        # Track timeline summary
        if r.get("track_first_seen") and r.get("track_last_seen"):
            first = r["track_first_seen"]
            last = r["track_last_seen"]
            fc = r.get("track_frame_count", 0)
            tl_text = f"Seen: {first}s → {last}s ({fc} frames)"
            draw.text((10, h - bar_h - 18), tl_text, fill=(0, 255, 255))

        if save:
            fname = f"result_{shown:02d}_{os.path.basename(path)}"
            out_path = os.path.join(save, fname)
            img.save(out_path)
            print(f"  Saved: {out_path}")
        else:
            img.show()

        shown += 1

    if shown == 0:
        print("[SightRAG] No results to visualize.")
    elif save:
        print(f"[SightRAG] {shown} annotated images saved to {save}/")


def _draw_mask(img, mask, label=""):
    """Draw semi-transparent mask overlay on image."""
    if not isinstance(mask, np.ndarray):
        return img

    img_array = np.array(img)
    h, w = img_array.shape[:2]

    # Resize mask if needed
    if mask.shape != (h, w):
        from PIL import Image as PILImage
        mask_pil = PILImage.fromarray(mask.astype(np.uint8) * 255)
        mask_pil = mask_pil.resize((w, h), PILImage.NEAREST)
        mask = np.array(mask_pil) > 127

    # Get color for label
    color = COLORS.get(label, DEFAULT_COLOR)

    # Create colored overlay
    overlay = np.zeros_like(img_array)
    overlay[mask] = color

    # Blend: 30% overlay
    alpha = 0.3
    img_array = np.where(
        mask[:, :, np.newaxis],
        (img_array * (1 - alpha) + overlay * alpha).astype(np.uint8),
        img_array
    )

    # Draw mask border
    from PIL import ImageFilter
    mask_img = Image.fromarray(mask.astype(np.uint8) * 255)
    edges = mask_img.filter(ImageFilter.FIND_EDGES)
    edge_array = np.array(edges) > 127
    img_array[edge_array] = color

    return Image.fromarray(img_array)


def show_timeline(timeline_data, save=None):
    """
    Visualize a track timeline as a horizontal strip.

    Shows key frames from the track's journey through the video.

    Args:
        timeline_data: dict from rag.timeline() with states, trajectory
        save: path to save the visualization
    """
    states = timeline_data.get("states", [])
    if not states:
        print("[SightRAG] No timeline states to visualize.")
        return

    track_id = timeline_data.get("track_id", "?")
    label = timeline_data.get("label", "")

    # Sample up to 8 key frames
    n_show = min(8, len(states))
    if len(states) > n_show:
        indices = np.linspace(0, len(states) - 1, n_show, dtype=int)
        samples = [states[i] for i in indices]
    else:
        samples = states

    # Create strip image
    thumb_w = 200
    thumb_h = 150
    margin = 5
    strip_w = (thumb_w + margin) * len(samples) + margin
    strip_h = thumb_h + 80

    strip = Image.new("RGB", (strip_w, strip_h), (30, 30, 30))
    draw = ImageDraw.Draw(strip)

    # Title
    title = f"Track #{track_id}: {label}"
    draw.text((margin, 5), title, fill=(255, 255, 255))

    time_range = (
        f"{timeline_data.get('first_timestamp', '?')}s → "
        f"{timeline_data.get('last_timestamp', '?')}s  "
        f"({timeline_data.get('frame_count', 0)} frames)"
    )
    draw.text((margin, 22), time_range, fill=(0, 255, 255))

    # Draw each sampled frame
    y_offset = 45
    for idx, state in enumerate(samples):
        x = margin + idx * (thumb_w + margin)

        video_path = timeline_data.get("video_path", "")
        bbox = state.get("bbox", [])
        ts = state.get("timestamp", "")

        # Try to load frame crop
        if os.path.exists(video_path) and bbox:
            try:
                from .utils.video import get_frame_at_timestamp
                frame = get_frame_at_timestamp(video_path, float(ts))
                if frame is not None:
                    x1, y1, x2, y2 = bbox
                    crop = frame.crop((
                        max(0, x1 - 20), max(0, y1 - 20),
                        min(frame.width, x2 + 20),
                        min(frame.height, y2 + 20)
                    ))
                    crop = crop.resize((thumb_w, thumb_h))
                    strip.paste(crop, (x, y_offset))
            except Exception:
                # Draw placeholder
                draw.rectangle(
                    [x, y_offset, x + thumb_w, y_offset + thumb_h],
                    fill=(60, 60, 60), outline=(100, 100, 100)
                )
        else:
            draw.rectangle(
                [x, y_offset, x + thumb_w, y_offset + thumb_h],
                fill=(60, 60, 60), outline=(100, 100, 100)
            )

        # Timestamp label
        draw.text(
            (x + 5, y_offset + thumb_h + 2),
            f"t={ts}s", fill=(255, 255, 0)
        )

        # Frame index
        fi = state.get("frame_idx", "")
        draw.text(
            (x + 5, y_offset + thumb_h + 16),
            f"F{fi}", fill=(180, 180, 180)
        )

    if save:
        os.makedirs(os.path.dirname(save) if os.path.dirname(save) else ".",
                     exist_ok=True)
        strip.save(save)
        print(f"[SightRAG] Timeline saved: {save}")
    else:
        strip.show()
