"""SightRAG result visualization — rag.show()"""

import os
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


def show_results(results, save=None, max_show=5):
    """
    Display or save results with clear bounding boxes.
    
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
        
        draw = ImageDraw.Draw(img)
        bbox = r.get("bbox", [])
        label = r.get("label", "")
        score = r.get("score", 0)
        
        w, h = img.size
        
        # Skip drawing if whole_image (full image bbox)
        is_whole = (label == "whole_image" or 
                    (bbox and bbox == [0, 0, w, h]))
        
        if bbox and len(bbox) == 4 and not is_whole:
            x1, y1, x2, y2 = bbox
            color = COLORS.get(label, DEFAULT_COLOR)
            
            # Thick bounding box with double border for contrast
            # Outer dark border
            draw.rectangle([x1-1, y1-1, x2+1, y2+1], outline=(0, 0, 0), width=4)
            # Inner colored border
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            
            # Label with background — high contrast
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
            # Colored text
            draw.text((x1 + 4, label_y + 2), text, fill=color)
            
            # Corner markers for extra clarity
            corner_len = min(20, (x2-x1)//4, (y2-y1)//4)
            for cx, cy, dx, dy in [
                (x1, y1, 1, 1), (x2, y1, -1, 1),
                (x1, y2, 1, -1), (x2, y2, -1, -1)
            ]:
                draw.line([(cx, cy), (cx + corner_len*dx, cy)], fill=color, width=3)
                draw.line([(cx, cy), (cx, cy + corner_len*dy)], fill=color, width=3)
        
        # Bottom info bar
        bar_h = 35
        draw.rectangle([0, h - bar_h, w, h], fill=(0, 0, 0))
        
        # Score bar (visual score indicator)
        score_w = int(score * (w - 20))
        score_color = (0, 255, 0) if score > 0.7 else (255, 255, 0) if score > 0.4 else (255, 100, 0)
        draw.rectangle([10, h - bar_h + 5, 10 + score_w, h - bar_h + 12], fill=score_color)
        
        # Info text
        info = f"score: {score:.4f} | {label} | {os.path.basename(path)}"
        draw.text((10, h - bar_h + 15), info, fill=(255, 255, 255))
        
        # Timestamp for video
        ts = r.get("timestamp", "")
        if ts:
            draw.text((w - 120, h - bar_h + 15), f"t={ts}", fill=(255, 255, 0))
        
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
