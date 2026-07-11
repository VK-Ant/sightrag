"""SightRAG result visualization — rag.show()"""

import os
from PIL import Image, ImageDraw, ImageFont

COLORS = {
    "person": "green", "car": "red", "truck": "red",
    "bottle": "orange", "cup": "orange", "chair": "brown",
    "dog": "yellow", "cat": "yellow", "horse": "yellow",
    "bus": "red", "motorcycle": "red", "bicycle": "blue",
    "whole_image": "cyan",
}


def show_results(results, save=None, max_show=5):
    """
    Display or save results with bounding boxes drawn.
    
    Usage:
        results = rag.query("find person")
        
        # Display on screen
        rag.show(results)
        
        # Save annotated images
        rag.show(results, save="./output/")
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
        color = COLORS.get(label, "cyan")
        
        if bbox and len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            
            # Bounding box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            
            # Label background
            text = f"{label} {score:.2f}"
            try:
                tw = draw.textlength(text)
            except:
                tw = len(text) * 7
            th = 14
            draw.rectangle([x1, y1 - th - 4, x1 + tw + 8, y1], fill=color)
            draw.text((x1 + 4, y1 - th - 2), text, fill="white")
        
        # Score bar at bottom
        w, h = img.size
        bar_h = 30
        draw.rectangle([0, h - bar_h, w, h], fill="black")
        info = f"score: {score:.4f} | {label} | {os.path.basename(path)}"
        draw.text((10, h - bar_h + 8), info, fill="white")
        
        # Timestamp for video
        ts = r.get("timestamp", "")
        if ts:
            draw.text((w - 150, h - bar_h + 8), f"t={ts}", fill="yellow")
        
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
