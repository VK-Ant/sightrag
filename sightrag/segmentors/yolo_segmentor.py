"""
YOLO-Seg segmentor - default segmentation using ultralytics YOLO11-seg.

Fast, lightweight, same 80 COCO classes as YOLO detection.
Produces instance masks alongside bounding boxes.
"""

import numpy as np
from PIL import Image
from .base import SegmentorBase


class YOLOSegmentor(SegmentorBase):
    """YOLO11-seg instance segmentation."""

    name = "yolo-seg"

    def __init__(self, model_name: str = "yolo11n-seg.pt"):
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics required for YOLO segmentation. "
                "Install: pip install ultralytics"
            )

        self.model = YOLO(model_name)
        self._min_area_ratio = 0.03

    def segment(self, image, confidence: float = 0.25):
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            pil_image = image

        img_w, img_h = pil_image.size
        img_area = img_w * img_h

        results = self.model(pil_image, verbose=False, conf=confidence)

        detections = []
        if not results or len(results) == 0:
            return detections

        result = results[0]

        if result.masks is None or result.boxes is None:
            return detections

        boxes = result.boxes
        masks = result.masks

        for i in range(len(boxes)):
            box = boxes[i]
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0].cpu())
            cls_id = int(box.cls[0].cpu())
            label = result.names.get(cls_id, f"class_{cls_id}")

            # Min area filter
            det_area = (x2 - x1) * (y2 - y1)
            if det_area < img_area * self._min_area_ratio:
                continue

            # Get mask for this instance
            mask_data = masks.data[i].cpu().numpy()
            # Resize mask to original image size
            if mask_data.shape != (img_h, img_w):
                from PIL import Image as PILImage
                mask_pil = PILImage.fromarray(
                    (mask_data * 255).astype(np.uint8)
                )
                mask_pil = mask_pil.resize((img_w, img_h), PILImage.NEAREST)
                mask_data = np.array(mask_pil) > 127

            # Crop
            x1c = max(0, x1)
            y1c = max(0, y1)
            x2c = min(img_w, x2)
            y2c = min(img_h, y2)
            crop = pil_image.crop((x1c, y1c, x2c, y2c))

            detections.append({
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "label": label,
                "confidence": round(conf, 4),
                "crop": crop,
                "mask": mask_data.astype(bool),
            })

        # Sort by area descending
        detections.sort(
            key=lambda d: (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1]),
            reverse=True
        )

        return detections
