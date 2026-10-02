"""
SAM2 segmentor - Segment Anything Model 2.

Segments ANY object with zero-shot capability.
Supports point prompts, box prompts, and automatic mask generation.
Uses HuggingFace transformers pipeline or local weights.
"""

import numpy as np
from PIL import Image
from .base import SegmentorBase


class SAM2Segmentor(SegmentorBase):
    """
    SAM2 (Segment Anything Model 2) segmentation.

    Modes:
        - "auto": automatic mask generation (no prompt needed)
        - "box": use bounding boxes from a detector as prompts
        - "point": use point clicks as prompts

    Usage:
        segmentor = SAM2Segmentor()  # default: auto mode
        segmentor = SAM2Segmentor(mode="box", detector=my_detector)
    """

    name = "sam2"

    def __init__(self, model_name: str = "facebook/sam2-hiera-small",
                 mode: str = "auto", detector=None, device: str = None):
        self.model_name = model_name
        self.mode = mode
        self._detector = detector
        self._device = device
        self._model = None
        self._processor = None
        self._mask_generator = None

    def _load_model(self):
        """Lazy load SAM2 model."""
        if self._model is not None:
            return

        try:
            from transformers import Sam2Model, Sam2Processor
        except ImportError:
            try:
                from transformers import SamModel as Sam2Model
                from transformers import SamProcessor as Sam2Processor
            except ImportError:
                raise ImportError(
                    "transformers required for SAM2. "
                    "Install: pip install transformers torch"
                )

        import torch

        if self._device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"[SightRAG] Loading SAM2: {self.model_name}")
        self._processor = Sam2Processor.from_pretrained(self.model_name)
        self._model = Sam2Model.from_pretrained(self.model_name)
        self._model.to(self._device)
        self._model.eval()

    def segment(self, image, confidence: float = 0.25):
        self._load_model()

        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            pil_image = image

        if self.mode == "box" and self._detector:
            return self._segment_with_boxes(pil_image, confidence)
        else:
            return self._segment_auto(pil_image, confidence)

    def _segment_auto(self, image, confidence):
        """Automatic mask generation - segments everything visible."""
        import torch

        img_w, img_h = image.size

        # Use grid of points as prompts for auto mode
        grid_points = []
        step = min(img_w, img_h) // 8
        for y in range(step, img_h, step):
            for x in range(step, img_w, step):
                grid_points.append([x, y])

        if not grid_points:
            grid_points = [[img_w // 2, img_h // 2]]

        detections = []

        # Process in batches of points
        batch_size = 16
        for i in range(0, len(grid_points), batch_size):
            batch_points = grid_points[i:i + batch_size]

            for point in batch_points:
                try:
                    inputs = self._processor(
                        images=image,
                        input_points=[[[point]]],
                        return_tensors="pt"
                    )
                    inputs = {k: v.to(self._device) for k, v in inputs.items()}

                    with torch.no_grad():
                        outputs = self._model(**inputs)

                    masks = self._processor.post_process_masks(
                        outputs.pred_masks,
                        inputs.get("original_sizes", [[img_h, img_w]]),
                        inputs.get("reshaped_input_sizes",
                                   [[img_h, img_w]])
                    )

                    if masks and len(masks) > 0:
                        scores = outputs.iou_scores[0][0]
                        mask_set = masks[0][0]

                        best_idx = scores.argmax().item()
                        score = float(scores[best_idx])

                        if score < confidence:
                            continue

                        mask = mask_set[best_idx].cpu().numpy().astype(bool)

                        if mask.ndim == 3:
                            mask = mask[0]

                        # Resize if needed
                        if mask.shape != (img_h, img_w):
                            mask_pil = Image.fromarray(
                                mask.astype(np.uint8) * 255
                            )
                            mask_pil = mask_pil.resize(
                                (img_w, img_h), Image.NEAREST
                            )
                            mask = np.array(mask_pil) > 127

                        # Get bbox from mask
                        ys, xs = np.where(mask)
                        if len(xs) == 0:
                            continue

                        x1, y1 = int(xs.min()), int(ys.min())
                        x2, y2 = int(xs.max()), int(ys.max())

                        # Min area
                        area = (x2 - x1) * (y2 - y1)
                        if area < img_w * img_h * 0.01:
                            continue

                        crop = image.crop((x1, y1, x2, y2))

                        detections.append({
                            "bbox": [x1, y1, x2, y2],
                            "label": "object",
                            "confidence": round(score, 4),
                            "crop": crop,
                            "mask": mask,
                        })

                except Exception:
                    continue

        # Deduplicate overlapping masks
        detections = self._deduplicate(detections)

        return detections

    def _segment_with_boxes(self, image, confidence):
        """Use detector bboxes as SAM2 prompts for precise masks."""
        import torch

        img_w, img_h = image.size

        # Get detections from the paired detector
        raw_dets = self._detector.detect(image, confidence=confidence)

        detections = []
        for det in raw_dets:
            bbox = det["bbox"]

            try:
                inputs = self._processor(
                    images=image,
                    input_boxes=[[[bbox]]],
                    return_tensors="pt"
                )
                inputs = {k: v.to(self._device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self._model(**inputs)

                masks = self._processor.post_process_masks(
                    outputs.pred_masks,
                    inputs.get("original_sizes", [[img_h, img_w]]),
                    inputs.get("reshaped_input_sizes", [[img_h, img_w]])
                )

                if masks and len(masks) > 0:
                    scores = outputs.iou_scores[0][0]
                    mask_set = masks[0][0]
                    best_idx = scores.argmax().item()
                    mask = mask_set[best_idx].cpu().numpy().astype(bool)

                    if mask.ndim == 3:
                        mask = mask[0]

                    if mask.shape != (img_h, img_w):
                        mask_pil = Image.fromarray(
                            mask.astype(np.uint8) * 255
                        )
                        mask_pil = mask_pil.resize(
                            (img_w, img_h), Image.NEAREST
                        )
                        mask = np.array(mask_pil) > 127
                else:
                    mask = self._bbox_to_mask(bbox, img_w, img_h)

            except Exception:
                mask = self._bbox_to_mask(bbox, img_w, img_h)

            crop = det.get("crop", image.crop(tuple(bbox)))

            detections.append({
                "bbox": bbox,
                "label": det.get("label", "object"),
                "confidence": det.get("confidence", 0.5),
                "crop": crop,
                "mask": mask,
            })

        return detections

    def segment_region(self, image, bbox):
        """Segment a specific region using bbox as prompt."""
        self._load_model()
        import torch

        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            pil_image = image

        img_w, img_h = pil_image.size

        try:
            inputs = self._processor(
                images=pil_image,
                input_boxes=[[[bbox]]],
                return_tensors="pt"
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model(**inputs)

            masks = self._processor.post_process_masks(
                outputs.pred_masks,
                inputs.get("original_sizes", [[img_h, img_w]]),
                inputs.get("reshaped_input_sizes", [[img_h, img_w]])
            )

            if masks and len(masks) > 0:
                scores = outputs.iou_scores[0][0]
                best_idx = scores.argmax().item()
                mask = masks[0][0][best_idx].cpu().numpy().astype(bool)
                if mask.ndim == 3:
                    mask = mask[0]
                return mask

        except Exception:
            pass

        return self._bbox_to_mask(bbox, img_w, img_h)

    @staticmethod
    def _bbox_to_mask(bbox, w, h):
        """Fallback: create rectangular mask from bbox."""
        mask = np.zeros((h, w), dtype=bool)
        x1, y1, x2, y2 = [int(v) for v in bbox]
        mask[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] = True
        return mask

    def _deduplicate(self, detections, iou_threshold=0.5):
        """Remove overlapping masks by IoU."""
        if len(detections) <= 1:
            return detections

        keep = []
        used = set()

        # Sort by confidence descending
        detections.sort(key=lambda d: d["confidence"], reverse=True)

        for i, det in enumerate(detections):
            if i in used:
                continue
            keep.append(det)
            for j in range(i + 1, len(detections)):
                if j in used:
                    continue
                iou = self._bbox_iou(det["bbox"], detections[j]["bbox"])
                if iou > iou_threshold:
                    used.add(j)

        return keep
