"""
SightRAG OCR — reads text on images during indexing.
Text stored alongside visual embeddings.
Query matches BOTH visual + text.

Usage:
    rag = SightRAG(ocr=True)
    rag.index("./store_photos/")
    results = rag.query("find Calgon")  # matches OCR text
"""

import os
import numpy as np
from PIL import Image


class OCREngine:
    """
    Reads text from image regions.
    Runs at INDEX time only — zero query overhead.
    Supports: EasyOCR (default), PaddleOCR, Tesseract.
    """
    
    def __init__(self, engine="easyocr", languages=None):
        self.engine_name = engine
        self.languages = languages or ["en"]
        self._engine = None
        self._load(engine)
    
    def _load(self, engine):
        if engine == "easyocr":
            try:
                import easyocr
                self._engine = easyocr.Reader(
                    self.languages, 
                    gpu=self._has_gpu(),
                    verbose=False
                )
                return
            except ImportError:
                pass
        
        if engine == "paddleocr" or self._engine is None:
            try:
                from paddleocr import PaddleOCR
                self._engine = PaddleOCR(
                    use_angle_cls=True,
                    lang="en",
                    show_log=False
                )
                self.engine_name = "paddleocr"
                return
            except ImportError:
                pass
        
        if self._engine is None:
            try:
                import pytesseract
                self._engine = "tesseract"
                self.engine_name = "tesseract"
                return
            except ImportError:
                raise ImportError(
                    "OCR requires one of:\n"
                    "  pip install easyocr         (recommended)\n"
                    "  pip install paddleocr\n"
                    "  pip install pytesseract"
                )
    
    def _has_gpu(self):
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def read(self, image):
        """
        Read text from image.
        Returns: {"text": "full text", "words": [...], "confidence": float}
        """
        if isinstance(image, str):
            image = Image.open(image)
        
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Skip very small images
        w, h = image.size
        if w < 20 or h < 20:
            return {"text": "", "words": [], "confidence": 0.0}
        
        try:
            if self.engine_name == "easyocr":
                return self._read_easyocr(image)
            elif self.engine_name == "paddleocr":
                return self._read_paddleocr(image)
            elif self.engine_name == "tesseract":
                return self._read_tesseract(image)
        except Exception:
            return {"text": "", "words": [], "confidence": 0.0}
    
    def _read_easyocr(self, image):
        import numpy as np
        img_array = np.array(image)
        results = self._engine.readtext(img_array)
        
        words = []
        confidences = []
        for (bbox, text, conf) in results:
            if conf > 0.3:  # skip low confidence
                words.append(text.strip())
                confidences.append(conf)
        
        full_text = " ".join(words)
        avg_conf = float(np.mean(confidences)) if confidences else 0.0
        
        return {"text": full_text, "words": words, "confidence": avg_conf}
    
    def _read_paddleocr(self, image):
        import numpy as np
        img_array = np.array(image)
        results = self._engine.ocr(img_array, cls=True)
        
        words = []
        confidences = []
        if results and results[0]:
            for line in results[0]:
                text = line[1][0]
                conf = line[1][1]
                if conf > 0.3:
                    words.append(text.strip())
                    confidences.append(conf)
        
        full_text = " ".join(words)
        avg_conf = float(np.mean(confidences)) if confidences else 0.0
        
        return {"text": full_text, "words": words, "confidence": avg_conf}
    
    def _read_tesseract(self, image):
        import pytesseract
        text = pytesseract.image_to_string(image).strip()
        words = [w for w in text.split() if len(w) > 1]
        return {"text": " ".join(words), "words": words, "confidence": 0.5}
