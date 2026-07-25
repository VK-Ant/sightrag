"""
SightRAG Multimodal — LLM understanding at query time.
Runs ONLY on top candidates (not all images).
Local models by default. API optional.

Usage:
    rag = SightRAG(ocr=True, multimodal="qwen2-vl")
    results = rag.query("find damaged product", understand=True)
"""

import os
import numpy as np
from PIL import Image


class MultimodalEngine:
    """
    Multimodal LLM for semantic understanding.
    Runs at QUERY time only — on top N candidates.
    Never runs on all images. Speed first.
    """
    
    def __init__(self, model="qwen2-vl", api_key=None):
        self.model_name = model
        self.api_key = api_key
        self._model = None
        self._processor = None
        self._load(model)
    
    def _load(self, model):
        if model in ("qwen2-vl", "qwen2-vl-2b"):
            self._load_qwen("Qwen/Qwen2-VL-2B-Instruct")
        elif model == "qwen2-vl-7b":
            self._load_qwen("Qwen/Qwen2-VL-7B-Instruct")
        elif model in ("gpt-4o", "gpt-4-vision"):
            self._setup_openai()
        elif model == "local-only":
            self._load_qwen("Qwen/Qwen2-VL-2B-Instruct")
        else:
            raise ValueError(
                f"Unknown model: {model}\n"
                f"Options: 'qwen2-vl', 'qwen2-vl-7b', 'gpt-4o'"
            )
    
    def _load_qwen(self, model_id):
        try:
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            import torch
            
            self._processor = AutoProcessor.from_pretrained(model_id)
            self._model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_id, torch_dtype=torch.float16
            )
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = self._model.to(device)
            self._model.eval()
            self.model_name = "qwen2-vl-local"
        except ImportError:
            raise ImportError(
                "Multimodal requires: pip install transformers torch\n"
                "Or: pip install sightrag[multimodal]"
            )
    
    def _setup_openai(self):
        try:
            import openai
            if not self.api_key:
                self.api_key = os.environ.get("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError("Set OPENAI_API_KEY or pass api_key=")
            self._client = openai.OpenAI(api_key=self.api_key)
            self.model_name = "gpt-4o"
        except ImportError:
            raise ImportError("pip install openai")
    
    def understand(self, image, query):
        """
        Ask multimodal LLM about an image.
        Returns: {"answer": str, "relevance": float}
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        
        if "local" in self.model_name or "qwen" in self.model_name:
            return self._understand_local(image, query)
        elif "gpt" in self.model_name:
            return self._understand_api(image, query)
        else:
            return {"answer": "", "relevance": 0.5}
    
    def _understand_local(self, image, query):
        import torch
        try:
            prompt = (
                f"Look at this image carefully. "
                f"User is searching for: '{query}'. "
                f"Rate relevance from 0.0 to 1.0. "
                f"Reply with ONLY a number like 0.85"
            )
            
            messages = [
                {"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]}
            ]
            
            text = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self._processor(
                text=[text], images=[image],
                return_tensors="pt", padding=True
            ).to(self._model.device)
            
            with torch.no_grad():
                output_ids = self._model.generate(
                    **inputs, max_new_tokens=10
                )
            
            response = self._processor.batch_decode(
                output_ids[:, inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )[0].strip()
            
            # Parse relevance score
            try:
                relevance = float(response)
                relevance = max(0.0, min(1.0, relevance))
            except ValueError:
                relevance = 0.5
            
            return {"answer": response, "relevance": relevance}
            
        except Exception:
            return {"answer": "", "relevance": 0.5}
    
    def _understand_api(self, image, query):
        import base64
        from io import BytesIO
        
        try:
            buf = BytesIO()
            image.save(buf, format="JPEG")
            b64 = base64.b64encode(buf.getvalue()).decode()
            
            response = self._client.chat.completions.create(
                model="gpt-4o",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/jpeg;base64,{b64}"
                        }},
                        {"type": "text", "text": (
                            f"User searches for: '{query}'. "
                            f"Rate image relevance 0.0 to 1.0. "
                            f"Reply ONLY a number."
                        )}
                    ]
                }],
                max_tokens=10
            )
            
            text = response.choices[0].message.content.strip()
            try:
                relevance = float(text)
                relevance = max(0.0, min(1.0, relevance))
            except ValueError:
                relevance = 0.5
            
            return {"answer": text, "relevance": relevance}
            
        except Exception:
            return {"answer": "", "relevance": 0.5}
    
    def rerank_with_understanding(self, query, candidates, top_k=5):
        """
        Re-rank candidates using multimodal understanding.
        Runs LLM on top candidates ONLY. Not all images.
        """
        scored = []
        
        for candidate in candidates:
            path = candidate.get("image_path", "")
            if not path or not os.path.exists(path):
                scored.append((candidate, candidate.get("score", 0)))
                continue
            
            try:
                image = Image.open(path).convert("RGB")
                
                # Crop to bbox if available
                bbox = candidate.get("bbox", [])
                if bbox and len(bbox) == 4:
                    x1, y1, x2, y2 = bbox
                    if (x2-x1) > 20 and (y2-y1) > 20:
                        image = image.crop((x1, y1, x2, y2))
                
                result = self.understand(image, query)
                relevance = result["relevance"]
                
                # Blend: 60% visual score + 40% LLM understanding
                visual_score = candidate.get("score", 0)
                blended = 0.6 * visual_score + 0.4 * relevance
                
                scored.append((candidate, blended))
            except Exception:
                scored.append((candidate, candidate.get("score", 0)))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for candidate, score in scored[:top_k]:
            candidate["score"] = round(score, 4)
            results.append(candidate)
        
        return results
