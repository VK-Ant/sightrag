<p align="center">
  <img src="https://raw.githubusercontent.com/VK-Ant/sightrag/main/assets/sightrag_banner.png" alt="SightRAG Banner" width="100%">
</p>

<h1 align="center">SightRAG</h1>
<h3 align="center">See. Search. Retrieve.</h3>

<p align="center">
  <a href="https://pypi.org/project/sightrag/"><img src="https://img.shields.io/pypi/v/sightrag" alt="PyPI"></a>
  <a href="https://github.com/VK-Ant/sightrag/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue" alt="License"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.9+-green" alt="Python"></a>
</p>

<p align="center">
  Search your images and videos with plain English. Three lines of code.
</p>

---

## Quick Start

```python
from sightrag import SightRAG

rag = SightRAG()
rag.index("./photos/")
results = rag.query("find empty shelf")
```

## Install

```bash
pip install sightrag
```

For REST API:
```bash
pip install sightrag[api]
```

## What SightRAG Does

Point SightRAG at any image folder, video file, or camera. Ask in plain English. Get back matched images with bounding boxes, timestamps, and confidence scores.

SightRAG is not a model. Not a wrapper. Not a framework plugin. It is a complete retrieval system that handles detection, embedding, indexing, and search — so you don't have to.

## Usage

### Image Folder

```python
from sightrag import SightRAG

rag = SightRAG()
rag.index("./shelf_photos/")
results = rag.query("find empty shelf near dairy")

for r in results:
    print(f"{r['image_path']} — score: {r['score']} — {r['label']}")
```

### Video File

```python
rag = SightRAG()
rag.index("./cctv_footage.mp4")
results = rag.query("person near exit door")

for r in results:
    print(f"Timestamp: {r['timestamp']} — score: {r['score']}")
```

### Live Camera

```python
rag = SightRAG()
rag.index(source="camera")              # default webcam
rag.index(source="camera", camera_id=1) # specific camera
results = rag.query("find person")
```

### Reference Image Query

```python
# Instead of text, search using a reference image
results = rag.query(reference="sample_shelf.jpg")
```

### Custom Domain (medical, industrial, satellite)

```python
rag = SightRAG(domain_hint="pcb defect solder joint circuit board")
rag.index("./circuit_boards/")
results = rag.query("find defective solder joint")
```

### SQLite Fallback (lightweight)

```python
# Use SQLite if ChromaDB can't install in your environment
rag = SightRAG(store="sqlite")
rag.index("./small_dataset/")
results = rag.query("find damaged product")
```

## Result Format

```python
[
    {
        "image_path":  "./photos/shelf_042.jpg",
        "score":       0.9134,
        "label":       "bottle",
        "confidence":  0.8721,
        "bbox":        [120, 45, 380, 290],
        "timestamp":   "",
        "source_type": "image"
    }
]
```

## How It Works

```
You bring               SightRAG does              You get
─────────────           ──────────────             ─────────────
Images                  Detects objects             Matched images
Videos                  Understands regions         Video timestamps
Camera                  Creates embeddings          Bounding boxes
Reference photo         Indexes everything          Confidence scores
Text query              Finds matches               Text answer
```

SightRAG uses YOLO for object detection and CLIP for semantic embeddings internally. You never configure or manage these models — SightRAG handles everything.

For custom domains where YOLO has no training data (medical, satellite, industrial), use `domain_hint` to guide CLIP embeddings. SightRAG falls back to whole-image CLIP embedding when YOLO detects nothing — it never fails silently.

## Storage

SightRAG uses ChromaDB as the default vector database — purpose-built for embeddings, fast at any scale, free, and local.

| Store | Scale | Cost | Usage |
|-------|-------|------|-------|
| ChromaDB (default) | Any scale | Free | `SightRAG()` |
| SQLite (fallback) | Up to 100k images | Free | `SightRAG(store="sqlite")` |

Enterprise connectors (Qdrant, Pinecone, Azure) coming in v2.

## REST API

```bash
pip install sightrag[api]
```

Start the server:

```bash
# Command line
sightrag-server

# Or from Python
from sightrag import serve
serve(port=8000)

# Or Docker (API starts automatically)
docker-compose up
```

API available at `http://localhost:8000`

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API info and available endpoints |
| GET | `/status` | Index count, store type, domain hint |
| POST | `/index/folder` | Index all images in a folder path |
| POST | `/index/video` | Index a video file |
| POST | `/index/upload` | Upload and index images directly |
| POST | `/query/text` | Search with plain English text |
| POST | `/query/reference` | Search with a reference image |
| DELETE | `/index` | Clear all indexed data |

### Examples

```bash
# Index a folder
curl -X POST http://localhost:8000/index/folder \
     -F "path=./data/"

# Search with text
curl -X POST http://localhost:8000/query/text \
     -F "text=find empty shelf" \
     -F "top_k=5"

# Upload and search with reference image
curl -X POST http://localhost:8000/query/reference \
     -F "file=@new_photo.jpg" \
     -F "top_k=5"

# Check status
curl http://localhost:8000/status
```

Interactive API docs available at `http://localhost:8000/docs` (Swagger UI).

## Docker

```bash
docker-compose up
```

This starts the REST API server on port 8000. See [Docker Guide](docs/DOCKER.md) for details.

## Honest Limitations (v0.1.0)

- Indexing ~500 images takes ~2 minutes on CPU (one-time cost — search is instant after)
- Custom domains (medical, satellite) use whole-image CLIP without region grounding
- Single webcam only (multiple cameras in v2)
- SQLite vector store loads all vectors into memory for search

## Roadmap

| Version | Features |
|---------|----------|
| v0.1 (current) | Image + Video + Camera + REST API + SQLite + ChromaDB |
| v0.2 | C++ core, CLI, Grounding DINO, Grounding SAM |
| v0.3 | Person Re-ID, scene graph, edge deployment |
| v1.0 | Jetson Orin, compliance modes (GDPR/HIPAA/DPDP) |

## Architecture

```
Input (images / video / camera)
        ↓
   Preprocessor (resize, validate, keyframe extract)
        ↓
   YOLO Detection (Tier 1 — standard domains)
        ↓ fallback if no detections
   CLIP Whole-Image (custom domain + domain_hint)
        ↓
   Embedding (CLIP-ViT-B/32)
        ↓
   Vector Store (SQLite default / ChromaDB optional)
        ↓
   Retrieval + Ranking (cosine similarity)
        ↓
Output (matched images, timestamps, bounding boxes, scores)
```

## Three Library Ecosystem

SightRAG is part of the VK-Ant AI ecosystem:

| Library | Purpose | Status |
|---------|---------|--------|
| [SightRAG](https://github.com/VK-Ant/sightrag) | Image & Video RAG | v0.1 |
| [adaptive-intelligence](https://pypi.org/project/adaptive-intelligence/) | RL-based RAG orchestration | v4.0 |
| [llmevalkit](https://pypi.org/project/llmevalkit/) | LLM evaluation (97+ tests) | Stable |

## License

Apache 2.0 — see [LICENSE](LICENSE)

## Author

Built by **Ant (VK-Ant)** — Kaggle Master, 8.5 years computer vision experience.

- GitHub: [VK-Ant](https://github.com/VK-Ant)
- LinkedIn: [VK-Ant](https://linkedin.com/in/vk-ant)
- Portfolio: [vk-ant.github.io](https://vk-ant.github.io/Venkatkumar)
