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

All models and indexes are stored in `~/.sightrag/` — your project folder stays clean.

## Project Structure

```
sightrag/
├── sightrag/                    ← main library (pip install)
│   ├── core.py                  ← SightRAG class
│   ├── detector.py              ← YOLO detection
│   ├── embedder.py              ← CLIP embeddings
│   ├── indexer.py               ← image/video/camera indexing
│   ├── retriever.py             ← text + reference queries
│   ├── api.py                   ← REST API (FastAPI)
│   ├── store/
│   │   ├── base.py              ← abstract store interface
│   │   ├── sqlite_store.py      ← SQLite fallback
│   │   └── chroma_store.py      ← ChromaDB (default)
│   └── utils/
│       ├── image.py             ← image loading
│       ├── video.py             ← frame extraction
│       └── camera.py            ← webcam capture
│
├── demo_sightrag/               ← test locally — run the scripts
│   ├── sightrag_images.py       ← demo: image folder indexing
│   ├── sightrag_video.py        ← demo: video file indexing
│   ├── sightrag_livecam.py      ← demo: live webcam capture
│   ├── sightrag_restapi.py      ← demo: REST API usage
│   ├── input_images/            ← sample images to index
│   ├── reference_images/        ← sample reference query images
│   ├── camera_captures/         ← webcam frames stored here
│   └── video_samples/           ← put your videos here
│
├── notebooks/                   ← test on Google Colab
│   └── SightRAG_Colab_Demo.ipynb
│
├── examples/                    ← code examples
│   ├── basic_usage.py
│   ├── camera_example.py
│   ├── custom_domain_example.py
│   └── rest_api_example.py
│
├── tests/                       ← unit tests
│   └── test_core.py
│
├── docs/                        ← documentation
│   └── DOCKER.md
│
├── assets/                      ← banner image
├── setup.py                     ← PyPI packaging
├── pyproject.toml               ← build config
├── requirements.txt             ← dependencies
├── Dockerfile                   ← container
├── docker-compose.yml           ← one command deploy
├── LICENSE                      ← Apache 2.0
└── test_sightrag.py             ← quick test script
```

## How To Test

### Quick Test (terminal)
```bash
python test_sightrag.py
```

### Demo Scripts (test each mode)
```bash
python demo_sightrag/sightrag_images.py     # image folder
python demo_sightrag/sightrag_video.py       # video files
python demo_sightrag/sightrag_livecam.py     # live webcam
python demo_sightrag/sightrag_restapi.py     # REST API
```

### Google Colab
Upload `notebooks/SightRAG_Colab_Demo.ipynb` to Google Colab → Run All

### Unit Tests
```bash
pip install pytest
python -m pytest tests/ -v
```

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

### Mixed Folder (images + videos)

```python
rag = SightRAG()
rag.index("./my_data/")  # automatically detects images AND videos
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
rag = SightRAG(store="sqlite")
rag.index("./small_dataset/")
```

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
| POST | `/index/folder` | Index all images and videos in a folder |
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

## Storage

SightRAG uses a built-in SQLite vector store by default — zero extra dependencies, works everywhere.

| Store | Scale | Cost | Usage |
|-------|-------|------|-------|
| SQLite (default) | Up to 100k images | Free | `SightRAG()` |
| ChromaDB (optional) | Large scale | Free | `SightRAG(store="chroma")` |

Enterprise connectors (Qdrant, Pinecone, Azure) coming in v2.

## Where SightRAG Stores Data

```
~/.sightrag/
├── models/      ← YOLO weights (auto-downloaded once)
└── index/       ← vector database (ChromaDB/SQLite)
```

Your project folder stays clean. No random `.pt` files or `sightrag_index/` folders appearing.

## Docker

```bash
docker-compose up
```

This starts the REST API server on port 8000. See [Docker Guide](docs/DOCKER.md) for details.

## Architecture

```
Input (images / video / camera / reference image)
        ↓
   Preprocessor (resize, validate, keyframe extract)
        ↓
   YOLO Detection (80 COCO classes + whole-image fallback)
        ↓
   CLIP Embedding (domain_hint enrichment for custom domains)
        ↓
   Vector Store (ChromaDB default / SQLite fallback)
        ↓
   Retrieval + Ranking (cosine similarity, confidence scoring)
        ↓
Output (matched images, timestamps, bounding boxes, scores)
```

## Honest Limitations (v0.1.0)

- Indexing ~500 images takes ~2 minutes on CPU (one-time cost — search is instant after)
- Custom domains (medical, satellite) use whole-image CLIP without region grounding
- Single webcam only (multiple cameras in v2)
- SQLite vector store loads all vectors into memory for search

## Roadmap

| Version | Features |
|---------|----------|
| v0.1 (current) | Image + Video + Camera + REST API + ChromaDB |
| v0.2 | C++ core, CLI, Grounding DINO, SAM |
| v0.3 | Person Re-ID, scene graph, edge deployment |
| v1.0 | Jetson Orin, compliance modes (GDPR/HIPAA/DPDP) |

## Three Library Ecosystem

SightRAG is part of the VK-Ant AI ecosystem:

| Library | Purpose | Status |
|---------|---------|--------|
| [SightRAG](https://github.com/VK-Ant/sightrag) | Image & Video RAG | v0.1 |
| [adaptive-intelligence](https://pypi.org/project/adaptive-intelligence/) | RL-based RAG orchestration | v4.0.7 |
| [llmevalkit](https://pypi.org/project/llmevalkit/) | LLM evaluation (97+ tests) | Stable |

## License

Apache 2.0 — [LICENSE](LICENSE)

## Author

Built by **Venkatkumar Rajan** — Kaggle Master, 8.5+ years computer vision experience.

- GitHub: [VK-Ant](https://github.com/VK-Ant)
- LinkedIn: [VK](https://linkedin.com/in/vk-ant)
- Portfolio: [vk-ant.github.io](https://vk-ant.github.io/Venkatkumar)
