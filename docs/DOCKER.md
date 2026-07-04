# SightRAG Docker Guide

## Quick Start

```bash
docker-compose up --build
```

## Run SightRAG in Docker

### Step 1 — Build

```bash
docker build -t sightrag .
```

### Step 2 — Run with your images

```bash
docker run -v $(pwd)/my_photos:/app/data \
           -v $(pwd)/sightrag_index:/app/sightrag_index \
           sightrag \
           python -c "
from sightrag import SightRAG
rag = SightRAG()
rag.index('/app/data/')
results = rag.query('find empty shelf')
print(results)
"
```

### Step 3 — Run interactive

```bash
docker run -it \
           -v $(pwd)/my_photos:/app/data \
           -v $(pwd)/sightrag_index:/app/sightrag_index \
           sightrag \
           python
```

Then inside Python:

```python
from sightrag import SightRAG
rag = SightRAG()
rag.index("/app/data/")
results = rag.query("find empty shelf")
```

## Volumes

| Volume | Purpose |
|--------|---------|
| `/app/data` | Your image/video files |
| `/app/sightrag_index` | Persistent vector store index |

## GPU Support

```bash
docker run --gpus all \
           -v $(pwd)/my_photos:/app/data \
           sightrag \
           python -c "
from sightrag import SightRAG
rag = SightRAG()
rag.index('/app/data/')
"
```

Requires NVIDIA Docker runtime installed.

## docker-compose.yml (default)

```yaml
version: "3.8"

services:
  sightrag:
    build: .
    volumes:
      - ./data:/app/data
      - ./sightrag_index:/app/sightrag_index
    environment:
      - PYTHONUNBUFFERED=1
```

## docker-compose with GPU

```yaml
version: "3.8"

services:
  sightrag:
    build: .
    volumes:
      - ./data:/app/data
      - ./sightrag_index:/app/sightrag_index
    environment:
      - PYTHONUNBUFFERED=1
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

## Camera in Docker

Camera access requires device passthrough:

```bash
docker run --device=/dev/video0 \
           -v $(pwd)/sightrag_index:/app/sightrag_index \
           sightrag \
           python -c "
from sightrag import SightRAG
rag = SightRAG()
rag.index(source='camera')
"
```

## Notes

- First run downloads YOLO (~6MB) and CLIP (~350MB) model weights
- Subsequent runs use cached models
- Index persists in `sightrag_index` volume between runs
- Index once, search forever — no re-indexing needed
