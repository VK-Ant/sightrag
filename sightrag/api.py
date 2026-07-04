# sightrag/api.py
# REST API — FastAPI based
# Run: sightrag-server or python -m sightrag.api

import os
import json
import shutil
import tempfile
from pathlib import Path

try:
    from fastapi import FastAPI, UploadFile, File, Form, HTTPException
    from fastapi.responses import JSONResponse
    import uvicorn
except ImportError:
    raise ImportError(
        "FastAPI not installed.\n"
        "Run: pip install sightrag[api]"
    )

from .core import SightRAG

app = FastAPI(
    title="SightRAG API",
    description="See. Search. Retrieve. — Image and Video RAG",
    version="0.1.0"
)

# Global SightRAG instance
rag = None


def get_rag():
    global rag
    if rag is None:
        store = os.getenv("SIGHTRAG_STORE", "sqlite")
        domain = os.getenv("SIGHTRAG_DOMAIN", None)
        index_path = os.getenv("SIGHTRAG_INDEX", "./sightrag_index")
        rag = SightRAG(store=store, domain_hint=domain, index_path=index_path)
    return rag


@app.get("/")
def root():
    return {
        "name": "SightRAG API",
        "version": "0.1.0",
        "tagline": "See. Search. Retrieve.",
        "endpoints": {
            "POST /index/folder": "Index an image folder",
            "POST /index/video": "Index a video file",
            "POST /index/upload": "Upload and index images",
            "POST /query/text": "Search with text",
            "POST /query/reference": "Search with reference image",
            "GET /status": "Index status",
            "DELETE /index": "Clear index"
        }
    }


@app.get("/status")
def status():
    r = get_rag()
    return {
        "indexed_regions": r.count(),
        "store": r._store_type,
        "domain_hint": r.domain_hint
    }


@app.post("/index/folder")
def index_folder(path: str = Form(...)):
    """Index all images in a folder."""
    r = get_rag()
    try:
        r.index(path)
        return {
            "status": "success",
            "indexed_regions": r.count(),
            "source": path
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/index/video")
def index_video(path: str = Form(...), fps: int = Form(1)):
    """Index a video file."""
    r = get_rag()
    try:
        r.index(path, fps=fps)
        return {
            "status": "success",
            "indexed_regions": r.count(),
            "source": path,
            "fps": fps
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/index/upload")
async def index_upload(files: list[UploadFile] = File(...)):
    """Upload and index images directly."""
    r = get_rag()
    upload_dir = tempfile.mkdtemp(prefix="sightrag_upload_")

    try:
        # Save uploaded files
        for f in files:
            file_path = os.path.join(upload_dir, f.filename)
            with open(file_path, "wb") as out:
                content = await f.read()
                out.write(content)

        # Index the upload folder
        r.index(upload_dir)

        return {
            "status": "success",
            "files_uploaded": len(files),
            "indexed_regions": r.count()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        shutil.rmtree(upload_dir, ignore_errors=True)


@app.post("/query/text")
def query_text(text: str = Form(...), top_k: int = Form(5)):
    """Search with plain English text."""
    r = get_rag()
    try:
        results = r.query(text=text, top_k=top_k)
        return {
            "query": text,
            "results": results,
            "count": len(results)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/query/reference")
async def query_reference(
    file: UploadFile = File(...),
    top_k: int = Form(5)
):
    """Search using a reference image."""
    r = get_rag()

    # Save reference temporarily
    tmp = tempfile.NamedTemporaryFile(
        delete=False, suffix=f"_{file.filename}"
    )
    try:
        content = await file.read()
        tmp.write(content)
        tmp.close()

        results = r.query(reference=tmp.name, top_k=top_k)
        return {
            "reference": file.filename,
            "results": results,
            "count": len(results)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        os.unlink(tmp.name)


@app.delete("/index")
def clear_index():
    """Clear all indexed data."""
    r = get_rag()
    r.clear()
    return {"status": "cleared", "indexed_regions": 0}


def serve(host: str = "0.0.0.0", port: int = 8000):
    """Start the SightRAG API server."""
    print(f"[SightRAG] Starting API server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    serve()
