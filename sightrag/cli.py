"""
SightRAG CLI — terminal commands for visual RAG.

Usage:
    sightrag index ./photos/
    sightrag index ./video.mp4 --fps 2
    sightrag query "find person near door"
    sightrag query --reference ./suspect.jpg
    sightrag show --query "find person" --save ./output/
    sightrag status
    sightrag clear
    sightrag serve --port 8000

Install: pip install sightrag[cli]
"""

import os
import sys
import json
import click


@click.group()
@click.version_option(version="0.3.0", prog_name="sightrag")
def cli():
    """SightRAG — See. Search. Retrieve."""
    pass


@cli.command()
@click.argument("path")
@click.option("--fps", default=1, help="Frames per second for video")
@click.option("--store", default="sqlite", help="Vector store: sqlite, chroma, qdrant")
@click.option("--detector", default=None, help="Detector: yolo, grounding-dino")
@click.option("--domain", default=None, help="Domain hint for better accuracy")
def index(path, fps, store, detector, domain):
    """Index images, video, or folder."""
    from sightrag import SightRAG
    
    kwargs = {"store": store}
    if domain:
        kwargs["domain_hint"] = domain
    if detector:
        if detector == "grounding-dino":
            from sightrag.detectors.grounding_dino import GroundingDINODetector
            kwargs["detector"] = GroundingDINODetector()
    
    rag = SightRAG(**kwargs)
    rag.index(path, fps=fps)


@cli.command()
@click.argument("text", required=False)
@click.option("--reference", "-r", default=None, help="Reference image path")
@click.option("--top-k", "-k", default=5, help="Number of results")
@click.option("--format", "fmt", default="text", help="Output: text, json")
def query(text, reference, top_k, fmt):
    """Search indexed content."""
    from sightrag import SightRAG
    
    rag = SightRAG()
    
    if reference:
        results = rag.query(reference=reference, top_k=top_k)
    elif text:
        results = rag.query(text=text, top_k=top_k)
    else:
        click.echo("Provide query text or --reference image")
        return
    
    if fmt == "json":
        click.echo(json.dumps(results, indent=2))
    else:
        for i, r in enumerate(results, 1):
            path = os.path.basename(r.get("image_path", ""))
            score = r.get("score", 0)
            label = r.get("label", "")
            ts = r.get("timestamp", "")
            line = f"  {i}. {path} — score: {score:.4f} | {label}"
            if ts:
                line += f" | t={ts}"
            click.echo(line)


@cli.command()
@click.option("--query", "-q", "query_text", required=True, help="Query text")
@click.option("--save", "-s", default="./output", help="Save annotated images to folder")
@click.option("--top-k", "-k", default=5, help="Number of results")
def show(query_text, save, top_k):
    """Visualize query results with bounding boxes."""
    from sightrag import SightRAG
    
    rag = SightRAG()
    results = rag.query(text=query_text, top_k=top_k)
    rag.show(results, save=save)


@cli.command()
def status():
    """Show index statistics."""
    from sightrag import SightRAG
    
    rag = SightRAG()
    click.echo(f"SightRAG Status:")
    click.echo(f"  Backend:  {rag._backend.name}")
    click.echo(f"  Regions:  {rag.count()}")
    click.echo(f"  Store:    {rag._store_type}")


@cli.command()
@click.confirmation_option(prompt="Clear all indexed data?")
def clear():
    """Clear all indexed data."""
    from sightrag import SightRAG
    
    rag = SightRAG()
    rag.clear()


@cli.command()
@click.option("--host", default="0.0.0.0", help="Server host")
@click.option("--port", default=8000, help="Server port")
def serve(host, port):
    """Start REST API server."""
    from sightrag import serve as start_server
    start_server(host=host, port=port)


def main():
    cli()


if __name__ == "__main__":
    main()
