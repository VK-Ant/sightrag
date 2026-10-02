"""
SightRAG CLI — terminal commands for visual RAG.

Usage:
    sightrag index ./photos/
    sightrag index ./video.mp4 --fps 2
    sightrag index ./video.mp4 --fps 5 --track
    sightrag query "find person near door"
    sightrag query --reference ./suspect.jpg
    sightrag track ./video.mp4 --query "person in red" --fps 5
    sightrag timeline 3
    sightrag tracks
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
@click.version_option(version="0.5.0", prog_name="sightrag")
def cli():
    """SightRAG — See. Search. Retrieve. Track."""
    pass


@cli.command()
@click.argument("path")
@click.option("--fps", default=1, help="Frames per second for video")
@click.option("--store", default="sqlite", help="Vector store backend")
@click.option("--detector", default=None, help="Custom detector")
@click.option("--domain", default=None, help="Domain hint for better accuracy")
@click.option("--track", is_flag=True, help="Enable object tracking")
@click.option("--tracker", default=None, help="Custom tracker")
@click.option("--segment", is_flag=True, help="Enable segmentation")
@click.option("--segmentor", default=None, help="Custom segmentor")
def index(path, fps, store, detector, domain, track, tracker, segment, segmentor):
    """Index images, video, or folder."""
    from sightrag import SightRAG

    kwargs = {"store": store}
    if domain:
        kwargs["domain_hint"] = domain
    if detector:
        kwargs["detector"] = detector
    if track:
        kwargs["track"] = True
    if tracker:
        kwargs["tracker"] = tracker
    if segment:
        kwargs["segment"] = True
    if segmentor:
        kwargs["segmentor"] = segmentor

    rag = SightRAG(**kwargs)
    rag.index(path, fps=fps)


@cli.command()
@click.argument("text", required=False)
@click.option("--reference", "-r", default=None, help="Reference image path")
@click.option("--top-k", "-k", default=5, help="Number of results")
@click.option("--format", "fmt", default="text", help="Output: text, json")
@click.option("--segment", is_flag=True, help="Include segmentation masks")
def query(text, reference, top_k, fmt, segment):
    """Search indexed content."""
    from sightrag import SightRAG

    kwargs = {}
    if segment:
        kwargs["segment"] = True

    rag = SightRAG(**kwargs)

    if reference:
        results = rag.query(reference=reference, top_k=top_k)
    elif text:
        results = rag.query(text=text, top_k=top_k)
    else:
        click.echo("Provide query text or --reference image")
        return

    if fmt == "json":
        # Remove non-serializable fields
        clean = []
        for r in results:
            cr = {k: v for k, v in r.items()
                  if k not in ("mask", "embedding")}
            clean.append(cr)
        click.echo(json.dumps(clean, indent=2))
    else:
        for i, r in enumerate(results, 1):
            path = os.path.basename(r.get("image_path", ""))
            score = r.get("score", 0)
            label = r.get("label", "")
            ts = r.get("timestamp", "")
            tid = r.get("track_id")
            line = f"  {i}. {path} — score: {score:.4f} | {label}"
            if tid is not None:
                line += f" | Track#{tid}"
            if ts:
                line += f" | t={ts}"
            click.echo(line)


@cli.command("find-track")
@click.argument("query_text")
@click.option("--video", "-v", default=None, help="Video path (indexes if given)")
@click.option("--fps", default=5, help="Frames per second for video indexing")
@click.option("--top-k", "-k", default=1, help="Number of tracks to find")
@click.option("--save", "-s", default=None, help="Save timeline visualization")
def find_track(query_text, video, fps, top_k, save):
    """Find object and track through video timeline."""
    from sightrag import SightRAG

    rag = SightRAG(track=True)

    if video:
        rag.index(video, fps=fps)

    timelines = rag.find_and_track(query_text, top_k=top_k)

    if not timelines:
        click.echo("No tracked objects found.")
        return

    for tl in timelines:
        click.echo(f"\nTrack #{tl['track_id']}: {tl['label']}")
        click.echo(f"  First seen: {tl['first_timestamp']}s (frame {tl['first_frame']})")
        click.echo(f"  Last seen:  {tl['last_timestamp']}s (frame {tl['last_frame']})")
        click.echo(f"  Duration:   {tl['frame_count']} frames")

        if save:
            from .visualizer import show_timeline
            out = os.path.join(save, f"track_{tl['track_id']}.png")
            show_timeline(tl, save=out)


@cli.command()
@click.argument("track_id", type=int)
@click.option("--format", "fmt", default="text", help="Output: text, json")
@click.option("--save", "-s", default=None, help="Save timeline visualization")
def timeline(track_id, fmt, save):
    """Show timeline for a tracked object."""
    from sightrag import SightRAG

    rag = SightRAG(track=True)

    try:
        tl = rag.timeline(track_id)
    except ValueError as e:
        click.echo(str(e))
        return

    if fmt == "json":
        click.echo(json.dumps(tl, indent=2))
    else:
        click.echo(f"Track #{tl['track_id']}: {tl['label']}")
        click.echo(f"  Video:      {tl['video_path']}")
        click.echo(f"  First seen: {tl['first_timestamp']}s (frame {tl['first_frame']})")
        click.echo(f"  Last seen:  {tl['last_timestamp']}s (frame {tl['last_frame']})")
        click.echo(f"  Duration:   {tl['frame_count']} frames")

        states = tl.get("states", [])
        if states:
            click.echo(f"  States:     {len(states)} frame detections")

    if save:
        from .visualizer import show_timeline
        show_timeline(tl, save=save)


@cli.command()
@click.option("--format", "fmt", default="text", help="Output: text, json")
def tracks(fmt):
    """List all tracked objects."""
    from sightrag import SightRAG

    rag = SightRAG(track=True)
    all_tracks = rag.get_all_tracks()

    if not all_tracks:
        click.echo("No tracks found. Index a video with --track first.")
        return

    if fmt == "json":
        click.echo(json.dumps(all_tracks, indent=2))
    else:
        click.echo(f"Tracked objects: {len(all_tracks)}")
        click.echo()
        for t in all_tracks:
            click.echo(
                f"  Track #{t['track_id']:3d} | {t['label']:12s} | "
                f"{t['first_timestamp']}s → {t['last_timestamp']}s | "
                f"{t['frame_count']} frames"
            )


@cli.command()
@click.option("--query", "-q", "query_text", required=True, help="Query text")
@click.option("--save", "-s", default="./output", help="Save annotated images to folder")
@click.option("--top-k", "-k", default=5, help="Number of results")
@click.option("--segment", is_flag=True, help="Include segmentation masks")
def show(query_text, save, top_k, segment):
    """Visualize query results with bounding boxes."""
    from sightrag import SightRAG

    kwargs = {}
    if segment:
        kwargs["segment"] = True

    rag = SightRAG(**kwargs)
    results = rag.query(text=query_text, top_k=top_k, segment=segment)
    rag.show(results, save=save)


@cli.command()
def status():
    """Show index statistics."""
    from sightrag import SightRAG

    rag = SightRAG()
    click.echo(f"SightRAG Status:")
    click.echo(f"  Regions:  {rag.count()}")

    # Track count
    try:
        tc = rag._store.track_count()
        if tc > 0:
            click.echo(f"  Tracks:   {tc}")
    except:
        pass


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
