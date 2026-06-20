"""Command-line interface: ingest, search, stats, serve."""

from __future__ import annotations

from pathlib import Path

import typer

from deepsearch.logging_utils import get_logger

app = typer.Typer(add_completion=False, help="Deep Search in Media Library (Gemma 4 + Qdrant).")
log = get_logger(__name__)


@app.command()
def ingest(
    path: str = typer.Argument(..., help="Media file or directory to ingest."),
    category: str = typer.Option("uncategorized", help="Category tag for filtering."),
):
    """Ingest a file or a whole directory into the index."""
    from deepsearch.ingestion import IngestionPipeline

    pipe = IngestionPipeline()
    target = Path(path)
    results = (
        pipe.ingest_directory(target, category)
        if target.is_dir()
        else [pipe.ingest_file(target, category)]
    )
    total = 0
    for r in results:
        if r.error:
            typer.secho(f"  FAIL {r.asset_name}: {r.error}", fg="red")
        else:
            total += r.fragments
            typer.secho(f"  OK {r.asset_name}: {r.fragments} fragments {r.by_modality}", fg="green")
    typer.secho(f"Done. {total} fragments indexed.", fg="cyan", bold=True)


@app.command()
def search(
    query: str = typer.Argument(..., help="Natural-language query."),
    top_k: int = typer.Option(10, help="Number of results."),
):
    """Run a one-off search from the terminal."""
    from deepsearch.query import get_engine

    resp = get_engine().search(query)
    if resp.blocked:
        typer.secho(f"Blocked: {resp.message}", fg="red")
        raise typer.Exit(1)
    typer.secho(f"intent={resp.intent.intent} weights={resp.intent.weights}", fg="yellow")
    for i, h in enumerate(resp.hits[:top_k], 1):
        typer.echo(
            f"{i:>2}. [{h.score:.3f}] {h.asset_name} @ {h.timestamp_label} "
            f"({h.modality})\n    {h.text[:140]}"
        )


@app.command()
def stats():
    """Show collection statistics."""
    from deepsearch.vectorstore.store import get_store

    s = get_store().stats()
    typer.secho(
        f"assets={s['assets']} fragments={s['fragments']} by_modality={s['by_modality']}",
        fg="cyan",
    )


@app.command()
def reset():
    """Delete and recreate the collection (destroys the index)."""
    from deepsearch.vectorstore.store import get_store

    typer.confirm("This deletes the entire index. Continue?", abort=True)
    get_store().ensure_collection(recreate=True)
    typer.secho("Collection recreated.", fg="green")


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(8000),
    reload: bool = typer.Option(False, help="Auto-reload on code changes (dev)."),
):
    """Launch the FastAPI backend (the Next.js frontend in web/ calls it)."""
    import uvicorn

    uvicorn.run("deepsearch.api.main:app", host=host, port=port, reload=reload)


@app.command()
def health():
    """Print the live Gemma 4 / Ollama health report."""
    import json as _json

    from deepsearch.models import get_model_client

    report = get_model_client().health()
    color = "green" if report["ok"] else "red"
    typer.secho(_json.dumps(report, indent=2), fg=color)
    raise typer.Exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    app()
