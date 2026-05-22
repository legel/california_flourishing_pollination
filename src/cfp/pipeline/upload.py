"""HF dataset upload stage.

Watches the shard directory, pushes each completed embedding shard to the
Hugging Face dataset repo, and deletes the local shard on success.

We use ``huggingface_hub.HfApi.upload_file`` rather than the shell `hf upload`
so retries and per-file commit messages can be controlled programmatically.

Resumable: shards already present in the remote repo are detected via
``list_repo_files`` and skipped.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import typer
from huggingface_hub import HfApi, CommitOperationAdd
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

app = typer.Typer(add_completion=False, help="HF dataset shard uploader.")
console = Console()


@app.command()
def upload(
    shard_dir: Path = typer.Option(Path("/home/legel/cfp_shards"), "--shard-dir"),
    repo: str = typer.Option("deepearth/california-flourishing-pollination", "--repo"),
    repo_type: str = typer.Option("dataset", "--repo-type"),
    remote_prefix: str = typer.Option("embeddings/", "--remote-prefix"),
    delete_after: bool = typer.Option(True, "--delete-after/--keep-local"),
    poll_seconds: float = typer.Option(0.0, "--poll-seconds",
                                       help="If >0, poll forever (until interrupted) instead of running once."),
    prov_dir: Path = typer.Option(Path("provenance"), "--prov-dir"),
) -> None:
    """Upload embedding shards to ``repo`` (dataset); delete on success."""
    api = HfApi()
    me = api.whoami()
    snapshot = datetime.now(timezone.utc).isoformat()
    prov_dir.mkdir(parents=True, exist_ok=True)
    prov_path = prov_dir / f"upload_{snapshot.replace(':','').replace('-','')[:15]}.jsonl"
    prov = prov_path.open("w")
    prov.write(json.dumps({
        "type": "run_meta", "stage": "pipeline.upload",
        "started_utc": snapshot, "repo": repo, "shard_dir": str(shard_dir),
        "user": me.get("name"), "remote_prefix": remote_prefix,
    }) + "\n")
    prov.flush()
    console.print(f"HF user: [bold]{me.get('name')}[/]  → repo {repo}")

    def _existing_remote() -> set[str]:
        try:
            files = api.list_repo_files(repo_id=repo, repo_type=repo_type)
            return {f for f in files if f.startswith(remote_prefix)}
        except Exception as e:
            console.print(f"[yellow]could not list repo files: {e}[/]")
            return set()

    def _upload_round() -> int:
        existing = _existing_remote()
        candidates = sorted(shard_dir.glob("embeddings_*.parquet"))
        n_pushed = 0
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as prog:
            tid = prog.add_task("uploading", total=len(candidates))
            for shard in candidates:
                remote_path = f"{remote_prefix}{shard.name}"
                if remote_path in existing:
                    prog.update(tid, advance=1)
                    continue
                try:
                    api.upload_file(
                        path_or_fileobj=str(shard),
                        path_in_repo=remote_path,
                        repo_id=repo,
                        repo_type=repo_type,
                        commit_message=f"embed: add {shard.name}",
                    )
                    prov.write(json.dumps({
                        "type": "upload",
                        "shard": shard.name,
                        "remote_path": remote_path,
                        "size_bytes": shard.stat().st_size,
                        "uploaded_utc": datetime.now(timezone.utc).isoformat(),
                    }) + "\n")
                    if delete_after:
                        shard.unlink(missing_ok=True)
                    n_pushed += 1
                except Exception as e:
                    prov.write(json.dumps({
                        "type": "upload_failed", "shard": shard.name, "error": str(e),
                    }) + "\n")
                    console.print(f"[red]upload failed[/] {shard.name}: {e}")
                prov.flush()
                prog.update(tid, advance=1)
        return n_pushed

    if poll_seconds > 0:
        console.print(f"[cyan]polling every {poll_seconds:g}s — Ctrl+C to stop[/]")
        try:
            while True:
                n = _upload_round()
                console.print(f"round complete: {n} shards pushed; sleeping {poll_seconds:g}s")
                time.sleep(poll_seconds)
        except KeyboardInterrupt:
            console.print("[yellow]interrupted; flushing provenance[/]")
    else:
        n = _upload_round()
        console.print(f"[bold green]done[/]: {n} shards pushed")

    prov.write(json.dumps({
        "type": "result", "finished_utc": datetime.now(timezone.utc).isoformat(),
    }) + "\n")
    prov.close()
