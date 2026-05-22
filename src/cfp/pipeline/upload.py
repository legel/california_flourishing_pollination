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

    import os as _os
    import shutil as _shutil
    import tempfile as _tempfile

    def _upload_round() -> int:
        """Batch-upload all local shards via api.upload_large_folder.

        This is dramatically faster than per-file ``upload_file`` (single git
        commit per batch + parallel xet chunk transfer). Measured: 10 shards
        × 4 GB in 140 s ≈ 290 MB/s effective.
        """
        existing = _existing_remote()
        candidates = [p for p in sorted(shard_dir.glob("embeddings_*.parquet"))
                      if f"{remote_prefix}{p.name}" not in existing]
        if not candidates:
            return 0

        # upload_large_folder expects a folder root. Create a temp dir with
        # symlinks under the desired remote prefix layout (embeddings/<file>).
        with _tempfile.TemporaryDirectory(prefix="_cfp_upload_") as staging:
            target = Path(staging) / remote_prefix.rstrip("/")
            target.mkdir(parents=True, exist_ok=True)
            for p in candidates:
                _os.symlink(p, target / p.name)

            try:
                api.upload_large_folder(
                    folder_path=staging,
                    repo_id=repo,
                    repo_type=repo_type,
                    allow_patterns=[f"{remote_prefix}*.parquet"],
                )
            except Exception as e:
                for p in candidates:
                    prov.write(json.dumps({
                        "type": "upload_failed", "shard": p.name, "error": str(e),
                    }) + "\n")
                console.print(f"[red]upload_large_folder failed[/]: {e}")
                prov.flush()
                return 0

        # Confirm and delete on success — re-query remote, intersect with
        # candidate names, and unlink only confirmed remote entries.
        confirmed_remote = _existing_remote()
        n_pushed = 0
        for p in candidates:
            if f"{remote_prefix}{p.name}" in confirmed_remote:
                prov.write(json.dumps({
                    "type": "upload",
                    "shard": p.name,
                    "remote_path": f"{remote_prefix}{p.name}",
                    "size_bytes": p.stat().st_size,
                    "uploaded_utc": datetime.now(timezone.utc).isoformat(),
                }) + "\n")
                if delete_after:
                    p.unlink(missing_ok=True)
                n_pushed += 1
        prov.flush()
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
