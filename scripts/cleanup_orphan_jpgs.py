"""Walk /home/legel/cfp_images and delete jpgs whose URL is already in
the embed checkpoint (i.e. orphans the embedder failed to unlink under
the pre-URL-keyed era). Safe to run concurrently with the live embedder:
we never delete a pending file, and the embedder skips checkpointed URLs.

Reads the embed checkpoint once into memory, walks the disk, deletes
matched orphans + their .json sidecars. Reports counts at the end.
"""
from __future__ import annotations
import json
import time
from pathlib import Path
import pandas as pd

IMG = Path("/home/legel/cfp_images")
CKPT = Path("/home/legel/california_flourishing_pollination/outputs/checkpoint_embedded.parquet")


def main() -> None:
    t0 = time.time()
    done = set(pd.read_parquet(CKPT)["image_url_large"].dropna().astype(str).tolist())
    print(f"[{time.time()-t0:.1f}s] embed checkpoint: {len(done):,} URLs", flush=True)

    n_total = n_deleted = n_pending = n_no_meta = n_bad_meta = 0
    bytes_freed = 0
    t_last_report = time.time()

    for bucket in sorted(IMG.iterdir()):
        if not bucket.is_dir():
            continue
        for img in bucket.iterdir():
            if img.suffix == ".json" or img.suffix.endswith(".tmp"):
                continue
            n_total += 1
            meta = img.with_suffix(".json")
            url = None
            if meta.exists():
                try:
                    d = json.loads(meta.read_text())
                    url = d.get("image_url_large") or d.get("url")
                except Exception:
                    n_bad_meta += 1
            else:
                n_no_meta += 1

            if url and url in done:
                try:
                    sz = img.stat().st_size
                    img.unlink(missing_ok=True)
                    meta.unlink(missing_ok=True)
                    bytes_freed += sz
                    n_deleted += 1
                except OSError:
                    pass
            else:
                n_pending += 1

            if time.time() - t_last_report > 30:
                print(f"  [{time.time()-t0:.0f}s] scanned={n_total:,}  deleted={n_deleted:,}  "
                      f"pending={n_pending:,}  freed={bytes_freed/1e9:.1f}GB", flush=True)
                t_last_report = time.time()

    print(f"\nDONE in {time.time()-t0:.0f}s")
    print(f"  total scanned: {n_total:,}")
    print(f"  deleted orphans: {n_deleted:,}  ({bytes_freed/1e9:.1f} GB freed)")
    print(f"  pending (untouched): {n_pending:,}")
    print(f"  no sidecar: {n_no_meta:,}")
    print(f"  bad sidecar: {n_bad_meta:,}")


if __name__ == "__main__":
    main()
