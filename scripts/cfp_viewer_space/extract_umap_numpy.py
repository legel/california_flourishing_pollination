"""Extract the trained UMAP encoder to a Python-version-independent numpy
file (no joblib class graph), upload to HF as `lookups/umap_numpy.npz`.

The Space then loads + uses sklearn.NearestNeighbors to approximate
UMAP's transform — cross-image semantic projection that actually works
under Python 3.13.

Approximate-UMAP transform formula:
  for query x in original 1024-D space:
    1. find K nearest training points (cosine distance)
    2. distance-weighted average of their stored 3-D embeddings
This is not identical to UMAP's iterative optimization-based transform
but gives the same cross-image color consistency for visualization.
"""
from __future__ import annotations
import time
import numpy as np
import joblib
from huggingface_hub import HfApi, hf_hub_download

REPO = "deepearth/california-flourishing-pollination"
SRC = "lookups/umap_encoder.joblib"
OUT = "umap_numpy.npz"


def main() -> None:
    t0 = time.time()
    p = hf_hub_download(REPO, SRC, repo_type="dataset")
    pack = joblib.load(p)
    enc = pack["encoder"]
    print(f"loaded encoder in {time.time()-t0:.0f}s", flush=True)
    print(f"  type: {type(enc).__name__}", flush=True)
    print(f"  attrs: {[a for a in dir(enc) if not a.startswith('__') and not callable(getattr(enc, a, None))][:15]}", flush=True)

    # Training data + embeddings — UMAP stores them as _raw_data and embedding_
    training_data = enc._raw_data
    training_embeddings = enc.embedding_
    n_neighbors = int(enc.n_neighbors)
    metric = str(enc.metric)
    print(f"  training_data: {training_data.shape} {training_data.dtype}", flush=True)
    print(f"  training_embeddings: {training_embeddings.shape} {training_embeddings.dtype}", flush=True)
    print(f"  n_neighbors: {n_neighbors}", flush=True)
    print(f"  metric: {metric}", flush=True)

    # Save fp16 to halve size
    np.savez(OUT,
              training_data=training_data.astype(np.float16),
              training_embeddings=training_embeddings.astype(np.float32),
              n_neighbors=np.int32(n_neighbors),
              metric=np.array(metric),
              channel_min=np.asarray(pack["channel_min"], dtype=np.float32),
              channel_max=np.asarray(pack["channel_max"], dtype=np.float32))

    sz_mb = __import__("os").path.getsize(OUT) / 1e6
    print(f"\nwrote {OUT}  {sz_mb:.1f} MB (was 1461 MB joblib)", flush=True)

    HfApi().upload_file(
        path_or_fileobj=OUT, path_in_repo="lookups/umap_numpy.npz",
        repo_id=REPO, repo_type="dataset",
        commit_message=(f"umap_numpy.npz: extracted UMAP training data + embeddings as "
                        f"fp16 numpy arrays (no class graph; loads under any Python "
                        f"version); approximate transform via sklearn.NearestNeighbors"),
    )
    print("uploaded · total", time.time()-t0, "s")


if __name__ == "__main__":
    main()
