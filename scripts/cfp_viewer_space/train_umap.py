"""Train a UMAP 1024→3 projection on sampled DINOv3 patch tokens for cross-
species/scene-semantic overlay coloring in the viewer Space.

We sample patch tokens (not CLS tokens) because we want patch-level
visualization. The trained encoder is uploaded to HF at
`lookups/umap_encoder.joblib` and lazy-loaded by the Space app.

Sample plan:
  5,000 random observations × 196 patches each, then sub-sample to 100,000
  vectors → fit UMAP(n_components=3). Once fit, transforming a single
  observation's 196 patches takes milliseconds (vs PCA which does a
  per-observation SVD).

Storage: trained encoder is a few MB joblib pickle.
"""
from __future__ import annotations
import time
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from huggingface_hub import HfApi
import umap

LOCAL_SHARDS = Path("/home/legel/cfp_shards")
SAMPLE_OBS = 5_000           # observations to sample
SAMPLE_PATCHES = 100_000     # final patch tokens for UMAP fit
N_COMPONENTS = 3
RANDOM_STATE = 0


def main() -> None:
    t0 = time.time()
    shards = sorted(LOCAL_SHARDS.glob("embeddings_*.parquet"))
    rng = np.random.default_rng(RANDOM_STATE)
    rng.shuffle(shards)
    print(f"local shards available: {len(shards)}", flush=True)

    patches_pool: list[np.ndarray] = []
    obs_count = 0
    for shard in shards:
        if obs_count >= SAMPLE_OBS:
            break
        df = pd.read_parquet(shard, columns=["patches_fp16", "patches_shape"])
        # Take a random subset of rows from this shard
        take = min(len(df), SAMPLE_OBS - obs_count)
        idx = rng.choice(len(df), size=take, replace=False)
        for i in idx:
            shape = tuple(df.iloc[i]["patches_shape"])
            buf = df.iloc[i]["patches_fp16"]
            arr = np.frombuffer(buf, dtype=np.float16).reshape(shape).astype(np.float32)
            patches_pool.append(arr.reshape(-1, arr.shape[-1]))
        obs_count += take
        print(f"  pooled {obs_count}/{SAMPLE_OBS} obs from {shard.name}", flush=True)

    all_patches = np.concatenate(patches_pool, axis=0)
    print(f"\ntotal patch tokens pooled: {len(all_patches):,}  "
          f"({time.time()-t0:.0f}s)", flush=True)

    # Sub-sample to final fit size
    if len(all_patches) > SAMPLE_PATCHES:
        sub_idx = rng.choice(len(all_patches), size=SAMPLE_PATCHES, replace=False)
        fit_data = all_patches[sub_idx]
    else:
        fit_data = all_patches
    print(f"fitting UMAP on {len(fit_data):,} × {fit_data.shape[1]}…", flush=True)

    encoder = umap.UMAP(
        n_components=N_COMPONENTS,
        n_neighbors=30,
        min_dist=0.1,
        metric="cosine",
        random_state=RANDOM_STATE,
        verbose=True,
        n_jobs=-1,
    )
    t_fit = time.time()
    embedded = encoder.fit_transform(fit_data)
    print(f"fit done in {time.time()-t_fit:.0f}s; embedded shape {embedded.shape}", flush=True)

    # Compute per-channel min/max so the Space can do consistent normalization
    chan_min = embedded.min(0).astype(np.float32).tolist()
    chan_max = embedded.max(0).astype(np.float32).tolist()
    print(f"channel ranges: min={chan_min}  max={chan_max}", flush=True)

    out = {
        "encoder": encoder,
        "channel_min": chan_min,
        "channel_max": chan_max,
        "n_components": N_COMPONENTS,
        "trained_on_patches": int(len(fit_data)),
        "trained_on_obs": int(obs_count),
    }
    joblib.dump(out, "umap_encoder.joblib")
    sz = Path("umap_encoder.joblib").stat().st_size / 1e6
    print(f"wrote umap_encoder.joblib ({sz:.1f} MB)", flush=True)

    HfApi().upload_file(
        path_or_fileobj="umap_encoder.joblib",
        path_in_repo="lookups/umap_encoder.joblib",
        repo_id="deepearth/california-flourishing-pollination",
        repo_type="dataset",
        commit_message=(f"umap_encoder.joblib: UMAP({N_COMPONENTS}) fit on "
                        f"{len(fit_data):,} DINOv3 patch tokens from "
                        f"{obs_count} random observations (cosine, "
                        f"n_neighbors=30, min_dist=0.1) for cross-species "
                        f"semantic overlay coloring"),
    )
    print("uploaded to HF", flush=True)
    print(f"\ntotal: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
