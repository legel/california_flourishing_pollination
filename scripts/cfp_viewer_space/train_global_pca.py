"""Train a global PCA(1024->3) on sampled DINOv3 patch tokens.

Why this exists alongside the UMAP encoder:
  The UMAP joblib pickle deserialization is fragile across Python versions
  (Space runs 3.13, training is 3.11). Global PCA needs no class
  deserialization — just a (1024, 3) numpy matrix, a (1024,) mean, and a
  (3,) per-channel min/max. ~12 KB total. Always loads.

Output: lookups/global_pca.npz with arrays:
  components  : (1024, 3) — top-3 principal components
  mean        : (1024,)
  channel_min : (3,)      — for normalization to [0,1]
  channel_max : (3,)
"""
from __future__ import annotations
import time
from pathlib import Path
import numpy as np
import pandas as pd
from huggingface_hub import HfApi

LOCAL_SHARDS = Path("/home/legel/cfp_shards")
SAMPLE_OBS = 5_000
SAMPLE_PATCHES = 500_000  # affordable since PCA scales linearly
RANDOM_STATE = 0


def main() -> None:
    t0 = time.time()
    shards = sorted(LOCAL_SHARDS.glob("embeddings_*.parquet"))
    rng = np.random.default_rng(RANDOM_STATE)
    rng.shuffle(shards)
    print(f"local shards: {len(shards)}", flush=True)

    pool: list[np.ndarray] = []
    obs_count = 0
    for shard in shards:
        if obs_count >= SAMPLE_OBS:
            break
        df = pd.read_parquet(shard, columns=["patches_fp16", "patches_shape"])
        take = min(len(df), SAMPLE_OBS - obs_count)
        idx = rng.choice(len(df), size=take, replace=False)
        for i in idx:
            shape = tuple(df.iloc[i]["patches_shape"])
            arr = np.frombuffer(df.iloc[i]["patches_fp16"], dtype=np.float16
                                ).reshape(shape).astype(np.float32)
            pool.append(arr.reshape(-1, arr.shape[-1]))
        obs_count += take

    all_patches = np.concatenate(pool, axis=0)
    print(f"pooled {len(all_patches):,} patches ({time.time()-t0:.0f}s)", flush=True)

    if len(all_patches) > SAMPLE_PATCHES:
        sub = rng.choice(len(all_patches), size=SAMPLE_PATCHES, replace=False)
        fit = all_patches[sub]
    else:
        fit = all_patches
    print(f"fitting PCA on {fit.shape}…", flush=True)

    mean = fit.mean(axis=0)
    centered = fit - mean
    # Use randomized SVD via sklearn for speed
    from sklearn.decomposition import PCA
    pca = PCA(n_components=3, svd_solver="randomized", random_state=RANDOM_STATE)
    proj = pca.fit_transform(centered)
    components = pca.components_.T.astype(np.float32)   # (1024, 3)
    print(f"  explained variance ratio: {pca.explained_variance_ratio_}",
          flush=True)
    print(f"  components shape: {components.shape}", flush=True)

    cmin = proj.min(axis=0).astype(np.float32)
    cmax = proj.max(axis=0).astype(np.float32)
    print(f"  channel min: {cmin}", flush=True)
    print(f"  channel max: {cmax}", flush=True)

    out_path = Path("global_pca.npz")
    np.savez(out_path, components=components, mean=mean.astype(np.float32),
              channel_min=cmin, channel_max=cmax,
              explained_variance_ratio=pca.explained_variance_ratio_.astype(np.float32),
              trained_on_patches=np.int32(len(fit)),
              trained_on_obs=np.int32(obs_count))
    print(f"wrote {out_path} ({out_path.stat().st_size/1e3:.1f} KB)", flush=True)

    HfApi().upload_file(
        path_or_fileobj=str(out_path),
        path_in_repo="lookups/global_pca.npz",
        repo_id="deepearth/california-flourishing-pollination",
        repo_type="dataset",
        commit_message=(f"global_pca.npz: top-3 PCA components fitted on "
                        f"{len(fit):,} DINOv3 patch tokens ({obs_count} obs); "
                        f"cross-image-consistent 3D RGB projection (12 KB, "
                        f"no class deserialization needed)"),
    )
    print(f"uploaded · total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
