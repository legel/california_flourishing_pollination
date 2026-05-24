"""Generate sample PNG overlays for visual QA of the embeddings + UMAP encoder.

For each of N sampled rows:
  - download the iNat photo
  - decode the DINOv3 patches from the shard
  - project to 3D RGB via the pretrained UMAP encoder (numpy npz)
  - render: side-by-side [photo | overlay | 50% blend]

Saved as /home/legel/cfp_sample_overlays/sample_*.png for the user to scp / view.
"""
from __future__ import annotations
import io
import time
from pathlib import Path
import numpy as np
import pandas as pd
import requests
import joblib
from PIL import Image

OUT_DIR = Path("/home/legel/cfp_sample_overlays")
OUT_DIR.mkdir(exist_ok=True)


def umap_setup():
    """Use the local joblib encoder (Python 3.11 here works fine)."""
    pack = joblib.load(Path.home() / ".cache/huggingface/hub" / "_umap_loc")
    return pack


def main() -> None:
    # Force numpy-knn path (matches what the Space uses) — joblib transform
    # is slower because it runs 100 epochs of optimization per query.
    p = None
    if p is None:
        # Build a tiny knn approximator from the numpy file (matches Space behavior)
        z = np.load("scripts/cfp_viewer_space/umap_numpy.npz") if Path("scripts/cfp_viewer_space/umap_numpy.npz").exists() else None
        if z is None:
            # Download fresh from HF
            from huggingface_hub import hf_hub_download
            local = hf_hub_download("deepearth/california-flourishing-pollination",
                                     "lookups/umap_numpy.npz", repo_type="dataset")
            z = np.load(local)
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=int(z["n_neighbors"]), metric=str(z["metric"]))
        nn.fit(z["training_data"].astype(np.float32))
        pack = {
            "nn": nn,
            "training_embeddings": z["training_embeddings"].astype(np.float32),
            "channel_min": z["channel_min"].astype(np.float32),
            "channel_max": z["channel_max"].astype(np.float32),
            "kind": "numpy_knn",
        }
        print(f"using numpy-extracted UMAP (sklearn NN), train={z['training_data'].shape}", flush=True)
    else:
        pack = {**p, "kind": "joblib"}
        print(f"using joblib UMAP encoder", flush=True)

    def umap_rgb(patches: np.ndarray) -> np.ndarray:
        h, w, d = patches.shape
        flat = patches.reshape(-1, d).astype(np.float32)
        if pack["kind"] == "joblib":
            proj = pack["encoder"].transform(flat)
        else:
            dists, idxs = pack["nn"].kneighbors(flat)
            weights = 1.0 / (dists + 1e-6)
            weights = weights / weights.sum(axis=1, keepdims=True)
            sel = pack["training_embeddings"][idxs]
            proj = (weights[..., None] * sel).sum(axis=1)
        cmin = np.asarray(pack["channel_min"], dtype=np.float32)
        cmax = np.asarray(pack["channel_max"], dtype=np.float32)
        proj = np.clip((proj - cmin) / (cmax - cmin + 1e-8), 0, 1)
        return (proj.reshape(h, w, 3) * 255).astype(np.uint8)

    # Pick a new-era shard (run_id 20260524T131149) for diverse samples
    shards = sorted(Path("/home/legel/cfp_shards").glob("embeddings_20260524T131149*.parquet"))
    if not shards:
        print("no new shards found, falling back to old shards"); shards = sorted(Path("/home/legel/cfp_shards").glob("embeddings_*.parquet"))
    print(f"using shard: {shards[0].name}", flush=True)
    df = pd.read_parquet(shards[0], columns=[
        "image_url_large","taxon_name","dataset_role","patches_fp16","patches_shape",
        "phenovision_flowering_prob","phenovision_fruiting_prob",
    ])
    print(f"shard rows: {len(df):,}", flush=True)

    # Sample diverse: 3 plants + 3 pollinators
    plants = df[df["dataset_role"] == "plant"].sample(3, random_state=42)
    polls = df[df["dataset_role"] == "pollinator"].sample(3, random_state=42)
    picks = pd.concat([plants, polls]).reset_index(drop=True)
    print(f"picked {len(picks)} samples", flush=True)

    for i, row in picks.iterrows():
        t0 = time.time()
        url = row["image_url_large"]
        try:
            r = requests.get(url, timeout=15)
            r.raise_for_status()
            photo = Image.open(io.BytesIO(r.content)).convert("RGB")
        except Exception as e:
            print(f"  {i}: fetch failed: {e}"); continue

        patches = np.frombuffer(row["patches_fp16"], dtype=np.float16).reshape(
            tuple(row["patches_shape"])).astype(np.float32)
        rgb = umap_rgb(patches)
        SZ = 672
        photo_s = photo.resize((SZ, SZ), Image.BILINEAR)
        overlay = Image.fromarray(rgb).resize((SZ, SZ), Image.BILINEAR)
        blend = Image.blend(photo_s, overlay, 0.5)

        # Compose 3-panel
        canvas = Image.new("RGB", (SZ * 3 + 30, SZ + 50), (16, 16, 16))
        canvas.paste(photo_s, (0, 50))
        canvas.paste(overlay, (SZ + 15, 50))
        canvas.paste(blend, (SZ * 2 + 30, 50))

        from PIL import ImageDraw, ImageFont
        d = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
        except Exception:
            font = ImageFont.load_default()
        d.text((10, 14), f"{row['taxon_name']} · {row['dataset_role']} · "
                f"flowering {row['phenovision_flowering_prob']:.2f} fruiting {row['phenovision_fruiting_prob']:.2f}",
                fill=(220, 220, 220), font=font)
        d.text((10, 50 + SZ - 26), "iNat photo (672²)", fill=(220, 220, 220), font=font)
        d.text((SZ + 25, 50 + SZ - 26), "DINOv3 patches → UMAP RGB", fill=(220, 220, 220), font=font)
        d.text((SZ * 2 + 40, 50 + SZ - 26), "50% blend", fill=(220, 220, 220), font=font)

        out = OUT_DIR / f"sample_{i:02d}_{row['dataset_role']}_{row['taxon_name'].replace(' ','_').replace('/','_')[:40]}.png"
        canvas.save(out, optimize=True)
        print(f"  {i:02d} [{time.time()-t0:.1f}s] -> {out.name}", flush=True)


if __name__ == "__main__":
    main()
