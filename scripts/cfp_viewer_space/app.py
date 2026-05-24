"""California Flourishing & Pollination — interactive viewer.

A Gradio Space that browses the deepearth/california-flourishing-pollination
HF dataset record-by-record:
  - photo from iNaturalist (browser-direct URL load — fast)
  - PhenoVision flowering / fruiting probabilities
  - DINOv3 14×14×1024 patch tokens → 3D RGB overlay
      * PCA: per-observation SVD (slower per request, always works)
      * UMAP: pretrained 1024→3 encoder (cross-species semantic, very fast)
  - opacity slider + overlay resolution slider
  - per-photo CC license + creator attribution + metadata

Storage: ships only the ~225 MB master manifest + 110 MB shard index +
~few-MB UMAP encoder. Embedding shards (3 GB each, 1,072 total) are
pulled on demand from the parent dataset via hf_hub_download and cached
on the Space's persistent disk.
"""
from __future__ import annotations
import io
import time
from typing import Optional

import numpy as np
import pandas as pd
import gradio as gr
import requests
from PIL import Image
from huggingface_hub import hf_hub_download

REPO = "deepearth/california-flourishing-pollination"
MANIFEST_FILE = "manifests/image_manifest.parquet"
SHARD_INDEX_FILE = "lookups/shard_index.parquet"
UMAP_FILE = "lookups/umap_encoder.joblib"
MAX_OBS = 200  # cap obs list per species


# ─── lazy loaders ──────────────────────────────────────────────────────────
_manifest: Optional[pd.DataFrame] = None
_shard_idx: Optional[pd.DataFrame] = None
_species: Optional[list[str]] = None
_umap_pack: Optional[dict] = None


def manifest() -> pd.DataFrame:
    global _manifest
    if _manifest is None:
        p = hf_hub_download(REPO, MANIFEST_FILE, repo_type="dataset")
        _manifest = pd.read_parquet(p, columns=[
            "gbif_occurrence_id", "image_url_large", "taxon_name",
            "taxon_name_verbatim", "gbif_taxon_key", "dataset_role", "family",
            "license", "rights_holder", "creator", "observed_on",
            "decimal_latitude", "decimal_longitude", "locality",
        ])
    return _manifest


def shard_idx() -> pd.DataFrame:
    global _shard_idx
    if _shard_idx is None:
        p = hf_hub_download(REPO, SHARD_INDEX_FILE, repo_type="dataset")
        _shard_idx = pd.read_parquet(p).set_index("image_url_large")
    return _shard_idx


def species_list() -> list[str]:
    global _species
    if _species is None:
        m = manifest()
        _species = sorted(m["taxon_name"].dropna().unique().tolist())
    return _species


def umap_pack() -> Optional[dict]:
    global _umap_pack
    if _umap_pack is None:
        try:
            import joblib
            p = hf_hub_download(REPO, UMAP_FILE, repo_type="dataset")
            _umap_pack = joblib.load(p)
        except Exception:
            _umap_pack = {}  # mark as tried, not available
    return _umap_pack if _umap_pack else None


# ─── shard row lookup ──────────────────────────────────────────────────────
_shard_cache: dict[str, pd.DataFrame] = {}


def fetch_shard_row(url: str) -> Optional[pd.Series]:
    """Lazy-load the relevant embedding shard and return the row matching URL."""
    idx = shard_idx()
    if url not in idx.index:
        return None
    shard_path = idx.loc[url, "shard_path"]
    if isinstance(shard_path, pd.Series):  # dup row in index
        shard_path = shard_path.iloc[0]
    if shard_path not in _shard_cache:
        local = hf_hub_download(REPO, shard_path, repo_type="dataset")
        _shard_cache[shard_path] = pd.read_parquet(local, columns=[
            "image_url_large", "patches_fp16", "patches_shape",
            "phenovision_flowering_prob", "phenovision_fruiting_prob",
        ])
    df = _shard_cache[shard_path]
    match = df[df["image_url_large"] == url]
    return match.iloc[0] if len(match) else None


# ─── projections ───────────────────────────────────────────────────────────
def pca_rgb(patches_hwd: np.ndarray) -> np.ndarray:
    """Per-observation PCA: 14×14×1024 → 14×14×3 normalized to [0,255]."""
    h, w, d = patches_hwd.shape
    flat = patches_hwd.reshape(-1, d).astype(np.float32)
    flat -= flat.mean(0)
    cov = flat.T @ flat / max(1, flat.shape[0] - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    top3 = eigvecs[:, -3:][:, ::-1]
    proj = flat @ top3
    rng = proj.max(0) - proj.min(0)
    proj = (proj - proj.min(0)) / (rng + 1e-8)
    return (proj.reshape(h, w, 3) * 255).astype(np.uint8)


def umap_rgb(patches_hwd: np.ndarray) -> Optional[np.ndarray]:
    """Pretrained UMAP: 14×14×1024 → 14×14×3 using GLOBAL channel range
    so colors are consistent across observations."""
    pack = umap_pack()
    if pack is None:
        return None
    h, w, d = patches_hwd.shape
    flat = patches_hwd.reshape(-1, d).astype(np.float32)
    proj = pack["encoder"].transform(flat)
    cmin = np.asarray(pack["channel_min"], dtype=np.float32)
    cmax = np.asarray(pack["channel_max"], dtype=np.float32)
    proj = np.clip((proj - cmin) / (cmax - cmin + 1e-8), 0, 1)
    return (proj.reshape(h, w, 3) * 255).astype(np.uint8)


def render_overlay(photo_bytes: bytes, patches_hwd: np.ndarray,
                    method: str, opacity: float, resolution: int) -> Image.Image:
    """Compose the photo with the patch-level projection overlay.

    `resolution` controls upscale: higher = sharper image but blockier overlay
    (since patches are 14×14)."""
    rgb = umap_rgb(patches_hwd) if method == "UMAP" else None
    if rgb is None:
        rgb = pca_rgb(patches_hwd)
    photo = Image.open(io.BytesIO(photo_bytes)).convert("RGB").resize(
        (resolution, resolution), Image.BILINEAR)
    overlay = Image.fromarray(rgb).resize((resolution, resolution), Image.BILINEAR)
    return Image.blend(photo, overlay, opacity)


# ─── core render ───────────────────────────────────────────────────────────
def render(url: str, method: str, opacity: float, resolution: int):
    """Return (photo_url, overlay_PIL, pheno_md, meta_md, status_md, url_state)."""
    if not url:
        return None, None, "", "", "", ""
    m = manifest()
    row = m[m["image_url_large"] == url]
    if row.empty:
        return None, None, "", "", "URL not found in manifest", ""
    row = row.iloc[0]

    meta = (f"### {row['taxon_name']}\n"
            f"*{row['taxon_name_verbatim']}*  \n"
            f"role: **{row['dataset_role']}** · family: {row['family']}  \n"
            f"observed: {row['observed_on']}  \n"
            f"locality: {row['locality']}  \n"
            f"📍 {row['decimal_latitude']:.4f}, {row['decimal_longitude']:.4f}  \n"
            f"📷 {row['creator']} — `{row['license']}`  \n"
            f"[iNaturalist photo source]({url})")

    srow = fetch_shard_row(url)
    if srow is None:
        return url, None, "_(no embedding shard found for this URL)_", meta, "", url

    pheno = (f"### PhenoVision  \n"
             f"🌸 flowering: **{srow['phenovision_flowering_prob']:.3f}**  \n"
             f"🍒 fruiting: **{srow['phenovision_fruiting_prob']:.3f}**")

    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        photo_bytes = r.content
    except Exception as e:
        return url, None, pheno, meta, f"iNat photo fetch failed: {e}", url

    patches = np.frombuffer(srow["patches_fp16"], dtype=np.float16).reshape(
        tuple(srow["patches_shape"])).astype(np.float32)
    overlay = render_overlay(photo_bytes, patches, method, opacity, resolution)
    return url, overlay, pheno, meta, f"_overlay: {method} @ {resolution}px, {int(opacity*100)}% opacity_", url


# ─── observation list builders ─────────────────────────────────────────────
def list_obs_for_species(name: str) -> pd.DataFrame:
    if not name:
        return pd.DataFrame(columns=["observed_on", "locality", "image_url_large"])
    m = manifest()
    sub = m[m["taxon_name"] == name].head(MAX_OBS)
    return sub[["observed_on", "locality", "image_url_large"]].reset_index(drop=True)


def on_species_change(name: str):
    df = list_obs_for_species(name)
    m = manifest()
    total = (m["taxon_name"] == name).sum() if name else 0
    return df, f"showing {len(df)} of {total:,} observations for **{name}**"


def random_any(method: str, opacity: float, resolution: int):
    m = manifest()
    row = m.sample(1).iloc[0]
    species = row["taxon_name"]
    df = list_obs_for_species(species)
    photo, overlay, pheno, meta, status, url = render(row["image_url_large"], method, opacity, resolution)
    total = (m["taxon_name"] == species).sum()
    return (species, df, f"random pick: **{species}** — showing {len(df)} of {total:,} observations",
            photo, overlay, pheno, meta, status, url)


def random_in_species(name: str, method: str, opacity: float, resolution: int):
    if not name:
        return random_any(method, opacity, resolution)
    m = manifest()
    sub = m[m["taxon_name"] == name]
    if sub.empty:
        return name, pd.DataFrame(), f"no observations for {name}", None, None, "", "", "", ""
    row = sub.sample(1).iloc[0]
    df = list_obs_for_species(name)
    photo, overlay, pheno, meta, status, url = render(row["image_url_large"], method, opacity, resolution)
    return (name, df, f"random in **{name}** — showing {len(df)} of {len(sub):,} observations",
            photo, overlay, pheno, meta, status, url)


def on_obs_select(evt: gr.SelectData, df: pd.DataFrame,
                   method: str, opacity: float, resolution: int):
    if df is None or len(df) == 0 or evt.index is None:
        return None, None, "", "", "", ""
    row_idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
    url = df.iloc[row_idx]["image_url_large"]
    return render(url, method, opacity, resolution)


def recompute(url_holder: str, method: str, opacity: float, resolution: int):
    """Re-render the current photo with new overlay settings."""
    if not url_holder:
        return None, None, "", "", "", ""
    return render(url_holder, method, opacity, resolution)


# ─── UI ────────────────────────────────────────────────────────────────────
with gr.Blocks(title="California Flourishing & Pollination",
                theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🌼 California Flourishing & Pollination — Viewer")
    gr.Markdown(
        "Browse **10M+ iNaturalist Research-grade observations** of "
        "California-native plants + flying pollinators, encoded with "
        "**DINOv3 ViT-L/16** spatial features + **PhenoVision** "
        "flowering/fruiting probabilities. "
        "Patch-grid (14×14×1024) projected to 3D RGB via PCA "
        "(per-observation) or UMAP (pretrained, cross-species semantic)."
    )

    current_url = gr.State("")

    with gr.Row():
        species = gr.Dropdown(
            label="Species (autocomplete — type to search 16,000+ taxa)",
            choices=[], allow_custom_value=True, scale=4,
        )
        random_any_btn = gr.Button("🎲 Random across all species", variant="primary", scale=2)
        random_sp_btn = gr.Button("🎲 Random in selected species", scale=2)

    with gr.Row():
        with gr.Column(scale=2):
            status = gr.Markdown()
            obs_table = gr.Dataframe(
                headers=["observed_on", "locality", "image_url_large"],
                datatype=["str", "str", "str"],
                row_count=(0, "dynamic"),
                label="Observations — click a row to view",
                interactive=False,
                wrap=True,
                max_height=550,
            )

        with gr.Column(scale=3):
            with gr.Row():
                photo = gr.Image(label="iNat photo", height=420)
                overlay = gr.Image(label="DINOv3 patch-grid overlay", height=420)
            with gr.Row():
                method = gr.Radio(["PCA", "UMAP"], value="PCA",
                                   label="Projection 1024→3", scale=2)
                opacity = gr.Slider(0.0, 1.0, value=0.5, step=0.05,
                                     label="Overlay opacity", scale=3)
                resolution = gr.Slider(224, 1120, value=672, step=112,
                                        label="Overlay resolution (px)", scale=3)
            render_status = gr.Markdown()
            pheno_md = gr.Markdown()
            meta_md = gr.Markdown()

    # ─── wiring ────────────────────────────────────────────────────────────
    demo.load(lambda: gr.Dropdown(choices=species_list()), outputs=species)

    species.change(on_species_change, inputs=species, outputs=[obs_table, status])

    obs_table.select(
        on_obs_select,
        inputs=[obs_table, method, opacity, resolution],
        outputs=[photo, overlay, pheno_md, meta_md, render_status, current_url],
    )

    random_any_btn.click(
        random_any, inputs=[method, opacity, resolution],
        outputs=[species, obs_table, status, photo, overlay, pheno_md, meta_md,
                 render_status, current_url],
    )

    random_sp_btn.click(
        random_in_species, inputs=[species, method, opacity, resolution],
        outputs=[species, obs_table, status, photo, overlay, pheno_md, meta_md,
                 render_status, current_url],
    )

    # Re-render when overlay settings change
    for ctrl in (method, opacity, resolution):
        ctrl.change(
            recompute, inputs=[current_url, method, opacity, resolution],
            outputs=[photo, overlay, pheno_md, meta_md, render_status, current_url],
        )


if __name__ == "__main__":
    demo.queue(max_size=20).launch()
