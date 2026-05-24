"""California Flourishing & Pollination — interactive viewer.

A Gradio Space that browses the deepearth/california-flourishing-pollination
HF dataset record-by-record:
  - photo from iNaturalist (browser-direct URL, fast)
  - DINOv3 14×14×1024 patches → 3D RGB overlay
      * PCA (per-observation SVD)
      * UMAP (pretrained 1024→3 encoder for cross-species semantic coloring)
  - Photo + overlay stacked via CSS so the **opacity slider is JS-driven**
    (no server roundtrip; instant)
  - Per-photo CC license + creator attribution + full metadata
  - PhenoVision flowering / fruiting probabilities (with old-shard label-swap
    correction — see PHENOVISION_FIX_TIMESTAMP below)

Storage strategy: ships only the ~225 MB master manifest + 110 MB shard
index + ~1.5 GB UMAP encoder. Embedding shards (3 GB each, 1,072 total)
are pulled on demand from the parent dataset and cached on the Space disk.
"""
from __future__ import annotations
import base64
import io
import re
from typing import Optional, Tuple

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
MAX_OBS = 200

# Shards with run_id >= this UTC stamp have the correct PhenoVision column
# order. Older shards have flowering / fruiting columns swapped (a bug in
# the original combined-extractor; vendor/phenovision/inference.py confirms
# class_names = ['fruiting', 'flowering']).
PHENOVISION_FIX_TIMESTAMP = "20260524T070916"


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


_umap_error: Optional[str] = None


def umap_pack() -> Optional[dict]:
    """Lazy-load the pretrained UMAP(1024->3) encoder + global channel ranges.

    On failure (missing dep, corrupt download, ...) the error is captured in
    _umap_error so the UI can surface "UMAP requested but failed: <reason>"
    instead of silently falling back to PCA."""
    global _umap_pack, _umap_error
    if _umap_pack is None:
        try:
            import joblib
            import umap  # noqa: F401  — needed for the deserialized UMAP object
            p = hf_hub_download(REPO, UMAP_FILE, repo_type="dataset")
            _umap_pack = joblib.load(p)
            _umap_error = None
        except Exception as e:
            _umap_pack = {}
            _umap_error = f"{type(e).__name__}: {e}"
            print(f"[umap] load failed — falling back to PCA: {_umap_error}", flush=True)
    return _umap_pack if _umap_pack else None


def umap_error() -> Optional[str]:
    """Surface the last UMAP load error (None if not yet attempted or success)."""
    return _umap_error


# ─── shard row lookup ──────────────────────────────────────────────────────
_shard_cache: dict[str, pd.DataFrame] = {}


def fetch_shard_row(url: str) -> Tuple[Optional[pd.Series], Optional[str]]:
    """Return (row, shard_path). shard_path tells us which PhenoVision label era."""
    idx = shard_idx()
    if url not in idx.index:
        return None, None
    shard_path = idx.loc[url, "shard_path"]
    if isinstance(shard_path, pd.Series):
        shard_path = shard_path.iloc[0]
    if shard_path not in _shard_cache:
        local = hf_hub_download(REPO, shard_path, repo_type="dataset")
        _shard_cache[shard_path] = pd.read_parquet(local, columns=[
            "image_url_large", "patches_fp16", "patches_shape",
            "phenovision_flowering_prob", "phenovision_fruiting_prob",
        ])
    df = _shard_cache[shard_path]
    match = df[df["image_url_large"] == url]
    return (match.iloc[0], shard_path) if len(match) else (None, shard_path)


def pheno_corrected(srow: pd.Series, shard_path: str) -> Tuple[float, float]:
    """Return (flowering, fruiting) with old-shard column-swap correction."""
    m = re.search(r"embeddings_(\d{8}T\d{6})_", shard_path or "")
    if m and m.group(1) >= PHENOVISION_FIX_TIMESTAMP:
        # New shards: column names are correct.
        return float(srow["phenovision_flowering_prob"]), float(srow["phenovision_fruiting_prob"])
    # Old shards have columns swapped: the value stored as "_flowering_prob"
    # is actually fruiting probability and vice versa.
    return float(srow["phenovision_fruiting_prob"]), float(srow["phenovision_flowering_prob"])


# ─── projections ───────────────────────────────────────────────────────────
def pca_rgb(patches_hwd: np.ndarray) -> np.ndarray:
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


def overlay_b64(patches_hwd: np.ndarray, method: str, resolution: int) -> Tuple[str, str]:
    """Return (data-URL of upscaled overlay PNG, method label actually used).

    If UMAP is requested but unavailable, the second tuple element will be
    "PCA (UMAP unavailable: <error>)" so the UI tells the truth."""
    actual = method
    rgb = None
    if method == "UMAP":
        rgb = umap_rgb(patches_hwd)
        if rgb is None:
            err = umap_error() or "encoder not loaded"
            actual = f"PCA (UMAP fallback — {err})"
    if rgb is None:
        rgb = pca_rgb(patches_hwd)
    img = Image.fromarray(rgb).resize((resolution, resolution), Image.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return ("data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(), actual)


# ─── HTML composition ──────────────────────────────────────────────────────
def stack_html(photo_url: str, overlay_url: Optional[str], opacity: float = 0.5) -> str:
    """Photo + overlay stacked via absolute positioning. The overlay's CSS
    opacity is mutated client-side by the slider's js callback."""
    base = (f'<div class="cfp-stack">'
            f'<img src="{photo_url}" class="cfp-photo" alt="iNaturalist photo">')
    if overlay_url:
        base += (f'<img src="{overlay_url}" id="cfp-overlay" class="cfp-overlay" '
                 f'style="opacity:{opacity:.2f}" alt="DINOv3 overlay">')
    base += "</div>"
    return base


# ─── core render ───────────────────────────────────────────────────────────
def render(url: str, method: str, opacity: float, resolution: int):
    """Return (html, pheno_md, meta_md, status_md, url_state)."""
    if not url:
        return "", "", "", "", ""
    m = manifest()
    row = m[m["image_url_large"] == url]
    if row.empty:
        return stack_html(url, None, opacity), "", "", "URL not found in manifest", ""
    row = row.iloc[0]

    meta = (f"### {row['taxon_name']}\n"
            f"*{row['taxon_name_verbatim']}*  \n"
            f"role: **{row['dataset_role']}** · family: {row['family']}  \n"
            f"observed: {row['observed_on']}  \n"
            f"locality: {row['locality']}  \n"
            f"📍 {row['decimal_latitude']:.4f}, {row['decimal_longitude']:.4f}  \n"
            f"📷 {row['creator']} — `{row['license']}`  \n"
            f"[iNaturalist photo source]({url})")

    srow, shard_path = fetch_shard_row(url)
    if srow is None:
        return stack_html(url, None, opacity), "", meta, \
               "_(no embedding shard found for this URL)_", url

    flowering, fruiting = pheno_corrected(srow, shard_path)
    pheno = (f"### PhenoVision\n"
             f"🌸 flowering: **{flowering:.3f}**  \n"
             f"🍒 fruiting: **{fruiting:.3f}**")

    patches = np.frombuffer(srow["patches_fp16"], dtype=np.float16).reshape(
        tuple(srow["patches_shape"])).astype(np.float32)
    ovl_url, actual_method = overlay_b64(patches, method, resolution)
    html = stack_html(url, ovl_url, opacity)
    status = f"_overlay: {actual_method} @ {resolution}px (opacity changes are client-side and instant)_"
    return html, pheno, meta, status, url


# ─── list builders / random / events ───────────────────────────────────────
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
    html, pheno, meta, status, url = render(row["image_url_large"], method, opacity, resolution)
    total = (m["taxon_name"] == species).sum()
    return (species, df, f"random pick: **{species}** — showing {len(df)} of {total:,} observations",
            html, pheno, meta, status, url)


def random_in_species(name: str, method: str, opacity: float, resolution: int):
    if not name:
        return random_any(method, opacity, resolution)
    m = manifest()
    sub = m[m["taxon_name"] == name]
    if sub.empty:
        return name, pd.DataFrame(), f"no observations for {name}", "", "", "", "", ""
    row = sub.sample(1).iloc[0]
    df = list_obs_for_species(name)
    html, pheno, meta, status, url = render(row["image_url_large"], method, opacity, resolution)
    return (name, df, f"random in **{name}** — showing {len(df)} of {len(sub):,} observations",
            html, pheno, meta, status, url)


def on_obs_select(evt: gr.SelectData, df: pd.DataFrame,
                   method: str, opacity: float, resolution: int):
    if df is None or len(df) == 0 or evt.index is None:
        return "", "", "", "", ""
    row_idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
    url = df.iloc[row_idx]["image_url_large"]
    return render(url, method, opacity, resolution)


def recompute(url_holder: str, method: str, opacity: float, resolution: int):
    """Re-render only when method or resolution changes (opacity is JS-driven)."""
    if not url_holder:
        return "", "", "", "", ""
    return render(url_holder, method, opacity, resolution)


# ─── UI ────────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Oxygen:wght@300;400;700&display=swap');
*, *::before, *::after { font-family: 'Oxygen', 'Helvetica Neue', sans-serif !important; }
code, pre, kbd, samp { font-family: 'Oxygen Mono', 'JetBrains Mono', 'Menlo', monospace !important; }
.cfp-stack {
    position: relative;
    display: inline-block;
    max-width: 100%;
    border-radius: 8px;
    overflow: hidden;
    background: #111;
}
.cfp-photo {
    display: block;
    max-width: 100%;
    width: auto;
    height: auto;
    border-radius: 8px;
}
.cfp-overlay {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    pointer-events: none;
    border-radius: 8px;
    mix-blend-mode: normal;
    image-rendering: auto;
    transition: opacity 0.05s linear;
}
"""

JS_OPACITY = """
(v) => {
  const el = document.getElementById('cfp-overlay');
  if (el) el.style.opacity = v;
  return v;
}
"""

with gr.Blocks(title="California Flourishing & Pollination",
                theme=gr.themes.Soft(font=[gr.themes.GoogleFont("Oxygen"), "sans-serif"]),
                css=CUSTOM_CSS) as demo:
    gr.Markdown("# 🌼 California Flourishing & Pollination")
    gr.Markdown(
        "Browse **10M+ iNaturalist Research-grade observations** of "
        "California-native plants + flying pollinators, encoded with "
        "**DINOv3 ViT-L/16** spatial features + **PhenoVision** "
        "flowering/fruiting probabilities. The 14×14×1024 patch grid is "
        "projected to 3D RGB and overlaid on the photo. "
        "*Opacity is client-side instant; method + resolution recompute the overlay.*"
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
                max_height=600,
            )

        with gr.Column(scale=3):
            stacked = gr.HTML(label="iNat photo + DINOv3 patch-grid overlay")
            with gr.Row():
                method = gr.Radio(["PCA", "UMAP"], value="UMAP",
                                   label="Projection (1024 → 3)", scale=2)
                opacity = gr.Slider(0.0, 1.0, value=0.5, step=0.01,
                                     label="Overlay opacity (instant)", scale=3)
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
        outputs=[stacked, pheno_md, meta_md, render_status, current_url],
    )

    random_any_btn.click(
        random_any, inputs=[method, opacity, resolution],
        outputs=[species, obs_table, status, stacked, pheno_md, meta_md,
                 render_status, current_url],
    )

    random_sp_btn.click(
        random_in_species, inputs=[species, method, opacity, resolution],
        outputs=[species, obs_table, status, stacked, pheno_md, meta_md,
                 render_status, current_url],
    )

    # Method + resolution → server recompute (need to rebuild overlay PNG)
    for ctrl in (method, resolution):
        ctrl.change(
            recompute, inputs=[current_url, method, opacity, resolution],
            outputs=[stacked, pheno_md, meta_md, render_status, current_url],
        )

    # Opacity → JS-only CSS update (no server roundtrip, no recompute)
    opacity.input(None, inputs=opacity, outputs=None, js=JS_OPACITY)


if __name__ == "__main__":
    demo.queue(max_size=20).launch()
