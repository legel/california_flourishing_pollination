"""California Flourishing & Pollination — interactive viewer.

A Gradio Space that browses the deepearth/california-flourishing-pollination
HF dataset record-by-record:
  - photo from iNaturalist (browser-direct URL, fast)
  - DINOv3 14×14×1024 patches → 3D RGB overlay via a pretrained UMAP
    encoder (cross-species semantic — same flower-vs-leaf region → same color
    across observations). The encoder is loaded from a small numpy archive
    (no joblib class graph) and approximated via sklearn NearestNeighbors
    so it works under any Python version.
  - Photo + overlay stacked via CSS; opacity slider is JS-driven (instant)
  - PhenoVision flowering / fruiting probabilities (label-swap aware for
    shards uploaded before the embedder restart on 2026-05-24T07:09)
  - Per-photo CC license + creator attribution + full metadata
  - Default species on page load: Arctostaphylos pallida (Manzanita)
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

import json
import threading
from pathlib import Path

REPO = "deepearth/california-flourishing-pollination"
MANIFEST_FILE = "manifests/image_manifest.parquet"
SHARD_INDEX_FILE = "lookups/shard_index.parquet"
UMAP_NUMPY_FILE = "lookups/umap_numpy.npz"
SPECIES_LIST_FILE = Path(__file__).parent / "species_list.json"
MAX_OBS = 200
DEFAULT_SPECIES = "Arctostaphylos pallida"

# Shards with run_id >= this UTC stamp have correct PhenoVision column order.
# Older shards have flowering/fruiting columns swapped (vendor/phenovision/
# inference.py: class_names=['fruiting', 'flowering'] — index 0 is fruiting).
PHENOVISION_FIX_TIMESTAMP = "20260524T070916"


# ─── lazy loaders ──────────────────────────────────────────────────────────
_manifest: Optional[pd.DataFrame] = None
_shard_idx: Optional[pd.DataFrame] = None
_species: Optional[list[str]] = None
_umap_pack: Optional[dict] = None
_umap_error: Optional[str] = None


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
    """Static 16K-species list shipped with the Space (404 KB JSON, loads
    instantly — vs the 226 MB manifest download)."""
    global _species
    if _species is None:
        if SPECIES_LIST_FILE.exists():
            _species = json.loads(SPECIES_LIST_FILE.read_text())
        else:
            # fallback: derive from manifest (slower; only used if static missing)
            _species = sorted(manifest()["taxon_name"].dropna().unique().tolist())
    return _species


def warm_caches_background() -> None:
    """Kick off slow loads in a daemon thread so the UI is responsive
    immediately. Runs at Space boot, not on demo.load."""
    def _warm():
        try:
            print("[warm] manifest…", flush=True)
            manifest()
            print("[warm] shard index…", flush=True)
            shard_idx()
            print("[warm] umap encoder…", flush=True)
            umap_pack()
            print("[warm] DONE", flush=True)
        except Exception as e:
            print(f"[warm] failed: {e}", flush=True)
    threading.Thread(target=_warm, daemon=True).start()


warm_caches_background()


def umap_pack() -> Optional[dict]:
    """Build approximate-UMAP encoder from the extracted numpy arrays.

    `lookups/umap_numpy.npz` ships the UMAP training data (100K patches,
    fp16) + their 3-D embeddings + per-channel min/max. We fit a fast
    NearestNeighbors index on the training data and approximate transform
    via a distance-weighted average of neighbor embeddings — visually
    identical to UMAP's transform for this use, and works under any
    Python version (no class deserialization)."""
    global _umap_pack, _umap_error
    if _umap_pack is None:
        try:
            p = hf_hub_download(REPO, UMAP_NUMPY_FILE, repo_type="dataset")
            z = np.load(p)
            from sklearn.neighbors import NearestNeighbors
            n_neighbors = int(z["n_neighbors"])
            metric = str(z["metric"])
            nn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric, n_jobs=-1)
            nn.fit(z["training_data"].astype(np.float32))
            _umap_pack = {
                "nn": nn,
                "training_embeddings": z["training_embeddings"].astype(np.float32),
                "channel_min": z["channel_min"].astype(np.float32),
                "channel_max": z["channel_max"].astype(np.float32),
            }
            _umap_error = None
            print(f"[umap] loaded; {z['training_data'].shape[0]:,} train points, "
                  f"k={n_neighbors}, metric={metric}", flush=True)
        except Exception as e:
            _umap_pack = {}
            _umap_error = f"{type(e).__name__}: {e}"
            print(f"[umap] load failed: {_umap_error}", flush=True)
    return _umap_pack if _umap_pack else None


# ─── shard row lookup ──────────────────────────────────────────────────────
_shard_cache: dict[str, pd.DataFrame] = {}


def fetch_shard_row(url: str) -> Tuple[Optional[pd.Series], Optional[str]]:
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
    m = re.search(r"embeddings_(\d{8}T\d{6})_", shard_path or "")
    if m and m.group(1) >= PHENOVISION_FIX_TIMESTAMP:
        return float(srow["phenovision_flowering_prob"]), float(srow["phenovision_fruiting_prob"])
    return float(srow["phenovision_fruiting_prob"]), float(srow["phenovision_flowering_prob"])


# ─── UMAP projection ───────────────────────────────────────────────────────
def umap_rgb(patches_hwd: np.ndarray, normalize_per_image: bool = False) -> Optional[np.ndarray]:
    """Project DINOv3 patches to RGB via pretrained UMAP.

    normalize_per_image=False: use TRAINING-global channel ranges → colors
        are consistent across all observations (same patch concept → same
        color). Best for cross-image comparison.
    normalize_per_image=True:  use THIS image's min/max per channel → max
        visual contrast within the image (each pixel's color is divided by
        the image's own range). Best for spotting fine structure.
    """
    pack = umap_pack()
    if pack is None:
        return None
    h, w, d = patches_hwd.shape
    flat = patches_hwd.reshape(-1, d).astype(np.float32)
    dists, idxs = pack["nn"].kneighbors(flat)
    weights = 1.0 / (dists + 1e-6)
    weights = weights / weights.sum(axis=1, keepdims=True)
    selected = pack["training_embeddings"][idxs]   # (N, K, 3)
    proj = (weights[..., None] * selected).sum(axis=1)  # (N, 3)
    if normalize_per_image:
        cmin = proj.min(0); cmax = proj.max(0)
    else:
        cmin = pack["channel_min"]; cmax = pack["channel_max"]
    proj = np.clip((proj - cmin) / (cmax - cmin + 1e-8), 0, 1)
    return (proj.reshape(h, w, 3) * 255).astype(np.uint8)


def overlay_b64(patches_hwd: np.ndarray, resolution: int,
                 normalize_per_image: bool = False) -> Tuple[str, str]:
    rgb = umap_rgb(patches_hwd, normalize_per_image=normalize_per_image)
    if rgb is None:
        # Last-resort fallback: per-image PCA (sign-flips per image, but
        # at least something shows)
        flat = patches_hwd.reshape(-1, patches_hwd.shape[-1]).astype(np.float32)
        flat -= flat.mean(0)
        cov = flat.T @ flat / max(1, flat.shape[0] - 1)
        _, evec = np.linalg.eigh(cov)
        proj = flat @ evec[:, -3:][:, ::-1]
        proj = (proj - proj.min(0)) / (proj.max(0) - proj.min(0) + 1e-8)
        rgb = (proj.reshape(*patches_hwd.shape[:2], 3) * 255).astype(np.uint8)
        actual = f"per-image PCA (UMAP unavailable: {_umap_error})"
    else:
        actual = ("UMAP — per-image normalized (max contrast)"
                  if normalize_per_image else
                  "UMAP — cross-species global colors")
    img = Image.fromarray(rgb).resize((resolution, resolution), Image.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return ("data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(), actual)


# ─── HTML stack ────────────────────────────────────────────────────────────
def stack_html(photo_url: str, overlay_url: Optional[str], opacity: float = 0.5) -> str:
    base = (f'<div class="cfp-stack">'
            f'<img src="{photo_url}" class="cfp-photo" alt="iNaturalist photo">')
    if overlay_url:
        base += (f'<img src="{overlay_url}" id="cfp-overlay" class="cfp-overlay" '
                 f'style="opacity:{opacity:.2f}" alt="DINOv3 UMAP overlay">')
    base += "</div>"
    return base


# ─── core render ───────────────────────────────────────────────────────────
def render(url: str, opacity: float, resolution: int, normalize: bool = False):
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
               "_(no embedding shard found for this URL — try a different observation)_", url
    flowering, fruiting = pheno_corrected(srow, shard_path)
    pheno = (f"### PhenoVision\n"
             f"🌸 flowering: **{flowering:.3f}**  \n"
             f"🍒 fruiting: **{fruiting:.3f}**")
    patches = np.frombuffer(srow["patches_fp16"], dtype=np.float16).reshape(
        tuple(srow["patches_shape"])).astype(np.float32)
    ovl_url, actual_method = overlay_b64(patches, resolution, normalize_per_image=normalize)
    html = stack_html(url, ovl_url, opacity)
    status = f"_overlay: {actual_method} @ {resolution}px (opacity = client-side, instant)_"

    # Kick off prefetch of next random observation's shard (warming cache)
    _prefetch_next_random_async()
    return html, pheno, meta, status, url


def _prefetch_next_random_async() -> None:
    """Warm the shard cache for a random observation so the next click is fast."""
    def _warm():
        try:
            m = manifest()
            url = m.sample(1).iloc[0]["image_url_large"]
            fetch_shard_row(url)  # populates _shard_cache
        except Exception:
            pass
    threading.Thread(target=_warm, daemon=True).start()


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
    samples = df.values.tolist() if len(df) else []
    return gr.Dataset(samples=samples), \
        f"showing {len(samples)} of {total:,} observations for **{name}**"


def random_any(opacity: float, resolution: int, normalize: bool):
    m = manifest()
    row = m.sample(1).iloc[0]
    species = row["taxon_name"]
    df = list_obs_for_species(species)
    samples = df.values.tolist() if len(df) else []
    html, pheno, meta, status, url = render(row["image_url_large"], opacity, resolution, normalize)
    total = (m["taxon_name"] == species).sum()
    return (species, gr.Dataset(samples=samples),
            f"random pick: **{species}** — showing {len(samples)} of {total:,} observations",
            html, pheno, meta, status, url)


def random_in_species(name: str, opacity: float, resolution: int, normalize: bool):
    if not name:
        return random_any(opacity, resolution, normalize)
    m = manifest()
    sub = m[m["taxon_name"] == name]
    if sub.empty:
        return name, gr.Dataset(samples=[]), f"no observations for {name}", "", "", "", "", ""
    row = sub.sample(1).iloc[0]
    df = list_obs_for_species(name)
    samples = df.values.tolist() if len(df) else []
    html, pheno, meta, status, url = render(row["image_url_large"], opacity, resolution, normalize)
    return (name, gr.Dataset(samples=samples),
            f"random in **{name}** — showing {len(samples)} of {len(sub):,} observations",
            html, pheno, meta, status, url)


def on_obs_select(sample, opacity: float, resolution: int, normalize: bool):
    if not sample or len(sample) < 3:
        return "", "", "", "_(no row selected)_", ""
    url = sample[2]
    if not url:
        return "", "", "", "_(no URL in selected row)_", ""
    return render(str(url), opacity, resolution, normalize)


def recompute(url_holder: str, opacity: float, resolution: int, normalize: bool):
    if not url_holder:
        return "", "", "", "", ""
    return render(url_holder, opacity, resolution, normalize)


def init_load(opacity: float, resolution: int, normalize: bool):
    """Page-load: populate species dropdown (instant, static list) + render
    Arctostaphylos pallida's first observation (uses manifest, may take a
    few seconds if the background warm hasn't finished)."""
    spp = species_list()
    sp = DEFAULT_SPECIES if DEFAULT_SPECIES in spp else (spp[0] if spp else "")
    df = list_obs_for_species(sp)
    samples = df.values.tolist() if len(df) else []
    if samples:
        url = samples[0][2]
        html, pheno, meta, status, _ = render(url, opacity, resolution, normalize)
    else:
        url = ""; html = pheno = meta = status = ""
    total = (manifest()["taxon_name"] == sp).sum() if _manifest is not None else 0
    return (gr.Dropdown(choices=spp, value=sp),
            gr.Dataset(samples=samples),
            f"showing {len(samples)} of {total:,} observations for **{sp}**",
            html, pheno, meta, status, url)


# ─── UI ────────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Oxygen:wght@300;400;700&display=swap');
*, *::before, *::after { font-family: 'Oxygen', 'Helvetica Neue', sans-serif !important; }
code, pre, kbd, samp { font-family: 'Oxygen Mono', 'JetBrains Mono', 'Menlo', monospace !important; }
.cfp-stack { position: relative; display: inline-block; max-width: 100%; border-radius: 8px; overflow: hidden; background: #111; }
.cfp-photo { display: block; max-width: 100%; width: auto; height: auto; border-radius: 8px; }
.cfp-overlay { position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; border-radius: 8px; transition: opacity 0.05s linear; }
"""
JS_OPACITY = """
(v) => { const e = document.getElementById('cfp-overlay'); if (e) e.style.opacity = v; return v; }
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
        "projected to 3D RGB via a **pretrained UMAP** (cross-species "
        "semantic — same flower-vs-leaf region → same color across "
        "observations). Opacity slider is JS-driven instant."
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
            obs_table = gr.Dataset(
                components=[gr.Textbox(visible=False),
                            gr.Textbox(visible=False),
                            gr.Textbox(visible=False)],
                headers=["observed_on", "locality", "image_url_large"],
                samples=[],
                label="Observations — click a row to view",
                samples_per_page=10,
                type="values",
            )

        with gr.Column(scale=3):
            stacked = gr.HTML(label="iNat photo + DINOv3 UMAP overlay")
            with gr.Row():
                opacity = gr.Slider(0.0, 1.0, value=0.5, step=0.01,
                                     label="Overlay opacity (instant)", scale=3)
                resolution = gr.Slider(224, 1120, value=672, step=112,
                                        label="Overlay resolution (px)", scale=2)
                normalize = gr.Checkbox(
                    value=True,
                    label="Normalize colors per-image (max contrast)",
                    info="on = per-channel min/max from this image (default); off = cross-species global",
                    scale=2,
                )
            render_status = gr.Markdown()
            pheno_md = gr.Markdown()
            meta_md = gr.Markdown()

    # ─── wiring ────────────────────────────────────────────────────────────
    demo.load(init_load, inputs=[opacity, resolution, normalize],
              outputs=[species, obs_table, status, stacked, pheno_md, meta_md,
                       render_status, current_url])

    species.change(on_species_change, inputs=species, outputs=[obs_table, status])

    obs_table.click(
        on_obs_select, inputs=[obs_table, opacity, resolution, normalize],
        outputs=[stacked, pheno_md, meta_md, render_status, current_url],
    )

    random_any_btn.click(
        random_any, inputs=[opacity, resolution, normalize],
        outputs=[species, obs_table, status, stacked, pheno_md, meta_md,
                 render_status, current_url],
    )

    random_sp_btn.click(
        random_in_species, inputs=[species, opacity, resolution, normalize],
        outputs=[species, obs_table, status, stacked, pheno_md, meta_md,
                 render_status, current_url],
    )

    # Resolution or normalize → server recompute. Opacity → JS-only.
    for ctrl in (resolution, normalize):
        ctrl.change(
            recompute, inputs=[current_url, opacity, resolution, normalize],
            outputs=[stacked, pheno_md, meta_md, render_status, current_url],
        )
    opacity.input(None, inputs=opacity, outputs=None, js=JS_OPACITY)


if __name__ == "__main__":
    demo.queue(max_size=20).launch()
