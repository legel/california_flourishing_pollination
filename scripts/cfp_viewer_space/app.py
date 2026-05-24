"""California Flourishing & Pollination — interactive viewer.

A Gradio Space that browses the deepearth/california-flourishing-pollination
HF dataset record-by-record, showing each photo with PhenoVision flowering /
fruiting probabilities and a 50% opacity DINOv3-PCA-RGB overlay derived from
the 14×14×1024 spatial patches.

Storage strategy: no embeddings are duplicated on the Space. We
  - ship a 200 MB master manifest as a Space file (`image_manifest.parquet`)
  - download embedding shards on demand via `hf_hub_download` (HF caches each
    shard locally after first fetch — ~3 GB per shard, persisted across requests)
  - require a `lookups/shard_index.parquet` on the dataset that maps
    `image_url_large -> shard_path`, so we can fetch only the relevant shard
    per query (build it once with `scripts/build_shard_index.py`).

Run locally:
    pip install gradio pandas pillow numpy huggingface_hub
    python app.py
"""
from __future__ import annotations
import io
import numpy as np
import pandas as pd
import gradio as gr
import requests
from PIL import Image
from huggingface_hub import hf_hub_download

REPO = "deepearth/california-flourishing-pollination"
MANIFEST_FILE = "manifests/image_manifest.parquet"
SHARD_INDEX_FILE = "lookups/shard_index.parquet"  # build via build_shard_index.py


# ─── lazy loaders (one-shot per Space worker) ─────────────────────────────
_manifest: pd.DataFrame | None = None
_shard_idx: pd.DataFrame | None = None
_species_list: list[str] | None = None


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
    global _species_list
    if _species_list is None:
        m = manifest()
        _species_list = sorted(m["taxon_name"].dropna().unique().tolist())
    return _species_list


# ─── render a single record ────────────────────────────────────────────────
def fetch_image(url: str) -> Image.Image:
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content)).convert("RGB")


def dinov3_pca_rgb(patches_hwd: np.ndarray) -> Image.Image:
    """PCA the 14×14×1024 patch grid to 3 channels, normalize to [0,255], to PIL."""
    h, w, d = patches_hwd.shape
    flat = patches_hwd.reshape(-1, d).astype(np.float32)
    flat -= flat.mean(0)
    cov = flat.T @ flat / max(1, flat.shape[0] - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    top3 = eigvecs[:, -3:][:, ::-1]
    proj = flat @ top3
    proj = (proj - proj.min(0)) / (proj.max(0) - proj.min(0) + 1e-8)
    grid = (proj.reshape(h, w, 3) * 255).astype(np.uint8)
    return Image.fromarray(grid).resize((224, 224), Image.NEAREST)


def overlay_dinov3(photo: Image.Image, patches_hwd: np.ndarray, alpha: float = 0.5) -> Image.Image:
    photo224 = photo.resize((224, 224), Image.BILINEAR)
    rgb = dinov3_pca_rgb(patches_hwd)
    return Image.blend(photo224, rgb, alpha)


def render_record(url: str):
    m = manifest()
    row = m[m["image_url_large"] == url]
    if row.empty:
        return None, None, "not found in manifest", "", ""
    row = row.iloc[0]

    # 1) fetch image from iNat
    try:
        img = fetch_image(url)
    except Exception as e:
        return None, None, f"iNat fetch failed: {e}", "", ""

    # 2) locate the shard, download (cached), pull the row's embedding
    idx_df = shard_idx()
    if url not in idx_df.index:
        # shard index not built yet — show image-only
        return img, None, "shard index not built yet (run build_shard_index.py)", \
               f"{row['taxon_name']} · {row['dataset_role']}", ""

    shard_path = idx_df.loc[url, "shard_path"]
    shard_local = hf_hub_download(REPO, shard_path, repo_type="dataset")
    s = pd.read_parquet(shard_local, columns=[
        "image_url_large", "patches_fp16", "patches_shape",
        "phenovision_flowering_prob", "phenovision_fruiting_prob",
    ])
    srow = s[s["image_url_large"] == url].iloc[0]
    patches = np.frombuffer(srow["patches_fp16"], dtype=np.float16).reshape(
        srow["patches_shape"]).astype(np.float32)

    overlay = overlay_dinov3(img, patches, alpha=0.5)
    pheno = f"flowering: {srow['phenovision_flowering_prob']:.2f}  ·  fruiting: {srow['phenovision_fruiting_prob']:.2f}"
    meta = (f"**{row['taxon_name']}** ({row['taxon_name_verbatim']})  \n"
            f"role: {row['dataset_role']} · family: {row['family']} · "
            f"observed: {row['observed_on']} · loc: {row['locality']}  \n"
            f"📍 {row['decimal_latitude']:.4f}, {row['decimal_longitude']:.4f}  \n"
            f"📷 {row['creator']} — {row['license']}  \n"
            f"[iNat photo]({url})")
    return img, overlay, pheno, row["taxon_name"], meta


# ─── UI ────────────────────────────────────────────────────────────────────
def by_species(name: str):
    m = manifest()
    matches = m[m["taxon_name"] == name].head(50)
    if matches.empty:
        return [], "no matches"
    urls = matches["image_url_large"].tolist()
    return urls, f"{len(urls)} of {(m['taxon_name']==name).sum()} obs"


def random_record():
    m = manifest()
    row = m.sample(1).iloc[0]
    return render_record(row["image_url_large"])


with gr.Blocks(title="CFP Viewer") as demo:
    gr.Markdown("# California Flourishing & Pollination — viewer")
    gr.Markdown("Browse 10M iNaturalist observations × DINOv3 ViT-L/16 + PhenoVision. "
                "DINOv3 14×14×1024 patches → PCA to 3 channels → 50% opacity overlay.")

    with gr.Row():
        with gr.Column(scale=1):
            sp = gr.Dropdown(label="Species (autocomplete)", choices=[], allow_custom_value=True)
            sp_btn = gr.Button("List obs for species")
            ramdom_btn = gr.Button("Random observation", variant="primary")
            url_in = gr.Textbox(label="image_url_large", placeholder="iNat /large.jpg URL")
            show_btn = gr.Button("Show this URL", variant="primary")
            urls_out = gr.Dataset(components=[gr.Textbox(visible=False)], samples=[], label="Pick one")
            status = gr.Markdown()

        with gr.Column(scale=2):
            with gr.Row():
                img_out = gr.Image(label="iNat photo", height=400)
                ovl_out = gr.Image(label="DINOv3 RGB overlay (50%)", height=400)
            pheno_out = gr.Markdown()
            taxon_out = gr.Markdown()
            meta_out = gr.Markdown()

    demo.load(fn=lambda: gr.Dropdown(choices=species_list()), outputs=sp)
    sp_btn.click(by_species, inputs=sp, outputs=[urls_out, status])
    ramdom_btn.click(random_record, outputs=[img_out, ovl_out, pheno_out, taxon_out, meta_out])
    show_btn.click(render_record, inputs=url_in,
                   outputs=[img_out, ovl_out, pheno_out, taxon_out, meta_out])
    urls_out.click(lambda sel: render_record(sel[0]) if sel else (None,) * 5,
                   inputs=urls_out, outputs=[img_out, ovl_out, pheno_out, taxon_out, meta_out])


if __name__ == "__main__":
    demo.queue(max_size=20).launch()
