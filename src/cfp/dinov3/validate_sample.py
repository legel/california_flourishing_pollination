"""10-image DINOv3 sanity check.

Pulls N California-native-plant observation photos from the iNaturalist v1 API
(at the requested image size), runs the DINOv3 ViT-B/16 spatial extractor,
projects per-image patch tokens to RGB via UMAP, and writes a triplet of PNGs
per image plus a single zip + a Markdown contact-sheet for visual review.

This is a GATE: do not proceed to the production ViT-L/16 streaming pass until
the user has visually confirmed the overlays segment plant structure (leaves,
flowers, stems) in a semantically coherent way.

Run:
    python -m cfp.dinov3 validate-sample \\
        --n 10 --backbone vitb16 --image-size 448 \\
        --outdir data/validation/dinov3_sanity

Provenance:
    Every taxon + observation + image URL queried is logged to
    ``<outdir>/_provenance.jsonl`` for traceability into PROVENANCE.md.
"""

from __future__ import annotations

import json
import shutil
import time
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import List, Optional

import requests
import typer
from PIL import Image
from rich.console import Console

from .extractor import DINOv3Extractor
from .visualize import make_overlay

app = typer.Typer(add_completion=False, help="DINOv3 10-image sanity check.")
console = Console()

INAT_API = "https://api.inaturalist.org/v1"
CALFLORA_TAXON_SCHEME_ID = 12  # iNaturalist taxon scheme #12 = Calflora (CA-native plants)
USER_AGENT = "DeepEarth-CFP/0.1 (validation sample; lance@3co.ai)"

# Default seed taxa for the sanity check — iconic California natives spanning
# diverse growth forms (geophyte, shrub, tree, perennial, succulent). Used only
# until the full CNPS/iNat-Calflora list is materialized; override with --taxa
# from the produced native list once available.
DEFAULT_TAXA: List[str] = [
    "Eschscholzia californica",   # California poppy
    "Arctostaphylos manzanita",   # Manzanita
    "Quercus agrifolia",          # Coast live oak
    "Cercis occidentalis",        # Western redbud
    "Eriogonum fasciculatum",     # California buckwheat
    "Diplacus aurantiacus",       # Bush monkeyflower
    "Heteromeles arbutifolia",    # Toyon
    "Baccharis pilularis",        # Coyote brush
    "Epilobium canum",            # California fuchsia
    "Romneya coulteri",           # Matilija poppy
]


def _resolve_taxon_id(name: str) -> Optional[int]:
    """Resolve a scientific name → iNaturalist taxon id (rank=species)."""
    r = requests.get(
        f"{INAT_API}/taxa",
        params={"q": name, "rank": "species", "is_active": "true"},
        headers={"User-Agent": USER_AGENT},
        timeout=30,
    )
    r.raise_for_status()
    results = r.json().get("results", [])
    for res in results:
        if res.get("name", "").lower() == name.lower():
            return int(res["id"])
    return int(results[0]["id"]) if results else None


def _pick_observation_photo(
    taxon_id: int,
    place_id_california: int = 14,
    quality: str = "research",
) -> Optional[dict]:
    """Return one Research-grade observation photo for the taxon in CA."""
    params = {
        "taxon_id": taxon_id,
        "place_id": place_id_california,
        "quality_grade": quality,
        "photos": "true",
        "per_page": 20,
        "order_by": "votes",
        "order": "desc",
    }
    r = requests.get(
        f"{INAT_API}/observations",
        params=params,
        headers={"User-Agent": USER_AGENT},
        timeout=30,
    )
    r.raise_for_status()
    results = r.json().get("results", [])
    for obs in results:
        for p in obs.get("photos", []) or []:
            url = p.get("url")
            if not url:
                continue
            # iNat returns the "square" thumbnail URL; swap to "large".
            large_url = url.replace("/square", "/large")
            return {
                "observation_id": obs["id"],
                "observation_uuid": obs.get("uuid"),
                "photo_id": p["id"],
                "image_url": large_url,
                "license_code": p.get("license_code") or obs.get("license_code"),
                "attribution": p.get("attribution"),
                "observed_on": obs.get("observed_on"),
                "place_guess": obs.get("place_guess"),
                "user_login": (obs.get("user") or {}).get("login"),
                "latitude": (obs.get("geojson") or {}).get("coordinates", [None, None])[1],
                "longitude": (obs.get("geojson") or {}).get("coordinates", [None, None])[0],
            }
    return None


def _download_image(url: str) -> Image.Image:
    r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=60)
    r.raise_for_status()
    return Image.open(BytesIO(r.content)).convert("RGB")


@app.callback(invoke_without_command=True)
def main(
    n: int = typer.Option(10, "--n", help="Number of sample images."),
    backbone: str = typer.Option("vitb16", "--backbone", help="DINOv3 backbone short name."),
    image_size: int = typer.Option(448, "--image-size", help="DINOv3 input size in pixels."),
    outdir: Path = typer.Option(
        Path("data/validation/dinov3_sanity"),
        "--outdir",
        help="Output directory for PNGs + zip + provenance.",
    ),
    alpha: float = typer.Option(0.5, "--alpha", help="Overlay opacity (0–1)."),
    taxa_file: Optional[Path] = typer.Option(
        None, "--taxa", help="Optional newline-delimited file of taxon names; defaults to a curated CA list."
    ),
    seed: int = typer.Option(0, "--seed", help="UMAP random_state."),
) -> None:
    """Run the 10-image DINOv3 sanity check."""
    taxa = (
        [line.strip() for line in taxa_file.read_text().splitlines() if line.strip()]
        if taxa_file
        else DEFAULT_TAXA
    )
    if len(taxa) < n:
        raise typer.BadParameter(f"Have {len(taxa)} taxa but need {n}")
    taxa = taxa[:n]

    outdir.mkdir(parents=True, exist_ok=True)
    workdir = outdir / "_workdir"
    workdir.mkdir(exist_ok=True)
    prov_path = outdir / "_provenance.jsonl"
    prov_f = prov_path.open("w")
    prov_meta = {
        "stage": "dinov3_validate_sample",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "backbone": backbone,
        "image_size": image_size,
        "alpha": alpha,
        "n": n,
        "seed": seed,
        "taxa": taxa,
        "inat_api": INAT_API,
        "taxon_scheme_id": CALFLORA_TAXON_SCHEME_ID,
    }
    prov_f.write(json.dumps({"type": "run_meta", **prov_meta}) + "\n")
    console.rule("[bold cyan]DINOv3 validation sample")
    console.print(prov_meta)

    # Step 1: download images
    samples = []
    for idx, name in enumerate(taxa, start=1):
        console.print(f"[cyan]({idx}/{n})[/] resolving [bold]{name}[/]")
        tid = _resolve_taxon_id(name)
        if tid is None:
            console.print(f"  [red]no taxon match — skipping[/]")
            continue
        photo = _pick_observation_photo(tid)
        if photo is None:
            console.print(f"  [yellow]no CA Research-grade photo — skipping[/]")
            continue
        img = _download_image(photo["image_url"])
        img_path = workdir / f"{idx}.png"
        img.save(img_path)
        samples.append({"idx": idx, "name": name, "taxon_id": tid, "image_path": str(img_path), **photo})
        prov_f.write(json.dumps({"type": "sample", "idx": idx, "name": name, "taxon_id": tid, **photo}) + "\n")
        console.print(f"  [green]saved[/] → {img_path}  ({img.size[0]}×{img.size[1]})")
        time.sleep(0.5)  # be polite to the iNat API

    if not samples:
        raise typer.Exit(code=2)

    # Step 2: load DINOv3 + embed
    console.rule("[bold cyan]Loading DINOv3")
    extractor = DINOv3Extractor(backbone=backbone, image_size=image_size)
    console.print(
        f"backbone=[bold]{backbone}[/]  embed_dim={extractor.embed_dim}  "
        f"patch={extractor.patch_size}  grid={extractor.grid_hw}  device={extractor.device}"
    )
    prov_f.write(
        json.dumps(
            {
                "type": "extractor",
                "backbone": backbone,
                "repo": extractor.repo,
                "embed_dim": extractor.embed_dim,
                "patch_size": extractor.patch_size,
                "grid_hw": list(extractor.grid_hw),
                "image_size": image_size,
                "device": str(extractor.device),
                "dtype": str(extractor.dtype),
            }
        )
        + "\n"
    )

    # Step 3: per-image overlay
    console.rule("[bold cyan]Per-image UMAP overlay")
    triplets = []
    for s in samples:
        img = Image.open(s["image_path"]).convert("RGB")
        out = extractor.embed([img])
        patches = out.patches[0]  # (H_p, W_p, D)
        overlay = make_overlay(img, patches, alpha=alpha, random_state=seed)
        stem = str(workdir / str(s["idx"]))
        p_orig, p_rgb, p_over = overlay.save_triplet(stem)
        triplets.append((s, p_orig, p_rgb, p_over))
        prov_f.write(
            json.dumps(
                {
                    "type": "overlay",
                    "idx": s["idx"],
                    "name": s["name"],
                    "image_url": s["image_url"],
                    "license_code": s["license_code"],
                    "files": {"original": p_orig, "rgb": p_rgb, "overlay": p_over},
                }
            )
            + "\n"
        )
        console.print(f"  [green]({s['idx']}) {s['name']}[/] → {Path(p_over).name}")

    prov_f.close()

    # Step 4: contact-sheet README + zip
    contact = ["# DINOv3 sanity check\n", f"_Generated {datetime.now(timezone.utc).isoformat()}_\n"]
    contact.append(
        f"Backbone `{backbone}` ({extractor.repo}) — embed_dim={extractor.embed_dim}, "
        f"patch={extractor.patch_size}, grid={extractor.grid_hw}, input={image_size}².\n"
    )
    contact.append("| # | Taxon | Original | UMAP→RGB | Overlay |\n|---|---|---|---|---|\n")
    for s, p_orig, p_rgb, p_over in triplets:
        contact.append(
            f"| {s['idx']} | _{s['name']}_ "
            f"| ![]({Path(p_orig).name}) "
            f"| ![]({Path(p_rgb).name}) "
            f"| ![]({Path(p_over).name}) |\n"
        )
    (workdir / "README.md").write_text("".join(contact))

    zip_path = outdir / f"dinov3_sanity_{backbone}_{image_size}.zip"
    shutil.make_archive(str(zip_path.with_suffix("")), "zip", workdir)
    console.rule("[bold green]Done")
    console.print(f"PNGs + README + provenance in: {workdir}")
    console.print(f"Zip: {zip_path}")
    console.print(f"Provenance log: {prov_path}")


if __name__ == "__main__":
    app()
