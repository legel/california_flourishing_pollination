"""UMAP-based RGB visualization of DINOv3 spatial features.

Given the (H_p, W_p, D) patch tokens for one image, we:

1. Flatten to (H_p * W_p, D) feature vectors.
2. Project with UMAP to 3-D.
3. Min-max normalize each channel to 0–255 → RGB.
4. Reshape back to (H_p, W_p, 3) and bilinearly upsample to the source image.
5. Alpha-blend with the source at the requested opacity (default 50%).

UMAP is fit independently per image by default — the resulting RGB encodes
within-image relative semantics, which is what we want for a "do these tokens
make sense?" sanity check. Pass ``shared_umap`` to fit one UMAP across many
images jointly (so colors are comparable across the set).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
from PIL import Image

import umap


def _normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    """Min-max normalize each channel to 0–255 uint8."""
    out = np.empty_like(arr, dtype=np.float32)
    for c in range(arr.shape[-1]):
        x = arr[..., c]
        lo, hi = float(np.min(x)), float(np.max(x))
        out[..., c] = (x - lo) / (hi - lo) if hi > lo else 0.0
    return (out * 255.0).clip(0, 255).astype(np.uint8)


def umap_patches_to_rgb(
    patches: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 0,
) -> np.ndarray:
    """Project a (H_p, W_p, D) patch tensor to a (H_p, W_p, 3) uint8 RGB grid."""
    h_p, w_p, d = patches.shape
    flat = patches.reshape(h_p * w_p, d).astype(np.float32)
    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=min(n_neighbors, h_p * w_p - 1),
        min_dist=min_dist,
        random_state=random_state,
        metric="cosine",
    )
    proj = reducer.fit_transform(flat)
    rgb = _normalize_to_uint8(proj).reshape(h_p, w_p, 3)
    return rgb


def overlay_rgb_on_image(
    original: Image.Image,
    rgb_grid: np.ndarray,
    alpha: float = 0.5,
    upsample: str = "bilinear",
) -> Image.Image:
    """Upsample a small RGB grid to the original image size and alpha-blend."""
    target_size = original.size  # (W, H)
    resample = Image.BILINEAR if upsample == "bilinear" else Image.NEAREST
    rgb_img = Image.fromarray(rgb_grid, mode="RGB").resize(target_size, resample=resample)
    base = original.convert("RGB")
    blended = Image.blend(base, rgb_img, alpha=float(alpha))
    return blended


@dataclass
class OverlayResult:
    original: Image.Image
    rgb_grid: np.ndarray             # (H_p, W_p, 3) uint8
    rgb_upsampled: Image.Image       # full-resolution RGB (no blending)
    overlay: Image.Image             # blended visualization at the requested alpha

    def save_triplet(self, stem: str) -> Tuple[str, str, str]:
        """Save original, RGB-only, and overlay PNGs sharing a common stem."""
        p_orig = f"{stem}.png"
        p_rgb = f"{stem}_deep_features_rgb.png"
        p_over = f"{stem}_deep_features.png"
        self.original.save(p_orig)
        self.rgb_upsampled.save(p_rgb)
        self.overlay.save(p_over)
        return p_orig, p_rgb, p_over


def make_overlay(
    original: Image.Image,
    patches: np.ndarray,
    alpha: float = 0.5,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 0,
) -> OverlayResult:
    rgb_grid = umap_patches_to_rgb(
        patches,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
    )
    rgb_up = Image.fromarray(rgb_grid, mode="RGB").resize(original.size, Image.BILINEAR)
    overlay = Image.blend(original.convert("RGB"), rgb_up, alpha=alpha)
    return OverlayResult(
        original=original.convert("RGB"),
        rgb_grid=rgb_grid,
        rgb_upsampled=rgb_up,
        overlay=overlay,
    )
