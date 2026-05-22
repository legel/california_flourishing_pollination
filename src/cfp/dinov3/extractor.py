"""DINOv3 spatial-embedding extractor.

DINOv3 (Meta AI) is a self-supervised ViT family. We use it as a frozen
feature extractor and persist two outputs per image:

  - CLS token         (global image embedding, used for image-level tasks)
  - spatial patches   (H/16 × W/16 grid of patch tokens, used for spatial tasks)

DINOv3 inserts 4 register tokens after the CLS token — these are discarded.
The exact token layout (per Meta's reference impl + Hugging Face port) is:
    last_hidden_state[:, 0]          = CLS
    last_hidden_state[:, 1:5]        = 4 register tokens   (discarded)
    last_hidden_state[:, 5:]         = (H/16 × W/16) patch tokens, row-major

For square 224 input → 14×14 = 196 patch tokens, dim 768 (ViT-B) or 1024 (ViT-L).
For square 448 input → 28×28 = 784 patch tokens. We default to 224 for production
throughput; the validation script may override to 448 for a higher-resolution
overlay map.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

BACKBONES = {
    "vits16": "facebook/dinov3-vits16-pretrain-lvd1689m",
    "vitb16": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "vitl16": "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "vith16plus": "facebook/dinov3-vith16plus-pretrain-lvd1689m",
}

# DINOv3 register-token count (per the reference implementation).
_N_REGISTER_TOKENS = 4


@dataclass
class DINOv3Output:
    """One DINOv3 forward pass result for a batch of images.

    Attributes
    ----------
    cls : np.ndarray
        Shape ``(N, D)``. CLS token per image — global embedding.
    patches : np.ndarray
        Shape ``(N, H_p, W_p, D)``. Patch tokens reshaped to a 2-D grid.
    grid_hw : tuple[int, int]
        ``(H_p, W_p)`` patch-grid dimensions (same for every image in the batch).
    image_size : tuple[int, int]
        ``(H, W)`` pixel size fed into the model.
    """

    cls: np.ndarray
    patches: np.ndarray
    grid_hw: Tuple[int, int]
    image_size: Tuple[int, int]


class DINOv3Extractor:
    """Frozen DINOv3 spatial-token extractor.

    Loads a Hugging Face DINOv3 checkpoint once and serves repeated batched
    forward passes. ``model.eval()`` is enforced and ``torch.inference_mode()``
    wraps every call — gradients never accumulate.

    Parameters
    ----------
    backbone : str
        Short name (vits16 / vitb16 / vitl16 / vith16plus) or full HF repo id.
    image_size : int
        Square input size in pixels. Default 224. Use 448 for the validation
        overlay so the UMAP→RGB grid covers the image at finer resolution.
    dtype : torch.dtype
        ``float16`` on GPU (default), ``float32`` on CPU.
    device : str
        ``cuda`` (default if available) or ``cpu``.
    """

    def __init__(
        self,
        backbone: str = "vitb16",
        image_size: int = 224,
        dtype: Optional[torch.dtype] = None,
        device: Optional[str] = None,
    ) -> None:
        repo = BACKBONES.get(backbone, backbone)
        self.repo = repo
        self.image_size = image_size
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        # DINOv3 ViT-L/16 in fp16 overflows in attention → NaN. bf16 has the same
        # memory + throughput as fp16 with the dynamic range of fp32; verified
        # numerically identical to fp32 to 3+ decimals on this model.
        self.dtype = dtype or (torch.bfloat16 if self.device.type == "cuda" else torch.float32)

        # AutoImageProcessor pulls the checkpoint's preprocessing config
        # (ImageNet mean/std, bicubic resize, center-crop). use_fast=True picks
        # the torchvision-backed processor — orders of magnitude faster than the
        # PIL+numpy default, and avoids starving the GPU on CPU preprocessing.
        try:
            self.processor = AutoImageProcessor.from_pretrained(repo, use_fast=True)
        except Exception:
            self.processor = AutoImageProcessor.from_pretrained(repo)
        self.processor.size = {"height": image_size, "width": image_size}
        if hasattr(self.processor, "do_center_crop"):
            self.processor.do_center_crop = False
        self.processor.crop_size = {"height": image_size, "width": image_size}

        self.model = AutoModel.from_pretrained(repo, torch_dtype=self.dtype).to(self.device).eval()
        self.embed_dim = int(self.model.config.hidden_size)
        self.patch_size = int(getattr(self.model.config, "patch_size", 16))
        self.grid_hw = (image_size // self.patch_size, image_size // self.patch_size)

    @torch.inference_mode()
    def embed(self, images: Sequence[Image.Image]) -> DINOv3Output:
        """Run one forward pass for a batch of PIL images."""
        if not images:
            raise ValueError("embed() received an empty image batch")

        inputs = self.processor(images=list(images), return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device, dtype=self.dtype, non_blocking=True)
        out = self.model(pixel_values=pixel_values)

        # last_hidden_state: (N, 1 + R + H_p*W_p, D)
        h = out.last_hidden_state
        n, total_tokens, d = h.shape
        h_p, w_p = self.grid_hw
        expected = 1 + _N_REGISTER_TOKENS + h_p * w_p
        # Some HF ports may not expose registers separately. Fall back gracefully.
        if total_tokens == expected:
            cls = h[:, 0]
            patches = h[:, 1 + _N_REGISTER_TOKENS :]
        elif total_tokens == 1 + h_p * w_p:
            cls = h[:, 0]
            patches = h[:, 1:]
        else:
            raise RuntimeError(
                f"Unexpected token count {total_tokens} for grid {self.grid_hw} "
                f"(expected {expected} or {1 + h_p * w_p}). Check image_size vs. patch_size."
            )

        patches = patches.reshape(n, h_p, w_p, d)

        return DINOv3Output(
            cls=cls.float().cpu().numpy(),
            patches=patches.float().cpu().numpy(),
            grid_hw=self.grid_hw,
            image_size=(self.image_size, self.image_size),
        )

    def embed_iter(
        self,
        images: Iterable[Image.Image],
        batch_size: int = 16,
    ) -> Iterable[DINOv3Output]:
        """Stream batched embeddings over an arbitrary iterable of PIL images."""
        batch: List[Image.Image] = []
        for img in images:
            batch.append(img)
            if len(batch) == batch_size:
                yield self.embed(batch)
                batch = []
        if batch:
            yield self.embed(batch)
