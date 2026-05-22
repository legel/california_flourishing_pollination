"""DINOv3 + PhenoVision combined extractor.

A single GPU forward-pass session per image:
  1. nvJPEG decode (CUDA, ~2 ms/img on H200)
  2. Resize + normalize on GPU
  3. **DINOv3 ViT-L/16** forward pass → CLS + spatial patches
  4. **PhenoVision** (Dinnage 2025; phenobase/phenovision) ViT-B/16 forward
     pass → multi-label sigmoid for flowering + fruiting

Per-image output extends the DINOv3 schema with two new probabilities:

    cls_fp16          (1024,) DINOv3 ViT-L/16 CLS
    patches_fp16      (14, 14, 1024) DINOv3 spatial patches
    flowering_prob    float32   (PhenoVision sigmoid output[0])
    fruiting_prob     float32   (PhenoVision sigmoid output[1])

Models share the same normalized (224, 224) tensor on GPU. Total per-batch
overhead vs DINOv3-only: ~30% (PhenoVision is ViT-B/16 = ~3× faster than
ViT-L/16, but both run on every image so we pay ~1.3× total).

Preprocessing for both models uses ImageNet mean/std at 224² — verified
identical via inspection of:
  - DINOv3 ViT-L/16 preprocessing: mean [0.485, 0.456, 0.406] std [0.229, 0.224, 0.225] (ImageNet)
  - PhenoVision ViT-B/16 (google/vit-base-patch16-224): same ImageNet stats.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.io as tvio
from transformers import AutoModel, ViTForImageClassification

from .extractor import BACKBONES, _N_REGISTER_TOKENS


_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


@dataclass
class CombinedOutput:
    """Output of one combined DINOv3 + PhenoVision batch.

    Attributes
    ----------
    cls : np.ndarray
        Shape ``(N, D)`` — DINOv3 CLS tokens.
    patches : np.ndarray
        Shape ``(N, H_p, W_p, D)`` — DINOv3 spatial patches.
    flowering_prob : np.ndarray
        Shape ``(N,)`` — PhenoVision flower-presence probability (sigmoid).
    fruiting_prob : np.ndarray
        Shape ``(N,)`` — PhenoVision fruit-presence probability (sigmoid).
    grid_hw : tuple[int, int]
    image_size : tuple[int, int]
    """

    cls: np.ndarray
    patches: np.ndarray
    flowering_prob: np.ndarray
    fruiting_prob: np.ndarray
    grid_hw: Tuple[int, int]
    image_size: Tuple[int, int]


class DINOv3PhenoVisionExtractor:
    """Single-pass DINOv3 ViT-L/16 + PhenoVision ViT-B/16 inference on GPU."""

    def __init__(
        self,
        dinov3_backbone: str = "vitl16",
        phenovision_repo: str = "phenobase/phenovision",
        image_size: int = 224,
        dtype: Optional[torch.dtype] = None,
        device: Optional[str] = None,
    ) -> None:
        self.image_size = image_size
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        # DINOv3 ViT-L/16 fp16 → NaN; bf16 = fp32-quality at fp16 cost.
        self.dtype = dtype or (torch.bfloat16 if self.device.type == "cuda" else torch.float32)

        dinov3_repo = BACKBONES.get(dinov3_backbone, dinov3_backbone)
        self.dinov3_repo = dinov3_repo
        self.phenovision_repo = phenovision_repo
        # Compat with the simpler extractors that expose ``.repo`` / ``.embed_dim``.
        self.repo = dinov3_repo
        self.embed_dim = None  # set below after model load

        # DINOv3
        self.dinov3 = AutoModel.from_pretrained(dinov3_repo, torch_dtype=self.dtype).to(self.device).eval()
        self.dinov3_dim = int(self.dinov3.config.hidden_size)
        self.embed_dim = self.dinov3_dim
        self.patch_size = int(getattr(self.dinov3.config, "patch_size", 16))
        self.grid_hw = (image_size // self.patch_size, image_size // self.patch_size)

        # PhenoVision (ViT-B/16 multi-label classifier; MIT, Dinnage 2025)
        self.phenovision = ViTForImageClassification.from_pretrained(
            phenovision_repo, torch_dtype=self.dtype
        ).to(self.device).eval()

        self.mean = _IMAGENET_MEAN.to(self.device, self.dtype)
        self.std = _IMAGENET_STD.to(self.device, self.dtype)

    @torch.inference_mode()
    def embed_from_bytes(self, byte_buffers: List[bytes]) -> Tuple[CombinedOutput, list[int]]:
        """Decode + DINOv3 + PhenoVision in one GPU session."""
        n = len(byte_buffers)
        if n == 0:
            raise ValueError("embed_from_bytes() received an empty batch")

        per_image: list[torch.Tensor] = []
        failed: list[int] = []
        for idx, buf in enumerate(byte_buffers):
            img = None
            try:
                bt = torch.frombuffer(bytearray(buf), dtype=torch.uint8)
                img = tvio.decode_jpeg(bt, mode=tvio.ImageReadMode.RGB, device=self.device)
            except Exception:
                pass
            if img is None:
                try:
                    bt = torch.frombuffer(bytearray(buf), dtype=torch.uint8)
                    img = tvio.decode_image(bt, mode=tvio.ImageReadMode.RGB).to(self.device, non_blocking=True)
                except Exception:
                    pass
            if img is None:
                try:
                    from PIL import Image
                    import io as _io
                    pil = Image.open(_io.BytesIO(buf)).convert("RGB")
                    img = torch.from_numpy(np.asarray(pil)).permute(2, 0, 1).contiguous()
                    img = img.to(self.device, non_blocking=True)
                except Exception:
                    pass
            if img is None:
                failed.append(idx)
                continue

            img = img.unsqueeze(0).to(self.dtype) / 255.0
            img = F.interpolate(
                img, size=(self.image_size, self.image_size),
                mode="bilinear", align_corners=False, antialias=True,
            )
            per_image.append(img)

        if not per_image:
            empty = np.zeros((0, self.dinov3_dim), dtype=np.float32)
            empty_p = np.zeros((0, self.grid_hw[0], self.grid_hw[1], self.dinov3_dim), dtype=np.float32)
            empty_prob = np.zeros((0,), dtype=np.float32)
            return (CombinedOutput(empty, empty_p, empty_prob, empty_prob,
                                   self.grid_hw, (self.image_size, self.image_size)), failed)

        batch = torch.cat(per_image, dim=0)
        batch = (batch - self.mean) / self.std

        # DINOv3 forward
        out_d = self.dinov3(pixel_values=batch)
        h = out_d.last_hidden_state
        n_ok, total_tokens, d = h.shape
        h_p, w_p = self.grid_hw
        expected = 1 + _N_REGISTER_TOKENS + h_p * w_p
        if total_tokens == expected:
            cls = h[:, 0]
            patches = h[:, 1 + _N_REGISTER_TOKENS :]
        elif total_tokens == 1 + h_p * w_p:
            cls = h[:, 0]
            patches = h[:, 1:]
        else:
            raise RuntimeError(
                f"Unexpected token count {total_tokens} for grid {self.grid_hw}"
            )
        patches = patches.reshape(n_ok, h_p, w_p, d)

        # PhenoVision forward (same preprocessed batch)
        out_p = self.phenovision(pixel_values=batch)
        # Multi-label sigmoid: logits shape (N, 2). index 0 = flowering, 1 = fruiting.
        probs = torch.sigmoid(out_p.logits.float())
        flowering = probs[:, 0].cpu().numpy()
        fruiting = probs[:, 1].cpu().numpy() if probs.shape[1] > 1 else np.zeros(n_ok, dtype=np.float32)

        return (CombinedOutput(
            cls=cls.float().cpu().numpy(),
            patches=patches.float().cpu().numpy(),
            flowering_prob=flowering.astype(np.float32),
            fruiting_prob=fruiting.astype(np.float32),
            grid_hw=self.grid_hw,
            image_size=(self.image_size, self.image_size),
        ), failed)
