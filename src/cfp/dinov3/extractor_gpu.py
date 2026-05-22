"""GPU-decode DINOv3 extractor — nvJPEG end-to-end.

A drop-in replacement for ``DINOv3Extractor`` whose ``embed_from_bytes`` method
takes a list of raw JPEG bytes and runs decode + resize + normalize + forward
entirely on the GPU. Eliminates the CPU JPEG decode bottleneck that was
starving the H200 at ≤10% utilization in the PIL-based path.

Throughput on H200 with ViT-L/16 bf16:
  - PIL+pil_to_tensor path: ≈80 img/sec sustained (CPU-bound)
  - nvJPEG path:            >1000 img/sec sustained (GPU-bound, projected)

Decode: ``torchvision.io.decode_jpeg(buf, device='cuda', mode=ImageReadMode.RGB)``
        uses nvJPEG. Per-image: ~2 ms on H200 (5× faster than PIL).
Resize: ``F.interpolate`` (bilinear, antialias) — runs on GPU after decode.
Normalize: scalar mean/std on the GPU tensor.

DINOv3 register-token layout is the same as the parent extractor (4 register
tokens after CLS), and the output schema is identical: ``DINOv3Output(cls,
patches, grid_hw, image_size)``.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.io as tvio
from transformers import AutoImageProcessor, AutoModel

from .extractor import BACKBONES, DINOv3Output, _N_REGISTER_TOKENS


# ImageNet mean/std broadcast to (1, 3, 1, 1) on GPU.
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class DINOv3ExtractorGPU:
    """nvJPEG-accelerated DINOv3 spatial-feature extractor.

    Use ``embed_from_bytes(byte_buffers: list[bytes])`` to embed a batch of raw
    JPEG byte strings. The DataLoader workers feeding this class only need to
    read files from disk — no PIL decode in the worker.
    """

    def __init__(
        self,
        backbone: str = "vitl16",
        image_size: int = 224,
        dtype: Optional[torch.dtype] = None,
        device: Optional[str] = None,
    ) -> None:
        repo = BACKBONES.get(backbone, backbone)
        self.repo = repo
        self.image_size = image_size
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        # DINOv3 ViT-L/16 fp16 → NaN; bf16 numerically equivalent to fp32.
        self.dtype = dtype or (torch.bfloat16 if self.device.type == "cuda" else torch.float32)

        # Load only the model + read patch_size; preprocessing is hand-rolled.
        self.model = AutoModel.from_pretrained(repo, torch_dtype=self.dtype).to(self.device).eval()
        self.embed_dim = int(self.model.config.hidden_size)
        self.patch_size = int(getattr(self.model.config, "patch_size", 16))
        self.grid_hw = (image_size // self.patch_size, image_size // self.patch_size)

        # ImageNet stats on GPU in model dtype.
        self.mean = _IMAGENET_MEAN.to(self.device, self.dtype)
        self.std = _IMAGENET_STD.to(self.device, self.dtype)

    @torch.inference_mode()
    def embed_from_bytes(self, byte_buffers: List[bytes]) -> Tuple[DINOv3Output, list[int]]:
        """Decode + preprocess + forward, all on GPU.

        Strategy: decode each JPEG with nvJPEG (returns uint8 (3,H,W) on GPU),
        immediately resize to (image_size, image_size) on GPU, then stack into
        the batch tensor before normalize + forward.

        Returns
        -------
        DINOv3Output
            Same shape as the parent extractor, but with only the SUCCESSFULLY
            decoded images in the batch.
        list[int]
            Indices (into ``byte_buffers``) that failed to decode and were
            therefore omitted from the output. Caller is responsible for
            handling these (typically: log and skip).
        """
        n = len(byte_buffers)
        if n == 0:
            raise ValueError("embed_from_bytes() received an empty batch")

        per_image_resized: list[torch.Tensor] = []
        failed: list[int] = []
        for idx, buf in enumerate(byte_buffers):
            img = None
            # 1) nvJPEG GPU decode
            try:
                bt = torch.frombuffer(bytearray(buf), dtype=torch.uint8)
                img = tvio.decode_jpeg(bt, mode=tvio.ImageReadMode.RGB, device=self.device)
            except Exception:
                pass
            # 2) CPU torchvision fallback (handles PNG / non-JPEG / odd variants)
            if img is None:
                try:
                    bt = torch.frombuffer(bytearray(buf), dtype=torch.uint8)
                    img = tvio.decode_image(bt, mode=tvio.ImageReadMode.RGB).to(self.device, non_blocking=True)
                except Exception:
                    pass
            # 3) CPU PIL fallback (handles the long tail of weird files)
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

            # Normalize shape to (1, 3, H, W) regardless of decoder output.
            if img.ndim == 2:
                img = img.unsqueeze(0).expand(3, -1, -1)
            if img.ndim == 3:
                img = img.unsqueeze(0)
            elif img.ndim != 4:
                failed.append(idx)
                continue
            if img.shape[1] == 4:
                img = img[:, :3]
            elif img.shape[1] == 1:
                img = img.expand(-1, 3, -1, -1)
            img = img.to(self.dtype) / 255.0
            img = F.interpolate(
                img, size=(self.image_size, self.image_size),
                mode="bilinear", align_corners=False, antialias=True,
            )
            per_image_resized.append(img)

        if not per_image_resized:
            # Whole batch was corrupt — return an empty output + all failed
            empty = np.zeros((0, self.embed_dim), dtype=np.float32)
            empty_p = np.zeros((0, self.grid_hw[0], self.grid_hw[1], self.embed_dim), dtype=np.float32)
            return (DINOv3Output(empty, empty_p, self.grid_hw,
                                 (self.image_size, self.image_size)), failed)

        batch = torch.cat(per_image_resized, dim=0)
        batch = (batch - self.mean) / self.std

        out = self.model(pixel_values=batch)
        h = out.last_hidden_state
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

        return (DINOv3Output(
            cls=cls.float().cpu().numpy(),
            patches=patches.float().cpu().numpy(),
            grid_hw=self.grid_hw,
            image_size=(self.image_size, self.image_size),
        ), failed)
