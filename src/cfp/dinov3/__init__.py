"""DINOv3 spatial-embedding extractor.

Validation backbone (sanity check): facebook/dinov3-vitb16-pretrain-lvd1689m
Production  backbone (Phase 1 emb): facebook/dinov3-vitl16-pretrain-lvd1689m

The model is GATED on Hugging Face — accept the license at the repo page first,
then `hf auth login` so the local cache can pull the weights.
"""

from .extractor import DINOv3Extractor, DINOv3Output  # noqa: F401
