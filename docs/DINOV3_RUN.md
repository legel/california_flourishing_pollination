# DINOv3 inference run — California Flourishing & Pollination

| | |
|---|---|
| Model | `facebook/dinov3-vitl16-pretrain-lvd1689m` (300 M params) |
| Precision | bf16 (fp16 → NaN in attention; bf16 numerically identical to fp32) |
| Input | 224 × 224 RGB, ImageNet-normalized |
| Output per image | CLS `(1024,)` + spatial patches `(14, 14, 1024)`, stored fp16 |
| Hardware | 1 × NVIDIA H200 |
| Decode → resize → forward | all on GPU (`torchvision.io.decode_jpeg(device='cuda')` → `F.interpolate` → DINOv3) |
| Throughput | ~170 img/sec sustained |

**Data: 9.85 M iNaturalist Research-grade images / 5.00 M observations / 16,400 species** (Calscape California-native plants + all California Insecta + Trochilidae + Chiroptera, sourced via two GBIF Occurrence Downloads — DOIs `10.15468/dl.pbgs4h` and `10.15468/dl.cvbfp4`). Multi-photo observations contribute one row per photo.

**Progress now: 3.65 M embeddings (37 %), 1.43 TB on Hugging Face.**

**Code:** https://github.com/legel/california_flourishing_pollination — see `src/cfp/dinov3/extractor_combined.py`
**Dataset:** https://huggingface.co/datasets/deepearth/california-flourishing-pollination
