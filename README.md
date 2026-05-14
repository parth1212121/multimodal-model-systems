# Multimodal Model Systems

Author: Parth Verma

This repository contains a collection of multimodal learning systems that connect images, text, and generation. It covers three related model families: contrastive vision-text representation learning, vision-language question answering, and latent diffusion for image reconstruction and text-conditioned generation.

The code is organized around runnable model pipelines rather than isolated model definitions. Each module includes inference entry points, artifact loading, preprocessing, and JSON-style input/output conventions so that trained components can be evaluated or composed without rewriting the surrounding infrastructure.

## System Overview

| Module | Role in the stack | Main capabilities |
| --- | --- | --- |
| `contrastive_vision_text/` | Learns and evaluates shared image-text embeddings | CLIP-style dual encoder, ViT image backbone, text tokenizer, retrieval, frozen-feature probing |
| `vision_language_qa/` | Connects a visual encoder to a language model interface | visual token projection, LLM adapter loading, CLEVR-style question answering |
| `latent_diffusion/` | Builds an image generation path in latent space | convolutional VAE, U-Net denoiser, Gaussian diffusion scheduler, CLIP text conditioning |

## Repository Layout

```text
.
├── contrastive_vision_text/
│   ├── contrastive/              # CLIP-style model, ViT backbone, tokenizer, data utilities
│   ├── inference_retrieval.py    # image-to-text and text-to-image retrieval
│   └── inference_linear_probing.py
├── vision_language_qa/
│   ├── vlm/                      # vision encoder loader and VLM projection components
│   └── inference.py              # question-answering inference
├── latent_diffusion/
│   ├── ldm/                      # VAE, U-Net, diffusion, text encoder, pipeline helpers
│   └── inference.py              # reconstruction and generation inference
└── requirements.txt
```

## Contrastive Vision-Text

Aim:

This module studies whether image encoders can learn useful scene representations from paired image-caption data or from self-supervised image-only training. It compares CLIP-style image-text contrastive learning with DINO-style self-distillation, then probes the learned embeddings through retrieval, linear classification, and t-SNE visualization.

The contrastive module provides a CLIP-style dual encoder for aligning image and text representations. It includes a Vision Transformer image encoder, a lightweight text encoder/tokenizer stack, model bundle loading, and retrieval utilities.

Supported workflows:

- Image-to-text retrieval over caption candidates.
- Text-to-image retrieval over image candidates.
- Feature extraction from frozen vision encoders.
- Linear probing on frozen image features for structured visual attributes such as object count and color.

Reported results:

- CLIP learned a strong shared embedding space for CLEVR-style image-caption pairs, with **90.12% Recall@1** and **97.79% Recall@3** for image-to-text retrieval.
- Text-to-image retrieval was similarly strong, reaching **91.20% Recall@1** and **98.52% Recall@3**.
- t-SNE visualizations showed that DINO embeddings organized object-count information more cleanly than CLIP, while CLIP better captured caption-level semantic attributes such as color, material, shape, and size.
- Linear probing indicated complementary behavior: CLIP was stronger for color semantics and retrieval, while DINO GAP features were stronger for object counting.

Retrieval example:

```bash
cd contrastive_vision_text
python inference_retrieval.py \
  --model_type clip \
  --model_dir /path/to/model_dir \
  --retrieval_task i2t \
  --data_path /path/to/retrieval_input.json \
  --output_file outputs/retrieval_predictions.json
```

Linear probing example:

```bash
python inference_linear_probing.py \
  --model_type clip \
  --model_dir /path/to/model_dir \
  --pooling_type cls \
  --probe_task count \
  --data_path /path/to/probe_input.json \
  --output_file outputs/probe_predictions.json
```

## Vision-Language QA

Aim:

This module turns frozen visual representations into a question-answering system by aligning image tokens with a language model. The goal is to test whether a pretrained/frozen vision encoder can be connected to an instruction-tuned LLM through a learned projector and lightweight LoRA adaptation, while still producing grounded answers for visual reasoning questions.

The QA module builds a vision-language inference path by combining a frozen visual encoder, a learned projector, and an LLM adapter. The visual stream produces image tokens, the projector maps them into the language model hidden space, and the language model produces answers for image-conditioned questions.

Key details:

- Loads vision encoder bundles and projector checkpoints independently.
- Supports LoRA adapter loading for the language model side.
- Includes prompt formatting and answer normalization for compact QA outputs.
- Operates on question JSON files with image paths and question records.

Reported results:

- Stage 1 trained only the visual projector for caption alignment while keeping both the vision encoder and Qwen frozen; on 5,000 validation examples it reached **82.613 BLEU** for generated captions.
- Stage 2 initialized from the caption-aligned projector, then trained the projector plus Qwen attention LoRA modules for QA.
- The best Stage 2 model was evaluated on the full validation split of **149,984** generated QA examples and achieved **88.74% normalized answer exact match**.
- The result suggests that frozen CLIP visual features plus a learned projector and LoRA-adapted Qwen can capture substantial CLEVR-style visual grounding.

Example:

```bash
cd vision_language_qa
python inference.py \
  --model_dir /path/to/model_dir \
  --data_path /path/to/questions.json \
  --output_file outputs/qa_predictions.json
```

## Latent Diffusion

Aim:

This module builds a two-stage generative image pipeline: first learning a compact continuous latent space with a VAE, then training a text-conditioned diffusion model to sample in that latent space. The purpose is to make caption-conditioned generation more efficient than pixel-space diffusion while preserving enough visual structure for reconstruction and synthesis.

The latent diffusion module separates image compression from conditional generation. A convolutional VAE maps images into a latent space, while a U-Net denoiser is trained to sample latents through a Gaussian diffusion process. Captions are encoded through a frozen text encoder and used as conditioning for generation.

Supported workflows:

- Reconstruct input images through the VAE.
- Generate images from caption records.
- Load model manifests, latent statistics, and checkpoint artifacts from a model directory.
- Save generated image grids or per-sample outputs to a target directory.

Reported results:

- The VAE compressed **128 x 128 x 3** images into a **16 x 16 x 4** latent representation and achieved a validation reconstruction **FID of 3.9582**.
- The latent diffusion model used a conditional U-Net, 500 diffusion timesteps, normalized VAE latents, and frozen CLIP text embeddings for conditioning.
- The best LDM checkpoint occurred at epoch 88 and achieved a full-validation generation **FID of 9.6165** over 10K caption-conditioned samples.
- A post-training sweep over classifier-free guidance scales and nearby checkpoints did not improve the original best checkpoint, so the reported generation result was retained.
- Qualitatively, VAE reconstructions preserved object layout and count well with mild blur, while LDM outputs captured broad CLEVR structure but were expectedly harder than reconstruction.

Reconstruction example:

```bash
cd latent_diffusion
python inference.py \
  --model_dir /path/to/model_dir \
  --task reconstruct \
  --data_path /path/to/images \
  --output_dir outputs/reconstructions
```

Generation example:

```bash
python inference.py \
  --model_dir /path/to/model_dir \
  --task generate \
  --data_path /path/to/captions.json \
  --output_dir outputs/generated
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Datasets, pretrained models, checkpoints, LoRA adapters, tokenizer files, and generated outputs are intentionally kept outside version control. The inference scripts expect those artifacts to be supplied through `--model_dir`, `--data_path`, and output path arguments.
