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

The contrastive module provides a CLIP-style dual encoder for aligning image and text representations. It includes a Vision Transformer image encoder, a lightweight text encoder/tokenizer stack, model bundle loading, and retrieval utilities.

Supported workflows:

- Image-to-text retrieval over caption candidates.
- Text-to-image retrieval over image candidates.
- Feature extraction from frozen vision encoders.
- Linear probing on frozen image features for structured visual attributes such as object count and color.

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

The QA module builds a vision-language inference path by combining a frozen visual encoder, a learned projector, and an LLM adapter. The visual stream produces image tokens, the projector maps them into the language model hidden space, and the language model produces answers for image-conditioned questions.

Key details:

- Loads vision encoder bundles and projector checkpoints independently.
- Supports LoRA adapter loading for the language model side.
- Includes prompt formatting and answer normalization for compact QA outputs.
- Operates on question JSON files with image paths and question records.

Example:

```bash
cd vision_language_qa
python inference.py \
  --model_dir /path/to/model_dir \
  --data_path /path/to/questions.json \
  --output_file outputs/qa_predictions.json
```

## Latent Diffusion

The latent diffusion module separates image compression from conditional generation. A convolutional VAE maps images into a latent space, while a U-Net denoiser is trained to sample latents through a Gaussian diffusion process. Captions are encoded through a frozen text encoder and used as conditioning for generation.

Supported workflows:

- Reconstruct input images through the VAE.
- Generate images from caption records.
- Load model manifests, latent statistics, and checkpoint artifacts from a model directory.
- Save generated image grids or per-sample outputs to a target directory.

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
