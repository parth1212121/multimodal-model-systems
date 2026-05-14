# Multimodal Model Systems

Author: Parth Verma

This repository contains PyTorch implementations for multimodal representation learning, vision-language question answering, and text-conditioned image generation.

## Modules

| Module | Focus | Entry points |
| --- | --- | --- |
| `contrastive_vision_text/` | CLIP-style image-text retrieval and vision encoder probing | `inference_retrieval.py`, `inference_linear_probing.py` |
| `vision_language_qa/` | Vision-language question answering with a frozen vision encoder and LLM adapter | `inference.py` |
| `latent_diffusion/` | VAE reconstruction and latent diffusion image generation | `inference.py` |

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

## Highlights

- CLIP-style dual encoder for image-text retrieval.
- Vision Transformer feature extraction with linear probes for object count and color prediction.
- Vision-language QA pipeline that projects visual tokens into an LLM interface.
- Convolutional VAE and latent diffusion model for reconstruction and text-conditioned generation.
- Modular inference scripts with JSON input/output support.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Datasets, checkpoints, adapters, and large model artifacts are expected to live outside the repository. Pass their local paths through the command-line interfaces below.

## Contrastive Vision-Text

Image-to-text or text-to-image retrieval:

```bash
cd contrastive_vision_text
python inference_retrieval.py \
  --model_type clip \
  --model_dir /path/to/model_dir \
  --retrieval_task i2t \
  --data_path /path/to/retrieval_input.json \
  --output_file outputs/retrieval_predictions.json
```

Linear probing on frozen image features:

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

```bash
cd vision_language_qa
python inference.py \
  --model_dir /path/to/model_dir \
  --data_path /path/to/questions.json \
  --output_file outputs/qa_predictions.json
```

## Latent Diffusion

Reconstruct images with the VAE:

```bash
cd latent_diffusion
python inference.py \
  --model_dir /path/to/model_dir \
  --task reconstruct \
  --data_path /path/to/images \
  --output_dir outputs/reconstructions
```

Generate images from captions:

```bash
python inference.py \
  --model_dir /path/to/model_dir \
  --task generate \
  --data_path /path/to/captions.json \
  --output_dir outputs/generated
```

