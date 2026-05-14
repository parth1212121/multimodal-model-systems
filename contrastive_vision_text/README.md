# Contrastive Vision-Text Models

CLIP-style image-text models for retrieval and frozen-feature probing.

## Entry Points

- `inference_retrieval.py`: image-to-text and text-to-image retrieval.
- `inference_linear_probing.py`: linear probes on frozen image encoder features.

## Example

```bash
python inference_retrieval.py \
  --model_type clip \
  --model_dir /path/to/model_dir \
  --retrieval_task i2t \
  --data_path /path/to/retrieval_input.json \
  --output_file outputs/retrieval_predictions.json
```

