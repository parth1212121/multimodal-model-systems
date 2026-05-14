# Latent Diffusion

VAE reconstruction and text-conditioned image generation utilities.

## Examples

```bash
python inference.py \
  --model_dir /path/to/model_dir \
  --task reconstruct \
  --data_path /path/to/images \
  --output_dir outputs/reconstructions
```

```bash
python inference.py \
  --model_dir /path/to/model_dir \
  --task generate \
  --data_path /path/to/captions.json \
  --output_dir outputs/generated
```

