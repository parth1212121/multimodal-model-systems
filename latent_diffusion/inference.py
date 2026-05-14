from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
VENDOR = ROOT / "_vendor"
if sys.platform.startswith("linux") and VENDOR.is_dir() and str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from ldm.config import load_config, load_json
from ldm.data import IMAGE_EXTENSIONS, caption_to_key, load_captions_from_json
from ldm.diffusion import GaussianDiffusion
from ldm.models.unet import LatentDiffusionModel
from ldm.models.vae import ConvVAE
from ldm.pipeline import (
    build_ldm_from_config,
    build_vae_from_config,
    latent_stats_to_tensors,
)
from ldm.text import FrozenClipTextEncoder
from ldm.training import get_device, save_tensor_image, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Latent diffusion inference")
    parser.add_argument("--model_dir", required=True, help="Directory containing latent diffusion model artifacts")
    parser.add_argument("--task", required=True, choices=["reconstruct", "generate"], help="Inference task")
    parser.add_argument("--data_path", required=True, help="Input image directory or caption JSON")
    parser.add_argument("--output_dir", required=True, help="Directory to save generated output images")
    return parser.parse_args()


def load_manifest(model_dir: Path) -> dict[str, Any]:
    manifest_path = model_dir / "manifest.json"
    if not manifest_path.exists():
        return {}
    with manifest_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_artifact(
    model_dir: Path,
    manifest: dict[str, Any],
    key: str,
    candidates: list[str],
    description: str,
    expect_dir: bool = False,
) -> Path:
    if key in manifest:
        candidate = model_dir / manifest[key]
        if expect_dir and candidate.is_dir():
            return candidate
        if not expect_dir and candidate.is_file():
            return candidate
        raise FileNotFoundError(f"Manifest entry '{key}' points to missing {description}: {candidate}")

    for relative in candidates:
        candidate = model_dir / relative
        if expect_dir and candidate.is_dir():
            return candidate
        if not expect_dir and candidate.is_file():
            return candidate

    searched = ", ".join(candidates)
    raise FileNotFoundError(f"Could not find {description} under {model_dir}. Tried: {searched}")


def load_checkpoint_state(path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location=device)
    if isinstance(payload, dict):
        if "model_state" in payload and isinstance(payload["model_state"], dict):
            return payload["model_state"]
        if "state_dict" in payload and isinstance(payload["state_dict"], dict):
            return payload["state_dict"]
    if isinstance(payload, dict):
        return payload
    raise TypeError(f"Unsupported checkpoint format at {path}")


def load_latent_stats_from_path(path: Path) -> dict[str, list[float]]:
    if path.suffix.lower() == ".json":
        return load_json(path)
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "mean" in payload and "std" in payload:
        return {
            "mean": [float(value) for value in payload["mean"]],
            "std": [float(value) for value in payload["std"]],
        }
    raise TypeError(f"Unsupported latent-stats format at {path}")


def load_vae_model(config: dict[str, Any], checkpoint_path: Path, device: torch.device) -> ConvVAE:
    model = build_vae_from_config(config).to(device)
    model.load_state_dict(load_checkpoint_state(checkpoint_path, device))
    model.eval()
    return model


def load_ldm_model(config: dict[str, Any], checkpoint_path: Path, device: torch.device) -> LatentDiffusionModel:
    model = build_ldm_from_config(config).to(device)
    model.load_state_dict(load_checkpoint_state(checkpoint_path, device))
    model.eval()
    return model


def list_input_images(data_dir: Path) -> list[Path]:
    if not data_dir.is_dir():
        raise NotADirectoryError(f"Expected an image directory for reconstruct task, got: {data_dir}")
    images = [
        path
        for path in sorted(data_dir.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not images:
        raise RuntimeError(f"No input images found directly inside {data_dir}")
    return images


def build_image_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Lambda(lambda tensor: tensor * 2.0 - 1.0),
        ]
    )


def reconstruct_images(
    vae_cfg: dict[str, Any],
    vae_model: ConvVAE,
    data_path: Path,
    output_dir: Path,
    batch_size: int,
    device: torch.device,
) -> None:
    image_size = int(vae_cfg.get("data", {}).get("image_size", 128))
    transform = build_image_transform(image_size)
    input_images = list_input_images(data_path)

    with torch.no_grad():
        for start in tqdm(range(0, len(input_images), batch_size), desc="Reconstructing"):
            batch_paths = input_images[start : start + batch_size]
            batch_tensor = torch.stack(
                [transform(Image.open(path).convert("RGB")) for path in batch_paths],
                dim=0,
            ).to(device)
            reconstructions = vae_model(batch_tensor, sample_posterior=False)["reconstruction"]
            for image_path, reconstruction in zip(batch_paths, reconstructions):
                destination = output_dir / f"{image_path.stem}.png"
                save_tensor_image(reconstruction, destination)


def precompute_text_embeddings(
    records: list[dict[str, str]],
    text_encoder_dir: Path,
    device: torch.device,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    unique_captions: list[str] = []
    seen: set[str] = set()
    for record in records:
        caption = record["caption_text"]
        key = caption_to_key(caption)
        if key in seen:
            continue
        seen.add(key)
        unique_captions.append(caption)

    encoder = FrozenClipTextEncoder(str(text_encoder_dir), device)
    return encoder.encode_texts(unique_captions, batch_size=batch_size)


def load_fixed_text_embeddings(
    records: list[dict[str, str]],
    fixed_embedding_path: Path,
) -> dict[str, torch.Tensor]:
    payload = torch.load(fixed_embedding_path, map_location="cpu")
    embedding = payload["embedding"] if isinstance(payload, dict) and "embedding" in payload else payload
    embedding = embedding.detach().float().cpu()
    embeddings: dict[str, torch.Tensor] = {}
    for record in records:
        embeddings[record["caption_key"]] = embedding
    return embeddings


def infer_latent_shape(ldm_cfg: dict[str, Any]) -> tuple[int, int]:
    latent_channels = int(ldm_cfg.get("model", {}).get("latent_channels", 4))
    if "latent_resolution" in ldm_cfg.get("model", {}):
        resolution = int(ldm_cfg["model"]["latent_resolution"])
        return latent_channels, resolution

    image_size = int(ldm_cfg.get("data", {}).get("image_size", 128))
    resolution = max(image_size // 8, 1)
    return latent_channels, resolution


def generate_images(
    vae_cfg: dict[str, Any],
    ldm_cfg: dict[str, Any],
    vae_model: ConvVAE,
    ldm_model: LatentDiffusionModel,
    diffusion: GaussianDiffusion,
    text_encoder_dir: Path,
    latent_stats_path: Path,
    data_path: Path,
    output_dir: Path,
    device: torch.device,
    manifest: dict[str, Any],
) -> None:
    records = load_captions_from_json(data_path)
    if not records:
        raise RuntimeError(f"No captions found in {data_path}")

    latent_stats = load_latent_stats_from_path(latent_stats_path)
    mean, std = latent_stats_to_tensors(latent_stats, device)

    inference_cfg = ldm_cfg.get("inference", {})
    batch_size = int(manifest.get("generate_batch_size", inference_cfg.get("batch_size", 16)))
    guidance_scale = float(manifest.get("guidance_scale", inference_cfg.get("guidance_scale", 4.0)))
    text_batch_size = int(manifest.get("text_batch_size", ldm_cfg.get("cache", {}).get("text_batch_size", 64)))
    latent_channels, latent_resolution = infer_latent_shape(ldm_cfg)
    fixed_embedding_path = Path(manifest.get("fixed_text_embedding", latent_stats_path.parent / "fixed_text_embedding.pt"))
    if fixed_embedding_path.exists():
        embeddings = load_fixed_text_embeddings(records, fixed_embedding_path)
    else:
        embeddings = precompute_text_embeddings(records, text_encoder_dir, device, text_batch_size)

    active_batch_size = max(1, batch_size)
    start = 0
    with torch.no_grad(), tqdm(total=len(records), desc="Generating") as progress:
        while start < len(records):
            chunk_size = min(active_batch_size, len(records) - start)
            batch_records = records[start : start + chunk_size]
            try:
                context = torch.stack(
                    [embeddings[record["caption_key"]] for record in batch_records],
                    dim=0,
                ).to(device)
                normalized_latents = diffusion.sample(
                    ldm_model,
                    shape=(len(batch_records), latent_channels, latent_resolution, latent_resolution),
                    context=context,
                    null_context=ldm_model.expanded_null_context(len(batch_records)),
                    device=device,
                    guidance_scale=guidance_scale,
                )
                images = vae_model.decode(normalized_latents * std + mean)
            except RuntimeError as error:
                if "out of memory" not in str(error).lower() or active_batch_size == 1:
                    raise
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                active_batch_size = max(1, active_batch_size // 2)
                print(f"CUDA OOM during generation; retrying with batch size {active_batch_size}", flush=True)
                continue

            for record, image in zip(batch_records, images):
                filename = f"{Path(record['image_filename']).stem}.png"
                save_tensor_image(image, output_dir / filename)
            start += chunk_size
            progress.update(chunk_size)


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir).resolve()
    data_path = Path(args.data_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(model_dir)

    vae_config_path = resolve_artifact(
        model_dir,
        manifest,
        "vae_config",
        ["vae_config.yaml", "configs/vae.yaml"],
        "VAE config",
    )
    ldm_config_path = resolve_artifact(
        model_dir,
        manifest,
        "ldm_config",
        ["ldm_config.yaml", "configs/ldm.yaml"],
        "LDM config",
    )
    vae_checkpoint_path = resolve_artifact(
        model_dir,
        manifest,
        "vae_checkpoint",
        ["vae.pt", "vae.pth", "checkpoints/vae.pt", "checkpoints/vae.pth"],
        "VAE checkpoint",
    )

    vae_cfg = load_config(vae_config_path)
    ldm_cfg = load_config(ldm_config_path)

    seed = int(manifest.get("seed", ldm_cfg.get("inference", {}).get("seed", 0)))
    set_seed(seed)
    device = get_device()
    vae_model = load_vae_model(vae_cfg, vae_checkpoint_path, device)

    if args.task == "reconstruct":
        batch_size = int(manifest.get("reconstruct_batch_size", vae_cfg.get("data", {}).get("val_batch_size", 32)))
        reconstruct_images(vae_cfg, vae_model, data_path, output_dir, batch_size, device)
        return

    ldm_checkpoint_path = resolve_artifact(
        model_dir,
        manifest,
        "ldm_checkpoint",
        ["ldm.pt", "ldm.pth", "checkpoints/ldm.pt", "checkpoints/ldm.pth"],
        "LDM checkpoint",
    )
    latent_stats_path = resolve_artifact(
        model_dir,
        manifest,
        "latent_stats",
        ["latent_stats.json", "latent_stats.pth", "latent_stats.pt", "checkpoints/latent_stats.json"],
        "latent statistics file",
    )
    text_encoder_dir = resolve_artifact(
        model_dir,
        manifest,
        "text_encoder",
        ["text_encoder", "clip_text_encoder", "clip-vit-base-patch32"],
        "text encoder directory",
        expect_dir=True,
    )

    ldm_model = load_ldm_model(ldm_cfg, ldm_checkpoint_path, device)
    diffusion_cfg = ldm_cfg.get("diffusion", {})
    diffusion = GaussianDiffusion(
        timesteps=int(diffusion_cfg.get("timesteps", 500)),
        schedule=str(diffusion_cfg.get("schedule", "cosine")),
    ).to(device)
    generate_images(
        vae_cfg=vae_cfg,
        ldm_cfg=ldm_cfg,
        vae_model=vae_model,
        ldm_model=ldm_model,
        diffusion=diffusion,
        text_encoder_dir=text_encoder_dir,
        latent_stats_path=latent_stats_path,
        data_path=data_path,
        output_dir=output_dir,
        device=device,
        manifest=manifest,
    )


if __name__ == "__main__":
    main()
