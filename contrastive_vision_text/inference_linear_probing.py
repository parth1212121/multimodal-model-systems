from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from contrastive.data import build_eval_transform
from contrastive.factory import load_vision_encoder_bundle


POOLING_ALIASES = {
    "cls": ("cls", "cls"),
    "gap": ("mean_patches", "gap"),
    "mean_patches": ("mean_patches", "gap"),
}


class ImagePathDataset(Dataset):
    def __init__(self, examples: list[dict[str, Any]], data_path: Path, transform) -> None:
        self.examples = examples
        self.data_path = data_path
        self.transform = transform

    def __len__(self) -> int:
        return len(self.examples)

    def _resolve_path(self, example: dict[str, Any]) -> Path:
        raw_path = example.get("image_path") or example.get("image_filename")
        if raw_path is None:
            raise KeyError("Each example must contain image_path or image_filename")
        path = Path(raw_path)
        if path.is_absolute() and path.exists():
            return path
        filename = example.get("image_filename", path.name)
        split = example.get("split", "val")
        candidates = [
            self.data_path.parent / path,
            self.data_path.parent / filename,
            Path(filename),
        ]
        for parent in [self.data_path.parent, *self.data_path.parents]:
            candidates.extend(
                [
                    parent / "Clevr_official" / "images" / split / filename,
                    parent / "Part_Aa" / "Clevr_official" / "images" / split / filename,
                    parent / "Part_A" / split / "images" / filename,
                    parent / "Part_A" / "val" / "images" / filename,
                    parent / "Part_A" / "train" / "images" / filename,
                ]
            )
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"Could not resolve image path for {example.get('image_filename', raw_path)}")

    def __getitem__(self, index: int) -> torch.Tensor:
        image = Image.open(self._resolve_path(self.examples[index])).convert("RGB")
        return self.transform(image)


class LinearProbeHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.linear(inputs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Linear-probe inference for vision encoders")
    parser.add_argument("--model_type", choices=["clip", "dino_student", "dino_teacher"], required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--pooling_type", choices=["cls", "gap", "mean_patches"], required=True)
    parser.add_argument("--probe_task", choices=["count", "color"], required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=16)
    return parser.parse_args()


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load_model_index(model_dir: Path) -> dict[str, Any]:
    index_path = model_dir / "parta_models.json"
    if index_path.exists():
        return load_json(index_path)
    return {
        "vision_bundles": {
            "clip": "clip_vision_encoder_best.pt",
            "dino_student": "dino_student_encoder_best.pt",
            "dino_teacher": "dino_teacher_encoder_best.pt",
        },
        "linear_probe_dir": "linear_probes",
    }


def normalize_model_dir(model_dir: Path) -> Path:
    if (model_dir / "parta_models.json").exists():
        return model_dir
    nested = model_dir / "models"
    if (nested / "parta_models.json").exists():
        return nested
    return model_dir


def resolve_model_file(model_dir: Path, relative_path: str) -> Path:
    path = model_dir / relative_path
    if not path.exists():
        raise FileNotFoundError(f"Missing model artifact: {path}")
    return path


def load_examples(data_path: Path) -> list[dict[str, Any]]:
    payload = load_json(data_path)
    if isinstance(payload, dict) and "examples" in payload:
        return payload["examples"]
    if isinstance(payload, list):
        return payload
    raise ValueError("Probe input must be a JSON object with examples or a list of examples")


def extract_features(
    model,
    bundle: dict[str, Any],
    examples: list[dict[str, Any]],
    data_path: Path,
    pooling: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> torch.Tensor:
    transform = build_eval_transform(
        image_size=bundle["config"]["image_size"],
        mean=bundle["normalization"]["mean"],
        std=bundle["normalization"]["std"],
    )
    loader = DataLoader(
        ImagePathDataset(examples, data_path, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    features = []
    with torch.no_grad():
        for images in loader:
            images = images.to(device, non_blocking=True)
            features.append(model(images, pooling=pooling).detach().cpu())
    return torch.cat(features, dim=0)


def main() -> None:
    args = parse_args()
    model_dir = normalize_model_dir(Path(args.model_dir))
    data_path = Path(args.data_path)
    output_file = Path(args.output_file)
    pooling, pooling_slug = POOLING_ALIASES[args.pooling_type]

    index = load_model_index(model_dir)
    bundle_path = resolve_model_file(model_dir, index["vision_bundles"][args.model_type])
    probe_path = resolve_model_file(
        model_dir,
        f"{index.get('linear_probe_dir', 'linear_probes')}/{args.model_type}_{args.probe_task}_{pooling_slug}.pt",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, bundle = load_vision_encoder_bundle(bundle_path, device=device)
    probe_bundle = torch.load(probe_path, map_location="cpu", weights_only=False)
    probe = LinearProbeHead(probe_bundle["input_dim"], probe_bundle["output_dim"]).to(device)
    probe.load_state_dict(probe_bundle["state_dict"])
    probe.eval()

    examples = load_examples(data_path)
    features = extract_features(
        model,
        bundle,
        examples,
        data_path,
        pooling,
        args.batch_size,
        args.num_workers,
        device,
    ).float()
    mean = probe_bundle["feature_mean"].float()
    std = probe_bundle["feature_std"].float().clamp_min(1e-6)
    features = (features - mean) / std

    filenames = [example.get("image_filename", Path(example.get("image_path", str(i))).name) for i, example in enumerate(examples)]
    with torch.no_grad():
        logits = probe(features.to(device)).cpu()

    if args.probe_task == "count":
        classes = probe_bundle.get("classes", list(range(probe_bundle["output_dim"])))
        pred_indices = logits.argmax(dim=1).tolist()
        predictions = {filename: int(classes[index]) for filename, index in zip(filenames, pred_indices)}
    else:
        color_vocab = probe_bundle["color_vocab"]
        threshold = float(probe_bundle.get("threshold", 0.5))
        probs = torch.sigmoid(logits)
        positive = probs >= threshold
        predictions = {}
        for filename, mask, row in zip(filenames, positive.tolist(), probs):
            colors = [color for color, keep in zip(color_vocab, mask) if keep]
            if not colors:
                colors = [color_vocab[int(row.argmax().item())]]
            predictions[filename] = colors

    save_json(output_file, predictions)


if __name__ == "__main__":
    main()
