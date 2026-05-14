from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from contrastive.data import build_eval_transform
from contrastive.factory import load_clip_full_model_bundle


class RetrievalImageDataset(Dataset):
    def __init__(self, examples: list[dict[str, Any]], data_path: Path, transform) -> None:
        self.examples = examples
        self.data_path = data_path
        self.transform = transform

    def __len__(self) -> int:
        return len(self.examples)

    def _resolve_path(self, example: dict[str, Any]) -> Path:
        raw_path = example.get("image_path") or example.get("image_filename")
        if raw_path is None:
            raise KeyError("Each retrieval item must contain image_path or image_filename")
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
                    parent / "images" / filename,
                    parent / "Part_A" / split / "images" / filename,
                    parent / "Part_A" / "val" / "images" / filename,
                    parent / "Part_A" / "train" / "images" / filename,
                    parent / "Part_Aa" / "Clevr_official" / "images" / split / filename,
                    parent / "Clevr_official" / "images" / split / filename,
                ]
            )
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"Could not resolve image path for {example.get('image_filename', raw_path)}")

    def __getitem__(self, index: int) -> torch.Tensor:
        image = Image.open(self._resolve_path(self.examples[index])).convert("RGB")
        return self.transform(image)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CLIP-style image-text retrieval inference")
    parser.add_argument("--model_type", choices=["clip"], required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--retrieval_task", choices=["i2t", "t2i"], required=True)
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
    return {"clip_full_bundle": "clip_full_model_best.pt"}


def normalize_model_dir(model_dir: Path) -> Path:
    if (model_dir / "parta_models.json").exists():
        return model_dir
    nested = model_dir / "models"
    if (nested / "parta_models.json").exists():
        return nested
    return model_dir


def load_caption_examples(data_path: Path) -> list[dict[str, Any]]:
    payload = load_json(data_path)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and "examples" in payload:
        return payload["examples"]
    raise ValueError("Caption input must be a list or a JSON object with examples")


def encode_images(
    model,
    bundle: dict[str, Any],
    examples: list[dict[str, Any]],
    data_path: Path,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> torch.Tensor:
    config = bundle["config"]
    transform = build_eval_transform(
        image_size=config["model"]["vision"]["image_size"],
        mean=config["normalization"]["mean"],
        std=config["normalization"]["std"],
    )
    loader = DataLoader(
        RetrievalImageDataset(examples, data_path, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    embeddings = []
    with torch.no_grad():
        for images in loader:
            images = images.to(device, non_blocking=True)
            embeddings.append(model.encode_image(images)["embeddings"].detach().cpu())
    return torch.cat(embeddings, dim=0)


def encode_texts(model, tokenizer, captions: list[str], batch_size: int, device: torch.device) -> torch.Tensor:
    embeddings = []
    with torch.no_grad():
        for start in range(0, len(captions), batch_size):
            batch = captions[start : start + batch_size]
            encoded = [tokenizer.encode(caption) for caption in batch]
            input_ids = torch.tensor([item.input_ids for item in encoded], dtype=torch.long, device=device)
            attention_mask = torch.tensor([item.attention_mask for item in encoded], dtype=torch.long, device=device)
            embeddings.append(model.encode_text(input_ids, attention_mask)["embeddings"].detach().cpu())
    return torch.cat(embeddings, dim=0)


def topk_indices(queries: torch.Tensor, candidates: torch.Tensor, k: int, batch_size: int) -> list[list[int]]:
    output: list[list[int]] = []
    for start in range(0, queries.size(0), batch_size):
        similarity = queries[start : start + batch_size] @ candidates.t()
        output.extend(similarity.topk(k, dim=1).indices.tolist())
    return output


def main() -> None:
    args = parse_args()
    model_dir = normalize_model_dir(Path(args.model_dir))
    data_path = Path(args.data_path)
    output_file = Path(args.output_file)

    index = load_model_index(model_dir)
    bundle_path = model_dir / index.get("clip_full_bundle", "clip_full_model_best.pt")
    if not bundle_path.exists():
        raise FileNotFoundError(f"Missing CLIP bundle: {bundle_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, bundle, tokenizer = load_clip_full_model_bundle(bundle_path, device=device)
    examples = load_caption_examples(data_path)
    captions = [example["caption"] for example in examples]
    filenames = [example.get("image_filename", Path(example.get("image_path", str(i))).name) for i, example in enumerate(examples)]

    image_embeddings = encode_images(model, bundle, examples, data_path, args.batch_size, args.num_workers, device)
    text_embeddings = encode_texts(model, tokenizer, captions, args.batch_size, device)

    if args.retrieval_task == "i2t":
        top = topk_indices(image_embeddings, text_embeddings, k=3, batch_size=args.batch_size)
        predictions = {filename: [captions[index] for index in indices] for filename, indices in zip(filenames, top)}
    else:
        top = topk_indices(text_embeddings, image_embeddings, k=3, batch_size=args.batch_size)
        predictions = {caption: [filenames[index] for index in indices] for caption, indices in zip(captions, top)}

    save_json(output_file, predictions)


if __name__ == "__main__":
    main()
