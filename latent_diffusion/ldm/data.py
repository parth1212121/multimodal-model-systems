from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def caption_to_key(caption: str) -> str:
    return hashlib.sha1(caption.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class CaptionExample:
    image_path: Path
    image_filename: str
    image_id: str
    caption_text: str
    caption_key: str


def _candidate_split_dirs(root: Path, split: str) -> list[Path]:
    return [
        root / split,
        root / split.lower(),
        root / split.upper(),
        root / split.capitalize(),
    ]


def _discover_split_dir(part_a_root: str | Path, split: str) -> Path:
    root = Path(part_a_root)
    for candidate in _candidate_split_dirs(root, split):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find split directory for '{split}' under {root}")


def _discover_image_dir(split_dir: Path) -> Path:
    candidates = [split_dir / "images", split_dir / "image", split_dir / "imgs", split_dir]
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        for extension in IMAGE_EXTENSIONS:
            if next(candidate.rglob(f"*{extension}"), None) is not None:
                return candidate
    raise FileNotFoundError(f"Could not locate an image directory inside {split_dir}")


def _discover_caption_file(split_dir: Path, split: str) -> Path:
    candidates = [
        split_dir / f"clevr_{split}_caption.json",
        split_dir / f"CLEVR_{split}_caption.json",
        split_dir / f"{split}_captions.json",
        split_dir / "captions.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    matches = sorted(split_dir.glob("*caption*.json"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Could not find a caption json under {split_dir}")


def _build_image_index(image_dir: Path) -> dict[str, Path]:
    image_index: dict[str, Path] = {}
    for path in image_dir.rglob("*"):
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        image_index[path.name] = path
        image_index[path.stem] = path
    return image_index


def _extract_caption_text(record: dict[str, Any]) -> str:
    for key in ("caption", "caption_text", "text", "sentence", "raw"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    captions = record.get("captions")
    if isinstance(captions, list) and captions:
        if isinstance(captions[0], str):
            return captions[0].strip()
        if isinstance(captions[0], dict):
            return _extract_caption_text(captions[0])

    raise KeyError(f"Could not find caption text in record keys={list(record.keys())}")


def _extract_image_name(record: dict[str, Any]) -> str:
    for key in ("image_filename", "filename", "file_name", "image", "image_name", "img", "image_path"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            image_name = Path(value).name
            if Path(image_name).suffix:
                return image_name
            return f"{image_name}.png"

    image_index = record.get("image_index")
    if image_index is not None:
        return f"{int(image_index):06d}.png"

    raise KeyError(f"Could not find image filename in record keys={list(record.keys())}")


def _normalize_annotation_list(raw_data: Any) -> list[dict[str, Any]]:
    if isinstance(raw_data, list):
        return [item for item in raw_data if isinstance(item, dict)]

    if isinstance(raw_data, dict):
        for key in ("annotations", "captions", "data", "items", "records"):
            value = raw_data.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]

        if raw_data and all(isinstance(key, str) and isinstance(value, str) for key, value in raw_data.items()):
            return [{"image_filename": key, "caption": value} for key, value in raw_data.items()]

    raise ValueError("Unsupported caption json structure")


def load_caption_examples(part_a_root: str | Path, split: str) -> list[CaptionExample]:
    split_dir = _discover_split_dir(part_a_root, split)
    image_dir = _discover_image_dir(split_dir)
    caption_file = _discover_caption_file(split_dir, split)
    image_index = _build_image_index(image_dir)

    with caption_file.open("r", encoding="utf-8") as handle:
        raw_data = json.load(handle)

    examples: list[CaptionExample] = []
    for record in _normalize_annotation_list(raw_data):
        caption_text = _extract_caption_text(record)
        image_name = _extract_image_name(record)
        image_path = image_index.get(image_name) or image_index.get(Path(image_name).stem)
        if image_path is None:
            continue
        examples.append(
            CaptionExample(
                image_path=image_path,
                image_filename=image_path.name,
                image_id=image_path.stem,
                caption_text=caption_text,
                caption_key=caption_to_key(caption_text),
            )
        )

    if not examples:
        raise RuntimeError(f"No image-caption pairs found for split '{split}' in {split_dir}")
    return examples


def load_captions_from_json(caption_json: str | Path) -> list[dict[str, str]]:
    with Path(caption_json).open("r", encoding="utf-8") as handle:
        raw_data = json.load(handle)
    records = _normalize_annotation_list(raw_data)
    caption_records: list[dict[str, str]] = []
    for record in records:
        caption_text = _extract_caption_text(record)
        image_name = _extract_image_name(record)
        caption_records.append(
            {
                "caption_text": caption_text,
                "caption_key": caption_to_key(caption_text),
                "image_filename": Path(image_name).name,
                "image_id": Path(image_name).stem,
            }
        )
    return caption_records


class ClevrPartADataset(Dataset):
    def __init__(self, part_a_root: str | Path, split: str, image_size: int = 128) -> None:
        self.examples = load_caption_examples(part_a_root, split)
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Lambda(lambda tensor: tensor * 2.0 - 1.0),
            ]
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = self.examples[index]
        image = Image.open(example.image_path).convert("RGB")
        return {
            "image_tensor": self.transform(image),
            "caption_text": example.caption_text,
            "caption_key": example.caption_key,
            "image_id": example.image_id,
            "image_filename": example.image_filename,
        }


class LatentCacheDataset(Dataset):
    def __init__(self, latent_cache_path: str | Path, text_cache_path: str | Path, latent_stats: dict[str, Any] | None = None) -> None:
        payload = torch.load(Path(latent_cache_path), map_location="cpu")
        text_cache = torch.load(Path(text_cache_path), map_location="cpu")
        self.latents = payload["latents"].float()
        self.image_ids = payload["image_ids"]
        self.image_filenames = payload["image_filenames"]
        self.caption_keys = payload["caption_keys"]
        self.caption_texts = payload["caption_texts"]
        self.text_embeddings = text_cache["embeddings"]
        self.stats = None
        if latent_stats is not None:
            mean = torch.tensor(latent_stats["mean"], dtype=torch.float32).view(1, -1, 1, 1)
            std = torch.tensor(latent_stats["std"], dtype=torch.float32).view(1, -1, 1, 1)
            self.stats = {"mean": mean, "std": std}

    def __len__(self) -> int:
        return self.latents.shape[0]

    def __getitem__(self, index: int) -> dict[str, Any]:
        latent = self.latents[index]
        if self.stats is not None:
            latent = (latent - self.stats["mean"][0]) / self.stats["std"][0]
        caption_key = self.caption_keys[index]
        return {
            "latent": latent,
            "text_embedding": self.text_embeddings[caption_key].float(),
            "caption_key": caption_key,
            "caption_text": self.caption_texts[index],
            "image_id": self.image_ids[index],
            "image_filename": self.image_filenames[index],
        }


def build_reference_directory(dataset: ClevrPartADataset, output_dir: str | Path) -> Path:
    reference_dir = Path(output_dir)
    reference_dir.mkdir(parents=True, exist_ok=True)
    for example in dataset.examples:
        destination = reference_dir / example.image_filename
        if destination.exists():
            continue
        image = Image.open(example.image_path).convert("RGB")
        tensor = dataset.transform(image)
        array = ((tensor.clamp(-1.0, 1.0) + 1.0) * 127.5).round().to(torch.uint8).permute(1, 2, 0).numpy()
        Image.fromarray(array).save(destination)
    return reference_dir
