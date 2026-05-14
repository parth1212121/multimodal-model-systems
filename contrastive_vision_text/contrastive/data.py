from __future__ import annotations

import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import BatchSampler, Dataset

from .augmentations import (
    CenterCrop,
    ColorJitter,
    Compose,
    GaussianBlur,
    Normalize,
    RandomApply,
    RandomGrayscale,
    RandomHorizontalFlip,
    RandomResizedCrop,
    RandomSolarize,
    Resize,
    ToTensor,
)
from .tokenization import SimpleTokenizer
from .utils import load_json


CAPTION_PREFIX_PATTERN = re.compile(r"an image with\s+(\d+)\s+objects?\s*:\s*(.+)", re.IGNORECASE)
OBJECT_PATTERN = re.compile(
    r"(\d+)\s+(small|large)\s+([a-z]+)\s+(metal|rubber)\s+([a-z]+)",
    re.IGNORECASE,
)


def build_clip_train_transform(
    image_size: int,
    mean: list[float],
    std: list[float],
    scale: tuple[float, float],
    flip_p: float,
) -> T.Compose:
    return Compose(
        [
            RandomResizedCrop(image_size, scale=scale),
            RandomHorizontalFlip(p=flip_p),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ]
    )


def build_eval_transform(image_size: int, mean: list[float], std: list[float]) -> Compose:
    return Compose(
        [
            Resize(image_size + 32),
            CenterCrop(image_size),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ]
    )


def discover_part_a_split(root: str | Path, split: str) -> tuple[Path, Path]:
    split_dir = Path(root) / "Part_A" / split
    images_dir = split_dir / "images"
    candidates = sorted(split_dir.glob("*captions.json"))
    if not candidates:
        raise FileNotFoundError(f"No caption manifest found under {split_dir}")
    return images_dir, candidates[0]


def load_captions_from_manifest(path: str | Path) -> list[dict[str, Any]]:
    payload = load_json(path)
    if not isinstance(payload, list):
        raise ValueError(f"Expected caption manifest list at {path}")
    return payload


class CLEVRCaptionDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str,
        transform: Compose,
        tokenizer: SimpleTokenizer | None = None,
    ) -> None:
        self.images_dir, self.caption_path = discover_part_a_split(root, split)
        self.transform = transform
        self.tokenizer = tokenizer
        self.samples = load_captions_from_manifest(self.caption_path)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        image = Image.open(self.images_dir / sample["image_filename"]).convert("RGB")
        item = {
            "image": self.transform(image),
            "caption": sample["caption"],
            "image_filename": sample["image_filename"],
            "image_index": sample["image_index"],
            "object_count": sample.get("object_count"),
        }
        if self.tokenizer is not None:
            encoded = self.tokenizer.encode(sample["caption"])
            item["input_ids"] = torch.tensor(encoded.input_ids, dtype=torch.long)
            item["attention_mask"] = torch.tensor(encoded.attention_mask, dtype=torch.long)
        return item


class CLEVRImageDataset(Dataset):
    def __init__(self, root: str | Path, split: str, transform: Any) -> None:
        self.images_dir, self.caption_path = discover_part_a_split(root, split)
        self.transform = transform
        self.samples = load_captions_from_manifest(self.caption_path)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        image = Image.open(self.images_dir / sample["image_filename"]).convert("RGB")
        transformed = self.transform(image)
        return {
            "views": transformed,
            "image_filename": sample["image_filename"],
            "image_index": sample["image_index"],
        }


class UniqueCaptionBatchSampler(BatchSampler):
    def __init__(self, captions: list[str], batch_size: int, drop_last: bool = False) -> None:
        self.captions = captions
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.groups: dict[str, list[int]] = defaultdict(list)
        for index, caption in enumerate(captions):
            self.groups[caption].append(index)
        self.group_keys = list(self.groups.keys())

    def __iter__(self):
        groups = {key: indices[:] for key, indices in self.groups.items()}
        for indices in groups.values():
            random.shuffle(indices)
        random.shuffle(self.group_keys)

        linearized: list[int] = []
        active = True
        while active:
            active = False
            round_indices: list[int] = []
            random.shuffle(self.group_keys)
            for key in self.group_keys:
                indices = groups[key]
                if indices:
                    round_indices.append(indices.pop())
                    active = True
            linearized.extend(round_indices)

        batch: list[int] = []
        for index in linearized:
            batch.append(index)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        size = len(self.captions)
        if self.drop_last:
            return size // self.batch_size
        return (size + self.batch_size - 1) // self.batch_size


def clevr_coarse_caption_key(caption: str) -> str:
    normalized = " ".join(caption.strip().lower().split()).rstrip(".")
    match = CAPTION_PREFIX_PATTERN.fullmatch(normalized)
    if match is None:
        return normalized

    colors: set[str] = set()
    shapes: set[str] = set()
    materials: set[str] = set()
    sizes: set[str] = set()
    for part in [piece.strip().rstrip(".") for piece in match.group(2).split(",") if piece.strip()]:
        object_match = OBJECT_PATTERN.fullmatch(part)
        if object_match is None:
            return normalized
        _multiplicity, size, color, material, shape = object_match.groups()
        colors.add(color)
        shapes.add(shape[:-1] if shape.endswith("s") else shape)
        materials.add(material)
        sizes.add(size)

    return "|".join(
        [
            f"count_{match.group(1)}",
            "colors_" + ",".join(sorted(colors)),
            "shapes_" + ",".join(sorted(shapes)),
            "materials_" + ",".join(sorted(materials)),
            "sizes_" + ",".join(sorted(sizes)),
        ]
    )


class DiverseCaptionBatchSampler(UniqueCaptionBatchSampler):
    def __init__(self, captions: list[str], batch_size: int, drop_last: bool = False) -> None:
        super().__init__(
            captions=[clevr_coarse_caption_key(caption) for caption in captions],
            batch_size=batch_size,
            drop_last=drop_last,
        )


class CLIPCollator:
    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "images": torch.stack([item["image"] for item in batch], dim=0),
            "input_ids": torch.stack([item["input_ids"] for item in batch], dim=0),
            "attention_mask": torch.stack([item["attention_mask"] for item in batch], dim=0),
            "captions": [item["caption"] for item in batch],
            "image_filenames": [item["image_filename"] for item in batch],
        }


class DINOCollator:
    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        views_per_sample = [item["views"] for item in batch]
        num_views = len(views_per_sample[0])
        stacked_views = []
        for view_index in range(num_views):
            stacked_views.append(torch.stack([views[view_index] for views in views_per_sample], dim=0))
        return {
            "views": stacked_views,
            "image_filenames": [item["image_filename"] for item in batch],
        }


class DINOAugmentation:
    def __init__(
        self,
        global_crop_scale: tuple[float, float],
        local_crop_scale: tuple[float, float],
        local_crop_size: int,
        local_crops_number: int,
        mean: list[float],
        std: list[float],
        global_size: int = 224,
    ) -> None:
        normalize = Compose([ToTensor(), Normalize(mean=mean, std=std)])
        color_jitter = RandomApply(
            [ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
            p=0.8,
        )
        blur = GaussianBlur(radius_min=0.1, radius_max=2.0)
        solarize = RandomSolarize(threshold=128, p=0.2)

        self.global_transform_1 = Compose(
            [
                RandomResizedCrop(global_size, scale=global_crop_scale),
                RandomHorizontalFlip(p=0.5),
                color_jitter,
                RandomGrayscale(p=0.2),
                blur,
                normalize,
            ]
        )
        self.global_transform_2 = Compose(
            [
                RandomResizedCrop(global_size, scale=global_crop_scale),
                RandomHorizontalFlip(p=0.5),
                color_jitter,
                RandomGrayscale(p=0.2),
                blur,
                solarize,
                normalize,
            ]
        )
        self.local_transform = Compose(
            [
                RandomResizedCrop(local_crop_size, scale=local_crop_scale),
                RandomHorizontalFlip(p=0.5),
                color_jitter,
                RandomGrayscale(p=0.2),
                blur,
                normalize,
            ]
        )
        self.local_crops_number = local_crops_number

    def __call__(self, image: Image.Image) -> list[torch.Tensor]:
        crops = [self.global_transform_1(image), self.global_transform_2(image)]
        for _ in range(self.local_crops_number):
            crops.append(self.local_transform(image))
        return crops
