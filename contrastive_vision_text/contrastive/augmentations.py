from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    if array.ndim == 2:
        array = np.expand_dims(array, axis=-1)
    array = np.transpose(array, (2, 0, 1))
    return torch.from_numpy(array)


class Compose:
    def __init__(self, transforms: list) -> None:
        self.transforms = transforms

    def __call__(self, image):
        for transform in self.transforms:
            image = transform(image)
        return image


class RandomApply:
    def __init__(self, transforms: list, p: float) -> None:
        self.transforms = transforms
        self.p = p

    def __call__(self, image):
        if random.random() >= self.p:
            return image
        for transform in self.transforms:
            image = transform(image)
        return image


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() < self.p:
            return ImageOps.mirror(image)
        return image


class Resize:
    def __init__(self, size: int) -> None:
        self.size = size

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        if width < height:
            new_width = self.size
            new_height = int(round(self.size * height / width))
        else:
            new_height = self.size
            new_width = int(round(self.size * width / height))
        return image.resize((new_width, new_height), Image.BICUBIC)


class CenterCrop:
    def __init__(self, size: int) -> None:
        self.size = size

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        left = max(0, int(round((width - self.size) / 2.0)))
        top = max(0, int(round((height - self.size) / 2.0)))
        return image.crop((left, top, left + self.size, top + self.size))


class RandomResizedCrop:
    def __init__(
        self,
        size: int,
        scale: tuple[float, float],
        ratio: tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0),
    ) -> None:
        self.size = size
        self.scale = scale
        self.ratio = ratio

    def _sample_params(self, width: int, height: int) -> tuple[int, int, int, int]:
        area = width * height
        log_ratio = (math.log(self.ratio[0]), math.log(self.ratio[1]))
        for _ in range(10):
            target_area = area * random.uniform(self.scale[0], self.scale[1])
            aspect_ratio = math.exp(random.uniform(log_ratio[0], log_ratio[1]))
            crop_w = int(round(math.sqrt(target_area * aspect_ratio)))
            crop_h = int(round(math.sqrt(target_area / aspect_ratio)))
            if 0 < crop_w <= width and 0 < crop_h <= height:
                top = random.randint(0, height - crop_h)
                left = random.randint(0, width - crop_w)
                return left, top, crop_w, crop_h

        min_side = min(width, height)
        crop_w = min_side
        crop_h = min_side
        top = (height - crop_h) // 2
        left = (width - crop_w) // 2
        return left, top, crop_w, crop_h

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        left, top, crop_w, crop_h = self._sample_params(width, height)
        cropped = image.crop((left, top, left + crop_w, top + crop_h))
        return cropped.resize((self.size, self.size), Image.BICUBIC)


class ColorJitter:
    def __init__(self, brightness: float, contrast: float, saturation: float, hue: float) -> None:
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, image: Image.Image) -> Image.Image:
        transforms = []
        if self.brightness > 0:
            factor = random.uniform(max(0, 1 - self.brightness), 1 + self.brightness)
            transforms.append(lambda img: ImageEnhance.Brightness(img).enhance(factor))
        if self.contrast > 0:
            factor = random.uniform(max(0, 1 - self.contrast), 1 + self.contrast)
            transforms.append(lambda img: ImageEnhance.Contrast(img).enhance(factor))
        if self.saturation > 0:
            factor = random.uniform(max(0, 1 - self.saturation), 1 + self.saturation)
            transforms.append(lambda img: ImageEnhance.Color(img).enhance(factor))
        random.shuffle(transforms)
        for transform in transforms:
            image = transform(image)
        if self.hue > 0:
            hue_delta = random.uniform(-self.hue, self.hue)
            hsv = np.array(image.convert("HSV"), dtype=np.uint8)
            hsv[..., 0] = (hsv[..., 0].astype(np.int16) + int(hue_delta * 255)) % 255
            image = Image.fromarray(hsv, mode="HSV").convert("RGB")
        return image


class RandomGrayscale:
    def __init__(self, p: float) -> None:
        self.p = p

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() < self.p:
            return ImageOps.grayscale(image).convert("RGB")
        return image


class GaussianBlur:
    def __init__(self, radius_min: float, radius_max: float) -> None:
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, image: Image.Image) -> Image.Image:
        radius = random.uniform(self.radius_min, self.radius_max)
        return image.filter(ImageFilter.GaussianBlur(radius=radius))


class RandomSolarize:
    def __init__(self, threshold: int = 128, p: float = 0.2) -> None:
        self.threshold = threshold
        self.p = p

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() < self.p:
            return ImageOps.solarize(image, threshold=self.threshold)
        return image


class ToTensor:
    def __call__(self, image: Image.Image) -> torch.Tensor:
        return pil_to_tensor(image)


@dataclass
class Normalize:
    mean: list[float]
    std: list[float]

    def __post_init__(self) -> None:
        self.mean_tensor = torch.tensor(self.mean, dtype=torch.float32).view(-1, 1, 1)
        self.std_tensor = torch.tensor(self.std, dtype=torch.float32).view(-1, 1, 1)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return (tensor - self.mean_tensor) / self.std_tensor
