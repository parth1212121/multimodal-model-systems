from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .utils import load_json, save_json


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+|[:,.;]")
CAPTION_PREFIX_PATTERN = re.compile(r"an image with\s+(\d+)\s+objects?\s*:\s*(.+)", re.IGNORECASE)
OBJECT_PATTERN = re.compile(
    r"(\d+)\s+(small|large)\s+([a-z]+)\s+(metal|rubber)\s+([a-z]+)",
    re.IGNORECASE,
)


@dataclass
class EncodedText:
    input_ids: list[int]
    attention_mask: list[int]


class SimpleTokenizer:
    PAD = "<pad>"
    BOS = "<bos>"
    EOS = "<eos>"
    UNK = "<unk>"

    def __init__(
        self,
        token_to_id: dict[str, int],
        context_length: int,
        lowercase: bool = True,
        preprocess_mode: str = "raw",
    ) -> None:
        self.token_to_id = token_to_id
        self.id_to_token = {value: key for key, value in token_to_id.items()}
        self.context_length = context_length
        self.lowercase = lowercase
        self.preprocess_mode = preprocess_mode

    @classmethod
    def build(
        cls,
        captions: list[str],
        context_length: int,
        min_freq: int = 1,
        max_vocab_size: int | None = None,
        lowercase: bool = True,
        preprocess_mode: str = "raw",
    ) -> "SimpleTokenizer":
        counter: Counter[str] = Counter()
        for caption in captions:
            normalized = cls._normalize_text(caption, lowercase=lowercase, preprocess_mode=preprocess_mode)
            counter.update(cls._tokenize(normalized))

        vocab = [cls.PAD, cls.BOS, cls.EOS, cls.UNK]
        items = [(token, freq) for token, freq in counter.items() if freq >= min_freq]
        items.sort(key=lambda item: (-item[1], item[0]))
        if max_vocab_size is not None:
            items = items[: max(0, max_vocab_size - len(vocab))]
        vocab.extend(token for token, _ in items)
        token_to_id = {token: idx for idx, token in enumerate(vocab)}
        return cls(
            token_to_id=token_to_id,
            context_length=context_length,
            lowercase=lowercase,
            preprocess_mode=preprocess_mode,
        )

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return TOKEN_PATTERN.findall(text)

    @staticmethod
    def _singularize_shape(shape: str) -> str:
        shape = shape.lower()
        if shape.endswith("s"):
            return shape[:-1]
        return shape

    @classmethod
    def _parse_clevr_caption(cls, text: str, lowercase: bool) -> tuple[str, list[tuple[str, str, str, str, str]]] | None:
        raw = " ".join(text.strip().split())
        normalized = raw.lower() if lowercase else raw
        match = CAPTION_PREFIX_PATTERN.fullmatch(normalized.rstrip("."))
        if match is None:
            return None
        object_count = match.group(1)
        object_specs = []
        for part in [piece.strip() for piece in match.group(2).split(",") if piece.strip()]:
            obj_match = OBJECT_PATTERN.fullmatch(part.rstrip("."))
            if obj_match is None:
                return None
            multiplicity, size, color, material, shape = obj_match.groups()
            object_specs.append(
                (
                    multiplicity,
                    size.lower(),
                    color.lower(),
                    material.lower(),
                    cls._singularize_shape(shape),
                )
            )
        return object_count, object_specs

    @classmethod
    def _normalize_clevr_compact(cls, text: str, lowercase: bool) -> str:
        parsed = cls._parse_clevr_caption(text, lowercase=lowercase)
        if parsed is None:
            normalized = " ".join(text.strip().split())
            return normalized.lower() if lowercase else normalized
        object_count, object_specs = parsed
        compact_specs = [
            f"obj_{multiplicity}_{size}_{color}_{material}_{shape}"
            for multiplicity, size, color, material, shape in object_specs
        ]
        compact_specs.sort()
        return "scene " + " ".join([f"count_{object_count}", *compact_specs])

    @classmethod
    def _normalize_clevr_compact_ordered(cls, text: str, lowercase: bool) -> str:
        parsed = cls._parse_clevr_caption(text, lowercase=lowercase)
        if parsed is None:
            normalized = " ".join(text.strip().split())
            return normalized.lower() if lowercase else normalized
        object_count, object_specs = parsed
        compact_specs = [
            f"obj_{idx}_{multiplicity}_{size}_{color}_{material}_{shape}"
            for idx, (multiplicity, size, color, material, shape) in enumerate(object_specs)
        ]
        return "scene " + " ".join([f"count_{object_count}", *compact_specs])

    @classmethod
    def _normalize_text(cls, text: str, lowercase: bool, preprocess_mode: str) -> str:
        if preprocess_mode == "clevr_compact":
            return cls._normalize_clevr_compact(text, lowercase=lowercase)
        if preprocess_mode == "clevr_compact_ordered":
            return cls._normalize_clevr_compact_ordered(text, lowercase=lowercase)
        normalized = " ".join(text.strip().split())
        if lowercase:
            normalized = normalized.lower()
        return normalized

    @property
    def pad_id(self) -> int:
        return self.token_to_id[self.PAD]

    @property
    def eos_id(self) -> int:
        return self.token_to_id[self.EOS]

    def normalize_text(self, text: str) -> str:
        return self._normalize_text(text, lowercase=self.lowercase, preprocess_mode=self.preprocess_mode)

    def encode(self, text: str) -> EncodedText:
        tokens = [self.BOS]
        tokens.extend(self._tokenize(self.normalize_text(text)))
        tokens.append(self.EOS)
        ids = [self.token_to_id.get(token, self.token_to_id[self.UNK]) for token in tokens]
        ids = ids[: self.context_length]
        if ids[-1] != self.eos_id:
            ids[-1] = self.eos_id
        attention_mask = [1] * len(ids)
        if len(ids) < self.context_length:
            pad_length = self.context_length - len(ids)
            ids.extend([self.pad_id] * pad_length)
            attention_mask.extend([0] * pad_length)
        return EncodedText(input_ids=ids, attention_mask=attention_mask)

    def to_dict(self) -> dict:
        return {
            "token_to_id": self.token_to_id,
            "context_length": self.context_length,
            "lowercase": self.lowercase,
            "preprocess_mode": self.preprocess_mode,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "SimpleTokenizer":
        return cls(
            token_to_id=payload["token_to_id"],
            context_length=payload["context_length"],
            lowercase=payload.get("lowercase", True),
            preprocess_mode=payload.get("preprocess_mode", "raw"),
        )

    def save(self, path: str | Path) -> None:
        save_json(path, self.to_dict())

    @classmethod
    def load(cls, path: str | Path) -> "SimpleTokenizer":
        return cls.from_dict(load_json(path))
