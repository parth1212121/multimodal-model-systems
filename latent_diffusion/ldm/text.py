from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
from transformers import CLIPTextModel, CLIPTokenizer

from ldm.data import caption_to_key


class FrozenClipTextEncoder:
    def __init__(self, model_name: str, device: torch.device) -> None:
        self.device = device
        self.model_name = model_name
        self.tokenizer = _load_pretrained_local_first(CLIPTokenizer, model_name)
        # Force safetensors loading for environments where torch.load safety defaults are strict,
        # while recent transformers versions reject pickle-based .bin weights.
        self.model = _load_pretrained_local_first(
            CLIPTextModel,
            model_name,
            use_safetensors=True,
        ).to(device)
        self.max_length = _resolve_text_max_length(self.tokenizer, self.model)
        self.tokenizer.model_max_length = self.max_length
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False

    @torch.no_grad()
    def encode_texts(self, texts: Iterable[str], batch_size: int = 64) -> dict[str, torch.Tensor]:
        text_list = list(texts)
        embeddings: dict[str, torch.Tensor] = {}
        for start in range(0, len(text_list), batch_size):
            batch_texts = text_list[start : start + batch_size]
            encoded = self.tokenizer(
                batch_texts,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            outputs = self.model(**encoded)
            hidden_states = outputs.last_hidden_state.detach().cpu()
            for text, embedding in zip(batch_texts, hidden_states):
                embeddings[caption_to_key(text)] = embedding
        return embeddings


def load_or_initialize_text_cache(path: str | Path, model_name: str) -> dict:
    cache_path = Path(path)
    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu")
        if payload.get("model_name") != model_name:
            raise ValueError(f"Text cache at {cache_path} was created for {payload.get('model_name')}, expected {model_name}")
        return payload
    return {"model_name": model_name, "embeddings": {}, "captions": {}}


def _load_pretrained_local_first(factory, model_name: str, **kwargs):
    try:
        return factory.from_pretrained(model_name, local_files_only=True, **kwargs)
    except OSError:
        return factory.from_pretrained(model_name, **kwargs)


def _resolve_text_max_length(tokenizer: CLIPTokenizer, model: CLIPTextModel, fallback: int = 77) -> int:
    model_limit = int(getattr(model.config, "max_position_embeddings", fallback))
    tokenizer_limit = getattr(tokenizer, "model_max_length", fallback)
    try:
        tokenizer_limit = int(tokenizer_limit)
    except (TypeError, ValueError, OverflowError):
        tokenizer_limit = fallback
    if tokenizer_limit <= 0 or tokenizer_limit > 10000:
        tokenizer_limit = fallback
    return min(model_limit, tokenizer_limit)
