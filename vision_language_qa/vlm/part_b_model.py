from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .factory import load_vision_encoder_bundle


IGNORE_INDEX = -100


class ReverseBottleneckProjector(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.net(tokens)


@dataclass
class PreparedInputs:
    inputs_embeds: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor | None


def build_chat_prompt(tokenizer: Any, user_text: str) -> str:
    messages = [{"role": "user", "content": user_text}]
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass
    return f"User: {user_text}\nAssistant:"


class PartBVLM(nn.Module):
    def __init__(
        self,
        vision_bundle_path: str,
        llm: nn.Module,
        vision_token_dim: int = 384,
        llm_hidden_size: int = 2560,
        projector_hidden_size: int = 2560,
        projector_dropout: float = 0.0,
        freeze_vision: bool = True,
    ) -> None:
        super().__init__()
        device = next(llm.parameters()).device
        self.vision_encoder, self.vision_bundle = load_vision_encoder_bundle(vision_bundle_path, device=device)
        self.llm = llm
        self.projector = ReverseBottleneckProjector(
            input_dim=vision_token_dim,
            hidden_dim=projector_hidden_size,
            output_dim=llm_hidden_size,
            dropout=projector_dropout,
        ).to(device)
        if freeze_vision:
            self.freeze_vision()

    @property
    def device(self) -> torch.device:
        return next(self.projector.parameters()).device

    def freeze_vision(self) -> None:
        self.vision_encoder.eval()
        for param in self.vision_encoder.parameters():
            param.requires_grad_(False)

    def freeze_llm(self) -> None:
        for param in self.llm.parameters():
            param.requires_grad_(False)

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [param for param in self.parameters() if param.requires_grad]

    def encode_image_tokens(self, images: torch.Tensor) -> torch.Tensor:
        images = images.to(self.device, non_blocking=True)
        self.vision_encoder.eval()
        with torch.no_grad():
            patch_tokens = self.vision_encoder.forward_features(images)["patches"]
        return self.projector(patch_tokens)

    def _embed_token_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.llm.get_input_embeddings()(input_ids)

    def _tokenize(self, tokenizer: Any, text: str) -> list[int]:
        encoded = tokenizer(text, add_special_tokens=False)
        return list(encoded["input_ids"])

    def _target_ids(self, tokenizer: Any, target: str) -> list[int]:
        eos = tokenizer.eos_token or ""
        return self._tokenize(tokenizer, target + eos)

    def prepare_inputs(
        self,
        images: torch.Tensor,
        prompts: list[str],
        targets: list[str] | None,
        tokenizer: Any,
        max_length: int,
    ) -> PreparedInputs:
        visual_embeds = self.encode_image_tokens(images)
        embed_layer = self.llm.get_input_embeddings()
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        pieces: list[torch.Tensor] = []
        labels: list[torch.Tensor] = []

        for row, prompt in enumerate(prompts):
            prompt_text = build_chat_prompt(tokenizer, prompt)
            prompt_ids = torch.tensor(self._tokenize(tokenizer, prompt_text), dtype=torch.long, device=self.device)
            prompt_embeds = embed_layer(prompt_ids).unsqueeze(0)
            image_embeds = visual_embeds[row : row + 1].to(dtype=prompt_embeds.dtype)
            if targets is None:
                sequence = torch.cat([prompt_embeds, image_embeds], dim=1).squeeze(0)
                pieces.append(sequence)
                continue

            target_ids_list = self._target_ids(tokenizer, targets[row])
            prefix_len = prompt_embeds.size(1) + image_embeds.size(1)
            available_target_len = max_length - prefix_len
            if available_target_len <= 0:
                raise ValueError(
                    f"max_length={max_length} is too small for prompt + {image_embeds.size(1)} visual tokens"
                )
            target_ids_list = target_ids_list[:available_target_len]
            target_ids = torch.tensor(target_ids_list, dtype=torch.long, device=self.device)
            target_embeds = embed_layer(target_ids).unsqueeze(0)
            sequence = torch.cat([prompt_embeds, image_embeds, target_embeds], dim=1).squeeze(0)
            label = torch.full((sequence.size(0),), IGNORE_INDEX, dtype=torch.long, device=self.device)
            label[prefix_len:] = target_ids
            pieces.append(sequence)
            labels.append(label)

        batch_size = len(pieces)
        hidden_size = pieces[0].size(-1)
        max_seq_len = min(max(piece.size(0) for piece in pieces), max_length)
        inputs_embeds = torch.zeros(batch_size, max_seq_len, hidden_size, dtype=pieces[0].dtype, device=self.device)
        attention_mask = torch.zeros(batch_size, max_seq_len, dtype=torch.long, device=self.device)
        padded_labels = None
        if targets is not None:
            padded_labels = torch.full((batch_size, max_seq_len), IGNORE_INDEX, dtype=torch.long, device=self.device)

        for row, piece in enumerate(pieces):
            length = min(piece.size(0), max_seq_len)
            inputs_embeds[row, :length] = piece[:length]
            attention_mask[row, :length] = 1
            if padded_labels is not None:
                padded_labels[row, :length] = labels[row][:length]

        if pad_id is None:
            pad_id = 0
        return PreparedInputs(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=padded_labels)

    def _left_pad_for_generation(self, prepared: PreparedInputs) -> PreparedInputs:
        batch_size, max_seq_len, hidden_size = prepared.inputs_embeds.shape
        left_embeds = torch.zeros(
            batch_size,
            max_seq_len,
            hidden_size,
            dtype=prepared.inputs_embeds.dtype,
            device=prepared.inputs_embeds.device,
        )
        left_mask = torch.zeros_like(prepared.attention_mask)
        lengths = prepared.attention_mask.sum(dim=1).to(dtype=torch.long)
        for row, length in enumerate(lengths.tolist()):
            if length <= 0:
                continue
            start = max_seq_len - length
            left_embeds[row, start:] = prepared.inputs_embeds[row, :length]
            left_mask[row, start:] = 1
        return PreparedInputs(inputs_embeds=left_embeds, attention_mask=left_mask, labels=None)

    def forward(
        self,
        images: torch.Tensor,
        prompts: list[str],
        targets: list[str],
        tokenizer: Any,
        max_length: int,
    ) -> Any:
        prepared = self.prepare_inputs(images, prompts, targets, tokenizer, max_length)
        return self.llm(
            inputs_embeds=prepared.inputs_embeds,
            attention_mask=prepared.attention_mask,
            labels=prepared.labels,
        )

    @torch.no_grad()
    def generate_text(
        self,
        images: torch.Tensor,
        prompts: list[str],
        tokenizer: Any,
        max_length: int,
        max_new_tokens: int,
    ) -> list[str]:
        was_training = self.training
        self.eval()
        prepared = self.prepare_inputs(images, prompts, None, tokenizer, max_length)
        prepared = self._left_pad_for_generation(prepared)
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        if pad_id is None:
            pad_id = 0
        input_ids = torch.full(
            prepared.attention_mask.shape,
            fill_value=pad_id,
            dtype=torch.long,
            device=prepared.attention_mask.device,
        )
        outputs = self.llm.generate(
            input_ids=input_ids,
            inputs_embeds=prepared.inputs_embeds,
            attention_mask=prepared.attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=pad_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        prefix_len = prepared.inputs_embeds.size(1)
        if outputs.size(1) > prefix_len:
            outputs = outputs[:, prefix_len:]
        texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        if was_training:
            self.train()
        return [text.strip() for text in texts]


def trainable_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    return {name: param.detach().cpu() for name, param in module.named_parameters() if param.requires_grad}


def load_trainable_state_dict(module: nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
    named_params = dict(module.named_parameters())
    missing = []
    for name, value in state_dict.items():
        param = named_params.get(name)
        if param is None:
            missing.append(name)
            continue
        param.data.copy_(value.to(device=param.device, dtype=param.dtype))
    if missing:
        raise ValueError(f"Checkpoint contains unknown trainable parameters: {missing[:5]}")
