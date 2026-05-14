from __future__ import annotations

import torch
from torch import nn

from .vit import TransformerEncoderBlock


class TextTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        hidden_dim: int,
        depth: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.context_length = context_length
        self.hidden_dim = hidden_dim
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, context_length, hidden_dim))
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [TransformerEncoderBlock(hidden_dim, num_heads, mlp_dim, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self._init_parameters()

    def _init_parameters(self) -> None:
        nn.init.trunc_normal_(self.token_embedding.weight, std=0.02)
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

    def _build_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        x = self.token_embedding(input_ids) + self.pos_embedding[:, : input_ids.size(1)]
        x = self.dropout(x)
        causal_mask = self._build_causal_mask(input_ids.size(1), input_ids.device)
        key_padding_mask = attention_mask == 0
        for block in self.blocks:
            x = block(x, attn_mask=causal_mask, key_padding_mask=key_padding_mask)
        x = self.norm(x)
        last_token_indices = attention_mask.sum(dim=1) - 1
        pooled = x[torch.arange(x.size(0), device=x.device), last_token_indices]
        return {"pooled": pooled, "tokens": x}
