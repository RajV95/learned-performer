# models/gpt.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.blocks.multihead import MultiHeadWrapper
from models.blocks.transformer_block import TransformerBlock


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        max_seq_len: int,
        attention_cls,
        d_ff: int = 2048,
        dropout: float = 0.1,
    ):
        """
        attention_cls: a callable that returns an attention module
                       e.g. StandardAttention(head_dim)
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        assert (
            d_model % num_heads == 0
        ), "d_model must be divisible by num_heads"

        # Embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        # Transformer blocks
        self.blocks = nn.ModuleList()
        for _ in range(num_layers):
            attention = attention_cls(self.head_dim)
            block = TransformerBlock(
                attention=MultiHeadWrapper(attention, num_heads),
                d_model=d_model,
                d_ff=d_ff,
                dropout=dropout,
            )
            self.blocks.append(block)

        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, targets=None):
        """
        input_ids: [B, T]
        targets:   [B, T]
        """
        B, T = input_ids.shape
        device = input_ids.device

        positions = torch.arange(0, T, device=device)
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        x = self.dropout(x)

        # causal mask
        mask = torch.tril(torch.ones(T, T, device=device)).unsqueeze(0).unsqueeze(0)

        for block in self.blocks:
            x = block(x, mask=mask)

        x = self.norm(x)
        logits = self.lm_head(x)

        if targets is None:
            return logits

        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-100,
        )

        return loss
