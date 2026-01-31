# models/blocks/multihead.py
import torch
import torch.nn as nn


class MultiHeadWrapper(nn.Module):
    def __init__(self, attention, num_heads):
        super().__init__()
        self.attention = attention
        self.num_heads = num_heads

    def forward(self, Q, K, V, mask=None):
        """
        Q, K, V: [B, T, D]
        """
        B, T, D = Q.shape
        H = self.num_heads
        head_dim = D // H

        # reshape to [B, H, T, head_dim]
        Q = Q.view(B, T, H, head_dim).transpose(1, 2)
        K = K.view(B, T, H, head_dim).transpose(1, 2)
        V = V.view(B, T, H, head_dim).transpose(1, 2)

        out = self.attention(Q, K, V, mask)

        # back to [B, T, D]
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return out
