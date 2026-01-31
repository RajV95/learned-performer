# models/attention/standard.py
import torch
import torch.nn as nn
import math

class StandardAttention(nn.Module):
    def __init__(self, head_dim: int):
        super().__init__()
        self.scale = 1.0 / math.sqrt(head_dim)

    def forward(self, Q, K, V, mask=None):
        """
        Q, K, V: [B, H, N, D]
        """
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)
        return out
