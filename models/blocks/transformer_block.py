# models/blocks/transformer_block.py
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, attention, d_model, d_ff=2048, dropout=0.1):
        super().__init__()
        self.attn = attention
        self.norm1 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Pre-norm attention
        x_norm = self.norm1(x)
        attn_out = self.attn(x_norm, x_norm, x_norm, mask)
        x = x + self.dropout(attn_out)

        # Feed-forward
        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm)
        x = x + self.dropout(ffn_out)

        return x
