# evaluation/sanity.py
import torch
from models.attention.standard import StandardAttention

def run_sanity():
    B, H, N, D = 2, 4, 16, 32
    Q = torch.randn(B, H, N, D)
    K = torch.randn(B, H, N, D)
    V = torch.randn(B, H, N, D)

    attn = StandardAttention(D)
    out = attn(Q, K, V)

    assert out.shape == (B, H, N, D)
    print("Sanity check passed")

if __name__ == "__main__":
    run_sanity()
