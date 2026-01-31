# evaluation/gpt_sanity.py
import torch

from models.gpt import GPT
from models.attention.standard import StandardAttention


def run_gpt_sanity():
    vocab_size = 100
    model = GPT(
        vocab_size=vocab_size,
        d_model=128,
        num_layers=2,
        num_heads=4,
        max_seq_len=32,
        attention_cls=StandardAttention,
    )

    x = torch.randint(0, vocab_size, (2, 32))
    loss = model(x, x)

    print("Loss:", loss.item())
    print("GPT sanity check passed")


if __name__ == "__main__":
    run_gpt_sanity()
