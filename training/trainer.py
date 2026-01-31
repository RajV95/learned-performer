# training/trainer.py
import torch
from utils.logger import log

def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0

    for step, batch in enumerate(dataloader):
        input_ids, targets = batch
        input_ids = input_ids.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        loss = model(input_ids, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if step % 100 == 0:
            log(f"step {step} | loss {loss.item():.4f}")

    return total_loss / len(dataloader)
