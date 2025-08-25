# -*- coding: utf-8 -*-
"""
通用训练 / 评估循环 —— 自动兼容 forward 输出为 tensor 或 (tensor, ...)
"""
import torch, torch.nn.functional as F


def _to_logits(out):
    """兼容 tuple / list 输出"""
    return out[0] if isinstance(out, (list, tuple)) else out


def train_epoch(model, loader, optimizer, device):
    model.train()
    epoch_loss, n, correct = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = _to_logits(model(x))
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * len(y)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item(); n += len(y)
    return epoch_loss / n, correct / n


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    loss, n, correct = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = _to_logits(model(x))
        loss += F.cross_entropy(logits, y, reduction='sum').item()
        pred = logits.argmax(1)
        correct += (pred == y).sum().item(); n += len(y)
    return loss / n, correct / n