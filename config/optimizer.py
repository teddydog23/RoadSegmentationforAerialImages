import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

def get_optimizer(model, lr=1e-4, weight_decay=1e-5):
    return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

def get_scheduler(optimizer, mode='min', patience=5, factor=0.5, min_lr=1e-6, verbose=True):
    return ReduceLROnPlateau(
        optimizer,
        mode=mode,
        factor=factor,
        patience=patience,
        min_lr=min_lr,
        verbose=verbose
    )
