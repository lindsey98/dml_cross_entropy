from functools import partial
from typing import NamedTuple, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from sacred import Experiment
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from visdom_logger import VisdomLogger
from utils.metrics import AverageMeter, recall_at_ks
import numpy as np

    
def train(model: nn.Module,
          loader: DataLoader,
          class_loss: nn.Module,
          optimizer: Optimizer,
          scheduler: _LRScheduler,
          epoch: int,
          callback: VisdomLogger,
          freq: int,
          ex: Experiment = None) -> None:
    
    model.train()
    device = next(model.parameters()).device
    to_device = lambda x: x.to(device, non_blocking=True)
    loader_length = len(loader)
    train_losses = AverageMeter(device=device, length=loader_length)

    pbar = tqdm(loader, ncols=80, desc='Training   [{:03d}]'.format(epoch))
    for i, (batch, labels, indices) in enumerate(pbar):
        batch, labels, indices = map(to_device, (batch, labels, indices))
#         x_hat, _, mean, log_var = model(batch)
#         loss = class_loss(x_hat, batch, mean, log_var)['loss'].mean()
        z, _, x_hat = model(batch)
        loss = class_loss(x_hat, batch)['loss'].mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        train_losses.append(loss)

        if callback is not None and not (i + 1) % freq:
            step = epoch + i / loader_length
            callback.scalar('train.loss', step, train_losses.last_avg, title='Train Losses')

    if ex is not None:
        for i, loss in enumerate(train_losses.values_list):
            step = epoch + i / loader_length
            ex.log_scalar('train.loss', loss, step=step)


class _Metrics(NamedTuple):
    loss: float
    

def evaluate(model: nn.Module,
             query_loader: DataLoader,
             class_loss: nn.Module) -> _Metrics:
    
    model.eval()
    device = next(model.parameters()).device
    to_device = lambda x: x.to(device, non_blocking=True)
    all_query_means = []
    all_query_logvar = []
    all_reconstruct_losses = []
    all_predictions = []

    with torch.no_grad():
        for batch, labels, _ in tqdm(query_loader, desc='Extracting query features', leave=False, ncols=80):
            batch, labels = map(to_device, (batch, labels))
#             x_hat, _, mean, log_var = model(batch)
#             reconstruct_loss = class_loss(x_hat, batch,mean, log_var)['reconstruct'].item()
            z, _, x_hat = model(batch)
            reconstruct_loss = class_loss(x_hat, batch)['reconstruct'].mean().item()
            all_reconstruct_losses.append(reconstruct_loss)

        torch.cuda.empty_cache()

        loss = np.mean(all_reconstruct_losses)

    return _Metrics(loss=loss)

