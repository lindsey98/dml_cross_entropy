import math
import os
import tempfile
from copy import deepcopy
from functools import partial
from pprint import pprint

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.backends import cudnn
from torch.optim import SGD, lr_scheduler

from utils import state_dict_to_cpu, SmoothCrossEntropy
from torch.utils.data import DataLoader
from typing import NamedTuple, Optional, Dict, List, Any
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torchvision import transforms

from utils.metrics import AverageMeter

from tqdm import tqdm
import logging
import yaml
import faiss

def get_optimizer_scheduler(cfg: Dict, parameters: Dict, loader_length: int) -> (Optimizer, _LRScheduler):
    '''
        Initialize optimizer and lr_scheduler
    '''
    epochs = cfg['Train']['epochs']
    lr = cfg['Train']['lr']
    momentum = cfg['Train']['momentum']
    nesterov = cfg['Train']['nesterov']
    weight_decay = cfg['Train']['weight_decay']
    scheduler = cfg['Train']['scheduler']
    lr_step = cfg['Train']['lr_step']
    
    optimizer = SGD(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay,
                    nesterov=True if nesterov and momentum else False)
    if epochs == 0:
        scheduler = None
    elif scheduler == 'cos':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * loader_length, eta_min=0)
    elif scheduler == 'warmcos':
        warm_cosine = lambda i: min((i + 1) / 100, (1 + math.cos(math.pi * i / (epochs * loader_length))) / 2)
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_cosine)
    elif scheduler == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, lr_step * loader_length)
    elif scheduler == 'warmstep':
        warm_step = lambda i: min((i + 1) / 100, 1) * 0.1 ** (i // (lr_step * loader_length))
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_step)
    else:
        scheduler = lr_scheduler.StepLR(optimizer, epochs * loader_length)
        
    return optimizer, scheduler

def training(model: nn.Module, loader: DataLoader,
          labeldict: Dict, class_loss: nn.Module,
          optimizer: Optimizer, scheduler: _LRScheduler,
          epoch: int, 
          logger) -> nn.Module:
    '''
        Train model for one epoch
    '''
    model.train()
    device = next(model.parameters()).device
    print('Device used: ', device)
    to_device = lambda x: x.to(device, non_blocking=True)
    loader_length = len(loader)
    train_losses = AverageMeter(device=device, length=loader_length)

    pbar = tqdm(loader, ncols=80, desc='Training   [{:03d}]'.format(epoch))
    for i, (batch, labels, indices) in enumerate(pbar):
        labels = torch.tensor([labeldict[x] for x in labels.numpy()])
#         print(labels)
        batch, labels, indices = map(to_device, (batch, labels, indices))
        logits, features = model(batch)
        loss = class_loss(logits, labels).mean()
#         print(loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        train_losses.append(loss)
    
    logger.info('Epoch {} train.loss = {}'.format(epoch, train_losses.last_avg))
        


