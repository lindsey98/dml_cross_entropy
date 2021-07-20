import math
import os
import tempfile
from copy import deepcopy
from functools import partial
from pprint import pprint

import sacred
import torch
import torch.nn as nn
from sacred import SETTINGS
from sacred.utils import apply_backspaces_and_linefeeds
from torch.backends import cudnn
from torch.optim import SGD, lr_scheduler
from visdom_logger import VisdomLogger

from models.vae_ingredient import model_ingredient, get_model
from utils import state_dict_to_cpu, SmoothCrossEntropy, VAELoss, AELoss
from utils.data.dataset_ingredient import data_ingredient, get_loaders
from utils.vae_train import train, evaluate

ex = sacred.Experiment('Autoencoder', ingredients=[data_ingredient, model_ingredient])
# Filter backspaces and linefeeds
SETTINGS.CAPTURE_MODE = 'sys'
ex.captured_out_filter = apply_backspaces_and_linefeeds


@ex.config
def config():
    epochs = 100
    lr = 0.02
    momentum = 0.
    nesterov = False
    weight_decay = 5e-4
    scheduler = 'warmcos'

    visdom_port = None
    visdom_freq = 20
    cpu = False  # Force training on CPU
    cudnn_flag = 'benchmark'
    temp_dir = 'checkpoints'

    no_bias_decay = True
    label_smoothing = 0.1


@ex.capture
def get_optimizer_scheduler(parameters, loader_length, epochs, lr, momentum, nesterov, weight_decay, scheduler,
                            lr_step=None):
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


@ex.automain
def main(epochs, cpu, cudnn_flag, visdom_port, visdom_freq, temp_dir, seed, no_bias_decay, label_smoothing):
    
    os.makedirs(temp_dir, exist_ok=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() and not cpu else 'cpu')
    callback = VisdomLogger(port=visdom_port) if visdom_port else None
    if cudnn_flag == 'deterministic':
        setattr(cudnn, cudnn_flag, True)

    torch.manual_seed(seed)
    loaders, recall_ks = get_loaders()

    torch.manual_seed(seed)
    model = get_model()
#     class_loss = VAELoss()
    class_loss = AELoss()

    model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    parameters = []
    if no_bias_decay:
        parameters.append({'params': [par for par in model.parameters() if par.dim() != 1]})
        parameters.append({'params': [par for par in model.parameters() if par.dim() == 1], 'weight_decay': 0})
    else:
        parameters.append({'params': model.parameters()})
    optimizer, scheduler = get_optimizer_scheduler(parameters=parameters, loader_length=len(loaders.train))

    # setup partial function to simplify call
    eval_function = partial(evaluate, model=model, query_loader=loaders.query, class_loss=class_loss)

    # setup best validation logger
    metrics = eval_function()
    if callback is not None:
        callback.scalars(['reconstruction'], 0, [metrics.loss], title='Reconstruction Loss')
    pprint(metrics.loss)
    best_val = (0, metrics.loss, deepcopy(model.state_dict()))

    torch.manual_seed(seed)
    for epoch in range(epochs):
        if cudnn_flag == 'benchmark':
            setattr(cudnn, cudnn_flag, True)

        train(model=model, loader=loaders.train, class_loss=class_loss, optimizer=optimizer,
              scheduler=scheduler, epoch=epoch, callback=callback, freq=visdom_freq, ex=ex)

        # validation
        if cudnn_flag == 'benchmark':
            setattr(cudnn, cudnn_flag, False)
        metrics = eval_function()
        print('Validation [{:03d}]'.format(epoch)), pprint(metrics.loss)
        ex.log_scalar('val.reconstruction', metrics.loss, step=epoch + 1)

        if callback is not None:
            callback.scalars(['reconstruction'], epoch + 1,
                             [metrics.loss], title='Reconstruction loss')

        # save model dict if the chosen validation metric is better
#         if metrics.loss <= best_val[1]:
        print('Update best model at epoch {}'.format(epoch+1)) # always update
        best_val = (epoch + 1, metrics.loss, deepcopy(model.state_dict()))

    # logging
    ex.info['reconstruction'] = best_val[1]

    # saving
    save_name = os.path.join(temp_dir, '{}_{}.pt'.format(ex.current_run.config['vae']['arch'], ex.current_run.config['dataset']['name']))
    torch.save(state_dict_to_cpu(best_val[2]), save_name)
    ex.add_artifact(save_name)

    if callback is not None:
        save_name = os.path.join(temp_dir, 'visdom_data_encoder.pt')
        callback.save(save_name)
        ex.add_artifact(save_name)

    return best_val[1]

