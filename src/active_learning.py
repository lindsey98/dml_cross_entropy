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
from src.data import MetricLoaders

from tqdm import tqdm
import logging
import yaml
import faiss
import numpy as np


def selector(all_gallery_features: torch.Tensor, all_gallery_labels: torch.Tensor, 
             all_pool_features: torch.Tensor, all_pool_labels: torch.Tensor, 
             strategy:str, selected_k:int) -> (np.ndarray, Any):
    '''
       AL unlabelled selector 
    '''
    # empirical estimate of p(y) estimated from training set
    (unique, counts) = np.unique(all_gallery_labels, return_counts=True)

    M = torch.zeros(len(unique), len(all_gallery_features))
    M[np.asarray([np.where(unique==x.item())[0].item() for x in all_gallery_labels]), 
                  torch.arange(all_gallery_features.shape[0])] = 1 # one-hot map

    centroids = torch.mm(M, all_gallery_features) # mean for each label
    centroids = F.normalize(centroids, p=2, dim=1).numpy() # normalize
    
    # Max
    if strategy == 'max':
        query_features = F.normalize(all_pool_features, p=2, dim=1).numpy()
        dist = np.dot(query_features, centroids.T) # dot product
        query2gallery = dist.max(1) # take maximum
        selected_indices = query2gallery.argsort()[:selected_k] # those data points that are farther away
        return (selected_indices, query2gallery)
        
    # Margin to centroids
    elif strategy == 'margin':
        query_features = F.normalize(all_pool_features, p=2, dim=1).numpy()
        dist = np.dot(query_features, centroids.T)
        ind = np.argsort(dist, axis=1)
        sorted_dist = np.take_along_axis(dist, ind, axis=1)
        margin_query2gallery = sorted_dist[:, -1] - sorted_dist[:, -2]
        selected_indices = margin_query2gallery.argsort()[:selected_k] # those data points that are near boundary
        return (selected_indices, margin_query2gallery)
    
    elif strategy == 'mixed':
        query_features = F.normalize(all_pool_features, p=2, dim=1).numpy()
        dist = np.dot(query_features, centroids.T)
        
        ind = np.argsort(dist, axis=1)
        sorted_dist = np.take_along_axis(dist, ind, axis=1)
        margin_query2gallery = sorted_dist[:, -1] - sorted_dist[:, -2]
        query2gallery = sorted_dist[:, -1]
        
        # first sort by margin, then sort by confidence to strike a balance between selecting FN and FP
        sorted_index_bymargin = margin_query2gallery.argsort()[:2*selected_k] # ascending order, select lowest 2*selected_k 
        sorted_index_byconf = query2gallery[sorted_index_bymargin].argsort() # nested sort, ascending order
        selected_indices = sorted_index_bymargin[sorted_index_byconf[:int(selected_k/2)]] + \
                           sorted_index_bymargin[sorted_index_byconf[-int(selected_k/2):]]
        return (selected_indices, None)
    
    elif strategy == 'entropy':
        query_features = F.normalize(all_pool_features, p=2, dim=1).numpy()
        dist = np.dot(query_features, centroids.T)
        prob = F.softmax(torch.from_numpy(dist), dim=1)
        entropy = torch.sum(-prob*torch.log2(prob), dim=1).numpy()
        selected_indices = entropy.argsort()[::-1][:selected_k] # those data points that are uncertain
        return (selected_indices, entropy)
    
    elif strategy == 'random':
        selected_indices = np.random.choice(len(all_pool_features), selected_k, replace=False)
        return (selected_indices, None)
    
    else:
        raise NotImplementedError
        
