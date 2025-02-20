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

from src.utils import state_dict_to_cpu, SmoothCrossEntropy
from torch.utils.data import DataLoader
from typing import NamedTuple, Optional, Dict, List, Any, Union
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torchvision import transforms

from src.metrics import AverageMeter
from src.data import MetricLoaders

from tqdm import tqdm
import logging
import yaml
import faiss
import numpy as np


def selector(all_gallery_features: torch.Tensor, all_gallery_labels: torch.Tensor, 
             all_pool_features: torch.Tensor, all_pool_labels: torch.Tensor, 
             strategy:str, 
             selected_k:int,
             threshold: float) -> (Union[List, np.ndarray], Any):
    '''
       AL unlabelled selector 
    '''
    all_gallery_features = F.normalize(all_gallery_features, p=2, dim=1)
    
    # empirical estimate of p(y) estimated from training set
    (unique, counts) = np.unique(all_gallery_labels, return_counts=True)

    M = torch.zeros(len(unique), len(all_gallery_features))
    M[np.asarray([np.where(unique==x.item())[0].item() for x in all_gallery_labels]), 
                  torch.arange(all_gallery_features.shape[0])] = 1 # one-hot map

    centroids = torch.mm(M, all_gallery_features) # mean for each label
    centroids = F.normalize(centroids, p=2, dim=1).numpy() # normalize
    
    # Max
    if strategy == 'max':
         # compute distance to centroids
        query_features = F.normalize(all_pool_features, p=2, dim=1).numpy()
        dist = np.dot(query_features, centroids.T) # dot product
        query2gallery = dist.max(1) # take maximum
        selected_indices = query2gallery.argsort()[:selected_k] # those data points that are farther away
        return selected_indices, query2gallery
        
    # Margin to centroids
    elif strategy == 'margin':
        query_features = F.normalize(all_pool_features, p=2, dim=1).numpy()
        dist = np.dot(query_features, centroids.T)
        ind = np.argsort(dist, axis=1)
        sorted_dist = np.take_along_axis(dist, ind, axis=1)
        margin_query2gallery = sorted_dist[:, -1] - sorted_dist[:, -2]
        selected_indices = margin_query2gallery.argsort()[:selected_k] # those data points that are near boundary
        return selected_indices, margin_query2gallery
    
    elif strategy == 'mixed':
        query_features = F.normalize(all_pool_features, p=2, dim=1).numpy()
        gallery_features = F.normalize(all_gallery_features, p=2, dim=1).numpy()
        
        # compute margin
        dist2c = np.dot(query_features, centroids.T)
        ind = np.argsort(dist2c, axis=1)
        sorted_dist = np.take_along_axis(dist2c, ind, axis=1)
        margin_query2gallery = sorted_dist[:, -1] - sorted_dist[:, -2]
        
        # compute distance to all gallery data (tried centroids but it couldnt cover all FNs)
        dist2g = np.dot(query_features, gallery_features.T)
        ind = np.argsort(dist2g, axis=1)
        sorted_dist = np.take_along_axis(dist2g, ind, axis=1)
        query2gallery = sorted_dist[:, -1]
        
        if np.sum(query2gallery >= threshold) == 0: # no negative already
            selected_indices = margin_query2gallery.argsort()[:selected_k] # purely select by margin
        else:
            # half selected by confidence, then half by margin to strike a balance between selecting FN and FP
            sorted_index_byconf = query2gallery.argsort()[:int(selected_k/2)]
            sorted_index_bymargin = margin_query2gallery.argsort()[:int(selected_k/2)] 
            selected_indices = list(set(sorted_index_byconf).union(set(sorted_index_bymargin)))
        
        return selected_indices, None
    
    elif strategy == 'entropy':
        query_features = F.normalize(all_pool_features, p=2, dim=1).numpy()
        dist = np.dot(query_features, centroids.T)
        prob = F.softmax(torch.from_numpy(dist), dim=1)
        entropy = torch.sum(-prob*torch.log2(prob), dim=1).numpy()
        selected_indices = entropy.argsort()[::-1][:selected_k] # those data points that are uncertain
        return selected_indices, entropy
    
    elif strategy == 'topology':
        intra_temp = 1000 # temperature smoothing 
        inter_temp = 0.1 # temperature smoothing 
        
        query_features = F.normalize(all_pool_features, p=2, dim=1)
        dist = torch.mm(query_features, all_gallery_features.T) # of shape N_pool x N_gallery

        all_intra_sum = torch.tensor([]) # weighted average of gallery set from a specific class of shape N_pool x C
        for cls in unique: # for each class in gallery set
            clsind = np.where(all_gallery_labels == cls)[0] # get corresponding indices
            intra_cls_prob = F.softmax(dist[:, clsind] / intra_temp, dim=1) # intra-class weights
            intra_cls_sum = torch.sum(intra_cls_prob * dist[:, clsind], dim=1) # intra-class weighted avg of similarity
            all_intra_sum = torch.cat((all_intra_sum, intra_cls_sum.unsqueeze(-1)), dim=-1) 

        inter_cls_prob = F.softmax(dist / inter_temp, dim=1)
        all_inter_sum = torch.sum(inter_cls_prob * dist, dim=1) # weighted avg to all gallery set, of shape N_pool

        derivative_proxy = torch.max(all_intra_sum, dim=1)[0] - all_inter_sum
        selected_indices = derivative_proxy.argsort()[:selected_k] # those data points that are uncertain
        return selected_indices, derivative_proxy
    
    elif strategy == 'random':
        selected_indices = np.random.choice(len(all_pool_features), selected_k, replace=False)
        return selected_indices, None
    
    else:
        raise NotImplementedError
        
