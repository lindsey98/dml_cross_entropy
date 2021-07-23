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
from src.active_learning import feature_extractor

from tqdm import tqdm
import logging
import yaml
import faiss
import numpy as np

def evaluator(query_features: torch.Tensor,
              query_labels: torch.Tensor,
              gallery_features: torch.Tensor,
              gallery_labels: torch.Tensor,
              ks: List[int], 
              threshold: float) -> (int, int, int, int):
    '''
        Evaluate number of TP/FP/TN/FN on a given model
    '''
    # query matching acc to gallery
    query_features = F.normalize(query_features, p=2, dim=1)
    gallery_features = F.normalize(gallery_features, p=2, dim=1)

    to_cpu_numpy = lambda x: x.cpu().numpy()
    q_f, q_l, g_f, g_l = map(to_cpu_numpy, [query_features, query_labels, gallery_features, gallery_labels])
    seen_labels = np.isin(q_l, g_l)
    q_f_novel, q_l_novel = q_f[~np.isin(q_l, g_l)], q_l[~np.isin(q_l, g_l)]
    q_f_seen, q_l_seen = q_f[np.isin(q_l, g_l)], q_l[np.isin(q_l, g_l)]
    
    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.device = 0

    max_k = max(ks)
    offset = 0
    index_function = faiss.GpuIndexFlatIP 
    index = index_function(res, g_f.shape[1], flat_config)
    index.add(g_f)
    
    # for seen classes
    distances_seen, closest_indices_seen = index.search(q_f_seen, max_k + offset)
    
    # for novel classes
    distances_novel, closest_indices_novel = index.search(q_f_novel, max_k + offset)

    tp = {}
    fn = {}
    fp = {}
    tn = {}
    for k in ks:
        indices = closest_indices_seen[:, offset:k + offset]
        tp[k] = ((q_l_seen[:, None] == g_l[indices]) & (distances_seen[:, offset:k + offset] >= threshold)).any(1)
        fn[k] = 1 - tp[k]
        tp[k] = tp[k].sum().item()
        fn[k] = fn[k].sum().item()
        
        indices = closest_indices_novel[:, offset:k + offset]
        fp[k] = ((distances_novel[:, offset:k + offset] >= threshold)).any(1)
        tn[k] = 1 - fp[k]        
        fp[k] = fp[k].sum().item()
        tn[k] = tn[k].sum().item()
        
    return (tp, fn, fp, tn)


def evaluator_aggregate(model: nn.Module, loaders: MetricLoaders, recall_ks: List[int]) -> float:
    '''
        Aggregate evaluation results over multiple thresholds and multiple k
    '''        
    
    gallery_features, gallery_labels, _ = feature_extractor(model=model, loaders=loaders.train_noshuffle)
    query_features, query_labels, _ = feature_extractor(model=model, loaders=loaders.query)
    
    # Evaluate
    eval_function = partial(evaluator, query_features=query_features, 
                            query_labels=query_labels,
                            gallery_features=gallery_features,
                            gallery_labels=gallery_labels,
                            ks = recall_ks)
    
    tp, fn, fp, tn = {}, {}, {}, {}
    for threshold in np.arange(0.7, 0.95, 0.05):
        tp[threshold], fn[threshold], fp[threshold], tn[threshold] = eval_function(threshold=threshold)
    
    torch.cuda.empty_cache()
    matching_acc = []
    for threshold in np.arange(0.7, 0.95, 0.05):
        for nn in recall_ks:
            acc = (tp[threshold][nn] + tn[threshold][nn]) / (fp[threshold][nn] + fn[threshold][nn] + tp[threshold][nn] + tn[threshold][nn])
            matching_acc.append(acc)
    matching_acc = np.mean(matching_acc)
    
    return matching_acc