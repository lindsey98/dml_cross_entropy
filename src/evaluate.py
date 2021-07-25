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

@torch.no_grad()
def feature_extractor(model: nn.Module, 
                      loaders: MetricLoaders) -> (torch.Tensor, torch.Tensor, List):
    '''
      Extract feature embeddings given a DataLoader
    '''
    device = next(model.parameters()).device

    all_query_labels = []
    all_query_features = []

    with torch.no_grad():
        for batch, labels, _ in tqdm(loaders, desc='Extracting features', leave=False, ncols=80):
            batch, labels = batch.to(device), labels.to(device)
            logits, features = model(batch)

            all_query_labels.append(labels)
            all_query_features.append(features)

    torch.cuda.empty_cache()
    all_query_labels = torch.cat(all_query_labels, 0)
    all_query_features = torch.cat(all_query_features, 0)
    all_query_samples = loaders.dataset.samples
    
    assert len(all_query_features) == len(all_query_labels) == len(all_query_samples)
    
    return all_query_features.detach().cpu(), all_query_labels.detach().cpu(), all_query_samples

def evaluator(query_features: torch.Tensor,
              query_labels: torch.Tensor,
              gallery_features: torch.Tensor,
              gallery_labels: torch.Tensor,
              ks: List[int], 
              offset: int,
              threshold: float) -> (int, int, int, int):
    '''
        Evaluate number of TP/FP/TN/FN on a given model 
    '''
    # query matching acc to gallery
    query_features = F.normalize(query_features, p=2, dim=1)
    gallery_features = F.normalize(gallery_features, p=2, dim=1)

    to_cpu_numpy = lambda x: x.cpu().numpy()
    q_f, q_l, g_f, g_l = map(to_cpu_numpy, [query_features, query_labels, gallery_features, gallery_labels])
    
    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.device = 0

    max_k = max(ks)
    index_function = faiss.GpuIndexFlatIP 
    index = index_function(res, g_f.shape[1], flat_config)
    index.add(g_f)
    
    distances, closest_indices = index.search(q_f, max_k + offset)

    tp = {}
    fn = {}
    fp = {}
    tn = {}
    # By right, you should observe that for a given threshold, the number of TPs increases as k increases, the number of FPs decreases as k increases, the number of TNs or the number of FNs remains constant when k increases. 
    # On the other hand, for a given k, the number of TPs increase as threshold decreases, the number of FPs increases as threshold decreases, the number of FNs decreases as threshold decreases, the number of TNs decreases as threshold decreases
    for k in ks:
        indices = closest_indices[:, offset:(k + offset)] # NN indices
        
        # TPs are those reported and reported as correct class by any neighbor
        tp[k] = ((q_l[:, None] == g_l[indices]) & (distances[:, offset:(k + offset)] >= threshold)).any(1)
        # FPs are those not TPs but has been reported by any neighbor
        fp[k] = (1 - tp[k]) & ((distances[:, offset:(k + offset)] >= threshold).any(1))
        assert (len(tp[k]) == len(q_f)) & (len(fp[k]) == len(q_f))
        tp[k] = tp[k].sum().item()
        fp[k] = fp[k].sum().item()
        
        # TNs are those not reported by any neighbor and belongs to novel classes
        tn[k] = (1 - np.isin(q_l, g_l)) & ((distances[:, offset:(k + offset)] < threshold).all(1))
        # FNs are those not reported by any neighbor but belongs to seen classes
        fn[k] = (np.isin(q_l, g_l)) & ((distances[:, offset:(k + offset)] < threshold).all(1))
        
        assert (len(tn[k]) == len(q_f)) & (len(fn[k]) == len(q_f))
  
        tn[k] = tn[k].sum().item()
        fn[k] = fn[k].sum().item()
        
        assert tn[k] + fn[k] + tp[k] + fp[k] == len(q_f)
        
    return (tp, fn, fp, tn)

def fpfn_extractor(query_features: torch.Tensor,
                  query_labels: torch.Tensor,
                  gallery_features: torch.Tensor,
                  gallery_labels: torch.Tensor,
                  ks: int, 
                  offset: int,
                  threshold: float) -> (np.ndarray, np.ndarray):
    '''
        Get FPs, FNs indices for query data
    '''
    # query matching acc to gallery
    query_features = F.normalize(query_features, p=2, dim=1)
    gallery_features = F.normalize(gallery_features, p=2, dim=1)

    to_cpu_numpy = lambda x: x.cpu().numpy()
    q_f, q_l, g_f, g_l = map(to_cpu_numpy, [query_features, query_labels, gallery_features, gallery_labels])
    
    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.device = 0

    index_function = faiss.GpuIndexFlatIP 
    index = index_function(res, g_f.shape[1], flat_config)
    index.add(g_f)
    
    distances, closest_indices = index.search(q_f, ks + offset)
    
    indices = closest_indices[:, offset:(ks + offset)]
    # TPs are those reported and reported as correct class
    tp = ((q_l[:, None] == g_l[indices]) & (distances[:, offset:(ks + offset)] >= threshold)).any(1)
    # FPs are those reported but reported as another class
    fp = (1 - tp) & ((distances[:, offset:(ks + offset)] >= threshold).any(1))
    
    # TNs are those not reported and from novel classes (not seen in gallery)
    tn = (1 - np.isin(q_l, g_l)) & ((distances[:, offset:(ks + offset)] < threshold).all(1))
    # FNs are those not reported and from known classes (seen in gallery)
    fn = (np.isin(q_l, g_l)) & ((distances[:, offset:(ks + offset)] < threshold).all(1))
    
    assert (len(fn) == len(q_l)) & (len(fp) == len(q_l))
    fn_indices = np.where(fn == True)[0]
    fp_indices = np.where(fp == True)[0]
        
    return fn_indices, fp_indices

@torch.no_grad()
def evaluator_aggregate(model: nn.Module, 
                        loaders: MetricLoaders, 
                        recall_ks: List[int]) -> float:
    '''
        Evaluator for current model on test set
        Aggregate evaluation results over multiple thresholds and multiple k
    '''        
    
    gallery_features, gallery_labels, _ = feature_extractor(model=model, loaders=loaders.train_noshuffle)
    query_features, query_labels, _ = feature_extractor(model=model, loaders=loaders.query)
    
    # Evaluate
    eval_function = partial(evaluator, query_features=query_features, 
                            query_labels=query_labels,
                            gallery_features=gallery_features,
                            gallery_labels=gallery_labels,
                            ks = recall_ks,
                            offset = 0)
    
    tp, fn, fp, tn = {}, {}, {}, {}
    for threshold in np.arange(0.5, 0.95, 0.025):
        tp[threshold], fn[threshold], fp[threshold], tn[threshold] = eval_function(threshold=threshold)
    
    torch.cuda.empty_cache()
    matching_acc = []
    avg_tp = []
    avg_fp = []
    avg_tn = []
    avg_fn = []
    
    for threshold in np.arange(0.5, 0.95, 0.025):
        for nn in recall_ks:
            total = (fp[threshold][nn] + fn[threshold][nn] + tp[threshold][nn] + tn[threshold][nn])
            correct = (tp[threshold][nn] + tn[threshold][nn])
            acc = correct / total
            matching_acc.append(acc)
            avg_tp.append(tp[threshold][nn])
            avg_fp.append(fp[threshold][nn])
            avg_tn.append(tn[threshold][nn])
            avg_fn.append(fn[threshold][nn])

    matching_acc = np.mean(matching_acc)
    avg_tp = np.mean(avg_tp)
    avg_fp = np.mean(avg_fp)
    avg_tn = np.mean(avg_tn)
    avg_fn = np.mean(avg_fn)

    return matching_acc, avg_tp, avg_fp, avg_tn, avg_fn



@torch.no_grad()
def evaluator_selector(model: nn.Module, 
                       loaders: MetricLoaders, 
                       selected_indices: Union[np.ndarray, List],
                       threshold: float) -> ((List, List), (int, int)):
    '''
        Evaluator for selector on unlabelled pool set
        Record how many FPs FNs are inside the selected indices
    '''
    gallery_features, gallery_labels, _ = feature_extractor(model=model, loaders=loaders.train_noshuffle)
    pool_features, pool_labels, _ = feature_extractor(model=model, loaders=loaders.pool)
    
    # Evaluate
    fn_indices, fp_indices = fpfn_extractor(query_features=pool_features, 
                                            query_labels=pool_labels,
                                            gallery_features=gallery_features,
                                            gallery_labels=gallery_labels,
                                            ks = 1,
                                            offset = 0, 
                                            threshold = threshold)
    
    # Get overlapping indices
    fn_selected_indices = list(set(fn_indices).intersection(set(selected_indices)))
    fp_selected_indices = list(set(fp_indices).intersection(set(selected_indices)))
    

    return (len(fn_indices), len(fp_indices)), (len(fn_selected_indices), len(fp_selected_indices))
    

@torch.no_grad()
def threshold_finder(model: nn.Module,
                    loaders: MetricLoaders) -> (float, float):
    '''
        Threshold finder on gallery set
        Decide which matching threshold is the best using gallery/training set only
    '''
    
    gallery_features, gallery_labels, _ = feature_extractor(model=model, loaders=loaders.train_noshuffle)

    # Get number of TPs, FPs, FNs, TNs(0 in this case cuz all classes are seen)
    best_f1 = 0
    best_threshold = 0.5
    for threshold in np.arange(0.5, 0.95, 0.025):
        (tp, fn, fp, tn) = evaluator(query_features=gallery_features,
                                      query_labels=gallery_labels,
                                      gallery_features=gallery_features,
                                      gallery_labels=gallery_labels,
                                      ks=[1], # look at 1st NN only
                                      offset = 1, # exclude itself
                                      threshold=threshold)
        precision = tp[1] / (tp[1] + fp[1])
        recall = tp[1] / (tp[1] + fn[1])
        f1 = 2*precision*recall / (precision + recall)
        print('Ts = {}, Precision = {}, Recall = {}, F1 = {}'.format(threshold, precision, recall, f1))
        
        if f1 >= best_f1:
            best_f1 = f1
            best_threshold = threshold
            
    return best_f1, best_threshold