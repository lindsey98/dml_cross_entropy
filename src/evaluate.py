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

from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score
import numpy as np


# def nmi_recall_evaluator(X: np.ndarray, 
#                          Y: np.ndarray, 
#                          Kset: np.ndarray) -> (float, List[float]):
#     '''
#         Evaluate NMI between clustering assignment and true labels
#         Evaluate Recall@K on a set of K (Kset)
#     '''
#     nmi = None
#     num = X.shape[0] # N
#     classN = np.max(Y)+1 # Number of classes
#     kmax = np.max(Kset) # Maximum number of NN we look at
#     recallK = np.zeros(len(Kset))
#     #compute NMI
#     kmeans = KMeans(n_clusters=classN).fit(X)
#     nmi = normalized_mutual_info_score(Y, kmeans.labels_, average_method='arithmetic')
    #How mutual information is calculated? I(X, Y) = \int_x \int_y {p(x,y) * log(p(x,y)/(p(x)p(y)))}
    #If X, Y are two sets then I(X, Y) = \sum_i \sum_j {|U_i intersect U_j|/N * log(N|U_i intersect U_j|/|U_i||U_j|))}
    
    #compute Recall@K: count if at least one of its K neighbors are correct
#     sim = X.dot(X.T)
#     minval = np.min(sim) - 1.
#     sim -= np.diag(np.diag(sim))
#     sim += np.diag(np.ones(num) * minval)
#     indices = np.argsort(-sim, axis=1)[:, : kmax] # sort according to similarity
#     YNN = Y[indices]
#     for i in range(0, len(Kset)):
#         pos = 0.
#         for j in range(0, num):
#             if Y[j] in YNN[j, :Kset[i]]:
#                 pos += 1. # at least 1 NN is correct
#         recallK[i] = pos/num 
#     return nmi, recallK

@torch.no_grad()
def recall_at_ks(query_features: torch.Tensor,
                 query_labels: torch.LongTensor,
                 ks: List[int],
                 gallery_features: Optional[torch.Tensor] = None,
                 gallery_labels: Optional[torch.Tensor] = None,
                 cosine: bool = False) -> Dict[int, float]:
    """
    Compute the recall between samples at each k. This function uses about 8GB of memory.

    Parameters
    ----------
    query_features : torch.Tensor
        Features for each query sample. shape: (num_queries, num_features)
    query_labels : torch.LongTensor
        Labels corresponding to the query features. shape: (num_queries,)
    ks : List[int]
        Values at which to compute the recall.
    gallery_features : torch.Tensor
        Features for each gallery sample. shape: (num_queries, num_features)
    gallery_labels : torch.LongTensor
        Labels corresponding to the gallery features. shape: (num_queries,)
    cosine : bool
        Use cosine distance between samples instead of euclidean distance.

    Returns
    -------
    recalls : Dict[int, float]
        Values of the recall at each k.

    """
    offset = 0
    if gallery_features is None and gallery_labels is None:
        offset = 1
        gallery_features = query_features
        gallery_labels = query_labels
    elif gallery_features is None or gallery_labels is None:
        raise ValueError('gallery_features and gallery_labels needs to be both None or both Tensors.')

    if cosine:
        query_features = F.normalize(query_features, p=2, dim=1)
        gallery_features = F.normalize(gallery_features, p=2, dim=1)

    to_cpu_numpy = lambda x: x.cpu().numpy()
    q_f, q_l, g_f, g_l = map(to_cpu_numpy, [query_features, query_labels, gallery_features, gallery_labels])

    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.device = 0

    max_k = max(ks)
    index_function = faiss.GpuIndexFlatIP if cosine else faiss.GpuIndexFlatL2
    index = index_function(res, g_f.shape[1], flat_config)
    index.add(g_f)
    closest_indices = index.search(q_f, max_k + offset)[1]

    recalls = {}
    for k in ks:
        indices = closest_indices[:, offset:k + offset]
        recalls[k] = (q_l[:, None] == g_l[indices]).any(1).mean()
    return {k: round(v * 100, 2) for k, v in recalls.items()}


@torch.no_grad()
def feature_extractor(model: nn.Module, 
                      loaders: DataLoader,
                      labeldict: Dict) -> (torch.Tensor, torch.Tensor, List):
    '''
      Extract feature embeddings given a DataLoader
    '''
    device = next(model.parameters()).device

    all_query_labels = []
    all_query_features = []

    with torch.no_grad():
        for batch, labels, _ in tqdm(loaders, desc='Extracting features', leave=False, ncols=80):
            labels = torch.tensor([labeldict[x] for x in labels.numpy()])
            batch, labels = batch.to(device), labels.to(device)
            logits, features = model(batch)
            all_query_labels.append(labels)
            all_query_features.append(features)

    torch.cuda.empty_cache()
    all_query_labels = torch.cat(all_query_labels, 0)
    all_query_features = torch.cat(all_query_features, 0)
    all_query_samples = loaders.dataset.samples
    print(len(all_query_labels))
    print(len(all_query_features))
    print(len(all_query_samples))
    
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
    tn_indices = np.where(tn == True)[0]
    tp_indices = np.where(tp == True)[0]
        
    return fn_indices, fp_indices, tn_indices, tp_indices

@torch.no_grad()
def evaluator_aggregate(model: nn.Module, 
                        loaders: MetricLoaders, 
                        recall_ks: List[int]) -> float:
    '''
        Evaluator for current model on test set
        Aggregate evaluation results over multiple thresholds and multiple k
    '''        
    
    gallery_features, gallery_labels, _ = feature_extractor(model=model, 
                                                            loaders=loaders.train_noshuffle, 
                                                            labeldict=loaders.labeldict)
    query_features, query_labels, _ = feature_extractor(model=model, 
                                                        loaders=loaders.query, 
                                                        labeldict=loaders.labeldict)
    
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
    gallery_features, gallery_labels, _ = feature_extractor(model=model, 
                                                            loaders=loaders.train_noshuffle, 
                                                            labeldict=loaders.labeldict)
    pool_features, pool_labels, _ = feature_extractor(model=model, 
                                                      loaders=loaders.pool, 
                                                      labeldict=loaders.labeldict)
    
    # Evaluate
    fn_indices, fp_indices, _, _ = fpfn_extractor(query_features=pool_features, 
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
    
    gallery_features, gallery_labels, _ = feature_extractor(model=model, 
                                                            loaders=loaders.train_noshuffle, 
                                                            labeldict=loaders.labeldict)

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