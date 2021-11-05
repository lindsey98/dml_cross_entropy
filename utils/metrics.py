from typing import Dict, List, Optional

import faiss
import numpy as np
import torch
import torch.nn.functional as F
import sklearn.cluster
import sklearn.metrics.cluster
from utils.map import *

class AverageMeter:
    """Computes and stores the average and current value on device"""

    def __init__(self, device, length):
        self.device = device
        self.length = length
        self.reset()

    def reset(self):
        self.values = torch.zeros(self.length, device=self.device, dtype=torch.float)
        self.counter = 0
        self.last_counter = 0

    def append(self, val):
        self.values[self.counter] = val.detach()
        self.counter += 1
        self.last_counter += 1

    @property
    def val(self):
        return self.values[self.counter - 1]

    @property
    def avg(self):
        return self.values[:self.counter].mean()

    @property
    def values_list(self):
        return self.values[:self.counter].cpu().tolist()

    @property
    def last_avg(self):
        if self.last_counter == 0:
            return self.latest_avg
        else:
            self.latest_avg = self.values[self.counter - self.last_counter:self.counter].mean()
            self.last_counter = 0
            return self.latest_avg


def cluster_by_kmeans(X, nb_clusters):
    """
    xs : embeddings with shape [nb_samples, nb_features]
    nb_clusters : in this case, must be equal to number of classes
    """
    # return sklearn.cluster.MiniBatchKMeans(nb_clusters, batch_size=32).fit(X).labels_
    X = X.detach().cpu().numpy()
    kmeans = faiss.Kmeans(d=X.shape[1], k=nb_clusters)
    kmeans.train(X.astype(np.float32))
    labels = kmeans.index.search(X.astype(np.float32), 1)[1]
    return np.squeeze(labels, 1)

def calc_normalized_mutual_information(ys, xs_clustered):
    return sklearn.metrics.cluster.normalized_mutual_info_score(xs_clustered, ys, average_method='geometric')


def calc_nmi(X, T, nb_classes):
    # calculate NMI with kmeans clustering
    nmi = calc_normalized_mutual_information(
        T,
        cluster_by_kmeans(
            X, nb_classes
        )
    )
    print(nmi)
    return nmi

def mapr(X, T):
    # MAP@R
    label_counts = get_label_match_counts(T, T) # get R
    # num_k = determine_k(num_reference_embeddings=len(T), embeddings_come_from_same_source=True) # equal to num_reference-1 (deduct itself)
    # num_k = determine_k(
    #     num_reference_embeddings=len(T), embeddings_come_from_same_source=True
    # ) # equal to num_reference-1 (deduct itself)

    num_k = max([count[1] for count in label_counts])
    knn_indices = get_knn(
        X, X, num_k, True
    )
    knn_labels = T[knn_indices] # get KNN indicies
    map_R = mean_average_precision_at_r(knn_labels=knn_labels,
                                        gt_labels=T[:, None],
                                        embeddings_come_from_same_source=True,
                                        label_counts=label_counts,
                                        avg_of_avgs=False,
                                        label_comparison_fn=torch.eq)
    logging.info("MAP@R:{:.3f}".format(map_R * 100))
    return map_R


def mapr_inshop(X_query, T_query, X_gallery, T_gallery):
    # MAP@R
    label_counts = get_label_match_counts(T_query, T_gallery)  # get R
    # num_k = determine_k(
    #     num_reference_embeddings=len(T_gallery), embeddings_come_from_same_source=False
    # )  # equal to num_reference
    num_k = max([count[1] for count in label_counts])
    knn_indices = get_knn(
        X_gallery, X_query, num_k, True
    )
    knn_labels = T_gallery[knn_indices]  # get KNN indicies
    map_R = mean_average_precision_at_r(knn_labels=knn_labels,
                                        gt_labels=T_query[:, None],
                                        embeddings_come_from_same_source=False,
                                        label_counts=label_counts,
                                        avg_of_avgs=False,
                                        label_comparison_fn=torch.eq)
    logging.info("MAP@R:{:.3f}".format(map_R * 100))
    return map_R


@torch.no_grad()
def recall_at_ks(query_features: torch.Tensor,
                 query_labels: torch.LongTensor,
                 ks: List[int],
                 gallery_features: Optional[torch.Tensor] = None,
                 gallery_labels: Optional[torch.Tensor] = None,
                 cosine: bool = True) -> Dict[int, float]:
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

    if hasattr(faiss, 'StandardGpuResources'):
        res = faiss.StandardGpuResources()
        flat_config = faiss.GpuIndexFlatConfig()
        flat_config.device = 0
        max_k = max(ks)
        index_function = faiss.GpuIndexFlatIP if cosine else faiss.GpuIndexFlatL2
        index = index_function(res, g_f.shape[1], flat_config)
    else:
        max_k = max(ks)
        index_function = faiss.IndexFlatIP if cosine else faiss.IndexFlatL2
        index = index_function(g_f.shape[1])

    index.add(g_f)
    closest_indices = index.search(q_f, max_k + offset)[1]

    recalls = {}
    for k in ks:
        indices = closest_indices[:, offset:k + offset]
        recalls[k] = (q_l[:, None] == g_l[indices]).any(1).mean()
    print(recalls)
    return {k: round(v * 100, 2) for k, v in recalls.items()}


@torch.no_grad()
def recall_at_ks_full(query_features: torch.Tensor,
                 query_labels: torch.LongTensor,
                 ks: List[int],
                 gallery_features: Optional[torch.Tensor] = None,
                 gallery_labels: Optional[torch.Tensor] = None,
                 cosine: bool = False,
                 threshold: Optional[float] = None) -> Dict[int, float]:
    """
    Compute the full recall, precision list between samples at each k.

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
    threshold: float
        Use to thresholding on the matched similarity, only consider cosine distance thresholding for now

    Returns
    -------
    recalls : Dict[List[bool]]
        List of recalls for all query data at each k.
        List of precisions for all query data at each k.

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
    distances, closest_indices = index.search(q_f, max_k + offset)


    recalls = {}
    precisions = {}
    for k in ks:
        indices = closest_indices[:, offset:k + offset]
        # Recall @ k
        recalls[k] = ((q_l[:, None] == g_l[indices]) * (distances[:, offset:k + offset] >= threshold)).any(1)
        precisions[k] = ((q_l[:, None] == g_l[indices]) * (distances[:, offset:k + offset] >= threshold)).sum(1) / \
                        ((distances[:, offset:k + offset] >= threshold).sum(1) + 1e-4)

    return recalls, precisions



@torch.no_grad()
def fp_fn_eval(query_features: torch.Tensor,
                 query_labels: torch.LongTensor,
                 ks: List[int],
                 gallery_features: Optional[torch.Tensor] = None,
                 gallery_labels: Optional[torch.Tensor] = None,
                 cosine: bool = False,
                 threshold: Optional[float] = None) -> Dict[int, float]:
    """
    Get FPs and FNs from query data
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
        threshold: float
            Use to thresholding on the matched similarity, only consider cosine distance thresholding for now

        Returns
        -------
        recalls : Dict[List[bool]]
            List of fns: true indicates a FN
            List of fps: true indicates a FP

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

    max_k = 1 # only look at the 1st neighbor to decide whether it is FP or FN
    index_function = faiss.GpuIndexFlatIP if cosine else faiss.GpuIndexFlatL2
    index = index_function(res, g_f.shape[1], flat_config)
    index.add(g_f)
    distances, closest_indices = index.search(q_f, max_k + offset)

    indices = closest_indices[:, offset:max_k + offset]
    # fns
    fns = ((distances[:, offset:max_k + offset] < threshold)).any(1)  # 1st neighbor is far --> FN
    fps = ((q_l[:, None] != g_l[indices]) * (distances[:, offset:max_k + offset] >= threshold)).any(1) # 1st neighbor is close but not having the same class label

    return fns, fps


