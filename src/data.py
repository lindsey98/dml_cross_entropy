import math
import os
import tempfile
from copy import deepcopy
from functools import partial
from pprint import pprint

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from typing import NamedTuple, Optional, Dict, List, Any
from torchvision import transforms

from src.data.image_dataset import ImageDataset
from src.data.utils import RandomReplacedIdentitySampler
from torch.utils.data import DataLoader, RandomSampler, BatchSampler

from tqdm import tqdm
import logging
import yaml
import faiss
from typing import Tuple, List


def get_transforms(crop_size: int, scale: Tuple[float], ratio: Tuple[float], 
                   resize=None, rotate=None, color_jitter=None) -> (transforms.Compose, transforms.Compose):
    '''
        Return list of transformations applied on training and testing
    '''
    train_transform, test_transform = [], []
    if resize is not None:
        test_transform.append(transforms.Resize(resize))
        train_transform.append(transforms.Resize(resize))
        
    if rotate is not None:
        train_transform.append(transforms.RandomRotation(rotate))
        
    if color_jitter is not None:
        train_transform.append(transforms.ColorJitter(color_jitter))
        
    train_transform.extend([transforms.RandomResizedCrop(size=crop_size, scale=scale, ratio=ratio),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor()])

    test_transform.extend([transforms.CenterCrop(size=crop_size),
                           transforms.ToTensor()])

    return transforms.Compose(train_transform), transforms.Compose(test_transform)



def read_file(filename: str) -> List:
    with open(filename) as f:
        lines = f.read().splitlines()
    return lines

def get_sets(cfg: Dict) -> (ImageDataset, ImageDataset, (ImageDataset, ImageDataset, ImageDataset, ImageDataset)):
    '''
        Prepared dataset
    '''
    
    data_path = cfg['DATA']['Path']['data_path']
    train_file = cfg['DATA']['Path']['train_file']
    test_file = cfg['DATA']['Path']['test_file']
    pool_file = cfg['DATA']['Path']['pool_file']
    preload = cfg['DATA']['Sampler']['preload']
    num_workers = cfg['DATA']['Sampler']['num_workers']
    
    crop_size=cfg['DATA']['Aug']['crop_size']
    scale=eval(cfg['DATA']['Aug']['scale'])
    ratio=eval(cfg['DATA']['Aug']['ratio'])
    resize=cfg['DATA']['Aug']['resize']
    rotate=cfg['DATA']['Aug']['rotate']
    color_jitter=cfg['DATA']['Aug']['color_jitter']
    
    train_transform, test_transform = get_transforms(crop_size, scale, ratio, resize, rotate,  color_jitter)

    train_lines = read_file(os.path.join(data_path, train_file))
    train_samples = [(os.path.join(data_path, line.split(',')[0]), int(line.split(',')[1])) for line in train_lines]
    known_classes = [int(line.split(',')[1]) for line in train_lines] # known classes already in training
    
    train_set = ImageDataset(train_samples, transform=train_transform, 
                             preload=preload, num_workers=num_workers)
    train_noshuffle_set = ImageDataset(train_samples, transform=test_transform, 
                             preload=preload, num_workers=num_workers)

    query_lines = read_file(os.path.join(data_path, test_file))
    query_samples = [(os.path.join(data_path, line.split(',')[0]), int(line.split(',')[1])) for line in query_lines]
    
    query_set = ImageDataset(query_samples, transform=test_transform, preload=preload, num_workers=num_workers)
    gallery_set = None
        
    # test set which only excluding known classes
    query_novel_samples = [(os.path.join(data_path, line.split(',')[0]), int(line.split(',')[1])) \
                               for line in query_lines if int(line.split(',')[1]) not in known_classes]  
    query_novel_set = ImageDataset(query_novel_samples, transform=test_transform, 
                                   preload=preload, num_workers=num_workers)

    # pool set
    pool_lines = read_file(os.path.join(data_path, pool_file))
    pool_samples = [(os.path.join(data_path, line.split(',')[0]), int(line.split(',')[1])) for line in pool_lines]
    pool_set = ImageDataset(pool_samples, transform=test_transform, preload=preload, num_workers=num_workers)

    return train_set, train_noshuffle_set, (query_set, gallery_set, pool_set, query_novel_set)

class MetricLoaders(NamedTuple):
    '''
        Data structure defined for dataloader
    '''
    train: DataLoader
    num_classes: int
    query: DataLoader
    query_novel: DataLoader
    pool: DataLoader
    train_noshuffle: DataLoader
    labeldict: Dict
    gallery: Optional[DataLoader] = None

def get_loaders(cfg: Dict) -> MetricLoaders:
    '''
        Prepare dataloader
    '''
    batch_size = cfg['DATA']['Sampler']['batch_size']
    test_batch_size = cfg['DATA']['Sampler']['test_batch_size']
    num_workers = cfg['DATA']['Sampler']['num_workers']
    pin_memory = cfg['DATA']['Sampler']['pin_memory']
    sampler = cfg['DATA']['Sampler']['sampler']
    recalls = cfg['DATA']['Aug']['recalls']
    num_iterations = cfg['DATA']['Sampler']['num_iterations']
    num_identities = cfg['DATA']['Sampler']['num_identities']
    train_set, train_noshuffle_set, (query_set, gallery_set, pool_set, query_novel_set) = get_sets(cfg)

    if sampler == 'random':
        train_sampler = BatchSampler(RandomSampler(train_set), batch_size=batch_size, drop_last=True)
    elif sampler == 'random_id':
        train_sampler = RandomReplacedIdentitySampler(train_set.targets, batch_size, num_identities, num_iterations)
    else:
        raise ValueError('Invalid choice of sampler ({}).'.format(sampler))
        
    train_loader = DataLoader(train_set, batch_sampler=train_sampler, 
                              num_workers=num_workers, pin_memory=pin_memory)
    
    train_noshuffle = DataLoader(train_noshuffle_set, batch_size=test_batch_size,
                                 num_workers=num_workers, pin_memory=pin_memory)
    
    query_loader = DataLoader(query_set, batch_size=test_batch_size, 
                              num_workers=num_workers, pin_memory=pin_memory)
    
    gallery_loader = None
    if gallery_set is not None:
        gallery_loader = DataLoader(gallery_set, batch_size=test_batch_size, 
                                    num_workers=num_workers, pin_memory=pin_memory)
        
    query_novel_loader = DataLoader(query_novel_set, batch_size=test_batch_size, 
                                    num_workers=num_workers, pin_memory=pin_memory)
    
    pool_loader = DataLoader(pool_set, batch_size=test_batch_size, 
                             num_workers=num_workers, pin_memory=pin_memory)
    
    # label is not continuous, need this dictionary to map it to continuous integer
    labeldict = {}
    i = 0
    for label in set(train_set.targets):
        if int(label) not in labeldict.keys():
            labeldict[int(label)] = i
            i += 1
            
    for label in set(pool_set.targets):
        if int(label) not in labeldict.keys():
            labeldict[int(label)] = i
            i += 1

    for label in set(query_set.targets):
        if int(label) not in labeldict.keys():
            labeldict[int(label)] = i
            i += 1
 
    print(labeldict)
    
    return MetricLoaders(train=train_loader, query=query_loader, pool=pool_loader, gallery=gallery_loader, 
                         train_noshuffle = train_noshuffle,
                         num_classes=len(set(train_set.targets)),
                         labeldict=labeldict,
                         query_novel=query_novel_loader), recalls