import math
import os
import tempfile
from copy import deepcopy
from functools import partial
from pprint import pprint

import torch
import torch.nn as nn

from models.ingredient import model_ingredient, get_model
from utils import state_dict_to_cpu, SmoothCrossEntropy
from utils.data.dataset_ingredient import data_ingredient, get_loaders
from utils.training import train, evaluate
from utils.data.dataset_ingredient import ImageDataset
from torch.utils.data import DataLoader, RandomSampler, BatchSampler
from torchvision import transforms

os.environ["CUDA_VISIBLE_DEVICES"]="1"
name = 'cars_512'
data_path = '/home/ruofan/PycharmProjects/ProxyNCA-/mnt/datasets/CARS_196'
train_file = 'train.txt'
test_file = 'test.txt'
num_classes = 98
temp_dir = 'checkpoints/resnet50_cars_512.pt'

arch = 'resnet50'
pretrained = True  # use a pretrained model from torchvision
num_features = 512  # dimensionality of the features produced by the feature extractor
dropout = 0.
norm_layer = 'batch'  # use a normalization layer (batchnorm or layernorm) for the features
remap = False  # remap features through a linear layer
detach = False  # detach features before feeding to the classification layer. Prevents training of the feature extractor with cross-entropy.
normalize = False  # normalize the features
set_bn_eval = True  # set bn in eval mode even in training
normalize_weight = False  # normalize the weights of the classification layer
crop_size = 224
scale = (0.16, 1)
ratio = (3. / 4., 4. / 3.)

resize = 256

def get_transforms(crop_size, scale, ratio, resize=None, rotate=None, color_jitter=None):
    train_transform, test_transform = [], []
    if resize is not None:
        test_transform.append(transforms.Resize(resize))
        train_transform.append(transforms.Resize(resize))
    if rotate is not None:
        train_transform.append(transforms.RandomRotation(rotate))
    if color_jitter is not None:
        train_transform.append(transforms.ColorJitter(*color_jitter))
    train_transform.extend([transforms.RandomResizedCrop(size=crop_size, scale=scale, ratio=ratio),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor()])

    test_transform.extend([transforms.CenterCrop(size=crop_size),
                           transforms.ToTensor()])

    return transforms.Compose(train_transform), transforms.Compose(test_transform)


def read_file(filename):
    with open(filename) as f:
        lines = f.read().splitlines()
    return lines

def get_sets(name, data_path, train_file, test_file, preload, num_workers):
    train_transform, test_transform = get_transforms(crop_size, scale, ratio, resize=resize)

    train_lines = read_file(os.path.join(data_path, train_file))
    train_samples = [(os.path.join(data_path, line.split(',')[0]), int(line.split(',')[1])) for line in train_lines]
    train_set = ImageDataset(train_samples, transform=train_transform, preload=preload, num_workers=num_workers)

    if isinstance(test_file, list) and len(test_file) == 2:
        query_lines = read_file(os.path.join(data_path, test_file[0]))
        gallery_lines = read_file(os.path.join(data_path, test_file[1]))
        query_samples = [(os.path.join(data_path, line.split(',')[0]), int(line.split(',')[1])) for line in query_lines]
        gallery_samples = [(os.path.join(data_path, line.split(',')[0]), int(line.split(',')[1])) for line in
                           gallery_lines]
        query_set = ImageDataset(query_samples, transform=test_transform, preload=preload, num_workers=num_workers)
        gallery_set = ImageDataset(gallery_samples, transform=test_transform, preload=preload, num_workers=num_workers)
    else:
        query_lines = read_file(os.path.join(data_path, test_file))
        query_samples = [(os.path.join(data_path, line.split(',')[0]), int(line.split(',')[1])) for line in query_lines]
        query_set = ImageDataset(query_samples, transform=test_transform, preload=preload, num_workers=num_workers)
        gallery_set = None

    return train_set, (query_set, gallery_set)

def main():

    device = torch.device('cuda')
    train_set, (query_set, gallery_set) = get_sets(name, data_path, train_file, test_file, False, 4)
    query_loader = DataLoader(query_set, batch_size=32, num_workers=4)
    gallery_loader = None
    if gallery_set is not None:
        gallery_loader = DataLoader(gallery_set, batch_size=32, num_workers=4)

    model = get_model(num_classes, arch, pretrained, num_features, norm_layer, detach, remap, normalize, normalize_weight,
                      set_bn_eval, dropout)

    model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model.load_state_dict(torch.load(os.path.join(temp_dir)))

    recall_ks = [1, 2, 4, 8]
    if 'sop' in name:
        recall_ks = [1, 10, 100, 1000]
    # setup partial function to simplify call
    eval_function = partial(evaluate, model=model, recall=recall_ks, query_loader=query_loader,
                            gallery_loader=gallery_loader)

    # setup best validation logger
    metrics = eval_function()
    print('Recall', metrics.recall)
    print('NMI', metrics.nmi)
    print('MAP@R', metrics.mapr)

if __name__ == '__main__':
    main()



