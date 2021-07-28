"""
    PyTorch Package for SoftTriple Loss

    Reference
    ICCV'19: "SoftTriple Loss: Deep Metric Learning Without Triplet Sampling"

    Copyright@Alibaba Group

"""

import argparse
import os
import torch
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import torchvision.datasets as datasets
import torch.nn as nn
from PIL import Image
import yaml
from src.model import get_model
from src.train import *
from src.data import *
from src.evaluate import *
from src.active_learning import *
from src.utils import SoftTriple
import argparse
import logging

from torch.utils.tensorboard import SummaryWriter
import numpy as np

def main(args):

    # load config
    ### initial model trained on initial training set ##### 
    with open(args.config_file) as file:
        cfg = yaml.safe_load(file)
        
    # get logger
    # logging
    os.makedirs("log", exist_ok=True)
    with open("log/train_{}_softtriple.log".format(cfg['DATA']['Name']), "w"):
        pass
    logging.basicConfig(filename="log/train_{}_softtriple.log".format(cfg['DATA']['Name']), level=logging.INFO)
    logger = logging.getLogger("trace")

    writer = SummaryWriter('runs/{}_{}'.format(cfg["MODEL"]["arch"], cfg['DATA']['Name']))

    device = torch.device("cuda" if torch.cuda.is_available() and not cfg["Train"]["cpu"] else "cpu")
    
    # load data
    loaders, recall_ks = get_loaders(cfg)
    # create model
    model = get_model(cfg, loaders.num_classes)
    model.to(device)
    model = nn.DataParallel(model)
    
    # define loss function (criterion) and optimizer
    criterion = SoftTriple(cfg['Train']['la'], cfg['Train']['gamma'], cfg['Train']['tau'], 
                           cfg['Train']['margin'], cfg['MODEL']['num_features'], 
                           loaders.num_classes, cfg['Train']['K'])
    # initialize proxies
    all_gallery_feat, all_gallery_labels, _ = feature_extractor(model, loaders.train_noshuffle, loaders.labeldict)
    criterion.proxy_initialize(all_gallery_feat, all_gallery_labels)
    criterion = criterion.to(device)
    print('Finish proxy initialization')
    logger.info('Finish proxy initialization')
    # saving initial proxies
    save_name = os.path.join(cfg['Train']['temp_dir'], "{}_{}_proxyK{}_epoch{}.pt".format(cfg["MODEL"]["arch"], 
                                                                                 cfg["DATA"]["Name"], 
                                                                                 cfg["Train"]["K"], -1))
    torch.save(criterion.state_dict(), save_name)
    measure = proxy_similarity(criterion)
    if criterion.K > 1:
        writer.add_histogram('Proxy intersimilarity', measure, -1)
    
    # define optimizer, model parameters and proxy weights are using different learning rate, proxy learning rate is higher, because model weights are pretrained needs lower learning rate
    optimizer = torch.optim.Adam([{"params": model.parameters(), "lr": cfg['Train']['modellr']},
                                  {"params": criterion.parameters(), "lr": cfg['Train']['centerlr']}],
                                     eps=cfg['Train']['eps'], weight_decay = cfg['Train']['weight_decay'])
    cudnn.benchmark = True
    
    # evaluate on test set
    recalls = validate(loaders.query, model, loaders.labeldict)
    print('Recall@1, 2, 4, 8: {recall[1]:.3f}, {recall[2]:.3f}, {recall[4]:.3f}, {recall[8]:.3f} \n'.format(recall=recalls))
    logger.info('Recall@1, 2, 4, 8: {recall[1]:.3f}, {recall[2]:.3f}, {recall[4]:.3f}, {recall[8]:.3f}'.format(recall=recalls))

    for epoch in range(cfg['Train']['epochs']):
        print('Training in Epoch[{}]'.format(epoch))
        adjust_learning_rate(optimizer, epoch, cfg) 

        # train for one epoch
        training(model, loaders.train, loaders.labeldict, criterion, optimizer, None, epoch, logger)
        
        # save model
        if epoch % 10 == 0:
            os.makedirs(cfg['Train']['temp_dir'], exist_ok=True)
            # saving
            save_name = os.path.join(cfg['Train']['temp_dir'], "{}_{}.pt".format(cfg["MODEL"]["arch"], 
                                                                                 cfg["DATA"]["Name"]))
            torch.save(model.state_dict(), save_name)
            # saving learnt proxies
            save_name = os.path.join(cfg['Train']['temp_dir'], "{}_{}_proxyK{}_epoch{}.pt".format(cfg["MODEL"]["arch"], 
                                                                                         cfg["DATA"]["Name"], 
                                                                                         cfg["Train"]["K"], epoch))
            torch.save(criterion.state_dict(), save_name)
            
            # evaluate on test set
            recalls = validate(loaders.query, model, loaders.labeldict)
            print('Recall@1, 2, 4, 8: {recall[1]:.3f}, {recall[2]:.3f}, {recall[4]:.3f}, {recall[8]:.3f} \n'.format(recall=recalls))
            logger.info('Recall@1, 2, 4, 8: {recall[1]:.3f}, {recall[2]:.3f}, {recall[4]:.3f}, {recall[8]:.3f}'.format(recall=recalls))
            writer.add_scalar('Recall@1', recalls[1], epoch)
            writer.add_scalar('Recall@2', recalls[2], epoch)
            writer.add_scalar('Recall@4', recalls[4], epoch)
            writer.add_scalar('Recall@8', recalls[8], epoch)
            
        measure = proxy_similarity(criterion)
        if criterion.K > 1:
            writer.add_histogram('Proxy intersimilarity', measure, epoch)
            
    # evaluate on test set
    recalls = validate(loaders.query, model, loaders.labeldict)
    print('Recall@1, 2, 4, 8: {recall[1]:.3f}, {recall[2]:.3f}, {recall[4]:.3f}, {recall[8]:.3f} \n'.format(recall=recalls))
    logger.info('Recall@1, 2, 4, 8: {recall[1]:.3f}, {recall[2]:.3f}, {recall[4]:.3f}, {recall[8]:.3f}'.format(recall=recalls))
    measure = proxy_similarity(criterion)
    if criterion.K > 1:
        writer.add_histogram('Proxy intersimilarity', measure, epoch)
    writer.add_scalar('Recall@1', recalls[1], epoch)
    writer.add_scalar('Recall@2', recalls[2], epoch)
    writer.add_scalar('Recall@4', recalls[4], epoch)
    writer.add_scalar('Recall@8', recalls[8], epoch)

    # saving
    save_name = os.path.join(cfg['Train']['temp_dir'], "{}_{}.pt".format(cfg["MODEL"]["arch"], cfg["DATA"]["Name"]))
    torch.save(model.state_dict(), save_name)
    
    # saving learnt proxies
    save_name = os.path.join(cfg['Train']['temp_dir'], "{}_{}_proxyK{}_epoch{}.pt".format(cfg["MODEL"]["arch"], 
                                                                                         cfg["DATA"]["Name"], 
                                                                                         cfg["Train"]["K"], epoch))
    torch.save(criterion.state_dict(), save_name)
            

def proxy_similarity(criterion: nn.Module) -> float:
    proxy = criterion.fc.detach().cpu()
    proxy = F.normalize(proxy, p=2, dim=0)
    simCenter = proxy.t().matmul(proxy) 
    
    measure = simCenter[criterion.weight]
    measure = measure.reshape(criterion.cN, -1)
    measure = measure.mean(1)
    
    return measure

def validate(test_loader: DataLoader, 
             model: nn.Module, 
             labeldict: Dict) -> Dict:
    '''
        Switch to evaluation mode
    '''
    device = next(model.parameters()).device
    model.eval()

    testdata, testlabel, _ = feature_extractor(model=model, 
                                              loaders=test_loader,
                                              labeldict=labeldict)
    
    recalls = recall_at_ks(query_features=testdata.to(device),
                          query_labels=testlabel.to(device),
                          ks=[1,2,4,8],
                          cosine=True)
    return recalls


def adjust_learning_rate(optimizer: Optimizer, epoch: int, cfg: Dict) -> None:
    ''' 
        Decay lr by 10 every 20 epochs
    '''
    if (epoch+1)%20 == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= cfg['Train']['rate']
            
    print('At current epoch {}, modellr = {:.5f}, centerlr = {:.5f}'.format(epoch, 
                                                                    optimizer.param_groups[0]['lr'], 
                                                                    optimizer.param_groups[1]['lr']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--config_file', required=True, help='path to config file')
    args = parser.parse_args()
    
    main(args)