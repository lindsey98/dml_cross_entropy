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

def main(args):

    # load config
    ### initial model trained on initial training set ##### 
    with open(args.config_file) as file:
        cfg = yaml.safe_load(file)
        
    # get logger
    # logging
    os.makedirs("log", exist_ok=True)
    with open("log/train_{}.log".format(cfg['DATA']['Name']), "w"): # TODO
        pass
    logging.basicConfig(filename="log/train_{}_softtriple.log".format(cfg['DATA']['Name']), level=logging.INFO)
    logger = logging.getLogger("trace")

    device = torch.device("cuda" if torch.cuda.is_available() and not cfg["Train"]["cpu"] else "cpu")
    
    # load data: TODO
    loaders, recall_ks = get_loaders(cfg)
    # create model
    model = get_model(cfg, loaders.num_classes)
    model.to(device)
    model = nn.DataParallel(model)
    
    # define loss function (criterion) and optimizer TODO
    criterion = SmoothCrossEntropy(epsilon=cfg['Train']['label_smoothing'])
    
    # define optimizer, model parameters and proxy weights are using different learning rate, proxy learning rate is higher, because model weights are pretrained needs lower learning rate
    optimizer = torch.optim.Adam("params": model.parameters(), lr = cfg['Train']['modellr'],
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
            
            # evaluate on test set
            recalls = validate(loaders.query, model, loaders.labeldict)
            print('Recall@1, 2, 4, 8: {recall[1]:.3f}, {recall[2]:.3f}, {recall[4]:.3f}, {recall[8]:.3f} \n'.format(recall=recalls))
            logger.info('Recall@1, 2, 4, 8: {recall[1]:.3f}, {recall[2]:.3f}, {recall[4]:.3f}, {recall[8]:.3f}'.format(recall=recalls))
            
    # evaluate on test set
    recalls = validate(loaders.query, model, loaders.labeldict)
    print('Recall@1, 2, 4, 8: {recall[1]:.3f}, {recall[2]:.3f}, {recall[4]:.3f}, {recall[8]:.3f} \n'.format(recall=recalls))
    logger.info('Recall@1, 2, 4, 8: {recall[1]:.3f}, {recall[2]:.3f}, {recall[4]:.3f}, {recall[8]:.3f}'.format(recall=recalls))
    # saving
    save_name = os.path.join(cfg['Train']['temp_dir'], "{}_{}.pt".format(cfg["MODEL"]["arch"], 
                                                                         cfg["DATA"]["Name"]))
    torch.save(model.state_dict(), save_name)



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