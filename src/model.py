from typing import Dict
import torch.nn as nn
from src.architectures import __all__, __dict__

def get_model(cfg: Dict, num_classes: int) -> nn.Module:
    '''
        Initialize model
    '''
    arch = cfg['MODEL']['arch']
    pretrained = cfg['MODEL']['pretrained']
    num_features = cfg['MODEL']['num_features']
    norm_layer = cfg['MODEL']['norm_layer']
    detach = cfg['MODEL']['detach']
    remap = cfg['MODEL']['remap']
    normalize = cfg['MODEL']['normalize']
    normalize_weight = cfg['MODEL']['normalize_weight']
    set_bn_eval = cfg['MODEL']['set_bn_eval']
    dropout = cfg['MODEL']['dropout']
    
    keys = list(map(lambda x: x.lower(), __all__))
    index = keys.index(arch.lower())
    arch = __all__[index]
    return __dict__[arch](num_classes=num_classes, num_features=num_features, 
                          pretrained=pretrained, dropout=dropout,
                          norm_layer=norm_layer, detach=detach, remap=remap, normalize=normalize,
                          normalize_weight=normalize_weight, set_bn_eval=set_bn_eval)


