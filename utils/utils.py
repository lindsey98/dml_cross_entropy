from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


def state_dict_to_cpu(state_dict: OrderedDict):
    """Moves a state_dict to cpu and removes the module. added by DataParallel.
    Parameters
    ----------
    state_dict : OrderedDict
        State_dict containing the tensors to move to cpu.
    Returns
    -------
    new_state_dict : OrderedDict
        State_dict on cpu.
    """
    new_state = OrderedDict()
    for k in state_dict.keys():
        newk = k.replace('module.', '')  # remove "module." if model was trained using DataParallel
        new_state[newk] = state_dict[k].cpu()
    return new_state


class SmoothCrossEntropy(nn.Module):
    def __init__(self, epsilon: float = 0.):
        super(SmoothCrossEntropy, self).__init__()
        self.epsilon = float(epsilon)

    def forward(self, logits: torch.Tensor, labels: torch.LongTensor) -> torch.Tensor:
        target_probs = torch.full_like(logits, self.epsilon / (logits.shape[1] - 1))
        target_probs.scatter_(1, labels.unsqueeze(1), 1 - self.epsilon)
        loss = F.kl_div(torch.log_softmax(logits, 1), target_probs, reduction='none').sum(1)
#         print(loss)
        return loss
    
    
class VAELoss(nn.Module):
    def __init__(self, kld_weight: float = 0.005):
        super(VAELoss, self).__init__()
        self.kld_weight = float(kld_weight)
    
    def forward(self, recons: torch.Tensor, input: torch.Tensor, mu: torch.Tensor, log_var: torch.Tensor) -> dict:
        
        recons_loss = F.binary_cross_entropy(recons, input)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        loss = recons_loss + self.kld_weight * kld_loss
        loss_dict = {'loss':loss, 'reconstruct':recons_loss, 'kld_loss': -kld_loss}
        print(loss_dict)
        return loss_dict
    
    
class AELoss(nn.Module):
    def __init__(self):
        super(AELoss, self).__init__()
    
    def forward(self, recons: torch.Tensor, input: torch.Tensor) -> dict:
        recons_loss = F.binary_cross_entropy(recons, input)
        loss = recons_loss 
        loss_dict = {'loss':loss, 'reconstruct':recons_loss}
#         print(loss_dict)
        return loss_dict




