from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import init
from tqdm import tqdm
from sklearn.cluster import KMeans


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
    def __init__(self, epsilon: float = 0.1):
        super(SmoothCrossEntropy, self).__init__()
        self.epsilon = float(epsilon)

    def forward(self, logits: torch.Tensor, labels: torch.LongTensor) -> torch.Tensor:
        
        # target probs is of shape [N x C], only the gt labels get values 1 - self.epsilon, 
        # other entries for other labels get values (self.epsilon)/(C - 1)
        target_probs = torch.full_like(logits, self.epsilon / (logits.shape[1] - 1))
        target_probs.scatter_(1, labels.unsqueeze(1), 1 - self.epsilon)
        
        # LogSoftMax for logits
        softmax_logits = F.softmax(logits, 1)
        logsoftmax_logits = torch.log(softmax_logits + 1e-5) # manually control underflow
        loss = F.kl_div(logsoftmax_logits, target_probs, reduction='none').sum(1) 
        # kl_divergence = \sum p(y) * log(p(yhat)/p(y)) while CE = - \sum p(y) * log(p(yhat))
        
        if torch.isnan(loss).any(): 
    #         print('labels:', labels)
            print(labels.shape)
            print('Labels Min: {}, Max: {}'.format(torch.min(labels), torch.max(labels)))
    #         print('target prob:', target_probs)
            print(target_probs.shape)
            print(torch.sum(target_probs == 0))
            print('Target probs Min: {}, Max: {}'.format(torch.min(target_probs), torch.max(target_probs)))
    #         print('log_softmax logits: ', torch.log_softmax(logits, 1))
            print(logsoftmax_logits.shape)
            print(torch.sum(logsoftmax_logits == 0))
            print('logsoftmax_logits Min: {}, Max: {}'.format(torch.min(logsoftmax_logits), torch.max(logsoftmax_logits)))
            print(loss)
            raise RuntimeError('Loss has nan values, probably because the log operations lead to -inf')

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


class SoftTriple(nn.Module):
    def __init__(self, la, gamma, tau, margin, dim, cN, K):
        super(SoftTriple, self).__init__()
        self.la = la # scaling factor in softmax loss
        self.gamma = 1./gamma # temperature scaling factor in q_k
        self.tau = tau # tradeoff param in center regularization
        self.margin = margin # delta
        self.cN = cN # number of classes
        self.K = K # number of centers per class
        self.dim = dim
        self.fc = Parameter(torch.Tensor(dim, cN*K)) # proxy embeddings (to be learnt)
        self.weight = torch.zeros(cN*K, cN*K, dtype=torch.bool).cuda() # q_k
        
        for i in range(0, cN): 
            for j in range(0, K):
                self.weight[i*K+j, i*K+j+1:(i+1)*K] = 1
        return
    
    def proxy_initialize(self, gallery_feat: torch.Tensor, gallery_labels: torch.Tensor):
        # use cluster centroids to initialize proxy for each class
        print('Begin proxy initialization')
        initialize_fc = torch.zeros(self.dim, self.cN*self.K)
        for c in tqdm(range(self.cN)): # this may take time
            subset_gallery_feat = gallery_feat[gallery_labels==c]
            kmeans = KMeans(n_clusters=self.K, random_state=0).fit(subset_gallery_feat.detach().cpu().numpy())
            initial_proxy_c = kmeans.cluster_centers_ # of shape K x dim
            initial_proxy_c = torch.from_numpy(initial_proxy_c)
            initial_proxy_c = F.normalize(initial_proxy_c, p=2, dim=1)
            initialize_fc[:, (c*self.K):((c+1)*self.K)] = initial_proxy_c.T
            
        self.fc = Parameter(initialize_fc)
        print(self.fc.shape)
        print(self.fc)
        
    def forward(self, input, target):
        centers = F.normalize(self.fc, p=2, dim=0) # proxy embeddings
        simInd = input.matmul(centers) # dot product xTw
        simStruc = simInd.reshape(-1, self.cN, self.K) # reshape to be dim x C x K
        
        prob = F.softmax(simStruc * self.gamma, dim=2) # soften probability by temperature gamma, gamma is 0.1, concentrated
        simClass = torch.sum(prob * simStruc, dim=2) # S'_i,c: weighted similarity (xTw) to each class weighted by exp(xTw)
        
        marginM = torch.zeros(simClass.shape).cuda()
        marginM[torch.arange(0, marginM.shape[0]), target] = self.margin # margin delta used to break ties
        lossClassify = F.cross_entropy(self.la * (simClass - marginM), target) # SoftTriple loss, la trades the hardness of triplets and the regularizer, temperature = 1/la = 0.2, less concentrated than intra-class temperature
        
        if self.tau > 0 and self.K > 1: 
            # reduce number of centers by pushing similar centers to be close
            simCenter = centers.t().matmul(centers) 
            reg = torch.sum(torch.sqrt(2.0 + 1e-5 - 2. * simCenter[self.weight]))/(self.cN * self.K * (self.K-1.))
            return lossClassify + self.tau*reg # tau is trade-off param on number of proxies regularizer
        else:
            return lossClassify




