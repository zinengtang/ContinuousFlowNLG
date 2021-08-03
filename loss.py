import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class EntropyLoss(nn.Module):
    def __init__(self, ):
        super(EntropyLoss, self).__init__()
    def forward(self, inputs, target):
        return - (torch.nn.functional.one_hot(target.long(), inputs.size(1)) * torch.log(F.softmax(inputs, 1) + 1e-12)).sum(1)

    
class RegressionLoss(nn.Module):
    def __init__(self, size_average=True):
        super(RegressionLoss, self).__init__()
        self.size_average = size_average
    def forward(self, inputs, target):
        if self.size_average:
            divider = inputs.size(0)*inputs.size(1)
        else:
            divider = 1.0
        return ((target - inputs)**2).sum()/divider
                
        
class Kl_Divergence(nn.Module):
    def __init__(self,):
        super(Kl_Divergence, self).__init__()

    def forward(self, z_mu, z_var):
        return 0.5 * torch.mean(torch.exp(z_var) + z_mu**2 - 1. - z_var)


class NLL(nn.Module):
    def __init__(self,):
        super(NLL, self).__init__()

    def forward(self, objective, pixels=32):
        return ((-objective) / float(np.log(2.) * pixels)).mean()

